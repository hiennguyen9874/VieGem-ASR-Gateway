"""Form validation of POST /v1/audio/transcriptions (docs/m4-design.vi.md §4.5), the chunk joiner and model ids."""

import json
from pathlib import Path

import pytest

from app import transcription
from app.errors import APIError
from app.transcription import MODEL_ID, Joiner, Params, model_object, parse_params, served_models
from tests.fake_engine import config

EXAMPLES = Path(__file__).resolve().parents[1] / "contract/examples"
CFG = config()
M = ("model", MODEL_ID)


def error(fields, cfg=CFG) -> tuple[int, str, str | None]:
    with pytest.raises(APIError) as e:
        parse_params(fields, cfg)
    return e.value.status, e.value.code, e.value.param


@pytest.mark.parametrize("fields,expected", [
    ([M], Params(MODEL_ID, "json", False)),
    ([M, ("language", "vi"), ("temperature", "0"), ("prompt", ""), ("response_format", "text"),
      ("stream", "false")], Params(MODEL_ID, "text", False)),
    ([("stream", "true"), M], Params(MODEL_ID, "json", True)),
    ([M, ("response_format", "text"), ("stream", "true")], Params(MODEL_ID, "text", True)),
    ([M, ("temperature", "0.0")], Params(MODEL_ID, "json", False)),
    ([M, ("temperature", "-0")], Params(MODEL_ID, "json", False)),
    *[([M, ("response_format", f)], Params(MODEL_ID, f, False)) for f in ("verbose_json", "srt", "vtt")],
])
def test_valid(fields, expected):
    assert parse_params(fields, CFG) == expected


@pytest.mark.parametrize("fields,expected", [
    # 1. unknown field, in the order sent, name cut to 64 characters
    ([M, ("chunking_strategy", "auto"), ("foo", "1")], (400, "unsupported_parameter", "chunking_strategy")),
    ([M, ("known_speaker_names[]", "a")], (400, "unsupported_parameter", "known_speaker_names[]")),
    ([M, ("include", "logprobs")], (400, "unsupported_parameter", "include")),
    ([M, ("x" * 100, "1")], (400, "unsupported_parameter", "x" * 64)),
    # 2. repeated single field
    ([M, M], (400, "invalid_value", "model")),
    ([M, ("language", "vi"), ("language", "vi")], (400, "invalid_value", "language")),
    # 3. model missing or empty
    ([], (400, "missing_required_parameter", "model")),
    ([("model", "")], (400, "missing_required_parameter", "model")),
    # 4. include[], any value
    ([M, ("include[]", "logprobs")], (400, "unsupported_parameter", "include[]")),
    # 5. granularity values
    *[([M, ("timestamp_granularities[]", v)], (400, "invalid_value", "timestamp_granularities[]"))
      for v in ("", "words", "Word", "token")],
    ([M, ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "x")],
     (400, "invalid_value", "timestamp_granularities[]")),
    # 6. language
    ([M, ("language", "en")], (400, "unsupported_language", "language")),
    ([M, ("language", "")], (400, "unsupported_language", "language")),
    ([M, ("language", "VI")], (400, "unsupported_language", "language")),
    # 7. temperature
    ([M, ("temperature", "abc")], (400, "invalid_value", "temperature")),
    ([M, ("temperature", "")], (400, "invalid_value", "temperature")),
    ([M, ("temperature", "nan")], (400, "invalid_value", "temperature")),
    ([M, ("temperature", "inf")], (400, "invalid_value", "temperature")),
    ([M, ("temperature", "0.2")], (400, "unsupported_parameter", "temperature")),
    # 8. prompt
    ([M, ("prompt", "xin chào")], (400, "unsupported_parameter", "prompt")),
    # 9. response formats
    *[([M, ("response_format", f)], (400, "unsupported_response_format", "response_format"))
      for f in ("diarized_json", "", "JSON", "SRT")],
    # 10. stream: valid values, then only json/text stream
    *[([M, ("stream", v)], (400, "invalid_value", "stream")) for v in ("True", "1", "yes", "")],
    *[([M, ("stream", "true"), ("response_format", f)], (400, "unsupported_parameter", "stream"))
      for f in ("verbose_json", "srt", "vtt")],
    # 11. granularities only with verbose_json
    *[([M, ("timestamp_granularities[]", "word"), *fmt], (400, "unsupported_parameter", "timestamp_granularities[]"))
      for fmt in ([], [("response_format", "json")], [("response_format", "srt")], [("response_format", "text")])],
    # 12. segment never (no Whisper segment fields); word not before the alignment gate
    *[([M, ("response_format", "verbose_json"), *(("timestamp_granularities[]", g) for g in gs)],
       (400, "unsupported_parameter", "timestamp_granularities[]")) for gs in (["segment"], ["word"], ["word", "segment"])],
    # 13. model not served
    ([("model", "whisper-1")], (404, "model_not_found", "model")),
    ([("model", "gpt-4o-transcribe")], (404, "model_not_found", "model")),
])
def test_errors(fields, expected):
    assert error(fields) == expected


def test_errors_come_in_table_order():
    """With every row violated at once, fixing the reported error reveals the next row, in table order."""
    def drop(name):
        return lambda f: [x for x in f if x[0] != name]

    fields = [("unknown", "1"), ("language", "en"), ("language", "en"), ("include[]", "x"),
              ("timestamp_granularities[]", "bad"), ("temperature", "x"), ("prompt", "p"),
              ("response_format", "srt2"), ("stream", "maybe")]
    steps = [
        (("unsupported_parameter", "unknown"), drop("unknown")),
        (("invalid_value", "language"), lambda f: f[:1] + f[2:]),
        (("missing_required_parameter", "model"), lambda f: f + [("model", "nope")]),
        (("unsupported_parameter", "include[]"), drop("include[]")),
        (("invalid_value", "timestamp_granularities[]"),
         lambda f: drop("timestamp_granularities[]")(f) + [("timestamp_granularities[]", "segment")]),
        (("unsupported_language", "language"), drop("language")),
        (("invalid_value", "temperature"), lambda f: drop("temperature")(f) + [("temperature", "0.5")]),
        (("unsupported_parameter", "temperature"), drop("temperature")),
        (("unsupported_parameter", "prompt"), drop("prompt")),
        (("unsupported_response_format", "response_format"),
         lambda f: drop("response_format")(f) + [("response_format", "srt")]),
        (("invalid_value", "stream"), lambda f: drop("stream")(f) + [("stream", "true")]),
        (("unsupported_parameter", "stream"), drop("stream")),
        (("unsupported_parameter", "timestamp_granularities[]"),
         lambda f: drop("response_format")(f) + [("response_format", "verbose_json")]),
        (("unsupported_parameter", "timestamp_granularities[]"), drop("timestamp_granularities[]")),
        (("model_not_found", "model"), None),
    ]
    for want, fix in steps:
        assert error(fields)[1:] == want, fields
        if fix:
            fields = fix(fields)


def test_repeated_array_fields_are_not_duplicates():
    # Repetition is legal for the [] fields; they are refused for their value, not for repeating.
    assert error([M, ("include[]", "a"), ("include[]", "b")]) == (400, "unsupported_parameter", "include[]")


def test_word_granularity_after_the_gate(monkeypatch):
    monkeypatch.setattr(transcription, "WORD_TIMESTAMPS", True)
    vj = ("response_format", "verbose_json")
    words = [M, vj, ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "word")]  # merged
    assert parse_params(words, CFG) == Params(MODEL_ID, "verbose_json", False, frozenset({"word"}))
    assert error([M, vj, ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]) == (
        400, "unsupported_parameter", "timestamp_granularities[]")
    assert error([M, ("timestamp_granularities[]", "word")]) == (
        400, "unsupported_parameter", "timestamp_granularities[]")


def test_whisper_alias_opt_in():
    alias = config(whisper_alias=True)
    assert served_models(CFG) == (MODEL_ID,)
    assert served_models(alias) == (MODEL_ID, "whisper-1")
    assert parse_params([("model", "whisper-1")], alias).model == "whisper-1"
    assert error([("model", "whisper-1")]) == (404, "model_not_found", "model")


def test_model_object_matches_contract_example():
    assert model_object(MODEL_ID) == json.loads((EXAMPLES / "model.json").read_text())
    assert {"object": "list", "data": [model_object(m) for m in served_models(CFG)]} == json.loads(
        (EXAMPLES / "models.json").read_text())


@pytest.mark.parametrize("chunks,deltas", [
    ([], []),
    ([""], []),
    (["a"], ["a"]),
    (["", "a", "b"], ["a", " b"]),  # empty first
    (["a", "", "b"], ["a", " b"]),  # empty middle
    (["a", "b", ""], ["a", " b"]),  # empty last
    (["", "", ""], []),
    (["một số", "loài", "voi"], ["một số", " loài", " voi"]),
])
def test_joiner(chunks, deltas):
    j = Joiner()
    got = [d for d in map(j.add, chunks) if d is not None]
    assert got == deltas
    assert "".join(got) == j.text == " ".join(c for c in chunks if c)
