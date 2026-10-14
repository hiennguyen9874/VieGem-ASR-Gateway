"""The examples in tests/contract/examples/ (docs/api-contract.vi.md) parse with the pinned OpenAI Python SDK.

HTTP examples go through the real SDK client over a mock transport, so this also pins what the SDK sends
(path, auth header, multipart field names) and how it parses JSON, text, SSE and errors.
Realtime examples are validated (not just constructed) against the SDK's pydantic event unions.
"""

import base64
import email.parser
import json
from pathlib import Path

import httpx2
import openai
import pydantic
import pytest
from openai.types.realtime import (
    RealtimeClientEvent,
    RealtimeServerEvent,
    RealtimeTranscriptionSessionCreateRequest,
)

EXAMPLES = Path(__file__).parent / "examples"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "vi_a.wav"
MODEL = "quan-transcribe-1.0"


def test_sdk_version_is_pinned():
    assert openai.__version__ == "3.31.0"


def parse_multipart(request) -> dict[str, list]:
    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.read()
    msg = email.parser.BytesParser().parsebytes(raw)
    fields: dict[str, list] = {}
    for part in msg.get_payload():
        name = part.get_param("name", header="content-disposition")
        value = part.get_filename() or part.get_payload(decode=True).decode()
        fields.setdefault(name, []).append(value)
    return fields


def client_with(status: int, body: bytes, content_type: str, seen: list):
    def handler(request):
        seen.append(request)
        return httpx2.Response(status, content=body, headers={"content-type": content_type})

    return openai.OpenAI(
        api_key="sk-test", base_url="http://gateway.test/v1", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def example(name: str) -> bytes:
    return (EXAMPLES / name).read_bytes()


def test_models_list_and_retrieve():
    seen = []
    c = client_with(200, example("models.json"), "application/json", seen)
    assert [m.id for m in c.models.list()] == [MODEL]
    c = client_with(200, example("model.json"), "application/json", seen)
    m = c.models.retrieve(MODEL)
    assert (m.id, m.object, m.owned_by, m.created) == (MODEL, "model", "qnguyen3", 1791950390)
    assert [r.url.path for r in seen] == ["/v1/models", f"/v1/models/{MODEL}"]
    assert all(r.headers["authorization"] == "Bearer sk-test" for r in seen)


def test_json_transcription_and_request_encoding():
    seen = []
    c = client_with(200, example("transcription.json"), "application/json", seen)
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, language="vi", temperature=0)
    assert isinstance(t, openai.types.audio.Transcription)
    assert t.text.startswith("các sản phẩm") and t.usage.type == "duration" and t.usage.seconds == 8.88
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/audio/transcriptions"
    fields = parse_multipart(req)
    assert fields == {"file": ["vi_a.wav"], "model": [MODEL], "language": ["vi"], "temperature": ["0"]}


def test_text_transcription_returns_body_verbatim():
    c = client_with(200, example("transcription.txt"), "text/plain; charset=utf-8", [])
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, response_format="text")
    assert t == example("transcription.txt").decode()


def test_array_fields_are_sent_as_repeated_bracket_fields():
    """The gateway must parse `include[]` / `timestamp_granularities[]` (repeated), not `include`."""
    seen = []
    c = client_with(200, example("transcription.json"), "application/json", seen)
    with FIXTURE.open("rb") as f:
        c.audio.transcriptions.create(
            file=f, model=MODEL, response_format="verbose_json",
            timestamp_granularities=["word", "segment"], include=["logprobs"],
        )
    fields = parse_multipart(seen[0])
    assert fields["timestamp_granularities[]"] == ["word", "segment"]
    assert fields["include[]"] == ["logprobs"]
    assert "include" not in fields and "timestamp_granularities" not in fields


def test_sse_stream_deltas_join_to_done():
    seen = []
    c = client_with(200, example("transcription_stream.sse"), "text/event-stream", seen)
    with FIXTURE.open("rb") as f:
        events = list(c.audio.transcriptions.create(file=f, model=MODEL, stream=True))
    assert parse_multipart(seen[0])["stream"] == ["true"]
    assert [e.type for e in events] == ["transcript.text.delta", "transcript.text.delta", "transcript.text.done"]
    assert "".join(e.delta for e in events[:-1]) == events[-1].text
    assert events[-1].usage is None  # done.usage only has a token variant; we omit it


def test_sse_error_after_headers_raises_and_has_no_done():
    c = client_with(200, example("transcription_stream_error.sse"), "text/event-stream", [])
    got = []
    with FIXTURE.open("rb") as f, pytest.raises(openai.APIError) as exc:
        for e in c.audio.transcriptions.create(file=f, model=MODEL, stream=True):
            got.append(e.type)
    assert got == ["transcript.text.delta"]
    assert exc.value.code == "inference_failed" and exc.value.type == "server_error"


def test_http_error_envelope():
    c = client_with(400, example("error.json"), "application/json", [])
    with FIXTURE.open("rb") as f, pytest.raises(openai.BadRequestError) as exc:
        c.audio.transcriptions.create(file=f, model=MODEL, language="en")
    e = exc.value
    assert (e.status_code, e.code, e.param, e.type) == (400, "unsupported_language", "language", "invalid_request_error")


def lines(name: str) -> list[dict]:
    return [json.loads(line) for line in example(name).decode().splitlines() if line.strip()]


def test_realtime_client_events_validate():
    adapter = pydantic.TypeAdapter(RealtimeClientEvent)
    events = [adapter.validate_python(e) for e in lines("realtime_client.jsonl")]
    assert isinstance(events[0].session, RealtimeTranscriptionSessionCreateRequest)
    pcm = base64.b64decode(events[1].audio, validate=True)
    assert len(pcm) % 2 == 0  # PCM16 little-endian


def test_realtime_server_events_validate_and_are_consistent():
    adapter = pydantic.TypeAdapter(RealtimeServerEvent)
    raw = lines("realtime_server.jsonl")
    events = [adapter.validate_python(e) for e in raw]
    for r, e in zip(raw, events):
        assert e.type == r["type"]
        assert e.model_dump(exclude_unset=True) == r  # no field silently dropped or renamed
    by_type = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)
    for e in by_type["session.created"] + by_type["session.updated"]:
        assert isinstance(e.session, RealtimeTranscriptionSessionCreateRequest)
        assert e.session.audio.input.format.rate == 24000
    committed = by_type["input_audio_buffer.committed"][0]
    added = by_type["conversation.item.added"][0]
    assert added.item.id == committed.item_id and added.previous_item_id == committed.previous_item_id
    deltas = by_type["conversation.item.input_audio_transcription.delta"]
    done = by_type["conversation.item.input_audio_transcription.completed"][0]
    assert all(d.item_id == done.item_id and d.content_index == done.content_index == 0 for d in deltas)
    assert "".join(d.delta for d in deltas) == done.transcript
    assert done.usage.type == "duration"
    assert len({e.event_id for e in events}) == len(events)
