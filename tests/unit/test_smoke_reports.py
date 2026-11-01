"""Smoke evidence must preserve comparisons without persisting recognized speech."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fake_engine import FakeEngine
from tests.live_server import KEY, serve

ROOT = Path(__file__).resolve().parents[2]


def assert_metadata_only(value):
    if isinstance(value, dict):
        for key in ("text", "texts", "text_body", "sse_joined", "sse_done", "baseline",
                    "transcribe", "empty_input_transcribe"):
            assert key not in value or isinstance(value[key], (int, float))
        for child in value.values():
            assert_metadata_only(child)
    elif isinstance(value, list):
        for child in value:
            assert_metadata_only(child)


@pytest.mark.parametrize(("errors", "passes"), [(17, True), (18, False)])
def test_fixture_wer_is_weighted_full_context_and_keeps_the_recorded_limit(errors, passes):
    from scripts.smoke_inference import score_fixtures

    score, failures = score_fixtures({
        "a:full": {"word_errors": 1, "reference_words": 25, "wer": 0.04},
        "b:full": {"word_errors": errors - 1, "reference_words": 304, "wer": (errors - 1) / 304},
        "a:streaming": {"word_errors": 25, "reference_words": 25, "wer": 1.0},
        "silence:full": {"text_sha256": "unused"},
    })
    assert score["word_errors"] == errors and score["reference_words"] == 329
    assert score["wer"] == round(errors / 329, 4)
    assert score["max_wer"] == 17 / 329
    assert score["passes"] is passes
    assert bool(failures) is not passes


def test_missing_reference_words_cannot_pass_the_wer_gate():
    from scripts.smoke_inference import score_fixtures

    score, failures = score_fixtures({})
    assert not score["passes"]
    assert failures == ["fixture WER: no scored reference words"]


def test_committed_baselines_are_metadata_only():
    for path in (ROOT / "docs/baseline").glob("*.json"):
        assert_metadata_only(json.loads(path.read_text()))


def test_server_smoke_report_and_stdout_do_not_contain_transcripts(tmp_path):
    marker = "private recognition sentinel"
    out = tmp_path / "server.json"
    with serve(FakeEngine(text_fn=lambda _: marker)) as srv:
        result = subprocess.run(
            [sys.executable, "scripts/smoke_server.py", "--base-url", srv.base_url,
             "--out", str(out)],
            cwd=ROOT, env={**os.environ, "OPENAI_API_KEY": KEY},
            capture_output=True, text=True, timeout=90,
        )
    # The deliberately wrong fake transcript must fail comparisons, not be hidden by sanitization.
    assert result.returncode == 1, result.stderr
    report = json.loads(out.read_text())
    assert report["failures"]
    assert not report["results"]["vi_a.wav"]["matches_baseline"]
    assert marker not in out.read_text() + result.stdout + result.stderr
    assert KEY not in out.read_text() + result.stdout + result.stderr
    assert_metadata_only(report)
