"""The real app on a GPU over a real Uvicorn server: transcripts through the HTTP API equal the offline baseline,
JSON/text/SSE agree, concurrent requests do not mix, the long fixture stays within the M2 WER bound, silence -> "".

Needs a CUDA GPU, staged assets and ffmpeg: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
import torch

from app.engine import QuanEngine
from scripts.smoke_inference import wer
from tests.live_server import serve

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("QUAN_MODELS_DIR", ROOT / "models"))
FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
URL = "/v1/audio/transcriptions"
MODEL = "quan-transcribe-1.0"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


@pytest.fixture(scope="module")
def srv():
    engine = QuanEngine(MODELS, "cuda", warmup_seconds=30.0)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS) as s:
        yield s


def post(srv, path: Path, **fields):
    r = srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data={"model": MODEL, **fields})
    assert r.status_code == 200, r.text
    return r


def stream(srv, path: Path) -> tuple[list[str], str]:
    events = [json.loads(b.removeprefix("data: ")) for b in post(srv, path, stream="true").text.split("\n\n")
              if b.strip()]
    assert events[-1]["type"] == "transcript.text.done", events[-1]
    return [e["delta"] for e in events[:-1]], events[-1]["text"]


def test_transcripts_match_baseline_in_every_format(srv):
    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    for name in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"):
        path = FIXTURES / name
        body = post(srv, path).json()
        assert body["text"] == baseline[f"{name}:full"]["text"], name
        assert post(srv, path, response_format="text").text == body["text"]
        deltas, done = stream(srv, path)
        assert "".join(deltas) == done == body["text"]
    assert stream(srv, FIXTURES / "silence_2s.wav") == ([], "")


def test_concurrent_requests_do_not_mix(srv):
    expected = {n: post(srv, FIXTURES / n).json()["text"] for n in ("vi_a.wav", "vi_b.wav")}
    results: dict[tuple[str, int], str] = {}

    def run(name: str, i: int):
        results[(name, i)] = post(srv, FIXTURES / name).json()["text"]

    threads = [threading.Thread(target=run, args=(n, i)) for i in range(3) for n in expected]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert results == {(n, i): expected[n] for i in range(3) for n in expected}


@pytest.mark.skipif(not LONG.exists(), reason="large fixture not fetched (scripts/fetch_fixtures.py)")
def test_long_m4a_within_wer_bound(srv, tmp_path):
    m4a = tmp_path / "long.m4a"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(LONG), "-c:a", "aac", str(m4a)],
                   check=True)
    ref = json.loads((FIXTURES / "manifest.json").read_text())["files"]["vi_long.wav"]["reference"]
    deltas, done = stream(srv, m4a)
    assert len(deltas) == 4  # 106.6 s at <= 30 s per chunk, none empty
    assert done == post(srv, m4a).json()["text"]
    assert wer(ref, done) <= 0.10
