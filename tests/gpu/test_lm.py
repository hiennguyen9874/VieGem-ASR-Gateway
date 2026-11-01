"""VIEGEM_LM on the real engine over a real Uvicorn server (ADR 0002): REST in every format and Realtime return the
LM text, equal to the pure LM decode of the same audio, and timed cues are aligned to that text (R2a gap checks
across a chunk boundary). Not the LM gate: WER, p95 decode time and RTF thresholds are measured separately.

Needs a CUDA GPU, staged assets (including the LM files) and ffmpeg: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch

from app.engine import VieGemEngine
from app.lm import ALPHA, BETA, decode_lattice, LM_DIR, LM_FILE, UNIGRAMS_FILE, UNK_OFFSET
from app.realtime import PCM_RATE, TurnBuffer
from app.viegem.lm_decode import LMDecoder
from tests.live_server import parse_cues, serve, SR, wav_file, WS

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("VIEGEM_MODELS_DIR", ROOT / "models"))
FIXTURES = ROOT / "tests/fixtures"
URL = "/v1/audio/transcriptions"
MODEL = "viegem-asr"
GAP_TOLERANCE_S = 1.5  # as tests/gpu/test_server.py (alignment probe)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / LM_DIR / LM_FILE).is_file(), reason="LM assets not staged"),
]


@pytest.fixture(scope="module")
def srv():
    engine = VieGemEngine(MODELS, "cuda", warmup_seconds=30.0, lm=True)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS, lm=True) as s:
        yield s


@pytest.fixture(scope="module")
def offline():
    """Pure LM decode of one chunk: a second engine instance (this thread) + the decoder, no server."""
    engine = VieGemEngine(MODELS, "cuda", warmup_seconds=30.0, lm=True)
    engine.load()
    words = (MODELS / LM_DIR / UNIGRAMS_FILE).read_text(encoding="utf-8").split()
    decoder = LMDecoder(MODELS / LM_DIR / LM_FILE, unigrams=words, alpha=ALPHA, beta=BETA, unk_offset=UNK_OFFSET)
    greedy = VieGemEngine(MODELS, "cuda", warmup_seconds=30.0)
    greedy._model, greedy._tokenizer, greedy._units, greedy._device = (
        engine._model, engine._tokenizer, engine._units, engine._device)
    pieces = engine.pieces()

    def run(wav: np.ndarray) -> tuple[str, str]:
        units, _, _ = decode_lattice(decoder, engine.transcribe(wav), pieces)
        return engine.text(units), greedy.transcribe(wav).text

    return run


def load(name: str) -> np.ndarray:
    return sf.read(FIXTURES / name, dtype="float32")[0]


def post(srv, path: Path, **fields):
    r = srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data={"model": MODEL, **fields})
    assert r.status_code == 200, r.text
    return r


def test_rest_formats_return_the_lm_text(srv, offline):
    for name in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"):
        path = FIXTURES / name
        text = post(srv, path).json()["text"]
        if name == "silence_2s.wav":
            assert text == ""  # digital zeros: empty without decoding
        else:
            assert text == offline(load(name))[0], name
        assert post(srv, path, response_format="text").text == text
        assert post(srv, path, response_format="verbose_json").json()["text"] == text
        events = [json.loads(b.removeprefix("data: ")) for b in post(srv, path, stream="true").text.split("\n\n")
                  if b.strip()]
        assert events[-1] == {"type": "transcript.text.done", "text": text}
        assert "".join(e["delta"] for e in events[:-1]) == text
        for fmt in ("srt", "vtt"):
            assert " ".join(c[2] for c in parse_cues(post(srv, path, response_format=fmt).text, fmt)) == text


def test_lm_changes_some_greedy_text(offline):
    """The LM is really applied: on the long fixture's first 30 s, the LM text differs from greedy."""
    long = ROOT / "fixtures-large/vi_long.wav"
    if not long.is_file():
        pytest.skip("fixtures-large not fetched")
    lm_text, greedy_text = offline(sf.read(long, dtype="float32")[0][: 30 * SR])
    assert lm_text and lm_text != greedy_text


def test_lm_cues_respect_known_silences_across_a_chunk_boundary(srv):
    a, b = load("vi_a.wav"), load("vi_b.wav")
    gap = np.zeros(SR, np.float32)
    parts = [a, gap, b, gap, a, gap, b]  # 37.7 s: two chunks, cut inside the last gap
    edges = np.cumsum([0] + [len(p) for p in parts]) / SR
    path = wav_file(Path(os.environ.get("TMPDIR", "/tmp")) / "viegem_lm_composite.wav", np.concatenate(parts))
    try:
        text = post(srv, path).json()["text"]
        for fmt in ("srt", "vtt"):
            cues = parse_cues(post(srv, path, response_format=fmt).text, fmt)
            assert " ".join(c[2] for c in cues) == text
            assert all(0 <= s < e <= edges[-1] for s, e, _ in cues)
            assert all(p[1] <= n[0] for p, n in zip(cues, cues[1:]))
            for g0, g1 in ((edges[i], edges[i + 1]) for i in (1, 3, 5)):
                before = [c for c in cues if c[0] < g0]
                after = [c for c in cues if c[0] >= g0]
                assert before and after
                assert g0 - GAP_TOLERANCE_S <= before[-1][1] <= g0 + 0.0005, (fmt, g0, before[-1])
                assert g1 - 0.0005 <= after[0][0] <= g1 + GAP_TOLERANCE_S, (fmt, g1, after[0])
    finally:
        path.unlink()


def test_realtime_items_use_the_lm(srv, offline, tmp_path):
    ws = WS(srv, timeout=120)
    try:
        for name in ("vi_a.wav", "vi_b.wav"):
            up = soxr.resample(load(name), SR, PCM_RATE, quality="HQ")
            pcm = np.round(np.clip(up, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
            ws.append(pcm, packet=PCM_RATE // 50 * 2)
            ws.send("input_audio_buffer.commit")
            _, deltas, done = ws.turn()
            assert done["type"] == "conversation.item.input_audio_transcription.completed", done
            buf = TurnBuffer(len(pcm))
            buf.append(pcm)
            wav = buf.commit()[0]
            lm_text, _ = offline(wav)
            assert done["transcript"] == "".join(deltas) == lm_text, name
            rest = post(srv, wav_file(tmp_path / f"c_{name}", wav)).json()["text"]
            assert done["transcript"] == rest
    finally:
        ws.close()
