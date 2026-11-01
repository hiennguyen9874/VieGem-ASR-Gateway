"""Real VieGemEngine behind the worker thread and scheduler: load, warm-up, concurrent lanes give the M0 baseline text.

Needs a CUDA GPU and staged assets (no network). Pick a free GPU: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from app.engine import EngineState, EngineWorker, VieGemEngine
from app.observability import Metrics
from app.scheduler import JobKind, Scheduler
from scripts.scoring import text_sha256
from tests.fake_engine import config

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("VIEGEM_MODELS_DIR", ROOT / "models"))
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


def test_real_engine_serves_concurrent_lanes_with_baseline_text():
    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    clips = {name: sf.read(ROOT / "tests/fixtures" / name, dtype="float32")[0] for name in ("vi_a.wav", "vi_b.wav")}

    async def main():
        cfg = config(models_dir=MODELS)
        metrics = Metrics()
        worker = EngineWorker(VieGemEngine(MODELS, "cuda", cfg.max_chunk_seconds), cfg.infer_hang_seconds, metrics)
        scheduler = Scheduler(worker, cfg, metrics)
        worker.start()
        scheduler.start()
        deadline = time.monotonic() + 300
        while worker.state == EngineState.LOADING:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)
        assert worker.state == EngineState.READY

        async def lane_run(kind, name):
            async with scheduler.open_lane(kind) as lane:
                return [(await (await lane.submit(clips[name]))).text for _ in range(3)]

        try:
            return await asyncio.gather(lane_run(JobKind.FILE, "vi_a.wav"), lane_run(JobKind.LIVE, "vi_b.wav"),
                                        lane_run(JobKind.FILE, "vi_b.wav"))
        finally:
            assert await scheduler.close(30)
            assert await asyncio.to_thread(worker.stop, 30)

    a, b_live, b_file = asyncio.run(main())
    assert [text_sha256(t) for t in a] == [baseline["vi_a.wav:full"]["text_sha256"]] * 3
    assert [text_sha256(t) for t in b_live] == [text_sha256(t) for t in b_file] == [baseline["vi_b.wav:full"]["text_sha256"]] * 3


def test_chunk_result_path_is_the_greedy_decode():
    """The engine's best path equals the vendored `greedy_decode` and its text is the baseline text."""
    from app.viegem.model import greedy_decode

    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    engine = VieGemEngine(MODELS, "cuda", warmup_seconds=1.0)
    engine.load()
    pieces = engine.pieces()
    assert len(pieces) == engine._model.cfg.vocab_size and pieces[:2] == ("", "")
    for name in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"):
        wav = sf.read(ROOT / "tests/fixtures" / name, dtype="float32")[0]
        result = engine.transcribe(wav)
        assert text_sha256(result.text) == baseline[f"{name}:full"]["text_sha256"], name
        logits, lens = engine.forward(wav)
        if name == "silence_2s.wav":
            assert result.text == "" and result.units == result.first == result.last == ()
        else:
            assert list(result.units) == greedy_decode(logits, lens)[0], name
        assert result.frames == int(lens[0]) == -(-(len(wav) - 160) // 640), name
        assert all(f <= l for f, l in zip(result.first, result.last))
        assert all(l < f for l, f in zip(result.last, result.first[1:]))
        assert not result.last or result.last[-1] < result.frames

    # Exact digital zeros are suppressed, but nonzero low-level audio still uses the greedy path.
    for wav in [np.zeros(n, np.float32) for n in (1601, 16000, 480000)] + [
        np.full(32000, 1e-5, np.float32),
        sf.read(ROOT / "tests/fixtures/vi_a.wav", dtype="float32")[0] * 1e-4,
    ]:
        result = engine.transcribe(wav)
        logits, lens = engine.forward(wav)
        assert result.frames == int(lens[0])
        if not np.any(wav):
            assert result.text == "" and result.units == result.first == result.last == ()
        else:
            expected = greedy_decode(logits, lens)[0]
            assert list(result.units) == expected
            assert result.text == engine._units.decode(engine._tokenizer, expected)
