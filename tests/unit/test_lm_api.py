"""VIEGEM_LM on CPU, end to end: real Uvicorn, real FFmpeg, the real spawn process pool with a tiny ARPA LM, and a
fake LM-on engine whose lattice lets the LM change the greedy text ("ac d" -> "ab d").

Failure modes covered: some format (json/text/SSE/timed/realtime) still returns greedy text; timed cues are not
aligned to the LM text or leave their chunk; the engine's job budget is held during the LM decode or LM work leaks
after a request; /ready turns true before the LM is loaded; a missing LM file is not a startup error; the pool
bounds are not enforced (overload must be 429, a deadline 503, a broken pool 500 + unhealthy).
"""

import asyncio
import concurrent.futures
import json
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from app import lm as lm_module
from app.config import ConfigError
from app.errors import APIError
from app.engine import EngineWorker
from app.lm import LMPool
from app.scheduler import JobKind, Scheduler
from app.viegem.lm_decode import LMDecoder
from app.main import create_app
from app.observability import Metrics
from app.transcription import MODEL_ID
from tests.fake_engine import config, fake_lattice, FakeEngine
from tests.lm_fixture import stage_tiny_lm, WORDS
from app.chunking import split
from tests.live_server import expected_turn, noise, parse_cues, pcm24, serve, SR, wait_for, wav_file, WS

URL = "/v1/audio/transcriptions"
CHUNK = 1.0


def lm_engine() -> FakeEngine:
    return FakeEngine(text_fn=lambda w: "ac d", near={"c": "b"})


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    models = stage_tiny_lm(tmp_path_factory.mktemp("models"))
    with serve(lm_engine(), ready_timeout=60, lm=True, lm_workers=2, models_dir=models,
               max_chunk_seconds=CHUNK) as s:
        yield s


def post(srv, path: Path, **fields):
    r = srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data={"model": MODEL_ID, **fields})
    assert r.status_code == 200, r.text
    return r


def test_greedy_path_of_the_fake_lattice_is_the_greedy_text():
    lat = fake_lattice("ac d", SR, {"c": "b"})
    best = [int(lat.ids[lat.offsets[t] + np.argmax(lat.logps[lat.offsets[t]:lat.offsets[t + 1]])])
            for t in range(lat.frames)]
    assert FakeEngine().text([u for i, u in enumerate(best) if i == 0 or u != best[i - 1]]) == "ac d"


def test_every_format_and_realtime_return_lm_text(srv, tmp_path):
    x = noise(2.5, seed=1)  # three chunks of at most 1 s
    path = wav_file(tmp_path / "a.wav", x)
    text = post(srv, path).json()["text"]
    chunks = len(split(x, int(CHUNK * SR)))
    assert chunks > 1 and text == " ".join(["ab d"] * chunks)  # greedy would be "ac d ..."
    assert post(srv, path, response_format="text").text == text
    assert post(srv, path, response_format="verbose_json").json()["text"] == text
    events = [json.loads(b.removeprefix("data: ")) for b in post(srv, path, stream="true").text.split("\n\n")
              if b.strip()]
    assert [e["type"] for e in events] == ["transcript.text.delta"] * chunks + ["transcript.text.done"]
    assert "".join(e["delta"] for e in events[:-1]) == events[-1]["text"] == text
    for fmt in ("srt", "vtt"):
        cues = parse_cues(post(srv, path, response_format=fmt).text, fmt)
        assert " ".join(c[2] for c in cues) == text
        assert all(0 <= s < e <= len(x) / SR for s, e, _ in cues)
        assert all(a[1] <= b[0] for a, b in zip(cues, cues[1:]))

    ws = WS(srv, timeout=30)
    try:
        pcm = pcm24(1.5, seed=2)
        ws.append(pcm)
        ws.send("input_audio_buffer.commit")
        _, deltas, done = ws.turn()
        assert done["type"] == "conversation.item.input_audio_transcription.completed", done
        assert done["transcript"] == "".join(deltas) == " ".join(["ab d"] * len(expected_turn(pcm, CHUNK)))
    finally:
        ws.close()

    scheduler = srv.app.state.scheduler
    wait_for(lambda: scheduler._jobs == 0 and not scheduler._lm_tasks and scheduler._lanes == 0)
    metrics = srv.app.state.metrics
    assert metrics.value("lm_decodes_total", outcome="ok") >= 6 * chunks + len(deltas)
    assert "viegem_lm_pending 0" in srv.client.get("/metrics").text


def test_ready_waits_for_the_lm_and_missing_lm_files_abort_startup(tmp_path):
    with pytest.raises(ConfigError, match="LM file not staged"):
        create_app(config(lm=True, models_dir=tmp_path), FakeEngine(near={}))
    models = stage_tiny_lm(tmp_path / "models")
    gate = threading.Event()
    original = lm_module.LMPool._warmup

    async def slow_warmup(self):
        await asyncio.to_thread(gate.wait, 10)
        await original(self)

    lm_module.LMPool._warmup = slow_warmup
    try:
        with serve(FakeEngine(near={}), wait_ready=False, lm=True, lm_workers=1, models_dir=models) as s:
            wait_for(lambda: s.app.state.worker.ready)
            assert s.client.get("/ready").status_code == 503
            gate.set()
            wait_for(lambda: s.client.get("/ready").status_code == 200, 30)
            assert s.client.get("/health").status_code == 200
    finally:
        lm_module.LMPool._warmup = original
        gate.set()


# -- pool bounds (thread executor in place of processes, so the decode can be held) ---------------------------------


def pool(tmp_path, max_pending=1, queue_timeout=0.2):
    p = LMPool(stage_tiny_lm(tmp_path), 1, max_pending, queue_timeout, Metrics())
    p._executor = concurrent.futures.ThreadPoolExecutor(2)
    p._slots = asyncio.Semaphore(max_pending)
    return p


LATTICE = fake_lattice("ac d", SR, {"c": "b"})


def test_full_pool_is_429_and_a_cancelled_decode_keeps_its_slot_until_it_ends(tmp_path, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(lm_module, "_decode", lambda *a: (gate.wait(5), ((), (), ()))[1])

    async def run():
        p = pool(tmp_path)
        first = asyncio.create_task(p.decode(LATTICE, FakeEngine().pieces(), None))
        await asyncio.sleep(0.05)
        with pytest.raises(APIError) as e:
            await p.decode(LATTICE, FakeEngine().pieces(), None)
        assert (e.value.status, e.value.code) == (429, "server_busy")
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert p._pending == 1  # still running in the pool
        gate.set()
        for _ in range(100):
            if p._pending == 0:
                break
            await asyncio.sleep(0.01)
        assert p._pending == 0
        assert await p.decode(LATTICE, FakeEngine().pieces(), None) == ((), (), ())
        p._executor.shutdown()

    asyncio.run(run())


def test_deadline_is_503_and_a_broken_pool_is_500_and_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr(lm_module, "_decode", lambda *a: (time.sleep(0.3), ((), (), ()))[1])

    async def run():
        p = pool(tmp_path, max_pending=2)
        with pytest.raises(APIError) as e:
            await p.decode(LATTICE, FakeEngine().pieces(), time.monotonic() + 0.05)
        assert (e.value.status, e.value.code) == (503, "request_timeout")

        def broken(*a):
            raise BrokenProcessPool("worker died")

        monkeypatch.setattr(lm_module, "_decode", broken)
        with pytest.raises(APIError) as e:
            await p.decode(LATTICE, FakeEngine().pieces(), None)
        assert (e.value.status, e.value.code) == (500, "inference_failed")
        assert not p.healthy
        p._executor.shutdown()

    asyncio.run(run())


def test_gpu_jobs_continue_while_lm_decodes_wait_and_lane_close_drops_lm_work(tmp_path, monkeypatch):
    """The LM stage runs after the job's inference: the engine is free for the next chunk meanwhile."""
    gate = threading.Event()
    real = lm_module._decode
    monkeypatch.setattr(lm_module, "_decode", lambda *a: (gate.wait(5), real(*a))[1])
    cfg = config(lm=True)
    engine = lm_engine()

    async def run():
        p = pool(tmp_path, max_pending=4)
        lm_module._decoder = LMDecoder(*p._paths[:1], unigrams=WORDS, alpha=lm_module.ALPHA,
                                       beta=lm_module.BETA, unk_offset=lm_module.UNK_OFFSET)
        metrics = Metrics()
        worker = EngineWorker(engine, cfg.infer_hang_seconds, metrics)
        scheduler = Scheduler(worker, cfg, metrics, p)
        worker.start()
        while not worker.ready:
            await asyncio.sleep(0.01)
        scheduler.start()
        try:
            a, b = scheduler.open_lane(JobKind.FILE), scheduler.open_lane(JobKind.LIVE)
            job_a = await a.submit(np.full(SR, 0.1, np.float32))
            job_b = await b.submit(np.full(SR, 0.2, np.float32))
            while len(engine.calls) < 2 or scheduler._jobs:  # both inferences done, both decodes held
                await asyncio.sleep(0.01)
            assert len(scheduler._lm_tasks) == 2 and not job_a.future.done()
            b.close()  # cancels job b while its decode is pending
            gate.set()
            result = await job_a
            assert result.text == "ab d" and len(result.units) == len(result.first) == 3
            with pytest.raises(asyncio.CancelledError):
                await job_b
            while scheduler._lm_tasks:
                await asyncio.sleep(0.01)
            a.close()
        finally:
            gate.set()
            await scheduler.close(5)
            assert worker.stop(5)
            p._executor.shutdown()

    asyncio.run(asyncio.wait_for(run(), 30))
