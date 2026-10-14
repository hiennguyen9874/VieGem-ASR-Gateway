"""Real Uvicorn server around `create_app` (fake engine by default), shared by CPU end-to-end and contract tests.

Audio is written as 16 kHz float WAV, which FFmpeg decodes bit-exactly, so tests can split it the way the server
does. `name` (the default fake `text_fn`) names a chunk by its first sample, identifying file and chunk position.
"""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
import soundfile as sf
import uvicorn

from app.chunking import split
from app.engine import Engine
from app.main import create_app
from app.transcription import Joiner
from tests.fake_engine import FakeEngine, config
from scripts.subtitles import parse_cues as _parse_cues

KEY = "test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}
SR = 16000


def name(w: np.ndarray) -> str:
    return f"{float(w[0]):.7f}"


def noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    envelope = 0.2 + 0.2 * np.sin(np.arange(n) / SR * 2 * np.pi * 0.7)  # quiet points for the splitter
    return (rng.uniform(-1, 1, n) * envelope).astype(np.float32)


def wav_file(path: Path, samples: np.ndarray) -> Path:
    sf.write(path, samples, SR, subtype="FLOAT")
    return path


def expected_deltas(x: np.ndarray, max_chunk_seconds: float) -> list[str]:
    """SSE deltas the server sends for `x` with the `name` text_fn (empty chunks never occur with noise)."""
    j = Joiner()
    return [j.add(name(x[s.start:s.end])) for s in split(x, int(max_chunk_seconds * SR))]


def parse_cues(body: str, fmt: str) -> list[tuple[float, float, str]]:
    """Strictly parse a gateway SRT/VTT body into (start s, end s, text) cues."""
    cues = _parse_cues(body, fmt)
    assert cues is not None, body
    return cues


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@contextmanager
def serve(engine: Engine | None = None, wait_ready: bool = True, ready_timeout: float = 5.0,
          http_timeout: float = 10.0, **overrides) -> Iterator[SimpleNamespace]:
    """Yield `client` (httpx, authorized), `sdk` (OpenAI client), `base_url`, `app`, `port`, `server`, `engine`.

    `engine=None` is a `FakeEngine(text_fn=name)`; config overrides apply to tests.fake_engine.config()."""
    engine = engine or FakeEngine(text_fn=name)
    app = create_app(config(**overrides), engine)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with httpx2.Client(base_url=base, headers=AUTH, timeout=http_timeout) as client:
        if wait_ready:
            wait_for(lambda: client.get("/ready").status_code == 200, ready_timeout)
        sdk = openai.OpenAI(api_key=KEY, base_url=base + "/v1", max_retries=0, timeout=http_timeout)
        try:
            yield SimpleNamespace(client=client, sdk=sdk, base_url=base, app=app, port=port, server=server,
                                  engine=engine)
        finally:
            if isinstance(engine, FakeEngine):
                engine.gate.set()
                engine.load_gate.set()
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive()
