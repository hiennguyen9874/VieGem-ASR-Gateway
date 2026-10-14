"""Engine stand-in for CPU tests: records concurrency, can block, fail or hang on demand.

A waveform's identity is its fill value: `wav(7)` transcribes to "7", so tests can check results are not mixed up.
"""

import threading
import time

import numpy as np

from app.config import Config


def wav(value: int, samples: int = 1600) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def config(**overrides) -> Config:
    defaults = dict(api_keys=("test-key-0123456789",), queue_timeout_seconds=2.0, infer_hang_seconds=30.0)
    return Config(**(defaults | overrides))


class FakeEngine:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[int] = []  # identities in execution order
        self.gate = threading.Event()  # transcribe() blocks until set
        self.gate.set()
        self.started = threading.Semaphore(0)  # released once per started call
        self.load_gate = threading.Event()
        self.load_gate.set()
        self.load_error: Exception | None = None
        self.fail_calls = 0  # next N transcribe() calls raise
        self.warmup_error: Exception | None = None
        self.warmups = 0

    def load(self) -> None:
        self.load_gate.wait(10)
        if self.load_error:
            raise self.load_error

    def warmup(self) -> None:
        self.warmups += 1
        if self.warmup_error:
            raise self.warmup_error

    def transcribe(self, wav: np.ndarray) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(int(wav[0]))
        self.started.release()
        try:
            self.gate.wait(10)
            time.sleep(self.delay)
            if self.fail_calls:
                self.fail_calls -= 1
                raise RuntimeError("CUDA error: simulated")
            return str(int(wav[0]))
        finally:
            with self.lock:
                self.active -= 1
