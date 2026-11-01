"""Engine stand-in for CPU tests: records concurrency, can block, fail or hang on demand.

A waveform's identity is its fill value: `wav(7)` transcribes to "7", so tests can check results are not mixed up.
Tests on real (decoded) audio pass `text_fn` to derive the text from the chunk instead.

Results carry a synthetic CTC path consistent with the text over a character vocabulary (`PIECES`): each word is a
"▁"-prefixed first character followed by one unit per character, spread evenly over the chunk's encoder frames.
"""

import threading
import time
from collections.abc import Callable

import numpy as np

from app.config import Config
from app.engine import ChunkResult
from app.lm import Lattice, prune

VI_LETTERS = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
CHARS = [chr(c) for c in range(33, 127)] + list(VI_LETTERS)
PIECES = ("", "") + tuple(p for c in CHARS for p in ("\u2581" + c, c))
UNIT = {p: i for i, p in enumerate(PIECES) if i >= 2}


def fake_result(text: str, samples: int) -> ChunkResult:
    """`text` (single-space separated) as the greedy result of a `samples`-long chunk."""
    pieces = [p for word in text.split(" ") if word for p in ("\u2581" + word[0], *word[1:])]
    units = tuple(UNIT[p] for p in pieces)
    frames = max(1, -(-(samples - 160) // 640))  # valid encoder frames: 640 k + 160 < samples
    n = len(units)
    first = tuple(i * frames // n for i in range(n))
    last = tuple(max(f, (i + 1) * frames // n - 1) for i, f in enumerate(first))
    return ChunkResult(text, units, first, last, frames)


def fake_lattice(text: str, samples: int, near: dict[str, str]) -> Lattice:
    """The LM-on engine output for `fake_result(text)`: each path unit has prob 0.5 on its frames and, for a
    character in `near`, its substitute 0.45 (greedy keeps `text`, a language model can pick the substitute)."""
    result = fake_result(text, samples)
    probs = np.full((result.frames, len(PIECES)), 1e-5)
    probs[:, 0] = 0.9
    for u, f, last in zip(result.units, result.first, result.last):
        probs[f : last + 1, 0] = 1e-3  # pruned: no blank inside a unit, so the LM cannot repeat it
        probs[f : last + 1, u] = 0.5
        piece = PIECES[u]
        if piece.lstrip("\u2581") in near:
            alt = piece[: len(piece) - 1] + near[piece.lstrip("\u2581")]
            probs[f : last + 1, UNIT[alt]] = 0.45
    lp = np.log(probs / probs.sum(-1, keepdims=True))
    return prune(lp.astype(np.float32))


def wav(value: int, samples: int = 1600) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def config(**overrides) -> Config:
    defaults = dict(api_keys=("test-key-0123456789",), queue_timeout_seconds=2.0, infer_hang_seconds=30.0)
    return Config(**(defaults | overrides))


class FakeEngine:
    def __init__(
        self,
        delay: float = 0.0,
        text_fn: Callable[[np.ndarray], str] | None = None,
        near: dict[str, str] | None = None,
    ):
        self.delay = delay
        self.text_fn = text_fn or (lambda w: str(int(w[0])))
        self.near = near  # set: LM-on engine, returns `fake_lattice(text, samples, near)`
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
        self.warmup_gate = threading.Event()
        self.warmup_gate.set()

    def load(self) -> None:
        self.load_gate.wait(10)
        if self.load_error:
            raise self.load_error

    def warmup(self) -> None:
        self.warmups += 1
        self.warmup_gate.wait(10)
        if self.warmup_error:
            raise self.warmup_error

    def pieces(self) -> tuple[str, ...]:
        return PIECES

    def text(self, units) -> str:
        return "".join(PIECES[u] for u in units).replace("\u2581", " ").strip()

    def transcribe(self, wav: np.ndarray) -> ChunkResult:
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
            if self.near is not None:
                return fake_lattice(self.text_fn(wav), len(wav), self.near)
            return fake_result(self.text_fn(wav), len(wav))
        finally:
            with self.lock:
                self.active -= 1
