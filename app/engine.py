"""The only code that calls the Model (VieGem-ASR), and the single thread allowed to run it.

`CTCModel._acts` is mutable hook state, so every load/warm-up/inference on the model instance runs on one dedicated
worker thread (`EngineWorker`). Endpoints and sessions never touch the model; they go through `app.scheduler`, which
hands the worker at most one job at a time.

Cancellation is soft: a call that has started always runs to completion (a CUDA kernel cannot be interrupted safely);
the caller discards the result. A call running longer than `hang_seconds` marks the worker hung: `/health` then fails
so the supervisor restarts the process.
"""

import asyncio
import concurrent.futures
import enum
import logging
import queue
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, TYPE_CHECKING

import numpy as np

from app.config import PRECISIONS, SAMPLE_RATE
from app.errors import inference_failed, model_not_ready
from app.observability import log, Metrics

if TYPE_CHECKING:
    from app.lm import Lattice  # app.lm imports this module

# The log-mel frontend crashes below 161 samples (docs/baseline.vi.md §7). Audio policy for short clips (return ""
# without inference) belongs to the callers; this is the hard safety floor.
MIN_FORWARD_SAMPLES = 161
BLANK = 0  # CTC unit ids: 0 = blank, 1 = <unk>, 2.. = Gemma tokens


class ChunkResult(NamedTuple):
    """CTC result of one chunk: the text and the path it was decoded from (no tensors). Greedy: the best path; with
    the LM (`app.lm`): the forced alignment of the LM units on the same log-probs.

    `units` is the collapsed path without blanks (<unk> kept, so it matches the path); unit i was emitted on encoder
    frames `first[i]..last[i]` (inclusive, relative to the chunk). `frames` is the number of valid encoder frames.
    All-zero chunks have empty text/path by policy, retaining the measured frame count.
    """

    text: str
    units: tuple[int, ...]
    first: tuple[int, ...]
    last: tuple[int, ...]
    frames: int


def ctc_path(best: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Collapse an argmax frame sequence like `app.viegem.model.greedy_decode`, keeping each unit's frames."""
    best = np.asarray(best)
    if best.size == 0:
        return (), (), ()
    starts = np.flatnonzero(np.concatenate(([True], best[1:] != best[:-1])))
    ends = np.concatenate((starts[1:], [len(best)])) - 1
    keep = best[starts] != BLANK
    return (
        tuple(best[starts][keep].tolist()),
        tuple(starts[keep].tolist()),
        tuple(ends[keep].tolist()),
    )


def force_align(
    log_probs: np.ndarray, units: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Best CTC path (Viterbi) of `log_probs` (T, V) that spells `units`, in the shape of `ctc_path`.

    Gives LM units the frames of the same inference (ADR 0002). Forcing the greedy units preserves their sequence,
    with monotonic, in-range frames; tied best paths may have different timestamps from the greedy path.
    Unnormalised logits give the same path as their log-softmax (the per-frame normaliser is shared by every path).
    -inf entries (pruned candidates) are allowed. Raises `ValueError`
    on malformed input or when no finite path exists (too many units for T, or only through -inf entries).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError("log_probs must be a 2-D (frames, vocab) array without NaN or +inf")
    frames, vocab = lp.shape
    if vocab == 0:
        raise ValueError("log_probs vocabulary must include the blank column")
    if not all(isinstance(u, int | np.integer) and not isinstance(u, bool) for u in units):
        raise ValueError("units must be integers")
    if any(not BLANK < u < vocab for u in units):
        raise ValueError(f"units must be non-blank ids below {vocab}")
    n = len(units)
    if n == 0 and frames == 0:
        return (), (), ()
    if frames == 0:
        raise ValueError("no CTC path: no frames")

    # Extended states: blank, u1, blank, u2, ..., blank. State s comes from s (stay), s-1, or s-2 (skip a blank
    # between two different units).
    ext = np.full(2 * n + 1, BLANK, dtype=np.int64)
    ext[1::2] = units
    states = len(ext)
    emit = lp[:, ext]
    can_skip = np.zeros(states, dtype=bool)
    can_skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    score = np.full(states, -np.inf)
    score[: min(2, states)] = emit[0, : min(2, states)]
    back = np.zeros((frames, states), dtype=np.int8)  # 0 stay, 1 step, 2 skip
    cand = np.full((3, states), -np.inf)
    for t in range(1, frames):
        cand[0] = score
        cand[1, 1:] = score[:-1]
        cand[2, 2:] = np.where(can_skip[2:], score[:-2], -np.inf)
        back[t] = cand.argmax(0)  # ties prefer stay, then step
        score = cand[back[t], np.arange(states)] + emit[t]

    end = states - 1 if n == 0 or score[-1] >= score[-2] else states - 2
    if not np.isfinite(score[end]):
        raise ValueError(f"no CTC path: {n} units do not fit {frames} frames with finite log-probs")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = end
        end -= int(back[t, end])

    # Each unit state is visited in one contiguous run, in order.
    on_unit = np.flatnonzero(path % 2 == 1)
    idx = path[on_unit] // 2
    order = np.arange(n)
    first = on_unit[np.searchsorted(idx, order, side="left")]
    last = on_unit[np.searchsorted(idx, order, side="right") - 1]
    return tuple(int(u) for u in units), tuple(first.tolist()), tuple(last.tolist())


class Engine(Protocol):
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(self, wav: np.ndarray) -> "ChunkResult | Lattice":
        """Greedy `ChunkResult`; with the LM on, the pruned `app.lm.Lattice` instead (all-zero chunks stay an empty
        `ChunkResult`)."""
        ...

    def pieces(self) -> tuple[str, ...]:
        """Unit id -> token piece ("" for blank and <unk>); called once after `load`."""
        ...

    def text(self, units: Sequence[int]) -> str:
        """Transcript of a unit sequence (tokenizer only, no Model call); used for LM units."""
        ...


class VieGemEngine:
    """Full-context decoding of one 16 kHz mono chunk: greedy, or with `lm=True` (`VIEGEM_LM`) the pruned
    log-probs for the LM pool (`app.lm`, ADR 0002). Beam search never runs here.

    `precision` (`VIEGEM_PRECISION`): "fp32" (default, lowest fixture WER) or "bf16" CUDA autocast; see
    `app.config.PRECISIONS` and docs/baseline.vi.md §8.
    """

    def __init__(
        self,
        models_dir: Path,
        device: str,
        warmup_seconds: float,
        precision: str = "fp32",
        lm: bool = False,
    ):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self._models_dir, self._warmup_seconds = models_dir, warmup_seconds
        self._device, self._precision, self._lm = device, precision, lm
        self._model = self._tokenizer = self._units = None

    def load(self) -> None:
        import torch

        from app.viegem_loader import load_viegem

        self._device = torch.device(self._device)
        if self._precision == "bf16" and self._device.type != "cuda":
            raise ValueError("bf16 precision needs a CUDA device (autocast)")
        self._model, self._tokenizer, self._units = load_viegem(self._models_dir, self._device)

    def warmup(self) -> None:
        """Also used to recover after a failed call. Running the longest allowed chunk reserves its peak memory."""
        import torch

        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self.transcribe(np.zeros(int(self._warmup_seconds * SAMPLE_RATE), dtype=np.float32))

    def forward(self, wav: np.ndarray):
        """Logits (1, T, V) and valid frame counts (1,) of one chunk, as in the model card's usage."""
        import torch

        x = torch.from_numpy(wav).unsqueeze(0).to(self._device)
        lens = torch.tensor([len(wav)], device=self._device)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._precision == "bf16"),
        ):
            return self._model(x, lens, streaming=False)

    def transcribe(self, wav: np.ndarray) -> "ChunkResult | Lattice":
        # Chunking is the caller's job; the greedy best path is kept for timestamps and the text is decoded from it.
        logits, lens = self.forward(wav)
        frames = int(lens[0])
        # The model hallucinates on digital zeros. Do not threshold quiet speech; keep warm-up real.
        if not np.any(wav):
            return ChunkResult("", (), (), (), frames)
        if self._lm:
            return self._lattice(logits[0, :frames])
        units, first, last = ctc_path(logits[0, :frames].argmax(-1).cpu().numpy())
        return ChunkResult(self.text(units), units, first, last, frames)

    def _lattice(self, logits) -> "Lattice":
        """Beam candidates of one chunk, pruned on the device so only they are copied (see `app.lm.prune`)."""
        import torch

        from app.lm import lattice_from_mask, TOKEN_MIN_LOGP

        lp = torch.log_softmax(logits.float(), -1)
        keep = lp >= TOKEN_MIN_LOGP
        keep[torch.arange(len(lp), device=lp.device), lp.argmax(-1)] = True
        rows, ids = keep.nonzero(as_tuple=True)
        return lattice_from_mask(
            rows.cpu().numpy(), ids.cpu().numpy(), lp[rows, ids].cpu().numpy(), len(lp)
        )

    def pieces(self) -> tuple[str, ...]:
        return ("", "") + tuple(self._tokenizer.convert_ids_to_tokens(self._units.gemma_ids))

    def text(self, units: Sequence[int]) -> str:
        return self._units.decode(self._tokenizer, list(units))


def check_waveform(wav: np.ndarray) -> None:
    if not isinstance(wav, np.ndarray) or wav.dtype != np.float32 or wav.ndim != 1:
        raise ValueError("waveform must be a 1-D float32 numpy array (16 kHz mono)")
    if len(wav) < MIN_FORWARD_SAMPLES:
        raise ValueError(
            f"waveform shorter than {MIN_FORWARD_SAMPLES} samples must not reach the model"
        )
    if not np.isfinite(wav).all():
        raise ValueError("waveform contains non-finite samples")


class EngineState(enum.StrEnum):
    LOADING = "loading"
    READY = "ready"
    RECOVERING = "recovering"
    FAILED = "failed"
    STOPPED = "stopped"


class EngineWorker:
    def __init__(self, engine: Engine, hang_seconds: float, metrics: Metrics):
        self._engine, self._hang_seconds, self._metrics = engine, hang_seconds, metrics
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._main, name="viegem-inference", daemon=True)
        self._state = EngineState.LOADING
        self._call_started: float | None = None
        self.pieces: tuple[str, ...] = ()  # set before the worker becomes ready
        metrics.gauge_fn(
            "ready",
            "1 when the model is loaded, warmed up and the worker is not hung.",
            lambda: float(self.ready),
        )
        metrics.gauge_fn(
            "worker_busy",
            "1 while an inference call is running.",
            lambda: float(self._call_started is not None),
        )
        metrics.gauge_fn(
            "worker_hung",
            "1 when the running call exceeded VIEGEM_INFER_HANG_SECONDS.",
            lambda: float(self.hung),
        )

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def hung(self) -> bool:
        started = self._call_started
        return started is not None and time.monotonic() - started > self._hang_seconds

    @property
    def ready(self) -> bool:
        return self._state == EngineState.READY and not self.hung

    @property
    def healthy(self) -> bool:
        """Liveness: false when the process needs a restart (load/recovery failed or a call hangs)."""
        return self._state not in (EngineState.FAILED, EngineState.STOPPED) and not self.hung

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float) -> bool:
        """Ask the thread to exit after its current call; False if it is still running after `timeout`."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    def text(self, units: Sequence[int]) -> str:
        """Transcript of LM units; the tokenizer is read-only, so this runs on the caller's thread."""
        return self._engine.text(units)

    async def transcribe(self, wav: np.ndarray) -> "ChunkResult | Lattice":
        check_waveform(wav)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((wav, fut))
        try:
            return await asyncio.wrap_future(fut)
        finally:
            wav = fut = None  # exception tracebacks must not keep completed audio alive

    def _set_state(self, state: EngineState) -> None:
        if state != self._state:
            self._state = state
            log(logging.INFO, "engine.state", state=state.value)

    def _main(self) -> None:
        t0 = time.monotonic()
        try:
            self._engine.load()
            self.pieces = tuple(self._engine.pieces())
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.load_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="load"
            )
            self._set_state(EngineState.FAILED)
        else:
            log(logging.INFO, "engine.loaded", seconds=round(time.monotonic() - t0, 3))
            self._set_state(EngineState.READY)

        while (item := self._queue.get()) is not None:
            wav, fut = item
            if not fut.set_running_or_notify_cancel():
                wav = item = fut = None
                continue
            if self._state != EngineState.READY:
                fut.set_exception(model_not_ready())
                wav = item = fut = None
                continue
            self._call_started = started = time.monotonic()
            try:
                result = self._engine.transcribe(wav)
            except Exception as e:
                self._call_started = None
                log(logging.ERROR, "engine.inference_failed", exc_info=True, samples=len(wav))
                self._metrics.inc(
                    "worker_failures_total", "Load, inference or recovery failures.", stage="infer"
                )
                err = inference_failed()
                err.__cause__ = e.with_traceback(
                    None
                )  # already logged; the traceback can retain audio
                fut.set_exception(err)
                self._recover()
            else:
                self._call_started = None
                self._metrics.observe(
                    "infer_seconds", "Duration of one inference call.", time.monotonic() - started
                )
                fut.set_result(result)
            finally:
                wav = item = fut = err = (
                    None  # idle worker must not pin completed audio or failure tracebacks
                )
        self._set_state(EngineState.STOPPED)

    def _warmup(self) -> None:
        self._call_started = time.monotonic()
        try:
            self._engine.warmup()
        finally:
            self._call_started = None

    def _recover(self) -> None:
        """Re-run warm-up on this thread; queued calls wait. If it fails the model stays unavailable."""
        self._set_state(EngineState.RECOVERING)
        try:
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.recovery_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="recover"
            )
            self._set_state(EngineState.FAILED)
        else:
            self._set_state(EngineState.READY)
