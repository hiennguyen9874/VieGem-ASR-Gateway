"""Language-model (LM) decoding off the engine thread (ADR 0002), switched on by `VIEGEM_LM`.

With the LM on, the engine thread only runs the Model and returns a `Lattice`: the log-softmax of each encoder frame
pruned to the beam-search candidates (log-prob >= `TOKEN_MIN_LOGP`, or the argmax when none is). `LMPool`, a bounded
pool of CPU processes, runs the vendored prefix beam search (`app.viegem.lm_decode`) with the model-card constants
and gives the LM units their frames by CTC forced alignment on the same lattice, so timed formats stay aligned to the
LM text (`app.alignment` is unchanged).

`decode_lattice` is the pure function behind the pool; it is testable on CPU with a tiny ARPA model.
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import time
from collections.abc import Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import NamedTuple

import numpy as np

from app.engine import BLANK, force_align
from app.errors import inference_failed, request_timeout, server_busy, shutting_down
from app.observability import log, Metrics

# Model-card decoding constants (VieGem-ASR README); not operator settings.
ALPHA = 0.35
BETA = 1.5
BEAM_WIDTH = 32
UNK_OFFSET = -10.0
TOKEN_MIN_LOGP = -5.0

LM_DIR = Path("viegem-asr/lm")  # under VIEGEM_MODELS_DIR, staged by scripts/stage_models.py
LM_FILE = "viegem_lm.bin"
UNIGRAMS_FILE = "viegem_lm.unigrams.txt"

UnitPath = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # units, first, last (as `ctc_path`)


class Lattice(NamedTuple):
    """Pruned log-probs of one chunk (CSR): frame t's candidates are `ids[offsets[t]:offsets[t + 1]]` (ascending)
    with log-probs `logps[...]`. `frames` is the number of valid encoder frames (`len(offsets) - 1`)."""

    offsets: np.ndarray  # int64 (frames + 1,)
    ids: np.ndarray  # int32
    logps: np.ndarray  # float32
    frames: int


def lattice_from_mask(rows: np.ndarray, ids: np.ndarray, logps: np.ndarray, frames: int) -> Lattice:
    """Build a `Lattice` from kept (row, id) pairs in row-major order (as `nonzero()` returns them)."""
    offsets = np.zeros(frames + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=frames), out=offsets[1:])
    return Lattice(offsets, ids.astype(np.int32), logps.astype(np.float32), frames)


def prune(log_probs: np.ndarray) -> Lattice:
    """The beam candidates of a (T, V) log-softmax matrix: log-prob >= TOKEN_MIN_LOGP, plus each frame's argmax
    (which is a candidate whenever any is, so this equals upstream's argmax fallback)."""
    lp = np.asarray(log_probs, dtype=np.float32)
    keep = lp >= TOKEN_MIN_LOGP
    keep[np.arange(len(lp)), lp.argmax(-1)] = True
    rows, ids = np.nonzero(keep)
    return lattice_from_mask(rows, ids, lp[rows, ids], len(lp))


def frames_of(lattice: Lattice):
    o = lattice.offsets
    return ((lattice.ids[o[t] : o[t + 1]], lattice.logps[o[t] : o[t + 1]]) for t in range(lattice.frames))


def decode_lattice(decoder, lattice: Lattice, pieces: Mapping[int, str] | Sequence[str]) -> UnitPath:
    """LM units of one chunk with their frames: prefix beam search, then forced alignment on the same lattice.

    `decoder` is an `app.viegem.lm_decode.LMDecoder`; `pieces[v]` is the text of every candidate id v. Raises
    `ValueError` if the units have no CTC path through the lattice (cannot happen for the search's own output).
    """
    units = decoder.decode(frames_of(lattice), pieces, blank=BLANK, beam_width=BEAM_WIDTH)
    # Forced alignment needs only the blank and the chosen units: a (T, 1 + k) matrix, -inf where pruned.
    cols = np.unique(np.asarray([BLANK, *units], dtype=np.int64))
    col = np.minimum(np.searchsorted(cols, lattice.ids), len(cols) - 1)
    hit = cols[col] == lattice.ids
    rows = np.repeat(np.arange(lattice.frames), np.diff(lattice.offsets))
    small = np.full((lattice.frames, len(cols)), -np.inf)
    small[rows[hit], col[hit]] = lattice.logps[hit]
    local = [int(i) for i in np.searchsorted(cols, units)]
    _, first, last = force_align(small, local)
    return tuple(int(u) for u in units), first, last


# -- pool process side ----------------------------------------------------------------------------------------------

_decoder = None


def _init(lm_path: str, unigrams_path: str) -> None:
    global _decoder
    from app.viegem.lm_decode import LMDecoder

    words = Path(unigrams_path).read_text(encoding="utf-8").split()
    _decoder = LMDecoder(lm_path, unigrams=words, alpha=ALPHA, beta=BETA, unk_offset=UNK_OFFSET)


def _decode(lattice: Lattice, pieces: dict[int, str]) -> UnitPath:
    return decode_lattice(_decoder, lattice, pieces)


def _ping() -> bool:
    return _decoder is not None


# -- event-loop side ------------------------------------------------------------------------------------------------


class LMPool:
    """Bounded CPU process pool for LM decoding. All methods but the done callbacks run on the event loop.

    At most `max_pending` chunks are queued or running; a decode waits up to `queue_timeout` for a slot, then gets
    429 `server_busy`. A deadline gives 503 `request_timeout`. A started decode cannot be interrupted: a cancelled
    caller stops waiting, but its slot is freed only when the process finishes. A broken pool (a worker process
    died or the LM failed to load) makes `healthy` false so the supervisor restarts the gateway.
    """

    def __init__(
        self,
        models_dir: Path,
        workers: int,
        max_pending: int,
        queue_timeout: float,
        metrics: Metrics,
    ):
        lm_dir = Path(models_dir) / LM_DIR
        self._paths = (str(lm_dir / LM_FILE), str(lm_dir / UNIGRAMS_FILE))
        for p in self._paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"LM file not staged (run scripts/stage_models.py): {p}")
        self._workers, self._queue_timeout, self._metrics = workers, queue_timeout, metrics
        self._max_pending = max_pending
        self._pending = 0
        self._slots: asyncio.Semaphore | None = None
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._warm: asyncio.Task | None = None
        self.ready = False
        self.failed = False
        metrics.gauge_fn(
            "lm_pending", "LM decodes queued or running in the CPU pool.", lambda: self._pending
        )
        metrics.gauge_fn("lm_ready", "1 when the LM pool is loaded.", lambda: float(self.ready))

    @property
    def healthy(self) -> bool:
        return not self.failed

    def start(self) -> None:
        # spawn: the parent runs CUDA and other threads, which fork() must not copy.
        self._executor = concurrent.futures.ProcessPoolExecutor(
            self._workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=self._paths,
        )
        self._slots = asyncio.Semaphore(self._max_pending)
        self._warm = asyncio.get_running_loop().create_task(self._warmup(), name="viegem-lm-warmup")

    async def _warmup(self) -> None:
        t0 = time.monotonic()
        try:
            # One call per worker starts every process (and loads the LM in each) before /ready.
            await asyncio.gather(
                *(asyncio.wrap_future(self._executor.submit(_ping)) for _ in range(self._workers))
            )
        except Exception:
            log(logging.ERROR, "lm.load_failed", exc_info=True)
            self.failed = True
        else:
            log(logging.INFO, "lm.loaded", workers=self._workers, seconds=round(time.monotonic() - t0, 3))
            self.ready = True

    def stop(self) -> None:
        """After scheduler drain, kill remaining work and reap workers within a shared five-second budget."""
        self.ready = False
        if self._executor is not None:
            # Python 3.12 has no public kill_workers(); snapshot before shutdown clears _processes.
            processes = list((self._executor._processes or {}).values())
            for process in processes:
                if process.is_alive():
                    process.kill()
            self._executor.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + 5
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))

    async def decode(self, lattice: Lattice, pieces: Sequence[str], deadline: float | None) -> UnitPath:
        """Units, first and last frames of the LM decode of `lattice`; `pieces` maps every unit id to its text."""
        if self._executor is None or self.failed:
            raise inference_failed()
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._slots.acquire()
        except TimeoutError:
            self._count("rejected")
            raise server_busy(retry_after=1) from None
        loop = asyncio.get_running_loop()
        self._pending += 1
        released = False

        def release(_=None) -> None:
            nonlocal released
            if not released:
                released = True
                self._pending -= 1
                self._slots.release()

        started = time.monotonic()
        try:
            needed = {int(i): pieces[int(i)] for i in np.unique(lattice.ids)}
            fut = self._executor.submit(_decode, lattice, needed)
        except BrokenProcessPool:  # before RuntimeError, its base class
            release()
            self._broken()
            raise inference_failed() from None
        except RuntimeError:  # executor shut down
            release()
            raise shutting_down() from None
        fut.add_done_callback(lambda _: loop.call_soon_threadsafe(release))
        try:
            timeout = None if deadline is None else deadline - time.monotonic()
            async with asyncio.timeout(timeout):
                result = await asyncio.shield(asyncio.wrap_future(fut))
        except TimeoutError:
            fut.cancel()  # only if still queued; a running decode finishes and then frees its slot
            self._count("expired")
            raise request_timeout() from None
        except asyncio.CancelledError:
            fut.cancel()
            self._count("cancelled")
            raise
        except BrokenProcessPool:
            self._broken()
            raise inference_failed() from None
        except Exception:
            log(logging.ERROR, "lm.decode_failed", exc_info=True)
            self._count("failed")
            raise inference_failed() from None
        self._count("ok")
        self._metrics.observe(
            "lm_decode_seconds", "From LM pool submit to result, per chunk.", time.monotonic() - started
        )
        return result

    def _broken(self) -> None:
        if not self.failed:
            log(logging.ERROR, "lm.pool_broken")
        self.failed, self.ready = True, False
        self._count("failed")

    def _count(self, outcome: str) -> None:
        self._metrics.inc("lm_decodes_total", "LM decodes by outcome.", outcome=outcome)
