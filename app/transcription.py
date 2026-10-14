"""OpenAI-compatible models and transcription endpoints (docs/api-contract.vi.md §3–5, docs/m3-design.vi.md).

POST /v1/audio/transcriptions, one scheduler lane per request:
  pre-check (draining / not ready) -> receive multipart (spool) -> parse_params -> open lane -> decode -> drop spool
  -> hold decoded audio (RAM budget) -> split -> run chunks (pipelined) -> json | text | SSE

- A raw ASGI handler, not `StreamingResponse`: once the body is read, a watcher cancels the work when the client
  disconnects (decode, queue wait, inference, streaming), and every send is bounded by `send_timeout_seconds`; a
  client that stops reading counts as disconnected.
- SSE headers are sent only after decoding succeeded. A later error is one `error` event and never a `done`.
- The lane is closed on every exit path: queued chunks are cancelled, a running chunk's result is discarded.
- `Joiner` is the only place chunk texts are joined, for every format.
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Coroutine, Sequence
from contextlib import aclosing
from typing import Any, NamedTuple, TypeVar

import numpy as np
from fastapi import FastAPI
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.types import Message, Receive, Scope, Send

from app.audio import AudioIngress
from app.chunking import Span, split
from app.config import SAMPLE_RATE, Config
from app.engine import EngineWorker
from app.errors import (APIError, internal_error, invalid_request, model_not_found, model_not_ready,
                        request_timeout, shutting_down)
from app.observability import Metrics, log
from app.scheduler import JobKind, Lane, Scheduler

MODEL_ID = "quan-transcribe-1.0"
WHISPER_ALIAS = "whisper-1"  # opt-in (QUAN_WHISPER_ALIAS); same QUAN backend, not Whisper output
MODEL_CREATED = 1791950390  # commit time of the pinned weights revision (docs/api-contract.vi.md §3)
MODEL_OWNER = "qnguyen3"

SINGLE_FIELDS = frozenset({"model", "language", "temperature", "prompt", "response_format", "stream"})
REPEATED_FIELDS = frozenset({"include[]", "timestamp_granularities[]"})
RESPONSE_FORMATS = ("json", "text")
MAX_PARAM_CHARS = 64

SSE_HEADERS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"x-accel-buffering", b"no"),
]

T = TypeVar("T")


# -- models -------------------------------------------------------------------------------------------------------

def served_models(config: Config) -> tuple[str, ...]:
    return (MODEL_ID, WHISPER_ALIAS) if config.whisper_alias else (MODEL_ID,)


def model_object(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": MODEL_CREATED, "owned_by": MODEL_OWNER}


# -- request parameters -------------------------------------------------------------------------------------------

class Params(NamedTuple):
    model: str
    response_format: str  # "json" | "text"
    stream: bool


def _unsupported(param: str) -> APIError:
    shown = param if param.isprintable() else "<invalid>"
    return invalid_request("unsupported_parameter", f"Unsupported parameter: '{shown}'.", param)


def parse_params(fields: Sequence[tuple[str, str]], config: Config) -> Params:
    """Validate the form fields (all but `file`) in a fixed order and return on the first error."""
    names = [name for name, _ in fields]
    for name in names:
        if name not in SINGLE_FIELDS and name not in REPEATED_FIELDS:
            raise _unsupported(name[:MAX_PARAM_CHARS])
    values: dict[str, str] = {}
    for name, value in fields:
        if name in SINGLE_FIELDS:
            if name in values:
                raise invalid_request("invalid_value", f"Parameter '{name}' must be sent once.", name)
            values[name] = value

    model = values.get("model", "")
    if not model:
        raise invalid_request("missing_required_parameter", "Missing required parameter: 'model'.", "model")
    for name in ("include[]", "timestamp_granularities[]"):
        if name in names:
            raise _unsupported(name)
    if "language" in values and values["language"] != "vi":
        raise invalid_request("unsupported_language", "Only language 'vi' is supported.", "language")
    if "temperature" in values:
        try:
            temperature = float(values["temperature"])
        except ValueError:
            temperature = math.nan
        if not math.isfinite(temperature):
            raise invalid_request("invalid_value", "Parameter 'temperature' must be a number.", "temperature")
        if temperature != 0:
            raise invalid_request("unsupported_parameter", "Only temperature 0 is supported.", "temperature")
    if values.get("prompt"):
        raise _unsupported("prompt")
    response_format = values.get("response_format", "json")
    if response_format not in RESPONSE_FORMATS:
        raise invalid_request("unsupported_response_format",
                              f"Supported response formats: {', '.join(RESPONSE_FORMATS)}.", "response_format")
    stream = values.get("stream", "false")
    if stream not in ("true", "false"):
        raise invalid_request("invalid_value", "Parameter 'stream' must be 'true' or 'false'.", "stream")
    # A served-model check comes last so a malformed request gets its 400 first.
    if model not in served_models(config):
        raise model_not_found(model)
    return Params(model, response_format, stream == "true")


# -- chunks -> text -----------------------------------------------------------------------------------------------

class Joiner:
    """Join chunk texts with one space; empty chunks add nothing. `"".join(deltas) == text`."""

    def __init__(self):
        self._parts: list[str] = []

    def add(self, text: str) -> str | None:
        """The delta for this chunk, or None when it adds nothing."""
        if not text:
            return None
        delta = " " + text if self._parts else text
        self._parts.append(delta)
        return delta

    @property
    def text(self) -> str:
        return "".join(self._parts)


async def _until(deadline: float, aw: Awaitable[T]) -> T:
    if time.monotonic() >= deadline:
        if isinstance(aw, Coroutine):
            aw.close()
        raise request_timeout()
    try:
        async with asyncio.timeout(deadline - time.monotonic()):
            return await aw
    except TimeoutError:
        raise request_timeout() from None


async def run_chunks(lane: Lane, wav: np.ndarray, spans: Sequence[Span], deadline: float,
                     window: int) -> AsyncIterator[str]:
    """Yield chunk texts in audio order; at most `window` jobs in flight.

    Admission runs alongside result delivery: a full queue must not delay an already-finished chunk. The
    pending submission counts toward the window. Each wait is deadline-bound (no timeout spans a yield).
    """
    todo = deque(spans)
    jobs = deque()
    admission = None

    def submit_next():
        s = todo.popleft()
        return asyncio.create_task(_until(deadline, lane.submit(wav[s.start:s.end], deadline)))

    try:
        while todo or jobs or admission is not None:
            if admission is None and todo and len(jobs) < window:
                admission = submit_next()
            if jobs and jobs[0].future.done():
                text = await _until(deadline, jobs.popleft())
                if admission is None and todo:
                    admission = submit_next()
                yield text
                continue
            if admission is not None and admission.done():
                jobs.append(admission.result())
                admission = None
                continue
            waiters = ([admission] if admission is not None else [])
            if jobs:
                waiters.append(jobs[0].future)
            await _until(deadline, asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED))
    finally:
        if admission is not None:
            admission.cancel()
            try:
                job = await admission
            except (asyncio.CancelledError, APIError):
                pass
            else:
                job.cancel()
        for job in jobs:
            job.cancel()
        wav = None  # failed job tracebacks must not retain the decoded backing array


# -- ASGI plumbing ------------------------------------------------------------------------------------------------

class Disconnected(Exception):
    """The client went away before the response was complete."""


class SendTimeout(Exception):
    """A send did not finish within `send_timeout_seconds` (the client stopped reading)."""


class DisconnectWatch:
    """Cancel work when the client disconnects. Start only after the request body has been read: from then on
    `receive()` can only return `http.disconnect`."""

    def __init__(self, receive: Receive):
        self._task = asyncio.create_task(self._wait(receive), name="quan-disconnect-watch")

    @staticmethod
    async def _wait(receive: Receive) -> None:
        while (await receive())["type"] != "http.disconnect":
            pass

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run `coro` as a task; raise `Disconnected` (after it has finished cleaning up) if the client leaves."""
        work = asyncio.ensure_future(coro)
        stopped = False
        try:
            await asyncio.wait((work, self._task), return_when=asyncio.FIRST_COMPLETED)
        finally:  # also when this task itself is cancelled
            if not work.done():
                stopped = True
                work.cancel()
                await asyncio.wait((work,))  # FFmpeg reaped, jobs cancelled
                if not work.cancelled():
                    work.exception()  # retrieved: the client is gone, nobody gets it
        if stopped:
            raise Disconnected
        return work.result()

    def close(self) -> None:
        self._task.cancel()


def _timed(send: Send, timeout: float) -> Send:
    async def timed(message: Message) -> None:
        try:
            async with asyncio.timeout(timeout):
                await send(message)
        except TimeoutError:
            raise SendTimeout from None

    return timed


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


class _Output:
    """Response writer of one request: an HTTP response, or SSE events once the stream has started."""

    def __init__(self, scope: Scope, receive: Receive, send: Send):
        self._scope, self._receive, self.send = scope, receive, send
        self.streaming = False

    async def respond(self, response: Response) -> None:
        await response(self._scope, self._receive, self.send)

    async def start_stream(self) -> None:
        self.streaming = True
        await self.send({"type": "http.response.start", "status": 200, "headers": list(SSE_HEADERS)})

    async def event(self, payload: dict, last: bool = False) -> None:
        await self.send({"type": "http.response.body", "body": _sse(payload), "more_body": not last})

    async def error(self, error: APIError) -> None:
        """HTTP error before the stream started; one `error` event (and the end of the stream) after."""
        try:
            if self.streaming:
                await self.event(error.body(), last=True)
            else:
                await self.respond(error.response())
        except SendTimeout:
            pass  # the client stopped reading; Uvicorn closes the connection


# -- endpoint -----------------------------------------------------------------------------------------------------

class TranscriptionEndpoint:
    """ASGI app for POST /v1/audio/transcriptions (routed by method in `install`)."""

    def __init__(self, config: Config, audio: AudioIngress, scheduler: Scheduler, worker: EngineWorker,
                 metrics: Metrics):
        self._config, self._audio, self._scheduler, self._worker = config, audio, scheduler, worker
        self._metrics = metrics
        self._window = config.queue_max_jobs_per_lane

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = time.monotonic()
        out = _Output(scope, receive, _timed(send, self._config.send_timeout_seconds))
        params: Params | None = None
        audio_s: float | None = None
        chunks: int | None = None
        outcome = "error"
        lane: Lane | None = None
        watch: DisconnectWatch | None = None
        release_audio = None
        wav = None
        try:
            # Refuse before reading a 25 MB body; the lane re-checks both below.
            if self._scheduler.closing:
                raise shutting_down()
            if not self._worker.ready:
                raise model_not_ready()
            deadline = started + self._config.request_timeout_seconds
            async with self._audio.receive(Request(scope, receive), deadline) as upload:
                params = parse_params(upload.fields, self._config)
                lane = self._scheduler.open_lane(JobKind.FILE)  # before FFmpeg: a full server costs no CPU
                watch = DisconnectWatch(receive)
                wav = await watch.run(self._audio.decode(upload.path, deadline))
            # The spool file is gone here; only the waveform is kept while waiting for the GPU.
            audio_s = len(wav) / SAMPLE_RATE
            self._metrics.inc("transcription_audio_seconds_total", "Decoded audio accepted for transcription.",
                              amount=audio_s)
            release_audio = self._audio.hold(wav)
            spans = split(wav, self._config.max_chunk_samples)
            chunks = len(spans)
            texts = run_chunks(lane, wav, spans, deadline, self._window)
            if params.stream:
                await _until(deadline, watch.run(self._stream(out, texts, started)))
            else:
                text = await _until(deadline, watch.run(_collect(texts)))
                await _until(deadline, out.respond(_final(params, text, audio_s)))
            outcome = "ok"
        except APIError as e:
            outcome = e.code
            await out.error(e)
        except (Disconnected, ClientDisconnect):
            outcome = "disconnected"  # nothing more is sent
        except SendTimeout:
            outcome = "send_timeout"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "internal_error"
            log(logging.ERROR, "transcription.unhandled_error", exc_info=True)
            await out.error(internal_error())
        finally:
            wav = None  # do not keep decoded RAM alive through an exception traceback
            if lane is not None:
                lane.close()
                if release_audio is not None:
                    lane.release_audio_when_idle(release_audio)
            if watch is not None:
                watch.close()
            fmt = params.response_format if params else "unknown"
            stream = str(params.stream).lower() if params else "unknown"
            self._metrics.inc("transcriptions_total", "Transcription requests by outcome.", response_format=fmt,
                              stream=stream, outcome=outcome)
            # Metadata only: no transcript, filename or field values.
            log(logging.INFO, "transcription.request", response_format=fmt, stream=stream,
                audio_s=None if audio_s is None else round(audio_s, 3), chunks=chunks, outcome=outcome,
                duration_ms=round((time.monotonic() - started) * 1000, 1))

    async def _stream(self, out: _Output, texts: AsyncIterator[str], started: float) -> None:
        joiner = Joiner()
        first = True

        async def event(payload: dict, last: bool = False) -> None:
            nonlocal first
            await out.event(payload, last)
            if first:
                first = False
                self._metrics.observe("sse_first_event_seconds", "From request start to the first SSE event.",
                                      time.monotonic() - started)

        await out.start_stream()
        async with aclosing(texts):
            async for text in texts:
                if (delta := joiner.add(text)) is not None:
                    await event({"type": "transcript.text.delta", "delta": delta})
        await event({"type": "transcript.text.done", "text": joiner.text}, last=True)


async def _collect(texts: AsyncIterator[str]) -> str:
    joiner = Joiner()
    async with aclosing(texts):
        async for text in texts:
            joiner.add(text)
    return joiner.text


def _final(params: Params, text: str, audio_s: float) -> Response:
    if params.response_format == "text":
        return Response(text, media_type="text/plain; charset=utf-8")
    return JSONResponse({"text": text, "usage": {"type": "duration", "seconds": audio_s}})


def install(app: FastAPI, config: Config, audio: AudioIngress, scheduler: Scheduler, worker: EngineWorker,
            metrics: Metrics) -> None:
    models = served_models(config)

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [model_object(m) for m in models]}

    @app.get("/v1/models/{model_id:path}")
    async def retrieve_model(model_id: str):
        if model_id not in models:
            raise model_not_found(model_id)
        return model_object(model_id)

    app.router.add_route("/v1/audio/transcriptions", TranscriptionEndpoint(config, audio, scheduler, worker, metrics),
                         methods=["POST"])
