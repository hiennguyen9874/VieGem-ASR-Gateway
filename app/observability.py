"""Structured JSON logs and in-process Prometheus-style metrics.

Only metadata is logged or measured: ids, sizes, durations, states. Never pass audio, transcripts, API keys,
tokens or client filenames to `log()` or as metric labels.
"""

import bisect
import contextvars
import json
import logging
import sys
import threading
from collections.abc import Callable

request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
logger = logging.getLogger("quan")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        rid = request_id.get()
        if rid:
            entry["request_id"] = rid
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def log(level: int, event: str, exc_info: bool = False, **fields: object) -> None:
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


Labels = tuple[tuple[str, str], ...]
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class Metrics:
    """Thread-safe counters, gauges and histograms rendered in Prometheus text format (0.0.4)."""

    def __init__(self, prefix: str = "quan_"):
        self._prefix = prefix
        self._lock = threading.Lock()
        self._help: dict[str, tuple[str, str]] = {}  # name -> (type, help)
        self._values: dict[str, dict[Labels, float]] = {}
        self._hists: dict[str, dict[Labels, list]] = {}  # labels -> [bucket counts, sum, count]
        self._gauge_fns: dict[str, Callable[[], float]] = {}

    def _declare(self, name: str, type_: str, help_: str) -> None:
        if self._help.setdefault(name, (type_, help_))[0] != type_:
            raise ValueError(f"metric {name} redeclared as {type_}")

    def inc(self, name: str, help_: str, amount: float = 1.0, **labels: str) -> None:
        self._declare(name, "counter", help_)
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._values.setdefault(name, {})
            series[key] = series.get(key, 0.0) + amount

    def set(self, name: str, help_: str, value: float, **labels: str) -> None:
        self._declare(name, "gauge", help_)
        with self._lock:
            self._values.setdefault(name, {})[tuple(sorted(labels.items()))] = float(value)

    def gauge_fn(self, name: str, help_: str, fn: Callable[[], float]) -> None:
        """Gauge sampled at render time (occupancy, readiness)."""
        self._declare(name, "gauge", help_)
        self._gauge_fns[name] = fn

    def observe(self, name: str, help_: str, value: float, **labels: str) -> None:
        self._declare(name, "histogram", help_)
        key = tuple(sorted(labels.items()))
        with self._lock:
            h = self._hists.setdefault(name, {}).setdefault(key, [[0] * len(DEFAULT_BUCKETS), 0.0, 0])
            i = bisect.bisect_left(DEFAULT_BUCKETS, value)
            if i < len(DEFAULT_BUCKETS):
                h[0][i] += 1
            h[1] += value
            h[2] += 1

    def value(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._values.get(name, {}).get(tuple(sorted(labels.items())), 0.0)

    def render(self) -> str:
        out = []
        sampled = {name: float(fn()) for name, fn in self._gauge_fns.items()}
        with self._lock:
            for name in sorted(self._help):
                type_, help_ = self._help[name]
                full = self._prefix + name
                out += [f"# HELP {full} {help_}", f"# TYPE {full} {type_}"]
                if name in sampled:
                    out.append(f"{full} {sampled[name]}")
                for key, v in sorted(self._values.get(name, {}).items()):
                    out.append(f"{full}{_labels(key)} {v}")
                for key, (buckets, total, count) in sorted(self._hists.get(name, {}).items()):
                    cumulative = 0
                    for bound, n in zip(DEFAULT_BUCKETS, buckets):
                        cumulative += n
                        out.append(f"{full}_bucket{_labels(key + (('le', str(bound)),))} {cumulative}")
                    out.append(f"{full}_bucket{_labels(key + (('le', '+Inf'),))} {count}")
                    out.append(f"{full}_sum{_labels(key)} {total}")
                    out.append(f"{full}_count{_labels(key)} {count}")
        return "\n".join(out) + "\n"


def _labels(key: Labels) -> str:
    if not key:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in key) + "}"
