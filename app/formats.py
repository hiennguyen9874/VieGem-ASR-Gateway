"""Serializers for the timed response formats (docs/api-contract.vi.md §4): pure functions over samples.

Times stay integer 16 kHz samples until here. JSON seconds are rounded to 3 decimals; SRT/VTT use whole
milliseconds `round(samples / 16)`. Both roundings are monotonic, so ordered times stay ordered.
"""

from collections.abc import Sequence

from app.alignment import Segment, Word
from app.config import SAMPLE_RATE

LANGUAGE = "vietnamese"  # Whisper verbose_json convention: the full language name


def seconds(samples: int) -> float:
    return round(samples / SAMPLE_RATE, 3)


def verbose_json(text: str, audio_s: float, words: Sequence[Word] | None = None) -> dict:
    """`words` is None when not requested (omitted, never null); an empty list stays `[]`."""
    body: dict = {"text": text, "language": LANGUAGE, "duration": audio_s}
    if words is not None:
        body["words"] = [{"word": w.text, "start": seconds(w.start), "end": seconds(w.end)} for w in words]
    body["usage"] = {"type": "duration", "seconds": audio_s}
    return body


def _clock(samples: int, sep: str) -> str:
    total_ms = round(samples * 1000 / SAMPLE_RATE)
    s, ms = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def srt(segments: Sequence[Segment]) -> str:
    return "".join(f"{i}\n{_clock(c.start, ',')} --> {_clock(c.end, ',')}\n{c.text}\n\n"
                   for i, c in enumerate(segments, 1))


def vtt(segments: Sequence[Segment]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_clock(c.start, '.')} --> {_clock(c.end, '.')}\n{c.text}\n\n"
                                  for c in segments)
