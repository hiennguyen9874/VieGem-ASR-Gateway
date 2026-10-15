"""Word and cue timestamps from the greedy CTC path of each chunk (docs/api-contract.vi.md §4).

The gateway decodes greedily, so the argmax path that produced the text is its alignment: no extra GPU work, no
forced alignment. Times are absolute 16 kHz sample offsets (ints) until serialization.

- A word is a maximal run of non-whitespace characters of the final text (a Vietnamese syllable, not a compound).
  Token pieces start a new word with "▁" and never contain it elsewhere, so each word is a run of units.
- Encoder frame k (40 ms) is centred on chunk sample 640 k and covers [640 k - 320, 640 k + 320), clipped to the
  chunk. A word spans the frames of its units.
- Any inconsistency (the path does not spell the text, frames out of order or outside the chunk) raises
  `AlignmentError`: timestamps are never guessed or spread evenly.
"""

import re
from collections.abc import Sequence
from typing import NamedTuple

from app.chunking import Span
from app.config import SAMPLE_RATE
from app.engine import ChunkResult

SAMPLES_PER_FRAME = 640  # two stride-2 convolutions over 10 ms mel frames
HALF_FRAME = SAMPLES_PER_FRAME // 2
# Constant spike delay compensation in samples. 0: the P0 probe found no constant delay to compensate
# (docs/baseline/alignment_probe.json).
EMISSION_SHIFT = 0
WORD_START = "\u2581"

# Cue (subtitle segment) limits, A6.
PAUSE_SAMPLES = SAMPLE_RATE // 2  # a silence of at least this between two words ends the cue
MAX_CUE_SAMPLES = 7 * SAMPLE_RATE
MAX_CUE_CHARS = 80

_WORD = re.compile(r"\S+")


class Word(NamedTuple):
    text: str
    start: int  # absolute samples at 16 kHz, end exclusive
    end: int


class Segment(NamedTuple):
    text: str
    start: int
    end: int


class AlignmentError(Exception):
    """The CTC path and the text disagree; the response must fail rather than carry made-up times."""


def words(span: Span, result: ChunkResult, pieces: Sequence[str]) -> list[Word]:
    """Words of one chunk with absolute times inside `span`."""
    n = span.end - span.start
    if not (len(result.units) == len(result.first) == len(result.last)):
        raise AlignmentError("path arrays differ in length")
    # Character range of each unit that emits text.
    chars: list[str] = []
    ranges: list[tuple[int, int, int, int]] = []  # (char start, char end, first frame, last frame)
    pos, prev_first, prev_last = 0, 0, 0
    for u, f, l in zip(result.units, result.first, result.last):
        # Real paths are strictly increasing; non-decreasing is all the times below need.
        if not (prev_first <= f <= l < result.frames and prev_last <= l):
            raise AlignmentError("frames out of order or out of range")
        prev_first, prev_last = f, l
        if not 0 <= u < len(pieces):
            raise AlignmentError("unit outside the vocabulary")
        piece = pieces[u].replace(WORD_START, " ")
        if piece:
            chars.append(piece)
            ranges.append((pos, pos + len(piece), f, l))
            pos += len(piece)
    spelled = "".join(chars)
    if spelled.strip() != result.text:
        raise AlignmentError("path does not spell the text")

    out: list[Word] = []
    j = 0
    for m in _WORD.finditer(spelled):
        a, b = m.span()
        while ranges[j][1] <= a:
            j += 1
        first, last, k = ranges[j][2], ranges[j][3], j
        while k < len(ranges) and ranges[k][0] < b:
            last = ranges[k][3]
            k += 1
        start = max(0, SAMPLES_PER_FRAME * first - HALF_FRAME + EMISSION_SHIFT)
        end = min(n, SAMPLES_PER_FRAME * last + HALF_FRAME + EMISSION_SHIFT)
        if not start < end:
            raise AlignmentError("word outside the chunk")
        out.append(Word(m.group(), span.start + start, span.start + end))
    if " ".join(w.text for w in out) != result.text:
        raise AlignmentError("words do not rebuild the text")
    return out


def align(chunks: Sequence[tuple[Span, ChunkResult]], pieces: Sequence[str]) -> list[Word]:
    """Words of a whole transcript, chunks in audio order; their texts joined by one space give the transcript."""
    out: list[Word] = []
    prev_end = 0
    for span, result in chunks:
        if span.start < prev_end:
            raise AlignmentError("chunks overlap or are out of order")
        prev_end = span.end
        out.extend(words(span, result, pieces))
    if any(b.start < a.start for a, b in zip(out, out[1:])):
        raise AlignmentError("word starts go backwards")
    return out


def segments(ws: Sequence[Word]) -> list[Segment]:
    """Group words into subtitle cues: a new cue after a pause of `PAUSE_SAMPLES`, or when the next word would make
    the cue longer than `MAX_CUE_SAMPLES` or `MAX_CUE_CHARS` (a single oversized word is a cue of its own)."""
    out: list[Segment] = []
    cue: list[Word] = []
    for w in ws:
        if cue and (w.start - cue[-1].end >= PAUSE_SAMPLES or w.end - cue[0].start > MAX_CUE_SAMPLES
                    or sum(len(x.text) + 1 for x in cue) + len(w.text) > MAX_CUE_CHARS):
            out.append(_cue(cue))
            cue = []
        cue.append(w)
    if cue:
        out.append(_cue(cue))
    return out


def _cue(ws: list[Word]) -> Segment:
    return Segment(" ".join(w.text for w in ws), ws[0].start, ws[-1].end)
