"""Timestamps from the greedy CTC path (app/alignment.py, docs/api-contract.vi.md §4), CPU only.

Failure modes covered: path extraction differs from the package's `greedy_decode` (repeats, blanks, edges); units
without text (<unk>, a lone "▁") create empty words or shift char ranges; a word gets another word's frames; frames are
not offset by the chunk start or leak outside the chunk; times go backwards across chunks; a path that does not spell
the text is accepted; cues straddle pauses or exceed their limits; cue/word texts do not rebuild the transcript.
"""

import random

import numpy as np
import pytest
import torch
from quan_transcribe.model import greedy_decode

from app.alignment import (MAX_CUE_CHARS, MAX_CUE_SAMPLES, PAUSE_SAMPLES, AlignmentError, Segment, Word, align,
                           segments, words)
from app.chunking import Span
from app.engine import ChunkResult, ctc_path
from app.transcription import Joiner
from tests.fake_engine import PIECES, fake_result

# unit ids: 0 blank, 1 unk
P = ("", "", "\u2581các", "\u2581sản", "\u2581ph", "ẩm", "\u2581", "a", "\u2581a", "\u2581b")
CAC, SAN, PH, AM, SPACE, A, WA, WB = range(2, 10)


def result(text, units, first, last=None, frames=None):
    last = last if last is not None else first
    return ChunkResult(text, tuple(units), tuple(first), tuple(last), frames or (max(last) + 1 if last else 1))


# -- path extraction ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("best,units,first,last", [
    ([], [], [], []),
    ([0, 0, 0], [], [], []),
    ([5], [5], [0], [0]),
    ([5, 5, 0, 5], [5, 5], [0, 3], [1, 3]),  # a ∅ a: two units
    ([5, 5, 5], [5], [0], [2]),  # a a: one unit
    ([0, 1, 1, 7, 0, 0], [1, 7], [1, 3], [2, 3]),  # unk kept
    ([3, 4, 0, 0, 4], [3, 4, 4], [0, 1, 4], [0, 1, 4]),
])
def test_ctc_path_cases(best, units, first, last):
    assert ctc_path(np.array(best, dtype=np.int64)) == (tuple(units), tuple(first), tuple(last))


def test_ctc_path_matches_greedy_decode_on_random_paths():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        best = rng.choice([0, 0, 0, 1, 2, 3], size=n)
        logits = torch.nn.functional.one_hot(torch.from_numpy(best), 6).float()[None]
        units, first, last = ctc_path(best)
        assert list(units) == greedy_decode(logits, torch.tensor([n]))[0]
        assert all(best[f] == u and best[l] == u for u, f, l in zip(units, first, last))
        assert all(f <= l < f2 for f, l, f2 in zip(first, last, first[1:] + (n,)))


# -- words ----------------------------------------------------------------------------------------------------------

def test_words_frames_to_absolute_samples():
    # "các sản phẩm": frame k covers [640k - 320, 640k + 320) of the chunk, offset by the chunk start.
    r = result("các sản phẩm", [CAC, SAN, PH, AM], [2, 5, 9, 11], [3, 5, 9, 12], frames=20)
    got = words(Span(16000, 16000 + 12800), r, P)
    assert got == [Word("các", 16000 + 960, 16000 + 2240), Word("sản", 16000 + 2880, 16000 + 3520),
                   Word("phẩm", 16000 + 5440, 16000 + 8000)]


def test_words_clip_to_the_chunk():
    r = result("a b", [WA, WB], [0, 9], frames=10)
    got = words(Span(1000, 1000 + 6000), r, P)  # frame 9 ends at 6080 > 6000
    assert got == [Word("a", 1000, 1000 + 320), Word("b", 1000 + 5440, 1000 + 6000)]


def test_units_without_text_make_no_words():
    # unk (1) emits nothing; a lone "▁" only separates words, so the next word gets only its own frames.
    r = result("các a", [1, CAC, SPACE, 1, A, 1], [0, 1, 3, 4, 6, 8], frames=10)
    assert words(Span(0, 6400), r, P) == [Word("các", 320, 960), Word("a", 3520, 4160)]
    # A unit without "▁" continues the previous word.
    r = result("các aa", [1, CAC, 1, WA, A, 1], [0, 1, 4, 6, 8, 9], frames=10)
    assert words(Span(0, 6400), r, P) == [Word("các", 320, 960), Word("aa", 3520, 5440)]


def test_repeated_unit_is_two_units():
    r = result("aa", [WA, A], [1, 4], [2, 4], frames=6)
    assert words(Span(0, 3840), r, P) == [Word("aa", 320, 2880)]


def test_empty_text_has_no_words():
    assert words(Span(0, 3200), result("", [], [], frames=5), P) == []
    assert words(Span(0, 3200), result("", [1, SPACE], [0, 2], frames=5), P) == []


@pytest.mark.parametrize("r", [
    result("các", [SAN], [1], frames=5),  # path spells another text
    result("các sản", [CAC], [1], frames=5),
    result("cácsản", [CAC, SAN], [1, 2], frames=5),
    result("các  sản", [CAC, SPACE, SAN], [1, 2, 3], frames=5),  # double space: words cannot rebuild the text
    result("các", [CAC], [5], frames=5),  # frame beyond the valid frames
    result("các", [99], [1], frames=5),  # unit outside the vocabulary
    result("các sản", [CAC, SAN], [3, 1], frames=5),  # frames go backwards
    result("các", [CAC], [3], [2], frames=5),  # last < first
])
def test_inconsistent_results_raise(r):
    with pytest.raises(AlignmentError):
        words(Span(0, 3200), r, P)


def test_frame_past_the_chunk_raises():
    # Frame 2 starts at 640 * 2 - 320 = 960: inside a 1000-sample chunk, past the end of a 900-sample one.
    assert words(Span(0, 1000), result("a", [WA], [2], frames=3), P) == [Word("a", 960, 1000)]
    with pytest.raises(AlignmentError):
        words(Span(0, 900), result("a", [WA], [2], frames=3), P)


def test_random_fake_results_round_trip():
    rnd = random.Random(1)
    letters = "aăâbcdđeêghiklmnoôơpqrstuưvxyàáảãạ0123456789-"
    for _ in range(300):
        text = " ".join("".join(rnd.choice(letters) for _ in range(rnd.randint(1, 6)))
                        for _ in range(rnd.randint(0, 12)))
        n = rnd.randint(161, 16000 * 5)
        start = rnd.randint(0, 10 ** 6)
        got = words(Span(start, start + n), fake_result(text, n), PIECES)
        assert " ".join(w.text for w in got) == text
        assert all(start <= w.start < w.end <= start + n for w in got)
        assert all(a.start <= b.start and a.end <= b.end for a, b in zip(got, got[1:]))


def test_align_concatenates_chunks_in_order():
    spans = [Span(0, 48000), Span(48000, 52000), Span(52000, 100000)]
    texts = ["một hai ba", "", "bốn năm"]
    chunks = [(s, fake_result(t, s.end - s.start)) for s, t in zip(spans, texts)]
    got = align(chunks, PIECES)
    j = Joiner()
    for t in texts:
        j.add(t)
    assert " ".join(w.text for w in got) == j.text == "một hai ba bốn năm"
    assert [w.text for w in got[3:]] == ["bốn", "năm"] and got[3].start >= 52000
    assert all(a.end <= b.start for a, b in zip(got, got[1:]))


def test_align_rejects_overlapping_chunks():
    chunks = [(Span(0, 32000), fake_result("a", 32000)), (Span(16000, 48000), fake_result("b", 32000))]
    with pytest.raises(AlignmentError):
        align(chunks, PIECES)


# -- segments -------------------------------------------------------------------------------------------------------

def w(text, start_s, end_s):
    return Word(text, round(start_s * 16000), round(end_s * 16000))


def test_segments_split_at_pauses():
    pause = PAUSE_SAMPLES / 16000
    ws = [w("a", 0, 0.2), w("b", 0.3, 0.5), w("c", 0.5 + pause, 1.5 + pause),  # gap == PAUSE: split
          w("d", 1.5 + pause + pause - 0.01, 3 + pause * 2)]  # gap just below PAUSE: same cue
    assert segments(ws) == [Segment("a b", 0, 8000), Segment("c d", ws[2].start, ws[3].end)]


def test_segments_respect_max_duration():
    # Words every 1 s, no pauses: a word that would end past MAX_CUE_SAMPLES after the cue start opens a new cue.
    ws = [w(str(i), i, i + 0.9) for i in range(20)]
    got = segments(ws)
    assert all(s.end - s.start <= MAX_CUE_SAMPLES for s in got)
    assert len(got) > 1 and got[0].text == " ".join(str(i) for i in range(7))  # 0..6 ends at 6.9 s; 7 ends at 7.9 s
    assert " ".join(s.text for s in got) == " ".join(x.text for x in ws)


def test_segments_respect_max_chars():
    ws = [w("x" * 9, i * 0.3, i * 0.3 + 0.2) for i in range(20)]
    got = segments(ws)
    assert all(len(s.text) <= MAX_CUE_CHARS for s in got)
    assert got[0].text == " ".join(["x" * 9] * 8)  # 8 words = 79 chars; a 9th would make 89


def test_oversized_word_is_its_own_cue():
    ws = [w("a", 0, 0.1), w("y" * (MAX_CUE_CHARS + 5), 0.2, 9.0), w("b", 9.1, 9.2)]
    assert [s.text for s in segments(ws)] == ["a", "y" * (MAX_CUE_CHARS + 5), "b"]


def test_segments_of_nothing():
    assert segments([]) == []
