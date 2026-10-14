"""Scoring of the R2b word-timestamp gate (scripts/eval_alignment.py, docs/m4-design.vi.md §7), CPU only.

Failure modes: ASR errors counted as timing errors (or timing computed on mismatched words), a systematic offset or
a few outliers passing, too little data passing, invalid or non-monotonic spans passing.
"""

import pytest

from scripts.eval_alignment import match, score


def utterance(n=20, offset=0.0, start=1.0):
    ref = [{"word": f"w{i}", "start": start + i * 0.3, "end": start + i * 0.3 + 0.2} for i in range(n)]
    hyp = [{"word": w["word"], "start": round(w["start"] + offset, 3), "end": round(w["end"] + offset, 3)}
           for w in ref]
    return ref, hyp, start + n * 0.3 + 1


def test_exact_timestamps_pass():
    got = score([utterance() for _ in range(10)])
    assert got["gate_passed"] and got["matched_words"] == 200 and got["start_median_abs_ms"] == 0


def test_too_little_data_never_passes():
    got = score([utterance() for _ in range(9)])
    assert not got["enough_data"] and not got["gate_passed"]
    got = score([utterance(n=19) for _ in range(10)])
    assert not got["gate_passed"]


def test_systematic_offset_fails():
    assert score([utterance(offset=0.04) for _ in range(10)])["gate_passed"]
    got = score([utterance(offset=0.12) for _ in range(10)])
    assert got["start_within_100ms"] == 0 and not got["gate_passed"]


def test_outliers_over_ten_percent_fail():
    utts = [utterance() for _ in range(10)]
    for ref, hyp, _ in utts[:2]:  # 2 of 10 utterances fully off: 80% within
        for w in hyp:
            w["start"] += 0.3
            w["end"] += 0.3
    assert not score(utts)["gate_passed"]


def test_asr_errors_are_reported_not_timed():
    utts = [utterance() for _ in range(10)]
    ref, hyp, _ = utts[0]
    hyp[3] = {**hyp[3], "word": "wrong"}  # a substitution: not timed, reported as unmatched
    got = score(utts)
    assert got["matched_words"] == 199 and got["unmatched_reference_share"] == 0.005
    assert got["start_within_100ms"] == 1.0 and got["gate_passed"]


def test_repeated_words_use_minimum_edit_alignment():
    ref = [{"word": w} for w in "a b c a b".split()]
    hyp = [{"word": w} for w in "a b x a b c".split()]
    pairs = match(ref, hyp)
    assert pairs == [(ref[0], hyp[0]), (ref[1], hyp[1]), (ref[3], hyp[3]), (ref[4], hyp[4])]
    assert match(ref, []) == []
    assert match([], hyp) == []
    assert match(ref, ref) == list(zip(ref, ref))


@pytest.mark.parametrize("offset, passes", [(0.05, True), (0.05004, False)])
def test_median_gate_uses_unrounded_error(offset, passes):
    ref = [{"word": f"w{i}", "start": 0.0, "end": 0.25} for i in range(20)]
    hyp = [{**w, "start": offset, "end": 0.25 + offset} for w in ref]
    got = score([(ref, hyp, 1.0)] * 10)
    assert got["start_median_abs_ms"] == 50.0
    assert got["gate_passed"] is passes


def test_percentage_gate_uses_unrounded_share():
    ref = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.25} for i in range(10009)]
    hyp = [{**w, "start": w["start"] + (0.125 if i < 1001 else 0),
            "end": w["end"] + (0.125 if i < 1001 else 0)} for i, w in enumerate(ref)]
    # Keep each utterance short: scoring aggregates the denominator across utterances.
    utterances = [(ref[i:i + 20], hyp[i:i + 20], 6000.0) for i in range(0, len(ref), 20)]
    got = score(utterances)
    assert got["start_within_100ms"] == 0.9
    assert got["all_spans_valid_and_monotonic"]
    assert not got["gate_passed"]


def test_invalid_spans_fail():
    utts = [utterance() for _ in range(10)]
    utts[0][1][5]["end"] = utts[0][1][5]["start"]  # empty span
    assert not score(utts)["gate_passed"]
    utts = [utterance() for _ in range(10)]
    hyp = utts[1][1]
    hyp[2], hyp[3] = {**hyp[2], "start": hyp[3]["start"]}, {**hyp[3], "start": hyp[2]["start"]}  # backwards
    assert not score(utts)["all_spans_valid_and_monotonic"]
