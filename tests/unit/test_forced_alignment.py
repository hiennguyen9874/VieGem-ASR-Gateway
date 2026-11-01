"""CTC forced alignment (`app.engine.force_align`), CPU only: the times of LM units (ADR 0002).

Failure modes covered: the forced path differs from the unique greedy path when forced with the greedy units (wrong
transitions, edges); tied best paths change units or yield invalid frames; repeated units merge without a blank
between them; frames leave the chunk or go backwards; an empty sequence is mishandled; a sequence too long for the frames, or only reachable through -inf
entries (pruned log-probs), returns a path instead of an error; malformed inputs (blank or out-of-range units, NaN,
wrong shape) are accepted.
"""

import numpy as np
import pytest

from app.engine import ctc_path, force_align

V = 6


def logp(rng, frames, vocab=V):
    x = rng.normal(size=(frames, vocab)) * 3
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def one_hot_path(best, vocab=V):
    lp = np.full((len(best), vocab), -10.0)
    lp[np.arange(len(best)), best] = 0.0
    return lp


def check_path(units, frames, path):
    assert path[0] == tuple(units)
    f, l = path[1], path[2]
    assert len(f) == len(l) == len(units)
    assert all(0 <= a <= b < frames for a, b in zip(f, l))
    assert all(b < a2 for b, a2 in zip(l, f[1:]))  # strictly after the previous unit


def test_forcing_greedy_units_reproduces_greedy_path_on_random_logprobs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        frames = int(rng.integers(1, 80))
        lp = logp(rng, frames)
        lp[:, 0] += rng.uniform(0, 4)  # blank-heavy matrices, like real CTC output
        greedy = ctc_path(lp.argmax(-1))
        assert force_align(lp, greedy[0]) == greedy


def test_forcing_greedy_units_reproduces_greedy_path_on_float32_logits():
    # Unnormalised fp32 logits give the same path: the per-frame normaliser is shared by every path.
    rng = np.random.default_rng(1)
    for _ in range(50):
        logits = rng.normal(size=(int(rng.integers(1, 200)), 40)).astype(np.float32)
        greedy = ctc_path(logits.argmax(-1))
        assert force_align(logits, greedy[0]) == greedy


@pytest.mark.parametrize("best", [
    [5],
    [5, 5, 0, 5],  # repeated unit with a blank between
    [3, 4, 4, 0, 4, 1, 1],  # <unk> kept as a unit
    [0, 0, 2, 0, 0],
    [2, 3, 2, 3],
])
def test_one_hot_paths(best):
    assert force_align(one_hot_path(best), ctc_path(np.array(best))[0]) == ctc_path(np.array(best))


def test_tied_greedy_units_get_an_optimal_path():
    lp = np.log([[0.5, 0.5], [0.25, 0.75]])
    greedy = ctc_path(lp.argmax(-1))
    forced = force_align(lp, greedy[0])
    check_path(greedy[0], len(lp), forced)
    # Both possible start frames have the same score; timestamps need not match greedy.
    assert forced[2] == (1,)
    assert forced[1] in ((0,), (1,))
    assert lp[0, 0] + lp[1, 1] == lp[0, 1] + lp[1, 1]


def test_non_greedy_units_get_a_valid_path():
    # An LM may choose units the argmax did not; they still get monotonic, in-range frames.
    rng = np.random.default_rng(2)
    for _ in range(200):
        frames = int(rng.integers(1, 60))
        n = int(rng.integers(0, frames // 2 + 1))
        units = tuple(int(u) for u in rng.integers(1, V, size=n))
        check_path(units, frames, force_align(logp(rng, frames), units))


def test_repeated_units_need_a_blank_between_them():
    # "a a" needs a blank frame: 3 frames suffice, 2 do not.
    assert force_align(one_hot_path([5, 0, 5]), (5, 5)) == ((5, 5), (0, 2), (0, 2))
    lp = np.log(np.full((3, V), 1 / V))
    check_path((5, 5), 3, force_align(lp, (5, 5)))
    with pytest.raises(ValueError, match="no CTC path"):
        force_align(lp[:2], (5, 5))
    # Distinct units can be adjacent: 2 frames suffice.
    assert force_align(lp[:2], (4, 5)) == ((4, 5), (0, 1), (0, 1))


def test_exact_fit_uses_one_frame_per_unit():
    lp = np.log(np.full((4, V), 1 / V))
    assert force_align(lp, (2, 3, 4, 5)) == ((2, 3, 4, 5), (0, 1, 2, 3), (0, 1, 2, 3))


def test_empty_sequence():
    rng = np.random.default_rng(3)
    assert force_align(logp(rng, 7), ()) == ((), (), ())
    assert force_align(np.zeros((0, V)), []) == ((), (), ())


@pytest.mark.parametrize("frames,units", [
    (0, (2,)),
    (2, (2, 3, 4)),
    (4, (2, 2, 2)),  # needs 5 frames
])
def test_too_long_sequences_are_rejected(frames, units):
    with pytest.raises(ValueError, match="no CTC path"):
        force_align(np.zeros((frames, V)), units)


def test_only_minus_inf_paths_are_rejected():
    # Pruned log-probs: unit 3 is never a candidate, so no finite path exists.
    lp = np.log(np.full((5, V), 1 / V))
    lp[:, 3] = -np.inf
    with pytest.raises(ValueError, match="no CTC path"):
        force_align(lp, (2, 3))
    # Unit 2 is allowed only on frame 4: the path must place it there.
    lp = np.log(np.full((5, V), 1 / V))
    lp[:4, 2] = -np.inf
    assert force_align(lp, (2,)) == ((2,), (4,), (4,))


@pytest.mark.parametrize("lp,units", [
    (np.zeros((3, V)), (0,)),  # blank is not a unit
    (np.zeros((3, V)), (V,)),  # out of range
    (np.zeros((3, V)), (-1,)),
    (np.zeros((3, V)), (2.5,)),
    (np.zeros(V), (2,)),  # not 2-D
    (np.zeros((1, 0)), ()),  # no blank column
    (np.zeros((0, 0)), ()),  # no blank column, even without frames
    (np.full((3, V), np.nan), (2,)),
    (np.full((3, V), np.inf), (2,)),
])
def test_malformed_inputs_are_rejected(lp, units):
    with pytest.raises(ValueError):
        force_align(lp, units)
