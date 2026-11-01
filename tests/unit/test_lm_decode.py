"""LM decoding as pure functions (`app.lm`, `app.viegem.lm_decode`), CPU only, on a tiny ARPA model.

Failure modes covered: the sparse (pruned) search differs from upstream's dense search (lost candidates, changed
iteration order or float rounding); pruning drops the argmax fallback or keeps non-candidates; the device pruning in
the engine disagrees with the reference pruning; the LM does not change the text where the acoustic scores are close;
LM units get frames outside the chunk, out of order, or through pruned entries; an all-blank chunk yields units.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from app.lm import ALPHA, BETA, BEAM_WIDTH, decode_lattice, frames_of, prune, TOKEN_MIN_LOGP, UNK_OFFSET
from app.viegem.lm_decode import LMDecoder
from tests.lm_fixture import WORDS, write_arpa

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "references/VieGem-ASR/asr/lm_decode.py"

# Unit ids: 0 blank, 1 <unk>, then pieces. "▁" starts a word.
PIECES = ["", "", "\u2581a", "b", "c", "\u2581d", "\u2581e", "a"]


@pytest.fixture(scope="module")
def lm_file(tmp_path_factory):
    return write_arpa(tmp_path_factory.mktemp("lm") / "tiny.arpa")


@pytest.fixture(scope="module")
def decoder(lm_file):
    return LMDecoder(lm_file, unigrams=WORDS, alpha=ALPHA, beta=BETA, unk_offset=UNK_OFFSET)


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)) - x.max(-1, keepdims=True))


def frames(*rows):
    """Log-probs (T, V) from per-frame {unit: prob} (the rest shares a tiny mass)."""
    out = np.full((len(rows), len(PIECES)), math.log(1e-4))
    for t, row in enumerate(rows):
        for u, p in row.items():
            out[t, u] = math.log(p)
    return log_softmax(out).astype(np.float32)


def test_prune_keeps_upstream_candidates_and_the_argmax_fallback():
    rng = np.random.default_rng(0)
    # 400 units: near-flat rows have no candidate (max log-prob ~ -ln 400 < -5), peaked rows have several.
    lp = log_softmax(rng.normal(size=(200, 400)) * rng.uniform(0.05, 6, size=(200, 1))).astype(np.float32)
    lat = prune(lp)
    assert lat.frames == 200 and lat.offsets[-1] == len(lat.ids) == len(lat.logps)
    fallback = 0
    for t, (ids, lps) in enumerate(frames_of(lat)):
        expected = np.flatnonzero(lp[t] >= TOKEN_MIN_LOGP)
        if len(expected) == 0:
            expected, fallback = np.array([lp[t].argmax()]), fallback + 1
        assert ids.tolist() == expected.tolist()
        assert lps.tolist() == lp[t, expected].tolist()
    assert fallback  # the flat rows exercised the argmax fallback


def test_engine_device_pruning_matches_the_reference():
    torch = pytest.importorskip("torch")
    from app.engine import VieGemEngine

    rng = np.random.default_rng(1)
    logits = (rng.normal(size=(120, 50)) * rng.uniform(0.5, 6, size=(120, 1))).astype(np.float32)
    got = VieGemEngine(Path("."), "cpu", 1.0, lm=True)._lattice(torch.from_numpy(logits))
    want = prune(torch.log_softmax(torch.from_numpy(logits), -1).numpy())
    assert got.frames == want.frames
    for a, b in zip(got[:3], want[:3]):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_lm_changes_close_acoustic_choice_and_forced_frames_follow_it(decoder):
    # Greedy reads "▁a c ▁d" ("ac d"); the LM prefers "ab d" and "c" is only slightly more likely than "b".
    lp = frames({0: 0.9}, {2: 0.9}, {0: 0.9}, {4: 0.5, 3: 0.45}, {4: 0.5, 3: 0.45}, {0: 0.9}, {5: 0.9}, {0: 0.9})
    greedy = [int(u) for u in lp.argmax(-1)]
    assert [u for i, u in enumerate(greedy) if u and (i == 0 or u != greedy[i - 1])] == [2, 4, 5]
    units, first, last = decode_lattice(decoder, prune(lp), PIECES)
    assert units == (2, 3, 5)
    assert first == (1, 3, 6) and last == (1, 4, 6)


def test_all_blank_chunk_has_no_units(decoder):
    lp = frames(*[{0: 0.99}] * 10)
    assert decode_lattice(decoder, prune(lp), PIECES) == ((), (), ())


def test_forced_frames_are_in_range_and_ordered_on_random_lattices(decoder):
    rng = np.random.default_rng(2)
    for _ in range(100):
        t = int(rng.integers(1, 60))
        x = rng.normal(size=(t, len(PIECES))) * 3
        x[:, 0] += rng.uniform(0, 4)
        lat = prune(log_softmax(x).astype(np.float32))
        units, first, last = decode_lattice(decoder, lat, PIECES)
        assert len(units) == len(first) == len(last)
        assert all(0 <= a <= b < t for a, b in zip(first, last))
        assert all(b < a for b, a in zip(last, first[1:]))
        for u, a, b in zip(units, first, last):  # each unit sits on frames where it was a candidate
            for f in range(a, b + 1):
                assert u in lat.ids[lat.offsets[f]:lat.offsets[f + 1]]


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="references/VieGem-ASR not checked out")
def test_sparse_search_equals_upstream_dense_search(lm_file):
    upstream = {}  # exec the source: importing it would write __pycache__ into the read-only reference
    exec(compile(UPSTREAM.read_text(), str(UPSTREAM), "exec"), upstream)
    ref = upstream["LMDecoder"](str(lm_file), unigrams=WORDS, alpha=ALPHA, beta=BETA, unk_offset=UNK_OFFSET)
    ours = LMDecoder(lm_file, unigrams=WORDS, alpha=ALPHA, beta=BETA, unk_offset=UNK_OFFSET)
    rng = np.random.default_rng(3)
    changed = 0
    for _ in range(200):
        t = int(rng.integers(1, 50))
        x = rng.normal(size=(t, len(PIECES))) * rng.uniform(0.5, 4)
        x[:, 0] += rng.uniform(0, 3)
        lp = log_softmax(x).astype(np.float32)
        want = ref.decode(lp, PIECES, beam_width=BEAM_WIDTH)
        assert ours.decode(frames_of(prune(lp)), PIECES, beam_width=BEAM_WIDTH) == want
        greedy = [int(u) for i, u in enumerate(lp.argmax(-1)) if u and (i == 0 or u != lp[i - 1].argmax())]
        changed += want != greedy
    assert changed  # the LM did change some outputs, so the comparison is not trivially greedy
