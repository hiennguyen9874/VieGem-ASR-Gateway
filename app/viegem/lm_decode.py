"""CTC prefix beam search with word-level KenLM shallow fusion (vendored from VieGem-ASR `asr/lm_decode.py`).

Units are Gemma tokenizer pieces; a piece starting with "▁" starts a new word. A word is scored by the LM when it is
complete (the next word starts, or at the end of the utterance, where </s> is also scored):

    score(prefix) = log P_ctc(prefix) + sum over completed words of (alpha * ln P_lm(word | history) + beta)

A unigram list (the LM vocabulary) gives words outside it, and partial words that are not a prefix of any listed
word, an extra `unk_offset`.

Gateway changes (see NOTICE): `decode` takes the per-frame candidates already pruned to `token_min_logp` (the engine
thread prunes them, ADR 0002) instead of the dense (T, V) matrix, and `pieces` only needs the candidate ids;
contextual biasing is removed. The search itself is unchanged.
"""

import math

LN10 = math.log(10.0)
NEG_INF = -float("inf")


def _logaddexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


class _Hyp:
    """One prefix: CTC log-probs ending in blank (pb) / non-blank (pnb) plus its LM context."""

    __slots__ = ("pb", "pnb", "state", "lm", "word", "nwords")

    def __init__(self, state, lm, word, nwords):
        self.pb, self.pnb = NEG_INF, NEG_INF
        self.state, self.lm, self.word, self.nwords = state, lm, word, nwords


class LMDecoder:
    def __init__(self, lm_path, unigrams=None, alpha=0.2, beta=1.5, unk_offset=-10.0):
        import kenlm

        self.kenlm = kenlm
        self.lm = kenlm.Model(str(lm_path))
        self.alpha, self.beta, self.unk_offset = alpha, beta, unk_offset
        self.vocab = set(unigrams) if unigrams else None
        self.prefixes = (
            {w[:i] for w in self.vocab for i in range(1, len(w) + 1)} if self.vocab else None
        )

    def _word(self, state, word):
        """LM transition for a completed word: (new state, weighted score)."""
        out = self.kenlm.State()
        s = self.alpha * LN10 * self.lm.BaseScore(state, word, out) + self.beta
        if self.vocab is not None and word not in self.vocab:
            s += self.unk_offset
        return out, s

    def _partial_penalty(self, word):
        if self.prefixes is None or not word or word in self.prefixes:
            return 0.0
        return self.unk_offset

    def _score(self, h):
        return _logaddexp(h.pb, h.pnb) + h.lm + self._partial_penalty(h.word)

    def _extend(self, h, piece):
        """LM context after appending one unit to prefix h."""
        if piece.startswith("▁"):
            if h.word:
                state, s = self._word(h.state, h.word)
                return _Hyp(state, h.lm + s, piece[1:], h.nwords + 1)
            return _Hyp(h.state, h.lm, piece[1:], h.nwords)
        return _Hyp(h.state, h.lm, h.word + piece, h.nwords)

    def decode(self, frames, pieces, blank=0, beam_width=32, beam_prune_logp=-10.0):
        """Best unit sequence.

        frames: one `(ids, log_probs)` pair per encoder frame, the log-softmax candidates with log-prob >=
        token_min_logp in ascending id order, or only the argmax when there is none (upstream's selection).
        pieces[v] is unit v's text for every candidate v ("" = blank/unk).
        """
        root = _Hyp(self.kenlm.State(), 0.0, "", 0)
        self.lm.BeginSentenceWrite(root.state)
        root.pb = 0.0
        beams = {(): root}
        for ids, lps in frames:
            nxt = {}

            def get(key, parent, unit):
                h = nxt.get(key)
                if h is None:
                    h = beams[key] if key in beams and unit is None else None
                    h = (
                        _Hyp(h.state, h.lm, h.word, h.nwords)
                        if h is not None
                        else self._extend(parent, pieces[unit])
                    )
                    nxt[key] = h
                return h

            for key, h in beams.items():
                total = _logaddexp(h.pb, h.pnb)
                for c, p in zip(ids, lps):
                    c, p = int(c), float(p)
                    if c == blank:
                        s = get(key, h, None)
                        s.pb = _logaddexp(s.pb, total + p)
                    elif key and c == key[-1]:
                        s = get(key, h, None)  # repeat without a blank between: same prefix
                        s.pnb = _logaddexp(s.pnb, h.pnb + p)
                        e = get(key + (c,), h, c)  # repeat after a blank: a new unit
                        e.pnb = _logaddexp(e.pnb, h.pb + p)
                    else:
                        e = get(key + (c,), h, c)
                        e.pnb = _logaddexp(e.pnb, total + p)
            scored = sorted(((self._score(h), k) for k, h in nxt.items()), reverse=True)
            best = scored[0][0]
            beams = {k: nxt[k] for s, k in scored[:beam_width] if s >= best + beam_prune_logp}

        def final(h):
            s = _logaddexp(h.pb, h.pnb) + h.lm
            state = h.state
            if h.word:
                state, w = self._word(state, h.word)
                s += w
            return s + self.alpha * LN10 * self.lm.BaseScore(state, "</s>", self.kenlm.State())

        return list(max(beams.items(), key=lambda kv: final(kv[1]))[0])
