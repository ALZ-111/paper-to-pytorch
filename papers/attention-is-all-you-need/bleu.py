"""
Corpus-level BLEU (Papineni et al., 2002), implemented from the definition.

    BLEU = BP * exp( sum_n w_n * log p_n ),   n = 1..4, w_n = 1/4
    p_n  = clipped n-gram matches / total candidate n-grams (summed over corpus)
    BP   = 1 if c > r else exp(1 - r / c)      (c = candidate length, r = reference length)

Matches sacrebleu's "tokenize=none" behaviour on pre-tokenised input, so numbers are
comparable to published Multi30k results computed on the same tokenisation.
"""

import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses, references, max_n=4):
    """hypotheses, references: lists of token lists. Returns BLEU in [0, 100]."""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h = _ngrams(hyp, n)
            r = _ngrams(ref, n)
            matches[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)

    if min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(log_precision)


if __name__ == "__main__":
    ref = "the cat sat on the mat".split()
    assert abs(corpus_bleu([ref], [ref]) - 100) < 1e-9
    print("half-match:", round(corpus_bleu(["the cat sat on a rug".split()], [ref]), 2))
