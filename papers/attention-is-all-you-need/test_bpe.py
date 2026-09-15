"""Tests for the from-scratch byte-pair encoder.  python -m pytest test_bpe.py -v"""

import random

from bpe import BPE, EOW


def test_learns_most_frequent_pair_first():
    # (l, o) appears in lo x3 + low x1 = 4; (o, </w>) 3; (o, w) 1; the rest <= 2.
    corpus = [["lo", "lo", "lo", "low", "newest", "widest"]]
    bpe = BPE.learn(corpus, num_merges=2, min_freq=1)
    assert bpe.merges == [("l", "o"), ("lo", EOW)]


def test_encode_applies_merges_in_learned_order():
    corpus = [["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3]
    bpe = BPE.learn(corpus, num_merges=20, min_freq=1)
    assert bpe.encode_word("low") == ["low" + EOW]           # fully merged, seen often
    pieces = bpe.encode_word("lowest")                      # unseen word: known pieces
    assert "".join(pieces) == "lowest" + EOW
    assert all(p in {"low", "est" + EOW} or len(p) <= 3 for p in pieces)


def test_decode_inverts_encode_exactly():
    random.seed(0)
    alphabet = "abcdefgh"
    words = ["".join(random.choice(alphabet) for _ in range(random.randint(1, 8))) for _ in range(500)]
    bpe = BPE.learn([words], num_merges=100, min_freq=1)
    new_words = ["".join(random.choice(alphabet) for _ in range(random.randint(1, 10))) for _ in range(200)]
    assert BPE.decode(bpe.encode(new_words)) == new_words
    assert BPE.decode(bpe.encode(words)) == words


def test_incremental_counts_match_naive_recount():
    """The indexed learner must produce the same merges as recounting from scratch."""
    random.seed(1)
    words = ["".join(random.choice("xyz") for _ in range(random.randint(2, 6))) for _ in range(300)]
    fast = BPE.learn([words], num_merges=30, min_freq=1)

    # Naive reference: recount every pair after every merge.
    from collections import Counter
    wf = Counter(words)
    vocab = {tuple(w) + (EOW,): c for w, c in wf.items()}
    naive = []
    for _ in range(30):
        pairs = Counter()
        for syms, c in vocab.items():
            for p in zip(syms, syms[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        naive.append(best)
        merged = best[0] + best[1]
        new_vocab = {}
        for syms, c in vocab.items():
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                    out.append(merged); i += 2
                else:
                    out.append(syms[i]); i += 1
            new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + c
        vocab = new_vocab
    assert fast.merges == naive


def test_save_load_roundtrip(tmp_path):
    bpe = BPE.learn([["hello", "help", "held", "hello"]], num_merges=5, min_freq=1)
    bpe.save(tmp_path / "m.json")
    loaded = BPE.load(tmp_path / "m.json")
    assert loaded.merges == bpe.merges
    assert loaded.encode_word("helper") == bpe.encode_word("helper")
