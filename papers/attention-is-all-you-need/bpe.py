"""
Byte-pair encoding (Sennrich, Haddow & Birch, 2016) from scratch.

The paper's WMT models use a shared 37k-token BPE vocabulary (Section 5.1). Word-level
vocabularies cannot represent unseen words at all; BPE splits rare words into pieces that
were seen, so any word can be encoded and the vocabulary stays small.

Algorithm (learn):
    1. Count word frequencies; represent each word as a tuple of characters plus an
       end-of-word marker "</w>" so that "low" and "lower" share "l o w" but end differently.
    2. Repeatedly find the most frequent adjacent symbol pair across the corpus and merge
       it into one symbol. Each merge is recorded; the merge list *is* the model.
Encode: apply the learned merges to a new word in the order they were learned.

The naive learner recounts every pair after each merge, O(merges x corpus). This one keeps
pair counts and an index from pair -> words containing it, so a merge only touches the
words it changes. 8,000 merges over Multi30k take ~20 s instead of ~15 min.
"""

import json
from collections import Counter, defaultdict

EOW = "</w>"


class BPE:
    def __init__(self, merges=None):
        self.merges = list(merges) if merges else []          # [(a, b), ...] in learned order
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}

    # ------------------------------------------------------------------ learn
    @classmethod
    def learn(cls, sentences, num_merges, min_freq=2, verbose=False):
        """sentences: iterable of token lists (already whitespace/punctuation split)."""
        word_freq = Counter(tok for s in sentences for tok in s)
        words = [tuple(w) + (EOW,) for w in word_freq]        # symbol sequences
        freqs = [word_freq[w] for w in word_freq]

        pair_counts = Counter()
        where = defaultdict(set)                                 # pair -> {word idx}
        for i, syms in enumerate(words):
            for a, b in zip(syms, syms[1:]):
                pair_counts[(a, b)] += freqs[i]
                where[(a, b)].add(i)

        merges = []
        for step in range(num_merges):
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if count < min_freq:
                break
            merges.append(best)
            merged = best[0] + best[1]

            for i in list(where[best]):
                old = words[i]
                new = _merge_word(old, best, merged)
                f = freqs[i]
                # Retract the old word's pairs, add the new word's pairs.
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    where[p].discard(i)
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        where.pop(p, None)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                words[i] = new
            if verbose and (step + 1) % 1000 == 0:
                print(f"  {step + 1} merges, last {best} x{count}")
        return cls(merges)

    # ----------------------------------------------------------------- encode
    def encode_word(self, word):
        """Apply merges to one word -> list of subword strings (last one ends in </w>)."""
        if word in self._cache:
            return self._cache[word]
        syms = tuple(word) + (EOW,)
        while len(syms) > 1:
            # Merge the pair with the lowest rank (earliest learned) present in the word.
            pairs = [(self.ranks[p], p) for p in zip(syms, syms[1:]) if p in self.ranks]
            if not pairs:
                break
            _, best = min(pairs)
            syms = _merge_word(syms, best, best[0] + best[1])
        out = list(syms)
        self._cache[word] = out
        return out

    def encode(self, tokens):
        """token list -> subword list."""
        return [piece for tok in tokens for piece in self.encode_word(tok)]

    @staticmethod
    def decode(pieces):
        """subword list -> token list (inverse of encode, exact)."""
        words, cur = [], ""
        for p in pieces:
            if p.endswith(EOW):
                words.append(cur + p[: -len(EOW)])
                cur = ""
            else:
                cur += p
        if cur:
            words.append(cur)
        return words

    # -------------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls([tuple(p) for p in json.load(f)])

    def __len__(self):
        return len(self.merges)


def _merge_word(syms, pair, merged):
    """Replace every non-overlapping occurrence of `pair` in `syms` with `merged`."""
    out, i = [], 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return tuple(out)


if __name__ == "__main__":
    corpus = [["low", "lower", "newest", "widest", "low", "low", "newer"]]
    bpe = BPE.learn(corpus, num_merges=10, min_freq=1)
    print("merges:", bpe.merges)
    print("lowest ->", bpe.encode_word("lowest"))
    print("decode :", BPE.decode(bpe.encode(["lowest", "wider"])))
