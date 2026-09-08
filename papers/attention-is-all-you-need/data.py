"""
Multi30k German -> English data pipeline.

Multi30k (Elliott et al., 2016) is the standard small-scale translation benchmark:
29,000 training pairs, 1,014 validation, 1,000 test, describing Flickr images.
Sentences are short (~13 tokens), which is what makes a from-scratch Transformer
trainable on a laptop CPU in under an hour.

Tokenisation is deliberately simple (lowercase, split punctuation) so there is no
dependency on spaCy or sentencepiece. Vocabulary is word-level with a frequency cutoff.

Usage:
    from data import load_multi30k
    train, val, test, src_vocab, tgt_vocab = load_multi30k()
"""

import gzip
import io
import os
import re
from collections import Counter

import requests
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
SPLITS = {"train": "train", "val": "val", "test": "test_2016_flickr"}

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    """Lowercase and split into words and individual punctuation marks."""
    return _TOKEN_RE.findall(text.lower())


def _download(split, lang):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{split}.{lang}")
    if not os.path.exists(path):
        url = f"{BASE_URL}/{SPLITS[split]}.{lang}.gz"
        print(f"downloading {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f, open(
            path, "w", encoding="utf-8"
        ) as out:
            out.write(f.read())
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class Vocab:
    def __init__(self, sentences, min_freq=2):
        counter = Counter(tok for s in sentences for tok in s)
        self.itos = SPECIALS + sorted(
            [w for w, c in counter.items() if c >= min_freq], key=lambda w: -counter[w]
        )
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids, strip_special=True):
        toks = []
        for i in ids:
            if strip_special and i == EOS:
                break
            if strip_special and i in (PAD, BOS):
                continue
            toks.append(self.itos[i])
        return toks


def load_multi30k(min_freq=2, max_len=50):
    """Returns (train, val, test, src_vocab, tgt_vocab).

    Each split is a list of (src_ids, tgt_ids) where tgt_ids already includes
    BOS and EOS. Sentences longer than max_len tokens are dropped from train only.
    """
    raw = {s: (_download(s, "de"), _download(s, "en")) for s in SPLITS}
    tok = {s: ([tokenize(x) for x in de], [tokenize(x) for x in en]) for s, (de, en) in raw.items()}

    src_vocab = Vocab(tok["train"][0], min_freq)
    tgt_vocab = Vocab(tok["train"][1], min_freq)

    def encode_split(split, filter_long):
        pairs = []
        for de, en in zip(*tok[split]):
            if filter_long and (len(de) > max_len or len(en) > max_len):
                continue
            pairs.append((src_vocab.encode(de), [BOS] + tgt_vocab.encode(en) + [EOS]))
        return pairs

    return (
        encode_split("train", True),
        encode_split("val", False),
        encode_split("test", False),
        src_vocab,
        tgt_vocab,
    )


def collate(batch):
    """Pad a list of (src_ids, tgt_ids) to tensors: src (B,S), tgt (B,T)."""
    src_len = max(len(s) for s, _ in batch)
    tgt_len = max(len(t) for _, t in batch)
    src = torch.full((len(batch), src_len), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt


def batches_by_length(pairs, max_tokens, shuffle=True):
    """Yield batches with roughly max_tokens tokens each, bucketing by length so
    padding is minimal (Section 5.1 of the paper batches by token count too)."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), len(pairs[i][1])))
    batches, cur, cur_max = [], [], 0
    for i in order:
        s, t = pairs[i]
        cur_max = max(cur_max, len(s), len(t))
        if cur and cur_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], max(len(s), len(t))
        cur.append(i)
    if cur:
        batches.append(cur)
    if shuffle:
        perm = torch.randperm(len(batches)).tolist()
        batches = [batches[j] for j in perm]
    for b in batches:
        yield collate([pairs[i] for i in b])


if __name__ == "__main__":
    train, val, test, sv, tv = load_multi30k()
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print(f"src vocab {len(sv)}  tgt vocab {len(tv)}")
    s, t = train[0]
    print("DE:", " ".join(sv.decode(s)))
    print("EN:", " ".join(tv.decode(t)))
    lens = [len(s) for s, _ in train]
    print(f"mean src len {sum(lens)/len(lens):.1f}, max {max(lens)}")
