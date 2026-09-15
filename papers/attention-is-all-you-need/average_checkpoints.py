"""
Checkpoint averaging (Section 6.1 of the paper: "we averaged the last 5 checkpoints").

Late in training the weights bounce around a good region of parameter space as the
learning rate decays; averaging several recent checkpoints lands closer to the centre of
that region than any single one and typically adds a few tenths of a BLEU point for free.
Only the parameters are averaged; the vocab and args are copied from the last checkpoint.

    python average_checkpoints.py checkpoints/multi30k_bpe_epoch1[6-9].pt checkpoints/multi30k_bpe_epoch20.pt \
        --out checkpoints/multi30k_bpe_avg5.pt
    python translate.py evaluate --tokenizer bpe --checkpoint checkpoints/multi30k_bpe_avg5.pt
"""

import argparse

import torch


def average_state_dicts(state_dicts):
    """Element-wise mean of matching float tensors; non-float buffers are taken from the
    first checkpoint (there are none in this model, but be safe)."""
    avg = {}
    for key, ref in state_dicts[0].items():
        if ref.is_floating_point():
            avg[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
            avg[key] = avg[key].to(ref.dtype)
        else:
            avg[key] = ref.clone()
    return avg


def average_checkpoints(paths, out):
    ckpts = [torch.load(p, map_location="cpu") for p in paths]
    merged = dict(ckpts[-1])
    merged["model"] = average_state_dicts([c["model"] for c in ckpts])
    merged["averaged_from"] = [c["epoch"] for c in ckpts]
    torch.save(merged, out)
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    m = average_checkpoints(args.checkpoints, args.out)
    print(f"averaged epochs {m['averaged_from']} -> {args.out}")
