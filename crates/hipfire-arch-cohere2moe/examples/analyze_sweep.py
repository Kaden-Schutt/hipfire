#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Offline KLD + PPL analysis for the North-Mini-Code-1.0 BF16/Q8/MQ6/MQ4 sweep.
#
# Reads per-position logit dumps produced by the `kld_logits --dump` example
# (binary: u32 n_pos, u32 vocab, then n_pos*vocab f32 LE) plus the token-id list
# used to produce them, and reports, for each tier:
#   * KL(oracle || tier) over next-token distributions (mean / median / p99), and
#   * wikitext perplexity = exp(mean NLL of the true next token).
# The oracle (first --dump) is the KLD reference.
#
# Usage:
#   analyze_sweep.py --tokens tokens.json \
#     --dump bf16=bf16.logits --dump q8=q8.logits \
#     --dump mq6=mq6.logits --dump mq4=mq4.logits [--ref bf16]
import argparse, json, struct, sys
import numpy as np


def load_dump(path):
    with open(path, "rb") as f:
        n_pos, vocab = struct.unpack("<II", f.read(8))
        arr = np.frombuffer(f.read(), dtype="<f4")
    return arr.reshape(n_pos, vocab)


def log_softmax(x):  # x: [n_pos, vocab]
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--dump", action="append", required=True,
                    help="name=path.logits (first is the KLD reference unless --ref)")
    ap.add_argument("--ref", default=None, help="tier name to use as KLD reference")
    args = ap.parse_args()

    tiers = {}
    order = []
    for d in args.dump:
        name, path = d.split("=", 1)
        tiers[name] = load_dump(path)
        order.append(name)
    ref = args.ref or order[0]
    toks = np.array(json.load(open(args.tokens)), dtype=np.int64)

    n_pos = min(v.shape[0] for v in tiers.values())
    # PPL uses positions 0..n_pos-2 (predict the *next* token); cap by token list.
    ppl_n = min(n_pos - 1, len(toks) - 1)
    ref_lsm = log_softmax(tiers[ref][:n_pos])
    ref_p = np.exp(ref_lsm)

    print(f"{'tier':>8} | {'KL.mean':>9} {'KL.med':>9} {'KL.p99':>9} | {'PPL':>9}  (ref={ref}, n_pos={n_pos})")
    print("-" * 64)
    for name in order:
        lsm = log_softmax(tiers[name][:n_pos])
        # KL(ref || tier) per position, then summary.
        kl = (ref_p * (ref_lsm - lsm)).sum(axis=1)
        kl = np.clip(kl, 0.0, None)
        # PPL: NLL of the true next token at each position.
        idx = toks[1:ppl_n + 1]
        nll = -lsm[np.arange(ppl_n), idx]
        ppl = float(np.exp(nll.mean()))
        print(f"{name:>8} | {kl.mean():9.5f} {np.median(kl):9.5f} "
              f"{np.percentile(kl, 99):9.5f} | {ppl:9.3f}")


if __name__ == "__main__":
    main()
