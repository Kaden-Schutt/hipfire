#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# FU2 bisect comparator: diff the HF reference dump (dump_hf_reference.py) against
# the hipfire dump (bisect_nano4b example) layer-by-layer at a prompt position,
# and report the FIRST layer where hipfire diverges from HF beyond tolerance.
# `dump_hf_reference.py` saves hidden_<L> in Hipfire-aligned form:
# embeddings, then each block output before final norm_f.
#
#   python3 benchmarks/nemotron/compare_bisect.py /tmp/nemo_hf_ref.npz /tmp/nemo_hipfire.bin

import struct
import sys

import numpy as np


def load_hipfire(path):
    with open(path, "rb") as f:
        n_caps, hidden, vocab = struct.unpack("<III", f.read(12))
        caps = np.frombuffer(f.read(n_caps * hidden * 4), dtype="<f4").reshape(n_caps, hidden)
        logits = np.frombuffer(f.read(vocab * 4), dtype="<f4")
    return caps, logits


def main():
    hf_path, hp_path = sys.argv[1], sys.argv[2]
    # which prompt position to compare (default 0 = isolates block math; pass
    # the position index to match CAP_POS=last in the dumper).
    pos = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    hf = np.load(hf_path)
    caps, hp_logits = load_hipfire(hp_path)
    print(f"comparing at prompt position {pos}")

    n = caps.shape[0]  # num_layers+1
    hf_hidden = [np.array(hf[f"hidden_{i}"][pos], dtype=np.float64, copy=True) for i in range(n)]
    hf_logits = np.array(hf["logits"][pos], dtype=np.float64, copy=True)
    hp = np.array(hp_logits, dtype=np.float64, copy=True)
    print(f"comparing {n} hidden layers (last position)")
    print(f"{'layer':>6} {'max|Δ|':>12} {'rel':>10} {'hf_std':>10} {'hp_std':>10}")
    first_bad = None
    for i in range(n):
        hf_h = hf_hidden[i]
        hp_h = caps[i].astype(np.float64)
        d = np.abs(hf_h - hp_h)
        denom = np.abs(hf_h).mean() + 1e-9
        rel = d.mean() / denom
        flag = ""
        if rel > 0.05 and first_bad is None:
            first_bad = i
            flag = "  <== FIRST DIVERGENCE"
        print(f"{i:>6} {d.max():>12.4e} {rel:>10.4f} {hf_h.std():>10.4f} {hp_h.std():>10.4f}{flag}")

    # logits
    hf_top5 = np.argsort(-hf_logits)[:5].tolist()
    hp_top5 = np.argsort(-hp)[:5].tolist()
    hf_mean_abs = np.abs(hf_logits).mean() + 1e-9
    logit_diff = np.abs(hf_logits - hp)
    print(f"\nlogits: max|Δ|={logit_diff.max():.4e} rel={logit_diff.mean() / hf_mean_abs:.4f}")
    print(f"  HF top5:      {hf_top5}")
    print(f"  hipfire top5: {hp_top5}")
    if first_bad is None:
        print("\nRESULT: no divergence > 5% rel — hipfire matches HF per layer.")
    else:
        print(
            f"\nRESULT: first divergence at hidden_{first_bad} "
            f"(= {'embeddings' if first_bad == 0 else f'after block {first_bad - 1}'})."
        )


if __name__ == "__main__":
    main()
