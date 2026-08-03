#!/usr/bin/env python3
"""Paired comparison of two flash_prefill_quality runs.

Record layout (44 bytes, little-endian):
    u32 next_tok | f32 nll | f32 lse | u32 top8[8]

The two runs score the IDENTICAL positions, so NLL can be compared pairwise —
which removes between-position variance and is far more sensitive than
comparing two independent means.
"""
import sys, math
import numpy as np

REC = np.dtype([("next_tok", "<u4"), ("nll", "<f4"), ("lse", "<f4"), ("top8", "<u4", 8)])


def load(p):
    return np.fromfile(p, dtype=REC)


def main(a_path, b_path, a_name="f32", b_name="f16-wmma"):
    a, b = load(a_path), load(b_path)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    assert (a["next_tok"] == b["next_tok"]).all(), "runs scored different positions"

    na, nb = a["nll"].astype(np.float64), b["nll"].astype(np.float64)
    ppl_a, ppl_b = math.exp(na.mean()), math.exp(nb.mean())

    # Paired delta: same positions, so the per-position difference cancels
    # content variance entirely.
    d = nb - na
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else float("nan")
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se

    top1 = (a["top8"][:, 0] == b["top8"][:, 0]).mean()
    ov = np.array([len(set(a["top8"][i]) & set(b["top8"][i])) for i in range(n)])

    print(f"scored positions: {n}  (paired, identical token stream)")
    print()
    print(f"  perplexity {a_name:>10}: {ppl_a:.4f}")
    print(f"  perplexity {b_name:>10}: {ppl_b:.4f}")
    print(f"  ppl ratio            : {ppl_b/ppl_a:.5f}")
    print()
    print(f"  mean NLL {a_name:>12}: {na.mean():.6f}")
    print(f"  mean NLL {b_name:>12}: {nb.mean():.6f}")
    print(f"  paired delta (b - a) : {d.mean():+.6f} nats")
    print(f"  std error            : {se:.6f}")
    print(f"  95% CI               : [{lo:+.6f}, {hi:+.6f}]")
    print(f"  t statistic          : {t:+.3f}")
    sig = abs(t) > 1.96
    print(f"  significant at 0.05  : {'YES' if sig else 'NO'}")
    if not sig:
        # Smallest effect this sample could have resolved.
        mde = 1.96 * se
        print(f"  -> resolvable effect : +-{mde:.6f} nats "
              f"({(math.exp(mde)-1)*100:.3f}% on ppl)")
    print()
    print(f"  top-1 agreement      : {top1*100:.2f}%")
    print(f"  top-8 overlap mean   : {ov.mean():.2f}/8   (>=6 in {(ov>=6).mean()*100:.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
