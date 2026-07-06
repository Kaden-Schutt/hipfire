#!/usr/bin/env python3
"""
Generate N-tier graded MoE tier-map files for Qwen3.6-35B-A3B.

Reads per-(layer,expert) contribution TSVs (agentic + wt2) from
/workspace/moe-awq-regress/, computes per-layer UNION ranking, and emits:
  /workspace/tier_map_T3-2L.txt  (MQ6 / MQ4 / MQ2L)
  /workspace/tier_map_T3-3L.txt  (MQ6 / MQ4 / MQ3L)

Algorithm:
  1. For each corpus, normalise sum_contrib per layer so all experts sum to 1.
  2. union_score(L,e) = agentic_norm[L][e] + wt2_norm[L][e]
  3. Rank DESC within layer (stable tiebreak: ascending expert index).
  4. Assign: ranks 0..51 -> MQ6, ranks 51..128 -> MQ4, ranks 128..256 -> COLD.
     COLD = MQ2L (T3-2L) or MQ3L (T3-3L).
  5. Emit 40*256 = 10240 lines per file: "LAYER EXPERT DTYPE".
"""

AG = "/workspace/moe-awq-regress/expert_stats_agentic.tsv"
WT = "/workspace/moe-awq-regress/expert_stats_wt2.tsv"

L, E = 40, 256

# Tier boundaries
MQ6_TOP = 51     # ranks 0..51   => top 20% of 256
MQ4_TOP = 128    # ranks 51..128 => next 30%
# ranks 128..256 => bottom 50% => COLD


def load(path):
    """Return dict (layer, expert) -> sum_contrib (col index 5, 0-based)."""
    d = {}
    with open(path) as f:
        next(f)  # skip header
        for ln in f:
            cols = ln.split()
            if len(cols) < 6:
                continue
            try:
                lay = int(cols[0])
                exp = int(cols[1])
                val = float(cols[5])  # sum_contrib
            except ValueError:
                continue
            d[(lay, exp)] = val
    return d


def norm_per_layer(d):
    """Return dict (layer, expert) -> normalised contrib (sums to 1 per layer)."""
    out = {}
    for lay in range(L):
        total = sum(d.get((lay, e), 0.0) for e in range(E))
        for exp in range(E):
            raw = d.get((lay, exp), 0.0)
            out[(lay, exp)] = (raw / total) if total > 0.0 else 0.0
    return out


def emit(path, cold_label, agn, wtn):
    """Write tier-map file with COLD mapped to cold_label."""
    with open(path, "w") as f:
        for lay in range(L):
            # Sort experts by union score DESC, tiebreak by expert index ASC
            order = sorted(
                range(E),
                key=lambda e: (-(agn[(lay, e)] + wtn[(lay, e)]), e)
            )
            tier = {}
            for rank, exp in enumerate(order):
                if rank < MQ6_TOP:
                    tier[exp] = "MQ6"
                elif rank < MQ4_TOP:
                    tier[exp] = "MQ4"
                else:
                    tier[exp] = cold_label
            # Emit in (layer, expert) order
            for exp in range(E):
                f.write(f"{lay} {exp} {tier[exp]}\n")
    print(f"Wrote {path}")


def report_histogram(path, label):
    """Print per-dtype counts and sample lines from a tier-map file."""
    counts = {}
    total = 0
    samples = []
    with open(path) as f:
        for i, ln in enumerate(f):
            total += 1
            dtype = ln.strip().split()[2]
            counts[dtype] = counts.get(dtype, 0) + 1
            if i < 5:
                samples.append(ln.rstrip())
    print(f"\n{label} ({path}):")
    print(f"  Total lines: {total}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("  First 5 lines:")
    for s in samples:
        print(f"    {s}")


if __name__ == "__main__":
    print(f"Loading {AG} ...")
    ag = load(AG)
    print(f"  Loaded {len(ag)} rows")

    print(f"Loading {WT} ...")
    wt = load(WT)
    print(f"  Loaded {len(wt)} rows")

    print("Normalising per layer ...")
    agn = norm_per_layer(ag)
    wtn = norm_per_layer(wt)

    print("Generating T3-2L (MQ6/MQ4/MQ2L) ...")
    emit("/workspace/tier_map_T3-2L.txt", "MQ2L", agn, wtn)

    print("Generating T3-3L (MQ6/MQ4/MQ3L) ...")
    emit("/workspace/tier_map_T3-3L.txt", "MQ3L", agn, wtn)

    print("Generating T3-3L-mfp3 (MQ6/MQ4/MFP3E8) ...")
    emit("/workspace/tier_map_T3-3L-mfp3.txt", "MFP3E8", agn, wtn)

    print("Generating T3-2L-mfp2 (MQ6/MQ4/MFP2E8) ...")
    emit("/workspace/tier_map_T3-2L-mfp2.txt", "MFP2E8", agn, wtn)

    report_histogram("/workspace/tier_map_T3-2L.txt", "T3-2L")
    report_histogram("/workspace/tier_map_T3-3L.txt", "T3-3L")
    report_histogram("/workspace/tier_map_T3-3L-mfp3.txt", "T3-3L-mfp3")
    report_histogram("/workspace/tier_map_T3-2L-mfp2.txt", "T3-2L-mfp2")

    print("\nDone.")
