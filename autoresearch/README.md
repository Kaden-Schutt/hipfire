# autoresearch — arch-gated RDNA kernel autoresearch

The **fixed-eval loop** that turns the bill-of-debt (which kernel is bound how,
per arch) into certified kernel wins. Karpathy single-experiment + fixed-eval
discipline: mutate one kernel, fight the champion under a fixed eval, keep the
winner, log every round. **The ledger IS the research.**

## The loop
1. **Bill-of-debt** (`scripts`/oracle_profile) names the lever per arch/kernel
   (DRAM-thrash → tile for L2; latency/occ-starved → raise occupancy; ALU → int8).
2. **Mutate** the kernel `.hip` (variant in `variants/`).
3. **`ab_certify.sh`** — the fixed eval: baseline vs variant, median-of-N
   `kernel_decode_tok_s` (daemon instrument) + coherence, warm + JIT-cleared.
4. **Certify** by rank SEPARATION, not a magnitude band: a WIN is when all N
   variant runs sit above all N baseline runs (clean separation) — a confident
   win at ANY magnitude, coherent + clock-matched. **Take every real small win,
   reject every real small loss, discard only genuine overlap** — the compound-
   interest engine (+0.3% ten times is a real +3%; a symmetric ±3% band throws
   both real small wins and losses away and compounds to zero).
5. **Log the WHY** (`PROFILE=1`): the target kernel's roofline (did occ/L2/mem
   move?) + its **VGPR/SGPR/LDS/scratch** (registers→occupancy = the mechanism) +
   a top-kernel wall% diff (kernel-level no-clobber / knock-on). Profile on wins.
6. **Ledger** every run → `ledger/<arch>_<kernel>.jsonl`. Champion source →
   `variants/`, then dispatch invoice (wire into `dispatch.rs`).

## The corpus — permanent, shared, queried
Everything is **git-tracked in the repo**, so `git pull` gives every contributor
the full research history (the history IS the research):
- `ledger/<arch>_<kernel>.jsonl` — append-only, one line per A/B (verdict · decode
  · clock · coherence · roofline · VGPR/LDS/scratch · knock-on). Boxes write to
  their local `~/hipfire/autoresearch/ledger/`; sync back + commit to share.
- `variants/*.hip` — winning kernel sources (reproducible).
- `oracle_db.py` — indexes/queries the ledgers:
  `oracle_db.py wins | best <arch> <k> | history <arch> <k> | kernel <arch> <k> | summary`

## Mechanism (embedded kernels)
Kernels are `include_str!`-embedded in `rdna-compute/src/kernels.rs`, so a variant
means: swap `kernels/src/<k>.hip` → **rebuild the daemon** (re-embeds it) → the
content-hash JIT re-compiles the changed kernel (belt-and-suspenders: evict its
`.hsaco`). Runs ON the GPU box against `~/hipfire`.

## Airworthiness gate (calibrate before trusting)
The harness must pass two shots before it may judge a novel variant — the same
"instruments calibrated before airworthiness" rule applied to the loop itself:
- **NO-OP** (an active kernel swapped for a comment-added, behavior-identical
  copy → forces a real re-JIT) → **Δ ≈ 0**, coherent. If a no-op "wins," the
  instrument lies (this is exactly how the JIT-cache false-positive is caught —
  a no-op reading +N% means we measured a stale binary).
- **KNOWN WIN** (`HIPFIRE_GEMV_ROWS=2`, the validated +4.24% R=2 multirow) →
  reproduce a real, above-noise, coherent win. If it can't recover a known win,
  it can't find a new one.

Only no-op-nulls ∩ known-win-reproduces certifies the loop airworthy.

## Usage
```
ab_certify.sh <arch> <dev> <card> <model> <kernel> <label> [B_env] [B_swap.hip]
```
