# 15 — Stage 2b PpMtp Phase B.2a cross-device overlap gate

Date: 2026-05-29
Branch: fix/q8-batched-masked-no-lds-cap (working tree at e6a25615 + bench)
Hardware: gfx906 (MI50, 32 GiB, HIP idx 0) + gfx1031 (RX 6700 XT, 12 GiB, HIP idx 1)
Bench: `crates/hipfire-runtime/examples/pp_overlap_microbench.rs`
Busy kernel: `gemm_hfq4g256_residual` M=K=5120 N=16, repeated per-band to mirror the trunk split.

## Question

The PpMtp perf gap is structural: the PP boundary in
`forward_prefill_batch_multi_with_caps` is fully serialized (gfx906 band
→ blocking peer copy → gfx1031 band; one card idle while the other
works). The only lever that can BEAT single-GPU decode is to pipeline
that boundary across decode steps. That's ~330-520 LOC of risky
stream/event/flip plumbing (plan merry-giggling-conway.md B.2b, 3 P0
review flaws). B.2a is the cheap physics gate first: **can a compute
kernel on gfx906 and a compute kernel on gfx1031 run CONCURRENTLY on
independent per-device streams, or does something serialize them?**

## Results

### Production split (48,16 iters)

| metric | µs |
|---|---|
| dev0 gfx906 (48 iters) alone | 9843.2 |
| dev1 gfx1031 (16 iters) alone | 2370.9 |
| sum of alones | 12214.1 |
| A: sequential (serialized boundary shape) | 22750.6 |
| B: concurrent (dual stream, one barrier) | 10263.9 |
| **overlap_ratio = A / B** | **2.217×** |

### Balanced split (32,32 iters)

| metric | µs |
|---|---|
| dev0 gfx906 (32 iters) alone | 6566.8 |
| dev1 gfx1031 (32 iters) alone | 4716.3 |
| sum of alones | 11283.1 |
| A: sequential | 23917.9 |
| B: concurrent | 6675.8 |
| A / B | 3.583× |

peer_access enabled bidirectionally: true.

## Read — overlap is REAL; gate PASSES, but use the conservative metric

**IMPORTANT — A/B overstates the win; lead with sum-of-alones vs B.** My
microbench's "sequential" mode does `issue(0); device_sync(0);
issue(1); device_sync(1)` — TWO full device-wide syncs. The real
`forward_prefill_batch_multi_with_caps` does `boundary_copy +
wait_boundary` (ONE cross-device wait), not two device syncs. So mode A
carries a barrier-tax artifact (A ≈ 1.86× the sum-of-alones in both
runs) that production does NOT pay. The A/B ratio (2.2× / 3.1×) is
therefore inflated as a proxy for the production win.

**The honest metric is sum-of-alones vs concurrent**, which isolates pure
cross-device compute overlap from that artifact:

| split | sum alones | concurrent | pure-overlap = sum/B |
|---|---|---|---|
| 48,16 (production) | 12214 | 10264 | **1.19×** |
| 32,32 (balanced) | 11283 | 6676 | **1.69×** |

1. **Cross-device concurrency WORKS.** Concurrent is below sum-of-alones
   in both runs → the two physically-separate cards genuinely run at
   once on independent streams. ROCm/PCIe/host does NOT force
   serialization. Worst-case "pipelining impossible" is RULED OUT.

2. **The ceiling tracks the work balance, as predicted.** Balanced 32,32
   hides nearly all of gfx1031's 4737 µs behind gfx906's 6564 (concurrent
   6797 ≈ gfx906-alone) → 1.66×. The skewed 48,16 can only hide the
   smaller 2371 µs → 1.19×. Ceiling = (T_big+T_small)/T_big.

3. **The split tension stands (note 14 vs here).** Pipelining wants
   BALANCE (→1.66× at 32,32) but: (a) gfx1031's 12 GiB can't hold 32
   trunk layers + MTP head + lm_head + KV at long ctx, and (b) the
   serialized-DECODE path (note 14) wanted gfx906-HEAVY (56,8). At the
   VRAM-feasible skew the pure-overlap ceiling is only ~1.19× (1.69× at
   the infeasible 32,32 balance).

4. **Realistic B.2b net is BELOW even 1.19×.** Add the true peer-copy
   data dependency (gfx1031 band N needs gfx906 band N's output) and the
   cross-step staleness/τ tax (draft_{N+1} before verify_N commits, plan
   P0-3). At the production split, 1.19× pure-compute ceiling minus those
   taxes is a thin margin for 330-520 LOC of high-risk plumbing — it does
   NOT clearly clear single-GPU's 22.5 tok/s from PpMtp's 14.2.

## Decision

**B.2a result is MIXED — overlap is physically real (premise confirmed),
but the VRAM-feasible-split ceiling is thin. Recommend NOT building the
full cross-step pipeline (B.2b) as the next step.** Reasoning:

- Pure cross-device overlap at the production 48,16 split is only ~1.19×
  (sum/B). Balanced 32,32 reaches 1.69× but is VRAM-infeasible on
  gfx1031's 12 GiB and fights the serialized-decode rebalance (note 14).
- B.2b's real net sits below 1.19× after the peer-copy data dependency
  and the cross-step staleness/τ tax — too thin to confidently clear
  single-GPU (22.5) from PpMtp's 14.2 for 330-520 LOC of P0-laden work.

**But there is a cheaper lever the bench surfaced:** in BOTH runs
sequential mode (A) ran ~1.86× the sum-of-alones — a barrier tax from the
per-band `device_synchronize` + bind_thread churn between bands. The
current `forward_prefill_batch_multi_with_caps` pays a structurally
similar cost: per chunk it does `boundary_copy + wait_boundary` (a
blocking cross-device wait) inline in the band loop, plus per-call
`PrefillBatchScratch::new`/free. Worth profiling whether those are
removable same-step costs BEFORE any cross-step pipelining. Forward-path
cleanup, not a spec-rollback change.

**Recommended order:**
1. **Opt 3 (no-replay rollback)** — highest value-per-LOC: kills PpMtp's
   2nd serialized boundary cross entirely (the `tape_captured=false` slow
   path), independent of the overlap-ceiling tension. Should get PpMtp
   below pp2-ar. The change llama.cpp/vLLM already made.
2. **Same-step boundary-wait/scratch cleanup** in the multi forward (cheap,
   measure first) — recovers some barrier tax without cross-step
   staleness risk.
3. **Full B.2b cross-step pipeline** — only if 1+2 leave a gap worth the
   risk, and only at a split where the overlap ceiling justifies it.

## Relation to note 14

Note 14 (Opt 2) found rebalancing-within-the-serialized-shape tops at
~14.2 tok/s. B.2a confirms the serialized shape leaves wall-clock on the
table, but the recoverable amount at a VRAM-feasible split is modest
(~1.19× pure overlap). The big-lever path remains Opt 3, not pipelining.

## Repro

```bash
source scripts/gpu-lock.sh && gpu_acquire pp-overlap-bench && \
  env HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16 \
  ./target/release/examples/pp_overlap_microbench && gpu_release
# BENCH_GFX906_ITERS / BENCH_GFX1031_ITERS override per-band volume.
```
