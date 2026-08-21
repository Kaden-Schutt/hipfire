# Decode-tuning campaign progress — gfx1100, Qwen3.5-4B + Qwen3.8-27B

**Date:** 2026-08-21 · **Branch:** `tune/iter3-gate-up-bt2` · **Scope:** plain AR
decode, q8 KV, MQ4 weights. No speculation, no quant-format changes (user
constraints).

## Goal and targets

Original goal: +50% decode. Bandwidth-floor analysis revised the targets:

| Model | Weights/token | AR floor @ 960 GB/s | Session start | Session end | Stated target |
|---|---|---|---|---|---|
| Qwen3.5-4B (2.59 GB) | 2.59 GB | 371 tok/s (2.70 ms) | 190.6 tok/s | **201.5 tok/s** | 288 (unreachable this arch gen) |
| Qwen3.8-27B (15.66 GB) | 15.66 GB | 61 tok/s (16.3 ms) | 48.5 tok/s | 48.5 tok/s | ~56 (user-revised from 72) |

72 tok/s on 27B was proven physically impossible: it requires 1127 GB/s
effective, above the 960 GB/s peak. The user approved revising to ~56.

## Shipped (4 commits, each fully certified)

| Commit | Change | Dispatches deleted/token | 4B gain |
|---|---|---|---|
| `57b189fd` | conv scalar_prep gate → all gfx1100 shapes | 24 (sigmoid_alpha) | +1.6% |
| `1a254267` | gated_norm_mq_rotate → admit 4B (dim 2560, 32 v-heads) | 24 (gated_norm + rotate) | +1.3% |
| `fb24c077` | fa_prep → admit 16Q/4K + rope FMA-contraction fix | 24 (deinterleave + 2×rmsnorm + rope) | +2.1% |
| `2b1d5dcb` | fa_epilogue → admit 16Q/4K | 8 (sigmoid_mul) | +1.0% |

**4B: 190.6 → 201.5 tok/s (5.11 → 4.82 ms/token, +5.7%)** on the bench
example; **207 tok/s** through the daemon (GPU-side sampling — the product
path and the honest number). 27B flat at 48.5 — it already had all four
fusions via its shape gates.

Certification per win (no exceptions): 3 fresh-process A/B pairs with every
ON run beating every OFF run, `test_kernels` 16/16, greedy `--temp 0` text
byte-identical, p99 ≈ p50.

### The pattern that produced all four wins

Every fused kernel was already shape-parameterized or grid-generic; the gates
hard-coded the exact certified shapes. Qwen3.5-4B matched the structural
requirements of all four (32 v-heads × 128 DeltaNet geometry, non-awq wo,
16Q/256 FA, fa_prep's grid-derived K workgroups). Check the structural
requirements before writing any new fusion.

### The one real bug: rope FMA contraction (`fb24c077`)

fa_prep admission initially flipped greedy tokens. `probe_fa_prep` (new
example: legacy deinterleave+rmsnorm+halfsplit sequence vs fused, per-buffer
bitwise diff bucketed by element index) localized ulp diffs to the rope
sin-branch only (dims 32-63). Root cause: clang contracted the identical
expression `x0*sin_a + x1*cos_a` as `fma(x0, sin_a, x1*cos_a)` in the
standalone rope kernel but `fma(x1, cos_a, x0*sin_a)` in the fused kernel.
Fix: explicit `__builtin_fmaf(x0, sin_a, x1*cos_a)` in the fused kernel;
probe then bitwise-identical on fa_q/fa_gate/fa_k. This also upgraded the 27B
certification from token-exact to bitwise-vs-legacy (same source, NQ
string-replace).

## Falsified levers (documented so nobody re-opens them blind)

| Lever | Attempts | Result |
|---|---|---|
| residual GEMV schedule (26% of 4B token) | 6: 8× unroll (−14% VGPR), dualrow (−8% TLP), prefetch (−0.5%), K-spec (null), K_SPLITS=2 (regression), rows2/LDS (−3.4×) | **408 GiB/s is the confirmed local optimum.** rows2/LDS (71 µs vs 21 µs) definitively falsifies the L2-x-traffic theory: halving L2 x re-reads at 8-wave occupancy is catastrophically worse — the kernel is latency/occupancy-bound. |
| qkvza schedule (14%) | 5: reduce_chain, hoist_x32, pair_buffer, K2560-spec, (k2048 family inapplicable at K=2560) | 517-538 GiB/s local optimum. K-specialization null — LLVM already strength-reduces runtime `groups_per_row`. |
| rmsnorm AWQ multi-CU (15%) | 2: handshake wavegrid (flat — ~2-4 µs spin cancels multi-CU gain), redundant-reduction (−4% — 32-thread workgroups pay the full 2560-load reduction at 8× worse MLP) | The reduction is irreducibly single-workgroup-latency-bound at K=2560-6144. Exact-tree handshake variant kept behind `HIPFIRE_AWQ_NORM_WAVEGRID=1` (token-exact, bit-exact probe). |
| Spec decode without drafts | ngram, dspark | ngram 138-185 vs AR 206.7 (draft overhead > acceptance); dspark flat. Trained drafts: Path C documented dead. |
| Graph replay | measured | No-op — launch overhead not binding (CPU stays ahead; wall = GPU kernel time + per-token sync). |
| sclk forcing | measured | Flat — decode is memory-bound. |
| K-specialization | residual + qkvza | Null both — compiler handles runtime `groups_per_row`. |

Full per-iteration data: `.codeinsight+research/kernel-tune/ledger/ledger.jsonl`
(iterations 5-12).

## Remaining levers (in scope, ranked)

1. **Consumer-fold of rms+awq+rotate into the qkvza/qkv/gate_up prologues** —
   ~10% on 4B, ~8% on 27B. The only remaining double-digit lever. Works where
   the standalone multi-CU variants failed because the redundant per-block
   reduction (2560 loads from L2 + ~2560 FLOP + a 40-op butterfly per 256
   values) amortizes inside memory-bound GEMVs. Multi-day: 3 consumer kernels
   × variant families + dispatch plumbing.
2. **kv_cache_write fold into flash attention** — ~1%. 8 FA layers write K/V
   then read them; the attention kernel could emit them as a side effect.
3. **gated_delta_net internals** (6.4% pool) — compact2_b2 exists; inner
   schedule unattacked. Maybe 1-2%.
4. **gemv_q8_0 (lm_head)** (15% at 816 GiB/s = 85% peak) — ~2% theoretical
   remainder (split-K, 128-bit loads).

## Measurement methodology (hard-won this session)

- **p99 ≈ p50 or the number is void.** Environmental VRAM contention
  (concurrent GPU processes; dmesg `svm_range_evict_svm_bo_worker`) produced
  0.5-4 s stalls that faked a "27B regression" and possibly earlier results.
  `HIPFIRE_SLOW_TOKEN_LOG=1` (new, in bench_qwen35_mq4) prints any token >50 ms.
- **`--temp 0` for every text A/B.** `hipfire run` defaults to
  `generation.temperature=0.3`; diffs are sampling noise otherwise.
- **Touch `crates/rdna-compute/src/kernels.rs` after editing any `.hip`.**
  Cargo does not rebuild dependents when an `include_str!`-embedded kernel
  changes; binaries silently JIT stale source. Bit this campaign twice.
- **Rebuild every binary under test** (daemon and examples each embed their
  own kernel copy).
- **Probe before E2E.** `probe_awq_wavegrid` / `probe_fa_prep` run
  legacy-vs-fused on synthetic data with per-buffer bitwise diffs and index
  bucketing — localizes a divergence to the exact expression in minutes
  instead of days of CLI bisecting.

## Tooling added

- `HIPFIRE_SLOW_TOKEN_LOG=1` — per-token stall log (step/pos/ms) in
  bench_qwen35_mq4.
- `probe_awq_wavegrid`, `probe_fa_prep` — bitwise kernel-certification probes
  (gated behind `required-features = ["deltanet"]`).
- `docs/kernel-tune-pipeline.md`, `scripts/kernel-tune-loop.sh` — the
  profile → one-lever → certify → decide loop this campaign ran.
- `HIPFIRE_AWQ_NORM_WAVEGRID=1` — exact-tree multi-CU AWQ norm (null result,
  kept as infrastructure).
