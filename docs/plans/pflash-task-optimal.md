# PFlash task-optimal design

**Status:** Draft 2026-05-21 — early-exit drafter forward shipped (commit `9479db1f`); remaining levers tracked.

## Premise

PFlash's drafter pass is *not chat*. It looks similar on the surface — load a small LM, run it over a prompt — but the workload it actually performs is mechanically different from chat in ways the current implementation does not exploit. Reusing chat kernels naively pays for compute pflash never consumes.

Specifically, the drafter in pflash:

| What pflash actually uses | What chat needs |
|---|---|
| K cache at *one* layer (`score_layer_idx`, the shallowest FullAttn) | K + V cache at every layer |
| One forward pass (no decode) | Prefill + N decode steps |
| No logits | Final logits every step |
| No final hidden state | Final hidden → lm_head |
| Tolerates coarser K precision (cosine integrates over head_dim) | Token-prediction-level precision |
| Disposable KV (used once, then released) | Persistent KV across the request |

Each row is an exploitable mismatch.

## Levers (ranked by impact × independence)

### A. Early-exit drafter forward at score_layer_idx — **SHIPPED (`9479db1f`)**

The drafter forward only needs to populate the K cache through the scoring layer. For Qwen3.5 hybrid drafters with `full_attention_interval=4`, that's layer 3 of 24 — meaning layers 4..23 plus the final norm + lm_head are wasted compute.

**Win:** ~6× drafter forward speedup (~80% of stack skipped) at every ctx length.
**Effort:** ~50 LOC, no new kernels — adds `max_layer: Option<usize>` to `forward_prefill_chunk` and `forward_prefill_batch_with_pbs`. Existing public `forward_prefill_batch` wrapper preserves its signature; only `pflash::drafter_prefill` opts in.
**Status:** Hybrid drafter wired. Plain drafter (`llama::forward_prefill_batch`) deferred to follow-up (~50 LOC).

### B. V-skip drafter — pending

The scorer reads K, never V. But the drafter still writes V cache at every FullAttention layer it visits (for chat-path correctness). Splitting V-projection from V-cache-write at the FullAttn layer body lets pflash compute V on-the-fly for in-pass attention and skip the cache write entirely.

**Win:** ~50% reduction in KV write bandwidth across the layers pflash *does* visit. At long ctx where KV writes dominate, expect 5-10% on compress.
**Effort:** Medium — kernel-level. `attention_flash_q8_0_*` family needs a `skip_v_cache_write` variant or flag. Affects 2-3 kernels.
**Stacks on:** A (V-skip applies to whichever layers A does run).

### C. Score-fused K-write — pending

The score kernel (`pflash_score_q8_kv`, ~3 ms on niah_4k) reads K cache twice in effect: once to compute per-block means, once to dot against `last_pos K`. If the per-block running mean is maintained as K positions are *written* to the cache (during the drafter forward), the second-pass scorer collapses to a tiny per-block cosine over precomputed sums.

**Win:** Eliminates `pflash_score_q8_kv` as a separable kernel — saves its ~3 ms (small) plus removes one full read sweep over the K cache (could be larger at long ctx).
**Effort:** Medium — extend `kv_cache_write_q8_0_batched` to also maintain `[n_blocks × kv_dim]` running f32 sums, plus a tiny finalize kernel for `sum / count` → cosine.
**Stacks on:** A, B (independent).

### D. Asym3 KV on the drafter — pending (handoff Lever 1)

Drafter is currently locked to Q8 KV by `assert!(kv.quant_q8)` at `pflash.rs:608`. At ctx > 15000 tokens, `attention_q8_0_kv_batched_masked` overflows the 56 KB usable LDS on gfx1100 and falls back to per-position single-token kernel calls. asym3 KV uses tiled partials-buffer reduction with no LDS cap.

**Win:** ~12× compress at 128K source (217 s → ~18 s). Irrelevant below 15K.
**Effort:** Medium — new `pflash_score_asym3_kv.hip` kernel (~120 LOC port of the Q8 score kernel with asym3 K dequant) + drop the Q8 assert + dispatch wiring. Spec in `docs/plans/pflash-drafter-asym3-handoff.md`.
**Stacks on:** A (early-exit composes with any KV mode).

### E. Tiled Q8 batched flash — pending (handoff Lever 2)

The other side of the 15K LDS cliff: a `attention_flash_q8_0_batched_tile.hip` kernel that uses partials-buffer reduction over Q8 KV so the chat-path also escapes the cliff. Broader than pflash (helps any Q8 long-ctx prefill, not just drafter), but unblocks the same regime D unblocks.

**Win:** 3-12× drafter prefill at >15K source. Same regime as D.
**Effort:** Larger — new kernel + reduce + dispatch. Spec in `docs/plans/pflash-drafter-asym3-handoff.md`.

### F. Sparse/chunked drafter attention — pending

At very long ctx, even with D or E unblocking the LDS cliff, the drafter still does O(L²) self-attention per FullAttn layer. The scorer's signal is robust to attention-pattern approximations — sliding window or sparse local attention in the drafter could trade some scoring fidelity for big throughput wins.

**Win:** Aggressive — possibly 5-10× at 128K. **Risk:** needle-miss if approximation drops too much positional signal.
**Effort:** Large — design + tune + validate against needle-recovery gate. Needs empirical study before commitment.

## Composition

Levers are mostly independent; the dispatch can stack them:

```
A (early-exit at score_layer_idx)
  └── reduces stack depth
  
B (V-skip in the layers A does run)
  └── reduces per-layer BW
  
C (score-fused K-write)
  └── eliminates the second-pass scorer kernel
  
D or E (long-ctx LDS-cliff escape)
  └── for ctx > 15K only — orthogonal to A/B/C at short ctx
  
F (sparse drafter attention)
  └── only worth chasing once D/E land, plus careful quality study
```

At short ctx (≤15K): A + B + C are the relevant stack. Expected combined win: ~7× compress.

At long ctx (>15K): A + (D or E) is the headline; B + C still apply on top.

## Open questions

1. Should pflash get its own subdirectory `kernels/src/pflash/` with isolated kernel files, or should pflash-specific variants live as flags on existing kernels? Subdir is cleaner if 3+ pflash-specific kernels land (B + C + D minimum); flags work if it stays to 1-2.

2. Should `pflash::drafter_prefill` route through `forward_prefill_batch_with_pbs` with a max_layer flag (current design, A) or through a brand-new `forward_drafter_for_pflash` function with no MoE / no logit branches at all? Current design preserves shared code with chat at the cost of one branch; new function would isolate concerns at the cost of ~200 LOC duplication.

3. Plain drafter path (`llama::forward_prefill_batch`): worth wiring max_layer? Score_layer_idx is 0 for Plain, which would skip even more compute (1 layer instead of all). But Plain drafters are uncommon — the hybrid path (Qwen3.5/3.6 family) covers most production usage.

## Empirical anchors

niah_4k on hipx (Radeon 8060S / gfx1151), target qwen3.5-27b.mq4, PFlash
drafter qwen3.5-0.8b.mq4, asym3 KV, --maxgen 64:

| keep_ratio | metric | Pre A | Post A (shipped) | Δ |
|---:|---|---:|---:|---:|
| 0.05 | compress ms | 492 | **92** | **-81% (5.35×)** |
| 0.05 | prefill ms | 1010 | 1004 | -1% (noise) |
| 0.05 | **TTFT ms** | **1505** | **1099** | **-27%** |
| 0.03 | compress ms | 493 | **91** | **-82% (5.42×)** |
| 0.03 | prefill ms | 598 | 598 | match |
| 0.03 | **TTFT ms** | **1095** | **692** | **-37%** |
| 0.01 | compress ms | 494 | **91** | **-82% (5.43×)** |
| 0.01 | prefill ms | 306 | 300 | -2% (noise) |
| 0.01 | **TTFT ms** | **803** | **394** | **-51%** |

source_tokens md5 `c1f8fa2c7634cced267143b6aecdadb0` IDENTICAL pre/post —
tokens unchanged. kept_spans pattern (sink + last block) IDENTICAL pre/post
— scoring behavior unchanged. Needle recovery at 0.05 still PASSES (1/1);
fail at 0.03/0.01 unchanged (separate scorer-at-short-ctx issue, see "F"
follow-up). Target prefill identical within noise — confirming early-exit
only affects drafter forward, not target.

Long-ctx (niah_128k) bench pending — A alone won't move the LDS-cliff
needle there; need D or E to land before that row can be filled.

This doc is a living plan; update the table as benches land.
