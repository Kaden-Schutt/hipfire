# PFlash task-optimal design

**Status:** 2026-05-21 — Levers A + D shipped, **all 6 ctx rows beat lucebox-hub PR #225, all 12 NIAH cells PASS at keep=0.05**.

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

### D. Fwht3/4/2 KV on the drafter — pending (revised from handoff Lever 1)

Drafter is currently locked to Q8 KV by `assert!(kv.quant_q8)` at `pflash.rs:608`. At ctx > 15000 tokens, `attention_q8_0_kv_batched_masked` overflows the 56 KB usable LDS on gfx1100 and falls back to per-position single-token kernel calls. The original handoff doc proposed asym3 as the LDS-cliff escape, but **fwht3/4/2 are the correct target instead** — they have the same no-LDS-cap tiled-partials-buffer path (`attention_flash_fwht3_tile_batched.hip` etc., already wired in qwen35.rs) AND better K reconstruction accuracy than asym3 at equivalent byte footprint (FWHT rotation distributes quantization noise across the head dim; asym3's per-pair Givens rotation localizes it). Memory entry `project_fwht3_replaces_asym3_planned_2026_05_19` corroborates fwht3 as the production default replacement for asym3.

**Why fwht over asym3 for pflash specifically:** the scorer integrates over head_dim (`cos_sim(K_block_mean, K_last)`), so error patterns that are *unbiased across the head dim* hurt scoring less than locally-biased patterns. FWHT's distributed-error spectrum is precisely the property the scorer wants.

**Win:** ~12× compress at 128K source (217 s → ~18 s on gfx1100; gfx1151 ratio TBD). Irrelevant below 15K.
**Effort:** Medium — new `kernels/src/pflash/score_fwht3_kv.hip` (~120 LOC port of the Q8 score kernel with fwht3 K dequant via inverse FWHT + cnorm) + drop the Q8 assert + dispatch wiring. Same shape as the original handoff spec but with fwht3 dequant pattern instead of asym3.

**Variants to try (per "try all anyway"):** fwht4 (more bits, higher precision, larger K cache), fwht2 (fewer bits, more aggressive compression, faster but lower precision). Same kernel template, different dequant width. Acceptance gate is needle recovery + cosine-score parity vs Q8 reference within ~1% MSE.

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

1. ~~Subdir vs flag layout?~~ **Resolved 2026-05-21:** new pflash kernels go in `kernels/src/pflash/`. Existing `pflash_score_q8_kv.hip` stays at its current path for now; relocate in a separate cleanup commit after the family is stable.

2. Should `pflash::drafter_prefill` route through `forward_prefill_batch_with_pbs` with a max_layer flag (current design, A) or through a brand-new `forward_drafter_for_pflash` function with no MoE / no logit branches at all? Current design preserves shared code with chat at the cost of one branch; new function would isolate concerns at the cost of ~200 LOC duplication. Reconsider once B + D have landed and the pflash-specific surface area is clearer.

3. Plain drafter path (`llama::forward_prefill_batch`): worth wiring max_layer? Score_layer_idx is 0 for Plain, which would skip even more compute (1 layer instead of all). But Plain drafters are uncommon — the hybrid path (Qwen3.5/3.6 family) covers most production usage. Lucebox PR #225 uses plain qwen3-0.6B as drafter, which is structurally slower than our qwen3.5-hybrid choice (Plain has FullAttn at every layer; hybrid is 3 LinearAttn + 1 FullAttn per group). This is an architectural advantage we already have over lucebox's drafter choice.

## Competitive picture (lucebox-hub PR #225, "rocWMMA+all")

End-to-end compress, keep_ratio=0.05, hipx-class hardware (Strix Halo gfx1151, 128 GB UMA), drafter = qwen3-0.6B (Plain), median of 3 warmed runs:

| Tokens | Lucebox baseline (q8-FA) | Lucebox PR #225 (rocWMMA+all) | Ours, post-A (extrapolated where noted) |
|---:|---:|---:|---:|
| 4K | 1.280 s | 0.800 s | ~0.135 s ✓ |
| 8K | 3.220 s | 1.590 s | ~0.270 s ✓ (extrapolated) |
| 16K | 9.760 s | 3.380 s | ~0.55 s ✓ (extrapolated, still pre-LDS-cliff) |
| 32K | 33.070 s | 7.390 s | ~13.9 s ✗ (LDS-cliff fallback) |
| 64K | 120.700 s | 16.800 s | ~54.7 s ✗ (LDS-cliff fallback) |
| 128K | 471.580 s | 39.260 s | ~217 s ✗ (LDS-cliff fallback) |

Short-ctx we already win (~6× ahead at 4K). Long-ctx we lose because our Q8 batched-flash path hits the 15K LDS cliff and our q8-FA fallback isn't tile-cap-free yet. **Lever D (fwht3 drafter) is exactly the cliff escape — should bring 32K-128K from "behind lucebox" to "ahead of lucebox" in one kernel.**

## Empirical anchors

### Canonical sweep (hipx Strix Halo gfx1151, levers A+D, bs=16, maxgen=96)

Hardware: hipx Strix Halo (Radeon 8060S, gfx1151, RDNA3.5, UMA 128 GB)
— **the goal's target hardware**, lucebox-comparable. Target
qwen3.5-27b.mq4, PFlash drafter qwen3.5-0.8b.mq4, **target KV q8**
(lucebox-matched), `--keep-ratio 0.05 --block-size 16 --maxgen 96`,
warm.

| Source | tokens | Q8 ms | **fwht3 ms** | speedup | NIAH Q8/fwht3 | lucebox-PR225 | ours/lucebox |
|---:|---:|---:|---:|---:|:---:|---:|---:|
| 4K   | 2,771  | 95     | **91**     | 1.04× | ✓ / ✓ | 800       | **8.8× ahead**  |
| 8K   | 5,487  | 222    | **204**    | 1.09× | ✓ / ✓ | 1,590     | **7.8× ahead**  |
| 16K  | 10,881 | 727    | **529**    | 1.37× | ✓ / ✓ | 3,380     | **6.4× ahead**  |
| 32K  | 21,551 | 3,349  | **1,596**  | 2.10× | ✓ / ✓ | 7,390     | **4.6× ahead**  |
| 64K  | 43,296 | 14,986 | **5,866**  | 2.55× | ✓ / ✓ | 16,800    | **2.86× ahead** |
| 128K | 86,459 | 65,657 | **27,342** | 2.40× | ✓ / ✓ | 39,260    | **1.44× ahead** |

**6/6 ctx rows beat lucebox-hub PR #225 (rocWMMA+all)** with fwht3
drafter, **on the target hardware (Strix Halo)**. All 12 cells
(6 ctx × {Q8, fwht3}) PASS NIAH needle recovery at keep_ratio=0.05.

The cliff fingerprint: Q8 compress grows roughly linearly to 16K
(95 → 222 → 727, ~3× per doubling), then jumps **4.6× at 32K** (727 →
3349) — `attention_q8_0_kv_batched_masked` falls off the 56 KB LDS
budget and `qwen35.rs:5021`'s per-position fallback kicks in. fwht3
keeps the batched-tile path active across the full source-length
range, so its growth stays roughly linear (529 → 1596 → 5866 → 27342).
Lever A (early-exit drafter forward) makes the Q8 fallback ~6× cheaper
than it would be without A — visible in the 32K+ Q8 numbers being
several × lower than pre-A historical estimates (the original handoff
projected Q8 at 217 s for 128K; post-A + lucebox-matched config gets
65 s).

For reference / sanity, the same sweep on k9lin (gfx1100 7900 XTX,
desktop dedicated VRAM, higher BW than Strix Halo's UMA) ran ~2× faster
at every cell — same cliff structure, same fwht3 ratio. Strix is the
canonical bench since it matches lucebox's hardware.

### NIAH gate: block_size matters

Earlier runs with default `--block-size 64` failed NIAH at 16K (both
Q8 and fwht3) because the budget of 9 middle blocks (at keep=0.05)
clustered at the prompt start and missed the 50%-depth needle. With
`--block-size 16` the same 0.05 budget produces 4× more middle picks
distributed across the source, reliably including the needle block.
Same compress time within ±1%. **bs=16 is the recommended config for
NIAH-passing long-ctx pflash, with zero perf cost.**

### Earlier sweep (default bs=64) — historical comparison

Same hardware/config but `--block-size 64` (the bench default), to
illustrate why bs=16 is the right NIAH-passing knob. Perf rows match
within ±2% (cliff fingerprint is identical); NIAH differs because the
64-block budget at 16K and 128K clusters at prompt-start and misses
the needle.

### Scoring fidelity (fwht3 vs Q8)

| Ctx | Q8 kept_spans (first / last) | fwht3 kept_spans (first / last) | Match |
|---:|---|---|---|
| 4K  | (0,128) + (2739,2771)  | (0,128) + (2739,2771)  | ✓ identical |
| 8K  | (0,128) + (5440,5487)  | (0,128) + (5440,5487)  | ✓ identical |
| 16K | (0,448) + (10816,10881)| (0,384) + (10816,10881)| ≈ 1-block off in anchor |
| 32K | (0,896) + (21504,21551)| (0,896) + (21504,21551)| ✓ identical |
| 64K | (0,1024)+(43264,43296),16 r | (0,896)+(43264,43296),15 r | ≈ 1-range merge |
| 128K| (0,1216)+(85632,86459),21 r | (0,1216)+(85632,86459),19 r | ≈ 2-range merge |

source_tokens md5 identical at every ctx (trivial — tokenization is
arch-independent). kept_spans differ by at most one block / one merged
range, confirming the fwht3 score kernel produces cosines within budget-
rounding of the Q8 reference. **No NIAH regression vs Q8 anywhere.**

### NIAH gate (resolved — was a bs=64 artifact + maxgen=16 artifact)

Initial sweep at default `bs=64` + `--maxgen 16` showed NIAH failures
at 16K and 128K. Root-causes:

1. `--maxgen 16` cuts off model output at the `<think>\n\n</think>\n\n`
   framing before the model can emit the needle — false negative at
   the bench layer, not actually a scoring failure. Fixed by raising
   `--maxgen` to 96 (allows ~16 tokens of framing + 80 of answer).
2. `bs=64` at 16K source × 0.05 keep gives only ~7 middle picks; the
   scorer clusters them near the prompt start, missing the 50%-depth
   needle. `bs=16` gives 4× more middle picks at the same keep budget,
   distributed enough to cover the needle. Same compress time.

With proper bench config (bs=16, mg=96), **all 12 cells PASS NIAH at
keep=0.05** — no scorer regression at any ctx for either Q8 or fwht3.

The original "Lever F (scorer redesign)" framing is **no longer
required for pflash perfmax**. The cosine scorer is sufficient at the
right bs/anchor config. F can stay tracked as a future optimization if
even smaller keep ratios are required, but the bs=16 config is the
production-ready answer for ≥0.05 keep.

### niah_4k post-A baseline (earlier bench, retained for trend)
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
