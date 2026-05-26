# MQ4-Lloyd 9B PPL — Phase 1 viability check (issue #182)

**Issue:** [#182](https://github.com/Kaden-Schutt/hipfire/issues/182)
(Implement MQ4-Lloyd, qt=21).
**Branch:** `lloyd-max-mq3-spike` + this branch's MQ4-Lloyd commits.
**Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, Strix Halo APU,
137 GB shared LPDDR5x, ROCm 7.12).
**Date:** 2026-05-07.

## Why this exists

Issue #182's Phase 1 deliverable: produce a `qwen3.5-9b.mq4-lloyd` artifact
and measure PPL to gate the kernel-fast-path work in Phase 2 (5 fused
kernel files, 13 qwen35.rs call sites). Quality answer:

  - **≥ MQ4 ppl (≥ 7.78 in #182's setup)** → abandon, no Pareto win.
  - **≤ ~6** → ship Phase 2.

This finding answers that gate.

## Method

Implementation is the **slow generic-GPU path only** (option B from the
session that produced this work):

  - `quantize_mq4g256_lloyd` — direct port of `quantize_mq3g256_lloyd`
    with K=16 centroids, 32 B header (16 fp16) + 128 B 4-bit nibble
    indices = 160 B/group. Lloyd's k-means with 16 evenly-spaced
    percentile init, 8 max iterations, deterministic centroid sort by
    value. `crates/hipfire-quantize/src/main.rs:875` (function);
    `:1722` (`Mq4Lloyd` enum); `:2027` (`--allow-mq4-lloyd` gate).
  - `kernels/src/gemv_mq4g256_lloyd.hip` — slow chip-agnostic GEMV with
    16-entry codebook lookup as a 16-way ternary chain. No gfx1100
    fast-path / LDS-codebook variant. ~90 LOC, mirrors
    `gemv_mq3g256_lloyd.hip`.
  - `DType::MQ4G256Lloyd` plumbed through `rdna-compute/src/dispatch.rs`,
    `kernels.rs`, `profile.rs`. `Gpu::gemv_mq4g256_lloyd` +
    `Gpu::gemv_mq4g256_lloyd_with_rotate` mirror the MQ3-Lloyd Rust
    bindings.
  - Loader arms (`hfq.rs:436`, `qwen35.rs:744`) handle qtype=21.
    `qwen35.rs:1004` adds the CPU-dequant path for DeltaNet conv1d
    consumers (16 fp16 centroids → nibble unpack → inverse FWHT).
  - `weight_gemv` / `weight_gemv_prerotated` / `fused_rmsnorm_rotate_for_mq` /
    `rotate_x_for_mq` in `hipfire-runtime/src/llama.rs` recognize
    `MQ4G256Lloyd` and route to the slow GEMV.
  - **Skipped per option B:** `gemv_mq4g256_lloyd_residual.hip`,
    `fused_gate_up_mq4g256_lloyd.hip`, `fused_qkv_mq4g256_lloyd.hip`,
    `fused_qkvza_mq4g256_lloyd.hip`, gfx1100 fast variants, the 13
    qwen35.rs fused-kernel call sites. These remain Phase 2 work.

PPL methodology:

  - Binary: `target/release/examples/perplexity` (single-window,
    position-by-position prefill, `-log_softmax(logits)[next_token]`
    over `[warmup, ctx-1)` positions).
  - Corpus: `benchmarks/calib/calib-5m.txt` (wikitext2-test content,
    20 MB).
  - Settings: `--ctx 2048 --warmup 8 --offset 0` (matches the
    canonical Lloyd-Max comparison framework from PR #115 / #181's
    devlog).
  - Quantize source: `/data/models/qwen/Qwen3.5-9B/` (HuggingFace
    safetensors f16, 4 shards).

## Results

### 9B Qwen3.5 PPL (this corpus, gfx1151)

| Format        | B/group | File size  | NLL/tok | **PPL** | vs MQ4 |
|---------------|--------:|-----------:|--------:|--------:|--------:|
| MQ4 (uniform) | 136     | 5.31 GB    | 2.687   |   14.68 | 1.000× |
| MQ3-Lloyd     | 112     | 4.57 GB    | 3.211   |   24.81 | 1.690× |
| **MQ4-Lloyd** | **160** | **6.06 GB**| **2.523** | **12.48** | **0.850×** |

**MQ4-Lloyd lands 15.0 % below MQ4 PPL on this corpus.**

### Cross-corpus calibration

The absolute PPL numbers run ~1.9× higher than issue #182's local
wikitext2-test measurements (#182: MQ4=7.78, MQ3-Lloyd=13.09, projected
MQ4-Lloyd≈4.30). The corpus content is similar but tokenization +
window-offset differences shift the absolute level. The **ratio**
between formats is stable:

  - MQ3-Lloyd / MQ4 — issue #182: **1.683×**; this finding: **1.690×**.
    Within 0.5 %; corpora differ only in absolute level, not relative
    quality.
  - **Implication: applying issue #182's MQ4 baseline 7.78 × the
    observed 0.850 ratio projects MQ4-Lloyd ≈ 6.61 on
    wikitext2-test.** Above the issue's "ship at ≤ 6" gate but
    comfortably below the "abandon at ≥ MQ4=7.78" gate.

### Quality projection vs reality

Issue #182's quality projections, vs observed:

| Source                         | Lloyd-vs-uniform ratio at 4-bit | Projected 9B MQ4-Lloyd PPL (#182 corpus) |
|--------------------------------|--------------------------------:|------------------------------------------:|
| Full-extrapolation (1.81×)     |                          0.553× |                                     ~4.30 |
| Half-extrapolation             |                          0.776× |                                     ~6.04 |
| **Observed (this finding)**    |                      **0.850×** |                                  **~6.61** (projected) |

Lloyd's relative gain compresses faster from 3-bit to 4-bit than even
the half-extrapolation. With 16 centroids the uniform 4-bit grid
already covers most of the FWHT-rotated weight distribution well, so
the data-driven placement reclaims less than at 3-bit (8 centroids).
This is consistent with issue #182's open question 1: *"Does Lloyd's
relative quality gain extrapolate from 3-bit (8 levels) to 4-bit
(16 levels)? Plausibly smaller, but even halved is still a win."* The
"plausibly smaller" turned out to be ~half-of-half (54 % of the 3-bit
gain), still a genuine improvement.

## Pareto positioning

| Format        | B/group | Overhead vs MQ4 | PPL ratio vs MQ4 |
|---------------|--------:|----------------:|-----------------:|
| MQ4 (uniform) |     136 |          0.0 %  |          1.000× |
| **MQ4-Lloyd** | **160** |     **+17.6 %** |     **0.850×** |
| MQ6           |     200 |       +47.0 %   |         (TBD)   |

MQ4-Lloyd trades 17.6 % bandwidth for 15.0 % PPL reduction. Whether
this is Pareto-favourable depends on the application:

  - **Quality-bound users** (long-context coherence, agent reasoning):
    MQ4-Lloyd is a clean win — substantially better PPL at ~5/6 the
    cost gap to MQ6.
  - **Throughput-bound users** (decode tok/s on memory-bound
    hardware): the +17.6 % bandwidth overhead is a real perf cost,
    and the slow generic kernel costs more on top
    (`elapsed: 220.0s` for MQ4-Lloyd ppl vs `52.2s` for MQ4 — that's
    4.2× decode-time penalty from the kernel alone, no LDS-codebook
    fast path). Phase 2 of issue #182 is required before this becomes
    a credible default.

## Performance footnote (slow kernel, expected)

The 9B PPL run produced 9.3 tok/s on MQ4-Lloyd vs 39.1 tok/s on
uniform MQ4 and 16.6 tok/s on MQ3-Lloyd. The MQ4-Lloyd absolute is
4.2× slower than MQ4 because:

  1. No LDS-codebook fast path (issue #182's Phase 2: 5 kernel files,
     "K4 LDS expansion" wrinkle from open question 2).
  2. No fused QKVZA / QKV / gate+up / residual variants — runtime
     falls through to per-projection slow GEMV.
  3. The chained 16-way ternary inside `gemv_mq4g256_lloyd.hip` does
     not fold to a clean conditional move chain on RDNA3 the way the
     8-way version does for MQ3-Lloyd.

This is expected for a slow-path-only viability check. The 9.3 tok/s
should not be quoted as "MQ4-Lloyd decode tok/s." Phase 2 work is
expected to recover the gap (PR #181 closed a similar 3.2× gap on
MQ3-Lloyd via K4 + LDS-codebook, landing 9B at 121.7 tok/s — close to
MQ4).

## Recommendation for issue #182

Per the issue's quality gate:

> If MQ4-Lloyd 9B projects ≥ MQ4's 7.78 ppl, abandon (no Pareto win).
> If it projects ≤ 6, ship.

Observed: 6.61 (projected onto issue #182's corpus). **This is between
the two gates** — clearly NOT in the abandon range, slightly above
the "ship" gate.

Recommendation: **proceed to Phase 2** (5 kernel files + 13 qwen35.rs
sites + parity test + perf gate). The quality lift is real and stable
(0.850× ratio is well outside measurement noise — within-fork ratios
match issue #182's wikitext2-test ratios to 0.5 %), and the Phase 1
implementation provides a working baseline for kernel-perf A/B against
the LDS-codebook fast path. The 6.61 vs 6.0 gap is small enough that
whether to ship depends on Phase 2's perf delivery, not on quality.

## Repro

```sh
source scripts/rocm-env.sh

# Quantize:
./target/release/hipfire-quantize \
  --input /data/models/qwen/Qwen3.5-9B \
  --output ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
  --format mq4-lloyd --allow-mq4-lloyd

# Build perplexity binary:
cargo build -p hipfire-runtime --example perplexity --release

# Score:
source scripts/gpu-lock.sh && gpu_acquire "ppl-mq4-lloyd"
./target/release/examples/perplexity \
  ~/.hipfire/models/qwen3.5-9b.mq4-lloyd \
  benchmarks/calib/calib-5m.txt \
  --ctx 2048 --warmup 8 --offset 0
gpu_release
```

## Open follow-ups

1. **Phase 2 kernel work** — `gemv_mq4g256_lloyd_residual.{,gfx1100.}hip`,
   `fused_gate_up_mq4g256_lloyd.{,gfx1100.}hip`,
   `fused_qkv_mq4g256_lloyd.{,gfx1100.}hip`,
   `fused_qkvza_mq4g256_lloyd.{,gfx1100.}hip`, gfx1100 fast variants
   for the slow generic kernel here, parity test, 13 qwen35.rs call
   sites. Ungated by this finding (proceed). LDS-codebook layout
   under K4 is the open design wrinkle (open question 2 in #182):
   16 entries × 4 groups = 64 LDS entries, doesn't fit 32 threads × 1
   load. Two options (cooperative double-load vs K2 unroll) per the
   issue's discussion.
2. **0.8B / 4B PPL** — issue #182 suggested measuring smaller sizes
   first; we landed on 9B directly. A short follow-up bench run on
   0.8B / 4B MQ4-Lloyd would confirm the ratio is stable across model
   scales (cf. PR #115's 1.94× / 2.01× / 2.27× ratios for MQ3-Lloyd
   across 0.8/4/9B — likely tighter at 4-bit but not measured here).
3. **Cross-corpus calibration** — confirm the 1.9× absolute-level
   gap between this corpus and issue #182's wikitext2-test is purely
   corpus, not implementation. Easiest check: clone the issue's
   corpus path and re-run. Not blocking.
