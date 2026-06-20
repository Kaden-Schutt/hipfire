# Quantization exploration — gfx1103 (RDNA3 UMA APU)

First-principles study of which quantization scheme best suits gfx1103, with
reproducible probes. Two questions: (E2) what gives the most quality per stored
byte while still mapping to the platform's native integer matmul, and (E1) does
reducing bytes/weight actually speed up decode on this UMA part.

Reproduce:
- `cargo run -p hipfire-quantize --example quant_explore` (CPU, SQNR sweep)
- `cargo run --release -p rdna-compute --example bench_gemv_dtype_bw` (GPU)

## Platform first principles

- **Bandwidth-bound decode.** gfx1103 shares LPDDR5 system DRAM (~90 GB/s,
  contended with the CPU). Per-token decode streams the whole weight matrix
  once → bytes/weight is the dominant decode cost. Quantization is primarily a
  *bandwidth* play here, not a compute play.
- **Native low-precision matmul:** WMMA/dot for f16, bf16, iu8, iu4 (no fp8).
  Symmetric uniform int maps directly to iu8/iu4; non-uniform/affine schemes
  need a dequant or a correction term.
- **No LDS** (firmware hang) → register-tiled kernels.

## E2 — Quality per bit (SQNR, higher is better)

`quant_explore` quantizes controlled weight distributions and reports SQNR and
effective bits/weight (incl. per-group scale/zp overhead). Representative run
(rows=256, k=4096):

| scheme | bits/w | gaussian | +1%×8 outl | +0.1%×20 outl | channel outl |
|--------|-------:|---------:|-----------:|--------------:|-------------:|
| int8 sym g128       | 8.13 | 43.8 | 36.6 | 36.5 | 36.1 |
| int4 sym per-row    | 4.00 | 16.0 |  4.6 |  2.7 |  3.1 |
| int4 sym g128       | 4.13 | 18.6 | 11.6 | 13.4 | 11.6 |
| int4 sym g32        | 4.50 | 20.3 | 16.3 | 18.0 | 16.7 |
| int4 affine g128    | 4.25 | 20.0 | 15.4 | 16.2 | 15.6 |
| **int4 sym+FWHT g256** | **4.06** | 18.0 | **18.5** | **18.8** | **18.7** |
| nf4 g128 (non-unif) | 4.13 | 20.4 | 15.5 | 16.0 | 15.2 |

Findings:
1. **Naive per-row int4 collapses on outliers** (2.7–4.6 dB) — one scale set by
   the max outlier wastes all resolution on the bulk. This is why low-bit needs
   help.
2. **On realistic outlier weights, FWHT-rotated symmetric int4 wins** — highest
   SQNR (18.5–18.8 dB) at the *lowest* overhead (4.06 bits/w), and it is stable
   across outlier types. The rotation spreads outliers so a single group scale
   fits the bulk.
3. **FWHT beats non-uniform NF4 on outliers** (18.5 vs 15.5) despite NF4 being
   non-uniform — NF4's per-group absmax is still wrecked by outliers, and NF4
   cannot use the native int path. So for this platform, rotation > codebook.
4. On *clean* Gaussian, NF4 / int4-g32 edge ahead (~20 dB) and FWHT is neutral —
   rotation's value is specifically outlier suppression, and real LLM weights
   have outliers.
5. **int8 is the robust fallback** everywhere (24–44 dB) for tensors too
   sensitive for 4-bit.

**Recommendation:** FWHT-rotated symmetric int4 with group scales is the
quality/byte sweet spot for gfx1103, and it maps to native iu4. This is exactly
the existing **MQ4** format — so E2 validates MQ4 from first principles for this
platform. int8 (≈ MQ8 / q8) is the safe per-tensor fallback. The engine already
FWHT-rotates activations at runtime, so the rotated-weight path is free on the
activation side.

## E1 — Does it actually speed up decode? (GEMV bandwidth)

`bench_gemv_dtype_bw` times the generic GEMV tier at M=K=8192 (128 MiB f16
weight, exceeds caches) on gfx1103:

| kernel | B/w | µs/call | GB/s (weight) | vs f16 |
|--------|----:|--------:|--------------:|-------:|
| f16→f32    | 2.0 | 3406 | 39.4 | 1.00× |
| bf16→bf16  | 2.0 | 3409 | 39.4 | 1.00× |
| iu8→i32    | 1.0 | 2051 | 32.7 | 1.66× |
| iu4→i32    | 0.5 | 1148 | 29.2 | 2.97× |

Findings:
1. **Fewer bytes → faster decode, monotonically** (int8 1.66×, int4 2.97×). The
   bandwidth thesis holds: quantization is the decode lever on this part.
2. **But scaling is sub-ideal** (int4 2.97×, not 4×) because the naive GEMV is
   **not yet bandwidth-bound**: f16 GEMV hits only ~39 GB/s, ~43% of the ~90
   GB/s DRAM ceiling. One wave32 per row with scalar 2-byte loads is
   latency/issue-bound, so per-element unpack overhead (worst for int4) eats
   into the byte savings.
3. Therefore the realized decode win from 4-bit is currently ~3×, with clear
   headroom: a bandwidth-optimal GEMV (128-bit vectorized loads, multiple waves
   or more rows per block) should raise absolute GB/s toward peak AND tighten
   the dtype scaling toward the ideal 2×/4×.

## E3 — Activation precision & the fused-iu4 (W4A4) path

`quant_wxax_explore` measures GEMM *output* SQNR (what model quality depends on)
for combinations of weight/activation precision, with and without FWHT rotation
of both operands. Activations modelled with strong per-channel outliers (the
realistic LLM regime); M=128 K=2048 B=64 g=128:

| scheme | no rotation | + FWHT (both) | fused path |
|--------|------------:|--------------:|-----------|
| W4A16 (deq→f16) | 14.5 | 18.1 | f16 wmma |
| W8A8            | 32.3 | 41.2 | iu8 wmma |
| W4A8 (mixed)    | 14.5 | 18.1 | upcast→iu8 |
| **W4A4**        | **9.3** | **16.0** | **iu4 wmma** |

Findings:
1. **Naive 4-bit activations cost ~5 dB** of output SQNR vs W4A16 (14.5→9.3):
   per-token int4 cannot absorb activation channel-outliers. Confirms the
   intuition that W4A4 hurts quality.
2. **8-bit activations are essentially free**: W4A8 ≈ W4A16 — the int4 *weight*
   is the binding constraint, not the activation.
3. **Rotation rescues W4A4**: +6.7 dB (9.3→16.0), within ~2 dB of W4A16+FWHT.
   The orthogonal rotation is exact (`Y = X·Wᵀ = (XQ)(WQ)ᵀ`) and Gaussianizes
   activations so int4 becomes tolerable (QuaRot/SpinQuant). The engine already
   FWHT-rotates activations.

Hardware caveats that shape the choice:
- **No mixed iu4×iu8 WMMA on RDNA3.** Fused int matmul is iu4×iu4 or iu8×iu8
  only. "W4A8" is therefore not one fused op — upcast the int4 weight to int8
  and run iu8 WMMA: keeps the 4-bit *storage/bandwidth*, gets iu8 *compute* and
  full weight quality.
- **Regime decides whether A4 helps at all.** Decode (GEMV, B=1) is
  memory-bound — A4 gives ~no speedup over A16 (the activation vector is tiny),
  only quality loss. The fused-iu4 W4A4 win is the **compute-bound**
  prefill/batched GEMM regime.

Resolved choice by regime:

| regime | choice | rationale |
|--------|--------|-----------|
| Decode (B=1) | **W4A16** (deq→f16) | bandwidth-bound; keep activations high-precision (free) |
| Prefill, quality-first | **W4A8 via iu8** (upcast) | full weight quality, int8 compute, 4-bit storage |
| Prefill, speed-first | **W4A4 fused iu4 + rotation** | fastest compute; rotation → within ~2 dB |

## E4 — How far can W4A4 quality be pushed? (all fused-iu4)

`quant_w4a4_improve` stacks iu4-preserving techniques on the W4A4 baseline and
measures output SQNR (M=128 K=2048 B=64, outlier-heavy activations):

| W4A4 scheme (fused iu4) | SQNR dB | Δ baseline |
|-------------------------|--------:|-----------:|
| baseline: FWHT256 + absmax, A=g128 | 16.0 | — |
| + clip-search scale | 17.1 | +1.0 |
| + clip + A=g32 | 17.3 | +1.3 |
| + full-K Hadamard | 16.8 | +0.8 |
| **+ SmoothQuant α0.5 + clip + g32** | **21.9** | **+5.9** |
| ref: W4A8 (upcast iu8), same front-end | 24.2 | |
| ref: W8A8 (ceiling), same front-end | 46.4 | |

Findings:
1. **W4A4 quality is very improvable.** Naive W4A4 was 9.3 dB (E3); the full
   iu4-preserving stack reaches **21.9 dB — +12.5 dB over naive (~18× less
   error)** — and closes most of the gap to W4A8 (24.2), still on the fused iu4
   path.
2. **SmoothQuant (per-channel migration) is the dominant lever** (+5.9 dB):
   `s_j = max|X_:,j|^α / max|W_:,j|^(1-α)`, then `X/s, W·s` (exact product). It
   moves activation channel-outliers into weights, which int4 tolerates far
   better. Activation scale is a runtime elementwise op (foldable into the
   preceding norm); weight scale is offline. iu4-compatible.
3. **Clip-search + fine activation groups are cheap additive wins** (~+1.3 dB):
   MSE-optimal per-group scale instead of absmax, A=g32.
4. **Rotation block size barely matters once SmoothQuant runs** (256-block ≈
   full-K) — both target the same outliers.
5. The residual ~2 dB to W4A8 is the intrinsic 4-bit-activation cost; closing it
   further needs GPTQ weight error-compensation (repo has `gptq.rs`) or
   outlier-channel mixed precision (breaks pure iu4 — needs a side GEMM).

So the fused-iu4 W4A4 path is quality-viable for compute-bound prefill with the
recipe **SmoothQuant → rotation → clip-search int4, A=g32**. Reproduce:
`cargo run -p hipfire-quantize --example quant_w4a4_improve`.

## E5 — End-to-end on-GPU validation of the W4A4 recipe

`validate_w4a4_recipe` (rdna-compute example) runs the full recipe through the
real fused iu4 kernel on gfx1103: SmoothQuant → FWHT-256 → clip-search int4
(g128), then per-K-group fused `gemm_iu4_i32_wmma` with per-group scale rescale
accumulated in f32. It checks GPU output against both the f32 reference (SQNR)
and a CPU sim of the identical scheme.

| M,K,B | CPU-sim SQNR | GPU SQNR | GPU vs CPU max-rel |
|-------|-------------:|---------:|-------------------:|
| 128,2048,64  | 21.29 | 21.29 | 8.6e-5 |
| 256,4096,32  | 20.06 | 20.06 | 4.7e-4 |
| 64,1024,128  | 22.58 | 22.58 | 2.1e-4 |
| 512,2048,16  | 21.45 | 21.45 | 4.8e-4 |

The fused-iu4 path realizes the recipe exactly (GPU == CPU sim) and the ~21 dB
W4A4 quality holds on real hardware — vs ~9 dB for naive W4A4. Grouped scales
are handled by K-tiling (one fused iu4 GEMM per K-group); a production kernel
would fold the per-group rescale into the epilogue instead of per-group launches.
Reproduce: `cargo run --release -p rdna-compute --example validate_w4a4_recipe`.

## E6 — Two named formats: Opus Quant and MQ+

`quant_opus_mqplus` compares the shipped MQ4 against two upgrades on identical
data (output SQNR dB; M=128 K=2048 B=64, representative):

| scheme | W | A | bits/w | compute | SQNR | vs MQ4 |
|--------|---|---|-------:|--------:|-----:|-------:|
| **MQ4** (as shipped) | affine4 | int8 | 4.25 | iu8 | 18.98 | — |
| **MQ+** | affine4 | int8 | 4.25 | iu8 | 24.82 | +5.8 |
| **Opus Quant** | sym4 | int4 | 4.13 | iu4 | 21.87 | +2.9 |
| Opus-A8 (ref) | sym4 | int8 | 4.13 | iu8 | 24.37 | +5.4 |

Stable across shapes (MQ+ +4.5…+8 dB over MQ4; Opus Quant ~2–3 dB below MQ+).

### MQ+  (= MQ4 + SmoothQuant + clip-search)
Keeps MQ4's affine-u4 / FWHT-256 / g256 format **and its iu8 GEMM kernel
unchanged**. Adds two offline/runtime-cheap steps:
- offline: clip-search the per-group range (MSE-optimal) instead of plain min/max;
- runtime: SmoothQuant per-channel activation rescale `X/s` (foldable into the
  preceding norm), with `W·s` folded offline.

Result: **+5.8 dB output SQNR for zero kernel work** — a drop-in quality upgrade
to MQ4 at the same 4.25 bits/w and the same W4A8-on-iu8 compute. This is the
quality-first prefill format.

### Opus Quant  (symmetric W4A4 on fused iu4 — "magnum → opus")
New compute path for max prefill throughput:
- symmetric signed-int4 weights (no zero-point) + clip-search, g128;
- SmoothQuant + FWHT-256 front-end (shared with MQ+);
- dynamic per-token **int4** activations, g32;
- fused `gemm_iu4_i32_wmma` (W4A4), per-group scale rescale in the epilogue.

~4.13 bits/w, validated end-to-end on gfx1103 (E5). It costs ~2–3 dB vs MQ+
(the intrinsic A4-vs-A8 gap) in exchange for the iu4 compute path. Affine-vs-
symmetric is a wash once SmoothQuant+clip are present (MQ+ ≈ Opus-A8), so the
symmetric choice is free — it exists only to enable signed iu4.

**Reuse map:** Opus Quant and MQ+ share the FWHT rotation, SmoothQuant, and
clip-search front-end. MQ+ reuses MQ4's storage + iu8 kernel verbatim. Opus
Quant forks the quantizer to symmetric and needs the new bits: a dynamic int4
activation quantizer and a grouped-iu4 GEMM with in-epilogue rescale.

Reproduce: `cargo run -p hipfire-quantize --example quant_opus_mqplus`.

## Net recommendation for gfx1103

- **Decode (memory-bound):** FWHT-rotated int4 weights, f16 activations (W4A16),
  dequant path — keep activation precision (free). The shipped MQ4 decode path
  already does this.
- **Prefill, quality-first:** **MQ+** (MQ4 + SmoothQuant + clip-search) — +5.8 dB
  over MQ4 for zero kernel work, same iu8 W4A8 compute. Low-risk, do this first.
- **Prefill, speed-first:** **Opus Quant** (symmetric W4A4 on fused iu4) — max
  throughput at ~2–3 dB below MQ+, validated end-to-end on gfx1103.
- **Fallback:** int8 (MQ8/q8) for outlier-sensitive tensors.
- **The next kernel lever is the GEMV**: optimize the generic
  GEMV for memory throughput (vectorized loads) to convert the 4-bit byte saving
  into the full ~4× decode speedup. Tracked as a follow-up to the generic kernel
  library.
- fp8/codebook (QTIP/Lloyd) explorations are lower priority here: fp8 has no
  gfx1103 hardware, and codebook schemes lose the native-int path while not
  beating rotation on outliers.
