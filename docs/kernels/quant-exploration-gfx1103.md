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

## Net recommendation for gfx1103

- **Weights: FWHT-rotated symmetric int4 (MQ4) at ~4.06 bits/w**, group scales,
  with int8 (MQ8/q8) fallback for outlier-sensitive tensors. Matches existing
  formats; no new format needed.
- **The next lever is the GEMV kernel, not the format**: optimize the generic
  GEMV for memory throughput (vectorized loads) to convert the 4-bit byte saving
  into the full ~4× decode speedup. Tracked as a follow-up to the generic kernel
  library.
- fp8/codebook (QTIP/Lloyd) explorations are lower priority here: fp8 has no
  gfx1103 hardware, and codebook schemes lose the native-int path while not
  beating rotation on outliers.
