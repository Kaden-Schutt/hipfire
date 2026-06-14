# MFMA HFQ4 GEMM proof-of-concept for gfx942 (MI300X)

## What

First MFMA-based kernel in hipfire's history. Establishes the
`V_MFMA_F32_16x16x16_F16` path for HFQ4 GEMM on gfx94x CDNA3.

Files (under `docs/investigations/2026-05-19-mfma-hfq4/`):
- `mfma_test.cpp` — minimal MFMA F32_16x16x16_F16 scaffold test
  (FP16 × FP16 GEMM, no quantization). Verifies the lane layout
  for inputs/output. **PASS at max_err = 8.9e-08** (float epsilon).
- `mfma_hfq4_test.cpp` — full HFQ4 dequant-in-register + MFMA test.
  Compares against scalar HFQ4 reference using same FP16 accumulator
  order. **PASS at max_rel_err = 2e-5** (within FP16 ULP).

Both are standalone hipcc programs:
```
hipcc -O2 --offload-arch=gfx942 mfma_hfq4_test.cpp -o mfma_hfq4_test
./mfma_hfq4_test
```

## Production kernel

`kernels/src/gemm_hfq4g256_residual_mfma.gfx942.hip` — the productionized
MFMA HFQ4 GEMM with residual add. Dispatch arm wired into
`gemm_hfq4g256_residual`, opt-in via `HIPFIRE_GFX942_MFMA_PREFILL=1`,
gated on `batch_size >= 16 && m % 16 == 0 && k % 256 == 0`.

## Benchmark vs rocBLAS Tensile (27B-3.6 mq4, batch=256 prefill)

| Config | prefill tok/s | wall ms |
|---|---:|---:|
| rocBLAS Tensile (default) | **1315** | 195 |
| MFMA-direct kernel (this work) | 1048 | 244 |
| Delta | **-20%** | +25% |

**Why slower than rocBLAS:** the v1 MFMA kernel is unoptimized vs years
of rocBLAS Tensile tuning. Specifically missing:
1. LDS shared B-tile loads (currently each lane loads B independently)
2. Larger output tiles per WG (currently 16x16 = 5120 WGs for our shape;
   should aim for 32x32 or 64x64 per WG)
3. K-direction prefetch / pipelining (currently MFMA stalls on loads)
4. Multiple MFMA chains per WG (4 wave64s could each do one chain)
5. `ds_read_b128` vectorized loads (currently scalar half loads)

## What we PROVED, even at 80% of rocBLAS

1. **MFMA intrinsics work on gfx942 in hipfire** (`__builtin_amdgcn_mfma_f32_16x16x16f16`)
2. **MFMA lane layout is documented and verified** — A operand m=l%16,
   B operand n=l%16, both k=(l/16)*4+r; OUTPUT m=(l/16)*4+r, n=l%16
3. **HFQ4 nibble unpacking + MFMA F32_16x16x16_F16 is a viable path**
   for skipping the FP16 dequant shadow buffer (saves 8.6 GB VRAM at
   64 layers)
4. **Channel-test methodology** for MFMA kernels works: scalar HFQ4
   reference + tolerance check vs FP16 accumulator order matches MFMA
   output at FP16 ULP precision

## Next steps (perf tuning)

To beat rocBLAS, the kernel needs:
1. **Larger tile per WG** — switch to 4 wave64s x 32x32 MFMA tile each
   = 64x64 output tile per WG. Cuts WG count 16x (5120 -> 320).
2. **LDS-cached B-tile** — load 64 cols x K_tile=8 of B into LDS once,
   shared across all 4 waves' MFMA calls in the WG.
3. **K-direction prefetch** — issue next-iter loads concurrent with
   current MFMA. CDNA3 MFMA has 8-cycle latency; lots of issue slots.
4. **`ds_read_b128`** — 16-byte LDS reads vs 2-byte scalar half loads.

Each is ~1-2 days of careful kernel work. Phase 2 of the MFMA workstream.
