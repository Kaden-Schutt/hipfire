# OCP FP8 recipe benchmark

`ocp_fp8_recipe_bench.hip` compares architecture-specific OCP FP8 WMMA
lowerings with a predecoded FP16 WMMA control:

- gfx11: OCP FP8 bytes are decoded to FP16 register fragments before the
  gfx11 FP16 WMMA builtin.
- gfx12: OCP FP8 bytes are consumed by the native gfx12 FP8/BF8 WMMA builtins.

The benchmark measures recipe-level kernels, not model serving performance.
Each row executes the same number of logical 16x16x16 WMMA operations within
one architecture. Absolute gfx11 and gfx12 times are not a direct GPU
comparison because their WMMA register layouts and devices differ.
`nominal_operand_gbps` is input bytes divided by kernel time, not a hardware
HBM counter.

`Relative to matching FP16` is computed as `FP16-control median / OCP median`,
so values above 1.0 mean that the OCP-input recipe is faster. The gfx11 FP16
control is not a previous OCP backend: it consumes operands predecoded to FP16
and calls the native gfx11 FP16 WMMA builtin. The new gfx11 OCP recipes instead
decode FP8 bytes inside the kernel before calling that same FP16 WMMA builtin.

## Correctness

Inputs are deterministic finite E4M3 or E5M2 values. Before reporting timing,
each OCP mode is compared with a matching predecoded FP16 reference over 4,096
tiles and every returned WMMA accumulator (`4096 * 32 * 8` values). The
validation launch is outside the timed region.

The gfx12 native instructions can differ slightly from staged FP16 WMMA due to
the arithmetic path. The gate uses `atol=0.5, rtol=1e-4` for E4M3 and
`atol=8192, rtol=1e-4` for E5M2, and reports the largest used fraction of that
tolerance as `max_tolerance_ratio`.

## Run

Check that the selected GPU is idle before starting. On a host with multiple
GPU architectures, set `ARCH` explicitly to match `HIP_VISIBLE_DEVICES`.

```bash
HIP_VISIBLE_DEVICES=1 \
ROCM_PATH=/opt/rocm/core-7.14 \
OUT=/tmp/gfx1100-w7900.csv \
./scripts/bench_ocp_fp8_recipes.sh
```

The default run uses 262,144 tiles, 20 warmup launches, 31 measured trials,
four launches per trial, and a five-second cooldown between modes. `MODES`,
`TILES`, `WARMUP`, `TRIALS`, `INNER`, and `COOLDOWN_SECS` are configurable.

## Results

ROCm 7.14 results collected on 2026-07-30:

| GPU | Mode | Median us | p10-p90 us | Relative to matching FP16 |
|---|---:|---:|---:|---:|
| W7900 / gfx1100 | E4M3 manual decode | 737.273 | 727.404-740.154 | 1.028x |
| W7900 / gfx1100 | E4M3 HIP convert | 720.694 | 716.424-732.314 | 1.052x |
| W7900 / gfx1100 | E5M2 bit decode | 384.162 | 383.772-387.222 | 1.973x |
| W7900 / gfx1100 | FP16 E4M3 control | 757.974 | 757.214-760.964 | 1.000x |
| W7900 / gfx1100 | FP16 E5M2 control | 757.934 | 757.314-761.114 | 1.000x |
| R9700 / gfx1201 | native E4M3 WMMA | 221.743 | 221.573-222.053 | 1.957x |
| R9700 / gfx1201 | native E5M2 WMMA | 221.773 | 221.583-222.013 | 1.949x |
| R9700 / gfx1201 | FP16 E4M3 control | 433.856 | 433.686-434.006 | 1.000x |
| R9700 / gfx1201 | FP16 E5M2 control | 432.366 | 432.146-432.706 | 1.000x |

The native gfx12 path retains almost the full twofold operand-size advantage.
On gfx11, E5M2's bit-shift decode also retains most of it, while E4M3 decode
cost consumes nearly all of the advantage. These data support exposing a
fail-closed gfx11 lowering recipe, but they do not justify selecting every
gfx11 format unconditionally in a production kernel.

The generated ISA contains gfx11 `v_wmma_f32_16x16x16_f16` and gfx12 native
`v_wmma_f32_16x16x16_fp8_fp8` / `bf8_bf8` instructions. Timed gfx11 kernels
use 26-38 VGPRs and 18 SGPRs; timed gfx12 kernels use 13-17 VGPRs and 12 SGPRs.
All audited variants are wave32 with zero private segment, SGPR spills, and
VGPR spills.

Raw outputs are stored under `bench/results/`.
