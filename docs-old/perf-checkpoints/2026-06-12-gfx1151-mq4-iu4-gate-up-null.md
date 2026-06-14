# gfx1151 MQ4 IU4 / gate_up MMQ checkpoint

Date: 2026-06-12
Arch: `gfx1151` / Radeon 8060S
Model focus: Qwen3.5/Qwen3.6 MQ4 prefill

## Profile Baseline

`profile_prefill_qwen35` on `qwen3.5-9b-mq4.hfq`, `--prefill 256 --warmup 0 --kv-mode asym3`, showed the dense MQ4 path is dominated by the existing HFQ4 MMQ integer kernels:

| Kernel | Calls | Total |
|---|---:|---:|
| `gemm_hfq4g256_mmq_set` | 184 | ~133-136 ms |
| `gemm_hfq4g256_residual_mmq` | 64 | ~62 ms |

Disabling MMQ with `HIPFIRE_MMQ=0` made the same profile much worse by falling through to FP16 WMMA projection kernels, so the dense MQ4 upgrade path should stay inside the integer MMQ family.

## IU4 WMMA Finding

The vendored AMD Matrix Instruction Calculator reports `gfx1151` supports:

- `v_wmma_i32_16x16x16_iu8`
- `v_wmma_i32_16x16x16_iu4`

It does not support `v_wmma_i32_16x16x32_iu4`.

A compiler probe also confirmed the ROCm builtin surface accepts:

```c
__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, int32x2_t_a, true, int32x2_t_b, int32x8_t_c, true)
```

The important constraint is that `iu4` is 4-bit on both A and B. Current HFQ4/MQ4 MMQ multiplies 4-bit packed weights by Q8_1 int8 activations. That means `iu4` is not a drop-in replacement for the deployed MQ4 prefill path unless the activation quantization is also changed to 4-bit, which would be a different quality/error envelope.

## S4/IU4 Synthetic MMQ Null Result

After the standalone correction-path probe passed, a resident synthetic
benchmark compared the current Q8_1 + IU8 MMQ control against the signed-Q4
activation + IU4 probe on the same HFQ4 weights and an A3B-like projection
shape:

```bash
cargo run --release -p rdna-compute --example bench_gfx1151_hfq4_s4_mmq
```

Output:

```text
shape M=4096 K=2048 N=128 trials=9
q8_1_iu8_control_median_ms=0.1620
s4_iu4_probe_median_ms=0.1660
s4_vs_q8_speedup=0.976x
s4_vs_q8_max_abs=3.004774e-1
s4_vs_q8_rms=1.717572e-1
s4_vs_q8_rel_rms=1.654034e-1
```

This blocks routing the S4/IU4 probe into Qwen: on the first production-shaped
synthetic case it is slightly slower than the Q8_1 control and introduces a
large activation-Q4 drift. Any follow-on needs a better activation quantizer or
a memory-traffic change, not just native IU4 WMMA substitution.

## Fused gate_up MMQ Null Result

Hypothesis: because the dense FFN path quantizes X once and then launches two `gemm_hfq4g256_mmq_set_prequant` kernels for gate and up, a gfx1151 helper that launches the same body over `grid.z = 2` might reduce launch overhead without changing math.

Prototype:

- added a direct-call HIP entry point that reused `gemm_hfq4g256_residual_mmq_full_body<false>`
- selected gate vs up by `blockIdx.z`
- kept the same `iu8` WMMA math and the same prequantized Q8_1 activation scratch
- compared output byte-for-byte against the old two-call path for `M=4096`, `K=1024`, `B=128`

Correctness passed byte-exact, but the direct microbench was neutral:

```text
old_two_launch=113.8us
fused_z_launch=114.5us
speedup=0.994x
```

The helper was not kept in the tree because it adds kernel and dispatch surface without improving the path. If this idea is revisited, it should use a structurally fused kernel that shares more than launch setup, not a `grid.z` wrapper around the same per-output body.

## Speed Gate Note

During this investigation, `tests/speed-gate.sh --fast` repeatedly reported pp32 prefill around `555-560 tok/s` against a `gfx1151` floor of `561.2 tok/s`, even after the default dispatch path was restored. Decode remained at or above baseline. Treat that as a host-state or noise warning for this session, not evidence that the removed `gate_up` prototype affected default inference.
