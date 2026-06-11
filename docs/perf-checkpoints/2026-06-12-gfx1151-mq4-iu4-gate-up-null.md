# gfx1151 MQ4 IU4 / gate_up MMQ checkpoint

Date: 2026-06-12
Arch: `gfx1151` / Radeon 8060S
Model focus: Qwen3.5/Qwen3.6 MQ4 prefill

## Profile Baseline

`profile_prefill_qwen35` on `qwen3.5-9b.mq4.hfq`, `--prefill 256 --warmup 0 --kv-mode asym3`, showed the dense MQ4 path is dominated by the existing HFQ4 MMQ integer kernels:

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
