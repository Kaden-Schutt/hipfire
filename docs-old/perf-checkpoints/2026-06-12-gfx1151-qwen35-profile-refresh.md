# gfx1151 Qwen3.5/3.6 Profile Refresh

Date: 2026-06-12
Branch: `chaingun`
GPU: `gfx1151` (Strix Halo / Radeon 8060S)

## Purpose

Refresh the gfx1151 Qwen3.5/Qwen3.6 kernel priority list after the
`rmsnorm`, `fused_rmsnorm_mq_rotate`, GEMV rows, and DeltaNet conv/tree
work. The goal was to check whether the remaining scalar-looking gaps
(`gated_norm`, `fused_silu_mul_mq_rotate`, DeltaNet gate/conv leftovers)
are still worth targeting before the matrix kernels.

## Commands

```bash
HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_DECODE=1 HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  cargo run --release --features deltanet -p hipfire-runtime --example bench_qwen35_speed -- \
  ~/.hipfire/models/qwen3.5-9b-mq4.hfq --prefill 256 --prefill-runs 1 --warmup 3 --gen 12

HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_DECODE=1 HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  cargo run --release --features deltanet -p hipfire-runtime --example bench_qwen35_speed -- \
  ~/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq --prefill 128 --prefill-runs 1 --warmup 2 --gen 6

HIPFIRE_PROFILE=1 HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_MOE_GROUPED_I8_4W={0,1} \
  target/release/examples/bench_qwen35_speed \
  ~/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq --prefill 128 --prefill-runs 1 --warmup 0 --gen 0
```

Raw logs:

- `/tmp/gfx1151-qwen35-9b-profile-20260612-100313.log`
- `/tmp/gfx1151-qwen36-a3b-profile-20260612-100343.log`
- `/tmp/gfx1151-a3b-k8-base-20260612-100453.log`
- `/tmp/gfx1151-a3b-k8-4w-20260612-100453.log`

## Qwen3.5-9B MQ4

Shape: `prefill=256`, `warmup=3`, `gen=12`

Prefill profile:

| Kernel | Calls | Total | Share |
|---|---:|---:|---:|
| `gemm_hfq4g256_mmq_set` | 184 | 133.6 ms | 60.7% |
| `gemm_hfq4g256_residual_mmq` | 64 | 61.9 ms | 28.1% |
| `gated_delta_net_q8_batch_seq` | 24 | 9.5 ms | 4.3% |
| `fused_silu_mul_mq_rotate_batched` | 32 | 4.8 ms | 2.2% |
| `fused_rmsnorm_mq_rotate_gfx1151_batched` | 64 | 2.3 ms | 1.0% |
| `conv1d_silu_split_f32_n_gfx1151` | 24 | 1.5 ms | 0.7% |
| `gated_norm_f32_batched` | 24 | 1.0 ms | 0.4% |

Decode profile:

| Kernel | Calls | Total | Share |
|---|---:|---:|---:|
| `gemv_hfq4g256_residual` | 768 | 69.6 ms | 38.4% |
| `fused_qkvza_hfq4g256` | 288 | 37.5 ms | 20.7% |
| `gemv_hfq4g256` | 12 | 29.6 ms | 16.3% |
| `fused_rmsnorm_mq_rotate_gfx1151` | 768 | 11.7 ms | 6.5% |
| `fused_qkv_hfq4g256` | 96 | 10.6 ms | 5.8% |
| `gated_norm_f32` | 288 | 2.2 ms | 1.2% |
| `fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151` | 288 | 2.1 ms | 1.2% |

Summary: scalar/norm leftovers are no longer the first-order target on 9B.
HFQ4 GEMV/fused-projection decode and HFQ4 MMQ prefill dominate.

## Qwen3.6-35B-A3B MQ4

Shape: `prefill=128`, `warmup=2`, `gen=6`

Prefill profile:

| Kernel | Calls | Total | Share |
|---|---:|---:|---:|
| `gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151` | 80 | 43.9 ms | 45.6% |
| `gemm_hfq4g256_mmq_gfx1151_set` | 230 | 20.1 ms | 20.9% |
| `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` | 40 | 7.2 ms | 7.5% |
| `gemm_hfq4g256_residual_mmq` | 40 | 6.6 ms | 6.9% |
| `gated_delta_net_q8_batch_seq` | 30 | 6.1 ms | 6.4% |
| `fused_silu_mul_mq_rotate_batched` | 80 | 0.9 ms | 0.9% |
| `gated_norm_f32_batched` | 30 | 0.7 ms | 0.7% |

Decode profile:

| Kernel | Calls | Total | Share |
|---|---:|---:|---:|
| `fused_qkvza_hfq4g256` | 180 | 12.4 ms | 15.7% |
| `gemv_hfq4g256_moe_gate_up_k8_indexed` | 240 | 11.7 ms | 14.8% |
| `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` | 240 | 8.3 ms | 10.6% |
| `gemv_hfq4g256` | 6 | 7.4 ms | 9.4% |
| `gemv_hfq4g256_residual` | 240 | 6.8 ms | 8.6% |
| `mq_rotate_x` | 726 | 4.1 ms | 5.2% |
| `fused_qkv_hfq4g256` | 60 | 3.2 ms | 4.1% |
| `gated_norm_f32` | 180 | 1.4 ms | 1.7% |
| `fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151` | 180 | 1.4 ms | 1.7% |

Summary: A3B has two active matrix priorities: grouped HFQ4 MMQ prefill and
HFQ4 indexed/fused GEMV decode. The norm/gate/conv leftovers remain useful
cleanup targets, but they do not outrank matrix work in the current profile.

## 4-Warp Grouped-MMQ Probe

The existing `HIPFIRE_MOE_GROUPED_I8_4W=1` experiment was retested on the
current A3B pp128 shape.

| Mode | Grouped kernel | Grouped total | Profile total | Prefill tok/s |
|---|---|---:|---:|---:|
| `HIPFIRE_MOE_GROUPED_I8_4W=0` | `gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151` | 44.0 ms | 96.7 ms | 663.9 |
| `HIPFIRE_MOE_GROUPED_I8_4W=1` | `gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151` | 50.1 ms | 103.1 ms | 642.6 |

Conclusion: keep the current k8 default. The 4-warp variant still loses on
the active A3B profile shape, matching the earlier A3B/122B sweeps.

## Updated Priority

1. HFQ4/MQ4 decode GEMV family on gfx1151: standalone, residual, fused QKV,
   fused QKVZA, and indexed MoE gate/down.
2. HFQ4 grouped MMQ prefill on gfx1151: k8 remains the best shipped route,
   but it is the top A3B prefill row.
3. Qwen3.5/3.6 scalar cleanup: `gated_norm`, `fused_silu_mul_mq_rotate`, and
   routed DeltaNet conv/gate BF16 policy remain open, but current profiles put
   them below matrix work.
