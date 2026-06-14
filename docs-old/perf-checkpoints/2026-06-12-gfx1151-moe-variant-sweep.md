# gfx1151 MoE grouped GEMM variant sweep

Date: 2026-06-12
Arch: `gfx1151` / Radeon 8060S
Model focus: Qwen3.5-122B-A10B MQ4 MoE prefill

## Result

The current gfx1151 defaults remain the best measured route for 122B pp64:

- HFQ6 grouped MoE keeps the 4-warp gfx1151 WMMA kernel enabled by default.
- HFQ4 grouped MoE keeps the k8 gfx1151 MMQ kernel enabled by default.
- HFQ6 v1/v2, HFQ4 k4, and HFQ4 k8-4w stay as opt-in experiment paths.

Command:

```bash
source scripts/gpu-lock.sh && gpu_acquire chaingun-gfx1151-moe-variant-profile
for spec in \
  "base::" \
  "hfq6_v1::HIPFIRE_MOE_HFQ6_4W=0" \
  "hfq6_v2::HIPFIRE_MOE_HFQ6_4W=0 HIPFIRE_MOE_HFQ6_V2=1" \
  "hfq4_k4::HIPFIRE_MOE_GROUPED_I8_K8=0 HIPFIRE_MOE_GROUPED_I8_K4=1" \
  "hfq4_4w::HIPFIRE_MOE_GROUPED_I8_4W=1"; do
  tag=${spec%%::*}
  envs=${spec#*::}
  if [ -n "$envs" ]; then
    env $envs cargo run --release --features deltanet -p hipfire-runtime \
      --example profile_prefill_qwen35 -- \
      ~/.hipfire/models/qwen3.5-122b-a10b-mq4.hfq \
      --prefill 64 --warmup 0 --kv-mode asym3
  else
    cargo run --release --features deltanet -p hipfire-runtime \
      --example profile_prefill_qwen35 -- \
      ~/.hipfire/models/qwen3.5-122b-a10b-mq4.hfq \
      --prefill 64 --warmup 0 --kv-mode asym3
  fi
done
gpu_release
```

Saved run log:

```text
/tmp/gfx1151-moe-variant-profile-20260612-092417.log
```

## Profile Summary

| Variant | Env | Total profiled prefill | Dominant MoE kernel | MoE kernel total |
|---|---|---:|---|---:|
| default | unset | 294.2 ms | `gemm_hfq6g256_moe_grouped_wmma_4w_gfx1151` | 86.8 ms |
| default | unset | 294.2 ms | `gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151` | 71.2 ms |
| HFQ6 v1 | `HIPFIRE_MOE_HFQ6_4W=0` | 345.2 ms | `gemm_hfq6g256_moe_grouped_wmma_gfx1151` | 144.1 ms |
| HFQ6 v2 | `HIPFIRE_MOE_HFQ6_4W=0 HIPFIRE_MOE_HFQ6_V2=1` | 374.1 ms | `gemm_hfq6g256_moe_grouped_wmma_v2_gfx1151` | 177.3 ms |
| HFQ4 k4 | `HIPFIRE_MOE_GROUPED_I8_K8=0 HIPFIRE_MOE_GROUPED_I8_K4=1` | 307.3 ms | `gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151` | 85.8 ms |
| HFQ4 k8 4w | `HIPFIRE_MOE_GROUPED_I8_4W=1` | 302.5 ms | `gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151` | 78.7 ms |

## Interpretation

The HFQ6 4-warp kernel is a large win on this 122B shape: disabling it regressed
the grouped HFQ6 bucket from 86.8 ms to 144.1 ms for v1 and 177.3 ms for v2.
The HFQ4 k8 path is also still the best HFQ4 grouped MoE route at pp64. The k4
variant cuts less inner-loop state but loses throughput, and the k8 4-warp
variant does not recover enough shared-X reuse to beat the single-wave k8
kernel.

The next MoE kernel work should move to a different lever rather than reusing
these env-gated variants. Good candidates are Q4-activation/IU4 prototypes for
uniform MQ4/HFQ4 or a new grouped layout that changes memory traffic rather than
only retiming the existing k4/k8 loops.
