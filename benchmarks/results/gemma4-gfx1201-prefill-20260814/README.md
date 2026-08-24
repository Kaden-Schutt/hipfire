# Gemma 4 E-Series gfx1201 Prefill Promotion

This evidence promotes three existing Gemma 4 E-series batched prefill routes on exact `gfx1201`: batched embedding lookup, batched PLE branch projections, and fused PLE activation. No new gfx12 kernel was added. The Q8 projection route uses the existing RDNA4 WMMA source; the embedding and activation kernels are architecture-generic.

## Environment

- GPU: AMD Radeon AI PRO R9700, `gfx1201`, 32 GiB
- HIP runtime: 7.14
- KV mode: Q8
- Q8 batched legacy path: disabled
- Batch: 64
- Workload: first 10 fixed GSM8K 8-shot prompts, 3 repeats, 16 output tokens
- Dataset SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- E2B artifact SHA256: `7dc98107c0e802036c30b1ee906f74a4214cbba75f81c7983e5bf184ee357a2d`
- E4B artifact SHA256: `0224a0e3240030c0b5d14dcbf4f3ace1c96bea3ae166e8372068c843f589e80a`
- Daemon SHA256: `d9d1ea4004614e9418940b3b0ad18c925d25f441dda294536b62eaa01950dc83`

## Result

| Model | Policy | Prefill median | Speedup | TTFT median | Decode median |
|---|---|---:|---:|---:|---:|
| E2B Q8 | explicit off | 1,365.125 tok/s | 1.00x | 574.190 ms | 96.97 tok/s |
| E2B Q8 | auto | 2,890.540 tok/s | 2.12x | 275.578 ms | 95.24 tok/s |
| E4B Q8 | explicit off | 989.360 tok/s | 1.00x | 790.524 ms | 65.04 tok/s |
| E4B Q8 | auto | 1,802.975 tok/s | 1.82x | 437.593 ms | 64.26 tok/s |

Direct sequential-versus-batched validation passed at batch 1, 2, 4, and 64 for both models, including last-token argmax and post-batch KV next-token argmax. A separate 30-question, 512-output-token GSM8K gate completed without errors and showed no measured accuracy regression. These are gfx1201-specific results; they do not validate gfx1200 or other gfx12 targets.

## Reproduction

Use `scripts/bench-gemma4-gfx12-prefill-routes-ab.sh` for isolated route A/B and `scripts/bench-gemma4-gfx12-prefill-quality-ab.sh` for the longer quality gate. Both scripts require the daemon to report `gfx1201`, pin the Q8 legacy policy off, and record model, daemon, and dataset hashes.
