# gfx1201 compact-head DeltaNet: 30 fewer AR dispatches

## Result

The gfx1201 Q8 DeltaNet decode path no longer materializes the model's 16
normalized Q/K heads as 32 repeated heads before recurrence. The recurrent
kernel maps state/value head `h` directly to Q/K head `h / 2`. This removes one
`repeat_interleave_qk_f32` launch and 32 KiB of Q/K write/read traffic from each
of the model's 30 linear-attention layers.

The retained-PM4 tape shrinks from 793 to 763 dispatches. The B2 occupancy
shape (`__launch_bounds__(32, 2)`) is enabled by default on gfx1201 Q8 plain AR;
`HIPFIRE_GDN_COMPACT2=0` restores the materialized path. Other architectures,
state formats, prefill, and speculative decode retain the existing path.

## Five-shape screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, five rows per shape:

| Minimum blocks/CU | Retained PM4 |
|---:|---:|
| 2 | **198.468 tok/s** |
| 4 | 194.886 tok/s |
| 8 | 194.678 tok/s |
| 12 | 197.859 tok/s |
| 16 | 194.494 tok/s |

B2 was then measured for 30 rows between two 30-row controls:

| Arm | Median tg128 |
|---|---:|
| materialized control | 195.744 tok/s |
| compact B2 | **196.677 tok/s** |
| materialized control, repeated | 196.044 tok/s |

Against the mean of the bracketing controls, the certified gain is +0.40%.
The five-row 198.468 result is retained as shape-selection evidence, not the
shipping claim.

## Correctness and cache boundary

- PM4 versus HIP shadow is exact over 15 consecutive positions.
- Logits, Q8 KV, recurrent state, and the reconstructed HIP kernarg-blob oracle
  all match.
- A separate materialized-control versus compact-B2 HIP pass matched every
  post-layer hidden byte at position 128 across all 40 layers (identical
  SHA-256 `7d65d90149649a9196b7d98f343ddbf8c1e534d2b93e30c86409a7a1f6f07aeb`).
- The sampled eight-turn session produced the exact same token counts and text
  in both arms: 8/8 normal stops, recall 3/3 on both recall turns, and zero
  runaway, empty, or attractor warnings.
- Default-kernel gfx1100 `.text` remains byte-identical before and after the
  parameterization (SHA-256
  `f634310a51dc6f59cefb6c2619e804036a9ad1cb7333ef725ace48f892357959`).
- The default non-compact gfx1201 kernel is likewise byte-identical (SHA-256
  `173ff813e89b48308406d375cea68ddc24a1dfc871b7ecb10b3fc2192a59137c`),
  so prefill/spec users of the original kernel do not inherit a codegen change.

The first PM4 replay attempt exposed a real boundary requirement. The removed
repeat kernel had also carried the vector-cache invalidation between
`fused_qk_l2_norm_scale_f32` and recurrence. A compute-idle wait alone produced
stale Q/K reads. Adding one gfx12 inter-node L0/L1 acquire at the new compact
GDN boundary restored exact parity; L2 remains resident.

## Eight-turn serve harness

The fixed-seed sampled session followed an identical 23.3k-context trajectory:

| Arm | Average decode | Median turn decode | Median prefill |
|---|---:|---:|---:|
| materialized control | 163.7 tok/s | 160.65 tok/s | 122.0 ms |
| compact B2 | **164.1 tok/s** | **161.0 tok/s** | 104.85 ms |

The decode improvement is consistent with the 30-row result. No prefill win is
claimed: this optimization is decode-only, and the control's turn-4 prefill was
a 482.8 ms outlier while every generated token remained identical.

Artifacts:

```text
hiptrx:/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/gdn-compact/
```
