# gfx1201 conv1d + Q/K normalization fusion: 201 tok/s AR

## Result

The gfx1201 Q8 plain-AR path now fuses `conv1d_silu_split_f32` with
`fused_qk_l2_norm_scale_f32`. Q/K conv results stage in 1 KiB LDS, retain the
existing wave reduction and two-step Q scaling order, and reach global memory
only once in normalized form. V remains in the same launch.

This removes one dispatch per linear-attention layer and approximately 32 KiB
of Q/K intermediate cache traffic per layer (the unnormalized write plus the
normalizer read), or about 960 KiB per token over A3B's 30 DeltaNet layers. The
retained-PM4 tape shrinks from 763 to 733 dispatches.

The B256 shape is enabled by default only for gfx1201, Q8 state, 128-wide Q/K
heads, and the lowered plain-AR path. `HIPFIRE_CONV_QKNORM=0` restores the old
two-kernel sequence. Prefill, speculative decode, other state formats, and all
other architectures remain on the existing kernels.

## Five-shape screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured tokens, five rows per shape:

| Workgroup | Retained PM4 | VGPR | SGPR | LDS |
|---:|---:|---:|---:|---:|
| 32 | 199.797 tok/s | 28 | 21 | 1 KiB |
| 64 | 200.107 tok/s | 30 | 22 | 1 KiB |
| 128 | 197.902 tok/s | 22 | 14 | 1 KiB |
| 256 | **201.668 tok/s** | 20 | 16 | 1 KiB |
| 512 | 199.775 tok/s | 20 | 18 | 2 KiB |

These are substantive shapes: 32/64/128-thread variants assign multiple Q and
K channels to each lane, B256 pairs one full Q/K head in one workgroup, and
B512 packs two heads per workgroup.

## Thirty-row certification

| Arm | Median tg128 |
|---|---:|
| two-kernel control | 196.520 tok/s |
| fused B256 | **201.174 tok/s** |
| two-kernel control, repeated | 198.212 tok/s |

The shipping result is +1.93% against the mean of the bracketing controls.

## Correctness and numerical drift

- PM4 versus HIP shadow is exact over 15 consecutive positions.
- The same 15-position shadow passes with the shipping default (both fusion
  environment variables unset), selecting B256 at 733 dispatches / 23 kernels.
- Replay capture is stable at 733 dispatches / 23 unique kernels; all replay
  contracts pass.
- The fusion is not byte-identical to the materialized two-kernel oracle
  because compiling conv and normalization together changes some conv results
  by roughly one ULP. Drift at position 128 is tightly bounded:
  - layer 0: max absolute `1.49e-8`, RMS `2.53e-9`, cosine effectively 1.0;
  - layer 31: max absolute `7.62e-5`, RMS `1.70e-5`, cosine `0.999999987`;
  - layer 39: max absolute `5.69e-4`, RMS `1.61e-4`, cosine `0.999999932`.
- No scratch/private-memory spill in any valid shape.
- Existing gfx1100 kernel sources are untouched; selection is runtime-gated to
  gfx1201, so the gfx1100 path remains byte-identical.

## Eight-turn sampled serve harness

Both Q8 arms used registry sampling, medium thinking, a 4096-token output cap,
and the fixed session seed.

| Arm | Average decode | Median decode | Median prefill |
|---|---:|---:|---:|
| control | 164.6 tok/s | 161.5 tok/s | 104.45 ms |
| fused B256 | **167.8 tok/s** | **165.1 tok/s** | 101.5 ms |

The tiny logit drift produced a different but coherent sampled trajectory, so
the session rates are not an equal-token A/B. The product gate passed: 8/8
normal stops, zero runaway/empty/attractor warnings, and recall 3/3 on both
recall turns. No prefill improvement is claimed because contexts diverged and
the fusion is decode-only.

Artifacts:

```text
hiptrx:/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/conv-qknorm/
```
