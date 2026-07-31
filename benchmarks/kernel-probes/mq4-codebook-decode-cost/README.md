# MQ4 codebook decode-cost probe

Answers one question: **does replacing MQ4G256's per-group affine
`(f32 scale, f32 min)` with a symmetric baked-codebook + g64 E4M3
sub-scale layout cost decode speed?**

Both layouts are **exactly 136 B per 256 weights**, so memory traffic is
identical by construction and any measured delta is pure ALU/LDS.

## Layouts compared

| | header (8 B) | payload | total |
|---|---|---|---|
| current | `[f32 scale][f32 min]` | 128 B nibbles | 136 B |
| candidate | `[f32 master][4 x u8 E4M3 g64]` | 128 B nibbles | 136 B |

The candidate is symmetric (the FWHT makes the zero-point nearly useless)
and spends the freed 4 B on 4x finer scale granularity. Because the
codebook absorbs the scale, the per-weight multiply collapses to one FMUL
per 8 weights instead of a per-weight FMA — which is why it can come out
*ahead* despite doing a table lookup.

## Files

| file | purpose |
|---|---|
| `lut_strategy_sweep.hip` | how to hold the 16-entry codebook: LDS+`__syncthreads`, LDS+`wave_barrier`, or `__constant__` indexed directly |
| `decode_cost_qwen35_0.8b.hip` | time-weighted verdict over the real Qwen3.5-0.8B per-token GEMV stack |

## Build + run

```
hipcc -O3 --offload-arch=gfx1201 -std=c++17 decode_cost_qwen35_0.8b.hip -o dc
./dc 640 5          # iters-per-shape, reps
```

## Methodology notes (learned the hard way)

1. **Interleave the variants inside the timing loop.** Timing kernel A to
   completion and then kernel B lets DPM/thermal ramp alias onto the
   variant. A sequential loop produced a +6% swing on identical code.
2. **Batch launches per timed region.** These per-layer GEMVs run ~5-10 us
   and ROCm launch overhead is ~3-5 us of that, so one-launch-then-sync
   measures the launch path, not the kernel. `B=64` back-to-back launches
   with a single sync also matches how decode actually enqueues.
3. **Randomize the weight payload.** A `hipMemset` constant fill gives every
   lane the same LUT index, which hides LDS bank conflicts entirely and
   flatters the codebook variant.
4. **Weight by per-token invocation count.** Per-shape percentages on an
   8 us kernel are noise; the decision metric is the weighted total.
5. Discard the first run — the first pass is still clock-ramping.

## Results (RX 9070 XT / gfx1201, 3 stable runs)

Weighted total, per token:

| accounting | delta |
|---|---|
| excl. lm_head (lm_head at Q8) | **-1.3% to -3.1%** |
| incl. lm_head at MQ4 | **-3.4% to -5.4%** |

Negative = faster than current. Budget was +1.0%.

Per op, with the codebook baked as immediates:

| op | delta |
|---|---|
| `fused_qkv [3072,1024]` | +0.4% .. +2.2% |
| `o_proj [1024,2048]` | -3.6% .. -4.5% |
| `fused_gate_up [7168,1024]` | -2.7% .. -3.4% |
| `down_proj [1024,3584]` | -6.4% .. -7.5% |
| `lm_head [248320,1024]` | -10.0% .. -10.7% |

## Two findings that constrain the implementation

1. **Bake the codebook as immediates; do NOT read it from `__constant__`.**
   Indexing `__constant__` with a divergent index compiles to a gather:
   +23% mean, up to +96% worst case. Even filling LDS *from*
   `__constant__` once per block costs real time on short kernels —
   `fused_qkv` went from +3.7..+4.8% down to +0.4..+2.2% purely by
   replacing that read with a VALU-only fill. The lookup was never the
   problem; the per-block memory read was.

2. **A 138 B group (g32 sub-scales) is dead on arrival.** Measured +5% to
   +60% depending on shape — far worse than the +1.47% byte delta implies,
   because the 138 B stride wrecks coalescing. Only 136 B variants qualify.

## Scope

This probe covers the **decode GEMV path only**. MQ4 also has a WMMA GEMM
family used for prefill (`gemm_*_mq4g256_lloyd_wmma*`); adopting a codebook
would need those too, and they are not measured here.

The codebook values in the probe are placeholders — the access pattern, not
the values, determines timing.
