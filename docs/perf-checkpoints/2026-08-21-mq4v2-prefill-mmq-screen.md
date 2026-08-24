# MQ4V2 prefill MMQ screen — gfx1100 / gfx1151 / gfx1201

**Date:** 2026-08-21  
**Lifecycle:** `historical`  
**Disposition:** mechanism screen only; not an admission decision or current default  
**Source:** `tools/quant-design/bench_mq4v2_prefill_mmq.hip`

## Question

Does MQ4V2's dual-FP16 affine header make a Q8_1 activation + i8-WMMA MMQ prefill path competitive with Hipfire's current F16-WMMA path, and can it explain the remaining full-model prefill gap?

The benchmark compares five arms over the same packed payload and output shape:

1. MQ4V2 F16-WMMA, preconverted F16 activation (`f16_compute`)
2. F32→F16 plus MQ4V2 F16-WMMA (`f16_e2e`)
3. MQ4V2 Q8_1 + i8-WMMA MMQ, prequantized activation (`mmq_compute`)
4. F32→Q8_1 plus MQ4V2 MMQ (`mmq_e2e`)
5. legacy HFQ4 F32-header MMQ (`hfq4_mmq_legacy`) as the header-representation control

The V2 MMQ affine correction is

```text
s_half * d_x * dot(q_w, q_x) + z_half * sum_real_x
```

with `(s0,z0)` for K positions 0–127 and `(s1,z1)` for positions 128–255.

## Method

- Fleet fatbin: `gfx1030,gfx1100,gfx1151,gfx1201`; unsupported gfx1030 exits without submitting a kernel.
- Correctness gate before timing: a dedicated M=32, K=512, N=16 allocation; canary immediately after the active 512-float output; intentionally disjoint V2 half headers; independent CPU f64 references.
- Q8_1 stores production-faithful `d * sum(rounded_q8)` metadata. F16-WMMA uses the original-F32 reference; MMQ uses the Q8_1-dequantized activation reference.
- A second header-control fixture uses identical nibble payloads and equivalent affine values: one legacy F32 `(s,z)` pair repeated into both V2 FP16 halves.
- Accepted rel-RMS on all measured architectures:
  - F16-WMMA disjoint-half fixture: `0.000243`
  - V2 MMQ disjoint-half fixture: `0.000120`
  - V2/legacy identical header-control fixture: `0.000197–0.000264`
- Timing: HIP events, 32 warmups, 100 measured iterations, five repeats; medians below.
- `ratio = mmq_e2e / f16_e2e`; below 1.0 means MMQ is faster.

Raw evidence, all from the revalidated hipx fleet:

- gfx1201 square: `/home/kaden/qcal/mq4v2-prefill-mmq-gfx1201-20260821.txt`
- gfx1201 FFN: `/home/kaden/qcal/mq4v2-prefill-mmq-ffn-gfx1201-20260821.txt`
- gfx1100 square: `/home/kaden/qcal/mq4v2-prefill-mmq-gfx1100-20260821.txt`
- gfx1100 FFN: `/home/kaden/qcal/mq4v2-prefill-mmq-ffn-gfx1100-20260821.txt`
- gfx1151 square: `/home/kaden/qcal/mq4v2-prefill-mmq-gfx1151-20260821.txt`
- gfx1151 FFN: `/home/kaden/qcal/mq4v2-prefill-mmq-ffn-gfx1151-20260821.txt`

A pre-fix gfx11 partial-tile run was rejected after a non-uniform barrier path. The accepted source uses uniform 256-thread barrier participation and zero-pads partial rows/columns; no pre-fix timing is included.

## Square projection — M=5120, K=5120

| GPU | N | F16 e2e µs | MMQ e2e µs | ratio | disposition |
|---|---:|---:|---:|---:|---|
| gfx1201 R9700 | 16 | 46.000 | 57.320 | 1.246 | F16 wins |
| gfx1201 R9700 | 128 | 181.981 | 248.281 | 1.364 | F16 wins |
| gfx1201 R9700 | 512 | 601.484 | 1,172.728 | 1.950 | F16 wins |
| gfx1201 R9700 | 2,048 | 2,547.139 | 5,329.381 | 2.092 | F16 wins |
| gfx1151 8060S | 16 | 71.013 | 282.008 | 3.971 | F16 wins |
| gfx1151 8060S | 128 | 404.979 | 386.925 | 0.955 | MMQ +4.7% |
| gfx1151 8060S | 256 | 819.997 | 758.862 | 0.925 | MMQ +8.1% |
| gfx1151 8060S | 512 | 1,696.058 | 1,559.362 | 0.919 | MMQ +8.8% |
| gfx1151 8060S | 1,024 | 3,535.946 | 3,909.487 | 1.106 | F16 wins |
| gfx1151 8060S | 2,048 | 7,236.241 | 7,740.005 | 1.070 | F16 wins |
| gfx1100 7900 XTX | 16 | 88.360 | 225.839 | 2.556 | F16 wins |
| gfx1100 7900 XTX | 256 | 410.898 | 400.958 | 0.976 | MMQ +2.5% |
| gfx1100 7900 XTX | 512 | 784.357 | 832.197 | 1.061 | F16 wins |
| gfx1100 7900 XTX | 1,024 | 1,525.533 | 1,638.713 | 1.074 | F16 wins |
| gfx1100 7900 XTX | 2,048 | 3,139.267 | 2,378.032 | 0.758 | MMQ +32.0% |

## FFN gate/up projection — M=17408, K=5120

This is the relevant large-output shape for Qwen3.8's dense FFN.

| GPU | N | F16 e2e µs | MMQ e2e µs | ratio | MMQ change |
|---|---:|---:|---:|---:|---:|
| gfx1201 R9700 | 128 | 553.166 | 1,309.754 | 2.368 | −57.8% |
| gfx1201 R9700 | 256 | 1,096.073 | 2,716.911 | 2.479 | −59.7% |
| gfx1201 R9700 | 512 | 2,191.366 | 5,205.803 | 2.375 | −57.9% |
| gfx1201 R9700 | 1,024 | 4,433.397 | 11,388.652 | 2.569 | −61.1% |
| gfx1201 R9700 | 2,048 | 8,916.007 | 23,313.830 | 2.615 | −61.8% |
| gfx1151 8060S | 128 | 1,976.925 | 1,821.052 | 0.921 | +8.6% |
| gfx1151 8060S | 256 | 4,106.115 | 3,320.603 | 0.809 | +23.7% |
| gfx1151 8060S | 512 | 8,530.746 | 6,445.819 | 0.756 | +32.3% |
| gfx1151 8060S | 1,024 | 17,572.289 | 12,176.998 | 0.693 | +44.3% |
| gfx1151 8060S | 2,048 | 35,760.789 | 23,473.045 | 0.656 | +52.3% |
| gfx1100 7900 XTX | 128 | 671.359 | 829.739 | 1.236 | −19.1% |
| gfx1100 7900 XTX | 256 | 1,293.979 | 1,246.759 | 0.964 | +3.8% |
| gfx1100 7900 XTX | 512 | 2,588.139 | 2,449.419 | 0.946 | +5.7% |
| gfx1100 7900 XTX | 1,024 | 5,340.380 | 4,038.521 | 0.756 | +32.2% |
| gfx1100 7900 XTX | 2,048 | 10,849.574 | 7,398.714 | 0.682 | +46.6% |

`MMQ change` is `(F16/MMQ)-1`, not one minus the raw ratio.

## Findings

1. **The header hypothesis is falsified as the primary cause.** Legacy F32-header and V2 dual-FP16-header MMQ timings are generally close. Header decode does not explain the full-model prefill gap.
2. **gfx1201 should stay on F16-WMMA.** The current MMQ implementation loses at every screened N, increasingly at long batches.
3. **gfx11 has a real large-output MMQ lever.** On the 17,408-row FFN shape, MMQ becomes useful at N≈256 on gfx1100 and N≈128 on gfx1151, reaching large wins at N≥1024.
4. **A global MMQ switch would regress short prompts and some square projections.** Any production experiment must be operation-, shape-, architecture-, and batch-gated.
5. **This screen does not explain llama.cpp-scale prompt throughput by itself.** The maximum wins apply to individual large-output GEMMs. Full-model profiling still owes attention/DeltaNet, launch count, activation transforms, and fused-family coverage.

## Candidate production experiment

Not a default: route only MQ4V2 dense FFN gate/up prefill to Q8_1 MMQ under a reversible gate:

- gfx1151: `N >= 128`
- gfx1100: `N >= 256`
- gfx1201: disabled

Then run a full-model, byte-identical prompt-length sweep before admission. Fused gate/up should quantize the activation once and reuse the Q8_1 buffer for both projections. Do not infer a product tok/s claim from this kernel-only screen.
