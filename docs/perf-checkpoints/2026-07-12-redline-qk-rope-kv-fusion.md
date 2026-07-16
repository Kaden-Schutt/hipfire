# gfx1201 Q/K norm + RoPE + K-cache fusion: exact, slower

## Verdict

Reverted. Two bit-exact implementations reduced the 833-dispatch retained tape
to 803 and 813 dispatches, respectively, but neither beat the single-stream PM4
control. The best exact shape was neutral-to-negative at tg128 and regressed the
eight-turn session. Full-attention Q/K norm + RoPE + K-cache fusion is closed
for this A3B/Q8 path in its current form.

## Live control topology

Each of A3B's ten full-attention layers executes:

```text
Q weighted RMSNorm
K weighted RMSNorm
half-split RoPE(Q, K)
Q8 K-cache write
Q8 V-cache write
attention
```

The requested fusion only affects single-token decode. Batched prefill uses a
separate source path and was unchanged throughout this experiment.

## Correctness investigation

The first per-head one-kernel implementation produced a stable 803-dispatch
tape and was PM4-vs-HIP exact, but differed from the control oracle. Binary
ablation localized the difference:

- fused RMSNorm + stock RoPE: control-identical logits, KV, and recurrent hashes;
- fused per-head RoPE + stock Q8 writer: different hashes;
- direct Q8 packing was therefore not the initial source of divergence.

Recomputing `pow/cos/sin` inside each per-head workgroup changed the compiled
RoPE path enough to violate the bit-exact gate even though the source formulas
matched. Two exact alternatives followed.

### Exact 803-dispatch arm

One 256-thread workgroup serialized all 18 Q/K head reductions, then wave zero
executed the stock RoPE loop order, and all eight waves packed K into Q8. This
preserved the stock execution shape and matched the control's 15-position
logits/KV/recurrent hashes exactly.

It lost because serializing 18 independent RMS reductions inside one workgroup
discarded more GPU parallelism than the 30 removed launches were worth:

| Arm | Ordinary HIP | Retained PM4 |
|---|---:|---:|
| control | **165.718 tok/s** | **192.896 tok/s** |
| exact one-workgroup fusion | 158.011 tok/s | 189.292 tok/s |

Retained PM4 regressed 1.87%.

### Exact 813-dispatch arm

The second design retained parallelism:

```text
fused Q/K RMSNorm: 18 independent head workgroups
fused stock-shaped RoPE + direct Q8 K-cache packing: one wave
ordinary Q8 V-cache write
```

This was exact against both PM4/HIP and the control oracle for all 15 positions.
It removes 20 launches per token. Two 30-row candidate runs bracketed an
intervening control:

| Order | Retained PM4 |
|---|---:|
| candidate | 192.556 tok/s |
| control | **193.608 tok/s** |
| candidate | 190.737 tok/s |

The first run was only 0.54% below the matched control, but the repeat fell
1.48%. Both miss the 195 tok/s product bar; there is no reproducible win.

## Eight-turn serve result

The exact two-stage candidate produced the same sampled token counts and
coherent outputs with zero runaways, empties, or attractors.

| Arm | Average decode | Final decode |
|---|---:|---:|
| retained-PM4 control | **166.013 tok/s** | **148.0 @ 20,069** |
| two-stage fusion | 164.575 tok/s | 147.0 @ 20,069 |

That is a 0.87% average regression and 0.68% at the final long-context turn.

The displayed session `prefill_ms` values are not a full-prefill benchmark:
they time only each turn's variable uncached suffix at the existing context.
No prefill performance claim is made. The candidate gate was decode-only, so
the batched prefill kernel sequence was source-identical to control.

## Mechanism and boundary

The retained PM4 tape has already made individual launch overhead small. The
fusion saves 20--30 dispatch packets, but exactness requires preserving the
stock RoPE execution shape and conservative producer visibility boundaries.
The remaining kernel work and fences offset the packet savings. Relaxing the
RoPE execution shape creates a numerically different path; serializing all
heads restores exactness but loses occupancy.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/fusion/qk-rope-kv/
```
