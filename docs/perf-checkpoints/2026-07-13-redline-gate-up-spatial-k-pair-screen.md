# gfx1201 gate/up pair-first spatial-K screen

## Verdict

Reverted. Seven LPG64 variants tested whether adjacent wave64 lanes could
reconstruct the shipping wave32 gate/up kernel's eight-element lane partial
closely enough to retain the spatial-K throughput prize without the original
model-coherence failure. None produced a shippable win.

The only arm that briefly touched the champion's range, chain-pair R1, was not
a repeatable performance win and reproduced the original LPG64 numerical
failure immediately. Per the experiment gate, no spatial-K QKVZA work was
started.

## Variants

Every 64-lane wave maps adjacent lanes onto the eight contiguous K elements
owned by one shipping wave32 lane. All variants retain four accumulator chains
and reduce the 32 even-lane pair leaders with the shipping wave32 tree.

- `gp-r1` / `gp-r2`: pair adjacent half-group results after every HFQ4 group,
  then accumulate the reconstructed values into four chains.
- `chain-r1` / `chain-r2`: accumulate four chains per half-lane, then pair the
  completed chains before their final combine.
- `total-r1` / `total-r2`: combine four chains per half-lane and pair once.
- `gp-r2-bcast`: the tight group-pair R2 form with each packed weight word
  loaded by the even lane and broadcast to its adjacent partner.

R1 owns one output row/workgroup; R2 owns two. Sources were compiled in a
dedicated `-mwavefrontsize64` code object and selected only on gfx1201. The
shipping wave32 source and all gfx1100 sources were unchanged.

## Product screen

Host `hiptrx`, R9700/gfx1201, Qwen 3.6 35B-A3B MQ4R, Q8 KV, single-stream
retained PM4, automatic clocks, ten 100-token warmups followed by five
100-token rows. The current branch champion is approximately 201.17 tok/s.

| Variant | HIP median | PM4 median | VGPR | scratch/workitem | Verdict |
|---|---:|---:|---:|---:|---|
| group-pair R1 | 171.655 | 198.563 | 51 | 0 B | slower |
| group-pair R2 | 160.168 | 197.593 | 60 | 0 B | slower |
| chain-pair R1 | 168.277 | 201.637 | 57 | 0 B | not repeatable; incoherent |
| chain-pair R2 | 167.886 | 157.630 | 96 | 16 B | spilled |
| total-pair R1 | 172.152 | 200.716 | 57 | 0 B | below champion; incoherent class |
| total-pair R2 | 167.018 | 167.124 | 96 | 16 B | spilled |
| group-pair R2 + packed broadcast | 167.751 | 193.466 | 61 | 0 B | slower |

A matched wave32 control measured 199.361 tok/s in the final pass. Repeating
chain-pair R1 immediately afterward produced 200.229 tok/s, versus 201.637 in
its first screen. It therefore did not establish a repeatable improvement over
the approximately 201.17 champion even before correctness was considered.

## Numerical gate

The best-looking performance arm, chain-pair R1, was compared against the
wave32 control with deterministic graph-off post-layer hidden dumps at the
same position. Two control runs in the earlier LPG64 investigation were
byte-identical; the candidate diverged at the first MoE layer:

| Layer | Max absolute | RMS | Cosine |
|---:|---:|---:|---:|
| 0 | 2.175e-2 | 5.750e-3 | 0.94673 |
| 1 | 8.460e-2 | 2.104e-2 | 0.66830 |
| 39 | 2.694 | 2.445e-1 | 0.83330 |

This is effectively the same immediate and accumulating failure as the prior
218.69 tok/s LPG64 arm. It is far outside the drift envelope required before a
sampled eight-turn serve run, so the expensive serve gate was not run.

## Mechanism

The original LPG64 arm destroyed the four-chain structure and the 32-lane
reduction tree, but those were not the dominant numerical difference. Even the
tightest group-pair form still splits one wave32 lane's eight-term dot into two
four-term partials before combining them. That rounding change is sufficient
to perturb the MoE trajectory immediately.

Reproducing the original lane dot more exactly would require sending the odd
lane's individual products to the even lane and replaying the final four adds,
or making the even lane execute all eight terms. Either removes the useful
compute side of the 64-lane spatial-K decomposition. The family therefore has
no demonstrated path to both the LPG64 throughput and acceptable model
coherence.

Artifacts remain on `hiptrx` under:

```text
/home/kaden/.redline-work/hipfire-pair-gate/.redline-work/pair-gate/
```
