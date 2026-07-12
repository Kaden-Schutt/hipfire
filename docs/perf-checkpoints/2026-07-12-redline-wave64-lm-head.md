# gfx1201 wave64 two-row LM head: exact, +0.31%, below the shipping bar

## Verdict

The apparent 200 tok/s R1 result was invalid: the two-row wide kernel was
launched with only 32 threads, so `warp_id` was always zero and every odd
vocabulary row was left unwritten. Launching the full 64 threads fixes the
output and returns the wave32 arm to about 193.6 tok/s.

Compiling that corrected 64-thread kernel as one real wave64 is bit-exact and
reproducibly adds about 0.31% to retained-PM4 decode, but it does not reach the
195 tok/s acceptance bar. The experimental LM-head selection was therefore
removed. This is not a route back to the invalid 200 tok/s number.

## Candidate

The existing `gemv_hfq4g256_wide` source was compiled as a separate whole
translation unit with `-mwavefrontsize64` and launched as block `[64,1,1]`.
Each 32-lane half computes one vocabulary row with the established four-chain,
pairwise-reduction arithmetic. No per-function wave-size attribute was used.

Loader metadata for the gfx1201 object:

```text
wavefront_size: 64
vgpr_count: 52
sgpr_count: 12
private_segment_fixed_size: 0
vgpr_spill_count: 0
sgpr_spill_count: 0
```

The default `gemv_hfq4g256_multirow_r2` control is wave32, 94 VGPR, 19 SGPR,
and zero-spill. It computes two rows inside one 32-lane wave and shares the x
register stream across them. The wave64 R1 variant instead runs two independent
row kernels in the two half-waves. Wave64 does not turn those halves into a
free dual-issued pair: its raw vector-register lane footprint is 52x64 versus
94x32 for R2, while the two halves duplicate the per-row work and x accesses.

## PM4 transport gate

Redline initially rejected the valid wave64 descriptor because its gfx12 PM4
encoder hard-coded `CS_W32_EN=1`. The encoder now derives `CS_W32_EN` from
`kernel_code_properties.ENABLE_WAVEFRONT_SIZE32` for every dispatch. This is a
general mixed-wave correctness fix and remains landed even though this
particular LM-head candidate missed its performance bar.

The public HSA loader accepted all 26 captured kernel ABIs. Fifteen consecutive
positions passed exact PM4/HIP shadow parity for logits, KV, recurrent state,
and captured kernarg blobs.

## Product A/B

Host: `hiptrx`, gfx1201, automatic clocks, Qwen 3.6 35B A3B MQ4R, Q8 KV,
MTP off. Each arm used 100 measured decode tokens, at least ten warmups, and 30
rows. The reversed confirmation used 15 warmups because the first wave64 pass
still ramped during its first four measured rows.

| Order | R2 wave32 control | R1 wave64 | Delta |
|---|---:|---:|---:|
| control then candidate | 193.243 tok/s | 193.834 tok/s | +0.306% |
| candidate then control | 193.426 tok/s | 194.023 tok/s | +0.309% |
| order-balanced mean | 193.334 tok/s | 193.929 tok/s | **+0.307%** |

The candidate's maximum measured row was 194.173 tok/s. It did not warrant an
eight-turn serve run because it failed the 195 tok/s tg128 gate. Prefill is
unchanged by construction: this experiment only selected the single-token
HFQ4 GEMV path; batched prefill continues to use the existing GEMM kernels.

## Shipping outcome

- Keep the `[64,1,1]` launch correction for `gemv_hfq4g256_wide`; `[32,1,1]`
  silently skipped odd rows on every architecture that selected the kernel.
- Keep descriptor-driven wave32/wave64 PM4 dispatch initiation.
- Do not enable or ship the wave64 R1 LM-head selection.
- The fake 200 tok/s result is explained by roughly halving LM-head weight
  traffic through missing rows, not by a realizable wave64 dual-issue gain.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/
  gemv-wide-wave64.gfx1201.hsaco
  gemv-wide-wave64-shadow15.json
  gemv-wide-wave64-control-30.json
  gemv-wide-wave64-candidate-30.json
  gemv-wide-wave64-reverse-candidate-30.json
  gemv-wide-wave64-reverse-control-30.json
```
