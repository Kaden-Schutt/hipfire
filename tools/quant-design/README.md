# quant-design — one-off GPU sweeps for quantization format design

## Why this exists

Format-design questions ("is 136 B better spent on outlier refinement or on a
second sub-block scale?") are embarrassingly parallel numeric sweeps over real
post-FWHT weight blocks. Run on the CPU in NumPy they are unusable: 64 candidate
level profiles × 278,528 blocks × 256 coefficients is ~4.6 billion nearest-level
searches *per configuration*, and a three-configuration cell times out.

`sweep_mq4_header_allocation.hip` evaluates **14 configurations in 2.9 seconds**
on a gfx1201. That is the difference between guessing from three data points and
seeing the whole tradeoff surface.

## Usage

```bash
# 1. dump real post-FWHT blocks + the codebook family (f32 binaries)
#    sweep_G.bin   : nblk × 256 f32, post-FWHT, engine sign seeds 42/1042
#    sweep_fam.bin : 64 × 16 f32 profile family
# 2. build and run; argv[1] is the tail threshold (99th pct of |G|)
/opt/rocm/core-7.14/bin/hipcc --offload-arch=gfx1201 -O3 \
    sweep_mq4_header_allocation.hip -o sweep_mq4
./sweep_mq4 2.869166e-02
```

One 256-thread workgroup per 256-weight group. Errors accumulate in **raw weight
space** — normalising and forgetting to multiply the scale back is the mistake
this harness exists to make impossible, since it inflates MSE by ~1e5 and looks
like a catastrophic result rather than a units bug.

## Reported metrics

- **overall MSE** — the metric that has repeatedly FAILED to predict KLD. Report,
  never rank on.
- **tail-1% MSE** — squared error restricted to coefficients with `|w| >=`
  threshold. This is the metric measured to track the observed KLD failure.
- **max-coefficient relative error** — error on each block's single largest
  coefficient. Uniform affine scores exactly 0.000% here by construction, because
  min/max fitting makes the block extreme representable; a codebook with a fixed
  outermost level does not.

## Adding a configuration

Extend the `cfgs[]` table in `main`. Each row is
`{name, p, nsub, refine_k, refine_bits, header_bytes, total_bytes}`:

- `p` — selection weight exponent: the profile minimising `Σ |z|^p (z-ẑ)²`.
  `p=0` is plain SSE and is **bulk-dominated**, since 255 of 256 coefficients are
  bulk; it systematically starves the tail.
- `nsub` — sub-blocks per 256-group, each with its own max and selector.
- `refine_k` / `refine_bits` — indexed top-k outlier refinement. Costs
  `k × (8-bit index + bits)`; index-free variants that identify outliers from the
  nibbles do NOT work, because the outer-two levels hold ~13.5 coefficients per
  block so scan-order slots miss the actual largest.
