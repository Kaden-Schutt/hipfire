# RoughQuant — Phase 0: fixtures & baselines

**VERDICT: PROCEED.** Fixtures verified, full pipeline works, the gate target is
established: RoughQuant at ~2.5 avg-bits must reach **PPL ≤ 29.08** (the
4-bit-uniform MQ4 anchor) on this corpus to pass.

## Fixtures (provenance)

- **Model:** `/srv/huggingface/models--Qwen--Qwen3.5-0.8B` — hybrid
  linear-attention arch (18 `linear_attn` layers + 6 full-attn `q/k/v/o_proj`
  layers = 24 layers).
- **Hessian sidecar:** `~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin` (HFHS v1,
  186 tensors, F32). Tensor `K ∈ {1024, 2048, 3584}` — **all divisible by 256**,
  so every quantizable 2D weight is eligible for the `*-sim` post-pass.
  Coverage: out_proj×18, in_proj_{qkv,z,b,a}×18 each, gate/up/down_proj×24 each,
  q/k/v/o_proj×6 each.
- **Corpus:** `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`
  (md5 `83b0205a304bf4e52172ecdb05f2e895`).
- **Pre-built .hfq:** `~/.hipfire/models/qwen3.5-0.8b-{bf16,mq4,mq6,qtip3sim-ldlq,qtip3sim-plain}.hfq`.
- Binaries `target/release/{hipfire-quantize, examples/perplexity}` already built.

## Baseline PPL (perplexity.rs, --ctx 2048 --warmup 8 --offset 0, 2039 scored)

| Model | ~avg-bits | NLL/tok | PPL | Δ vs bf16 |
|---|---|---|---|---|
| bf16 (floor)        | 16  | 3.2648 | **26.17** | —     |
| **mq4 (GATE target)** | ~4  | 3.3700 | **29.08** | +11%  |
| qtip3sim-ldlq       | ~3  | 3.4476 | 31.42 | +20%  |
| qtip3sim-plain      | ~3  | 3.5384 | 34.41 | +31%  |

Observations:
- 3-bit QTIP (sim) is *worse* than 4-bit MQ4 here, as expected.
- LDLQ (output-aware) buys ~3 PPL over plain QTIP at 3-bit (34.41 → 31.42),
  confirming the Hessian sidecar + LDLQ path is live and effective.

## The sim methodology (established pattern, reused)

`hipfire-quantize --format <fmt>` has a `*-sim` family (`qtip2-sim`,
`qtip3-sim`): every 2D weight is staged as BF16, then a post-pass
(`main.rs:8912`) bakes the quantize→dequant error back into BF16. PPL is then
measured via the *normal* bf16 forward — no GPU kernel needed. The post-pass
pulls the per-tensor k×k Hessian from `HIPFIRE_QTIP_HESSIAN` and runs LDLQ
(`ldlq::qtip_ldlq_dequant_bits`) or plain QTIP (`qtip_simquant_nbit`).

**RoughQuant reuses this exact vehicle:** a new `roughquant-sim` format adds a
post-pass that (Phase 1) protects the top-k highest-`diag(H)` input columns at
exact BF16 and quantizes the rest, then (Phase 2) adds PCA rotation + per-tier
binning. This keeps the de-risk on the CPU/sim side until the frontier is proven.

## Next: Phase 1

Implement `roughquant-sim` top-k column protection (no rotation). Rank the `k`
input columns of each weight by `diag(H)`, protect top-k at BF16, quantize the
rest to a low-bit grid, dequant, re-emit BF16. Sweep k (fraction of columns
protected) and the bulk bit-width. Gate: does protecting a tiny top-k move PPL
per the super-weight thesis?
