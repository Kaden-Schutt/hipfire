# RoughQuant — Phase 2f: reverse-search knee via KLD (quantize + void)

**VERDICT: confirms the negative, cleanly.** Searching from the lossless bf16
floor and increasing the quantized/voided fraction of the lowest-energy channels,
measured with **monotonic KLD** (immune to PPL's pointwise wiggle): there is only
a *soft* quantization knee (mild concentration), and **no weights are truly
unneeded** (voiding even 1–2% is catastrophic). Consistent with the spread-energy
finding (phase2e): the model's channels are densely informative.

## Method (user's reverse-search direction)

Start at bf16 + fp32 DeltaNet state (the 26.17/24.05 floor), quantize (or void)
the **lowest-energy** channels first, increasing the fraction; watch where the
metric departs from flat = the knee. Metric: **KLD(bf16‖candidate)** via the new
`perplexity --dump-ref/--kld-ref` (top-128), which is monotonic in weight error —
PPL was too pointwise-noisy to resolve the knee (non-monotonic across fracs even
though bit-exact deterministic). bf16 self-KLD = 0.000000 (exact sanity).

## Quantize (mq4) knee — soft

| quantize % (lowest-energy first) | PPL | KLD |
|---|---|---|
| 2%   | 24.10 | 0.0004 |
| 10%  | 24.38 | 0.0036 |
| 20%  | 24.42 | 0.0074 |
| 40%  | 24.68 | 0.0156 |
| 60%  | 24.84 | 0.0287 |
| 80%  | 25.26 | 0.0507 |
| 100% | 27.38 | 0.1608 |

KLD rises monotonically; slope steepens in the last 20% (top-energy channels):
~0.0004/% early → ~0.0055/% at the end. **Soft knee at ~80%:** quantizing the
lowest-energy 80% costs only ~1/3 of the total KLD (0.051 vs 0.161). Mild
concentration (top 20% of channels → 2/3 of divergence), matching the raw energy
CDF (top 20% = 56% energy). Not sharp enough for a bit-efficient win: protecting
20% at bf16 ≈ 6.6 avg-bits, worse than just using mq6.

## Void (prune) knee — catastrophic, no dead weights

| void % (lowest-energy first) | PPL | KLD | vs mq4 same % |
|---|---|---|---|
| 1%   | 24.44 | 0.0285 | — |
| 2%   | 25.54 | 0.0698 | **175×** worse |
| 5%   | 28.93 | 0.2165 | — |
| 10%  | 44.26 | 0.598  | 166× |
| 20%  | 161.5 | 1.790  | — |
| 40%  | 2678  | 4.511  | 290× |

Voiding the lowest-energy **2%** (KLD 0.070) is already worse than 4-bit
quantizing the *entire* model (0.161). **No weights are truly unneeded** — the
channels need their *value* (mq4's 4-bit captures it), but cannot be *dropped*.
The spec's `void` tier (prune the dead tail) has no dead tail to prune here.

## Synthesis (with phase2e)

- Quantize = keep approximate value → cheap (soft knee). Void = drop value →
  catastrophic. The model's weights are densely informative; energy is spread
  (raw CDF ~linear), so every channel matters but 4-bit suffices per channel.
- The only sharp concentration is in the per-weight eigenbasis (phase2e), which
  doesn't fold. So no foldable scheme — protect, permute, channel-consistent,
  prune — beats mq4.
- KLD is the right metric here and should be the default for future quant-quality
  work (PPL's pointwise wiggle masked everything sub-~1-PPL).

## Tooling delivered

- `perplexity --dump-ref/--kld-ref --top-k` (combined PPL+KLD, self-contained,
  works for bf16 sim candidates; `eval_hipfire` can't on gfx1151).
- `HIPFIRE_RQ4_BULK=void` (structured prune), `=mq4` (real bulk); `roughquant_energy_cdf.py`.

## Remaining avenue

Cross-model (7B/9B): energy may concentrate differently at scale; needs a fresh
Hessian (~1–3h collect) + the KLD harness (now ready). The bigger-model quantize
would benefit from CPU→GPU quant math IF the PCA path is revisited (else the fast
mq4/void/channel path is CPU-fine). Awaits a go decision.
