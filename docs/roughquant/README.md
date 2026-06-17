# RoughQuant — investigation summary & next steps

Navigable index + final verdict for the RoughQuant sim study on Qwen3.5-0.8B
(2026-06-17). Spec: `../roughquant-spec.md`. Autonomous prompt:
`../roughquant-autonomous-prompt.md`.

## TL;DR (final, reconciled)

**Positive result:** foldable protection of the **shared ~75-channel outlier set**
(bf16; persistent outlier dims recurring across layers) **halves mq4's KLD
(0.162→0.084) at +0.6 bits AND improves coherence** (beats the protect-0%
control, toward mq4). Literature-consistent (AWQ/super-weight outliers).

**Not yet shippable** — the bf16 SIM is faithful for KLD/PPL (teacher-forced) but
NOT for generation (different GEMV kernel than the real packed format; `rq-mq4path`
at protect-0% generates worse than real mq4). Honest coherence + perf need the
REAL packed format. Cross-model (7B/9B) confirmatory.

## Phase index

| doc | what | verdict |
|---|---|---|
| `phase0` | fixtures + baselines | mq4 PPL 29.08 gate (later: KLD default) |
| `phase1` | top-k protection (no rotation) | protection premise confirmed; rotation needed sub-4-bit |
| `phase2` | PCA rotation frontier | (later corrected) PCA win was a bug artifact |
| `phase2d` | permutation + channel-consistent | foldable variants vs mq4 |
| `phase2e` | energy CDF | **WRONG** "energy spread" (aggregation artifact) — superseded |
| `phase2f` | reverse-search knee (KLD), void | KLD knee soft; void catastrophic (no dead weights) |
| `phase2g` | **CORRECTION** | outliers real, sharp, SHARED (~75 dims), foldable |
| `phase2h` | foldable win + coherence | KLD halved + coherence improved; Q8-protect out; sim not generation-faithful |

## Five false-negative artifacts (all surfaced by user skepticism)

1. Non-monotonic zeroing in protected quantizers (fixed → overwrite).
2. Energy aggregation flattened the shared-outlier structure into a fake-linear CDF.
3. bf16 protection wasted half its bits (bf16 ≈ 8-bit mantissa); but Q8 weight-
   protection *degrades* generation — protect sensitive outliers at full precision.
4. PPL pointwise non-monotonicity masked sub-~1-PPL effects → KLD is the metric.
5. Q8-DeltaNet-state default + bf16-sim generation-fidelity faked "coherence fail".

**Meta-lesson (re-learned 3×): coherence batteries gate, not PPL/KLD. Generation
quality is non-monotonic in weight precision; teacher-forced metrics don't predict it.**

## Tooling delivered

- `perplexity --dump-ref/--kld-ref --top-k` — combined PPL+KLD (self-KLD=0;
  works for bf16 sim candidates). Default quant-quality metric.
- `scripts/roughquant_energy_cdf.py` — energy/eigenvalue concentration (no GPU).
- `scripts/roughquant_coherence_battery.sh` — FP32-state, detector-based
  attractor/repetition battery with a protect-0% control.
- `hipfire-quantize` formats/knobs: `roughquant{,2,3,4}-sim`,
  `HIPFIRE_RQ4_{BULK=mq4|void, MQ_BITS=N, PROTECT_FRAC, PROTECT_Q8, SALIENCY,
  Q8_EMBED}`, `HIPFIRE_RQ2_*`, `HIPFIRE_RQ_*`.
- `.githooks/pre-commit` — docs/markdown no longer trip the heavy GPU gates.

## Concrete next steps (productionization)

1. **Real packed format** (the make-or-break): `mq4` bulk + a **bf16 sidecar for
   the shared ~75 outlier residual channels** (read-columns + write-rows), with
   the offline fold of the shared protected-channel set into the producing/
   consuming weights. Then evaluate KLD + **coherence on the real GEMV** (not the
   sim) + fresh-probe tok/s. This settles absolute coherence and the perf cost of
   the sidecar.
2. **Selection**: try persistence-based outlier selection (channels that are
   top-k outliers in the most layers) vs the current aggregated-energy ranking;
   likely ≥ as good, more principled. Cheap.
3. **Cross-model (7B/9B)**: collect a Hessian (~1–3h), re-run energy CDF + KLD +
   coherence. Confirmatory — the shared-outlier mechanism should generalize.
4. **Do NOT** use Q8 weight-protection (degrades generation); protect at bf16.

## Fixtures (provenance, not committed)

- Model: `/srv/huggingface/models--Qwen--Qwen3.5-0.8B`.
- Hessian: `~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin` (HFHS, 186 tensors).
- KLD ref: `~/.hipfire/eval-results/refs/qwen3.5-0.8b-bf16.kldref.hfq` (q8 KV) and
  the self-contained `/tmp/bf16.pkld` (perplexity --dump-ref).
- Sim models: `~/.hipfire/models/qwen3.5-0.8b-rq-*.hfq`.
