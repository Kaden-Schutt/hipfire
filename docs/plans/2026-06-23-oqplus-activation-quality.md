# Plan 1: OQ+ (W4A8) quality — activation-aware + weight-error-feedback levers

Date: 2026-06-23. Branch: chaingun. Follows the recovery-FT exploration, which
established that **FT-based recovery is NOT the lever** for OQ+ on qwen3.5
(block-local norm recovery was a wash end-to-end: ppl 27.25→27.28; see
[[project_qwen35_norm_recovery_phaseA]]). The quality must come from the
**offline quantizer**, not post-hoc fine-tuning. AWQ already recovers OQ+ ~1.9×
(KLD 0.1536→0.0813, commit cf387d42). This plan pushes further.

## Goal

Minimize OQ+ (W4A8) KLD-vs-bf16 toward the W8A8 floor (oq8 = 0.00156), and decide
whether OQ+ is production-viable vs the mq4+ baseline. Headline gap to close:
OQ+awq **0.0813** → ideally <0.03 (the gap from W4 weight quant, since A8 is clean).

## Key hypothesis (the cheap high-value test)

**LDLQ (full-Hessian GPTQ/OBS error-feedback weight quant) should help OQ+ even
though it did NOT help oq4 (W4A4).** For W4A4 the dominant error was the runtime
int4 ACTIVATION quant (AWQ's target), so LDLQ-on-weights added ~nothing over AWQ
(RTN 32.16 / LDLQ 30.74 / AWQ 29.45 / LDLQ+AWQ 29.48 ppl; see
[[project_opus_w4a4_status]]). OQ+ is W4A8 — activations are clean int8, so the
**weight** int4 quant is now the dominant error (cf387d42 sweep: W8A8 0.00156 vs
OQ+ W4A8 0.1536 — the entire gap is the W8→W4 weight step). Weight error-feedback
should therefore pay off here. **If true, LDLQ+AWQ is the new best OQ+ recipe.**

## What's already built (no new codec work to start)

- `HfqInputFormat::OqPlus` (`--format oq+`, qt=33) — plain W4A8.
- `OqPlusTiered` (`--format oq+t`, qt=34) and `OqPlusCompact` (`--format oq+c`,
  qt=36) — magnitude-tiered: int4 bulk + sparse int8 outliers (`N_out =
  round(w8_frac·256)`/group); loaders qt 34/36 expand+overlay. `quantize_oqplus_compact`.
- `--awq` (AWQ/SmoothQuant sidecar) and `--ldlq` (full-Hessian) both compose with
  all three: `ldlq::oq4_ldlq_pack` (plain), `oqplus_compact_ldlq_pack`,
  `oqplus_tiered_ldlq_pack` (main.rs ~3124/3222/3224). `--hessian <h>` feeds both.
- Hessian: `~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin` (full [K,K], `HfhsFull`).

So the front-end exists; the deliverable is the **eval program** that finds the
best recipe and validates the LDLQ-helps-W4A8 hypothesis.

## Eval methodology (per the perf/quality rules)

- **KLD-vs-bf16** (primary, low-noise): `build_kld_ref*` from bf16
  (`crates/hipfire-eval/src/quality.rs`), then KLD per recipe. ≥16 chunks (top-K
  KLD is noisy at 2 — the lesson from the KV work).
- **ppl ctx=2048** (`perplexity` example, default lowered path — NOT
  FORWARD_LOWERED=0, which breaks scoring; a 512-tok window is too noisy, use 2048).
- **Coherence**: `./scripts/coherence-gate.sh` on the winning recipe (no attractor /
  list-primes loop — the failure mode plain oq4 had).

## Steps (priority order)

1. **Recipe sweep on qwen3.5-0.8b** (cheap, all front-end exists):
   `oq+ {RTN, AWQ, LDLQ, LDLQ+AWQ}` × baseline. KLD + ppl2048 each. Confirms or
   kills the LDLQ-helps-W4A8 hypothesis directly. ~8 quantize runs + evals.
2. **Tiered/compact sweep:** `oq+c` (and `oq+t`) at `w8_frac ∈ {0.01, 0.03, 0.06}`
   × {AWQ, LDLQ+AWQ}. Measures the quality/byte Pareto of sparse-int8-outlier
   protection (the natural fix if pure W4 leaves a floor). Plot KLD vs bits/weight.
3. **Pick the knee** of the Pareto; run coherence gate; compare to mq4+ (the
   incumbent W4A8 — same iu8 kernel, affine-u4+clip+SmoothQuant) on equal footing.
4. **Only if headroom remains after 1–3** (new work, gated on the sweep):
   - AWQ α search (current α is fixed; sweep α∈{0.25,0.5,0.75} — `compute_awq_scales`).
   - Per-group (vs per-token) int8 activation quant — tighter A8, if A8 turns out
     non-negligible at the W4-recovered operating point.
   - Better/learned rotation in place of fixed FWHT-256 (only if rotation shows as
     the residual bottleneck via an SQNR-by-stage breakdown).

## Decision this produces

A single recommended OQ+ recipe + its KLD/ppl/coherence, and a go/no-go vs mq4+:
- If LDLQ+AWQ (or compact) gets OQ+ KLD into mq4+ territory at ≤ mq4+ bits → OQ+
  ships as the W4A8 production format (symmetric-int4, the cleaner iu8 path).
- If it can't beat mq4+ → OQ+ stays a research format; mq4+ remains production W4A8.

## Cross-cutting note

This is **offline-quant** quality work — orthogonal to and cheaper than the
FT-recovery path that this session showed doesn't transfer. No GPU training loop,
no capture; just quantize → eval. Reuses the validated codecs + Hessian infra.
