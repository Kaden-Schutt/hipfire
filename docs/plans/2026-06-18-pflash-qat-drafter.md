# Plan: QAT'd tiny PFlash drafter — purpose-built token-importance model

Status: PLAN / exploration (not started). Date 2026-06-18.

## Goal

Replace PFlash's "borrow an off-the-shelf small LM" drafter with a **purpose-built,
QAT'd, quantization-native** token-importance model: as small and low-compute as
possible, trained to reproduce the *target's* source-block importance, then
quantized to the fastest RDNA format (int8/Q8-WMMA on RDNA3, int4-WMMA on RDNA4)
with no quality cliff. Feeds PFlash's existing cosine-K scoring/selection.

## What PFlash actually needs from the drafter (grounded in pflash.rs)

- Runs a forward over the **source tokens** (>32K context trigger;
  `threshold_tokens=32768`), in the **target's tokenizer/vocab** (hard
  constraint — `vocab=248320` for qwen3.5/3.6; qwen3-0.6b is rejected as
  incompatible).
- Produces a **K cache at the shallowest full-attention layer**
  (`score_layer_idx`) — shallow on purpose to dodge the long-context RoPE-OOD NaN
  cascade deep layers hit past the drafter's trained window.
- Scoring (`compute_scores_batched*`): **mean-pool K per 128-token block, score =
  cosine(block_mean_K, last_token_K)**; keep top `keep_ratio≈0.05` blocks +
  mandatory anchors (sink + recent + chat boundaries), `min_keep_tokens≈2048`.
- So the drafter is a **representation model** (good K geometry), NOT a generator
  — **no lm_head needed**.

PFlash consumption is currently **scaffolding only** (`maybe_compress_prompt`
always `Bypass`); the scoring functions exist but selection/gather/wiring don't.

## Design decisions

### D1 — Drafter shape: tiny transformer + SHARED target embedding (no own vocab)

The 248K vocab is the enemy of "tiny": a fresh embedding is ~127M params at
hidden=512 — bigger than the model. Resolve by **sharing the target's embedding
read-only** and giving the drafter only a few tiny transformer layers that emit
K at one shallow layer. Net incremental params ≈ a handful of attention+MLP
blocks; the embedding is already resident. Vocab-matched for free, genuinely
tiny, memory-light. (Departs from "load a standalone small LM as drafter" — the
drafter becomes a small head over the target's embedding table.)

Alternative kept in reserve: a standalone tiny LM at the target vocab (simpler
PFlash interface, but pays the embedding cost).

### D2 — Training signal: distill the target's per-block K-cosine importance

The drafter should reproduce the importance ranking PFlash *would* get from the
target itself. So the supervision is the **target's own block scores**: run the
target over long-context calibration data, capture its K at a scoring layer,
compute `cosine(block_mean_K, last_K)` per block → that's the ground-truth
ranking. Train the drafter so its K-cosine scores match (rank/cosine-correlation
loss, or KL over the softmaxed block-score distribution).

Why this over alternatives:
- vs distilling target *attention weights*: the K-cosine signal is exactly what
  PFlash consumes, so we train to the metric that's actually used.
- vs keep/drop oracle (ablate each block, measure target-output change): correct
  but O(blocks) target forwards per example — far more expensive to label.
Start with D2; keep the oracle as a validation check on a few examples.

### D3 — QAT in hipfire-train: fake-quant + straight-through estimator

Add a fake-quant op (quantize→dequant to the target format in the forward, STE in
the backward — gradient flows as identity). Train the drafter with its weights
fake-quantized to **Q8** (the shipped fast format) so the final real quant has no
surprise. Small, well-scoped addition to the existing op set; gradcheck the STE.

### D4 — Fast format: Q8 WMMA (exists), DOT4 as a follow-on

hipfire already has `gemm_q8_0_wmma` (Q8 WMMA GEMM, gfx12 + gfx1151 4-warp
variants) → the drafter quantizes to Q8 and serves on that path today. Per the
FSR4 RDNA3 notes, a `dot4add_i8packed` (DOT4) int8 path may beat WMMA on
RDNA3/3.5 specifically — a measured follow-on, not a blocker. RDNA4: revisit
int4-WMMA. The drafter is the ideal first consumer of a "fastest-compute format"
because its quality bar is low (ranking, not generation).

## What has to be built vs reused

Reused (already exists):
- **hipfire-train** — fp32 forward+backward + AdamW + the verified op set
  (attention, rmsnorm, linear, rope, softmax, distill, cross-entropy).
- **Calibration capture** — GPU `ActivationCapture` + reduction kernels; the
  hook pattern for capturing per-forward signals (but currently captures GEMM
  inputs for Hessians, NOT K — see gap).
- **Q8 WMMA GEMM** — the serve path for the quantized drafter.
- **PFlash scoring math** — `compute_scores_batched*` (cosine-K) exists.

To build:
- **Target K-score capture** (new): hook the target's K projection at the scoring
  layer during long-context forwards → per-block cosine scores as training labels.
  (The calib capture hooks GEMM *inputs*; this needs the K *output* — similar
  mechanism, new hook point.)
- **hipfire-train: fake-quant/STE op** (D3) + the **embedding-shared tiny-drafter
  arch** (D1) in training form (a few blocks producing K; reuses existing ops).
- **PFlash consumption build-out** (greenfield): drafter forward → K → cosine
  scores → block selection + anchors → token gather → compressed stream to
  prefill. (`maybe_compress_prompt` is bypass today.)

## Milestones

- **M0 — feasibility probe (cheap, do first).** On a long-context example,
  compute the *target's* per-block K-cosine ranking, then check: does a SHALLOW
  layer's K already rank blocks usefully (it must, since PFlash relies on it)? And
  how much does a small generic LM's ranking already correlate with the target's?
  This sizes the headroom a trained drafter can capture before building anything.
- **M1 — target K-score capture.** New hook → emit per-block cosine scores over a
  long-context calibration corpus as `(embeddings → block_scores)` training data.
- **M2 — QAT op in hipfire-train.** fake-quant + STE op, gradchecked. (Q8 sim.)
- **M3 — train the drafter.** Embedding-shared tiny transformer → K → cosine
  scores; distill the target's block ranking (D2), weights fake-quantized to Q8
  (D3). Metric: rank-correlation / top-5%-block recall vs target.
- **M4 — quantize + verify.** Real Q8 quant of the drafter; confirm the ranking
  quality (M3 metric) survives. Serve via `gemm_q8_0_wmma`.
- **M5 — PFlash consumption.** Build scoring→selection→gather→compressed-prefill;
  wire the drafter; flip `maybe_compress_prompt`. Measure end-to-end: long-context
  prefill speedup AND output-quality delta vs full context (the gate that matters).

M0–M4 are training-side (hipfire-train + calib, our turf). M5 is engine
greenfield (PFlash hot-path; coherence-gated).

## Open questions / risks (decide before/within M0)

1. **Does a tiny QAT'd drafter rank well enough at 5% keep?** The core empirical
   risk. M0 sizes it; if a generic small LM already ranks ~as well as the target,
   a trained drafter buys little. If there's a real gap, it's worth it.
2. **D1 embedding-shared vs standalone tiny LM** — the former is genuinely tiny
   but changes PFlash's drafter interface (feed target embeddings); the latter is
   a drop-in but pays the embedding cost. Pick based on how small we need it.
3. **Per-target asset.** A drafter is specific to a target (vocab + distilled from
   that target). So this is one-drafter-per-target-family, not universal.
4. **Long-context numerics.** The shallow-layer-for-NaN-avoidance trick is a
   target-model property; a trained drafter must stay finite past its trained
   window (or be trained to the target's context length).
5. **M5 is a big greenfield chunk** independent of the drafter — PFlash's whole
   consumption side. Worth knowing the drafter is only half the work.

## Recommendation

Do **M0 first** — it's cheap, needs no new code (compute the target's block
ranking + a generic small LM's ranking, measure correlation), and it's the
go/no-go: it tells us whether a trained drafter has headroom to capture before we
invest in M1–M5. Everything downstream is gated on that number.

## M0 RESULT (2026-06-18) — recency-confounded, GO NOT established

Probe: `crates/hipfire-train/examples/pflash_m0_probe.rs`. Supra-50M, fp32
partial forward (hipfire-train, no PyTorch), SEQ=1024 (Supra max_position),
shallow L1 vs deep L8, block-cosine importance = PFlash's own metric
`cosine(block_mean_K, last_token_K)`. Run on gfx1103.

First-pass headline looked like a clean GO (shallow↔deep Spearman **0.991**,
top-k recall 1.00). **It was a synthetic-win trap.** Adding a recency baseline
(score = block index) collapses the claim:

| block | shallow↔deep | recency↔deep | recency↔shallow |
|------:|-------------:|-------------:|----------------:|
| 16    | +0.983       | **+0.981**   | +0.988          |
| 32    | +0.991       | **+0.985**   | +0.994          |
| 64    | +0.991       | **+0.991**   | +1.000          |

**The shallow layer adds ~zero lift over block-index recency** at reproducing the
deep ranking. PFlash's cosine-K importance score is, on this proxy, almost
entirely recency: recent blocks resemble the last token at *every* layer, so the
trivial "keep recent blocks + anchors" heuristic already captures nearly all of
it. A *learned* drafter has little headroom to beat that here.

**Two distinct takeaways:**
1. **Drafter (M1–M5): not justified by this number.** Do not invest in the QAT
   drafter on the strength of M0. The cheap signal a drafter would learn is
   recency, which we already have for free.
2. **PFlash metric critique (independent, useful):** `cosine(block_mean_K,
   last_token_K)` is recency-dominated by construction. That's worth knowing for
   PFlash selection design regardless of the drafter.

**Caveats — why this is "not established," not a hard NO:**
- Supra-50M at SEQ=1024 on local prose is a *weak proxy*. PFlash's real regime is
  a 9B/27B target at **>32K** context, where importance may decouple from recency
  much more (long-range retrieval, anchors far from the tail).
- Tiny models / short context / locally-coherent prose all bias toward recency.
- The faithful next step is to re-run the **same recency-confound test on a real
  target at real long context** before either building the drafter or redesigning
  the metric. That's the gate — not the inflated 0.991.

Bottom line: **M0 falsifies the easy GO.** Park M1–M5; the next cheap experiment
is the confound test at scale (large target, >32K), and/or rethinking PFlash's
importance metric to add a non-recency component.
