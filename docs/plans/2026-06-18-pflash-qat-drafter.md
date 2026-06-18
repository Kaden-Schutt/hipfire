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

## DECISIONS LOCKED (2026-06-18, post-M0b)

**Build the custom scorer.** M0b shows real, scale-growing non-recency importance
→ enough headroom to justify the machinery.

**Arch — shared embedding + small ATTENTION body + token-ID prior bias; NOT a
token-ID-gated MoE body.**
- Shared target embedding, frozen, resident (D1 confirmed). Only thing that makes
  "tiny" possible at 248K vocab.
- A few (2–4) SMALL attention+MLP layers. Attention is non-negotiable: the M0b
  needle is important only because the tail query asks for it — importance is
  CONTEXTUAL and needs attention depth (shallow K barely sees it, mid-layer does).
- A learned token-ID importance-prior bias (`[vocab]→scalar`, additive to the
  head) captures the cheap *static/lexical* slice (sinks, rare content tokens,
  punctuation — the StreamingLLM/H2O regime). This is the salvageable part of the
  token-ID-MoE idea.
- **Rejected: token-ID-gated MoE as the body.** It's a static per-token mechanism,
  but the shared embedding we already reuse IS a 248K token-ID lookup — an MoE
  would add capacity to the already-solved lexical part while contributing nothing
  to the hard contextual-coupling part (which only attention provides). Kept in
  reserve ONLY as a possible attention-free "fast lexical-prior tier" that
  knowingly cannot catch contextual needles.

**Train target — the target's MID-LAYER cosine-K block ranking (drop-in).**
Chosen over reproducing PFlash's shallow cosine-K (the weak +0.34 signal M0b
exposed) and over the causal oracle (gold but O(blocks) forwards/example).
Rationale: mid-layer K is the strong +0.81 signal, costs ONE target forward to
label, and the scorer keeps emitting K so PFlash's existing cosine scoring
consumes it unchanged (minimal engine change). The scorer's job is precisely the
M0b headroom: *shallow-cost output that ranks like the target's mid layer.* Causal
oracle reserved as a small validation set (does it catch the distant needle?).

**Strategy — prototype entirely in hipfire-train on a LOADABLE Llama target
first.** The real PFlash target (qwen3.5-9b/27b, vocab 248320) is hybrid-arch and
hipfire-train can't load it; capturing ITS mid-layer K needs a daemon-side hook
(the M1 engine gap). But we can prove the whole concept TODAY on a dense-Llama
stand-in target (e.g. Llama-3.2-3B) that hipfire-train loads: capture its
mid-layer K labels, build the shared-embedding tiny-attention drafter, train it to
reproduce the ranking, and measure vs the target's own shallow K (PFlash's
baseline). Only after that proto succeeds do we build the daemon K-capture hook +
PFlash consumption for the real qwen target.

### Revised milestones (supersede the M0–M5 list above for the chosen path)

- **P1 — label capture (hipfire-train, mostly exists).** Run a Llama stand-in
  target over a corpus; capture mid-layer K; per-block cosine ranking = labels.
  Reuses `model_forward` (returns all `layer_acts`) + `block_scores`.
- **P2 — drafter arch (hipfire-train).** Shared frozen target embedding + N small
  attention+MLP layers (reuse `block_forward` ops) emitting K, + token-ID prior
  bias. New constructor; no own vocab.
- **P3 — train + measure.** Distill the ranking (listwise/cosine-correlation or
  MSE on block-cosine scores). Gate: drafter's K ranking vs target mid-layer
  ranking, AND does it beat the target's OWN shallow K (PFlash's current baseline)?
  Validate distant-needle capture against the causal oracle on a few examples.
- **P4 — QAT (Q8 fake-quant + STE)** once fp32 proto clears P3.
- **P5 — real target.** Daemon-side mid-layer K-capture hook for qwen3.5; retrain
  the drafter against the real target.
- **P6 — PFlash consumption.** Drafter forward → K → existing cosine scoring →
  block selection + anchors → gather → compressed prefill; flip
  `maybe_compress_prompt`. Coherence-gated; measure long-ctx speedup + quality.

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

## M0b — SCALING + CAUSAL-ORACLE follow-up (2026-06-18) — REVISES the M0 verdict

M0 was scoped to Supra-50M and used shallow-vs-deep self-correlation (both layers
share the RoPE recency envelope → trivially correlated → measures little). The
405B question ("does this change with scale?") demanded a better design. Probe:
`crates/hipfire-train/examples/pflash_scaling_probe.rs`, dense-Llama ladder, fp32,
SEQ=512, 8 blocks, **planted distant dependency** (a needle fact in early block 1;
the tail query needs it), with a **causal keep/drop oracle** (ablate each context
block → KL shift in last-token logits = "does this block matter for the next
token"). Compares PFlash's shallow cosine-K metric AND a mid-layer metric against
the oracle, partialling out recency.

| model         | layers | partial(**shallowK**,oracle\|rec) | partial(**midK**,oracle\|rec) | needle rank shallow/mid/oracle |
|---------------|-------:|----------------------------------:|------------------------------:|--------------------------------|
| Supra-50M     | 12     | +0.000                            | −0.222                        | 6/7 · 5/7 · 2/7                |
| Llama-3.2-1B  | 16     | +0.154                            | +0.394                        | 7/7 · 5/7 · 1/7                |
| Llama-3.2-3B  | 28     | +0.335                            | **+0.810**                    | 5/7 · 3/7 · 2/7                |

(8B fp32 OOMs the 45GB gfx1103 box — the train loader has no quantized path; 8B/
70B/405B are the natural next rungs on hipx 96GB / hiptrx. partial = recency-
partialled Spearman of the metric vs the causal oracle; needle rank 1=best of 7
context blocks.)

**This REVISES M0, in the direction the 405B intuition predicted:**

1. **Recency dominance is a small-model artifact.** Shallow-K's non-recency
   importance (partial vs the causal oracle) rises **monotonically +0.00 → +0.15 →
   +0.34** across 60× scale. M0's "it's just recency" held only at 50M. By 3B the
   shallow metric already beats recency at finding what causally matters, and the
   trend extrapolates favourably toward 405B.
2. **Depth carries far more signal than scale.** Mid-layer K is dramatically better
   at every rung (3B: **+0.810** mid vs +0.335 shallow; −0.22→+0.39→+0.81). PFlash
   scores at the *shallowest* full-attn layer (to dodge long-ctx RoPE-OOD NaN) and
   is therefore leaving most of the importance signal on the table. **The gap
   between cheap-shallow and rich-mid is the real headroom** — and it's exactly the
   niche a trained drafter could fill: emit a mid-quality ranking at shallow cost
   without the deep-layer NaN.
3. **Caveat — the single distant needle is still hard.** Even at 3B the shallow
   metric ranks the one critical far-back block 5/7; the partial-correlation gains
   come from ranking the bulk, not nailing THE block 5%-keep depends on. Mid-layer
   does better (3/7 at 3B) but isn't crisp either at this size.

**Revised recommendation (supersedes M0's "park"):** the drafter is back on the
table, but its target is **not** "beat recency at shallow" (small win) — it's
**reproduce a mid-layer-quality importance ranking cheaply**, which is where the
signal actually lives and where the shallow-NaN constraint blocks PFlash today.
Before building M1–M5, the two decisive open tests are: (a) confirm the trend at
8B/70B (big boxes) — does shallow-K's partial keep climbing?; and (b) the **>32K
operating point** — this probe is 512 tokens; the recency envelope and OOD-NaN
behaviour at true long context (PFlash's actual trigger) are still untested.
Caveat on the retrieval readout: 3B-Instruct p(answer)=0.13 (vs 1B-base 0.80) is
confounded by base-vs-instruct chat formatting on a raw prompt; the causal oracle
is robust to it (KL shift regardless of literal answer token).
