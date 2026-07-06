<!-- Copyright (c) 2026 Kaden Schutt -->
# DFlash drafter-context bloat: a TriAttention eviction sidecar for the *drafter*

**Status:** PARKED idea / design sketch (2026-06-26). Not started. Gated on a
cheap decisive measurement (drafter MRL — see §5).

## 1. Problem — the DFlash long-ctx throughput wall

DFlash spec-decode is net-positive at short context but **net-negative at long
context**. The drafter re-attends over the full target-derived context
(`DflashScratch.k_ctx_cached` / `v_ctx_cached`, derived from the extracted
target hidden states) on **every** spec step → **O(ctx) work per step**. As ctx
grows the per-step draft cost grows linearly, τ falls, and DFlash crosses below
AR (measured: ~24k ctx → DFlash ≈3.1 tok/s, τ≈0.95 — a loss).

MTP does **not** hit this wall: its head reads the trunk's *final hidden state*,
not the whole context, so its per-step cost is ctx-independent. That asymmetry
is exactly why MTP wins long-ctx decode while DFlash degrades. The bloat is the
single thing standing between DFlash and long-ctx parity with MTP.

Note the bloat is a **throughput** wall (O(ctx)/step re-attention), not only a
memory one. Bounding the drafter's *attended* context is what removes it.

## 2. What already exists (so this is smaller than it looks)

- **Eviction MECHANISM is present.** `apply_eviction_retain_to_draft(gpu,
  draft_scratch, retain_mask, ne, h, pre_phys)` (`speculative.rs`) already
  compacts the drafter's cached context to a bounded budget
  (`budget = retain_mask.len()`). The O(ctx)→O(budget) plumbing is done.
- **But the drafter has no eviction policy of its own.** `DflashSpeculator::
  on_evict` (`dflash_spec.rs`) just *mirrors* the target's eviction: it applies
  whatever `retain_mask` the **target's** FlashCASK policy produced. So today the
  only way to bound the drafter is to evict the **target**.

Two consequences make that insufficient:
1. **Coupling to target correctness.** Bounding the drafter requires evicting the
   target — a correctness hit you take purely to speed up the draft. You often do
   *not* want to evict the target.
2. **Wrong keys.** A CASK-mirror keeps the keys the **target** needs. The drafter
   is a different (small, distilled) model with a different attention pattern, so
   it may need different keys → mirroring the target's retain mask costs draft τ.

## 3. Proposal — a drafter-calibrated TriAttention eviction sidecar

TriAttention (`triattn.rs`; Mao et al. 2026, arXiv:2604.04921) is a **KV
importance scorer**: pre-RoPE Q/K concentrate around fixed centers (MRL R≈1), so
the RoPE attention logit at query–key distance Δ collapses to a cheap
trigonometric series in Δ, letting you **predict a cached key's importance
without recomputing attention** — to decide eviction. The sidecar stores the
calibrated per-(layer,head,band) Q-centers `E[q_f]`.

The idea: **calibrate a TriAttention sidecar for the DRAFTER** and use it as the
missing drafter-side `retain_mask` *source*:

1. Trig-score the drafter's cached context (`k_ctx_cached`) against the drafter's
   own Q-centers each step (cheap — no attention recompute).
2. Produce a **drafter-optimal** retain mask at a **fixed budget** B.
3. Feed it into the existing `apply_eviction_retain_to_draft` plumbing.

Result: the drafter re-attends **O(B)/step instead of O(ctx)/step** — a constant
that closes the long-ctx gap to MTP — and, because the scoring is calibrated to
the *drafter's* attention, τ degrades far less than under a CASK-mirror.

### Why it beats the CASK-mirror

- **Target-independent.** Bound the drafter without evicting the target — keep
  full target KV for correctness, bound only the draft for speed.
- **Drafter-optimal keys.** Keep the keys the **draft** attends to (its pattern,
  not the target's), so acceptance holds.

## 4. Where it slots in

- New: a `DrafterEvictionPolicy` source that, given the drafter's cached context +
  the drafter triattn sidecar + a budget B, emits a `retain_mask`. Reuse the
  triattn scoring kernels (already used for the target).
- `DflashSpeculator::step` (or a pre-step hook): when the drafter's cached rows
  exceed B, run the policy → `apply_eviction_retain_to_draft`. Decoupled from
  `on_evict` (which stays as the target-mirror path for when CASK *is* on).
- Sidecar load: extend the loader's triattn sidecar path to also carry an
  optional **drafter** sidecar (`<draft>.triattn.*`), gated like the target one.
- Calibration: extend `hipfire sidecar-gen` to run the existing pre-RoPE Q/K tap
  (`triattn::record_prerope_qk`) on the **drafter** forward over a corpus and fit
  `E[q_f]` — same pipeline as the target sidecar, pointed at the draft model.

## 5. The decisive feasibility gate — drafter MRL

TriAttention's entire premise is **R≈1** (pre-RoPE Q/K concentration). It holds
for the *target* (why target sidecars work). The drafter is a small **distilled**
model — its Q/K geometry may not concentrate, and **if R is diffuse the trig
score won't predict its attention** and this whole approach fails.

**Cheap, decisive first step (do this before building anything):** wire the
existing `triattn` tap (`record_prerope_qk`) onto the **drafter's** forward, run a
corpus, and compute the drafter's per-(layer,head) MRL.

- **R ≈ 1 on most heads** → calibrate the drafter sidecar; proceed with §3/§4.
- **R diffuse** → trig-scoring is the wrong scorer for the draft. Fall back to a
  drafter-side scorer that does not assume concentration (see §6).

## 6. Fallbacks if the drafter MRL is diffuse

- **Learned tiny eviction head** on the drafter: a small MLP over (key, query-
  center, Δ) trained to predict the drafter's attention mass → retain mask. More
  capable than trig, needs training data + a tiny runtime head.
- **Attention-rollout from the extract layers**: aggregate the drafter's recent
  realized attention to score keys (uses observed attention, no model assumption)
  — cheaper to prototype, but reactive (scores last-step attention, not next).
- **Fixed sink+window** (streaming-LLM style) as a no-calibration baseline:
  always-keep first-k sinks + a sliding window of B-k recent — bounds the drafter
  trivially; measure how much τ a *dumb* bound costs to set the bar the smart
  scorer must beat.

## 7. Risks / open questions

- **Drafter is already τ-fragile at long ctx.** Any eviction loses context. The
  net win requires (throughput gain from O(B)) > (τ loss from eviction). Sweep B
  vs τ; there may be a budget floor below which DFlash stays net-negative.
- **Composition with target CASK.** When both evict, the drafter retain set must
  be a subset of the (post-CASK) available context. Define precedence.
- **Per-turn vs streaming eviction.** Evict once per turn (cheap, coarse) vs a
  rolling budget (smoother, more bookkeeping). Start per-turn.
- **Calibration corpus.** Reuse the target sidecar corpus discipline (Hermes/
  Aureth-class; wikitext sidecars were falsified — see memory
  `feedback_wikitext_triattn_sidecar_garbage`).
- **Validation.** Long-ctx (≥16k/24k/32k) DFlash tok/s + τ *and* coherence (the
  three-tier DFlash gate) at each budget B — a bounded drafter must not
  reintroduce attractors. Measure on the **daemon**, never a demo harness.

## 8. Success criterion

DFlash at ≥24k ctx goes from net-negative (≈0.95× AR) to **≥1.0× AR with τ>1**,
closing the long-ctx gap to MTP, at a fixed drafter budget B, coherence-clean.

## References
- `crates/hipfire-runtime/src/triattn.rs` — the trig-scoring + sidecar format.
- `apply_eviction_retain_to_draft`, `DflashSpeculator::on_evict` — the eviction
  plumbing this reuses.
- `docs/investigations/2026-05-15-dflash-prose-tau-research/` — DFlash long-ctx τ.
- Prior session finding: MTP wins long-ctx decode because its per-step cost is
  ctx-independent; DFlash's O(ctx) re-attention is the gap this closes.
