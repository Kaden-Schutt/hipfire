# Plan: rejection-sampling tree verify for DFlash ddtree (temp>0 distribution-correct acceptance)

- **Branch:** `feature/speculator-ddtree` (off `fivetide/feature/speculator-abstraction`)
- **Depends on / relates to:** `feat/ddtree-banded-build` @ `e0767840` — the banded-build
  knob (`HIPFIRE_DDTREE_MINNODES`), merged into this branch. The floor stays a
  near-no-op at both temps (it rarely binds under an active cutoff; see the A/B
  below). NOTE: the original "breadth only pays under sampling" premise was
  *falsified* by the A/B — tree breadth via budget/topk raises acceptance under
  greedy too. The floor specifically, not breadth in general, is the no-op.
- **Date:** 2026-06-26
- **Status:** IMPLEMENTED (Tasks 1–3, demo path) + A/B verified. Production
  `SpecTarget::step` temp plumbing (Task 2 daemon arm) deferred — see Remaining.

## Implementation note — the correct algorithm is *naive sampling*, not rejection

While scoping, a counterexample showed the obvious "rejection sampling over the
top-k siblings" scheme is NOT distribution-preserving: a deterministically-first
candidate is proposed with probability 1 (not `q`), so `min(1,p/q)` over-accepts
it (e.g. target wants token0 at 0.20 but the scheme emits it at 0.40). Leviathan's
guarantee needs the candidate to be a genuine *draft sample*; hipfire's ddtree is
built from top-k *marginals*, so there is no draft sample to anchor to.

The correct scheme for a top-k tree is SpecInfer's **naive sampling**: at each
slot draw a token `x` from the **target** distribution; if `x` matches a drafted
child, accept and descend; else `x` is the bonus. Every emitted token is a target
draw ⇒ distribution-preserving EXACTLY, at any temperature, with no dependence on
the draft probabilities. `DdNode.logw` is therefore unused by the sampler (kept as
metadata). Implemented as `ddtree::sample_verified_tree`; reduces to
`follow_verified_tree` at temp→0. Unit tests: temp→0 byte-equivalence + a 400k-run
Monte-Carlo confirming TV(emitted, target) < 0.01 incl. an uncovered target mode.

## A/B verification (27B-3.6 DFlash, q8 KV, --no-chatml, PEP-8 LRU md5 df5dedc8, warm)

Greedy (temp 0) vs sampled (temp 0.7), narrow (b8-k2) vs wide (b22-k4), no cutoff:

| cell                | accept | tau  | tok/s | mean_nodes |
|---------------------|--------|------|-------|-----------|
| temp0  narrow b8-k2 | 0.290  | 4.35 | 25.6  | 8         |
| temp0  wide  b22-k4 | 0.392  | 5.88 | 24.1  | 22        |
| temp0.7 narrow b8-k2| 0.284  | 4.26 | 32.7  | 8         |
| temp0.7 wide  b22-k4| 0.359  | 5.39 | 22.6  | 22        |

Findings:
1. **The sampled tree verify works and is coherent** — temp0.7 outputs pass the
   attractor detector (last-128 unique-ratio 0.34/0.48, max-freq 0.09), τ≈5.4.
   temp=0 path unregressed (coherence-gate-dflash 4/4 OK).
2. **Breadth raises acceptance at BOTH temperatures** (wide ≫ narrow: +0.10 accept
   at temp0, +0.075 at temp0.7). This *corrects* the earlier framing that "greedy
   ignores breadth" — greedy tree verify also benefits, because a wider tree has
   more chances to contain the argmax-matching child deeper. Breadth's real lever
   is **budget/topk**, and it is NOT temperature-gated.
3. **temp0.7 sampled accept is slightly below temp0 greedy** (0.359 vs 0.392 wide)
   — expected: sampling draws higher-entropy targets the draft tree covers less
   often. The payoff of temp>0 is **distribution-correctness**, not speed.

**Verdict:** the deliverable's value is correctness — temp>0 ddtree DFlash now
samples the target distribution properly instead of silently falling back to
greedy. It does NOT vindicate the banded `MINNODES` floor (still a near-no-op at
both temps; the floor rarely binds under an active cutoff). Breadth via
budget/topk was already helping greedy and continues to.

### τ impact — sampled vs old greedy-fallback, isolated (temp 0.7)

Apples-to-apples at the SAME temp/seed/tree, only the accept rule differs
(`HIPFIRE_DDTREE_GREEDY_VERIFY=1` forces the old greedy walk):

| tree        | greedy τ (old) | sampled τ (new) | Δτ     | tok/s          |
|-------------|----------------|-----------------|--------|----------------|
| narrow b8-k2| 4.353          | 4.256           | −2.2%  | 34.5 → 32.7    |
| wide  b22-k4| 5.885          | 5.390           | −8.4%  | 24.1 → 22.6    |

The greedy column reproduces the temp=0 baseline exactly (greedy accept is
temperature-invariant), confirming the isolation. So distribution-preserving
sampling costs **~2–8% τ** (more on wider trees) — the price of correctness:
sampling draws higher-entropy targets the draft tree covers slightly less often.
Modest and bounded; it buys correct temp>0 output where the engine previously
emitted argmax-biased (greedy) tokens regardless of the requested temperature.

## Remaining
- Plumb request temperature through `SpecTarget::step` so the daemon/serve path
  uses sampled verify at temp>0 (today production passes `temp=0.0`, greedy).
- Add a temp>0 arm to `coherence-gate-dflash.sh` (seeded RNG) as a standing guard.

## Problem

The qwen35 DFlash ddtree verify (`spec_step_ddtree_batched`, `crates/hipfire-arch-qwen35/src/speculative.rs`)
is **greedy/argmax-only**. At `temp>0` it prints
`WARNING: --ddtree with temp>0 falls back to greedy on the verify side ... rejection-sampling
integration is deferred` and accepts only the target's argmax path per slot
(`ddtree::follow_verified_tree`).

This is the **T→0 special case** of the canonical method. The literature
(Leviathan et al. 2023, Chen et al. 2023) defines speculative decoding around a
**modified rejection sampling** that provably preserves the target distribution at
*any* temperature: accept a candidate `c` with probability `min(1, p(c)/q(c))`,
else subtract the draft mass and resample from the residual. Tree methods
(SpecInfer, Sequoia, EAGLE) are built around this **sampled multi-candidate**
acceptance — which is exactly why tree *breadth* (siblings per position) is
wasted under greedy verify and load-bearing under sampling.

**Goal:** implement distribution-preserving rejection-sampling acceptance on the
ddtree verify path, so `temp>0` DFlash is correct and tree breadth (incl. the
banded floor) starts buying acceptance.

## What already exists (the scaffold — ~70% done)

- `DflashVerifyOutput.logits_per_pos: Vec<f32>` (speculative.rs:2203) — full per-slot
  target logits, "Only populated when `want_full_logits=true` (i.e. temperature sampling)".
- `verify_dflash_block_tree(... want_full_logits: bool ...)` (speculative.rs:2263) — flag
  already threaded; the batched lm-head branch has a live arm commented
  *"Rejection-sampling path needs full target distribution. Cost: B × vocab × 4 bytes
  D2H per verify (~15 MB at B=16 × 248K)"* (speculative.rs:~2620).
- Tree structure carries everything the walk needs: `DdTree.child_maps[slot]` = the
  sibling candidate set at each node; `follow_verified_tree` already walks it greedily.
- Divergent-path commit machinery: topk>1 accepts that diverge from the linear prefix
  already fall to a slow re-verify/commit branch ("step 10") — sampled non-argmax
  accepts reuse it, **no new kernel**.
- Demo/daemon already thread `temp` + a `seed` (demo default 42).

The current call passes the hardcoded `false` for `want_full_logits` (speculative.rs:4857)
and walks with `follow_verified_tree` (4865). That bool and that walk are the seam.

## Design

Rejection-sampling walk (SpecInfer/Leviathan sequential variant), per accepted position:

```
current = root; p = softmax(logits_per_pos[slot(current)] / temp)
loop:
  C = children(current)   # child_maps[slot] → (token, draft cond prob q)
  accepted_child = none
  for c in C ordered by draft rank (desc q):
      if uniform(0,1) < min(1, p[c.token] / q(c)):
          accepted_child = c; break
      else:
          p[c.token] = max(0, p[c.token] - q(c)); renormalize(p)
  if accepted_child: accept it; current = accepted_child; continue
  else: bonus = sample(p); stop          # residual draw
```

- **Back-compat:** at `temp→0`, `p` is a point mass on the argmax ⇒ accept iff
  `c.token == argmax`, bonus = argmax. Identical to `follow_verified_tree`.
- `q(c)` = conditional draft prob = `exp(node.logw − parent.logw)` — requires storing
  the cumulative draft log-prob per node (new `DdNode.logw`).
- Residual renorm + bonus sampling are O(vocab) CPU per rejected sibling — negligible
  vs the forward.

## Tasks

1. **`ddtree.rs` — node prob + sampled walk** (no GPU)
   - Add `pub logw: f32` (cumulative draft log-prob) to `DdNode`; populate in
     `build_ddtree_tree_bounded` from `HeapEntry.logw` at node creation.
   - New `pub fn sample_verified_tree(tree: &DdTree, logits_per_pos: &[f32], vocab: usize,
     temp: f32, rng: &mut impl Rng) -> (Vec<usize>, u32)` implementing the walk above.
   - Unit tests: (a) `temp=0` ⇒ identical to `follow_verified_tree` on random trees;
     (b) **Monte-Carlo distribution preservation** — over a toy tree, the accepted-token
     histogram matches the target `p` within tolerance.

2. **`spec_step_ddtree_batched` (speculative.rs:4635) — wire the seam**
   - Add params `temp: f32`, `rng: &mut StdRng`.
   - `want_full_logits = temp > 0.0` (flip the `false` at :4857).
   - `temp>0` ⇒ `sample_verified_tree(&tree, &verify_out.logits_per_pos, vocab, temp, rng)`
     in place of `follow_verified_tree` (:4865); route divergent accepts through the
     existing topk>1 slow commit branch.
   - Remove the temp>0 greedy-fallback warning (demo:1284).
   - Apply the same seam to `spec_step_ddtree` and `spec_step_ddtree_path_c` if/when
     they need temp>0 (path_c is main-path-linear; lower priority).

3. **Callers** — thread `temp` + seeded `StdRng` from `dflash_spec_demo` (has temp/seed)
   and the daemon serve path into `spec_step_ddtree_batched`.

4. **Gates + re-A/B**
   - `coherence-gate-dflash.sh`: add a `temp>0` ddtree arm (must stay coherent;
     seed the RNG for determinism).
   - **Re-run the banding A/B at temp=0.7** (rebase `feat/ddtree-banded-build` in first):
     MINNODES 0 vs 8, ddtree-b22-k4. *Now* τ/accept should move — the real test of the
     banding hypothesis. Use the canonical fixture (PEP-8 LRU, md5 `df5dedc8`), warm,
     fresh process/measure, GPU lock.

## Cost / risk

- **+~15 MB D2H per verify at temp>0** (full logits), ~3–5 ms/iter (scaffold's own
  estimate). **Greedy path untouched** (GPU argmax, 4 B D2H).
- Sampled accepts lean on the slow commit path ⇒ temp>0 tok/s trails greedy. The win is
  **distribution-correctness** + breadth finally paying (τ↑), not raw tok/s. Measure net.
- RNG must be seeded for reproducible gates.

## Effort

~1–2 focused sessions. The hard 70% (full-logits plumbing, batched tree-attention verify,
divergent-path slow commit) already exists. New code is one `DdNode` field + one CPU
accept function + flipping a bool + threading temp/RNG.

## Out of scope (this plan)

- 63fc72d2's faithful single-pass *forward layout* (`linearize_tree_for_verify`) — that's
  the orthogonal forward-geometry axis (still greedy), not the acceptance rule. Independent.
- deepseek4 MTP / minimax tree verify (same acceptance rule could be lifted later via the
  shared `ddtree` module once proven on qwen35).

## References

- Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding*.
- Chen et al. 2023, *Accelerating LLM Decoding with Speculative Sampling* (arXiv:2302.01318).
- SpecInfer (arXiv:2305.09781), Sequoia (arXiv:2402.12374) — tree topology vs temperature,
  sampling-without-replacement verification.
