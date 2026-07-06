# Plan — shared greedy-accept core + qwen-DFlash relocation + ChainSpeculator + generic guard

**Branch:** `feature/speculator-abstraction` (on top of `c0986b83`)
**Goal:** make the speculative-decode *acceptance rule* a single tested source of
truth across all drafters, move all qwen-specific drafter code into the qwen
crate, and give a generic per-arch slot-guard — so a future llama-DFlash / MTP
builds on the seam with no daemon/loader edits.

## Design (settled with the user after a 4-site accept audit)

The four greedy-accept sites share only a 3-line inner comparison; each wraps it
with genuinely-different concerns (DFlash: sampling/n-gram/repeat; MTP:
EOS-early-stop; deepseek4: per-position grammar mask + matcher advance). The
**precompute-then-match** core avoids a kitchen-sink: each arch computes its own
`target_pick[i]` *before* calling the core; the core does only prefix-match +
EOS-stop + bonus.

```rust
// hipfire-runtime/src/spec.rs
pub struct GreedyAccept { pub committed: Vec<u32>, pub accepted: usize, pub hit_eos: bool }
/// longest i where target_pick[i] == drafts[i]; if eos=Some and an accepted
/// token == eos, stop and skip bonus; else bonus = target_pick[accepted].
pub fn accept_greedy_prefix(drafts: &[u32], target_pick: &[u32], eos: Option<u32>) -> GreedyAccept
```

Captures: DFlash-greedy (`eos=None`), MTP `greedy_trunk_spine_accept` (`eos=Some`),
n-gram (`eos=None`), deepseek4 **non-grammar** path. Stays bespoke: DFlash
`temp>0` rejection-sampling (separate non-greedy fn), deepseek4 **grammar** path
(stateful per-position masking — `target_pick[i+1]` depends on accepted `i`).

## Steps (each → verify)

1. **Accept core** — add `GreedyAccept` + `accept_greedy_prefix` to `spec.rs` with
   unit tests (eos=None full/partial; eos=Some stop-mid-prefix; bonus==eos). →
   verify: `cargo test -p hipfire-runtime spec::`.
2. **n-gram onto core** — `NgramSpeculator::step` uses `accept_greedy_prefix`
   (eos=None). → verify: qwen35-9b + qwen3-0.6b-llama committed-ids byte-identical
   to current (fresh daemon).
3. **DFlash greedy onto core** — in `spec_step_dflash`, the greedy branch builds
   `target_pick` (its argmax + n-gram override) then calls the core; sampling
   branch untouched. → verify: coherence-gate-dflash + 27B byte-identical greedy.
4. **MTP onto core** — replace `greedy_trunk_spine_accept` body with the core
   (eos=Some). → verify: MTP coherence (mtp gate / deepseek4-or-qwen35-MTP run).
5. **deepseek4 non-grammar onto core**; grammar path stays sequential. → verify:
   deepseek4 coherence (no-tools + tool-call). **DONE (e8fb7191).** Routed the
   non-grammar accept through `accept_greedy_prefix` by widening verify K→K+1
   (the (K+1)'th position is the full-accept bonus the core requires); grammar
   path stays sequential + appends the same bonus. Spec output byte-identical to
   fresh greedy decode; coherence-gate-deepseek4-mtp --full OK on all 6. Default
   `spec_k` flipped 3→2: K+1 makes k=2 highest-accept AND highest-throughput
   (before→after: k2 code +28%/prose +22%/math +5%; k3 regresses 9-12% so it's
   no longer the default, but k2-after beats the old k3 default everywhere).
6. **Unify `SpecStepResult`** — single struct in runtime (`{drafted, accepted,
   bonus, committed}`); qwen35 + deepseek4 lower from it. → verify: build all.
7. **`ChainSpeculator<BlockDrafter>`** — `BlockDrafter::propose(emitted, seed, k)
   -> Vec<u32>`; `NgramSpeculator` becomes a `BlockDrafter`; `ChainSpeculator`
   does prefill/verify_block/accept-core/commit_prefix. → verify: byte-identical
   to step 2.
8. **Move qwen35 DFlash → qwen crate** — `DflashState`, `DdtreeState`,
   `load_dflash_state`, `lower_qwen35`, `DflashSpeculator`,
   `build_dflash_speculator` → `hipfire_arch_qwen35::dflash_spec` (runtime::dflash
   + qwen35::speculative types only, no loader types). Loader keeps `Qwen35SlotGuard`
   + the `build_speculator` registry. → verify: build + serve-multiturn DFlash arm.
   **DONE (e2e92a84).** Byte-identical relocation; builds + 6 accept-core tests
   green. NOTE: `Qwen35SlotGuard` substance was NOT split into qwen35 — the guard
   fundamentally needs `ModelState` (a loader type), so splitting it into a
   loader-wrapper + qwen35-substance would make two shallow modules (interface ≈
   guts). Kept whole in the loader per the deep-module principle; the seam goal is
   unaffected (see step 9).
9. **Generic `SpecTargetGuard`** — trait in runtime; loader `spec_target_guard()`
   dispatch (qwen35 → `Qwen35SlotGuard`, llama → new `LlamaSlotGuard`); daemon
   drops the `SpecSlotGuard` enum. → verify: serve-multiturn (AR+DFlash) + llama
   n-gram still route. **DONE.** serve-multiturn PASS (AR qwen35 + DFlash 27B, all
   4 cross-session requests coherent, uniq 0.81–0.86); llama n-gram route via
   `coherence_probe` on qwen3-0.6b-llama + `HIPFIRE_NGRAM_DRAFT=1` → OK (0 hard, 0
   soft). A future llama-DFlash/MTP adds one arm to `spec_target_guard` + a
   `SpecTarget` impl — no daemon edits.

## Validation (mandatory)
- Greedy byte-identical checks use **fresh daemon** (rebuild daemon + probe — see
  [[coherence-probe-stale-daemon]]).
- `scripts/coherence-gate.sh`, `scripts/coherence-gate-dflash.sh`,
  `scripts/serve-multiturn-gate.sh`. deepseek4 coherence for step 5.
- **NEVER `cargo fmt`** / `fmt-changed.sh` on this long branch — per-file
  `rustfmt --edition 2021 --config skip_children=true` on ONLY edited files;
  do not touch llama.rs (legacy debt). See [[rustfmt-changed-files-only]].

## Risk
Steps 3–5 touch coherence-sensitive spec paths. The precompute-core preserves
each site's exact semantics (the arch still computes target_pick its own way), so
greedy output must stay byte-identical — that's the regression guard. Commit per
step so a regression bisects cleanly.
