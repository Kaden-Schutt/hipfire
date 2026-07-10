---
title: Daemon god-struct collapse → ModelParallel + ArchDispatch (design APPROVED, pre-impl) — the one structural unification independent of Step-decomposition
date: 2026-07-09
tags: [device-mesh, god-struct, loadedmodel, archdispatch, modelparallel, sessionstate, 462, state-bleed, daemon, dispatch, refactor, design, phase3, trait-object, parity-harness]
---

**Branch:** `feature/device-mesh` (worktree `.claude/worktrees/feature+device-mesh`). Full spec (gitignored-local per branch convention): `docs/superpowers/specs/2026-07-09-daemon-god-struct-archdispatch-design.md`. Parent: [[device-mesh-pivot-execute-steps-spine]] ("the one structural unification still worth doing"). **Independent of** the MoE/DeltaNet Step-decomposition thread (P-D/P-E): that delivers parallelism *transparency*; this collapses the daemon dispatch god-struct + makes #462 state-bleed structurally impossible.

## Decisions locked (bjoern, 2026-07-09)
- **Goal = BOTH, sequenced:** kill the #462 bleed surface AND collapse the god-struct; **#462-safety is the acceptance test** gating every increment.
- **Approach = C (full `Box<dyn ArchDispatch>` trait object), design-full / land-incremental.** Realized as **shared driver + hooks**, NOT a monolithic `fn ar_generate()` (the anatomy mapper flagged a single opaque method with per-phase if-chains as worse than per-arch hook impls). Every arch behind one `dyn ArchDispatch`; the daemon keeps ONE `ar_generate` driver owning the arch-invariant phases.
- **Inc 1 arch = qwen35 DeltaNet** (the catastrophic #462 case — recurrent state + DFlash checkpoints; prove the hardest lifecycle first).
- **Parity harness = dual-run env-gated shadow assert** (`HIPFIRE_ARCHDISPATCH_PARITY=1` runs old+new loops per request, asserts token-identical, off in prod). The safety spine that lets C land without a big-bang cutover.

## Target shape (~50 fields → ~6)
```rust
struct LoadedModel {
    parallel:   ModelParallel,          // where: was pp/pp_gpus/pp_scratch_set/pp_dn_la_to_device/ep/tp/pp_dense
    arch:       Box<dyn ArchDispatch>,  // what:  was state(ModelState)+arch_id+*_eos_tok+mtp_head+dots/vision+rec_* ladder+feature wiring
    session:    SessionState,           // request: was seq_pos/conversation_tokens/prefill+dflash_checkpoints/asst_turn_cache/kv_adaptive/eviction
    speculator: Option<Box<dyn Speculator>>,  // unchanged (already clean)
    tokenizer, model_path, chat_template,     // immutable infra
}
enum ModelParallel { Single, Tp(TpModel), PpDense(PpModel), PpQwen35{..}, Ep(EpState) }
```
`ArchDispatch` = AR-decode analog of load-time `Carrier` + TP/PP `DenseServed`. Hooks: `arch_id/eos_token/sampling_defaults/features/reset` + AR phases `frame_and_tokenize/prefill(lcp)/forward_step/stream_parser()->Option/finalize` + `as_spec_target()->Option<&mut dyn SpecTarget>` (bridges the EXISTING spec seam; SpecTarget/Speculator/ModelState/Carrier/DenseServed all stay, ArchDispatch composes).

## #462 mechanism (why bleed becomes unrepresentable)
Today reset is spread across **5+ sites** (per-handler overflow guard, `reset` cmd, VL entry guard, decode abort, EP abort), each an `if let Some(ModelState::Arch(..))` chain a new arch must remember to touch — correctness by vigilance. Target: ONE `LoadedModel::reset_context(gpu)` = `session.clear()` (**total by ownership** — a field not in SessionState is config or arch-owned) + `arch.reset(gpu)` (**total by the compiler** — impl ArchDispatch forces a reset arm) + speculator.reset. Adding an arch can't skip a reset. Guard = `serve-multiturn-gate.sh`.

## Increment plan (each gated: parity-shadow token-identical + serve-multiturn + coherence + build/test)
- **Inc 0:** new types alongside existing fields (dual-state, zero behavior change); `ar_generate` written unrouted.
- **Inc 1:** qwen35 (5/6) through `ar_generate`+ArchDispatch behind shadow-parity. Extracts the embedded qwen35 AR loop from `generate()` body (~daemon.rs:8588) — highest risk, done first.
- **Inc 2..N:** one arch/increment, delete old loop when green. Order: qwen2 → llama(+DenseServed axis seam) → minimax/cohere2/lfm2moe → deepseek4(DSML StreamParser) → qwen35-MTP → VL(qwen35-vl, dots-ocr).
- **Inc N+1:** ModelParallel collapse — axis-first tree → `match model.parallel`; migrate `load_model_{tp,pp,ep}`. Sequenced LAST to avoid colliding with live D-series load-path work; rebase on device-mesh HEAD first.
- **Inc N+2:** delete migrated direct fields + arch_id ladders + split *_eos_tok → 6-field target.

## Grounding (3 read-only subagent maps, 2026-07-09)
- **God-struct:** `LoadedModel` `crates/hipfire-loader/src/lib.rs:273`, ~50 fields. `generate()` `daemon.rs:6851` axis-first→arch-second, **25+ arch_id match sites** (sampling ladder ~1807-1881, EOS ~3079, feature wiring w/ silenced-unused-param warts).
- **Loops:** 9 AR paths, **12-phase skeleton, structurally isomorphic / operationally divergent**; variance concentrated in phases 3/5/9-10 (LCP/prefix-cache, prefill, per-token sample+forward) — ds4 DSML parser, minimax partial-LCP, qwen2 greedy-only. Two loops EMBEDDED in `generate()` (qwen35 ~8588, llama ~9734).
- **#462 today:** NO active bleed (all sites currently reset correctly) — the win is removing the *latent* surface (vigilance→construction), not fixing a live bug.
- **Existing seams:** `Carrier` `loader_api.rs:115`; `SpecTarget`/`Speculator` `spec.rs:155/582`; `ModelState` enum `lib.rs:242` (7 bundles: Qwen2/Qwen35/Llama/Lfm2Moe/Minimax/Cohere2Moe/Deepseek4).

## Non-goals / risks
Non-goals: NOT the Step-decomposition thread; NOT changing SpecTarget/Speculator/ModelState/Carrier/DenseServed; NOT a perf change (byte-identical target — any tok/s delta is a regression); NOT new functionality. Risks: (1) embedded qwen35/llama loop extraction entangled with axis dispatch (Inc 1 guard = parity assert); (2) hooks too coarse for ds4/minimax/qwen35-adaptive-KV → may add 1-2 optional hooks after Inc 1 + deepseek4; (3) AR-vs-spec routing must match today via `as_spec_target`; (4) Inc N+1 merge friction w/ D-series; (5) VL migrated last.

**Next:** `writing-plans` → Inc 0 + Inc 1 implementation plan.

## STATUS 2026-07-10 — Inc 0 DONE + Inc 1 Task 1.4 DONE (qwen35 through ar_generate)
Inc 0 complete (@1c78d297). Inc 1 Task 1.4 complete: expanded `ArchDispatch` (added tangle hooks maybe_evict/maybe_adaptive_downshift/take_prefill_checkpoint/abort_zero_recurrent/sample + GrammarMatcher::attractor_detected — the plan's "5 hooks" was insufficient; the "arch-neutral tangle" touches the arch bundle's kv/dn so it needs hooks too, and the runtime trait can't name LoadedModel so the dispatcher OWNS &mut m per Task 1.2). Generic `ar_generate` extracted (commits 380e7bea/036f07cf/e1d1b97a), dual-run shadow-parity wired (8a6b5dc7), GPU-parity PROVEN token-identical (5 prompts × single-GPU + emulated-2, FP32 DeltaNet + HIPFIRE_DETERMINISTIC=1), then flipped to prod + legacy arm deleted (−1212 lines), coherence-gate validated. Ledger: `.superpowers/sdd/progress.md`.

## INC 2..N ABSORPTION PROGRAM (bjoern 2026-07-10 "sequentially absorb all architectures") — live status
Goal: every arch's AR path onto the single ar_generate/ArchDispatch driver. UPLIFT decision (bjoern): migrated arches ADOPT ar_generate's richer feature set (n-gram loop guard, think-cap, stop-seqs, generic sampler) — NOT byte-identical to the simpler legacy loops; validate via clean-prompt/temp0 parity + coherence, not strict all-prompt parity. Sequence + status:
- **Inc 2 qwen2 (arch 7) DONE** (27bfadf3 Qwen2Dispatch + e1ce5a8b is_eos hook + c2ccf933 flip). Clean split port, ZERO new hooks (proved ar_generate generic). Added `is_eos` hook (per-arch stop set: qwen35=eos||terminator / qwen2=eos_token_ids config set). Coherent; qwen2 still feeds RAW prompt (pre-existing framing degeneration — [[vibethinker-framing-rootcause]], now fixable via ar_generate).
- **Inc 3 llama (arch 0/1) DONE** (0baca0b9 LlamaDispatch + 26be61d1 flip, path-a). llama arm was FUSED+minimal (forward_scratch samples internally, no features). Path-a: batched forward_prefill_batch B=1 + generic sampler. PERF-NEUTRAL A/B (0.6b 256→258, 8b 18.0→18.4, within noise — B=1 concern falsified). Coherent. NOT byte-identical (intentional uplift).
- **Inc 4 minimax (arch 10) GROUNDWORK ONLY** (66247ae5 MinimaxDispatch dead-code, build-clean; NOT wired/flipped). Medium port: HOST-side sampling (legacy deepseek4::sampling::sample_token temp/top_p+Xorshift → MinimaxDispatch::sample downloads state.logits[GPU-resident] + sampler::sample_cpu; temp0=argmax parity, temp>0 uplift); interleaved-thinking primed_think "<think>\n" must emit in preamble BEFORE ar_generate; LCP partial-reuse prefill preamble kept; model_reset_context DOES reset MiniMaxState (parity re-run sound). REMAINING: wire dual-run in generate_minimax(11680-12072) + validate temp0-parity/coherence on ornith-35b-aeon.mq6 (35b, ~28 tok/s, SLOW) + flip. The wiring is intricate (LCP/primed_think) — a focused pass, not a one-shot.
- **Inc 5 cohere2moe (arch 12)**: generate_cohere2moe@11842. North-Mini-Code models on box. MoE + sliding-window. Split-friendly, qwen2-ish pattern (expect medium).
- **Inc 6 lfm2moe (arch 11)**: generate_lfm2moe@11146. lfm2.5-8b-a1b.mq4. Adds CONV-STATE recurrent → abort_zero_recurrent/take_prefill_checkpoint per-arch bodies (not no-op).
- **Inc 7 deepseek4 (arch 9)**: generate_deepseek4@10484. HARDEST — DSML grammar (deepseek4::grammar::Matcher → init_grammar + ds4 newtype) + EP multi-GPU + MTP. Own increment.
- **DEFER**: VL (dots-ocr@13288 / qwen35-vl@12678 image embeds) + all spec-decode (generate_spec/_dflash/_deepseek4_spec — the "fold spec into ArchDispatch" follow-up #2 below).
PATTERN per arch: fork implements <Arch>Dispatch + dual-run parity wiring (build-only) → controller GPU-validates (temp0/clean parity + coherence + tok/s A/B) → flip (delete legacy loop) → commit. Separate-fn arches keep their own preamble + call ar_generate with its outputs (pflash_*=None). No trait changes needed so far except is_eos (Inc 2). CLEANUP: generate()-level `pflash_done_fragment`@daemon.rs:9478 is now orphaned dead-code (qwen35 flip) — remove/underscore.

## FOLLOW-UPS surfaced during Inc 1 (bjoern 2026-07-10 — do not forget)
1. **Grammar hook cross-arch adoption.** `ArchDispatch::init_grammar → Option<Box<dyn GrammarMatcher>>` (+ `GrammarMatcher{token_mask/advance/is_free/attractor_detected}`) is qwen35-ONLY today (newtype `Qwen35GrammarMatcher` over `qwen35::grammar::Matcher`). deepseek4 has its OWN grammar Matcher (`deepseek4::grammar::Matcher`, DSML tool-call path); qwen2 tool-calls too. When those arches move onto `ar_generate` (Inc 2..N), wire each arch's Matcher through `init_grammar` via a per-arch newtype so grammar is arch-generic in the driver (no per-arch grammar branch). ACTION: inventory which arches have grammar/tool-call Matchers, unify under the hook.
2. **Expand ArchDispatch beyond AR — fold in speculation.** Inc 1 deliberately scoped ArchDispatch to the AR arm ONLY; spec-decode stays on separate loops (`generate_spec` qwen35 DFlash / `generate_deepseek4_spec` DSpark-MTP), and `as_spec_target()` is stubbed `None` because the qwen35 SpecTarget is GUARD-based RAII (`Qwen35Carrier::spec_target_guard → Box<dyn SpecTargetGuard>`, moves the bundle out of m.state + restores on Drop) which does NOT fit the trait's `Option<&mut dyn SpecTarget>`. FUTURE INCREMENT: reshape the trait's spec bridge to yield the guard, OR route the spec loop through ArchDispatch, so ONE generate driver handles AR + spec generically (unified-generate endgame). This is the real payoff of `as_spec_target`.
3. **Port all other arches to Steps dispatch (SEPARATE thread — this note's non-goal).** `execute_steps` (Step IR) reach is scoped to llama arch 0/1; minimax/lfm2moe/cohere2moe/dots-ocr/deepseek4-EP/qwen35-MoE+DeltaNet still on bare `execute_steps` or hand-blocks. Decompose MoE/DeltaNet into single Steps per the DIRECTION AGREED (split WHAT from HOW/WHERE). Tracked in [[moe-deltanet-decompose-into-steps]] + [[device-mesh-pivot-execute-steps-spine]]. ORTHOGONAL to ArchDispatch: ArchDispatch = decode-LOOP dispatch (which arch's generate); Steps = per-LAYER op dispatch (how a forward runs across the mesh). They compose — an arch can be on ar_generate AND still hand-block its forward until Step-decomposed.
