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
