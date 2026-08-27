# Task 4 Report: Gemma4 Typed Step Lowering

## Status

Complete. The additive Step route is implemented; the existing SuperOp and hand routes remain selected exactly as before.

## Commits

- `44a5d313d` — `refactor(gemma4): lower decoder layers to typed Steps`

## Files

- `crates/hipfire-arch-gemma4/src/lowered.rs`

## Red/green evidence

Red (before the production descriptors existed):

```text
$ cargo test -p hipfire-arch-gemma4 sliding_dense_step_order_is_total --lib
error[E0432]: unresolved imports `super::gemma4_op_sequence`, `super::Gemma4Op`
```

Green:

```text
$ cargo test -p hipfire-arch-gemma4 step_order_is_total --lib
1 passed
$ cargo test -p hipfire-arch-gemma4 full_attention_copies_k --lib
1 passed
$ cargo test -p hipfire-arch-gemma4 moe_replaces_dense_post_ffn_norm_once --lib
1 passed
$ cargo test -p hipfire-arch-gemma4 --lib
20 passed
$ cargo test -p hipfire-dispatch gelu_expert_step --lib
2 passed
```

The focused commands emitted only the two pre-existing `sliding_cap` unused-variable warnings.

## Exact operation sequences

`Gemma4Op` is private architecture metadata and never enters dispatch or is encoded.

- `SlidingDense` — **23**: `NormInput, ProjQ, ProjK, ProjV, NormQ, NormK, NormV, ScaleQ, RopeFull, Attend, ProjO, NormPostAttn, ResidualAddAttn, SaveResidual, NormPreFfn, ProjGate, ProjUp, GeluTanhMul, ProjDown, NormPostFfn, RestoreResidual, ResidualAddFfn, ScaleLayer`.
- `FullDense` — **23**: `NormInput, ProjQ, ProjK, CopyKToV, NormQ, NormK, NormV, ScaleQ, RopePartial, Attend, ProjO, NormPostAttn, ResidualAddAttn, SaveResidual, NormPreFfn, ProjGate, ProjUp, GeluTanhMul, ProjDown, NormPostFfn, RestoreResidual, ResidualAddFfn, ScaleLayer`.
- `SlidingMoe` — **32**: `NormInput, ProjQ, ProjK, ProjV, NormQ, NormK, NormV, ScaleQ, RopeFull, Attend, ProjO, NormPostAttn, ResidualAddAttn, SaveResidual, NormPreFfn, ProjGate, ProjUp, GeluTanhMul, ProjDown, NormPostFfn1, NormPreFfn2, NormRouter, ScaleRouter, ProjRouter, MoeSoftmaxTopK, MoeExperts, NormPostFfn2, ResidualAddMoe, NormOuterFfn, RestoreResidual, ResidualAddFfn, ScaleLayer`.
- `FullMoe` — **32**: `NormInput, ProjQ, ProjK, CopyKToV, NormQ, NormK, NormV, ScaleQ, RopePartial, Attend, ProjO, NormPostAttn, ResidualAddAttn, SaveResidual, NormPreFfn, ProjGate, ProjUp, GeluTanhMul, ProjDown, NormPostFfn1, NormPreFfn2, NormRouter, ScaleRouter, ProjRouter, MoeSoftmaxTopK, MoeExperts, NormPostFfn2, ResidualAddMoe, NormOuterFfn, RestoreResidual, ResidualAddFfn, ScaleLayer`.

The full variants copy pre-normalization K into V; MoE variants replace the dense `NormPostFfn` with the two branch norms, router, typed GELU expert step, MoE combine, and outer norm.

## Borrow/allocation proof

Each variant helper creates a fixed `[Step<'_>; 23]` or `[Step<'_>; 32]` local on the stack. Projection `WeightRef` locals, `MoeGeluExpertsRef`, `KvTierPlan`, and `AttnParams` borrow model/scratch/cache storage for the same scope. The array is passed immediately to `execute_bound_gemma4_steps`, which accepts a borrowed slice and returns before those locals leave scope. No Step program is returned or stored, no per-token `Vec` is constructed, and the Task 4 code contains no `unsafe`, self-referential storage, callback, architecture opcode, or fallback route.

## Concerns

- The new entry point is intentionally not selected by production forwarding in Task 4; Task 5 owns hand-versus-Step numerical/capture parity wiring.
- No GPU model run was performed, per the brief. Focused compile/unit evidence covers the structure and typed dispatch contracts; runtime parity remains Task 5 work.

## Self-review

- Validation runs before `DispatchCtx`, `WeightRef`, or `Step` construction and reports the required configured/actual layer-type message.
- MoE lowering rejects `top_k_experts != 8` before execution.
- Sliding uses q/k/v normalization, Q scaling, full RoPE, and windowed attention; full uses K-to-V copy before q/k/v normalization and partial RoPE.
- Attention residual add/save, FFN residual restore/add, and layer scalar ordering match the required sequences.
- Existing `Gemma4Bindings`, SuperOp opcodes, `lower_variant`, hand layer functions, and production route selection were not removed or changed.
- No formatter, linter, project-wide suite, model run, or unrelated build was run.
