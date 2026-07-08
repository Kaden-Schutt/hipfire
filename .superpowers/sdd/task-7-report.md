# Task 7 report — minimax `forward_ep` decomposed via `execute_steps_parallel`, gated `HIPFIRE_MOE_STEP`

## Status: DONE (decode byte-identity proven on emulated EP-2). Prefill: per-token prefill covered byte-identically; batched-grouped-EP prefill scoped out (see Concerns).

## What shipped
- **`HIPFIRE_MOE_STEP` gate** (`minimax_moe_step_enabled()`, default OFF) + a parity
  override (`set_minimax_moe_step_override(Option<bool>)`) so a single-process
  harness can A/B both arms after ONE 79 GB load (the env read is `OnceLock`-cached).
- **Decomposed arm `minimax_ep_moe_step`** (additive; primitive arm stays default):
  - Phase 1 (pre-down): rmsnorm → input FWHT → router GEMV → sigmoid → bias-aware
    top-K → indexed gate/up → fused silu*mul+rotate. Runs the SAME direct arch
    kernels in the SAME order as the primitive's `minimax_moe_block` (these fused
    ops have no bit-identical Step twin).
  - Phase 2 (down + reduce): a single-Step per-rank list — `Step::IndexedMoeGemv`
    with `MoeProj::DownResidual` (MQ3-Lloyd residual-scaled kernel, folds the
    weighted combine straight into the per-rank EP partial) — driven through
    `execute_steps_parallel` with `zero_before=[true]` (mirrors the primitive
    memset) and `StepCollective::AllReduce { kind: Ep, dim: hidden }`. EP mesh is
    `DeviceMesh::rect(&[(Ep, n)])` so `group_along(Ep) == group`.
  - Phase 3 (residual fold): `state.h += partial` per rank (== primitive `AddResidual`).
- **`execute_steps_parallel`, `StepCollective`, `MoeProj` re-exported** from
  `hipfire_dispatch::pipeline` (Task 7 is their first external consumer).
- **Parity harness** `crates/hipfire-arch-minimax/examples/moe_step_ep_parity.rs`
  (placed in the minimax crate, NOT hipfire-runtime as the brief listed —
  hipfire-runtime cannot depend on an arch crate; `forward_ep` lives here and the
  `ep_minimax.rs` template is here).

## FNVs observed (emulated EP-2: `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1`, `--tp 2 --max 32`, prompt "The capital of France is")
- **flag OFF (primitive):** `0x887c2e7717e9c3bf` (32 tok)
- **flag ON  (decomposed):** `0x887c2e7717e9c3bf` (32 tok)
- **matched:** YES. **first_div:** `None`. Decoded text identical + fluent
  (" Paris. The capital of Germany is Berlin. The capital of Italy is Rome...").

The A/B is byte-exact: same process, same loaded weights, same fresh state, only
the EP-MoE arm differs. The prompt's 5-token prefill runs through the decomposed
`forward_ep` per-token in both arms, so per-token EP prefill is included in this
byte-identical result.

Note on the brief's quoted reference `0x31ede7c1d1cf140e`: my flag-OFF baseline is
`0x887c2e7717e9c3bf`. The reference was a DIFFERENT config — the brief's run
command under-specifies `--tp/--max/--prompt`; EP-2 vs EP-4 differ in reduction
order (fp non-associativity) and the template `ep_minimax.rs` defaults `--tp 4`.
The load-bearing claim (flag-ON reproduces flag-OFF EXACTLY on identical config)
is what I observed and is what byte-identity requires — I did not observe, and do
not claim, the `0x31ed…` number.

## Decode / Prefill
- **Decode:** byte-identical (above).
- **Prefill (per-token):** byte-identical — every prompt token goes through the
  decomposed `forward_ep` and produces the byte-identical generation above.
- **Prefill (batched grouped Steps, brief Step 5):** NOT built. Rationale: the
  existing `forward_ep` is per-token only; there is NO batched-EP primitive in the
  codebase to assert byte-identity against. A batched grouped MoE
  (scatter/GroupedMoeGemm/unscatter + `MoeCombine{inverse_perm:Some}`) uses
  DIFFERENT kernels and accumulation order than per-token indexed MoE, so a
  "prefill-batch FNV == primitive prefill" assertion is unachievable by
  construction. Deferred as a separate capability rather than shipping an
  unverifiable comparison.

## Verification
- `cargo build --release --workspace --all-targets --locked` — green.
- `cargo test -p hipfire-dispatch --lib` — 168 passed (incl. `execute_steps_parallel`
  arg-validation tests).
- rustfmt clean on all three changed files (forward.rs check = no whole-file debt).
- clippy: no errors; only advisory nits.

## Concerns
1. **Pre-down ops stay arch kernels, not Steps.** sigmoid, fused rmsnorm+rotate,
   and fused silu*mul+rotate have no bit-identical Step variant; forcing them into
   `SiluMul`+separate-rotate etc. risks breaking byte-identity (different kernels).
   The executor drives the down-projection + EP collective (the part that needs the
   parallel executor, `zero_before`, and `AllReduce{Ep}`). Full step-ification of
   the fused pre-down ops is deferred pending bit-identical Step variants —
   consistent with the P-D "deferred decomposition" stance.
2. **Absolute FNV ≠ brief's reference** (config difference, see above).
3. **`peer_access_enabled=false` on emulated single-GPU** — the peer all-reduce is a
   same-device copy; correct + byte-identical here, but real 2-GPU HW is untested
   (no hardware; consistent with the branch's standing "ZERO real 2-GPU HW" note).
4. **Harness location** differs from the brief (minimax crate, not hipfire-runtime)
   due to the crate dependency direction.

## Report path
`/home/bjoern/hipfire/.claude/worktrees/feature+device-mesh/.superpowers/sdd/task-7-report.md`
