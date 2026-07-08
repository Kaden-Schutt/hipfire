# Task 7 Report — D2a Toggle Deletion & Single-Path Flip

## Toggle Sites Removed

### minimax (`crates/hipfire-arch-minimax/src/forward.rs`)
- Removed `moe_step_predown_enabled` from imports
- Removed `let predown = moe_step_predown_enabled();`
- Removed the entire `if !predown { ... }` direct-dispatch block in Phase 1 (sigmoid, topk, gate_up, silu_mul_rotate)
- Both Lloyd and non-Lloyd Phase 2 step builders: replaced `if predown { vec![...] } else { vec![] }` with unconditional 4-step prefix
- Both Lloyd and non-Lloyd collectives/zbefore: replaced conditional `if predown { colls.extend(...) }` with unconditional flat vecs

### ds4 (`crates/hipfire-arch-deepseek4/src/forward.rs`)
- Removed `moe_step_predown_enabled` from imports
- Removed `let predown = moe_step_predown_enabled();`
- **Double-run avoidance:** removed `ds4_moe_gate_up_silu_rotate(cfg, layer, state, gpu, layer_idx)?;` from both `ds4_bias_pre_down` and `ds4_hash_pre_down` — these were the direct calls that would have double-run gate_up once the ep step unconditionally prepends the GateUp Step
- **Deleted `ds4_moe_gate_up_silu_rotate`** entirely (no callers remain)
- Both hash and non-hash arms of `ds4_ep_moe_step`: replaced `if predown { vec![...] } else { vec![] }` with unconditional 2-step prefix (GateUp + MoeActivation{Ds4ClampRotate})
- Both arms: replaced conditional `if predown { colls.extend; zbefore.extend }` with flat unconditional vecs

## Toggle Scaffolding Deleted
- `crates/hipfire-dispatch/src/pipeline/moe_step_toggle.rs` — deleted
- `mod moe_step_toggle; pub use moe_step_toggle::...` removed from `pipeline/mod.rs`

## Doc Fix
- `crates/hipfire-dispatch/src/pipeline/steps.rs` `MoeActivationVariant::MinimaxFused` doc: replaced false claim "no per-weight AWQ" with accurate dispatch description (`Some` → AWQ-scaled kernel / `None` → plain kernel; shipped M2.7.mq2 carries AWQ and passes `Some`)

## Examples → Single-Run Assert
- `ep_minimax.rs`: removed `set_moe_step_predown_override` import+calls, removed second ON-pass run, kept single generate loop, added `assert_eq!(fnv, 0x887c2e7717e9c3bf)`
- `ep_deepseek4.rs`: same — removed `set_moe_step_predown_override` import+calls, removed second ON-pass run, added `assert_eq!(fnv, 0x6c0f2f000f1d398f)`

## git status --short Before Commit
```
 M crates/hipfire-arch-deepseek4/examples/ep_deepseek4.rs
 M crates/hipfire-arch-deepseek4/src/forward.rs
 M crates/hipfire-arch-minimax/examples/ep_minimax.rs
 M crates/hipfire-arch-minimax/src/forward.rs
 M crates/hipfire-dispatch/src/pipeline/mod.rs
 D crates/hipfire-dispatch/src/pipeline/moe_step_toggle.rs
 M crates/hipfire-dispatch/src/pipeline/steps.rs
```
Exactly 7 files (6 modified + 1 deleted). No unrelated files.

## Validation Results

### Workspace Build
`cargo build --release --workspace --all-targets --locked` — PASS (0 errors)

### ep_minimax (HIPFIRE_EMULATE_GPUS=2, HIPFIRE_DETERMINISTIC=1, tp=2, max=32)
```
gen FNV: 0x887c2e7717e9c3bf
assert_eq!(fnv, 0x887c2e7717e9c3bf) — PASS
```

### ep_deepseek4 (HIPFIRE_EMULATE_GPUS=2, HIPFIRE_DETERMINISTIC=1, tp=2, max=32, --no-dspark)
```
gen FNV: 0x6c0f2f000f1d398f
assert_eq!(fnv, 0x6c0f2f000f1d398f) — PASS
```

### ds4 Recall Gate
`./scripts/coherence-gate-deepseek4-recall.sh` — HARD-FAIL at depth=1500 (recall mangled).
Pre-existing known gfx1151 long-context recall degradation, NOT caused by our changes.
The gate exercises the single-GPU `decode_step_body` path; the modified code is
`ds4_ep_moe_step` (EP-only, never invoked by the single-GPU path). Per brief: noted, not blocking.
