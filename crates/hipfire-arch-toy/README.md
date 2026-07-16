# hipfire-arch-toy

A minimum-viable [`Architecture`] trait implementation. Use as a starting
template for new arch crates.

## What this crate is

The smallest impl that compiles, type-checks against
`hipfire_runtime::arch::Architecture`, and demonstrates every required
method. Each method is a one-liner with a doc-comment explaining what a
real arch would do.

The three optional override structs (`SamplerOverrides`,
`PromptFrameOverrides`, `EosFilterOverrides`) are
shown with their default value plus an example of when you'd override.

## What this crate is not

- Not a real model. `config_from_hfq` ignores its input and returns
  hardcoded constants; `load_weights` returns a zero-initialized
  embedding table; `new_state` returns a bare counter. There is no
  forward pass.
- Not consumed by the daemon, runtime examples, or any binary. It
  exists purely as a template. The workspace builds it to keep the
  template from rotting, but nothing depends on it at runtime.
- Not a vehicle for shared scaffolding. If a piece of code is useful
  to more than one arch crate, it belongs in `hipfire-runtime`, not
  here.

## How to use this as a template

1. Copy `crates/hipfire-arch-toy/` to `crates/hipfire-arch-<your-name>/`.
2. Update `Cargo.toml`: `name`, `description`. Add the new crate to
   the workspace `Cargo.toml` `members` list and to
   `crates/hipfire-runtime/Cargo.toml`'s `[dev-dependencies]` if your
   arch is consumed by a runtime example/binary.
3. Replace the stub types in `src/toy_model.rs`:
   - `ToyConfig` → your arch's config (parsed out of
     `hfq.metadata_json`). Reference: `Qwen35Config::from_hfq` in
     `hipfire-arch-qwen35`.
   - `ToyWeights` → your arch's GPU-resident weight handles. Use
     `WeightTensor` from `hipfire-runtime::llama` for upload + dispatch
     plumbing; arch crates do not implement quant unpack themselves.
   - `ToyState` → your arch's per-step scratch (KV cache, attention
     workspace, recurrent state for hybrid archs).
4. Update `src/arch.rs` `impl Architecture for Toy` to call your new
   types. The trait surface is fixed: same five required methods
   (`arch_id`, `name`, `config_from_hfq`, `load_weights`, `new_state`)
   plus the four optional overrides.
5. Implement your forward pass as free functions in your model module
   (e.g. `forward`, `forward_prefill_batch`). The trait deliberately
   does **not** route forward through dyn dispatch — see the rationale
   in `crates/hipfire-arch-qwen35/src/arch.rs` module docs.
6. Add an `arch_id` constant to `docs/architecture-ids.md` (when that
   file exists) or coordinate via PR review to claim an unused id.

Rough effort estimate for a real arch port:
- Bring-up triple (config / weights / state): a few hundred lines.
- Forward pass: a couple thousand lines for a dense LLaMA-style model;
  more for hybrid attention or MoE.
- Kernel work for any new ops: stays in `kernels/src/*.hip` and
  `crates/rdna-compute`, **not** in your arch crate. See
  `CONTRIBUTING.md` "Crate topology" decision tree.
- Coherence + speed-gate validation: see `scripts/coherence-gate.sh`
  and `scripts/speed-gate.sh`.

## Production reference

Read `crates/hipfire-arch-qwen35/` for a complete arch implementation
with hybrid DeltaNet attention, MoE expert routing, weight paging,
speculative decoding, and PFlash long-context paging. That's the bar.

[`Architecture`]: ../../crates/hipfire-runtime/src/arch.rs
