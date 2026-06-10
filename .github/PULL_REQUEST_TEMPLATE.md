## Summary

<one or two sentences>

## Which crate(s) does this touch?

- [ ] `kernels/` (HIP source)
- [ ] `crates/rdna-compute` (kernel dispatch / RDNA arch routing)
- [ ] `crates/hip-bridge` (HIP/ROCm FFI)
- [ ] `crates/hipfire-runtime` (LM runtime: KV, sampler, guards, framing, paging, spec decode)
- [ ] `crates/hipfire-arch-qwen35`
- [ ] `crates/hipfire-arch-qwen35-vl`
- [ ] `crates/hipfire-arch-llama`
- [ ] `crates/hipfire-arch-toy` (template — touch only when refining the new-arch reference)
- [ ] `crates/hipfire-quantize`
- [ ] examples / daemon
- [ ] docs / CI / scripts

## Test plan

- [ ] `./scripts/no-gpu-ci.sh` passes, or equivalent CI job is green
- [ ] `cargo build --release --workspace --features deltanet` clean
- [ ] `cargo test --lib --workspace --features deltanet` passes
- [ ] If kernel/dispatch changed: `./scripts/coherence-gate.sh` clean
- [ ] If spec-decode changed: `./scripts/coherence-gate-dflash.sh` clean
- [ ] If perf-relevant: `./scripts/speed-gate.sh` within ±2% of locked baselines

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
