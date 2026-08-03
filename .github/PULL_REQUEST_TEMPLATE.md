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
- [ ] **Change gate run and telemetry pasted below.** `python3 -m tools.change_gate plan --base beta`
      shows what your diff actually owes and what it costs; `run --md gate.md` executes it and emits the
      report. The gate selects routes from the diff, so an unrelated change does not owe a model battery.
- [ ] If perf-relevant: `./scripts/speed-gate.sh` within ±2% of locked baselines
- [ ] Any route reported `blocked` (absent model, wrong arch) is **acknowledged below**, not ignored —
      a blocked route makes the gate verdict `incomplete`, which is not a pass.

<details><summary>change_gate telemetry</summary>

```
paste `python3 -m tools.change_gate run --base beta --md -` output here
```

</details>

Route selection is defined in [`tools/change_gate/routes.py`](../tools/change_gate/routes.py) and route
policy in [`docs/VALIDATION.md`](../docs/VALIDATION.md). The retired `scripts/coherence-gate*.sh`
batteries are **not** acceptance evidence and no longer exist in-tree. Adding coverage means adding a
`Route` + `Rule` there, not resurrecting a fixed battery.

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
