## Summary

<one or two sentences: what changes, in behavioral terms>

## Which surface(s) does this touch?

Tick the surfaces this PR changes so a reviewer can match evidence to the claim map in `docs/VALIDATION.md`. Optional hw-gate automation uses the same buckets (`scripts/hw-gate/select.py`) when it runs.

- [ ] **kernel** — `kernels/`, `crates/rdna-compute`, `crates/hipfire-dispatch`, `crates/hip-bridge`, `crates/saddle-core`
- [ ] **load** — `crates/hipfire-loader`, `crates/hipfire-daemon`, runtime load path (`model_load`, `hfq`, `loader_api`, `config`, `safetensors_source`, `weight_backend`, `multi_gpu`), arch `load*`/`weights*`/`carrier.rs`, `hipfire-config`, `hipfire-registry`, `registry/`, Cargo manifests
- [ ] **serve** — `crates/hipfire-engine`, `crates/hipfire-generate`, daemon slots/serve, runtime emit/eos/dflash/dspark/spec/reset/triattn
- [ ] **arch crate(s)**: <list, e.g. `hipfire-arch-qwen35`, `hipfire-arch-gemma4`, `hipfire-arch-lfm2moe`>
- [ ] `crates/hipfire-quantize` / quant formats (update `docs/quant-formats/qt-register.txt`)
- [ ] control plane — `hipfire-cli`, `hipfire-client`, `hipfire-tui`
- [ ] docs / CI / scripts only (no hardware route)
- [ ] **policy files** — `.github/workflows/`, `CODEOWNERS`, `scripts/hw-gate/`, `leanup-thresholds.txt`, `layering.txt`, `registry/` (always human-owned; automation must not merge these)

## Test plan

- [ ] `./scripts/no-gpu-ci.sh` passes, or the CI jobs are green
- [ ] `cargo build --release` clean
- [ ] `cargo test --lib --workspace` passes
- [ ] **load / serve / kernel changes:** I ran `python3 scripts/serve_harness.py --model <flagship artifact> --mode battery --out battery.json` on hardware myself and attached `battery.json` below. A `hipfire run` transcript is not evidence.
- [ ] If perf-relevant: `./scripts/speed-gate.sh` within ±2% of locked baselines
- [ ] If this raises a ceiling in `scripts/leanup-thresholds.txt`: the commit message carries `RATCHET-RAISE: <metric> <old> -> <new>, traded for <reason>` **and** the PR carries the `ratchet-raise` label (CI fails without both)

<details><summary>local serve_harness battery.json (load / serve / kernel changes)</summary>

```json
paste the harness --out JSON here (per-turn rows with assistant_content, attractor, empty, finish, expected_substrings), plus the artifact sha256 and daemon md5
```

</details>

## Hardware validation request (optional)

Optional. Name registry artifacts and the claim they should prove. When hw-gate automation runs, Sol treats the claim as a claim and runs the routes you name (tags must exist on the runner; unknown or absent tags are reported, not failed). Leave the block out to rely on manual harness attachments and/or the automation's default fixtures for touched surfaces. Either path is evidence for direct review — not a required CI pass.

<!-- hw-gate-request -->
```json
{
  "routes": [
    {"mode": "battery", "tag": "qwen3.6:27b"},
    {"mode": "chain",   "tag": "ornith-1.5:35b-a3b-mq4r"}
  ],
  "claim": "loads the Qwen3.5-family text artifacts; no regression on the load path"
}
```

## How this merges (direct review)

Merge authority is **direct maintainer review** plus the required no-GPU CI checks. hw-gate automation is optional evidence delivery, not a prerequisite or substitute.

1. **Required CI** (must stay green): `build (workspace, no GPU)`, `unit tests (lib, no GPU)`, `gates (ratchets, layering, registers)` from [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
2. **Required review**: one approving maintainer review. The reviewer judges claim-matched evidence for the surfaces you ticked — static read for docs/control-plane-only work; hardware/model proof when load/serve/kernel/runtime behavior changes.
3. **Evidence you owe**: pick routes from [`docs/VALIDATION.md`](../docs/VALIDATION.md). Prefer attaching local harness output (`serve_harness.py`, `redline_daemon_harness.py`, `test_kernels`, etc.). A lone `hipfire run` transcript is not evidence.
4. **Optional automation**: when hw-gate runs, Sol/Fable may post seat commentary and hardware fixtures under `hipfire-sol[bot]` / `hipfire-fable[bot]`. A maintainer's `hw-run` label can force a hardware pass. Seat output informs the human reviewer; it does **not** auto-approve, auto-merge, or promote `master`, and hw-gate is **not** a required status check.
5. **Retired paths**: `scripts/coherence-gate*.sh` and `tools/change_gate` / agentic-review are historical only — never acceptance evidence. No local planning tool is merge evidence.

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
