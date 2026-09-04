## Summary

<one or two sentences: what changes, in behavioral terms>

## Which surface(s) does this touch?

hw-gate selects hardware routes from the diff (`scripts/hw-gate/select.py`); tick what applies so a reviewer can check the selection.

- [ ] **kernel** — `kernels/`, `crates/rdna-compute`, `crates/hipfire-dispatch`, `crates/hip-bridge`, `crates/saddle-core`
- [ ] **load** — `crates/hipfire-loader`, `crates/hipfire-daemon`, runtime load path (`model_load`, `hfq`, `loader_api`, `config`, `safetensors_source`, `weight_backend`, `multi_gpu`), arch `load*`/`weights*`/`carrier.rs`, `hipfire-config`, `hipfire-registry`, `registry/`, Cargo manifests
- [ ] **serve** — `crates/hipfire-engine`, `crates/hipfire-generate`, daemon slots/serve, runtime emit/eos/dflash/dspark/spec/reset/triattn
- [ ] **arch crate(s)**: <list, e.g. `hipfire-arch-qwen35`, `hipfire-arch-gemma4`, `hipfire-arch-lfm2moe`>
- [ ] `crates/hipfire-quantize` / quant formats (update `docs/quant-formats/qt-register.txt`)
- [ ] control plane — `hipfire-cli`, `hipfire-client`, `hipfire-tui`
- [ ] docs / CI / scripts only (no hardware route)
- [ ] **policy files** — `.github/workflows/`, `CODEOWNERS`, `scripts/hw-gate/`, `leanup-thresholds.txt`, `layering.txt`, `registry/` (hard floor: no seat can merge these; a human does)

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

Tell the gate which registry artifacts prove your change, and what you claim. Sol reads the claim as a claim and runs the routes you name (tags must exist on the runner; unknown or absent tags are reported, not failed). Leave the block out and the gate runs the mandatory fixtures for the surfaces you touched.

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

## How this merges (hw-gate)

Two model seats, one human owner. Every decision is announced on the PR.

1. **Sol reads the diff** (read-only) and decides whether your code runs on the maintainer's hardware and which routes run — the mandatory fixtures for the surfaces you touched, plus the routes you requested, plus any Sol adds. Only Sol decides this; a maintainer's `hw-run` label can force a run, nothing can silently block one. Skipped on drafts.
2. **Hardware runs**: the PR is built and every route is driven through `serve_harness.py` on gfx1201. Every turn's decoded text is posted verbatim in the evidence comment. A missing or mismatched pinned fixture, an attractor, an empty turn, or a missed expect-substring is a failure.
3. **Sol's verdict**: `greenlight` / `needs-human` / `block` on diff + evidence, with regressions cited by `file:line` and fixture. Sol never merges.
4. **Fable investigates and decides**: with a shell in a sandboxed checkout of your head on the hardware (all five hiptrx GPUs, base branch built for A/B), Fable runs whatever proves your change — multi-GPU loads, refusal sequences, parity, A/B — and returns `merge-staging` / `hold` / `block` with an investigation table and every evidence file. It may veto a greenlight or override a needs-human, and says why. On `merge-staging` Fable merges your head into **`beta`** (staging); `master` is promoted by the maintainer. Neither seat can override the hard floor: a failed fixture, an attractor, a policy-file change, or an unlabelled `RATCHET-RAISE`.
5. **The `hw-gate` status** is green only on `merge-staging`; `hold` turns green when a maintainer applies `human-reviewed`; `block` clears only with a new commit.

The seats act as `hipfire-sol[bot]` and `hipfire-fable[bot]`. Route policy: [`docs/VALIDATION.md`](../docs/VALIDATION.md) § hw-gate — request coverage with the `<!-- hw-gate-request -->` block above; no local planning tool is CI evidence. The retired `scripts/coherence-gate*.sh` batteries no longer exist.

## Architecture-trait change?

If this PR changes the `Architecture` trait surface in
`crates/hipfire-runtime/src/arch.rs`, note here. Trait changes ripple
to every arch crate.
