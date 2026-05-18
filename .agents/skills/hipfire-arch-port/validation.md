# Validation procedure for arch ports

Three gates. All three must pass before merging an arch port. Two
of them require **real hardware for the target arch** — there is no
emulator/simulator path. If you don't have hardware, you'll need a
contributor who does (`contributor-onboarding.md`).

## Gate 1: Channel-test (correctness)

This is the load-bearing one. WMMA / MFMA kernels have arch-specific
per-lane mappings, and getting the C-mapping wrong silently
corrupts all matrix outputs. The 6-week gfx11 WMMA bug (fixed in
commit `b7ac66a` — "wmma correctness fix + MQ6 family + cross-arch
prefill + gate framework") is a documented case of this passing the
speed-gate AND coherence-gate while silently producing garbage.
Channel-test caught it; the other two gates didn't.

### What it tests

`crates/hipfire-runtime/examples/test_kernels.rs` (and `test_kernelsQA.rs`)
runs each registered GPU kernel through a battery of small, golden-
output cases. Inputs are deterministic; outputs are compared
element-wise against a CPU reference within a tight tolerance.

### How to run

On the target hardware, with a fresh hipfire checkout:

```bash
cd hipfire
cargo build --release --features deltanet -p hipfire-runtime \
  --example test_kernels --example test_kernelsQA

./target/release/examples/test_kernels        # smoke battery
./target/release/examples/test_kernelsQA      # full QA matrix
```

### What "pass" looks like

Every dispatched kernel emits `OK` (or `PASS`) at the end of its
case. Any `FAIL` / `MISMATCH` is a hard block — the arch port is
not merge-ready.

### Hot tip

If `test_kernels` doesn't currently exercise the new arch's WMMA
path (because dispatch is gated on the arch and the arch isn't
listed), **add a test case first** that targets the new kernel
explicitly, then run the suite. Don't merge a port that has no
correctness coverage on its specific kernel.

## Gate 2: Coherence-gate (output sanity)

`scripts/coherence-gate.sh` runs a small fixed AR prompt matrix through
the daemon and writes a markdown report. `scripts/coherence-gate-dflash.sh`
is the canonical correctness gate for changes touching kernels, quant
formats, dispatch, fusion, rotation, rmsnorm, or the spec-decode path.

### What it tests

- Daemon doesn't panic on the arch.
- Generation emits non-zero tokens within timeout.
- Outputs are coherent (English, on-topic, not stuck in a loop).

The hard-fail conditions are panic / zero-tokens / timeout. Soft
output diffs (different but still correct answers) are not block-
ing — the human / agent reviewer reads the report.

### How to run

```bash
./scripts/coherence-gate.sh           # AR smoke — 0.8b/4b/9b dense
./scripts/coherence-gate.sh --full    # AR smoke plus A3B MoE
./scripts/coherence-gate-dflash.sh    # canonical gate for affected code
```

### Important note for arch ports

A *new* arch may not have its WMMA path exercised by the
coherence-gate matrix if the gate is running models that fall
through to the dispatch fallback. Check the daemon's stderr — if
you don't see `[verify-graph] captured...` for prefill or the
arch-specific dispatch print, you're not actually testing the
ported path. Either:

- Force the arch's path via env (`HIPFIRE_FORCE_ARCH=gfx1201` if
  such a knob exists; otherwise temporarily edit dispatch.rs to
  remove the fallback for the gate-run only), OR
- Add a model + prompt to the gate matrix that's known to require
  the arch-specific kernel.

## Gate 3: Speed-gate (no regression on the baseline arch)

`scripts/speed-gate.sh --fast` benches 4b prefill + decode against
the committed ground-floor baselines in `tests/speed-baselines/`.

### Why this matters even for an arch port

Every change to `dispatch.rs` MUST run the speed-gate on the
baseline arch (gfx1100). The pre-commit hook enforces this; do not
bypass it.

In this session, commit `a048544` (a "no-op" predicate refactor)
was reported by the speed-gate as a 50% prefill regression on
gfx1100 and was reverted in `1f3bad3`. **Re-tested 2026-04-27 in
commit 6e100c2 with forced rebuild — the regression was a stale-
binary measurement artifact**, not a real codegen difference. The
gate's `ensure_build()` is a no-op when the bench binary already
exists, so a "stash and re-bench" flow leaves the post-change
binary in place during the "before" measurement, and both runs
measure the same binary.

**The fix is simple but mandatory:** `rm
target/release/examples/bench_qwen35_mq4` before re-running the
gate to compare diffs. The gate's `ensure_build()` will detect
the missing binary and rebuild from current sources. Without
this, your "regression" measurement might be measuring the wrong
code entirely.

### How to run

```bash
./scripts/speed-gate.sh --fast        # 4b only, ~30s
./scripts/speed-gate.sh                # all sizes (0.8b/4b/9b/27b)
./scripts/speed-gate.sh --verbose      # full bench output
```

Tolerance is ±5% from the committed baseline. The pre-commit hook
runs `--fast` automatically when the staged diff touches kernel,
dispatch, forward-pass, or engine code.

### If you legitimately trade speed for something else

If your arch port intentionally regresses gfx1100 perf (highly
unusual but possible — e.g., a refactor that enables a much faster
gfx12 path at the cost of a small gfx11 slowdown), justify it AND
re-baseline:

```bash
./scripts/speed-gate.sh --update-baselines
git add tests/speed-baselines/
git commit                            # runs hook again, now passes
```

The baseline change must be in the SAME commit as the regressing
diff so reviewers see the trade-off explicitly.

## What you do NOT do

- **Do not bypass the speed-gate with `--no-verify`.** The repo's
  contract: this is a red line unless explicitly authorized by the
  maintainer in writing. (Documented this session: `--no-verify`
  bypass on commit `a048544` was a contract violation; the bypass
  hid that the change actually DID regress perf, leading to a
  revert.)

- **Do not assume "my change is functionally identical so the gate
  doesn't apply."** It might not be. Run the gate.

- **Do not merge without all three gates green** unless you have
  the maintainer's written sign-off on a known limitation (e.g.
  "channel-test deferred to follow-up because hardware not
  available").

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Channel-test FAIL on a new arch | Per-lane C-mapping wrong | Add `eprintln!` of `(tid, acc[j])` for first warp, compare to CPU reference, derive correct mapping. See commit `b7ac66a` for the gfx11 case. |
| Coherence-gate panic | Codegen failure, missing kernel file, bad dispatch | Read the panic message; usually a stack trace from `dispatch.rs` or `kernels.rs` |
| Coherence-gate zero-tokens | Daemon stops at EOS immediately, often a tokenizer / chat-template / KV-init bug | Check `m.seq_pos` and `prompt_tokens` in the daemon's generate path. Common cause: chatml wrap missing the trailing `\n` after `<|im_end|>`. |
| Speed-gate regress on gfx1100 from "should-be-no-op" refactor | **Most common cause**: gate's `ensure_build` is a no-op when the bench binary exists, so "stash and re-bench" measures the same binary twice. Less common: real codegen difference, GPU thermal drift, firmware shadowing. | (1) `rm target/release/examples/bench_qwen35_mq4` to force a clean bench-binary rebuild (the gate detects missing exe and rebuilds); (2) `cargo clean -p rdna-compute && cargo build --release ...` to invalidate the dispatch crate; (3) `cat /sys/class/drm/card*/device/pp_dpm_sclk` to check DPM state; (4) `dmesg \| tail -40` for firmware errors; (5) re-run gate. If still regressed, revert the diff and bisect |
| Speed-gate regress with system in known-good state | Firmware shadowing — `/lib/firmware/updates/amdgpu` overrides kernel firmware → SMU IF mismatch | `sudo mv /lib/firmware/updates/amdgpu /lib/firmware/updates/amdgpu.bak && sudo reboot` |

## Last verified

This procedure was used to validate gfx1100 (RDNA3) and gfx1030
(RDNA2) ports. Current checkouts include gfx12 WMMA sibling kernels and
selectors for multiple paths, while some MQ3-Lloyd gfx12 fast paths remain
explicitly env-gated (`HIPFIRE_LLOYD_GFX12`) until external RDNA4 parity and
coherence data is accepted. Before changing an arch route, inspect the current
`kernels.rs` selector and `dispatch.rs` branch rather than relying on old
issue status.
