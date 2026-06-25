# Further work: tiny-model testing

Open follow-ups for the tiny-model testing system (the seeded random-init
fixtures + the `tiny_quant` eval battery / `tests/tiny-quant-gate.sh`). The system
itself is landed and documented; adding a family is covered by
[`docs/howto/add-tiny-quant-family.md`](../howto/add-tiny-quant-family.md). These
are the loose ends, roughly in priority order.

## 1. No automated gate runs the tiny-quant *quality* matrix (highest value)

CI (`tests/no-gpu-ci.sh` → `.github/workflows/no-gpu-ci.yml`) gates only the **CPU
plumbing**: the fixture unit tests + the emit→quantize→arch-detect roundtrip
(`tests/fixture-roundtrip-nogpu.sh`, all families) + compile-checks of the battery
and `tiny_quant_probe`. The pre-commit hook now uses Git's staged path list via
`tests/tiny-affected-gate.sh` to run the smallest covered `tiny_quant` family set
first, then escalates to the large coherence battery on failure, missing tiny
coverage, or inconclusive rows. What is still missing is a **GPU CI / scheduled
fleet** run of the KLD / collect matrix — the part that caught the cross-arch MoE
NaN — on validation boxes.

So a future dequant-kernel or quant regression would not be caught automatically.

- **Do:** add a **GPU CI or scheduled fleet job** that runs
  `tiny-affected-gate.sh --base <protected-branch> --require-coverage` on PRs and
  `tiny-quant-gate.sh` as a periodic full matrix on the validation boxes
  (gfx1103 + halo/gfx1151). Model it on however the other GPU gates are triggered.
- **Don't:** put the GPU matrix in the pre-commit hook — per-commit GPU minutes +
  GPU-lock contention train people to `git commit --no-verify`, which defeats the
  correctness tripwires that legitimately live there.
- Optional, low value: a scoped `cargo test -p hipfire-quantize fixture` in the
  pre-commit front tier (≈1s, gated on `fixture.rs`/`tiny_harness.rs`/
  `executor_tinyquant.rs` being staged) — only helps no-GPU dev boxes.

## 2. minimax topk GPU-faults on gfx1151

`minimax` (arch 10) faults the GPU on gfx1151 in `deepseek4_moe_topk_bias_aware`
(VMFault / coredump). It works fine on gfx1103. It's currently excluded from the
gfx1151 record via `HIPFIRE_TINYQUANT_FAMILIES`, so minimax has **no gfx1151
baseline**.

- **Caveat to rule out first:** the failure was observed from an SSH `git worktree`
  whose kernel cache forced a JIT recompile (`pre-compiled blob has no hash file,
  recompiling`). It may be a bad-recompile artifact rather than a true kernel bug.
- **Do:** reproduce from halo's **canonical pre-warmed-cache environment** (not an
  ad-hoc worktree — a fault wedges the shared GPU). If it still faults with the
  cached kernel, it's a real gfx1151 topk bug; fix it and drop the
  `HIPFIRE_TINYQUANT_FAMILIES` exclusion, then record the gfx1151 minimax baseline.

## 3. llama has a gfx1103 baseline only

`llama` (arch 0) was added with gfx1103 baselines. Record gfx1151 on halo when
convenient — the toolchain env needed for a non-interactive SSH record is in the
HOWTO ("Recording baselines on another GPU"). gfx1151 currently covers the four
non-minimax, non-llama families.

## 4. Pre-commit hooks not enabled in this clone

`git config core.hooksPath` is unset, so `.githooks/pre-commit` (the GPU
fixture-golden tripwire + coherence battery) doesn't run locally. Enable with
`git config core.hooksPath .githooks` if you want the local tripwire active. Note
the pre-commit hook is **not** wired to the tiny tests (its only tiny step is the
GPU golden tripwire); see item 1.

## 5. Open design choice: missing-baseline status

A tiny-quant cell with no committed baseline currently reports **Skip** ("no
committed baseline (run --record)"), `Pass` only under `--record`. The hard-fail
checks (non-finite / zero-token / crash) still fire regardless. The stricter
alternative is **Fail** unless `--record` or an explicit allow-missing env var —
forces baselines to exist but blocks a brand-new GPU arch from going green until
recorded. We chose Skip (non-blocking, honest, mirrors fixture-golden's
"inconclusive"). Revisit if drift-without-a-baseline becomes a real foot-gun.

## 6. More families

`DeepSeek4` (9 — Q/O LoRA + hyper-connections + compressed-KV indexer),
`LFM2-MoE` (11 — hybrid short-conv + attention + MoE), and the VL families
(`dots.ocr` 8, `gemma3-vl` 13 — need image inputs) are not yet covered. Each is
harder than the dense families; start from the loader-recon checklist in the HOWTO.
Training-path testing is also out of scope so far.
