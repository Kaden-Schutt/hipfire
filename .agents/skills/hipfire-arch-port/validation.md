<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Arch-port validation (thin route)

This skill file is **not** a second validation authority. Route selection lives
in [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). Arch-port claim class
and the cheap forward-proof method live in
[`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md).

| Field | Value |
|---|---|
| Role | Arch-port evidence checklist + pointers |
| Route selector | `docs/VALIDATION.md` |
| Arch-port method | `docs/methodology/arch-port-validation.md` |
| Perf protocol | `docs/methodology/perf-benchmarking.md` |
| Admissions | `docs/admissions.yml` (schema only; `records: []`) |

## Rules (fail closed)

1. Pick the **narrowest** route in `VALIDATION.md` for the changed surface.
2. There is **no universal replacement gate**.
3. Automatic no-GPU CI is never GPU/kernel correctness.
4. `serve_harness.py` is **user-facing serve semantics only** — not numerical
   or state parity.
5. Fixed `scripts/coherence-gate-*.sh` batteries are **retired as acceptance**.
   Historical reproduction only; never merge/promotion evidence.
6. A green route does **not** create an `admissions.yml` row.

## Arch-port evidence stack

For a **GPU architecture port** (new gfx / WMMA-MFMA routing / arch-conditional
dispatch), minimum evidence is the **Arch port** row in `VALIDATION.md`:

| Layer | Path / command | Proves | Does not prove |
|---|---|---|---|
| **Channel (numeric)** | Build + run `target/release/examples/test_kernels` (`cargo build --release --features deltanet --example test_kernels -p hipfire-runtime`). Add arch-specific WMMA examples under `crates/hipfire-runtime/examples/test_wmma_*.rs` when the generic battery does not hit the new builtin. | Kernel outputs vs CPU/reference on the **detected** GPU | Dispatch bind coverage; model coherence; product admission |
| **Bind invariant** | `.githooks/pre-commit` → `scripts/verify-bind-thread.sh` (or run that script manually) | Every public `dispatch.rs` entry binds the HIP thread | Numerics |
| **Speed floor (when policy applies)** | `scripts/speed-gate.sh` (`--fast` for the short arm) vs committed `tests/speed-baselines/<arch>.txt`; ±5% (`TOLERANCE=0.05`). Re-baseline only with `--update-baselines` **in the same change** as an intentional trade-off. | Only the scripted baseline metrics it runs (`bench_qwen35_mq4` plus its named DFlash arms) — not every touched dispatch/kernel path. If the changed path is not exercised, run separate matched measurement or mark speed coverage blocked | Correctness of the new arch; admission; unexercised paths |
| **Forward / state parity** | Arch-owned oracle when one exists (`dump_*_hidden_states`, graph/parity examples, kernel channel tests). Method: `docs/methodology/arch-port-validation.md` (tiny oracle → per-layer cosine → bisect → precision sweep). | Hidden/logit/KV/conv parity for that surface | Serve framing; Redline product route |
| **Serve semantics (if user-visible)** | `scripts/serve_harness.py` with the exact model and registry tag; use `--sampling recipe:nothink` for LFM non-thinking framing | Finish reasons, empty/runaway, timing hooks, LFM framing | Numerical parity; Redline proof |
| **Optional wrapper** | `scripts/gates.sh --model …` | Manual Redline capture + serve battery + optional `probe_commits.sh` perf arm | Universal gate; retired coherence batteries |
| **Perf claim** | `docs/methodology/perf-benchmarking.md` + matched fresh-process runs; identity hashes | Measured delta under stated conditions | Floor, default, or admission |
| **Redline / PM4-AQL** | `scripts/redline_daemon_harness.py` **and** ladder in `docs/REDLINE.md` | Capture/fingerprint evidence under manual env | Installed product timed-arm without certification |

Hardware for the **target** arch is required for channel evidence. There is no
emulator path for WMMA/MFMA lane mapping.

## What to add when porting a kernel path

1. Ensure `test_kernels` (or a dedicated `test_wmma_*` example) **actually
   launches** the new arch’s builtin/path. Dispatch fallbacks that never select
   the new kernel are not coverage.
2. Confirm routing with `ArchCaps` predicates in
   `crates/rdna-compute/src/arch_caps.rs` (`has_wmma_w32` vs
   `has_wmma_w32_gfx12`, `is_rdna4`, per-chip atoms) — inspect current
   `kernels.rs` / `gemm.rs` / `attention.rs` branches; do not trust old issue
   text.
3. Env-gated paths stay gated until parity is earned (example:
   `HIPFIRE_LLOYD_GFX12=1` for Lloyd MQ3/MQ4 WMMA on gfx1200/gfx1201). Default-off
   is not “shipped default-on.”
4. If no path-specific parity oracle exists for the surface → **blocked** on
   numerical/state claims (`VALIDATION.md`).

## Explicit non-routes (arch-port)

| Anti-pattern | Disposition |
|---|---|
| Require `scripts/coherence-gate-*.sh` for merge/promotion | **Rejected** (retired acceptance) |
| Treat `test_kernelsQA` as the sole canonical CI name | Prefer the `VALIDATION.md` entry (`test_kernels`); QA example is optional extra local matrix if present |
| Green no-GPU CI as GPU correctness | **Rejected** |
| `serve_harness` as parity or Redline proof | **Rejected** |
| Bench number without protocol + binary/model identity | **Rejected** as promotion evidence |
| Inferred admission from a green harness | **Rejected** — keep `admissions.yml` empty until schema/policy defines records |

## Troubleshooting (channel / speed only)

| Symptom | Likely cause | Action |
|---|---|---|
| Channel FAIL on new WMMA path | Wrong C-lane mapping or K half-tile | Dump first-warp `(tid, acc[j])` vs CPU; see `wmma-matrix.md` and gfx11 fix history (`b7ac66a`) |
| Path never selected | `ArchCaps` / env gate / wrong sibling `.gfx12.hip` | Log arch string + cap bits; open the dispatch arm |
| Speed-gate “regression” after no-op | Stale bench binary (`ensure_build` no-op if exe exists) | Remove the bench example binary so the gate rebuilds; re-run on clean DPM |
| ~50% drop with known-good tree | Firmware shadowing / SMU IF | Check `dmesg`; system fix outside hipfire |

## Related

- Navigation: `docs/INDEX.md`
- Spec-decode after AR is correct: `speculation.md` (this skill)
- Operand / builtin matrix: `wmma-matrix.md`
