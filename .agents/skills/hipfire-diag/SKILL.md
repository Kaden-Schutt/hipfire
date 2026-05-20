---
name: hipfire-diag
description: Run and interpret hipfire GPU diagnostics for ROCm/HIP bring-up, missing kernels, test_kernels failures, inference smoke failures, and install/runtime environment problems. Use when a user asks to diagnose hipfire, check GPU readiness, run baseline tests, or explain diagnostic output.
---

# hipfire-diag

Use this skill for read-only diagnosis of a hipfire checkout or installed
runtime. It gathers GPU, ROCm, kernel-blob, build, and optional inference-test
signals, then maps failures to concrete next steps.

## Workflow

1. Run from the repo root:

```bash
.agents/skills/hipfire-diag/run-diagnostics.sh [MODEL_PATH]
```

2. Read `interpret.md` for failure mapping and `fix-suggestions.md` for
   platform-specific remediation.
3. Report the failing subsystem first: GPU visibility, ROCm/HIP install,
   missing test binaries, kernel test failures, inference failures, or stale
   build artifacts.

## Guardrails

- Do not install packages, reboot, or edit user shell config unless the user
  explicitly approves that remediation.
- If diagnostics touch kernels, quant formats, dispatch, fusion, rotation,
  rmsnorm, or DFlash/spec-decode behavior, verify with
  `./scripts/coherence-gate-dflash.sh` before claiming correctness.
- Treat one smoke run as diagnosis only, not a performance claim.

