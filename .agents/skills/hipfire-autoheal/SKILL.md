---
name: hipfire-autoheal
description: Triage and repair hipfire runtime failures such as daemon hangs, stale serve.pid, port 11435 conflicts, ROCm include-path problems, missing precompiled kernels, VRAM OOM, kernel JIT failures, and multi-turn recall regressions. Use after diagnostics identify a likely runtime issue or when the user asks to fix a broken hipfire serve/run flow.
---

# hipfire-autoheal

Use this skill after `hipfire-diag` or when the symptom is clearly a runtime
failure. Start with evidence, apply the smallest targeted fix, then verify the
specific user-facing path that failed.

## Workflow

1. Gather triage:

```bash
.agents/skills/hipfire-autoheal/triage.sh
.agents/skills/hipfire-autoheal/triage.sh --json
```

2. Read `playbook.md` and work the fix catalog in order unless the evidence
   clearly points to a later item.
3. Check `known-issues.md` for hardware/model-specific caveats.
4. Use `bisection.md` only for repeatable hangs or regressions that survive
   the standard fix catalog.

## Guardrails

- Ask for approval before killing user processes, deleting files, changing
  persistent config, installing packages, rebooting, or running privileged
  commands.
- Prefer one-run environment overrides while bisecting, for example
  `HIPFIRE_KV_MODE=q8` or `HIPFIRE_ATTN_FLASH=never`, before changing config.
- If the fix changes code or runtime behavior touching kernels, quant formats,
  dispatch, fusion, rotation, rmsnorm, or spec-decode, run
  `./scripts/coherence-gate-dflash.sh` before claiming the repair is correct.

