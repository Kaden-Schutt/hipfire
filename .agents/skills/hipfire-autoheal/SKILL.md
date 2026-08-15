---
name: hipfire-autoheal
description: Triage and repair hipfire runtime failures such as daemon hangs, stale serve.pid, port 11435 conflicts, ROCm include-path problems, missing precompiled kernels, VRAM OOM, kernel JIT failures, and multi-turn recall regressions. Use after diagnostics identify a likely runtime issue or when the user asks to fix a broken hipfire serve/run flow.
---

# hipfire-autoheal

Runtime repair after evidence. Prefer diagnosis and the smallest reversible
fix; do not kill processes, delete files, or reinstall until ownership and
symptom match are confirmed.

For GPU/ROCm inventory and cold bring-up, start with `hipfire-diag` (or
`hipfire diag`). Use this skill when the install looks present but serve/run
is stuck, crashing, OOM, or semantically wrong.

## Workflow

1. **Evidence first**

```bash
.agents/skills/hipfire-autoheal/triage.sh
.agents/skills/hipfire-autoheal/triage.sh --json
hipfire ps
hipfire diag          # optional fuller inventory
tail -n 80 ~/.hipfire/serve.log
```

2. Map `likely_issues` + log lines through `playbook.md` (symptom → diagnosis
   → minimal repair). Do not skip earlier catalog rows without ruling them out.
3. Check `known-issues.md` for **current** hardware/model caveats (dated /
   ref-pinned when available). Rows marked **historical** are fixed,
   superseded, retained measurement debt, or unverified older observations —
   do not treat them as live defaults without fresh evidence.
4. Use `bisection.md` only when the catalog does not localize the failure.
   Daemon-side env bisections require a confirmed local path (see playbook /
   bisection guardrails) — prefixing `hipfire run` alone does not reconfigure
   a resident serve.
5. Verify the **same user path** that failed (serve health, `hipfire run`, or
   the exact chat/completions request). Route any correctness/perf claim per
   [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) — there is no universal
   gate.

## Safe vs destructive actions

| Class | Examples | When |
|---|---|---|
| **Safe automatic** | Read-only triage/diag; `hipfire ps`; `curl` `/health`; one-shot env on a **confirmed local** daemon path (`HIPFIRE_LOCAL=1` / no resident serve — see bisection); `HIPFIRE_HIPCC_EXTRA_FLAGS`; `tail` logs | Always allowed while diagnosing |
| **Safe controlled recovery** | Plain `hipfire stop` **only when** it targets the tracked serve pid that passes CLI ownership validation; wait on cold JIT while tailing `serve.log` | Prefer these before any force-reap path |
| **Destructive / manual — need explicit user approval** | `hipfire restart -d` (stop leg force-reaps name-matched daemons and `fuser -k` on the port — can kill an unvalidated port owner); `hipfire stop --force` / `--all`; `scripts/serve-restart.sh` (kills by pattern + `fuser -k`); bare `pkill`/`kill -9`; deleting `serve.pid`/`daemon.pid` outside CLI validation; wiping kernel cache; `hipfire config set …`; `hipfire update`; package installs; reboot; any privileged command | Only after **port/process-owner inspection** (`hipfire ps`, pidfile, port listener, `/health`) and explicit user approval |

Never blind-kill or delete:

- Confirm the tracked serve via `hipfire ps`, pidfile record, port owner, and/or
  `/health` token before any kill path. Plain `hipfire stop` refuses the
  **tracked pid** on failed ownership validation (`crates/hipfire-cli/src/main.rs`);
  that guarantee does **not** cover force-reap / restart port cleanup
  (`fuser -k`, exact-name daemon reap). Agents must inspect owners before
  requesting approval for restart/force.
- Do not `rm` JIT/kernel caches unless diagnosing a confirmed hash/arch
  mismatch. Cache root is `HIPFIRE_KERNEL_CACHE` (default cwd-relative
  `.hipfire_kernels/<arch>/` — see [`docs/env-vars.md`](../../../docs/env-vars.md)).
- Do not reinstall ROCm/hipfire or rebuild the tree as a first step.

## Canonical recovery paths

| Intent | Path |
|---|---|
| Tracked stop (ownership-validated pid only) | `hipfire stop` |
| Stop + orphan reap + free port | `hipfire stop --force` (**approval** + owner inspection first); optional `--all` also reaps quantize jobs |
| Stop then start with same flags | `hipfire restart -d` (**approval** — force-reap semantics on the stop leg; inspect port/process owners first) |
| Scripted kill/free/optional relaunch | `scripts/serve-restart.sh [port] [--kill-only] [-- serve args…]` — destructive; see script header + [`docs/SERVE.md`](../../../docs/SERVE.md) |
| Read-only / diagnostic GPU/install inventory | `hipfire diag` or `.agents/skills/hipfire-diag/` (GPU/JIT side effects disclosed there) |
| Serve API / lifecycle prose | [`docs/SERVE.md`](../../../docs/SERVE.md), [`docs/CLI.md`](../../../docs/CLI.md) |
| Config / env owners | [`docs/CONFIG.md`](../../../docs/CONFIG.md), [`docs/env-vars.md`](../../../docs/env-vars.md) |
| Claim → validation route | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |

Default bind: `0.0.0.0:11435`. Pid/log: `~/.hipfire/serve.pid`,
`~/.hipfire/serve.log`. Install root: `~/.hipfire/` (`bin/daemon`, `models/`,
`config.toml`).

## References in this skill

| File | Role |
|---|---|
| `triage.sh` | Structured evidence gatherer (`--json` optional) |
| `playbook.md` | Symptom → diagnosis → repair catalog |
| `known-issues.md` | Dated current + historical/unknown caveats |
| `bisection.md` | Localize hangs/regressions after the catalog |

## Out of scope

- Performance claims or route admissions (methodology + `VALIDATION.md` /
  `docs/admissions.yml` only).
- Blind coherence-gate runs: `scripts/coherence-gate-*.sh` are **historical
  reproduction only**, not acceptance ([`docs/VALIDATION.md`](../../../docs/VALIDATION.md)).
- Editing product code unless the user asked for a code fix; this skill is
  runtime ops first.
