---
name: hipfire-diag
description: Run and interpret hipfire GPU diagnostics for ROCm/HIP bring-up, missing kernels, test_kernels failures, inference smoke failures, and install/runtime environment problems. Use when a user asks to diagnose hipfire, check GPU readiness, run baseline tests, or explain diagnostic output.
---

# hipfire-diag

Non-destructive GPU/install diagnostics for a **source checkout**. Gather
evidence, classify the failing subsystem, suggest bounded next steps. Do not
repair runtime state here — hand established serve/runtime failures to
`hipfire-autoheal`.

Product CLI inventory for an **installed** tree is separate: `hipfire diag`
(and `hipfire diag --json`). See [docs/CLI.md](../../../docs/CLI.md) and
[docs/GETTING_STARTED.md](../../../docs/GETTING_STARTED.md).

## When to use

- Fresh ROCm/HIP or GPU visibility check
- Missing / empty pre-compiled kernel blobs
- `test_kernels` or `test_inference` smoke failures
- Explaining JSON from this skill’s runner
- Before filing an issue about “hipfire won’t run on this machine”

**Not this skill:** daemon hangs, stale `serve.pid`, port 11435 fights,
multi-turn recall, mid-gen HipError after a healthy inventory →
`.agents/skills/hipfire-autoheal/`.

## Workflow

1. From the **repo root**, disclose side effects (below), then:

```bash
.agents/skills/hipfire-diag/run-diagnostics.sh
# optional inference smoke (file must exist):
.agents/skills/hipfire-diag/run-diagnostics.sh /path/to/model.hfq
```

2. Parse the JSON with [interpret.md](interpret.md). Treat `passed: 0` /
   `failed: 0`, missing `failures[]`, timeouts, and discarded nonzero status
   as **unknown** — not pass (see contract notes).
3. Map concrete symptoms to safe commands in [fix-suggestions.md](fix-suggestions.md).
4. Report in order: **what works → what fails → unknowns → one recommended
   action per confirmed failure**.
5. If evidence is a runtime/serve failure (not inventory), stop diagnosing and
   open `hipfire-autoheal` (`triage.sh` + playbook). Do not invent fixes here.

## What the runner measures

Script: [run-diagnostics.sh](run-diagnostics.sh) (cwd = repo root). Emits one
JSON object (`tool: hipfire-diag`, `version: 0.0.1`):

| Key | Source | Notes |
|---|---|---|
| `gpu` | `/dev/kfd`, `rocm-smi`, optional `test_kernels` line `GPU:` | `kfd: false` → no AMD KFD node |
| `kernels` | `kernels/compiled/{gfx1010,gfx1030,gfx1100,gfx1200,gfx1201}/` | Per-arch `blobs` (`.hsaco`) and `hashes` (`.hash`) counts |
| `kernel_tests` | `target/release/examples/test_kernels` (60s timeout); fallback `test_kernelsQA` | See **result contract** below — do not treat bare counts as pass/fail without status |
| `inference_tests` | `test_inference <model>` when argv model path exists (120s) | Same contract; may be `skipped` or path error |
| `build` | Presence of `infer` and `infer_hfq` release examples | Boolean flags only |

Missing binaries produce an `error` string with the exact `cargo build`
line the script embeds — prefer that over guessing features.

### Result contract (fail closed)

The current runner discards some timeouts and nonzero exits with `|| true`.
Therefore:

- `passed: 0` and `failed: 0` together → **unknown** (crash, timeout, or
  empty parse), **not** a clean pass.
- Missing `failures[]` (including the `test_kernelsQA` fallback path) →
  failure detail is **unknown**; do not invent a failures list.
- Inference fields with the same masked-status pattern → **unknown** unless
  an explicit success/error string is present.
- Do not promote any of the above to “healthy inventory.”

Prefer reporting: works / fails / **unknown**, then one action per confirmed
failure.

### Side effects (disclose before invoke)

This skill is **non-destructive diagnostic execution**, not a pure filesystem
inventory:

- When `test_kernels` / `test_inference` run, they call `Gpu::init`, may create
  `.hipfire_kernels/<arch>/`, and may JIT/write cache objects.
- They launch real GPU work for the probe/suite duration.
- They do **not** stop daemons, delete unrelated files, reinstall packages, or
  edit shell rc — those remain approval-gated and owned by autoheal/user.

If the user needs inventory-only with zero GPU/JIT, stick to path/presence
checks (`ls` kernels dirs, `rocm-smi`, `/dev/kfd`) and skip the binary suite
until they opt in after this disclosure.

## Guardrails

- **Non-destructive by default (no repair).** Do not install packages, reboot,
  edit shell rc, `pkill`, delete kernel caches, or run `hipfire update` unless
  the user explicitly approves that step. GPU/JIT execution above is allowed
  diagnostic work after disclosure — still not automatic repair.
- **No automatic repair.** This skill diagnoses and proposes; `hipfire-autoheal`
  owns repair playbooks after approval.
- **No universal gate.** Coherence-gate scripts are retired as acceptance
  ([docs/VALIDATION.md](../../../docs/VALIDATION.md)). Do not require
  `coherence-gate-*.sh` to “finish” diagnosis.
- **No perf claims from one smoke.** `inference_tests.tok_s` is a single timed
  sample. Floors, baselines, and promotion live in
  [docs/methodology/perf-benchmarking.md](../../../docs/methodology/perf-benchmarking.md)
  and measured owners under [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md) /
  [docs/perf-checkpoints/](../../../docs/perf-checkpoints/) — not here.
- **Fail closed on unknown symptoms and unknown counts.** If JSON is healthy
  but the user still fails, or counts are 0/0 / missing failures, hand off to
  autoheal triage or file an issue with the full JSON — do not invent a root
  cause or mark unknown as pass.
- **Mutable inventories** (supported arches, model tags, VRAM tables, env
  vars) stay in their docs owners (`docs/MODELS.md`, `docs/env-vars.md`,
  `docs/architecture-ids.md`, etc.). Link; do not fork lists into this skill.

## Related

| Path | Role |
|---|---|
| [run-diagnostics.sh](run-diagnostics.sh) | Executable diagnostic runner (GPU/JIT side effects) |
| [interpret.md](interpret.md) | Field-by-field reading order |
| [fix-suggestions.md](fix-suggestions.md) | Bounded, opt-in remediation snippets |
| [../hipfire-autoheal/](../hipfire-autoheal/) | Runtime/serve repair after diagnosis |
| [docs/VALIDATION.md](../../../docs/VALIDATION.md) | Claim → validation route selector |
| [docs/CLI.md](../../../docs/CLI.md) | `hipfire diag` product surface |
