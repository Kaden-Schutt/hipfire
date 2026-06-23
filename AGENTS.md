# AGENTS.md — hipfire agent instructions

For detailed notices, project operating guides, and testing playbooks, see the corresponding sections in `README.md`.

## Project Rules (Core Invariants)
1. **No Python in the inference hot path.** Python is allowed for tooling, benchmarks, comparison baselines.
2. **Commit meaningful experiment states.** Document failures explicitly.
3. **Portability matters.** Every decision should consider: will this work on RDNA2? RDNA3? RDNA4?
4. **No Vulkan / wgpu / cross-vendor compute backend.** hipfire ships a single HIP/ROCm-direct backend.
5. **One lock primitive: `hipfire-lock` `flock(2)`. Never roll your own.** Every cross-process mutex (GPU/NPU/CPU resource lease, daemon singleton, any future device or file lock) MUST go through the **one** canonical surface:
   - **Rust callers** use the `hipfire-lock` crate directly — `FlockGuard` plus the shared path contracts `resource_lock_root()` / `resource_lock_path(resource)` / `gpu_resource_lock_path()` (= `resource_lock_path("hip-gpu-0")`, the canonical single-GPU lock).
   - **Shell / non-Rust / external callers** use the **`hipfire lock {acquire,release,status}` CLI** (alias `gpu-lock`). It is a thin wrapper over the same `FlockGuard` on `gpu_resource_lock_path()` — the SAME inode the daemon's GPU resource lease uses, so a CLI holder and the daemon coordinate. `acquire` spawns a detached holder that keeps the flock until the watched pid dies. Do NOT reach for `flock(1)`, a sentinel file, or a bespoke lockfile from a script.

   Do NOT introduce a second mechanism anywhere: no `std::fs::create_dir`-as-mutex, no `owner.json`/pidfile + pid-liveness reclamation, no env-var sentinel, no ad-hoc lockfile, no shell-only mutex, and no *second lock file* for the same resource (the CLI lock and the daemon GPU lease MUST be the same inode). Rationale: `flock` is inode-keyed (one mutex shared across processes/languages) and kernel-released on holder exit (no stale-lock cleanup), and a single primitive+CLI prevents divergent/duplicate locks that silently fail to coordinate — the cautionary cases: the `create_dir` resource-lease that drifted out of sync (fixed in `f24313c09`), the separate `hipfire-gpu.lock` file that did not coordinate with the daemon's `hip-gpu-0` lease (unified when `gpu-lock` was renamed to `lock`), and the removed `scripts/gpu-lock.sh` shell adapter. Need a new lock? Add a path helper to `hipfire-lock` and expose it via the CLI if scripts need it — don't fork the mechanism.

## Branch Policy
- **Use `chaingun` as the reference branch for all further work.** New work should either happen directly on `chaingun` or be explicitly based on and compared against `chaingun`; do not treat `master` as the active development baseline unless the user says so.
- **Keep git moving as you work.** Pull/rebase from the `chaingun` reference before starting meaningful changes, commit coherent work states with descriptive messages, and push the active branch regularly. If the worktree contains unrelated user changes, preserve them and only stage/commit the files that belong to the current task.

## Testing & Coherence Gates
- **Coherence-gate-dflash is the canonical correctness gate.** Run `./tests/coherence-gate-dflash.sh` after any change touching kernels, quant formats, dispatch, fusion, rotation, rmsnorm, or the spec-decode path.
- **Model/runtime evidence belongs in `hipfire-eval`.** When adding or repairing speed, coherence, DFlash/DDTree/Path C, PFlash, agentic/tool-call, long-context, quality, or server-runtime admission tests, add or update `hipfire-eval` batteries/suites first. Keep shell gates only as enforcement wrappers where they still provide baseline comparison or hook integration.
- **Prompt structure dictates τ.** ALWAYS use byte-identical prompts via `benchmarks/prompts/*.txt`.
- **Run the no-GPU subset.** `./tests/no-gpu-ci.sh` before handing off workflow-only changes.
- **Resource Lock Protocol** (mechanism per Core Invariant 5 — `hipfire-lock` flock, never a second lock): `hipfire-daemon` acquires `flock(2)` leases (one `<resource>.lock` file per GPU/NPU/CPU resource, via `resource_lock_path()`) under `~/.hipfire/locks/` before HIP init, releasing on exit; the daemon singleton uses the same primitive on `~/.hipfire/daemon.pid`. Override the resource-lock root with `HIPFIRE_RESOURCE_LOCK_DIR`; use `HIPFIRE_RESOURCE_LOCK_WAIT_MS` to wait for busy leases. Non-daemon GPU binaries (cargo `--example` benches, `hipfire eval`, `hipfire-quantize`) do not self-lock — coordinate them with the native CLI mutex `hipfire lock {acquire,release,status}` (alias `gpu-lock`), which flocks `gpu_resource_lock_path()` — the same `hip-gpu-0.lock` inode the daemon's GPU lease holds, so they actually coordinate. The legacy `scripts/gpu-lock.sh` shell adapter has been removed.

## Skills (`.agents/skills/`)
Reusable how-tos live in `.agents/skills/<name>/`. Each has a `SKILL.md` (and optionally `skill.json`) with full context. Load one by reading its files when the situation matches.

| Skill | When to use |
|-------|-------------|
| `astrea` | Quant calibration, imatrix-driven experiments, KLD/PPL quality eval |
| `hipfire-amd-matrix-calculator` | Vendored AMD Matrix Instruction Calculator queries |
| `hipfire-arch-port` | Porting compute kernels to a new RDNA/CDNA arch |
| `hipfire-autoheal` | Triage daemon hangs, kernel JIT failures, port conflicts |
| `hipfire-diag` | GPU diagnostics — interpret results and suggest fixes |
| `hipfire-eval-harness` | Running or interpreting the unified eval harness |
| `hipfire-kernel-atlas` | Phase-aware Kernel Atlas rows and ISA Fit View |
| `hipfire-kernel-tuning` | Optimize HIP kernels — tuning levers, multi-row, WMMA, wave-size |
| `hipfire-tester` | Bring-up, smoke tests, DFlash opt-in, MQ format checks |
| `npu-kernel-build` | Compile MLIR-AIE kernels for the XDNA NPU via IRON + aiecc |
| `rebase-onto-modular` | Port feature branches from pre-0.1.20 master |
| `run-model` | Load a model and generate tokens via the daemon JSON-lines protocol |

## Hipfire artifact naming convention
Canonical shape:
`<family>-<version>-<size>[-<variant>][-<role>]-<format>[+<features>].<ext>`

Rules:
- Use `.hfq` for hipfire container artifacts, including MQ-family models.
- Use dotted model versions such as `qwen3.5`.
- Put calibration / transform modifiers before the quant token: `awq-mq4`, `lloyd-mq3`.
- Use `+feature` only when bundled: `mq4+mtp`, `mq4+dflash`.
- Use role sidecars when loaded independently: `.mtp.hfq`, `.dflash.hfq`, `.triattn.hfq`.
- Use `.triattn.hfq` for TriAttention sidecars even though they are not weight tensors; do not introduce `.triattn.bin` for new files.
- Do not use dotted quant artifact suffixes. The quant token belongs before
  the `.hfq` extension with a hyphen separator; Lloyd MQ2 uses
  `-lloyd-mq2.hfq`.
- When a script, gate, registry, or doc is found using an older artifact spelling,
  update it to the canonical naming convention as part of the fix.
- Backwards compatibility is a separate, explicit decision: add legacy-name
  fallback only when the task or migration risk calls for it, and document why.

---

@./AGENTS.local.md
