# AGENTS.md — hipfire agent instructions

For detailed notices, project operating guides, and testing playbooks, see the corresponding sections in `README.md`.

## Project Rules (Core Invariants)
1. **No Python in the inference hot path.** Python is allowed for tooling, benchmarks, comparison baselines.
2. **Commit meaningful experiment states.** Document failures explicitly.
3. **Portability matters.** Every decision should consider: will this work on RDNA2? RDNA3? RDNA4?
4. **No Vulkan / wgpu / cross-vendor compute backend.** hipfire ships a single HIP/ROCm-direct backend.

## Branch Policy
- **Use `chaingun` as the reference branch for all further work.** New work should either happen directly on `chaingun` or be explicitly based on and compared against `chaingun`; do not treat `master` as the active development baseline unless the user says so.
- **Keep git moving as you work.** Pull/rebase from the `chaingun` reference before starting meaningful changes, commit coherent work states with descriptive messages, and push the active branch regularly. If the worktree contains unrelated user changes, preserve them and only stage/commit the files that belong to the current task.

## Testing & Coherence Gates
- **Coherence-gate-dflash is the canonical correctness gate.** Run `./tests/coherence-gate-dflash.sh` after any change touching kernels, quant formats, dispatch, fusion, rotation, rmsnorm, or the spec-decode path.
- **Model/runtime evidence belongs in `hipfire-eval`.** When adding or repairing speed, coherence, DFlash/DDTree/Path C, PFlash, agentic/tool-call, long-context, quality, or server-runtime admission tests, add or update `hipfire-eval` batteries/suites first. Keep shell gates only as enforcement wrappers where they still provide baseline comparison or hook integration.
- **Prompt structure dictates τ.** ALWAYS use byte-identical prompts via `benchmarks/prompts/*.txt`.
- **Run the no-GPU subset.** `./tests/no-gpu-ci.sh` before handing off workflow-only changes.
- **Resource Lock Protocol:** `hipfire-daemon` acquires `/tmp/hipfire-resource-locks/*.lock` leases before HIP init. Use `HIPFIRE_RESOURCE_LOCK_WAIT_MS` to wait for busy GPU/NPU/CPU leases. Non-daemon GPU binaries (cargo `--example` benches, `hipfire eval`, `hipfire-quantize`) do not self-lock — coordinate them with the native CLI mutex `hipfire gpu-lock {acquire,release,status}` (the legacy `scripts/gpu-lock.sh` shell adapter has been removed).

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
