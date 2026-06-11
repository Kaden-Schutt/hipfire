# AGENTS.md — hipfire agent instructions

For detailed notices, project operating guides, and testing playbooks, see the corresponding sections in `README.md`.

## Project Rules (Core Invariants)
1. **No Python in the inference hot path.** Python is allowed for tooling, benchmarks, comparison baselines.
2. **Commit meaningful experiment states.** Document failures explicitly.
3. **Portability matters.** Every decision should consider: will this work on RDNA2? RDNA3? RDNA4?
4. **No Vulkan / wgpu / cross-vendor compute backend.** hipfire ships a single HIP/ROCm-direct backend.

## Testing & Coherence Gates
- **Coherence-gate-dflash is the canonical correctness gate.** Run `./tests/coherence-gate-dflash.sh` after any change touching kernels, quant formats, dispatch, fusion, rotation, rmsnorm, or the spec-decode path.
- **Prompt structure dictates τ.** ALWAYS use byte-identical prompts via `benchmarks/prompts/*.txt`.
- **Run the no-GPU subset.** `./tests/no-gpu-ci.sh` before handing off workflow-only changes.
- **GPU Lock Protocol:** Coordinate GPU access through `gpu-lock.sh` (`source gpu-lock.sh && gpu_acquire "<branch>"`).

## Skills (`docs/skills/`)
Reusable how-tos live in `docs/skills/` to keep this root file focused. Reach for it by name when the situation matches.

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

---

@AGENTS.local.md
