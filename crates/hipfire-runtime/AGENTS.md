# AGENTS.md - hipfire-runtime

This crate is part of the inference hot path. Keep runtime changes narrow,
portable, and covered by the hardware gates that match their blast radius.

## Runtime Rules

- Do not introduce Python, PyTorch, or out-of-process scripting into model
  execution.
- Preserve portability across RDNA2, RDNA3, and RDNA4 when changing dispatch,
  fusion, rotation, rmsnorm, KV/cache behavior, or speculative decode.
- Run `./tests/coherence-gate-dflash.sh` after changes touching dispatch,
  fusion, rotation, rmsnorm, quant formats, kernels, or the spec-decode path.
- Runtime examples and microbenches are non-daemon GPU binaries. Document or
  wrap them with `hipfire gpu-lock` when they use GPU resources.
- Keep artifact parsing and sidecar behavior aligned with the root artifact
  naming convention.
