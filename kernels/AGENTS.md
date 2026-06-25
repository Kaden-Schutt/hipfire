# AGENTS.md - HIP kernels

This subtree owns HIP kernel sources. Kernel changes are high risk because small
layout or launch-shape edits can silently change model behavior.

## Kernel Rules

- Keep kernels HIP/ROCm-direct. Do not introduce Vulkan, wgpu, or cross-vendor
  compute code here.
- Consider RDNA2, RDNA3, and RDNA4 before accepting an optimization. If a path is
  arch-specific, guard and document it explicitly.
- For WMMA/MFMA/lane-layout work, use the AMD matrix calculator skill instead of
  relying on memory.
- After kernel edits, run the narrow relevant kernel/unit check if one exists,
  then `./tests/coherence-gate-dflash.sh` for behavior-facing changes.
