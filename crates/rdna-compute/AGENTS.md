# AGENTS.md - rdna-compute

This crate owns RDNA compute dispatch, kernel management, feature flags, and
low-level GPU execution glue.

## Dispatch And Arch Policy

- HIP/ROCm-direct is the only backend. Do not route work through Vulkan, wgpu,
  or cross-vendor abstractions.
- Arch-specific paths must be explicit and portable. Prefer capability/arch
  checks over assuming the current development GPU.
- When changing WMMA/MFMA, lane mappings, launch shapes, or arch-specific
  dispatch, use the AMD matrix calculator skill when instruction details matter.
- Run `./tests/coherence-gate-dflash.sh` after dispatch or kernel-routing
  changes that can affect model behavior.
