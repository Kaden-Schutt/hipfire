# AGENTS.md - hipfire-rdna

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

## GPU Scratch Lifetime

- Per-call (transient) GPU scratch uses RAII: allocate with `alloc_owned` /
  `zeros_owned` / `upload_owned_f32` (returns `OwnedTensor`) and drain at the
  forward/loop boundary with `reclaim_pending()`. Do not add new
  `alloc_tensor`/`zeros` + manual `free_tensor` for per-call scratch — that is the
  error-path leak class `OwnedTensor` exists to remove. `reclaim_pending` self-gates
  on `graph_state_live()`, so it is capture/replay-safe.
- `GpuTensor` stays the non-owning view / kernel-arg type (no `Drop`). Keep using
  it for `sub_offset`/`alias` views, escaping return values (callers own them),
  persistent hoisted/state scratch (freed in a `free_gpu`), and weights. Raw
  `hip.malloc` buffers (e.g. `pos_buf`) are not pooled — free them explicitly.
- See `docs/plans/2026-06-29-owned-tensor-raii-scratch.md` for the design and the
  rejected alternatives.
