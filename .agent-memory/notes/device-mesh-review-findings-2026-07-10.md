---
title: Device-mesh external review findings (2026-07-10)
tags: [device-mesh, review, pp, tp, vram]
created: 2026-07-10
updated: 2026-07-11
---

# External review findings — device-mesh branch (2026-07-10)

External reviewer graded the branch "8/10 foundation, 4/10 integration —
emulated proofs labeled as production." 6 findings, all code-verified against
HEAD (c9d36d0e). Status as of 2026-07-11:

## #1 Dense PP emulation-only (REAL, known follow-up)

pp_serve.rs loads whole weights on out_dev; module doc admitted real-HW per-stage
banding is a follow-up. No capacity benefit on real multi-GPU; cross-device pointer
deref possible on real HW. Status: **STRUCTURALLY DONE FOR PP** — Tasks 1–4 (dense-pp-residency)
implemented and GPU-validated distributed weights (Layout::from_gpus) + distributed KV
(new_gpu_q8_multi) + free_gpu_multi. Byte-identical on emulated-2 (max|Δ|=0).
Real-HW capacity split (per-device weight pages on physically separate devices)
still requires 2-GPU HW to prove — that is a real-HW-only validation by construction.
NOT fixed structurally for real-HW; latent until 2-GPU box available.

## #2 TP/PP unload leaked all VRAM (REAL, NEW — FIXED)

GpuTensor/DeviceBuffer/GpuPool had no freeing Drop; Gpu::drop only re-binds.
unload_model's drop(tp)/drop(pp) freed nothing (EP arm already freed explicitly).
Fix: TpModel::free/PpModel::free (typed frees→drain_pool→drop Gpus) + stream_destroy
for TP streams (TP had a residual 50 MB/cycle leak from undestroyed active_stream).
GPU-validated: TP drift 0 MB / 4 cycles; PP drift -1 MB / 4 cycles. **CLOSED.**
Commits: eafd8663 (initial TpModel::free+PpModel::free) + stream_destroy fix.

## #3 Peer access enabled before allocations (REAL, NEW — FIXED)

hipfire-hardware doc mandates enable_peer_all AFTER allocs (else hipMemcpyPeer
silently writes nothing on real multi-GPU). TP/PP called it right after from_mesh.
Moved after all allocs in tp_serve/pp_serve. Emulation-invisible. **FIXED @ 17fc1c4c.**

## #4 Manifest replication wrong for EP (REAL, latent)

weight_manifest.rs:68 `_ => group_along(Tp)` → singleton on Ep-only mesh (a test
asserts device-0-only). Live EP uses hand forward_ep, not the manifest, so latent.
NOT fixed. Status: **OPEN — LATENT.**

## #5 ArchDispatch "dead end/unrouted" (STALE)

Superseded — ar_generate is the live generic driver, all 9 arches flipped onto it
(Axis A/B + folds). Self-ref sidestepped: dispatch structs built transiently per-call,
not stored in LoadedModel. Reviewer's cited lines were a think-cap loop (stale rev).
Status: **STALE — NOT APPLICABLE.**

## #6 Parity self-check proves only the comparator (FIXED — honesty)

Per-flip dual-run is deleted post-flip; in prod all ar_generate callers pass tape:None.
Relabeled harness comment truthfully + strengthened --self-check-parity to test both
comparator directions. **FIXED @ 17fc1c4c.**
