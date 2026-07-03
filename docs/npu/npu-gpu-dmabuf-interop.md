# NPU ⇄ GPU dma-buf + fence interop (Strix Halo)

Pointer/summary doc. The **canonical, detailed notes** live next to the bench
that proved the paths:

- Notes: [`benchmarks/npu_gpu_dmabuf/NPU_XRT_DMA_SYNC_NOTES.md`](../../benchmarks/npu_gpu_dmabuf/NPU_XRT_DMA_SYNC_NOTES.md)
- Bench + probe: [`benchmarks/npu_gpu_dmabuf/`](../../benchmarks/npu_gpu_dmabuf/)
  (`npu_gpu_dmabuf_bench.hip`, `syncobj_probe.cpp`, `build.sh`, `run_matrix.sh`,
  `profile_data_path.sh`)
- Captured results (2026-06-02):
  [`benchmarks/results/npu-gpu-dmabuf-2026-06-02/`](../../benchmarks/results/npu-gpu-dmabuf-2026-06-02/),
  [`…-profile-2026-06-02/`](../../benchmarks/results/npu-gpu-dmabuf-profile-2026-06-02/),
  [`…-syncobj-2026-06-02/`](../../benchmarks/results/npu-gpu-dmabuf-syncobj-2026-06-02/)
- Consumer / motivation: [`docs/todo/npu-gpu-heterogeneous-prefill.md`](../todo/npu-gpu-heterogeneous-prefill.md)

## Why this matters

Concurrent NPU + GPU prefill needs two cross-device primitives: a **shared
buffer** both engines can read/write without a serializing host round-trip, and
a **fence** so the GPU stream can wait on NPU completion without a CPU polling
thread. Both are proven on this exact host below.

## Hardware / stack the results were captured on

Ryzen AI MAX+ 395 · Radeon 8060S (gfx1151, 40 CU) · NPU Strix Halo (aie2p, 6×8) ·
XRT 2.25.0 · amdxdna 2.25.0_20260601 · NPU FW 1.1.2.65 · kernel 7.0.0-15.
Nodes: `/dev/dri/renderD128` (amdgpu), `/dev/accel/accel0` (XDNA).

> These results are pinned to that XRT/amdxdna/FW stack. Re-run `syncobj_probe`
> to re-verify fence behavior after any driver/XRT update.

## Proven data path (dma-buf, zero host round-trip)

```
amdgpu GTT BO → export dma-buf fd → XRT imports fd as NPU output BO
→ XDNA kernel writes → HIP imports the (separately duplicated) fd → GPU reads
```

Proven end-to-end with the bundled df-bw payload. Details that bite:

- Output BO is `AMDGPU_GEM_DOMAIN_GTT`, **not** userptr, **not** `VM_ALWAYS_VALID`.
- The dma-buf fd must be **duplicated separately** for the XRT and HIP imports.
- This proves UMA/GTT dma-buf interop — **not** direct NPU DMA into GPU-private
  VRAM. On Strix Halo the traffic is physically DDR/GTT-backed; treat any
  cache-residency win as unmeasured until profiled.

## Proven sync path (no CPU polling)

```
XDNA context drm_syncobj timeline → export syncobj fd → import into amdgpu
→ amdgpu CS waits on the timeline point → GPU compute-ring PM4 NOP completes
```

The meaningful test is `amdgpu_cs_wait_nop_ib` (a real BO-backed PM4 NOP IB with
an `AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT` chunk) — **pass**. `amdgpu_cs_wait_only`
fails by design (amdgpu rejects a wait chunk with no work chunk).

## Key findings

- **Doorbells are the wrong cross-device ordering primitive.** Keep them
  engine-local (they just notify a queue that work exists). Use `dma_fence` /
  `drm_syncobj` / `sync_file` for NPU ↔ GPU ordering.
- The clean syncobj export currently requires reaching through **private
  `shim_xdna::hwctx` state** and scanning `/proc/self/fd` to match the accel
  node — works, but not a stable API. The notes list the exact XRT/XDNA patch
  requests to make it clean.
- The installed `AMDXDNA_EXEC_CMD` path only took plain exec buffers; the newer
  driver tree exposes `SUBMIT_EXEC_BUF` / `SUBMIT_DEPENDENCY` / `SUBMIT_SIGNAL`
  with syncobj dependency plumbing — the source for a backport if GPU→NPU
  dependency waits are needed.
- Message passing: shared GTT mailbox BO (magic/version/producer_seq/
  consumer_seq/payload) + syncobj timeline points in each direction.

## Scope captured so far

In: NPU→GPU DMA into GPU-visible memory, NPU↔GPU control via shared memory,
DRM fence/syncobj ordering. Out (intentionally, for now): bulk GPU→NPU DMA and
full transformer offload.
