# NPU/XRT dma-buf and sync notes

These notes capture the Strix Halo NPU/GPU interop findings so we can
return focus to the GPU MQ2/MQ3/MQ4/MQ6/MQ8 path without losing the XDNA
work.

## Scope

Near-term useful NPU work is limited to:

- NPU -> GPU DMA into GPU-visible memory.
- NPU <-> GPU control messages through shared memory.
- DRM fence/syncobj ordering between XDNA and amdgpu.

Bulk GPU -> NPU DMA and full transformer offload are intentionally out
of scope for now.

## Local Hardware and Stack

Observed on this host:

```text
CPU/APU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
GPU:     Radeon 8060S Graphics, gfx1151, 40 CUs
NPU:     NPU Strix Halo, aie2p, topology 6x8
XRT:     2.25.0
amdxdna: 2.25.0_20260601, 627cee46c6c40fd92147ba64cb1c596538aad750
NPU FW:  1.1.2.65
Kernel:  7.0.0-15-generic
```

Useful device nodes:

```text
/dev/dri/renderD128  amdgpu render node
/dev/accel/accel0    XDNA accel node
```

Useful payloads:

```text
/home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin
/home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf
```

## Proven Data Path

The preferred NPU -> GPU data path is:

```text
amdgpu GTT BO
  -> export as dma-buf fd
  -> XRT imports fd as NPU output BO
  -> XDNA kernel writes output
  -> HIP imports duplicated dma-buf fd
  -> GPU reads/checks output
```

The benchmark under this directory proved that path end to end for the
bundled df-bw payload:

```text
DDR source BO -> XDNA/XRT NPU kernel -> amdgpu GTT dma-buf -> HIP GPU compare
```

Important data-plane details:

- The output BO is allocated by amdgpu in `AMDGPU_GEM_DOMAIN_GTT`.
- It is not userptr.
- It is not `VM_ALWAYS_VALID`.
- The dma-buf fd must be duplicated separately for XRT and HIP imports.
- This proves UMA/GTT dma-buf interoperability. It does not prove direct
  NPU DMA into GPU-private VRAM.
- On Strix Halo this is still physically DDR-backed traffic. Treat cache
  residency benefits as incidental until measured separately.

Useful command:

```bash
benchmarks/npu_gpu_dmabuf/build.sh
benchmarks/npu_gpu_dmabuf/run_matrix.sh
```

## Proven Sync Path: NPU -> GPU

The cleaner synchronization path is also proven:

```text
XDNA context drm_syncobj timeline
  -> export syncobj fd
  -> import into amdgpu
  -> amdgpu CS waits on timeline point
  -> GPU compute-ring PM4 NOP completes
```

Evidence is recorded in:

```text
benchmarks/results/npu-gpu-dmabuf-syncobj-2026-06-02/
```

The meaningful stage is `amdgpu_cs_wait_nop_ib`: a real BO-backed PM4
NOP IB is submitted on the amdgpu compute ring with an
`AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT` chunk that waits on the imported
XDNA context timeline point.

Observed pass summary:

```text
export_xdna_context_syncobj          pass
amdgpu_import_xdna_syncobj           pass
npu_start                           pass
amdgpu_cs_wait_only                  fail, Invalid argument
amdgpu_cs_wait_nop_ib                pass
amdgpu_cpu_timeline_wait             pass
xrt_run_wait_after_probe             pass
```

The wait-only CS failure is expected. amdgpu rejects a submission that
contains a wait chunk but no work chunk. The BO-backed PM4 NOP IB is the
real scheduler test.

Useful command:

```bash
mkdir -p benchmarks/results/npu-gpu-dmabuf-syncobj-$(date -u +%F)
target/npu_gpu_dmabuf/syncobj_probe \
  --json benchmarks/results/npu-gpu-dmabuf-syncobj-$(date -u +%F)/syncobj-probe.json
```

## XRT and XDNA Details

The public XRT surface was enough for the dma-buf data path:

- `xrt::device{0}`
- `device.register_xclbin(xclbin)`
- `xrt::hw_context`
- `xrt::module`
- `xrt::ext::kernel`
- `xrt::bo(device, export_handle_fd)` for importing the amdgpu dma-buf.

The public XRT surface was not enough for the direct scheduler sync
experiment. The probe currently reaches through private XDNA shim state:

```cpp
auto* base = static_cast<xrt_core::hwctx_handle*>(hwctx);
auto* xdna = dynamic_cast<shim_xdna::hwctx*>(base);
uint32_t syncobj = xdna->get_syncobj();
uint32_t slotidx = xdna->get_slotidx();
```

The syncobj handle is a DRM handle owned by the XRT-opened
`/dev/accel/accel0` file. DRM syncobj handles are per DRM file, so the
probe scans `/proc/self/fd`, finds the XRT accel fd whose `st_rdev`
matches `/dev/accel/accel0`, and calls:

```cpp
drmSyncobjHandleToFD(xdna_fd, syncobj, &shared_fd);
drmSyncobjFDToHandle(amdgpu_fd, shared_fd, &amdgpu_handle);
```

That works, but it is not a stable API shape.

## Why Not Public `xrt::fence`

The local XDNA shim can export XRT fences, but that path is not the
same thing as exporting the hardware-context completion syncobj.

The local `submit_signal(fence)` path creates a separate XRT fence and
uses a host pending thread to signal it after the NPU command completes.
That is useful API-level interop, but it is not the direct DRM scheduler
wait path.

For the no-CPU-polling path, prefer:

```text
XDNA context completion syncobj + timeline point
```

not:

```text
host thread waits for XDNA completion, then signals another fence
```

## Message Passing Model

For NPU <-> GPU messages, use shared GTT BOs plus syncobjs:

```text
NPU -> GPU message:
  NPU writes mailbox/data BO
  XDNA context syncobj signals point N
  amdgpu CS waits point N
  GPU reads mailbox/data BO

GPU -> NPU message:
  GPU writes mailbox BO
  amdgpu CS signals syncobj point M
  XDNA submission waits point M
  NPU reads mailbox BO
```

Doorbells should stay engine-local. They notify an engine queue that
new work exists; they are not the right cross-device ordering primitive.
Use `dma_fence`, `drm_syncobj`, or `sync_file` for cross-device ordering.

For the current project direction, the only required bulk data plane is
NPU -> GPU DMA. GPU -> NPU can be just small mailbox/control records if
needed.

## Driver/UAPI Observations

The active-looking accel driver path under:

```text
/home/sadara/xdna-driver/drivers/accel/amdxdna/
```

creates a context syncobj and adds timeline points for submitted jobs:

```text
amdxdna_drm_create_hwctx.syncobj_handle
drm_syncobj_add_point(hwctx->priv->syncobj, chain, job->out_fence, seq)
```

In that path, `AMDXDNA_EXEC_CMD` handling only accepted ordinary exec
buffer submission in the code inspected during this experiment.

The newer-looking tree under:

```text
/home/sadara/xdna-driver/src/driver/amdxdna/
```

has UAPI and implementation code for:

```text
AMDXDNA_CMD_SUBMIT_EXEC_BUF
AMDXDNA_CMD_SUBMIT_DEPENDENCY
AMDXDNA_CMD_SUBMIT_SIGNAL
```

and implements dependency/signal plumbing around syncobj handles and
timeline points. If the installed driver does not expose that behavior,
that tree is the obvious source for a custom-driver backport or upstream
patch request.

## Patch Requests Worth Making Later

Public XRT/XDNA API:

- Expose the XDNA hardware-context completion syncobj as an fd.
- Return the submitted timeline point for each run/command.
- Avoid requiring private `shim_xdna::hwctx` casts or `/proc/self/fd`
  scanning.

XDNA dependency wait:

- Let XDNA command submission wait on imported DRM syncobj timeline
  points.
- Expose that through XRT.
- This is needed for clean GPU -> NPU message dependencies.

Optional convenience:

- Add an XRT helper that imports/exports syncobj fds directly.
- Add diagnostics showing which syncobj point corresponds to which XRT
  run.

## Profiling Notes

Useful preflight:

```bash
xrt-smi examine
xrt-smi validate -r df-bw -p /home/sadara/xdna-driver/build/vtd_extract/strx
```

Useful profiling wrapper:

```bash
benchmarks/npu_gpu_dmabuf/profile_data_path.sh
```

That wrapper records benchmark variants that isolate:

- NPU write path.
- HIP import without compare.
- HIP compare.
- CPU spot-check effects.

With `kernel.perf_event_paranoid=0` and uProf installed, the next useful
measurement is DDR/data-fabric traffic around:

```text
NPU-only write into amdgpu GTT BO
NPU write + HIP import only
NPU write + HIP compare/read
```

The expected physical path on Strix Halo is DDR/GTT, not direct NPU
DMA into GPU-private VRAM.

## Practical Next Experiments If Resumed

1. Keep using the existing `syncobj_probe` to guard NPU -> GPU fence
   behavior after driver/XRT changes.
2. Add a small mailbox BO layout:

   ```text
   magic
   version
   producer_seq
   consumer_seq
   payload_bytes
   payload[]
   ```

3. Have the NPU payload write the mailbox and data BO, then have amdgpu
   wait on the XDNA context syncobj before reading the mailbox.
4. Probe the reverse message direction only if needed:
   amdgpu CS signals a syncobj point, then XDNA submission waits on it.

Until then, the main project should stay on the GPU MQ path.
