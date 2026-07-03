# NPU to GPU dma-buf streaming benchmark

This benchmark is a Strix Halo/XDNA harness for proving the UMA/GTT
streaming path end to end:

```text
DDR source BO -> XDNA/XRT NPU kernel -> amdgpu GTT dma-buf -> HIP GPU compare
```

The preferred pass is:

```text
amdgpu GTT BO exported -> XRT imports and NPU writes -> HIP imports and GPU reads -> compare passes
```

This proves GTT dma-buf interoperability across XDNA and HIP. It does
not prove direct NPU DMA into GPU VRAM.

## Build

```bash
benchmarks/npu_gpu_dmabuf/build.sh
```

The script uses `hipcc`, `libdrm_amdgpu`, `libdrm`, XRT headers/libs,
`xrt_coreutil`, `xrt_core`, `xrt++`, and `vxdna`. The binaries are
written to:

```text
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench
target/npu_gpu_dmabuf/syncobj_probe
```

`syncobj_probe` also includes private XDNA shim headers from
`/home/sadara/xdna-driver/` so it can inspect the XRT-created hardware
context syncobj. It is an experiment, not a stable public XRT API sample.

## Single run

```bash
target/npu_gpu_dmabuf/npu_gpu_dmabuf_bench \
  --mode single \
  --size 4K \
  --iters 1 \
  --xclbin /home/sadara/xdna-driver/build/vtd_extract/strx/validate_df_bandwidth.xclbin \
  --elf /home/sadara/xdna-driver/build/vtd_extract/strx/df_bw.elf \
  --json /tmp/npu-gpu-dmabuf-single-4k.json
```

Useful profiling switches:

- `--skip-hip-compare`: import the dma-buf into HIP but do not run the
  GPU compare kernel. This isolates the NPU write path.
- `--skip-cpu-spot`: skip the tiny CPU spot-check after the NPU run.
  This keeps the measured window free of CPU reads from the output BO.

## Matrix run

```bash
benchmarks/npu_gpu_dmabuf/run_matrix.sh
```

By default this writes JSON and raw logs under:

```text
benchmarks/results/npu-gpu-dmabuf-<utc-timestamp>/
```

The matrix script records:

- `xrt-smi examine --batch`
- the requested `xrt-smi validate -r df-bw -p ...` preflight attempt
- single-mode correctness runs for `4K`, `64K`, `1M`, `16M`, and `64M`
- ping-pong throughput runs for `1M`, `16M`, and `64M`

If the installed `xrt-smi` does not support the `df-bw` validation
syntax, the failure is recorded and the direct benchmark still runs.

The bundled `df_bw.elf` profile advertises 1 GiB `ifm`/`ofm` buffers.
For that payload, the binary allocates at least 1 GiB for the XRT and
amdgpu BOs, even when `--size` is smaller. `--size` remains the prefix
that is initialized, validated, and used for the reported application
bytes.

## Result labels

The JSON `result` field uses these labels:

- `pass`: XRT import, NPU run, XRT sync, HIP import, and HIP compare passed.
- `partial`: NPU wrote the amdgpu BO and CPU spot-check passed, but HIP
  import or HIP compare did not pass.
- `fail`: the first required stage before useful visibility failed.

If XRT cannot import the amdgpu dma-buf, the binary runs one inverse
visibility diagnostic: an XRT-owned shared BO exported to HIP. That is
reported as `visibility_only` and is not treated as equivalent to the
preferred GPU-owned output path.

## Syncobj probe

`syncobj_probe` tests the cleaner synchronization path for NPU -> GPU
handoff without a CPU thread polling mailboxes or doorbells:

```text
XDNA hwctx drm_syncobj timeline -> exported syncobj fd -> amdgpu import -> amdgpu CS timeline wait
```

Run it with:

```bash
mkdir -p benchmarks/results/npu-gpu-dmabuf-syncobj-$(date -u +%F)
target/npu_gpu_dmabuf/syncobj_probe \
  --json benchmarks/results/npu-gpu-dmabuf-syncobj-$(date -u +%F)/syncobj-probe.json
```

What it does:

- Creates the same XRT/XDNA context used by the dma-buf benchmark.
- Reads the underlying `shim_xdna::hwctx` context syncobj with private
  XRT shim headers.
- Finds the XRT-owned `/dev/accel/accel0` fd in `/proc/self/fd` and
  exports that syncobj as a shared DRM syncobj fd.
- Imports the shared syncobj into `/dev/dri/renderD128`.
- Starts the bundled df-bw NPU payload.
- Submits a tiny BO-backed PM4 NOP IB on the amdgpu compute ring with
  `AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT` on the imported XDNA timeline.

The probe intentionally does not use `xrt::fence` for this path. In the
local XDNA shim, `submit_signal(fence)` creates a separate XRT fence and
uses a host pending thread to signal it after the NPU command completes.
That is useful API-level interop, but it is not the direct DRM scheduler
wait path this probe is trying to isolate.

Passing this probe proves amdgpu can wait in its command submission path
on the XDNA context syncobj. It does not prove HIP stream-level external
semaphore support for the same syncobj.
