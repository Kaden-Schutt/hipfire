# NPU to amdgpu syncobj probe - 2026-06-02

## Conclusion

The direct DRM scheduler path passed on this host:

```text
XDNA context drm_syncobj timeline -> exported fd -> amdgpu import -> amdgpu CS timeline wait -> compute-ring PM4 NOP completes
```

This avoids a CPU polling bridge for the NPU-to-GPU handoff. The probe
still uses a private XRT shim cast to discover the XDNA hardware-context
syncobj because public XRT does not currently expose that context
syncobj/fd.

## Command

```bash
target/npu_gpu_dmabuf/syncobj_probe \
  --json benchmarks/results/npu-gpu-dmabuf-syncobj-2026-06-02/syncobj-probe.json
```

Additional runs were also recorded as:

```text
syncobj-probe-run2.json
syncobj-probe-run3.json
syncobj-probe-rebuild.json
syncobj-probe-final.json
```

## Result Summary

All recorded runs returned `result: pass`.

Common stage result:

```text
xrt_context                         pass
export_xdna_context_syncobj          pass
amdgpu_open                         pass
amdgpu_import_xdna_syncobj           pass
npu_start                           pass
amdgpu_cs_wait_only                  fail, Invalid argument
amdgpu_cs_wait_nop_ib                pass
amdgpu_cpu_timeline_wait             pass
xrt_run_wait_after_probe             pass
```

The wait-only CS failure is expected: amdgpu rejected a submission with
only a wait chunk and no work chunk. The BO-backed PM4 NOP IB is the
meaningful scheduler test.

Observed `amdgpu_cs_wait_nop_ib` durations:

```text
run1: 36666 us
run2: 32505 us
run3: 32913 us
rebuild: 33420 us
final: 32002 us
```

The CPU-side `drmSyncobjTimelineWait` on the imported amdgpu handle
returned immediately after the CS wait stage because the same XDNA
timeline point had already completed.

## Caveats

- The sync point is hardcoded to timeline point 0 for a fresh XRT
  hardware context and the bundled df-bw command.
- This is a DRM/amdgpu command submission proof, not HIP stream external
  semaphore interop.
- It uses `/proc/self/fd` to find the XRT-owned `/dev/accel/accel0` fd,
  because DRM syncobj handles are per DRM file.
- Public `xrt::fence` was intentionally avoided. The local XDNA shim's
  `submit_signal(fence)` path uses an XRT host pending thread to signal a
  separate fence after the NPU command completes.
