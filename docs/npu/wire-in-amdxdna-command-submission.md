# Wiring W4A8 NPU kernels into hipfire — the amdxdna command-submission path

Scope: how to take the W4A8 gemv/gemm kernels (built + characterized in
`benchmarks/npu_gemm_tuning/`, which emit an `xclbin` + `instr.bin` via mlir-aie)
and actually run them from the hipfire runtime. This is the "wire them in" half.

## Why not the existing path

Two NPU runtime paths exist today, both unusable for new kernels:

1. **`hipfire-arch-qwen35/src/xdna1_ffi.rs`** dlopens `libhipfire_xdna1.so` — a
   binary blob with reverse-engineered symbols (swiglu/rmsnorm/rope/…). It is
   **not on disk here**, and being a blob we cannot add a gemm/gemv symbol to it.
2. **`hipfire-xdna`** talks to the `amdxdna` kernel driver directly via ioctl on
   `/dev/accel/accel0`, but scope today is **read-only telemetry** (`GET_INFO`:
   sensors/clocks/resource-info). Its own header comment names command submission
   as future work.

hipfire deliberately links **no XRT** (no `xrt_coreutil`, no vendor runtime — same
philosophy as HIP-direct on the GPU side). So the in-spirit path is to extend
`hipfire-xdna` with **direct amdxdna command submission via ioctl**, i.e. a minimal
XRT-equivalent in pure Rust. `/dev/accel/accel0` is world-accessible (crw-rw-rw),
so no privilege barrier.

## The amdxdna ioctl ABI (confirmed against `/usr/include/drm/amdxdna_accel.h`)

ioctl numbers (`DRM_IOWR(DRM_COMMAND_BASE=0x40 + n, struct)`), `DRM_TYPE='d'`:

| n | ioctl | struct | role |
|---|---|---|---|
| 0 | CREATE_HWCTX | `amdxdna_drm_create_hwctx` | make a hardware context (tiles, QoS) |
| 1 | DESTROY_HWCTX | `amdxdna_drm_destroy_hwctx` | teardown |
| 2 | CONFIG_HWCTX | `amdxdna_drm_config_hwctx` | load CU config (the xclbin partition) |
| 3 | CREATE_BO | `amdxdna_drm_create_bo` | alloc a buffer object |
| 4 | GET_BO_INFO | `amdxdna_drm_get_bo_info` | get `map_offset` (mmap) + `xdna_addr` |
| 5 | SYNC_BO | `amdxdna_drm_sync_bo` | cache sync to/from device |
| 6 | EXEC_CMD | `amdxdna_drm_exec_cmd` | submit a command, returns `seq` |
| 7 | GET_INFO | (already impl) | telemetry (existing) |

BO types: `SHMEM`, `DEV_HEAP`, `DEV`, `CMD`. Sync directions: `TO_DEVICE=0`,
`FROM_DEVICE=1`. `EXEC_CMD` takes `cmd_handles` (CMD BOs) + `args` (data BO
handles) and returns a `seq`; completion is waited via the hwctx `syncobj_handle`
(DRM syncobj wait) — not a separate amdxdna ioctl.

## The run sequence (what XRT does internally, to replicate)

1. `open("/dev/accel/accel0")`.
2. **Load the xclbin**: parse the AXLF container (`xclbin2` magic) → extract the
   `AIE_PARTITION` / `AIE_METADATA` sections + the PDI. Register the partition
   with the driver; `CONFIG_HWCTX(DRM_AMDXDNA_HWCTX_CONFIG_CU, cu_configs)` points
   the hwctx at the CU (the compiled tile program).
3. `CREATE_HWCTX{ num_tiles, mem_size, qos, max_opc }` → `handle`, `syncobj_handle`.
4. **BOs**: `CREATE_BO(DEV_HEAP)` once for the heap; then per buffer
   `CREATE_BO(SHMEM)` for inputs/outputs and `CREATE_BO(DEV)` where device-resident;
   `GET_BO_INFO` → `mmap(fd, map_offset)` to get a userspace pointer; fill inputs.
5. **Instruction BO**: load `instr.bin` (the mlir-aie NPU instruction stream) into
   a BO — this is the per-run control program the mlir-aie flow emits alongside the
   xclbin.
6. **Command BO** (`CREATE_BO(CMD)`): write the **ERT command packet** — opcode
   `ERT_START_NPU`/`ERT_START_CU`, a pointer to the instr BO, and the arg BO
   handles in the packet payload. **This packet layout is the hard part** — it is
   the firmware ABI (from XRT `ert.h` / `ert_ctrlpkt`), not in the DRM uapi header.
7. `SYNC_BO(TO_DEVICE)` inputs → `EXEC_CMD{ hwctx, type=SUBMIT_EXEC_BUF,
   cmd_handles=[cmdbo], args=[argbos] }` → `seq`.
8. Wait on `syncobj_handle` (DRM `SYNCOBJ_WAIT`), then `SYNC_BO(FROM_DEVICE)`
   outputs, read results.

## Work breakdown

- **W1 — ABI layer** (mechanical, ~1 day): the 7 submission structs + ioctl
  numbers + `size_of` asserts in a new `hipfire-xdna::submit` module, mirroring the
  existing `GET_INFO` style. Low risk; the header pins every field.
- **W2 — BO + mmap + sync** (~1 day): safe wrappers for CREATE_BO/GET_BO_INFO/mmap/
  SYNC_BO; a `DeviceBuffer` type.
- **W3 — xclbin AXLF parse + CONFIG_HWCTX** (~2–3 days): parse the AXLF sections,
  register the partition, config the CU. The AXLF format is documented (XRT
  `xclbin.h`); tedious but bounded.
- **W4 — ERT command packet** (the hard/risky part, ~2–5 days): reproduce the
  firmware command-packet layout for `ERT_START_NPU` with the instr-BO pointer and
  arg handles. Reference: XRT `core/common/api/xrt_kernel.cpp` +
  `runtime_src/core/include/ert.h`; cross-check by dumping the CMD BO that a working
  mlir-aie `test.py` run produces on this box and matching bytes.
- **W5 — dispatch integration** (~1–2 days): a `NpuGemm` op that resolves the
  W4A8 xclbin/instr for a shape (like `npu_xclbin_for`), binds weight/activation/
  output BOs, and slots into the runtime GEMM dispatch behind an admission gate.

## De-risking

The cheapest way to nail W4 (the risky part) is to **capture a known-good command
BO**: run a working mlir-aie kernel through its Python host on this box under
`strace`/a BO dump, snapshot the exact bytes submitted to `EXEC_CMD`, and match the
Rust packet builder against it byte-for-byte before trusting it. That converts the
firmware-ABI unknown into a diff.

## Status

Kernels (benchmarks/npu_gemm_tuning): decode GEMV (R3a) works and is bandwidth-
bound; prefill GEMM needs the weight-broadcast array dataflow (R3c → R4) to be
compute-bound. Wire-in: this plan; W1 (ABI layer) is the next concrete code step.
Both remaining halves (R4 array kernel, W1–W5 submission subsystem) are multi-day
builds — this doc makes the wire-in half fully scoped and actionable.

## W3c findings (hwctx creation frontier)

Implemented `create_hwctx`/`destroy_hwctx` + the BO heap. On-hardware probing of
`/dev/accel/accel0` pinned the exact prerequisites and the current frontier:

1. **DEV_HEAP required first.** A bare CREATE_HWCTX returns `-ENOENT` ("dev heap
   object not exist"): `aie2_hwctx_init` needs `client->dev_heap`. Allocating a
   `AMDXDNA_BO_DEV_HEAP` BO first clears it. (W2's `alloc_buffer` does this.)
2. **Then EINVAL at the resource solver.** With the heap present, CREATE_HWCTX
   reaches `aie2_alloc_resource` → `xrs_allocate_resource` (or, if the device is
   `AIE2_TEMPORAL_ONLY`, `aie2_create_context`, a firmware call) and returns
   `-EINVAL` for every `num_tiles`/QoS tried. So it needs more of what XRT sets up
   before context creation — almost certainly a **registered AIE partition / PDI**
   (the driver must know the partition to reserve columns) and/or the temporal-only
   firmware create path with the right config.

**This is the firmware frontier.** Everything past here — the partition/PDI
registration, CONFIG_HWCTX firmware load, and the W4 ERT command packet + EXEC_CMD
— is a deep reverse-engineering effort against the amdxdna firmware protocol
(reimplementing XRT's aie2 submission core), and EXEC_CMD runs a real program on
the NPU where a malformed packet can hang the device. The right next move is the
documented **capture-based de-risking**: run a working mlir-aie kernel through XRT
under instrumentation (strace + BO dumps), snapshot the exact CREATE_HWCTX args,
the partition-registration path, and the CMD BO bytes, and match the Rust path to
them — not further blind ioctl guessing. This warrants a human call on approach
(from-scratch capture-based build vs. reviving the dormant libhipfire_xdna1.so).

Landed so far (all tested): W1 ABI, W2 BO alloc/mmap/sync (hardware-validated),
W3a AXLF parse, W3b AIE_PARTITION/PDI extract, W3c create/destroy_hwctx + dev_heap
(reaches the resource-solver frontier).
