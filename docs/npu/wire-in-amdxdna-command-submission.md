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

**W1–W4 DONE — first hipfire-driven NPU dispatch runs end-to-end.** The
`run_smoke` example takes an mlir-aie cache dir (`final.xclbin` + `insts.bin`),
parses the PDI, loads it via `CONFIG_HWCTX(CONFIG_CU)`, stages the instruction
stream + A/W/C data BOs, builds the ERT DPU command, submits it (`EXEC_CMD`),
waits on the hwctx timeline, and reads back C. Verified on halo running the R2a
single-core W4A8 GEMM: A=all-1s int8 × W=all-1s int4 yields the exact expected
`16 × (INNER+1) = 1040` per lane. The whole submission ABI is captured byte-exact
from XRT (see recipe below) and locked with a unit test.

**W5 DONE — reusable `NpuKernel` API.** `crates/hipfire-xdna/src/kernel.rs` hoists
the dispatch flow behind `NpuKernel::load(xclbin, insts)` → `alloc_arg` → `dispatch`.
It opens the device, sets up the heap/hwctx and loads the PDI + instruction stream
once, then dispatches repeatedly. Validated on halo: two back-to-back dispatches of
the R2a W4A8 GEMM with A=1 then A=2 give 1040 then 2080 (correct math, hwctx reuse,
no BO-handle leak — `DeviceBuffer::drop` now `GEM_CLOSE`s). `run_smoke` is now a
thin driver over this API.

Kernels (benchmarks/npu_gemm_tuning): decode GEMV (R3a) works and is bandwidth-
bound; prefill GEMM needs the weight-broadcast array dataflow (R3c → R4) to be
compute-bound. Remaining: a shape→cached-xclbin registry (`npu_xclbin_for`) and
the runtime call site that offloads prefill GEMMs to `NpuKernel` (the actual hot-
path integration, to be designed against the GPU/NPU concurrency model).

### How W4 was de-risked

The command-packet layout was the risky part, so it was captured byte-exact from a
working pyxrt run under an `LD_PRELOAD` ioctl tracer rather than guessed:

- **Data BOs** (A/W/C) are `SHMEM`; their regmap address is the **host mmap VA**
  (the NPU reaches host memory at the same VA via PASID). `xdna_addr` is invalid
  for SHMEM.
- **Instruction BO** is `DEV` (heap); regmap address is its `xdna_addr`, and
  `ninstr` is the instruction-stream **byte size**.
- DEV BOs are not mmap-able (`GET_BO_INFO` map_offset = INVALID); fill them by
  writing through the heap mapping at `bo.xdna_addr - heap.xdna_addr`, then
  `SYNC_BO` flushes those heap pages.
- `EXEC_CMD` args = `[A, W, C, instr]` handles (residency); the PDI/cu_bo is *not*
  listed (it's loaded via CONFIG_CU). For `cmd_count==1`, `cmd_handles` is the BO
  handle directly, not a pointer.
- Wait via `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` on the hwctx syncobj at
  `point = EXEC seq`, `flags = WAIT_FOR_SUBMIT`, `timeout = i64::MAX`.
- Output readback needs **no** sync: SHMEM is coherent once the timeline signals.
  `SYNC_DIRECT_FROM_DEVICE` is reserved for debug BOs and EINVALs on data BOs.

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

## Captured submission recipe (from a working pyxrt run)

Traced a working mlir-aie/pyxrt run of my own kernel on `/dev/accel/accel0`
(strace + an LD_PRELOAD ioctl dumper). This is the byte-exact ground truth for the
whole path — the "capture-based de-risking" the plan called for:

Ordered ioctl sequence (fd = accel device):
1. `CREATE_BO(DEV_HEAP, 64 MB, flags=0, vaddr=0)` + `GET_BO_INFO` → `map_offset=0x1_0000_0000`
2. `CREATE_BO(SHMEM, in)` , `CREATE_BO(SHMEM, out)` (+ GET_BO_INFO each)
3. `GET_INFO` (device query)
4. **`CREATE_HWCTX{ num_tiles=32, mem_size=0, max_opc=0x800, qos=all-zeros }`** → handle, syncobj
5. `CREATE_BO(DEV, instr.bin)` → the instruction BO
6. `CONFIG_HWCTX{ param_type=0 (CONFIG_CU), param_val=ptr, size=16 }` → loads the CU/PDI
7. `CREATE_BO(SHMEM)`, `CREATE_BO(CMD, 144 B)` → the ERT command BO
8. `CONFIG_HWCTX{ param_type=1 }` → **-EINVAL, expected** (XRT probes a debug-buf option)
9. `CREATE_BO(DEV)`, `CREATE_BO(CMD)`
10. **`EXEC_CMD{ hwctx, type=0 (SUBMIT_EXEC_BUF), cmd_handles=[cmd_bo], args=[3 BOs], cmd_cnt=1, arg_cnt=3 }`** → seq
11. `SYNCOBJ_TIMELINE_WAIT` (on the hwctx syncobj) → completion
12. `GEM_CLOSE`×, `SYNCOBJ_DESTROY`, `DESTROY_HWCTX`

Command packet (W4): the DPU kernel arg vector is in every xclbin's
EMBEDDED_METADATA — `{opcode:u64, instr:ptr, ninstr:u32, bo0, bo1, …}`,
`dpu_kernel_id=0x901` (same for FLM and my kernels). So the CMD BO is an ERT
packet wrapping those args; `EXEC_CMD` passes `cmd_handles=[cmd_bo]` +
`args=[data BOs]`.

### The one remaining gap: DEV_HEAP must be MAP_FIXED at a device-matched VA

`create_hwctx` in `hipfire-xdna` is now **byte-identical** to pyxrt's call
(confirmed under the ioctl dumper: same num_tiles/mem_size/max_opc/qos, same
CREATE_BO flags, same `map_offset=0x1_0000_0000`). It still `-EINVAL`s, and dmesg
pins it to `aie2_hwctx_init: Map host buffer failed` — the firmware rejects mapping
the dev_heap. Ruled out: args (identical), BO flags (identical), map offset
(identical), page population, MAP_LOCKED (memlock is unlimited). The sole remaining
difference is that XRT mmaps the heap **MAP_FIXED at a specific VA**
(`mmap(0x7b20_5000_0000, 64M, RW, MAP_SHARED|MAP_FIXED|MAP_LOCKED, fd, 0x1_0000_0000)`),
whereas a kernel-chosen VA is rejected by the firmware host-buffer map. Next step:
work out how XRT computes that heap VA (its amdxdna shim / device-VA scheme) and
mmap MAP_FIXED there. Everything downstream (CONFIG_HWCTX, ERT packet, EXEC_CMD) is
now a known quantity from the capture above.

Tooling: the LD_PRELOAD ioctl dumper + strace method is the reusable capture path
for any remaining byte-matching.

## RESOLVED: CREATE_HWCTX works — the gap was the DEV_HEAP mmap VA

The "Map host buffer failed" was the DEV_HEAP mmap placement, not the args. Fix:
mmap the heap **MAP_FIXED at a fixed, moderate, 2 MiB-aligned VA** (hipfire-xdna
uses `0x7000_0000_0000`). Confirmed:
- MAP_FIXED at `0x7b20_5000_0000` (pyxrt's VA) → CREATE_HWCTX **ok**; `0x7000..` also ok.
- The kernel's default placement (~`0x7f6a..`) is rejected by the firmware
  host-buffer map, even though DMA is 64-bit — it's an IOMMU-aperture / placement
  requirement, not a bit-width one.
- The VA is **not** hardcoded in `libxrt_driver_xdna.so`, so XRT derives it the
  same way (force a fixed in-range VA); the driver's heap `dev_addr = AIE2_DEVM_BASE
  (0x400_0000)` is separate from this host VA.

`XdnaDevice::create_hwctx` now succeeds on real hardware (`examples/hwctx_smoke`:
alloc 64 MB dev_heap + 2 SHMEM BOs → CREATE_HWCTX → DESTROY_HWCTX, no dmesg errors).
So W3c is done. Remaining: W3d CONFIG_HWCTX(CONFIG_CU) to load the PDI (arg struct
captured), then W4 build the ERT command BO ({opcode,instr,ninstr,bo*}) + EXEC_CMD
+ syncobj wait — all now known quantities from the captured recipe.
