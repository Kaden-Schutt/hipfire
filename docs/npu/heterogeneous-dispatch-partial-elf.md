# Heterogeneous NPU: zero-copy dma-buf, async dispatch, and the partial-ELF gap (gfx1103 / XDNA1)

Status: research complete; **zero-copy proven, dispatch-on-new-firmware is a scoped TODO**.
Scope: Phoenix-class APU (gfx1103 GPU + XDNA1 `accel0`). Companion to
`docs/npu/npu-gpu-dmabuf-interop.md` (the Strix Halo/XRT interop) and
`docs/todo/npu-gpu-heterogeneous-prefill.md` (the motivation). Detailed R-by-R log lives in
the benchmark memory; this is the pick-up doc.

## Why this exists

Concurrent GPU + NPU work (parallel prefill offload, or a DSpark/MTP draft running on the NPU
while the GPU verifies) needs three primitives: **async NPU dispatch**, a **zero-copy shared
buffer** both engines read/write, and a **cross-engine fence**. On this box the GPU∥NPU overlap
is real (measured ~48% efficiency, each engine keeps ~70–78% under concurrency), so the payoff
is there. The blocker is a **driver/firmware pairing** on this specific box, detailed below.

## The two-stack split (the core constraint)

Two amdxdna drivers exist, and each pairs with a different firmware **and a different command
ABI**. On this box you cannot currently have both dma-buf import *and* working dispatch at once:

| | dma-buf import | dispatch | firmware | command ABI |
|---|---|---|---|---|
| **stock** (in-tree, kernel 6.17, srcversion `2DBDA7…`) | ❌ (no `va_tbl`/`BO_SHARE`) | ✅ | `npu.sbin` **1.5.2.380** | `ERT_START_CU`, raw `insts.bin` |
| **out-of-tree** (`~/xdna-driver` `drivers/accel/amdxdna`, 0.15.0) | ✅ | ⚠️ needs partial-ELF | `npu.dev.sbin` **1.5.5.391** | `ERT_START_NPU`, partial-ELF |

Proven on hardware: the stock stack dispatches correctly (hipfire gets bit-exact results); the
out-of-tree stack imports dma-bufs zero-copy and boots cleanly with the matched firmware, but
its dispatch needs the partial-ELF instruction format (below).

## Firmware pairing (this bit is easy to get wrong)

The out-of-tree driver loads `npu.dev.sbin` **first**, falling back to the stock `npu.sbin`. Its
required version is declared in `~/xdna-driver/tools/info.json` — for npu1 (device `1502`):

- version **`1.5.5.391`**, from the **gitlab.com** mirror (not freedesktop):
  `https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/1502_00/npu.sbin.1.5.5.391`
- install as `/lib/firmware/amdnpu/1502_00/npu.dev.sbin`

Gotchas learned the hard way:
- The freedesktop `drm/firmware` mirror only has `1.4.2.313/.323` for `1502`, which **do not boot**
  this driver ("Invalid mbox magic / firmware is not alive / hardware init -22"). Use gitlab.com +
  the `info.json` version.
- `1.5.5.391` is *newer* than the stock `1.5.2.380` — a different release track, not a downgrade.
- The stock driver ignores `npu.dev.sbin`, so installing it is safe for the stock stack.

Build + load the out-of-tree module (matches the running kernel's vermagic):
```
cd ~/xdna-driver/drivers/accel/amdxdna
OUT=config_kernel.h KERNEL_VER=$(uname -r) bash ../tools/configure_kernel.sh
make modules            # -> amdxdna.ko
sudo rmmod amdxdna && sudo insmod ./amdxdna.ko   # loads npu.dev.sbin (must be idle: no serve on accel0)
# revert: sudo rmmod amdxdna && sudo modprobe amdxdna
```
xclbin packaging needs XRT `xclbinutil` on `PATH` + a user-space `libboost_program_options.1.83`
on `LD_LIBRARY_PATH` (cache at `~/.cache/hipfire-npu-deps/lib`); `r5_build.sh` auto-adds both.

## Primitive 1 — zero-copy dma-buf import (PROVEN, raw path, no XRT)

`XdnaDevice::import_dmabuf(fd, size, map)` (hipfire-xdna): `CREATE_BO(AMDXDNA_BO_SHARE)` with
`vaddr` → `struct amdxdna_drm_va_tbl { i32 dmabuf_fd; u32 num_entries=0 }`. The driver
`dma_buf_get`s the fd and `amdxdna_gem_prime_import`s it. Flow (see `examples/dmabuf_probe.rs`):

```
amdgpu renderD128: GEM_CREATE(GTT, CPU_ACCESS) -> GEM_MMAP -> PRIME_HANDLE_TO_FD (DRM_RDWR|CLOEXEC)
  -> amdxdna import_dmabuf(fd) -> both engines address the same physical GTT pages
```
Proven bidirectional on the out-of-tree stack: NPU reads the GPU's marker and the GPU reads the
NPU's (`share r/w = yes/yes`). The raw amdxdna path also owns the hwctx **syncobj** directly
(`create_hwctx` returns `syncobj_handle`), so a device-side fence needs only
`drmSyncobjHandleToFD` → amdgpu import — no XRT private-`shim_xdna::hwctx` hack (unlike the halo
notes). `NpuInFlight.seq` from `submit()` is exactly the timeline point to export.

## Primitive 2 — async dispatch (COMMITTED, stock-safe)

`NpuKernel::submit`/`poll`/`wait` (+ `syncobj_poll`, non-blocking `ETIME`-aware). `NpuInFlight`
carries `seq` (per-hwctx order) and a caller `tag` (microbatch/layer/expert id) so a scheduler
can poll-by-handle, correlate-by-tag, order-by-seq. Each `submit` owns its command BO (the
single-slot cache can't back multiple in-flight). Validated on the stock stack.

## The dispatch ABI: ERT_START_CU vs ERT_START_NPU

`dpu_cmd_packet(instr_addr, instr_size, arg_addrs, ert_npu)` builds one of two 144-byte packets.
The `ert_npu` flag is set from `HIPFIRE_XDNA_ERT_NPU` (default off = stock). **This code is
written and held uncommitted** (per decision) pending working dispatch on the new stack.

Stock — **ERT_START_CU** (opcode in the packet):
```
@0x00 u32 header 0x30010001   @0x04 cu_mask=1   @0x08 u64 opcode=3
@0x10 u64 instr_addr   @0x18 u32 instr_size   @0x1c args (u64 each)
```
Out-of-tree — **ERT_START_NPU** (opcode in the header, `struct amdxdna_cmd_start_npu` payload):
```
@0x00 u32 header = STATE(NEW=1) | COUNT(5+2n)<<12 | OPCODE(20)<<23   // GENMASK 3:0 / 22:12 / 27:23
@0x04 u32 cu_mask[0]=1
@0x08 u64 buffer        // instruction-stream device address
@0x10 u32 buffer_size   @0x14 u32 prop_count=0   @0x18 prop_args (arg addrs, u64 each)
```
Verified byte-exact vs XRT on the out-of-tree driver (`amdxdna_cmd_get_op`/`get_cu_idx`/
`get_payload` in `amdxdna_ctx.c`; `aie2_cmdlist_fill_npu_dpu` in `aie2_message.c`). The driver's
`npu_exec_message_ops` (fw-version-gated, `aie2_pci.c`) routes `ERT_START_NPU` →
`fill_npu_dpu`, which **hardcodes `EXEC_NPU_TYPE_PARTIAL_ELF`**.

Coherence: the out-of-tree driver maps SHMEM **cached** (`create_shmem_object` → `map_wc=false`),
so outputs need a `SYNC_BO` clflush after the fence. Use `SYNC_DIRECT_TO_DEVICE` — its `drm_clflush`
invalidates on x86, and `FROM_DEVICE` additionally hits an `amdxdna_hwctx_sync_debug_bo` that
rejects normal BOs. Gated on `ert_npu`; no-op-cheap on the coherent stock driver.

## The gap: partial-ELF instruction format

The new firmware's `EXEC_NPU_TYPE_PARTIAL_ELF` does **not** accept the raw `insts.bin` nor a raw
ELF blob (both dispatch with no `-22` but produce no output — the firmware silently no-ops). XRT
applies the ELF relocations **in userspace** (`xrt::elf` → `xrt::module` → `ext::kernel`) and
sends **patched `.ctrltext`**.

The toolchain already emits the ELF — **no upgrade needed** (installed mlir-aie `1.3.3.dev13`):
```
aiecc … --aie-generate-elf --elf-name insts.elf
```
`insts.elf` is AIE `ELF32` (machine `WE32100`) with:
- `.ctrltext` — the control/instruction code, **same byte size as the raw `insts.bin`**.
- `.rela.dyn` — one address relocation (type 5) **per kernel arg**; `.dynsym` names them by arg
  index with buffer sizes. For a 3-arg kernel: relocs at offsets `0x20`, `0x98`, `0x110`, symbols
  1/2/3 → arg 0/1/2 (A/W/C).
- `.note.xrt.UID`.

### Per-dispatch reloc plan (the remaining work)

To dispatch on the new firmware, hipfire must replicate XRT's userspace reloc:

1. Build kernels with `--aie-generate-elf` (add to `r5_build.sh`; keep `--aie-generate-npu-insts`
   for the stock path).
2. `NpuKernel::load` (when `ert_npu`): parse `insts.elf` once → keep the `.ctrltext` bytes and a
   `[(offset, arg_index)]` reloc table (from `.rela.dyn` + `.dynsym`).
3. Per dispatch: copy `.ctrltext`, write `args[arg_index].xdna/host_addr` (u64) at each reloc
   offset, write the patched stream into the instruction DEV BO, `SYNC_BO`, then `exec`. Use
   `inst_size = .ctrltext size`; likely empty `prop_args` (the args are now baked in — verify).

Open unknowns (need hardware iteration, each = a driver swap): patched-`.ctrltext` vs a ctrlpkt
variant; whether `prop_args` must still be passed; host-VA vs device-VA for the patched addresses.
A Rust ELF reader (`object`/`goblin`) covers the parse; the reloc itself is 3 × u64 writes.

## Current state of the code

- Committed / stock-safe: `import_dmabuf`, `syncobj_poll`, `submit/poll/wait` + `NpuInFlight`,
  `dmabuf_probe`, `async_smoke`.
- Held uncommitted (per decision): env-gated `ERT_START_NPU` `dpu_cmd_packet`, `NpuKernel.ert_npu`,
  gated post-exec clflush + `sync_from_device`, ELF-selection in `npu_gemm_bench`.
- On disk for pick-up: `npu.dev.sbin`=1.5.5.391 installed, out-of-tree `amdxdna.ko` built.

## Bottom line

The heterogeneous thesis is proven at every primitive — GPU∥NPU overlap, async dispatch, and
zero-copy import all work. The one remaining gap to a productizable path on this box is the
**per-dispatch partial-ELF relocation** (a bounded, XRT-equivalent implementation), plus its
hardware iteration. Best taken on deliberately with the box idle, not under a live server.
