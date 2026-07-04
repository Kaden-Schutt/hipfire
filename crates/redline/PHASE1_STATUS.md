# Redline Phase 1 — Status & Gap Analysis

**Goal:** eliminate HIP's ~10 µs per-launch overhead on gfx1100 (7900 XTX) so
the ~36 non-GEMV launches per forward pass stop costing 360 µs of pure
framework tax.

## System inventory

| Component | Version / State |
|-----------|-----------------|
| GPU | AMD Radeon RX 7900 XTX (Navi 31, gfx1100, 96 CUs, 25.7 GB VRAM) |
| Kernel | Linux 6.17.0, `amdgpu` loaded, DRM version 3.64 |
| ROCm | 6.3.4 (`/opt/rocm-6.3.4`) |
| `libhsa-runtime64.so` | 1.14.60304 — present and working |
| `libhsakmt.a` | present (static lib only; we'd statically link or use HSA instead) |
| `libamdhip64.so` | 6.3.60304 (what hip-bridge dlopens today) |
| `/dev/kfd` | present; user has render/video group access |
| `libdrm_amdgpu` | working (Redline already dlopens it) |

## Current Redline capabilities

Path in use today: **PM4 over `libdrm_amdgpu` `cs_submit` ioctl** (no HSA, no
KFD, no doorbell). All submission goes through a kernel ioctl per dispatch.

### What works (gfx1010 + gfx1100)

- `Device::open` — DRM render-node open, `amdgpu_device_initialize`, GPU info
- `alloc_vram` / `upload` / `download` — BO alloc, VA mapping, CPU map copy
- `HsacoModule::from_bytes` — parses `.hsaco` (incl. `__CLANG_OFFLOAD_BUNDLE__`),
  finds `.kd` symbols, extracts `pgm_rsrc1/2`, `kernarg_size`,
  `group_segment_size`, code entry VA
- `Kernel::from_meta` — decodes `kernel_code_properties` for user-SGPR layout
- `CommandBuffer::dispatch` — PM4 builder for a single dispatch (COMPUTE_PGM_LO/HI,
  RSRC1/2/3, USER_DATA, DISPATCH_DIRECT)
- `CommandBuffer::dispatch_with_lds` — dispatch with dynamic LDS size override
- `DispatchQueue::dispatch` — single-dispatch path with cs_submit + fence wait
- `FastDispatch::dispatch` — persistent IB mapping, persistent kernarg mapping,
  persistent BO list; eliminates per-dispatch Vec allocs and bo_list_create
- `KernargBuilder` — typed kernarg packing

### What's broken on gfx1100 (worked on gfx1010)

- `CommandBuffer::barrier` (RELEASE_MEM + WAIT_REG_MEM) — **fails with `-62`
  (ETIME)** on gfx1100. The packet encoding was verified in C against
  `gfx_v10_0.c` for RDNA1. RDNA3's MEC accepts a different event/GCR mask
  layout. Every chain-dispatch test in `bench_dispatch.rs` returns -62.
- `src/kfd.rs::AqlQueue::new` — **`KFD_IOC_CREATE_QUEUE` returns EINVAL** on
  gfx1100. On gfx1010 the failure was blamed on `local_mem_size=0`; on gfx1100
  that excuse is gone (25.7 GB VRAM reported). The remaining suspect is the
  **userptr-backed ring buffer**: discrete GPUs almost certainly need a
  VRAM-backed ring (MQD is placed in the BO the ring points at), plus a
  different `ctx_save_restore_size`/`ctl_stack_size` for RDNA3.

### Measured per-dispatch overhead (vector_add, 256 elements, 5000 iter)

Baseline on 7900 XTX (`redline::examples::bench_dispatch`):

```
DispatchQueue (cs_submit + fence wait per call):
  median 63.5 µs    mean 65.2 µs    p99 82.8 µs    min 59.6 µs

FastDispatch  (persistent map + BO list, still cs_submit per call):
  median 33.3 µs    mean 33.6 µs    p99 38.6 µs    min 29.0 µs

Chained IB (10/50/100/200 dispatches in one submit):
  ALL FAILED with -62 ETIME (barrier packet broken on RDNA3)
```

HIP (from bandwidth-ceiling Phase 3.5 profiling):

```
hipLaunchKernelGGL: ~10 µs per call (dominated by runtime wrappers + signal)
```

**The current Redline path is 3-6× slower than HIP, not faster.** The
`cs_submit` ioctl + fence wait alone is ~30 µs on gfx1100. There is no way
to beat HIP while we keep paying an ioctl per dispatch.

## The real gap

To beat HIP we need one of these three dispatch paths working on gfx1100:

### Path A — HSA FFI (recommended primary)

Link/dlopen `libhsa-runtime64.so` (already on the system) and use its
user-mode AQL queue directly:

```
hsa_init()
hsa_iterate_agents() → find gfx1100 agent
hsa_queue_create(agent, 1024, HSA_QUEUE_TYPE_SINGLE, ...) → user-mode ring
hsa_queue_load_write_index_relaxed(queue)
write 64-byte AQL dispatch packet to ring
hsa_queue_store_write_index_release(queue, idx+1)
*doorbell = idx+1                         ← MMIO, no syscall
hsa_signal_wait_scacquire(completion_signal, ...) ← only when we actually need to sync
```

**Why this wins:**

- ROCr handles the ugly parts (CREATE_QUEUE, VRAM-backed MQD, CWSR sizing,
  hw-specific doorbell paging, signal event plumbing)
- Doorbell ring is a single MMIO write, no kernel transition
- We can batch completion: submit N packets, wait on a single counting signal
  at the end — per-dispatch overhead approaches the MMIO write cost (<1 µs)
- Works on any ROCm-supported card (gfx1100, gfx1030+, datacenter) without
  per-arch packet wizardry
- HIP sits ON TOP of this exact layer — our 10 µs floor is HIP's wrapping,
  not the underlying AQL path

**Cost:** thin `hsa-bridge` crate parallel to `hip-bridge`, ~15 FFI entry
points, re-uses existing `.hsaco` loading from `redline::hsaco`. Kernel code
objects are the same files hipcc already produces; only the dispatch path
changes.

**Risk:** dependency on ROCm being installed. Same constraint hip-bridge
already has. If we want a ROCm-less path, Path B is the fallback.

### Path B — Raw KFD AQL queue (current `src/kfd.rs`, needs fixes for gfx1100)

Fix `AqlQueue::new` by:
1. Allocating the ring buffer, write-ptr, read-ptr, EOP, and CWSR via
   `KFD_IOC_ALLOC_MEMORY_OF_GPU` with `KFD_IOC_ALLOC_MEM_FLAGS_VRAM`
   (not `USERPTR`).
2. Using RDNA3-appropriate CWSR/ctl-stack sizes (current: 2.5 MB / 12 KB —
   likely wrong for RDNA3).
3. Calling `KFD_IOC_CREATE_QUEUE` with queue_type `COMPUTE_AQL=0x2`.
4. `mmap`ing the doorbell page returned in `doorbell_offset`.

**Why it's worth the effort:** no ROCm dependency at all. A Rust-native user
of Redline could ship without `libhsa-runtime64.so` present. Same dispatch
cost as Path A once it works.

**Cost:** debugging gfx1100 KFD semantics from scratch (no headers that tell
us exactly what RDNA3 wants). Likely needs reference to `rocr-runtime` source.
Much more work than Path A for the same dispatch win.

**Recommendation:** defer to after Path A validates the launch-overhead
hypothesis.

### Path C — Fix PM4 chain dispatch on gfx1100 (current `CommandBuffer::barrier`)

The `RELEASE_MEM` / `WAIT_REG_MEM` packet encoding in `src/dispatch.rs:263`
was reverse-engineered against `gfx_v10_0.c` (RDNA1). RDNA3 needs
`gfx_v11_0.c` encoding. Fix the packets → chain N dispatches into one
`cs_submit` → amortize the ~30 µs ioctl cost over N dispatches.

**Why it's still worth considering:**

- Pure libdrm path (no ROCm, no KFD). Portable to any card with `amdgpu` KMD.
- For the specific case of "dispatch 50 small kernels in a row with fences
  between them," it can beat even HSA because there's no per-dispatch
  doorbell write or signal wait — the GPU walks the IB from dispatch to
  dispatch on its own.

**Cost:** RDNA3 packet decoding (RELEASE_MEM event type, GCR_CNTL layout,
WAIT_REG_MEM flags). Mechanical work once we have the right kernel source.

**Recommendation:** pursue as Path A's insurance policy. If Path A bogs down
or we need a ROCm-less distribution, Path C is the fallback.

## Strategy decision

**Primary:** Path A (HSA FFI). Lowest-effort way to hit the <3 µs per-launch
target. Uses the exact same layer HIP sits on — we lose only the HIP
wrappers, signal-wait conservatism, and cross-stream synchronization cost.

**Shadow:** Path C (PM4 chain fix). Fixing RDNA3 barrier packets is
independently valuable (it unlocks the existing chain-dispatch test and
validates the raw-libdrm path for future arches). Not a blocker for the
launch-overhead win.

**Deferred:** Path B (raw KFD AQL). Same goal as Path A but much more work.
Only pursue if we want a ROCm-less dispatch path later.

## Phase 2 revised plan

1. Add `hsa-bridge` crate (thin dlopen wrapper over `libhsa-runtime64.so`).
   Mirror the structure of `hip-bridge`. Expose:
   - `HsaRuntime::init`
   - `HsaRuntime::find_gpu_agent(gfx_arch)`
   - `HsaQueue::create(agent)`
   - `HsaQueue::submit_dispatch(packet, completion_signal)` — user-mode doorbell
   - `HsaSignal::create/wait/destroy`
   - `HsaExecutable::load_code_object(hsaco_bytes)` — returns a map from kernel
     name → `kernel_object` (kernel descriptor VA for AQL packets)
   - `HsaMemory::alloc_device_memory(agent, size)` — for kernel input/output VAs
2. Pick a small single-launch kernel (`add_inplace_f32` or `scale_f32`). Build a
   test harness that:
   - Loads the same `.hsaco` that `hipfire-rdna` builds today.
   - Uses the **same input buffers** (just reuse the GpuTensors from
     `hipfire-rdna` — we don't need to re-allocate).
   - Fires the kernel once through HIP (baseline) and once through HSA.
   - Measures latency with `std::time::Instant` inside a 10 000-iter loop.
3. Target: HSA median < 3 µs, both paths produce identical output. If HSA is
   still 8+ µs, the next suspect is completion-signal wait, and we should
   switch to a counting-signal pattern (fire-and-forget, wait once at the end).
4. Deliverable: measured A/B comparison for one kernel, PR-sized commit.

If Phase 2 lands under 3 µs/dispatch, Phase 3 scales to all small kernels,
Phase 4 to the GEMVs. If it doesn't, fall back to Path C (fix RDNA3 barrier
packets) and try chain-dispatch amortization instead.

## Gap list (concrete to-do for Phase 2)

- [ ] Create `crates/hsa-bridge/` crate, mirror `hip-bridge` structure
- [ ] Add minimal FFI bindings to `libhsa-runtime64.so` (subset of hsa.h)
- [ ] Extend HSA bindings with `hsa_ext_amd.h` for AMD-specific memory alloc
- [ ] Provide `HsaExecutable::load_from_memory` that feeds a `.hsaco` into
      `hsa_code_object_reader_create_from_memory` / `hsa_executable_load_agent_code_object`
- [ ] Provide a `kernel_object` query — the VA that goes in AQL packet [32:39]
- [ ] Implement `submit_dispatch`:
      - load write index, spin until queue has space
      - write AQL packet to ring (header written LAST with release fence)
      - store write index, write doorbell
- [ ] Implement `signal_wait_scacquire` wrapper
- [ ] Thread the AQL path through `hipfire-rdna::dispatch` for ONE kernel
      (keep HIP path as fallback; gate on a `use_hsa: bool`)
- [ ] Run the same in-process hipEvent profiler against HIP and HSA, compare
- [ ] Document: launch latency, total forward delta, any correctness diff
