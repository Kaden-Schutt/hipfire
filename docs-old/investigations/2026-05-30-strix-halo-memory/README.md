# Strix Halo (gfx1151) beyond-carveout memory: can hipfire run >VRAM models without Vulkan?

**Status (2026-05-30):** Investigated on hipx (AMD Ryzen AI Max+ 395 / Radeon
8060S, gfx1151, 128 GB unified LPDDR5x). **Finding: yes — the KMD GTT-GEM path
works and reaches memory beyond the VRAM carveout at ~55–65% of carveout read
bandwidth, no Vulkan required.** `hipMallocManaged` is dead on RDNA3.5;
`hipHostMalloc` works but is slow; direct `AMDGPU_GEM_DOMAIN_GTT` allocation via
libdrm_amdgpu + dma-buf + `hipImportExternalMemory` is the viable route.

## Why this matters

MiniMax-M2.7 mq2-lloyd (86 GB) runs on a single Strix Halo today because it fits
the **96 GB VRAM carveout**; `hipMalloc` uses the carveout and OOMs above it.
The bigger tiers (mq3 102, mq3-lloyd 109, mq4 124 GB) exceed the carveout. The
question: can hipfire address memory beyond the carveout — like llama.cpp's
Vulkan backend appears to — through its ROCm/KMD-direct stack, without adopting
Vulkan (out of scope, issue #44)?

## Hardware memory topology (hipx, measured)

- Unified 128 GB LPDDR5x (~256 GB/s peak), shared CPU+GPU (it's an APU).
- BIOS split today: **96 GiB VRAM carveout** + ~30 GB system RAM.
- amdgpu **GTT pool = 15.2 GiB** (`mem_info_gtt_total`, `amdgpu.gttsize` default).
- HSA pools: the gfx1151 agent exposes **only** the 96 GiB carveout
  (COARSE-GRAINED + EXTENDED-FINE-GRAINED, "Accessible by all: FALSE"). There is
  **no GTT pool on the GPU agent**; system RAM is only on the CPU agent as a
  FINE-GRAINED (coherent) pool. So no HSA-level fast path beyond the carveout.
- gfx1151 = HIP **device 1**, renderD129 (device 0 / renderD128 = a 7–8 GB RX
  5700 XT, gfx1010). Pin `HIP_VISIBLE_DEVICES=1` ALONE (compounding with
  `ROCR_VISIBLE_DEVICES` → empty device set).

## Allocation paths tested (byte-strided kernel; ratios are the signal)

| path | works? | WRITE | READ | capacity | notes |
|---|---|---|---|---|---|
| `hipMalloc` (carveout) | ✅ | 70–119 | 86–108 GB/s | 96 GB | the fast tier; the only thing `hipMalloc` reaches |
| `hipMallocManaged` | ❌ | — | — | — | **OOM even at 50 GB** (inside carveout). `gfx1151:xnack+` is an invalid target — **RDNA3.5 has no XNACK**. Managed memory is non-functional here. |
| `hipHostMalloc` mapped (coherent) | ✅ | ~30 | ~50 GB/s | system RAM | dev ptr == host ptr (zero-copy); slow coherent aperture |
| `hipHostMalloc` write-combined | ✅ | ~26 | ~58 GB/s | system RAM | flag didn't help; alloc slow (1.4–5.9 s/20 GB) |
| **GTT-GEM (KMD + dma-buf + HIP import)** | ✅ | 29–36 | **58–69 GB/s** | GTT (15 GB, growable) | `AMDGPU_GEM_DOMAIN_GTT` BO → dma-buf → `hipImportExternalMemory`. The viable beyond-carveout path. |

Apples-to-apples at 10 GB: carveout READ **107.8** vs GTT-GEM READ **69.1**
(~64%); GTT-GEM at 14 GB drops to 57.7 (~54%), converging toward the host-pinned
~58. Absolute peaks are undermeasured (uint8 strided kernel — latency/occupancy
bound, not peak BW); the **carveout-vs-GTT ratio (~55–65% read)** is the result.

## Interpretation

- **There is a real GPU→system-RAM fabric bottleneck (~58–69 GB/s) on gfx1151**,
  independent of allocator. The carveout (GPU-local) is the fast tier; anything
  in system RAM (GTT or host-pinned) tops out ~60–65% of carveout read. GART
  mapping (GTT-GEM) beats the coherent host-pinned aperture at small sizes but
  converges at scale.
- **Vulkan has no advantage here.** radv hits the same fabric and the same
  carveout-vs-system split. The reason a Strix Halo box runs a 105 GB Q3 model
  at ~32 t/s is almost certainly a **larger BIOS carveout** (model in fast VRAM),
  not fast GTT — i.e. the carveout-size lever, which both Vulkan and hipfire
  share equally. hipfire is **not behind on the mechanism.**
- **GTT-GEM is a viable capacity extension at a read penalty**, best used as a
  hybrid: dense/hot tensors (attention, router, lm_head) in the carveout; cold
  routed experts in GTT-GEM. A MoE reads only k/N experts per token, so the
  ~35–45% read penalty only touches expert reads — the path to run **mq4
  (124 GB, exceeds any single carveout)** on one Strix Halo without Vulkan.

## Recommendation / levers

1. **mq3 / mq3-lloyd (≤~112 GB): raise the BIOS VRAM carveout** so they fit fast
   VRAM. Zero code; the same thing the fast-benchmark boxes do. (Deferred —
   needs physical/UEFI access; hipx reportedly caps at 96 GB, to be confirmed.)
2. **mq4 / anything > max carveout: carveout + GTT-GEM hybrid** placement. The
   real hipfire-native unlock; `redline/src/drm.rs` already has the
   libdrm_amdgpu `AMDGPU_GEM_DOMAIN_GTT` binding + CS structs. Work: wire a
   GTT-GEM allocation path into the `rdna-compute` allocator (alloc GTT BO →
   dma-buf export → `hipImportExternalMemory` → device ptr; kernels unchanged),
   behind a hot/cold placement policy in the weight loader.
3. **Dead end: managed memory** (`hipMallocManaged`) — non-functional on RDNA3.5
   (no XNACK). Do not pursue.

## Reproduction

Test programs (this dir): `gtt_probe.cpp` (KMD GTT-GEM + dma-buf + HIP import),
`managed_spike.cpp` (`hipMallocManaged`), `host_spike.cpp` (`hipHostMalloc`),
`dev_bw.cpp` (carveout baseline). Build: `hipcc --offload-arch=gfx1151 -O3 X.cpp
-o X [-I/usr/include/libdrm -ldrm_amdgpu]`. Run pinned: `HIP_VISIBLE_DEVICES=1
./gtt_probe 10`. Needs membership in the `render` group for /dev/dri/renderD129.

## Open items

- Vectorized (uint4/float4) bandwidth test to firm the absolute ceiling + confirm
  the ~60% ratio holds at peak.
- `amdgpu.gttsize` bump (reboot) to grow GTT past 15 GB (bounded by system RAM;
  trades against carveout size).
- Confirm hipx's BIOS max carveout (lever #1).

## Separate: NPU

The AI Max's XDNA2 NPU (~50 TOPS INT8) is unused. Different driver (`amdxdna`) +
toolchain (IRON/Peano), not HIP; strong for INT8 GEMM but limited local memory
and a dataflow programming model — non-obvious fit for a 229 B MoE trunk, more
plausible as a drafter / spec-decode offload. Scope separately, not folded here.
