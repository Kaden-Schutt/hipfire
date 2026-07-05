# GPU ISA + lever map — shape kernels to the target ISA, use ALL the leverage

**Principle (the hipfire thesis, do not violate):** AMD consumer RDNA does NOT self-optimize the way the
CUDA/NVIDIA reflex assumes. The compiler will NOT auto-manage cache residency, will NOT pick the cheapest
instruction, will NOT saturate bandwidth for you. Every hardware capability is a LEVER you must pull
EXPLICITLY, per-arch. This is why `gfx11 != gfx12` kernels exist — each gen inherits + changes the ISA, and
a kernel shaped for one arch's instruction set / cache / issue rules is wrong for another.

**How to use this doc:** in the brainstorm phase it is the capability frame (what CAN this chip do that we
aren't using?). In the autoresearch/rollover phase it is cross-referenced with the census: **a low instrument
reading names the lever that is being wasted.** Do not propose a lever without (a) a capability from this map
and (b) an instrument that says it's currently unused. Look up exact mnemonics in the emitted ISA
(`llvm-objdump -d` on the `.hsaco`) and the LLVM AMDGPU backend (`llvm/lib/Target/AMDGPU/*.td`) / AMD RDNA ISA docs.

## Instrument → lever map (the census tells you which lever)
| census reading | what it means | LEVER to pull |
|---|---|---|
| **L2-hit LOW (<30%) + DRAM-miss HIGH** | reads thrash DRAM; reused data (x/KV) evicted by the write-once weight stream | **cache-residency**: non-temporal (SLC/streaming) loads on write-once weights so they bypass cache; keep x/KV/scales resident; engage MALL. *(THE gfx11 ratio bug: qkvza 16% hit vs gfx12 78%.)* |
| occ ~0.5% + mem_busy moderate | pathologically under-occupied, NO thread-level parallelism, latency fully exposed | **per-wave MLP**: register-ring prefetch + deferred reduction (the v3 attention win). ONLY wins here. |
| occ moderate (30-50%) + mem_busy moderate + L2-hit LOW | TLP already hides latency; NOT MLP-limited | **cache-residency** (above), NOT more MLP (v4 proved per-wave MLP no-ops here). |
| mem_busy ~90% + occ ~90% | at the memory roofline | leave it; it's saturated. |
| high VALU + low mem | compute-bound | **cheaper instructions**: fp8/fp4 WMMA (gfx12), dp4a/sdot4 (int8), VOPD dual-issue (gfx11), packed math. |
| VGPR at the occupancy cliff | a wave dropped = regression | `global_load_lds` (frees VGPR + prefetches), fewer temporaries, scalarize wave-uniform values. |

## Per-arch capability map

### gfx1100 — Radeon 7900XTX, RDNA3 / Navi31
- 96 CU, wave32, 32 waves/CU, 64KB LDS, 2526 MHz, 24GB, **~960 GB/s**. **6MB L2 + 96MB Infinity Cache (MALL). 128B cacheline.**
- **CACHE IS THE PROBLEM HERE:** measured L2-hit 12-20% on the GEMV/MoE kernels = DRAM-thrash (vs gfx12 ~77%).
  RDNA3's cache/MALL is NOT self-managing this workload. Levers: SLC/streaming loads on the write-once
  expert-weight stream (stop it evicting x/KV), cache-hint the reused activation, verify the 96MB MALL is even
  engaged (16% hit with 96MB is suspicious — check driver/`amdgpu` MALL enable + the access pattern that triggers it).
- Compute levers: **VOPD dual-issue** (2 independent FP32 ops/cycle — pack independent FMA pairs), WMMA
  fp16/bf16/int8 (present but scalar MMQ chosen for HFQ4 after iu8-WMMA lost; fp16-WMMA UNEXPLORED),
  `global_load_lds` (direct global→LDS, async, frees VGPR), fdot2/dp4a, `v_perm_b32`/`v_cvt_f32_ubyte0..3` (fast nibble/byte dequant).
- **NO native FP8/FP4 compute.** int8/fp16 is the floor.

### gfx1201 — Radeon AI PRO R9700, RDNA4 / Navi48
- 64 CU, wave32, 32 waves/CU, 64KB LDS, ~2350+ MHz, 32GB, **~640 GB/s**. L2 + ~64MB MALL. **256B cacheline.**
- **Cache is EFFECTIVE here** (L2-hit ~64-78% on the same kernels) — this is WHY gfx12 keeps pace with gfx11's
  1.5x bandwidth: gfx11 pays DRAM latency, gfx12 pays cache latency. Coalesce to 256B.
- Compute levers: native **WMMA** (grouped GEMM uses it), and the BIG unused one: **FP8 (E4M3/E5M2) + FP4**
  WMMA + conversion instructions. hipfire has always gone fp16 → int8 and never used the fp8 path. For 4-bit
  quant (mq4r) the intermediate precision budget easily tolerates fp8 accumulate → potential ~2x GEMM
  throughput vs fp16 where the kernel is compute-bound. **UNEXPLORED — the cheapest-instruction lever we keep
  believing in but haven't landed.** Confirm the exact `v_wmma_*_fp8`/`v_cvt_*_fp8` mnemonics from the RDNA4 ISA.

### Fleet (validate before assuming a gfx11/12 win transfers)
- gfx1151 Strix Halo (RDNA3.5), gfx1030 (RDNA2), gfx1010 (RDNA1): each inherits+changes the ISA (RDNA1 has no
  WMMA/dp4a; RDNA2 no WMMA; RDNA3.5 ~= RDNA3). A shared-base kernel must be clobber-checked per arch
  (r2lds was coherent on gfx1201, INCOHERENT on gfx1151). Fork to `<k>.<arch>.hip` when the ISA diverges.

## Cross-gen inheritance (gfx11 → gfx12 deltas)
| capability | gfx11 (RDNA3) | gfx12 (RDNA4) |
|---|---|---|
| cacheline | 128B | 256B |
| cache residency (this workload) | **DRAM-thrash 12-20% L2-hit** | L2-resident ~77% |
| matrix | WMMA fp16/bf16/int8 | + **FP8/FP4 WMMA** |
| dual-issue | VOPD | improved issue |
| quant precision floor | int8 | **fp8/fp4** |

The point: a kernel that is optimal on gfx12 (256B-coalesced, fp8-WMMA, cache-resident) is wrong on gfx11
(128B, no fp8, must MANUALLY manage residency). Shape to the target ISA. Never assume the stack does it for you.
