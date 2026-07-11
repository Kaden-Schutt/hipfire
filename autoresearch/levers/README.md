# autoresearch/levers/ — shape kernels to the target ISA, use ALL the leverage

**The principle (the hipfire thesis — do not violate):** AMD consumer RDNA does NOT self-optimize the way
the CUDA/NVIDIA reflex assumes. The compiler/driver/runtime will NOT manage cache residency, will NOT pick
the cheapest instruction, will NOT saturate bandwidth for you. Every hardware capability is a LEVER you pull
EXPLICITLY, per-arch. This is why `gfx11 != gfx12` kernels exist — each gen inherits + changes the ISA, and a
kernel shaped for one arch's instructions / cache / issue rules is wrong for another. hipfire is the
invalidation of the "it'll self-optimize" reflex; the loop must encode that, not re-learn it every run.

## Files
- **This guide** — the arch-agnostic method + the instrument→lever map.
- **[gfx1100.md](gfx1100.md)** — RDNA3 / Radeon 7900XTX (96CU, 960 GB/s, 96MB MALL) — the DRAM-thrash arch.
- **[gfx1201.md](gfx1201.md)** — RDNA4 / R9700 (64CU, 640 GB/s, cache-resident, FP8/FP4).
- *(to add as fleet campaigns run: gfx1151 Strix Halo, gfx1030 RDNA2, gfx1010 RDNA1, gfx942 CDNA.)*

Each arch file is a **living map**: every campaign appends what it learned (the measured bottleneck, the
levers that won/died, why). Treat it like the kernel forks — form-fitting per silicon.

## How the loop uses this
1. **Brainstorm phase** — the target arch's file is the CONTEXT capability frame: *what can THIS silicon do
   that we aren't using?* Do not propose a lever without (a) a capability from the arch file AND (b) an
   instrument that says it's idle.
2. **Autoresearch phase** — cross-reference the census: a low instrument reading NAMES the wasted lever. The
   certify's `profile_feedback` surfaces the agent's own variant's occ/VGPR/L2-hit/mem_busy vs baseline,
   annotated with the lever, so it steers instead of thrashing (thrashing = every verdict DEAD).
3. **Fold phase** — an arch-specific win FORKS to `<k>.<arch>.hip` (see `harness/v3-queue/promote_fork_prompt.txt`),
   never clobbers the shared `<k>.hip`. Universal wins (cross-arch re-verified) may go shared.

## Instrument → lever map (the census reads out the lever; arch-agnostic)
| census reading | meaning | LEVER |
|---|---|---|
| **L2-hit LOW (<30%) + DRAM-miss HIGH** | reads thrash DRAM; reused x/KV evicted by the write-once weight stream | **cache-residency**: SLC/streaming loads on write-once weights (bypass), keep x/KV/scales resident, engage MALL |
| occ ~0.5% + mem_busy moderate | pathologically under-occupied, NO TLP, latency fully exposed | **per-wave MLP**: register-ring prefetch + deferred reduction (v3 attention win). ONLY wins here. |
| occ 30-50% + mem_busy moderate + L2-hit low | TLP already hides latency; NOT MLP-limited | **cache-residency**, NOT more MLP (v4 proved per-wave MLP no-ops here). |
| mem_busy ~90% + occ ~90% | at the memory roofline | leave it, saturated. |
| high VALU + low mem, **prefill/GEMM (batch>1)** | compute-bound matrix×matrix | **matrix cores**: fp8/fp4 WMMA (RDNA4), WMMA fp16/int8. PREFILL ONLY — decode is batch-1 GEMV, wastes ~15/16 of a matrix core. |
| high VALU + low mem, **decode/GEMV (batch-1)** | compute-bound matrix×vector | **cheaper scalar/vector**: dp4a/sdot4 (int8), fdot2 (fp16), VOPD dual-issue (RDNA3), packed math — NOT WMMA. |
| VGPR at the occupancy cliff | a wave dropped = regression (the row-reuse failure mode) | `global_load_lds` (frees VGPR + prefetches), fewer temporaries, scalarize wave-uniform values. |

**Look up exact mnemonics** in the emitted ISA (`llvm-objdump -d` on the `.hsaco`) and the LLVM AMDGPU
backend (`llvm/lib/Target/AMDGPU/*.td`) / AMD RDNA ISA docs. The map is the *what*; the ISA is the *how*.
