# MQ4 (HFQ4-G256) 128B weight-layout alignment — DOCUMENTED NEGATIVE

**TL;DR — the cacheline-alignment lever is FALSIFIED. Do not re-chase.**
Separating the MQ4 scale/zp out so the 128B nibble blocks land on 128B
boundaries does **not** speed up decode: the actual DRAM traffic is byte-for-byte
identical (measured), the residual effect is a size-fragile L1-request artifact,
and at the *real* a3b decode working-set scale the aligned layout is **noise on
gfx1201 and a net −12 % REGRESSION on gfx1100**. Both layouts are bit-exact.

## The hypothesis

hipfire's MQ4 (HFQ4-G256) weight group is **136 bytes, interleaved**:
`[4B scale][4B zp][128B nibble-block (256 nibbles)]`. The 128B nibble block —
the bulk of every weight load — sits at **offset 8**, so it is NOT 128B-aligned.
Claim: on gfx1100 (128B lines) / gfx1201 (256B lines) the misaligned 128B block
straddles two cachelines → extra fetches → higher decode load-latency (the
83–94 % DEP_WAIT MoE GEMVs, ~42 % of decode). Proposed fix: split scale/zp into
a separate array so the 128B nibble blocks are 128B-aligned.

Two layouts, identical dequant math and values (`mq4_align_bench.hip`):
- **A (current):** 136B interleaved. Warp's 32×4B nibble load spans
  `[group_base+8, group_base+136)` — a 128B window at a non-128-aligned offset.
- **B (aligned):** nibble blocks in their own 128B-stride array (aligned) +
  scales/zp in a separate array. Warp's nibble load is a fully aligned 128B span.

`mq4_align_moe.hip` is a scattered-multi-expert variant (grid=(rows,E), each
block reads a different expert base) mirroring `gemv_hfq4g256_moe_gate_up_k8_indexed`.

Build/run:
```
hipcc -O3 --offload-arch=gfx1201 mq4_align_bench.hip -o mq4b
HIP_VISIBLE_DEVICES=<gfx1201-dev> ./mq4b -M 16384 -K 4096 -iters 200 -runs 5
```

## Why it's a negative — the measurement

Parity is **bit-exact** (`max|A-B| = 0.000e+00`) at every size on both archs —
the re-layout changes only memory placement, not values.

### 1. DRAM traffic is IDENTICAL (the strong-form hypothesis is falsified)

rocprofv3 (`profile_standard` perf level, gfx1201, M=4096 K=2048, per-dispatch):

| counter | A (136B) | B (aligned) | reading |
|---|---:|---:|---|
| **GL2C_MISS** (L2→DRAM fetches) | **17505** | **17505** | **identical → same DRAM bytes** |
| GL2C_HIT | 30696 | 43065 | B re-references L2 more (extra scale stream) but all hit |
| TCP_REQ (L1 requests) | 624640 | 561152 | A +11.3 % more L1 requests |
| SQ_INSTS_TEX_LOAD | 131072 | 98304 | A issues more VMEM loads |

**Root cause.** The 136B group is *densely* read — scale(4) + zp(4) + all 128B
of nibbles, no gaps. So when the warp's 128B nibble load straddles two 128B L2
lines, the **second line's bytes are still consumed** (by the neighbouring
group's scale/zp), i.e. it is already resident — a **cache hit, not an extra
DRAM fetch**. `GL2C_MISS` proves it: the actual bottleneck (DRAM traffic) is
unchanged. The misalignment costs only extra *L1 request slots* (+11 % TCP_REQ),
not memory bandwidth. The hypothesis's causal chain (misalignment → extra
fetches → DEP_WAIT) is broken at "extra fetches."

### 2. Wall-time benefit is size-fragile and REVERSES at real scale

gfx1201 (R9700, L2=8MB), B-vs-A, best-of-5, AUTO clock:

| working set | gpr | Δ (B faster) | note |
|---|---:|---:|---|
| 1.06 MB | 8 | +0.7 % | ≈ single a3b expert gate_up — **noise** |
| 1.59 MB | 8 | +1.3 % | noise band |
| **4.25 MB** | 8 | **+39.2 %** | **L2-aliasing resonance of the 136B non-pow2 stride — artifact** |
| 8.50 MB | 8 | +4.7 % | ≈ one 8-expert decode gate_up launch |
| 17.0 MB | 16 | −0.5 % | noise |
| 34.0 MB | 16 | +1.6 % | DRAM-bound |
| **68.0 MB** | 16 | **−16.7 %** | **B SLOWER** (two-stream layout pathology) |

gfx1100 (7900 XTX, L2=6MB, RDNA3 128B lines — the arch the hypothesis targets):

| working set | gpr | Δ (B faster) | note |
|---|---:|---:|---|
| 1.06 MB | 8 | +24.8 % | tiny, L2-resident aliasing (not the decode regime) |
| 4.25 MB | 8 | +5.3 % | |
| **8.50 MB** | 8 | **−12.5 %** | **≈ real 8-expert decode gate_up → B is a REGRESSION** |
| **34.0 MB** | 16 | **−13.3 %** | **DRAM-bound → B is a REGRESSION** |

**The direction flips with working set.** The "+" only appears when the whole
matrix is L2-resident *and* small (an L2-set-mapping quirk of the 136B non-power-
of-2 stride, not a per-load straddle — see the razor-thin +39 % spike at exactly
4.25 MB). At the sizes decode actually touches — ~8.5 MB scattered across 8
experts, partly DRAM-bound — the aligned two-region layout is **worse**, because
splitting nibbles and scales into two streams costs more than the aligned L1 load
saves (two DRAM access streams / more row activations vs one contiguous 136B
burst). On gfx1100 this is a clean **−12 to −13 % regression** at real scale.

## Real a3b decode regime (settles it)

`qwen3.6-35b-a3b.mq4r` confirmed **MQ4G256 / 136B** (routed experts
`experts.N.gate_up_proj.weight`, no PaRo sidecars → `load_moe_ffn` HFQ path).
`moe_intermediate=512` → per-expert gate_up = **1.06 MB** (M=1024, K=2048,
gpr=8); top-8 → **~8.5 MB scattered per gate_up launch**. That is exactly the
row where gfx1201 = +4.7 % (largely the aliasing artifact) and **gfx1100 = −12.5 %**.
Net expected end-to-end decode effect: **≤ noise on gfx1201, negative on gfx1100.**

## Verdict

The 128B weight-layout alignment is **not** a decode lever. It does not reduce
DRAM traffic (the real bottleneck), its microbench "wins" are an L2-aliasing
artifact of the 136B stride that vanishes or reverses at decode scale, and on the
very arch the hypothesis targeted (gfx1100) it is a net regression. A real-path
encoder/loader/kernel prototype was scoped but **deliberately not landed** — the
expected end-to-end value straddles zero (worse on gfx1100), not worth the MQ4
wire-format churn. Recorded here so it is never re-chased.
