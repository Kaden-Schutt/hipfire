# Tuning levers catalog

Patterns hipfire has used on real kernels. Pick **one** per commit after
`playbook.md` root-cause. Paths are under `kernels/src/` unless noted.
Commits are historical anchors — verify the hash exists in your checkout
before citing externally.

Perf numbers are **measured** snapshots, not floors or admissions. Protocol:
[`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md).
Variant allowlisting:
[`docs/methodology/perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md).

## How to choose

| Diagnosed bottleneck | Start here |
|---|---|
| Wave32 kernel on wave64-native CDNA | §1 Wave-size port |
| Decode launch-bound / low GB/s GEMV | §2 Multi-row GEMV |
| Prefill GEMM K-loop overhead | §3 K-tile depth |
| Prefill batch with idle matrix units | §4 WMMA / MFMA |
| Decode L2 misses on gfx12 weights | §5 `s_prefetch_data` |
| Multiple projections share `x` | §6 Fused projections |
| One kernel wants special hipcc flags | §7 Per-kernel hipcc flags |
| Large-M CDNA prefill | §8 rocBLAS / library GEMM |
| Multi-wave WG, BW% ≪ peak, barrier in loop | §10 Barrier-free / nosync |
| Already tried below | §9 Negative results — new mechanism only |

---

## 1. Wave-size port (CDNA ⇄ RDNA)

**When:** target is wave64-native (gfx94x / selected GCN) but the hot kernel
is wave32 — half the lanes mask out; output can still be correct at ~½
throughput.

**Pattern:** separate `*.wave64.hip` (or chip `.gfx942.hip`) with wave64 lane
decomposition; dispatch only via wave64-native capability, not as a perf
tweak on RDNA.

**Shipped anchor:** `4105035` — full wave64 port of hot HFQ4 kernels; MI300X
decode roughly 2× on the measured A3B setup (`case-studies.md` §1).

**Tree examples:** `fused_qkv_hfq4g256_wave64.hip`,
`gemv_hfq4g256_residual` wave64 siblings, `moe_router_softmax_topk_k8_wave64.hip`.

**Don't:** force wave64 mode on RDNA inference paths expecting free wins —
occupancy cost usually dominates.

---

## 2. Multi-row GEMV

**When:** decode is launch- or latency-bound on GEMV; profile shows modest
GB/s and high invoke count. Share `x` across R output rows per warp.

**Pattern:** R=2/4/8 entry points; tune R per arch (VGPR budget).

**Tree (verify before edit):**

```
kernels/src/gemv_hfq4g256_multirow.hip
kernels/src/gemv_hfq4g256_multirow.gfx1100.hip
kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip
```

Dispatch maps multirow module names to `r2`/`r4`/`r8` symbols in
`crates/rdna-compute` (see `gemv_hfq4g256_multirow_*` registration). Env
knobs such as `HIPFIRE_GEMV_ROWS` may select R — confirm in current
`feature_flags` / Atlas dispatch provenance before assuming defaults.

**Trade-off:** higher R → VGPR pressure → spills. Past comfortable R the win
vanishes (often ≤8 on RDNA3 dGPU; tighter on smaller VGPR budgets).

**Don't reinvent** — fork the closest existing multirow file.

---

## 3. K-tile depth (K2 / K4 / K-split)

**When:** WMMA prefill GEMM inner K-loop dominates; deeper unroll amortizes
tile setup until registers spill.

**Pattern:** K2 soft-pipeline is the common baseline; K4 / ksplit are
opt-in or shape-selected variants.

**Tree examples:**

```
kernels/src/gemm_hfq4g256_residual_wmma.hip          # baseline family
kernels/src/gemm_gate_up_hfq4g256_wmma_k4.hip        # HIPFIRE_GATE_UP_VARIANT=k4
kernels/src/gemm_hfq4g256_residual_wmma_ksplit_det.hip
```

**Positive:** residual/gate_up/qkv HFQ4 WMMA K2 paths are production workhorses.

**Null result anchor:** `f670e16` — k2x32 wider-row lm_head **slower** from
register pressure (`case-studies.md` §3). Revisit only with an LDS/B-share
plan and a fresh ISA budget.

**Policy:** K-split / K4 selection must be measured per arch class — do not
key off a broad `is_rdna3p5()` without a ledger-style note
(`perf-arch-discipline.md`).

---

## 4. WMMA / MFMA matrix engine

**When:** prefill (batch ≫ 1) on an arch with matrix engines. Hipfire uses
WMMA on gfx11/gfx12 and MFMA on gfx94x; adjacent fallbacks are
**packed-FP16** (e.g. gfx1010/gfx1013 HFQ4 batched QKV/gate-up) and **dot2**
(gfx1030-class), with **scalar** only as the final kernel-family-dependent
baseline.

**Pattern:** 16×16×16 fp16→fp32 tiles are common; **builtin name and C-mapping
differ by family**. gfx12 needs `_w32_gfx12` sisters — gfx11 builtins do not
lower (`has_wmma_w32` vs `has_wmma_w32_gfx12` in `arch_caps.rs`).

**Tree examples:**

```
kernels/src/gemm_gate_up_hfq4g256_wmma.hip
kernels/src/gemm_gate_up_hfq4g256_wmma.gfx12.hip
kernels/src/gemm_qkv_hfq4g256_wmma*.hip
kernels/src/gemm_hfq4g256_moe_grouped_wmma.gfx12.hip
```

**Port skill:** new ISA shapes → `.agents/skills/hipfire-arch-port/`, not this
catalog alone.

**Correctness:** channel-test element-wise vs reference is mandatory. The
gfx11 WMMA C-mapping bug was **fixed in `b7ac66a`** after ~6 weeks latent;
speed/serve-shaped checks had stayed green (`case-studies.md` §4).

---

## 5. Software prefetch (`s_prefetch_data`)

**When:** hot decode weight streaming on **gfx12 / RDNA4**; L2 miss dominated.

**Pattern:** chip or family override using RDNA4 prefetch intrinsic.

**Tree:**

```
kernels/src/gemv_hfq4g256.gfx1201.hip
```

Header comments document VGPR/occupancy trade for lookahead groups. Prefer
`.gfx1201.hip` for 9070-class chip tuning; `.gfx12.hip` when the same binary
must cover gfx1200+gfx1201 without chip splits (`cross-arch.md` resolution
order).

**Don't:** expect the same intrinsic on gfx11.

---

## 6. Fused projections

**When:** Q/K/V (and DeltaNet z/β/α) or gate+up share one `x`; separate
launches dominate small-M decode.

**Pattern:** multi-output kernels — `fused_qkv_*`, `fused_qkvza_*`,
`gemm_gate_up_*`, `fused_gate_up_*`.

**Anchor:** fused QKV consolidation work (e.g. `9d05c9f` family) is a major
reason decode can beat unfused baselines on small models — one weight pass,
multiple Y buffers.

**Trade-off:** larger live ranges; on VRAM/occupancy-tight chips the unfused
path can win. Keep fall-through in dispatch.

---

## 7. Per-kernel hipcc flags

**When:** one kernel wants non-default compile flags without global `-Xclang`
risk.

**Pattern:** magic comment picked up by the kernel JIT, e.g.
`// HIPFIRE_COMPILER_FLAGS: ...` at top of the `.hip` file (plumbing anchor `5f65005`).
Rebuild kernel hashes if your tree still uses
`scripts/write-kernel-hashes.sh` for the path you touched.

Validate with the claim-scoped correctness route + fresh-process measure —
flags can change numerics under fast-math-like options.

---

## 8. rocBLAS / library GEMM fallback

**When:** very large-M prefill on CDNA3 where library MFMA beats hand-rolled
paths.

**Pattern:** optional rocBLAS route behind env kill-switches (historically
`HIPFIRE_ROCBLAS_OFF`, `HIPFIRE_ROCBLAS_ALL_ARCHS` — confirm names in
`docs/env-vars.md` / `feature_flags` before teaching defaults).

**Default posture:** measured enablement per arch; do not broad-enable on
RDNA because CDNA won.

---

## 9. Negative results (do not re-burn casually)

### Nontemporal weight loads on gfx1100

- Candidate `0532579` claimed small within-session gain; revert `34eb024`
  showed large decode regression vs clean baseline.
- Mechanism guess: nontemporal broke beneficial coalescing/cache behavior.
- **Revisit only** with a different cache-control mechanism + fresh-process
  proof.

### k2x32 wider-row lm_head

- `f670e16` — large slowdown from register pressure at huge M.
- Kept as experiment/opt-in history; not an auto path.

### Always-on hipGraph capture

- Series (`33b8861` / `5705a59` / …): default-on capture produced garbage
  from dangling kernargs in some forward paths.
- Opt-in only (`HIPFIRE_GRAPH`-class flags); not a free decode win.

### LDS-staged X on gate_up (ldsx)

- `gemm_gate_up_hfq4g256_wmma_ldsx.hip`, opt-in `HIPFIRE_GATE_UP_VARIANT=ldsx`.
- ISA looked cleaner; wall-clock prefill **regressed** on measured gfx1100
  shapes (`case-studies.md` §7).
- Revisit only with a different LDS role (e.g. weights not X) or on archs
  with different prefetch (gfx12).

### Capability-inherited perf variants

- Not a kernel micro-opt — a **dispatch** anti-pattern.
- `ldscoop` measured best on gfx115x, then selected via `is_rdna3()` and
  harmed gfx1100 (~14% DFlash class regression; fix narrative `24e4baa9`).
- Rule: capability ≠ perf allowlist. See `perf-arch-discipline.md`.

### MoE grouped dead ends (campaign-specific)

- Indexed `_k8` GEMV decode paths and some `m2` / `i8` grouped variants have
  been **falsified or negative** on specific campaigns (e.g. LFM/Qwen MoE
  prefill notes). Before retrying, read the owning plan/ledger and current
  `run_moe_prefill` Path-2 grouped WMMA substrate — do not resurrect from
  memory.

---

## 10. Barrier-free / nosync (LDS → direct global)

**When:** multi-wave workgroup stages through LDS + `__syncthreads()` inside
a loop, yet effective BW% is low — barriers serialize more than redundant
global loads cost.

**Pattern:** remove shared LDS stage and both barriers; each warp loads from
global. Accept redundant reads when they are cheaper than barrier rounds.

**Shipped-style anchors (verify wiring in your tree):**

| Kernel stem | Notes |
|---|---|
| `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync` | Grouped MoE barrier-free work |
| `gemm_gate_up_hfq4g256_wmma_ldscoop_nosync` | HFQ4 gate_up nosync sister |
| `gemm_gate_up_mq4g256_lloyd_wmma*_nosync` | Lloyd MQ4 family; some gfx1151 tags |
| `gemm_gate_up_mq3g256_lloyd_wmma*_nosync` | Lloyd MQ3 family |

Env gates seen in tree: `HIPFIRE_GATE_UP_NOSYNC`, variant selectors under
`HIPFIRE_GATE_UP_VARIANT`. **Unwired** nosync files may pass channel tests
without production dispatch — check forward/gemm call sites before claiming
product impact.

**Prerequisites:**

1. Low BW utilization relative to arch peak on the hot kernel.
2. LDS data is per-warp or redundant-cheap — not true cross-warp shares
   that need a different algorithm (e.g. some attention V pages).
3. Barrier is **inside** a loop, not a one-shot phase fence.

**Anti-pattern:** "barriers are free on single-wave blocks." A 1-wave
`__syncthreads` is largely a compiler fence; nosync variants there do not
apply the multi-wave lesson.

---

## Variant selection checklist (all levers)

Before enabling a sub-variant on an arch:

1. **Capability legal?** (`arch_caps` / ISA builtin exists)
2. **Measured on this arch atom or class?** If no → portable default only.
3. **Env kill-switch** for A/B when landing experimental paths.
4. **Atlas or rocprof proof** the intended symbol runs in the timed arm.
5. **Rejection log** if it loses — update this section or the commit body.

---

## After the lever

Return to `playbook.md` steps 5–9: claim-scoped correctness, fresh-process
measure, adjacent-arch story, promote or reject with hashes. Use Kernel Atlas
`suggest` / `task` / `eval` when you want structured experiment ledgers
([`docs/methodology/kernel-atlas.md`](../../../docs/methodology/kernel-atlas.md)) —
suggestions are queues, not predicted wins.
