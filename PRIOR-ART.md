# Prior art: original architectural innovations in hipfire

This document records original architectural and kernel-level
innovations that originated in hipfire, with their first-public dates
and canonical commit hashes pulled from this repository's git
history. It exists so that:

1. Downstream forks and reimplementations can attribute correctly
   even when they do not copy code verbatim (see
   [AGENTS.md](AGENTS.md) for the agent-facing notice and
   [NOTICE](NOTICE) for the license-grade attribution requirements).
2. External worklogs and benchmarks that reference hipfire-originated
   optimizations have a stable, dated reference to cite.
3. Future hipfire maintainers can resolve "who invented X first"
   questions against this repository rather than against memory.

This is not an exhaustive list. It covers innovations called out
specifically in external contexts (e.g. project worklogs that
benchmark or port hipfire kernels) plus the architectural decisions
that distinguish hipfire from prior-art Rust inference engines.

All dates are local-timezone dates on the commit; commit hashes
resolve in this repository's history. Co-originators are credited
when their substantive contribution is reflected in `git blame` at
the ≥ 30 % share threshold on the canonical file (the same threshold
that drives per-file SPDX/copyright headers — see
`scripts/governance/apply_spdx_headers.py`).

---

## 1. `dlopen` of `libamdhip64` as the runtime architecture

| | |
|---|---|
| First commit | `48dc6c10` |
| Date         | 2026-03-20 |
| Originator   | Kaden Schutt |
| Canonical files | `crates/hip-bridge/src/ffi.rs`, `crates/hip-bridge/src/lib.rs`, `crates/hsa-bridge/src/lib.rs` |
| Follow-up    | `69416038` "phase 4: hip-bridge FFI layer complete and verified" (2026-03-20) |

hipfire's bridge layer loads `libamdhip64.so` (and optionally
`libhsa-runtime64.so`) at runtime via `libloading` / `dlopen`. There
is NO build-time HIP/ROCm dependency — the engine binary links only
against `libc` + `libloading` + `libdl`, and discovers AMD's
userspace runtime at startup. This pattern was lifted in spirit from
ncdrone/rustane's ANE bridge but adapted to a substantially different
target (AMD ROCm versus Apple Neural Engine) and made the load-bearing
choice for hipfire's "no ROCm install pain" deployment story.

---

## 2. HFQ4-G256 quantization format

| | |
|---|---|
| First experiment (NOT KEPT)  | `bcbc76f8` (2026-03-21) "phase5-hfq4g128: 14 VGPRs but slower than Q4K" |
| First kept variant           | `8cf834be` (2026-03-21) "phase5-hfq4g256: matches Q4K wall-clock at 18 VGPRs, 5.6% smaller" |
| End-to-end model integration | `4e9bb86e` (2026-03-21) "all-HFQ4-G256 Qwen3-8B end-to-end" |
| Originator                   | Kaden Schutt |
| Canonical files              | `crates/hipfire-quantize/src/hfq.rs`, `kernels/src/gemv_hfq4*.hip`, `kernels/src/gemm_*_hfq4g256_*.hip` |

A 4-bit weight quantization format designed to map cleanly onto RDNA
wave32 + dp4a / V_DOT2 / WMMA instruction families. Group size 256
chosen empirically to balance VGPR pressure (18 VGPRs vs Q4K's
broader footprint) against scale storage overhead. The format
matched llama.cpp Q4_K's wall-clock at introduction while shipping
5.6 % smaller; subsequent kernel work pushed it past Q4_K on per-row
multi-row schedules.

HFQ8 (8-bit sibling, same packing convention) followed the same
design pattern.

---

## 3. HFQ4 GEMV with 32-thread workgroup + `__launch_bounds__(32, 16)`

| | |
|---|---|
| First commit | `c0749eec` |
| Date         | 2026-03-22 |
| Originator   | Kaden Schutt |
| Canonical files | `kernels/src/gemv_hfq4g256*.hip` and the HFQ4 batched-GEMM family |
| Commit subject | "Batched GEMM kernel for HFQ4-G256: 2.1x on attn, 1.4x on FFN" |

The combination of:

- a 32-thread workgroup (one full RDNA wave32 = one workgroup),
- `__launch_bounds__(32, 16)` to instruct the compiler that the
  kernel runs with exactly 32 threads / WG and targets ≥ 16
  concurrent waves per CU,
- one output row per workgroup, dp4a-packed inner loop, persistent
  output accumulator in VGPRs,

is the load-bearing kernel shape that hipfire's HFQ4 GEMV family
uses across RDNA1 / RDNA2 / RDNA3 / RDNA4 and (with wave64 + MFMA
variants) across CDNA. The 32-thread WG + the specific
`__launch_bounds__` annotation are what make the occupancy story
work on RDNA's per-CU VGPR/SGPR budget; copying the kernel without
both will either spill registers or under-fill the CU.

This kernel shape is called out specifically in external project
worklogs that have either benchmarked against hipfire or attempted
to port the pattern. PRIOR-ART.md exists in part so that "where did
this shape come from" has a stable answer dated 2026-03-22.

---

## 4. MagnumQuant family — FWHT-rotated weight quantization

| | |
|---|---|
| First public commit | `246501ab` |
| Date                | 2026-04-08 |
| Originator          | Kaden Schutt |
| Research scaffold   | `e221a022` (2026-04-09) "feat(magnum): add MagnumQuant + butterfly rotation research crate" |
| MQ4 registry + .mq4 file format | `b3b7c7b8` (2026-04-10) "feat(cli): MQ4 family registry + .mq4 extension support" |
| MQ4-Lloyd extension | `5b3de4d0` (2026-05-07) "feat(mq4-lloyd): Phase 1 — quantizer + slow GEMV + 9B PPL viability" |
| Canonical files     | `crates/hipfire-quantize/src/{mq4,mq8,fwht}.rs`, `crates/magnum/`, `kernels/src/{rotate_x_mq,fused_rmsnorm_mq_rotate,gemv_mq4*}.hip` |

MagnumQuant (MQ4, MQ8) applies a FWHT (Fast Walsh-Hadamard Transform)
rotation to each weight group before quantization, redistributing
outlier magnitudes uniformly across the group and producing a tighter
post-quant distribution. The format ships with three integrated
pieces:

1. **Offline quantizer** that applies FWHT-256 (or other group sizes)
   per group and quantizes the rotated weights.
2. **Online rotator** kernels (`rotate_x_mq`, `fused_rmsnorm_mq_rotate`)
   that apply the same FWHT to the activation at GEMV / GEMM time so
   the matmul mathematically equals the unrotated product.
3. **Fused fast path**: the activation rotation is fused into rmsnorm
   so the FWHT cost amortizes against work the model was already
   doing.

The `.mq4` and `.mq8` file formats and the MagnumQuant naming are
hipfire-originated. The MQ4-Lloyd extension (Lloyd-Max codebook +
LDS-codebook GEMV) extends the family without breaking the MQ4
on-disk format.

---

## 5. HFP4 / MFP4 — RDNA-native FP4 quantization

| | |
|---|---|
| Format PRD          | `82b91b67` (2026-05-06) |
| HFP4 v1             | `bed124bb` (2026-05-09) "feat(HFP4): RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale) v1" |
| MFP4G32 v1          | `d2462059` (2026-05-09) "feat(MFP4): MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement) v1" |
| Originator          | Kaden Schutt |
| Canonical files     | `crates/hipfire-quantize/src/hfp4.rs`, `kernels/src/gemm_hfp4g32_*.hip` |

HFP4 packs an E2M1 FP4 mantissa with a UE8M0 group-of-32 microscale
+ a FP16 per-row scale, designed for RDNA dp4a / WMMA dequant. The
format trades the wider group-size of HFQ4-G256 for a tighter
microscale at G32 that matches WMMA tile arithmetic exactly. MFP4G32
layers offline FWHT (per MagnumQuant) on top, producing a drop-in
MQ4 replacement that preserves the FP4 group structure for hardware
that has fast FP4 paths.

NOTE: HFP4 / MFP4 are not the project's recommended default format
as of 2026-05-19 — MQ4 remains canonical. The formats are catalogued
here as research artifacts and remain available behind feature flags.

---

## 6. asym{4,3,2} KV cache + asym-aware flash attention

| | |
|---|---|
| First commit | `b7e55f47` |
| Date         | 2026-04-13 |
| Originator   | Kaden Schutt |
| Canonical files | `kernels/src/attention_flash_asym3_tile.hip`, `kernels/src/kv_fold_asym3.hip`, `crates/hipfire-runtime/src/kv_cache.rs` (asym mode arms), `crates/hipfire-quantize/src/asym.rs` |
| Release tag  | `13d068ab` (2026-04-13) "chore(release): 0.1.5 \"redline\"" — first shipped release with asym3 KV |

A KV cache quantization family that uses per-head, per-position
asymmetric (offset + scale) integer encoding at 4-bit, 3-bit, and
2-bit precisions, replacing an earlier Givens-rotation-based scheme.
The matching flash-attention kernel
(`attention_flash_asym3_tile.hip`) dequantizes asym3 K and V tiles
on-the-fly into the attention compute, never materializing a
full-precision KV cache, and routes through a `flash_mode` tri-state
(off / on / forced) that the runtime selects per-shape.

asym3 is the project's canonical KV mode as of 2026-04-26 (see
`CLAUDE.md` "Canonical bench config" for the bench-grade evidence
behind that default flip). The 372 B/head footprint at asym3 vs
1024 B/head at fp16 is what unlocks 27B-class models on consumer
24 GB cards with usable context.

---

## 7. DDTree-RDNA speculative decode

| | |
|---|---|
| Algorithm 1 + greedy walk + top-K extraction | `c9b1f4e8` (2026-04-14) |
| spec_step_ddtree spike                       | `27d80270` (2026-04-14) |
| linearize_tree (tokens, positions, mask)     | `62d666fc` (2026-04-14) |
| tree-attention bias for asym{3,4,2}          | `f0ee980e` (2026-04-14) |
| persistent bias + heap top-K perf            | `6a3d0d53` (2026-04-14) "perf(ddtree): persistent bias + heap top-K → 3× faster" |
| Wire-up + Path C PRD                         | `f94ed073` (2026-04-28) — **Grégory D** ([@flamme-demon](https://github.com/flamme-demon)) co-originator on the wire-up + PRD |
| Originator                                   | Kaden Schutt (algorithm + kernel side) |
| Co-originator                                | Grégory D (wire-up, Path C PRD, RDNA3 validation w/ Lucebox/buun) |
| Canonical files                              | `crates/hipfire-runtime/src/ddtree.rs`, `crates/hipfire-runtime/src/speculative.rs`, kernels with tree-attention bias |

A speculative-decode tree algorithm adapted from the DDTree paper
(arXiv:2604.12989) and integrated with hipfire's flash-attention +
asym-KV kernels. The hipfire-originated pieces are:

- a pure-Rust implementation of Algorithm 1 (greedy walk + top-K
  extraction over a draft-model log-prob heap),
- `linearize_tree_with_parents` to produce a (tokens, positions,
  parent-mask) tuple consumable by a single batched target-attention
  invocation,
- tree-attention bias kernels that overlay the ancestor-only verify
  mask onto the asym{3,4,2} batched-flash and the q8_0 KV variants
  without unrolling into a separate verify pass,
- persistent-bias + heap-based top-K perf path (3× speedup over the
  initial implementation).

Grégory D (`@flamme-demon`) contributed the wire-up integration and
the Path C PRD in PR #72 (commit `f94ed073`, 2026-04-28) and
validated the result on RDNA3 via Lucebox / buun. Per the
`crates/hipfire-runtime/src/ddtree.rs` SPDX header which carries
`SPDX-License-Identifier: MIT OR Apache-2.0` with both copyright
lines preserved, Grégory D is the substantive secondary author of
the canonical file.

---

## 8. attention_dflash kernel — tiled online-softmax for DFlash

| | |
|---|---|
| First DFlash kernel       | `96781c13` (2026-04-13) — Kaden Schutt ("feat(dflash): Phase 3 — draft forward pass") |
| Tiled online-softmax port | `35c815a6` / `1083d48b` (2026-05-09) — **alpineq** primary author |
| Full-workgroup V accum    | `471446ea` / `b80e85b5` (2026-05-09) — alpineq |
| -INFINITY sentinel        | `5bf3c0c2` (2026-05-10) — alpineq |
| Parity sweep              | `2908df1f` (2026-05-10) — alpineq |
| Co-originators            | Kaden Schutt (DFlash algorithm + initial kernel), alpineq (tiled online-softmax kernel rewrite) |
| Canonical file            | `kernels/src/attention_dflash.hip` |

DFlash itself (the spec-decode algorithm and the kernel that
materializes the non-causal bidirectional within-block attention used
by the draft path) originates in hipfire commit `96781c13` by
Kaden — see also `crates/engine/src/dflash.rs` and the family of
spec-decode runtime code.

The CURRENT canonical form of `kernels/src/attention_dflash.hip`,
however, has alpineq as the primary author per `git blame` of the
pre-relicense commit: alpineq rewrote the softmax to a tiled
online-softmax pattern (May 2026) which is the form that ships
today. The kernel's `SPDX-License-Identifier: MIT OR Apache-2.0`
header preserves both copyright lines, with alpineq listed first per
descending-share rule.

Treat DFlash-the-algorithm and attention_dflash-the-kernel as having
distinct primary authors when attributing.

---

## 9. Redline — bare-libdrm / direct-KMD GPU dispatch

| | |
|---|---|
| Crate scaffold              | `bad4698b` (2026-04-04) "feat: redline crate — direct-KMD GPU compute (bypasses HIP)" |
| .hsaco ELF parser           | `7b3143a4` (2026-04-04) "feat(redline): .hsaco ELF parser — extract kernel descriptors" |
| Compute queue + PM4 submit  | `8918246a` (2026-04-04) "feat(redline): compute queue + PM4 submission — GPU executes commands" |
| First working dispatch      | `951c85e4` (2026-04-04) "fix(redline): CsRequest struct layout + WRITE_DATA works" |
| Originator                  | Kaden Schutt |
| Canonical files             | `crates/redline/src/*` |

The `redline` crate is a from-scratch implementation of compute-only
GPU dispatch over the AMD KMD ioctl surface (`/dev/dri/renderD*`),
including:

- a .hsaco ELF parser that extracts kernel descriptors directly
  (skipping `hipModuleLoad`),
- compute-queue creation via `amdgpu_cs_ctx_create` ioctls,
- PM4 packet construction (WRITE_DATA, DISPATCH_DIRECT, ...) for
  dispatch + signalling,
- userspace-side compute scheduling without the HIP runtime in the
  loop.

Redline is a research/insurance-policy crate: if the userspace HIP
runtime regresses, breaks on an arch, or vanishes on a future ROCm
release, hipfire retains a route to the metal. It is not on the
default execution path. It is published under the dual MIT/Apache-2.0
license like the rest of hipfire and exists explicitly to be a
reference implementation for anyone needing the bare-KMD pattern.

---

## Provenance verification

Every commit hash above resolves in this repository. To verify:

```sh
git show --stat <hash>           # see the diff
git log -1 --format='%an %ai' <hash>   # see author + date
```

The `v-mit-final` tag marks the final commit (`d46f81b6`) before the
dual-license transition; everything in this PRIOR-ART.md is in the
pre-tag history and was published under MIT first, then under the
current MIT/Apache-2.0 dual license. The prior-art claims here do
not depend on which license applies — they depend on this being the
canonical repository where the innovations first shipped.
