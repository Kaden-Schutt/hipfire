<!--
SPDX-License-Identifier: Apache-2.0
hipfire — see LICENSE and NOTICE in the project root.
-->
# hipfire model support — source of truth

This is the **canonical model-support matrix** for hipfire. It tracks what is
*actually implemented and routed* per architecture, with the flagship
**qwen3.5** as the reference for full feature parity.

- Ground truth for arch IDs: `crates/hipfire-model/src/lib.rs` (`ARCH_ID_*`).
- Ground truth for routing/gating: `crates/hipfire-serving-core/src/generate.rs`
  and `load.rs` (where unsupported features are explicitly refused per `arch_id`).
- This table reflects **implemented + served** capability, not the forward-looking
  family roster (that lives in `docs/plans/2026-06-19-arch-roster-feature-matrix.md`).
- **Quant formats** and their weight/activation/calibration tradeoffs:
  `docs/quant-formats/opus-mqplus-eval-plan.md`. Canonical names: **Magnum**=MQ4 /
  **Magnum Plus**=MQ4+ · **Opus Quant**=OQ4 (W4A4) / OQ8 (W8A8) and **Opus
  Plus**=OQ4+ / OQ8+. First plus = clip-search, SmoothQuant/AWQ, or comparable
  activation-aware calibration; second plus = Hessian/LDLQ feedback. Opus =
  symmetric quantized weights.
- **Per-quant × per-GPU-arch kernel coverage** (which formats have tuned
  decode/prefill/WMMA kernels vs generic fallback): see "Kernel coverage" below.

**Last verified:** 2026-06-26 (against `chaingun`).

Legend: ✅ full · 🟡 partial / limited · ❌ not implemented (explicitly refused at load/serve) · — not applicable (e.g. expert sharding on a dense arch)

## Generated capability matrix

The tables below are generated from `docs/model-support.toml` (the single source
of truth, shared with `crates/hipfire-model/src/model_support_generated.rs`). Do
not hand-edit between the markers — run `cargo run -p hipfire-cli -- gen-model-support`.
The richer hand-maintained roster (microbatch / PP / EP columns) follows.

<!-- BEGIN GENERATED model-support (source: docs/model-support.toml — run `cargo run -p hipfire-cli -- gen-model-support`) -->

### Capability matrix (generated)

Machine-readable subset consumed by `arch_features` / admission. Edit `docs/model-support.toml`.

| Arch (arch_id) | Batched prefill | DFlash spec | MTP spec | KV quant | Vision |
|---|---|---|---|---|---|
| qwen3.5 (5, 6) | ✅ | ✅ | ✅ | full | 🟡 |
| deepseek4 (9) | ✅ | ❌ | 🟡 | fp32 | ❌ |
| minimax (10) | 🟡 | ❌ | 🟡 | fp32 | ❌ |
| lfm2-moe (11) | 🟡 | ❌ | ❌ | fp32 | ❌ |
| nemotron_h (14) | ✅ | ❌ | ❌ | fp32 | ❌ |
| mamba2 (15) | ✅ | ❌ | ❌ | no-kv | ❌ |
| zaya (16) | ❌ | ❌ | ❌ | none | ❌ |
| gemma3 (12) | ✅ | ❌ | ❌ | fp32+q8 | ❌ |
| gemma3-vl (13) | ✅ | ❌ | ❌ | fp32+q8 | ✅ |
| qwen2 (7) | ✅ | ❌ | ❌ | fp32 | ❌ |
| dots-ocr (8) | ✅ | ❌ | ❌ | fp32 | ✅ |
| llama (0, 1) | 🟡 | ❌ | ❌ | fp32 | ❌ |

### Quant formats (generated)

| Quant | Weight bits | Act bits | Status |
|---|---|---|---|
| bf16 (BF16 (unquantized)) | 16 | 16 | stable |
| q8 (Q8 (W8A16)) | 8 | 16 | stable |
| mq4 (Magnum / MQ4 (W4A16)) | 4 | 16 | stable |
| mq6 (Magnum / MQ6 (W6A16)) | 6 | 16 | stable |
| oq4 (Opus Quant / OQ4 (W4A4, int4 activations)) | 4 | 4 | opt-in |
| oq4+ (Opus Quant Plus / OQ4+ (W4A8, calibrated OQ4 weights)) | 4 | 8 | opt-in |
| oq8 (Opus Quant / OQ8 (W8A8, int8 activations)) | 8 | 8 | opt-in |
| oq8+ (Opus Quant Plus / OQ8+ (W8A8, calibrated OQ8 weights)) | 8 | 8 | opt-in |
| mq3 (Magnum / MQ3 (W3A16, mixed-precision only)) | 3 | 16 | experimental |

### Intentional gates (generated)

Per-quant overrides of an arch capability (admission consults these before green-lighting).

| Arch | Quant | Feature | Support | Note |
|---|---|---|---|---|
| 5 | oq4 | prefill | 🟡 | OQ4 W4A4 batched prefill is parity-gated / opt-in (iu4 WMMA path) |
| 5 | oq4+ | prefill | 🟡 | OQ4+ W4A8 prefill uses the int8-activation path; full quality admission is still gated |
| 5 | oq8 | prefill | 🟡 | OQ8 W8A8 route is experimental / parity-gated |
| 5 | oq8+ | prefill | 🟡 | OQ8+ calibrated W8A8 route shares OQ8 kernels; quality admission is still gated |
| 11 | oq4 | prefill | 🟡 | LFM2 OQ4 W4A4 prefill routes through iu4 WMMA; current evidence is 350M smoke/parity |
| 11 | oq4+ | prefill | 🟡 | LFM2 OQ4+ W4A8 prefill routes through int8 activation MMQ; full calibration/quality pending |
| 11 | oq8 | prefill | 🟡 | LFM2 OQ8 W8A8 prefill routes through iu8 WMMA; current evidence is 350M smoke/parity |
| 11 | oq8+ | prefill | 🟡 | LFM2 OQ8+ shares OQ8 runtime kernels; calibrated plus artifact quality is pending |
<!-- END GENERATED model-support -->

## Feature matrix vs flagship qwen3.5

| Arch (arch_id) | Decode | Batched prefill | Server microbatch | DFlash spec | MTP spec | KV quant modes | Lowered/superop pipeline | Layer shard (PP) | Expert shard (EP/TP) | Vision |
|---|---|---|---|---|---|---|---|---|---|---|
| **qwen3.5 dense / MoE (5 / 6)** — flagship | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ full menu | ✅ | ✅ | ✅ (MoE) | via qwen35-vl |
| qwen3.5-VL (5/6 + splice) | ✅ | ✅ | ✅ (family) | ✅ (family) | ✅ (family) | ✅ full | ✅ | ✅ (family) | ✅ (MoE) | ✅ |
| deepseek4-flash (9) | ✅ | ✅ (own kernels) | ❌ | ❌ | 🟡 native MTP head loads, not wired to spec-serving | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| minimax-m2 (10) | ✅ | ❌ per-token | ❌ | ❌ | 🟡 config plumbing only | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| lfm2-moe (11) | ✅ | 🟡 in-request batched prefill | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| nemotron_h (14) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | 🟡 SimpleAr seam | ❌ | ❌ (MoE, unsharded) | ❌ |
| gemma3 text (12) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | — (dense) | ❌ |
| gemma3-VL / medgemma (13) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | — (dense) | ✅ |
| qwen2 (7) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | — (dense) | ❌ |
| dots-ocr (8) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | 🟡 | ❌ | — (dense) | ✅ (OCR) |
| llama / mistral (0), qwen3-legacy (1) | ✅ | 🟡 (llama path) | ❌ | ❌ | ❌ | 🟡 fp32 | 🟡 | ❌ | — (dense) | ❌ |
| toy | test fixture only | — | — | — | — | — | — | — | — | — |

> **Server microbatch** = serving many *concurrent* request streams batched
> together (continuous batching), distinct from in-request *batched prefill* (one
> prompt, many tokens). It's a bespoke qwen35 subsystem (`Qwen35RequestSessionState`
> + `qwen35_decode_batch`), gated to arch 5/6; the grouped-MoE fused batch worker
> requires `arch_id=6`. All other archs run single-session AR only and emit
> `generate_batch_prefill_unsupported`.

## The headline

**qwen3.5 is the only arch that gets the full inference stack.** Three capability
tiers are hard-gated to the qwen3.5 family (`is_qwen35_family_arch_id`, arch 5/6),
and every other arch *explicitly errors* if asked for them:

- **DFlash spec-decode** (the big tok/s lever) — qwen35 only.
- **MTP spec-decode serving** — qwen35 only. (deepseek4 *has* a native MTP head
  that loads, but it isn't routed to the spec path yet; minimax has config
  plumbing only.)
- **CASK eviction + pipeline-parallel (pp>1)** — qwen35 only.
- **Full KV-quant menu** (q8 / asym3 / asym4 / FWHT / KVarN / hierarchical) —
  qwen35 only. gemma3 family adds q8; everyone else is fp32-only.

## Where each arch sits relative to flagship

- **Closest to flagship: deepseek4 (9)** — own batched prefill + decode kernels,
  lowered pipeline, native MTP head present. Missing: spec-decode serving, KV
  quant, CASK/PP.
- **gemma3-VL (13)** — strongest *multimodal* arch (vision grounding + batched
  prefill + q8 KV), but no spec-decode and no lowered pipeline.
- **minimax (10)** — solid validated decode + lowered pipeline, but still
  **per-token prefill** (slow long-context ingest) and fp32-only KV.
- **lfm2-moe (11)** — decode + lowered pipeline plus in-request batched prefill
  are routed. OQ4/OQ4+/OQ8/OQ8+ prefill is still partial because current evidence
  is 350M smoke/parity plus local sidecar experiments, not full admission.
- **nemotron_h (14)** — hybrid Mamba-2 / GQA / ReLU² / MoE AR path with model-level
  batched prefill and fp32 KV. Missing: server microbatch, DFlash/MTP, KV quant,
  CASK/PP/EP.
- **qwen2 (7) / dots-ocr (8)** — basic AR decode + batched prefill, fp32 KV, no
  fast paths.
- **llama / legacy (0 / 1)** — the original baseline path; functional decode, none
  of the modern levers.

## Multi-GPU & sharding (layer / expert / host)

Like the fast paths above, **all sharding is qwen3.5-family-only and single-host.**

| Sharding axis | Status | Scope | Where |
|---|---|---|---|
| **Layer sharding (pipeline-parallel, PP)** | ✅ implemented | qwen35 family only (5/6); `pp>1` explicitly refused for all other archs | `hipfire-runtime/src/multi_gpu.rs` (`Gpus`: layer bands, boundary copy, peer-access). `HIPFIRE_PP_LAYERS=48,16` sets per-device bands. Issue #58 Stage 7. |
| **Expert sharding (expert-parallel, EP)** | ✅ implemented | qwen35-**MoE** (arch 6) — each rank computes only its owned experts + shared expert on rank 0, then all-reduce combine | `hipfire-runtime/src/ep.rs`. `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` for peer-direct combine. |
| **Tensor sharding (TP: Q/KV heads, weight sub-ranges)** | ✅ implemented | qwen35-MoE A3B (Qwen3.5-30B-A3B); `expert_to_rank[e] = e % tp_size` (or contiguous); KV replicated when `tp_size > n_kv_heads` (TP=4 on A3B) | `hipfire-runtime/src/tp_shard.rs`; see `docs/plans/multi-gpu-tp-a3b.md`. |
| **Collectives backend** | ✅ implemented | single-node, multi-GPU | `hip-bridge/src/rccl.rs` — RCCL (AMD NCCL) FFI: `ncclCommInitAll` over local device ids; backs `Gpus::all_reduce_sum`. ~3× faster than a host-driven ring on gfx1201. |
| **Across hosts (multi-node / cross-node)** | ❌ **not implemented** | — | No `ncclCommInitRank` / `ncclUniqueId` / TCP bootstrap / `node_rank`. RCCL init is single-node only. Multi-host inference is unsupported. |

**Summary:** layer + expert + tensor sharding all work **across GPUs on one host**,
and only for the qwen3.5 family (TP/EP tuned specifically for the 30B-A3B MoE).
**Nothing shards across hosts** — there is no cross-node communicator or bootstrap.
HIP work is also single-threaded (one OS thread for all `Gpu::*` calls), so the
multi-GPU orchestrator drives devices from a single host thread.

## Biggest gaps to close (flagship-parity order)

1. **Batched prefill for minimax; broaden/gate LFM2 prefill** — minimax is still
   per-token, while LFM2 needs multi-model quality/perf admission beyond 350M OQ
   smoke/parity.
2. **KV quantization beyond qwen35** — only gemma3 has q8; no asym/FWHT kernels
   for any non-qwen35 arch.
3. **Spec-decode generalization** — DFlash/MTP are architecturally welded to
   qwen35; deepseek4's native MTP head is the cheapest candidate to wire next.
4. **Lowered pipeline for gemma3 / gemma3-vl** — the only "modern" archs still on
   the legacy forward path.
5. **Server microbatch + sharding generalization** — continuous batching, PP, and
   EP/TP are all welded to qwen35 (`Qwen35RequestSessionState`, `qwen35_decode_batch`,
   `*_qwen35` planners). A generic per-arch session/shard abstraction is needed
   before any other arch can microbatch or shard.
6. **Multi-host inference** — no cross-node communicator exists (RCCL is single-node
   `ncclCommInitAll`). Would need rank/uniqueId bootstrap + a transport.

## Kernel coverage (per-quant × per-GPU-arch tuned kernels)

Whether a quant format runs on a *tuned* kernel vs a *generic fallback* is set by
three things: the **decode** GEMV arm (`weight_gemv`), the **prefill** GEMM arm
(`weight_gemm`, batched), and whether it has **`_for_arch` selectors** (per-GPU-arch
WMMA + multi-batch mb2/mb4 variants — the deeply-tuned tier). Missing prefill arms
fall to `W8A8Ref` generic reference (instrumented by `warn_generic_once`, silence
with `HIPFIRE_WARN_GENERIC=0`).

Ground truth: `crates/hipfire-runtime/src/weights.rs` (dispatch arms),
`crates/rdna-compute/src/{dispatch,kernels,generic_warn}.rs`.

| Quant (DType) | Decode GEMV | Fused decode | Prefill GEMM (batched) | Per-arch tuned (`_for_arch`) | Primary model-archs |
|---|---|---|---|---|---|
| **MQ4G256-Lloyd** | ✅ | ✅ resid/swiglu | ✅ WMMA mb2/mb4 + fused QKV/gate-up | ✅ | qwen35 |
| **MQ3G256-Lloyd** | ✅ | ✅ | ✅ WMMA mb4 + fused QKV/gate-up | ✅ | qwen35 |
| **HFQ4G256** | ✅ | ✅ resid | ✅ `gemm_hfq4g256` | ✅ | qwen35, general |
| **HFQ3G256** | ✅ | ✅ resid | ✅ residual WMMA | ✅ | qwen35 |
| HFP4G32 | ✅ | — | 🟡 | ✅ | fp4 path |
| MQ4G256 (plain "magnum") | ✅ | ✅ resid/swiglu | 🟡 via Lloyd/generic | partial | qwen35, lfm2 experts |
| ParoQ4G128 | ✅ | ✅ resid/swiglu | ✅ paro gemm | ❌ | PARO variants |
| MQ8G256 | ✅ | ✅ prerotated | 🟡 | ❌ | q8 weights |
| MQ6G256 / HFQ6G256 | ✅ (HFQ6 indexed-MoE) | 🟡 | 🟡 | ❌ | lfm2 experts (mq6e) |
| MQ2G256(-Lloyd) | ✅ | ✅ (Lloyd) | 🟡 | ❌ | minimax MoE |
| **Oq4G256 (opus/OQ4/OQ4+)** | ✅ | ✅ qkvza/gate-up | ✅ W4A4 iu4 WMMA; OQ4+ W4A8 via int8-activation MMQ/F16 WMMA | partial | qwen35, lfm2 |
| **Oq8G256 (opus/OQ8/OQ8+)** | ✅ | ✅ qkvza/gate-up | ✅ W8A8 iu8 WMMA | partial | qwen35, lfm2 |
| MFP4G32 | ✅ | — | 🟡 | ❌ | microscaling fp4 |
| Q4F16 g32/g64 | ✅ | — | 🟡 | ❌ | gguf-ish |
| Qtip3G256 | 🟡 | — | ❌ | ❌ | trellis |
| W8A8Ref | generic ref (fallback) | — | generic | ❌ | fallback only |

**GPU-arch tuning:** the `_for_arch` tuned variants target **gfx1100 (RDNA3 dGPU)**
and **gfx1151 (RDNA3.5 Strix Halo APU)**. `arch_caps` also recognizes gfx1103,
gfx1152, RDNA4, CDNA3, gfx906, but those JIT the same kernel without a dedicated
tuned variant. wave32 (RDNA) vs wave64 (CDNA/GCN) is auto-selected.

**Takeaways:**
- The deeply-tuned tier (per-arch WMMA, mb2/mb4 batched prefill, fused QKV/gate-up/
  residual) is **MQ4/MQ3-Lloyd + HFQ4/HFQ3 + HFP4**, and lives in **qwen35** — another
  qwen35 concentration.
- **Prefill is still the coverage cliff:** many formats have decode GEMV, fewer
  have tuned batched prefill; the rest hit `W8A8Ref` generic or family-specific
  fallback paths.
- **Opus is no longer decode-only:** OQ4/OQ4+/OQ8/OQ8+ have batched prefill
  routes, but non-qwen35 admission is still smoke/parity gated.
- Non-qwen35 archs mostly run generic/decode paths: lfm2 experts (MQ4/MQ6, HFQ6
  indexed-MoE), minimax (MQ2-Lloyd MoE), gemma3 (bf16/Q8).

## Maintaining this file

This is a **living source of truth** — update it whenever arch support changes:

- Adding/removing an `arch_id` or routing a new feature to an arch → update the
  matrix row + the relevant prose.
- Re-verify against `generate.rs` / `load.rs` gating (search for
  `not supported on arch_id=`) and bump **Last verified**.
- Keep the forward-looking *roster* (planned families, audio/omni/diffusion) in
  `docs/plans/2026-06-19-arch-roster-feature-matrix.md`; this file is
  **shipped capability only**.
