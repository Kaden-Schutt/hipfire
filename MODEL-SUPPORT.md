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

**Last verified:** 2026-06-23 (against `chaingun`).

Legend: ✅ full · 🟡 partial / limited · ❌ not implemented (explicitly refused at load/serve)

## Feature matrix vs flagship qwen3.5

| Arch (arch_id) | Decode | Batched prefill | Server microbatch | DFlash spec | MTP spec | KV quant modes | Lowered/superop pipeline | Multi-GPU shard (PP / EP-TP) | Vision |
|---|---|---|---|---|---|---|---|---|---|
| **qwen3.5 dense / MoE (5 / 6)** — flagship | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ full menu | ✅ | ✅ (PP; EP/TP on MoE) | via qwen35-vl |
| qwen3.5-VL (5/6 + splice) | ✅ | ✅ | ✅ (family) | ✅ (family) | ✅ (family) | ✅ full | ✅ | ✅ (family) | ✅ |
| deepseek4-flash (9) | ✅ | ✅ (own kernels) | ❌ | ❌ | 🟡 native MTP head loads, not wired to spec-serving | 🟡 fp32 only | ✅ | ❌ | ❌ |
| minimax-m2 (10) | ✅ | ❌ per-token | ❌ | ❌ | 🟡 config plumbing only | 🟡 fp32 only | ✅ | ❌ | ❌ |
| lfm2-moe (11) | ✅ | ❌ per-token | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ |
| gemma3 text (12) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | ❌ |
| gemma3-VL / medgemma (13) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | ✅ |
| qwen2 (7) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ |
| dots-ocr (8) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | 🟡 | ❌ | ✅ (OCR) |
| llama / mistral (0), qwen3-legacy (1) | ✅ | 🟡 (llama path) | ❌ | ❌ | ❌ | 🟡 fp32 | 🟡 | ❌ | ❌ |
| toy | test fixture only | — | — | — | — | — | — | — | — |

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
- **minimax (10) / lfm2-moe (11)** — solid validated decode + lowered pipeline,
  but **per-token prefill** (no batching → slow long-context ingest) and fp32-only
  KV. Both are recent minimal-AR bring-ups.
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

1. **Batched prefill for minimax + lfm2** — both per-token today; long-prompt
   ingest is the visible cost.
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

## Maintaining this file

This is a **living source of truth** — update it whenever arch support changes:

- Adding/removing an `arch_id` or routing a new feature to an arch → update the
  matrix row + the relevant prose.
- Re-verify against `generate.rs` / `load.rs` gating (search for
  `not supported on arch_id=`) and bump **Last verified**.
- Keep the forward-looking *roster* (planned families, audio/omni/diffusion) in
  `docs/plans/2026-06-19-arch-roster-feature-matrix.md`; this file is
  **shipped capability only**.
