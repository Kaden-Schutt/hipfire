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

**Last verified:** 2026-06-22 (against `chaingun`).

Legend: ✅ full · 🟡 partial / limited · ❌ not implemented (explicitly refused at load/serve)

## Feature matrix vs flagship qwen3.5

| Arch (arch_id) | Decode | Batched prefill | DFlash spec | MTP spec | KV quant modes | Lowered/superop pipeline | CASK evict / PP | Vision |
|---|---|---|---|---|---|---|---|---|
| **qwen3.5 dense / MoE (5 / 6)** — flagship | ✅ | ✅ | ✅ | ✅ | ✅ full menu | ✅ | ✅ | via qwen35-vl |
| qwen3.5-VL (5/6 + splice) | ✅ | ✅ | ✅ (family) | ✅ (family) | ✅ full | ✅ | ✅ | ✅ |
| deepseek4-flash (9) | ✅ | ✅ (own kernels) | ❌ | 🟡 native MTP head loads, not wired to spec-serving | 🟡 fp32 only | ✅ | ❌ | ❌ |
| minimax-m2 (10) | ✅ | ❌ per-token | ❌ | 🟡 config plumbing only | 🟡 fp32 only | ✅ | ❌ | ❌ |
| lfm2-moe (11) | ✅ | ❌ per-token | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ |
| gemma3 text (12) | ✅ | ✅ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | ❌ |
| gemma3-VL / medgemma (13) | ✅ | ✅ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ✅ | ✅ |
| qwen2 (7) | ✅ | ✅ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ |
| dots-ocr (8) | ✅ | ✅ | ❌ | ❌ | 🟡 fp32 only | 🟡 | ❌ | ✅ (OCR) |
| llama / mistral (0), qwen3-legacy (1) | ✅ | 🟡 (llama path) | ❌ | ❌ | 🟡 fp32 | 🟡 | ❌ | ❌ |
| toy | test fixture only | — | — | — | — | — | — | — |

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

## Biggest gaps to close (flagship-parity order)

1. **Batched prefill for minimax + lfm2** — both per-token today; long-prompt
   ingest is the visible cost.
2. **KV quantization beyond qwen35** — only gemma3 has q8; no asym/FWHT kernels
   for any non-qwen35 arch.
3. **Spec-decode generalization** — DFlash/MTP are architecturally welded to
   qwen35; deepseek4's native MTP head is the cheapest candidate to wire next.
4. **Lowered pipeline for gemma3 / gemma3-vl** — the only "modern" archs still on
   the legacy forward path.

## Maintaining this file

This is a **living source of truth** — update it whenever arch support changes:

- Adding/removing an `arch_id` or routing a new feature to an arch → update the
  matrix row + the relevant prose.
- Re-verify against `generate.rs` / `load.rs` gating (search for
  `not supported on arch_id=`) and bump **Last verified**.
- Keep the forward-looking *roster* (planned families, audio/omni/diffusion) in
  `docs/plans/2026-06-19-arch-roster-feature-matrix.md`; this file is
  **shipped capability only**.
