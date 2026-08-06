# Architecture ID registry

Canonical `arch_id` values used by HFQ headers (`HfqFile::arch_id`), safetensors
directory derivation (`derive_arch_id` in
`crates/hipfire-runtime/src/safetensors_source.rs`), and the loader carrier
registry (`crates/hipfire-loader`). Trait markers come from
`Architecture::arch_id()` in each arch crate.

The capability entries below are policy targets, not claims that future
CAP-001 behavior is currently enforced. CAP-001 will enforce refusal of
planned cells and dense-EP normalization before mesh, device, allocation,
or collective creation. Until then, current loader behavior remains the
existing architecture-specific gates characterized by PAR-001.

| arch_id | family | crate | Single | PP | TP | EP | notes |
|---|---|---|---|---|---|---|---|
| 0 | LLaMA / Mistral | `hipfire-arch-llama` | implemented | implemented code; HW-003 pending | partial; `has_qk_norm=true` / Qwen3-family metadata eligible; non-qk-norm LLaMA/Mistral refused; HW-006 pending for eligible artifacts | normalized-to-single(CAP-001) | dense FA |
| 1 | plain Qwen3 | `hipfire-arch-llama` | implemented | implemented code; HW-003 pending | implemented code; HW-006 pending | normalized-to-single(CAP-001) | covered by llama's `config_from_hfq` branch |
| 5 | Qwen3.5 dense | `hipfire-arch-qwen35` | implemented | partial (GEN-001; HW-004 pending) | planned (AXIS-002; HW-007 pending) | normalized-to-single(CAP-001) | hybrid DeltaNet + dense FFN |
| 5 (VL extension) | Qwen3.5-VL | `hipfire-arch-qwen35-vl` | implemented | planned (AXIS-004; HW-012 pending) | planned (AXIS-004; HW-013 pending) | normalized-to-single(CAP-001) | current `pp > 1` HFQ path bypasses VL detection and does not explicitly refuse at load; CAP-001 must make the admission error explicit; no PP+VL support claim |
| 6 | Qwen3.5 / 3.6 MoE / A3B | `hipfire-arch-qwen35` | implemented | partial (GEN-001; HW-004 pending) | planned (AXIS-002; HW-007 pending) | planned (AXIS-002; HW-011 pending) | MoE expert routing |
| 7 | Qwen2 dense (standalone) | `hipfire-arch-qwen2` | implemented | planned (AXIS-001; HW-003 pending) | planned (AXIS-001; HW-006 pending) | normalized-to-single(CAP-001) | rev 0 skeleton; full bring-up in `docs/plans/dots-ocr-prd.md` phase 1 |
| 8 | Qwen2-VL family (dots.ocr) | `hipfire-arch-dots-ocr` | implemented | planned (AXIS-004; HW-012 pending) | planned (AXIS-004; HW-013 pending) | normalized-to-single(CAP-001) | vision tower + Strategy A E2E OCR validated 2026-05-21; daemon plumbing pending in `docs/plans/dots-ocr-prd.md` phase 3 |
| 9 | DeepSeek V4 Flash | `hipfire-arch-deepseek4` | implemented | planned (AXIS-003; HW-008 pending) | planned (AXIS-003; HW-008 pending) | implemented code; HW-001 pending | Hyper-Connections, compressed-KV indexer, tail-only RoPE, raw SWA; optional `mtp.0.*` MTP layer. |
| 10 | MiniMax-M2 | `hipfire-arch-minimax` | implemented | planned (AXIS-003; HW-009 pending) | planned (AXIS-003; HW-009 pending) | implemented code; HW-002 pending | Mixtral-style MoE: GQA + per-layer QK-norm + partial rotate_half RoPE + sigmoid+bias 256-expert routing. |
| 11 | LFM2.5 dense | `hipfire-arch-lfm2moe` | currently refused; planned admission AXIS-003 | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | normalized-to-single(CAP-001) | hybrid short-conv + GQA-attn; dense SwiGLU; tied embeddings. |
| 11 | LFM2.5-MoE | `hipfire-arch-lfm2moe` | implemented | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | hybrid short-conv + GQA-attn; top-4 MoE (sigmoid+expert_bias); tied embeddings. |
| 12 | Cohere2-MoE (North-Mini-Code) | `hipfire-arch-cohere2moe` | implemented | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | parallel block (single mean-centered Cohere2LayerNorm feeds attn+ffn); interleaved sliding(RoPE)/global(NoPE) GQA; sigmoid 128-expert MoE (`norm_topk_prob=false`, no bias, no shared); dense layer-0 (`first_k_dense_replace=1`); tied embeddings. |
| 0xFF | toy / template | `hipfire-arch-toy` | out-of-scope | out-of-scope | out-of-scope | out-of-scope | never shipped; daemon refuses to dispatch |

One crate may cover multiple runtime ids. The trait’s canonical marker is the
family default; the file/dir id on the loaded source is authoritative for
dispatch.

**Routing availability ≠ product admission.** A primary id, source derivation, or
carrier match means the loader can route that artifact — not that any model is
product-admitted. The sole admission owner is [`admissions.yml`](admissions.yml)
(schema v2; exactly one evidence-bound record). Product-use and admission claims
remain fail-closed for every route outside that exact row; source-derived routing capability is independent.

- The trait doc at `crates/hipfire-runtime/src/arch.rs:81-89` calls out
  that one crate may cover multiple ids — e.g. `Llama::arch_id() == 0`
  but the LLaMA crate's `config_from_hfq` handles HFQ files with
  `arch_id ∈ {0, 1}` by branching on metadata.
- TP×EP composition is explicitly refused (COMP-001): `--tp` and `--ep` cannot both exceed one. Reopening it requires a concrete deployment requirement and separate implementation/physical-validation task.
- A future PR may migrate `arch_id = 1` from the LLaMA crate to
  `hipfire-arch-qwen2` once the latter is mature; until then, both
  arch_ids coexist with non-overlapping ownership.
- Daemon dispatch sites that branch on arch_id:
  `daemon.rs:672, 1081, 1163, 1448, 1494, 1719, 3158, 3516`. Any new
  arch_id needs explicit handling at the VL-gating sites
  (`:1494, :1719, :3158, :3516`) if it carries a vision tower.

## Primary model ids

| arch_id | Family | Crate | Carrier | Notes |
|---:|---|---|---|---|
| 0 | LLaMA / Mistral | `hipfire-arch-llama` | `LlamaCarrier` | Dense FA. Trait marker `Llama::arch_id() == 0`. |
| 1 | plain Qwen3 | `hipfire-arch-llama` | `LlamaCarrier` | Same crate; `config_from_hfq` branches on metadata. Dir `model_type`/`architectures` qwen3 → 1. |
| 5 | Qwen3.5 / 3.6 dense | `hipfire-arch-qwen35` | `Qwen35Carrier` | Hybrid DeltaNet + dense FFN. Trait marker returns 5. Optional VL via `hipfire-arch-qwen35-vl` under the same ids. |
| 6 | Qwen3.5 / 3.6 MoE (A3B) | `hipfire-arch-qwen35` | `Qwen35Carrier` | MoE expert routing; same crate as 5. |
| 7 | Qwen2 dense | `hipfire-arch-qwen2` | `Qwen2Carrier` | Standalone path so Q/K/V `attention_bias` loads (llama dir path would drop them). Dir `qwen2` → 7. |
| 8 | dots.ocr (Qwen2-VL family) | `hipfire-arch-dots-ocr` | `DotsOcrCarrier` | Vision tower + Qwen2 text decoder fields on `LoadedModel`. |
| 9 | DeepSeek V4 Flash | `hipfire-arch-deepseek4` | `Deepseek4Carrier` | Hyper-Connections, compressed-KV indexer, tail-only RoPE, raw SWA; optional in-trunk / sidecar MTP; EP via `load_model_ep`. |
| 10 | MiniMax-M2 | `hipfire-arch-minimax` | `MinimaxCarrier` | Mixtral-style MoE (GQA, per-layer QK-norm, partial rotate_half RoPE, sigmoid+bias expert route). EP via `load_model_ep`. |
| 11 | LFM2.5 / LFM2.5-MoE | `hipfire-arch-lfm2moe` | `Lfm2MoeCarrier` | Hybrid short-conv + GQA; dense SwiGLU and/or top-4 MoE; tied embeddings. Dir `lfm2` / `lfm2_moe` → 11. |
| 12 | Cohere2-MoE (North-Mini-Code) | `hipfire-arch-cohere2moe` | `Cohere2MoeCarrier` | Parallel block, interleaved sliding(RoPE)/global(NoPE) GQA, sigmoid MoE, dense layer-0, tied embeddings. |

Ids **2–4** are unused in the carrier registry (not claimed). Do not invent
assignments without a carrier + source mapping.

## Sidecar / reserved ids

These are **not** primary `load_model` trunk targets. Carriers must not claim
them for ordinary dispatch.

| arch_id | Role | Where defined | Notes |
|---:|---|---|---|
| 20 | DFlash draft artifact | `hipfire-quantize` `dflash_convert` (`ARCH_ID_DFLASH_DRAFT`); loader draft open checks `draft_hfq.arch_id == 20` | Loaded as a draft alongside a compatible target, not as a standalone serve model via the primary registry. |
| 21 | Qwen3.5 MTP head | `hipfire-quantize` `mtp_extract` (`ARCH_ID_QWEN35_MTP_HEAD`); consumed as `.mq4-mtp` trailer / `.mtp` sidecar on qwen35 trunks | Attached onto arch 5/6 loads (`LoadedModel.qwen35_mtp_head`), not a carrier `arch_id`. |
| 0xFF | Toy / template | `hipfire-arch-toy` | Never ship; daemon must not dispatch. No current carrier claims it. Registry disjointness tests include `0xFF` and assert at most one claimer — they do **not** assert zero claimers / hard-reserved. |
| `u32::MAX` | Unclaimed dir sentinel | `safetensors_source::UNCLAIMED_ARCH_ID` | Emitted for unrecognized `model_type`; no carrier matches → fail closed. |

## Source namespaces

| Namespace | Origin | Examples |
|---|---|---|
| HFQ header | `HfqFile::arch_id` written at quantize/pack time | Primary table above; sidecars 20/21 |
| Safetensors dir | `derive_arch_id(&config)` | `llama`/`mistral`→0, `qwen3`→1, `qwen3.5`/`qwen3.6` (+experts→6 else 5), `qwen2`→7, `dots_ocr`→8, `deepseek_v4`→9, `minimax_m2`→10, `lfm2`/`lfm2_moe`→11, `cohere2_moe`→12 |

`Carrier::claims_arch_id(arch_id, is_dir)` may distinguish the two namespaces.
Today’s carriers are disjoint on bare id for the ids they claim; registry unit
tests sweep `0..=64` plus reserved 20 and `0xFF` for both `is_dir` values and
require at most one claimer (disjointness only — not a zero-claimer guarantee
for `0xFF`).

## Dispatch sites (maintainers)

| Concern | Location |
|---|---|
| Load (single arch-dispatch) | `hipfire_loader::load_model` → `REGISTRY` probe |
| EP load | `hipfire_loader::load_model_ep` (9, 10) |
| Spec target / emitter by id | `hipfire_loader::carrier_for` |
| Generate short-circuits | `crates/hipfire-runtime/examples/daemon.rs` `generate` (EP first; then 7, 9, 11, 12, 10, 8; else qwen35/llama body) |
| Trait contract | `crates/hipfire-runtime/src/arch.rs` |

Vision-capable loads: Qwen3.5-VL rides ids **5/6** with optional vision
weights; dots.ocr is id **8**. New vision arches need explicit generate and
HTTP image gating in the daemon, not only a carrier.

## Related

- Crate / data-flow map: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Model catalog (tags ≠ admission): [`MODELS.md`](MODELS.md)
- Docs ownership / truth states: [`INDEX.md`](INDEX.md)
