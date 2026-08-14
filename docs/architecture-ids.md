# Architecture ID registry

Canonical `arch_id` values used by HFQ headers (`HfqFile::arch_id`), safetensors
directory derivation (`derive_arch_id` in
`crates/hipfire-runtime/src/safetensors_source.rs`), and the loader carrier
registry (`crates/hipfire-loader`). Trait markers come from
`Architecture::arch_id()` in each arch crate.

One crate may cover multiple runtime ids. The trait’s canonical marker is the
family default; the file/dir id on the loaded source is authoritative for
dispatch.

**Routing availability ≠ product admission.** A primary id, source derivation, or
carrier match means the loader can route that artifact — not that any model is
product-admitted. The sole admission owner is [`admissions.yml`](admissions.yml)
(schema v2; exactly one evidence-bound record). Product-use and admission claims
remain fail-closed for every route outside that exact row; source-derived routing capability is independent.

## Primary model ids

| arch_id | Family | Crate | Carrier | Notes |
|---:|---|---|---|---|
| 0 | LLaMA / Mistral | `hipfire-arch-llama` | `LlamaCarrier` | Dense FA. Trait marker `Llama::arch_id() == 0`. |
| 1 | plain Qwen3 | `hipfire-arch-llama` | `LlamaCarrier` | Same crate; `config_from_hfq` branches on metadata. Dir `model_type`/`architectures` qwen3 → 1. |
| 5 | Qwen3.5 / 3.6 / 3.8 dense | `hipfire-arch-qwen35` | `Qwen35Carrier` | Hybrid DeltaNet + dense FFN. Trait marker returns 5. Optional VL via `hipfire-arch-qwen35-vl` under the same ids. |
| 6 | Qwen3.5 / 3.6 / 3.8 MoE (A3B) | `hipfire-arch-qwen35` | `Qwen35Carrier` | MoE expert routing; same crate as 5. |
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
| 20 | DFlash draft artifact (generic) | `hipfire-quantize` `dflash_convert` (`ARCH_ID_DFLASH_DRAFT`); loader draft open checks `draft_hfq.arch_id == 20` | Generic Qwen-family diffusion draft (fc/hidden_norm/norm naming, `dflash` metadata). Loaded as a draft alongside a compatible target, not as a standalone serve model. |
| 21 | Qwen3.5 MTP head | `hipfire-quantize` `mtp_extract` (`ARCH_ID_QWEN35_MTP_HEAD`); consumed as `.mq4-mtp` trailer / `.mtp` sidecar on qwen35 trunks | Attached onto arch 5/6 loads (`LoadedModel.qwen35_mtp_head`), not a carrier `arch_id`. |
| 22 | Gemma4 EAGLE draft | `hipfire-arch-gemma4` `DRAFTER_ARCH_ID` | `model_type gemma4_unified_assistant` — hidden-state-consuming EAGLE (pre/post projections, KV-shared layers). Loaded via `params.drafter` on arch 13. |
| 23 | Muse Glimmer DFlash draft | `hipfire-arch-muse-glimmer` `GLIMMER_DRAFTER_ARCH_ID` | `model_type muse_glimmer_assistant` — 5-layer block-diffusion draft (encoder.fc/output_norm_enc, block 16, target_layer_ids [1,13,25,37,49], mask 201818). Loaded via `HIPFIRE_DFLASH_DRAFT` on arch 14. NOT a `dflash_convert` 20 artifact (different tensor naming: encoder.fc vs fc, separate `config` envelope). Quantizer auto-maps `muse_glimmer_assistant` → 23 (`hipfire-quantize/src/main.rs:8607`, `hipfire-runtime/src/safetensors_source.rs:271`); HFQ on disk is `arch 23`. |
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
