# Architecture ID registry

Canonical `arch_id` values used by `HfqFile::arch_id` and routed through
the daemon's `load_model` dispatcher. Each entry lists the canonical
trait-impl marker (returned by `Architecture::arch_id()`) and the HFQ
file ids it actually loads.

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

## Notes

- The trait doc at `crates/hipfire-runtime/src/arch.rs:81-89` calls out
  that one crate may cover multiple ids — e.g. `Llama::arch_id() == 0`
  but the LLaMA crate's `config_from_hfq` handles HFQ files with
  `arch_id ∈ {0, 1}` by branching on metadata.
- A future PR may migrate `arch_id = 1` from the LLaMA crate to
  `hipfire-arch-qwen2` once the latter is mature; until then, both
  arch_ids coexist with non-overlapping ownership.
- Daemon dispatch sites that branch on arch_id:
  `daemon.rs:672, 1081, 1163, 1448, 1494, 1719, 3158, 3516`. Any new
  arch_id needs explicit handling at the VL-gating sites
  (`:1494, :1719, :3158, :3516`) if it carries a vision tower.
