# Architecture ID registry

Canonical `arch_id` values used by `HfqFile::arch_id` and routed through
the daemon's `load_model` dispatcher. Each entry lists the canonical
trait-impl marker (returned by `Architecture::arch_id()`) and the HFQ
file ids it actually loads.

| arch_id | family | crate | notes |
|---|---|---|---|
| 0 | LLaMA / Mistral | `hipfire-arch-llama` | dense FA |
| 1 | plain Qwen3 / Qwen2 | `hipfire-arch-llama` | covered by llama's `config_from_hfq` branch |
| 5 | Qwen3.5 dense | `hipfire-arch-qwen35` | hybrid DeltaNet + dense FFN |
| 6 | Qwen3.5 / 3.6 MoE / A3B | `hipfire-arch-qwen35` | MoE expert routing |
| 7 | Qwen2 dense (standalone) | `hipfire-arch-qwen2` | rev 0 skeleton; full bring-up in `docs/plans/dots-ocr-prd.md` phase 1 |
| 8 | Qwen2-VL family (dots.ocr) | `hipfire-arch-dots-ocr` | vision tower + Strategy A E2E OCR validated 2026-05-21; daemon plumbing pending in `docs/plans/dots-ocr-prd.md` phase 3 |
| 9 | DeepSeek V4 Flash | `hipfire-arch-deepseek4` | Hyper-Connections, compressed-KV indexer, tail-only RoPE, raw SWA; optional `mtp.0.*` MTP layer. |
| 0xFF | toy / template | `hipfire-arch-toy` | never shipped; daemon refuses to dispatch |

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
