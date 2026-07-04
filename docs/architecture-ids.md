# Architecture IDs

The `arch_id` is the stable numeric identity of a model architecture family. It
is stamped into the `.hfq` header at quantize time and drives dispatch in the
daemon, serving-core, and quantizer.

## Where the constants live

The canonical `ARCH_ID_*` constants are defined **once** in the capability
layer, `hipfire-arch-api`:

- Source of truth: `crates/hipfire-arch-api/src/lib.rs` (the `ARCH_ID_*` block,
  next to `ArchId`).
- `hipfire-model` re-exports them (`pub use hipfire_arch_api::{...}`), so every
  existing `hipfire_model::ARCH_ID_*` caller keeps working unchanged.
- `hipfire-arch-api` is a leaf, CPU-only crate. That lets the quantizer and
  other capability-layer consumers reference the ids without pulling in
  `hipfire-model` / runtime deps.

**Rule:** reference the named constant, never a bare `arch_id == N` literal or a
raw match arm (`"gemma3" => 12`). Import from `hipfire_arch_api` on the
capability layer (quantizer, arch crates) and from `hipfire_model` in the
runtime/daemon/serving path — both resolve to the same values.

The ids are stable (they are an on-disk header value) and gap-tolerant: 2, 3,
and 4 are historically retired and must not be reused.

## Registered ids

| id | constant | family label | serving crate | detection `arch_str` |
|----|----------|--------------|---------------|----------------------|
| 0  | `ARCH_ID_LLAMA_MISTRAL`      | llama        | `hipfire-arch-llama`    | `llama` (+ unknown-arch default) |
| 1  | `ARCH_ID_QWEN3_QWEN2_LEGACY` | llama        | `hipfire-arch-llama`    | `qwen3`, `qwen2` |
| 5  | `ARCH_ID_QWEN35_DENSE`       | qwen3.5      | `hipfire-arch-qwen35`   | `qwen3_5`, `qwen3_5_text` |
| 6  | `ARCH_ID_QWEN35_MOE`         | qwen3.5      | `hipfire-arch-qwen35`   | `qwen3_5_moe`, `qwen3_5_moe_text`, GGUF `qwen3moe` |
| 7  | `ARCH_ID_QWEN2`              | qwen2        | `hipfire-arch-qwen2`    | via `--arch-id 7` (auto-detect maps plain Qwen2 to 1) |
| 8  | `ARCH_ID_DOTS_OCR`          | dots-ocr     | `hipfire-arch-dots-ocr` | `dots_ocr` |
| 9  | `ARCH_ID_DEEPSEEK4_FLASH`   | deepseek4    | `hipfire-arch-deepseek4`| `deepseek_v4` |
| 10 | `ARCH_ID_MINIMAX_M2`        | minimax      | `hipfire-arch-minimax`  | `minimax_m2` |
| 11 | `ARCH_ID_LFM2_MOE`          | lfm2-moe     | `hipfire-arch-lfm2moe`  | `lfm2_moe`, `lfm2` |
| 12 | `ARCH_ID_GEMMA3_TEXT`       | gemma3       | `hipfire-arch-gemma3`   | `gemma3_text`, `gemma3` |
| 13 | `ARCH_ID_GEMMA3_VL`         | gemma3-vl    | `hipfire-arch-gemma3-vl`| `gemma3` + `vision_config` (auto-promoted from 12) |
| 14 | `ARCH_ID_NEMOTRON_H`        | nemotron_h   | `hipfire-arch-nemotron` | `nemotron_h` |
| 15 | `ARCH_ID_MAMBA2`            | mamba2       | `hipfire-arch-nemotron` | `mamba2` (+ `is_mamba2_config`) |
| 16 | `ARCH_ID_ZAYA`             | zaya         | `hipfire-arch-zaya`     | `zaya` |

The per-arch capability matrix (prefill / dflash / mtp / kv / vision support)
lives in `docs/model-support.toml`, keyed by these same ids.

### Tooling-only ids

These sidecar ids are defined locally in the quantizer binaries (not in the
capability layer) because they never reach runtime dispatch:

| id | constant | where | purpose |
|----|----------|-------|---------|
| 20 | `ARCH_ID_DFLASH_DRAFT`     | `hipfire-quantize` `bin/dflash_convert.rs` | DFlash draft-head sidecar artifact |
| 21 | `ARCH_ID_QWEN35_MTP_HEAD`  | `hipfire-quantize` `bin/mtp_extract.rs`    | Qwen3.5 MTP-head sidecar artifact |

## Adding a new architecture

1. Add `pub const ARCH_ID_<FAMILY>: u32 = <next free id>;` to
   `crates/hipfire-arch-api/src/lib.rs` and to the `pub use` re-export in
   `crates/hipfire-model/src/lib.rs`.
2. Register the family name + `model_arch_family` arm in `hipfire-model`.
3. Add the detection `arch_str` arm(s) in the quantizer
   (`hipfire-quantize/src/main.rs` and, if GGUF-importable,
   `gguf_import.rs`) using the new constant — not a bare literal.
4. Add an `[[arch]]` entry to `docs/model-support.toml`.
5. Never reuse a retired id (2, 3, 4).
