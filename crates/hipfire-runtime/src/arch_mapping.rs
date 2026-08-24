// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Single source of truth for `model_type` / `general.architecture` → `arch_id`.
//!
//! Why this table exists: three independent `model_type -> arch_id` maps
//! drifted (safetensors_source.rs, quantize/src/pipeline.rs,
//! quantize/src/pipeline_gguf.rs). One silently defaulted to llama (0) on
//! unknown input, another returned UNCLAIMED, the third lacked entries. This
//! module is the sole authority; the other sites call [`lookup_model_type`].
//!
//! The numeric ids are the HFQ `arch_id` stamped into the file header and
//! claimed by `Carrier::claims_arch_id`. Changing any assignment is a
//! wire-format / routing break, so keep them byte-identical.

/// Canonical `model_type` (HF) / `general.architecture` (GGUF) → `arch_id`.
///
/// Covers the union of every string previously recognised by the three
/// consumers. Strings absent from this table are *unknown* and must fail
/// closed (not silently become llama 0). The qwen2 entry is intentionally
/// `7` (Qwen2Carrier, loads Q/K/V biases); earlier `hipfire-quantize` builds
/// mapped it to `1` (LLaMA) which dropped those biases — that was a bug and
/// is corrected here. See `safetensors_source.rs` commit 9002d7f8b.
///
/// Sorted by `arch_id` then alphabetically for auditability.
pub const MODEL_TYPE_TO_ARCH_ID: &[(&str, u32)] = &[
    // arch 0 — llama family
    ("llama", 0),
    ("mistral", 0),
    // arch 1 — qwen3 (llama-family loader, no bias)
    ("qwen3", 1),
    // arch 5 — qwen3.5 dense (qwen3.5/qwen3.6 share the same loader, 5 dense / 6 MoE)
    ("qwen3.5", 5),
    ("qwen3.6", 5),
    ("qwen35", 5),
    ("qwen3_5", 5),
    ("qwen3_5_text", 5),
    ("qwen3_6", 5),
    // arch 6 — qwen3.5 MoE (explicit model_type strings; the safetensors path also
    // derives 6 from has_experts==true for the qwen3.5/3.6 family)
    ("qwen3_5_moe", 6),
    ("qwen3_5_moe_text", 6),
    ("qwen3moe", 6),
    // arch 7 — qwen2 (Qwen2Carrier, loads attention biases the llama loader drops)
    ("qwen2", 7),
    // arch 8 — dots.ocr
    ("dots_ocr", 8),
    // arch 9 — deepseek_v4
    ("deepseek_v4", 9),
    // arch 10 — minimax_m2
    ("minimax_m2", 10),
    // arch 11 — lfm2 (dense) + lfm2_moe (MoE); both route to hipfire-arch-lfm2moe/11
    ("lfm2", 11),
    ("lfm2_moe", 11),
    // arch 12 — cohere2_moe
    ("cohere2_moe", 12),
    // arch 13 — gemma4 family (dense + MoE unified; text decoder only). The
    // four strings mirror pipeline.rs; gguf's old `starts_with("gemma4")`
    // catch-all is intentionally replaced by this exact list so unknown
    // `gemma4*` variants fail closed instead of silently becoming 13.
    ("gemma4", 13),
    ("gemma4_text", 13),
    ("gemma4_unified", 13),
    ("gemma4_unified_text", 13),
    // arch 14 — muse_glimmer dense (52-layer + ViT)
    ("muse_glimmer", 14),
    ("muse_glimmer_text", 14),
    // arch 22 — gemma4 EAGLE drafter (single-block spec-decode head for arch 13)
    ("gemma4_unified_assistant", 22),
    // arch 23 — muse_glimmer DFlash drafter
    ("muse_glimmer_assistant", 23),
];

/// Look up an `arch_id` for a `model_type` / GGUF `general.architecture` string.
///
/// Returns `None` for unknown inputs — callers must fail closed (error
/// naming the unrecognised string and listing `supported_model_types()`).
/// The lookup is an exact string compare; no prefix or substring fallback,
/// so a typo does not silently route to an unrelated arch.
pub fn lookup_model_type(model_type: &str) -> Option<u32> {
    for (k, v) in MODEL_TYPE_TO_ARCH_ID {
        if *k == model_type {
            return Some(*v);
        }
    }
    None
}

/// Sorted list of every recognised `model_type` / architecture string, for
/// error messages. Computed from [`MODEL_TYPE_TO_ARCH_ID`] so it cannot drift.
pub fn supported_model_types() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = MODEL_TYPE_TO_ARCH_ID.iter().map(|(k, _)| *k).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Human-readable, comma-joined list for `eprintln!` diagnostics.
pub fn supported_model_types_display() -> String {
    supported_model_types().join(", ")
}
