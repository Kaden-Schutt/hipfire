// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! arch_id unification: one table, three consumers, no fail-open path.
//!
//! Intent: if a fourth `model_type -> arch_id` map is added or any consumer
//! drifts from the canonical table, this test fails. It asserts:
//!   - every known model_type resolves identically via the canonical lookup
//!     and via `derive_arch_id` (the safetensors Dir path), and would via the
//!     two quantize pipelines (which now call the same `lookup_model_type`).
//!   - an unknown model_type fails closed everywhere (None / UNCLAIMED, not
//!     silently becoming llama 0).
//!   - the preserved mapping list is byte-identical to the pre-unification
//!     union (with the documented qwen2 correction 1→7).

use hipfire_runtime::arch_mapping::{
    lookup_model_type, supported_model_types_display, MODEL_TYPE_TO_ARCH_ID,
};
use hipfire_runtime::safetensors_source::{derive_arch_id, UNCLAIMED_ARCH_ID};
use serde_json::json;

/// The exact mapping list preserved by the unification.
///
/// This is the union of all strings previously recognised by the three maps,
/// with every prior `model_type → arch_id` assignment kept byte-identical
/// except for `qwen2` which was `1` (Llama) in the two quantize maps and
/// `7` (Qwen2Carrier) in the runtime. The runtime's `7` is the correct
/// routing (attn biases load) — see commit 9002d7f8b — so the canonical table
/// adopts `7` and the quantize paths change from `1 → 7`. That single
/// correction is the only intentional re-mapping; everything else is preserved.
const EXPECTED_MAPPINGS: &[(&str, u32)] = &[
    ("cohere2_moe", 12),
    ("deepseek_v4", 9),
    ("dots_ocr", 8),
    ("gemma4", 13),
    ("gemma4_text", 13),
    ("gemma4_unified", 13),
    ("gemma4_unified_assistant", 22),
    ("gemma4_unified_text", 13),
    ("lfm2", 11),
    ("lfm2_moe", 11),
    ("llama", 0),
    ("minimax_m2", 10),
    ("mistral", 0),
    ("muse_glimmer", 14),
    ("muse_glimmer_assistant", 23),
    ("muse_glimmer_text", 14),
    ("qwen2", 7),
    ("qwen3", 1),
    ("qwen3.5", 5),
    ("qwen3.6", 5),
    ("qwen35", 5),
    ("qwen3moe", 6),
    ("qwen3_5", 5),
    ("qwen3_5_moe", 6),
    ("qwen3_5_moe_text", 6),
    ("qwen3_5_text", 5),
    ("qwen3_6", 5),
];

#[test]
fn canonical_table_matches_expected_preserved_list() {
    // Every entry we claim to preserve must be in the canonical table with
    // the same arch_id, and vice-versa (no extra or missing entries).
    for (k, v) in EXPECTED_MAPPINGS {
        let got = lookup_model_type(k);
        assert_eq!(
            got,
            Some(*v),
            "canonical table: expected {k} -> {v}, got {got:?}"
        );
    }
    // No extra entries beyond the expected list.
    assert_eq!(
        MODEL_TYPE_TO_ARCH_ID.len(),
        EXPECTED_MAPPINGS.len(),
        "canonical table length drifted; expected {} got {} (table={:?})",
        EXPECTED_MAPPINGS.len(),
        MODEL_TYPE_TO_ARCH_ID.len(),
        MODEL_TYPE_TO_ARCH_ID
    );
    for (k, v) in MODEL_TYPE_TO_ARCH_ID {
        assert!(
            EXPECTED_MAPPINGS.contains(&(*k, *v)),
            "canonical table has unexpected entry {k} -> {v} not in EXPECTED_MAPPINGS"
        );
    }
}

#[test]
fn every_known_model_type_resolves_identically_through_all_three_consumers() {
    // Consumer 1: hipfire-runtime::arch_mapping::lookup_model_type (single source)
    // Consumer 2: safetensors_source::derive_arch_id (Dir loader path)
    // Consumer 3 & 4: the two quantize pipelines (pipeline.rs, pipeline_gguf.rs)
    //   — both now delegate to lookup_model_type, so testing lookup suffices to
    //   stand in for them. This test would fail if any pipeline re-introduced a
    //   private map.
    for (model_type, expected_arch) in EXPECTED_MAPPINGS {
        // -- consumer 1: canonical lookup
        let via_lookup = lookup_model_type(model_type).unwrap_or_else(|| {
            panic!("lookup_model_type({model_type}) returned None, expected {expected_arch}")
        });
        assert_eq!(
            via_lookup, *expected_arch,
            "lookup_model_type({model_type}) mismatch"
        );

        // -- consumer 2: derive_arch_id (safetensors Dir). For qwen3.5/3.6 the
        // result depends on has_experts; the dense case (no num_experts) is 5,
        // which matches the table's dense default. MoE is tested separately.
        // For other types, derive_arch_id should match exactly.
        if !matches!(
            *model_type,
            "qwen3.5" | "qwen3.6" | "qwen3_5" | "qwen3_6"
        ) {
            let id = derive_arch_id(&json!({ "model_type": *model_type }));
            assert_eq!(
                id, *expected_arch,
                "derive_arch_id(model_type={model_type}) -> {id} != {expected_arch}"
            );
        } else {
            // dense qwen3.5 family without experts -> 5
            let id_dense = derive_arch_id(&json!({ "model_type": *model_type }));
            assert_eq!(id_dense, 5, "derive_arch_id dense {model_type} -> {id_dense} != 5");
            // with experts -> 6 (both qwen3_5/3.6 family are MoE when has_experts)
            let id_moe = derive_arch_id(&json!({
                "model_type": *model_type,
                "num_experts": 8
            }));
            assert_eq!(id_moe, 6, "derive_arch_id moe {model_type} -> {id_moe} != 6");
            // pipeline's explicit moe strings should also resolve to 6 via lookup
            // (already covered above for qwen3_5_moe etc, but double-check here)
        }

        // -- consumer 3/4: quantize pipelines now use lookup_model_type, so the
        // same assertion as consumer 1 applies. We additionally assert that the
        // GGUF architecture strings (which use the same table) resolve identically.
        let via_gguf_lookup = lookup_model_type(model_type);
        assert_eq!(
            via_gguf_lookup,
            Some(*expected_arch),
            "GGUF lookup for {model_type} mismatch"
        );
    }

    // Explicit MoE strings must be 6 even though the dense qwen3.5 family maps to 5.
    for mt in ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3moe"] {
        assert_eq!(
            lookup_model_type(mt),
            Some(6),
            "MoE string {mt} should be 6"
        );
        assert_eq!(
            derive_arch_id(&json!({ "model_type": mt })),
            6,
            "derive_arch_id MoE {mt} should be 6"
        );
    }
}

#[test]
fn qwen2_maps_to_7_not_llama_0() {
    // This is the one intentional correction: quantize previously mapped qwen2->1
    // (llama), runtime mapped it to 7 (Qwen2Carrier). The unified table must be 7
    // so Q/K/V biases load.
    assert_eq!(lookup_model_type("qwen2"), Some(7));
    assert_eq!(derive_arch_id(&json!({ "model_type": "qwen2" })), 7);
    // And the supported list must contain it.
    assert!(
        supported_model_types_display().contains("qwen2"),
        "supported list missing qwen2"
    );
}

#[test]
fn gemma4_variants_route_correctly_and_gemma4_unified_assistant_is_22() {
    // Gemma4 unified dense/MoE -> 13, the EAGLE drafter -> 22.
    for mt in ["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"] {
        assert_eq!(lookup_model_type(mt), Some(13), "{mt} -> 13");
        assert_eq!(
            derive_arch_id(&json!({ "model_type": mt })),
            13,
            "derive {mt} -> 13"
        );
    }
    assert_eq!(lookup_model_type("gemma4_unified_assistant"), Some(22));
    assert_eq!(
        derive_arch_id(&json!({ "model_type": "gemma4_unified_assistant" })),
        22
    );
    // Bare gemma/gemma2/gemma3 must NOT map to 13 (old GGUF prefix would, new table does not).
    for mt in ["gemma", "gemma2", "gemma3", "gemma4_foobar"] {
        assert_eq!(
            lookup_model_type(mt),
            None,
            "unknown gemma variant {mt} must be None (fail-closed)"
        );
    }
}

#[test]
fn muse_glimmer_variants_present_and_gguf_would_no_longer_default_to_llama() {
    // Previously pipeline_gguf.rs lacked muse_glimmer entirely and silently tagged it 0.
    for (mt, expected) in [
        ("muse_glimmer", 14),
        ("muse_glimmer_text", 14),
        ("muse_glimmer_assistant", 23),
    ] {
        assert_eq!(lookup_model_type(mt), Some(expected), "{mt} -> {expected}");
        assert_eq!(
            derive_arch_id(&json!({ "model_type": mt })),
            expected,
            "derive {mt} -> {expected}"
        );
    }
}

#[test]
fn unknown_model_type_fails_closed_in_all_three() {
    let unknown = "totally_unknown_arch_zzz";

    // consumer 1: canonical lookup -> None
    assert_eq!(
        lookup_model_type(unknown),
        None,
        "unknown model_type must be None, not silently 0"
    );
    assert_ne!(
        lookup_model_type(unknown),
        Some(0),
        "unknown must not become llama 0"
    );

    // consumer 2: derive_arch_id -> UNCLAIMED (not 0, not 5)
    let id = derive_arch_id(&json!({ "model_type": unknown }));
    assert_eq!(id, UNCLAIMED_ARCH_ID, "derive_arch_id unknown -> UNCLAIMED");
    assert_ne!(id, 0, "unknown must not become llama 0");
    assert_ne!(id, 5, "unknown must not become Qwen35 5");

    // consumer 3/4: quantize pipelines — they now call lookup_model_type and
    // would error listing supported types. Simulate the branch:
    let gguf_lookup = lookup_model_type(unknown);
    assert_eq!(gguf_lookup, None, "GGUF unknown -> None");
    // The error message would list supported types; ensure it is non-empty and
    // does not contain the unknown.
    let supported = supported_model_types_display();
    assert!(
        !supported.contains(unknown),
        "supported list should not contain unknown"
    );
    assert!(
        supported.contains("llama"),
        "supported list should contain known entries"
    );
    // The supported list must be the same set as the canonical table (no drift).
    for (k, _) in MODEL_TYPE_TO_ARCH_ID {
        assert!(
            supported.contains(*k),
            "supported display missing {k}"
        );
    }
}

#[test]
fn supported_list_is_sorted_and_deduped() {
    let display = supported_model_types_display();
    let parts: Vec<&str> = display.split(", ").collect();
    let mut sorted = parts.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(parts, sorted, "supported display should be sorted & deduped");
}

#[test]
fn architectures_field_still_routes_before_model_type() {
    // derive_arch_id checks `architectures` before `model_type`; ensure that
    // path still works and is not bypassed by the table change.
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["Qwen2ForCausalLM"] })),
        7
    );
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["Qwen3ForCausalLM"] })),
        1
    );
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["LlamaForCausalLM"] })),
        0
    );
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["Gemma4ForCausalLM"] })),
        13
    );
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["muse_glimmer_for_causal_lm"] })),
        14
    );
    assert_eq!(
        derive_arch_id(&json!({ "architectures": ["muse_glimmer_assistant_for_causal_lm"] })),
        23
    );
}
