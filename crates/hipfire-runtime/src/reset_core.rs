// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Architecture rollback / reset-core inventory for serve-hardening retry.
//!
//! Retry may only target routes whose reset core covers every residual that
//! would otherwise bleed across attempts (recurrent/conv, default-on EF,
//! KV/aux caches, graph warmup, drafter scratch, adaptive controller, host
//! sequence position, conversation tokens). Routes that still lack a complete
//! core remain **explicitly retry-ineligible** rather than faking a reset.
//!
//! This module is the architecture-side inventory only. Daemon/client reset
//! wire schema, checked acknowledgements, and the retry loop live elsewhere.

/// Whether a route's architecture reset-core is complete enough for retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryResetEligibility {
    /// Reset core covers every residual listed in [`ResetCoreCoverage`].
    Eligible,
    /// Incomplete or unproven — must not be retried until the gap is closed.
    Ineligible { reason: &'static str },
}

/// Per-architecture checklist of residual state a cold retry must clear.
///
/// Host fields (`seq_pos`, `conversation_len`) are owned by the daemon's
/// `LoadedModel` and are listed here so the inventory stays complete even when
/// the arch `SpecTarget::reset_recurrent` hook only touches GPU/recurrent
/// state. Graph invalidate is likewise often paired at the daemon call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetCoreCoverage {
    pub arch: &'static str,
    /// DeltaNet S/scale/conv, LFM conv rings, or "n/a pure attention".
    pub recurrent_or_conv: bool,
    /// Qwen3.5 default-on `s_ef_residual` (false + n/a for pure-attn arches).
    pub s_ef_residual: bool,
    /// KV rings / SWA / compressed / indexer caches zeroed or position-rewound.
    pub kv_or_aux_caches: bool,
    /// Captured HIP graph / AR warmup invalidated or forced re-warmup.
    pub graphs: bool,
    /// Spec/drafter scratch not retained across cold reset (or n/a).
    pub drafter: bool,
    /// Adaptive KV controller reset-or-sticky-poison handled (or n/a).
    pub adaptive: bool,
    /// Host `seq_pos` / conversation token buffer cleared by the reset path.
    pub host_position_and_conversation: bool,
    pub eligibility: RetryResetEligibility,
}

impl ResetCoreCoverage {
    #[inline]
    pub fn is_retry_eligible(self) -> bool {
        matches!(self.eligibility, RetryResetEligibility::Eligible)
    }
}

/// Static inventory of retry-candidate architectures (Qwen3.5, DeepSeek V4,
/// LLaMA/Qwen3 dense). Other SpecTarget impls stay ineligible until audited.
pub fn retry_candidate_reset_inventory() -> &'static [ResetCoreCoverage] {
    // Keep reasons short and greppable — tests pin eligibility, not prose.
    const QWEN35: ResetCoreCoverage = ResetCoreCoverage {
        arch: "qwen35",
        // ModelSlot::reset_state / reset_recurrent → DeltaNetState::reset
        // (S/scale/conv + s_ef_residual) + kv compact_offset.
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        // Daemon pairs invalidate_graph_state on cold reset / abort.
        graphs: true,
        // Spec scratch is request-local; DFlash state rebuilt on next prefill.
        drafter: true,
        // Daemon cold-reset resets non-poisoned kv_adaptive with the cache.
        adaptive: true,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Eligible,
    };
    const DEEPSEEK4: ResetCoreCoverage = ResetCoreCoverage {
        arch: "deepseek4",
        // No DeltaNet; host n_tokens + mtp_last_hidden + ar_forward_warmed_up.
        recurrent_or_conv: true,
        s_ef_residual: true, // n/a — no EF residual; treated covered
        // SpecTarget::reset_recurrent now calls zero_decode_caches.
        kv_or_aux_caches: true,
        // ar_forward_warmed_up=false; daemon pairs invalidate_graph_state.
        graphs: true,
        drafter: true,
        adaptive: true, // n/a for DS4 adaptive controller
        host_position_and_conversation: true,
        // Coverage rows stay complete; retry eligibility waits on GPU fault
        // injection / fresh-process snapshot parity (Task 14).
        eligibility: RetryResetEligibility::Ineligible {
            reason: "GPU fault parity pending",
        },
    };
    const LLAMA: ResetCoreCoverage = ResetCoreCoverage {
        arch: "llama",
        recurrent_or_conv: true, // pure attention — no recurrent residual
        s_ef_residual: true,     // n/a
        kv_or_aux_caches: true,  // compact_offset rewind; absolute KV overwrite
        graphs: true,
        drafter: true,
        adaptive: true,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "GPU fault parity pending",
        },
    };
    // Explicitly ineligible until a full audit lands (do not fake Eligible).
    const QWEN2: ResetCoreCoverage = ResetCoreCoverage {
        arch: "qwen2",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "qwen2 reset-core not fully audited for graphs/adaptive pairing",
        },
    };
    const COHERE2MOE: ResetCoreCoverage = ResetCoreCoverage {
        arch: "cohere2moe",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "cohere2moe not a serve-hardening retry candidate yet",
        },
    };
    const DOTS_OCR: ResetCoreCoverage = ResetCoreCoverage {
        arch: "dots-ocr",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "dots-ocr not a serve-hardening retry candidate yet",
        },
    };
    const MINIMAX: ResetCoreCoverage = ResetCoreCoverage {
        arch: "minimax",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "minimax not a serve-hardening retry candidate yet",
        },
    };
    const LFM2MOE: ResetCoreCoverage = ResetCoreCoverage {
        arch: "lfm2moe",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "lfm2moe not a serve-hardening retry candidate yet",
        },
    };
    &[
        QWEN35, DEEPSEEK4, LLAMA, QWEN2, COHERE2MOE, DOTS_OCR, MINIMAX, LFM2MOE,
    ]
}

/// Look up inventory row by arch id (`"qwen35"`, `"deepseek4"`, …).
pub fn reset_coverage_for(arch: &str) -> Option<ResetCoreCoverage> {
    retry_candidate_reset_inventory()
        .iter()
        .copied()
        .find(|c| c.arch == arch)
}

/// Single enforced retry-eligibility matrix lookup.
///
/// Unknown arches are ineligible. Only inventory rows marked
/// [`RetryResetEligibility::Eligible`] return true — never fake eligibility.
pub fn is_retry_reset_eligible(arch: &str) -> bool {
    reset_coverage_for(arch)
        .map(|c| c.is_retry_eligible())
        .unwrap_or(false)
}

/// Routes that accept `test_fault_after_prefill` under `serve-fault-inject`.
///
/// Only qwen35 has a local GPU fixture + full snapshot/hash surface today.
/// Other arches keep inventory coverage rows but are not fault-inject eligible.
pub fn fault_inject_eligible_routes(arch: &str) -> &'static [&'static str] {
    match arch {
        "qwen35" => &["qwen_ar", "qwen_dflash"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_qwen35_is_retry_eligible() {
        let c = reset_coverage_for("qwen35").expect("qwen35");
        assert!(c.is_retry_eligible(), "qwen35 must remain retry-eligible");
        assert!(c.recurrent_or_conv);
        assert!(c.s_ef_residual);
        assert!(c.kv_or_aux_caches);
        assert!(c.graphs);
        assert!(c.drafter);
        assert!(c.adaptive);
        assert!(c.host_position_and_conversation);
        assert!(is_retry_reset_eligible("qwen35"));
    }

    #[test]
    fn ds4_and_llama_coverage_rows_gpu_parity_pending() {
        for arch in ["deepseek4", "llama"] {
            let c = reset_coverage_for(arch).unwrap_or_else(|| panic!("missing {arch}"));
            assert!(
                !c.is_retry_eligible(),
                "{arch} must be ineligible until GPU fault parity lands"
            );
            // Coverage checklist stays complete so the gap is eligibility-only.
            assert!(c.recurrent_or_conv, "{arch} recurrent/conv");
            assert!(c.s_ef_residual, "{arch} s_ef (or n/a covered)");
            assert!(c.kv_or_aux_caches, "{arch} kv/aux");
            assert!(c.graphs, "{arch} graphs");
            assert!(c.drafter, "{arch} drafter");
            assert!(c.adaptive, "{arch} adaptive");
            assert!(
                c.host_position_and_conversation,
                "{arch} host pos/conversation"
            );
            match c.eligibility {
                RetryResetEligibility::Ineligible { reason } => {
                    assert_eq!(reason, "GPU fault parity pending");
                }
                RetryResetEligibility::Eligible => panic!("{arch} should not be Eligible"),
            }
            assert!(!is_retry_reset_eligible(arch));
        }
    }

    #[test]
    fn unaudited_routes_stay_explicitly_ineligible() {
        for arch in ["qwen2", "cohere2moe", "dots-ocr", "minimax", "lfm2moe"] {
            let c = reset_coverage_for(arch).expect(arch);
            assert!(
                !c.is_retry_eligible(),
                "{arch} must remain explicitly retry-ineligible"
            );
            match c.eligibility {
                RetryResetEligibility::Ineligible { reason } => {
                    assert!(!reason.is_empty(), "{arch} needs a non-empty reason");
                }
                RetryResetEligibility::Eligible => panic!("{arch} should not be Eligible"),
            }
        }
    }

    #[test]
    fn inventory_has_unique_arch_ids() {
        let mut seen = std::collections::BTreeSet::new();
        for c in retry_candidate_reset_inventory() {
            assert!(seen.insert(c.arch), "duplicate arch id {}", c.arch);
        }
    }

    #[test]
    fn single_matrix_helper_matches_inventory() {
        assert!(is_retry_reset_eligible("qwen35"));
        assert!(!is_retry_reset_eligible("deepseek4"));
        assert!(!is_retry_reset_eligible("llama"));
        assert!(!is_retry_reset_eligible("qwen2"));
        assert!(!is_retry_reset_eligible("unknown-arch"));
    }

    #[test]
    fn fault_inject_eligible_routes_are_qwen_only() {
        assert_eq!(
            fault_inject_eligible_routes("qwen35"),
            &["qwen_ar", "qwen_dflash"][..]
        );
        assert!(fault_inject_eligible_routes("deepseek4").is_empty());
        assert!(fault_inject_eligible_routes("llama").is_empty());
        assert!(fault_inject_eligible_routes("unknown").is_empty());
    }
}
