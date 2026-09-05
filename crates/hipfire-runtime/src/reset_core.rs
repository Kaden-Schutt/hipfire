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
    const GEMMA4: ResetCoreCoverage = ResetCoreCoverage {
        arch: "gemma4",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "gemma4 not a serve-hardening retry candidate yet",
        },
    };
    const MUSE_GLIMMER: ResetCoreCoverage = ResetCoreCoverage {
        arch: "muse_glimmer",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "muse_glimmer not a serve-hardening retry candidate yet",
        },
    };
    const MAPLE: ResetCoreCoverage = ResetCoreCoverage {
        arch: "maple",
        recurrent_or_conv: true,
        s_ef_residual: true,
        kv_or_aux_caches: true,
        graphs: false,
        drafter: true,
        adaptive: false,
        host_position_and_conversation: true,
        eligibility: RetryResetEligibility::Ineligible {
            reason: "maple not a serve-hardening retry candidate yet",
        },
    };
    &[
        QWEN35,
        DEEPSEEK4,
        LLAMA,
        QWEN2,
        COHERE2MOE,
        DOTS_OCR,
        MINIMAX,
        LFM2MOE,
        GEMMA4,
        MUSE_GLIMMER,
        MAPLE,
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
/// Sticky GPU-fault poison latch (serve-hardening, complements the retry
/// inventory above: retry eligibility assumes a *live* context; these codes
/// mean the context itself is dead and no reset can revive it).
///
/// `hipErrorIllegalMemoryAccess` (700) and `hipErrorLaunchFailure` (719) are
/// sticky: once reported, every subsequent device op on the context fails the
/// same way until process (or device) teardown. The tree documents both
/// behaviours in the wild — 700 surfacing late after freed-memory reads
/// (`rdna-compute/src/graph.rs`), 719 from bound-violating launches
/// (`rdna-compute/src/gemv.rs`) — and a gate run showed the same 719
/// repeating across requests (`reset_recurrent` memsets on a dead context),
/// each burning a full prefill that could not succeed.
///
/// Arch hooks call [`note_hip_error`] (or [`note_hip_result`]) on error paths
/// that observe a device op result; the daemon checks [`gpu_poison`] before
/// dispatching generate and fails fast ("unload/reload" cannot revive a
/// sticky-dead primary context — only process restart does — so the latch is
/// never cleared automatically; [`clear_gpu_poison`] exists for tests and
/// explicit operator recovery paths).
pub const STICKY_GPU_FAULT_ILLEGAL_ACCESS: u32 = 700;
/// See [`STICKY_GPU_FAULT_ILLEGAL_ACCESS`]; 719 is `hipErrorLaunchFailure`.
pub const STICKY_GPU_FAULT_LAUNCH_FAILURE: u32 = 719;

/// A latched sticky fault: the device code and the arch call site that first
/// observed it (first observation wins — later reports are knock-on effects
/// of the same dead context, not independent faults).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPoison {
    /// Raw HIP error code ([`STICKY_GPU_FAULT_ILLEGAL_ACCESS`] / [`STICKY_GPU_FAULT_LAUNCH_FAILURE`]).
    pub code: u32,
    /// Arch call-site label that first observed the fault (e.g.
    /// `"qwen35::DeltaNetState::reset"`).
    pub site: &'static str,
}

static GPU_POISON: std::sync::Mutex<Option<GpuPoison>> = std::sync::Mutex::new(None);

/// Whether a raw HIP error code is sticky (context-dead) as opposed to an
/// ordinary per-call failure (OOM, invalid value, unsupported) that a reset
/// or retry may legitimately recover from.
#[inline]
pub fn is_sticky_gpu_fault(code: u32) -> bool {
    code == STICKY_GPU_FAULT_ILLEGAL_ACCESS || code == STICKY_GPU_FAULT_LAUNCH_FAILURE
}

/// Latch [`GpuPoison`] when `err` carries a sticky code. Returns true when
/// latched. First observation wins; later sticky reports from the same dead
/// context are ignored (but still returned to the caller for its own handling).
pub fn note_hip_error(err: &hip_bridge::HipError, site: &'static str) -> bool {
    if !is_sticky_gpu_fault(err.code) {
        return false;
    }
    let mut guard = GPU_POISON.lock().unwrap_or_else(|error| error.into_inner());
    if guard.is_none() {
        *guard = Some(GpuPoison {
            code: err.code,
            site,
        });
    }
    true
}

/// [`note_hip_error`] over a full result: passes `Ok` through untouched and
/// latches-then-returns `Err` unchanged, so call sites wrap a single
/// expression with no behaviour change on any path.
pub fn note_hip_result<T>(
    result: hip_bridge::HipResult<T>,
    site: &'static str,
) -> hip_bridge::HipResult<T> {
    if let Err(err) = &result {
        note_hip_error(err, site);
    }
    result
}

/// Copy out the latched sticky fault, if any. Does not clear: the context
/// stays dead until process teardown, so clearing on read would let the next
/// request burn another doomed prefill.
pub fn gpu_poison() -> Option<GpuPoison> {
    *GPU_POISON.lock().unwrap_or_else(|error| error.into_inner())
}

/// Clear the latch. Only for tests and explicit operator recovery paths that
/// re-establish a live context out-of-band (model unload/reload does NOT:
/// it frees buffers without resetting the primary context). Never called
/// automatically.
pub fn clear_gpu_poison() {
    *GPU_POISON.lock().unwrap_or_else(|error| error.into_inner()) = None;
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
        for arch in [
            "qwen2",
            "cohere2moe",
            "dots-ocr",
            "minimax",
            "lfm2moe",
            "gemma4",
            "muse_glimmer",
        ] {
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
    fn gemma4_and_muse_glimmer_rows_are_explicitly_ineligible() {
        for arch in ["gemma4", "muse_glimmer"] {
            let c = reset_coverage_for(arch).unwrap_or_else(|| panic!("missing {arch}"));
            assert!(!c.is_retry_eligible(), "{arch} must be ineligible");
            // Must be explicitly present, not silently missing (missing would also be
            // ineligible, so we pin that the row exists and carries a reason).
            match c.eligibility {
                RetryResetEligibility::Ineligible { reason } => {
                    assert!(
                        reason.contains(arch) || reason.contains("retry"),
                        "{arch} reason should name the audit it lacks, got: {reason}"
                    );
                }
                RetryResetEligibility::Eligible => panic!("{arch} must not be Eligible"),
            }
            assert!(!is_retry_reset_eligible(arch));
            // Coverage checklist for unaudited arches: graphs/adaptive false, rest true.
            assert!(c.recurrent_or_conv);
            assert!(c.s_ef_residual);
            assert!(c.kv_or_aux_caches);
            assert!(!c.graphs, "{arch} graphs not yet audited");
            assert!(c.drafter);
            assert!(!c.adaptive, "{arch} adaptive not yet audited");
            assert!(c.host_position_and_conversation);
        }
    }

    #[test]
    fn unknown_architectures_remain_ineligible() {
        for arch in [
            "unknown-arch",
            "toy",
            "gemma4_unified_assistant",
            "muse_glimmer_assistant",
        ] {
            assert!(
                reset_coverage_for(arch).is_none(),
                "{arch} must have no inventory row"
            );
            assert!(
                !is_retry_reset_eligible(arch),
                "{arch} must be retry-ineligible"
            );
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
        assert!(!is_retry_reset_eligible("gemma4"));
        assert!(!is_retry_reset_eligible("muse_glimmer"));
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
        assert!(fault_inject_eligible_routes("gemma4").is_empty());
        assert!(fault_inject_eligible_routes("muse_glimmer").is_empty());
        assert!(fault_inject_eligible_routes("unknown").is_empty());
    }

    #[test]
    fn inventory_covers_every_arch_in_arch_mapping() {
        // Makes a missing inventory row a loud build failure instead of silent
        // ineligibility. Derives the expected set of arch_keys from the
        // canonical `MODEL_TYPE_TO_ARCH_ID` table so a new architecture that
        // adds a `model_type -> arch_id` entry without a curated
        // `ResetCoreCoverage` row fails this test rather than being quietly
        // ineligible. Reachable without a crate cycle because `arch_mapping`
        // lives in this crate; the loader's registry would be the alternative
        // but would require a dev-dependency and is not needed here.
        use crate::arch_mapping::MODEL_TYPE_TO_ARCH_ID;
        use std::collections::{BTreeMap, BTreeSet};

        fn arch_key_for_id(id: u32) -> Option<&'static str> {
            match id {
                0 | 1 => Some("llama"),
                5 | 6 => Some("qwen35"),
                7 => Some("qwen2"),
                8 => Some("dots-ocr"),
                9 => Some("deepseek4"),
                10 => Some("minimax"),
                11 => Some("lfm2moe"),
                12 => Some("cohere2moe"),
                13 => Some("gemma4"),
                14 => Some("muse_glimmer"),
                15 => Some("maple"),
                // Drafter sidecars (22, 23) are intentionally not retry
                // candidates and have no inventory row.
                22 | 23 => None,
                _ => None,
            }
        }

        let drafters: BTreeSet<u32> = [22, 23].into_iter().collect();

        let mut expected_keys = BTreeSet::new();
        let mut id_to_key = BTreeMap::new();
        for (_, id) in MODEL_TYPE_TO_ARCH_ID.iter() {
            if drafters.contains(id) {
                continue;
            }
            let key = arch_key_for_id(*id).unwrap_or_else(|| {
                panic!(
                    "arch_mapping contains arch_id {id} with no arch_key mapping — add it to arch_key_for_id and to retry_candidate_reset_inventory"
                )
            });
            expected_keys.insert(key);
            id_to_key.insert(*id, key);
        }

        let inventory_keys: BTreeSet<&str> = retry_candidate_reset_inventory()
            .iter()
            .map(|c| c.arch)
            .collect();

        let missing: Vec<&&str> = expected_keys.difference(&inventory_keys).collect();
        assert!(
            missing.is_empty(),
            "missing ResetCoreCoverage row for arch_keys {missing:?} — derived from MODEL_TYPE_TO_ARCH_ID {id_to_key:?}; add an explicit Ineligible row rather than staying silently ineligible"
        );

        let extra: Vec<&&str> = inventory_keys.difference(&expected_keys).collect();
        assert!(
            extra.is_empty(),
            "inventory contains arch_keys {extra:?} with no MODEL_TYPE_TO_ARCH_ID entry — remove or reconcile"
        );

        for key in &expected_keys {
            assert!(
                reset_coverage_for(key).is_some(),
                "reset_coverage_for({key}) missing — arch_key spelling must match inventory's `arch` field"
            );
        }

        // Unknowns must stay ineligible (fail-closed).
        assert!(!is_retry_reset_eligible("unknown-arch"));
        assert!(reset_coverage_for("unknown-arch").is_none());
    }

    #[test]
    fn sticky_poison_latch_lifecycle() {
        // Single test (not one per case): the latch is process-global, so
        // sequential steps here avoid cross-test races by construction.
        clear_gpu_poison();
        assert_eq!(gpu_poison(), None);
        // Classification: only 700/719 latch.
        assert!(is_sticky_gpu_fault(700));
        assert!(is_sticky_gpu_fault(719));
        for code in [0, 1, 2, 11, 200, 217, 999] {
            assert!(!is_sticky_gpu_fault(code), "code {code} must not latch");
        }
        // Non-sticky errors never latch and pass through untouched.
        let oom: hip_bridge::HipResult<()> = Err(hip_bridge::HipError::new(2, "hipOutOfMemory"));
        let oom = note_hip_result(oom, "test::oom");
        assert!(oom.is_err());
        assert_eq!(gpu_poison(), None);
        // First sticky observation wins and preserves the error value.
        let first: hip_bridge::HipResult<()> =
            Err(hip_bridge::HipError::new(719, "hipMemcpy H2D offset"));
        let first = note_hip_result(first, "test::first");
        assert!(first.is_err());
        assert_eq!(
            gpu_poison(),
            Some(GpuPoison {
                code: 719,
                site: "test::first"
            })
        );
        let second: hip_bridge::HipResult<()> =
            Err(hip_bridge::HipError::new(700, "hipMemsetAsync"));
        let second = note_hip_result(second, "test::second");
        assert!(second.is_err());
        assert_eq!(
            gpu_poison(),
            Some(GpuPoison {
                code: 719,
                site: "test::first"
            }),
            "knock-on reports must not overwrite the root-cause latch"
        );
        clear_gpu_poison();
        let ok: hip_bridge::HipResult<u32> = Ok(7);
        assert_eq!(note_hip_result(ok, "test::ok").unwrap(), 7);
        assert_eq!(gpu_poison(), None);
        clear_gpu_poison();
    }
}
