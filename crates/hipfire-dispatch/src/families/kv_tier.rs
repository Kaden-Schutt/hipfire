// SPDX-License-Identifier: MIT OR Apache-2.0
//! KV-tier paired plan type (Phase 0.3). Derived once per attention step from
//! the live KV-cache state. Carries both the write key and attend key together
//! so they can never diverge (the #30-class drift guard).

use crate::types::KernelKey;

/// GPU-free scalar inputs for tier derivation. NO runtime types (avoids the
/// dep cycle — `hipfire-dispatch` cannot depend on `hipfire-runtime`).
/// The arch-side code constructs this from a `&KvCache` at each attention step.
#[derive(Clone, Copy, Debug)]
pub struct KvTierInputs {
    pub quant_asym4: bool,
    pub quant_asym3: bool,
    pub quant_asym2: bool,
    pub quant_q8: bool,
    pub quant_fwht: bool,
    pub v_mode_bits: i32,
    // q8 use_flash heuristic inputs (moved from qwen35.rs:12885)
    pub pos: usize,
    pub flash_mode: usize,
    pub capture_mode: bool,
}

/// Paired KV write + attend plan. Derived from `KvTierInputs` by
/// `KvTierPlan::derive`. Both keys are produced by a single derivation so
/// they always agree on tier.
#[derive(Clone, Copy, Debug)]
pub struct KvTierPlan {
    pub write_key: KernelKey,
    pub attend_key: KernelKey,
    /// Shared sub-plan: V-quant mode kernarg (8=Q8, 2/3/4=Lloyd-V).
    pub v_mode_bits: i32,
    /// Shared sub-plan: needs givens_cos/sin buffers.
    pub uses_givens: bool,
}

impl KvTierPlan {
    /// Derive the paired (write, attend) key plan from scalar KV-cache state.
    /// GPU-free, unit-testable. The q8 `use_flash` heuristic is folded in:
    /// it selects between `AttnFlashQ8_0` and `AttnQ8_0Kv`.
    ///
    /// Panics (debug_assert) if the inputs are contradictory (e.g. two
    /// quant flags set simultaneously).
    pub fn derive(inputs: KvTierInputs) -> Self {
        let KvTierInputs {
            quant_asym4,
            quant_asym3,
            quant_asym2,
            quant_q8,
            quant_fwht,
            v_mode_bits,
            pos,
            flash_mode,
            capture_mode,
        } = inputs;

        // At most one quant tier flag should be set.
        debug_assert!(
            [quant_asym4, quant_asym3, quant_asym2, quant_q8]
                .iter()
                .filter(|&&b| b)
                .count() <= 1,
            "at most one KV quant tier flag should be set"
        );

        let (write_key, attend_key, uses_givens) = if quant_asym4 {
            if quant_fwht {
                (KernelKey::KvWriteAsym4Fwht, KernelKey::AttnFlashAsym4Fwht, true)
            } else {
                (KernelKey::KvWriteAsym4, KernelKey::AttnFlashAsym4, true)
            }
        } else if quant_asym3 {
            if quant_fwht {
                (KernelKey::KvWriteAsym3Fwht, KernelKey::AttnFlashAsym3Fwht, true)
            } else {
                (KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym3, true)
            }
        } else if quant_asym2 {
            if quant_fwht {
                (KernelKey::KvWriteAsym2Fwht, KernelKey::AttnFlashAsym2Fwht, true)
            } else {
                (KernelKey::KvWriteAsym2, KernelKey::AttnFlashAsym2, true)
            }
        } else if quant_q8 {
            let use_flash = capture_mode
                || flash_mode == 2
                || (flash_mode == 1 && pos + 1 >= 2048)
                || pos + 1 > 15000;
            let attend = if use_flash {
                KernelKey::AttnFlashQ8_0
            } else {
                KernelKey::AttnQ8_0Kv
            };
            (KernelKey::KvWriteQ8_0, attend, false)
        } else {
            // F32 fallback
            (KernelKey::KvWriteF32, KernelKey::AttnF32, false)
        };

        // Phase 0.3 drift guard: write and attend keys must agree on tier.
        debug_assert!(
            tiers_match(write_key, attend_key),
            "KvTierPlan tier mismatch: write={:?}, attend={:?}",
            write_key,
            attend_key,
        );

        Self {
            write_key,
            attend_key,
            v_mode_bits,
            uses_givens,
        }
    }
}

/// Check that the write and attend keys agree on tier. This is the Phase 0.3
/// #30-class drift guard — a first-class assert on a first-class type.
fn tiers_match(write: KernelKey, attend: KernelKey) -> bool {
    use KernelKey::*;
    matches!(
        (write, attend),
        // asym4
        (KvWriteAsym4, AttnFlashAsym4)
        | (KvWriteAsym4Fwht, AttnFlashAsym4Fwht)
        // asym3
        | (KvWriteAsym3, AttnFlashAsym3)
        | (KvWriteAsym3Fwht, AttnFlashAsym3Fwht)
        // asym2
        | (KvWriteAsym2, AttnFlashAsym2)
        | (KvWriteAsym2Fwht, AttnFlashAsym2Fwht)
        // q8
        | (KvWriteQ8_0, AttnFlashQ8_0)
        | (KvWriteQ8_0, AttnQ8_0Kv)
        // f32
        | (KvWriteF32, AttnF32)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build default inputs (all-false = F32 tier).
    fn default_inputs() -> KvTierInputs {
        KvTierInputs {
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_q8: false,
            quant_fwht: false,
            v_mode_bits: 8,
            pos: 0,
            flash_mode: 0,
            capture_mode: false,
        }
    }

    // ── Tier derivation tests ──

    #[test]
    fn f32_tier() {
        let plan = KvTierPlan::derive(default_inputs());
        assert_eq!(plan.write_key, KernelKey::KvWriteF32);
        assert_eq!(plan.attend_key, KernelKey::AttnF32);
        assert!(!plan.uses_givens);
    }

    #[test]
    fn q8_non_flash_short_context() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 100,
            flash_mode: 0,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0Kv);
    }

    #[test]
    fn q8_flash_mode_2() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 10,
            flash_mode: 2,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_flash_long_context() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 2047, // pos + 1 = 2048 >= 2048
            flash_mode: 1,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_flash_very_long_context() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 15000, // pos + 1 = 15001 > 15000
            flash_mode: 0,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_flash_capture_mode() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 0,
            flash_mode: 0,
            capture_mode: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_non_flash_flash_mode_1_short() {
        // flash_mode=1 but pos < 2048 → non-flash
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 100,
            flash_mode: 1,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0Kv);
    }

    #[test]
    fn asym4_givens() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym4);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4);
        assert!(plan.uses_givens);
    }

    #[test]
    fn asym4_fwht() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym4Fwht);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4Fwht);
        assert!(plan.uses_givens);
    }

    #[test]
    fn asym3_givens() {
        let inputs = KvTierInputs {
            quant_asym3: true,
            quant_fwht: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym3);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym3);
        assert!(plan.uses_givens);
    }

    #[test]
    fn asym3_fwht() {
        let inputs = KvTierInputs {
            quant_asym3: true,
            quant_fwht: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym3Fwht);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym3Fwht);
        assert!(plan.uses_givens);
    }

    #[test]
    fn asym2_givens() {
        let inputs = KvTierInputs {
            quant_asym2: true,
            quant_fwht: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym2);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym2);
        assert!(plan.uses_givens);
    }

    #[test]
    fn asym2_fwht() {
        let inputs = KvTierInputs {
            quant_asym2: true,
            quant_fwht: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym2Fwht);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym2Fwht);
        assert!(plan.uses_givens);
    }

    // ── v_mode_bits pass-through ──

    #[test]
    fn v_mode_bits_passed_through() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            v_mode_bits: 3,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs);
        assert_eq!(plan.v_mode_bits, 3);
    }

    // ── tiers_match guard ──

    #[test]
    fn tiers_match_valid_pairs() {
        assert!(tiers_match(KernelKey::KvWriteF32, KernelKey::AttnF32));
        assert!(tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnFlashQ8_0));
        assert!(tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnQ8_0Kv));
        assert!(tiers_match(KernelKey::KvWriteAsym4, KernelKey::AttnFlashAsym4));
        assert!(tiers_match(KernelKey::KvWriteAsym4Fwht, KernelKey::AttnFlashAsym4Fwht));
        assert!(tiers_match(KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym3));
        assert!(tiers_match(KernelKey::KvWriteAsym3Fwht, KernelKey::AttnFlashAsym3Fwht));
        assert!(tiers_match(KernelKey::KvWriteAsym2, KernelKey::AttnFlashAsym2));
        assert!(tiers_match(KernelKey::KvWriteAsym2Fwht, KernelKey::AttnFlashAsym2Fwht));
    }

    #[test]
    fn tiers_match_rejects_cross_tier() {
        assert!(!tiers_match(KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym4));
        assert!(!tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnF32));
        assert!(!tiers_match(KernelKey::KvWriteF32, KernelKey::AttnFlashAsym3Fwht));
    }
}
