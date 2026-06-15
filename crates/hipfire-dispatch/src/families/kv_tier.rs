// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
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
    pub quant_hfq4: bool,   // llama legacy HFQ4 KV mode
    pub quant_q4: bool,     // llama legacy Q4 KV mode
    pub v_mode_bits: i32,
    // q8 use_flash heuristic inputs (moved from qwen35.rs:12885)
    pub pos: usize,
    pub flash_mode: usize,
    pub capture_mode: bool,
    // ── Batched / tree / boundary (ship 3.2) ──
    /// Token batch size. `1` for decode/per-token, `>1` for batched prefill.
    pub batch_size: usize,
    /// True when tree-verify is active (`tree_bias.is_some()`).
    pub is_tree: bool,
    /// True for boundary layers (pinned to Q8 regardless of global tier).
    /// Inert until the boundary-layer producer populates `layer_is_boundary`.
    pub is_boundary: bool,
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
    /// Token batch size (for ShapeInfo threading).
    pub batch_size: usize,
}

/// Error returned by `KvTierPlan::derive` when the combination of inputs
/// is unsupported (e.g. 2-bit tier + tree-verify has no masked kernel).
#[derive(Debug)]
pub struct UnsupportedTreeTier {
    pub write_key: KernelKey,
    pub attend_key: KernelKey,
    pub batch_size: usize,
}

impl std::fmt::Display for UnsupportedTreeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tree-verify not supported for tier {:?}+{:?} at batch_size={}",
            self.write_key, self.attend_key, self.batch_size
        )
    }
}

impl KvTierPlan {
    /// Derive the paired (write, attend) key plan from scalar KV-cache state.
    /// GPU-free, unit-testable. The q8 `use_flash` heuristic is folded in:
    /// it selects between `AttnFlashQ8_0` and `AttnQ8_0Kv`.
    ///
    /// Returns `Err(UnsupportedTreeTier)` when `batch_size > 1 && is_tree`
    /// but the tier has no `_batched_masked` variant (currently: 2-bit).
    ///
    /// Panics (debug_assert) if the inputs are contradictory (e.g. two
    /// quant flags set simultaneously).
    pub fn derive(inputs: KvTierInputs) -> Result<Self, UnsupportedTreeTier> {
        let KvTierInputs {
            quant_asym4,
            quant_asym3,
            quant_asym2,
            quant_q8,
            quant_fwht,
            quant_hfq4,
            quant_q4,
            v_mode_bits,
            pos,
            flash_mode,
            capture_mode,
            batch_size,
            is_tree,
            is_boundary,
        } = inputs;

        // At most one quant tier flag should be set.
        debug_assert!(
            [quant_asym4, quant_asym3, quant_asym2, quant_q8, quant_hfq4, quant_q4]
                .iter()
                .filter(|&&b| b)
                .count() <= 1,
            "at most one KV quant tier flag should be set"
        );

        let (write_single, attend_single, uses_givens) = if is_boundary {
            // Boundary layers pin to Q8 regardless of global tier.
            let attend = q8_attend_key(pos, flash_mode, capture_mode);
            (KernelKey::KvWriteQ8_0, attend, false)
        } else if quant_asym4 {
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
            let attend = q8_attend_key(pos, flash_mode, capture_mode);
            (KernelKey::KvWriteQ8_0, attend, false)
        } else if quant_hfq4 {
            (KernelKey::KvWriteHfq4, KernelKey::AttnHfq4Kv, false)
        } else if quant_q4 {
            (KernelKey::KvWriteQ4, KernelKey::AttnQ4Kv, false)
        } else {
            // F32 fallback
            (KernelKey::KvWriteF32, KernelKey::AttnF32, false)
        };

        // Select batched keys when batch_size > 1.
        let (write_key, attend_key) = if batch_size > 1 {
            let (w, a) = batched_keys(write_single, attend_single, is_tree, batch_size)?;;
            (w, a)
        } else {
            (write_single, attend_single)
        };

        // Phase 0.3 drift guard: write and attend keys must agree on tier.
        debug_assert!(
            tiers_match(write_key, attend_key),
            "KvTierPlan tier mismatch: write={:?}, attend={:?}",
            write_key,
            attend_key,
        );

        Ok(Self {
            write_key,
            attend_key,
            v_mode_bits,
            uses_givens,
            batch_size,
        })
    }
}

/// Q8 decode attend-key heuristic: select between flash and non-flash Q8
/// attention based on context length and capture mode. Shared between the
/// `is_boundary` and `quant_q8` branches of `derive`.
fn q8_attend_key(pos: usize, flash_mode: usize, capture_mode: bool) -> KernelKey {
    let use_flash = capture_mode
        || flash_mode == 2
        || (flash_mode == 1 && pos + 1 >= 2048)
        || pos + 1 > 15000;
    if use_flash {
        KernelKey::AttnFlashQ8_0
    } else {
        KernelKey::AttnQ8_0Kv
    }
}

/// Map single-token keys to their batched counterparts.
/// Returns `Err(UnsupportedTreeTier)` for 2-bit + tree-verify (no _masked kernel).
fn batched_keys(
    write_single: KernelKey,
    attend_single: KernelKey,
    is_tree: bool,
    batch_size: usize,
) -> Result<(KernelKey, KernelKey), UnsupportedTreeTier> {
    use KernelKey::*;
    match (write_single, attend_single) {
        // asym4 → batched-masked (serves both causal and tree-verify)
        (KvWriteAsym4, AttnFlashAsym4) => {
            Ok((KvWriteAsym4Batched, AttnFlashAsym4BatchedMasked))
        }
        (KvWriteAsym4Fwht, AttnFlashAsym4Fwht) => {
            Ok((KvWriteAsym4FwhtBatched, AttnFlashAsym4FwhtBatchedMasked))
        }
        // asym3 → batched-masked
        (KvWriteAsym3, AttnFlashAsym3) => {
            Ok((KvWriteAsym3Batched, AttnFlashAsym3BatchedMasked))
        }
        (KvWriteAsym3Fwht, AttnFlashAsym3Fwht) => {
            Ok((KvWriteAsym3FwhtBatched, AttnFlashAsym3FwhtBatchedMasked))
        }
        // asym2 → batched ONLY (no _masked — 2-bit tree-verify gap)
        (KvWriteAsym2, AttnFlashAsym2) => {
            if is_tree {
                Err(UnsupportedTreeTier {
                    write_key: write_single,
                    attend_key: attend_single,
                    batch_size,
                })
            } else {
                Ok((KvWriteAsym2Batched, AttnFlashAsym2Batched))
            }
        }
        (KvWriteAsym2Fwht, AttnFlashAsym2Fwht) => {
            if is_tree {
                Err(UnsupportedTreeTier {
                    write_key: write_single,
                    attend_key: attend_single,
                    batch_size,
                })
            } else {
                Ok((KvWriteAsym2FwhtBatched, AttnFlashAsym2FwhtBatched))
            }
        }
        // Q8 → batched-masked (P-1 no-LDS-cap tiled kernel)
        (KvWriteQ8_0, AttnFlashQ8_0) | (KvWriteQ8_0, AttnQ8_0Kv) => {
            Ok((KvWriteQ8_0Batched, AttnQ8_0KvBatchedMasked))
        }
        // F32 → no batched keys exist. Returning single-token keys with
        // batch_size > 1 will cause MissingImpl at resolve (BatchEq(1) gate).
        // Intentionally fall through to the default arm rather than silently
        // returning single-token keys that can never resolve batched.
        (KvWriteF32, AttnF32) => {
            Ok((KvWriteF32, AttnF32))
        }
        _ => Ok((write_single, attend_single)),
    }
}

/// Check that the write and attend keys agree on tier. This is the Phase 0.3
/// #30-class drift guard — a first-class assert on a first-class type.
/// Covers both single-token and batched pairs.
fn tiers_match(write: KernelKey, attend: KernelKey) -> bool {
    use KernelKey::*;
    matches!(
        (write, attend),
        // asym4 single-token
        (KvWriteAsym4, AttnFlashAsym4)
        | (KvWriteAsym4Fwht, AttnFlashAsym4Fwht)
        // asym3 single-token
        | (KvWriteAsym3, AttnFlashAsym3)
        | (KvWriteAsym3Fwht, AttnFlashAsym3Fwht)
        // asym2 single-token
        | (KvWriteAsym2, AttnFlashAsym2)
        | (KvWriteAsym2Fwht, AttnFlashAsym2Fwht)
        // q8 single-token
        | (KvWriteQ8_0, AttnFlashQ8_0)
        | (KvWriteQ8_0, AttnQ8_0Kv)
        // hfq4 single-token (llama legacy)
        | (KvWriteHfq4, AttnHfq4Kv)
        // q4 single-token (llama legacy)
        | (KvWriteQ4, AttnQ4Kv)
        // f32 single-token
        | (KvWriteF32, AttnF32)
        // asym4 batched
        | (KvWriteAsym4Batched, AttnFlashAsym4BatchedMasked)
        | (KvWriteAsym4FwhtBatched, AttnFlashAsym4FwhtBatchedMasked)
        // asym3 batched
        | (KvWriteAsym3Batched, AttnFlashAsym3BatchedMasked)
        | (KvWriteAsym3FwhtBatched, AttnFlashAsym3FwhtBatchedMasked)
        // asym2 batched (no _masked)
        | (KvWriteAsym2Batched, AttnFlashAsym2Batched)
        | (KvWriteAsym2FwhtBatched, AttnFlashAsym2FwhtBatched)
        // q8 batched
        | (KvWriteQ8_0Batched, AttnQ8_0KvBatchedMasked)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build default inputs (all-false = F32 tier, batch_size=1).
    fn default_inputs() -> KvTierInputs {
        KvTierInputs {
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_q8: false,
            quant_fwht: false,
            quant_hfq4: false,
            quant_q4: false,
            v_mode_bits: 8,
            pos: 0,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
        }
    }

    /// Helper for batched inputs.
    fn batched_inputs() -> KvTierInputs {
        KvTierInputs {
            batch_size: 128,
            ..default_inputs()
        }
    }

    // ── Single-token tier derivation tests (unchanged from 3.1) ──

    #[test]
    fn f32_tier() {
        let plan = KvTierPlan::derive(default_inputs()).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteF32);
        assert_eq!(plan.attend_key, KernelKey::AttnF32);
        assert!(!plan.uses_givens);
        assert_eq!(plan.batch_size, 1);
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn hfq4_tier() {
        let inputs = KvTierInputs {
            quant_hfq4: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteHfq4);
        assert_eq!(plan.attend_key, KernelKey::AttnHfq4Kv);
        assert!(!plan.uses_givens);
    }

    #[test]
    fn q4_tier() {
        let inputs = KvTierInputs {
            quant_q4: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ4);
        assert_eq!(plan.attend_key, KernelKey::AttnQ4Kv);
        assert!(!plan.uses_givens);
    }

    #[test]
    fn q8_flash_long_context() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 2047,
            flash_mode: 1,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_flash_very_long_context() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 15000,
            flash_mode: 0,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0);
    }

    #[test]
    fn q8_non_flash_flash_mode_1_short() {
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 100,
            flash_mode: 1,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
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
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym2Fwht);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym2Fwht);
        assert!(plan.uses_givens);
    }

    #[test]
    fn v_mode_bits_passed_through() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            v_mode_bits: 3,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.v_mode_bits, 3);
    }

    // ── Batched key selection tests ──

    #[test]
    fn batched_asym4() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym4Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4BatchedMasked);
        assert_eq!(plan.batch_size, 128);
    }

    #[test]
    fn batched_asym4_fwht() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym4FwhtBatched);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym4FwhtBatchedMasked);
    }

    #[test]
    fn batched_asym3() {
        let inputs = KvTierInputs {
            quant_asym3: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym3Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym3BatchedMasked);
    }

    #[test]
    fn batched_asym3_fwht() {
        let inputs = KvTierInputs {
            quant_asym3: true,
            quant_fwht: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym3FwhtBatched);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym3FwhtBatchedMasked);
    }

    #[test]
    fn batched_asym2_causal() {
        let inputs = KvTierInputs {
            quant_asym2: true,
            is_tree: false,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteAsym2Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnFlashAsym2Batched);
    }

    #[test]
    fn batched_asym2_tree_rejected() {
        let inputs = KvTierInputs {
            quant_asym2: true,
            is_tree: true,
            ..batched_inputs()
        };
        let err = KvTierPlan::derive(inputs).unwrap_err();
        assert_eq!(err.write_key, KernelKey::KvWriteAsym2);
        assert_eq!(err.attend_key, KernelKey::AttnFlashAsym2);
    }

    #[test]
    fn batched_asym2_fwht_tree_rejected() {
        let inputs = KvTierInputs {
            quant_asym2: true,
            quant_fwht: true,
            is_tree: true,
            ..batched_inputs()
        };
        assert!(KvTierPlan::derive(inputs).is_err());
    }

    #[test]
    fn batched_q8() {
        let inputs = KvTierInputs {
            quant_q8: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0KvBatchedMasked);
    }

    #[test]
    fn batched_q8_tree() {
        let inputs = KvTierInputs {
            quant_q8: true,
            is_tree: true,
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0KvBatchedMasked);
    }

    #[test]
    fn batched_f32_returns_single_token_keys() {
        // F32 has no batched keys — returns single-token for per-token fallback.
        let inputs = KvTierInputs {
            ..batched_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteF32);
        assert_eq!(plan.attend_key, KernelKey::AttnF32);
    }

    // ── Boundary layer tests ──

    #[test]
    fn boundary_pins_q8_regardless_of_tier() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            is_boundary: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0Kv);
    }

    #[test]
    fn boundary_batched_pins_q8_batched() {
        let inputs = KvTierInputs {
            quant_asym4: true,
            quant_fwht: true,
            is_boundary: true,
            batch_size: 64,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0Batched);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0KvBatchedMasked);
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
        // batched
        assert!(tiers_match(KernelKey::KvWriteAsym4Batched, KernelKey::AttnFlashAsym4BatchedMasked));
        assert!(tiers_match(KernelKey::KvWriteAsym3Batched, KernelKey::AttnFlashAsym3BatchedMasked));
        assert!(tiers_match(KernelKey::KvWriteAsym2Batched, KernelKey::AttnFlashAsym2Batched));
        assert!(tiers_match(KernelKey::KvWriteQ8_0Batched, KernelKey::AttnQ8_0KvBatchedMasked));
    }

    #[test]
    fn tiers_match_rejects_cross_tier() {
        assert!(!tiers_match(KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym4));
        assert!(!tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnF32));
        assert!(!tiers_match(KernelKey::KvWriteF32, KernelKey::AttnFlashAsym3Fwht));
    }
}
