// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! KV-tier paired plan type (Phase 0.3). Derived once per attention step from
//! the live KV-cache state. Carries both the write key and attend key together
//! so they can never diverge (the #30-class drift guard).

use crate::types::KernelKey;

/// F32-KV decode attention policy. Default `Simple` = the plain `attention_f32`
/// kernel (llama/qwen35 F32 decode unchanged). `Gqa` = qwen2's 4-way GQA-flash
/// selector (algorithm/occupancy choice, not a quant tier). The arch reads
/// `HIPFIRE_GQA_FUSED` once per forward and passes it as `fused` — `derive`
/// itself stays pure/GPU-free/unit-testable (no env reads).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum F32AttnPolicy {
    #[default]
    Simple,
    Gqa {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        fused: bool,
    },
}

/// Sealed decode of the KV-cache storage tier. Produced ONLY by `classify`.
/// Carries every distinction any consumer branches on, including the
/// Givens-vs-FWHT rotation distinction (the `fwht` field on asym variants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KTier {
    F32,
    Q8,
    Hfq4,  // llama legacy
    Q4,    // llama legacy
    Int8c, // llama INT8-per-column
    Hfq8,  // llama HFQ8 flat-layout
    Asym4 { fwht: bool },
    Asym3 { fwht: bool },
    Asym2 { fwht: bool },
}

/// The single bool→tier decode. The only sanctioned producer of `KTier`.
#[allow(clippy::too_many_arguments)]
pub fn classify(
    quant_q8: bool,
    quant_asym4: bool,
    quant_asym3: bool,
    quant_asym2: bool,
    quant_hfq4: bool,
    quant_q4: bool,
    quant_int8: bool,
    quant_hfq8: bool,
    quant_fwht: bool,
) -> KTier {
    debug_assert!(
        [
            quant_asym4,
            quant_asym3,
            quant_asym2,
            quant_q8,
            quant_hfq4,
            quant_q4,
            quant_int8,
            quant_hfq8
        ]
        .iter()
        .filter(|&&b| b)
        .count()
            <= 1,
        "at most one KV quant tier flag should be set"
    );
    if quant_asym4 {
        KTier::Asym4 { fwht: quant_fwht }
    } else if quant_asym3 {
        KTier::Asym3 { fwht: quant_fwht }
    } else if quant_asym2 {
        KTier::Asym2 { fwht: quant_fwht }
    } else if quant_q8 {
        KTier::Q8
    } else if quant_int8 {
        KTier::Int8c
    } else if quant_hfq8 {
        KTier::Hfq8
    } else if quant_hfq4 {
        KTier::Hfq4
    } else if quant_q4 {
        KTier::Q4
    } else {
        KTier::F32
    }
}

impl KTier {
    /// K bytes per position for the compaction gather kernels (cask/triattn).
    /// Callers MUST gate with `is_compactable()` first.
    pub fn k_bytes_per_pos(self, n_kv_heads: usize, head_dim: usize) -> usize {
        match self {
            KTier::Q8 => n_kv_heads * (head_dim / 32) * 34,
            KTier::Asym4 { .. } => n_kv_heads * (4 + head_dim / 2),
            KTier::Asym3 { .. } => n_kv_heads * (4 + (head_dim * 3) / 8),
            KTier::Asym2 { .. } => n_kv_heads * (4 + head_dim / 4),
            KTier::F32 | KTier::Hfq4 | KTier::Q4 | KTier::Int8c | KTier::Hfq8 => {
                panic!("k_bytes_per_pos undefined for {self:?}")
            }
        }
    }

    /// Tiers the TriAttn/CASK compaction kernels support. Collapses `fwht`.
    pub fn is_compactable(self) -> bool {
        matches!(
            self,
            KTier::Q8 | KTier::Asym4 { .. } | KTier::Asym3 { .. } | KTier::Asym2 { .. }
        )
    }

    /// Drafter-scoring storage must be Q8 or FWHT-asym (Givens-asym banned).
    pub fn storage_ok_for_pflash(self) -> bool {
        matches!(
            self,
            KTier::Q8
                | KTier::Asym4 { fwht: true }
                | KTier::Asym3 { fwht: true }
                | KTier::Asym2 { fwht: true }
        )
    }

    pub fn is_q8(self) -> bool {
        matches!(self, KTier::Q8)
    }
}

/// GPU-free scalar inputs for tier derivation. NO runtime types (avoids the
/// dep cycle — `hipfire-dispatch` cannot depend on `hipfire-runtime`).
/// The arch-side code constructs this from a `&KvCache` at each attention step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvTierInputs {
    pub quant_asym4: bool,
    pub quant_asym3: bool,
    pub quant_asym2: bool,
    pub quant_q8: bool,
    pub quant_fwht: bool,
    pub quant_hfq4: bool, // llama legacy HFQ4 KV mode
    pub quant_q4: bool,   // llama legacy Q4 KV mode
    pub quant_int8: bool, // llama INT8-per-column KV mode
    pub quant_hfq8: bool, // llama HFQ8 flat-layout KV mode
    /// F32-KV attention policy (Simple = attention_f32; Gqa = qwen2 selector).
    pub f32_policy: F32AttnPolicy,
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
    /// cohere2moe: use the sliding-window Q8 flash kernel unconditionally
    /// (NOT the q8_attend_key flash/non-flash heuristic — the non-windowed,
    /// non-flash AttnQ8_0Kv drops the window → wrong attention at ctx>window).
    pub q8_windowed: bool,
    /// Sliding-window size kernarg for the windowed Q8 kernel. `0` = full
    /// (windowed kernel with window=0 == plain flash). Only read when
    /// `q8_windowed` (or any future windowed tier) is set.
    pub window: i32,
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
    /// Sliding-window kernarg (cohere2moe windowed Q8). `0` = full attention.
    pub window: i32,
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
            quant_int8,
            quant_hfq8,
            f32_policy,
            v_mode_bits,
            pos,
            flash_mode,
            capture_mode,
            batch_size,
            is_tree,
            is_boundary,
            q8_windowed,
            window,
        } = inputs;

        let (write_single, attend_single, uses_givens) = if is_boundary {
            // Boundary layers pin to Q8 regardless of global tier.
            let attend = q8_attend_key(pos, flash_mode, capture_mode);
            (KernelKey::KvWriteQ8_0, attend, false)
        } else {
            match classify(
                quant_q8,
                quant_asym4,
                quant_asym3,
                quant_asym2,
                quant_hfq4,
                quant_q4,
                quant_int8,
                quant_hfq8,
                quant_fwht,
            ) {
                KTier::Asym4 { fwht: true } => (
                    KernelKey::KvWriteAsym4Fwht,
                    KernelKey::AttnFlashAsym4Fwht,
                    true,
                ),
                KTier::Asym4 { fwht: false } => {
                    (KernelKey::KvWriteAsym4, KernelKey::AttnFlashAsym4, true)
                }
                KTier::Asym3 { fwht: true } => (
                    KernelKey::KvWriteAsym3Fwht,
                    KernelKey::AttnFlashAsym3Fwht,
                    true,
                ),
                KTier::Asym3 { fwht: false } => {
                    (KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym3, true)
                }
                KTier::Asym2 { fwht: true } => (
                    KernelKey::KvWriteAsym2Fwht,
                    KernelKey::AttnFlashAsym2Fwht,
                    true,
                ),
                KTier::Asym2 { fwht: false } => {
                    (KernelKey::KvWriteAsym2, KernelKey::AttnFlashAsym2, true)
                }
                KTier::Q8 if q8_windowed => {
                    // cohere2moe: windowed flash unconditionally (window=0 on
                    // full layers == plain flash; >0 applies the sliding mask).
                    // NOT q8_attend_key — the non-windowed AttnQ8_0Kv drops the
                    // window and would attend the full context at ctx>window.
                    (
                        KernelKey::KvWriteQ8_0,
                        KernelKey::AttnFlashQ8_0Windowed,
                        false,
                    )
                }
                KTier::Q8 => (
                    KernelKey::KvWriteQ8_0,
                    q8_attend_key(pos, flash_mode, capture_mode),
                    false,
                ),
                KTier::Hfq4 => (KernelKey::KvWriteHfq4, KernelKey::AttnHfq4Kv, false),
                KTier::Q4 => (KernelKey::KvWriteQ4, KernelKey::AttnQ4Kv, false),
                KTier::Int8c => (KernelKey::KvWriteInt8c, KernelKey::AttnInt8cKv, false),
                KTier::Hfq8 => (KernelKey::KvWriteHfq8, KernelKey::AttnHfq8Kv, false),
                KTier::F32 => {
                    // F32 decode: Simple = plain attention_f32 (llama/qwen35);
                    // Gqa = qwen2's 4-way selector (exact mirror of
                    // Qwen2Dense::attend, qwen2.rs:1756-1809). seq_len == pos+1.
                    let attend = match f32_policy {
                        F32AttnPolicy::Simple => KernelKey::AttnF32,
                        F32AttnPolicy::Gqa {
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            fused,
                        } => {
                            let gqa = n_kv_heads < n_heads;
                            if fused && gqa {
                                KernelKey::AttnGqaFused
                            } else if gqa && head_dim == 128 && pos + 1 >= 4096 {
                                KernelKey::AttnGqaWarp
                            } else if gqa && pos + 1 >= 4096 {
                                KernelKey::AttnFlashGqa
                            } else {
                                KernelKey::AttnFlash
                            }
                        }
                    };
                    (KernelKey::KvWriteF32, attend, false)
                }
            }
        };

        // Select batched keys when batch_size > 1.
        let (write_key, attend_key) = if batch_size > 1 {
            let (w, a) = batched_keys(write_single, attend_single, is_tree, batch_size)?;
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
            window,
        })
    }
}

/// Q8 decode attend-key heuristic: select between flash and non-flash Q8
/// attention based on context length and capture mode. Shared between the
/// `is_boundary` and `quant_q8` branches of `derive`.
fn q8_attend_key(pos: usize, flash_mode: usize, capture_mode: bool) -> KernelKey {
    let use_flash =
        capture_mode || flash_mode == 2 || (flash_mode == 1 && pos + 1 >= 2048);
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
        (KvWriteAsym4, AttnFlashAsym4) => Ok((KvWriteAsym4Batched, AttnFlashAsym4BatchedMasked)),
        (KvWriteAsym4Fwht, AttnFlashAsym4Fwht) => {
            Ok((KvWriteAsym4FwhtBatched, AttnFlashAsym4FwhtBatchedMasked))
        }
        // asym3 → batched-masked
        (KvWriteAsym3, AttnFlashAsym3) => Ok((KvWriteAsym3Batched, AttnFlashAsym3BatchedMasked)),
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
        // cohere2moe windowed batched (sliding-window prefill).
        (KvWriteQ8_0, AttnFlashQ8_0Windowed) => {
            Ok((KvWriteQ8_0Batched, AttnQ8_0KvBatchedMaskedWindowed))
        }
        // F32 → no batched keys exist. Returning single-token keys with
        // batch_size > 1 will cause MissingImpl at resolve (BatchEq(1) gate).
        // Intentionally fall through to the default arm rather than silently
        // returning single-token keys that can never resolve batched.
        (KvWriteF32, AttnF32) => Ok((KvWriteF32, AttnF32)),
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
        | (KvWriteQ8_0, AttnFlashQ8_0Windowed)
        // hfq4 single-token (llama legacy)
        | (KvWriteHfq4, AttnHfq4Kv)
        // q4 single-token (llama legacy)
        | (KvWriteQ4, AttnQ4Kv)
        // int8c / hfq8 single-token (llama)
        | (KvWriteInt8c, AttnInt8cKv)
        | (KvWriteHfq8, AttnHfq8Kv)
        // f32 single-token (Simple + qwen2 GQA-flash family)
        | (KvWriteF32, AttnF32)
        | (KvWriteF32, AttnGqaFused)
        | (KvWriteF32, AttnGqaWarp)
        | (KvWriteF32, AttnFlashGqa)
        | (KvWriteF32, AttnFlash)
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
        | (KvWriteQ8_0Batched, AttnQ8_0KvBatchedMaskedWindowed)
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
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: F32AttnPolicy::Simple,
            v_mode_bits: 8,
            pos: 0,
            flash_mode: 0,
            capture_mode: false,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
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
        // Simple policy (default; llama/qwen35) → plain attention_f32, unchanged.
        assert_eq!(plan.attend_key, KernelKey::AttnF32);
        assert!(!plan.uses_givens);
        assert_eq!(plan.batch_size, 1);
    }

    // ── qwen2 F32 GQA-flash selector (exact mirror of Qwen2Dense::attend) ──
    fn gqa_inputs(n_heads: usize, n_kv_heads: usize, head_dim: usize, fused: bool) -> KvTierInputs {
        KvTierInputs {
            f32_policy: F32AttnPolicy::Gqa {
                n_heads,
                n_kv_heads,
                head_dim,
                fused,
            },
            ..default_inputs()
        }
    }

    #[test]
    fn f32_gqa_fused() {
        // fused && gqa → AttnGqaFused (write stays KvWriteF32).
        let p = KvTierPlan::derive(gqa_inputs(14, 2, 128, true)).unwrap();
        assert_eq!(p.write_key, KernelKey::KvWriteF32);
        assert_eq!(p.attend_key, KernelKey::AttnGqaFused);
    }

    #[test]
    fn f32_gqa_fused_needs_gqa() {
        // fused but n_kv==n_heads (not gqa) → falls through to AttnFlash.
        let p = KvTierPlan::derive(gqa_inputs(14, 14, 128, true)).unwrap();
        assert_eq!(p.attend_key, KernelKey::AttnFlash);
    }

    #[test]
    fn f32_gqa_warp_head128_long() {
        // gqa && head_dim==128 && pos+1>=4096 → AttnGqaWarp.
        let inputs = KvTierInputs {
            pos: 4095,
            ..gqa_inputs(14, 2, 128, false)
        };
        assert_eq!(
            KvTierPlan::derive(inputs).unwrap().attend_key,
            KernelKey::AttnGqaWarp
        );
    }

    #[test]
    fn f32_flash_gqa_long_non128() {
        // gqa && pos+1>=4096 but head_dim!=128 → AttnFlashGqa.
        let inputs = KvTierInputs {
            pos: 5000,
            ..gqa_inputs(14, 2, 64, false)
        };
        assert_eq!(
            KvTierPlan::derive(inputs).unwrap().attend_key,
            KernelKey::AttnFlashGqa
        );
    }

    #[test]
    fn f32_flash_short_ctx() {
        // gqa but pos+1<4096 → AttnFlash (the else).
        let inputs = KvTierInputs {
            pos: 100,
            ..gqa_inputs(14, 2, 128, false)
        };
        assert_eq!(
            KvTierPlan::derive(inputs).unwrap().attend_key,
            KernelKey::AttnFlash
        );
    }

    #[test]
    fn f32_flash_non_gqa_long() {
        // not gqa (n_kv==n_heads), long ctx → AttnFlash.
        let inputs = KvTierInputs {
            pos: 5000,
            ..gqa_inputs(14, 14, 128, false)
        };
        assert_eq!(
            KvTierPlan::derive(inputs).unwrap().attend_key,
            KernelKey::AttnFlash
        );
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
    fn int8c_tier() {
        let inputs = KvTierInputs {
            quant_int8: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteInt8c);
        assert_eq!(plan.attend_key, KernelKey::AttnInt8cKv);
        assert!(!plan.uses_givens);
    }

    #[test]
    fn hfq8_tier() {
        let inputs = KvTierInputs {
            quant_hfq8: true,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteHfq8);
        assert_eq!(plan.attend_key, KernelKey::AttnHfq8Kv);
        assert!(!plan.uses_givens);
    }

    #[test]
    fn q8_windowed_bypasses_flash_heuristic() {
        // cohere2moe: q8_windowed picks the windowed key UNCONDITIONALLY (not
        // AttnQ8_0Kv/AttnFlashQ8_0 from q8_attend_key), carrying the window.
        for (pos, window) in [(10usize, 0i32), (10, 4096), (20000, 4096)] {
            let inputs = KvTierInputs {
                quant_q8: true,
                q8_windowed: true,
                window,
                pos,
                ..default_inputs()
            };
            let plan = KvTierPlan::derive(inputs).unwrap();
            assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
            assert_eq!(plan.attend_key, KernelKey::AttnFlashQ8_0Windowed);
            assert_eq!(plan.window, window);
        }
        // Without the flag, q8 still uses the flash/non-flash heuristic.
        let plain = KvTierPlan::derive(KvTierInputs {
            quant_q8: true,
            pos: 10,
            ..default_inputs()
        })
        .unwrap();
        assert_eq!(plain.attend_key, KernelKey::AttnQ8_0Kv);

        // Batched (prefill): q8_windowed → batched windowed key, window carried.
        let batched = KvTierPlan::derive(KvTierInputs {
            quant_q8: true,
            q8_windowed: true,
            window: 4096,
            batch_size: 128,
            ..default_inputs()
        })
        .unwrap();
        assert_eq!(batched.write_key, KernelKey::KvWriteQ8_0Batched);
        assert_eq!(
            batched.attend_key,
            KernelKey::AttnQ8_0KvBatchedMaskedWindowed
        );
        assert_eq!(batched.window, 4096);
    }

    #[test]
    fn tiers_match_int8c_hfq8() {
        assert!(tiers_match(KernelKey::KvWriteInt8c, KernelKey::AttnInt8cKv));
        assert!(tiers_match(KernelKey::KvWriteHfq8, KernelKey::AttnHfq8Kv));
        assert!(!tiers_match(KernelKey::KvWriteInt8c, KernelKey::AttnHfq8Kv));
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
    fn q8_non_flash_very_long_context() {
        // flash_mode=0 never selects flash merely because pos is large;
        // only capture / flash_mode=2 / (flash_mode=1 && long) do.
        let inputs = KvTierInputs {
            quant_q8: true,
            pos: 15000,
            flash_mode: 0,
            capture_mode: false,
            ..default_inputs()
        };
        let plan = KvTierPlan::derive(inputs).unwrap();
        assert_eq!(plan.write_key, KernelKey::KvWriteQ8_0);
        assert_eq!(plan.attend_key, KernelKey::AttnQ8_0Kv);
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
        let inputs = KvTierInputs { ..batched_inputs() };
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
        assert!(tiers_match(
            KernelKey::KvWriteQ8_0,
            KernelKey::AttnFlashQ8_0
        ));
        assert!(tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnQ8_0Kv));
        assert!(tiers_match(
            KernelKey::KvWriteAsym4,
            KernelKey::AttnFlashAsym4
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym4Fwht,
            KernelKey::AttnFlashAsym4Fwht
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym3,
            KernelKey::AttnFlashAsym3
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym3Fwht,
            KernelKey::AttnFlashAsym3Fwht
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym2,
            KernelKey::AttnFlashAsym2
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym2Fwht,
            KernelKey::AttnFlashAsym2Fwht
        ));
        // batched
        assert!(tiers_match(
            KernelKey::KvWriteAsym4Batched,
            KernelKey::AttnFlashAsym4BatchedMasked
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym3Batched,
            KernelKey::AttnFlashAsym3BatchedMasked
        ));
        assert!(tiers_match(
            KernelKey::KvWriteAsym2Batched,
            KernelKey::AttnFlashAsym2Batched
        ));
        assert!(tiers_match(
            KernelKey::KvWriteQ8_0Batched,
            KernelKey::AttnQ8_0KvBatchedMasked
        ));
    }

    #[test]
    fn tiers_match_rejects_cross_tier() {
        assert!(!tiers_match(
            KernelKey::KvWriteAsym3,
            KernelKey::AttnFlashAsym4
        ));
        assert!(!tiers_match(KernelKey::KvWriteQ8_0, KernelKey::AttnF32));
        assert!(!tiers_match(
            KernelKey::KvWriteF32,
            KernelKey::AttnFlashAsym3Fwht
        ));
    }

    // ── KTier enum and classify tests ──

    #[test]
    fn classify_carries_fwht_bit() {
        assert_eq!(
            classify(false, false, true, false, false, false, false, false, true),
            KTier::Asym3 { fwht: true }
        );
        assert_eq!(
            classify(false, false, true, false, false, false, false, false, false),
            KTier::Asym3 { fwht: false }
        );
        assert_eq!(
            classify(true, false, false, false, false, false, false, false, false),
            KTier::Q8
        );
        assert_eq!(
            classify(false, false, false, false, false, false, false, false, false),
            KTier::F32
        );
        assert_eq!(
            classify(false, false, false, false, false, false, true, false, false),
            KTier::Int8c
        );
        assert_eq!(
            classify(false, false, false, false, false, false, false, true, false),
            KTier::Hfq8
        );
    }

    #[test]
    fn ktier_queries() {
        assert!(
            KTier::Q8.is_compactable() && KTier::Q8.is_q8() && KTier::Q8.storage_ok_for_pflash()
        );
        assert!(KTier::Asym3 { fwht: true }.storage_ok_for_pflash());
        assert!(!KTier::Asym3 { fwht: false }.storage_ok_for_pflash());
        assert!(!KTier::F32.is_compactable());
    }
}
