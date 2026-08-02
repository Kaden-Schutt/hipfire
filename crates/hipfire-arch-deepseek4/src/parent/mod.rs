// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # NOT A CALIBRATION REFERENCE (2026-08-02)
//!
//! End-to-end on `tokens.bin` (1024 ids, sha `48b0f834...`):
//!
//! | system | PPL | top-1 | mean KLD vs ref_fp8 |
//! |--------|----:|------:|--------------------:|
//! | **torch teacher (fp8 shim)** | **4.693** | 0.640 | 0 (self) |
//! | torch teacher (exact f32) | 4.624 | 0.649 | 0.040 |
//! | **this `parent/*` backend** | **59.507** | ~0.28-0.48 | **2.718** |
//! | mq2r student (prior) | 14.703 | - | - |
//!
//! That is a **12.7x PPL gap** against the teacher and ~67x the teacher's
//! own fp8-vs-exact KLD control. Component gates (load, HC, MoE, attn smoke,
//! residual **norms**, L0/L2 stage floors, **head path**) can and do pass
//! while the full forward remains badly wrong -- do not read a green gate
//! matrix as "parent is faithful."
//!
//! **Must not be used as a calibration reference for Gates 6-9 (or any
//! Hessian / GPTQ / activation capture).** The production teacher is the
//! torch residual/PPL harness under `reference_oracle/` (`ref_ppl_e2e.py`,
//! `ref_fp8_*.plog`). See:
//! - `reference_oracle/REF_PPL_E2E.md` -- teacher PPL + residual-magnitude note
//! - `reference_oracle/HEAD_PATH_CONTENT.md` -- head path closed at floor
//! - `reference_oracle/ATTN_OUT_QUANT_FLOOR.md` -- L0 attn_out at quant floor
//! - `docs/investigations/2026-08-02-ds4-parent-not-calibration-ref.md`
//!
//! Investigation stopped after the head-path check (host + GPU) matched the
//! teacher at the BF16-act floor on identical residuals. Deep-layer stage
//! bisect was deliberately not pursued; residual content / full-seq path
//! remains open but is out of scope for this backend until explicitly reopened.
//!
//! DeepSeek V4 Flash **parent-checkpoint** calibration backend.
//!
//! This module exists to execute the *original* mixed-precision DeepSeek V4
//! Flash checkpoint — `F8_E4M3` dense weights with `F8_E8M0` 128×128 block
//! scales, plus `I8`-packed `E2M1` routed experts with per-32-K `F8_E8M0`
//! scales — rather than a hipfire-quantized derivative of it.
//!
//! The distinction is load-bearing. A previous capture drove 554 activation
//! tensors and their Gram matrices through the *quantized* MQ2R P3 artifact
//! and labelled the result "native"; the buffers were F32, but the
//! distribution was the quantized model's, so the Hessians were rejected as
//! GPTQ input. See
//! `docs/investigations/2026-08-01-ds4-parent-hessian-handoff.md`.
//!
//! Everything here is therefore **fail-closed**: a missing scale companion,
//! an unexpected dtype or shape, or a non-gfx942 device is an error, never a
//! fallback to [`rdna_compute::DType::Raw`], to the MQ2R path, or to a
//! generic RDNA route. Silence is the failure mode this module is designed to
//! make impossible.
//!
//! Backend identity belongs to the loaded weights, not to the process-wide
//! GPU context — same rule as [`crate::backend`]'s `Mq2rBackend`, so a model
//! swap cannot inherit parent-calibration policy.

pub mod attention;
pub mod codec;
pub mod compressor;
pub mod forward;
pub mod gemm_ref;
pub mod hc;
pub mod head;
pub mod hessian;
pub mod indexer;
pub mod inventory;
pub mod layer_ref;
pub mod linear;
pub mod manifest;
pub mod model;
pub mod moe;
pub mod plog;
pub mod weights;

use hipfire_runtime::model_source::ModelSource;
use rdna_compute::Gpu;

/// Quantization contract the parent checkpoint must declare, verbatim.
///
/// These are the values in `config.json → quantization_config` of
/// `DeepSeek-V4-Flash-0731`. Admission compares against them exactly; a
/// checkpoint that differs in any field is a different format and is refused
/// rather than reinterpreted.
pub const PARENT_MODEL_TYPE: &str = "deepseek_v4";
pub const PARENT_QUANT_METHOD: &str = "fp8";
pub const PARENT_WEIGHT_FMT: &str = "e4m3";
pub const PARENT_SCALE_FMT: &str = "ue8m0";
pub const PARENT_EXPERT_DTYPE: &str = "fp4";
pub const PARENT_WEIGHT_BLOCK: [usize; 2] = [128, 128];
/// Routed-expert FP4 scale group along K. Note this is *not* the dense
/// `weight_block_size`; experts carry one E8M0 scale per row per 32 K.
pub const PARENT_EXPERT_SCALE_GROUP: usize = 32;

/// Model-side proof that the original parent checkpoint was admitted on an
/// exact gfx942 device.
///
/// The field is private and every operation reacquires the rdna-compute
/// gfx942 borrow, so moving these weights to another device fails closed
/// instead of quietly selecting an RDNA or portable kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ds4ParentBackend {
    _sealed: (),
}

/// The subset of `config.json` that admission checks. Parsed straight from
/// the source metadata JSON so admission does not depend on
/// [`crate::DeepseekV4Config`] having preserved these fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentQuantConfig {
    pub model_type: String,
    pub quant_method: String,
    pub fmt: String,
    pub scale_fmt: String,
    pub expert_dtype: String,
    pub weight_block_size: [usize; 2],
    pub num_hidden_layers: usize,
    pub num_hash_layers: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub compress_ratios: Vec<usize>,
}

impl ParentQuantConfig {
    /// Parse and validate the parent quant contract from a source's metadata
    /// JSON. Every field is required; a missing or mismatched field is an
    /// error, because each one changes how bytes must be decoded.
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("deepseek4 parent: metadata_json not valid JSON: {e}"))?;
        // Both the HFQ wrapper and `SafetensorsSource` nest the HF config
        // under `config`; tolerate a bare config for direct callers.
        let cfg = meta.get("config").unwrap_or(&meta);

        let want_str = |key: &str, want: &str| -> Result<String, String> {
            let got = cfg
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("deepseek4 parent: config.{key} missing or not a string"))?;
            if got != want {
                return Err(format!(
                    "deepseek4 parent: config.{key} = {got:?}, parent calibration requires {want:?}"
                ));
            }
            Ok(got.to_owned())
        };
        let want_usize = |key: &str| -> Result<usize, String> {
            cfg.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("deepseek4 parent: config.{key} missing or not an integer"))
        };

        let model_type = want_str("model_type", PARENT_MODEL_TYPE)?;
        let expert_dtype = want_str("expert_dtype", PARENT_EXPERT_DTYPE)?;

        let qc = cfg.get("quantization_config").ok_or_else(|| {
            "deepseek4 parent: config.quantization_config missing — this is not a \
             quantized parent checkpoint"
                .to_owned()
        })?;
        let qc_str = |key: &str, want: &str| -> Result<String, String> {
            let got = qc.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
                format!("deepseek4 parent: quantization_config.{key} missing or not a string")
            })?;
            if got != want {
                return Err(format!(
                    "deepseek4 parent: quantization_config.{key} = {got:?}, parent \
                     calibration requires {want:?}"
                ));
            }
            Ok(got.to_owned())
        };
        let quant_method = qc_str("quant_method", PARENT_QUANT_METHOD)?;
        let fmt = qc_str("fmt", PARENT_WEIGHT_FMT)?;
        let scale_fmt = qc_str("scale_fmt", PARENT_SCALE_FMT)?;

        let wbs: Vec<usize> = qc
            .get("weight_block_size")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        if wbs.as_slice() != PARENT_WEIGHT_BLOCK {
            return Err(format!(
                "deepseek4 parent: quantization_config.weight_block_size = {wbs:?}, \
                 parent calibration requires {PARENT_WEIGHT_BLOCK:?}"
            ));
        }

        let compress_ratios: Vec<usize> = cfg
            .get("compress_ratios")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "deepseek4 parent: config.compress_ratios missing".to_owned())?
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .collect();

        Ok(Self {
            model_type,
            quant_method,
            fmt,
            scale_fmt,
            expert_dtype,
            weight_block_size: PARENT_WEIGHT_BLOCK,
            num_hidden_layers: want_usize("num_hidden_layers")?,
            num_hash_layers: want_usize("num_hash_layers")?,
            n_routed_experts: want_usize("n_routed_experts")?,
            num_experts_per_tok: want_usize("num_experts_per_tok")?,
            compress_ratios,
        })
    }

    /// Per-layer KV compression ratio. Layers with ratio `4` additionally
    /// carry the indexer sub-module; ratio `0` layers carry no compressor.
    pub fn compress_ratio(&self, layer: usize) -> usize {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }
}

impl Ds4ParentBackend {
    /// Admit the original parent checkpoint, or refuse.
    ///
    /// Admission requires the full declared quant contract *and* an exact
    /// gfx942 device. There is deliberately no environment-variable override
    /// and no "best effort" arm: the entire point of this backend is that a
    /// bad admission cannot silently produce plausible-looking numbers.
    ///
    /// MTP is excluded from parent calibration loads (the handoff's gate
    /// order), so callers must not request it.
    pub fn admit(
        source: &dyn ModelSource,
        gpu: &mut Gpu,
    ) -> Result<(Self, ParentQuantConfig), String> {
        let cfg = ParentQuantConfig::from_metadata_json(source.metadata_json())?;
        if gpu.try_gfx942().is_none() {
            return Err(
                "deepseek4 parent: parent-checkpoint calibration requires an exact gfx942 \
                 device; there is no portable or RDNA fallback for the E4M3/E2M1 decode path"
                    .to_owned(),
            );
        }
        Ok((Self { _sealed: () }, cfg))
    }

    /// Reacquire the gfx942 guarantee at use time. Every operation that
    /// touches parent weights goes through this, so a device change between
    /// admission and execution fails closed.
    pub fn ensure_device(self, gpu: &mut Gpu) -> Result<(), String> {
        if gpu.try_gfx942().is_none() {
            return Err(
                "deepseek4 parent: admitted parent weights cannot execute on this GPU \
                 (gfx942 required)"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `DeepSeek-V4-Flash-0731` `config.json`, trimmed to the fields
    /// admission reads. Values transcribed from the checkpoint on `mi300x`.
    fn parent_config_json() -> String {
        serde_json::json!({
            "config": {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "num_hidden_layers": 43,
                "num_hash_layers": 3,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 0, 0, 0],
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128]
                }
            }
        })
        .to_string()
    }

    #[test]
    fn admits_the_real_parent_config() {
        let cfg = ParentQuantConfig::from_metadata_json(&parent_config_json())
            .expect("real parent config must be admitted");
        assert_eq!(cfg.num_hidden_layers, 43);
        assert_eq!(cfg.num_hash_layers, 3);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 6);
        // Layer 2 is the first compressed layer, and ratio 4 is the indexer
        // marker; layer 3 is ratio 128 (compressor, no indexer).
        assert_eq!(cfg.compress_ratio(0), 0);
        assert_eq!(cfg.compress_ratio(2), 4);
        assert_eq!(cfg.compress_ratio(3), 128);
        // 43 main layers + 3 MTP/DSpark entries.
        assert_eq!(cfg.compress_ratios.len(), 46);
    }

    /// Each of these mutations changes how bytes must be decoded, so each one
    /// must refuse rather than be reinterpreted.
    #[test]
    fn refuses_every_contract_deviation() {
        let cases: [(&str, serde_json::Value); 6] = [
            ("model_type", serde_json::json!("deepseek_v3")),
            ("expert_dtype", serde_json::json!("fp8")),
            ("quantization_config.fmt", serde_json::json!("e5m2")),
            ("quantization_config.scale_fmt", serde_json::json!("fp32")),
            ("quantization_config.quant_method", serde_json::json!("awq")),
            (
                "quantization_config.weight_block_size",
                serde_json::json!([64, 64]),
            ),
        ];
        for (path, bad) in cases {
            let mut v: serde_json::Value =
                serde_json::from_str(&parent_config_json()).expect("fixture parses");
            let cfg = v.get_mut("config").expect("config node");
            match path.split_once('.') {
                Some((outer, inner)) => {
                    cfg[outer][inner] = bad;
                }
                None => cfg[path] = bad,
            }
            let err = ParentQuantConfig::from_metadata_json(&v.to_string())
                .expect_err(&format!("{path} deviation must be refused"));
            assert!(
                err.starts_with("deepseek4 parent:"),
                "{path}: unprefixed error {err:?}"
            );
        }
    }

    #[test]
    fn refuses_a_checkpoint_with_no_quantization_config() {
        let mut v: serde_json::Value =
            serde_json::from_str(&parent_config_json()).expect("fixture parses");
        v["config"]
            .as_object_mut()
            .expect("config object")
            .remove("quantization_config");
        let err = ParentQuantConfig::from_metadata_json(&v.to_string())
            .expect_err("an unquantized checkpoint is not a parent checkpoint");
        assert!(err.contains("quantization_config"), "{err}");
    }
}
