// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed inventory of a DeepSeek V4 Flash **parent** checkpoint.
//!
//! Enumerates every tensor the source offers, pairs each native quantized
//! weight with exactly one valid scale companion, and excludes MTP from the
//! main-tower load set while still validating MTP tensors. See
//! `docs/investigations/2026-08-01-ds4-parent-hessian-handoff.md` and
//! Gate 1 of the parent-checkpoint calibration backend.

use hipfire_runtime::model_source::ModelSource;
use std::collections::{HashMap, HashSet};

use super::{ParentQuantConfig, PARENT_EXPERT_SCALE_GROUP, PARENT_WEIGHT_BLOCK};

/// Classification of a parent-checkpoint tensor for the load / dequant path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParentTensorClass {
    /// Dense `F8_E4M3` weight with a 128×128 `F8_E8M0` scale companion.
    DenseFp8,
    /// Routed-expert `I8`-packed E2M1 weight with per-row / group-32 `F8_E8M0` scale.
    ExpertFp4,
    Bf16,
    F32,
    I64,
}

/// One main-tower tensor admitted into the parent inventory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentTensorEntry {
    pub name: String,
    pub class: ParentTensorClass,
    /// Safetensors dtype string (`F8_E4M3`, `I8`, `BF16`, …).
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Equals [`Self::shape`] except [`ParentTensorClass::ExpertFp4`], where
    /// the logical K dimension is `2 * Kpacked`.
    pub logical_shape: Vec<usize>,
    /// Present iff `class` is [`ParentTensorClass::DenseFp8`] or
    /// [`ParentTensorClass::ExpertFp4`].
    pub scale: Option<ParentScaleRef>,
    pub is_mtp: bool,
}

/// Scale companion of a quantized weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentScaleRef {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Main-tower inventory plus the MTP names that were validated and skipped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentInventory {
    /// Main-tower tensors only (`is_mtp == false`).
    pub entries: Vec<ParentTensorEntry>,
    /// MTP tensor names skipped for parent calibration loads (for the manifest).
    pub excluded_mtp: Vec<String>,
    pub totals: ParentInventoryTotals,
}

/// Aggregate counts measured while building the inventory.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ParentInventoryTotals {
    /// Every name offered by the source (main tower + MTP).
    pub tensors_seen: usize,
    pub dense_fp8: usize,
    pub expert_fp4: usize,
    pub bf16: usize,
    pub f32: usize,
    pub i64: usize,
    pub mtp_excluded: usize,
    /// Main-tower payload bytes only (weights + scales + unquantized).
    pub payload_bytes: u64,
}

impl ParentInventory {
    /// Build a fail-closed inventory of `source` under the admitted parent
    /// quant contract in `cfg`.
    ///
    /// Every tensor the source reports is classified and validated. Quantized
    /// weights must carry a correctly shaped scale sibling; unquantized
    /// tensors must not; scales must not be orphaned; unknown dtypes are
    /// refused. MTP names are recorded in [`Self::excluded_mtp`] and kept out
    /// of [`Self::entries`], but a broken MTP tensor still fails the build.
    pub fn build(source: &dyn ModelSource, cfg: &ParentQuantConfig) -> Result<Self, String> {
        let _ = cfg; // contract already admitted; kept for call-site symmetry / future probes

        let names = source.tensor_names();
        let tensors_seen = names.len();

        // Materialize infos once so scale pairing is O(1) and we can detect
        // orphan scales after walking every non-scale tensor.
        let mut infos: HashMap<String, (String, Vec<usize>, u64)> =
            HashMap::with_capacity(tensors_seen);
        for name in &names {
            let info = source.tensor_info(name).ok_or_else(|| {
                format!(
                    "deepseek4 parent: tensor {name:?} listed by source but tensor_info returned None"
                )
            })?;
            let nbytes = info.data_size as u64;
            infos.insert(
                name.to_string(),
                (info.dtype.clone(), info.shape.clone(), nbytes),
            );
        }

        let mut scale_claimed: HashSet<String> = HashSet::new();
        let mut entries: Vec<ParentTensorEntry> = Vec::new();
        let mut excluded_mtp: Vec<String> = Vec::new();
        let mut totals = ParentInventoryTotals {
            tensors_seen,
            ..ParentInventoryTotals::default()
        };

        // First pass: classify every non-scale tensor and claim its scale.
        let mut non_scale_names: Vec<String> = infos
            .keys()
            .filter(|n| !n.ends_with(".scale"))
            .cloned()
            .collect();
        non_scale_names.sort();

        for name in non_scale_names {
            let (dtype, shape, nbytes) = infos.get(&name).expect("key from infos").clone();
            let is_mtp = name.starts_with("mtp.");

            match dtype.as_str() {
                "F8_E4M3" => {
                    let entry = classify_dense_fp8(&name, &shape, &infos, &mut scale_claimed)?;
                    bump_class(
                        &mut totals,
                        ParentTensorClass::DenseFp8,
                        is_mtp,
                        nbytes,
                    );
                    if is_mtp {
                        excluded_mtp.push(name);
                    } else {
                        entries.push(entry);
                    }
                }
                "I8" => {
                    let entry = classify_expert_fp4(&name, &shape, &infos, &mut scale_claimed)?;
                    bump_class(
                        &mut totals,
                        ParentTensorClass::ExpertFp4,
                        is_mtp,
                        nbytes,
                    );
                    if is_mtp {
                        excluded_mtp.push(name);
                    } else {
                        entries.push(entry);
                    }
                }
                "BF16" => {
                    reject_scale_sibling(&name, &infos)?;
                    let entry = ParentTensorEntry {
                        name: name.clone(),
                        class: ParentTensorClass::Bf16,
                        dtype,
                        shape: shape.clone(),
                        logical_shape: shape,
                        scale: None,
                        is_mtp,
                    };
                    bump_class(&mut totals, ParentTensorClass::Bf16, is_mtp, nbytes);
                    if is_mtp {
                        excluded_mtp.push(name);
                    } else {
                        entries.push(entry);
                    }
                }
                "F32" => {
                    reject_scale_sibling(&name, &infos)?;
                    let entry = ParentTensorEntry {
                        name: name.clone(),
                        class: ParentTensorClass::F32,
                        dtype,
                        shape: shape.clone(),
                        logical_shape: shape,
                        scale: None,
                        is_mtp,
                    };
                    bump_class(&mut totals, ParentTensorClass::F32, is_mtp, nbytes);
                    if is_mtp {
                        excluded_mtp.push(name);
                    } else {
                        entries.push(entry);
                    }
                }
                "I64" => {
                    reject_scale_sibling(&name, &infos)?;
                    let entry = ParentTensorEntry {
                        name: name.clone(),
                        class: ParentTensorClass::I64,
                        dtype,
                        shape: shape.clone(),
                        logical_shape: shape,
                        scale: None,
                        is_mtp,
                    };
                    bump_class(&mut totals, ParentTensorClass::I64, is_mtp, nbytes);
                    if is_mtp {
                        excluded_mtp.push(name);
                    } else {
                        entries.push(entry);
                    }
                }
                "F8_E8M0" => {
                    // Scales are only valid as `.scale` companions of quantized
                    // weights. A bare F8_E8M0 that is not named `.scale` is
                    // refuse-closed; named scales are claimed in the weight pass
                    // and orphan-checked below.
                    if !name.ends_with(".scale") {
                        return Err(format!(
                            "deepseek4 parent: tensor {name:?} has dtype F8_E8M0 but is not a .scale companion"
                        ));
                    }
                }
                other => {
                    return Err(format!(
                        "deepseek4 parent: tensor {name:?} has unsupported dtype {other:?}; \
                         parent inventory accepts only F8_E4M3, F8_E8M0, I8, BF16, F32, I64"
                    ));
                }
            }
        }

        // Orphan / unclaimed scales (rule 4), including MTP.
        let mut scale_names: Vec<String> = infos
            .keys()
            .filter(|n| n.ends_with(".scale"))
            .cloned()
            .collect();
        scale_names.sort();
        for scale_name in scale_names {
            let (dtype, _shape, nbytes) = infos.get(&scale_name).expect("key from infos");
            if dtype != "F8_E8M0" {
                return Err(format!(
                    "deepseek4 parent: scale tensor {scale_name:?} has dtype {dtype:?}, expected F8_E8M0"
                ));
            }
            if !scale_claimed.contains(&scale_name) {
                let weight_name = scale_to_weight_name(&scale_name);
                if !infos.contains_key(weight_name.as_str()) {
                    return Err(format!(
                        "deepseek4 parent: orphan scale {scale_name:?} has no corresponding .weight"
                    ));
                }
                return Err(format!(
                    "deepseek4 parent: orphan scale {scale_name:?} was not claimed by a quantized weight"
                ));
            }
            let is_mtp = scale_name.starts_with("mtp.");
            if is_mtp {
                // Scales for MTP quantized weights are excluded with the weight;
                // record the scale name too so the manifest can list every skipped
                // source tensor, not only primary weights.
                excluded_mtp.push(scale_name.clone());
            } else {
                // Main-tower scale bytes count toward payload; class counters
                // already counted the weight side only.
                totals.payload_bytes = totals.payload_bytes.saturating_add(*nbytes);
            }
        }

        // Sort for deterministic downstream consumption.
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        excluded_mtp.sort();
        totals.mtp_excluded = excluded_mtp.len();

        Ok(Self {
            entries,
            excluded_mtp,
            totals,
        })
    }

    /// Fail unless `tensors_seen` equals the expected source total (e.g. 72_317).
    pub fn assert_complete(&self, expected_total: usize) -> Result<(), String> {
        if self.totals.tensors_seen != expected_total {
            return Err(format!(
                "deepseek4 parent: inventory tensors_seen = {}, expected {expected_total}",
                self.totals.tensors_seen
            ));
        }
        Ok(())
    }
}

fn scale_name_for_weight(weight: &str) -> Option<String> {
    weight
        .strip_suffix(".weight")
        .map(|stem| format!("{stem}.scale"))
}

fn scale_to_weight_name(scale: &str) -> String {
    match scale.strip_suffix(".scale") {
        Some(stem) => format!("{stem}.weight"),
        None => format!("{scale}.weight"),
    }
}

fn is_routed_expert_weight(name: &str) -> bool {
    // `...ffn.experts.<e>.w{1,2,3}.weight` (optionally under `mtp.{m}.`)
    if !name.contains("ffn.experts.") || !name.ends_with(".weight") {
        return false;
    }
    let Some(rest) = name.split("ffn.experts.").nth(1) else {
        return false;
    };
    let mut parts = rest.split('.');
    let Some(expert_idx) = parts.next() else {
        return false;
    };
    if expert_idx.is_empty() || !expert_idx.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let Some(w) = parts.next() else {
        return false;
    };
    if w != "w1" && w != "w2" && w != "w3" {
        return false;
    }
    let Some(tail) = parts.next() else {
        return false;
    };
    tail == "weight" && parts.next().is_none()
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn classify_dense_fp8(
    name: &str,
    shape: &[usize],
    infos: &HashMap<String, (String, Vec<usize>, u64)>,
    scale_claimed: &mut HashSet<String>,
) -> Result<ParentTensorEntry, String> {
    if !name.ends_with(".weight") {
        return Err(format!(
            "deepseek4 parent: F8_E4M3 tensor {name:?} is not a .weight; dense FP8 weights must end in .weight"
        ));
    }
    if shape.len() != 2 {
        return Err(format!(
            "deepseek4 parent: F8_E4M3 weight {name:?} has shape {shape:?}, expected rank-2 [M, K]"
        ));
    }
    let m = shape[0];
    let k = shape[1];
    let [bm, bk] = PARENT_WEIGHT_BLOCK;
    let expect_scale = vec![ceil_div(m, bm), ceil_div(k, bk)];

    let scale_name = scale_name_for_weight(name).expect("checked .weight suffix");
    let (scale_dtype, scale_shape, _) = infos.get(&scale_name).ok_or_else(|| {
        format!(
            "deepseek4 parent: F8_E4M3 weight {name:?} is missing required scale companion {scale_name:?}"
        )
    })?;
    if scale_dtype != "F8_E8M0" {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} for F8_E4M3 weight {name:?} has dtype {scale_dtype:?}, expected F8_E8M0"
        ));
    }
    if scale_shape.as_slice() != expect_scale.as_slice() {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} for F8_E4M3 weight {name:?} has shape {scale_shape:?}, \
             expected {expect_scale:?} (= ceil([M,K]/{PARENT_WEIGHT_BLOCK:?}) for weight shape {shape:?})"
        ));
    }
    if !scale_claimed.insert(scale_name.clone()) {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} claimed by multiple weights"
        ));
    }

    let is_mtp = name.starts_with("mtp.");
    Ok(ParentTensorEntry {
        name: name.to_owned(),
        class: ParentTensorClass::DenseFp8,
        dtype: "F8_E4M3".to_owned(),
        shape: shape.to_vec(),
        logical_shape: shape.to_vec(),
        scale: Some(ParentScaleRef {
            name: scale_name,
            shape: expect_scale,
        }),
        is_mtp,
    })
}

fn classify_expert_fp4(
    name: &str,
    shape: &[usize],
    infos: &HashMap<String, (String, Vec<usize>, u64)>,
    scale_claimed: &mut HashSet<String>,
) -> Result<ParentTensorEntry, String> {
    if !is_routed_expert_weight(name) {
        return Err(format!(
            "deepseek4 parent: I8 tensor {name:?} is not a routed-expert weight \
             (expected ...ffn.experts.<e>.w{{1,2,3}}.weight)"
        ));
    }
    if shape.len() != 2 {
        return Err(format!(
            "deepseek4 parent: I8 expert weight {name:?} has shape {shape:?}, expected rank-2 [M, Kpacked]"
        ));
    }
    let m = shape[0];
    let k_packed = shape[1];
    let logical_k = k_packed.checked_mul(2).ok_or_else(|| {
        format!(
            "deepseek4 parent: I8 expert weight {name:?} Kpacked overflow doubling to logical K"
        )
    })?;
    // Scale shape [M, (2*Kpacked)/32] == [M, logical_k / PARENT_EXPERT_SCALE_GROUP]
    if logical_k % PARENT_EXPERT_SCALE_GROUP != 0 {
        return Err(format!(
            "deepseek4 parent: I8 expert weight {name:?} logical K={logical_k} is not divisible by \
             expert scale group {PARENT_EXPERT_SCALE_GROUP}"
        ));
    }
    let expect_scale = vec![m, logical_k / PARENT_EXPERT_SCALE_GROUP];

    let scale_name = scale_name_for_weight(name).expect("expert weight ends in .weight");
    let (scale_dtype, scale_shape, _) = infos.get(&scale_name).ok_or_else(|| {
        format!(
            "deepseek4 parent: I8 expert weight {name:?} is missing required scale companion {scale_name:?}"
        )
    })?;
    if scale_dtype != "F8_E8M0" {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} for I8 expert weight {name:?} has dtype {scale_dtype:?}, expected F8_E8M0"
        ));
    }
    if scale_shape.as_slice() != expect_scale.as_slice() {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} for I8 expert weight {name:?} has shape {scale_shape:?}, \
             expected {expect_scale:?} (= [M, (2*Kpacked)/{PARENT_EXPERT_SCALE_GROUP}] for packed shape {shape:?})"
        ));
    }
    if !scale_claimed.insert(scale_name.clone()) {
        return Err(format!(
            "deepseek4 parent: scale {scale_name:?} claimed by multiple weights"
        ));
    }

    let is_mtp = name.starts_with("mtp.");
    Ok(ParentTensorEntry {
        name: name.to_owned(),
        class: ParentTensorClass::ExpertFp4,
        dtype: "I8".to_owned(),
        shape: shape.to_vec(),
        logical_shape: vec![m, logical_k],
        scale: Some(ParentScaleRef {
            name: scale_name,
            shape: expect_scale,
        }),
        is_mtp,
    })
}

fn reject_scale_sibling(
    name: &str,
    infos: &HashMap<String, (String, Vec<usize>, u64)>,
) -> Result<(), String> {
    let Some(scale_name) = scale_name_for_weight(name) else {
        // Non-`.weight` unquantized tensors (attn_sink, hc_*, ape, …) have no
        // scale naming convention; nothing to reject.
        return Ok(());
    };
    if infos.contains_key(&scale_name) {
        return Err(format!(
            "deepseek4 parent: unquantized tensor {name:?} must not carry scale companion {scale_name:?}"
        ));
    }
    Ok(())
}

fn bump_class(
    totals: &mut ParentInventoryTotals,
    class: ParentTensorClass,
    is_mtp: bool,
    weight_nbytes: u64,
) {
    if is_mtp {
        return;
    }
    match class {
        ParentTensorClass::DenseFp8 => totals.dense_fp8 += 1,
        ParentTensorClass::ExpertFp4 => totals.expert_fp4 += 1,
        ParentTensorClass::Bf16 => totals.bf16 += 1,
        ParentTensorClass::F32 => totals.f32 += 1,
        ParentTensorClass::I64 => totals.i64 += 1,
    }
    totals.payload_bytes = totals.payload_bytes.saturating_add(weight_nbytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::model_source::{ModelSource, QuantConfig, TensorInfo};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// In-memory `ModelSource` fixture for inventory unit tests.
    struct FixtureSource {
        tensors: BTreeMap<String, TensorInfo>,
        meta: String,
    }

    impl FixtureSource {
        fn new(pairs: Vec<(&str, &str, Vec<usize>)>) -> Self {
            let mut tensors = BTreeMap::new();
            let mut offset = 0usize;
            for (name, dtype, shape) in pairs {
                let elem = dtype_nbytes(dtype);
                let n_elem: usize = shape.iter().product();
                let data_size = n_elem.saturating_mul(elem);
                tensors.insert(
                    name.to_owned(),
                    TensorInfo {
                        name: name.to_owned(),
                        dtype: dtype.to_owned(),
                        shape,
                        quant_type: 0xFF,
                        data_offset: offset,
                        data_size,
                    },
                );
                offset += data_size;
            }
            Self {
                tensors,
                meta: String::new(),
            }
        }
    }

    fn dtype_nbytes(dtype: &str) -> usize {
        match dtype {
            "F8_E4M3" | "F8_E8M0" | "I8" => 1,
            "BF16" | "F16" => 2,
            "F32" => 4,
            "I64" => 8,
            // Unknown dtypes still need a size so the fixture can surface the
            // inventory's rule-5 error instead of panicking in test setup.
            _ => 1,
        }
    }

    impl ModelSource for FixtureSource {
        fn metadata_json(&self) -> &str {
            &self.meta
        }
        fn arch_id(&self) -> u32 {
            0
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
        fn tensor_data(&self, _name: &str) -> Option<(&TensorInfo, &[u8])> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.tensors.keys().map(String::as_str).collect()
        }
        fn path(&self) -> &Path {
            Path::new("/tmp/ds4-parent-inventory-fixture")
        }
    }

    fn test_cfg() -> ParentQuantConfig {
        ParentQuantConfig {
            model_type: crate::PARENT_MODEL_TYPE.to_owned(),
            quant_method: crate::PARENT_QUANT_METHOD.to_owned(),
            fmt: crate::PARENT_WEIGHT_FMT.to_owned(),
            scale_fmt: crate::PARENT_SCALE_FMT.to_owned(),
            expert_dtype: crate::PARENT_EXPERT_DTYPE.to_owned(),
            weight_block_size: crate::PARENT_WEIGHT_BLOCK,
            num_hidden_layers: 2,
            num_hash_layers: 1,
            n_routed_experts: 2,
            num_experts_per_tok: 2,
            compress_ratios: vec![0, 4],
        }
    }

    /// Minimal happy-path fixture: one dense FP8, one expert FP4, one of each
    /// unquantized class, plus an MTP dense pair that must be excluded.
    fn happy_pairs() -> Vec<(&'static str, &'static str, Vec<usize>)> {
        vec![
            ("embed.weight", "BF16", vec![128, 64]),
            ("layers.0.attn.wq_a.weight", "F8_E4M3", vec![256, 128]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![2, 1]), // ceil(256/128)=2, ceil(128/128)=1
            (
                "layers.0.ffn.experts.0.w1.weight",
                "I8",
                vec![64, 32], // logical K = 64
            ),
            (
                "layers.0.ffn.experts.0.w1.scale",
                "F8_E8M0",
                vec![64, 2], // [M, logical_k/32] = [64, 64/32]
            ),
            ("layers.0.attn.attn_sink", "F32", vec![8]),
            ("layers.0.ffn.gate.tid2eid", "I64", vec![16, 2]),
            // MTP — validated then excluded
            ("mtp.0.attn.wq_a.weight", "F8_E4M3", vec![128, 128]),
            ("mtp.0.attn.wq_a.scale", "F8_E8M0", vec![1, 1]),
            ("mtp.0.norm.weight", "BF16", vec![64]),
        ]
    }

    #[test]
    fn happy_path_classifies_and_excludes_mtp() {
        let src = FixtureSource::new(happy_pairs());
        let inv = ParentInventory::build(&src, &test_cfg()).expect("happy path");

        assert_eq!(inv.totals.tensors_seen, happy_pairs().len());
        inv.assert_complete(happy_pairs().len()).unwrap();

        // Main tower: embed, dense w, expert w, sink, tid2eid = 5 primary entries
        // (scales are companions, not entries).
        assert_eq!(inv.entries.len(), 5);
        assert!(inv.entries.iter().all(|e| !e.is_mtp));

        assert_eq!(inv.totals.dense_fp8, 1);
        assert_eq!(inv.totals.expert_fp4, 1);
        assert_eq!(inv.totals.bf16, 1);
        assert_eq!(inv.totals.f32, 1);
        assert_eq!(inv.totals.i64, 1);

        // MTP names excluded: weight, scale, norm
        assert_eq!(inv.totals.mtp_excluded, 3);
        assert_eq!(
            inv.excluded_mtp,
            vec![
                "mtp.0.attn.wq_a.scale".to_owned(),
                "mtp.0.attn.wq_a.weight".to_owned(),
                "mtp.0.norm.weight".to_owned(),
            ]
        );

        let dense = inv
            .entries
            .iter()
            .find(|e| e.name == "layers.0.attn.wq_a.weight")
            .expect("dense entry");
        assert_eq!(dense.class, ParentTensorClass::DenseFp8);
        assert_eq!(dense.logical_shape, dense.shape);
        let scale = dense.scale.as_ref().expect("dense scale");
        assert_eq!(scale.name, "layers.0.attn.wq_a.scale");
        assert_eq!(scale.shape, vec![2, 1]);

        let expert = inv
            .entries
            .iter()
            .find(|e| e.name == "layers.0.ffn.experts.0.w1.weight")
            .expect("expert entry");
        assert_eq!(expert.class, ParentTensorClass::ExpertFp4);
        assert_eq!(expert.shape, vec![64, 32]);
        assert_eq!(expert.logical_shape, vec![64, 64]);
        let es = expert.scale.as_ref().expect("expert scale");
        assert_eq!(es.shape, vec![64, 2]);

        // Payload = main weights + main scales only.
        let expected_payload = {
            let main = [
                ("embed.weight", "BF16", vec![128usize, 64]),
                ("layers.0.attn.wq_a.weight", "F8_E4M3", vec![256, 128]),
                ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![2, 1]),
                ("layers.0.ffn.experts.0.w1.weight", "I8", vec![64, 32]),
                ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", vec![64, 2]),
                ("layers.0.attn.attn_sink", "F32", vec![8]),
                ("layers.0.ffn.gate.tid2eid", "I64", vec![16, 2]),
            ];
            main.iter()
                .map(|(_, d, s)| {
                    let n: usize = s.iter().product();
                    (n * dtype_nbytes(d)) as u64
                })
                .sum::<u64>()
        };
        assert_eq!(inv.totals.payload_bytes, expected_payload);
    }

    #[test]
    fn rule1_missing_dense_scale_errors() {
        let mut pairs = happy_pairs();
        pairs.retain(|(n, _, _)| *n != "layers.0.attn.wq_a.scale");
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("missing scale");
        assert!(
            err.contains("missing required scale companion") && err.contains("wq_a.weight"),
            "unexpected err: {err}"
        );
        assert!(err.starts_with("deepseek4 parent:"), "{err}");
    }

    #[test]
    fn rule1_wrong_dense_scale_shape_errors() {
        let mut pairs = happy_pairs();
        for (n, _, s) in &mut pairs {
            if *n == "layers.0.attn.wq_a.scale" {
                *s = vec![1, 1]; // wrong: should be [2,1]
            }
        }
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("bad scale shape");
        assert!(
            err.contains("layers.0.attn.wq_a.scale") && err.contains("expected"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule1_wrong_dense_scale_dtype_errors() {
        let mut pairs = happy_pairs();
        for (n, d, _) in &mut pairs {
            if *n == "layers.0.attn.wq_a.scale" {
                *d = "F32";
            }
        }
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("bad scale dtype");
        assert!(
            err.contains("expected F8_E8M0") && err.contains("wq_a.scale"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule2_non_expert_i8_errors() {
        let mut pairs = happy_pairs();
        pairs.push(("layers.0.ffn.shared_experts.w1.weight", "I8", vec![64, 32]));
        pairs.push((
            "layers.0.ffn.shared_experts.w1.scale",
            "F8_E8M0",
            vec![64, 2],
        ));
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("non-expert I8");
        assert!(
            err.contains("not a routed-expert weight")
                && err.contains("shared_experts.w1.weight"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule2_missing_expert_scale_errors() {
        let mut pairs = happy_pairs();
        pairs.retain(|(n, _, _)| *n != "layers.0.ffn.experts.0.w1.scale");
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("missing expert scale");
        assert!(
            err.contains("missing required scale companion")
                && err.contains("experts.0.w1.weight"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule2_wrong_expert_scale_shape_errors() {
        let mut pairs = happy_pairs();
        for (n, _, s) in &mut pairs {
            if *n == "layers.0.ffn.experts.0.w1.scale" {
                *s = vec![64, 1]; // wrong: should be [64,2]
            }
        }
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("bad expert scale shape");
        assert!(
            err.contains("experts.0.w1.scale") && err.contains("expected"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule2_expert_logical_shape_doubles_k() {
        let src = FixtureSource::new(happy_pairs());
        let inv = ParentInventory::build(&src, &test_cfg()).unwrap();
        let e = inv
            .entries
            .iter()
            .find(|e| e.class == ParentTensorClass::ExpertFp4)
            .unwrap();
        assert_eq!(e.shape, vec![64, 32]);
        assert_eq!(e.logical_shape, vec![64, 64]);
    }

    #[test]
    fn rule3_unquantized_with_scale_errors() {
        let mut pairs = happy_pairs();
        pairs.push(("embed.scale", "F8_E8M0", vec![1, 1]));
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("bf16 with scale");
        assert!(
            err.contains("unquantized tensor") && err.contains("embed.weight"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule4_orphan_scale_errors() {
        let mut pairs = happy_pairs();
        pairs.push(("layers.9.attn.wo_a.scale", "F8_E8M0", vec![1, 1]));
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("orphan scale");
        assert!(
            err.contains("orphan scale") && err.contains("layers.9.attn.wo_a.scale"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn rule5_unknown_dtype_errors() {
        let mut pairs = happy_pairs();
        pairs.push(("layers.0.mystery", "F16", vec![4]));
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("unknown dtype");
        assert!(
            err.contains("unsupported dtype")
                && err.contains("F16")
                && err.contains("layers.0.mystery"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn assert_complete_rejects_mismatch() {
        let src = FixtureSource::new(happy_pairs());
        let inv = ParentInventory::build(&src, &test_cfg()).unwrap();
        let err = inv.assert_complete(72_317).expect_err("mismatch");
        assert!(err.contains("tensors_seen"), "{err}");
    }

    #[test]
    fn broken_mtp_still_fails_closed() {
        // MTP dense weight without its scale must fail even though MTP is excluded.
        let mut pairs = happy_pairs();
        pairs.retain(|(n, _, _)| *n != "mtp.0.attn.wq_a.scale");
        let src = FixtureSource::new(pairs);
        let err = ParentInventory::build(&src, &test_cfg()).expect_err("broken mtp");
        assert!(
            err.contains("mtp.0.attn.wq_a.weight") && err.contains("missing required scale"),
            "unexpected err: {err}"
        );
    }
}
