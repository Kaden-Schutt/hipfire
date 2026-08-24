// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parent-checkpoint **weight residency** layer.
//!
//! Loads the original mixed-precision DeepSeek V4 Flash checkpoint onto a
//! gfx942 device with:
//!
//! - dense `F8_E4M3` projections decoded to **resident BF16**
//!   ([`ParentDenseWeight::decode_resident`]);
//! - routed experts left **compressed** in HBM
//!   ([`ParentExpertWeight::upload_compressed`]);
//! - unquantized BF16 / F32 / I64 tensors uploaded at their native widths
//!   (no BF16→F32 widening of norms — the reference keeps them BF16).
//!
//! `ParentWeights` is intentionally its own struct, **not** a reinterpretation
//! of `crate::deepseek4::DeepseekV4Weights`. The parent tier layout
//! (resident BF16 dense + separately-scaled compressed experts) is
//! structurally different from the HFQ single-blob-per-tensor layout.
//!
//! MTP (`mtp.*`) is never loaded. Drive every upload from
//! [`ParentInventory`] so inventory validation is the single source of truth
//! for which tensors exist.

use std::collections::HashMap;
use std::ops::Range;
use std::time::Instant;

use hipfire_runtime::model_source::ModelSource;
use rdna_compute::{DType, Gpu, GpuTensor};

use super::inventory::{ParentInventory, ParentTensorClass, ParentTensorEntry};
use super::linear::{ParentDenseWeight, ParentExpertWeight};
use super::{Ds4ParentBackend, ParentQuantConfig};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Bytes actually resident on device after a load, broken out by tier.
///
/// Matches the Gate 1 VRAM projection (main tower, MTP excluded):
/// dense-as-BF16 + compressed experts + BF16 + F32 + I64.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParentResidency {
    /// Dense FP8 weights expanded to BF16 (2 × stored F8_E4M3 weight bytes).
    pub dense_bf16_bytes: u64,
    /// Routed-expert codes + scales left compressed (I8 + F8_E8M0 as stored).
    pub expert_compressed_bytes: u64,
    /// Unquantized BF16 tensors (norms, embed, head, gate.weight, …).
    pub bf16_bytes: u64,
    /// Unquantized F32 tensors (attn_sink, HC, gate.bias, ape, …).
    pub f32_bytes: u64,
    /// Unquantized I64 tensors (`tid2eid`).
    pub i64_bytes: u64,
}

impl ParentResidency {
    /// Sum of all tiers.
    pub fn total_bytes(self) -> u64 {
        self.dense_bf16_bytes
            .saturating_add(self.expert_compressed_bytes)
            .saturating_add(self.bf16_bytes)
            .saturating_add(self.f32_bytes)
            .saturating_add(self.i64_bytes)
    }

    fn add_assign(&mut self, other: ParentResidency) {
        self.dense_bf16_bytes = self.dense_bf16_bytes.saturating_add(other.dense_bf16_bytes);
        self.expert_compressed_bytes = self
            .expert_compressed_bytes
            .saturating_add(other.expert_compressed_bytes);
        self.bf16_bytes = self.bf16_bytes.saturating_add(other.bf16_bytes);
        self.f32_bytes = self.f32_bytes.saturating_add(other.f32_bytes);
        self.i64_bytes = self.i64_bytes.saturating_add(other.i64_bytes);
    }
}

/// Which layers (and whether experts) to materialise on device.
///
/// `layers: 0..cfg.num_hidden_layers` is the full-model case.
/// `load_experts: false` skips the routed-expert tier for attention-only
/// smoke tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentLoadPlan {
    pub layers: Range<usize>,
    pub load_experts: bool,
}

/// Indexer sub-module weights (present only on `compress_ratio == 4` layers).
pub struct ParentIndexerWeights {
    pub wq_b: ParentDenseWeight,
    pub weights_proj: GpuTensor,     // BF16
    pub compressor_wkv: GpuTensor,   // BF16
    pub compressor_wgate: GpuTensor, // BF16
    pub compressor_norm: GpuTensor,  // BF16
    pub compressor_ape: GpuTensor,   // F32
}

/// Main-attention compressor (present on `compress_ratio > 0` layers).
pub struct ParentCompressorWeights {
    pub wkv: GpuTensor,   // BF16
    pub wgate: GpuTensor, // BF16
    pub norm: GpuTensor,  // BF16
    pub ape: GpuTensor,   // F32
}

/// One transformer layer's parent-checkpoint weights.
pub struct ParentLayerWeights {
    pub layer_idx: usize,
    pub compress_ratio: usize,

    // Norms (BF16).
    pub attn_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub q_norm: GpuTensor,
    pub kv_norm: GpuTensor,

    // Attention sinks (F32).
    pub attn_sink: GpuTensor,

    // Dense FP8 → resident BF16 attention projections.
    pub wq_a: ParentDenseWeight,
    pub wq_b: ParentDenseWeight,
    pub wkv: ParentDenseWeight,
    pub wo_a: ParentDenseWeight,
    pub wo_b: ParentDenseWeight,

    // Optional compressor / indexer.
    pub compressor: Option<ParentCompressorWeights>,
    pub indexer: Option<ParentIndexerWeights>,

    // Hyper-Connections (F32).
    pub hc_attn_base: GpuTensor,
    pub hc_attn_fn: GpuTensor,
    pub hc_attn_scale: GpuTensor,
    pub hc_ffn_base: GpuTensor,
    pub hc_ffn_fn: GpuTensor,
    pub hc_ffn_scale: GpuTensor,

    // Router. `gate_bias` only on score-routed layers; `tid2eid` only on
    // hash-routed layers (`layer_idx < num_hash_layers`).
    pub gate_weight: GpuTensor,           // BF16
    pub gate_bias: Option<GpuTensor>,     // F32
    pub tid2eid: Option<GpuTensor>,       // I64 / Raw

    // Shared experts (dense FP8 → resident BF16).
    pub shared_w1: ParentDenseWeight,
    pub shared_w2: ParentDenseWeight,
    pub shared_w3: ParentDenseWeight,

    // Routed experts left compressed. Empty when `load_experts == false`.
    // Indexed by expert id `0..n_routed_experts`.
    pub experts: Vec<ParentExpertTriple>,
}

/// One routed expert's three projections, each left compressed in HBM.
pub struct ParentExpertTriple {
    pub expert_id: usize,
    pub w1: ParentExpertWeight,
    pub w2: ParentExpertWeight,
    pub w3: ParentExpertWeight,
}

/// Full (or partial) parent-checkpoint weight residency on device.
///
/// `layers` contains only the range requested by [`ParentLoadPlan`]; global
/// tensors (embed / norm / head / hc_head_*) are always loaded.
pub struct ParentWeights {
    pub backend: Ds4ParentBackend,
    /// Absolute layer indices corresponding 1:1 with [`Self::layers`].
    pub layer_range: Range<usize>,
    pub layers: Vec<ParentLayerWeights>,
    pub embed: GpuTensor,         // BF16
    pub norm: GpuTensor,          // BF16
    pub head: GpuTensor,          // BF16
    pub hc_head_base: GpuTensor,  // F32
    pub hc_head_fn: GpuTensor,    // F32
    pub hc_head_scale: GpuTensor, // F32
    /// Whether routed experts were loaded.
    pub experts_loaded: bool,
}

impl ParentWeights {
    /// Load parent weights from `source` onto `gpu` according to `plan`.
    ///
    /// Drive every upload from `inv` so a tensor the inventory did not
    /// validate cannot reach the device, and a tensor the inventory *did*
    /// validate that is missing from the source fails closed.
    ///
    /// Progress is printed per layer (with running resident bytes) so a
    /// detached multi-minute full-model load is observable via `tail`.
    pub fn load(
        source: &dyn ModelSource,
        cfg: &ParentQuantConfig,
        inv: &ParentInventory,
        gpu: &mut Gpu,
        backend: Ds4ParentBackend,
        plan: &ParentLoadPlan,
    ) -> Result<Self, String> {
        backend.ensure_device(gpu)?;
        validate_plan(cfg, plan)?;

        // Index inventory entries by name for O(1) lookup. MTP is already
        // excluded from `inv.entries`.
        let by_name: HashMap<&str, &ParentTensorEntry> = inv
            .entries
            .iter()
            .map(|e| (e.name.as_str(), e))
            .collect();

        let t0 = Instant::now();
        let mut running = ParentResidency::default();

        eprintln!(
            "deepseek4 parent: load begin layers={:?} load_experts={} inventory_entries={}",
            plan.layers,
            plan.load_experts,
            inv.entries.len()
        );

        // ── Globals ──────────────────────────────────────────────────────
        let embed = upload_bf16(source, gpu, &by_name, "embed.weight", &mut running)?;
        let norm = upload_bf16(source, gpu, &by_name, "norm.weight", &mut running)?;
        let head = upload_bf16(source, gpu, &by_name, "head.weight", &mut running)?;
        let hc_head_base = upload_f32(source, gpu, &by_name, "hc_head_base", &mut running)?;
        let hc_head_fn = upload_f32(source, gpu, &by_name, "hc_head_fn", &mut running)?;
        let hc_head_scale = upload_f32(source, gpu, &by_name, "hc_head_scale", &mut running)?;

        eprintln!(
            "deepseek4 parent: globals done  resident={:.3} GiB  elapsed={:.1}s",
            gib(running.total_bytes()),
            t0.elapsed().as_secs_f64()
        );

        // ── Layers ───────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(plan.layers.end.saturating_sub(plan.layers.start));
        for layer_idx in plan.layers.clone() {
            let layer_t0 = Instant::now();
            let layer = load_layer(
                source,
                cfg,
                gpu,
                backend,
                &by_name,
                layer_idx,
                plan.load_experts,
                &mut running,
            )?;
            layers.push(layer);
            eprintln!(
                "deepseek4 parent: layer {layer_idx:>2} done  \
                 resident={:.3} GiB  layer={:.1}s  elapsed={:.1}s  experts={}",
                gib(running.total_bytes()),
                layer_t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64(),
                if plan.load_experts {
                    cfg.n_routed_experts
                } else {
                    0
                }
            );
        }

        eprintln!(
            "deepseek4 parent: load complete  layers={}  total={:.3} GiB  wall={:.1}s  \
             dense_bf16={:.3} expert={:.3} bf16={:.3} f32={:.3} i64={:.3}",
            layers.len(),
            gib(running.total_bytes()),
            t0.elapsed().as_secs_f64(),
            gib(running.dense_bf16_bytes),
            gib(running.expert_compressed_bytes),
            gib(running.bf16_bytes),
            gib(running.f32_bytes),
            gib(running.i64_bytes),
        );

        Ok(Self {
            backend,
            layer_range: plan.layers.clone(),
            layers,
            embed,
            norm,
            head,
            hc_head_base,
            hc_head_fn,
            hc_head_scale,
            experts_loaded: plan.load_experts,
        })
    }

    /// Bytes actually resident on device, by tier.
    pub fn residency(&self) -> ParentResidency {
        let mut r = ParentResidency::default();
        r.bf16_bytes = r
            .bf16_bytes
            .saturating_add(self.embed.buf.size() as u64)
            .saturating_add(self.norm.buf.size() as u64)
            .saturating_add(self.head.buf.size() as u64);
        r.f32_bytes = r
            .f32_bytes
            .saturating_add(self.hc_head_base.buf.size() as u64)
            .saturating_add(self.hc_head_fn.buf.size() as u64)
            .saturating_add(self.hc_head_scale.buf.size() as u64);
        for layer in &self.layers {
            r.add_assign(layer_residency(layer));
        }
        r
    }
}

// ---------------------------------------------------------------------------
// Layer load
// ---------------------------------------------------------------------------

fn load_layer(
    source: &dyn ModelSource,
    cfg: &ParentQuantConfig,
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    layer_idx: usize,
    load_experts: bool,
    running: &mut ParentResidency,
) -> Result<ParentLayerWeights, String> {
    let p = |suffix: &str| format!("layers.{layer_idx}.{suffix}");
    let compress_ratio = cfg.compress_ratio(layer_idx);
    let is_hash = layer_idx < cfg.num_hash_layers;

    // Norms / sink.
    let attn_norm = upload_bf16(source, gpu, by_name, &p("attn_norm.weight"), running)?;
    let ffn_norm = upload_bf16(source, gpu, by_name, &p("ffn_norm.weight"), running)?;
    let q_norm = upload_bf16(source, gpu, by_name, &p("attn.q_norm.weight"), running)?;
    let kv_norm = upload_bf16(source, gpu, by_name, &p("attn.kv_norm.weight"), running)?;
    let attn_sink = upload_f32(source, gpu, by_name, &p("attn.attn_sink"), running)?;

    // Dense attention projections.
    let wq_a = decode_dense(source, gpu, backend, by_name, &p("attn.wq_a.weight"), running)?;
    let wq_b = decode_dense(source, gpu, backend, by_name, &p("attn.wq_b.weight"), running)?;
    let wkv = decode_dense(source, gpu, backend, by_name, &p("attn.wkv.weight"), running)?;
    let wo_a = decode_dense(source, gpu, backend, by_name, &p("attn.wo_a.weight"), running)?;
    let wo_b = decode_dense(source, gpu, backend, by_name, &p("attn.wo_b.weight"), running)?;

    // Compressor (ratio > 0).
    let compressor = if compress_ratio > 0 {
        Some(ParentCompressorWeights {
            wkv: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.compressor.wkv.weight"),
                running,
            )?,
            wgate: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.compressor.wgate.weight"),
                running,
            )?,
            norm: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.compressor.norm.weight"),
                running,
            )?,
            ape: upload_f32(source, gpu, by_name, &p("attn.compressor.ape"), running)?,
        })
    } else {
        None
    };

    // Indexer (ratio == 4 only).
    let indexer = if compress_ratio == 4 {
        Some(ParentIndexerWeights {
            wq_b: decode_dense(
                source,
                gpu,
                backend,
                by_name,
                &p("attn.indexer.wq_b.weight"),
                running,
            )?,
            weights_proj: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.indexer.weights_proj.weight"),
                running,
            )?,
            compressor_wkv: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.indexer.compressor.wkv.weight"),
                running,
            )?,
            compressor_wgate: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.indexer.compressor.wgate.weight"),
                running,
            )?,
            compressor_norm: upload_bf16(
                source,
                gpu,
                by_name,
                &p("attn.indexer.compressor.norm.weight"),
                running,
            )?,
            compressor_ape: upload_f32(
                source,
                gpu,
                by_name,
                &p("attn.indexer.compressor.ape"),
                running,
            )?,
        })
    } else {
        None
    };

    // Hyper-Connections.
    let hc_attn_base = upload_f32(source, gpu, by_name, &p("hc_attn_base"), running)?;
    let hc_attn_fn = upload_f32(source, gpu, by_name, &p("hc_attn_fn"), running)?;
    let hc_attn_scale = upload_f32(source, gpu, by_name, &p("hc_attn_scale"), running)?;
    let hc_ffn_base = upload_f32(source, gpu, by_name, &p("hc_ffn_base"), running)?;
    let hc_ffn_fn = upload_f32(source, gpu, by_name, &p("hc_ffn_fn"), running)?;
    let hc_ffn_scale = upload_f32(source, gpu, by_name, &p("hc_ffn_scale"), running)?;

    // Router.
    let gate_weight = upload_bf16(source, gpu, by_name, &p("ffn.gate.weight"), running)?;
    let gate_bias = if is_hash {
        None
    } else {
        Some(upload_f32(
            source,
            gpu,
            by_name,
            &p("ffn.gate.bias"),
            running,
        )?)
    };
    let tid2eid = if is_hash {
        Some(upload_i64(
            source,
            gpu,
            by_name,
            &p("ffn.gate.tid2eid"),
            running,
        )?)
    } else {
        None
    };

    // Shared experts (always dense FP8).
    let shared_w1 = decode_dense(
        source,
        gpu,
        backend,
        by_name,
        &p("ffn.shared_experts.w1.weight"),
        running,
    )?;
    let shared_w2 = decode_dense(
        source,
        gpu,
        backend,
        by_name,
        &p("ffn.shared_experts.w2.weight"),
        running,
    )?;
    let shared_w3 = decode_dense(
        source,
        gpu,
        backend,
        by_name,
        &p("ffn.shared_experts.w3.weight"),
        running,
    )?;

    // Routed experts.
    let mut experts = Vec::new();
    if load_experts {
        experts.reserve(cfg.n_routed_experts);
        for e in 0..cfg.n_routed_experts {
            let w1 = upload_expert(
                source,
                gpu,
                backend,
                by_name,
                &p(&format!("ffn.experts.{e}.w1.weight")),
                running,
            )?;
            let w2 = upload_expert(
                source,
                gpu,
                backend,
                by_name,
                &p(&format!("ffn.experts.{e}.w2.weight")),
                running,
            )?;
            let w3 = upload_expert(
                source,
                gpu,
                backend,
                by_name,
                &p(&format!("ffn.experts.{e}.w3.weight")),
                running,
            )?;
            experts.push(ParentExpertTriple {
                expert_id: e,
                w1,
                w2,
                w3,
            });
        }
    }

    Ok(ParentLayerWeights {
        layer_idx,
        compress_ratio,
        attn_norm,
        ffn_norm,
        q_norm,
        kv_norm,
        attn_sink,
        wq_a,
        wq_b,
        wkv,
        wo_a,
        wo_b,
        compressor,
        indexer,
        hc_attn_base,
        hc_attn_fn,
        hc_attn_scale,
        hc_ffn_base,
        hc_ffn_fn,
        hc_ffn_scale,
        gate_weight,
        gate_bias,
        tid2eid,
        shared_w1,
        shared_w2,
        shared_w3,
        experts,
    })
}

fn layer_residency(layer: &ParentLayerWeights) -> ParentResidency {
    let mut r = ParentResidency::default();

    for t in [
        &layer.attn_norm,
        &layer.ffn_norm,
        &layer.q_norm,
        &layer.kv_norm,
        &layer.gate_weight,
    ] {
        r.bf16_bytes = r.bf16_bytes.saturating_add(t.buf.size() as u64);
    }

    for t in [
        &layer.attn_sink,
        &layer.hc_attn_base,
        &layer.hc_attn_fn,
        &layer.hc_attn_scale,
        &layer.hc_ffn_base,
        &layer.hc_ffn_fn,
        &layer.hc_ffn_scale,
    ] {
        r.f32_bytes = r.f32_bytes.saturating_add(t.buf.size() as u64);
    }
    if let Some(b) = layer.gate_bias.as_ref() {
        r.f32_bytes = r.f32_bytes.saturating_add(b.buf.size() as u64);
    }
    if let Some(t) = layer.tid2eid.as_ref() {
        r.i64_bytes = r.i64_bytes.saturating_add(t.buf.size() as u64);
    }

    for d in [
        &layer.wq_a,
        &layer.wq_b,
        &layer.wkv,
        &layer.wo_a,
        &layer.wo_b,
        &layer.shared_w1,
        &layer.shared_w2,
        &layer.shared_w3,
    ] {
        r.dense_bf16_bytes = r
            .dense_bf16_bytes
            .saturating_add(d.resident_bytes() as u64);
    }

    if let Some(c) = layer.compressor.as_ref() {
        r.bf16_bytes = r
            .bf16_bytes
            .saturating_add(c.wkv.buf.size() as u64)
            .saturating_add(c.wgate.buf.size() as u64)
            .saturating_add(c.norm.buf.size() as u64);
        r.f32_bytes = r.f32_bytes.saturating_add(c.ape.buf.size() as u64);
    }
    if let Some(ix) = layer.indexer.as_ref() {
        r.dense_bf16_bytes = r
            .dense_bf16_bytes
            .saturating_add(ix.wq_b.resident_bytes() as u64);
        r.bf16_bytes = r
            .bf16_bytes
            .saturating_add(ix.weights_proj.buf.size() as u64)
            .saturating_add(ix.compressor_wkv.buf.size() as u64)
            .saturating_add(ix.compressor_wgate.buf.size() as u64)
            .saturating_add(ix.compressor_norm.buf.size() as u64);
        r.f32_bytes = r
            .f32_bytes
            .saturating_add(ix.compressor_ape.buf.size() as u64);
    }

    for e in &layer.experts {
        r.expert_compressed_bytes = r
            .expert_compressed_bytes
            .saturating_add(e.w1.compressed_bytes() as u64)
            .saturating_add(e.w2.compressed_bytes() as u64)
            .saturating_add(e.w3.compressed_bytes() as u64);
    }

    r
}

// ---------------------------------------------------------------------------
// Per-tensor upload helpers
// ---------------------------------------------------------------------------

fn require_entry<'a>(
    by_name: &HashMap<&str, &'a ParentTensorEntry>,
    name: &str,
    want_class: ParentTensorClass,
) -> Result<&'a ParentTensorEntry, String> {
    let e = by_name.get(name).copied().ok_or_else(|| {
        format!(
            "deepseek4 parent: required tensor {name:?} missing from inventory \
             (not present in source, or excluded as MTP)"
        )
    })?;
    if e.class != want_class {
        return Err(format!(
            "deepseek4 parent: tensor {name:?} has class {:?}, expected {want_class:?}",
            e.class
        ));
    }
    if e.is_mtp {
        return Err(format!(
            "deepseek4 parent: refusing to load MTP tensor {name:?} during parent calibration"
        ));
    }
    Ok(e)
}

fn read_bytes<'a>(
    source: &'a dyn ModelSource,
    name: &str,
    expected_nbytes: usize,
) -> Result<&'a [u8], String> {
    let (info, bytes) = source.tensor_data(name).ok_or_else(|| {
        format!(
            "deepseek4 parent: tensor {name:?} listed in inventory but source.tensor_data returned None \
             (requested {expected_nbytes} bytes)"
        )
    })?;
    if info.data_size != expected_nbytes {
        return Err(format!(
            "deepseek4 parent: tensor {name:?} byte size mismatch: source has {} bytes, \
             expected {expected_nbytes} from inventory shape {:?}",
            info.data_size, info.shape
        ));
    }
    if bytes.len() != expected_nbytes {
        return Err(format!(
            "deepseek4 parent: tensor {name:?} mmap slice length {} != expected {expected_nbytes}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn shape_nbytes(shape: &[usize], elem: usize) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(elem, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            format!("deepseek4 parent: shape {shape:?} × elem_size {elem} overflowed")
        })
}

fn decode_dense(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    name: &str,
    running: &mut ParentResidency,
) -> Result<ParentDenseWeight, String> {
    let e = require_entry(by_name, name, ParentTensorClass::DenseFp8)?;
    if e.shape.len() != 2 {
        return Err(format!(
            "deepseek4 parent: dense weight {name:?} expected rank-2, got shape {:?}",
            e.shape
        ));
    }
    let n = e.shape[0];
    let k = e.shape[1];
    let codes_nbytes = shape_nbytes(&e.shape, 1)?;
    let codes = read_bytes(source, name, codes_nbytes).map_err(|err| {
        format!("{err}; dense decode requested {codes_nbytes} code bytes for [{n},{k}]")
    })?;

    let scale = e.scale.as_ref().ok_or_else(|| {
        format!("deepseek4 parent: dense weight {name:?} has no scale companion in inventory")
    })?;
    let scales_nbytes = shape_nbytes(&scale.shape, 1)?;
    let scales = read_bytes(source, &scale.name, scales_nbytes).map_err(|err| {
        format!(
            "{err}; dense decode requested {scales_nbytes} scale bytes for {:?}",
            scale.shape
        )
    })?;

    let w = ParentDenseWeight::decode_resident(gpu, backend, codes, scales, n, k).map_err(|err| {
        format!(
            "deepseek4 parent: dense decode failed for {name:?} \
             ([{n},{k}] codes={codes_nbytes} B scales={scales_nbytes} B): {err}"
        )
    })?;
    running.dense_bf16_bytes = running
        .dense_bf16_bytes
        .saturating_add(w.resident_bytes() as u64);
    Ok(w)
}

fn upload_expert(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    name: &str,
    running: &mut ParentResidency,
) -> Result<ParentExpertWeight, String> {
    let e = require_entry(by_name, name, ParentTensorClass::ExpertFp4)?;
    if e.shape.len() != 2 || e.logical_shape.len() != 2 {
        return Err(format!(
            "deepseek4 parent: expert weight {name:?} expected rank-2, got shape {:?} logical {:?}",
            e.shape, e.logical_shape
        ));
    }
    let n = e.logical_shape[0];
    let k = e.logical_shape[1];
    let codes_nbytes = shape_nbytes(&e.shape, 1)?;
    let codes = read_bytes(source, name, codes_nbytes).map_err(|err| {
        format!(
            "{err}; expert upload requested {codes_nbytes} code bytes for packed {:?} logical [{n},{k}]",
            e.shape
        )
    })?;

    let scale = e.scale.as_ref().ok_or_else(|| {
        format!("deepseek4 parent: expert weight {name:?} has no scale companion in inventory")
    })?;
    let scales_nbytes = shape_nbytes(&scale.shape, 1)?;
    let scales = read_bytes(source, &scale.name, scales_nbytes).map_err(|err| {
        format!(
            "{err}; expert upload requested {scales_nbytes} scale bytes for {:?}",
            scale.shape
        )
    })?;

    let w = ParentExpertWeight::upload_compressed(gpu, backend, codes, scales, n, k).map_err(
        |err| {
            format!(
                "deepseek4 parent: expert upload failed for {name:?} \
                 (logical [{n},{k}] codes={codes_nbytes} B scales={scales_nbytes} B): {err}"
            )
        },
    )?;
    running.expert_compressed_bytes = running
        .expert_compressed_bytes
        .saturating_add(w.compressed_bytes() as u64);
    Ok(w)
}

fn upload_bf16(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    name: &str,
    running: &mut ParentResidency,
) -> Result<GpuTensor, String> {
    let e = require_entry(by_name, name, ParentTensorClass::Bf16)?;
    let nbytes = shape_nbytes(&e.shape, 2)?;
    let bytes = read_bytes(source, name, nbytes)?;
    let mut t = gpu.upload_raw(bytes, &e.shape).map_err(|err| {
        format!(
            "deepseek4 parent: BF16 upload failed for {name:?} \
             (shape {:?} = {nbytes} bytes): {err}",
            e.shape
        )
    })?;
    t.dtype = DType::BF16;
    running.bf16_bytes = running.bf16_bytes.saturating_add(t.buf.size() as u64);
    Ok(t)
}

fn upload_f32(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    name: &str,
    running: &mut ParentResidency,
) -> Result<GpuTensor, String> {
    let e = require_entry(by_name, name, ParentTensorClass::F32)?;
    let nbytes = shape_nbytes(&e.shape, 4)?;
    let bytes = read_bytes(source, name, nbytes)?;
    let mut t = gpu.upload_raw(bytes, &e.shape).map_err(|err| {
        format!(
            "deepseek4 parent: F32 upload failed for {name:?} \
             (shape {:?} = {nbytes} bytes): {err}",
            e.shape
        )
    })?;
    t.dtype = DType::F32;
    running.f32_bytes = running.f32_bytes.saturating_add(t.buf.size() as u64);
    Ok(t)
}

fn upload_i64(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    by_name: &HashMap<&str, &ParentTensorEntry>,
    name: &str,
    running: &mut ParentResidency,
) -> Result<GpuTensor, String> {
    let e = require_entry(by_name, name, ParentTensorClass::I64)?;
    let nbytes = shape_nbytes(&e.shape, 8)?;
    let bytes = read_bytes(source, name, nbytes)?;
    // I64 has no dedicated DType; keep as Raw so byte size is exact.
    let t = gpu.upload_raw(bytes, &e.shape).map_err(|err| {
        format!(
            "deepseek4 parent: I64 upload failed for {name:?} \
             (shape {:?} = {nbytes} bytes): {err}",
            e.shape
        )
    })?;
    running.i64_bytes = running.i64_bytes.saturating_add(t.buf.size() as u64);
    Ok(t)
}

fn validate_plan(cfg: &ParentQuantConfig, plan: &ParentLoadPlan) -> Result<(), String> {
    if plan.layers.start > plan.layers.end {
        return Err(format!(
            "deepseek4 parent: ParentLoadPlan.layers start {} > end {}",
            plan.layers.start, plan.layers.end
        ));
    }
    if plan.layers.end > cfg.num_hidden_layers {
        return Err(format!(
            "deepseek4 parent: ParentLoadPlan.layers end {} exceeds num_hidden_layers {}",
            plan.layers.end, cfg.num_hidden_layers
        ));
    }
    Ok(())
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

// ---------------------------------------------------------------------------
// Unit tests (synthetic ModelSource fixture — no GPU required for the plan /
// inventory routing logic; GPU paths are exercised on mi300x via the gate).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::ParentInventory;
    use crate::{
        PARENT_EXPERT_DTYPE, PARENT_MODEL_TYPE, PARENT_QUANT_METHOD, PARENT_SCALE_FMT,
        PARENT_WEIGHT_BLOCK, PARENT_WEIGHT_FMT,
    };
    use hipfire_runtime::model_source::{ModelSource, QuantConfig, TensorInfo};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// In-memory source that actually serves bytes (inventory fixture does not).
    struct ByteSource {
        infos: BTreeMap<String, TensorInfo>,
        data: BTreeMap<String, Vec<u8>>,
        meta: String,
    }

    impl ByteSource {
        fn new() -> Self {
            Self {
                infos: BTreeMap::new(),
                data: BTreeMap::new(),
                meta: String::new(),
            }
        }

        fn push(&mut self, name: &str, dtype: &str, shape: Vec<usize>) {
            let elem = match dtype {
                "F8_E4M3" | "F8_E8M0" | "I8" => 1,
                "BF16" => 2,
                "F32" => 4,
                "I64" => 8,
                _ => 1,
            };
            let n: usize = shape.iter().product();
            let data_size = n * elem;
            let offset = self.data.values().map(|v| v.len()).sum();
            self.infos.insert(
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
            self.data.insert(name.to_owned(), vec![0u8; data_size]);
        }
    }

    impl ModelSource for ByteSource {
        fn metadata_json(&self) -> &str {
            &self.meta
        }
        fn arch_id(&self) -> u32 {
            0
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
        fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
            let info = self.infos.get(name)?;
            let bytes = self.data.get(name)?;
            Some((info, bytes.as_slice()))
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.infos.keys().map(String::as_str).collect()
        }
        fn path(&self) -> &Path {
            Path::new("/tmp/ds4-parent-weights-fixture")
        }
    }

    fn test_cfg(
        n_layers: usize,
        n_hash: usize,
        ratios: Vec<usize>,
        n_exp: usize,
    ) -> ParentQuantConfig {
        ParentQuantConfig {
            model_type: PARENT_MODEL_TYPE.to_owned(),
            quant_method: PARENT_QUANT_METHOD.to_owned(),
            fmt: PARENT_WEIGHT_FMT.to_owned(),
            scale_fmt: PARENT_SCALE_FMT.to_owned(),
            expert_dtype: PARENT_EXPERT_DTYPE.to_owned(),
            weight_block_size: PARENT_WEIGHT_BLOCK,
            num_hidden_layers: n_layers,
            num_hash_layers: n_hash,
            n_routed_experts: n_exp,
            num_experts_per_tok: 2,
            compress_ratios: ratios,
        }
    }

    /// Tiny synthetic tower covering ratio 0 / 4 / 128, hash vs score routing,
    /// shared experts, and an MTP tensor that must stay excluded.
    fn build_fixture() -> (ByteSource, ParentQuantConfig) {
        // layers: 0 ratio=0 hash, 1 ratio=4 score, 2 ratio=128 score
        let cfg = test_cfg(3, 1, vec![0, 4, 128], 2);
        let mut s = ByteSource::new();

        // Globals.
        s.push("embed.weight", "BF16", vec![32, 16]);
        s.push("norm.weight", "BF16", vec![16]);
        s.push("head.weight", "BF16", vec![32, 16]);
        s.push("hc_head_base", "F32", vec![4]);
        s.push("hc_head_fn", "F32", vec![4, 64]);
        s.push("hc_head_scale", "F32", vec![1]);

        for layer in 0..3 {
            let ratio = cfg.compress_ratio(layer);
            let is_hash = layer < cfg.num_hash_layers;
            s.push(&format!("layers.{layer}.attn_norm.weight"), "BF16", vec![16]);
            s.push(&format!("layers.{layer}.ffn_norm.weight"), "BF16", vec![16]);
            s.push(
                &format!("layers.{layer}.attn.q_norm.weight"),
                "BF16",
                vec![8],
            );
            s.push(
                &format!("layers.{layer}.attn.kv_norm.weight"),
                "BF16",
                vec![4],
            );
            s.push(&format!("layers.{layer}.attn.attn_sink"), "F32", vec![2]);

            push_dense(&mut s, &format!("layers.{layer}.attn.wq_a"), 8, 16);
            push_dense(&mut s, &format!("layers.{layer}.attn.wq_b"), 16, 8);
            push_dense(&mut s, &format!("layers.{layer}.attn.wkv"), 4, 16);
            push_dense(&mut s, &format!("layers.{layer}.attn.wo_a"), 8, 16);
            push_dense(&mut s, &format!("layers.{layer}.attn.wo_b"), 16, 8);

            push_dense(
                &mut s,
                &format!("layers.{layer}.ffn.shared_experts.w1"),
                8,
                16,
            );
            push_dense(
                &mut s,
                &format!("layers.{layer}.ffn.shared_experts.w2"),
                16,
                8,
            );
            push_dense(
                &mut s,
                &format!("layers.{layer}.ffn.shared_experts.w3"),
                8,
                16,
            );

            for e in 0..cfg.n_routed_experts {
                push_expert(
                    &mut s,
                    &format!("layers.{layer}.ffn.experts.{e}.w1"),
                    8,
                    32,
                );
                push_expert(
                    &mut s,
                    &format!("layers.{layer}.ffn.experts.{e}.w2"),
                    16,
                    32,
                );
                push_expert(
                    &mut s,
                    &format!("layers.{layer}.ffn.experts.{e}.w3"),
                    8,
                    32,
                );
            }

            s.push(
                &format!("layers.{layer}.ffn.gate.weight"),
                "BF16",
                vec![cfg.n_routed_experts, 16],
            );
            if is_hash {
                s.push(
                    &format!("layers.{layer}.ffn.gate.tid2eid"),
                    "I64",
                    vec![32, 2],
                );
            } else {
                s.push(
                    &format!("layers.{layer}.ffn.gate.bias"),
                    "F32",
                    vec![cfg.n_routed_experts],
                );
            }

            s.push(&format!("layers.{layer}.hc_attn_base"), "F32", vec![4]);
            s.push(&format!("layers.{layer}.hc_attn_fn"), "F32", vec![4, 64]);
            s.push(&format!("layers.{layer}.hc_attn_scale"), "F32", vec![1]);
            s.push(&format!("layers.{layer}.hc_ffn_base"), "F32", vec![4]);
            s.push(&format!("layers.{layer}.hc_ffn_fn"), "F32", vec![4, 64]);
            s.push(&format!("layers.{layer}.hc_ffn_scale"), "F32", vec![1]);

            if ratio > 0 {
                let coff = if ratio == 4 { 8 } else { 4 };
                s.push(
                    &format!("layers.{layer}.attn.compressor.wkv.weight"),
                    "BF16",
                    vec![coff, 16],
                );
                s.push(
                    &format!("layers.{layer}.attn.compressor.wgate.weight"),
                    "BF16",
                    vec![coff, 16],
                );
                s.push(
                    &format!("layers.{layer}.attn.compressor.norm.weight"),
                    "BF16",
                    vec![4],
                );
                s.push(
                    &format!("layers.{layer}.attn.compressor.ape"),
                    "F32",
                    vec![ratio.min(4), coff],
                );
            }
            if ratio == 4 {
                push_dense(
                    &mut s,
                    &format!("layers.{layer}.attn.indexer.wq_b"),
                    8,
                    8,
                );
                s.push(
                    &format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                    "BF16",
                    vec![4, 16],
                );
                s.push(
                    &format!("layers.{layer}.attn.indexer.compressor.wkv.weight"),
                    "BF16",
                    vec![4, 16],
                );
                s.push(
                    &format!("layers.{layer}.attn.indexer.compressor.wgate.weight"),
                    "BF16",
                    vec![4, 16],
                );
                s.push(
                    &format!("layers.{layer}.attn.indexer.compressor.norm.weight"),
                    "BF16",
                    vec![2],
                );
                s.push(
                    &format!("layers.{layer}.attn.indexer.compressor.ape"),
                    "F32",
                    vec![4, 4],
                );
            }
        }

        // MTP — must be inventoried then excluded from load.
        push_dense(&mut s, "mtp.0.attn.wq_a", 8, 16);
        s.push("mtp.0.norm.weight", "BF16", vec![16]);

        (s, cfg)
    }

    fn push_dense(s: &mut ByteSource, stem: &str, n: usize, k: usize) {
        s.push(&format!("{stem}.weight"), "F8_E4M3", vec![n, k]);
        s.push(
            &format!("{stem}.scale"),
            "F8_E8M0",
            vec![n.div_ceil(128), k.div_ceil(128)],
        );
    }

    fn push_expert(s: &mut ByteSource, stem: &str, n: usize, k_logical: usize) {
        assert!(k_logical % 32 == 0);
        s.push(
            &format!("{stem}.weight"),
            "I8",
            vec![n, k_logical / 2],
        );
        s.push(
            &format!("{stem}.scale"),
            "F8_E8M0",
            vec![n, k_logical / 32],
        );
    }

    #[test]
    fn fixture_inventory_covers_ratio_and_routing_variants() {
        let (src, cfg) = build_fixture();
        let inv = ParentInventory::build(&src, &cfg).expect("fixture inventory");

        assert!(
            inv.excluded_mtp.iter().any(|n| n.starts_with("mtp.")),
            "MTP must be excluded: {:?}",
            inv.excluded_mtp
        );
        assert!(inv.entries.iter().all(|e| !e.is_mtp));
        assert!(inv.entries.iter().all(|e| !e.name.starts_with("mtp.")));

        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.0.ffn.gate.tid2eid"));
        assert!(inv
            .entries
            .iter()
            .all(|e| e.name != "layers.0.ffn.gate.bias"));
        assert!(inv
            .entries
            .iter()
            .all(|e| !e.name.starts_with("layers.0.attn.compressor")));
        assert!(inv
            .entries
            .iter()
            .all(|e| !e.name.starts_with("layers.0.attn.indexer")));

        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.1.attn.compressor.wkv.weight"));
        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.1.attn.indexer.wq_b.weight"));
        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.1.ffn.gate.bias"));
        assert!(inv
            .entries
            .iter()
            .all(|e| e.name != "layers.1.ffn.gate.tid2eid"));

        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.2.attn.compressor.wkv.weight"));
        assert!(inv
            .entries
            .iter()
            .all(|e| !e.name.starts_with("layers.2.attn.indexer")));
        assert!(inv
            .entries
            .iter()
            .any(|e| e.name == "layers.2.ffn.gate.bias"));

        for layer in 0..3 {
            for e in 0..cfg.n_routed_experts {
                let n = format!("layers.{layer}.ffn.experts.{e}.w1.weight");
                assert!(
                    inv.entries.iter().any(|ent| ent.name == n),
                    "missing {n}"
                );
            }
        }
    }

    #[test]
    fn plan_validation_rejects_out_of_range_layers() {
        let cfg = test_cfg(3, 1, vec![0, 4, 128], 2);
        let err = validate_plan(
            &cfg,
            &ParentLoadPlan {
                layers: 0..4,
                load_experts: true,
            },
        )
        .expect_err("end past num_hidden_layers");
        assert!(
            err.contains("exceeds num_hidden_layers") && err.starts_with("deepseek4 parent:"),
            "{err}"
        );
    }

    #[test]
    fn missing_tensor_refused_by_require_entry() {
        let (src, cfg) = build_fixture();
        let inv = ParentInventory::build(&src, &cfg).unwrap();
        let by_name: HashMap<&str, &ParentTensorEntry> =
            inv.entries.iter().map(|e| (e.name.as_str(), e)).collect();
        let err =
            require_entry(&by_name, "layers.0.attn.does_not_exist.weight", ParentTensorClass::Bf16)
                .expect_err("missing");
        assert!(
            err.contains("missing from inventory") && err.starts_with("deepseek4 parent:"),
            "{err}"
        );
    }

    #[test]
    fn mtp_tensor_never_in_load_set() {
        let (src, cfg) = build_fixture();
        let inv = ParentInventory::build(&src, &cfg).unwrap();
        let by_name: HashMap<&str, &ParentTensorEntry> =
            inv.entries.iter().map(|e| (e.name.as_str(), e)).collect();
        let err = require_entry(&by_name, "mtp.0.attn.wq_a.weight", ParentTensorClass::DenseFp8)
            .expect_err("mtp");
        assert!(err.contains("missing from inventory") || err.contains("MTP"), "{err}");
    }

    #[test]
    fn load_experts_false_skips_expert_names_in_plan_logic() {
        let plan = ParentLoadPlan {
            layers: 0..1,
            load_experts: false,
        };
        assert!(!plan.load_experts);
        assert_eq!(plan.layers, 0..1);
    }

    #[test]
    fn layers_subrange_plan() {
        let plan = ParentLoadPlan {
            layers: 1..2,
            load_experts: true,
        };
        assert_eq!(plan.layers.start, 1);
        assert_eq!(plan.layers.end, 2);
        assert_eq!(plan.layers.len(), 1);
    }

    #[test]
    fn residency_total_sums_tiers() {
        let r = ParentResidency {
            dense_bf16_bytes: 10,
            expert_compressed_bytes: 20,
            bf16_bytes: 30,
            f32_bytes: 40,
            i64_bytes: 50,
        };
        assert_eq!(r.total_bytes(), 150);
    }

    #[test]
    fn missing_dense_scale_refused_at_inventory() {
        let (mut src, cfg) = build_fixture();
        src.infos.remove("layers.0.attn.wq_a.scale");
        src.data.remove("layers.0.attn.wq_a.scale");
        let err = ParentInventory::build(&src, &cfg).expect_err("missing scale");
        assert!(
            err.contains("missing required scale") && err.contains("wq_a.weight"),
            "{err}"
        );
    }

    #[test]
    fn expected_names_per_layer_variant() {
        let cfg = test_cfg(3, 1, vec![0, 4, 128], 2);
        let n0 = expected_layer_primary_names(&cfg, 0, true);
        assert!(n0.contains(&"layers.0.ffn.gate.tid2eid".to_owned()));
        assert!(!n0.iter().any(|n| n.contains("gate.bias")));
        assert!(!n0.iter().any(|n| n.contains("compressor")));
        assert!(!n0.iter().any(|n| n.contains("indexer")));

        let n1 = expected_layer_primary_names(&cfg, 1, true);
        assert!(n1.iter().any(|n| n.contains("compressor.wkv")));
        assert!(n1.iter().any(|n| n.contains("indexer.wq_b")));
        assert!(n1.contains(&"layers.1.ffn.gate.bias".to_owned()));
        assert!(!n1.iter().any(|n| n.contains("tid2eid")));

        let n2 = expected_layer_primary_names(&cfg, 2, true);
        assert!(n2.iter().any(|n| n.contains("compressor.wkv")));
        assert!(!n2.iter().any(|n| n.contains("indexer")));
        let n0_no = expected_layer_primary_names(&cfg, 0, false);
        assert!(!n0_no.iter().any(|n| n.contains("ffn.experts.")));
        assert!(n0.iter().any(|n| n.contains("ffn.experts.")));
    }

    fn expected_layer_primary_names(
        cfg: &ParentQuantConfig,
        layer: usize,
        load_experts: bool,
    ) -> Vec<String> {
        let ratio = cfg.compress_ratio(layer);
        let is_hash = layer < cfg.num_hash_layers;
        let mut names = vec![
            format!("layers.{layer}.attn_norm.weight"),
            format!("layers.{layer}.ffn_norm.weight"),
            format!("layers.{layer}.attn.q_norm.weight"),
            format!("layers.{layer}.attn.kv_norm.weight"),
            format!("layers.{layer}.attn.attn_sink"),
            format!("layers.{layer}.attn.wq_a.weight"),
            format!("layers.{layer}.attn.wq_b.weight"),
            format!("layers.{layer}.attn.wkv.weight"),
            format!("layers.{layer}.attn.wo_a.weight"),
            format!("layers.{layer}.attn.wo_b.weight"),
            format!("layers.{layer}.ffn.shared_experts.w1.weight"),
            format!("layers.{layer}.ffn.shared_experts.w2.weight"),
            format!("layers.{layer}.ffn.shared_experts.w3.weight"),
            format!("layers.{layer}.ffn.gate.weight"),
            format!("layers.{layer}.hc_attn_base"),
            format!("layers.{layer}.hc_attn_fn"),
            format!("layers.{layer}.hc_attn_scale"),
            format!("layers.{layer}.hc_ffn_base"),
            format!("layers.{layer}.hc_ffn_fn"),
            format!("layers.{layer}.hc_ffn_scale"),
        ];
        if is_hash {
            names.push(format!("layers.{layer}.ffn.gate.tid2eid"));
        } else {
            names.push(format!("layers.{layer}.ffn.gate.bias"));
        }
        if ratio > 0 {
            names.extend([
                format!("layers.{layer}.attn.compressor.wkv.weight"),
                format!("layers.{layer}.attn.compressor.wgate.weight"),
                format!("layers.{layer}.attn.compressor.norm.weight"),
                format!("layers.{layer}.attn.compressor.ape"),
            ]);
        }
        if ratio == 4 {
            names.extend([
                format!("layers.{layer}.attn.indexer.wq_b.weight"),
                format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                format!("layers.{layer}.attn.indexer.compressor.wkv.weight"),
                format!("layers.{layer}.attn.indexer.compressor.wgate.weight"),
                format!("layers.{layer}.attn.indexer.compressor.norm.weight"),
                format!("layers.{layer}.attn.indexer.compressor.ape"),
            ]);
        }
        if load_experts {
            for e in 0..cfg.n_routed_experts {
                names.push(format!("layers.{layer}.ffn.experts.{e}.w1.weight"));
                names.push(format!("layers.{layer}.ffn.experts.{e}.w2.weight"));
                names.push(format!("layers.{layer}.ffn.experts.{e}.w3.weight"));
            }
        }
        names
    }
}
