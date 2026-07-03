// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic GPU KV cache for autoregressive generation.
//!
//! `KvCache` — the per-layer key/value GPU buffers plus the quantized-KV
//! tiers (q8 / asym / FWHT-rotated) and their constructors — is arch-agnostic;
//! every attention arch fills and reads it. It historically lived in
//! `llama.rs`; relocated here as part of the de-llama-ify cleanup, together
//! with its per-device alloc + givens/FWHT-sign replication helpers.

use crate::multi_gpu::Gpus;
use hip_bridge::HipResult;
use hipfire_primitives::fwht;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Number of F32 slots needed to back `bytes` of packed KV data (round up).
///
/// Quantized KV caches store bytes but allocate F32 buffers, so a partial
/// trailing F32 word must still be reserved. Single-sources the `(bytes + 3)/4`
/// ceil-div that was inlined at every quantized-KV constructor — an off-by-one
/// here under-allocates the cache and corrupts the tail on write.
#[inline]
fn kv_f32_elems_for_bytes(bytes: usize) -> usize {
    bytes.div_ceil(4)
}

/// Whether KV layer `kv_ordinal` is a boundary layer (bounds-checked index into
/// the per-layer boundary flags). Pure so the index logic is unit-testable
/// without constructing a full [`KvCache`].
#[inline]
fn is_boundary_ordinal(layer_is_boundary: &[bool], kv_ordinal: usize) -> bool {
    kv_ordinal < layer_is_boundary.len() && layer_is_boundary[kv_ordinal]
}

/// GPU-resident KV cache for autoregressive generation.
///
/// Two capacity axes live here:
///   * `max_seq`       — advertised absolute-position range (used for RoPE phase,
///                       attention masks, and anything that reasons about the
///                       user-visible context window).
///   * `physical_cap`  — actual buffer size along the token axis (drives
///                       allocation + kernel strides). When eviction is active,
///                       `physical_cap << max_seq` so the buffer stays bounded
///                       even as the absolute position grows past it.
///
/// The KV-cache quantization mode, as a single typed value.
///
/// `KvCache` currently stores this as nine parallel `quant_*` booleans (a
/// historical layout the review flagged, 3.10). This enum is the canonical
/// mutually-exclusive view of them: [`KvCache::quant_mode`] derives it, and
/// [`kv_quant_mode_from_flags`] pins the boolean→mode mapping in a pure,
/// unit-tested function. New code should switch on this rather than reading the
/// raw booleans; the eventual field-replacement (booleans → this enum + a
/// `KvCacheSpec` builder) is a hot-path change that must be validated under the
/// GPU coherence gate.
///
/// Superset of the arch-level `KvMode` (qwen35 speculative): it also covers the
/// `KvCache`-only `Int8`, `Hfq4`, and `Kvarn` modes that never reach that enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    /// Unquantized FP32 K/V (`quantized == false`).
    Unquantized,
    /// INT8 co-located K and V (Q8_0).
    Q8,
    /// INT8 with separate scales.
    Int8,
    /// HFQ4 co-located blocks.
    Hfq4,
    /// Givens-rotated 4-bit K + Q8 V.
    Asym4,
    /// Givens-rotated 3-bit K + Q8 V.
    Asym3,
    /// Givens-rotated 2-bit K + Q8 V.
    Asym2,
    /// Signed-FWHT-rotated 4-bit K + Q8 V (byte-identical storage to `Asym4`).
    Fwht4,
    /// Signed-FWHT-rotated 3-bit K + Q8 V (byte-identical storage to `Asym3`).
    Fwht3,
    /// Signed-FWHT-rotated 2-bit K + Q8 V (byte-identical storage to `Asym2`).
    Fwht2,
    /// KVarN variance-normalized 4-bit K blocks + Q8 V.
    Kvarn,
}

/// Derive the canonical [`KvQuantMode`] from `KvCache`'s raw quant flags.
///
/// Pure and total so it is unit-testable without a GPU. Mirrors the exclusive
/// flag pattern every `new_gpu*` constructor sets: `quantized` plus exactly one
/// of `{q8, int8, hfq4, asym{4,3,2}, kvarn}`, with `fwht` selecting the
/// FWHT-rotated variant of the asym tiers. `quantized == false` is
/// `Unquantized` regardless of the other flags.
#[allow(clippy::too_many_arguments)]
pub fn kv_quant_mode_from_flags(
    quantized: bool,
    q8: bool,
    int8: bool,
    hfq4: bool,
    asym4: bool,
    asym3: bool,
    asym2: bool,
    fwht: bool,
    kvarn: bool,
) -> KvQuantMode {
    if !quantized {
        return KvQuantMode::Unquantized;
    }
    if kvarn {
        return KvQuantMode::Kvarn;
    }
    if q8 {
        return KvQuantMode::Q8;
    }
    if int8 {
        return KvQuantMode::Int8;
    }
    if hfq4 {
        return KvQuantMode::Hfq4;
    }
    if asym4 {
        return if fwht {
            KvQuantMode::Fwht4
        } else {
            KvQuantMode::Asym4
        };
    }
    if asym3 {
        return if fwht {
            KvQuantMode::Fwht3
        } else {
            KvQuantMode::Asym3
        };
    }
    if asym2 {
        return if fwht {
            KvQuantMode::Fwht2
        } else {
            KvQuantMode::Asym2
        };
    }
    // `quantized` with no tier flag set is not produced by any constructor;
    // treat as unquantized rather than panic in the hot path.
    KvQuantMode::Unquantized
}

/// Back-compat: constructors that do not take `physical_cap` set it equal to
/// `max_seq`, preserving existing behaviour.
pub struct KvCache {
    pub k_gpu: Vec<GpuTensor>,    // [n_layers] key values (FP32 or int8)
    pub v_gpu: Vec<GpuTensor>,    // [n_layers] value values (FP32 or int8)
    pub k_scales: Vec<GpuTensor>, // [n_layers] key scales (for INT8 mode)
    pub v_scales: Vec<GpuTensor>, // [n_layers] value scales (for INT8 mode)
    pub kv_dim: usize,
    pub max_seq: usize,
    /// Physical capacity of each per-layer k/v buffer in *tokens*.
    /// Equals `max_seq` unless the buffer was sized for eviction-bounded use.
    pub physical_cap: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub quantized: bool,
    pub quant_q8: bool,
    pub quant_int8: bool,    // true = INT8 with separate scales
    pub quant_hfq4: bool,    // true = HFQ4 co-located blocks (72 bytes/head)
    pub quant_asym4: bool,   // true = K at 4-bit rotated, V at Q8_0 — RotorQuant planar4 asymmetric
    pub quant_asym3: bool, // true = K at givens3 (rotated 3-bit Lloyd-Max), V at Q8_0 — best-quality rotated K per RotorQuant
    pub quant_asym2: bool, // true = K at givens2 (rotated 2-bit), V at Q8_0 (normal space)
    pub boundary_layers: u8, // number of boundary layers at each end (default 2)
    // KV rotation parameter buffers. Field names are historical — in the
    // Givens-rotated asym{2,3,4} modes (`quant_fwht == false`) these hold the
    // per-block cos/sin tables. In the signed-FWHT-rotated fwht{2,3,4} modes
    // (`quant_fwht == true`) the SAME slots hold signs1/signs2 ±1 vectors.
    // Both are [n_blocks × f32] in shape, so the storage is fungible; the
    // dispatcher reads `quant_fwht` to know which kernel signature to use.
    pub givens_cos: Option<GpuTensor>,
    pub givens_sin: Option<GpuTensor>,
    /// True when the rotation primitive is signed-FWHT (matches Fwht{2,3,4}
    /// KvMode values). False when Givens (matches Asym{2,3,4}).
    pub quant_fwht: bool,
    /// Per-layer flag: true = this layer uses Q8 (boundary layer)
    pub layer_is_boundary: Vec<bool>,
    /// TriAttention compaction bookkeeping. After each eviction we leave the
    /// retained keys in physical slots `0..budget` with their baked-in RoPE
    /// phases intact, but the forward pass still counts absolute positions
    /// for new writes. `compact_offset = absolute_seq_len - physical_seq_len`
    /// — added to `pos` before RoPE so the new query/key get the correct
    /// absolute phase, and the cache write still lands at `pos` (physical).
    /// Zero when no compaction has happened.
    pub compact_offset: usize,
    /// True = KVarN mode: K stored as variance-normalized 4-bit block records
    /// (`kvarn.rs` tile = `[head_dim × GROUP]`, GROUP=128) for full 128-token
    /// blocks, plus an f32 recent-window ring for the partial trailing block;
    /// V stays Q8_0 (reuses the asym4 V layout). See `new_gpu_kvarn_capped`.
    pub quant_kvarn: bool,
    /// KVarN recent-window staging ring: `[n_layers]` buffers, each `GROUP × kv_dim`
    /// f32, holding the K rows of the not-yet-quantized trailing block. Empty
    /// unless `quant_kvarn`. A block is flush-quantized into `k_gpu` once full.
    pub k_window: Vec<GpuTensor>,
    /// KVarN write-side scratch, reused across every (layer, position) so the
    /// attention path does NOT allocate per call (GpuTensor has no pool-return
    /// Drop — per-call `alloc_tensor` would leak and wedge the GPU). Lazily
    /// allocated on first KVarN attention: `kvarn_tiles` = [n_kv_heads × head_dim
    /// × GROUP] f32 gather staging for one block flush.
    /// `kvarn_shadow` is reserved (the v1 read path materialized a [physical_cap ×
    /// kv_dim] f16 shadow K here; the Phase-D2 fused flash reads records in place,
    /// so it stays `None` — kept for an optional shadow-build fallback).
    pub kvarn_shadow: Option<GpuTensor>,
    pub kvarn_tiles: Option<GpuTensor>,
    /// Deferred-hierarchical two-tier KV (flag-gated `HIPFIRE_KV_HIERARCHICAL=1`).
    /// `None` until lazily built at the first KVarN dispatch (needs `n_heads` from
    /// the model config). When `Some(s)` with `s.enabled`, the KVarN decode path
    /// uses the hot-ring + 4-bit cold-segment two-tier read. See `kv_hier`.
    pub hier: Option<crate::kv_hier::HierKvState>,
}

impl KvCache {
    /// Check if a given KV layer ordinal is a boundary layer (first N + last N).
    pub fn is_boundary(&self, kv_ordinal: usize) -> bool {
        is_boundary_ordinal(&self.layer_is_boundary, kv_ordinal)
    }
}

impl KvCache {
    /// Canonical quantization mode of this cache, derived from the raw
    /// `quant_*` boolean flags via [`kv_quant_mode_from_flags`]. Prefer this
    /// over reading the individual booleans in new code.
    pub fn quant_mode(&self) -> KvQuantMode {
        kv_quant_mode_from_flags(
            self.quantized,
            self.quant_q8,
            self.quant_int8,
            self.quant_hfq4,
            self.quant_asym4,
            self.quant_asym3,
            self.quant_asym2,
            self.quant_fwht,
            self.quant_kvarn,
        )
    }

    pub fn new_gpu(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// FP32 KV cache that skips allocation for layers flagged as non-KV.
    pub fn new_gpu_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = physical_cap * kv_dim;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, cache_size, cache_size, is_kv_layer)?;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fp32 ({n_kv}/{} layers carry KV, others placeholder, physical_cap={physical_cap} / max_seq={max_seq_len})",
            is_kv_layer.len(),
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create quantized KV cache (HFQ4-G128). 3.56x smaller than FP32.
    pub fn new_gpu_q4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        // Per position per head: 8 bytes (scale+zero) + head_dim/2 bytes (nibbles)
        let bytes_per_head = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_head;
        let cache_bytes = max_seq_len * bytes_per_pos;
        // Allocate as raw bytes (use F32 dtype but size in bytes)
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes); // round up to F32 elements
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create Q8_0 quantized KV cache (GGML Q8_0 format). 3.76x smaller than FP32.
    /// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
    /// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
    pub fn new_gpu_q8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Same as [`new_gpu_q8`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_q8_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Helper: allocate K/V Vecs, skipping layers where is_kv_layer[i] is false
    /// by inserting a 1-element placeholder. Saves VRAM for hybrid arches
    /// (Qwen 3.5 DeltaNet + FullAttention) where 75% of layers don't carry
    /// KV in this cache — their state lives in [`crate::qwen35::DeltaNetState`].
    /// Per-layer index is preserved so downstream code can index by absolute
    /// layer_idx unchanged.
    fn alloc_k_v_filtered(
        gpu: &mut Gpu,
        k_elems: usize,
        v_elems: usize,
        is_kv_layer: &[bool],
    ) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
        let n = is_kv_layer.len();
        let mut k_gpu = Vec::with_capacity(n);
        let mut v_gpu = Vec::with_capacity(n);
        for &is_kv in is_kv_layer {
            if is_kv {
                k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
                v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
            } else {
                k_gpu.push(gpu.zeros(&[1], DType::F32)?);
                v_gpu.push(gpu.zeros(&[1], DType::F32)?);
            }
        }
        Ok((k_gpu, v_gpu))
    }

    /// Q8_0 KV cache that skips allocation for layers flagged as non-KV.
    /// Each `is_kv_layer[i] == false` slot gets a 1-element placeholder
    /// (~4 bytes) instead of the full `cache_elems × 4` allocation.
    ///
    /// For Qwen 3.5 hybrid (48 DeltaNet + 16 FullAttention layers), saves
    /// 48 × cache_elems × 4 bytes per cache — at ctx=64K this is multi-GB.
    pub fn new_gpu_q8_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped variant of [`new_gpu_q8_filtered`].
    pub fn new_gpu_q8_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, cache_elems, cache_elems, is_kv_layer)?;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: q8 ({n_kv}/{} layers carry KV, others placeholder)",
            is_kv_layer.len()
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create INT8 co-located KV cache: [f32 scale][pad 4B][int8 × head_dim] = 136 bytes per head.
    pub fn new_gpu_int8c(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bph = 8 + head_dim; // 136 for head_dim=128 (8-byte header + data)
        let bpp = n_kv_heads * bph;
        let cache_bytes = max_seq_len * bpp;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create HFQ4 KV cache: co-located blocks. 72 bytes per head (scale+zero+nibbles).
    pub fn new_gpu_hfq4kv(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_block = 8 + head_dim / 2; // 72 for head_dim=128
        let bytes_per_pos = n_kv_heads * bytes_per_block;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: true,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create HFQ8 KV cache: FP32 scale+zero per head, contiguous uint8 data.
    pub fn new_gpu_hfq8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4; // uint8 data, rounded to f32
        let scale_elems = max_seq_len * n_kv_heads * 2; // scale + zero per head per pos
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        let mut k_scales = Vec::with_capacity(n_layers);
        let mut v_scales = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            k_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
            v_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create INT8 KV cache with separate scale arrays. Clean contiguous layout.
    pub fn new_gpu_int8(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        // Values: max_seq × kv_dim bytes (int8). Round up to f32 elements for alloc.
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        // Scales: max_seq × n_kv_heads floats
        let scale_elems = max_seq_len * n_kv_heads;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        let mut k_scales = Vec::with_capacity(n_layers);
        let mut v_scales = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[val_elems], DType::F32)?);
            k_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
            v_scales.push(gpu.zeros(&[scale_elems], DType::F32)?);
        }
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Generate deterministic Givens rotation angles from a seed.
    /// Returns (cos_theta, sin_theta) each of length n_blocks.
    pub fn gen_givens_angles(seed: u32, n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut cos_vals = Vec::with_capacity(n_blocks);
        let mut sin_vals = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            let angle = (state as f64 / 0x7fffffff as f64) * std::f64::consts::TAU;
            cos_vals.push(angle.cos() as f32);
            sin_vals.push(angle.sin() as f32);
        }
        (cos_vals, sin_vals)
    }

    /// Create asym4 KV cache: K at 4-bit rotated (Givens + Lloyd-Max), V at Q8_0.
    /// head_dim=256 → K=132 B/head, V=272 B/head → 404 B/head total (5.1× vs fp32).
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_asym4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym4_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym4`]: skips KV alloc for non-KV layers.
    pub fn new_gpu_asym4_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym4 filtered ({n_kv}/{} layers carry KV; K rotated-4b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Filtered variant of [`new_gpu_fwht4`]: skips KV alloc for non-KV layers.
    /// Mirrors `new_gpu_asym4_filtered` byte-for-byte except the rotation
    /// parameter buffers hold signs1/signs2 (FWHT) instead of cos/sin (Givens)
    /// and `quant_fwht` is set true. K-cache byte layout is identical to
    /// asym4 so scoring kernels are shared.
    pub fn new_gpu_fwht4_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        // fwht_shfl_forward operates on 128 elements regardless of head_dim;
        // signs are shared across the hd=256 two-half rotation. Seeds (42,
        // 1042) match the MQ4 weight-FWHT convention.
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht4 filtered ({n_kv}/{} layers carry KV; K FWHT-4b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Same as [`new_gpu_asym4`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym4_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: asym4 (K rotated-4b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create fwht4 KV cache: K at 4-bit signed-FWHT-rotated (Lloyd-Max
    /// post-FWHT N(0, 1/128)), V at Q8_0 in normal space. Byte-identical
    /// storage to asym4 — only the rotation primitive differs.
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_fwht4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht4_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Same as [`new_gpu_fwht4`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_fwht4_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        // fwht_shfl_forward operates on 128 elements regardless of head_dim
        // (hd=256 is processed as 2 halves with the same signs reused).
        // Seeds (42, 1042) match the established MQ4 weight-FWHT convention
        // (see crates/hipfire-quantize/src/bin/dflash_convert.rs:600 and
        // crates/hipfire-arch-qwen35/src/qwen35.rs:872 — same PRNG family).
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: fwht4 (K FWHT-4b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// KVarN block grouping: a K record covers `GROUP` consecutive tokens
    /// (one `[head_dim × GROUP]` tile per kv-head). Must match the GROUP used by
    /// the write/read kernels and `kvarn.rs`.
    pub const KVARN_GROUP: usize = 128;

    /// Byte length of one KVarN K record (`[head_dim × GROUP]` tile): packed
    /// 4-bit codes + fp16 per-channel scale_abs/zp_abs + fp16 per-token s_col.
    /// Mirrors `hipfire_quantize::kvarn::kvarn_record_bytes(head_dim, GROUP)`
    /// (inlined to avoid a runtime→quantize dep).
    pub fn kvarn_k_record_bytes(head_dim: usize) -> usize {
        let (r, c) = (head_dim, Self::KVARN_GROUP);
        (r * c).div_ceil(2) + r * 2 * 2 + c * 2
    }

    /// Create KVarN KV cache: K stored as variance-normalized 4-bit block
    /// records (`[head_dim × GROUP]` tiles, one per kv-head per 128-token
    /// block) plus an fp16 recent-window ring for the trailing partial block;
    /// V at Q8_0 (identical layout to asym4's V). Back-compat wrapper:
    /// `physical_cap == max_seq_len`. See [`new_gpu_kvarn_capped`].
    pub fn new_gpu_kvarn(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_kvarn_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Same as [`new_gpu_kvarn`] with an explicit `physical_cap`. Eviction-aware.
    pub fn new_gpu_kvarn_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "kvarn requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        // K: block-tiled KVarN records. One record per (block, kv_head); a block
        // spans GROUP tokens. The trailing partial block lives in `k_window`
        // (fp16) until it fills, so allocate ceil(cap/GROUP) record slots.
        let group = Self::KVARN_GROUP;
        let n_blocks = physical_cap.div_ceil(group);
        let rec_bytes = Self::kvarn_k_record_bytes(head_dim);
        let k_bytes = n_blocks * n_kv_heads * rec_bytes;
        let k_elems = k_bytes.div_ceil(4); // store as F32 buffer (byte-addressed by kernels)
                                           // V: Q8_0, identical layout to asym4 (34 bytes per 32-elem block).
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp).div_ceil(4);

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        let mut k_window = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
            // Recent-window staging ring: GROUP tokens × kv_dim, stored F32 so
            // the existing `kv_cache_write_f32_batched` can append rows and the
            // gather/quantize kernels share one input dtype. It holds at most one
            // 128-token block, so the f32-vs-f16 size cost is negligible.
            k_window.push(gpu.zeros(&[group * kv_dim], DType::F32)?);
        }
        let k_bph = rec_bytes / n_kv_heads.max(1); // informational; record is per-head already
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: kvarn (K 4b var-norm block records {rec_bytes}B/tile [{}-tok blocks] + fp16 window + V Q8 {v_bph}B/head)",
            group,
        );
        let _ = k_bph;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: true,
            k_window,
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create asym3 KV cache: K at 3-bit rotated (Lloyd-Max N(0, 1/256)), V at Q8_0.
    /// head_dim=256 → K=100 B/head, V=272 B/head → 372 B/head (5.5× vs fp32).
    /// Back-compat wrapper: allocates physical_cap == max_seq_len slots per layer.
    pub fn new_gpu_asym3(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym3`]: skips KV allocation for layers
    /// flagged as non-KV (LinearAttention/DeltaNet in hybrid arches). See
    /// [`alloc_k_v_filtered`].
    pub fn new_gpu_asym3_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered asym3 — saves multi-GB at long ctx for Qwen 3.5 hybrid.
    pub fn new_gpu_asym3_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym3 filtered ({n_kv}/{} layers carry KV; K rotated-3b {k_bph}B + V Q8 {v_bph}B = {} B/head, physical_cap={physical_cap} / max_seq={max_seq_len})",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Filtered variant of fwht3 — signed-FWHT-256 K-rotation, 3-bit centroid,
    /// V at Q8_0. Same byte layout as asym3_filtered; rotation primitive swapped
    /// to fwht_shfl_forward_256 which expects 256-element signs1/signs2.
    pub fn new_gpu_fwht3_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        // fwht_shfl_forward_256 reads signs[tid*8..tid*8+7], so 256 floats each.
        let n_signs = 256;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht3 filtered ({n_kv}/{} layers carry KV; K FWHT-3b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Same as [`new_gpu_asym3`] but with an explicit physical capacity. When
    /// `physical_cap < max_seq_len`, the cache is sized for `physical_cap`
    /// tokens along the time axis; the caller is responsible for triggering
    /// TriAttention/CASK eviction before the physical position overruns
    /// `physical_cap`. `max_seq_len` is retained for RoPE/mask purposes.
    pub fn new_gpu_asym3_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!("KV cache: asym3 (K rotated-3b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32, physical_cap={physical_cap} / max_seq={max_seq_len})",
            k_bph + v_bph, (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64);
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Create asym2 KV cache: K at 2-bit rotated, V at Q8_0.
    /// head_dim=256 → K=68 B/head, V=272 B/head → 340 B/head (6.0× vs fp32).
    /// Back-compat wrapper: `physical_cap == max_seq_len`.
    pub fn new_gpu_asym2(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym2_capped(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Filtered variant of [`new_gpu_asym2`]: skips KV alloc for non-KV layers.
    pub fn new_gpu_asym2_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: asym2 filtered ({n_kv}/{} layers carry KV; K rotated-2b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Filtered variant of fwht2 — signed-FWHT-128 K-rotation, 2-bit centroid,
    /// V at Q8_0. Same 2-pass-over-128 structure as fwht4, signs are 128 floats.
    pub fn new_gpu_fwht2_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        let physical_cap = max_seq_len;
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = Self::alloc_k_v_filtered(gpu, k_elems, v_elems, is_kv_layer)?;
        let n_signs = 128;
        let s1_vals = Self::gen_fwht_signs(42, n_signs);
        let s2_vals = Self::gen_fwht_signs(1042, n_signs);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = gpu.alloc_tensor(&[n_signs], DType::F32)?;
        gpu.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        gpu.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        let v_bph = v_bpp / n_kv_heads;
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!(
            "KV cache: fwht2 filtered ({n_kv}/{} layers carry KV; K FWHT-2b {k_bph}B + V Q8 {v_bph}B = {} B/head)",
            is_kv_layer.len(),
            k_bph + v_bph,
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Same as [`new_gpu_asym2`] with an explicit physical_cap. Eviction-aware.
    pub fn new_gpu_asym2_capped(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(
            physical_cap > 0 && physical_cap <= max_seq_len,
            "physical_cap ({physical_cap}) must be in (0, max_seq_len={max_seq_len}]"
        );
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;

        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
            v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
        }
        let n_blocks = head_dim / 2;
        let (cos_vals, sin_vals) = Self::gen_givens_angles(42, n_blocks);
        let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ct = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = gpu.alloc_tensor(&[n_blocks], DType::F32)?;
        gpu.hip.memcpy_htod(&ct.buf, &cb)?;
        gpu.hip.memcpy_htod(&st.buf, &sb)?;
        let v_bph = v_bpp / n_kv_heads;
        eprintln!(
            "KV cache: asym2 (K rotated-2b {k_bph}B + V Q8 {v_bph}B = {} B/head, {:.1}x vs fp32)",
            k_bph + v_bph,
            (head_dim * 4 * 2) as f64 / (k_bph + v_bph) as f64
        );
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: Some(ct),
            givens_sin: Some(st),
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    /// Generate deterministic ±1 sign array for FWHT.
    pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
        fwht::gen_fwht_signs(seed, n)
    }

    /// Free all GPU tensors in this cache. Call before drop to return VRAM.
    /// After calling, follow with gpu.drain_pool() to actually release memory.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.k_gpu {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_gpu {
            let _ = gpu.free_tensor(t);
        }
        for t in self.k_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_scales {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.givens_cos {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.givens_sin {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Store K, V at position `pos` in layer cache (CPU → GPU copy into cache slot).
    pub fn store_kv_pub(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> HipResult<()> {
        self.store_kv(gpu, layer, pos, k, v)
    }

    fn store_kv(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> HipResult<()> {
        let byte_offset = pos * self.kv_dim * 4; // float = 4 bytes
        let k_bytes =
            unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4) };
        let v_bytes =
            unsafe { std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4) };
        gpu.hip
            .memcpy_htod_offset(&self.k_gpu[layer].buf, byte_offset, k_bytes)?;
        gpu.hip
            .memcpy_htod_offset(&self.v_gpu[layer].buf, byte_offset, v_bytes)?;
        Ok(())
    }

    // ── Multi-GPU constructors (Stage 5 of issue #58) ───────────────────
    //
    // Each `_multi` variant places the per-layer K/V slot on
    // `gpus.devices[gpus.device_for_layer(i)]`. asym{2,3,4} variants
    // additionally replicate the rotation tables to every device by
    // populating `gpus.givens_cos_per_dev` / `gpus.givens_sin_per_dev`.
    //
    // The KvCache.givens_cos / .givens_sin fields stay `None` in multi mode
    // — Stage 6 forward dispatch reads from the per-device replicas in
    // `Gpus` instead.

    /// Free all per-layer GPU tensors on their owning devices. Mirror of
    /// `free_gpu` for the multi-GPU layout. Givens replicas stay owned by
    /// `Gpus`; freeing them is the orchestrator's responsibility.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (i, t) in self.k_gpu.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.v_gpu.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.k_scales.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
        for (i, t) in self.v_scales.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let _ = gpus.devices[dev_idx].free_tensor(t);
        }
    }

    pub fn new_gpu_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_size, cache_size)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: false,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_q4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_head = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_head;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_q8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_q8_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_q8_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let blocks_per_head = head_dim / 32;
        let total_blocks = n_kv_heads * blocks_per_head;
        let cache_bytes = physical_cap * total_blocks * 34;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_int8c_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bph = 8 + head_dim;
        let bpp = n_kv_heads * bph;
        let cache_bytes = max_seq_len * bpp;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_hfq4kv_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let bytes_per_block = 8 + head_dim / 2;
        let bytes_per_pos = n_kv_heads * bytes_per_block;
        let cache_bytes = max_seq_len * bytes_per_pos;
        let cache_elems = kv_f32_elems_for_bytes(cache_bytes);
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, cache_elems, cache_elems)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: true,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_hfq8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        let scale_elems = max_seq_len * n_kv_heads * 2;
        let (k_gpu, v_gpu, k_scales, v_scales) = alloc_kv_with_scales_per_layer_multi(
            gpus,
            n_layers,
            val_elems,
            val_elems,
            scale_elems,
            scale_elems,
        )?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_int8_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let val_elems = (max_seq_len * kv_dim + 3) / 4;
        let scale_elems = max_seq_len * n_kv_heads;
        let (k_gpu, v_gpu, k_scales, v_scales) = alloc_kv_with_scales_per_layer_multi(
            gpus,
            n_layers,
            val_elems,
            val_elems,
            scale_elems,
            scale_elems,
        )?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales,
            v_scales,
            kv_dim,
            max_seq: max_seq_len,
            physical_cap: max_seq_len,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: true,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_asym4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym4_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym4_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_asym3_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym3_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "asym3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_asym2_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym2_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_asym2_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "asym2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_givens_to_all_devices(gpus, head_dim / 2, 42)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    // ── fwht multi-GPU constructors ──────────────────────────────────
    // Mirror the asym{4,3,2}_multi shape. Per-device signs1/signs2
    // replicated via replicate_fwht_signs_to_all_devices. The KvCache
    // struct keeps givens_cos/sin = None in multi mode (per-device slots
    // live on the Gpus struct); `quant_fwht: true` tells the dispatcher
    // to read from gpus.givens_cos_per_dev / .givens_sin_per_dev as
    // signs1/signs2 instead of cos/sin.

    pub fn new_gpu_fwht4_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht4_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht4_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht4 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 2;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_fwht3_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht3_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht3_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + (head_dim * 3) / 8;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        // fwht_shfl_forward_256 needs 256-element signs1/signs2.
        replicate_fwht_signs_to_all_devices(gpus, 256)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: true,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn new_gpu_fwht2_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_fwht2_capped_multi(
            gpus,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    pub fn new_gpu_fwht2_capped_multi(
        gpus: &mut Gpus,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 128 || head_dim == 256,
            "fwht2 requires head_dim=128 or 256"
        );
        assert!(head_dim % 32 == 0);
        assert!(physical_cap > 0 && physical_cap <= max_seq_len);
        let kv_dim = n_kv_heads * head_dim;
        let k_bph = 4 + head_dim / 4;
        let k_elems = (physical_cap * n_kv_heads * k_bph + 3) / 4;
        let v_blocks_per_head = head_dim / 32;
        let v_bpp = n_kv_heads * v_blocks_per_head * 34;
        let v_elems = (physical_cap * v_bpp + 3) / 4;
        let (k_gpu, v_gpu) = alloc_kv_per_layer_multi(gpus, n_layers, k_elems, v_elems)?;
        replicate_fwht_signs_to_all_devices(gpus, 128)?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: true,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            quant_kvarn: false,
            k_window: vec![],
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }
}

// ── Stage 5 helpers: per-device KV alloc + givens replication ────────

fn alloc_kv_per_layer_multi(
    gpus: &mut Gpus,
    n_layers: usize,
    k_elems: usize,
    v_elems: usize,
) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
    let mut k_gpu = Vec::with_capacity(n_layers);
    let mut v_gpu = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let dev_idx = gpus.device_for_layer(i);
        let g = &mut gpus.devices[dev_idx];
        k_gpu.push(g.zeros(&[k_elems], DType::F32)?);
        v_gpu.push(g.zeros(&[v_elems], DType::F32)?);
    }
    Ok((k_gpu, v_gpu))
}

fn alloc_kv_with_scales_per_layer_multi(
    gpus: &mut Gpus,
    n_layers: usize,
    k_elems: usize,
    v_elems: usize,
    k_scale_elems: usize,
    v_scale_elems: usize,
) -> HipResult<(
    Vec<GpuTensor>,
    Vec<GpuTensor>,
    Vec<GpuTensor>,
    Vec<GpuTensor>,
)> {
    let mut k_gpu = Vec::with_capacity(n_layers);
    let mut v_gpu = Vec::with_capacity(n_layers);
    let mut k_scales = Vec::with_capacity(n_layers);
    let mut v_scales = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let dev_idx = gpus.device_for_layer(i);
        let g = &mut gpus.devices[dev_idx];
        k_gpu.push(g.zeros(&[k_elems], DType::F32)?);
        v_gpu.push(g.zeros(&[v_elems], DType::F32)?);
        k_scales.push(g.zeros(&[k_scale_elems], DType::F32)?);
        v_scales.push(g.zeros(&[v_scale_elems], DType::F32)?);
    }
    Ok((k_gpu, v_gpu, k_scales, v_scales))
}

/// Asym{2,3,4} KV-rotation tables replicated to every device. Replaces any
/// previous contents of `gpus.givens_*_per_dev`. Stage 6 forward dispatch
/// reads `gpus.givens_*_per_dev[layer_to_device[i]]` per layer.
fn replicate_givens_to_all_devices(gpus: &mut Gpus, n_blocks: usize, seed: u32) -> HipResult<()> {
    let (cos_vals, sin_vals) = KvCache::gen_givens_angles(seed, n_blocks);
    let cb: Vec<u8> = cos_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let sb: Vec<u8> = sin_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let prev_cos = std::mem::take(&mut gpus.givens_cos_per_dev);
    let prev_sin = std::mem::take(&mut gpus.givens_sin_per_dev);
    for (i, t) in prev_cos.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }
    for (i, t) in prev_sin.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }

    for dev_idx in 0..gpus.devices.len() {
        let g = &mut gpus.devices[dev_idx];
        let ct = g.alloc_tensor(&[n_blocks], DType::F32)?;
        let st = g.alloc_tensor(&[n_blocks], DType::F32)?;
        g.hip.memcpy_htod(&ct.buf, &cb)?;
        g.hip.memcpy_htod(&st.buf, &sb)?;
        gpus.givens_cos_per_dev.push(ct);
        gpus.givens_sin_per_dev.push(st);
    }
    Ok(())
}

/// Multi-device replication of signed-FWHT sign vectors. Mirrors
/// `replicate_givens_to_all_devices` but uses gen_fwht_signs (seeds
/// 42/1042, matching the single-GPU `new_gpu_fwht*_filtered` ctors and
/// the MQ4 weight-FWHT convention). signs1/signs2 occupy the same
/// per-device slots as cos/sin — dispatcher branches on `quant_fwht`
/// to pick the kernel signature.
fn replicate_fwht_signs_to_all_devices(gpus: &mut Gpus, n_signs: usize) -> HipResult<()> {
    let s1_vals = KvCache::gen_fwht_signs(42, n_signs);
    let s2_vals = KvCache::gen_fwht_signs(1042, n_signs);
    let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let prev_cos = std::mem::take(&mut gpus.givens_cos_per_dev);
    let prev_sin = std::mem::take(&mut gpus.givens_sin_per_dev);
    for (i, t) in prev_cos.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }
    for (i, t) in prev_sin.into_iter().enumerate() {
        if i < gpus.devices.len() {
            let _ = gpus.devices[i].free_tensor(t);
        }
    }

    for dev_idx in 0..gpus.devices.len() {
        let g = &mut gpus.devices[dev_idx];
        let s1 = g.alloc_tensor(&[n_signs], DType::F32)?;
        let s2 = g.alloc_tensor(&[n_signs], DType::F32)?;
        g.hip.memcpy_htod(&s1.buf, &s1_bytes)?;
        g.hip.memcpy_htod(&s2.buf, &s2_bytes)?;
        gpus.givens_cos_per_dev.push(s1);
        gpus.givens_sin_per_dev.push(s2);
    }
    Ok(())
}

#[cfg(test)]
mod index_math_tests {
    use super::{is_boundary_ordinal, kv_f32_elems_for_bytes, kv_quant_mode_from_flags};

    #[test]
    fn f32_elems_rounds_bytes_up_to_whole_words() {
        // 4 bytes = 1 F32; a partial word must still reserve a whole slot.
        assert_eq!(kv_f32_elems_for_bytes(0), 0);
        assert_eq!(kv_f32_elems_for_bytes(1), 1);
        assert_eq!(kv_f32_elems_for_bytes(3), 1);
        assert_eq!(kv_f32_elems_for_bytes(4), 1);
        assert_eq!(kv_f32_elems_for_bytes(5), 2);
        assert_eq!(kv_f32_elems_for_bytes(8), 2);
        assert_eq!(kv_f32_elems_for_bytes(9), 3);
    }

    #[test]
    fn f32_elems_matches_the_legacy_plus3_div4_formula() {
        // Equivalence guard: div_ceil(4) must be bit-for-bit the old
        // `(bytes + 3) / 4` for every byte count the cache can produce.
        for bytes in 0..4096usize {
            assert_eq!(kv_f32_elems_for_bytes(bytes), (bytes + 3) / 4, "bytes={bytes}");
        }
    }

    #[test]
    fn f32_elems_does_not_overflow_near_usize_max() {
        // div_ceil saturates the +3 internally; the old (x+3) would wrap. Guard
        // that huge (nonsensical but non-panicking) inputs stay monotonic.
        assert_eq!(kv_f32_elems_for_bytes(usize::MAX), usize::MAX / 4 + 1);
    }

    #[test]
    fn boundary_ordinal_is_bounds_checked() {
        let flags = [true, false, false, true];
        assert!(is_boundary_ordinal(&flags, 0));
        assert!(!is_boundary_ordinal(&flags, 1));
        assert!(is_boundary_ordinal(&flags, 3));
        // Out-of-range ordinal must be false, not a panic.
        assert!(!is_boundary_ordinal(&flags, 4));
        assert!(!is_boundary_ordinal(&flags, usize::MAX));
        assert!(!is_boundary_ordinal(&[], 0));
    }

    // 3.10: lock the boolean→KvQuantMode mapping so the eventual field-level
    // refactor (booleans → enum) has a tested oracle. Flag order:
    // (quantized, q8, int8, hfq4, asym4, asym3, asym2, fwht, kvarn).
    #[test]
    fn kv_quant_mode_derives_each_exclusive_tier() {
        use super::KvQuantMode::*;
        let f = kv_quant_mode_from_flags;
        // Unquantized ignores every other flag.
        assert_eq!(f(false, true, true, true, true, true, true, true, true), Unquantized);
        assert_eq!(f(false, false, false, false, false, false, false, false, false), Unquantized);
        // Each co-located / separate-scale tier.
        assert_eq!(f(true, true, false, false, false, false, false, false, false), Q8);
        assert_eq!(f(true, false, true, false, false, false, false, false, false), Int8);
        assert_eq!(f(true, false, false, true, false, false, false, false, false), Hfq4);
        assert_eq!(f(true, false, false, false, false, false, false, false, true), Kvarn);
        // Asym tiers, Givens (fwht=false) vs signed-FWHT (fwht=true).
        assert_eq!(f(true, false, false, false, true, false, false, false, false), Asym4);
        assert_eq!(f(true, false, false, false, true, false, false, true, false), Fwht4);
        assert_eq!(f(true, false, false, false, false, true, false, false, false), Asym3);
        assert_eq!(f(true, false, false, false, false, true, false, true, false), Fwht3);
        assert_eq!(f(true, false, false, false, false, false, true, false, false), Asym2);
        assert_eq!(f(true, false, false, false, false, false, true, true, false), Fwht2);
    }

    #[test]
    fn kv_quant_mode_precedence_matches_constructor_priority() {
        use super::KvQuantMode::*;
        // kvarn wins over any asym flag (the KVarN constructors leave asym
        // false, but the precedence must be explicit and stable).
        assert_eq!(
            kv_quant_mode_from_flags(true, false, false, false, true, false, false, false, true),
            Kvarn
        );
        // fwht is a modifier, not a standalone tier: fwht=true with no asym tier
        // set does not fabricate an Fwht* mode.
        assert_eq!(
            kv_quant_mode_from_flags(true, true, false, false, false, false, false, true, false),
            Q8
        );
    }
}
