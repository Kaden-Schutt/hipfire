// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KV cache, extracted from hipfire-runtime/src/llama.rs.
//! This module is the canonical home for `KvCache` and its supporting
//! types. `hipfire-runtime::llama` re-exports these for backward
//! compatibility; new code should import from `saddle_core::kv`.

use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

/// The resolved, validated KV-cache mode (plus one resolver-internal sentinel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMode {
    Q8,
    Asym2,
    Asym3,
    Asym4,
    Fwht2,
    Fwht3,
    Fwht4,
    /// SENTINEL — not an allocatable mode. Emitted only by site 3's
    /// `normalize_dir` for its `auto`-set (`"" | "auto" | "turbo" | "turbo3"`).
    /// `resolve` collapses it to `Asym3` (head_dim == 256) or `Q8` (else)
    /// *before returning*, so `KvCache::from_mode` never sees it.
    Asym3Auto,
}

/// KV storage backend selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvBackend {
    #[default]
    Contiguous,
    Vmm,
}

impl KvBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Vmm => "vmm",
        }
    }
}

impl std::fmt::Display for KvBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseKvBackendError {
    value: String,
}

pub const KV_BACKEND_NAMES: &[&str] = &["contiguous", "vmm"];

pub const DEFAULT_KV_CHUNK_TOKENS: usize = 64;
pub const DEFAULT_VMM_PHYSICAL_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvMapGrowth {
    pub offset_bytes: usize,
    pub size_bytes: usize,
    pub token_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvChunkPlan {
    bytes_per_token: usize,
    max_tokens: usize,
    target_chunk_tokens: usize,
    granularity: usize,
    reserve_bytes: usize,
    growth_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvChunkPlanError {
    Zero(&'static str),
    Overflow,
    RequiredTokens { required: usize, max: usize },
    InvalidMappedBytes { mapped: usize, reserve: usize },
}

impl std::fmt::Display for KvChunkPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(field) => write!(f, "{field} must be greater than zero"),
            Self::Overflow => f.write_str("KV chunk byte calculation overflowed"),
            Self::RequiredTokens { required, max } => {
                write!(f, "required token count {required} exceeds maximum {max}")
            }
            Self::InvalidMappedBytes { mapped, reserve } => write!(
                f,
                "current mapped byte count {mapped} is not page-aligned or exceeds reserve {reserve}"
            ),
        }
    }
}

impl std::error::Error for KvChunkPlanError {}

impl KvChunkPlan {
    pub fn new(
        bytes_per_token: usize,
        max_tokens: usize,
        target_chunk_tokens: usize,
        granularity: usize,
        minimum_growth_bytes: usize,
    ) -> Result<Self, KvChunkPlanError> {
        for (name, value) in [
            ("bytes_per_token", bytes_per_token),
            ("max_tokens", max_tokens),
            ("target_chunk_tokens", target_chunk_tokens),
            ("granularity", granularity),
            ("minimum_growth_bytes", minimum_growth_bytes),
        ] {
            if value == 0 {
                return Err(KvChunkPlanError::Zero(name));
            }
        }

        let logical_bytes = bytes_per_token
            .checked_mul(max_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        let target_bytes = bytes_per_token
            .checked_mul(target_chunk_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        let reserve_bytes = checked_round_up(logical_bytes, granularity)?;
        let growth_bytes = checked_round_up(target_bytes.max(minimum_growth_bytes), granularity)?;

        Ok(Self {
            bytes_per_token,
            max_tokens,
            target_chunk_tokens,
            granularity,
            reserve_bytes,
            growth_bytes,
        })
    }

    pub const fn reserve_bytes(self) -> usize {
        self.reserve_bytes
    }

    pub const fn growth_bytes(self) -> usize {
        self.growth_bytes
    }

    pub const fn bytes_per_token(self) -> usize {
        self.bytes_per_token
    }

    pub const fn target_chunk_tokens(self) -> usize {
        self.target_chunk_tokens
    }

    pub const fn granularity(self) -> usize {
        self.granularity
    }

    pub fn mapped_bytes_for_tokens(
        self,
        required_tokens: usize,
    ) -> Result<usize, KvChunkPlanError> {
        if required_tokens > self.max_tokens {
            return Err(KvChunkPlanError::RequiredTokens {
                required: required_tokens,
                max: self.max_tokens,
            });
        }
        if required_tokens == 0 {
            return Ok(0);
        }

        let required_bytes = self
            .bytes_per_token
            .checked_mul(required_tokens)
            .ok_or(KvChunkPlanError::Overflow)?;
        Ok(checked_round_up(required_bytes, self.growth_bytes)?.min(self.reserve_bytes))
    }

    pub fn token_capacity(self, mapped_bytes: usize) -> usize {
        (mapped_bytes / self.bytes_per_token).min(self.max_tokens)
    }

    pub fn growth(
        self,
        mapped_bytes: usize,
        required_tokens: usize,
    ) -> Result<Option<KvMapGrowth>, KvChunkPlanError> {
        if mapped_bytes > self.reserve_bytes || !mapped_bytes.is_multiple_of(self.granularity) {
            return Err(KvChunkPlanError::InvalidMappedBytes {
                mapped: mapped_bytes,
                reserve: self.reserve_bytes,
            });
        }
        let target = self.mapped_bytes_for_tokens(required_tokens)?;
        if target <= mapped_bytes {
            return Ok(None);
        }
        Ok(Some(KvMapGrowth {
            offset_bytes: mapped_bytes,
            size_bytes: target - mapped_bytes,
            token_capacity: self.token_capacity(target),
        }))
    }
}

fn checked_round_up(value: usize, alignment: usize) -> Result<usize, KvChunkPlanError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(KvChunkPlanError::Overflow)
    }
}

impl std::fmt::Display for ParseKvBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown KV backend {:?}; expected one of: {}",
            self.value,
            KV_BACKEND_NAMES.join(", ")
        )
    }
}

impl std::error::Error for ParseKvBackendError {}

impl std::str::FromStr for KvBackend {
    type Err = ParseKvBackendError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "contiguous" => Ok(Self::Contiguous),
            "vmm" => Ok(Self::Vmm),
            other => Err(ParseKvBackendError {
                value: other.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMode {
    Q8,
    Lloyd2,
    Lloyd3,
    Lloyd4,
}

impl VMode {
    /// Kernarg value: the per-element bit count (8 for Q8). Drives both kernel
    /// dispatch branches and byte-layout arithmetic.
    pub fn bits(self) -> i32 {
        match self {
            VMode::Q8 => 8,
            VMode::Lloyd2 => 2,
            VMode::Lloyd3 => 3,
            VMode::Lloyd4 => 4,
        }
    }
}

/// Two capacity axes live here:
///   * `max_seq`       — advertised absolute-position range (used for RoPE phase,
///                       attention masks, and anything that reasons about the
///                       user-visible context window).
///   * `physical_cap`  — actual buffer size along the token axis (drives
///                       allocation + kernel strides). When eviction is active,
///                       `physical_cap << max_seq` so the buffer stays bounded
///                       even as the absolute position grows past it.
///
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
    /// V-cache quantization mode (independent of the K mode). Defaults to Q8.
    pub v_mode: VMode,
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
}

/// Layer addressing for [`KvCache::from_mode`]: a per-layer "is this a
/// full-attention layer" mask (→ `_filtered` family) OR a flat layer count
/// (→ plain / flat-`_capped` family).
pub enum KvLayers {
    /// `is_kv_layer` mask — sites 1, 2, 6.
    Mask(Vec<bool>),
    /// `n_layers` — sites 3, 4, 5.
    Flat(usize),
}

/// Geometry + cap inputs shared by every `new_gpu_*` constructor.
pub struct KvDims {
    pub layers: KvLayers,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// For minimax (site 5a) this MUST be the CLAMPED value (12288), not the
    /// raw `ctx.max_seq`, or the allocation size changes.
    pub max_seq: usize,
    /// `Some(cap)` → request a `_capped` form. HONORED ONLY for modes that have
    /// one (q8/asym3/fwht2/fwht3 on Mask sites; q8/asym3/asym4 on Flat sites);
    /// silently DROPPED for asym2/asym4/fwht4 on Mask sites — faithful to today.
    pub physical_cap: Option<usize>,
}

/// Single- vs multi-GPU dispatch for [`KvCache::from_mode`].

/// Single source of truth for VMM K/V byte layout.
///
/// - `k_bytes_per_token` / `v_bytes_per_token` are **current** encoding strides
///   (drive mapped-token capacity, growth, and live source-prefix sizing).
/// - `k_reserve_*` describe the virtual VA arena size (static: reserve == current;
///   adaptive may reserve at floor while current encoding is FWHT4/Q8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VmmKvLayout {
    mode: KvMode,
    v_mode: VMode,
    n_kv_heads: usize,
    head_dim: usize,
    physical_cap: usize,
    kv_dim: usize,
    k_bytes_per_head: usize,
    v_bytes_per_head: usize,
    k_bytes_per_token: usize,
    v_bytes_per_token: usize,
    /// Virtual reserve bytes at the reserve tier (may differ from current).
    k_reserve_bytes: usize,
    v_reserve_bytes: usize,
    k_reserve_elems: usize,
    v_reserve_elems: usize,
    /// Sign/angle table width. 0 = none (Q8). 128 = Givens or FWHT-128.
    /// 256 = FWHT-256 (fwht3, or any FWHT+lloyd-V).
    rotation_table_len: usize,
    uses_fwht_signs: bool,
}

impl VmmKvLayout {
    /// Checked live K source-prefix bytes for `n_positions` at the **current**
    /// K stride. Independent of virtual reserve size — used by adaptive
    /// transcode scratch sizing and map-before-read guards.
    fn prefix_k_bytes(self, n_positions: usize) -> HipResult<usize> {
        KvCache::checked_vmm_product("K source prefix", &[n_positions, self.k_bytes_per_token])
    }

    /// Checked live V source-prefix bytes for `n_positions` at the **current**
    /// V stride. Independent of virtual reserve size.
    fn prefix_v_bytes(self, n_positions: usize) -> HipResult<usize> {
        KvCache::checked_vmm_product("V source prefix", &[n_positions, self.v_bytes_per_token])
    }
}

impl KvCache {
    fn checked_vmm_product(label: &str, factors: &[usize]) -> HipResult<usize> {
        factors
            .iter()
            .try_fold(1usize, |product, &factor| product.checked_mul(factor))
            .ok_or_else(|| hip_bridge::HipError::new(0, &format!("VMM KV {label} size overflowed")))
    }

    fn bytes_to_f32_elems(label: &str, bytes: usize) -> HipResult<usize> {
        bytes.checked_add(3).map(|value| value / 4).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("VMM KV {label} element count overflowed"))
        })
    }

    /// Packed K bytes-per-head for Q8 / asym{2,3,4} / fwht{2,3,4}.
    /// Asym and FWHT share storage at the same bit width; only rotation tables differ.
    fn vmm_k_bytes_per_head(mode: KvMode, head_dim: usize) -> HipResult<usize> {
        match mode {
            KvMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM q8 requires a non-zero head_dim divisible by 32 (got head_dim={head_dim})"
                        ),
                    ));
                }
                Self::checked_vmm_product("q8 K head stride", &[head_dim / 32, 34])
            }
            KvMode::Asym2 | KvMode::Fwht2 => head_dim
                .checked_div(4)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 2-bit K head stride overflowed")),
            KvMode::Asym3 | KvMode::Fwht3 => head_dim
                .checked_mul(3)
                .and_then(|n| n.checked_div(8))
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 3-bit K head stride overflowed")),
            KvMode::Asym4 | KvMode::Fwht4 => head_dim
                .checked_div(2)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| hip_bridge::HipError::new(0, "VMM 4-bit K head stride overflowed")),
            KvMode::Asym3Auto => Err(hip_bridge::HipError::new(
                0,
                "KV mode Asym3Auto must be resolved before VMM layout",
            )),
        }
    }

    /// Packed V bytes-per-head for Q8 / Lloyd{2,3,4}.
    fn vmm_v_bytes_per_head(v_mode: VMode, head_dim: usize) -> HipResult<usize> {
        match v_mode {
            VMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM Q8-V requires a non-zero head_dim divisible by 32 (got head_dim={head_dim})"
                        ),
                    ));
                }
                Self::checked_vmm_product("q8 V head stride", &[head_dim / 32, 34])
            }
            VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                let bits = v_mode.bits() as usize;
                head_dim
                    .checked_mul(bits)
                    .and_then(|n| n.checked_div(8))
                    .and_then(|n| n.checked_add(4))
                    .ok_or_else(|| {
                        hip_bridge::HipError::new(
                            0,
                            &format!("VMM lloyd{bits} V head stride overflowed"),
                        )
                    })
            }
        }
    }

    /// Geometry gates shared by every static VMM mode (and legal Lloyd-V pairs).
    fn validate_vmm_static_geometry(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        if n_kv_heads == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("VMM {mode:?} requires n_kv_heads>0 (got 0)"),
            ));
        }
        match mode {
            KvMode::Q8 => {
                if head_dim == 0 || !head_dim.is_multiple_of(32) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM q8 requires n_kv_heads>0 and a non-zero head_dim divisible by 32 (got n_kv_heads={n_kv_heads}, head_dim={head_dim})"
                        ),
                    ));
                }
                if !matches!(v_mode, VMode::Q8) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM q8 only supports VMode::Q8 (lloyd-V requires an FWHT K mode)",
                    ));
                }
            }
            KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4 => {
                let ok_hd = match mode {
                    KvMode::Asym3 => head_dim == 256,
                    _ => head_dim == 128 || head_dim == 256,
                };
                if !ok_hd {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM {mode:?} requires head_dim={} (got {head_dim})",
                            if matches!(mode, KvMode::Asym3) {
                                "256"
                            } else {
                                "128 or 256"
                            }
                        ),
                    ));
                }
                if !matches!(v_mode, VMode::Q8) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM {mode:?} only supports VMode::Q8 (lloyd-V requires an FWHT K mode)"
                        ),
                    ));
                }
            }
            KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4 => match v_mode {
                VMode::Q8 => {
                    let ok_hd = match mode {
                        KvMode::Fwht3 => head_dim == 256,
                        _ => head_dim == 128 || head_dim == 256,
                    };
                    if !ok_hd {
                        return Err(hip_bridge::HipError::new(
                            0,
                            &format!(
                                "VMM {mode:?} requires head_dim={} (got {head_dim})",
                                if matches!(mode, KvMode::Fwht3) {
                                    "256"
                                } else {
                                    "128 or 256"
                                }
                            ),
                        ));
                    }
                }
                VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                    if head_dim != 256 {
                        return Err(hip_bridge::HipError::new(
                            0,
                            &format!(
                                "VMM lloyd-V requires head_dim=256 with FWHT K (got mode={mode:?}, head_dim={head_dim})"
                            ),
                        ));
                    }
                }
            },
            KvMode::Asym3Auto => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "KV mode Asym3Auto must be resolved before VMM layout",
                ));
            }
        }
        Ok(())
    }

    /// Build a static VMM layout (reserve tier == current tier).
    fn vmm_static_layout(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<VmmKvLayout> {
        Self::validate_vmm_static_geometry(mode, v_mode, n_kv_heads, head_dim)?;
        if physical_cap == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM physical_cap must be > 0"));
        }
        let kv_dim = Self::checked_vmm_product("logical", &[n_kv_heads, head_dim])?;
        let k_bytes_per_head = Self::vmm_k_bytes_per_head(mode, head_dim)?;
        let v_bytes_per_head = Self::vmm_v_bytes_per_head(v_mode, head_dim)?;
        let k_bytes_per_token =
            Self::checked_vmm_product("K token stride", &[n_kv_heads, k_bytes_per_head])?;
        let v_bytes_per_token =
            Self::checked_vmm_product("V token stride", &[n_kv_heads, v_bytes_per_head])?;
        let k_reserve_bytes =
            Self::checked_vmm_product("K reserve", &[physical_cap, k_bytes_per_token])?;
        let v_reserve_bytes =
            Self::checked_vmm_product("V reserve", &[physical_cap, v_bytes_per_token])?;
        let rotation_table_len = match mode {
            KvMode::Q8 => 0,
            KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4 => head_dim / 2,
            KvMode::Fwht3 => 256,
            KvMode::Fwht2 | KvMode::Fwht4 => {
                // Q8-V uses 128-wide signs; lloyd-V needs 256-wide up front so
                // constructors never replace owners after publish.
                if matches!(v_mode, VMode::Q8) {
                    128
                } else {
                    256
                }
            }
            KvMode::Asym3Auto => 0,
        };
        let uses_fwht_signs = matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4);
        Ok(VmmKvLayout {
            mode,
            v_mode,
            n_kv_heads,
            head_dim,
            physical_cap,
            kv_dim,
            k_bytes_per_head,
            v_bytes_per_head,
            k_bytes_per_token,
            v_bytes_per_token,
            k_reserve_bytes,
            v_reserve_bytes,
            k_reserve_elems: Self::bytes_to_f32_elems("K reserve", k_reserve_bytes)?,
            v_reserve_elems: Self::bytes_to_f32_elems("V reserve", v_reserve_bytes)?,
            rotation_table_len,
            uses_fwht_signs,
        })
    }

    /// Floor-reserved layout: current encoding strides may differ from reserve.
    /// Adaptive uses this with start FWHT4/Q8 and floor reserve bph/v_mode.
    fn vmm_layout_with_reserve(
        mode: KvMode,
        v_mode: VMode,
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
        reserve_k_bytes_per_head: usize,
        reserve_v_mode: VMode,
    ) -> HipResult<VmmKvLayout> {
        Self::validate_vmm_static_geometry(mode, v_mode, n_kv_heads, head_dim)?;
        if physical_cap == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM physical_cap must be > 0"));
        }
        if reserve_k_bytes_per_head == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "VMM reserve K bytes-per-head must be > 0",
            ));
        }
        let kv_dim = Self::checked_vmm_product("logical", &[n_kv_heads, head_dim])?;
        let k_bytes_per_head = Self::vmm_k_bytes_per_head(mode, head_dim)?;
        let v_bytes_per_head = Self::vmm_v_bytes_per_head(v_mode, head_dim)?;
        let reserve_v_bph = Self::vmm_v_bytes_per_head(reserve_v_mode, head_dim)?;
        let k_bytes_per_token =
            Self::checked_vmm_product("K token stride", &[n_kv_heads, k_bytes_per_head])?;
        let v_bytes_per_token =
            Self::checked_vmm_product("V token stride", &[n_kv_heads, v_bytes_per_head])?;
        let k_reserve_token =
            Self::checked_vmm_product("K reserve token", &[n_kv_heads, reserve_k_bytes_per_head])?;
        let v_reserve_token =
            Self::checked_vmm_product("V reserve token", &[n_kv_heads, reserve_v_bph])?;
        let k_reserve_bytes =
            Self::checked_vmm_product("K reserve", &[physical_cap, k_reserve_token])?;
        let v_reserve_bytes =
            Self::checked_vmm_product("V reserve", &[physical_cap, v_reserve_token])?;
        // Adaptive / lloyd paths need 256-wide signs (q8→lloyd, optional fwht4→fwht3).
        let rotation_table_len = if matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
            || !matches!(reserve_v_mode, VMode::Q8)
            || !matches!(v_mode, VMode::Q8)
        {
            256
        } else if matches!(mode, KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4) {
            head_dim / 2
        } else {
            0
        };
        let uses_fwht_signs = matches!(mode, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
            || !matches!(reserve_v_mode, VMode::Q8);
        Ok(VmmKvLayout {
            mode,
            v_mode,
            n_kv_heads,
            head_dim,
            physical_cap,
            kv_dim,
            k_bytes_per_head,
            v_bytes_per_head,
            k_bytes_per_token,
            v_bytes_per_token,
            k_reserve_bytes,
            v_reserve_bytes,
            k_reserve_elems: Self::bytes_to_f32_elems("K reserve", k_reserve_bytes)?,
            v_reserve_elems: Self::bytes_to_f32_elems("V reserve", v_reserve_bytes)?,
            rotation_table_len,
            uses_fwht_signs,
        })
    }

    /// Back-compat wrappers used by older call sites / tests.
    fn q8_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Q8, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.k_bytes_per_token,
        ))
    }

    fn asym3_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Asym3, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        ))
    }

    /// FWHT3 reuses the Asym3 packed-K / Q8-V byte layout; only the rotation
    /// tables and `quant_fwht` flag differ from Asym3 VMM.
    fn fwht3_vmm_layout(
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
    ) -> HipResult<(usize, usize, usize, usize, usize)> {
        let layout =
            Self::vmm_static_layout(KvMode::Fwht3, VMode::Q8, n_kv_heads, head_dim, physical_cap)?;
        Ok((
            layout.kv_dim,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        ))
    }

    /// Flag bundle applied by the unified static VMM constructor.
    /// Returns (quant_q8, quant_asym4, quant_asym3, quant_asym2, quant_fwht).
    /// Pure classification of a `KvMode` into its five VMM layout flags.
    /// Public so callers one layer up can build a cache in a known layout
    /// without duplicating the mapping.
    pub fn vmm_mode_flags(mode: KvMode) -> (bool, bool, bool, bool, bool) {
        match mode {
            KvMode::Q8 => (true, false, false, false, false),
            KvMode::Asym2 => (false, false, false, true, false),
            KvMode::Asym3 => (false, false, true, false, false),
            KvMode::Asym4 => (false, true, false, false, false),
            KvMode::Fwht2 => (false, false, false, true, true),
            KvMode::Fwht3 => (false, false, true, false, true),
            KvMode::Fwht4 => (false, true, false, false, true),
            KvMode::Asym3Auto => (false, false, false, false, false),
        }
    }

    /// Validate a backend request without touching GPU memory.
    pub fn validate_mode_with_backend(
        mode: KvMode,
        backend: KvBackend,
        single_gpu: bool,
        dims: &KvDims,
    ) -> HipResult<()> {
        if mode == KvMode::Asym3Auto {
            return Err(hip_bridge::HipError::new(
                0,
                "KV mode Asym3Auto must be resolved before allocation",
            ));
        }
        if matches!(mode, KvMode::Asym4 | KvMode::Asym3) && dims.head_dim != 256 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "{} KV cache requires head_dim=256; head_dim={} is unsupported",
                    if mode == KvMode::Asym4 {
                        "asym4"
                    } else {
                        "asym3"
                    },
                    dims.head_dim
                ),
            ));
        }
        if backend != KvBackend::Vmm {
            return Ok(());
        }
        if !single_gpu {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' currently supports single-GPU qwen3.5 only",
            ));
        }
        let KvLayers::Mask(is_kv_layer) = &dims.layers else {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' requires qwen3.5's filtered FullAttention layer mask",
            ));
        };
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                "KV backend 'vmm' requires at least one FullAttention layer",
            ));
        }
        let physical_cap = dims.physical_cap.ok_or_else(|| {
            hip_bridge::HipError::new(0, "KV backend 'vmm' requires a physical token capacity")
        })?;
        if physical_cap == 0 || physical_cap > dims.max_seq {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM physical_cap must be in 1..=max_seq (got {physical_cap}, max_seq={})",
                    dims.max_seq
                ),
            ));
        }
        match mode {
            KvMode::Q8
            | KvMode::Asym2
            | KvMode::Asym3
            | KvMode::Asym4
            | KvMode::Fwht2
            | KvMode::Fwht3
            | KvMode::Fwht4 => {
                // Default static V is Q8; Lloyd-V is validated when a constructor
                // is invoked with an explicit v_mode.
                Self::vmm_static_layout(
                    mode,
                    VMode::Q8,
                    dims.n_kv_heads,
                    dims.head_dim,
                    physical_cap,
                )?;
            }
            other => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "KV backend 'vmm' supports kv_mode=q8|asym2|asym3|asym4|fwht2|fwht3|fwht4 only (got {other:?})"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Non-owning view of one lane inside a parent contiguous Q8 cache.
    ///
    /// `self` must have been allocated with total physical capacity
    /// `lanes * lane_capacity`. The returned cache presents ordinary
    /// single-sequence Q8 addressing, so existing prefill kernels can seed one
    /// lane without copying weights or allocating a second cache. The view must
    /// never be passed to [`KvCache::free_gpu`].
    ///
    /// Rejects asym / FWHT / INT8 / HFQ4 / VMM layouts rather than fabricating
    /// support: continuous-batch only targets exact contiguous Q8 HIP.
    pub fn q8_lane_view(&self, lane: usize, lane_capacity: usize) -> HipResult<Self> {
        if !self.quant_q8
            || self.quant_int8
            || self.quant_hfq4
            || self.quant_asym4
            || self.quant_asym3
            || self.quant_asym2
            || self.quant_fwht
            || self.v_mode != VMode::Q8
            || lane_capacity == 0
        {
            return Err(hip_bridge::HipError::new(
                0,
                "q8_lane_view requires a contiguous Q8 cache and non-zero lane capacity",
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(32) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "q8_lane_view requires head_dim divisible by 32 (got {})",
                    self.head_dim
                ),
            ));
        }
        // Fail closed on VMM-backed parent storage: lane views only slice
        // contiguous hipMalloc buffers via borrowed DeviceBuffer pointers.
        for t in self.k_gpu.iter().chain(self.v_gpu.iter()) {
            if t.buf.is_vmm_owner() {
                return Err(hip_bridge::HipError::new(
                    0,
                    "q8_lane_view does not support VMM-backed KV caches",
                ));
            }
        }
        let lane_end = lane
            .checked_add(1)
            .and_then(|n| n.checked_mul(lane_capacity))
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view capacity overflow"))?;
        if lane_end > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                "q8_lane_view lane exceeds backing cache capacity",
            ));
        }
        let blocks_per_pos = self
            .n_kv_heads
            .checked_mul(self.head_dim / 32)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view blocks_per_pos overflow"))?;
        let bytes_per_pos = blocks_per_pos
            .checked_mul(34)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view bytes_per_pos overflow"))?;
        let byte_offset = lane
            .checked_mul(lane_capacity)
            .and_then(|n| n.checked_mul(bytes_per_pos))
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view byte_offset overflow"))?;
        let lane_bytes = lane_capacity
            .checked_mul(bytes_per_pos)
            .ok_or_else(|| hip_bridge::HipError::new(0, "q8_lane_view lane_bytes overflow"))?;
        let lane_elems = Self::bytes_to_f32_elems("q8_lane_view lane", lane_bytes)?;
        let view = |t: &GpuTensor| -> HipResult<GpuTensor> {
            if t.numel() <= 1 {
                // Filtered placeholder layer — keep a non-owning alias.
                return Ok(t.shallow_clone());
            }
            let parent_bytes = t.buf.size();
            let end = byte_offset.checked_add(lane_bytes).ok_or_else(|| {
                hip_bridge::HipError::new(0, "q8_lane_view parent byte range overflow")
            })?;
            if end > parent_bytes {
                return Err(hip_bridge::HipError::new(
                    0,
                    "q8_lane_view lane byte range exceeds parent buffer",
                ));
            }
            let ptr =
                unsafe { (t.buf.as_ptr() as *mut u8).add(byte_offset) as *mut std::ffi::c_void };
            Ok(GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, lane_bytes) },
                shape: vec![lane_elems],
                dtype: DType::F32,
            })
        };
        let k_gpu = self.k_gpu.iter().map(view).collect::<HipResult<Vec<_>>>()?;
        let v_gpu = self.v_gpu.iter().map(view).collect::<HipResult<Vec<_>>>()?;
        Ok(Self {
            k_gpu,
            v_gpu,
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            kv_dim: self.kv_dim,
            max_seq: lane_capacity,
            physical_cap: lane_capacity,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            boundary_layers: self.boundary_layers,
            givens_cos: None,
            givens_sin: None,
            quant_fwht: false,
            v_mode: VMode::Q8,
            layer_is_boundary: self.layer_is_boundary.clone(),
            compact_offset: 0,
        })
    }

    /// Check if a given KV layer ordinal is a boundary layer (first N + last N).
    pub fn is_boundary(&self, kv_ordinal: usize) -> bool {
        kv_ordinal < self.layer_is_boundary.len() && self.layer_is_boundary[kv_ordinal]
    }

    /// Zero every per-layer K/V (and scale) buffer on the GPU. Defense-in-depth
    /// for arch `reset()`: positional KV is normally overwritten by the next
    /// prefill (so the stale tail is never attended), but this guarantees no
    /// prior-conversation bytes can survive a reset even under a future
    /// window/LCP edge that reads an un-rewritten slot. Sub-millisecond memset
    /// of the cache buffers; callers MUST also clear their token mirror so a
    /// zeroed slot can never be stale-LCP-reused.
    pub fn clear_gpu(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for t in self
            .k_gpu
            .iter()
            .chain(self.v_gpu.iter())
            .chain(self.k_scales.iter())
            .chain(self.v_scales.iter())
        {
            let bytes = gpu.vmm_mapped_bytes(t).unwrap_or_else(|| t.buf.size());
            gpu.hip.memset(&t.buf, 0, bytes)?;
        }
        Ok(())
    }

    /// The single dispatcher over the `new_gpu_*` constructor family. Each
    /// non-error arm corresponds 1:1 to a line that exists in a load-site ladder
    /// today (the byte-identical contract). Unreachable `(mode × layers × cap ×
    /// target)` cells return `Err` rather than panic, so a future policy mis-wire
    /// surfaces as a clean load failure.
    pub fn from_mode(mode: KvMode, gpu: &mut Gpu, dims: &KvDims) -> HipResult<Self> {
        Self::from_mode_with_backend(mode, KvBackend::Contiguous, gpu, dims)
    }

    pub fn from_mode_with_backend(
        mode: KvMode,
        backend: KvBackend,
        gpu: &mut Gpu,
        dims: &KvDims,
    ) -> HipResult<Self> {
        Self::validate_mode_with_backend(mode, backend, true, dims)?;
        match backend {
            KvBackend::Contiguous => Self::from_mode_single(mode, gpu, dims),
            KvBackend::Vmm => Self::from_mode_single_vmm(mode, gpu, dims),
        }
    }

    fn from_mode_single_vmm(mode: KvMode, gpu: &mut Gpu, dims: &KvDims) -> HipResult<Self> {
        let KvLayers::Mask(is_kv_layer) = &dims.layers else {
            unreachable!("VMM layer shape validated before dispatch")
        };
        let physical_cap = dims
            .physical_cap
            .expect("VMM physical capacity validated before dispatch");
        // Static path defaults V to Q8. Carrier may call
        // `new_gpu_vmm_capped_filtered` directly with Lloyd-V for FWHT-K.
        match mode {
            KvMode::Q8
            | KvMode::Asym2
            | KvMode::Asym3
            | KvMode::Asym4
            | KvMode::Fwht2
            | KvMode::Fwht3
            | KvMode::Fwht4 => Self::new_gpu_vmm_capped_filtered(
                gpu,
                is_kv_layer,
                dims.n_kv_heads,
                dims.head_dim,
                dims.max_seq,
                physical_cap,
                mode,
                VMode::Q8,
            ),
            other => unreachable!("VMM mode {other:?} validated before dispatch"),
        }
    }

    fn from_mode_single(mode: KvMode, gpu: &mut Gpu, dims: &KvDims) -> HipResult<Self> {
        use KvLayers::*;
        let nh = dims.n_kv_heads;
        let hd = dims.head_dim;
        let ms = dims.max_seq;
        match (mode, &dims.layers, dims.physical_cap) {
            // Mask + Some(cap): _capped_filtered (only q8/asym3/fwht2/fwht3 have it).
            (KvMode::Q8, Mask(m), Some(cap)) => Self::new_gpu_q8_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Asym3, Mask(m), Some(cap)) => Self::new_gpu_asym3_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Fwht2, Mask(m), Some(cap)) => Self::new_gpu_fwht2_capped_filtered(gpu, m, nh, hd, ms, cap),
            (KvMode::Fwht3, Mask(m), Some(cap)) => Self::new_gpu_fwht3_capped_filtered(gpu, m, nh, hd, ms, cap),
            // Mask + cap-but-no-capped-variant: cap DROPPED, use _filtered (faithful).
            (KvMode::Asym2, Mask(m), _) => Self::new_gpu_asym2_filtered(gpu, m, nh, hd, ms),
            (KvMode::Asym4, Mask(m), _) => Self::new_gpu_asym4_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht4, Mask(m), _) => Self::new_gpu_fwht4_filtered(gpu, m, nh, hd, ms),
            // Mask + None for the capped-capable modes: plain _filtered.
            (KvMode::Q8, Mask(m), None) => Self::new_gpu_q8_filtered(gpu, m, nh, hd, ms),
            (KvMode::Asym3, Mask(m), None) => Self::new_gpu_asym3_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht2, Mask(m), None) => Self::new_gpu_fwht2_filtered(gpu, m, nh, hd, ms),
            (KvMode::Fwht3, Mask(m), None) => Self::new_gpu_fwht3_filtered(gpu, m, nh, hd, ms),
            // Flat + Some(cap): _capped (only q8/asym3/asym4).
            (KvMode::Q8, Flat(n), Some(cap)) => Self::new_gpu_q8_capped(gpu, *n, nh, hd, ms, cap),
            (KvMode::Asym3, Flat(n), Some(cap)) => Self::new_gpu_asym3_capped(gpu, *n, nh, hd, ms, cap),
            (KvMode::Asym4, Flat(n), Some(cap)) => Self::new_gpu_asym4_capped(gpu, *n, nh, hd, ms, cap),
            // Flat + None: plain (only q8/asym3/asym4).
            (KvMode::Q8, Flat(n), None) => Self::new_gpu_q8(gpu, *n, nh, hd, ms),
            (KvMode::Asym3, Flat(n), None) => Self::new_gpu_asym3(gpu, *n, nh, hd, ms),
            (KvMode::Asym4, Flat(n), None) => Self::new_gpu_asym4(gpu, *n, nh, hd, ms),
            // No constructor exists for this combination.
            (m, l, c) => Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KvCache::from_mode_single: no constructor for (mode={m:?}, layers={}, cap={c:?}); \
                     unreachable under current policies — a policy/accepted-set mis-wire, not a user error",
                    match l {
                        Mask(_) => "Mask",
                        Flat(_) => "Flat",
                    },
                ),
            )),
        }
    }
}

impl KvCache {
    /// F32 K/V cache with a per-tensor adoption hook. Every K and V tensor is
    /// staged immediately after its allocation, then `hook` runs with the
    /// zero-based tensor index (layer `l`'s K at `2*l`, its V at `2*l + 1`)
    /// BEFORE the next allocation. A hook error — or an allocation error —
    /// rolls back every staged tensor on this GPU and returns the original
    /// error, so a faulted construction never leaks (GpuTensor has no freeing
    /// Drop). The hook is a test-only deterministic fault seam in production
    /// builds: callers that arm no fault pass a no-op.
    pub fn new_gpu_with_hook(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        mut hook: impl FnMut(usize) -> HipResult<()>,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        // Transactional allocation, mirroring `new_gpu_q8_capped`: stage every
        // tensor immediately and, on any mid-loop failure — including the
        // adoption-hook seam below — free every staged tensor on this GPU
        // before returning the original error.
        let result = (|| -> HipResult<()> {
            for layer in 0..n_layers {
                let k = gpu.zeros(&[cache_size], DType::F32)?;
                k_gpu.push(k);
                hook(2 * layer)?;
                let v = gpu.zeros(&[cache_size], DType::F32)?;
                v_gpu.push(v);
                hook(2 * layer + 1)?;
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.into_iter().chain(v_gpu) {
                // Immediate release to the system, not the reuse pool: the
                // failed-construction contract measures free VRAM right after
                // the error returns, before any pool drain. Staged buffers are
                // plain contiguous allocations (gpu.zeros), never VMM owners.
                if let Err(free_err) = gpu.hip.free(tensor.buf) {
                    eprintln!("kv F32 rollback: staged tensor free failed: {free_err}");
                }
            }
            return Err(err);
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
            v_mode: VMode::Q8,
        })
    }

    pub fn new_gpu(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        // No-op adoption hook: identical behavior to the historical direct
        // loop, but the mid-loop rollback now also covers plain allocation
        // failures (strictly safer; success path unchanged).
        Self::new_gpu_with_hook(gpu, n_layers, n_kv_heads, head_dim, max_seq_len, |_| Ok(()))
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
        let cache_elems = (cache_bytes + 3) / 4; // round up to F32 elements
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
            v_mode: VMode::Q8,
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
        let cache_elems = (cache_bytes + 3) / 4;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        // Transactional allocation, mirroring alloc_k_v_filtered: stage every
        // tensor immediately and, on any mid-loop failure — including the
        // test fault seam below — free every staged tensor on this GPU before
        // returning the original error (GpuTensor has no freeing Drop).
        let result = (|| -> HipResult<()> {
            for _ in 0..n_layers {
                k_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
                v_gpu.push(gpu.zeros(&[cache_elems], DType::F32)?);
                // Test-only fault seam (Oracle phase-1 follow-up):
                // deterministic mid-allocation failure after this layer's K/V
                // are owned but before the constructor succeeds. The staging
                // rollback above frees the tensors already pushed. Compiled
                // out of production builds; one-shot.
                #[cfg(test)]
                if fault_seam::consume() {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "kv q8 fault seam: forced allocation failure after first KV layer",
                    ));
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                // Immediate release to the system, not the reuse pool: the
                // failed-construction contract measures free VRAM right after
                // the error returns, before any pool drain. Staged buffers are
                // plain contiguous allocations (gpu.zeros), never VMM owners.
                if let Err(free_err) = gpu.hip.free(tensor.buf) {
                    eprintln!("kv q8 rollback: staged tensor free failed: {free_err}");
                }
            }
            return Err(err);
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
            v_mode: VMode::Q8,
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
        // Contiguous path mirrors alloc_k_v_vmm_filtered: on any mid-loop
        // failure free every tensor already pushed so a partial build never
        // leaks device memory (GpuTensor has no freeing Drop).
        let result = (|| -> HipResult<()> {
            for &is_kv in is_kv_layer {
                if is_kv {
                    k_gpu.push(gpu.zeros(&[k_elems], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[v_elems], DType::F32)?);
                } else {
                    k_gpu.push(gpu.zeros(&[1], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[1], DType::F32)?);
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok((k_gpu, v_gpu))
    }

    /// Reserve dense virtual K/V tensors while mapping physical pages only as
    /// the request grows. Placeholder tensors preserve absolute layer indices.
    fn alloc_k_v_vmm_filtered(
        gpu: &mut Gpu,
        k_elems: usize,
        v_elems: usize,
        is_kv_layer: &[bool],
    ) -> HipResult<(Vec<GpuTensor>, Vec<GpuTensor>)> {
        let mut k_gpu = Vec::with_capacity(is_kv_layer.len());
        let mut v_gpu = Vec::with_capacity(is_kv_layer.len());
        let device_id = gpu.device_id;
        let result = (|| -> HipResult<()> {
            for &is_kv in is_kv_layer {
                if is_kv {
                    // SAFETY: Qwen3.5 mapped-prefix guards run before every
                    // write, attention dispatch, and graph capture/replay.
                    k_gpu.push(unsafe {
                        gpu.alloc_vmm_tensor(&[k_elems], DType::F32, 0, &[device_id])?
                    });
                    v_gpu.push(unsafe {
                        gpu.alloc_vmm_tensor(&[v_elems], DType::F32, 0, &[device_id])?
                    });
                } else {
                    k_gpu.push(gpu.zeros(&[1], DType::F32)?);
                    v_gpu.push(gpu.zeros(&[1], DType::F32)?);
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok((k_gpu, v_gpu))
    }

    /// Resolve the current K `KvMode` from live cache flags.
    /// Read-only: the mode this cache is currently laid out for.
    pub fn current_kv_mode(&self) -> HipResult<KvMode> {
        if self.quant_q8 {
            return Ok(KvMode::Q8);
        }
        if self.quant_asym4 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht4
            } else {
                KvMode::Asym4
            });
        }
        if self.quant_asym3 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht3
            } else {
                KvMode::Asym3
            });
        }
        if self.quant_asym2 {
            return Ok(if self.quant_fwht {
                KvMode::Fwht2
            } else {
                KvMode::Asym2
            });
        }
        Err(hip_bridge::HipError::new(
            0,
            "VMM KV cache encoding flags are not a recognized static mode",
        ))
    }

    /// Current K/V bytes-per-token from live flags / V mode.
    /// Independent K and V strides — capacity is always min-of-two.
    fn vmm_bytes_per_token(&self) -> HipResult<(usize, usize)> {
        if self.n_kv_heads == 0 {
            return Err(hip_bridge::HipError::new(0, "VMM KV requires n_kv_heads>0"));
        }
        let mode = self.current_kv_mode()?;
        // Geometry for the live encoding; lloyd-V only legal on FWHT-K.
        Self::validate_vmm_static_geometry(mode, self.v_mode, self.n_kv_heads, self.head_dim)?;
        let k_bph = Self::vmm_k_bytes_per_head(mode, self.head_dim)?;
        let v_bph = Self::vmm_v_bytes_per_head(self.v_mode, self.head_dim)?;
        let k = Self::checked_vmm_product("K token stride", &[self.n_kv_heads, k_bph])?;
        let v = Self::checked_vmm_product("V token stride", &[self.n_kv_heads, v_bph])?;
        Ok((k, v))
    }

    pub fn uses_vmm_backend(&self) -> bool {
        self.k_gpu
            .iter()
            .chain(self.v_gpu.iter())
            .find(|tensor| tensor.numel() > 1)
            .is_some_and(|tensor| tensor.buf.is_vmm_owner())
    }

    fn fast_mapped_token_capacity(&self) -> HipResult<Option<usize>> {
        if !self.uses_vmm_backend() {
            return Ok(None);
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        let capacity = [
            ("K", self.k_gpu.as_slice(), k_bytes_per_token),
            ("V", self.v_gpu.as_slice(), v_bytes_per_token),
        ]
        .into_iter()
        .try_fold(self.physical_cap, |capacity, (label, tensors, stride)| {
            let tensor = tensors
                .iter()
                .rev()
                .find(|tensor| tensor.numel() > 1)
                .ok_or_else(|| {
                    hip_bridge::HipError::new(
                        0,
                        &format!("VMM KV cache has no real {label} tensor"),
                    )
                })?;
            if !tensor.buf.is_vmm_owner() {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("VMM KV cache has a non-VMM real {label} tensor"),
                ));
            }
            Ok(capacity.min(tensor.buf.size() / stride))
        })?;
        Ok(Some(capacity))
    }

    pub fn mapped_token_capacity(&self) -> HipResult<Option<usize>> {
        if !self.uses_vmm_backend() {
            return Ok(None);
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        let mut capacity = self.physical_cap;
        for (label, tensors, bytes_per_token) in [
            ("K", self.k_gpu.as_slice(), k_bytes_per_token),
            ("V", self.v_gpu.as_slice(), v_bytes_per_token),
        ] {
            for tensor in tensors {
                if tensor.buf.is_vmm_owner() {
                    capacity = capacity.min(tensor.buf.size() / bytes_per_token);
                } else if tensor.numel() > 1 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "VMM KV cache mixes a non-VMM real {label} tensor with VMM tensors"
                        ),
                    ));
                }
            }
        }
        Ok(Some(capacity))
    }

    pub fn require_mapped_capacity(&self, required_tokens: usize) -> HipResult<()> {
        let Some(mapped_tokens) = self.mapped_token_capacity()? else {
            return Ok(());
        };
        if required_tokens > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KV write requires {required_tokens} tokens but physical_cap is {}",
                    self.physical_cap
                ),
            ));
        }
        if required_tokens > mapped_tokens {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM KV mapped prefix holds {mapped_tokens} tokens but the operation requires {required_tokens}"
                ),
            ));
        }
        Ok(())
    }

    /// Grow VMM tensors for one side (K or V).
    ///
    /// `bytes_per_token` is the **current** encoding stride. `physical_cap` is
    /// the constructor token horizon. Per-tensor max tokens is
    /// `min(physical_cap, reserve_bytes / current_stride)` so adaptive
    /// floor-reserved arenas never plan growth past the stable VA.
    fn grow_vmm_tensors(
        gpu: &mut Gpu,
        tensors: &mut [GpuTensor],
        bytes_per_token: usize,
        physical_cap: usize,
        required_tokens: usize,
    ) -> HipResult<usize> {
        if bytes_per_token == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "VMM growth requires a non-zero current byte stride",
            ));
        }
        let mut grown = 0usize;
        let device_id = gpu.device_id;
        let minimum_growth_bytes = DEFAULT_VMM_PHYSICAL_CHUNK_BYTES;
        for tensor in tensors {
            if !tensor.buf.is_vmm_owner() {
                if tensor.numel() > 1 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM KV cache contains a non-VMM real tensor",
                    ));
                }
                continue;
            }
            let mapped = gpu.vmm_mapped_bytes(tensor).ok_or_else(|| {
                hip_bridge::HipError::new(0, "VMM KV tensor is not registered with its GPU")
            })?;
            let granularity = gpu.vmm_granularity(tensor).ok_or_else(|| {
                hip_bridge::HipError::new(0, "VMM KV tensor has no allocation granularity")
            })?;
            // Full reserved VA size (shape × dtype), independent of mapped prefix.
            let reserve_bytes = tensor.byte_size();
            let side_cap = (reserve_bytes / bytes_per_token).min(physical_cap);
            if side_cap == 0 {
                return Err(hip_bridge::HipError::new(
                    0,
                    "VMM KV reserve is smaller than one token at the current stride",
                ));
            }
            if required_tokens > side_cap {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "VMM KV current-stride capacity is {side_cap} tokens but the operation requires {required_tokens}"
                    ),
                ));
            }
            let plan = KvChunkPlan::new(
                bytes_per_token,
                side_cap,
                DEFAULT_KV_CHUNK_TOKENS,
                granularity,
                minimum_growth_bytes,
            )
            .map_err(|err| hip_bridge::HipError::new(0, &err.to_string()))?;
            if let Some(growth) = plan
                .growth(mapped, required_tokens)
                .map_err(|err| hip_bridge::HipError::new(0, &err.to_string()))?
            {
                if gpu.graphs.capture_mode {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "VMM KV growth requested during graph capture",
                    ));
                }
                // Map before any subsequent write; owners stay at the same VA.
                gpu.grow_vmm_tensor(tensor, growth.size_bytes, &[device_id])?;
                grown += 1;
            }
        }
        Ok(grown)
    }

    /// Ensure both K and V mapped prefixes cover `required_tokens` at the
    /// **current** strides. Growth is independent per side and completes before
    /// the caller may write. Never replaces VMM owners.
    pub fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> HipResult<()> {
        let Some(fast_capacity) = self.fast_mapped_token_capacity()? else {
            return Ok(());
        };
        if required_tokens > self.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KV write requires {required_tokens} tokens but physical_cap is {}",
                    self.physical_cap
                ),
            ));
        }
        if required_tokens <= fast_capacity {
            return Ok(());
        }
        let (k_bytes_per_token, v_bytes_per_token) = self.vmm_bytes_per_token()?;
        // Growth is monotonic: if a later map fails, the next call keeps the
        // completed prefixes and fills the lagging tensors before the final guard.
        // K and V are grown independently at their current strides.
        Self::grow_vmm_tensors(
            gpu,
            &mut self.k_gpu,
            k_bytes_per_token,
            self.physical_cap,
            required_tokens,
        )?;
        Self::grow_vmm_tensors(
            gpu,
            &mut self.v_gpu,
            v_bytes_per_token,
            self.physical_cap,
            required_tokens,
        )?;
        self.require_mapped_capacity(required_tokens)?;
        Ok(())
    }

    /// Bytes of V-cache per token-position (all heads) for a given V mode.
    /// Q8 = n_kv_heads * (head_dim/32) * 34. Lloyd = n_kv_heads * (4 + head_dim*bits/8).
    fn v_bytes_per_pos(n_kv_heads: usize, head_dim: usize, v_mode: VMode) -> usize {
        match v_mode {
            VMode::Q8 => n_kv_heads * (head_dim / 32) * 34,
            VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                n_kv_heads * (4 + (head_dim * v_mode.bits() as usize) / 8)
            }
        }
    }

    /// V-mode bit-count to pass as a kernarg.
    pub fn v_mode_bits(&self) -> i32 {
        self.v_mode.bits()
    }

    /// Mode-derived `KvTierInputs` with per-call fields zero-filled. Each
    /// attention dispatch site sets `pos`/`flash_mode`/`capture_mode`/
    /// `batch_size`/`is_tree`/`is_boundary` after this returns (via functional
    /// update). Single source of truth for the cache-stable tier flags,
    /// replacing the four hand-copied literals.
    /// The "quantized but no known format and no separate scales" residual —
    /// the llama legacy Q4 KV path. Shared by `tier_inputs()` and `k_tier()`
    /// so the two derivations can never silently diverge.
    pub fn quant_q4_residual(&self) -> bool {
        // The llama-legacy "plain Q4" KV tier is the residual: quantized but
        // none of the named tiers. MUST also exclude asym{2,3,4} — those set
        // `quantized:true` with empty `k_scales`, so without this guard an asym
        // cache reports BOTH its asym flag AND quant_q4, tripping classify()'s
        // "at most one tier flag" debug_assert (debug-build panic on the qwen35
        // asym3 default) and breaking the byte-identical-to-legacy-literal
        // invariant (the legacy qwen35 literals hardcoded quant_q4 = false).
        // Release classify() output is unchanged either way (asym is matched
        // before q4), so this is a true no-op for kernel selection.
        self.quantized
            && !self.quant_hfq4
            && !self.quant_q8
            && !self.quant_int8
            && !self.quant_asym4
            && !self.quant_asym3
            && !self.quant_asym2
            && self.k_scales.is_empty()
    }

    /// HFQ8 flat-layout KV: quantized with a separate per-block scale table,
    /// and none of the other named tiers. Mirrors the hand ladder's hfq8 branch
    /// condition (`llama.rs` `llama_kv_write_attend` `quantized && !k_scales.is_empty()
    /// && !int8 && !q8`), extended with the exclusions that keep exactly one tier
    /// flag set (asym caches set `quantized` with EMPTY `k_scales`; hfq4 is matched
    /// earlier) so classify()'s "at most one tier flag" debug_assert holds.
    pub fn is_hfq8_kv(&self) -> bool {
        self.quantized
            && !self.k_scales.is_empty()
            && !self.quant_int8
            && !self.quant_q8
            && !self.quant_hfq4
            && !self.quant_asym4
            && !self.quant_asym3
            && !self.quant_asym2
    }

    /// The sealed decode of this cache's storage tier. The sanctioned way to
    /// branch on KV quant mode — replaces hand-rolled `if kv.quant_q8 { … }` ladders.

    fn resize_real_tensors_zeroed(
        gpu: &mut Gpu,
        tensors: &mut [GpuTensor],
        elems: usize,
    ) -> HipResult<()> {
        let real: Vec<usize> = tensors
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t.numel() > 1).then_some(i))
            .collect();
        for &i in &real {
            let placeholder = gpu.zeros(&[1], DType::F32)?;
            let old = std::mem::replace(&mut tensors[i], placeholder);
            let _ = gpu.free_tensor(old);
        }
        gpu.drain_pool();
        for &i in &real {
            let new_tensor = gpu.zeros(&[elems], DType::F32)?;
            let placeholder = std::mem::replace(&mut tensors[i], new_tensor);
            let _ = gpu.free_tensor(placeholder);
        }
        gpu.drain_pool();
        Ok(())
    }

    /// Reallocate the V buffers for a new V mode (used by eval/bench to set an
    /// independent V quant after construction). Re-sizes only real KV layers
    /// (placeholder 1-element buffers for non-KV layers are left as-is).
    /// K buffers and rotation tables are untouched except when enabling lloyd-V
    /// on fwht2/4-K caches (128-element signs → reallocated to 256; the 128-wide
    /// K rotation reads only indices 0..127 so the LCG prefix is byte-identical).
    /// Note: single-GPU only; multi-GPU V-mode wiring is deferred (plan Task 9).
    pub fn set_v_mode_realloc(&mut self, gpu: &mut Gpu, v_mode: VMode) -> HipResult<()> {
        assert!(
            ((self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht)
                || matches!(v_mode, VMode::Q8),
            "lloyd-V is 256-wide and requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht); got a different K mode — would corrupt the V cache"
        );
        if !matches!(v_mode, VMode::Q8) {
            assert!(self.head_dim == 256, "lloyd-V requires head_dim == 256");
        }
        // For fwht2/4-K caches the sign tables are 128-element (the K rotation
        // is 128-wide). Lloyd-V is 256-wide and needs 256-element tables.
        // gen_fwht_signs is a pure LCG: gen_fwht_signs(seed,256)[0..128] ==
        // gen_fwht_signs(seed,128), so the K path remains byte-identical after
        // realloc. Skip when signs are already 256 (fwht3) or when givens_cos
        // is None (multi-GPU cache — sign realloc deferred to Task 9).
        if !matches!(v_mode, VMode::Q8) {
            let need_realloc = self.givens_cos.as_ref().map_or(false, |t| t.numel() < 256);
            if need_realloc {
                let n = 256usize;
                let s1v = Self::gen_fwht_signs(42, n);
                let s2v = Self::gen_fwht_signs(1042, n);
                let s1b: Vec<u8> = s1v.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let s2b: Vec<u8> = s2v.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
                let s2 = gpu.alloc_tensor(&[n], DType::F32)?;
                gpu.hip.memcpy_htod(&s1.buf, &s1b)?;
                gpu.hip.memcpy_htod(&s2.buf, &s2b)?;
                if let Some(old) = self.givens_cos.take() {
                    let _ = gpu.free_tensor(old);
                }
                if let Some(old) = self.givens_sin.take() {
                    let _ = gpu.free_tensor(old);
                }
                self.givens_cos = Some(s1);
                self.givens_sin = Some(s2);
            }
        }
        let v_bpp = Self::v_bytes_per_pos(self.n_kv_heads, self.head_dim, v_mode);
        let v_elems = (self.physical_cap * v_bpp + 3) / 4;
        Self::resize_real_tensors_zeroed(gpu, &mut self.v_gpu, v_elems)?;
        self.v_mode = v_mode;
        Ok(())
    }

    /// Adaptive-KV load setup: size the V buffer at the V FLOOR (so the fixed
    /// buffer holds `physical_cap` tokens at the floor; FEWER at higher tiers)
    /// and ensure 256-wide FWHT signs (so the q8→lloyd4 transcode is safe).
    /// `v_mode` STAYS Q8 — the fast, highest-precision start tier the controller
    /// runs until the first threshold. K is untouched (the caller loads K at the
    /// fwht4 footprint). Mirrors the sign-upgrade block of `set_v_mode_realloc`
    /// but reallocs each real V layer to the FLOOR size, not the current mode's
    /// size. Because the floor record (e.g. lloyd2 = 68 B/head) is smaller than
    /// the q8 record (272 B/head), the q8 phase physically holds only
    /// ~physical_cap*68/272 ≈ 0.25*physical_cap positions — exactly why the
    /// controller transcodes before that cap. Single-GPU only (matches
    /// set_v_mode_realloc).
    /// `k_floor_bph` is the K bytes-per-head at the K FLOOR tier (e.g. fwht2 =
    /// 68 @hd=256, fwht4 = 132 @hd=256). When it is SMALLER than the current K
    /// mode's footprint (i.e. the floor is below fwht4), the K buffers are
    /// reallocated to `physical_cap * n_kv_heads * k_floor_bph` so we actually
    /// save K VRAM. K data is still WRITTEN at the fwht4 stride (132 @256) until
    /// `transcode_k_step` runs, so the floor-sized K buffer physically holds
    /// ~physical_cap * k_floor_bph / 132 positions at fwht4 — the controller
    /// transcodes K→fwht2 before that cap. Pass the current K footprint (132 for
    /// a fwht4 cache) to leave K unresized (V-only presets).
    pub fn set_adaptive_floor_alloc(
        &mut self,
        gpu: &mut Gpu,
        v_floor: VMode,
        k_floor_bph: usize,
    ) -> HipResult<()> {
        // VMM owners are never replaced after publication. Adaptive VMM must be
        // constructed via `new_gpu_vmm_adaptive_filtered` with floor reserve.
        if self.uses_vmm_backend() {
            return Err(hip_bridge::HipError::new(
                0,
                "set_adaptive_floor_alloc refuses VMM owners; construct adaptive VMM with floor reserve up front",
            ));
        }
        // Mirror the set_v_mode_realloc guard: lloyd-V is 256-wide and requires
        // an FWHT K mode + head_dim == 256.
        assert!(
            (self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht,
            "adaptive-KV requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht)"
        );
        assert!(self.head_dim == 256, "adaptive-KV requires head_dim == 256");
        assert!(
            !matches!(v_floor, VMode::Q8),
            "adaptive-KV V floor must be a lloyd tier (got Q8); nothing to size down to"
        );
        // Upgrade the FWHT signs to 256-wide (copy of the need_realloc block from
        // set_v_mode_realloc): fwht2/4-K caches allocate only 128-element sign
        // tables; the q8→lloyd4 transcode runs fwht_shfl_forward_256 (reads
        // signs[0..255]). gen_fwht_signs is a pure LCG, so the first 128 entries
        // are byte-identical and the 128-wide K reads are unaffected.
        let need_realloc = self.givens_cos.as_ref().map_or(false, |t| t.numel() < 256);
        if need_realloc {
            let n = 256usize;
            let s1v = Self::gen_fwht_signs(42, n);
            let s2v = Self::gen_fwht_signs(1042, n);
            let s1b: Vec<u8> = s1v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = s2v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
            let s2 = gpu.alloc_tensor(&[n], DType::F32)?;
            gpu.hip.memcpy_htod(&s1.buf, &s1b)?;
            gpu.hip.memcpy_htod(&s2.buf, &s2b)?;
            if let Some(old) = self.givens_cos.take() {
                let _ = gpu.free_tensor(old);
            }
            if let Some(old) = self.givens_sin.take() {
                let _ = gpu.free_tensor(old);
            }
            self.givens_cos = Some(s1);
            self.givens_sin = Some(s2);
        }
        // Size V at the FLOOR tier (not the current v_mode). The q8 start phase
        // simply fits fewer positions in this smaller buffer.
        let v_bpp_floor = Self::v_bytes_per_pos(self.n_kv_heads, self.head_dim, v_floor);
        let v_elems = (self.physical_cap * v_bpp_floor + 3) / 4;
        Self::resize_real_tensors_zeroed(gpu, &mut self.v_gpu, v_elems)?;
        // v_mode STAYS at its current value (Q8 fast start tier); only the buffer
        // size changed.

        // Size K at the K FLOOR (fwht2=68 or fwht3=100 @256) when the floor is
        // below the current fwht4 footprint (132 @256). K data is still WRITTEN
        // at the fwht4 stride until the K transcode fires (fwht4→fwht2 remap, or
        // fwht4→fwht3 re-rotation for k_floor=fwht3), exactly mirroring the V
        // side. The 256-wide sign upgrade above ALSO satisfies the fwht4→fwht3
        // re-rotation's sign-width requirement (inverse-128 + forward-256). The
        // K-mode booleans STAY at fwht4 (start tier).
        let k_bph_cur = 4 + self.head_dim / 2; // fwht4 footprint @ this head_dim
        if k_floor_bph < k_bph_cur {
            let k_bpp_floor = self.n_kv_heads * k_floor_bph;
            let k_elems = (self.physical_cap * k_bpp_floor + 3) / 4;
            Self::resize_real_tensors_zeroed(gpu, &mut self.k_gpu, k_elems)?;
        }
        Ok(())
    }

    /// Restore cache mode flags to the adaptive start tier (K=fwht4, V=q8)
    /// without touching buffer owners. Used by atomic controller+cache reset.
    pub fn restore_adaptive_start_flags(&mut self) {
        self.quant_q8 = false;
        self.quant_int8 = false;
        self.quant_hfq4 = false;
        self.quant_asym4 = true;
        self.quant_asym3 = false;
        self.quant_asym2 = false;
        self.quant_fwht = true;
        self.v_mode = VMode::Q8;
        self.quantized = true;
    }

    /// Adaptive-KV: re-quantize the EXISTING V cache (all written positions of
    /// every real KV layer) from the current `v_mode` to a lower `target` tier,
    /// in place. No realloc — the V buffers are floor-sized (allocated at the V
    /// floor) and the lloyd record is smaller than the q8/higher-lloyd record, so
    /// the in-place ascending transcode is byte-safe (dst stride < src stride;
    /// see the per-kernel headers).
    ///
    /// Supported transitions: Q8→Lloyd4 (FWHT), Lloyd4→Lloyd3, Lloyd4→Lloyd2,
    /// Lloyd3→Lloyd2 (rotated-space remap, no FWHT). `n_positions` is the number
    /// of token positions currently written (seq_pos+1, or physical_cap if
    /// compacted). Reuses self.givens_cos/givens_sin (256-wide FWHT signs) for
    /// the q8→lloyd4 FWHT.
    pub fn transcode_v_step(
        &mut self,
        gpu: &mut Gpu,
        target: VMode,
        n_positions: usize,
    ) -> HipResult<()> {
        // Mirror set_v_mode_realloc's guard: lloyd-V is 256-wide and needs an
        // FWHT K mode + head_dim==256.
        assert!(
            (self.quant_asym2 || self.quant_asym3 || self.quant_asym4) && self.quant_fwht,
            "lloyd-V transcode requires an FWHT K mode (quant_asym{{2,3,4}} && quant_fwht)"
        );
        assert!(
            self.head_dim == 256,
            "lloyd-V transcode requires head_dim == 256"
        );
        assert!(
            !matches!(target, VMode::Q8),
            "transcode_v_step only downshifts (target != Q8)"
        );

        if n_positions == 0 {
            self.v_mode = target;
            gpu.invalidate_for_kv_mode_switch();
            return Ok(());
        }

        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;

        // Determine the kernel for the (current → target) transition.
        #[derive(Clone, Copy)]
        enum Op {
            Q8ToL4,
            Down(i32, i32),
        }
        let op = match (self.v_mode, target) {
            (VMode::Q8, VMode::Lloyd4) => Op::Q8ToL4,
            (VMode::Lloyd4, VMode::Lloyd3) => Op::Down(4, 3),
            (VMode::Lloyd4, VMode::Lloyd2) => Op::Down(4, 2),
            (VMode::Lloyd3, VMode::Lloyd2) => Op::Down(3, 2),
            (cur, tgt) => panic!("unsupported V transcode {cur:?} -> {tgt:?}"),
        };

        // q8→lloyd4 needs the 256-wide FWHT signs (already reallocated to 256 by
        // set_v_mode_realloc when lloyd-V was enabled at load). Take non-owning
        // views so we don't borrow `self` across the v_gpu iteration below.
        let (signs1, signs2) = match op {
            Op::Q8ToL4 => {
                let s1 = self
                    .givens_cos
                    .as_ref()
                    .expect("q8→lloyd4 transcode needs 256-wide FWHT signs");
                let s2 = self
                    .givens_sin
                    .as_ref()
                    .expect("q8→lloyd4 transcode needs 256-wide FWHT signs");
                // The q8→lloyd4 kernel runs fwht_shfl_forward_256 → reads
                // signs[0..255]. fwht2/fwht4 K caches allocate only 128-element
                // sign tables; adaptive's load path MUST upgrade them to 256
                // (LCG prefix keeps the K-side 128-wide reads byte-identical)
                // before the first transcode. Fail loud rather than OOB-read
                // phantom signs and silently corrupt every position's cnorm.
                assert!(
                    s1.numel() >= 256 && s2.numel() >= 256,
                    "q8→lloyd4 transcode requires 256-wide FWHT signs (got {}); \
                     upgrade fwht2/4 signs to 256 at adaptive load before transcode_v_step",
                    s1.numel()
                );
                (
                    Some(s1.sub_offset(0, s1.numel())),
                    Some(s2.sub_offset(0, s2.numel())),
                )
            }
            Op::Down(_, _) => (None, None),
        };

        // Map the live prefix at the CURRENT V stride before any read/write.
        // Floor-reserved VMM VA may be larger than the mapped prefix — never
        // size or copy from the full virtual reserve.
        self.ensure_mapped_capacity(gpu, n_positions)?;
        let (cur_k_bpt, cur_v_bpt) = if self.uses_vmm_backend() {
            self.vmm_bytes_per_token()?
        } else {
            // Contiguous: full buffer is mapped; keep prior full-layer scratch size.
            (0, 0)
        };
        let prefix_v_bytes = if self.uses_vmm_backend() {
            Self::checked_vmm_product("V source prefix", &[n_positions, cur_v_bpt])?
        } else {
            0
        };
        let prefix_v_elems = if self.uses_vmm_backend() {
            Self::bytes_to_f32_elems("V source prefix", prefix_v_bytes)?
        } else {
            0
        };

        // 1-layer scratch: VMM uses live source-prefix elems only; contiguous
        // keeps full-layer elems (prior behavior). Copy layer→scratch then
        // transcode scratch→layer (non-aliasing).
        let src_elems = if self.uses_vmm_backend() {
            if prefix_v_elems > 0 {
                Some(prefix_v_elems)
            } else {
                None
            }
        } else {
            self.v_gpu
                .iter()
                .map(|t| t.numel())
                .filter(|&n| n > 1)
                .max()
        };
        let scratch = match src_elems {
            Some(n) => Some(gpu.zeros(&[n], DType::F32)?),
            None => None,
        };
        let _ = cur_k_bpt; // K stride unused on V step

        // Free the scratch on EVERY exit path (GpuTensor has no Drop): capture
        // the first error, break, free, then propagate — a HIP failure mid-pass
        // must not leak the per-layer scratch (multi-MB at long context).
        let mut pending: HipResult<()> = Ok(());
        for t in self.v_gpu.iter() {
            // Skip 1-element placeholder buffers for non-KV layers.
            if t.numel() <= 1 {
                continue;
            }
            let scratch = scratch.as_ref().unwrap();
            // Copy the live source prefix into scratch (device-to-device), then
            // read from scratch and write the compacted record back into the layer.
            let nbytes = if self.uses_vmm_backend() {
                prefix_v_bytes
            } else {
                t.byte_size()
            };
            pending = gpu.hip.memcpy_dtod(&scratch.buf, &t.buf, nbytes);
            if pending.is_err() {
                break;
            }
            pending = match op {
                Op::Q8ToL4 => gpu.transcode_v_q8_to_lloyd4(
                    t,
                    scratch,
                    signs1.as_ref().unwrap(),
                    signs2.as_ref().unwrap(),
                    n_kv_heads,
                    head_dim,
                    n_positions,
                ),
                Op::Down(sb, db) => gpu.transcode_v_lloyd_down(
                    t,
                    scratch,
                    n_kv_heads,
                    head_dim,
                    n_positions,
                    sb,
                    db,
                ),
            };
            if pending.is_err() {
                break;
            }
        }

        if let Some(s) = scratch {
            let _ = gpu.free_tensor(s);
        }
        pending?;
        self.v_mode = target;
        gpu.invalidate_for_kv_mode_switch();
        Ok(())
    }

    /// Adaptive-KV: re-quantize the EXISTING K cache to a lower tier
    /// (`target_bits` ∈ {2,3}). Two supported transitions, both from the fwht4
    /// start tier:
    ///   * fwht4 → fwht2 (`target_bits==2`): SAME-WIDTH 128-LUT remap (the
    ///     balanced/aggressive presets' only K step). Reconstructs from the
    ///     fwht4 record and re-quantizes each rotated dim at 2-bit (128-family);
    ///     no FWHT.
    ///   * fwht4 → fwht3 (`target_bits==3`): RE-ROTATION (128-wide → 256-wide).
    ///     Reconstructs normal-space K (dequant + inverse-128), re-rotates
    ///     256-wide, quantizes to TURBO_C3_256. Engaged only by the advanced
    ///     selector with k_floor=fwht3. fwht3→fwht2 never occurs (K starts at
    ///     fwht4 and balanced_steps adds at most one K step), so it is not
    ///     implemented; any other request errors clearly.
    ///
    /// Per real KV layer: copy the K layer into a 1-layer scratch (d2d), then
    /// transcode scratch→layer (separate buffers, never aliased). cnorm is
    /// recomputed per (head, pos). Then flips the K-mode booleans (clears
    /// quant_asym4; sets quant_asym2 OR quant_asym3; quant_fwht stays true) so
    /// the next forward dispatches the right attention kernel, and invalidates
    /// captured graphs.
    ///
    /// Sign tables: fwht4→fwht2 are both 128-wide and share the SAME signs
    /// (gen_fwht_signs(42/1042); the first 128 entries of a 256-wide table equal
    /// the 128-wide table). fwht4→fwht3 RE-ROTATION needs 256-wide signs (the
    /// inverse-128 reads [0..127], forward-256 reads [0..255]) — adaptive's load
    /// path (set_adaptive_floor_alloc with k_floor=fwht3) upgrades them to 256.
    ///
    /// `n_positions` is the number of token positions currently written. The K
    /// buffer is floor-sized at adaptive load (fwht2=68 or fwht3=100 B/head), so
    /// the fwht4 phase physically holds only ~physical_cap*floor_bph/132
    /// positions — the controller transcodes K before that cap.
    pub fn transcode_k_step(
        &mut self,
        gpu: &mut Gpu,
        target_bits: u32,
        n_positions: usize,
    ) -> HipResult<()> {
        // Determine the current K mode. Adaptive starts at fwht4
        // (quant_asym4 && quant_fwht). Supported steps: fwht4->fwht2 (remap),
        // fwht4->fwht3 (re-rotation).
        let src_is_fwht4 = self.quant_asym4 && self.quant_fwht;
        let cur_label = if !self.quant_fwht {
            "non-fwht"
        } else if self.quant_asym4 {
            "fwht4"
        } else if self.quant_asym3 {
            "fwht3"
        } else if self.quant_asym2 {
            "fwht2"
        } else {
            "unknown"
        };
        if !(src_is_fwht4 && (target_bits == 2 || target_bits == 3)) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "K transcode {cur_label}->fwht{target_bits} not implemented \
                     (only fwht4->fwht2 same-width remap and fwht4->fwht3 re-rotation are supported)"
                ),
            ));
        }
        assert!(
            self.head_dim % 128 == 0,
            "fwht K transcode requires head_dim multiple of 128"
        );
        // fwht4->fwht3 re-rotation is hard-wired to the 128↔256 width crossing.
        if target_bits == 3 {
            assert!(
                self.head_dim == 256,
                "fwht4->fwht3 re-rotation requires head_dim == 256"
            );
        }

        if n_positions == 0 {
            self.quant_asym4 = false;
            if target_bits == 3 {
                self.quant_asym3 = true;
            } else {
                self.quant_asym2 = true;
            }
            // quant_fwht stays true.
            gpu.invalidate_for_kv_mode_switch();
            return Ok(());
        }

        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;

        // For the re-rotation (fwht4->fwht3) the kernel runs fwht_shfl_inverse
        // (128-wide, reads signs[0..127]) then fwht_shfl_forward_256 (reads
        // signs[0..255]). The cache signs MUST be 256-wide — adaptive's load
        // (set_adaptive_floor_alloc with k_floor=fwht3) upgrades them. Fail loud
        // rather than OOB-read phantom signs and silently corrupt every record.
        if target_bits == 3 {
            let n1 = self.givens_cos.as_ref().map_or(0, |t| t.numel());
            let n2 = self.givens_sin.as_ref().map_or(0, |t| t.numel());
            assert!(
                n1 >= 256 && n2 >= 256,
                "fwht4->fwht3 transcode requires 256-wide FWHT signs (got {n1}); \
                 set_adaptive_floor_alloc(k_floor=fwht3) must upgrade signs to 256 first"
            );
        }
        // Take non-owning views of the 256-wide signs for the re-rotation so we
        // don't borrow `self` across the k_gpu iteration below.
        let signs = if target_bits == 3 {
            let s1 = self.givens_cos.as_ref().unwrap();
            let s2 = self.givens_sin.as_ref().unwrap();
            Some((s1.sub_offset(0, s1.numel()), s2.sub_offset(0, s2.numel())))
        } else {
            None
        };

        // Map the live prefix at the CURRENT K stride before any read/write.
        // Floor-reserved VMM VA may be larger than the mapped prefix — never
        // size or copy from the full virtual reserve.
        self.ensure_mapped_capacity(gpu, n_positions)?;
        let (cur_k_bpt, _cur_v_bpt) = if self.uses_vmm_backend() {
            self.vmm_bytes_per_token()?
        } else {
            (0, 0)
        };
        let prefix_k_bytes = if self.uses_vmm_backend() {
            Self::checked_vmm_product("K source prefix", &[n_positions, cur_k_bpt])?
        } else {
            0
        };
        let prefix_k_elems = if self.uses_vmm_backend() {
            Self::bytes_to_f32_elems("K source prefix", prefix_k_bytes)?
        } else {
            0
        };

        // 1-layer scratch: VMM uses live source-prefix elems only; contiguous
        // keeps full-layer elems (prior behavior).
        let src_elems = if self.uses_vmm_backend() {
            if prefix_k_elems > 0 {
                Some(prefix_k_elems)
            } else {
                None
            }
        } else {
            self.k_gpu
                .iter()
                .map(|t| t.numel())
                .filter(|&n| n > 1)
                .max()
        };
        let scratch = match src_elems {
            Some(n) => Some(gpu.zeros(&[n], DType::F32)?),
            None => None,
        };

        // Free the scratch on EVERY exit path (GpuTensor has no Drop): capture
        // the first error, break, free, then propagate.
        let mut pending: HipResult<()> = Ok(());
        for t in self.k_gpu.iter() {
            // Skip 1-element placeholder buffers for non-KV layers.
            if t.numel() <= 1 {
                continue;
            }
            let scratch = scratch.as_ref().unwrap();
            let nbytes = if self.uses_vmm_backend() {
                prefix_k_bytes
            } else {
                t.byte_size()
            };
            pending = gpu.hip.memcpy_dtod(&scratch.buf, &t.buf, nbytes);
            if pending.is_err() {
                break;
            }
            pending = if target_bits == 3 {
                let (s1, s2) = signs.as_ref().unwrap();
                gpu.transcode_k_fwht4_to_fwht3(
                    t,
                    scratch,
                    s1,
                    s2,
                    n_kv_heads,
                    head_dim,
                    n_positions,
                )
            } else {
                gpu.transcode_k_fwht4_to_fwht2(t, scratch, n_kv_heads, head_dim, n_positions)
            };
            if pending.is_err() {
                break;
            }
        }

        if let Some(s) = scratch {
            let _ = gpu.free_tensor(s);
        }
        pending?;

        // Flip the K-mode booleans so the next forward dispatches the target
        // attention kernel. quant_fwht stays true (still FWHT-rotated K).
        self.quant_asym4 = false;
        if target_bits == 3 {
            self.quant_asym3 = true;
        } else {
            self.quant_asym2 = true;
        }
        gpu.invalidate_for_kv_mode_switch();
        Ok(())
    }

    /// Adaptive VMM constructor: reserve at K/V floors, start encoding FWHT4/Q8.
    /// One stable K and one stable V owner per real layer for the load lifetime.
    /// Built on `vmm_layout_with_reserve`; never replaces owners after publish.
    pub fn new_gpu_vmm_adaptive_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        k_floor_bph: usize,
        v_floor: VMode,
    ) -> HipResult<Self> {
        if max_seq == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM requires max_seq > 0",
            ));
        }
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM requires at least one KV-carrying layer",
            ));
        }
        if matches!(v_floor, VMode::Q8) {
            return Err(hip_bridge::HipError::new(
                0,
                "adaptive VMM V floor must be a lloyd tier (got Q8)",
            ));
        }
        if head_dim != 256 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("adaptive VMM requires head_dim=256 (got {head_dim})"),
            ));
        }
        let layout = Self::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            max_seq,
            k_floor_bph,
            v_floor,
        )?;
        let (mut k_gpu, mut v_gpu) = Self::alloc_k_v_vmm_filtered(
            gpu,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            is_kv_layer,
        )?;

        // Adaptive always needs 256-wide FWHT signs (q8→lloyd and optional fwht4→fwht3).
        let n = layout.rotation_table_len.max(256);
        let s1_vals = Self::gen_fwht_signs(42, n);
        let s2_vals = Self::gen_fwht_signs(1042, n);
        let s1_bytes: Vec<u8> = s1_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2_bytes: Vec<u8> = s2_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let tables = (|| -> HipResult<(GpuTensor, GpuTensor)> {
            let s1 = gpu.alloc_tensor(&[n], DType::F32)?;
            let s2 = match gpu.alloc_tensor(&[n], DType::F32) {
                Ok(s2) => s2,
                Err(err) => {
                    let _ = gpu.free_tensor(s1);
                    return Err(err);
                }
            };
            if let Err(err) = gpu.hip.memcpy_htod(&s1.buf, &s1_bytes) {
                let _ = gpu.free_tensor(s1);
                let _ = gpu.free_tensor(s2);
                return Err(err);
            }
            if let Err(err) = gpu.hip.memcpy_htod(&s2.buf, &s2_bytes) {
                let _ = gpu.free_tensor(s1);
                let _ = gpu.free_tensor(s2);
                return Err(err);
            }
            Ok((s1, s2))
        })();
        let (s1, s2) = match tables {
            Ok(pair) => pair,
            Err(err) => {
                for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                    let _ = gpu.free_tensor(tensor);
                }
                return Err(err);
            }
        };

        let mut cache = Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: layout.kv_dim,
            max_seq,
            physical_cap: layout.physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: false,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: true, // FWHT4 start
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: true,
            boundary_layers: 0,
            givens_cos: Some(s1),
            givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8, // start tier
        };
        if let Err(err) = cache.ensure_mapped_capacity(gpu, 1) {
            let _ = cache.free_gpu(gpu);
            return Err(err);
        }
        let n_kv = is_kv_layer.iter().filter(|is_kv| **is_kv).count();
        let mapped = match cache.mapped_token_capacity() {
            Ok(capacity) => capacity.unwrap_or(0),
            Err(err) => {
                let _ = cache.free_gpu(gpu);
                return Err(err);
            }
        };
        eprintln!(
            "KV cache: adaptive vmm ({n_kv}/{} layers; start FWHT4/Q8; K floor {}B/head V floor {:?}; reserve_k={}B reserve_v={}B; mapped_prefix={mapped} / max_seq={max_seq})",
            is_kv_layer.len(),
            k_floor_bph,
            v_floor,
            layout.k_reserve_bytes,
            layout.v_reserve_bytes,
        );
        Ok(cache)
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
        let cache_elems = (cache_bytes + 3) / 4;
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
            v_mode: VMode::Q8,
        })
    }

    /// Unified static VMM constructor for every legal (K mode × V mode) pair.
    ///
    /// Reserves stable K/V VAs from [`vmm_static_layout`], maps an initial
    /// prefix, and never replaces owners afterward. Lloyd-V is accepted only
    /// with FWHT-K and head_dim=256 (validated by the layout table).
    pub fn new_gpu_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
        mode: KvMode,
        v_mode: VMode,
    ) -> HipResult<Self> {
        if physical_cap == 0 || physical_cap > max_seq_len {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "VMM {mode:?} physical_cap must be in 1..=max_seq_len (got physical_cap={physical_cap}, max_seq_len={max_seq_len})"
                ),
            ));
        }
        if !is_kv_layer.iter().any(|is_kv| *is_kv) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("VMM {mode:?} requires at least one KV-carrying layer"),
            ));
        }
        let layout = Self::vmm_static_layout(mode, v_mode, n_kv_heads, head_dim, physical_cap)?;
        let (mut k_gpu, mut v_gpu) = Self::alloc_k_v_vmm_filtered(
            gpu,
            layout.k_reserve_elems,
            layout.v_reserve_elems,
            is_kv_layer,
        )?;

        // Optional rotation tables (Givens angles or FWHT signs). Built before
        // the cache is published so a table failure can roll back arenas.
        let rotation = if layout.rotation_table_len == 0 {
            None
        } else {
            let n = layout.rotation_table_len;
            let (a_vals, b_vals) = if layout.uses_fwht_signs {
                (Self::gen_fwht_signs(42, n), Self::gen_fwht_signs(1042, n))
            } else {
                Self::gen_givens_angles(42, n)
            };
            let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let tables = (|| -> HipResult<(GpuTensor, GpuTensor)> {
                let a = gpu.alloc_tensor(&[n], DType::F32)?;
                let b = match gpu.alloc_tensor(&[n], DType::F32) {
                    Ok(b) => b,
                    Err(err) => {
                        let _ = gpu.free_tensor(a);
                        return Err(err);
                    }
                };
                if let Err(err) = gpu.hip.memcpy_htod(&a.buf, &a_bytes) {
                    let _ = gpu.free_tensor(a);
                    let _ = gpu.free_tensor(b);
                    return Err(err);
                }
                if let Err(err) = gpu.hip.memcpy_htod(&b.buf, &b_bytes) {
                    let _ = gpu.free_tensor(a);
                    let _ = gpu.free_tensor(b);
                    return Err(err);
                }
                Ok((a, b))
            })();
            match tables {
                Ok(pair) => Some(pair),
                Err(err) => {
                    for tensor in k_gpu.drain(..).chain(v_gpu.drain(..)) {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(err);
                }
            }
        };

        let (quant_q8, quant_asym4, quant_asym3, quant_asym2, quant_fwht) =
            Self::vmm_mode_flags(mode);
        let (givens_cos, givens_sin) = match rotation {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };

        let mut cache = Self {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: layout.kv_dim,
            max_seq: max_seq_len,
            physical_cap,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4,
            quant_asym3,
            quant_asym2,
            quant_fwht,
            boundary_layers: 0,
            givens_cos,
            givens_sin,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode,
        };
        // Map initial prefix before any write; roll back on failure.
        if let Err(err) = cache.ensure_mapped_capacity(gpu, 1) {
            let _ = cache.free_gpu(gpu);
            return Err(err);
        }
        let n_kv = is_kv_layer.iter().filter(|is_kv| **is_kv).count();
        let mapped = match cache.mapped_token_capacity() {
            Ok(capacity) => capacity.unwrap_or(0),
            Err(err) => {
                let _ = cache.free_gpu(gpu);
                return Err(err);
            }
        };
        let v_label = match v_mode {
            VMode::Q8 => "Q8".to_string(),
            VMode::Lloyd2 => "lloyd2".to_string(),
            VMode::Lloyd3 => "lloyd3".to_string(),
            VMode::Lloyd4 => "lloyd4".to_string(),
        };
        eprintln!(
            "KV cache: {mode:?} vmm ({n_kv}/{} layers carry KV; K {}B/head + V {v_label} {}B/head; mapped_prefix={mapped} / physical_cap={physical_cap} / max_seq={max_seq_len})",
            is_kv_layer.len(),
            layout.k_bytes_per_head,
            layout.v_bytes_per_head,
        );
        Ok(cache)
    }

    /// VMM-backed Q8 cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_q8_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Q8,
            VMode::Q8,
        )
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
        let cache_elems = (cache_bytes + 3) / 4;
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
            v_mode: VMode::Q8,
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
        let cache_elems = (cache_bytes + 3) / 4;
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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

    /// Gemma4 variant of `new_gpu_asym3` (global 512). Delegates to the
    /// gemma4-capped door so the shared 256-only guard stays unchanged.
    pub fn new_gpu_asym3_gemma4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_asym3_capped_gemma4(
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
            v_mode: VMode::Q8,
        })
    }

    /// VMM-backed Asym3-K/Q8-V cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_asym3_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Asym3,
            VMode::Q8,
        )
    }

    /// VMM-backed FWHT3-K/Q8-V cache for Qwen3.5 hybrid stacks.
    /// Thin wrapper over [`Self::new_gpu_vmm_capped_filtered`].
    pub fn new_gpu_fwht3_vmm_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        Self::new_gpu_vmm_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
            KvMode::Fwht3,
            VMode::Q8,
        )
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
        Self::new_gpu_fwht3_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered fwht3 — layer-filtered (skips non-KV layers) with an
    /// explicit `physical_cap` for TriAttention/CASK eviction. Default path has
    /// `physical_cap == max_seq_len` (no eviction). Byte layout identical to
    /// `asym3_capped_filtered`; rotation primitive is signed-FWHT-256.
    pub fn new_gpu_fwht3_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256,
            "fwht3 currently requires head_dim=256 (Qwen 3.5)"
        );
        Self::new_gpu_fwht3_capped_filtered_inner(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    /// Gemma 4 door for the global-attention KV, whose `global_head_dim` is
    /// 512. Deliberately a SEPARATE entry point: the shared constructor above
    /// keeps its exact 256-only guard for Qwen 3.5/3.6 and every other
    /// architecture. Gemma gets its own door rather than a bound widened on
    /// everyone's behalf — a relaxed shared guard silently admits misconfigured
    /// geometry for architectures that were previously protected from it.
    pub fn new_gpu_fwht3_capped_filtered_gemma4(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256 || head_dim == 512,
            "fwht3 (gemma4) requires head_dim=256 or 512 (got {head_dim})"
        );
        Self::new_gpu_fwht3_capped_filtered_inner(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    fn new_gpu_fwht3_capped_filtered_inner(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
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
            v_mode: VMode::Q8,
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
        Self::new_gpu_asym3_capped_inner(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    /// Gemma 4 door for the global-attention KV, whose `global_head_dim` is
    /// 512. Separate entry point so the shared constructor above keeps its
    /// exact 256-only guard for Qwen 3.5/3.6 and every other architecture.
    pub fn new_gpu_asym3_capped_gemma4(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
        assert!(
            head_dim == 256 || head_dim == 512,
            "asym3 (gemma4) requires head_dim=256 or 512 (got {head_dim})"
        );
        Self::new_gpu_asym3_capped_inner(
            gpu,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            physical_cap,
        )
    }

    fn new_gpu_asym3_capped_inner(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        physical_cap: usize,
    ) -> HipResult<Self> {
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
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
        Self::new_gpu_fwht2_capped_filtered(
            gpu,
            is_kv_layer,
            n_kv_heads,
            head_dim,
            max_seq_len,
            max_seq_len,
        )
    }

    /// Capped + filtered fwht2 — layer-filtered with explicit `physical_cap`
    /// for TriAttention/CASK eviction. Byte layout identical to
    /// `asym2_capped`; rotation primitive is signed-FWHT-128.
    pub fn new_gpu_fwht2_capped_filtered(
        gpu: &mut Gpu,
        is_kv_layer: &[bool],
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
            v_mode: VMode::Q8,
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
            v_mode: VMode::Q8,
        })
    }

    /// Generate deterministic ±1 sign array for FWHT.
    pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                if (state >> 16) & 1 == 1 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect()
    }

    /// Free all GPU tensors in this cache. Call before drop to return VRAM.
    /// After calling, follow with gpu.drain_pool() to actually release memory.
    ///
    /// Contiguous frees keep prior log-and-continue behavior. VMM teardown
    /// failures are aggregated and returned so unload cannot claim success
    /// while arenas remain registered (retry via `Gpu::ensure_vmm_cleaned`).
    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<hip_bridge::HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(err) = r {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        };
        for t in self.k_gpu {
            note(gpu.free_tensor(t));
        }
        for t in self.v_gpu {
            note(gpu.free_tensor(t));
        }
        for t in self.k_scales {
            note(gpu.free_tensor(t));
        }
        for t in self.v_scales {
            note(gpu.free_tensor(t));
        }
        if let Some(t) = self.givens_cos {
            note(gpu.free_tensor(t));
        }
        if let Some(t) = self.givens_sin {
            note(gpu.free_tensor(t));
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
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
}

/// KV VMM-layout and adaptive-reset contract tests.
///
/// These ten tests were dropped when `KvCache` moved out of
/// `hipfire-runtime::llama` (wave 1, C1) and were never restored. Recovered
/// verbatim from `8510ca5f2`. They live inside this module rather than in
/// `tests/` because they exercise private layout internals — moving them out
/// would have meant widening `KvCache`'s public API to keep its own tests.
#[cfg(test)]
mod vmm_layout_tests {
    use super::*;
    use rdna_compute::Gpu;

    fn expected_k_bph(mode: KvMode, head_dim: usize) -> usize {
        match mode {
            KvMode::Q8 => (head_dim / 32) * 34,
            KvMode::Asym2 | KvMode::Fwht2 => 4 + head_dim / 4,
            KvMode::Asym3 | KvMode::Fwht3 => 4 + (head_dim * 3) / 8,
            KvMode::Asym4 | KvMode::Fwht4 => 4 + head_dim / 2,
            KvMode::Asym3Auto => panic!("Asym3Auto is not a layout mode"),
        }
    }

    fn expected_v_bph(v_mode: VMode, head_dim: usize) -> usize {
        match v_mode {
            VMode::Q8 => (head_dim / 32) * 34,
            VMode::Lloyd2 => 4 + head_dim / 4,
            VMode::Lloyd3 => 4 + (head_dim * 3) / 8,
            VMode::Lloyd4 => 4 + head_dim / 2,
        }
    }

    fn flag_standin(mode: KvMode, v_mode: VMode, n_kv_heads: usize, head_dim: usize) -> KvCache {
        let (q8, a4, a3, a2, fwht) = KvCache::vmm_mode_flags(mode);
        KvCache {
            k_gpu: vec![],
            v_gpu: vec![],
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: n_kv_heads * head_dim,
            max_seq: 128,
            physical_cap: 128,
            n_kv_heads,
            head_dim,
            quantized: true,
            quant_q8: q8,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: a4,
            quant_asym3: a3,
            quant_asym2: a2,
            quant_fwht: fwht,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode,
        }
    }

    fn vmm_mask_dims(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        physical_cap: usize,
    ) -> KvDims {
        KvDims {
            layers: KvLayers::Mask(vec![true, false, true]),
            n_kv_heads,
            head_dim,
            max_seq,
            physical_cap: Some(physical_cap),
        }
    }

    #[test]
    fn fwht3_vmm_layout_matches_asym3_byte_geometry() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 128;
        let asym = KvCache::asym3_vmm_layout(n_kv_heads, head_dim, physical_cap).unwrap();
        let fwht = KvCache::fwht3_vmm_layout(n_kv_heads, head_dim, physical_cap).unwrap();
        assert_eq!(fwht, asym, "fwht3 VMM must reuse asym3 K/V byte layout");
        // Explicit stride math: K 100 B/head, V Q8 272 B/head at head_dim=256.
        assert_eq!(fwht.3, 4 + (head_dim * 3) / 8);
        assert_eq!(fwht.4, (head_dim / 32) * 34);
        let k_bytes = physical_cap * n_kv_heads * fwht.3;
        let v_bytes = physical_cap * n_kv_heads * fwht.4;
        assert_eq!(fwht.1, (k_bytes + 3) / 4);
        assert_eq!(fwht.2, (v_bytes + 3) / 4);
        assert_eq!(fwht.0, n_kv_heads * head_dim);
    }

    #[test]
    fn independent_k_v_capacity_math_from_reserve_and_current() {
        // Simulate lopsided adaptive start: K floor 68, V floor 68, current K132/V272.
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 1000;
        let layout = KvCache::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            physical_cap,
            expected_k_bph(KvMode::Fwht2, head_dim),
            VMode::Lloyd2,
        )
        .unwrap();
        let k_tokens = layout.k_reserve_bytes / layout.k_bytes_per_token;
        let v_tokens = layout.v_reserve_bytes / layout.v_bytes_per_token;
        // Must NOT sum K+V bytes into a shared pool.
        let naive_shared = (layout.k_reserve_bytes + layout.v_reserve_bytes)
            / (layout.k_bytes_per_token + layout.v_bytes_per_token);
        assert_ne!(
            k_tokens.min(v_tokens),
            naive_shared,
            "min-of-two must differ from shared-pool sum"
        );
        assert_eq!(k_tokens.min(v_tokens), v_tokens);
    }

    #[test]
    fn validate_mode_admits_all_seven_static_vmm_modes() {
        let dims = vmm_mask_dims(4, 256, 4096, 1024);
        for mode in [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ] {
            KvCache::validate_mode_with_backend(mode, KvBackend::Vmm, true, &dims)
                .unwrap_or_else(|e| panic!("mode={mode:?}: {e}"));
            // Contiguous admission remains open (no VMM-only gate).
            KvCache::validate_mode_with_backend(mode, KvBackend::Contiguous, true, &dims)
                .unwrap_or_else(|e| panic!("contiguous mode={mode:?}: {e}"));
        }
    }

    #[test]
    fn validate_mode_rejects_multi_gpu_flat_and_asym3auto_vmm() {
        let mask = vmm_mask_dims(4, 256, 4096, 1024);
        let err = KvCache::validate_mode_with_backend(KvMode::Fwht3, KvBackend::Vmm, false, &mask)
            .unwrap_err()
            .to_string();
        assert!(err.contains("single-GPU"), "{err}");

        let flat = KvDims {
            layers: KvLayers::Flat(8),
            n_kv_heads: 4,
            head_dim: 256,
            max_seq: 4096,
            physical_cap: Some(1024),
        };
        let err = KvCache::validate_mode_with_backend(KvMode::Fwht3, KvBackend::Vmm, true, &flat)
            .unwrap_err()
            .to_string();
        assert!(err.contains("filtered"), "{err}");

        let err =
            KvCache::validate_mode_with_backend(KvMode::Asym3Auto, KvBackend::Vmm, true, &mask)
                .unwrap_err()
                .to_string();
        assert!(err.contains("Asym3Auto"), "{err}");
    }

    #[test]
    fn vmm_bytes_per_token_matches_layout_for_all_static_modes() {
        let n_kv_heads = 4;
        let head_dim = 256;
        for mode in [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ] {
            let cache = flag_standin(mode, VMode::Q8, n_kv_heads, head_dim);
            let (k, v) = cache.vmm_bytes_per_token().unwrap();
            assert_eq!(k, n_kv_heads * expected_k_bph(mode, head_dim), "{mode:?}");
            assert_eq!(
                v,
                n_kv_heads * expected_v_bph(VMode::Q8, head_dim),
                "{mode:?}"
            );
        }
        // FWHT-K + Lloyd-V current strides.
        for v_mode in [VMode::Lloyd2, VMode::Lloyd3, VMode::Lloyd4] {
            let cache = flag_standin(KvMode::Fwht4, v_mode, n_kv_heads, head_dim);
            let (k, v) = cache.vmm_bytes_per_token().unwrap();
            assert_eq!(
                k,
                n_kv_heads * expected_k_bph(KvMode::Fwht4, head_dim),
                "{v_mode:?}"
            );
            assert_eq!(
                v,
                n_kv_heads * expected_v_bph(v_mode, head_dim),
                "{v_mode:?}"
            );
        }
        // Asym-K + Lloyd-V must fail (illegal pair).
        let bad = flag_standin(KvMode::Asym3, VMode::Lloyd3, n_kv_heads, head_dim);
        assert!(bad.vmm_bytes_per_token().is_err());
    }

    #[test]
    fn vmm_layout_rejects_illegal_pairs() {
        // Asym-K + Lloyd-V is illegal.
        for mode in [KvMode::Asym2, KvMode::Asym3, KvMode::Asym4, KvMode::Q8] {
            let err = KvCache::vmm_static_layout(mode, VMode::Lloyd4, 4, 256, 64)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("lloyd-V") || err.contains("VMode::Q8"),
                "mode={mode:?} err={err}"
            );
        }
        // FWHT3 / Asym3 require head_dim=256.
        for mode in [KvMode::Fwht3, KvMode::Asym3] {
            let err = KvCache::vmm_static_layout(mode, VMode::Q8, 4, 128, 64)
                .unwrap_err()
                .to_string();
            assert!(err.contains("head_dim"), "mode={mode:?} err={err}");
        }
        // n_kv_heads == 0.
        let err = KvCache::vmm_static_layout(KvMode::Fwht3, VMode::Q8, 0, 256, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("n_kv_heads>0"), "{err}");
        // Lloyd-V with FWHT-K but head_dim != 256.
        let err = KvCache::vmm_static_layout(KvMode::Fwht4, VMode::Lloyd2, 4, 128, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("head_dim=256"), "{err}");
    }

    #[test]
    fn vmm_layout_with_reserve_separates_current_and_floor() {
        // Adaptive-style: current FWHT4/Q8, reserve at fwht2/lloyd2 floors.
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 1000;
        let k_floor_bph = expected_k_bph(KvMode::Fwht2, head_dim); // 68
        let layout = KvCache::vmm_layout_with_reserve(
            KvMode::Fwht4,
            VMode::Q8,
            n_kv_heads,
            head_dim,
            physical_cap,
            k_floor_bph,
            VMode::Lloyd2,
        )
        .unwrap();
        // Current strides are start encoding.
        assert_eq!(
            layout.k_bytes_per_token,
            n_kv_heads * expected_k_bph(KvMode::Fwht4, head_dim)
        );
        assert_eq!(
            layout.v_bytes_per_token,
            n_kv_heads * expected_v_bph(VMode::Q8, head_dim)
        );
        // Reserve is floor-sized.
        assert_eq!(
            layout.k_reserve_bytes,
            physical_cap * n_kv_heads * k_floor_bph
        );
        assert_eq!(
            layout.v_reserve_bytes,
            physical_cap * n_kv_heads * expected_v_bph(VMode::Lloyd2, head_dim)
        );
        // Token capacity at start is min of reserve/current (V binds: 68/272).
        let k_cap = layout.k_reserve_bytes / layout.k_bytes_per_token;
        let v_cap = layout.v_reserve_bytes / layout.v_bytes_per_token;
        assert_eq!(
            k_cap,
            physical_cap * k_floor_bph / expected_k_bph(KvMode::Fwht4, head_dim)
        );
        assert_eq!(
            v_cap,
            physical_cap * expected_v_bph(VMode::Lloyd2, head_dim)
                / expected_v_bph(VMode::Q8, head_dim)
        );
        assert!(v_cap < k_cap, "V should bind at FWHT4/Q8 start");
        // Source-prefix bytes remain current-stride × n_pos (not floor).
        assert_eq!(
            layout.prefix_k_bytes(10).unwrap(),
            10 * layout.k_bytes_per_token
        );
        assert_eq!(
            layout.prefix_v_bytes(10).unwrap(),
            10 * layout.v_bytes_per_token
        );
        assert_eq!(layout.rotation_table_len, 256);
    }

    #[test]
    fn vmm_static_layout_covers_all_seven_k_modes_with_q8_v() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 128;
        let modes = [
            KvMode::Q8,
            KvMode::Asym2,
            KvMode::Asym3,
            KvMode::Asym4,
            KvMode::Fwht2,
            KvMode::Fwht3,
            KvMode::Fwht4,
        ];
        for mode in modes {
            let layout =
                KvCache::vmm_static_layout(mode, VMode::Q8, n_kv_heads, head_dim, physical_cap)
                    .unwrap_or_else(|e| panic!("mode={mode:?}: {e}"));
            let k_bph = expected_k_bph(mode, head_dim);
            let v_bph = expected_v_bph(VMode::Q8, head_dim);
            assert_eq!(layout.k_bytes_per_head, k_bph, "mode={mode:?}");
            assert_eq!(layout.v_bytes_per_head, v_bph, "mode={mode:?}");
            assert_eq!(
                layout.k_bytes_per_token,
                n_kv_heads * k_bph,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_bytes_per_token,
                n_kv_heads * v_bph,
                "mode={mode:?}"
            );
            // Static: reserve == current.
            assert_eq!(
                layout.k_reserve_bytes,
                physical_cap * layout.k_bytes_per_token,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_reserve_bytes,
                physical_cap * layout.v_bytes_per_token,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.k_reserve_elems,
                (layout.k_reserve_bytes + 3) / 4,
                "mode={mode:?}"
            );
            assert_eq!(
                layout.v_reserve_elems,
                (layout.v_reserve_bytes + 3) / 4,
                "mode={mode:?}"
            );
            assert_eq!(layout.kv_dim, n_kv_heads * head_dim);
            // Independent K vs V capacity at equal physical_cap is just physical_cap.
            let k_cap = layout.k_reserve_bytes / layout.k_bytes_per_token;
            let v_cap = layout.v_reserve_bytes / layout.v_bytes_per_token;
            assert_eq!(k_cap.min(v_cap), physical_cap, "mode={mode:?}");
        }
    }

    #[test]
    fn vmm_static_layout_covers_fwht_k_with_lloyd_v() {
        let n_kv_heads = 4;
        let head_dim = 256;
        let physical_cap = 64;
        let k_modes = [KvMode::Fwht2, KvMode::Fwht3, KvMode::Fwht4];
        let v_modes = [VMode::Lloyd2, VMode::Lloyd3, VMode::Lloyd4];
        for mode in k_modes {
            for v_mode in v_modes {
                let layout =
                    KvCache::vmm_static_layout(mode, v_mode, n_kv_heads, head_dim, physical_cap)
                        .unwrap_or_else(|e| panic!("{mode:?}/{v_mode:?}: {e}"));
                assert_eq!(layout.k_bytes_per_head, expected_k_bph(mode, head_dim));
                assert_eq!(layout.v_bytes_per_head, expected_v_bph(v_mode, head_dim));
                // Lloyd-V forces 256-wide signs even for fwht2/4.
                assert_eq!(layout.rotation_table_len, 256, "{mode:?}/{v_mode:?}");
                assert!(layout.uses_fwht_signs, "{mode:?}/{v_mode:?}");
                // Prefix helpers use current stride, not reserve.
                let n_pos = 17;
                assert_eq!(
                    layout.prefix_k_bytes(n_pos).unwrap(),
                    n_pos * layout.k_bytes_per_token
                );
                assert_eq!(
                    layout.prefix_v_bytes(n_pos).unwrap(),
                    n_pos * layout.v_bytes_per_token
                );
            }
        }
    }
}

/// Test-only one-shot fault hook for the single-device Q8 KV allocation path
/// (compiled out of production builds). When armed, the next
/// `KvCache::new_gpu_q8[_capped]` fails deterministically at the
/// mid-allocation seam in `new_gpu_q8_capped` — after the first layer's K/V
/// tensors are owned — and the arm is consumed, so at most one construction
/// attempt can observe it. Atomic, so full-suite isolation needs no global
/// lock beyond the tests' own `GPU_TEST_LOCK`.
#[cfg(test)]
mod fault_seam {
    use std::sync::atomic::{AtomicBool, Ordering};

    static FAIL_NEXT_Q8_ALLOC: AtomicBool = AtomicBool::new(false);

    /// Arm the next single-device Q8 KV construction to fail at the
    /// mid-allocation seam.
    pub fn arm_fail_next_q8_alloc() {
        FAIL_NEXT_Q8_ALLOC.store(true, Ordering::SeqCst);
    }

    /// One-shot consume: returns true exactly once per arm.
    pub fn consume() -> bool {
        FAIL_NEXT_Q8_ALLOC.swap(false, Ordering::SeqCst)
    }

    /// True while an arm is pending (tests assert the arm is consumed).
    pub fn is_armed() -> bool {
        FAIL_NEXT_Q8_ALLOC.load(Ordering::SeqCst)
    }

    /// Clear any pending arm (suite-isolation backstop).
    pub fn reset() {
        FAIL_NEXT_Q8_ALLOC.store(false, Ordering::SeqCst);
    }
}

/// Oracle phase-1 follow-up RED contract (single-device Q8 KV): the Q8 KV
/// constructor must be transactional under partial-allocation failure. The
/// armed seam fires after the first layer's K/V tensors are owned (a ~136 MiB
/// leak at the test geometry), so a correct constructor must free them before
/// returning the error. `new_gpu_q8_capped` stages every tensor and rolls
/// back on failure (same checked pattern as `alloc_k_v_filtered`), so the
/// mid-loop failure reclaims all owned allocations. Uses the real-gfx1151
/// VRAM observation convention from the tp_serve/pp_serve rollback tests
/// (warm-up cycle, pool drain, 64 MiB tolerance).
#[cfg(test)]
mod allocation_fault_tests {
    use super::*;
    use rdna_compute::Gpu;

    /// Serializes GPU-touching tests in this crate: device binding and VRAM
    /// state are process-global (same pattern as the per-module
    /// `GPU_TEST_LOCK`s elsewhere in the workspace).
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// VRAM recovery tolerance in bytes — matches the tp_serve/pp_serve
    /// rollback tests (64 MiB) so driver/allocator noise cannot flip the
    /// assertion.
    const VRAM_TOLERANCE: usize = 64 * 1024 * 1024;

    /// Free VRAM in bytes on `gpu`'s physical device.
    fn vram_free(gpu: &Gpu) -> usize {
        gpu.hip.get_vram_info().expect("hipMemGetInfo").0
    }

    #[test]
    fn failed_q8_kv_construction_reclaims_owned_allocations() {
        // Same gate as the other GPU tests: skip cleanly on GPU-less CI.
        if Gpu::init().is_err() {
            eprintln!("skip: no GPU");
            return;
        }
        // Poison-recovery acquisition: a panicking GPU test cannot cascade
        // lock panics into the rest of the suite.
        let _lock = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut gpu = Gpu::init().expect("GPU required for the Q8 KV rollback contract");

        // Geometry: head_dim=128 → 4 Q8 blocks/head × 34 B; 8 KV heads × 8
        // layers at 64K positions → ~68 MiB per K/V tensor, so even the
        // single armed mid-loop layer leak (~136 MiB) far exceeds the 64 MiB
        // tolerance.
        const N_LAYERS: usize = 8;
        const N_KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        const MAX_SEQ: usize = 65_536;

        // Warm-up cycle: one successful construction + free absorbs one-time
        // driver / allocator residency, so the measured baseline is stable.
        {
            let kv = KvCache::new_gpu_q8(&mut gpu, N_LAYERS, N_KV_HEADS, HEAD_DIM, MAX_SEQ)
                .expect("warm-up Q8 KV must succeed");
            kv.free_gpu(&mut gpu).expect("warm-up free must succeed");
        }
        gpu.drain_pool();
        let baseline = vram_free(&gpu);

        // Forced mid-construction failure at the allocation seam.
        fault_seam::reset();
        fault_seam::arm_fail_next_q8_alloc();
        let err = match KvCache::new_gpu_q8(&mut gpu, N_LAYERS, N_KV_HEADS, HEAD_DIM, MAX_SEQ) {
            Ok(_) => panic!("armed Q8 KV fault must fail the construction"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("fault seam"),
            "expected the mid-allocation seam to fail, got: {err}"
        );
        assert!(
            !fault_seam::is_armed(),
            "fault arm must be one-shot (consumed by the failed construction)"
        );

        // Every allocation owned at the seam must be reclaimed on the device.
        let after = vram_free(&gpu);
        assert!(
            baseline.abs_diff(after) < VRAM_TOLERANCE,
            "VRAM not reclaimed after failed Q8 KV construction: baseline={baseline} \
             after={after} delta={} — every mid-loop tensor already pushed must \
             be freed on its device",
            baseline.saturating_sub(after)
        );

        // Success cycle after the failure: the GPU must remain usable and an
        // unarmed construction + free must return VRAM to the same baseline.
        {
            let kv = KvCache::new_gpu_q8(&mut gpu, N_LAYERS, N_KV_HEADS, HEAD_DIM, MAX_SEQ)
                .expect("Q8 KV must succeed after the forced failure");
            kv.free_gpu(&mut gpu)
                .expect("free after the forced failure must succeed");
        }
        gpu.drain_pool();
        let after = vram_free(&gpu);
        assert!(
            baseline.abs_diff(after) < VRAM_TOLERANCE,
            "VRAM not recovered after the post-failure success cycle: \
             baseline={baseline} after={after}"
        );
    }

    /// The F32 `new_gpu_with_hook` constructor is transactional under a
    /// mid-loop adoption-hook failure: every K/V tensor allocated before the
    /// armed hook index is freed on the device, the original hook error stays
    /// primary, and the hook fires exactly once per adopted tensor (indices
    /// `2*l` / `2*l + 1` per layer).
    #[test]
    fn f32_kv_adoption_hook_failure_reclaims_owned_allocations() {
        // Same gate as the other GPU tests: skip cleanly on GPU-less CI.
        if Gpu::init().is_err() {
            eprintln!("skip: no GPU");
            return;
        }
        // Poison-recovery acquisition: a panicking GPU test cannot cascade
        // lock panics into the rest of the suite.
        let _lock = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut gpu = Gpu::init().expect("GPU required for the F32 KV hook rollback contract");

        const N_LAYERS: usize = 8;
        const N_KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        const MAX_SEQ: usize = 4096;
        // Hook indices: layer l K at 2*l, V at 2*l+1 → 2*N_LAYERS tensors.
        const HOOK_INDEX_COUNT: usize = 2 * N_LAYERS;

        // Warm-up cycle: one successful construction + free absorbs one-time
        // driver / allocator residency, so the measured baseline is stable.
        {
            let kv = KvCache::new_gpu(&mut gpu, N_LAYERS, N_KV_HEADS, HEAD_DIM, MAX_SEQ)
                .expect("warm-up F32 KV must succeed");
            kv.free_gpu(&mut gpu).expect("warm-up free must succeed");
        }
        gpu.drain_pool();
        let baseline = vram_free(&gpu);

        // Sweep every hook index: indices 0..count-1 must fail the
        // construction (each with every earlier tensor reclaimed), and the
        // index exactly at the count must succeed — the sentinel that proves
        // the hook fires once per tensor.
        let mut success = false;
        for fail_at in 0..=HOOK_INDEX_COUNT {
            let mut calls = 0usize;
            let result = KvCache::new_gpu_with_hook(
                &mut gpu,
                N_LAYERS,
                N_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ,
                |index| {
                    calls += 1;
                    assert_eq!(
                        calls - 1,
                        index,
                        "hook must be invoked once per adopted tensor with its \
                         zero-based index"
                    );
                    if index == fail_at {
                        Err(HipError::new(0, "forced F32 KV adoption-hook failure"))
                    } else {
                        Ok(())
                    }
                },
            );
            match result {
                Ok(kv) => {
                    kv.free_gpu(&mut gpu).expect("sentinel free must succeed");
                    gpu.drain_pool();
                    assert_eq!(
                        fail_at, HOOK_INDEX_COUNT,
                        "the success sentinel must be exactly one past the last \
                         F32 KV tensor"
                    );
                    let after = vram_free(&gpu);
                    assert!(
                        baseline.abs_diff(after) < VRAM_TOLERANCE,
                        "VRAM not recovered at the F32 KV success sentinel: \
                         baseline={baseline} after={after}"
                    );
                    success = true;
                    break;
                }
                Err(err) => {
                    assert_eq!(
                        calls,
                        fail_at + 1,
                        "the hook must fire through the armed index and no further"
                    );
                    assert!(
                        err.to_string()
                            .contains("forced F32 KV adoption-hook failure"),
                        "the original hook error must stay primary, got: {err}"
                    );
                    gpu.drain_pool();
                    let after = vram_free(&gpu);
                    assert!(
                        baseline.abs_diff(after) < VRAM_TOLERANCE,
                        "VRAM not reclaimed after F32 KV hook rollback at index \
                         {fail_at}: baseline={baseline} after={after} delta={}",
                        baseline.saturating_sub(after)
                    );
                }
            }
        }
        assert!(
            success,
            "F32 KV hook sweep did not reach the constructor-success sentinel"
        );
    }
}
