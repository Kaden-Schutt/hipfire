// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use hip_bridge::DeviceBuffer;
use rdna_compute::{Gpu, GpuTensor};

pub struct AttnParams<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    /// TODO(ship 3.1b): llama HFQ8/INT8 attend scales
    pub k_scales: Option<&'a GpuTensor>,
    /// TODO(ship 3.1b): llama HFQ8/INT8 attend scales
    pub v_scales: Option<&'a GpuTensor>,
    // ── Position (dual-type coexistence per D4/F8) ──
    /// Single-token position buffer. Used when `batch_size == 1`.
    /// Ignored when `batch_size > 1` (use `positions` instead).
    pub pos_buf: &'a DeviceBuffer,
    /// 0-based physical position index. `dispatch_*` internally computes
    /// `seq_len = pos + 1`. Callers MUST pass `pos`, never `pos + 1`.
    /// Used only for single-token (`batch_size == 1`).
    pub pos: usize,
    /// Batched position tensor `[n]` i32. Used when `batch_size > 1`.
    /// `None` when `batch_size == 1` (use `pos_buf` instead).
    pub positions: Option<&'a GpuTensor>,
    // ── Dimensions ──
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Maximum KV cache capacity (= max_seq for batched kernels).
    pub physical_cap: usize,
    /// Batch size. REQUIRED: `1` for decode/per-token, `>1` for batched prefill.
    pub batch_size: usize,
    /// Batched attend loop bound (= start_pos + n). `0` when `batch_size == 1`.
    pub max_ctx_len: usize,
    // ── Flash attention scratch ──
    pub flash_partials: Option<&'a GpuTensor>,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    // ── Tree-verify (spec-decode) ──
    /// `[n×n]` additive bias matrix. `Some` → tree-verify, `None` → causal.
    pub tree_bias: Option<&'a GpuTensor>,
    /// Tree window start (0 for plain causal).
    pub block_start: usize,
    /// Tree window cols (0 for plain causal).
    pub block_cols: usize,
    pub output: &'a GpuTensor,
}

impl<'a> AttnParams<'a> {
    /// Returns the batched positions tensor, asserting `batch_size > 1`.
    pub fn positions(&self) -> &'a GpuTensor {
        debug_assert!(
            self.batch_size > 1,
            "positions() called with batch_size == 1"
        );
        self.positions.expect("positions required for batch_size > 1")
    }
}

pub struct AttentionFamily {
    registry: KernelRegistry,
}

impl AttentionFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        super::super::tables::attention_table::populate(&mut registry);
        registry.validate().expect("attention kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn resolve(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        self.registry.resolve(key, ctx, shape)
    }

    /// Paired write-then-attend entry point (Phase 0.3). Takes a `KvTierPlan`
    /// carrying both the write key and attend key derived from the same
    /// `KvTierInputs`. Enforces the tier-match debug_assert before dispatch.
    /// Threads `ShapeInfo` derived from `plan.batch_size` into `resolve()`
    /// so that `BatchGt(1)`/`BatchEq(1)` shape gates actually fire.
    pub fn run_attention(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        plan: &crate::families::kv_tier::KvTierPlan,
        io: &AttnParams,
    ) -> Result<(), DispatchError> {
        let shape = ShapeInfo {
            batch_size: plan.batch_size,
            head_dim: io.head_dim,
            m: 0,
            is_tree: io.tree_bias.is_some(),
        };
        let write_var = self.resolve(plan.write_key, ctx, Some(&shape))?;
        dispatch_kv_write(gpu, plan.write_key, plan, io)?;
        let attend_var = self.resolve(plan.attend_key, ctx, Some(&shape))?;
        dispatch_attend(gpu, plan.attend_key, attend_var.tile, plan, io)
    }
}

impl KernelFamily for AttentionFamily {
    fn name(&self) -> &'static str {
        "attention"
    }
}

macro_rules! hip {
    ($e:expr) => {
        $e.map_err(|e| DispatchError::Hip(e.to_string()))
    };
}

// ── KV Cache Write dispatch ────────────────────────────

fn dispatch_kv_write(
    gpu: &mut Gpu,
    key: KernelKey,
    plan: &crate::families::kv_tier::KvTierPlan,
    io: &AttnParams,
) -> Result<(), DispatchError> {
    match key {
        // ── Single-token (decode / per-token fallback) ──
        KernelKey::KvWriteF32 => {
            debug_assert_eq!(plan.batch_size, 1);
            let kv_dim = io.n_kv_heads * io.head_dim;
            hip!(gpu.kv_cache_write(io.k_cache, io.k, io.pos_buf, kv_dim))?;
            hip!(gpu.kv_cache_write(io.v_cache, io.v, io.pos_buf, kv_dim))
        }
        KernelKey::KvWriteQ8_0 => {
            debug_assert_eq!(plan.batch_size, 1);
            hip!(gpu.kv_cache_write_q8_0(io.k_cache, io.k, io.pos_buf, io.n_kv_heads, io.head_dim))?;
            hip!(gpu.kv_cache_write_q8_0(io.v_cache, io.v, io.pos_buf, io.n_kv_heads, io.head_dim))
        }
        KernelKey::KvWriteAsym4 => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym4_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim,
            ))
        }
        KernelKey::KvWriteAsym4Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht4_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim, plan.v_mode_bits,
            ))
        }
        KernelKey::KvWriteAsym3 => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym3_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim,
            ))
        }
        KernelKey::KvWriteAsym3Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht3_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim, plan.v_mode_bits,
            ))
        }
        KernelKey::KvWriteAsym2 => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym2_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim,
            ))
        }
        KernelKey::KvWriteAsym2Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht2_fused(
                io.k_cache, io.v_cache, io.k, io.v, io.pos_buf,
                ct, st, io.n_kv_heads, io.head_dim, plan.v_mode_bits,
            ))
        }

        // ── Batched (prefill / tree-verify) ──
        KernelKey::KvWriteAsym4Batched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym4_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size,
            ))
        }
        KernelKey::KvWriteAsym4FwhtBatched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht4_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size, plan.v_mode_bits,
            ))
        }
        KernelKey::KvWriteAsym3Batched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym3_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size,
            ))
        }
        KernelKey::KvWriteAsym3FwhtBatched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht3_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size, plan.v_mode_bits,
            ))
        }
        KernelKey::KvWriteAsym2Batched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_asym2_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size,
            ))
        }
        KernelKey::KvWriteAsym2FwhtBatched => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            hip!(gpu.kv_cache_write_fwht2_batched(
                io.k_cache, io.v_cache, io.k, io.v, io.positions(),
                ct, st, io.n_kv_heads, io.head_dim, io.batch_size, plan.v_mode_bits,
            ))
        }
        KernelKey::KvWriteQ8_0Batched => {
            // Q8 batched write is called twice (K, then V) — not fused.
            let pos = io.positions();
            hip!(gpu.kv_cache_write_q8_0_batched(
                io.k_cache, io.k, pos, io.n_kv_heads, io.head_dim, io.batch_size,
            ))?;
            hip!(gpu.kv_cache_write_q8_0_batched(
                io.v_cache, io.v, pos, io.n_kv_heads, io.head_dim, io.batch_size,
            ))
        }

        _ => Err(DispatchError::UnsupportedVariant {
            family: "attention/kv_write",
            variant: "unhandled key — missing dispatch arm",
            arch: "",
            quant: "",
        }),
    }
}

// ── Attention dispatch ─────────────────────────────────

fn dispatch_attend(
    gpu: &mut Gpu,
    key: KernelKey,
    tile: TileImpl,
    plan: &crate::families::kv_tier::KvTierPlan,
    io: &AttnParams,
) -> Result<(), DispatchError> {
    // Tile-first dispatch: tile variants get their own arms, key-only dispatch
    // lives under TileImpl::None.
    match tile {
        TileImpl::Asym4WmmaTile => {
            debug_assert_eq!(key, KernelKey::AttnFlashAsym4BatchedMasked);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym4_wmma_tile_batched(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols,
            ))
        }
        TileImpl::Asym4WmmaTileGfx12 => {
            debug_assert_eq!(key, KernelKey::AttnFlashAsym4BatchedMasked);
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym4_wmma_tile_batched_gfx12(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols,
            ))
        }
        TileImpl::None => match key {
        // ── Single-token (decode / per-token fallback) ──
        KernelKey::AttnF32 => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            hip!(gpu.attention_f32(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap,
            ))
        }
        KernelKey::AttnFlashQ8_0 => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_q8_0(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
            ))
        }
        KernelKey::AttnQ8_0Kv => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            hip!(gpu.attention_q8_0_kv(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap,
            ))
        }
        KernelKey::AttnFlashAsym4 => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym4(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym4Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht4(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
                plan.v_mode_bits,
            ))
        }
        KernelKey::AttnFlashAsym3 => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym3(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym3Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht3(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
                plan.v_mode_bits,
            ))
        }
        KernelKey::AttnFlashAsym2 => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym2(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
            ))
        }
        KernelKey::AttnFlashAsym2Fwht => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht2(
                io.q, io.k_cache, io.v_cache, io.output, io.pos_buf,
                ct, st, seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap, fp,
                plan.v_mode_bits,
            ))
        }
        KernelKey::AttnGqaFused => {
            debug_assert_eq!(plan.batch_size, 1);
            let seq_len = io.pos + 1;
            hip!(gpu.attention_flash_gqa_fused(
                io.q, io.k_cache, io.v_cache, io.output,
                seq_len, io.n_heads, io.n_kv_heads, io.head_dim, io.physical_cap,
            ))
        }

        // ── Batched (prefill / tree-verify) ──
        KernelKey::AttnFlashAsym4BatchedMasked => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym4_batched_masked(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols,
            ))
        }
        KernelKey::AttnFlashAsym4FwhtBatchedMasked => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht4_batched_masked(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols, plan.v_mode_bits,
            ))
        }
        KernelKey::AttnFlashAsym3BatchedMasked => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym3_batched_masked(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols,
            ))
        }
        KernelKey::AttnFlashAsym3FwhtBatchedMasked => {
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht3_batched_masked(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols, plan.v_mode_bits,
            ))
        }
        // 2-bit: _batched only (no _masked — tree-verify gap)
        KernelKey::AttnFlashAsym2Batched => {
            debug_assert!(io.tree_bias.is_none(), "asym2 has no _batched_masked variant");
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_asym2_batched(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
            ))
        }
        KernelKey::AttnFlashAsym2FwhtBatched => {
            debug_assert!(io.tree_bias.is_none(), "asym2 fwht has no _batched_masked variant");
            let ct = io.givens_cos.unwrap();
            let st = io.givens_sin.unwrap();
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_fwht2_batched(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                ct, st, io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                plan.v_mode_bits,
            ))
        }
        // Q8_0: P-1 no-LDS-cap tiled kernel
        KernelKey::AttnQ8_0KvBatchedMasked => {
            let fp = io.flash_partials.unwrap();
            hip!(gpu.attention_flash_q8_0_batched_masked(
                io.q, io.k_cache, io.v_cache, io.output, io.positions(),
                io.n_heads, io.n_kv_heads, io.head_dim,
                io.physical_cap, io.max_ctx_len, io.batch_size, fp,
                io.tree_bias, io.block_start, io.block_cols,
            ))
        }

        _ => Err(DispatchError::UnsupportedVariant {
            family: "attention/attend",
            variant: "unhandled key — missing dispatch arm",
            arch: "",
            quant: "",
        }),
        }  // close match key

        // Unhandled tile variants (should not reach here without an arm)
        _ => Err(DispatchError::UnsupportedVariant {
            family: "attention/attend",
            variant: "unhandled tile variant",
            arch: "",
            quant: "",
        }),
    }  // close match tile
}

// ── Dispatch key constants for completeness tests ──────

/// All `KernelKey` variants handled by `dispatch_kv_write`.
/// If you add a new KV write key and forget to add a dispatch arm, the
/// completeness test will fail.
pub(crate) const DISPATCHED_KV_WRITE_KEYS: &[KernelKey] = &[
    // Single-token
    KernelKey::KvWriteF32,
    KernelKey::KvWriteQ8_0,
    KernelKey::KvWriteAsym4,
    KernelKey::KvWriteAsym4Fwht,
    KernelKey::KvWriteAsym3,
    KernelKey::KvWriteAsym3Fwht,
    KernelKey::KvWriteAsym2,
    KernelKey::KvWriteAsym2Fwht,
    // Batched
    KernelKey::KvWriteAsym4Batched,
    KernelKey::KvWriteAsym4FwhtBatched,
    KernelKey::KvWriteAsym3Batched,
    KernelKey::KvWriteAsym3FwhtBatched,
    KernelKey::KvWriteAsym2Batched,
    KernelKey::KvWriteAsym2FwhtBatched,
    KernelKey::KvWriteQ8_0Batched,
];

/// All `KernelKey` variants handled by `dispatch_attend`.
pub(crate) const DISPATCHED_ATTEND_KEYS: &[KernelKey] = &[
    // Single-token
    KernelKey::AttnF32,
    KernelKey::AttnFlashQ8_0,
    KernelKey::AttnQ8_0Kv,
    KernelKey::AttnFlashAsym4,
    KernelKey::AttnFlashAsym4Fwht,
    KernelKey::AttnFlashAsym3,
    KernelKey::AttnFlashAsym3Fwht,
    KernelKey::AttnFlashAsym2,
    KernelKey::AttnFlashAsym2Fwht,
    KernelKey::AttnGqaFused,
    // Batched
    KernelKey::AttnFlashAsym4BatchedMasked,
    KernelKey::AttnFlashAsym4FwhtBatchedMasked,
    KernelKey::AttnFlashAsym3BatchedMasked,
    KernelKey::AttnFlashAsym3FwhtBatchedMasked,
    KernelKey::AttnFlashAsym2Batched,
    KernelKey::AttnFlashAsym2FwhtBatched,
    KernelKey::AttnQ8_0KvBatchedMasked,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Bidirectional completeness check for `dispatch_kv_write`.
    /// Every registered KV write key must have an arm, and every arm key
    /// must be registered.
    #[test]
    fn dispatch_kv_write_has_arms_for_all_registered_keys() {
        let family = AttentionFamily::new();
        let ctx = DispatchCtx::for_test("gfx1100");

        let dispatched_set: std::collections::HashSet<KernelKey> =
            DISPATCHED_KV_WRITE_KEYS.iter().copied().collect();

        for &key in DISPATCHED_KV_WRITE_KEYS {
            let batch = if is_batched_kv_key(key) { 16 } else { 1 };
            let shape = ShapeInfo { batch_size: batch, head_dim: 128, m: 0, is_tree: false };
            assert!(
                family.resolve(key, &ctx, Some(&shape)).is_ok(),
                "DISPATCHED_KV_WRITE_KEYS contains {:?} but it is NOT registered — stale entry",
                key
            );
        }

        // Ensure no registered KV write key is missing from the dispatched set.
        for &key in DISPATCHED_KV_WRITE_KEYS {
            let batch = if is_batched_kv_key(key) { 16 } else { 1 };
            let shape = ShapeInfo { batch_size: batch, head_dim: 128, m: 0, is_tree: false };
            if family.resolve(key, &ctx, Some(&shape)).is_ok() {
                assert!(
                    dispatched_set.contains(&key),
                    "registered key {:?} is not in DISPATCHED_KV_WRITE_KEYS — missing arm",
                    key
                );
            }
        }
    }

    /// Helper: is this key a batched KV write key?
    fn is_batched_kv_key(key: KernelKey) -> bool {
        use KernelKey::*;
        matches!(
            key,
            KvWriteAsym4Batched
            | KvWriteAsym4FwhtBatched
            | KvWriteAsym3Batched
            | KvWriteAsym3FwhtBatched
            | KvWriteAsym2Batched
            | KvWriteAsym2FwhtBatched
            | KvWriteQ8_0Batched
        )
    }

    /// Bidirectional completeness check for `dispatch_attend`.
    #[test]
    fn dispatch_attend_has_arms_for_all_registered_keys() {
        let family = AttentionFamily::new();
        let ctx = DispatchCtx::for_test("gfx1100");

        let dispatched_set: std::collections::HashSet<KernelKey> =
            DISPATCHED_ATTEND_KEYS.iter().copied().collect();

        for &key in DISPATCHED_ATTEND_KEYS {
            // Single-token keys resolve at batch_size=1, batched at batch_size>1
            let batch = if is_batched_key(key) { 16 } else { 1 };
            let shape = ShapeInfo { batch_size: batch, head_dim: 128, m: 0, is_tree: false };
            assert!(
                family.resolve(key, &ctx, Some(&shape)).is_ok(),
                "DISPATCHED_ATTEND_KEYS contains {:?} but it is NOT registered — stale entry",
                key
            );
        }

        for &key in DISPATCHED_ATTEND_KEYS {
            let batch = if is_batched_key(key) { 16 } else { 1 };
            let shape = ShapeInfo { batch_size: batch, head_dim: 128, m: 0, is_tree: false };
            if family.resolve(key, &ctx, Some(&shape)).is_ok() {
                assert!(
                    dispatched_set.contains(&key),
                    "registered key {:?} is not in DISPATCHED_ATTEND_KEYS — missing arm",
                    key
                );
            }
        }
    }

    /// Helper: is this key a batched key (needs batch_size > 1 to resolve)?
    fn is_batched_key(key: KernelKey) -> bool {
        use KernelKey::*;
        matches!(
            key,
            AttnFlashAsym4BatchedMasked
            | AttnFlashAsym4FwhtBatchedMasked
            | AttnFlashAsym3BatchedMasked
            | AttnFlashAsym3FwhtBatchedMasked
            | AttnFlashAsym2Batched
            | AttnFlashAsym2FwhtBatched
            | AttnQ8_0KvBatchedMasked
        )
    }
}
