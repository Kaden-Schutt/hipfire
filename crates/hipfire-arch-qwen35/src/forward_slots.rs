// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// forward_batch_slots — the N-slot forward pass (SP3 Task 2).
//
// A PARALLEL entry point to `qwen35::forward_prefill_batch_with_pbs_opts`,
// deliberately NOT a modification of it. That function carries hipGraph
// capture eligibility, tree-verify, GDN-tape and MTP interactions whose
// interaction with a slot axis is unknown, and every current caller (chat,
// spec decode, MTP, the TUI) depends on it being unchanged. This file
// mirrors its per-layer kernel sequence (see `sp3-task-2-report.md` for the
// enumeration) but routes attention and KV-write through the slot-aware
// `_slots` entry points SP1 built, and DeltaNet through a per-slot loop
// over SP2's per-slot `DeltaNetState`.
//
// Scope: dense layers stay Q8_0 only. Every weight a `DeltaNet`/`FullAttn`
// (dense) layer touches (QKV/QKVZA, wo, gate/up, down, and the lm_head) must
// be `DType::Q8_0`, the KV cache is Q8_0-quantized, and DeltaNet state is
// `StateQuant::Q8`. This matches the ABI the multi-slot infrastructure was
// actually built against — `SlotPool`'s per-slot addressing is documented as
// a Q8_0 ABI (asym3 is explicitly exempted because its K/V strides differ and
// it cannot share `k_base`/`v_base`), `kv_cache_write_q8_0_batched_slots` and
// both `attention_*_batched_masked_slots` entry points are Q8_0-named, and
// `gated_delta_net_q8_batch_seq` is the Q8 recurrence. The KV cache tier is
// unconditionally Q8_0 for EVERY layer this file drives, dense or MoE — slot
// addressing is a KV-cache-tier property, not a weight-quant property, so the
// attend/KV-write steps below never change no matter what a layer's
// projection weights are quantized to.
//
// `DeltaNetMoe`/`FullAttnMoe` layers (qwen3.6-35b-a3b and similar A3B
// checkpoints) additionally admit uniformly-`MQ4G256` projection weights,
// mirroring `forward_prefill_chunk`'s own `is_mq` dispatch fork for these
// layer kinds (rmsnorm+FWHT-rotate via `fused_rmsnorm_rotate_mq_batched_for`
// / `rotate_x_mq_batched_for`, then the same `*Hfq4G256` kernel keys the
// dense HFQ4G256 path already uses — MQ4G256 is byte-identical to HFQ4G256,
// only the input activations are pre-rotated). The MoE FFN itself is
// stateless per row (no kv_cache, no dn_state, no positions — confirmed by
// reading `moe_ffn_decode`'s signature), so it needs no slot machinery at
// all: `run_deltanet_moe_layer_slots`/`run_fullattn_moe_layer_slots` call
// the reference's own `prefill_moe_ffn_body_batched` directly over the flat
// N-row batch, gated by the reference's own `moe_ffn_batched_admissible`.
// MQ6/PARO/Lloyd/E8/mixed-dtype MoE attention or FFN weights are out of
// scope here (see `require_batchable_deltanet_moe_layer` /
// `require_batchable_fullattn_moe_layer` / `require_batchable_moe_ffn`) —
// each returns a clear `HipError` rather than guessing at an untested path.
//
// DeltaNet slot-state note: only the GDN recurrence and the conv1d causal
// state are sequential-per-slot (both carry state across steps: the S
// matrix and the conv1d ring buffer respectively). The brief's Step 3 only
// spells out the GDN call, but the SAME hazard applies to conv1d — its
// ring buffer seeds the causal window for a slot's FIRST 1-2 rows from the
// PREVIOUS call, so batching rows from different slots into one launch
// would let a slot's first row convolve over a neighbouring slot's last
// rows instead of its own history. Both are looped per slot; every other
// per-layer step (rmsnorm, the QKVZA/QKV projections, sigmoid/alpha-gate,
// the Q/K L2-norm, gated-norm, wo+residual, the FFN) is stateless per row
// and runs once across all N rows regardless of slot boundaries, exactly
// like the FullAttention layer body (whose only slot-aware steps are the
// KV write and the attend call, both single launches via the `_slots`
// entry points — RoPE is slot-agnostic per SP2 Task 2 and needs no split).

use crate::qwen35::{
    mq6_batched_admit_enabled_from_env, moe_ffn_batched_admissible, prefill_moe_ffn_body_batched,
    q8_prefill_wmma_enabled, run_fused_gate_up_key, run_fused_qkv_key, run_fused_qkvza_key,
    run_plain_gemm_key, run_residual_gemm_key, DeltaNetLayerWeights, DeltaNetMoeLayerWeights,
    DeltaNetState, FullAttnLayerWeights, FullAttnMoeLayerWeights, LayerType, LayerWeights,
    MoeFfnWeights, PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::slot_batch::SlotBatch;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_mq_batched_for, rotate_x_mq_batched_for, EmbeddingFormat, WeightTensor,
};
use rdna_compute::kv_slots::{build_tiles, KvSlotDesc};
use rdna_compute::slot_pool::SlotPool;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Pack a `KvSlotDesc` table byte-identically to `kernels/src/kv_slot_desc.h`
/// (`k_base: u64, v_base: u64, seq_len: i32, cap: i32`, 24 bytes, no padding
/// between fields on this target). Mirrors the packer SP1's Task 7 harness
/// uses (`rdna-compute/examples/test_batched_attn_slots.rs::pack_descs`) —
/// duplicated rather than shared because that packer lives in `examples/`
/// (test-only) and this is production `src/`.
fn pack_descs(descs: &[KvSlotDesc]) -> Vec<u8> {
    let mut out = Vec::with_capacity(descs.len() * 24);
    for d in descs {
        out.extend_from_slice(&d.k_base.to_ne_bytes());
        out.extend_from_slice(&d.v_base.to_ne_bytes());
        out.extend_from_slice(&d.seq_len.to_ne_bytes());
        out.extend_from_slice(&d.cap.to_ne_bytes());
    }
    out
}

/// Persistent device staging for the two slot-addressing tables every
/// `_slots` kernel call needs: the `KvSlotDesc` table and the per-row
/// `row_slot` map. Owned by the caller across steps (like
/// `PrefillBatchScratch`) so re-uploading is a `memcpy_htod` into a fixed
/// allocation, not an alloc/free pair every step.
///
/// `descs_dev` is re-uploaded only when `pool.descriptors_dirty()` — most
/// steps touch at least one slot's `seq_len` so this rarely skips work, but
/// it's cheap to check and is what the brief asks for. `row_slot_dev` is
/// re-uploaded every step unconditionally: batch composition (which rows
/// belong to which slot) legitimately changes step to step, and `SlotBatch`
/// carries no dirty-tracking of its own.
pub struct SlotDescStaging {
    pub descs_dev: GpuTensor,
    pub row_slot_dev: GpuTensor,
    /// Flat row-tile lists for the WMMA flash-prefill kernel
    /// (`attention_q8_0_flash_prefill_wmma_slots`) — the only `_slots` kernel
    /// with `BR > 1` (fixed `M_TILE = 16`) that this file drives, so it is
    /// the only one that needs tile arrays rather than a plain `row_slot`
    /// map. Capacity is `max_rows` i32 entries: a tile always owns >= 1 row,
    /// so the tile count can never exceed the row count. Rebuilt and
    /// re-uploaded every step (like `row_slot_dev`) whenever more than one
    /// slot is active this step — batch composition legitimately changes
    /// step to step and `SlotBatch` carries no dirty-tracking of its own.
    /// Only their PREFIX (`0..n_tiles_this_step`) is meaningful; callers
    /// must track `n_tiles` themselves (see `forward_batch_slots_with_max_layer`).
    pub tile_slot_dev: GpuTensor,
    pub tile_row0_dev: GpuTensor,
    pub tile_qbase_dev: GpuTensor,
    n_slots: usize,
    max_rows: usize,
}

impl SlotDescStaging {
    pub fn new(gpu: &mut Gpu, n_slots: usize, max_rows: usize) -> HipResult<Self> {
        // Allocate all five staging buffers, freeing whatever already
        // succeeded if a later allocation fails partway through.
        let mut allocated: Vec<GpuTensor> = Vec::new();
        let shapes: [&[usize]; 5] = [
            &[n_slots * 24], // descs_dev
            &[max_rows * 4], // row_slot_dev
            &[max_rows * 4], // tile_slot_dev
            &[max_rows * 4], // tile_row0_dev
            &[max_rows * 4], // tile_qbase_dev
        ];
        for shape in shapes {
            match gpu.alloc_tensor(shape, DType::Raw) {
                Ok(t) => allocated.push(t),
                Err(e) => {
                    for t in allocated.drain(..) {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(e);
                }
            }
        }
        let mut it = allocated.into_iter();
        Ok(Self {
            descs_dev: it.next().unwrap(),
            row_slot_dev: it.next().unwrap(),
            tile_slot_dev: it.next().unwrap(),
            tile_row0_dev: it.next().unwrap(),
            tile_qbase_dev: it.next().unwrap(),
            n_slots,
            max_rows,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.descs_dev);
        let _ = gpu.free_tensor(self.row_slot_dev);
        let _ = gpu.free_tensor(self.tile_slot_dev);
        let _ = gpu.free_tensor(self.tile_row0_dev);
        let _ = gpu.free_tensor(self.tile_qbase_dev);
    }
}

/// Q8-only weight-dtype gate for a `DeltaNetLayerWeights`. All eight
/// projections must be `Q8_0` — a mixed-dtype layer would silently
/// misroute through a Q8-stride kernel against a differently-strided
/// weight, the exact corruption class the dense batched-prefill path
/// guards against at every one of its own dtype branches.
fn require_q8_deltanet_layer(layer: &DeltaNetLayerWeights) -> HipResult<()> {
    let ok = matches!(layer.wqkv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wz.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_gate.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_up.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_down.gpu_dtype, DType::Q8_0);
    if ok {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            "forward_batch_slots: DeltaNet layer has a non-Q8_0 weight; the \
             multi-slot batched path is Q8_0-only (see sp3-task-2-report.md)",
        ))
    }
}

fn require_q8_fullattn_layer(layer: &FullAttnLayerWeights) -> HipResult<()> {
    let ok = matches!(layer.wq.gpu_dtype, DType::Q8_0)
        && matches!(layer.wk.gpu_dtype, DType::Q8_0)
        && matches!(layer.wv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_gate.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_up.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_down.gpu_dtype, DType::Q8_0);
    if ok {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            "forward_batch_slots: FullAttention layer has a non-Q8_0 weight; \
             the multi-slot batched path is Q8_0-only (see sp3-task-2-report.md)",
        ))
    }
}

/// Which batched projection dispatch a MoE attention body should take.
/// Mirrors the `is_q8` / `is_mq` dtype forks `forward_prefill_chunk` applies
/// to `DeltaNetMoe`/`FullAttnMoe` layers (qwen35.rs's DeltaNetMoe LA branch
/// and FullAttnMoe FA branch) — narrowed to the two dtypes this file
/// actually implements a slot-aware body for. MQ6G256/HFQ6G256, ParoQ4G128,
/// and mixed-dtype-within-layer are real paths in the reference but are NOT
/// ported here; see `require_batchable_deltanet_moe_layer` /
/// `require_batchable_fullattn_moe_layer`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MoeAttnDtype {
    Q8_0,
    Mq4G256,
}

/// Q8_0-or-MQ4G256 weight-dtype gate for a `DeltaNetMoeLayerWeights`,
/// uniform across all five attention projections (wqkv/wz/w_beta/w_alpha/wo)
/// — a mixed Q8/MQ4 layer would misroute through a single-stride fused
/// kernel against differently-strided weights, the same corruption class
/// `require_q8_deltanet_layer` guards against for the dense path. MQ4G256
/// requires the caller to additionally rotate activations via
/// `fused_rmsnorm_rotate_mq_batched_for`/`rotate_x_mq_batched_for` before
/// each GEMM — see `run_deltanet_moe_layer_slots`.
fn require_batchable_deltanet_moe_layer(layer: &DeltaNetMoeLayerWeights) -> HipResult<MoeAttnDtype> {
    let all_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wz.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0);
    if all_q8 {
        return Ok(MoeAttnDtype::Q8_0);
    }
    let all_mq4 = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wz.gpu_dtype, DType::MQ4G256)
        && matches!(layer.w_beta.gpu_dtype, DType::MQ4G256)
        && matches!(layer.w_alpha.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wo.gpu_dtype, DType::MQ4G256);
    if all_mq4 {
        return Ok(MoeAttnDtype::Mq4G256);
    }
    Err(HipError::new(
        0,
        "forward_batch_slots: DeltaNetMoe layer attention weights must be \
         uniformly Q8_0 or uniformly MQ4G256 (mirrors forward_prefill_chunk's \
         is_q8/is_mq dispatch); MQ6G256/HFQ6G256, ParoQ4G128, and mixed-dtype \
         MoE attention are out of scope for the multi-slot batched path",
    ))
}

/// Same gate as [`require_batchable_deltanet_moe_layer`] for a
/// `FullAttnMoeLayerWeights` (wq/wk/wv/wo).
fn require_batchable_fullattn_moe_layer(layer: &FullAttnMoeLayerWeights) -> HipResult<MoeAttnDtype> {
    let all_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0)
        && matches!(layer.wk.gpu_dtype, DType::Q8_0)
        && matches!(layer.wv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0);
    if all_q8 {
        return Ok(MoeAttnDtype::Q8_0);
    }
    let all_mq4 = matches!(layer.wq.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wk.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wv.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wo.gpu_dtype, DType::MQ4G256);
    if all_mq4 {
        return Ok(MoeAttnDtype::Mq4G256);
    }
    Err(HipError::new(
        0,
        "forward_batch_slots: FullAttnMoe layer attention weights must be \
         uniformly Q8_0 or uniformly MQ4G256 (mirrors forward_prefill_chunk's \
         qkv_is_q8/qkv_is_mq dispatch); MQ6G256/HFQ6G256, ParoQ4G128, and \
         mixed-dtype MoE attention are out of scope for the multi-slot \
         batched path",
    ))
}

/// MoE-FFN weight-dtype admissibility gate, delegating to the reference's own
/// `moe_ffn_batched_admissible` (same predicate `prefill_batch_pbs_eligible`
/// checks before ever entering `forward_prefill_chunk`'s MoE branches) so
/// this file's notion of "batchable MoE FFN" can never drift from the
/// function it is about to call (`prefill_moe_ffn_body_batched`). `admit_mq6`
/// is computed identically to the reference's own call site
/// (`prefill_batch_pbs_eligible`, qwen35.rs) via the same env-keyed helper.
fn require_batchable_moe_ffn(gpu: &Gpu, ffn: &MoeFfnWeights) -> HipResult<()> {
    let arch = gpu.arch.as_str();
    let admit_mq6 = mq6_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
            .ok()
            .as_deref(),
        arch,
    );
    if moe_ffn_batched_admissible(ffn, admit_mq6, arch) {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            "forward_batch_slots: MoE FFN weight dtypes are not admissible for \
             the batched prefill path (see moe_ffn_batched_admissible); this \
             file requires the same admission the reference's own \
             prefill_batch_pbs_eligible checks before entering the batched MoE \
             branches",
        ))
    }
}

/// FWHT-rotated MQ4G256 residual projection: `y[0..n*m] += w · FWHT(x[0..n*k])`.
/// Mirrors the reference's default (non-Q8/non-6bit/non-PARO) wo / w_down
/// dispatch fork for `DeltaNetMoe`/`FullAttnMoe` layers — `GemmHfq4G256Residual`
/// against a `rotate_x_mq_batched_for`-rotated input. `scratch` is the
/// caller's dead buffer to rotate into (mirrors `pbs.dn_normed_rot_batch` /
/// `pbs.fa_attn_out_rot_batch` reuse in the dense Q8 path's `q8_residual_proj`).
fn mq4_residual_proj(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    rotate_x_mq_batched_for(gpu, w, x, scratch, w.k, n)?;
    let y_n = y.sub_offset(0, n * w.m);
    run_residual_gemm_key(
        gpu,
        KernelKey::GemmHfq4G256Residual,
        &w.buf,
        w.gpu_dtype,
        scratch,
        &y_n,
        w.m,
        w.k,
        n,
    )
}

/// `y[0..n*m] += w · x[0..n*k]`, dispatched through the same
/// `GemmQ8_0ResidualWmma` / `GemmQ8_0BatchedChunked`+`add_inplace_f32`
/// fork the dense batched-prefill path uses for every Q8 residual
/// projection (wo, w_down, and — via a plain non-residual variant below —
/// the lm_head). `scratch` is the non-WMMA fallback's landing buffer for
/// `w · x` before the explicit add; callers pass a buffer that is dead at
/// this point in the layer (mirrors the reference's reuse of
/// `x_rot_batch`/`x_batch` for exactly this purpose).
#[allow(clippy::too_many_arguments)]
fn q8_residual_proj(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    if q8_wmma_arch {
        let y_n = y.sub_offset(0, n * w.m);
        run_residual_gemm_key(
            gpu,
            KernelKey::GemmQ8_0ResidualWmma,
            &w.buf,
            w.gpu_dtype,
            x,
            &y_n,
            w.m,
            w.k,
            n,
        )
    } else {
        let s = scratch.sub_offset(0, n * w.m);
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w.buf,
            w.gpu_dtype,
            x,
            &s,
            w.m,
            w.k,
            n,
        )?;
        let y_n = y.sub_offset(0, n * w.m);
        gpu.add_inplace_f32(&y_n, &s)
    }
}

/// Q8 gate+up FFN projection, WMMA-fused when available else two plain
/// batched GEMMs. Mirrors the `ffn_is_q8` fork of the dense path.
#[allow(clippy::too_many_arguments)]
fn q8_gate_up_proj(
    gpu: &mut Gpu,
    w_gate: &hipfire_runtime::llama::WeightTensor,
    w_up: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    if q8_wmma_arch {
        run_fused_gate_up_key(
            gpu,
            KernelKey::FusedGateUpQ8_0,
            &w_gate.buf,
            &w_up.buf,
            x,
            y_gate,
            y_up,
            w_gate.m,
            w_up.m,
            w_gate.k,
            n,
        )
    } else {
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w_gate.buf,
            w_gate.gpu_dtype,
            x,
            y_gate,
            w_gate.m,
            w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w_up.buf,
            w_up.gpu_dtype,
            x,
            y_up,
            w_up.m,
            w_up.k,
            n,
        )
    }
}

/// Run one `LinearAttention` (DeltaNet) layer across the whole step.
///
/// The stateless pieces (rmsnorm, the 4-way QKVZA projection,
/// sigmoid/alpha-gate, the Q/K L2-norm, gated-norm, wo+residual, FFN) run
/// once over all `n` rows. The stateful pieces (conv1d, the GDN recurrence)
/// loop per slot with that slot's own state — see the module doc.
#[allow(clippy::too_many_arguments)]
fn run_deltanet_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &DeltaNetLayerWeights,
    batch: &SlotBatch,
    dn_states: &mut [DeltaNetState],
    pbs: &PrefillBatchScratch,
    q8_wmma_arch: bool,
    n: usize,
    delta_layer_idx: usize,
) -> HipResult<()> {
    require_q8_deltanet_layer(layer)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // 1. rmsnorm (plain — Q8 needs no FWHT rotate).
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &layer.attn_norm,
        &pbs.x_rot_batch,
        n,
        config.dim,
        config.norm_eps,
    )?;

    // 2. Batched 4-way QKVZA projection.
    if q8_wmma_arch {
        run_fused_qkvza_key(
            gpu,
            KernelKey::FusedQkvzaQ8_0,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else {
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.wqkv.buf,
            layer.wqkv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.wz.buf,
            layer.wz.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.w_beta.buf,
            layer.w_beta.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.w_alpha.buf,
            layer.w_alpha.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
    }

    // 3. Fused sigmoid(beta) + alpha_gate(alpha) — stateless, batched over N.
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;

    // 4. conv1d — STATEFUL (ring buffer), looped per slot.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            let q_out = pbs.dn_q_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let k_out = pbs.dn_k_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let v_out = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let input = pbs.dn_qkv_batch.sub_offset(row_off * qkv_dim, m * qkv_dim);
            gpu.conv1d_silu_split_f32_n(
                &q_out,
                &k_out,
                &v_out,
                &input,
                &layer.conv_weight,
                &dn_states[s].conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                m,
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 5. Q/K L2-norm(+scale, +repeat-interleave) — stateless, batched over N.
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
    } else {
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_num_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }

    // 6. Gated Delta Net recurrence — STATEFUL (S matrix), looped per slot.
    // Existing, unmodified `gated_delta_net_q8_batch_seq`; each launch
    // advances exactly one slot's own `DeltaNetState`. Do NOT use
    // `gated_delta_net_q8_batch_seq_slots` — it now asserts
    // `s_stride_elems == 0` (the stride design was found unsound; see
    // 8fb0e38e). No stride is needed here: one launch per slot already
    // addresses that slot's state directly.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if !matches!(dn_states[s].quant, StateQuant::Q8) {
                return Err(HipError::new(
                    0,
                    "forward_batch_slots: DeltaNetState must be StateQuant::Q8 \
                     (the multi-slot batched path is Q8-only)",
                ));
            }
            let q_view = pbs.dn_q_batch.sub_offset(row_off * v_dim, m * v_dim);
            let k_view = pbs.dn_k_batch.sub_offset(row_off * v_dim, m * v_dim);
            let v_view = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let gate_view = pbs.dn_alpha_batch.sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs.dn_beta_batch.sub_offset(row_off * n_v_heads, m * n_v_heads);
            let out_view = pbs.dn_attn_out_batch.sub_offset(row_off * v_dim, m * v_dim);
            gpu.gated_delta_net_q8_batch_seq(
                &q_view,
                &k_view,
                &v_view,
                &gate_view,
                &beta_view,
                &dn_states[s].s_matrices[delta_layer_idx],
                &dn_states[s].s_scales[delta_layer_idx],
                &out_view,
                m,
                n_v_heads,
                config.linear_value_head_dim,
                dn_states[s].ef_residual(delta_layer_idx),
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 7. Batched gated output norm — stateless, batched over N.
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;

    // 8. wo + residual.
    q8_residual_proj(
        gpu,
        &layer.wo,
        &pbs.dn_normed_batch,
        &pbs.x_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 9. FFN: rmsnorm, gate+up, silu_mul, w_down + residual.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &layer.ffn_norm,
        &pbs.x_rot_batch,
        n,
        config.dim,
        config.norm_eps,
    )?;
    q8_gate_up_proj(
        gpu,
        &layer.w_gate,
        &layer.w_up,
        &pbs.x_rot_batch,
        &pbs.gate_ffn_batch,
        &pbs.up_batch,
        n,
        q8_wmma_arch,
    )?;
    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
    q8_residual_proj(
        gpu,
        &layer.w_down,
        &pbs.ffn_hidden_batch,
        &pbs.x_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )
}

/// Run one `LinearAttention` (DeltaNet) + MoE layer across the whole step.
///
/// Same shape as [`run_deltanet_layer_slots`] — the stateless attention
/// pieces run once over all `n` rows, the stateful pieces (conv1d, GDN) loop
/// per slot — except: (a) the QKVZA projection and wo admit MQ4G256 as well
/// as Q8_0 (see `require_batchable_deltanet_moe_layer`), and (b) the dense
/// FFN (rmsnorm+gate/up+silu_mul+down) is replaced by a call to the
/// reference's own `prefill_moe_ffn_body_batched`. The MoE FFN is stateless
/// per row — it takes no `kv_cache`, `dn_state`, or `positions` — so no
/// slot-aware variant is needed: the flat `[n × dim]` `pbs.x_batch` this
/// function already threads through the attention body is exactly the input
/// shape `prefill_moe_ffn_body_batched` expects.
#[allow(clippy::too_many_arguments)]
fn run_deltanet_moe_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &DeltaNetMoeLayerWeights,
    batch: &SlotBatch,
    dn_states: &mut [DeltaNetState],
    pbs: &PrefillBatchScratch,
    q8_wmma_arch: bool,
    n: usize,
    delta_layer_idx: usize,
    // Whole-MODEL flag (not per-layer): true when ANY MoE layer anywhere in
    // the model has an MQ6 FFN projection. Threaded straight through to
    // `prefill_moe_ffn_body_batched`'s `model_has_mq6_moe` — see that
    // parameter's use at qwen35.rs:8488 (`force_mq4_grouped_fp16`), a
    // cross-layer kernel-selection consistency knob on gfx1151. Passing this
    // layer's own (uniform-MQ4G256, per `require_batchable_moe_ffn`) dtype
    // instead of the model-wide flag would silently diverge from the
    // reference whenever the SAME model mixes an MQ6 layer elsewhere.
    weights_moe_has_mq6: bool,
) -> HipResult<()> {
    let attn_dtype = require_batchable_deltanet_moe_layer(layer)?;
    require_batchable_moe_ffn(gpu, &layer.ffn)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // 1-2. rmsnorm(+FWHT-rotate for MQ4) then the batched 4-way QKVZA
    // projection. Mirrors forward_prefill_chunk's DeltaNetMoe `is_mq`/`is_q8`
    // forks (qwen35.rs) — MQ4G256 shares HFQ4G256's byte layout, so the only
    // difference from HFQ4G256 is the FWHT-rotated input.
    match attn_dtype {
        MoeAttnDtype::Mq4G256 => {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.attn_norm,
                &layer.wqkv,
                &pbs.x_rot_batch,
                config.dim,
                config.norm_eps,
                n,
            )?;
            run_fused_qkvza_key(
                gpu,
                KernelKey::FusedQkvzaHfq4G256,
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_rot_batch,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
            )?;
        }
        MoeAttnDtype::Q8_0 => {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                n,
                config.dim,
                config.norm_eps,
            )?;
            if q8_wmma_arch {
                run_fused_qkvza_key(
                    gpu,
                    KernelKey::FusedQkvzaQ8_0,
                    &layer.wqkv.buf,
                    &layer.wz.buf,
                    &layer.w_beta.buf,
                    &layer.w_alpha.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch,
                    &pbs.dn_z_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    layer.wqkv.m,
                    layer.wz.m,
                    layer.w_beta.m,
                    layer.w_alpha.m,
                    layer.wqkv.k,
                    n,
                )?;
            } else {
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.wqkv.buf,
                    layer.wqkv.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch,
                    layer.wqkv.m,
                    layer.wqkv.k,
                    n,
                )?;
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.wz.buf,
                    layer.wz.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.dn_z_batch,
                    layer.wz.m,
                    layer.wz.k,
                    n,
                )?;
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.w_beta.buf,
                    layer.w_beta.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.dn_beta_batch,
                    layer.w_beta.m,
                    layer.w_beta.k,
                    n,
                )?;
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.w_alpha.buf,
                    layer.w_alpha.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.dn_alpha_batch,
                    layer.w_alpha.m,
                    layer.w_alpha.k,
                    n,
                )?;
            }
        }
    }

    // 3. Fused sigmoid(beta) + alpha_gate(alpha) — stateless, batched over N.
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;

    // 4. conv1d — STATEFUL (ring buffer), looped per slot. Byte-identical to
    // run_deltanet_layer_slots step 4 — conv_weight is a dequantized F32
    // GpuTensor regardless of the layer's projection weight dtype.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            let q_out = pbs.dn_q_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let k_out = pbs.dn_k_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let v_out = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let input = pbs.dn_qkv_batch.sub_offset(row_off * qkv_dim, m * qkv_dim);
            gpu.conv1d_silu_split_f32_n(
                &q_out,
                &k_out,
                &v_out,
                &input,
                &layer.conv_weight,
                &dn_states[s].conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                m,
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 5. Q/K L2-norm(+scale, +repeat-interleave) — stateless, batched over N.
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
    } else {
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_num_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }

    // 6. Gated Delta Net recurrence — STATEFUL (S matrix), looped per slot.
    // Same Q8-only per-slot dispatch as run_deltanet_layer_slots step 6 —
    // the GDN recurrence dtype is a DeltaNetState property, independent of
    // this layer's projection weight dtype.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if !matches!(dn_states[s].quant, StateQuant::Q8) {
                return Err(HipError::new(
                    0,
                    "forward_batch_slots: DeltaNetState must be StateQuant::Q8 \
                     (the multi-slot batched path is Q8-only)",
                ));
            }
            let q_view = pbs.dn_q_batch.sub_offset(row_off * v_dim, m * v_dim);
            let k_view = pbs.dn_k_batch.sub_offset(row_off * v_dim, m * v_dim);
            let v_view = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let gate_view = pbs.dn_alpha_batch.sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs.dn_beta_batch.sub_offset(row_off * n_v_heads, m * n_v_heads);
            let out_view = pbs.dn_attn_out_batch.sub_offset(row_off * v_dim, m * v_dim);
            gpu.gated_delta_net_q8_batch_seq(
                &q_view,
                &k_view,
                &v_view,
                &gate_view,
                &beta_view,
                &dn_states[s].s_matrices[delta_layer_idx],
                &dn_states[s].s_scales[delta_layer_idx],
                &out_view,
                m,
                n_v_heads,
                config.linear_value_head_dim,
                dn_states[s].ef_residual(delta_layer_idx),
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 7. Batched gated output norm — stateless, batched over N.
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;

    // 8. wo + residual.
    match attn_dtype {
        MoeAttnDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.x_batch,
            &pbs.dn_normed_rot_batch,
            n,
        )?,
        MoeAttnDtype::Q8_0 => q8_residual_proj(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.x_batch,
            &pbs.dn_normed_rot_batch,
            n,
            q8_wmma_arch,
        )?,
    }

    // 9. Batched MoE FFN replaces the dense (rmsnorm + gate+up + silu_mul +
    // w_down) block — stateless per row, no slot machinery needed. Takes
    // pbs.x_batch as input and accumulates the FFN output residual back
    // into it (same contract as the reference's own call site).
    let ctx = DispatchCtx::new(gpu);
    prefill_moe_ffn_body_batched(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights_moe_has_mq6,
        /*routed_out=*/ None,
    )
}

/// Mirrors the `flash_optin && wmma_ok` gate that
/// `hipfire-dispatch/src/families/attention.rs` applies to
/// `KernelKey::AttnQ8_0KvBatchedMasked` BEFORE it ever reaches the LDS/tiled
/// crossover below — on gfx11xx (RDNA3/3.5) this gate is DEFAULT ON (not an
/// opt-in corner case), so any batched Q8 prefill on that hardware (any
/// `n > 1`) actually runs through `attention_q8_0_flash_prefill_wmma`
/// (f16-accumulate WMMA flash-prefill), never the LDS-backed masked kernel
/// this file's crossover assumed was the whole story. Confirmed empirically
/// (see `sp3-defect-report.md`): forcing `HIPFIRE_FLASH_PREFILL=0` on the
/// reference cuts the golden test's worst-element error from 20.49x to
/// 3.93x tolerance, and a layer-by-layer bisection shows the FIRST nonzero
/// hidden-state divergence appears exactly at the first `FullAttention`
/// layer — both point at this kernel-selection gap, not a per-layer op
/// bug.
///
/// Mirrors ONLY the unconditional gfx11 branch of the reference's
/// `flash_default_on = arch.starts_with("gfx11") || gfx12_query16_route_ok`:
/// the gfx12 disjunct reads three `hipfire-dispatch`-private eligibility
/// predicates (one needs a live `DispatchCtx`) this crate has no visibility
/// into. Restricting to `has_wmma_w32() && !has_wmma_w32_gfx12()` means this
/// NEVER opts in on gfx12 even if `HIPFIRE_FLASH_PREFILL=1` forces the
/// reference on there too — a strict narrowing (documented scope gap, same
/// category as the crossover below already carries for the scalar
/// flash-prefill ladder), not a guess at gfx12's real eligibility window.
fn q8_flash_prefill_wmma_eligible(gpu: &Gpu, head_dim: usize, batch_size: usize) -> bool {
    if batch_size <= 1 {
        return false;
    }
    let default_on = gpu.arch.starts_with("gfx11");
    let flash_optin = match hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL")
        .ok()
        .as_deref()
    {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => default_on,
    };
    if !flash_optin {
        return false;
    }
    let variant = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_KERNEL")
        .unwrap_or_else(|_| "wmma".to_owned());
    variant != "scalar"
        && gpu.arch_caps.has_wmma_w32()
        && !gpu.arch_caps.has_wmma_w32_gfx12()
        && head_dim % 32 == 0
        && head_dim <= 256
}

/// Crossover between the LDS-backed masked kernel (no context ceiling issue
/// below the crossover) and the tiled no-LDS-cap flash kernel, mirroring the
/// fallback ladder in `hipfire-dispatch/src/families/attention.rs`'s
/// `AttnQ8_0KvBatchedMasked` arm. Deliberately narrower than that arm in one
/// remaining respect: the scalar flash-prefill opt-in variant (gated by
/// `HIPFIRE_FLASH_PREFILL_KERNEL=scalar` + a long-context minimum) has no
/// `_slots` port, so that one case still falls through to this crossover —
/// SP1 built the two families used here, plus (below) the WMMA flash-prefill
/// family, now ported for BOTH the single-active-slot reduction and the
/// genuinely-multi-slot case (`attention_q8_0_flash_prefill_wmma_slots`).
///
/// `single_slot`, when `Some((k_base, slab_bytes))`, means exactly one slot
/// is active in this step (true for every `n_slots == 1` call, and for a
/// larger pool whenever only one slot has live rows this step). `k_base` is
/// that slot's byte offset into the shared arena (`SlotPool` guarantees
/// `v_base == k_base`); slot 0 of a fresh pool always has `k_base == 0`, so
/// this reduces byte-for-byte to the reference's own `k_cache`/`v_cache`
/// addressing when `n_slots == 1`. This path is kept (rather than folded into
/// the general multi-slot path below) because it is strictly cheaper: no
/// tile-array build/upload, no descriptor indirection, no extra kernarg
/// pointers or per-tile table lookups — just the plain legacy kernel against
/// a pointer-shifted view, address-identical to what SP3 verified
/// bit-identical (0.000x tolerance) against the reference.
///
/// `multi_slot_tiles`, when `Some((tile_slot_dev, tile_row0_dev,
/// tile_qbase_dev, n_tiles))`, means MORE than one slot is active this step
/// AND the reference's own gfx11 WMMA-flash-prefill gate
/// (`q8_flash_prefill_wmma_eligible`) would fire for this shape — the exact
/// condition under which the reference's single-sequence dispatch (which has
/// no concept of "slots" at all, just a flat batch) would have run this
/// kernel. `tile_*_dev` are persistent per-step staging buffers (see
/// `SlotDescStaging`) sized to `desc_staging.max_rows`; only their first
/// `n_tiles` entries are valid this step, hence the `sub_offset` views built
/// below (which exist purely to give the launcher's `tile_slot.numel()` grid
/// sizing the correct `n_tiles`, not `max_rows`).
#[allow(clippy::too_many_arguments)]
fn q8_attend_slots(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    out: &GpuTensor,
    positions: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    batch_size: usize,
    flash_partials: &GpuTensor,
    descs_dev: &GpuTensor,
    row_slot_dev: &GpuTensor,
    single_slot: Option<(u64, usize)>,
    multi_slot_tiles: Option<(&GpuTensor, &GpuTensor, &GpuTensor, usize)>,
) -> HipResult<()> {
    if let Some((k_base, slab_bytes)) = single_slot {
        if q8_flash_prefill_wmma_eligible(gpu, head_dim, batch_size) {
            let k_view = k_cache.sub_offset(k_base as usize, slab_bytes);
            let v_view = v_cache.sub_offset(k_base as usize, slab_bytes);
            return gpu.attention_q8_0_flash_prefill_wmma(
                q,
                &k_view,
                &v_view,
                out,
                positions,
                n_heads,
                n_kv_heads,
                head_dim,
                batch_size,
            );
        }
    } else if let Some((tile_slot_dev, tile_row0_dev, tile_qbase_dev, n_tiles)) = multi_slot_tiles
    {
        // Caller (forward_batch_slots_with_max_layer) only populates
        // multi_slot_tiles when q8_flash_prefill_wmma_eligible already held —
        // re-check anyway so this function's own contract doesn't depend on
        // the caller getting that right, matching this crate's usual
        // defense-in-depth style (see require_q8_fullattn_layer et al.).
        if n_tiles > 0 && q8_flash_prefill_wmma_eligible(gpu, head_dim, batch_size) {
            // Views exist only so `.numel()` reports n_tiles, not the
            // persistent buffers' max_rows capacity — see this function's
            // doc comment. Byte length as reported by the view is otherwise
            // unused (only the pointer is read downstream).
            let tile_slot_view = tile_slot_dev.sub_offset(0, n_tiles);
            let tile_row0_view = tile_row0_dev.sub_offset(0, n_tiles);
            let tile_qbase_view = tile_qbase_dev.sub_offset(0, n_tiles);
            return gpu.attention_q8_0_flash_prefill_wmma_slots(
                q,
                k_cache,
                v_cache,
                out,
                positions,
                n_heads,
                n_kv_heads,
                head_dim,
                batch_size,
                Some(descs_dev),
                Some(&tile_slot_view),
                Some(&tile_row0_view),
                Some(&tile_qbase_view),
            );
        }
    }
    let crossover = if gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201() {
        4096
    } else {
        8192
    };
    if max_ctx_len <= crossover {
        gpu.attention_q8_0_kv_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
            Some(descs_dev),
            Some(row_slot_dev),
        )
    } else {
        gpu.attention_flash_q8_0_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            Some(descs_dev),
            Some(row_slot_dev),
        )
    }
}

/// Run one `FullAttention` layer across the whole step. Every step here is
/// stateless per row and runs once over all `n` rows — the KV write and
/// attend calls are each a SINGLE launch across every slot via the
/// `_slots` entry points (that's the whole point of the descriptor table),
/// and RoPE is slot-agnostic (SP2 Task 2).
#[allow(clippy::too_many_arguments)]
fn run_fullattn_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &FullAttnLayerWeights,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    desc_staging: &SlotDescStaging,
    q8_wmma_arch: bool,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    single_slot: Option<(u64, usize)>,
    n_tiles: Option<usize>,
) -> HipResult<()> {
    require_q8_fullattn_layer(layer)?;

    let dim = config.dim;

    // 1. rmsnorm.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &layer.attn_norm,
        &pbs.x_rot_batch,
        n,
        dim,
        config.norm_eps,
    )?;

    // 2. Batched 3-way QKV projection.
    if q8_wmma_arch {
        run_fused_qkv_key(
            gpu,
            KernelKey::FusedQkvQ8_0,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else {
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.wq.buf,
            layer.wq.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.wk.buf,
            layer.wk.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &layer.wv.buf,
            layer.wv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    }

    // 3. Deinterleave Q + gate.
    gpu.deinterleave_f32_batched(
        &pbs.fa_q_full_batch,
        &pbs.fa_q_batch,
        &pbs.fa_gate_batch,
        config.n_heads,
        config.head_dim,
        n,
    )?;

    // 4. Per-head Q/K rmsnorm.
    gpu.rmsnorm_batched(
        &pbs.fa_q_batch,
        &layer.q_norm,
        &pbs.fa_q_batch,
        n * config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &pbs.fa_k_batch,
        &layer.k_norm,
        &pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    // 5. RoPE — slot-agnostic (SP2 Task 2): indexes by global flat row via
    // `pbs.positions`, which SlotBatch already fills per-slot-absolute and
    // global-row-indexed. No compaction in the slot path yet, so
    // pos_offset is always 0.
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32_batched(
        &pbs.fa_q_batch,
        &pbs.fa_k_batch,
        &pbs.positions,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
        n,
        0,
    )?;

    // 6. Batched KV write — slot-aware, one launch each for K and V across
    // every slot. Both arenas resolve through `k_base`/`v_base`; correct
    // under the Q8_0 ABI (`SlotPool` enforces `v_base == k_base`). asym3
    // cannot use this path — its K/V strides differ — and this file never
    // routes asym3 through it (Q8_0-only scope).
    gpu.kv_cache_write_q8_0_batched_slots(
        k_cache,
        &pbs.fa_k_batch,
        &pbs.positions,
        config.n_kv_heads,
        config.head_dim,
        n,
        Some(&desc_staging.descs_dev),
        Some(&desc_staging.row_slot_dev),
    )?;
    gpu.kv_cache_write_q8_0_batched_slots(
        v_cache,
        &pbs.fa_v_batch,
        &pbs.positions,
        config.n_kv_heads,
        config.head_dim,
        n,
        Some(&desc_staging.descs_dev),
        Some(&desc_staging.row_slot_dev),
    )?;

    // 7. Batched attend — slot-aware, one launch across every slot.
    // `positions[]` (not any `desc.seq_len`) is authoritative for the
    // causal bound inside the ported kernels — SP1's only Critical defect
    // came from conflating the two on a tile kernel bounded by
    // `desc.seq_len` while the shared reduce kernel stayed bounded by
    // `positions[]`. `tree_bias` is never combined with descriptors here
    // (asserted out of SP1 scope).
    q8_attend_slots(
        gpu,
        &pbs.fa_q_batch,
        k_cache,
        v_cache,
        &pbs.fa_attn_out_batch,
        &pbs.positions,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        physical_cap,
        max_ctx_len,
        n,
        &s.flash_partials,
        &desc_staging.descs_dev,
        &desc_staging.row_slot_dev,
        single_slot,
        n_tiles.map(|nt| {
            (
                &desc_staging.tile_slot_dev,
                &desc_staging.tile_row0_dev,
                &desc_staging.tile_qbase_dev,
                nt,
            )
        }),
    )?;

    // 8. sigmoid(gate) * attn_out.
    gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;

    // 9. wo + residual.
    q8_residual_proj(
        gpu,
        &layer.wo,
        &pbs.fa_attn_out_batch,
        &pbs.x_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 10. FFN: rmsnorm, gate+up, silu_mul, w_down + residual.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &layer.ffn_norm,
        &pbs.x_rot_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    q8_gate_up_proj(
        gpu,
        &layer.w_gate,
        &layer.w_up,
        &pbs.x_rot_batch,
        &pbs.gate_ffn_batch,
        &pbs.up_batch,
        n,
        q8_wmma_arch,
    )?;
    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
    q8_residual_proj(
        gpu,
        &layer.w_down,
        &pbs.ffn_hidden_batch,
        &pbs.x_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )
}

/// Run one `FullAttention` + MoE layer across the whole step. Same shape as
/// [`run_fullattn_layer_slots`] — attention (KV write + attend) is a SINGLE
/// slot-aware launch across every slot via the `_slots` entry points,
/// unconditionally on the Q8_0 KV-cache tier regardless of this layer's
/// projection weight dtype — except: (a) the QKV projection and wo admit
/// MQ4G256 as well as Q8_0 (see `require_batchable_fullattn_moe_layer`), and
/// (b) the dense FFN is replaced by the reference's own
/// `prefill_moe_ffn_body_batched` (stateless per row, no slot machinery
/// needed — see `run_deltanet_moe_layer_slots`'s doc comment).
#[allow(clippy::too_many_arguments)]
fn run_fullattn_moe_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &FullAttnMoeLayerWeights,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    desc_staging: &SlotDescStaging,
    q8_wmma_arch: bool,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    single_slot: Option<(u64, usize)>,
    n_tiles: Option<usize>,
    weights_moe_has_mq6: bool,
) -> HipResult<()> {
    let attn_dtype = require_batchable_fullattn_moe_layer(layer)?;
    require_batchable_moe_ffn(gpu, &layer.ffn)?;

    let dim = config.dim;

    // 1-2. rmsnorm(+FWHT-rotate for MQ4) then the batched 3-way QKV
    // projection. Mirrors forward_prefill_chunk's FullAttnMoe
    // `qkv_is_mq`/`qkv_is_q8` forks (qwen35.rs).
    match attn_dtype {
        MoeAttnDtype::Mq4G256 => {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.attn_norm,
                &layer.wq,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
            run_fused_qkv_key(
                gpu,
                KernelKey::FusedQkvHfq4G256,
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }
        MoeAttnDtype::Q8_0 => {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.attn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
            if q8_wmma_arch {
                run_fused_qkv_key(
                    gpu,
                    KernelKey::FusedQkvQ8_0,
                    &layer.wq.buf,
                    &layer.wk.buf,
                    &layer.wv.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_k_batch,
                    &pbs.fa_v_batch,
                    layer.wq.m,
                    layer.wk.m,
                    layer.wv.m,
                    layer.wq.k,
                    n,
                )?;
            } else {
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.wq.buf,
                    layer.wq.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    layer.wq.m,
                    layer.wq.k,
                    n,
                )?;
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.wk.buf,
                    layer.wk.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    layer.wk.m,
                    layer.wk.k,
                    n,
                )?;
                run_plain_gemm_key(
                    gpu,
                    KernelKey::GemmQ8_0BatchedChunked,
                    &layer.wv.buf,
                    layer.wv.gpu_dtype,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    layer.wv.m,
                    layer.wv.k,
                    n,
                )?;
            }
        }
    }

    // 3. Deinterleave Q + gate.
    gpu.deinterleave_f32_batched(
        &pbs.fa_q_full_batch,
        &pbs.fa_q_batch,
        &pbs.fa_gate_batch,
        config.n_heads,
        config.head_dim,
        n,
    )?;

    // 4. Per-head Q/K rmsnorm.
    gpu.rmsnorm_batched(
        &pbs.fa_q_batch,
        &layer.q_norm,
        &pbs.fa_q_batch,
        n * config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &pbs.fa_k_batch,
        &layer.k_norm,
        &pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    // 5. RoPE — slot-agnostic (SP2 Task 2).
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32_batched(
        &pbs.fa_q_batch,
        &pbs.fa_k_batch,
        &pbs.positions,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
        n,
        0,
    )?;

    // 6. Batched KV write — slot-aware, Q8_0 KV-cache tier regardless of
    // this layer's projection weight dtype (see module doc).
    gpu.kv_cache_write_q8_0_batched_slots(
        k_cache,
        &pbs.fa_k_batch,
        &pbs.positions,
        config.n_kv_heads,
        config.head_dim,
        n,
        Some(&desc_staging.descs_dev),
        Some(&desc_staging.row_slot_dev),
    )?;
    gpu.kv_cache_write_q8_0_batched_slots(
        v_cache,
        &pbs.fa_v_batch,
        &pbs.positions,
        config.n_kv_heads,
        config.head_dim,
        n,
        Some(&desc_staging.descs_dev),
        Some(&desc_staging.row_slot_dev),
    )?;

    // 7. Batched attend — slot-aware, one launch across every slot.
    q8_attend_slots(
        gpu,
        &pbs.fa_q_batch,
        k_cache,
        v_cache,
        &pbs.fa_attn_out_batch,
        &pbs.positions,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        physical_cap,
        max_ctx_len,
        n,
        &s.flash_partials,
        &desc_staging.descs_dev,
        &desc_staging.row_slot_dev,
        single_slot,
        n_tiles.map(|nt| {
            (
                &desc_staging.tile_slot_dev,
                &desc_staging.tile_row0_dev,
                &desc_staging.tile_qbase_dev,
                nt,
            )
        }),
    )?;

    // 8. sigmoid(gate) * attn_out.
    gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;

    // 9. wo + residual.
    match attn_dtype {
        MoeAttnDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.x_batch,
            &pbs.fa_attn_out_rot_batch,
            n,
        )?,
        MoeAttnDtype::Q8_0 => q8_residual_proj(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.x_batch,
            &pbs.fa_attn_out_rot_batch,
            n,
            q8_wmma_arch,
        )?,
    }

    // 10. Batched MoE FFN — stateless per row, no slot machinery needed.
    let ctx = DispatchCtx::new(gpu);
    prefill_moe_ffn_body_batched(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights_moe_has_mq6,
        /*routed_out=*/ None,
    )
}

/// Final output norm + per-slot last-token logits. Gathers only the last
/// row of each ACTIVE slot (`m_per_slot[s] > 0`) into a compact
/// `[n_slots × dim]` block before rmsnorm + the lm_head GEMM, rather than
/// normalizing and projecting all N rows — mirrors the reference's
/// "legacy path: only last-token logits" shortcut, generalized from one
/// row to `n_slots` rows. Idle slots' rows in `logits_out` are left
/// whatever was already there; callers must not sample them (this is the
/// same contract `SlotBatch.m_per_slot` already establishes).
///
/// Per-active-slot reduction: for EVERY active slot (not just when the pool
/// has exactly one), run the reference's own single-vector legacy-path
/// kernels (`rmsnorm_f32` + `Step::Gemv`, `qwen35.rs:11984-12008`)
/// byte-for-byte, rather than a single batched `rmsnorm_batched` +
/// `GemmQ8_0BatchedChunked(n_slots)` call. Originally an `n_slots == 1`
/// special case (SP3's fix for root cause #2, confirmed by a layer-by-layer
/// bisection — see `sp3-defect-report.md`: with the FullAttention
/// kernel-selection fix in `q8_attend_slots` also in place, EVERY layer's
/// hidden state is bit-identical between this file and the reference up to
/// and including the last layer, so the only divergence was here, a
/// `rmsnorm_batched`+`GemmQ8_0BatchedChunked` (n=1) vs `rmsnorm_f32`+GEMV
/// numeric mismatch between two mathematically-equivalent but
/// differently-implemented kernel families, not a logic bug). Generalized
/// to every slot count once the WMMA flash-prefill port (root cause #1)
/// stopped masking the same mismatch at `n_slots >= 2` — see the loop body
/// below for the empirical confirmation.
fn final_logits_per_slot(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
) -> HipResult<()> {
    // The lm_head goes through `Step::Gemv` + `weights.output.dispatch_ref()`
    // below, which is exactly what the reference does (qwen35.rs, the
    // `weights.output` GEMV) and is dtype-generic — the dispatcher picks the
    // kernel from `gpu_dtype`. A Q8_0-only gate here was therefore
    // over-restrictive and blocked A3B, whose untied lm_head is MQ4G256, even
    // though the very next lines would have dispatched it correctly.
    //
    // Kept as an allow-list rather than removed: an unsupported dtype should
    // still fail here with a clear message naming the lm_head, not deep inside
    // the dispatcher.
    if !matches!(weights.output.gpu_dtype, DType::Q8_0 | DType::MQ4G256) {
        return Err(HipError::new(
            0,
            &format!(
                "forward_batch_slots: lm_head (weights.output) dtype {:?} is not \
                 supported by the multi-slot path (expected Q8_0 or MQ4G256)",
                weights.output.gpu_dtype
            ),
        ));
    }
    let dim = config.dim;
    let n_slots = batch.m_per_slot.len();
    assert!(
        n_slots <= pbs.max_batch,
        "forward_batch_slots: n_slots ({n_slots}) exceeds pbs.max_batch ({})",
        pbs.max_batch
    );
    assert!(
        logits_out.numel() >= n_slots * config.vocab_size,
        "forward_batch_slots: logits_out has {} elements, need >= {} (n_slots * vocab_size)",
        logits_out.numel(),
        n_slots * config.vocab_size
    );

    // Per-active-slot reduction: run the reference's own single-vector
    // legacy-path kernels (`rmsnorm_f32` + `Step::Gemv`, `qwen35.rs:11984-
    // 12008`) once per active slot, byte-for-byte, rather than a single
    // batched `rmsnorm_batched`+`GemmQ8_0BatchedChunked(n_slots)` call.
    //
    // This generalizes what was originally an `n_slots == 1` special case
    // (SP3's fix for root cause #2 — see sp3-defect-report.md) to every
    // slot count. That generalization is necessary, not merely tidier: the
    // REFERENCE this file is checked against (`run_reference_for_slot` in
    // `test_forward_slots_golden.rs`) always computes each slot through an
    // independent single-sequence `forward_prefill_batch` call, which is
    // ALWAYS one row through the GEVM path — regardless of how many slots
    // this file's own SlotBatch happens to have. A batched GEMM over M =
    // n_slots rows is a mathematically-equivalent but numerically DIFFERENT
    // kernel from M independent GEVMs (different accumulation order), so it
    // was never going to match at n_slots >= 2 either, once the FullAttention
    // kernel-selection gap (root cause #1, the WMMA flash-prefill port)
    // stopped masking it. Confirmed empirically: before this change, fixing
    // only the WMMA gap left n_slots=2 failing at ~4.4x tolerance — the same
    // residual root-cause-#2 numbers SP3 first saw at n_slots=1 before this
    // exact fix.
    let mut row_off = 0usize;
    for (slot, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            let last_row = row_off + m - 1;
            let dim_row_bytes = dim * 4;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last_row * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            let ctx = DispatchCtx::new(gpu);
            let wr = weights.output.dispatch_ref();
            let logits_view = logits_out.sub_offset(slot * config.vocab_size, config.vocab_size);
            let step = Step::Gemv {
                w: &wr,
                input: GemvInput::Raw(&s.tmp),
                out: &logits_view,
            };
            execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, batch.total_rows());
    Ok(())
}

/// Advance every active slot in `batch` by one step: embed, run every
/// layer, and write per-slot last-token logits into `logits_out`
/// (`[n_slots × vocab_size]` f32 — row `s` is valid iff
/// `batch.m_per_slot[s] > 0`; sampling it is the caller's job, e.g. via
/// `Gpu::sample_per_slot`, Task 3's harness).
///
/// `k_arenas`/`v_arenas` hold one arena tensor per `FullAttention` layer
/// (in model layer order), each sized to `pool.arena_bytes()` — the whole
/// multi-slot arena for that layer, addressed via `pool.descriptors()`'s
/// `k_base`/`v_base` byte offsets. `dn_states` holds one `DeltaNetState`
/// per slot (indexed the same as `pool`/`desc_staging`).
///
/// Dense layers are Q8_0-only; `DeltaNetMoe`/`FullAttnMoe` layers additionally
/// admit uniform MQ4G256 (see module doc). A non-admitted weight dtype, a
/// non-Q8 `DeltaNetState`, or a MoE FFN dtype combination
/// `moe_ffn_batched_admissible` rejects returns `Err` rather than guessing at
/// an untested path.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
) -> HipResult<()> {
    forward_batch_slots_with_max_layer(
        gpu,
        weights,
        config,
        batch,
        pool,
        dn_states,
        k_arenas,
        v_arenas,
        desc_staging,
        pbs,
        s,
        logits_out,
        None,
    )
}

/// Debugging/bisection variant of [`forward_batch_slots`]: early-exits the
/// layer loop at `max_layer` (exclusive), mirroring the reference's own
/// `max_layer` parameter on `forward_prefill_batch_with_pbs_opts`
/// (`qwen35.rs:6554`). `pbs.x_batch[0..n*dim]` holds the post-layer hidden
/// state on return when `max_layer` is `Some` — read it directly (it is
/// `pub`) to diff against the reference's own `pbs.x_batch` after an
/// identically-bounded call. Skips the final norm/lm_head AND the per-slot
/// KV-length advance when `max_layer` is `Some`, exactly as the reference
/// skips `do_lm_head` — an early-exit call must not mutate `pool`'s
/// bookkeeping with a partial forward's positions.
///
/// Not `#[cfg(test)]`-gated: kept as a normal `pub fn` so
/// `test_forward_slots_golden` (an `examples/` binary in a different crate)
/// can call it without a feature flag plumbing exercise. `forward_batch_slots`
/// above is the real entry point every other caller keeps using unchanged.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_with_max_layer(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    max_layer: Option<usize>,
) -> HipResult<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let n = batch.total_rows();
    let n_slots = pool.descriptors().len();
    assert_eq!(
        batch.m_per_slot.len(),
        n_slots,
        "forward_batch_slots: batch has {} slots, pool has {}",
        batch.m_per_slot.len(),
        n_slots
    );
    assert_eq!(
        dn_states.len(),
        n_slots,
        "forward_batch_slots: dn_states.len() ({}) must equal n_slots ({n_slots})",
        dn_states.len()
    );
    assert_eq!(
        desc_staging.n_slots, n_slots,
        "forward_batch_slots: desc_staging was built for a different n_slots"
    );
    assert!(
        n <= pbs.max_batch,
        "forward_batch_slots: batch.total_rows() ({n}) exceeds pbs.max_batch ({})",
        pbs.max_batch
    );
    assert!(
        n <= desc_staging.max_rows,
        "forward_batch_slots: batch.total_rows() ({n}) exceeds desc_staging.max_rows ({})",
        desc_staging.max_rows
    );
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    assert_eq!(
        k_arenas.len(),
        n_fa_layers,
        "forward_batch_slots: k_arenas.len() must equal the model's FullAttention layer count"
    );
    assert_eq!(v_arenas.len(), k_arenas.len());

    let dim = config.dim;

    // ── 1. Embed tokens (Q8_0 only) ──────────────────────────────────────
    if !matches!(weights.embd_format, EmbeddingFormat::Q8_0) {
        return Err(HipError::new(
            0,
            "forward_batch_slots: embedding table must be Q8_0 (the multi-slot \
             batched path is Q8_0-only, see sp3-task-2-report.md)",
        ));
    }
    let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &pbs.x_batch, &pbs.tokens, n, dim)?;

    // ── 2. Upload positions ──────────────────────────────────────────────
    // Per-row ABSOLUTE position within that row's own slot — authoritative
    // for the causal bound everywhere downstream (RoPE angle, KV write
    // slot-relative index, and the attend kernels' per-row seq_len). Never
    // `desc.seq_len` — see SP1's only Critical defect.
    let positions_host: Vec<i32> = batch.positions.clone();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;

    // ── 3. Upload row_slot (every step) and the descriptor table (only
    // when dirty) — once per step, not once per layer. ──────────────────
    let row_slot_bytes: Vec<u8> = batch.row_slot.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&desc_staging.row_slot_dev.buf, &row_slot_bytes)?;
    if pool.descriptors_dirty() {
        let desc_bytes = pack_descs(pool.descriptors());
        gpu.hip.memcpy_htod(&desc_staging.descs_dev.buf, &desc_bytes)?;
        pool.mark_uploaded();
    }

    let physical_cap = pool.descriptors()[0].cap as usize;
    let max_ctx_len = (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
        .min(physical_cap)
        .max(1);
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);

    // Exactly one active slot this step? (Always true for n_slots == 1;
    // also true for a larger pool when every row this step belongs to the
    // same slot.) When so, `q8_attend_slots` can safely reduce to the
    // reference's own WMMA flash-prefill kernel (see its doc comment) —
    // that kernel has no slot-descriptor concept, so it is unsafe to use
    // whenever more than one slot has live rows in the same call.
    let active_slots = batch.m_per_slot.iter().filter(|&&m| m > 0).count();
    let single_slot = if active_slots == 1 {
        let slot_idx = batch
            .m_per_slot
            .iter()
            .position(|&m| m > 0)
            .expect("active_slots == 1 implies exactly one m_per_slot entry > 0");
        let k_base = pool.descriptors()[slot_idx].k_base;
        let slab_bytes = pool.arena_bytes() / n_slots;
        Some((k_base, slab_bytes))
    } else {
        None
    };

    // Genuinely multi-slot AND the reference's own gfx11 WMMA-flash-prefill
    // gate (q8_flash_prefill_wmma_eligible) would fire for this shape: the
    // exact condition under which the reference's flat, slot-unaware
    // dispatch would have run attention_q8_0_flash_prefill_wmma. Built once
    // per step here (not once per FullAttention layer — head_dim/batch_size
    // are layer-invariant config, and re-uploading per layer would repeat
    // identical work), mirroring row_slot_dev's "every step, not every
    // layer" upload policy.
    let n_tiles: Option<usize> = if single_slot.is_none()
        && active_slots > 1
        && q8_flash_prefill_wmma_eligible(gpu, config.head_dim, n)
    {
        // Fixed M_TILE = 16: the WMMA kernel's WMMA-fragment tile size, not
        // the scalar kernel's tunable BR.
        const WMMA_M_TILE: usize = 16;
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&batch.m_per_slot, WMMA_M_TILE);
        let nt = tile_slot.len();
        assert!(
            nt <= desc_staging.max_rows,
            "forward_batch_slots: n_tiles ({nt}) exceeds desc_staging.max_rows \
             ({}) capacity — a tile always owns >= 1 row so this should be \
             unreachable unless max_rows was undersized",
            desc_staging.max_rows
        );
        if nt > 0 {
            let to_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
            gpu.hip
                .memcpy_htod(&desc_staging.tile_slot_dev.buf, &to_bytes(&tile_slot))?;
            gpu.hip
                .memcpy_htod(&desc_staging.tile_row0_dev.buf, &to_bytes(&tile_row0))?;
            gpu.hip
                .memcpy_htod(&desc_staging.tile_qbase_dev.buf, &to_bytes(&tile_qbase))?;
        }
        Some(nt)
    } else {
        None
    };

    // ── 4. Per-layer loop ─────────────────────────────────────────────────
    let layer_end = config.n_layers.min(max_layer.unwrap_or(usize::MAX));
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                run_deltanet_layer_slots(
                    gpu,
                    config,
                    layer,
                    batch,
                    dn_states,
                    pbs,
                    q8_wmma_arch,
                    n,
                    delta_layer_idx,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                run_fullattn_layer_slots(
                    gpu,
                    config,
                    layer,
                    pbs,
                    s,
                    &k_arenas[kv_layer_idx],
                    &v_arenas[kv_layer_idx],
                    desc_staging,
                    q8_wmma_arch,
                    n,
                    physical_cap,
                    max_ctx_len,
                    single_slot,
                    n_tiles,
                )?;
                kv_layer_idx += 1;
            }
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                run_deltanet_moe_layer_slots(
                    gpu,
                    config,
                    layer,
                    batch,
                    dn_states,
                    pbs,
                    q8_wmma_arch,
                    n,
                    delta_layer_idx,
                    weights.moe_has_mq6,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                run_fullattn_moe_layer_slots(
                    gpu,
                    config,
                    layer,
                    pbs,
                    s,
                    &k_arenas[kv_layer_idx],
                    &v_arenas[kv_layer_idx],
                    desc_staging,
                    q8_wmma_arch,
                    n,
                    physical_cap,
                    max_ctx_len,
                    single_slot,
                    n_tiles,
                    weights.moe_has_mq6,
                )?;
                kv_layer_idx += 1;
            }
            (_, lt) => {
                return Err(HipError::new(
                    0,
                    &format!(
                        "forward_batch_slots: layer {layer_idx} weight/type mismatch \
                         (layer_type={lt:?})"
                    ),
                ));
            }
        }
    }

    if max_layer.is_some() {
        // Early-exit for bisection: mirror the reference's `do_lm_head =
        // ... && max_layer.is_none()` — skip the final norm/lm_head AND the
        // KV-length advance below (a partial forward must not tell `pool`
        // this step's positions were fully written).
        return Ok(());
    }

    // ── 5. Final norm + per-slot last-token logits ──────────────────────
    final_logits_per_slot(gpu, weights, config, batch, pbs, s, logits_out)?;

    // ── 6. Advance each slot's logical KV length ────────────────────────
    //
    // This step exists because SP1 removed the device-side
    // `positions[row] + 1 <= desc.seq_len` guard: it shipped in release
    // (compiler.rs never passes -DNDEBUG) and cost 64 bytes/lane of scratch on
    // four kernels. Removing it was right for occupancy, but it left the
    // invariant unenforced — and SP3 Task 3's review found that nothing was
    // keeping `seq_len` in sync with a slot's real history either. A caller who
    // forgot to update it got no error, just quietly-wrong metadata that the
    // next step would read as the slot's length.
    //
    // Maintaining it HERE rather than asking callers to remember is
    // correct-by-construction: this function is the only thing that advances a
    // slot's KV, so it is the only thing that can get the length right.
    for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 {
            continue;
        }
        // The slot's new length is one past its highest written position. Rows
        // are packed in slot order, so this slot's last row is the highest.
        let last_row = batch
            .row_slot
            .iter()
            .rposition(|&s| s as usize == slot_ix)
            .expect("m_per_slot > 0 implies at least one row for this slot");
        let new_len = (batch.positions[last_row] + 1) as usize;
        pool.set_seq_len(rdna_compute::slot_pool::SlotId(slot_ix), new_len)
            .map_err(|e| HipError::new(0, &format!("forward_batch_slots: {e}")))?;
    }
    Ok(())
}
