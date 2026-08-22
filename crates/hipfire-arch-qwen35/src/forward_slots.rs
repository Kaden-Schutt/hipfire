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
// Scope: dense layers admit uniform `Q8_0` or uniform `MQ4G256`. Every weight
// a `DeltaNet`/`FullAttn` (dense) layer touches (QKV/QKVZA, wo, gate/up, down,
// and the lm_head) must share one of those two dtypes; the KV cache is
// Q8_0-quantized and DeltaNet state is `StateQuant::Q8` regardless. MQ4G256
// differs only in that activations are FWHT-rotated before each GEMM and the
// `*Hfq4G256` kernel keys are used (MQ4 shares HFQ4's byte layout) — the same
// treatment the MoE bodies already gave their attention projections. This is
// what lets a dense MQ4 checkpoint such as `qwen3.6-27b.mq4` run multi-slot.
// MQ6/HFQ6, MQ3, ParoQ4G128 and mixed-dtype-within-layer remain unported. This matches the ABI the multi-slot infrastructure was
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
    moe_ffn_batched_admissible, mq6_batched_admit_enabled_from_env, prefill_moe_ffn_body_batched,
    q8_prefill_wmma_enabled, run_fused_gate_up_key, run_fused_qkv_key, run_fused_qkvza_key,
    run_plain_gemm_key, run_residual_gemm_key, DeltaNetLayerWeights, DeltaNetMoeLayerWeights,
    DeltaNetState, FullAttnLayerWeights, FullAttnMoeLayerWeights, LayerType, LayerWeights,
    MoeFfnWeights, PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::slot_batch::SlotBatch;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::{GemvFamily, RotateInputs};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_batched_for,
    rotate_x_mq_batched_for, EmbeddingFormat, WeightTensor,
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

/// Uniform-dtype gate for a `DeltaNetLayerWeights`. All eight projections must
/// share ONE dtype — `Q8_0` or `MQ4G256`. Uniformity is the load-bearing part:
/// a mixed-dtype layer would silently misroute through one dtype's stride
/// against a differently-strided weight, the exact corruption class the dense
/// batched-prefill path guards against at every one of its own dtype branches.
fn require_batchable_deltanet_layer(layer: &DeltaNetLayerWeights) -> HipResult<AttnProjDtype> {
    let all = |d: DType| {
        layer.wqkv.gpu_dtype == d
            && layer.wz.gpu_dtype == d
            && layer.w_beta.gpu_dtype == d
            && layer.w_alpha.gpu_dtype == d
            && layer.wo.gpu_dtype == d
            && layer.w_gate.gpu_dtype == d
            && layer.w_up.gpu_dtype == d
            && layer.w_down.gpu_dtype == d
    };
    if all(DType::Q8_0) {
        return Ok(AttnProjDtype::Q8_0);
    }
    if all(DType::MQ4G256) {
        return Ok(AttnProjDtype::Mq4G256);
    }
    Err(HipError::new(
        0,
        "forward_batch_slots: dense DeltaNet layer is neither uniformly Q8_0 nor \
         uniformly MQ4G256; the multi-slot batched path implements those two \
         (MQ6/HFQ6, MQ3, ParoQ4G128 and mixed-dtype-within-layer are real \
         reference paths but are not ported here)",
    ))
}

fn require_batchable_fullattn_layer(layer: &FullAttnLayerWeights) -> HipResult<AttnProjDtype> {
    let all = |d: DType| {
        layer.wq.gpu_dtype == d
            && layer.wk.gpu_dtype == d
            && layer.wv.gpu_dtype == d
            && layer.wo.gpu_dtype == d
            && layer.w_gate.gpu_dtype == d
            && layer.w_up.gpu_dtype == d
            && layer.w_down.gpu_dtype == d
    };
    if all(DType::Q8_0) {
        return Ok(AttnProjDtype::Q8_0);
    }
    if all(DType::MQ4G256) {
        return Ok(AttnProjDtype::Mq4G256);
    }
    Err(HipError::new(
        0,
        "forward_batch_slots: dense FullAttention layer is neither uniformly \
         Q8_0 nor uniformly MQ4G256; see require_batchable_deltanet_layer",
    ))
}

/// Which batched projection dispatch an attention body should take. Applies to
/// dense and MoE layers alike: the projections are the same weights either way,
/// and only the FFN downstream differs.
/// Mirrors the `is_q8` / `is_mq` dtype forks `forward_prefill_chunk` applies
/// to `DeltaNetMoe`/`FullAttnMoe` layers (qwen35.rs's DeltaNetMoe LA branch
/// and FullAttnMoe FA branch) — narrowed to the two dtypes this file
/// actually implements a slot-aware body for. MQ6G256/HFQ6G256, ParoQ4G128,
/// and mixed-dtype-within-layer are real paths in the reference but are NOT
/// ported here; see `require_batchable_deltanet_moe_layer` /
/// `require_batchable_fullattn_moe_layer`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AttnProjDtype {
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
fn require_batchable_deltanet_moe_layer(
    layer: &DeltaNetMoeLayerWeights,
) -> HipResult<AttnProjDtype> {
    let all_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wz.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0);
    if all_q8 {
        return Ok(AttnProjDtype::Q8_0);
    }
    let all_mq4 = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wz.gpu_dtype, DType::MQ4G256)
        && matches!(layer.w_beta.gpu_dtype, DType::MQ4G256)
        && matches!(layer.w_alpha.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wo.gpu_dtype, DType::MQ4G256);
    if all_mq4 {
        return Ok(AttnProjDtype::Mq4G256);
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
fn require_batchable_fullattn_moe_layer(
    layer: &FullAttnMoeLayerWeights,
) -> HipResult<AttnProjDtype> {
    let all_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0)
        && matches!(layer.wk.gpu_dtype, DType::Q8_0)
        && matches!(layer.wv.gpu_dtype, DType::Q8_0)
        && matches!(layer.wo.gpu_dtype, DType::Q8_0);
    if all_q8 {
        return Ok(AttnProjDtype::Q8_0);
    }
    let all_mq4 = matches!(layer.wq.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wk.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wv.gpu_dtype, DType::MQ4G256)
        && matches!(layer.wo.gpu_dtype, DType::MQ4G256);
    if all_mq4 {
        return Ok(AttnProjDtype::Mq4G256);
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

/// Dense FFN body, dtype-forked. Mirrors the reference's dense
/// `ffn_is_mq`/`ffn_is_q8` forks (qwen35.rs LA and FA branches) exactly:
///
///   * MQ4G256 — `fused_rmsnorm_rotate_mq_batched_for(w_gate)`, the
///     `FusedGateUpHfq4G256` fused gate+up, then **`fused_silu_mul_rotate_mq_
///     batched_for(w_down)`** and a plain `GemmHfq4G256Residual`. The rotate
///     for `w_down` is FUSED INTO the SwiGLU, so this must NOT route through
///     `mq4_residual_proj` (which rotates its own input) — that would rotate
///     twice and is the one place the two dtypes are not symmetric.
///   * Q8_0 — plain rmsnorm, `q8_gate_up_proj`, plain `silu_mul_f32`,
///     `q8_residual_proj`.
#[allow(clippy::too_many_arguments)]
fn dense_ffn_body_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    ffn_norm: &GpuTensor,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    w_down: &WeightTensor,
    pbs: &PrefillBatchScratch,
    dtype: AttnProjDtype,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    match dtype {
        AttnProjDtype::Mq4G256 => {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                ffn_norm,
                w_gate,
                &pbs.x_rot_batch,
                config.dim,
                config.norm_eps,
                n,
            )?;
            run_fused_gate_up_key(
                gpu,
                KernelKey::FusedGateUpHfq4G256,
                &w_gate.buf,
                &w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                w_gate.m,
                w_up.m,
                w_gate.k,
                n,
            )?;
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                w_down.k,
                n,
            )?;
            run_residual_gemm_key(
                gpu,
                KernelKey::GemmHfq4G256Residual,
                &w_down.buf,
                w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                w_down.m,
                w_down.k,
                n,
            )
        }
        AttnProjDtype::Q8_0 => {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                ffn_norm,
                &pbs.x_rot_batch,
                n,
                config.dim,
                config.norm_eps,
            )?;
            q8_gate_up_proj(
                gpu,
                w_gate,
                w_up,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                n,
                q8_wmma_arch,
            )?;
            gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
            q8_residual_proj(
                gpu,
                w_down,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                &pbs.x_rot_batch,
                n,
                q8_wmma_arch,
            )
        }
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
    let attn_dtype = require_batchable_deltanet_layer(layer)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // 1-2. rmsnorm (+FWHT-rotate for MQ4) then the batched 4-way QKVZA
    // projection. Mirrors the reference's dense LA `is_mq` fork: MQ4G256
    // shares HFQ4G256's byte layout, so the only difference is the rotated
    // input (qwen35.rs's `FusedQkvzaHfq4G256` default branch).
    if attn_dtype == AttnProjDtype::Mq4G256 {
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
    } else {
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
            let gate_view = pbs
                .dn_alpha_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs
                .dn_beta_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
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

    // 8. wo + residual. MQ4 rotates its input for the pre-rotated weights
    // (the reference's `wo_is_mq` fork); Q8 projects directly.
    match attn_dtype {
        AttnProjDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.x_batch,
            &pbs.dn_normed_rot_batch,
            n,
        )?,
        AttnProjDtype::Q8_0 => q8_residual_proj(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.x_batch,
            &pbs.x_rot_batch,
            n,
            q8_wmma_arch,
        )?,
    }

    // 9. FFN: rmsnorm, gate+up, silu_mul, w_down + residual — dtype-forked.
    dense_ffn_body_slots(
        gpu,
        config,
        &layer.ffn_norm,
        &layer.w_gate,
        &layer.w_up,
        &layer.w_down,
        pbs,
        attn_dtype,
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
        AttnProjDtype::Mq4G256 => {
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
        AttnProjDtype::Q8_0 => {
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
            let gate_view = pbs
                .dn_alpha_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs
                .dn_beta_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
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
        AttnProjDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.x_batch,
            &pbs.dn_normed_rot_batch,
            n,
        )?,
        AttnProjDtype::Q8_0 => q8_residual_proj(
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
/// The WMMA flash-prefill kernel's query-row tile size (its WMMA-fragment
/// tile, not the scalar kernel's tunable BR). Both the eligibility floor and
/// `build_tiles` must use the same value.
const WMMA_M_TILE: usize = 16;

fn q8_flash_prefill_wmma_eligible(gpu: &Gpu, head_dim: usize, batch_size: usize) -> bool {
    // Mirrors the reference's own gate exactly (`batch_size > 1`), and must
    // keep doing so: the single-active-slot reduction below uses this answer
    // to decide whether it may drop into the reference's own WMMA kernel, and
    // any divergence there breaks byte-exactness with the reference arm.
    // The decode-batch floor lives on the multi-slot tile path instead, which
    // has no reference counterpart -- see MULTI_SLOT_TILE_MIN_ROWS.
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
                q, &k_view, &v_view, out, positions, n_heads, n_kv_heads, head_dim, batch_size,
            );
        }
    } else if let Some((tile_slot_dev, tile_row0_dev, tile_qbase_dev, n_tiles)) = multi_slot_tiles {
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
    // Context length above which the flash (context-split) kernel beats the
    // scalar one. Overridable via HIPFIRE_SLOTS_ATTN_CROSSOVER.
    //
    // gfx1151's 2048 is measured on this path, not inherited: at 4096-token
    // context the scalar kernel reads 17.8 MiB of KV per FA layer in 586 us
    // (~30 GB/s) because it launches only n_kv_heads x slots = 64 workgroups
    // and each scans the whole context serially. The flash kernel splits the
    // context and recovers that parallelism. Interleaved A/B, 35B-A3B, 4
    // slots, 4096-token context, 3 reps each, no overlap between arms:
    //
    //   scalar 38.02 / 38.60 / 38.18 ms per decode step  (mean 38.27)
    //   flash  35.66 / 35.94 / 36.53 ms                  (mean 36.04)
    //
    // Flash also won at 1, 2 and 3 slots (21.46->18.24, 29.19->26.85,
    // 34.49->31.25), so this is not a batching effect -- the old 8192 was
    // simply on the wrong side for this path. At ~1086 tokens the two arms
    // were within run-to-run noise, hence 2048 rather than 0.
    //
    // Left at 8192 for other non-gfx120x parts: the measurement is gfx1151's
    // and does not transfer. gfx1200/gfx1201 keep 4096, which already puts a
    // 4096-token context on the flash side.
    let crossover = gpu.slots_attn_crossover().unwrap_or({
        if gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201() {
            4096
        } else if gpu.arch_caps.is_gfx1151() {
            2048
        } else {
            8192
        }
    });
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
    mrope_pos3: Option<&GpuTensor>,
) -> HipResult<()> {
    let attn_dtype = require_batchable_fullattn_layer(layer)?;

    let dim = config.dim;

    // 1-2. rmsnorm (+FWHT-rotate for MQ4) then the batched 3-way QKV
    // projection — the reference's dense FA `is_mq` fork.
    if attn_dtype == AttnProjDtype::Mq4G256 {
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
    } else {
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
    if let Some(pos3) = mrope_pos3 {
        // VL prefill: 3D mrope with per-row (t, h, w). Batched twin of the
        // sequential path's rope_mrope_halfsplit_f32.
        gpu.rope_mrope_halfsplit_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pos3.buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
            config.mrope_section,
        )?;
    } else {
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
    }

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

    // 8. sigmoid(gate) * attn_out, over the `n` LIVE rows only.
    //
    // `fa_attn_out_batch` / `fa_gate_batch` are sized for the largest prefill
    // batch, and `sigmoid_mul_f32` derives its element count from the tensor
    // it is handed. Passing the whole buffer therefore did the same work on a
    // 4-row decode step as on a full prefill chunk: profiled at 8388608
    // elements (2048 rows x n_heads*head_dim) and 427.6 us per call, 10 calls
    // per step -- 4.28 ms, 10.3% of a 4-slot decode step, on 2044 dead rows.
    //
    // The reference does the same thing at its own batched call sites, but
    // only ever reaches them for prefill, where n is the batch. Its decode
    // path uses the single-row `s.fa_attn_out` scratch instead. This path is
    // the one that runs a batched buffer at decode row counts.
    //
    // Rows >= n keep whatever they held rather than being overwritten with
    // sigmoid(garbage)*garbage; nothing downstream reads them (o_proj takes
    // n rows), and the golden gate holds at 0.000x.
    let gate_elems = n * config.n_heads * config.head_dim;
    gpu.sigmoid_mul_f32(
        &pbs.fa_attn_out_batch.sub_offset(0, gate_elems),
        &pbs.fa_gate_batch.sub_offset(0, gate_elems),
    )?;

    // 9. wo + residual — MQ4 rotates into the pre-rotated weights first.
    match attn_dtype {
        AttnProjDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.x_batch,
            &pbs.fa_attn_out_rot_batch,
            n,
        )?,
        AttnProjDtype::Q8_0 => q8_residual_proj(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.x_batch,
            &pbs.x_rot_batch,
            n,
            q8_wmma_arch,
        )?,
    }

    // 10. FFN: rmsnorm, gate+up, silu_mul, w_down + residual — dtype-forked.
    dense_ffn_body_slots(
        gpu,
        config,
        &layer.ffn_norm,
        &layer.w_gate,
        &layer.w_up,
        &layer.w_down,
        pbs,
        attn_dtype,
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
    mrope_pos3: Option<&GpuTensor>,
) -> HipResult<()> {
    let attn_dtype = require_batchable_fullattn_moe_layer(layer)?;
    require_batchable_moe_ffn(gpu, &layer.ffn)?;

    let dim = config.dim;

    // 1-2. rmsnorm(+FWHT-rotate for MQ4) then the batched 3-way QKV
    // projection. Mirrors forward_prefill_chunk's FullAttnMoe
    // `qkv_is_mq`/`qkv_is_q8` forks (qwen35.rs).
    match attn_dtype {
        AttnProjDtype::Mq4G256 => {
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
        AttnProjDtype::Q8_0 => {
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
    if let Some(pos3) = mrope_pos3 {
        // VL prefill: 3D mrope with per-row (t, h, w). Batched twin of the
        // sequential path's rope_mrope_halfsplit_f32.
        gpu.rope_mrope_halfsplit_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pos3.buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
            config.mrope_section,
        )?;
    } else {
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
    }

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

    // 8. sigmoid(gate) * attn_out, over the `n` LIVE rows only.
    //
    // `fa_attn_out_batch` / `fa_gate_batch` are sized for the largest prefill
    // batch, and `sigmoid_mul_f32` derives its element count from the tensor
    // it is handed. Passing the whole buffer therefore did the same work on a
    // 4-row decode step as on a full prefill chunk: profiled at 8388608
    // elements (2048 rows x n_heads*head_dim) and 427.6 us per call, 10 calls
    // per step -- 4.28 ms, 10.3% of a 4-slot decode step, on 2044 dead rows.
    //
    // The reference does the same thing at its own batched call sites, but
    // only ever reaches them for prefill, where n is the batch. Its decode
    // path uses the single-row `s.fa_attn_out` scratch instead. This path is
    // the one that runs a batched buffer at decode row counts.
    //
    // Rows >= n keep whatever they held rather than being overwritten with
    // sigmoid(garbage)*garbage; nothing downstream reads them (o_proj takes
    // n rows), and the golden gate holds at 0.000x.
    let gate_elems = n * config.n_heads * config.head_dim;
    gpu.sigmoid_mul_f32(
        &pbs.fa_attn_out_batch.sub_offset(0, gate_elems),
        &pbs.fa_gate_batch.sub_offset(0, gate_elems),
    )?;

    // 9. wo + residual.
    match attn_dtype {
        AttnProjDtype::Mq4G256 => mq4_residual_proj(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.x_batch,
            &pbs.fa_attn_out_rot_batch,
            n,
        )?,
        AttnProjDtype::Q8_0 => q8_residual_proj(
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
    // ── x-batched fast path ──────────────────────────────────────────────
    // The loop below is the last per-slot-serialised op in this file:
    // everything else is one launch across all rows. Each iteration re-reads
    // the whole lm_head weight matrix (270 MiB at vocab 248320 x dim 2048,
    // MQ4G256), so a 4-slot decode step reads it four times for four
    // different activation vectors.
    //
    // `gemv_hfq4g256_xbatch` reads it once and dots it against all B vectors.
    // It is bitwise identical to running the single-vector GEMV B times --
    // gated by `test_gemv_hfq4g256_xbatch`, which checks against the actual
    // hand-tuned gfx1151 lm_head kernel this shape selects, not a generic
    // stand-in. Measured there at M=248320 K=2048:
    //
    //   B=1 1.265 -> 1.312 ms (0.96x)   B=3 3.973 -> 1.457 ms (2.73x)
    //   B=2 2.576 -> 1.325 ms (1.94x)   B=4 5.104 -> 1.868 ms (2.73x)
    //
    // B=1 is a small loss, hence the `>= 2` gate: one slot keeps the tuned
    // single-vector kernel untouched.
    //
    // Only the GEMV is batched. The rmsnorm and the FWHT rotation stay
    // exactly as the per-slot path runs them -- same `rmsnorm_f32`, same
    // `GemvFamily::rotate` with `RotateInputs::default()`, which is what
    // `Step::Gemv { input: Raw }` does internally and is what carries the
    // weight's AWQ sidecar. They are ~8 KiB of work per slot; the 270 MiB
    // weight pass is the whole cost, and reproducing their numerics exactly
    // is worth more than batching them.
    //
    // Requires every slot active so batch row `b` IS slot `b` and the kernel
    // writes straight into `logits_out` with no scatter. That holds on every
    // decode step, which is the only place this matters.
    let all_active = batch.m_per_slot.iter().all(|&m| m > 0);
    if matches!(weights.output.gpu_dtype, DType::MQ4G256)
        && all_active
        && (2..=Gpu::HFQ4G256_XBATCH_MAX).contains(&n_slots)
        && pbs.x_rot_batch.numel() >= n_slots * dim
    {
        let dim_row_bytes = dim * 4;
        let wr = weights.output.dispatch_ref();
        let family = GemvFamily::new();
        let mut row_off = 0usize;
        for (slot, &m) in batch.m_per_slot.iter().enumerate() {
            let last_row = row_off + m - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last_row * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            let ctx = DispatchCtx::new(gpu);
            let rot = family
                .rotate(&ctx, gpu, &wr, &s.tmp, &RotateInputs::default())
                .map_err(|e| HipError::new(0, &e.to_string()))?;
            // `rotate` hands back an alias of a single shared scratch buffer,
            // so each slot's result must be copied out before the next
            // iteration overwrites it.
            gpu.memcpy_dtod_at_auto(
                &pbs.x_rot_batch.buf,
                slot * dim_row_bytes,
                &rot.buf().buf,
                0,
                dim_row_bytes,
            )?;
            row_off += m;
        }
        debug_assert_eq!(row_off, batch.total_rows());
        let x_rot_all = pbs.x_rot_batch.sub_offset(0, n_slots * dim);
        gpu.gemv_hfq4g256_xbatch(
            wr.buf,
            &x_rot_all,
            logits_out,
            config.vocab_size,
            dim,
            n_slots,
        )?;
        return Ok(());
    }

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
/// Dense and MoE layers alike admit uniform `Q8_0` or uniform `MQ4G256`
/// (see module doc). A non-admitted weight dtype, a
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
/// Context-length bucket for the decode graph cache, in tokens.
///
/// A captured graph bakes `max_ctx_len` into kernargs, but it grows by one
/// every decode step, so capturing per step would be pointless. Instead the
/// bound is rounded up to a bucket and the graph is reused until the context
/// crosses into the next one -- one capture per 256 generated tokens.
///
/// Over-scanning inside a bucket is correct, not merely tolerable:
/// `positions[]` is the authoritative causal bound everywhere downstream
/// (SP1's only Critical defect), so positions past a row's own are masked.
/// It costs at most one bucket of extra KV scan per step.
pub const DECODE_GRAPH_CTX_BUCKET: usize = 256;

/// Identity of a captured decode graph. Any change invalidates it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DecodeGraphKey {
    n_slots: usize,
    n_rows: usize,
    ctx_bucket: usize,
    max_layer: Option<usize>,
}

/// A hipGraph of one pure-decode step, reused across steps.
///
/// Why: a 4-slot decode step issues ~1390 kernel launches. Profiled on
/// gfx1151 at 4096-token context, that is 4.16 ms/step of host submission
/// (2.99 us per launch) against 9.29 ms/step of measured GPU idle -- the GPU
/// spends 23% of the step waiting to be fed. Replaying one graph replaces all
/// of those submissions with a single launch.
///
/// Only pure-decode steps are captured. Prefill batches vary in shape step to
/// step, and the win is concentrated in decode anyway: prefill launches the
/// same number of kernels but each does far more work, so submission is not
/// the bottleneck there.
#[derive(Default)]
pub struct SlotDecodeGraph {
    exec: Option<hip_bridge::GraphExec>,
    graph: Option<hip_bridge::Graph>,
    key: Option<DecodeGraphKey>,
    captures: usize,
    replays: usize,
}

impl SlotDecodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many captures and replays this cache has served. A healthy decode
    /// run shows captures ~= generated_tokens / DECODE_GRAPH_CTX_BUCKET.
    pub fn stats(&self) -> (usize, usize) {
        (self.captures, self.replays)
    }

    fn release(&mut self, gpu: &Gpu) {
        if let Some(exec) = self.exec.take() {
            let _ = gpu.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.graph.take() {
            let _ = gpu.hip.graph_destroy(graph);
        }
        self.key = None;
    }
}

/// `forward_batch_slots` with hipGraph capture/replay for pure-decode steps.
///
/// Falls back to the plain path whenever the step is not pure decode, the
/// feature is off, or capture fails -- so this is never load-bearing for
/// correctness, only for speed.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_graphed(
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
    cache: &mut SlotDecodeGraph,
    vl_active: bool,
) -> HipResult<()> {
    let pure_decode = !batch.is_empty() && batch.m_per_slot.iter().all(|&m| m == 1);
    // VL prefill carries embedding overrides + mrope positions, which the
    // captured-graph path cannot represent — always take the plain path.
    if !gpu.slots_decode_graph() || !pure_decode || vl_active {
        return forward_batch_slots(
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
        );
    }

    let physical_cap = pool.descriptors()[0].cap as usize;
    let true_ctx = (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
        .min(physical_cap)
        .max(1);
    let ctx_bucket = true_ctx
        .div_ceil(DECODE_GRAPH_CTX_BUCKET)
        .saturating_mul(DECODE_GRAPH_CTX_BUCKET)
        .min(physical_cap)
        .max(1);
    let key = DecodeGraphKey {
        n_slots: pool.descriptors().len(),
        n_rows: batch.total_rows(),
        ctx_bucket,
        max_layer: None,
    };

    // The per-step inputs always go up outside the graph: their host source
    // buffers are temporaries, and a captured memcpy node bakes its source
    // pointer. The captured kernels read the device buffers these fill, so
    // fresh contents reach a replay without re-capturing.
    upload_step_inputs(gpu, batch, pool, desc_staging, pbs)?;

    if cache.key != Some(key) {
        cache.release(gpu);
        gpu.ensure_capture_stream()?;
        // Everything must already be warm: a kernel compile inside capture is
        // exactly what ThreadLocal capture mode forbids. One uncaptured step
        // at this shape guarantees it.
        forward_batch_slots_opts(
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
            SlotStepOpts {
                skip_uploads: true,
                ctx_override: Some(ctx_bucket),
                embed_override: None,
                mrope_pos3: None,
            },
        )?;
        // The warm-up step above already advanced the slot lengths; re-running
        // the capture body would advance them a second time, so roll back to
        // the pre-step lengths before capturing.
        rewind_slot_seq_lens(batch, pool)?;

        gpu.begin_stream_capture()?;
        let captured = forward_batch_slots_opts(
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
            SlotStepOpts {
                skip_uploads: true,
                ctx_override: Some(ctx_bucket),
                embed_override: None,
                mrope_pos3: None,
            },
        );
        let graph = gpu.end_stream_capture()?;
        captured?;
        let exec = gpu.hip.graph_instantiate(&graph)?;
        cache.graph = Some(graph);
        cache.exec = Some(exec);
        cache.key = Some(key);
        cache.captures += 1;
        // Capture does not execute, and the warm-up step above already did
        // this step's work, so nothing is launched here.
        return Ok(());
    }

    let exec = cache
        .exec
        .take()
        .expect("cache.key set implies an instantiated exec");
    let launched = gpu.launch_graph(&exec);
    cache.exec = Some(exec);
    launched?;
    cache.replays += 1;
    // Host-side bookkeeping the replay cannot do for itself.
    advance_slot_seq_lens(batch, pool)?;
    Ok(())
}

/// Undo `advance_slot_seq_lens` for this batch: restore each active slot to
/// the length it had before the step. Used only by the graph path, which runs
/// the same step body twice (once to warm, once to capture).
fn rewind_slot_seq_lens(batch: &SlotBatch, pool: &mut SlotPool) -> HipResult<()> {
    for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let first_row = batch
            .row_slot
            .iter()
            .position(|&s| s as usize == slot_ix)
            .expect("m_per_slot > 0 implies at least one row for this slot");
        let old_len = batch.positions[first_row] as usize;
        pool.set_seq_len(rdna_compute::slot_pool::SlotId(slot_ix), old_len)
            .map_err(|e| HipError::new(0, &format!("rewind_slot_seq_lens: {e}")))?;
    }
    Ok(())
}

/// Knobs the graph path needs and no other caller does.
#[derive(Clone, Default)]
pub struct SlotStepOpts {
    /// The per-step H2D uploads (tokens, positions, row_slot, descriptors)
    /// were already done by the caller. Set when capturing into a hipGraph:
    /// those copies read from short-lived host buffers, and a captured memcpy
    /// node bakes its source pointer -- replaying one would read freed
    /// memory. The caller does them per step, outside the graph.
    pub skip_uploads: bool,
    /// Use this context bound instead of deriving it from `batch.positions`.
    /// A captured graph bakes `max_ctx_len` into kernargs, but it grows every
    /// decode step, so the graph path rounds it up to a bucket and re-captures
    /// only when the bucket changes. Over-scanning within a bucket is safe:
    /// `positions[]` is the authoritative causal bound (SP1's Critical
    /// defect), so the extra positions are masked out.
    pub ctx_override: Option<usize>,
    /// VL prefill: visual embeddings to write over `x_batch` rows after the
    /// token-embedding lookup. Host layout `[count × dim]`, written to rows
    /// `row0..row0+count` of this step. Rows outside the span keep their
    /// token embeddings.
    pub embed_override: Option<(Vec<f32>, usize)>,
    /// VL prefill: packed 3D mrope positions `[n_rows][3]` i32 for this
    /// step's rows. When set, FA RoPE dispatches to the batched mrope kernel
    /// instead of the 1D one; DeltaNet layers ignore rope entirely.
    pub mrope_pos3: Option<Vec<i32>>,
}

/// The per-step H2D uploads, hoisted so the graph path can run them outside
/// the captured region. Must run before every step, captured or not.
pub fn upload_step_inputs(
    gpu: &mut Gpu,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    desc_staging: &mut SlotDescStaging,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    let n = batch.total_rows();
    let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;

    let positions_host: Vec<i32> = batch.positions.clone();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;

    let row_slot_bytes: Vec<u8> = batch
        .row_slot
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&desc_staging.row_slot_dev.buf, &row_slot_bytes)?;
    if pool.descriptors_dirty() {
        let desc_bytes = pack_descs(pool.descriptors());
        gpu.hip
            .memcpy_htod(&desc_staging.descs_dev.buf, &desc_bytes)?;
        pool.mark_uploaded();
    }
    Ok(())
}

/// Post-step host bookkeeping: advance each active slot's logical KV length.
/// Hoisted for the same reason as the uploads -- it is host code, so a graph
/// replay does not run it, and the caller must.
pub fn advance_slot_seq_lens(batch: &SlotBatch, pool: &mut SlotPool) -> HipResult<()> {
    for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 {
            continue;
        }
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
    forward_batch_slots_opts(
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
        max_layer,
        SlotStepOpts::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_opts(
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
    opts: SlotStepOpts,
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
    if !opts.skip_uploads {
        let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
        let tokens_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    }
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &pbs.x_batch, &pbs.tokens, n, dim)?;

    // ── 1b. VL embedding override ─────────────────────────────────────────
    // Overwrite the image-pad rows with vision-tower embeddings before any
    // layer sees them. Rows outside the span keep their token embeddings.
    if let Some((embs, row0)) = &opts.embed_override {
        eprintln!("[vl-slot] embedding override: row0={row0} elems={} sum={:.4}", embs.len(), embs.iter().sum::<f32>());
        let dim_bytes = dim * 4;
        gpu.hip.memcpy_htod_offset(
            &pbs.x_batch.buf,
            row0 * dim_bytes,
            unsafe { std::slice::from_raw_parts(embs.as_ptr() as *const u8, embs.len() * 4) },
        )?;
    }

    // ── 2. Upload positions ──────────────────────────────────────────────
    // Per-row ABSOLUTE position within that row's own slot — authoritative
    // for the causal bound everywhere downstream (RoPE angle, KV write
    // slot-relative index, and the attend kernels' per-row seq_len). Never
    // `desc.seq_len` — see SP1's only Critical defect.
    if !opts.skip_uploads {
        let positions_host: Vec<i32> = batch.positions.clone();
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    // ── 3. Upload row_slot (every step) and the descriptor table (only
    // when dirty) — once per step, not once per layer. ──────────────────
    if !opts.skip_uploads {
        let row_slot_bytes: Vec<u8> = batch
            .row_slot
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();
        gpu.hip
            .memcpy_htod(&desc_staging.row_slot_dev.buf, &row_slot_bytes)?;
        if pool.descriptors_dirty() {
            let desc_bytes = pack_descs(pool.descriptors());
            gpu.hip
                .memcpy_htod(&desc_staging.descs_dev.buf, &desc_bytes)?;
            pool.mark_uploaded();
        }
    }

    let physical_cap = pool.descriptors()[0].cap as usize;
    let max_ctx_len = opts.ctx_override.unwrap_or(
        (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
            .min(physical_cap)
            .max(1),
    );
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

    // `n > active_slots` means at least one slot contributes more than one row
    // this step -- i.e. there is prompt left to prefill. A step where every
    // active slot contributes exactly one row is pure decode, and there the
    // WMMA *prefill* kernel is both wrong to use and slow:
    //
    //   - Wrong: the reference decodes each sequence with batch_size == 1, so
    //     its own `batch_size <= 1` gate puts it on the scalar path. Batching
    //     N sequences into N rows and then firing the WMMA kernel because
    //     N > 1 diverges from what the reference did for those same tokens.
    //   - Slow: it tiles query rows in WMMA_M_TILE (16) blocks, so 2-4 decode
    //     rows launch a full, mostly-padded tile. Measured on gfx1151,
    //     35B-A3B, 4096-token context, per decode step:
    //
    //       slots | firing on decode | pure-decode scalar | aggregate decode
    //         2   |     57.23 ms     |      33.64 ms      | 34.95 -> 59.45 tok/s
    //         3   |     61.77 ms     |      40.41 ms      | 48.57 -> 74.23 tok/s
    //         4   |     68.60 ms     |      48.47 ms      | 58.31 -> 82.52 tok/s
    //
    //     A flat ~21 ms penalty that made two concurrent users slower *in
    //     aggregate* than one. Prefill still takes the kernel and still wants
    //     it: the 4-slot run prefills in 14.7 s either way, against 19.2 s
    //     with the path disabled outright.
    //
    // Gating on row count instead (n >= 16) was tried and is wrong: it also
    // pushes small *prefill* batches (the golden gate's 14-row, 2-slot case)
    // onto the scalar kernel, which is not bit-identical to the reference's
    // WMMA kernel, and the golden gate fails at 225x tolerance.
    //
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
        && n > active_slots
        && q8_flash_prefill_wmma_eligible(gpu, config.head_dim, n)
    {
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
            let to_bytes =
                |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
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

    // ── 3b. VL mrope pos3 upload (layer-invariant, once per step) ────────
    let mrope_pos3_dev = match &opts.mrope_pos3 {
        Some(host) => {
            let t = gpu.alloc_tensor(
                &[host.len()],
                DType::F32, // i32-in-f32 pattern used by `positions`
            )?;
            let bytes: Vec<u8> = host.iter().flat_map(|x| x.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&t.buf, &bytes)?;
            Some(t)
        }
        None => None,
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
                    mrope_pos3_dev.as_ref(),
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
                    mrope_pos3_dev.as_ref(),
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
    advance_slot_seq_lens(batch, pool)?;
    Ok(())
}
