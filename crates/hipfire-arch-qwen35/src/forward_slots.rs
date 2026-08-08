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
// Scope: Q8_0 only. Every weight this path touches (QKV/QKVZA, wo,
// gate/up, down, and the lm_head) must be `DType::Q8_0`, the KV cache is
// Q8_0-quantized, and DeltaNet state is `StateQuant::Q8`. This matches the
// ABI the multi-slot infrastructure was actually built against —
// `SlotPool`'s per-slot addressing is documented as a Q8_0 ABI (asym3 is
// explicitly exempted because its K/V strides differ and it cannot share
// `k_base`/`v_base`), `kv_cache_write_q8_0_batched_slots` and both
// `attention_*_batched_masked_slots` entry points are Q8_0-named, and
// `gated_delta_net_q8_batch_seq` is the Q8 recurrence. Extending this to
// other dtypes is future work, not silently guessed at here — a non-Q8
// weight or a MoE layer returns a clear `HipError` rather than attempting
// a dtype branch nothing in SP1/SP2 covers.
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
    q8_prefill_wmma_enabled, run_fused_gate_up_key, run_fused_qkv_key, run_fused_qkvza_key,
    run_plain_gemm_key, run_residual_gemm_key, DeltaNetLayerWeights, DeltaNetState,
    FullAttnLayerWeights, LayerType, LayerWeights, PrefillBatchScratch, Qwen35Config,
    Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::slot_batch::SlotBatch;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::llama::EmbeddingFormat;
use rdna_compute::kv_slots::KvSlotDesc;
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
    n_slots: usize,
    max_rows: usize,
}

impl SlotDescStaging {
    pub fn new(gpu: &mut Gpu, n_slots: usize, max_rows: usize) -> HipResult<Self> {
        let descs_dev = gpu.alloc_tensor(&[n_slots * 24], DType::Raw)?;
        let row_slot_dev = match gpu.alloc_tensor(&[max_rows * 4], DType::Raw) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(descs_dev);
                return Err(e);
            }
        };
        Ok(Self {
            descs_dev,
            row_slot_dev,
            n_slots,
            max_rows,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.descs_dev);
        let _ = gpu.free_tensor(self.row_slot_dev);
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

/// Crossover between the LDS-backed masked kernel (no context ceiling issue
/// below the crossover) and the tiled no-LDS-cap flash kernel, mirroring the
/// fallback ladder in `hipfire-dispatch/src/families/attention.rs`'s
/// `AttnQ8_0KvBatchedMasked` arm. Deliberately narrower than that arm: the
/// WMMA flash-prefill and scalar flash-prefill opt-in variants it also tries
/// have no `_slots` port (SP1 built only the two families used here), so
/// this uses only those two.
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
) -> HipResult<()> {
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

/// Final output norm + per-slot last-token logits. Gathers only the last
/// row of each ACTIVE slot (`m_per_slot[s] > 0`) into a compact
/// `[n_slots × dim]` block before rmsnorm + the lm_head GEMM, rather than
/// normalizing and projecting all N rows — mirrors the reference's
/// "legacy path: only last-token logits" shortcut, generalized from one
/// row to `n_slots` rows. Idle slots' rows in `logits_out` are left
/// whatever was already there; callers must not sample them (this is the
/// same contract `SlotBatch.m_per_slot` already establishes).
fn final_logits_per_slot(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pbs: &PrefillBatchScratch,
    logits_out: &GpuTensor,
) -> HipResult<()> {
    if !matches!(weights.output.gpu_dtype, DType::Q8_0) {
        return Err(HipError::new(
            0,
            "forward_batch_slots: lm_head (weights.output) must be Q8_0",
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

    // x_rot_batch and x_norm_batch are both dead at this point (their last
    // use was this layer's FFN, already consumed by the w_down residual
    // add) — reused here as compact last-token-per-slot scratch.
    let mut row_off = 0usize;
    for (slot, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            let last_row = row_off + m - 1;
            gpu.hip.memcpy_dtod_at(
                &pbs.x_rot_batch.buf,
                slot * dim * 4,
                &pbs.x_batch.buf,
                last_row * dim * 4,
                dim * 4,
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, batch.total_rows());

    gpu.rmsnorm_batched(
        &pbs.x_rot_batch,
        &weights.output_norm,
        &pbs.x_norm_batch,
        n_slots,
        dim,
        config.norm_eps,
    )?;
    run_plain_gemm_key(
        gpu,
        KernelKey::GemmQ8_0BatchedChunked,
        &weights.output.buf,
        weights.output.gpu_dtype,
        &pbs.x_norm_batch,
        logits_out,
        weights.output.m,
        weights.output.k,
        n_slots,
    )
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
/// Q8_0-only (see module doc). A non-Q8_0 weight, a non-Q8 `DeltaNetState`,
/// or a MoE layer returns `Err` rather than guessing at an untested path.
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

    // ── 4. Per-layer loop ─────────────────────────────────────────────────
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
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
                )?;
                kv_layer_idx += 1;
            }
            (LayerWeights::DeltaNetMoe(_), _) | (LayerWeights::FullAttnMoe(_), _) => {
                return Err(HipError::new(
                    0,
                    "forward_batch_slots: MoE layers are out of scope for the \
                     multi-slot batched path (see sp3-task-2-report.md)",
                ));
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

    // ── 5. Final norm + per-slot last-token logits ──────────────────────
    final_logits_per_slot(gpu, weights, config, batch, pbs, logits_out)
}
