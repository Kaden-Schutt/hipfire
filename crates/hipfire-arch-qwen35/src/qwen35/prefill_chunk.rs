// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 prefill chunk executor: the batched per-chunk prefill layer loop
//! (`forward_prefill_chunk`) plus its batched MoE FFN body
//! (`prefill_moe_ffn_body_batched`) and full-attention layer body
//! (`run_fa_layer_body`). On the prefill hot path.

use super::*;

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - `moe_ffn_batched_admissible(ffn, arch)` returns true: router +
///   shared_expert_gate may be MQ4G256 or Q8_0; all other MoE weights must
///   use an arch-supported MoE quant family.
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok == 8` and `config.num_experts <= 1024`
///   (hard limits of the batched top-K kernel)
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    layer_idx: usize,
    ctx: &DispatchCtx,
    // EP (Ship 6 substrate-EP prefill): when `Some`, the routed combine writes
    // into this zeroed `[n × dim]` partial instead of `pbs.x_batch` (the EP
    // driver all-reduce-sums it across ranks and adds into x_batch). The shared
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");
    let dtypes = MoePrefillDtypes::from_ffn(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert dtype metadata for batched prefill"))?;
    let expert_shape = moe_expert_shape(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert shape metadata for batched prefill"))?;

    // ── 1. Split rmsnorm vs FWHT rotate ──
    //
    // A3B (and every other MoE here) leaves router + shared_expert_gate
    // as Q8_0 in the quantizer — these tiny tensors lose too much
    // accuracy at 4-bit, so the engine never reduces them. Q8 weights
    // are quantized against the un-rotated rmsnorm output, while the
    // MQ-family siblings (shared_expert.{gate,up,down} +
    // experts.{gate_up,down}) expect FWHT(rmsnorm(x) / awq_scale). Populate both:
    //   x_norm_batch ← rmsnorm(x_batch)
    //   x_rot_batch  ← FWHT(x_norm_batch / awq_scale)  (only if any
    //                  downstream MQ weight is present)
    //
    // Pick `shared_expert.gate` as the AWQ representative (instead of
    // the previous `ffn.router`). Per the F1 imatrix scope every gate-side
    // MQ4 sibling shares the same input basis and therefore an identical
    // awq_scale, but the router itself is excluded from F1 (it stays Q8).
    // Reading awq_scale from router would silently drop AWQ rotation in
    // v3 AWQ runs — latent until this predicate widened.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        ffn_norm,
        &pbs.x_norm_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    // PARO mode (shared_expert.gate is ParoQ4G128): each weight carries its
    // own Givens rotation table (paro.pairs / theta / channel_scales). The
    // shared MQ4-style FWHT pre-rotation here would be wrong — skip it. The
    // ParoQ4G128 dispatch arms below run per-weight Givens rotation in-place
    // before each GEMM, using pbs.x_rot_batch as the rotation destination.
    let paro_mode =
        config.has_shared_expert && matches!(ffn.shared_expert.gate.gpu_dtype, DType::ParoQ4G128);
    if !paro_mode {
        if config.has_shared_expert {
            rotate_x_mq_batched_for(
                gpu,
                &ffn.shared_expert.gate,
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                dim,
                n,
            )?;
        } else {
            gpu.rotate_x_mq_batched(&pbs.x_norm_batch, &pbs.x_rot_batch, dim, n)?;
        }
    }

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // Per-dtype dispatch — Q8 reads `x_norm_batch`, MQ4 reads
    // `x_rot_batch`. The natural 4-way fuse via `gemm_qkvza_hfq4g256`
    // is not applicable when router/shared_expert_gate are Q8 (mixed
    // strides). Four separate launches; +3 per MoE layer over the fused
    // ideal, acceptable for the structural unlock.
    match ffn.router.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::MQ4G256 => gpu.gemm_hfq4g256(
            &ffn.router.buf,
            &pbs.x_rot_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::F32 => gpu.gemm_f32_register_tiled(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::F16 => gpu.gemm_f16_x_f32_wmma(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
            &ffn.router.buf,
            &pbs.x_norm_batch,
            router_logits,
            ffn.router.m,
            ffn.router.k,
            n,
        )?,
        other => panic!(
            "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32, F16, BF16"
        ),
    }
    if config.has_shared_expert {
        match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::MQ4G256 => gpu.gemm_hfq4g256(
                &ffn.shared_expert_gate.buf,
                &pbs.x_rot_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::F32 => gpu.gemm_f32_register_tiled(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::F16 => gpu.gemm_f16_x_f32_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                             — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32, F16, BF16"
            ),
        }
    }
    // #397 Ship 5.2 PILOT: route the router GEMM through GemmFamily::run_key.
    // Each arm uses the *dispatcher-entry* KernelKey (GemmQ8_0BatchedChunked /
    // GemmHfq4G256 / GemmF32Batched) so run_key dispatches to the IDENTICAL
    // gpu.gemm_* method the prior direct call used — preserving each method's
    // own internal arch routing (RDNA4-WMMA / gfx906-dp4a / CDNA-rocBLAS / …)
    // byte-for-byte. The x input still differs per dtype (Q8/F32 read
    // x_norm_batch; MQ4 reads x_rot_batch), exactly as before. The three keys
    // are registered ArchPredicate::Always, so run_key never rejects.
    {
        use hipfire_dispatch::families::gemm::GemmParams;
        let ctx = DispatchCtx::new(gpu);
        let (key, x_in): (hipfire_dispatch::types::KernelKey, &GpuTensor) =
            match ffn.router.gpu_dtype {
                DType::Q8_0 => (
                    hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                    &pbs.x_norm_batch,
                ),
                DType::MQ4G256 => (
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &pbs.x_rot_batch,
                ),
                DType::F32 => (
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &pbs.x_norm_batch,
                ),
                other => panic!(
                    "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admitted admits MQ4G256, Q8_0, F32"
                ),
            };
        let w = WeightRef {
            buf: &ffn.router.buf,
            dtype: ffn.router.gpu_dtype,
            m: ffn.router.m,
            k: ffn.router.k,
            row_stride: ffn.router.k,
            rotation: None,
            awq_scale: None,
        };
        let params = GemmParams {
            w: &w,
            x: x_in,
            y: router_logits,
            batch_size: n,
        };
        hipfire_runtime::dispatch::gemm_family()
            .run_key(key, &ctx, gpu, &params)
            .map_err(HipError::from)?;
    }
    // DIAG: dump MoE router logits (batched)
    dump_hidden_localize(gpu, router_logits, n, 0, ffn.router.m, 0, "router_b");
    // #397 Ship 5.2 slice1: route the shared-expert-gate GEMM through
    // GemmFamily::run_key. Same dtype-routed dispatcher-entry keys as the router
    // match above (Q8/F32 read x_norm_batch, MQ4 reads x_rot_batch) → identical
    // gpu.gemm_* method, byte-for-byte.
    {
        use hipfire_dispatch::types::KernelKey;
        let (key, x_in): (KernelKey, &GpuTensor) = match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => (KernelKey::GemmQ8_0BatchedChunked, &pbs.x_norm_batch),
            DType::MQ4G256 => (KernelKey::GemmHfq4G256, &pbs.x_rot_batch),
            DType::F32 => (KernelKey::GemmF32Batched, &pbs.x_norm_batch),
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32"
            ),
        };
        run_plain_gemm_key(
            gpu,
            key,
            &ffn.shared_expert_gate.buf,
            ffn.shared_expert_gate.gpu_dtype,
            x_in,
            shared_scalar,
            ffn.shared_expert_gate.m,
            ffn.shared_expert_gate.k,
            n,
        )?;
    }
    // Fused gate+up dispatch for the shared expert — halves the kernel
    // launch count vs back-to-back gemm_hfq*g256 (~75µs/launch × 40
    // MoE layers = ~3ms saved on R9700 A3B prefill at bs=256).
    // Per-projection dispatch: gate AND up share the same dtype (predicate
    // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout fused kernels.
    if config.has_shared_expert {
        match ffn.shared_expert.gate.gpu_dtype {
            // #397 Ship 5.2 slice 2: shared-expert fused gate+up → FusedQkvFamily
            // (batched-prefill gate+up variant). Same batched kernel, behavior-preserving.
            DType::MQ4G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            DType::MQ6G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            // Phase 2: PARO shared_expert.gate + up. Each weight has its own
            // Givens rotation table — rotate x_norm_batch into x_rot_batch using
            // gate's tables, GEMM, then re-rotate using up's tables, GEMM. Total
            // 4 dispatches vs the MQ4 path's 1 fused gemm_gate_up — acceptable
            // overhead for the per-token-loop elimination win. Phase 4 could
            // collapse this into a single fused kernel
            // (gemm_gate_up_paro_q4g128_batched) if measurement shows it matters.
            DType::ParoQ4G128 => {
                let paro_gate = ffn
                    .shared_expert
                    .gate
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.gate missing paro metadata");
                let paro_up = ffn
                    .shared_expert
                    .up
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.up missing paro metadata");
                // Gate: rotate x_norm by gate's Givens → x_rot, then HFQ4G128 GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_gate.pairs,
                    &paro_gate.theta,
                    &paro_gate.channel_scales,
                    n,
                    dim,
                    paro_gate.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.gate.buf,
                    ffn.shared_expert.gate.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                // Up: re-rotate x_norm by up's Givens → x_rot (overwrite), GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_up.pairs,
                    &paro_up.theta,
                    &paro_up.channel_scales,
                    n,
                    dim,
                    paro_up.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.up.buf,
                    ffn.shared_expert.up.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::F16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::F16,
                    "shared_expert.gate/up dtype predicate should keep F16 paired"
                );
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::BF16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::BF16,
                    "shared_expert.gate/up dtype predicate should keep BF16 paired"
                );
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.gate dtype {other:?} \
                             — admit predicate should have rejected this layer"
            ),
        }
    }

    // ── 3. GPU softmax + top-K + renorm, batched over N tokens ──
    //
    // Same Path B split as the decode call site: split the fused
    // softmax+topk+renorm into gpu.softmax_f32 + moe_topk_renorm_k8_batched
    // so prefill activations match the CPU-reference softmax math
    // exactly. router_logits is allocated 1D as [n × n_exp]; alias it
    // into a 2D view so gpu.softmax_f32 takes rows = n.
    let router_logits_2d = GpuTensor {
        buf: unsafe { router_logits.buf.alias() },
        shape: vec![n, n_exp],
        dtype: DType::F32,
    };
    gpu.softmax_f32(&router_logits_2d)?;
    let cpu_topk = if k_top == 8 {
        gpu.moe_topk_renorm_k8_batched(
            router_logits,
            topk_indices,
            topk_weights,
            n_exp,
            config.norm_topk_prob,
            n,
        )?;
        None
    } else {
        let probs = gpu.download_f32(router_logits)?;
        let (indices, weights) =
            cpu_topk_from_softmaxed_rows(&probs, n, n_exp, k_top, config.norm_topk_prob)?;
        upload_cpu_topk_to_device(gpu, &indices, &weights, topk_indices, topk_weights)?;
        Some((indices, weights))
    };
    if moe_router_histogram_active() {
        let (indices, weights) = if let Some((indices, weights)) = cpu_topk.as_ref() {
            (indices.clone(), weights.clone())
        } else {
            (
                download_i32_tensor(gpu, topk_indices, n * k_top)?
                    .into_iter()
                    .map(router_index_i32_to_usize)
                    .collect::<Vec<_>>(),
                gpu.download_f32(topk_weights)?,
            )
        };
        for token_idx in 0..n {
            let start = token_idx * k_top;
            let end = start + k_top;
            record_moe_router_selection(layer_idx, &indices[start..end], &weights[start..end]);
        }
    }
    let paged_topk_indices = if ffn.experts.is_empty() {
        let indices = if let Some((indices, _)) = cpu_topk.as_ref() {
            indices.clone()
        } else {
            download_i32_tensor(gpu, topk_indices, n * k_top)?
                .into_iter()
                .map(router_index_i32_to_usize)
                .collect::<Vec<_>>()
        };
        Some(indices)
    } else {
        None
    };
    let paged_expert_buckets = if let Some(indices) = paged_topk_indices.as_ref() {
        Some(build_paged_moe_expert_buckets(indices, n, k_top, n_exp)?)
    } else {
        None
    };

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    // F2: AWQ-aware silu_mul+rotate for the batched shared-expert down input.
    // PARO: shared_expert.down has its own Givens rotation tables (paro.*);
    // use the dedicated fused kernel (commit 50198daa). It takes a per-weight
    // (pairs, theta, channel_scales, krot) tuple instead of the MQ4 FWHT
    // convention. Same shape: gate/up [N × smi] → shared_rot [N × smi].
    if config.has_shared_expert {
        if paro_mode {
            let paro_down = ffn
                .shared_expert
                .down
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.down missing paro metadata");
            gpu.fused_silu_mul_givens_rotate_f32(
                shared_gate,
                shared_up,
                shared_rot,
                &paro_down.pairs,
                &paro_down.theta,
                &paro_down.channel_scales,
                n,
                smi,
                paro_down.krot as usize,
            )?;
        } else if matches!(ffn.shared_expert.down.gpu_dtype, DType::F16 | DType::BF16) {
            gpu.silu_mul_f32(shared_gate, shared_up, shared_rot)?;
        } else {
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &ffn.shared_expert.down,
                shared_gate,
                shared_up,
                shared_rot,
                smi,
                n,
            )?;
        }
    }

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and += sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row]. (Note: HFQ4 sister uses += not
    // atomicAdd; each (bid, row) writes a unique cell.)
    // Per-projection dispatch: MQ4/MQ3/MQ6 route to their HFQ-layout sisters.
    if config.has_shared_expert {
        match ffn.shared_expert.down.gpu_dtype {
            DType::MQ4G256 => gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ6G256 => gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256 => gpu.gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256 => gpu.gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ8G256 => gpu.gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            // Phase 2: HFQ4G128 batched residual+sigmoid-scaled kernel. Single
            // launch, same semantics as the HFQ4G256 sister — reads shared_rot
            // (already silu-mul-rotated by the PARO fused kernel above), GEMVs
            // against W_down, applies sigmoid(shared_scalar[token]) × output,
            // accumulates into pbs.x_batch.
            DType::ParoQ4G128 => gpu.gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::F16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            DType::BF16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.down dtype {other:?} \
                         — admit predicate should have rejected this layer"
            ),
        }
    }

    // ── 6. Routed experts: batched gate_up → SwiGLU+FWHT → down ──
    //
    // Gate/up for top-K experts (per token) → [N × K_TOP × mi]. Each
    // output row reads topk_indices[token × K_TOP + krank] to pick its
    // expert weight base from the device-side expert_gate_up_ptrs table.
    let down_m = expert_shape.down_m;
    let down_k = expert_shape.down_k;
    let gate_up_k = expert_shape.gate_up_k;

    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) — default ON for
    // gfx11/gfx12, where the grouped-WMMA kernel is validated (gfx11 routes
    // to `gemm_hfq4g256_moe_grouped_wmma_k2` via the base w32 WMMA builtin,
    // gfx12 to the `_gfx12` variant). Empirical lift on Qwen3.5-A3B mq4
    // prefill=256: gfx1100 7900 XTX 1396 → 2983 tok/s (+114%); gfx1201
    // R9700 1016 → 2966 tok/s (uniform-mq4.hfq, +192%). CDNA wave64 (gfx9*)
    // and pre-WMMA RDNA (gfx10*) stay on the per-token indexed_batched
    // GEMV path. Opt out with `HIPFIRE_MOE_GROUPED_GEMM=0`.
    // Cached read — getenv on every layer × MoE call adds up.
    static USE_PATH2_GATE_UP: OnceLock<bool> = OnceLock::new();
    let use_path2 = *USE_PATH2_GATE_UP.get_or_init(|| {
        moe_grouped_gemm_path2_enabled_from_env(
            std::env::var("HIPFIRE_MOE_GROUPED_GEMM").ok().as_deref(),
        )
    });
    let path2_eligible = moe_grouped_gemm_path2_eligible_for_dtype(
        dtypes.expert_gate_up,
        &gpu.arch,
        use_path2 && (!ffn.experts.is_empty() || paged_expert_buckets.is_some()),
    );
    // m_total — computed during gate_up scatter, reused for down. Avoids
    // a second dtoh sync per MoE layer.
    let mut path2_m_total: usize = 0;
    let path2_shape = moe_grouped_path2_shape(n, k_top, n_exp);
    if paged_expert_buckets.is_some() && !path2_eligible {
        return Err(HipError::new(
            0,
            "paged grouped-MoE prefill requires grouped GEMM path2 support",
        ));
    }
    moe_prefill_prepare_routed_gate_up_input(
        gpu,
        ffn,
        &dtypes,
        &pbs.x_norm_batch,
        &pbs.x_rot_batch,
        dim,
        n,
    )?;
    if path2_eligible {
        // Stage 1 scatter pipeline. The scratch buffers are sized for
        // worst-case max_batch. Runtime launch bounds use the tighter live
        // slot upper bound below. Block size 16 (the WMMA tile row count).
        const BLOCK_M: usize = MOE_GROUPED_BLOCK_M;
        let counts = pbs.moe_expert_token_counts.as_ref().expect("path2 scratch");
        let offsets = pbs.moe_expert_offsets.as_ref().expect("path2 scratch");
        let sorted = pbs.moe_sorted_slot_index.as_ref().expect("path2 scratch");
        let inverse_perm = pbs.moe_inverse_perm.as_ref().expect("path2 scratch");
        let tile_ids = pbs.moe_expert_tile_ids.as_ref().expect("path2 scratch");
        let y_gu_grouped = pbs.moe_y_gate_up_grouped.as_ref().expect("path2 scratch");
        if let Some(buckets) = paged_expert_buckets.as_ref() {
            if dtypes.expert_gate_up != DType::MQ6G256 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "paged grouped-MoE prefill currently supports MQ6 routed experts only, got {:?}",
                        dtypes.expert_gate_up
                    ),
                ));
            }
            // Load all active experts once before the per-bucket loops so the
            // down phase doesn't need a second round of page-ins.
            let active_experts: Vec<usize> = buckets.iter().map(|b| b.expert as usize).collect();
            ensure_paged_experts_resident(gpu, pager, ffn, &active_experts)?;
            for bucket in buckets {
                upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    bucket.m_total,
                    path2_shape.gate_up_source_rows,
                )?;
                gpu.moe_gate_up_unscatter_k8(
                    y_gu_grouped,
                    sorted,
                    gate_batch,
                    up_batch,
                    mi,
                    k_top,
                    bucket.m_total,
                )?;
            }
        } else {
            // m_total upper bound — scratch is sized in PrefillBatchScratch::new
            // with the all-experts worst case, while this launch only needs slots
            // plus padding for experts that can be non-empty at this N.
            // The scatter fused kernel pre-fills every tile id in this aligned
            // bound with -1; grouped GEMM early-returns on sentinel tiles, so we
            // can skip the m_total dtoh sync entirely. Saves ~50µs/layer.
            let m_total_max = path2_shape.m_total_bound;

            // Fused scatter pipeline: one launch replaces histogram + offsets
            // + permute. Saves 2 launches × ~75µs × MoE layers.
            gpu.moe_scatter_fused_k8(
                topk_indices,
                counts,
                offsets,
                sorted,
                tile_ids,
                inverse_perm,
                path2_shape.total_slots,
                n_exp,
                m_total_max,
                BLOCK_M,
            )?;

            // Use m_total_max as the upper bound for grid sizing — the kernel
            // early-returns on expert_tile_ids[tile_y] == -1 for the
            // pre-sentinel'd unused-tile range.
            path2_m_total = m_total_max;
            let m_total = m_total_max;

            // Stage 2 grouped GEMM (gate_up). Writes Y_grouped[m_total × 2*mi] direct.
            // x_src = x_rot_batch [N × dim], x_row_div = K_TOP.
            // Per-dtype dispatch: experts uniform per layer (admit predicate
            // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout grouped WMMA
            // sisters.
            match dtypes.expert_gate_up {
                DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_rot_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::MQ2G256Lloyd => {
                    if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?
                    } else {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?
                    }
                }
                DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_norm_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_gate_up_ptrs,
                    tile_ids,
                    sorted,
                    &pbs.x_norm_batch,
                    y_gu_grouped,
                    2 * mi,
                    gate_up_k,
                    path2_shape.gate_up_x_row_div,
                    m_total,
                    path2_shape.gate_up_source_rows,
                )?,
                // Phase 4: Path 2 ParoQ4G128 grouped-WMMA. All 256 routed
                // experts at this layer share one gate_up Givens rotation
                // sidecar (ffn.paro_shared.gate_up_*); rotate x_norm into
                // x_rot ONCE, then dispatch the HFQ4G128 grouped WMMA. The
                // kernel auto-converts the F32 x_rot to F16 internally via
                // ensure_fp16_x, same as the G256 sister.
                //
                // gfx1151 i8 MMQ opt-in (HIPFIRE_MOE_PARO_I8=1): routes to the
                // HFQ4G128 i8 MMQ kernel which doubles compute throughput on
                // Strix Halo (~140 vs ~71 TFLOPS). Compute-bound regime per
                // Phase 4 attribution (gemm_paro_q4g128_moe_grouped_wmma_k2
                // = 68.5% GPU time, 25.8 GiB/s — far from BW roof).
                DType::ParoQ4G128 => {
                    let paro = ffn
                        .paro_shared
                        .as_ref()
                        .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro.gate_up_pairs,
                        &paro.gate_up_theta,
                        &paro.gate_up_channel_scales,
                        n,
                        dim,
                        paro.krot as usize,
                    )?;
                    // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                    // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                    // (clean decode, finite logits) + coherence-gate (MQ4 paths
                    // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                    let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                        gpu.arch.as_str(),
                        std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                    );
                    let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                        use_paro_i8,
                        std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                    );
                    if use_paro_i8_k8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    } else if use_paro_i8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    } else {
                        gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                            &ffn.expert_gate_up_ptrs,
                            tile_ids,
                            sorted,
                            &pbs.x_rot_batch,
                            y_gu_grouped,
                            2 * mi,
                            gate_up_k,
                            path2_shape.gate_up_x_row_div,
                            m_total,
                            path2_shape.gate_up_source_rows,
                        )?;
                    }
                }
                other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].gate_up dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
            }

            // Stage 3 unscatter combine. Fans Y_grouped → gate_batch + up_batch.
            gpu.moe_gate_up_unscatter_k8(
                y_gu_grouped,
                sorted,
                gate_batch,
                up_batch,
                mi,
                k_top,
                m_total,
            )?;
        }
    } else {
        // Path 1 fallback (CDNA/gfx10): per-token indexed GEMV, batched
        // over the N tokens via grid.z. The dispatch is dtype-keyed because
        // the kernel reads the weight nibble layout directly (HFQ4G256:
        // 136 B/group; HFQ4G128/PARO: 72 B/group).
        match dtypes.expert_gate_up {
            DType::MQ4G256 => gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ6G256 => gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ2G256 => gpu.gemv_hfq2g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ8G256 => gpu.gemv_hfq8g256_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched(
                &ffn.expert_gate_up_ptrs,
                topk_indices,
                &pbs.x_rot_batch,
                gate_batch,
                up_batch,
                2 * mi,
                gate_up_k,
                k_top,
                n,
            )?,
            // Phase 3 PARO routed-expert: apply the layer's shared gate_up
            // Givens rotation to x_norm_batch into x_rot_batch ONCE, then
            // dispatch the HFQ4G128 indexed batched kernel. All 256 experts
            // at this layer share the same gate_up rotation sidecar
            // (ffn.paro_shared, populated by paro_load_moe_shared_sidecars).
            DType::ParoQ4G128 => {
                let paro = ffn
                    .paro_shared
                    .as_ref()
                    .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro.gate_up_pairs,
                    &paro.gate_up_theta,
                    &paro.gate_up_channel_scales,
                    n,
                    dim,
                    paro.krot as usize,
                )?;
                gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                             experts[0].gate_up dtype {other:?} — admit predicate should \
                             have rejected this layer"
            ),
        }
    }

    // SwiGLU + FWHT over [N*K_TOP × mi] — batch flatten across tokens and
    // expert ranks, k=mi is per-row width.
    // F2: AWQ-aware silu_mul+rotate; experts[0].down is representative (all
    // experts at this layer share imatrix at the same residual basis).
    // PARO branch (Phase 3): the layer-shared `down` rotation sidecar lives
    // on ffn.paro_shared (not per-expert; all 256 experts alias the same
    // tuple). Apply via fused_silu_mul_givens_rotate_f32 over the flattened
    // [n*k_top × mi] grid.
    if paro_mode {
        let paro = ffn
            .paro_shared
            .as_ref()
            .expect("ParoQ4G128 routed experts require paro_shared sidecars");
        gpu.fused_silu_mul_givens_rotate_f32(
            gate_batch,
            up_batch,
            rot_batch,
            &paro.down_pairs,
            &paro.down_theta,
            &paro.down_channel_scales,
            n * k_top,
            mi,
            paro.krot as usize,
        )?;
    } else if matches!(dtypes.expert_down, DType::F16 | DType::BF16) {
        gpu.silu_mul_f32(gate_batch, up_batch, rot_batch)?;
    } else if ffn.experts.is_empty() {
        gpu.fused_silu_mul_rotate_mq_batched(gate_batch, up_batch, rot_batch, mi, n * k_top)?;
    } else {
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &ffn.experts[0].down,
            gate_batch,
            up_batch,
            rot_batch,
            mi,
            n * k_top,
        )?;
    }

    // Down projection. Three paths:
    //   Path 2 (HIPFIRE_MOE_GROUPED_GEMM=1, RDNA): grouped-WMMA-GEMM
    //     reusing the gate_up scatter + inverse_perm + a non-atomic combine.
    //   Path 1 (RDNA, default): atomic-free expanded GEMV write + combine.
    //   Path 0 (CDNA wave64 fallback): residual_scaled atomic GEMV.
    //
    // Path 1: K_TOP-way atomicAdd contention per output cell — 387 GiB/s
    // observed vs 954 on the sister gate_up. Path 2 amortizes weights via
    // WMMA across the m_total tokens routed to each expert; ~67ms saved on
    // the down kernel for A3B prefill at batch 256 (R9700).
    // CDNA (wave64, HBM2/3) stays on Path 0 — cheap HBM atomics +
    // expanded scratch cost makes the GEMV pattern competitive.
    if path2_eligible {
        let y_down_grouped = pbs.moe_y_down_grouped.as_ref().expect("path2 scratch");
        let inverse_perm = pbs.moe_inverse_perm.as_ref().expect("path2 scratch");
        let sorted = pbs.moe_sorted_slot_index.as_ref().expect("path2 scratch");
        let tile_ids = pbs.moe_expert_tile_ids.as_ref().expect("path2 scratch");
        if let Some(buckets) = paged_expert_buckets.as_ref() {
            if dtypes.expert_down != DType::MQ6G256 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "paged grouped-MoE prefill currently supports MQ6 routed down experts only, got {:?}",
                        dtypes.expert_down
                    ),
                ));
            }
            for bucket in buckets {
                upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    bucket.m_total,
                    path2_shape.down_source_rows,
                )?;
                gpu.moe_down_combine_grouped_k8(
                    y_down_grouped,
                    inverse_perm,
                    topk_weights,
                    &pbs.x_batch,
                    down_m,
                    k_top,
                    n,
                )?;
            }
        } else {
            // m_total already computed during gate_up scatter — reuse to skip
            // a second dtoh sync per MoE layer (~50µs each × 40 layers = 2ms).
            let m_total = path2_m_total;

            // Grouped GEMM on down: x_src = rot_batch [N*K_TOP × mi], x_row_div = 1
            // (sorted_slot_index[slot] directly indexes the source row).
            // Per-dtype dispatch: experts uniform per layer. MQ4 → HFQ4-layout;
            // MQ6 → HFQ6 sister (shipped via feat/hfq6-moe-grouped-wmma).
            match dtypes.expert_down {
                DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::MQ2G256Lloyd => {
                    if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?
                    } else {
                        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?
                    }
                }
                DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                    &ffn.expert_down_ptrs,
                    tile_ids,
                    sorted,
                    rot_batch,
                    y_down_grouped,
                    down_m,
                    down_k,
                    path2_shape.down_x_row_div,
                    m_total,
                    path2_shape.down_source_rows,
                )?,
                // Phase 4: Path 2 ParoQ4G128 down grouped-WMMA (with i8 MMQ
                // opt-in for gfx1151 — see gate_up arm above). rot_batch was
                // already Givens-rotated by paro_shared.down_* via the PARO
                // fused_silu_mul_givens_rotate_f32 step above; the kernel is
                // rotation-agnostic. Same kernel for gate_up + down — only
                // shape parameters and x_row_div differ.
                DType::ParoQ4G128 => {
                    // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                    // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                    // (clean decode, finite logits) + coherence-gate (MQ4 paths
                    // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                    let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                        gpu.arch.as_str(),
                        std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                    );
                    let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                        use_paro_i8,
                        std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                    );
                    if use_paro_i8_k8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    } else if use_paro_i8 {
                        gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    } else {
                        gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                            &ffn.expert_down_ptrs,
                            tile_ids,
                            sorted,
                            rot_batch,
                            y_down_grouped,
                            down_m,
                            down_k,
                            path2_shape.down_x_row_div,
                            m_total,
                            path2_shape.down_source_rows,
                        )?;
                    }
                }
                other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].down dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
            }
            // Non-atomic combine via inverse_perm + topk_weights.
            gpu.moe_down_combine_grouped_k8(
                y_down_grouped,
                inverse_perm,
                topk_weights,
                &pbs.x_batch,
                down_m,
                k_top,
                n,
            )?;
        }
    } else {
        let use_atomic_free_down = !gpu.arch.starts_with("gfx9");
        if use_atomic_free_down {
            // Path 1 expanded-down: per-token-per-rank GEMV writes to a
            // [N × K_TOP × M] scratch, then a separate combine kernel folds
            // it back into pbs.x_batch with topk weights. The expanded
            // kernel is dtype-keyed; the combine kernel is dtype-agnostic.
            match dtypes.expert_down {
                DType::MQ4G256 => gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ6G256 => gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256 => gpu.gemv_hfq2g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ8G256 => gpu.gemv_hfq8g256_moe_down_k8_indexed_batched_expanded(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256Lloyd => gpu
                    .gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                DType::MQ3G256Lloyd => gpu
                    .gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                // Phase 3 PARO down: the layer-shared `down` Givens rotation
                // has already been applied to rot_batch by the
                // fused_silu_mul_givens_rotate_f32 call above. The HFQ4G128
                // indexed kernel (existing, shipped in 7c00970d) is
                // rotation-agnostic; same dispatch shape as G256 sister.
                DType::ParoQ4G128 => gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    rot_batch,
                    down_expanded,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?,
                other => panic!(
                    "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                                 experts[0].down dtype {other:?} — admit predicate should \
                                 have rejected this layer"
                ),
            }
            gpu.moe_down_combine_k8_batched(
                down_expanded,
                topk_weights,
                &pbs.x_batch,
                down_m,
                k_top,
                n,
            )?;
        } else {
            gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                &ffn.expert_down_ptrs,
                topk_indices,
                topk_weights,
                rot_batch,
                &pbs.x_batch,
                down_m,
                down_k,
                k_top,
                n,
            )?;
        }
    }
    // ── 6. Routed experts: delegated to MoeFamily::run_prefill (Ship 4.2) ──
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    let total_slots = n * k_top;
    let m_total_max = moe_grouped_m_total_bound(total_slots, n_exp);

    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        experts_all_gate_up_mq4: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256),
        routed_gate_up: ffn.experts[0].gate_up.gpu_dtype,
        routed_down: ffn.experts[0].down.gpu_dtype,
        has_paro_shared: ffn.paro_shared.is_some(),
    };

    let paro_gate_up =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.gate_up_pairs,
                theta: &paro.gate_up_theta,
                scales: &paro.gate_up_channel_scales,
                krot: paro.krot as usize,
            });
    let paro_down =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.down_pairs,
                theta: &paro.down_theta,
                scales: &paro.down_channel_scales,
                krot: paro.krot as usize,
            });
    let down_awq_scale = ffn.experts[0].down.awq_scale.as_ref();

    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        dtypes: moe_dtypes,
        batch_size: n,
        mi,
        down_m,
        down_k,
        gate_up_k,
        k_top,
        n_exp,
        m_total_max,
        topk_indices,
        topk_weights,
        x_batch: &pbs.x_batch,
        x_norm_batch: &pbs.x_norm_batch,
        x_rot_batch: &pbs.x_rot_batch,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
        expert_token_counts: pbs.moe_expert_token_counts.as_ref().expect("moe scratch"),
        expert_offsets: pbs.moe_expert_offsets.as_ref().expect("moe scratch"),
        sorted_slot_index: pbs.moe_sorted_slot_index.as_ref().expect("moe scratch"),
        expert_tile_ids: pbs.moe_expert_tile_ids.as_ref().expect("moe scratch"),
        inverse_perm: pbs.moe_inverse_perm.as_ref().expect("moe scratch"),
        y_gate_up_grouped: pbs.moe_y_gate_up_grouped.as_ref().expect("moe scratch"),
        y_down_grouped: pbs.moe_y_down_grouped.as_ref().expect("moe scratch"),
        paro_gate_up,
        paro_down,
        down_awq_scale,
        routed_out,
    };
    hipfire_runtime::dispatch::moe_family()
        .run_prefill(ctx, gpu, &moe_prefill_params)
        .map_err(HipError::from)?;

    Ok(())
}

/// Band view for `forward_prefill_chunk`. `None` (the default) means the
/// chunk processes the whole stack: embedding → all layers → final norm
/// + lm_head. `Some(b)` restricts the chunk to layers `b.layer_start..
/// b.layer_end`, skips the embedding when `!b.is_first_band` (input is
/// already in `pbs.x_batch` from a prior peer-copy), and skips the final
/// norm + lm_head when `!b.is_last_band` (output activation stays in
/// `pbs.x_batch` for the next band's peer-copy).
///
/// Counter offsets seed the running per-LA / per-KV / per-FA counters so
/// the band's first DeltaNet/FullAttn layer indexes the correct
/// `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot.
pub(crate) struct PrefillBandCtx<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub delta_layer_offset: usize,
    pub kv_layer_offset: usize,
    pub fa_layer_offset: usize,
    pub is_first_band: bool,
    pub is_last_band: bool,
    /// Per-device asym{2,3,4} givens replicas. When `Some`, the chunk's
    /// FA-layer batched KV writers use these instead of `kv_cache.givens_*`
    /// (which is `None` in multi-GPU mode by design — each device needs its
    /// own copy of the rotation tables).
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
}

#[allow(clippy::too_many_arguments)]
/// Debug localization hook (no-op unless `HIPFIRE_DUMP_HIDDEN` is set to a file
/// prefix). Appends the post-layer hidden row for the target absolute position
/// to `{HIPFIRE_DUMP_HIDDEN}.{tag}` as `u32 layer_idx` followed by `dim`
/// little-endian f32. The target absolute position is `HIPFIRE_DUMP_HIDDEN_POS`
/// (default 0); `abs_pos_of_row0` is the absolute sequence position of row 0 of
/// `x` (`start_pos` for the batched residual `pbs.x_batch`, `pos` for the
/// single-row per-token `s.x`). Used to localize the PARO batched-prefill
/// divergence by diffing `.batched` vs `.pertoken` per layer. Requires
/// `HIPFIRE_GRAPH=0` (does a synchronous D2H readback, which is illegal under
/// graph capture).
pub(crate) fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match std::env::var("HIPFIRE_DUMP_HIDDEN") {
        Ok(p) => p,
        Err(_) => return,
    };
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    // Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row of `x`
    // as raw [dim] f32 each (no per-row header) — so one prefill yields n_rows
    // real-activation samples for an offline rotation/quant study, AND a per-token
    // decode appends its single row each call → the file accumulates the whole
    // sequence. This mode IGNORES the single-position POS gate (which only makes
    // sense for the localize path below); it must fire at every position.
    if std::env::var("HIPFIRE_DUMP_HIDDEN_ALL").as_deref() == Ok("1") {
        // Two sub-modes:
        //  - default: restrict to one target layer (HIPFIRE_DUMP_HIDDEN_LAYER,
        //    default 0), one file `{prefix}.{tag}` (the kv-compression study path).
        //  - HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer
        //    files `{prefix}.{tag}.L{layer_idx}` — the Phase-A block-local
        //    recovery capture (residual-stream in/mid/out for all blocks).
        let all_layers = std::env::var("HIPFIRE_DUMP_HIDDEN_ALLLAYERS").as_deref() == Ok("1");
        let layer_path = if all_layers {
            format!("{path}.L{layer_idx}")
        } else {
            let want_layer: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_LAYER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if layer_idx != want_layer {
                return;
            }
            path.clone()
        };
        if gpu.hip.device_synchronize().is_err() {
            return;
        }
        let all = match gpu.download_f32(x) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&layer_path)
        {
            let take = (n_rows * dim).min(all.len());
            let bytes: Vec<u8> = all[..take].iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = f.write_all(&bytes);
        }
        return;
    }
    // Single-position localize path (PARO batched-vs-pertoken diff).
    let target: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows {
        return;
    }
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let off = row * dim;
    if off + dim > all.len() {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(&(layer_idx as u32).to_le_bytes());
        let mut bytes = Vec::with_capacity(dim * 4);
        for v in &all[off..off + dim] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

pub(crate) fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    positions_override: Option<&[usize]>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    force_q8_gdn_per_token: bool,
    // EP (Ship 6 substrate-EP prefill): per-MoE-layer routed partial. ONLY set
    // by the EP driver, which calls this with a SINGLE-layer band so the routed
    // combine of that one MoE layer lands in the zeroed partial (all-reduced by
    // the driver after the call). Always `None` for multi-layer bands (PP /
    // single-GPU full stack) — a shared partial across >1 MoE layer would be wrong.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    debug_assert!(
        routed_out.is_none()
            || band
                .map(|b| b.layer_end - b.layer_start <= 1)
                .unwrap_or(false),
        "forward_prefill_chunk: routed_out requires a single-layer band (EP driver invariant)",
    );

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let do_embed = band.map(|b| b.is_first_band).unwrap_or(true);
    let layer_start = band.map(|b| b.layer_start).unwrap_or(0);
    // `max_layer = Some(N)` early-exits at layer N (exclusive). pflash uses
    // this with N = score_layer_idx + 1: the drafter forward only needs to
    // populate the K cache through the scoring layer (the shallowest
    // FullAttention layer, typically layer 3 of 24 in Qwen3.5 hybrid),
    // since `pflash_score_q8_kv` reads exactly that layer's K. Layers
    // beyond it and the final norm + lm_head are wasted compute for
    // pflash. Saves ~80% of drafter forward time on hybrid drafters.
    let layer_end = band
        .map(|b| b.layer_end)
        .unwrap_or(config.n_layers)
        .min(max_layer.unwrap_or(usize::MAX));
    // Skip final norm + lm_head when the caller early-exits — they produce
    // logits the caller doesn't read, and require running through the full
    // layer stack anyway.
    let do_lm_head = band.map(|b| b.is_last_band).unwrap_or(true) && max_layer.is_none();
    let debug_stop_after_la_layer = std::env::var("HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage_layer = std::env::var("HIPFIRE_PREFILL_STOP_STAGE_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage = std::env::var("HIPFIRE_PREFILL_STOP_STAGE").ok();
    macro_rules! debug_stop_after {
        ($stage:literal, $layer_idx:expr) => {
            if debug_stop_stage_layer == Some($layer_idx)
                && debug_stop_stage.as_deref() == Some($stage)
            {
                return Ok(());
            }
        };
    }
    // Per-call-site `givens_cos_view` / `givens_sin_view` macros below
    // resolve to either the band-supplied per-device replica (multi-GPU
    // mode where `kv_cache.givens_*` is `None` by design) or the
    // kv_cache's own table (single-GPU). Held as macros, not top-level
    // bindings, so the immutable borrow on `kv_cache.givens_*` doesn't
    // outlive the kernel-call statement and conflict with later
    // mutable borrows of `kv_cache` (e.g. inside `run_fa_layer_body`).
    macro_rules! givens_cos_view {
        () => {
            band.and_then(|b| b.givens_cos)
                .or(kv_cache.givens_cos.as_ref())
        };
    }
    macro_rules! givens_sin_view {
        () => {
            band.and_then(|b| b.givens_sin)
                .or(kv_cache.givens_sin.as_ref())
        };
    }

    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    if do_embed
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0 | EmbeddingFormat::F32
        )
    {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup_f32_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            _ => unreachable!(),
        }
    } else if do_embed {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                _ => panic!("unsupported embedding format"),
            }
            gpu.memcpy_dtod_at_auto(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // ── 1a. Apply MaskEmbedOverride (MTP probe hook) ─────────────────────
    //
    // Overwrite a single batch slot's embedding row in `pbs.x_batch` after
    // the embedding-lookup kernel populated it but BEFORE the layer loop
    // (or any subsequent kernel) reads it. The Qualcomm MTP probe uses this
    // to replace the embedding-table value at a "mask token" position with
    // a prompt-mean vector. Default callers pass `None` → zero overhead.
    //
    // Multi-GPU band-mode: skip on non-first bands; pbs.x_batch already
    // holds the peer-copied activation from the previous band, so an
    // override applied at band 0 has already propagated through the layer
    // stack on that device — re-applying here would clobber the partial
    // forward state.
    if do_embed {
        if let Some(ovr) = mask_override {
            assert!(
                ovr.slot < n,
                "MaskEmbedOverride.slot ({}) must be < n ({})",
                ovr.slot,
                n,
            );
            assert_eq!(
                ovr.embed.len(),
                dim,
                "MaskEmbedOverride.embed.len() ({}) must equal config.dim ({})",
                ovr.embed.len(),
                dim,
            );
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
            let offset = ovr.slot * dim_row_bytes;
            gpu.hip
                .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
        }
    }

    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). Default callers
    // use flat linear `start_pos .. start_pos + n`; the dense server-prefill
    // session-batch worker can pass explicit per-row positions for independent
    // sessions. Siblings in DDTree mode get DISTINCT slots via the default
    // linear path — no write race — and the stored K carries a RoPE angle that
    // matches the physical slot, which keeps subsequent cycles' attention
    // reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    if !pre_uploaded {
        let positions_host: Vec<i32> = if let Some(positions) = positions_override {
            assert_eq!(
                positions.len(),
                n,
                "positions_override length {} must equal tokens.len() {}",
                positions.len(),
                n,
            );
            positions.iter().map(|&p| p as i32).collect()
        } else {
            (0..n).map(|i| (start_pos + i) as i32).collect()
        };
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    // Decide whether the FA layers can take the batched path. Requires
    // (a) all FA weights to be MQ4G256 or HFQ4G256 (the batched gemm_qkv
    // + wo GEMMs are dtype-agnostic; the rmsnorm+rotate / silu_mul kernels
    // differ by dtype and we branch on that at each layer) and (b) a Q8_0
    // or givens KV cache. If the check fails, FA layers fall back to
    // per-token gather/scatter via run_fa_layer_body.
    let fa_arch = gpu.arch.as_str();
    // Q8 WMMA gate: the fused Q8 WMMA family (gemm_qkv/qkvza/gate_up/residual
    // _q8_0_wmma) uses the gfx11 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
    // builtin; the sibling `*.gfx12.hip` kernels use the `_w32_gfx12` variant
    // (silicon-validated on R9700, 2026-05-14, 4/4 unit tests PASS). Each
    // call site below selects the right variant via an `arch.starts_with`
    // branch. On non-WMMA archs we keep the Tier 2 chunked-substrate path.
    let q8_wmma_arch = gpu.arch_caps.has_wmma();
    let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
    let fa_batched_ok = (!kv_cache.quantized
        || kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_gate.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_up.gpu_dtype, fa_arch)
                    && is_batchable_la(l.w_down.gpu_dtype, fa_arch)
            }
            // MoE variant: attention weights must be MQ4-class (FFN is
            // checked separately by moe_ffn_batched_admissible in the eligibility gate).
            LayerWeights::FullAttnMoe(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch)
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch)
            }
            _ => true, // LA layers don't gate this check
        });
    // Under hipGraph capture, scalar kernargs get BAKED into the kernarg blob
    // at capture time. `max_ctx_len = start_pos + n` grows per cycle, so the
    // captured value would be stale on replay — the attention kernel would
    // allocate too-small LDS for `scores[]` and over-read. Bake the physical
    // cap instead (LDS sized for the worst case). The kernel still iterates
    // over the actual `positions[b] + 1` per-row seq_len from a device buffer,
    // so correctness is preserved; only the LDS allocation is over-provisioned.
    let max_ctx_len = if gpu.capture_mode {
        kv_cache.physical_cap
    } else if let Some(positions) = positions_override {
        positions.iter().copied().max().unwrap_or(start_pos) + 1
    } else {
        start_pos + n
    };
    let position_at_row = |row: usize| -> usize {
        positions_override
            .map(|p| p[row])
            .unwrap_or(start_pos + row)
    };

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    // Multi-GPU band-mode: counters seed from the band's running offsets so
    // the band's first DeltaNet/FullAttn layer reads the correct
    // `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot. Single-GPU
    // (band==None) seeds zeros — original behavior.
    let mut delta_layer_idx = band.map(|b| b.delta_layer_offset).unwrap_or(0);
    let mut kv_layer_idx = band.map(|b| b.kv_layer_offset).unwrap_or(0);
    // Path B: per-FA-layer counter, drives the index into
    // tree_verify.pre_rope_k_capture[]. Increments alongside each
    // FullAttention layer iteration regardless of MoE/non-MoE variant.
    let mut fa_layer_idx = band.map(|b| b.fa_layer_offset).unwrap_or(0);
    let use_q8_gdn_per_token =
        force_q8_gdn_per_token || (gdn_tape.is_some() && q8_gdn_verify_per_token_enabled());
    let q8_gdn_serial_frame_base = if use_q8_gdn_per_token
        && q8_gdn_verify_serial_frames_enabled()
        && gdn_tape.is_some()
        && tree_verify.is_none()
        && band.is_none()
    {
        Some(gpu.debug_gdn_requant_frame())
    } else {
        None
    };
    let q8_gdn_serial_frame_layers = config
        .layer_types
        .iter()
        .filter(|&&lt| lt == LayerType::LinearAttention)
        .count();
    if let Some(frame_base) = q8_gdn_serial_frame_base {
        if let Some(tape) = gdn_tape.as_deref_mut() {
            tape.q8_requant_frame_base = Some(frame_base);
            tape.q8_requant_frame_layers = q8_gdn_serial_frame_layers;
        }
    }
    let ctx = DispatchCtx::new(gpu); // hoisted — arch-constant, safe to reuse per-layer

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
                // activation to match its pre-rotated weights; HFQ4 uses
                // plain rmsnormed activations. The GEMM kernels themselves
                // are dtype-agnostic — they just consume whatever [N × K]
                // activation buffer we point them at.
                // GAP NOTE: this matcher (and the 7 sibling dense LA/FA
                // matchers in this file) wires MQ3G256Lloyd through the
                // gemm_*_mq3g256_lloyd_wmma family. MQ2G256Lloyd remains
                // unwired — to add it, update is_batchable_la, ALL 8 is_mq*
                // matchers, AND add a Lloyd-MQ2-specific GEMM dispatch arm
                // together (the all-together corruption-prevention rule from
                // docs/plans/mq-lloyd-batched-prefill-followup.md). MQ4-Lloyd
                // is wired in a separate PR (issue #182).
                let is_mq = matches!(
                    layer.wqkv.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        // Opus W4A4: weights FWHT-rotated offline → x must be
                        // FWHT(+AWQ)-rotated by the shared mq rotate path before
                        // the int4 activation quantize (decode parity:
                        // rotate_x_mq[_awq] → quantize_act_oq4).
                        | DType::Oq4G256
                );
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_mq3 = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256);
                let is_mq3_lloyd = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256Lloyd);
                let is_fp4 = matches!(layer.wqkv.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let is_oq4 = matches!(layer.wqkv.gpu_dtype, DType::Oq4G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                let is_f32 = matches!(layer.wqkv.gpu_dtype, DType::F32);
                let is_f16 = matches!(layer.wqkv.gpu_dtype, DType::F16 | DType::BF16);

                // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
                // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
                // we reuse x_rot_batch as the "normed, unrotated" output
                // so the subsequent GEMM can read it the same way.
                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
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
                }

                // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
                if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
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
                } else if is_q8 && q8_wmma_arch {
                    // `is_q8` only inspects `wqkv` (the routing anchor). The fused
                    // kernel assumes ALL four weights share the Q8_0 stride; a
                    // mixed-dtype layer would silently re-introduce the Tier-1
                    // kernel-vs-stride corruption mode.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
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
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha) → GemmFamily::run_key with the
                    // GemmQ8_0BatchedChunked dispatcher-entry key → identical
                    // gpu.gemm_q8_0_batched_chunked method, byte-for-byte.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if is_f32 {
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::F32)
                            && matches!(layer.w_beta.gpu_dtype, DType::F32)
                            && matches!(layer.w_alpha.gpu_dtype, DType::F32),
                        "LA qkvza F32 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.wqkv.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.wz.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_beta.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if is_f16 {
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::F16 | DType::BF16)
                            && matches!(layer.w_beta.gpu_dtype, DType::F16 | DType::BF16)
                            && matches!(layer.w_alpha.gpu_dtype, DType::F16 | DType::BF16),
                        "LA qkvza F16/BF16 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wqkv.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_qkv_batch,
                            layer.wqkv.m,
                            layer.wqkv.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wz.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_z_batch,
                            layer.wz.m,
                            layer.wz.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_beta.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_beta_batch,
                            layer.w_beta.m,
                            layer.w_beta.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_alpha.buf,
                            &pbs.x_rot_batch,
                            &pbs.dn_alpha_batch,
                            layer.w_alpha.m,
                            layer.w_alpha.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_qkvza_f16_xf32_batched(
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
                } else if is_mq3_lloyd {
                    // 112 B/group Lloyd-MQ3 stride; X is already FWHT-rotated.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaMq3G256Lloyd,
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
                } else if is_mq3 {
                    // 104 B/group HFQ3-stride; X is already FWHT-rotated by
                    // fused_rmsnorm_rotate_mq_batched above. The FusedQkvzaHfq3G256
                    // run-arm replicates the call-site WMMA-vs-base arch split
                    // internally (gemm_qkvza_hfq3g256_wmma on has_wmma() else the
                    // base cross-arch ladder), so the same kernel runs.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq3G256,
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
                } else if is_fp4 {
                    // HFP4G32: 17-B blocks (vs HFQ4's 136-B groups), per-row 16-B header.
                    // MFP4G32: same storage as HFP4 + offline-FWHT weights; X is already
                    // rotated above when is_mq, so this branch handles both unrotated
                    // (HFP4) and post-rotation (MFP4) activations identically.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfp4G32,
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
                } else if is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (is_mq).
                    // The FusedQkvzaOq4G256 run-arm int4-quantizes it once then
                    // runs the batched grouped-WMMA fused kernel.
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaOq4G256,
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
                } else if gdn_tape.is_some() {
                    gpu.gemm_qkvza_hfq4g256_exact(
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
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
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

                if let Some(tape) = gdn_tape.as_ref() {
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_x = tape_offset * x_in_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_x = n * x_in_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        off_x,
                        &pbs.x_rot_batch.buf,
                        0,
                        copy_x,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }

                // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;

                // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
                // for this layer into the per-layer tape slots. The next LA
                // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
                // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
                // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }

                // Tree-aware dispatch gate: when the caller provides
                // parent_indices (Phase 3b+ of Task #101), swap the linear
                // conv1d + GDN for tree-walking variants that eliminate
                // sibling-subtree state cross-contamination. The tree
                // kernels are READ-ONLY on dn_state (don't advance it) —
                // caller runs linear replay on the accepted spine
                // post-acceptance to commit the trajectory.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    let off_q_raw = tape_offset * q_raw_row_bytes;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_q_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_k_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_v_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }

                // Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave
                // when n_key_heads < n_v_heads. One launch instead of two —
                // ~200µs saved per LA layer × ~30 LA layers ≈ 6ms per prefill
                // on A3B (R9700/gfx1201).
                //
                // The fused kernel reads q_raw/k_raw (unchanged on exit), so
                // the conv1d output is preserved if downstream readers need it
                // (no current consumer reads _raw after this).
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
                    // n_key_heads == n_v_heads → no replication; keep the
                    // original sequence (norm in place, then memcpy).
                    gpu.fused_qk_l2_norm_scale_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                        n,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let q_row_bytes = tape.v_dim * 4;
                    let off_q = tape_offset * q_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_q_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_k_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                }

                // Gated Delta Net — tree variant reads per-token S from
                // s_tape[parent] (or pre-block s_q8_init at root); linear
                // variant advances dn_state.s_matrices in place.
                if let Some(parents) = tree_parents {
                    if matches!(dn_state.quant, StateQuant::FP32) {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "FP32-state batched prefill does not support tree DeltaNet replay yet",
                        ));
                    }
                    let tape_q8 = pbs.dn_s_tape_q8.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch (check PrefillBatchScratch::new)");
                    let tape_sc = pbs.dn_s_tape_scales.as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch (check PrefillBatchScratch::new)");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8,
                        tape_sc,
                        parents,
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if matches!(dn_state.quant, StateQuant::FP32) {
                    gpu.gated_delta_net_f32_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if use_q8_gdn_per_token {
                    for step in 0..n {
                        if let Some(frame_base) = q8_gdn_serial_frame_base {
                            gpu.debug_set_gdn_requant_frame(frame_base.wrapping_add(
                                (step * q8_gdn_serial_frame_layers + delta_layer_idx) as u32,
                            ));
                        }
                        let q = pbs.dn_q_batch.sub_offset(step * v_dim, v_dim);
                        let k = pbs.dn_k_batch.sub_offset(step * v_dim, v_dim);
                        let v = pbs.dn_v_batch.sub_offset(step * v_dim, v_dim);
                        let alpha = pbs.dn_alpha_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let beta = pbs.dn_beta_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let out = pbs.dn_attn_out_batch.sub_offset(step * v_dim, v_dim);
                        gpu.gated_delta_net_q8(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?;
                    }
                    if let Some(frame_base) = q8_gdn_serial_frame_base {
                        gpu.debug_set_gdn_requant_frame(
                            frame_base.wrapping_add((n * q8_gdn_serial_frame_layers) as u32),
                        );
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_attn_out_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                    // EXPERIMENT (not #417): mirror the state-quant dispatch the
                    // decode siblings already do (forward_scratch_layers:13194),
                    // so the captured/eager batched prefill honours FP32/Q4 state
                    // instead of forcing the Q8 kernel onto non-Q8 buffers.
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                }

                // Batched gated output norm.
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

                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_normed_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched wo + residual.
                //
                // For MQ weights, the decode path's weight_gemv_residual
                // internally FWHT-rotates dn_normed into mq_x_rot before
                // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
                // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
                // For HFQ weights no rotation is needed — the activation
                // feeds gemm_hfq{4,6}g256_residual directly.
                let wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
                let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
                let wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
                let wo_input = if wo_is_mq {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    &pbs.dn_normed_batch
                };
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        off_v,
                        &wo_input.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_oq4 {
                    // Opus W4A4: wo_input is FWHT(+AWQ)-rotated above (wo_is_mq).
                    // No fused oq4 residual kernel → grouped-WMMA GEMM into scratch
                    // + add into the residual stream (pbs.x_batch).
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_q8 {
                    // Tier 2 fallback (non-WMMA archs): GEMM into x_rot_batch as
                    // scratch (safe — next consumer is the FFN rmsnorm), then
                    // add into residual.
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if wo_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.wo.buf,
                            wo_input,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.wo.buf,
                            wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_hfq4g256_residual_exact(
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // FFN: rmsnorm (+ rotate for MQ).
                let ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
                let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                let ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
                let ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
                if ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_rot_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched gate+up projection.
                // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
                // (batched-prefill gate+up variant) via run_fused_gate_up_key.
                // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
                // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
                // split is folded into the FusedGateUpHfq3G256 run-arm, which
                // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
                if ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if ffn_is_f32 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F32),
                        "LA FFN F32 dispatch requires both w_gate and w_up to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.w_gate.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if ffn_is_f16 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                        "LA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_gate.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            layer.w_gate.m,
                            layer.w_gate.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.up_batch,
                            layer.w_up.m,
                            layer.w_up.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_gate_up_f16_xf32_batched(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            &pbs.up_batch,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                            n,
                        )?;
                    }
                } else if ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if ffn_is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (ffn_is_mq).
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_gate_up_hfq4g256_exact(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    let off_ffn = tape_offset * ffn_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_gate_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.gate_ffn_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_up_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.up_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                }

                // SwiGLU activation feeding w_down. For MQ, we need the
                // output FWHT-rotated so it matches the pre-rotated w_down
                // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
                // is purely element-wise and uses numel() as its length,
                // so a [N × hidden_dim] tensor processes all rows in one
                // launch with no batch offset needed.
                let w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
                let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                let w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
                let w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
                if w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    let off_ffn = tape_offset * ffn_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_input_bufs[delta_layer_idx].buf,
                        off_ffn,
                        &pbs.ffn_hidden_batch.buf,
                        0,
                        n * ffn_row_bytes,
                    )?;
                }

                // Batched w_down + residual.
                if w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_oq4 {
                    // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above
                    // (fused_silu_mul_rotate_mq, w_down_is_mq). grouped-WMMA GEMM
                    // into scratch + residual add into the hidden stream.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if w_down_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if gdn_tape.is_some() {
                    gpu.gemm_hfq4g256_residual_exact(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.layer_out_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = is_mq; // retained above for potential future use
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Fully batched FA layer. Mirrors the FA branch of
                // forward_scratch_layers kernel-for-kernel, but every
                // launch covers all N tokens at once.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let _q_dim = config.n_heads * config.head_dim;
                let qkv_is_mq = matches!(
                    layer.wq.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
                let qkv_is_mq3_lloyd = matches!(layer.wq.gpu_dtype, DType::MQ3G256Lloyd);
                let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let qkv_is_oq4 = matches!(layer.wq.gpu_dtype, DType::Oq4G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                let qkv_is_f32 = matches!(layer.wq.gpu_dtype, DType::F32);
                let qkv_is_f16 = matches!(layer.wq.gpu_dtype, DType::F16 | DType::BF16);
                // Fused QKV kernels require all three weights to share a
                // dtype — they treat wq/wk/wv as same-stride byte arrays.
                // When kmap mode 2 promotes only `v_proj` (issue #249), the
                // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
                // and produces silent NaN. Gate the fused kernels here.
                //
                // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
                // dispatches a Q8-stride kernel per weight, so it needs the
                // same gate when wk/wv aren't Q8.
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;
                let fa_bridge_tape_active = gdn_tape.as_ref().is_some_and(|tape| {
                    delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                });
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_input_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
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
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_x_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_rot_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 2. Batched 3-way QKV projection (wq+wk+wv).
                if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
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
                } else if qkv_is_mq3_lloyd && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvMq3G256Lloyd,
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
                } else if qkv_is_mq3 && qkv_same_dtype {
                    // X is already FWHT-rotated by fused_rmsnorm_rotate_mq_batched
                    // above; call the bare HFQ3 GEMM (no second rotation). The
                    // FusedQkvHfq3G256 run-arm replicates the call-site WMMA-vs-base
                    // arch split internally (gemm_qkv_hfq3g256_wmma on has_wmma()
                    // else the base cross-arch ladder), so the same kernel runs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq3G256,
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
                } else if qkv_is_fp4 && qkv_same_dtype {
                    // HFP4G32 / MFP4G32 FP4 batched WMMA. X is already
                    // rotated above for MFP4 (is_mq path) — same kernel
                    // covers both unrotated HFP4 and rotated MFP4 inputs.
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfp4G32,
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
                } else if qkv_is_oq4 && qkv_same_dtype {
                    // OQ4+ batched prefill FA QKV: int8-WMMA MMQ (n>=64) quantizing
                    // the shared FWHT(+AWQ)-rotated activation to q8_1 ONCE across
                    // q/k/v; gemm_oq4_qkv_mmq falls back to the f16 grouped path for
                    // tiny batches internally via gemm_oq4_grouped_act_batched.
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Oq4G256)
                            && matches!(layer.wv.gpu_dtype, DType::Oq4G256),
                        "FA qkv Oq4 dispatch requires all of wq/wk/wv to be Oq4G256",
                    );
                    if n >= 64 {
                        gpu.gemm_oq4_qkv_mmq(
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
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wq.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            layer.wq.m,
                            layer.wq.k,
                            n,
                        )?;
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wk.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_k_batch,
                            layer.wk.m,
                            layer.wk.k,
                            n,
                        )?;
                        gpu.gemm_oq4_grouped_act_batched(
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_v_batch,
                            layer.wv.m,
                            layer.wv.k,
                            n,
                        )?;
                    }
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
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
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_f32 && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::F32)
                            && matches!(layer.wv.gpu_dtype, DType::F32),
                        "FA qkv F32 dispatch requires all of wq/wk/wv to be F32",
                    );
                } else if qkv_is_f16 && qkv_same_dtype {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wq.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            layer.wq.m,
                            layer.wq.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wk.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_k_batch,
                            layer.wk.m,
                            layer.wk.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_v_batch,
                            layer.wv.m,
                            layer.wv.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_qkvza_f16_xf32_batched(
                            &layer.wq.buf,
                            &layer.wk.buf,
                            &layer.wv.buf,
                            &layer.wv.buf,
                            &pbs.x_rot_batch,
                            &pbs.fa_q_full_batch,
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.dn_alpha_batch,
                            layer.wq.m,
                            layer.wk.m,
                            layer.wv.m,
                            0,
                            layer.wq.k,
                            n,
                        )?;
                    }
                } else if qkv_same_dtype {
                    if fa_bridge_tape_active {
                        gpu.gemm_qkv_hfq4g256_exact(
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
                        gpu.gemm_qkv_hfq4g256(
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
                } else {
                    // Mixed-format fallback (issue #249): wq/wk/wv don't all
                    // share a dtype. Dispatch each weight to its own
                    // single-weight batched GEMM, dropping the fused-kernel
                    // launch-overhead optimization for correctness.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_full_row_bytes = tape.fa_q_full_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_full_bufs[delta_layer_idx].buf,
                            tape_offset * q_full_row_bytes,
                            &pbs.fa_q_full_batch.buf,
                            0,
                            n * q_full_row_bytes,
                        )?;
                    }
                }

                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    n,
                )?;

                // 4. Per-head Q/K rmsnorm. rmsnorm_batched uses batch =
                // number of "rows" of head_dim. For [N × n_heads × head_dim]
                // that's batch = N * n_heads.
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_norm_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_q_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                    }
                }
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;

                if hipfire_runtime::triattn::tap_enabled() {
                    // Try GPU path first: dispatches a reduce kernel on the
                    // device-resident Q tensor, zero PCIe transfer. Only
                    // succeeds when install_tap_gpu() was used. Falls through
                    // to CPU path otherwise.
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }

                // Path B pre-RoPE K capture (slow-path-kill, WIP).
                // The next line mutates pbs.fa_k_batch in place — capture
                // BEFORE so the slow path has the unrotated K available
                // and can apply RoPE for the COMMITTED slot phases instead
                // of these linearization-slot phases. Capture is None
                // unless the env gate + the per-FA-layer scratch are both
                // wired through TreeVerifyCtx.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        // Use _auto so the memcpy is recorded onto the
                        // active stream when one exists (matches the
                        // existing GdnTape capture pattern at line ~3193).
                        // Plain gpu.hip.memcpy_dtod_at runs on the null
                        // stream and sync-blocks pending async kernels,
                        // changing kernel-launch order in ways that
                        // perturb DDTree's ksplit-atomic nondeterminism
                        // — output diverges even though no data is
                        // actually changed.
                        gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
                    }
                }

                // 5. Batched partial-interleaved RoPE (per-row positions).
                // pbs.positions stays physical for the KV write below; the
                // offset rotates new Q/K at absolute phase after compaction.
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
                    kv_cache.compact_offset as i32,
                )?;
                // KV-compression study capture: post-RoPE FA Q/K/V for a target FA
                // layer (HIPFIRE_DUMP_HIDDEN_ALL=1 + HIPFIRE_DUMP_HIDDEN_LAYER).
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_q_batch,
                    n,
                    start_pos,
                    config.n_heads * config.head_dim,
                    layer_idx,
                    "faq",
                );
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_k_batch,
                    n,
                    start_pos,
                    config.n_kv_heads * config.head_dim,
                    layer_idx,
                    "fak",
                );
                dump_hidden_localize(
                    gpu,
                    &pbs.fa_v_batch,
                    n,
                    start_pos,
                    config.n_kv_heads * config.head_dim,
                    layer_idx,
                    "fav",
                );
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        let kv_row_bytes = tape.fa_kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_q_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_k_bufs[delta_layer_idx].buf,
                            tape_offset * kv_row_bytes,
                            &pbs.fa_k_batch.buf,
                            0,
                            n * kv_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_v_bufs[delta_layer_idx].buf,
                            tape_offset * kv_row_bytes,
                            &pbs.fa_v_batch.buf,
                            0,
                            n * kv_row_bytes,
                        )?;
                    }
                }

                let use_kld_direct_f16kv_attention = kld_direct_f16kv_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                );
                let use_kld_fp32_gqa4_attention = kld_fp32_gqa4_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                    n,
                );

                // 6. Batched KV cache writes (per-row positions).
                if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
                    assert!(
                        tree_verify.is_none(),
                        "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
                    );
                    // Diagnostic: defer KV writes to the row-serial attention
                    // loop below so write/read ordering matches serial decode.
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                } else if !use_kld_direct_f16kv_attention && !use_kld_fp32_gqa4_attention {
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                }

                // 7. Batched causal attention (or tree-attention if tree_verify is set).
                // asym{4,3,2}: batched flash (K rotated-quantized + V Q8 in normal space).
                // Q8: batched kernel unless ctx > 15K (LDS overflow), then per-position flash.
                //
                // Tree-verify mode: `block_start = start_pos`, `block_cols = n`.
                // The bias buffer is `[n × n]`; each query row applies its
                // corresponding bias row to in-block keys. Long-context Q8
                // tiled fallback isn't supported in tree mode (we caught
                // that as an assert above — tree blocks are small).
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                // 6–7. Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if use_kld_direct_f16kv_attention {
                    gpu.attention_dflash_wmma_causal_f32(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        n,
                        n,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else if use_kld_fp32_gqa4_attention {
                    gpu.attention_f32_batched_gqa4(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                        n,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
                    assert!(
                        tree_verify.is_none(),
                        "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
                    );
                    let q_dim = config.n_heads * config.head_dim;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let k_b = pbs.fa_k_batch.sub_offset(b * kv_dim, kv_dim);
                            let v_b = pbs.fa_v_batch.sub_offset(b * kv_dim, kv_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.kv_cache_write_q8_0(
                                &kv_cache.k_gpu[layer_idx],
                                &k_b,
                                &pos_buf_tmp,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.kv_cache_write_q8_0(
                                &kv_cache.v_gpu[layer_idx],
                                &v_b,
                                &pos_buf_tmp,
                                config.n_kv_heads,
                                config.head_dim,
                            )?;
                            gpu.attention_q8_0_kv(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                pos_b + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len,
                        LDS_CTX_LIMIT,
                    );
                    // Per-position flash Q8 attention for long-context prefill.
                    //
                    // `pbs.positions` is raw i32 bits in an F32 slot
                    // (slot-cosmetic, see PrefillBatchScratch::new).
                    // `download_f32` would reinterpret those bytes as floats —
                    // i32 15000 = 0x3A98 round-trips through f32 as ~1e-3
                    // subnormal, which casts to 0. Reconstruct from the
                    // host-side row position directly.
                    let q_dim = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let seq_len_b = pos_b + 1;
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.attention_flash_q8_0(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                seq_len_b,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && q8_fa_attention_scalar_loop_enabled() {
                    let q_dim = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                            let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                            gpu.attention_q8_0_kv(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                pos_b + 1,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 && q8_fa_attention_row_loop_enabled() {
                    let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                        None
                    } else {
                        tree_bias
                    };
                    let q_dim = config.n_heads * config.head_dim;
                    for b in 0..n {
                        let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                        let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                        let pos_b = pbs.positions.sub_offset(b, 1);
                        let bias_b =
                            q8_tree_bias.map(|bias| bias.sub_offset(b * block_cols, block_cols));
                        gpu.attention_q8_0_kv_batched_masked(
                            &q_b,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &out_b,
                            &pos_b,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            1,
                            bias_b.as_ref(),
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                        None
                    } else {
                        tree_bias
                    };
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        q8_tree_bias,
                        block_start,
                        block_cols,
                    )?;
                } else {
                    gpu.attention_f32_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_raw_bufs[delta_layer_idx].buf,
                            tape_offset * q_row_bytes,
                            &pbs.fa_attn_out_batch.buf,
                            0,
                            n * q_row_bytes,
                        )?;
                    }
                }
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    quant_asym4: kv_cache.quant_asym4,
                    quant_asym3: kv_cache.quant_asym3,
                    quant_asym2: kv_cache.quant_asym2,
                    quant_q8: kv_cache.quant_q8,
                    quant_fwht: kv_cache.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: 0,
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.capture_mode,
                    batch_size: n,
                    is_tree,
                    is_boundary: false,
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;

                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_out_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.fa_attn_out_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
                // Same MQ rotation requirement as the LA wo path.
                let fa_wo_is_mq = matches!(
                    layer.wo.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
                let fa_wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
                let fa_wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
                let fa_wo_input = if fa_wo_is_mq {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    &pbs.fa_attn_out_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_oq4 {
                    // Opus W4A4: fa_wo_input is FWHT(+AWQ)-rotated above.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.wo.buf,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.wo.buf,
                        fa_wo_input,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.wo.buf,
                            fa_wo_input,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.wo.buf,
                            fa_wo_input,
                            &pbs.x_batch,
                            layer.wo.m,
                            layer.wo.k,
                            n,
                        )?;
                    }
                } else if fa_wo_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_wo_residual_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
                // (+ rotate for MQ), w_down residual.
                let fa_ffn_is_mq = matches!(
                    layer.w_gate.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_ffn_is_6bit =
                    matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
                let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_ffn_is_fp4 =
                    matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
                let fa_ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
                let fa_ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
                let fa_ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
                if fa_ffn_is_mq {
                    // AWQ-aware: next linear is w_gate (FA-FFN, gate/up share input).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &layer.w_gate,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.ffn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                // #397 Ship 5.2 slice 2: FA-FFN fused gate+up → FusedQkvFamily
                // (batched-prefill gate+up variant), mirroring the LA-FFN block
                // above. Q8-non-WMMA stays as two plain GEMMs; HFQ3 WMMA-vs-base
                // is folded into the FusedGateUpHfq3G256 run-arm.
                if fa_ffn_is_6bit {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_oq4 {
                    // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (fa_ffn_is_mq).
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 && q8_wmma_arch {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                        "FA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
                    );
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_q8 {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_gate.buf,
                        layer.w_gate.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_up.buf,
                        layer.w_up.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if fa_ffn_is_f32 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F32),
                        "FA FFN F32 dispatch requires both w_gate and w_up to be F32",
                    );
                    gpu.gemm_f32_register_tiled(
                        &layer.w_gate.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    gpu.gemm_f32_register_tiled(
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.up_batch,
                        layer.w_up.m,
                        layer.w_up.k,
                        n,
                    )?;
                } else if fa_ffn_is_f16 {
                    debug_assert!(
                        matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                        "FA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
                    );
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_gate.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            layer.w_gate.m,
                            layer.w_gate.k,
                            n,
                        )?;
                        gemm_fp16_or_bf16_x_f32_wmma(
                            gpu,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.up_batch,
                            layer.w_up.m,
                            layer.w_up.k,
                            n,
                        )?;
                    } else {
                        gpu.fused_gate_up_f16_xf32_batched(
                            &layer.w_gate.buf,
                            &layer.w_up.buf,
                            &pbs.x_rot_batch,
                            &pbs.gate_ffn_batch,
                            &pbs.up_batch,
                            layer.w_gate.m,
                            layer.w_up.m,
                            layer.w_gate.k,
                            n,
                        )?;
                    }
                } else if fa_ffn_is_mq3_lloyd {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_mq3 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else if fa_ffn_is_fp4 {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                } else {
                    run_fused_gate_up_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &pbs.x_rot_batch,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                        n,
                    )?;
                }
                let fa_w_down_is_mq = matches!(
                    layer.w_down.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::MQ3G256
                        | DType::MQ3G256Lloyd
                        | DType::MFP4G32
                        | DType::Oq4G256
                );
                let fa_w_down_is_6bit =
                    matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let fa_w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
                let fa_w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
                let fa_w_down_is_fp4 =
                    matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
                let fa_w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
                let fa_w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
                let fa_w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
                let fa_w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
                if fa_w_down_is_mq {
                    // F2: AWQ-aware silu_mul+rotate for FullAttention w_down input.
                    fused_silu_mul_rotate_mq_batched_for(
                        gpu,
                        &layer.w_down,
                        &pbs.gate_ffn_batch,
                        &pbs.up_batch,
                        &pbs.ffn_hidden_batch,
                        hidden_dim,
                        n,
                    )?;
                } else {
                    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                }
                if fa_w_down_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_oq4 {
                    // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above.
                    gpu.gemm_oq4_grouped_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &x_n,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_q8 {
                    let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &scratch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_w_down_is_f32 {
                    gemm_f32_residual_batched(
                        gpu,
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        &pbs.x_rot_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_f16 {
                    if f16_prefill_wmma {
                        gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
                            gpu,
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            &pbs.x_rot_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    } else {
                        gpu.gemv_f16_xf32_residual_batched(
                            &layer.w_down.buf,
                            &pbs.ffn_hidden_batch,
                            &pbs.x_batch,
                            layer.w_down.m,
                            layer.w_down.k,
                            n,
                        )?;
                    }
                } else if fa_w_down_is_mq3_lloyd {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_mq3 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else if fa_w_down_is_fp4 {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.w_down.buf,
                        layer.w_down.gpu_dtype,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_layer_out_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                // Silence unused warning if kv_dim ends up shadowed.
                let _ = kv_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // Per-token gather/scatter fallback for FA layers that don't
                // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
                for i in 0..n {
                    let pos = start_pos + i;
                    gpu.hip.memcpy_dtod_at(
                        &s.x.buf,
                        0,
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        dim_row_bytes,
                    )?;
                    let pos_i32 = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes())?;
                    run_fa_layer_body(
                        gpu,
                        weights,
                        config,
                        layer_idx,
                        kv_layer_idx,
                        pos,
                        kv_cache,
                        s,
                    )?;
                    gpu.hip.memcpy_dtod_at(
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        &s.x.buf,
                        0,
                        dim_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path. After
                // the per-token loop, pbs.x_batch has the full layer output
                // for all N tokens (last copy-back finishes each row).
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // Batched MoE LA layer. LA body is the same as DeltaNet
                // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
                // repeat_interleave + GDN + gated_norm + wo+residual);
                // only the FFN differs. Duplicated inline for now — can
                // be factored into a `prefill_la_body_batched` helper
                // when dense and MoE LA paths are proven byte-exact.
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wqkv.gpu_dtype` is restricted here to MQ4G256 /
                // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
                // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
                // attention weights are Q8 — engine quantizer keeps q/k/v/o
                // at Q8 alongside the Q8 router + shared_expert_gate).
                let is_mq = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
                // ParoQ4G128 (each with its own Givens rotation tables);
                // w_alpha/w_beta are F32 (no rotation, no quantization).
                // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
                // direct gemm_f32_batched for w_alpha and w_beta. Same shape
                // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
                // dn_alpha_batch, dn_beta_batch).
                let is_paro = matches!(layer.wqkv.gpu_dtype, DType::ParoQ4G128);
                let q8_wmma_arch = gpu.arch_caps.has_wmma();

                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if is_paro {
                    // PARO: need un-rotated x_norm available for per-weight
                    // Givens rotation. Write rmsnorm into x_norm_batch (the
                    // dedicated normalized buffer); x_rot_batch becomes the
                    // per-weight rotation scratch (overwritten per GEMM).
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
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
                }
                debug_stop_after!("attn_norm", layer_idx);
                if is_paro {
                    // PARO 4-way unfused dispatch. wqkv and wz are
                    // ParoQ4G128 with their own Givens rotation tables;
                    // w_alpha and w_beta are F32 with no rotation.
                    let paro_wqkv = layer.wqkv.paro.as_ref().unwrap_or_else(|| {
                        panic!(
                            "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — paro_load_wt() loader regression?"
                        )
                    });
                    let paro_wz = layer.wz.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
                    });
                    // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wqkv.pairs,
                        &paro_wqkv.theta,
                        &paro_wqkv.channel_scales,
                        n,
                        dim,
                        paro_wqkv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wz.pairs,
                        &paro_wz.theta,
                        &paro_wz.channel_scales,
                        n,
                        dim,
                        paro_wz.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                } else if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
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
                } else if is_q8 && q8_wmma_arch {
                    // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
                    // stride; mixed Q8/other layers within DNMoe are rejected
                    // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
                    // shared_expert + experts must be MQ4) and would otherwise
                    // re-introduce Tier-1 stride corruption.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
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
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
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
                debug_stop_after!("qkvza", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_x = tape_offset * x_in_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_x = n * x_in_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        off_x,
                        &pbs.x_rot_batch.buf,
                        0,
                        copy_x,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;
                debug_stop_after!("sigmoid_alpha", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                // Same tree-aware dispatch gate as dense LA branch above.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }
                debug_stop_after!("conv", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    let off_q_raw = tape_offset * q_raw_row_bytes;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_q_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_k_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_v_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    n,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        n,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }
                debug_stop_after!("qk_repeat", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_row_bytes = tape.v_dim * 4;
                    let off_q = tape_offset * q_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_q_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_k_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                }
                // DIAG: dump GDN inputs (batched, MoE branch)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * hd;
                    dump_hidden_localize(gpu, &pbs.dn_q_batch, n, start_pos, qk_dim, 0, "q_b");
                    dump_hidden_localize(gpu, &pbs.dn_k_batch, n, start_pos, qk_dim, 0, "k_b");
                    dump_hidden_localize(gpu, &pbs.dn_v_batch, n, start_pos, v_dim, 0, "v_b");
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_alpha_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "alpha_b",
                    );
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_beta_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "beta_b",
                    );
                }
                if let Some(parents) = tree_parents {
                    if matches!(dn_state.quant, StateQuant::FP32) {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "FP32-state batched prefill does not support tree DeltaNet replay yet",
                        ));
                    }
                    let tape_q8 = pbs
                        .dn_s_tape_q8
                        .as_ref()
                        .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                    let tape_sc = pbs
                        .dn_s_tape_scales
                        .as_ref()
                        .expect("tree-aware LA requires dn_s_tape_scales scratch");
                    gpu.gated_delta_net_q8_tree_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        tape_q8,
                        tape_sc,
                        parents,
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if use_q8_gdn_per_token {
                    for step in 0..n {
                        if let Some(frame_base) = q8_gdn_serial_frame_base {
                            gpu.debug_set_gdn_requant_frame(frame_base.wrapping_add(
                                (step * q8_gdn_serial_frame_layers + delta_layer_idx) as u32,
                            ));
                        }
                        let q = pbs.dn_q_batch.sub_offset(step * v_dim, v_dim);
                        let k = pbs.dn_k_batch.sub_offset(step * v_dim, v_dim);
                        let v = pbs.dn_v_batch.sub_offset(step * v_dim, v_dim);
                        let alpha = pbs.dn_alpha_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let beta = pbs.dn_beta_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let out = pbs.dn_attn_out_batch.sub_offset(step * v_dim, v_dim);
                        gpu.gated_delta_net_q8(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?;
                    }
                    if let Some(frame_base) = q8_gdn_serial_frame_base {
                        gpu.debug_set_gdn_requant_frame(
                            frame_base.wrapping_add((n * q8_gdn_serial_frame_layers) as u32),
                        );
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                }
                debug_stop_after!("gdn", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_attn_out_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    // DIAG: dump GDN attention output at layer 0
                    if layer_idx == 0 {
                        dump_hidden_localize(
                            gpu,
                            &pbs.dn_attn_out_batch,
                            n,
                            start_pos,
                            n_v_heads * config.linear_value_head_dim,
                            0,
                            "gdn_b",
                        );
                    }
                }
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
                debug_stop_after!("gated_norm", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_normed_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                // wo + residual. Q8 wo lands un-rotated (Q8 weights were
                // quantized against un-rotated activations); MQ4/MQ6 wo
                // require FWHT(awq_scale-adjusted) rotation. Mirrors the
                // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
                // branch is required for AWQ A3B where 4/40 LA layers
                // ship MQ6 wo and would otherwise corrupt the residual
                // stream when dispatched through the HFQ4 kernel against
                // 200 B/group MQ6-layout bytes.
                let dn_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let dn_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let dn_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let dn_wo_input = if dn_wo_is_q8 {
                    &pbs.dn_normed_batch
                } else if dn_wo_is_paro {
                    // PARO wo: rotate dn_normed by wo's own Givens tables
                    // into dn_normed_rot_batch. Same scratch layout as MQ4
                    // (since dn_normed_rot_batch is unused on the Q8 path).
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                };
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        off_v,
                        &dn_wo_input.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if dn_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 {
                    // Non-WMMA Q8: gemm into a scratch then add into x_batch.
                    // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
                    // path didn't run here) as the GEMM scratch.
                    let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if dn_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free at
                    // this point — used earlier for the QKVZA stage; not
                    // needed for the rest of this layer) as the scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }
                debug_stop_after!("wo", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
                // silu_mul + w_down) block. Takes pbs.x_batch as input AND
                // accumulates the FFN output residual back into it via the
                // batched indexed down kernel's atomicAdd path.
                if debug_stop_after_la_layer == Some(layer_idx) {
                    return Ok(());
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Batched MoE FA layer. FA body is the same as FullAttn
                // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
                // kv_write + attention + sigmoid_mul + wo+residual);
                // only the FFN differs. Duplicated inline — will be
                // consolidated with the dense FA batched body once the
                // MoE path is proven byte-exact.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wq.gpu_dtype` is restricted to MQ4G256 / HFQ4G256
                // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
                // the QKV dispatch is insufficient — the wo path below
                // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
                // wiring lives in a separate PR (see followup issue).
                let qkv_is_mq = matches!(layer.wq.gpu_dtype, DType::MQ4G256 | DType::MQ6G256);
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
                // (each with its own Givens rotation tables). The fused-QKV
                // kernels can't handle this — they assume one shared
                // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
                // per projection) matches the LA QKVZA Phase 1.5 pattern.
                let qkv_is_paro = matches!(layer.wq.gpu_dtype, DType::ParoQ4G128);
                // Fused QKV requires uniform dtype — see issue #249 for
                // the dense FA variant. Gate the same way here.
                let q8_wmma_arch = gpu.arch_caps.has_wmma();
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
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
                } else if qkv_is_paro {
                    // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
                    // is reused as the per-weight rotation scratch.
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
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
                }
                if qkv_is_paro {
                    // PARO 3-way unfused dispatch (wq, wk, wv each with own
                    // Givens rotation). Same shape outputs as the fused
                    // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
                    let paro_wq = layer.wq.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wk = layer.wk.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wv = layer.wv.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
                    });
                    // wq
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wq.pairs,
                        &paro_wq.theta,
                        &paro_wq.channel_scales,
                        n,
                        dim,
                        paro_wq.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    // wk
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wk.pairs,
                        &paro_wk.theta,
                        &paro_wk.channel_scales,
                        n,
                        dim,
                        paro_wk.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    // wv
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wv.pairs,
                        &paro_wv.theta,
                        &paro_wv.channel_scales,
                        n,
                        dim,
                        paro_wv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
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
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
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
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
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
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq4G256,
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
                    // Mixed-format fallback (issue #249). batched_gemm_single_weight
                    // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
                    // routes here.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    n,
                )?;
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
                if hipfire_runtime::triattn::tap_enabled() {
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }
                // Path B pre-RoPE K capture (MoE FA variant). See same
                // block in the FullAttn branch for rationale.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
                    }
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // pbs.positions stays physical for the KV write below; the
                // offset rotates new Q/K at absolute phase after compaction.
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
                    kv_cache.compact_offset as i32,
                )?;

                let use_kld_direct_f16kv_attention = kld_direct_f16kv_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                );
                let use_kld_fp32_gqa4_attention = kld_fp32_gqa4_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                    n,
                );

                if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                } else if !use_kld_direct_f16kv_attention && !use_kld_fp32_gqa4_attention {
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                }
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                // Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if use_kld_direct_f16kv_attention {
                    gpu.attention_dflash_wmma_causal_f32(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        n,
                        n,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else if use_kld_fp32_gqa4_attention {
                    gpu.attention_f32_batched_gqa4(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                        n,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len,
                        LDS_CTX_LIMIT,
                    );
                    // See dense FullAttn branch above for the i32-vs-f32 slot
                    // rationale; reconstruct positions from the host-side row
                    // position directly.
                    let q_dim_local = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let seq_len_b = pos_b + 1;
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim_local, q_dim_local);
                            let out_b = pbs
                                .fa_attn_out_batch
                                .sub_offset(b * q_dim_local, q_dim_local);
                            gpu.attention_flash_q8_0(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                seq_len_b,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 {
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                } else {
                    gpu.attention_f32_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                }
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                let plan = KvTierPlan::derive(KvTierInputs {
                    quant_asym4: kv_cache.quant_asym4,
                    quant_asym3: kv_cache.quant_asym3,
                    quant_asym2: kv_cache.quant_asym2,
                    quant_q8: kv_cache.quant_q8,
                    quant_fwht: kv_cache.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    v_mode_bits: 0,
                    pos: start_pos,
                    flash_mode: s.flash_mode as usize,
                    capture_mode: gpu.capture_mode,
                    batch_size: n,
                    is_tree,
                    is_boundary: false,
                })
                .map_err(|e| HipError::new(0, &e.to_string()))?;
                let io = AttnParams {
                    q: &pbs.fa_q_batch,
                    k: &pbs.fa_k_batch,
                    v: &pbs.fa_v_batch,
                    k_cache: &kv_cache.k_gpu[layer_idx],
                    v_cache: &kv_cache.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &s.pos_buf,
                    pos: start_pos,
                    positions: Some(&pbs.positions),
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    physical_cap: kv_cache.physical_cap,
                    batch_size: n,
                    max_ctx_len,
                    flash_partials: Some(&s.flash_partials),
                    givens_cos: kv_cache.givens_cos.as_ref(),
                    givens_sin: kv_cache.givens_sin.as_ref(),
                    tree_bias,
                    block_start,
                    block_cols,
                    output: &pbs.fa_attn_out_batch,
                };
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // wo + residual. Mirrors the dense FA wo dispatch at
                // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
                // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
                // MQ6 wo has its own branch: feeding MQ6 bytes to the MQ4
                // residual kernel would read 200 B/group data as 136 B/group
                // HFQ4 layout and catastrophically mis-stride.
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
                // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
                // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
                // scratch, then add into x_batch.
                let fa_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let fa_wo_input = if fa_wo_is_q8 {
                    &pbs.fa_attn_out_batch
                } else if fa_wo_is_paro {
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
                    // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
                    // didn't run here) as scratch.
                    let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free since
                    // QKVZA is done — the MoE FFN body below rewrites it
                    // as its first action) as the gemm output scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // Batched MoE FFN.
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = kv_dim;
                let _ = q_dim;
                kv_layer_idx += 1;
                fa_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
        dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
    }

    // ── 3. Final output norm + logits ───────────────────────────────────
    // Multi-GPU band-mode: skip when this is not the last band — the
    // running activation in `pbs.x_batch` is what the next band's
    // peer-copy reads. `weights.output_norm` and `weights.output` only
    // live on the last band's device anyway.
    if do_lm_head {
        // If the caller requested per-token hidden output (DFlash verify path),
        // run rmsnorm over all N rows into their buffer. Otherwise use the
        // legacy last-token-only path.
        if let Some((dst, offset_rows)) = per_token_hidden_out {
            let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &weights.output_norm,
                &dst_view,
                n,
                dim,
                config.norm_eps,
            )?;
            if prefill_should_emit_last_token_logits(true, needs_last_token_logits) {
                // Still populate s.logits with the last-token logits for
                // callers that rely on it (the legacy prefill post-condition).
                let last = n - 1;
                let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
                {
                    let wr = weights.output.dispatch_ref();
                    let step = Step::Gemv {
                        w: &wr,
                        input: GemvInput::Raw(&last_view),
                        out: &s.logits,
                    };
                    execute_steps(gpu, &ctx, &[step])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
            }
        } else {
            // Legacy path: only last-token logits.
            // Use _auto so the D→D copy routes through the active stream
            // during hipGraph capture (bare memcpy_dtod_at uses the legacy
            // null stream and breaks capture: HIP error 906).
            let last = n - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            {
                let wr = weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
    }

    Ok(())
}

/// Run a single FullAttn layer body on s.x at position `pos`. Extracted
/// for use from the batched prefill path's FA-layer fallback. Byte-exact
/// with the FA branch of forward_scratch_layers.
#[allow(clippy::too_many_arguments)]
fn run_fa_layer_body(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    _kv_layer_idx: usize,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let layer = match &weights.layers[layer_idx] {
        LayerWeights::FullAttn(l) => l,
        _ => unreachable!(),
    };

    // Fused rmsnorm + FWHT rotation for wq/wk/wv (MQ-family).
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.wq,
        &s.x,
        &layer.attn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Lever 1 — Fused rmsnorm + PARO per-group rotation for wq.
    // x_rot_paro is valid ONLY for wq (PARO rotation uses wq's pairs/theta/channel_scales);
    // wk and wv will run their own rotation via the standard weight_gemv path. The fused
    // kernel ALSO writes s.tmp (post-rmsnorm) so wk/wv get correct input. Saves 1 launch
    // per FA block (rmsnorm+wq rotate folded into one kernel). Default on; opt out via
    // HIPFIRE_PARO_FUSE_RMSNORM=0.
    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
        && layer.wq.gpu_dtype == DType::ParoQ4G128
        && layer.wq.k % 128 == 0
        && layer.wq.m % 8 == 0
    {
        fused_rmsnorm_rotate_for_paro(
            gpu,
            &layer.wq,
            &s.x,
            &layer.attn_norm,
            &s.tmp,
            &s.x_rot,
            config.norm_eps,
        )?
    } else {
        None
    };

    // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
    let dt = layer.wq.gpu_dtype;
    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
    let fused_fa3_f16 = config.attn_output_gate && fa3_same_dtype && dt == DType::F16;
    let fused_fa3_mq4 = fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_fa3_hfq6 = config.attn_output_gate
        && fa3_same_dtype
        && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_fa3_f16 {
        gpu.fused_qkvza_f16_xf32(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &layer.wq.buf,
            &s.tmp,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            &s.o,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            0,
            layer.wq.k,
        )?;
    } else if fused_fa3_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq4g256(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq3g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq4g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq6g256_dp4a(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else {
        // Lever 1 fast path: when fused_rmsnorm_rotate_for_paro produced x_rot_paro,
        // wq has its rotated x already — call the prerotated GEMV directly (saves the
        // standalone paro4g128t_rotate launch for wq). wk and wv MUST do their own
        // rotation since PARO pairs/theta differ per linear; they consume s.tmp
        // (post-rmsnorm) via the standard weight_gemv path.
        if let Some(xr_q) = x_rot_paro {
            gpu.gemv_paro4g128t_prerotated(
                &layer.wq.buf,
                xr_q,
                &s.fa_q_full,
                layer.wq.m,
                layer.wq.k,
            )?;
        } else {
            weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
        }
        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
    }

    qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    let npu_hnr_ok = if hipfire_runtime::triattn::tap_enabled() {
        false
    } else {
        try_npu_headnorm_rope(
            gpu,
            layer_idx,
            &s.fa_q,
            &s.fa_k,
            &layer.q_norm,
            &layer.k_norm,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            pos,
        )?
    };
    if !npu_hnr_ok {
        gpu.rmsnorm_batched(
            &s.fa_q,
            &layer.q_norm,
            &s.fa_q,
            config.n_heads,
            config.head_dim,
            config.norm_eps,
        )?;
        gpu.rmsnorm_batched(
            &s.fa_k,
            &layer.k_norm,
            &s.fa_k,
            config.n_kv_heads,
            config.head_dim,
            config.norm_eps,
        )?;
        if hipfire_runtime::triattn::tap_enabled() {
            // Try GPU path first (matches the batched FA tap at line ~3499 in
            // forward_prefill_batch). When the calibration tap is GPU-resident
            // (CalibrateGpu) we MUST dispatch the kernel here — falling
            // through to record_prerope_qk would either silently drop the
            // sample (pre-Phase-2) or panic (post-Phase-2).
            let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                gpu,
                layer_idx,
                &s.fa_q.buf,
                1,
                config.n_heads,
                config.head_dim,
            )?;
            if !gpu_handled {
                let n_q = config.n_heads * config.head_dim;
                let q_cpu = gpu.download_f32(&s.fa_q)?;
                if hipfire_runtime::triattn::tap_needs_k() {
                    let n_k = config.n_kv_heads * config.head_dim;
                    let k_cpu = gpu.download_f32(&s.fa_k)?;
                    hipfire_runtime::triattn::record_prerope_qk(
                        layer_idx,
                        &q_cpu[..n_q],
                        Some(&k_cpu[..n_k]),
                    );
                } else {
                    hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
                }
            }
        }
        // If TriAttention has compacted the cache, absolute RoPE phase diverges
        // from the physical cache index. Temporarily load the absolute position
        // into pos_buf for the rope call, then restore the physical position
        // for kv_cache_write + flash attention (which both want the write slot).
        if kv_cache.compact_offset > 0 {
            let abs = (pos + kv_cache.compact_offset) as i32;
            gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
        }
        gpu.rope_partial_interleaved_f32(
            &s.fa_q,
            &s.fa_k,
            &s.pos_buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
        )?;
    }
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
    }

    if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht4_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht4(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym4_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym4(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht3_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht3(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym3_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym3(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if kv_cache.quant_fwht {
            gpu.kv_cache_write_fwht2_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
                0,
            )?;
            gpu.attention_flash_fwht2(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
                0,
            )?;
        } else {
            gpu.kv_cache_write_asym2_fused(
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_k,
                &s.fa_v,
                &s.pos_buf,
                ct,
                st,
                config.n_kv_heads,
                config.head_dim,
            )?;
            gpu.attention_flash_asym2(
                &s.fa_q,
                &kv_cache.k_gpu[layer_idx],
                &kv_cache.v_gpu[layer_idx],
                &s.fa_attn_out,
                &s.pos_buf,
                ct,
                st,
                pos + 1,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                kv_cache.physical_cap,
                &s.flash_partials,
            )?;
        }
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(
            &kv_cache.k_gpu[layer_idx],
            &s.fa_k,
            &s.pos_buf,
            config.n_kv_heads,
            config.head_dim,
        )?;
        gpu.kv_cache_write_q8_0(
            &kv_cache.v_gpu[layer_idx],
            &s.fa_v,
            &s.pos_buf,
            config.n_kv_heads,
            config.head_dim,
        )?;
        gpu.attention_q8_0_kv(
            &s.fa_q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out,
            &s.pos_buf,
            pos + 1,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_cache.physical_cap,
        )?;
    } else {
        gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &s.fa_q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &s.fa_attn_out,
            &s.pos_buf,
            pos + 1,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_cache.physical_cap,
        )?;
    }

    qwen35_apply_fa_gate(gpu, config, &s.fa_attn_out, &s.fa_gate)?;
    qwen35_attention_wo_residual(
        gpu,
        config,
        layer_idx,
        &layer.wo,
        &s.fa_attn_out,
        &s.x,
        &s.o,
    )?;
    let ctx = DispatchCtx::new(gpu);
    kv_cache_attention_dispatch(&ctx, gpu, kv_cache, s, config, layer_idx, pos)?;

    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    {
        let wr = layer.wo.dispatch_ref();
        execute_steps(
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wr,
                input: GemvInput::Raw(&s.fa_attn_out),
                residual: &s.x,
                out: &s.x,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // FFN: fused rmsnorm + rotate for w_gate/w_up.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.w_gate,
        &s.x,
        &layer.ffn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Lever 1 — Fused rmsnorm + PARO per-group rotation for w_gate.
    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
        && layer.w_gate.gpu_dtype == DType::ParoQ4G128
        && layer.w_gate.k % 128 == 0
        && layer.w_gate.m % 8 == 0
    {
        fused_rmsnorm_rotate_for_paro(
            gpu,
            &layer.w_gate,
            &s.x,
            &layer.ffn_norm,
            &s.tmp,
            &s.x_rot,
            config.norm_eps,
        )?
    } else {
        None
    };
    let dt_g = layer.w_gate.gpu_dtype;
    let same_dtype = layer.w_up.gpu_dtype == dt_g;
    let fused_gu_mq4 = same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
    let fused_gu_f16 = same_dtype && dt_g == DType::F16;
    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_gu_hfq6 = same_dtype
        && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_gu_f16 {
        gpu.fused_gate_up_f16_xf32(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &s.tmp,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq4g256(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq3g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq4g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq6g256_dp4a(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else {
        if let Some(xr_first) = x_rot_paro {
            gpu.gemv_paro4g128t_prerotated(
                &layer.w_gate.buf,
                xr_first,
                &s.gate_ffn,
                layer.w_gate.m,
                layer.w_gate.k,
            )?;
        } else {
            weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
        }
        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
    }
    weight_gemv_swiglu_residual_bf16_probe(
        gpu,
        layer_idx,
        &layer.w_down,
        &layer.bf16_down_shadow,
        &s.gate_ffn,
        &s.up,
        &s.ffn_hidden,
        &s.x,
    )?;

    Ok(())
}
