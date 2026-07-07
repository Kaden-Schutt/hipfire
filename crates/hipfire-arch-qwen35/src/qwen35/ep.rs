// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 multi-GPU / expert-parallel (EP) forward paths: replicated N-rank
//! decode + prefill (`forward_ep`, `forward_prefill_batch_ep`), the sharded
//! multi-GPU layer loop (`forward_scratch_layers_multi`), and the multi-device
//! entry points (`forward_scratch_multi`, `forward_prefill_batch_multi*`).

use super::*;

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
///
/// Every rank holds **full replicated** weights / scratch / KV / DeltaNet
/// state EXCEPT the MoE routed experts, which were sharded per rank at load by
/// [`shard_moe_experts`]. Behaviorally this mirrors the single-GPU
/// [`forward_scratch`] → [`forward_scratch_layers_lowered`] pipeline (embed →
/// per-layer `LayerProgram` → final norm + lm_head), but runs each layer's
/// program through the EP executor ([`hipfire_runtime::ep::run_layer_program_ep`]):
/// the `Moe` super-op is all-reduce-EP'd across ranks (each rank computes only
/// its owned experts into a zeroed routed partial, the partials are
/// all-reduce-summed, then added into each rank's residual); every other
/// super-op runs **replicated** and stays bit-identical across ranks.
///
/// Logits land in `scratch_per_rank[0].logits` (rank 0 = `output_device`); the
/// caller reads them with `gpu.download_f32` after this returns (this fn
/// device-synchronizes every rank before returning, so the read is safe even
/// though work ran on each rank's `active_stream`).
///
/// All parallel slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`,
/// `scratch_per_rank`, `partials`) must have length `gpus.devices.len()`, with
/// element `r` allocated on `gpus.devices[r]`. Every device must have an
/// `active_stream` set ([`hipfire_runtime::ep::ensure_rank_streams`]).
///
/// TP=1 is the degenerate reference: one rank owns all experts (no zero-dummy),
/// the all-reduce short-circuits to identity, and the result is the same as the
/// single-GPU lowered decode (validated byte-/argmax-identical on the fleet).
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_per_rank: &mut [kv::KvCache],
    dn_per_rank: &[DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_ep: weights_per_rank.len() != n_ranks"
    );
    assert_eq!(
        kv_per_rank.len(),
        n,
        "forward_ep: kv_per_rank.len() != n_ranks"
    );
    assert_eq!(
        dn_per_rank.len(),
        n,
        "forward_ep: dn_per_rank.len() != n_ranks"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n,
        "forward_ep: scratch_per_rank.len() != n_ranks"
    );
    assert_eq!(partials.len(), n, "forward_ep: partials.len() != n_ranks");

    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let pos_i32 = pos as i32;

    // 1. Embed token + write pos on each rank (replicated; deterministic, since
    //    weights are byte-identical replicas → s.x is bit-identical per rank).
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        let w = &weights_per_rank[r];
        let s = &scratch_per_rank[r];
        let gpu = &mut gpus.devices[r];
        match w.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&w.token_embd, &s.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&w.token_embd, &s.x, token, dim)?,
            EmbeddingFormat::F32 => gpu.embedding_lookup(&w.token_embd, &s.x, token, dim)?,
            other => {
                return Err(HipError::new(
                    0,
                    &format!("forward_ep: unsupported embedding format {other:?}"),
                ))
            }
        }
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;
    }

    // 2. Per-layer EP program. Variant + delta-layer counter are replicated
    //    (sharding frees experts but never changes the layer variant), so rank 0
    //    is authoritative for both.
    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        let program = lower_variant(variant_of(&weights_per_rank[0].layers[layer_idx]));
        // Build the N per-rank bindings. `kv_per_rank.iter_mut()` yields the
        // disjoint `&mut KvCache` each binding needs; weights/scratch/dn are
        // shared `&`. This Vec is dropped at the end of the iteration, releasing
        // the mutable KV borrows before the next layer's `iter_mut`.
        let mut binds: Vec<Qwen35Bindings> = Vec::with_capacity(n);
        for (((w, s), kv), dn) in weights_per_rank
            .iter()
            .zip(scratch_per_rank.iter())
            .zip(kv_per_rank.iter_mut())
            .zip(dn_per_rank.iter())
        {
            binds.push(Qwen35Bindings {
                layer: &w.layers[layer_idx],
                s,
                config,
                kv_cache: kv,
                dn_state: dn,
                pos,
                layer_idx,
                delta_layer_idx,
                k_dim,
                v_dim,
                n_v_heads,
                hd,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            binds.as_mut_slice(),
            partials,
            &program,
            dim,
        )
        .map_err(|e| HipError::new(0, &e.to_string()))?;
        if matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
    }

    // 3. Final norm + lm_head on rank 0 (output_device). Logits → rank0 scratch.
    {
        gpus.devices[0].bind_thread()?;
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // 4. Sync every rank — work ran on each device's active_stream, so a host
    //    download of rank 0's logits (on the null stream) would otherwise race.
    for r in 0..n {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    Ok(())
}

/// EP (Ship 6 substrate-EP) **WMMA batched prefill** for qwen3.x-A3B (E6b).
///
/// The batched analog of [`forward_ep`]: processes all `tokens` as one batch
/// through the WMMA/grouped-GEMM prefill kernels (NOT token-by-token), replicated
/// across `gpus.devices.len()` EP ranks, with MoE experts sharded per rank.
///
/// Driven **layer-granularly** by calling [`forward_prefill_chunk`] with a
/// single-layer band per rank, because EP needs a per-MoE-layer all-reduce: the
/// next layer's replicated attention must read the FULL (cross-rank-summed)
/// residual. For each layer:
///   1. (MoE only) zero each rank's `[n × dim]` routed partial,
///   2. run the layer's batched chunk on every rank — the **shared** expert
///      accumulates into `pbs.x_batch` (replicated, added once per rank), the
///      **routed** combine into the zeroed partial (owned experts only; non-owned
///      read load-time zero-dummy → 0),
///   3. (MoE only) `all_reduce_sum_f32` the `[n × dim]` partials across ranks and
///      add into each rank's `pbs.x_batch`.
/// Non-MoE (dense DeltaNet / FullAttn) layers run replicated, no partial, no
/// all-reduce. Final norm + lm_head (last token) run on rank 0 → `scratch_per_rank[0].logits`.
///
/// **v1 constraints:** the whole prompt must fit one batch (`tokens.len() <=
/// pbs.max_batch`; no chunk loop yet) and KV must be a non-asym mode (q8/q4/…)
/// so no per-rank Givens replicas are needed (asym EP prefill = future work). The
/// per-layer chunk dispatch trades some launch overhead for the per-layer
/// all-reduce seam; a fused EP prefill layer loop is a later perf refinement.
///
/// Slices (`weights_per_rank`, `kv_per_rank`, `dn_per_rank`, `scratch_per_rank`,
/// `pbs_per_rank`, `partials`) must have length `gpus.devices.len()`; element `r`
/// lives on `gpus.devices[r]`. Each `partials[r]` must hold >= `n × dim` f32.
/// Every device must have an `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_ep(
    gpus: &mut Gpus,
    weights_per_rank: &[Qwen35Weights],
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_per_rank: &mut [kv::KvCache],
    dn_per_rank: &mut [DeltaNetState],
    scratch_per_rank: &[Qwen35Scratch],
    pbs_per_rank: &[PrefillBatchScratch],
    partials: &[GpuTensor],
) -> HipResult<()> {
    let n_rank = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: weights_per_rank len"
    );
    assert_eq!(
        kv_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: kv_per_rank len"
    );
    assert_eq!(
        dn_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: dn_per_rank len"
    );
    assert_eq!(
        scratch_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: scratch_per_rank len"
    );
    assert_eq!(
        pbs_per_rank.len(),
        n_rank,
        "forward_prefill_batch_ep: pbs_per_rank len"
    );
    assert_eq!(
        partials.len(),
        n_rank,
        "forward_prefill_batch_ep: partials len"
    );

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let dim = config.dim;
    assert!(
        n <= pbs_per_rank[0].max_batch,
        "forward_prefill_batch_ep v1: prompt ({n} toks) must fit one batch (max_batch={}); \
         chunked EP prefill is future work",
        pbs_per_rank[0].max_batch,
    );

    // Per-layer cumulative LA / FA counters (replicated → identical across ranks;
    // they index dn_state.s_matrices / kv_cache.k_gpu exactly like the band
    // offsets the PP driver threads). kv_layer_offset == fa_layer_offset.
    let mut delta_off = 0usize;
    let mut fa_off = 0usize;

    let ep_timing = std::env::var("HIPFIRE_EP_PREFILL_TIMING").is_ok();
    let ep_skip_ar = std::env::var("HIPFIRE_EP_SKIP_ALLREDUCE").is_ok(); // DIAGNOSTIC ONLY (wrong output)
                                                                         // Peer-direct all-reduce (bypass RCCL): the routed-partial sum goes through
                                                                         // Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms
                                                                         // vs RCCL's ~40 ms/call on hiptrx (gfx1201, PCIe). DEFAULT ON; opt back to
                                                                         // RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared
                                                                         // with TP), lazily sized to the largest count seen.
    let ep_peer_ar = std::env::var("HIPFIRE_EP_PEER_ALLREDUCE").as_deref() != Ok("0");
    let mut t_chunk = 0.0f64;
    let mut t_ar = 0.0f64;
    let mut t_add = 0.0f64;
    for layer_idx in 0..config.n_layers {
        let is_moe = matches!(
            &weights_per_rank[0].layers[layer_idx],
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        );

        // 1. Zero each rank's routed partial (on its active_stream, so it's
        //    ordered before the chunk's routed combine that writes into it).
        if is_moe {
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let stream = gpus.devices[r].active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        "forward_prefill_batch_ep: no active_stream (call ensure_rank_streams)",
                    )
                })?;
                gpus.devices[r]
                    .hip
                    .memset_async(&partials[r].buf, 0, n * dim * 4, stream)?;
            }
        }

        // 2. Run the layer's batched chunk on every rank (single-layer band).
        let t_c = std::time::Instant::now();
        for r in 0..n_rank {
            gpus.devices[r].bind_thread()?;
            let band = PrefillBandCtx {
                layer_start: layer_idx,
                layer_end: layer_idx + 1,
                delta_layer_offset: delta_off,
                kv_layer_offset: fa_off,
                fa_layer_offset: fa_off,
                is_first_band: layer_idx == 0,
                is_last_band: false, // final norm + lm_head done explicitly below
                // v1 EP prefill is q8/non-asym KV → no per-rank Givens replicas.
                givens_cos: None,
                givens_sin: None,
            };
            let routed_out = if is_moe { Some(&partials[r]) } else { None };
            forward_prefill_chunk(
                &mut gpus.devices[r],
                &weights_per_rank[r],
                config,
                tokens,
                start_pos,
                &mut kv_per_rank[r],
                &mut dn_per_rank[r],
                &scratch_per_rank[r],
                &pbs_per_rank[r],
                None,  // hidden_rb
                None,  // per_token_hidden_out
                None,  // gdn_tape
                0,     // tape_offset
                None,  // tree_verify
                false, // pre_uploaded
                Some(&band),
                None,  // mask_override
                None,  // positions_override
                false, // needs_last_token_logits (no lm_head in band)
                None,  // max_layer
                false, // force_q8_gdn_per_token
                routed_out,
            )?;
        }

        if ep_timing {
            t_chunk += t_c.elapsed().as_secs_f64() * 1000.0;
        }

        // 3. All-reduce the routed partials, add into each rank's residual.
        if is_moe && !ep_skip_ar {
            let t_a = std::time::Instant::now();
            let refs: Vec<&hip_bridge::DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
            if ep_peer_ar {
                gpus.all_reduce_sum_f32_peer(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            } else {
                gpus.all_reduce_sum_f32(&refs, n * dim)
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
            }
            if ep_timing {
                t_ar += t_a.elapsed().as_secs_f64() * 1000.0;
            }
            let t_d = std::time::Instant::now();
            for r in 0..n_rank {
                gpus.devices[r].bind_thread()?;
                let x_n = pbs_per_rank[r].x_batch.sub_offset(0, n * dim);
                let p_n = partials[r].sub_offset(0, n * dim);
                gpus.devices[r].add_inplace_f32(&x_n, &p_n)?;
            }
            if ep_timing {
                t_add += t_d.elapsed().as_secs_f64() * 1000.0;
            }
        }

        match config.layer_types[layer_idx] {
            LayerType::LinearAttention => delta_off += 1,
            LayerType::FullAttention => fa_off += 1,
        }
    }

    // Final norm + lm_head on rank 0 (last token) → scratch_per_rank[0].logits.
    // Done explicitly (not via the chunk) so it runs AFTER the last layer's
    // all-reduce — the last MoE layer's routed output is only in x_batch after
    // step 3, so an in-chunk lm_head would read an incomplete residual.
    {
        gpus.devices[0].bind_thread()?;
        let gpu = &mut gpus.devices[0];
        let w = &weights_per_rank[0];
        let s = &scratch_per_rank[0];
        let pbs = &pbs_per_rank[0];
        let last_x = pbs.x_batch.sub_offset((n - 1) * dim, dim);
        gpu.rmsnorm_f32(&last_x, &w.output_norm, &s.tmp, config.norm_eps)?;
        let ctx = DispatchCtx::new(gpu);
        let wr = w.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s.tmp),
            out: &s.logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    // Sync every rank — work ran on active_streams; the host logits read on rank
    // 0 (null stream) would otherwise race.
    let t_s = std::time::Instant::now();
    for r in 0..n_rank {
        gpus.devices[r].bind_thread()?;
        gpus.devices[r].hip.device_synchronize()?;
    }
    if ep_timing {
        let t_sync = t_s.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "EP-PREFILL-TIMING (host ms): chunk-loop={t_chunk:.1} all_reduce={t_ar:.1} add={t_add:.1} final-sync={t_sync:.1}",
        );
    }
    Ok(())
}

/// Multi-GPU layer-loop dispatcher (Stage 5 of multi-GPU pp migration #58).
/// Mirrors `forward_scratch_layers` but routes per-layer work to
/// `gpus.devices[gpus.device_for_layer(i)]` and copies the residual
/// stream `s.x` across band boundaries via `Gpus::boundary_copy`.
/// Final `output_norm + lm_head` runs on `gpus.output_device`
/// (Variant 2 — no copy back to dev_0). Spec-decode `hidden_rb` is
/// not threaded — refused at load time when pp > 1.
#[allow(unused_variables, unused_assignments)]
fn forward_scratch_layers_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let _ = qkv_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let mut delta_layer_idx = 0usize;
    let mut prev_dev: Option<usize> = None;

    for layer_idx in 0..config.n_layers {
        let dev_idx = gpus.device_for_layer(layer_idx);

        if let Some(pd) = prev_dev {
            if dev_idx != pd {
                let src_buf = &scratch_set.per_device[pd].x.buf;
                let dst_buf = &scratch_set.per_device[dev_idx].x.buf;
                let evt = gpus.boundary_copy(pd, dev_idx, src_buf, dst_buf, dim * 4)?;
                gpus.wait_boundary(evt)?;
            }
        }

        {
            let s = &scratch_set.per_device[dev_idx];
            let givens_cos_dev = gpus.givens_cos_per_dev.get(dev_idx);
            let givens_sin_dev = gpus.givens_sin_per_dev.get(dev_idx);
            let gpu = &mut gpus.devices[dev_idx];

            // Resolve givens lazily — asym{2,3,4} branches use these,
            // others don't. Multi-GPU prefers the per-device replica
            // populated by the KV ctor; fall back to kv_cache.givens_*
            // for single-GPU shape compatibility (shouldn't fire in
            // pp > 1 since asym ctors always populate per-device).
            macro_rules! ct {
                () => {
                    givens_cos_dev.unwrap_or_else(|| kv_cache.givens_cos.as_ref().unwrap())
                };
            }
            macro_rules! st {
                () => {
                    givens_sin_dev.unwrap_or_else(|| kv_cache.givens_sin.as_ref().unwrap())
                };
            }

            match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
                (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    // Lever 1 — Fused rmsnorm + PARO per-group rotation for wqkv.
                    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                        && layer.wqkv.gpu_dtype == DType::ParoQ4G128
                        && layer.wqkv.k % 128 == 0
                        && layer.wqkv.m % 8 == 0
                    {
                        fused_rmsnorm_rotate_for_paro(
                            gpu,
                            &layer.wqkv,
                            &s.x,
                            &layer.attn_norm,
                            &s.tmp,
                            &s.x_rot,
                            config.norm_eps,
                        )?
                    } else {
                        None
                    };
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    let fused_la4_paro4t = la4_same_dtype
                        && dt == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var_os("HIPFIRE_PARO_LA4_FUSED").is_some();
                    let fused_la2_paro4t = dt == DType::ParoQ4G128
                        && layer.wz.gpu_dtype == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var_os("HIPFIRE_PARO_LA2_FUSED").is_some();
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_paro4t {
                        gpu.fused_qkvza_paro4g128t(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            &s.tmp,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            &s.x_rot,
                            &s.ffn_hidden,
                            &s.ffn_out,
                            &s.o,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la2_paro4t {
                        gpu.fused_qkvza_paro4g128t(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.wqkv.buf,
                            &layer.wqkv.buf,
                            &s.tmp,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            &s.x_rot,
                            &s.ffn_hidden,
                            &s.ffn_out,
                            &s.o,
                            layer.wqkv.m,
                            layer.wz.m,
                            0,
                            0,
                            layer.wqkv.k,
                        )?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    } else {
                        if let Some(xr_first) = x_rot_paro {
                            gpu.gemv_paro4g128t_prerotated(
                                &layer.wqkv.buf,
                                xr_first,
                                &s.dn_qkv,
                                layer.wqkv.m,
                                layer.wqkv.k,
                            )?;
                        } else {
                            weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        }
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_conv1d_silu_split_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        n_v_heads,
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

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
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    let fused_gu_paro4t = same_dtype
                        && dt_g == DType::ParoQ4G128
                        && layer.w_gate.m == layer.w_up.m
                        && layer.w_gate.k == layer.w_up.k
                        && x_rot_paro.is_none()
                        && std::env::var("HIPFIRE_PARO_GATE_UP_FUSED")
                            .map(|v| v != "0")
                            .unwrap_or(true);
                    if fused_gu_mq4 {
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
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
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
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 = config.attn_output_gate
                        && fa3_same_dtype
                        && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 =
                        config.attn_output_gate && fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 =
                        config.attn_output_gate && fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    let fused_fa3_paro4t = config.attn_output_gate
                        && fa3_same_dtype
                        && dt == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var("HIPFIRE_PARO_FA3_FUSED")
                            .map(|v| v != "0")
                            .unwrap_or(true);
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
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
                    } else {
                        if let Some(xr_first) = x_rot_paro {
                            gpu.gemv_paro4g128t_prerotated(
                                &layer.wq.buf,
                                xr_first,
                                &s.fa_q_full,
                                layer.wq.m,
                                layer.wq.k,
                            )?;
                        } else {
                            weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
                        }
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
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
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
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
                        let ct = ct!();
                        let st = st!();
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
                        let ct = ct!();
                        let st = st!();
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
                        let use_flash = gpu.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
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
                                &s.flash_partials,
                            )?;
                        } else {
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
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
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
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
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
                    let fused_gu_mq4 =
                        same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                    let fused_gu_paro4t = same_dtype
                        && dt_g == DType::ParoQ4G128
                        && layer.w_gate.m == layer.w_up.m
                        && layer.w_gate.k == layer.w_up.k
                        && x_rot_paro.is_none()
                        && std::env::var("HIPFIRE_PARO_GATE_UP_FUSED")
                            .map(|v| v != "0")
                            .unwrap_or(true);
                    if fused_gu_mq4 {
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
                }

                (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                    let x_rot = fused_rmsnorm_rotate_for_mq(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?;
                    // Lever 1 — Fused rmsnorm + PARO per-group rotation for wqkv.
                    let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                        && layer.wqkv.gpu_dtype == DType::ParoQ4G128
                        && layer.wqkv.k % 128 == 0
                        && layer.wqkv.m % 8 == 0
                    {
                        fused_rmsnorm_rotate_for_paro(
                            gpu,
                            &layer.wqkv,
                            &s.x,
                            &layer.attn_norm,
                            &s.tmp,
                            &s.x_rot,
                            config.norm_eps,
                        )?
                    } else {
                        None
                    };
                    let dt = layer.wqkv.gpu_dtype;
                    let la4_same_dtype = layer.wz.gpu_dtype == dt
                        && layer.w_beta.gpu_dtype == dt
                        && layer.w_alpha.gpu_dtype == dt;
                    let fused_la4_mq4 =
                        la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                    let fused_la4_paro4t = la4_same_dtype
                        && dt == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var_os("HIPFIRE_PARO_LA4_FUSED").is_some();
                    let fused_la2_paro4t = dt == DType::ParoQ4G128
                        && layer.wz.gpu_dtype == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var_os("HIPFIRE_PARO_LA2_FUSED").is_some();
                    if fused_la4_mq4 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_hfq4g256(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_lloyd_mq3 {
                        let eff_x = match x_rot {
                            Some(xr) => xr,
                            None => &s.tmp,
                        };
                        gpu.fused_qkvza_mq3g256_lloyd(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            eff_x,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la4_paro4t {
                        gpu.fused_qkvza_paro4g128t(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.w_beta.buf,
                            &layer.w_alpha.buf,
                            &s.tmp,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            &s.x_rot,
                            &s.ffn_hidden,
                            &s.ffn_out,
                            &s.o,
                            layer.wqkv.m,
                            layer.wz.m,
                            layer.w_beta.m,
                            layer.w_alpha.m,
                            layer.wqkv.k,
                        )?;
                    } else if fused_la2_paro4t {
                        gpu.fused_qkvza_paro4g128t(
                            &layer.wqkv.buf,
                            &layer.wz.buf,
                            &layer.wqkv.buf,
                            &layer.wqkv.buf,
                            &s.tmp,
                            &s.dn_qkv,
                            &s.dn_z,
                            &s.dn_beta,
                            &s.dn_alpha,
                            &s.x_rot,
                            &s.ffn_hidden,
                            &s.ffn_out,
                            &s.o,
                            layer.wqkv.m,
                            layer.wz.m,
                            0,
                            0,
                            layer.wqkv.k,
                        )?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    } else {
                        if let Some(xr_first) = x_rot_paro {
                            gpu.gemv_paro4g128t_prerotated(
                                &layer.wqkv.buf,
                                xr_first,
                                &s.dn_qkv,
                                layer.wqkv.m,
                                layer.wqkv.k,
                            )?;
                        } else {
                            weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                        }
                        weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                        weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                        weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                    }
                    gpu.fused_sigmoid_alpha_gate_conv1d_silu_split_f32(
                        &s.dn_beta,
                        &s.dn_alpha,
                        &layer.dt_bias,
                        &layer.a_log,
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_v,
                        &s.dn_qkv,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        n_v_heads,
                        k_dim,
                        v_dim,
                    )?;
                    gpu.fused_qk_l2_norm_scale_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        config.linear_num_key_heads,
                        hd,
                        1.0 / (hd as f32).sqrt(),
                        config.norm_eps,
                    )?;
                    if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        gpu.repeat_interleave_qk_f32(
                            &s.dn_q_raw,
                            &s.dn_k_raw,
                            &s.dn_q,
                            &s.dn_k,
                            config.linear_num_key_heads,
                            ratio,
                            hd,
                        )?;
                    } else {
                        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                    }
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &s.dn_q,
                            &s.dn_k,
                            &s.dn_v,
                            &s.dn_alpha,
                            &s.dn_beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &s.dn_attn_out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?,
                    }
                    gpu.gated_norm_f32(
                        &s.dn_attn_out,
                        &s.dn_z,
                        &layer.norm_weight,
                        &s.dn_normed,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                    )?;
                    {
                        let ctx = DispatchCtx::new(gpu);
                        let wr = layer.wo.dispatch_ref();
                        execute_steps(
                            gpu,
                            &ctx,
                            &[Step::GemvResidual {
                                w: &wr,
                                input: GemvInput::Raw(&s.dn_normed),
                                residual: &s.x,
                                out: &s.x,
                            }],
                        )
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    }

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.x,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    } else if ffn_routed_mq2_lloyd_plain_prerotate_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq_plain(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            &s.tmp,
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.tmp,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.tmp,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    }
                    delta_layer_idx += 1;
                }

                (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
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
                    let dt = layer.wq.gpu_dtype;
                    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                    let fused_fa3_mq4 = config.attn_output_gate
                        && fa3_same_dtype
                        && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 =
                        config.attn_output_gate && fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 =
                        config.attn_output_gate && fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    let fused_fa3_paro4t = config.attn_output_gate
                        && fa3_same_dtype
                        && dt == DType::ParoQ4G128
                        && x_rot_paro.is_none()
                        && std::env::var("HIPFIRE_PARO_FA3_FUSED")
                            .map(|v| v != "0")
                            .unwrap_or(true);
                    let fused_fa3_mq4 =
                        fa3_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                    if fused_fa3_mq4 {
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
                    } else {
                        if let Some(xr_first) = x_rot_paro {
                            gpu.gemv_paro4g128t_prerotated(
                                &layer.wq.buf,
                                xr_first,
                                &s.fa_q_full,
                                layer.wq.m,
                                layer.wq.k,
                            )?;
                        } else {
                            weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
                        }
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;

                        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    }
                    qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    let kv_dim = config.n_kv_heads * config.head_dim;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;

                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
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
                    if kv_cache.compact_offset > 0 {
                        let phys = pos as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                    }

                    if kv_cache.quant_asym4 {
                        let ct = ct!();
                        let st = st!();
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
                        let ct = ct!();
                        let st = st!();
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
                        let ct = ct!();
                        let st = st!();
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
                        let use_flash = gpu.capture_mode
                            || s.flash_mode == 2
                            || (s.flash_mode == 1 && pos + 1 >= 2048)
                            || pos + 1 > 15000;
                        if use_flash {
                            gpu.attention_flash_q8_0(
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
                                &s.flash_partials,
                            )?;
                        } else {
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
                        }
                    } else {
                        gpu.kv_cache_write(
                            &kv_cache.k_gpu[layer_idx],
                            &s.fa_k,
                            &s.pos_buf,
                            kv_dim,
                        )?;
                        gpu.kv_cache_write(
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_v,
                            &s.pos_buf,
                            kv_dim,
                        )?;
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
                    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                    {
                        let ctx = DispatchCtx::new(gpu);
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

                    if ffn_all_mq4_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.x,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    } else if ffn_routed_mq2_lloyd_plain_prerotate_for_moe(&layer.ffn) {
                        gpu.fused_rmsnorm_rotate_mq_plain(
                            &s.x,
                            &layer.ffn_norm,
                            s.moe_x_rot.as_ref().expect("MoE scratch"),
                            &s.tmp,
                            config.dim,
                            config.norm_eps,
                        )?;
                        moe_ffn_decode_with_scratch_prerotated(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.tmp,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    } else {
                        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                        moe_ffn_decode_with_scratch(
                            gpu,
                            weights.pager.as_ref(),
                            &layer.ffn,
                            &s.tmp,
                            &s.x,
                            config,
                            s,
                            layer_idx,
                        )?;
                    }
                }

                _ => panic!("layer type mismatch at layer {layer_idx}"),
            }
        }

        prev_dev = Some(dev_idx);
    }

    let dev_last = gpus.output_device;
    let s_last = &scratch_set.per_device[dev_last];
    let gpu_last = &mut gpus.devices[dev_last];
    gpu_last.rmsnorm_f32(
        &s_last.x,
        &weights.output_norm,
        &s_last.tmp,
        config.norm_eps,
    )?;
    {
        let ctx = DispatchCtx::new(gpu_last);
        let wr = weights.output.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&s_last.tmp),
            out: &s_last.logits,
        };
        execute_steps(gpu_last, &ctx, &[step])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    Ok(())
}

/// Multi-GPU decode forward (Stage 5 of multi-GPU pp migration #58).
/// Embedding lookup on dev 0 (token_embd lives there per Stage 4 placement),
/// then the layer loop via `forward_scratch_layers_multi`. `s.logits` ends
/// up on `gpus.output_device`. hipGraph capture is bypassed for pp > 1.
pub fn forward_scratch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    // F3 (review): asym{2,3,4} KV requires per-device givens replicas. The
    // ct!()/st!() macros in forward_scratch_layers_multi fall back to
    // kv_cache.givens_* if the per-device replica is None — which silently
    // hands a wrong-device tensor to attention kernels. Refuse up-front.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != gpus.devices.len()
            || gpus.givens_sin_per_dev.len() != gpus.devices.len())
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_scratch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let dim = config.dim;
    let pos_bytes = (pos as i32).to_ne_bytes();
    {
        let gpu0 = &mut gpus.devices[0];
        let s0 = &scratch_set.per_device[0];
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu0.embedding_lookup_hfq4g256(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu0.embedding_lookup_hfq4g128(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::Q8_0 => {
                gpu0.embedding_lookup_q8(&weights.token_embd, &s0.x, token, dim)?
            }
            EmbeddingFormat::F32 => {
                gpu0.embedding_lookup(&weights.token_embd, &s0.x, token, dim)?
            }
            _ => panic!("unsupported embedding format"),
        }
    }
    // pos_buf written to every device's scratch — every band reads it inside
    // RoPE / KV write for FullAttention layers. F1 (review): bind_thread
    // before each raw gpu.hip.memcpy_htod — HipRuntime methods bypass the
    // Stage 2b bind audit, so without explicit bind the writes land on
    // whatever device was last bound (dev 0 from the embedding lookup above).
    for dev_idx in 0..gpus.devices.len() {
        let gpu = &mut gpus.devices[dev_idx];
        gpu.bind_thread()?;
        let s = &scratch_set.per_device[dev_idx];
        gpu.hip.memcpy_htod(&s.pos_buf, &pos_bytes)?;
    }
    forward_scratch_layers_multi(gpus, weights, config, pos, kv_cache, dn_state, scratch_set)
}

/// Multi-GPU batched prefill (Stage 6 of #58 — multi-gpu pipeline-parallel).
/// Closes the daemon-time pp=1 vs pp=2 divergence — single-GPU
/// `forward_prefill_batch` runs through the WMMA-batched fast path, while
/// pp=2 was previously stuck on per-token `forward_scratch_multi` (a
/// different kernel sequence with a different reduction order). This
/// routes both paths through the same `forward_prefill_chunk` body, just
/// band-restricted via `PrefillBandCtx`.
///
/// Flow per chunk of up to `max_batch` tokens:
///   1. Allocate per-band `PrefillBatchScratch` on each device's pbs.
///   2. Run `forward_prefill_chunk` on dev 0 with band 0 layers,
///      `is_first_band=true` (does the embedding) and
///      `is_last_band=(n_bands==1)`.
///   3. peer-copy band 0's `pbs.x_batch` into band 1's `pbs.x_batch`.
///   4. Run `forward_prefill_chunk` on dev 1 with band 1 layers,
///      `is_first_band=false` (skips embedding, reads already-populated
///      `x_batch`) and `is_last_band=true` (does final norm + lm_head).
///   5. Repeat for any further bands.
///
/// `tree_verify`, DFlash hidden-rb, GdnTape, and per_token_hidden_out
/// are pp=1 only in v1. They've been refused at the daemon load-time
/// gate, so this function does not accept them as parameters.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_multi(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
) -> HipResult<()> {
    forward_prefill_batch_multi_with_caps(
        gpus,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch_set,
        None,
        None,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_multi_with_caps(
    gpus: &mut Gpus,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch_set: &Qwen35ScratchSet,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape_shards: Option<&mut crate::speculative::GdnTapeShards>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    needs_last_token_logits: bool,
) -> HipResult<()> {
    assert!(
        tree_verify.is_none(),
        "forward_prefill_batch_multi_with_caps: tree_verify under PP is not implemented in v1",
    );
    let n_total = tokens.len();
    if n_total == 0 {
        return Ok(());
    }

    let n_bands = gpus.devices.len();
    if n_bands == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: no devices",
        ));
    }

    // F3 (review-pattern from forward_scratch_multi): asym{2,3,4} KV requires
    // per-device givens replicas. Refuse up-front — the band-mode macros in
    // forward_prefill_chunk fall back to kv_cache.givens_* if the band's
    // givens override is None, which silently hands a wrong-device tensor
    // to attention kernels.
    if (kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && (gpus.givens_cos_per_dev.len() != n_bands || gpus.givens_sin_per_dev.len() != n_bands)
    {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_multi: asym KV mode requires gpus.givens_*_per_dev \
             populated for every device. Construct KvCache via the *_multi ctor \
             (e.g. KvCache::new_gpu_asym3_capped_multi) — single-GPU ctors leave \
             gpus.givens_*_per_dev empty.",
        ));
    }

    let max_batch: usize = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(PREFILL_MAX_BATCH);

    let force_fallback = std::env::var("HIPFIRE_PREFILL_BATCHED").ok().as_deref() == Some("0");

    // Eligibility: same checks as `forward_prefill_batch_with_pbs`. If any
    // layer fails the batched gate, fall back to per-token forward —
    // correctness preserved at the cost of per-token kernel sequence.
    let arch0 = gpus.devices[0].arch.as_str();
    let moe_topk_ok = config.num_experts_per_tok == 8 && config.num_experts <= 1024;
    let eligible = !force_fallback
        && n_total >= 2
        && dn_state.quant == StateQuant::Q8
        && weights
            .layers
            .iter()
            .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),))
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::DeltaNet(l) => {
                is_batchable_la(l.wqkv.gpu_dtype, arch0)
                    && is_batchable_la(l.wz.gpu_dtype, arch0)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch0)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, arch0)
                    && is_batchable_la(l.wk.gpu_dtype, arch0)
                    && is_batchable_la(l.wv.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && is_batchable_la(l.w_gate.gpu_dtype, arch0)
                    && is_batchable_la(l.w_up.gpu_dtype, arch0)
                    && is_batchable_la(l.w_down.gpu_dtype, arch0)
            }
            LayerWeights::DeltaNetMoe(l) => {
                moe_topk_ok
                    && is_batchable_la(l.wqkv.gpu_dtype, arch0)
                    && is_batchable_la(l.wz.gpu_dtype, arch0)
                    && is_batchable_la(l.w_beta.gpu_dtype, arch0)
                    && is_batchable_la(l.w_alpha.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && moe_ffn_batched_admissible(&l.ffn, arch0)
            }
            LayerWeights::FullAttnMoe(l) => {
                moe_topk_ok
                    && is_batchable_la(l.wq.gpu_dtype, arch0)
                    && is_batchable_la(l.wk.gpu_dtype, arch0)
                    && is_batchable_la(l.wv.gpu_dtype, arch0)
                    && is_batchable_la(l.wo.gpu_dtype, arch0)
                    && moe_ffn_batched_admissible(&l.ffn, arch0)
            }
        });

    if !eligible {
        // Per-token fallback. Correctness over speed when the batched
        // path's preconditions are not met.
        for (i, &tok) in tokens.iter().enumerate() {
            forward_scratch_multi(
                gpus,
                weights,
                config,
                tok,
                start_pos + i,
                kv_cache,
                dn_state,
                scratch_set,
            )?;
        }
        return Ok(());
    }

    // Per-band cumulative offsets into LA / FA layer indices. The band's
    // first layer of a given type (DeltaNet or FullAttn) reads
    // `dn_state.s_matrices[delta_off]` / `kv_cache.k_caches[fa_off]`.
    let mut delta_off_per_band = vec![0usize; n_bands];
    let mut fa_off_per_band = vec![0usize; n_bands];
    {
        let mut delta_run = 0usize;
        let mut fa_run = 0usize;
        for b in 0..n_bands {
            delta_off_per_band[b] = delta_run;
            fa_off_per_band[b] = fa_run;
            let band_start = gpus.band_starts[b];
            let band_end = if b + 1 < n_bands {
                gpus.band_starts[b + 1]
            } else {
                config.n_layers
            };
            for li in band_start..band_end {
                match config.layer_types[li] {
                    LayerType::LinearAttention => delta_run += 1,
                    LayerType::FullAttention => fa_run += 1,
                }
            }
        }
    }

    // Allocate one PrefillBatchScratch per band. Each lives on the band's
    // device. Freed at the end of the call (matches forward_prefill_batch's
    // own_pbs pattern). Future opt: cache on Qwen35ScratchSet.
    let mut pbs_per_band: Vec<PrefillBatchScratch> = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        let g = &mut gpus.devices[b];
        g.bind_thread()?;
        pbs_per_band.push(PrefillBatchScratch::new(g, config, max_batch)?);
    }

    let dim = config.dim;
    let dim_row_bytes = dim * 4;
    let mut gdn_tape_shards = gdn_tape_shards;
    let last_band = n_bands - 1;

    let result = (|| -> HipResult<()> {
        let mut chunk_start = 0usize;
        while chunk_start < n_total {
            let chunk_end = (chunk_start + max_batch).min(n_total);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();

            for b in 0..n_bands {
                let band_layer_start = gpus.band_starts[b];
                let band_layer_end = if b + 1 < n_bands {
                    gpus.band_starts[b + 1]
                } else {
                    config.n_layers
                };
                let givens_cos = gpus.givens_cos_per_dev.get(b);
                let givens_sin = gpus.givens_sin_per_dev.get(b);
                let band_ctx = PrefillBandCtx {
                    layer_start: band_layer_start,
                    layer_end: band_layer_end,
                    delta_layer_offset: delta_off_per_band[b],
                    kv_layer_offset: fa_off_per_band[b],
                    fa_layer_offset: fa_off_per_band[b],
                    is_first_band: b == 0,
                    is_last_band: b + 1 == n_bands,
                    givens_cos,
                    givens_sin,
                };
                let pth_for_band = if b == last_band {
                    per_token_hidden_out.map(|t| (t, chunk_start))
                } else {
                    None
                };
                let tape_for_band: Option<&mut crate::speculative::GdnTape> =
                    gdn_tape_shards.as_mut().map(|shards| shards.shard_mut(b));
                {
                    let pbs_b: &PrefillBatchScratch = &pbs_per_band[b];
                    let s_b = &scratch_set.per_device[b];
                    let g_b = &mut gpus.devices[b];
                    forward_prefill_chunk(
                        g_b,
                        weights,
                        config,
                        chunk,
                        start_pos + chunk_start,
                        kv_cache,
                        dn_state,
                        s_b,
                        pbs_b,
                        None, // hidden_rb: pp=1 only
                        pth_for_band,
                        tape_for_band,
                        0,
                        None,  // tree_verify: pp=1 only
                        false, // pre_uploaded
                        Some(&band_ctx),
                        None, // mask_override: multi-GPU PP path doesn't use the MTP probe hook
                        None, // positions_override: PP path uses linear positions
                        needs_last_token_logits,
                        None,  // max_layer: multi-GPU PP path runs full stack
                        false, // force_q8_gdn_per_token
                        None,  // routed_out: PP bands are multi-layer, not EP
                    )?;
                }

                if b + 1 < n_bands {
                    // Hand off the chunk's residual stream to the next band.
                    // pbs.x_batch holds [N × dim] f32 — copy `chunk_n` rows
                    // from band b to band b+1. wait_boundary makes the dst
                    // device wait on the copy's completion event before the
                    // next forward_prefill_chunk dispatch reads x_batch.
                    let copy_bytes = chunk_n * dim_row_bytes;
                    let (left, right) = pbs_per_band.split_at(b + 1);
                    let pbs_src = &left[b];
                    let pbs_dst = &right[0];
                    let evt = gpus.boundary_copy(
                        b,
                        b + 1,
                        &pbs_src.x_batch.buf,
                        &pbs_dst.x_batch.buf,
                        copy_bytes,
                    )?;
                    gpus.wait_boundary(evt)?;
                }
            }

            chunk_start = chunk_end;
        }
        Ok(())
    })();

    for (b, pbs) in pbs_per_band.into_iter().enumerate() {
        let g = &mut gpus.devices[b];
        let _ = g.bind_thread();
        pbs.free_gpu(g);
    }

    result
}
