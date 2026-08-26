// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use crate::config_cache;
use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{
    acquire_moe_authority_mesh, compressor_cache_uses_vmm, ds4_attn_block, ds4_ep_moe_step,
    ds4_lower_program, final_norm_and_head, init_residual_streams,
    refresh_compressor_cache_shard_tables, validate_mesh_entry_policy, Deepseek4Bindings,
};
use crate::forward::{
    ensure_compressor_capacity, precompute_positions, precompute_token_id, update_attn_state_host,
    update_pos_array_host, update_token_id_host,
};
use rdna_compute::GpuTensor;

// ───────────────────────── Ship 6 substrate-EP (DeepSeek-V4) ─────────────────
//
// Mirror of the qwen35 / MiniMax EP wiring. DeepSeek packs all routed experts
// into ONE blob per projection (too big to load-then-free on a 32 GB card), so
// sharding is done at LOAD time: `DeepseekV4::load_weights_sharded(.., shard,
// rank)` uploads only the rank-owned experts (non-owned → zeroed gate_up dummy).
// UNLIKE MiniMax, DeepSeek has a SHARED expert (ffn_stub) and the HC FFN mix:
//   - the shared expert stays replicated in `state.ffn_out` (every rank),
//   - only the ROUTED combine crosses ranks (redirected into the per-rank
//     partial, all-reduced), and
//   - `hc_ffn_mix` is DEFERRED to `ep_add_into_residual` (runs after the
//     all-reduce assembles `ffn_out = shared + routed`).
// See `Deepseek4Bindings::run_moe_ep` / `ep_add_into_residual` + `ds4_moe_block_core`.
// MLA attention (latent KV) is replicated per rank → no attention-sharding seam.

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
///
/// Mirror of `decode_step` + `decode_step_body_lowered`, fanned across
/// `gpus.devices.len()` ranks: every rank replicates embed / positions /
/// token-id / residual-stream init and the per-layer `[Attend, Moe]` program
/// (Attend replicated, Moe all-reduce-EP'd) via
/// [`hipfire_runtime::ep::run_layer_program_ep`], then final norm + head run on
/// rank 0 → `state_per_rank[0].logits` (caller downloads). Every device must
/// have an `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]); peer
/// access enabled for the fast peer-direct all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    partials_i64: &[GpuTensor],
    policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    // Policy validation + MoE authority BEFORE any GPU work: the caller
    // policy must be the exact Ep kind + mesh/epoch binding of these Gpus —
    // enforced even when MoE is disabled/shared-only. The authority is
    // acquired EXACTLY ONCE for this forward, under the exact
    // caller/loader-owned policy and the SINGLE enable snapshot.
    validate_mesh_entry_policy(
        gpus,
        policy,
        hipfire_runtime::moe_plan::MoEExecutionKind::Ep,
    )?;
    assert_eq!(partials_i64.len(), n, "ds4 forward_ep: partials_i64 len");
    let moe_on = config_cache::moe_on();
    let authority = acquire_moe_authority_mesh(weights_per_rank, cfg, policy, moe_on)?;
    if state_per_rank.len() == n
        && gpus
            .devices
            .iter()
            .all(|device| compressor_cache_uses_vmm(device))
    {
        for rank in 0..n {
            gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("ds4 TP{n} cache bind rank {rank}: {error:?}"))?;
            ensure_compressor_capacity(
                cfg,
                &mut state_per_rank[rank],
                &mut gpus.devices[rank],
                (position as usize).saturating_add(1),
            )?;
        }
        refresh_compressor_cache_shard_tables(state_per_rank)?;
    }
    let graph_slots = cfg.num_hidden_layers * 2;
    let tp_graph_admitted = matches!(n, 3 | 4)
        && weights_per_rank.len() == n
        && state_per_rank.len() == n
        && partials.len() == n
        && cfg.num_hidden_layers > 0
        && cfg.mq2r
        && !cfg.mq2rxt
        && gpus.peer_access_enabled
        && gpus.tp_graph_signals_ready(graph_slots)
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && weights_per_rank.iter().all(|weights| {
            let layer = weights.resolve_layer(0);
            layer.attn_tp_size == n && layer.shared_tp_size == n
        })
        // The dump synchronizes and downloads inside the layer loop; it is a
        // deliberately direct diagnostic route, never a captured product path.
        && hipfire_config::developer_var("HIPFIRE_EP_DUMP_POS").is_err();

    if tp_graph_admitted {
        // Mainline captured TP graph route (gfx1201 MQ2R TP3/TP4): the
        // host-binding super-op program is the certified captured behavior —
        // kept intact (authority-free host MoE inside the captured body).
        forward_ep_tp_graph(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            partials_i64,
            policy,
            authority,
            token,
            position,
        )
    } else {
        // Sealed EP decode: per-layer attend (replicated) + routed MoE
        // through the lowered parallel program (i64 down + AllReduce{Ep})
        // executed by the sealed executor.
        forward_ep_direct(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            partials_i64,
            policy,
            authority,
            token,
            position,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_ep_tp_graph_body(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    let program = ds4_lower_program();
    let skip_ffn = config_cache::skip_ffn();
    for layer_idx in 0..cfg.num_hidden_layers {
        let mut bindings: Vec<Deepseek4Bindings> = Vec::with_capacity(n);
        for (rank, state) in state_per_rank.iter_mut().enumerate() {
            bindings.push(Deepseek4Bindings {
                cfg,
                weights: &weights_per_rank[rank],
                state,
                layer_idx,
                position,
                token_id: token,
                skip_ffn,
                authority: None,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            bindings.as_mut_slice(),
            partials,
            &program,
            cfg.hidden_size,
        )
        .map_err(|error| format!("ds4 TP{n} graph run_layer_program_ep L{layer_idx}: {error}"))?;
    }

    gpus.devices[0]
        .bind_thread()
        .map_err(|error| format!("ds4 TP{n} graph final bind: {error:?}"))?;
    final_norm_and_head(
        cfg,
        &weights_per_rank[0],
        &mut state_per_rank[0],
        &mut gpus.devices[0],
    )
}

fn sync_ep_ranks(gpus: &mut hipfire_runtime::multi_gpu::Gpus, label: &str) -> Result<(), String> {
    for rank in 0..gpus.devices.len() {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|error| format!("{label} bind rank {rank}: {error:?}"))?;
        gpus.devices[rank]
            .hip
            .device_synchronize()
            .map_err(|error| format!("{label} sync rank {rank}: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_ep_tp_graph(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    partials_i64: &[GpuTensor],
    policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
    authority: crate::forward::MoeAuthority<'_>,
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    for rank in 0..n {
        ensure_compressor_capacity(
            cfg,
            &mut state_per_rank[rank],
            &mut gpus.devices[rank],
            (position as usize).saturating_add(1),
        )?;
    }

    // One ordinary pass admits every lazy allocation and JIT module before
    // four-device capture. Layout growth resets this flag on every rank.
    if state_per_rank
        .iter()
        .any(|state| !state.ar_forward_warmed_up)
    {
        for state in state_per_rank.iter_mut() {
            state.ar_forward_warmed_up = true;
        }
        return forward_ep_direct(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            partials_i64,
            policy,
            authority,
            token,
            position,
        );
    }

    // Dynamic embedding stays outside the fixed graph body. Position, cache
    // state, and token-id uploads are captured from stable host allocations,
    // exactly like the certified single-device DS4 graph route.
    for rank in 0..n {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|error| format!("ds4 TP{n} graph stage bind {rank}: {error:?}"))?;
        init_residual_streams(
            cfg,
            &weights_per_rank[rank],
            &mut state_per_rank[rank],
            &mut gpus.devices[rank],
            token,
        )?;
    }

    let captured = gpus
        .devices
        .iter()
        .filter(|device| device.graphs.graph_exec.is_some())
        .count();
    if captured != 0 && captured != n {
        return Err(format!(
            "ds4 TP{n} graph state is partial: {captured}/{n} rank graphs exist"
        ));
    }

    if captured == 0 {
        gpus.begin_tp_graph_signal_capture()
            .map_err(|error| format!("ds4 TP{n} graph signal capture: {error:?}"))?;
        for rank in 0..n {
            let gpu = &mut gpus.devices[rank];
            gpu.graphs
                .begin_graph_capture_relaxed(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().ok_or_else(|| {
                        format!("ds4 TP{n} graph rank {rank} has no active stream")
                    })?,
                )
                .map_err(|error| format!("ds4 TP{n} graph begin rank {rank}: {error:?}"))?;
        }
        for rank in 0..n {
            precompute_positions(
                cfg,
                &mut state_per_rank[rank],
                &mut gpus.devices[rank],
                position,
            )?;
            precompute_token_id(&mut state_per_rank[rank], &mut gpus.devices[rank], token)?;
        }
        forward_ep_tp_graph_body(
            gpus,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            token,
            position,
        )?;

        let captured_signals = gpus.tp_graph_captured_signal_count();
        let expected_signals = cfg.num_hidden_layers * 2;
        if captured_signals != expected_signals {
            return Err(format!(
                "ds4 TP{n} graph captured {captured_signals} barriers, expected {expected_signals}"
            ));
        }
        for rank in 0..n {
            let gpu = &mut gpus.devices[rank];
            gpu.graphs
                .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
                .map_err(|error| format!("ds4 TP{n} graph end rank {rank}: {error:?}"))?;
        }
        let blobs: usize = gpus
            .devices
            .iter()
            .map(|device| device.graphs.ar_forward_blobs.len())
            .sum();
        eprintln!(
            "[DeepSeek V4 gfx1201 TP{n} hipGraph] captured {n} ranks, {captured_signals} barriers, {blobs} kernarg blobs"
        );
    } else {
        for state in state_per_rank.iter_mut() {
            update_pos_array_host(cfg, state, position);
            update_attn_state_host(cfg, state, state.n_tokens as u32);
            update_token_id_host(state, token);
        }
    }

    // Clear all 86 epochs on every rank before any graph launch. Synchronous
    // signal-memory resets are intentional: async per-rank resets race a fast
    // peer store against another rank still holding the prior epoch's value.
    gpus.reset_tp_graph_signals()
        .map_err(|error| format!("ds4 TP{n} graph reset signals: {error:?}"))?;
    for rank in 0..n {
        let gpu = &gpus.devices[rank];
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|error| format!("ds4 TP{n} graph launch rank {rank}: {error:?}"))?;
    }
    sync_ep_ranks(gpus, &format!("ds4 TP{n} graph"))?;
    for state in state_per_rank.iter_mut() {
        state.n_tokens += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_ep_direct(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    partials_i64: &[GpuTensor],
    policy: &hipfire_runtime::moe_plan::MoEExecutionPolicy,
    authority: crate::forward::MoeAuthority<'_>,
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "ds4 forward_ep: weights_per_rank len"
    );
    assert_eq!(
        state_per_rank.len(),
        n,
        "ds4 forward_ep: state_per_rank len"
    );
    assert_eq!(partials.len(), n, "ds4 forward_ep: partials len");
    assert_eq!(partials_i64.len(), n, "ds4 forward_ep: partials_i64 len");
    let skip_ffn = config_cache::skip_ffn();

    // 1. Per-rank embed + position + token-id staging + residual-stream init
    //    (replicated, deterministic functions of the token → bit-identical
    //    across ranks). Mirrors `decode_step`'s preamble.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep bind {r}: {e:?}"))?;
        precompute_positions(cfg, &mut state_per_rank[r], &mut gpus.devices[r], position)?;
        precompute_token_id(&mut state_per_rank[r], &mut gpus.devices[r], token)?;
        init_residual_streams(
            cfg,
            &weights_per_rank[r],
            &mut state_per_rank[r],
            &mut gpus.devices[r],
            token,
        )?;
    }

    // 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd). Rebuild
    //    the N per-rank bindings each layer (disjoint `iter_mut` mutable state
    //    borrows), exactly like the single-GPU lowered loop advances per layer.
    let timing = hipfire_config::developer_var("HIPFIRE_EP_DECODE_TIMING").is_ok();
    // Divergence-localization dump: HIPFIRE_EP_DUMP_POS="0,64,...,302" prints a
    // per-(position, layer, rank) fingerprint of the residual streams so EP
    // forwards can be compared across tp counts / arches. Diagnostic only.
    let dump_pos_hit = hipfire_config::developer_var("HIPFIRE_EP_DUMP_POS")
        .ok()
        .map(|s| {
            s.split(',')
                .any(|x| x.trim().parse::<u32>() == Ok(position))
        })
        .unwrap_or(false);
    let t_layers = std::time::Instant::now();
    for l in 0..cfg.num_hidden_layers {
        // Attend replicated: every rank holds full MLA weights + full KV, so
        // the per-rank attention is a deterministic function of replicated
        // inputs and stays bit-identical across ranks (the only EP divergence
        // is Moe).
        for r in 0..n {
            gpus.devices[r]
                .bind_thread()
                .map_err(|e| format!("ds4 forward_ep attn bind {r} L{l}: {e:?}"))?;
            ds4_attn_block(
                cfg,
                &weights_per_rank[r],
                &mut state_per_rank[r],
                &mut gpus.devices[r],
                l,
                position,
            )
            .map_err(|e| format!("ds4 forward_ep attn L{l} r{r}: {e}"))?;
        }
        // Moe all-reduce EP through the sealed lowered program: each rank
        // runs its owned routed experts (+ the replicated shared expert) into
        // an i64 partial, the partials all-reduce over the Ep group, and each
        // rank folds the reduced routed sum into ffn_out before the HC mix
        // (tail order [AddRouted, HcMix] — see `ds4_tail_actions`). The
        // authoritative borrowed plan for this layer is borrowed once per
        // forward (an Enabled state with a missing plan is an explicit error
        // — cached resolution failure, never a silent skip).
        let plan = match authority.entry() {
            Some(entry) => Some(entry.plan(l).ok_or_else(|| {
                format!("ds4 forward_ep moe-step L{l}: no plan in the authority entry")
            })?),
            None => None,
        };
        ds4_ep_moe_step(
            gpus,
            plan,
            weights_per_rank,
            cfg,
            state_per_rank,
            partials,
            partials_i64,
            policy,
            l,
            token,
            skip_ffn,
        )
        .map_err(|e| format!("ds4 forward_ep moe-step L{l}: {e}"))?;
        if dump_pos_hit {
            for r in 0..n {
                gpus.devices[r]
                    .bind_thread()
                    .map_err(|e| format!("ds4 EPDUMP bind {r}: {e:?}"))?;
                gpus.devices[r]
                    .hip
                    .device_synchronize()
                    .map_err(|e| format!("ds4 EPDUMP sync {r}: {e:?}"))?;
                if let Some(t) = state_per_rank[r].residual_streams.as_ref() {
                    let v = gpus.devices[r].download_f32(t).unwrap_or_default();
                    let l2: f64 = v
                        .iter()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum::<f64>()
                        .sqrt();
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &x in &v {
                        for b in x.to_bits().to_le_bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                    }
                    eprintln!(
                        "EPDUMP pos={position} layer={l} rank={r} l2={l2:.9e} fnv=0x{h:016x} f0={:.6e} f1={:.6e}",
                        v.first().copied().unwrap_or(0.0),
                        v.get(1).copied().unwrap_or(0.0),
                    );
                }
                // Deeper DSA-path dump (rank 0 only): compressor caches, indexer
                // scores, and selected top-k indices — discriminates a
                // systematically-divergent compressor kernel from near-tie
                // top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable.
                if r == 0
                    && hipfire_config::developer_var("HIPFIRE_EP_DUMP_IDX")
                        .ok()
                        .as_deref()
                        == Some("1")
                {
                    let fp = |gpu: &mut rdna_compute::Gpu,
                              t: &Option<rdna_compute::GpuTensor>|
                     -> String {
                        match t {
                            Some(t) => match gpu.download_f32(t) {
                                Ok(v) => {
                                    let l2: f64 = v
                                        .iter()
                                        .map(|&x| (x as f64) * (x as f64))
                                        .sum::<f64>()
                                        .sqrt();
                                    let mut h: u64 = 0xcbf29ce484222325;
                                    for &x in &v {
                                        for b in x.to_bits().to_le_bytes() {
                                            h ^= b as u64;
                                            h = h.wrapping_mul(0x100000001b3);
                                        }
                                    }
                                    format!("l2={l2:.9e} fnv=0x{h:016x}")
                                }
                                Err(_) => "dl-err".to_string(),
                            },
                            None => "none".to_string(),
                        }
                    };
                    let idx = &state_per_rank[0]._indexer[l];
                    let score_fp = fp(&mut gpus.devices[0], &idx.index_score);
                    let ikv_fp = fp(&mut gpus.devices[0], &idx.indexer_kv_cache);
                    let mkv_fp = fp(&mut gpus.devices[0], &idx.main_kv_cache);
                    let topk_head: String = match idx.topk_idx_indices.as_ref() {
                        Some(t) => match gpus.devices[0].download_f32(t) {
                            Ok(v) => v
                                .iter()
                                .take(24)
                                .map(|x| (x.to_bits() as i32).to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                            Err(_) => "dl-err".to_string(),
                        },
                        None => "none".to_string(),
                    };
                    eprintln!(
                        "EPIDX pos={position} layer={l} score[{score_fp}] ikv[{ikv_fp}] mkv[{mkv_fp}] topk={topk_head}"
                    );
                }
            }
        }
    }

    // 3. Final norm + head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep bind0: {e:?}"))?;
        final_norm_and_head(
            cfg,
            &weights_per_rank[0],
            &mut state_per_rank[0],
            &mut gpus.devices[0],
        )?;
    }

    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    // 4. Sync every rank (work ran on active_streams; host logits read races otherwise).
    let t_sync = std::time::Instant::now();
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("ds4 forward_ep sync {r}: {e:?}"))?;
    }
    if timing {
        eprintln!(
            "EP-DECODE-TIMING: layers(host)={layers_ms:.2} ms  final-sync(gpu)={:.2} ms",
            t_sync.elapsed().as_secs_f64() * 1000.0,
        );
    }
    for s in state_per_rank.iter_mut() {
        s.n_tokens += 1;
    }
    Ok(())
}
