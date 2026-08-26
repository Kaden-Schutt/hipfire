// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch staging, moved out of the daemon load handler.
//!
//! This block constructs `Qwen35DecodeBatchState`, `Lfm2DecodeBatchState` and
//! the EP `Qwen35DecodeBatchEpState`, and it was the single largest remaining
//! reason the daemon named architecture types at all (17 of 87 uses).
//!
//! It belongs here rather than in the daemon because `LoadedModel` already
//! owns the typed fields these write (`qwen35_decode_batch`,
//! `lfm2_decode_batch`, `EpArch::Qwen35 { batch, .. }`) — only the
//! *construction* had leaked upward.
//!
//! `ContinuousBatchScheduler` lives in `hipfire-engine`, which is *above* this
//! crate, so staging returns the two numbers the scheduler needs and the
//! caller constructs it. That keeps the dependency edge pointing one way.
//!
//! The body is a verbatim move: identical branch order, identical fail-closed
//! EP invariants, identical operator-visible `[daemon]` / `[daemon][EP]`
//! strings. Only the local `staged_*` variables became [`BatchStaging`]
//! fields.

use crate::LoadedModel;
use rdna_compute::Gpu;

/// Hard lane cap shared by every decode batch state: lanes are addressed
/// with u64 masks (`valid_lane_mask` in the archs), so more than 64 lanes
/// can never be constructed or migrated to. Staging refuses larger sizes
/// before any GPU allocation.
pub const CONTINUOUS_BATCH_MAX_LANES: usize = 64;

/// True when `requested` fits the u64 lane-mask hard cap.
pub fn continuous_batch_size_within_lane_cap(requested: usize) -> bool {
    requested <= CONTINUOUS_BATCH_MAX_LANES
}

/// What the load handler needs to know after staging.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStaging {
    /// Continuous batching is live for this model.
    pub capable: bool,
    /// Scheduler slot count (0 when not staged).
    pub slots: usize,
    /// Per-lane KV capacity, clamped to what independent attention admits.
    pub lane_capacity: usize,
    /// The expert-parallel (TP=4) route was taken rather than single-GPU.
    pub ep: bool,
    /// EP slot count, mirrored for the receipt.
    pub ep_slots: usize,
    /// EP lane capacity, mirrored for the receipt.
    pub ep_lane_cap: usize,
}

/// True when embedding and lm_head formats admit the batched decode kernels.
pub fn qwen_batch_weight_formats_supported(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
) -> bool {
    use hipfire_runtime::llama::EmbeddingFormat;
    use rdna_compute::DType;
    let embd_ok = matches!(
        weights.embd_format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
    );
    let lm_ok = matches!(
        weights.output.gpu_dtype,
        DType::Q8_0
            | DType::HFQ4G256
            | DType::MQ4G256
            | DType::HFQ6G256
            | DType::MQ6G256
            | DType::MQ3G256
    );
    embd_ok && lm_ok
}

/// EP uses the same admission rule as single-GPU.
pub fn qwen_ep_batch_weight_formats_supported(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
) -> bool {
    qwen_batch_weight_formats_supported(weights)
}

/// Stage continuous batching for `m`, returning what the caller must publish.
pub fn stage_continuous_batch(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    requested: usize,
) -> BatchStaging {
    let mut out = BatchStaging::default();
    // ── Continuous batch staging (must be before `loaded` ack) ──
    // Stage Qwen35DecodeBatchState / hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState (single-GPU) or
    // Qwen35DecodeBatchEpState (EP TP=4 pure gfx1201) + host scheduler.
    // `continuous_batch_capable` reflects the newly staged state, not the previous.
    // EP is batch-only: TP must be 4 and exactly 4×gfx1201, else fail closed.
    // Allocation failure advertises false and preserves sequential/poison handling.
    if requested > 1 && m.pp == 1 && m.ep.is_none() {
        match crate::continuous_batch_route(m.arch_id) {
            Some(crate::ContinuousBatchRoute::Qwen35) => {
                // Immutable borrow of `m` ends after this extraction; mutable borrow for batch field later is disjoint.
                let qwen_info = m.qwen35().map(|b| {
                    (
                        qwen_batch_weight_formats_supported(&b.weights),
                        b.scratch.repeat_buf.buf.size(),
                        b.config.head_dim,
                        b.config.clone(),
                        b.weights.embd_format,
                        b.weights.output.gpu_dtype,
                    )
                });
                if let Some((
                    weight_ok,
                    scratch_size,
                    head_dim,
                    config_clone,
                    embd_fmt,
                    out_dtype,
                )) = qwen_info
                {
                    if !weight_ok {
                        eprintln!(
                                            "[daemon] continuous batch requested but weight formats unsupported (embd={:?} lm_head={:?}) — fallback to sequential",
                                            embd_fmt, out_dtype
                                        );
                    } else {
                        let repeat_cap = (scratch_size / 4).max(1);
                        let max_attention_lane =
                            gpu.attention_q8_0_kv_independent_max_lane_capacity(head_dim);
                        let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                        if batch_lane_capacity == 0 {
                            eprintln!(
                                                "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                            );
                        } else {
                            // Hard lane cap: decode batch states address lanes with
                            // u64 masks (`valid_lane_mask`), so >64 lanes can never be
                            // constructed. Refuse before any GPU allocation (mirrors the
                            // EP arm's fail-closed max_batch bound).
                            if !continuous_batch_size_within_lane_cap(requested) {
                                eprintln!(
                                                    "[daemon] continuous batch requested size {} exceeds the 64-lane hard cap (u64 lane masks) — fallback to sequential",
                                                    requested
                                                );
                            } else {
                                match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchState::new(
                                    gpu,
                                    &config_clone,
                                    requested,
                                    batch_lane_capacity,
                                    repeat_cap,
                                ) {
                                    Ok(batch_state) => {
                                        m.qwen35_mut().unwrap().qwen35_decode_batch =
                                            Some(batch_state);
                                        out.slots = requested;
                                        out.lane_capacity = batch_lane_capacity;
                                        out.capable = true;
                                        eprintln!(
                                                            "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                            requested,
                                                            batch_lane_capacity,
                                                            repeat_cap
                                                        );
                                    }
                                    Err(e) => {
                                        eprintln!(
                                                            "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                        );
                                    }
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("[daemon] continuous batch requested but model state not Qwen35 — fallback to sequential");
                }
            }
            Some(crate::ContinuousBatchRoute::Lfm2Moe) => {
                if m.lfm2moe().is_none() {
                    eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                } else if !m.lfm2moe().unwrap().config.is_dense() {
                    eprintln!(
                                        "[daemon] continuous batch requested but LFM MoE not supported (dense only) — fallback to sequential"
                                    );
                } else if let Err(reason) = hipfire_arch_lfm2moe::batch_weight_formats_supported(
                    &m.lfm2moe().unwrap().weights,
                ) {
                    eprintln!(
                                        "[daemon] continuous batch requested but weight formats unsupported: {} — fallback to sequential",
                                        reason
                                    );
                } else {
                    let repeat_cap = 2048usize.max(1);
                    let max_attention_lane = {
                        let b = m.lfm2moe().unwrap();
                        gpu.attention_q8_0_kv_independent_max_lane_capacity(b.config.head_dim)
                    };
                    let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                    if batch_lane_capacity == 0 {
                        eprintln!(
                                            "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                        );
                    } else {
                        if batch_lane_capacity < m.max_seq {
                            eprintln!(
                                                "[daemon] continuous batch lane capacity clamped: requested={} supported={}",
                                                m.max_seq,
                                                batch_lane_capacity
                                            );
                        }
                        // Hard lane cap: decode batch states address lanes with
                        // u64 masks, so >64 lanes can never be constructed. Refuse
                        // before any GPU allocation (mirrors the Qwen/EP arms).
                        if !continuous_batch_size_within_lane_cap(requested) {
                            eprintln!(
                                                "[daemon] continuous batch requested size {} exceeds the 64-lane hard cap (u64 lane masks) — fallback to sequential",
                                                requested
                                            );
                        } else {
                            // Clone config for the call so the immutable borrow ends before the mutable one.
                            let cfg = m.lfm2moe().unwrap().config.clone();
                            match hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState::new(
                                gpu,
                                &cfg,
                                requested,
                                batch_lane_capacity,
                                repeat_cap,
                            ) {
                                Ok(batch_state) => {
                                    if let Some(b) = m.lfm2moe_mut() {
                                        b.lfm2_decode_batch = Some(batch_state);
                                        out.slots = requested;
                                        out.lane_capacity = batch_lane_capacity;
                                        out.capable = true;
                                        eprintln!(
                                                        "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                        requested,
                                                        batch_lane_capacity,
                                                        repeat_cap
                                                    );
                                    } else {
                                        // Should be unreachable (we checked is_some above), but free to avoid leak.
                                        batch_state.free_gpu(gpu);
                                        eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                                    "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                );
                                }
                            }
                        }
                    }
                }
            }
            None => {
                eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
            }
        }
    } else if requested > 1 && m.pp == 1 && m.ep.is_some() {
        // EP Qwen35 pure expert-parallel batch route: TP=4, 4×gfx1201, batch-only.
        let tp_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.len() == 4)
                .unwrap_or(false);
        let gfx_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.iter().all(|d| d.arch_caps.is_gfx1201()))
                .unwrap_or(false);
        let arch_ok = matches!(m.arch_id, 5 | 6);
        if !arch_ok || !tp_ok || !gfx_ok {
            eprintln!("[daemon][EP] continuous batch requires arch 5/6, TP=4, 4×gfx1201 (arch_ok={arch_ok} tp_ok={tp_ok} gfx_ok={gfx_ok}) — fail closed");
            out.capable = false;
        } else if let Some(ep) = m.ep.as_mut() {
            if let crate::EpArch::Qwen35 {
                config,
                weights,
                batch,
            } = &mut ep.inner
            {
                if !qwen_ep_batch_weight_formats_supported(&weights[0]) {
                    eprintln!(
                        "[daemon][EP] continuous batch weight formats unsupported — fail closed"
                    );
                } else {
                    // Derive capacities similar to single-GPU but via EP Gpus handle when possible.
                    let max_attention_lane = ep.gpus.devices[0]
                        .attention_q8_0_kv_independent_max_lane_capacity(config.head_dim);
                    let batch_lane_capacity = m.max_seq.min(max_attention_lane).max(1);
                    let repeat_cap = 128usize.max(1);
                    let prefill_chunk = 512usize;
                    if batch_lane_capacity == 0 || batch_lane_capacity >= m.max_seq + 1 {
                        eprintln!(
                            "[daemon][EP] continuous batch lane capacity invalid — fail closed"
                        );
                    } else {
                        let load_cfg = hipfire_arch_qwen35::qwen35::Qwen35BatchLoadConfig::new(
                            requested,
                            batch_lane_capacity,
                            repeat_cap,
                            prefill_chunk,
                        );
                        // Fail-closed validation before allocation.
                        match hipfire_arch_qwen35::qwen35::validate_ep_batch_compatibility(
                            &ep.gpus, weights, config, &load_cfg,
                        ) {
                            Ok(compat) => {
                                // Enforce frozen invariants.
                                if compat.rank_count() != 4 || compat.rank_mask() != 0x0f || compat.reduce() != hipfire_arch_qwen35::qwen35::Qwen35EpReduce::PeerRootedF32 || compat.topology() != hipfire_arch_qwen35::qwen35::Qwen35EpTopology::ExpertParallel {
                                                        eprintln!("[daemon][EP] compat invariants violated — fail closed: rank_count={} mask={:#x} reduce={:?} topo={:?}", compat.rank_count(), compat.rank_mask(), compat.reduce(), compat.topology());
                                                    } else {
                                                        match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchEpState::new(&mut ep.gpus, weights, config, &load_cfg) {
                                                            Ok(ep_batch) => {
                                                                // Attest receipt getters work before publishing.
                                                                let _ = ep_batch.max_batch();
                                                                let _ = ep_batch.lane_capacity();
                                                                // Peer access MUST follow every peer-visible batch
                                                                // allocation (partials + leased scratch); ROCm may
                                                                // not retroactively map late allocs.
                                                                match ep.gpus.enable_peer_all() {
                                                                    Ok(peer_access) => {
                                                                        *batch = Some(ep_batch);
                                                                        out.slots = requested;
 out.lane_capacity = batch_lane_capacity;
                                                                        out.capable = true;
                                                                        out.ep = true;
                                                                        out.ep_slots = requested;
                                                                        out.ep_lane_cap = batch_lane_capacity;
                                                                        eprintln!("[daemon][EP] expert-parallel batch staged: slots={} lane_cap={} repeat_cap={} prefill_chunk={} reduce=peer_rooted_f32 rank_count=4 peer_access={}", requested, batch_lane_capacity, repeat_cap, prefill_chunk, peer_access);
                                                                    }
                                                                    Err(enable_err) => {
                                                                        match ep_batch.free_gpu(&mut ep.gpus) {
                                                                            Ok(()) => {
                                                                                eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?} — fail closed (batch freed)");
                                                                            }
                                                                            Err(cleanup_err) => {
                                                                                eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?}; cleanup also failed: {cleanup_err:?} — fail closed");
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            Err(e) => {
                                                                eprintln!("[daemon][EP] expert-parallel batch allocation failed: {e} — fail closed");
                                                            }
                                                        }
                                                    }
                            }
                            Err(e) => {
                                eprintln!("[daemon][EP] expert-parallel batch compatibility failed: {e} — fail closed");
                            }
                        }
                    }
                }
            } else {
                eprintln!(
                    "[daemon][EP] continuous batch requested but EP arch not Qwen35 — fail closed"
                );
            }
        }
    } else if requested > 1 {
        eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
    }
    out
}
#[cfg(test)]
mod tests {
    use super::{continuous_batch_size_within_lane_cap, CONTINUOUS_BATCH_MAX_LANES};

    #[test]
    fn lane_cap_boundary_is_exactly_64() {
        // u64 lane masks address at most 64 lanes; staging refuses anything
        // larger before any GPU allocation.
        assert_eq!(CONTINUOUS_BATCH_MAX_LANES, 64);
        assert!(continuous_batch_size_within_lane_cap(0));
        assert!(continuous_batch_size_within_lane_cap(1));
        assert!(continuous_batch_size_within_lane_cap(64));
        assert!(!continuous_batch_size_within_lane_cap(65));
        assert!(!continuous_batch_size_within_lane_cap(128));
        assert!(!continuous_batch_size_within_lane_cap(usize::MAX));
    }
}
