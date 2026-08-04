// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 impl of the arch-generic `hipfire_runtime::spec::SpecTarget`.
//!
//! [`Deepseek4Bundle`] owns the model pieces the daemon + MTP drafter need
//! (config + weights + recurrent state + eos) so deepseek4 can be borrowed as a
//! `&mut dyn SpecTarget` exactly like the qwen35 `ModelSlot` — the prerequisite
//! for routing it through the unified spec loop. The MTP draft+verify itself is
//! the [`crate::spec_decode`] fused step, reached by downcasting this bundle in
//! the deepseek4 `MtpDrafter` impl; deepseek4 never pairs with the model-free
//! n-gram drafter, so the n-gram-verify primitives are intentional error stubs.
//!
//! The four DSpark-specific `SpecTarget` hooks (`new_spec_scratch`,
//! `verify_block`, `commit_prefix`, `capture_seed_main_hidden`) ARE
//! implemented here so the generic `DsparkDrafter` in `dspark_core` can
//! route verify + bootstrap through the trait without downcasting — the
//! byte-identical gate depends on these hitting the same kernel paths as
//! the old inline `Deepseek4DsparkDrafter`.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{
    self, dspark_assemble_main_hidden, final_norm_and_argmax_all_batched,
    final_norm_and_argmax_all_batched_lazy, final_norm_and_sample_all_batched_lazy,
    forward_prefill_batch_chunk, forward_prefill_batch_chunk_preuploaded,
    upload_prefill_batch_inputs, PrefillBatchScratch,
};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::replay::ReplayState;
use rdna_compute::{Gpu, GpuTensor};

/// Owned deepseek4 model state — the future `ModelState::Deepseek4` payload and
/// the spec-decode target. Bundles config + weights + recurrent state + eos so
/// the daemon can borrow it as `&mut dyn SpecTarget`.
pub struct Deepseek4Bundle {
    pub config: DeepseekV4Config,
    pub weights: DeepseekV4Weights,
    pub state: DeepseekV4State,
    pub eos_tok: u32,
}

/// Thin verify scratch for the DSpark `DsparkDrafter` path. DeepSeek V4's SWA
/// attention is stateless (no recurrent rewind needed between verify and
/// commit_prefix), so the scratch carries no GPU buffers — the PBS lives in
/// `state.dspark_verify_pbs` and is reused across windows.
pub struct Deepseek4DsparkScratch;

impl SpecScratch for Deepseek4DsparkScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // No GPU buffers owned by this scratch.
    }
}

/// Max batch for the trunk-side verify PBS (bootstrap 1-token + verify up to
/// block+1 tokens). Mirror of `Deepseek4DsparkDrafter::pbs_max_batch`.
fn dspark_verify_pbs_max_batch() -> usize {
    hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}

/// DSpark's adaptive controller settles at two draft tokens on the gfx1151
/// MoE route, so the hot verify shape is B=3 (drafts plus the target bonus
/// token). Capturing every transient startup shape pays graph construction for
/// little or no reuse. `0` remains a diagnostic opt-in for the old all-B mode.
fn dspark_verify_graph_batch_from_value(value: Option<&str>) -> usize {
    value.and_then(|s| s.parse().ok()).unwrap_or(3)
}

fn dspark_verify_graph_batch() -> usize {
    dspark_verify_graph_batch_from_value(
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_GRAPH_BATCH")
            .ok()
            .as_deref(),
    )
}

impl Deepseek4Bundle {
    /// Shared verify forward for `verify_block` / `verify_block_capture_gpu`:
    /// arms hidden capture into `state.dspark_caps`, runs one batched trunk
    /// forward over `block` at `position`, and leaves the head unapplied. When
    /// `refresh_layers`, re-reads the extract-layer ids from the sidecar (needed
    /// on steady-state windows that never called `capture_seed_main_hidden`).
    fn dspark_verify_forward(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        refresh_layers: bool,
    ) -> Result<(), String> {
        let n_verify = block.len();
        {
            let pbs_ref = self.state.dspark_verify_pbs.as_ref().ok_or(
                "Deepseek4Bundle::verify_block: dspark_verify_pbs not allocated (call new_spec_scratch first)",
            )?;
            if pbs_ref.max_batch < n_verify {
                return Err(format!(
                    "Deepseek4Bundle::verify_block: PBS max_batch ({}) < block len ({})",
                    pbs_ref.max_batch, n_verify
                ));
            }
        }
        if refresh_layers {
            if let Some(ref dspark) = self.weights.dspark {
                self.state.dspark_target_layers = dspark.cfg.target_layer_ids.clone();
            }
        }
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable + mutable borrow collision.
        let mut pbs = self.state.dspark_verify_pbs.take().unwrap();
        let graph_batch = dspark_verify_graph_batch();
        let pm4_enabled = gpu.arch == "gfx1151"
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_PM4")
                .ok()
                .as_deref()
                == Some("1");
        let aql_enabled = !pm4_enabled
            && gpu.arch == "gfx1151"
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_AQL")
                .ok()
                .as_deref()
                == Some("1");
        let retained_enabled = pm4_enabled || aql_enabled;
        let graph_enabled = !retained_enabled
            && gpu.arch == "gfx1151"
            && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSPARK_VERIFY_GRAPH")
                .ok()
                .as_deref()
                == Some("1")
            && (graph_batch == 0 || graph_batch == n_verify);
        let fwd_result = if graph_enabled {
            (|| -> Result<(), String> {
                upload_prefill_batch_inputs(&self.config, gpu, &pbs, block, position as u32)?;
                if gpu.active_stream.is_none() {
                    return Err(
                        "DSpark verify graph: upload did not create an active stream".into(),
                    );
                }
                if gpu.graphs.verify_has_graph(n_verify) {
                    gpu.graphs
                        .verify_graph_launch(
                            &gpu.hip,
                            gpu.device_id,
                            gpu.active_stream.as_ref().unwrap(),
                            n_verify,
                        )
                        .map_err(|e| format!("DSpark verify graph replay B={n_verify}: {e:?}"))
                } else if gpu.graphs.verify_needs_warmup(n_verify) {
                    gpu.graphs.verify_mark_warmup_done(n_verify);
                    let result = forward_prefill_batch_chunk_preuploaded(
                        &self.config,
                        &self.weights,
                        &mut self.state,
                        gpu,
                        &pbs,
                        block,
                        position as u32,
                    );
                    if result.is_ok() {
                        eprintln!("[ds4-dspark-verify-graph] warmup B={n_verify} complete");
                    }
                    result
                } else {
                    gpu.graphs
                        .begin_verify_graph_capture(
                            &gpu.hip,
                            gpu.device_id,
                            gpu.active_stream.as_ref().unwrap(),
                            n_verify,
                        )
                        .map_err(|e| format!("DSpark verify graph begin B={n_verify}: {e:?}"))?;
                    let result = forward_prefill_batch_chunk_preuploaded(
                        &self.config,
                        &self.weights,
                        &mut self.state,
                        gpu,
                        &pbs,
                        block,
                        position as u32,
                    );
                    if let Err(error) = result {
                        let _ = gpu.hip.stream_end_capture(
                            gpu.active_stream
                                .as_ref()
                                .expect("DSpark verify graph stream disappeared"),
                        );
                        gpu.graphs.capture_mode = false;
                        gpu.graphs.capture_blobs.clear();
                        Err(error)
                    } else {
                        let stream = gpu
                            .active_stream
                            .as_ref()
                            .expect("DSpark verify graph stream disappeared");
                        gpu.graphs
                            .end_verify_graph_capture(&gpu.hip, gpu.device_id, stream)
                            .map_err(|e| format!("DSpark verify graph end B={n_verify}: {e:?}"))?;
                        // Stream capture records but does not execute the forward.
                        // Launch once so this verify window observes current inputs.
                        gpu.graphs
                            .verify_graph_launch(&gpu.hip, gpu.device_id, stream, n_verify)
                            .map_err(|e| {
                                format!("DSpark verify graph first launch B={n_verify}: {e:?}")
                            })?;
                        eprintln!(
                            "[ds4-dspark-verify-graph] captured B={n_verify} entries={}",
                            gpu.graphs.verify_graph_count()
                        );
                        Ok(())
                    }
                }
            })()
        } else if retained_enabled {
            (|| -> Result<(), String> {
                upload_prefill_batch_inputs(&self.config, gpu, &pbs, block, position as u32)?;

                let mut controller = pbs.dspark_verify_pm4.remove(&n_verify).unwrap_or_else(|| {
                    if aql_enabled {
                        rdna_compute::replay::ReplayController::new_manual_aql()
                    } else {
                        rdna_compute::replay::ReplayController::new_manual_pm4()
                    }
                });
                std::mem::swap(&mut gpu.replay, &mut controller);
                let result = (|| -> Result<(), String> {
                    if gpu.replay.should_route_pm4() {
                        // Outside graph capture, `upload_prefill_batch_inputs`
                        // uses synchronous H2D copies for every dynamic input
                        // consumed by this retained body. The preceding DSpark
                        // proposal and the prior verify result are likewise
                        // materialized before the controller reaches this
                        // call. A host stream synchronization here therefore
                        // adds one full GPU pipeline bubble per verify window
                        // without establishing any additional dependency.
                        let first_observed = gpu.replay.replay_observation().count == 0;
                        unsafe { gpu.replay.replay_pm4(position) }.map_err(|reason| {
                            format!("DSpark verify PM4 replay B={n_verify}: {reason}")
                        })?;
                        if first_observed {
                            eprintln!(
                                "[ds4-dspark-verify-pm4] observed replay B={n_verify} position={position} identity={:?}",
                                gpu.replay.prepared_route_identity()
                            );
                        }
                        return Ok(());
                    }
                    if gpu.replay.should_route_aql() {
                        let first_observed = gpu.replay.replay_observation().count == 0;
                        unsafe { gpu.replay.replay_linear_aql(position) }.map_err(|reason| {
                            format!("DSpark verify AQL replay B={n_verify}: {reason}")
                        })?;
                        if first_observed {
                            eprintln!(
                                "[ds4-dspark-verify-aql] observed replay B={n_verify} position={position} identity={:?}",
                                gpu.replay.prepared_route_identity()
                            );
                        }
                        return Ok(());
                    }

                    if matches!(gpu.replay.state(), ReplayState::Fallback | ReplayState::Hip) {
                        return forward_prefill_batch_chunk_preuploaded(
                            &self.config,
                            &self.weights,
                            &mut self.state,
                            gpu,
                            &pbs,
                            block,
                            position as u32,
                        );
                    }

                    gpu.replay.begin_capture().map_err(|reason| {
                        format!("DSpark verify PM4 begin capture B={n_verify}: {reason}")
                    })?;
                    let direct = forward_prefill_batch_chunk_preuploaded(
                        &self.config,
                        &self.weights,
                        &mut self.state,
                        gpu,
                        &pbs,
                        block,
                        position as u32,
                    );
                    if let Err(error) = direct {
                        gpu.replay.poison(format!(
                            "DSpark verify PM4 capture body B={n_verify}: {error}"
                        ));
                        return Err(error);
                    }
                    gpu.hip.device_synchronize().map_err(|e| {
                        format!("DSpark verify PM4 capture sync B={n_verify}: {e:?}")
                    })?;
                    let capture = gpu.replay.finish_capture().map_err(|reason| {
                        format!("DSpark verify PM4 finish capture B={n_verify}: {reason}")
                    })?;
                    let launches = gpu.replay.recorded_launches().len();
                    if hipfire_config::developer_var("HIPFIRE_DS4_REPLAY_INVENTORY")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        let mut inventory = std::collections::BTreeMap::<String, usize>::new();
                        for launch in gpu.replay.recorded_launches() {
                            *inventory.entry(launch.kernel.clone()).or_default() += 1;
                        }
                        for (kernel, count) in inventory {
                            eprintln!(
                                "DS4_DSPARK_PM4_INVENTORY B={n_verify} kernel={kernel} count={count}"
                            );
                        }
                    }
                    if hipfire_config::developer_var("HIPFIRE_DS4_REPLAY_SEQUENCE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        for (index, launch) in gpu.replay.recorded_launches().iter().enumerate() {
                            eprintln!(
                                "DS4_DSPARK_PM4_SEQUENCE B={n_verify} index={index} kernel={}",
                                launch.kernel
                            );
                        }
                    }
                    let contracts = match gpu.replay.probe_aql_contracts(gpu.device_id as usize) {
                        Ok(contracts) => contracts,
                        Err(reason) => {
                            gpu.replay.poison(format!(
                                "DSpark verify PM4 AQL contracts B={n_verify}: {reason}"
                            ));
                            eprintln!(
                                "[ds4-dspark-verify-pm4] rejected B={n_verify}: contract probe: {reason}"
                            );
                            return Ok(());
                        }
                    };
                    let prepared = if aql_enabled {
                        gpu.replay
                            .prepare_linear_aql_prefix(gpu.device_id as usize, launches)
                            .map(|_| ())
                    } else {
                        gpu.replay
                            .prepare_pm4_prefix(gpu.device_id as usize, launches)
                            .map(|_| ())
                    };
                    match prepared {
                        Ok(_) => eprintln!(
                            "[ds4-dspark-verify-{}] ready B={n_verify} capture={capture:?} contracts={}/{} identity={:?}",
                            if aql_enabled { "aql" } else { "pm4" },
                            contracts.len(),
                            capture.unique_kernel_count,
                            gpu.replay.prepared_route_identity()
                        ),
                        Err(reason) => {
                            gpu.replay.poison(format!(
                                "DSpark verify retained prepare B={n_verify}: {reason}"
                            ));
                            eprintln!(
                                "[ds4-dspark-verify-{}] rejected B={n_verify}: prepare: {reason}",
                                if aql_enabled { "aql" } else { "pm4" },
                            );
                        }
                    }
                    Ok(())
                })();
                std::mem::swap(&mut gpu.replay, &mut controller);
                pbs.dspark_verify_pm4.insert(n_verify, controller);
                result
            })()
        } else {
            forward_prefill_batch_chunk(
                &self.config,
                &self.weights,
                &mut self.state,
                gpu,
                &pbs,
                block,
                position as u32,
            )
        };
        // Restore the PBS before propagating any error so state stays consistent.
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result.map_err(|e| format!("Deepseek4Bundle::verify_block forward: {e}"))
    }

    /// Apply the trunk final-norm + lm_head + per-position argmax over the
    /// `n_verify` hidden rows left in the verify PBS by `dspark_verify_forward`.
    fn dspark_verify_argmax(&mut self, gpu: &mut Gpu, n_verify: usize) -> Result<Vec<u32>, String> {
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let argmax_result = final_norm_and_argmax_all_batched(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            n_verify,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        argmax_result.map_err(|e| format!("Deepseek4Bundle::verify_block head+argmax: {e}"))
    }

    /// LAZY twin of `dspark_verify_argmax`: greedy argmax per position with a
    /// prefix stop against the drafted `block` (skips heads after the first
    /// mismatch). Byte-identical committed output, fewer lm_head GEMVs.
    fn dspark_verify_argmax_lazy(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
    ) -> Result<Vec<u32>, String> {
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let res = final_norm_and_argmax_all_batched_lazy(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            block,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        res.map_err(|e| format!("Deepseek4Bundle::verify_block head+argmax(lazy): {e}"))
    }

    /// temp>0 twin of `dspark_verify_argmax`: fused GPU sample per position with
    /// LAZY prefix stop against the drafted `block` (samples ~τ heads/window, not
    /// all n). Advances `rng_state`.
    fn dspark_verify_sample_lazy(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        temp: f32,
        top_p: f32,
        top_k: usize,
        cactus_delta: f32,
        rng_state: &mut u64,
    ) -> Result<Vec<u32>, String> {
        let result_buf = gpu
            .alloc_tensor(&[2], rdna_compute::DType::F32)
            .map_err(|e| format!("dspark_verify_sample_lazy result_buf: {e:?}"))?;
        let repeat_buf = gpu
            .alloc_tensor(&[1], rdna_compute::DType::F32)
            .map_err(|e| format!("dspark_verify_sample_lazy repeat_buf: {e:?}"))?;
        let mut rng32 = *rng_state as u32;
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let res = final_norm_and_sample_all_batched_lazy(
            &self.config,
            &self.weights,
            &mut self.state,
            &pbs,
            gpu,
            block,
            temp,
            top_p,
            top_k,
            cactus_delta,
            &mut rng32,
            &result_buf,
            &repeat_buf,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        *rng_state = rng32 as u64;
        let _ = gpu.free_tensor(result_buf);
        let _ = gpu.free_tensor(repeat_buf);
        res.map_err(|e| format!("Deepseek4Bundle::verify_block head+sample: {e}"))
    }
}

impl SpecTarget for Deepseek4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Host counters + MTP residual + graph-warmup flag, then zero every
        // position-indexed SWA/full/compressed/indexer cache so a fresh
        // conversation cannot bleed prior-turn residue (pairs with the
        // daemon's `gpu.invalidate_graph_state()` after this hook).
        self.state.reset();
        self.state.zero_decode_caches(gpu);
        Ok(())
    }

    fn retry_reset_eligible(&self) -> bool {
        // reset() + zero_decode_caches; daemon pairs invalidate_graph_state.
        true
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.config.max_position_embeddings
    }

    // ── n-gram-verify primitives (intentionally unsupported) ────────────────
    // deepseek4's MTP drafter downcasts this bundle and runs `spec_decode` —
    // those paths never hit these hooks. The DSpark drafter DOES use
    // `new_spec_scratch` / `verify_block` / `commit_prefix`; see below.

    /// Advance the trunk over `tokens` from `start_pos`, returning the greedy
    /// argmax at the last position. Used by `DsparkDrafter::mtp_prefill` to
    /// run the prompt through the trunk in a single pass.
    ///
    /// `reset` is always `false` here — the caller (`DsparkDrafter::mtp_prefill`)
    /// calls `reset_recurrent` separately on cache miss. `abort` and `hidden_out`
    /// are ignored for this arch (deepseek4 is not abort-capable in this path
    /// and does not expose hidden states via `spec_advance`).
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        _reset: bool,
        _abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        // Lazily allocate the trunk-sized PBS.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle::spec_advance: alloc PBS: {e}"))?,
            );
        }
        // Take the PBS out of state to avoid a simultaneous immutable + mutable
        // borrow of self.state (forward_prefill_batch_chunked takes &mut state).
        // Restore it afterward (it is always Some after the lazy alloc above).
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let result = forward::forward_prefill_batch_chunked(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            tokens,
            start_pos as u32,
            &pbs,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        let last_logits =
            result.map_err(|e| format!("Deepseek4Bundle::spec_advance prefill: {e}"))?;
        let last_argmax = crate::spec_decode::logits_argmax(&last_logits) as u32;
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(last_logits),
        })
    }

    // ── DSpark verify primitives ──────────────────────────────────────────
    //
    // The generic `DsparkDrafter` in `dspark_core` calls these three methods
    // to verify draft tokens against the trunk. They route to the IDENTICAL
    // kernel paths the old inline `Deepseek4DsparkDrafter` used —
    // `forward_prefill_batch_chunk` + `final_norm_and_argmax_all_batched` —
    // so the byte-identical gate passes without any numeric change.

    /// Allocate the thin DSpark verify scratch. The PBS lives in
    /// `state.dspark_verify_pbs` (lazily allocated here on first call);
    /// `Deepseek4DsparkScratch` itself carries no GPU buffers.
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| format!("Deepseek4Bundle: alloc dspark_verify_pbs: {e}"))?,
            );
        }
        Ok(Box::new(Deepseek4DsparkScratch))
    }

    /// Run the trunk forward over `block` at absolute `position`, returning
    /// per-slot target argmaxes. Mirrors `Deepseek4DsparkDrafter::mtp_step`
    /// steps 3–4 exactly: capture armed, `forward_prefill_batch_chunk` then
    /// `final_norm_and_argmax_all_batched`.
    ///
    /// **Stage 3 hidden_out capture:** when `hidden_out` is `Some`, downloads
    /// the per-position captured main-hidden from `state.dspark_caps` and writes
    /// `n * n_targets * hidden` floats into it (row-major, one `n_targets * hidden`
    /// row per verified position). This is the multi-slot context the generic
    /// `DsparkDrafter` uses to skip bootstrap in steady-state windows.
    /// `dspark_target_layers` is set from the DSpark sidecar config before each
    /// capture so it remains valid even after the initial bootstrap.
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let n_verify = block.len();
        // Arm capture + run the trunk forward. refresh_layers when hidden_out is
        // Some (steady-state windows re-read the sidecar extract layers).
        self.dspark_verify_forward(gpu, block, position, hidden_out.is_some())?;

        // ── Stage 3: download captured hidden for multi-slot context update ──
        // dspark_caps layout: [max_batch, n_targets, hidden] flat. Positions
        // 0..n_verify are contiguous at offset 0, so a single d2h suffices.
        if let Some(out) = hidden_out {
            let n_targets = self.state.dspark_target_layers.len();
            let hidden = self.config.hidden_size;
            if n_targets > 0 {
                let n_floats = n_verify * n_targets * hidden;
                let mut raw = vec![0.0f32; n_floats];
                if let Some(caps) = self.state.dspark_caps.as_ref() {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, n_floats * 4)
                    };
                    gpu.hip
                        .memcpy_dtoh(bytes, &caps.buf)
                        .map_err(|e| format!("Deepseek4Bundle::verify_block caps d2h: {e:?}"))?;
                }
                *out = raw;
            }
        }

        self.dspark_verify_argmax(gpu, n_verify)
    }

    /// GPU-resident variant of [`verify_block`]: captures the accepted-prefix
    /// hidden straight into the caller-owned `hidden_gpu` (GPU→GPU) instead of
    /// downloading it to a host `Vec` and re-uploading. deepseek4's batched
    /// forward captures every position, so `captured` is always true once the
    /// drafter has extract layers configured.
    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let n_verify = block.len();
        self.dspark_verify_forward(gpu, block, position, true)?;

        let n_targets = self.state.dspark_target_layers.len();
        let hidden = self.config.hidden_size;
        let captured = n_targets > 0;
        if captured {
            let n_floats = n_verify * n_targets * hidden;
            if let Some(caps) = self.state.dspark_caps.as_ref() {
                gpu.memcpy_dtod_auto(&hidden_gpu.buf, &caps.buf, n_floats * 4)
                    .map_err(|e| {
                        format!("Deepseek4Bundle::verify_block_capture_gpu caps dtod: {e:?}")
                    })?;
            }
        }

        let picks = self.dspark_verify_argmax_lazy(gpu, block)?;
        Ok((picks, captured))
    }

    /// temp>0 twin of [`verify_block_capture_gpu`]: same batched forward + GPU
    /// hidden capture, but draws each position with the fused GPU sampler and
    /// LAZY prefix stop (distribution-identical to AR temp-T decoding).
    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        cactus_delta: f32,
        rng_state: &mut u64,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let n_verify = block.len();
        self.dspark_verify_forward(gpu, block, position, true)?;

        let n_targets = self.state.dspark_target_layers.len();
        let hidden = self.config.hidden_size;
        let captured = n_targets > 0;
        if captured {
            let n_floats = n_verify * n_targets * hidden;
            if let Some(caps) = self.state.dspark_caps.as_ref() {
                gpu.memcpy_dtod_auto(&hidden_gpu.buf, &caps.buf, n_floats * 4)
                    .map_err(|e| {
                        format!(
                            "Deepseek4Bundle::verify_block_sampled_capture_gpu caps dtod: {e:?}"
                        )
                    })?;
            }
        }

        let picks = self.dspark_verify_sample_lazy(
            gpu,
            block,
            temp,
            top_p,
            top_k,
            cactus_delta,
            rng_state,
        )?;
        Ok((picks, captured))
    }

    /// Advance `state.n_tokens` to reflect the committed prefix. DeepSeek
    /// V4's SWA attention is stateless so no recurrent rewind is needed;
    /// the next verify forward simply overwrites the rejected tail slots.
    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        accept_len: usize,
        position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Mirrors the old inline drafter:
        // `bundle.state.n_tokens = (position + committed.len()) as u64`
        // where `committed.len() = accept_len + 1` (accepted drafts + bonus).
        self.state.n_tokens = (position + accept_len + 1) as u64;
        Ok(())
    }

    // ── DSpark bootstrap primitive ─────────────────────────────────────────

    /// Run a 1-token trunk forward with capture armed at `layers`, assemble
    /// the concatenated `[layers.len() * hidden]` main-hidden vector, and
    /// return it as a host-side `Vec<f32>`. Mirrors the bootstrap step of
    /// the old `Deepseek4DsparkDrafter::mtp_step` (steps 1a–1c) exactly.
    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        // Lazily allocate the trunk-sized verify PBS if not yet present.
        if self.state.dspark_verify_pbs.is_none() {
            self.state.dspark_verify_pbs = Some(
                PrefillBatchScratch::new(gpu, &self.config, dspark_verify_pbs_max_batch())
                    .map_err(|e| {
                        format!("Deepseek4Bundle: alloc dspark_verify_pbs (bootstrap): {e}")
                    })?,
            );
        }

        self.state.dspark_target_layers = layers.to_vec();
        self.state.dspark_capture_active = true;
        // Take the PBS out of state to avoid immutable+mutable borrow conflict.
        let pbs = self.state.dspark_verify_pbs.take().unwrap();
        let fwd_result = forward_prefill_batch_chunk(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            &pbs,
            &[seed],
            position as u32,
        );
        self.state.dspark_verify_pbs = Some(pbs);
        fwd_result
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden forward: {e}"))?;

        dspark_assemble_main_hidden(&mut self.state, gpu, &self.config, 0)
            .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden assemble: {e}"))?;

        let n = layers.len() * self.config.hidden_size;
        let mut host = vec![0.0f32; n];
        {
            let main_hidden = self
                .state
                .dspark_main_hidden
                .as_ref()
                .ok_or("Deepseek4Bundle::capture_seed_main_hidden: dspark_main_hidden is None after assemble")?;
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &main_hidden.buf)
                .map_err(|e| format!("Deepseek4Bundle::capture_seed_main_hidden d2h: {e:?}"))?;
        }
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::dspark_verify_graph_batch_from_value;

    #[test]
    fn dspark_verify_graph_defaults_to_settled_b3() {
        assert_eq!(dspark_verify_graph_batch_from_value(None), 3);
        assert_eq!(dspark_verify_graph_batch_from_value(Some("5")), 5);
        assert_eq!(dspark_verify_graph_batch_from_value(Some("0")), 0);
        assert_eq!(dspark_verify_graph_batch_from_value(Some("bad")), 3);
    }
}
