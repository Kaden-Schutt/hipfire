// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotEngine — the multi-slot rig owned by one background thread.
//
// `hipfire serve` already speaks OpenAI-compatible HTTP, and HTTP is already
// concurrent. What serialises requests today is the single `Engine` behind a
// `Mutex<ServeRuntime>`. This engine is fed by a channel and lives outside any
// such lock, so N requests are in flight together and share one batched
// forward per step.
//
// The thread owns the GPU rig exclusively: nothing else touches `Gpu`,
// `SlotPool`, the arenas or the DeltaNet states. All interaction is by message,
// which is what makes "no lock" safe rather than merely fast.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use hipfire_runtime::admission::{AdmissionController, ModelFootprint};
use hipfire_runtime::serve::{send_event, DoneReason, EngineStats, Event, SubmitRequest};
use hipfire_runtime::session_table::{SessionId, SessionTable};
use hipfire_runtime::swap::snapshot::{capture_slot, restore_slot, SnapshotStamp};
use hipfire_runtime::swap::SwapManager;
use rdna_compute::sampling::SlotSampleParams;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::forward_slots::{forward_batch_slots_graphed, SlotDecodeGraph, SlotDescStaging};
use crate::qwen35::{
    self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
};
use crate::scheduler::{PendingWork, Scheduler};

pub struct EngineConfig {
    pub model_path: PathBuf,
    pub n_slots: usize,
    pub cap_tokens: usize,
    /// Prefill tokens taken from one slot per step. Bounds the batch scratch
    /// (`n_slots × prefill_chunk` rows, NOT `cap_tokens`) and keeps a long
    /// prompt from blocking other slots for its whole prefill — the property
    /// the scheduler was built around. Sizing scratch by `cap_tokens` put a
    /// 16k-ctx A3B out of reach of a 24 GB card.
    pub prefill_chunk: usize,
    pub host_budget_bytes: u64,
    pub swap_dir: PathBuf,
}

pub struct SlotEngine {
    tx: Option<Sender<SubmitRequest>>,
    stats: Arc<Mutex<EngineStats>>,
    handle: Option<JoinHandle<()>>,
}

impl SlotEngine {
    pub fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(req)
            .map_err(|_| "engine thread is gone".to_string())
    }

    pub fn stats(&self) -> EngineStats {
        *self.stats.lock().expect("stats mutex poisoned")
    }

    /// Build the rig on a new thread and start serving. Returns once the model
    /// is loaded, so a caller that gets `Ok` can submit immediately.
    pub fn spawn(cfg: EngineConfig) -> Result<SlotEngine, String> {
        let (tx, rx) = channel::<SubmitRequest>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);

        let handle = std::thread::Builder::new()
            .name("hipfire-slot-engine".to_string())
            .spawn(move || match Rig::build(&cfg) {
                Ok(rig) => {
                    let _ = ready_tx.send(Ok(()));
                    run_loop(rig, rx, stats_thread);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| format!("spawn engine thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(SlotEngine {
                tx: Some(tx),
                stats,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("engine thread died during startup".to_string()),
        }
    }
}

impl Drop for SlotEngine {
    fn drop(&mut self) {
        // Closing the channel is the shutdown signal; the loop exits when it
        // has no work left and the sender is gone.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Everything the engine thread owns.
struct Rig {
    gpu: Gpu,
    weights: Qwen35Weights,
    config: qwen35::Qwen35Config,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    pool: SlotPool,
    k_arenas: Vec<GpuTensor>,
    v_arenas: Vec<GpuTensor>,
    dn_states: Vec<DeltaNetState>,
    desc_staging: SlotDescStaging,
    pbs: PrefillBatchScratch,
    scratch: Qwen35Scratch,
    logits_out: GpuTensor,
    out_tokens: GpuTensor,
    sample_params: Vec<SlotSampleParams>,
    sessions: SessionTable,
    adm: AdmissionController,
    swap: SwapManager,
    stamp: SnapshotStamp,
    n_slots: usize,
    cap_tokens: usize,
    prefill_chunk: usize,
}

fn dn_buffers(dn: &DeltaNetState) -> Vec<&GpuTensor> {
    let mut v: Vec<&GpuTensor> = Vec::new();
    v.extend(dn.s_matrices.iter());
    v.extend(dn.s_scales.iter());
    v.extend(dn.conv_states.iter());
    v.extend(dn.s_ef_residual.iter());
    v
}

impl Rig {
    fn build(cfg: &EngineConfig) -> Result<Rig, String> {
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::tokenizer::Tokenizer;
        use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};

        let mut hfq = HfqFile::open(&cfg.model_path).map_err(|e| format!("open model: {e}"))?;
        let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("config: {e}"))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let n_fa_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34;
        let prefill_chunk = cfg.prefill_chunk.max(1).min(cfg.cap_tokens.max(1));
        let max_batch = (prefill_chunk * cfg.n_slots).max(cfg.n_slots);

        let weight_bytes = std::fs::metadata(&cfg.model_path)
            .map_err(|e| format!("stat model: {e}"))?
            .len();
        let cap_rounded = cfg.cap_tokens.div_ceil(128) * 128;
        let kv_bytes = (n_fa_layers as u64)
            * 2
            * (cfg.n_slots as u64)
            * (cap_rounded as u64)
            * (per_pos_bytes as u64);
        let planned = weight_bytes + kv_bytes + 768 * 1024 * 1024;
        preflight_alloc(planned, R9700_VRAM_BYTES, "SlotEngine")
            .map_err(|e| format!("preflight refused: {e}"))?;

        let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
        let weights: Qwen35Weights = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .map_err(|e| format!("load weights: {e}"))?;

        let pool = SlotPool::new(cfg.n_slots, cfg.cap_tokens, per_pos_bytes)
            .map_err(|e| format!("SlotPool: {e}"))?;
        let arena_bytes = pool.arena_bytes();
        let mut k_arenas = Vec::with_capacity(n_fa_layers);
        let mut v_arenas = Vec::with_capacity(n_fa_layers);
        for _ in 0..n_fa_layers {
            k_arenas.push(
                gpu.zeros(&[arena_bytes], DType::Raw)
                    .map_err(|e| format!("k arena: {e}"))?,
            );
            v_arenas.push(
                gpu.zeros(&[arena_bytes], DType::Raw)
                    .map_err(|e| format!("v arena: {e}"))?,
            );
        }
        let mut dn_states = Vec::with_capacity(cfg.n_slots);
        for _ in 0..cfg.n_slots {
            dn_states.push(DeltaNetState::new(&mut gpu, &config).map_err(|e| format!("dn: {e}"))?);
        }
        let desc_staging = SlotDescStaging::new(&mut gpu, cfg.n_slots, max_batch)
            .map_err(|e| format!("staging: {e}"))?;
        // Slots do plain prefill only — never tree-verify — so skip the GDN
        // S-tape: at max_batch=cap_tokens it is tens of GB (16k → ~21 GB).
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, max_batch, false)
            .map_err(|e| format!("pbs: {e}"))?;
        let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, cfg.cap_tokens)
            .map_err(|e| format!("scratch: {e}"))?;
        let logits_out = gpu
            .zeros(&[cfg.n_slots * config.vocab_size], DType::F32)
            .map_err(|e| format!("logits: {e}"))?;
        let out_tokens = gpu
            .zeros(&[cfg.n_slots], DType::F32)
            .map_err(|e| format!("out_tokens: {e}"))?;
        let sample_params = (0..cfg.n_slots)
            .map(|_| SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
            })
            .collect();

        let dn_bytes: u64 = dn_buffers(&dn_states[0])
            .iter()
            .map(|t| t.buf.size() as u64)
            .sum();
        let stamp = SnapshotStamp {
            model_hash: weight_bytes,
            kv_dtype_tag: 1,
            per_pos_bytes: per_pos_bytes as u32,
            n_fa_layers: n_fa_layers as u32,
            dn_layout_version: 1,
            cap: pool.descriptors()[0].cap as u32,
            dn_bytes,
        };

        let mut adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: weight_bytes,
                kv_bytes_per_token: (n_fa_layers * 2 * per_pos_bytes) as u64,
            },
            rdna_compute::kv_slots::R9700_VRAM_BYTES,
        );
        adm.set_host_budget(cfg.host_budget_bytes);
        let swap = SwapManager::new(cfg.swap_dir.clone(), cfg.host_budget_bytes)
            .map_err(|e| format!("SwapManager: {e}"))?;

        Ok(Rig {
            gpu,
            weights,
            config,
            tokenizer,
            pool,
            k_arenas,
            v_arenas,
            dn_states,
            desc_staging,
            pbs,
            scratch,
            logits_out,
            out_tokens,
            sample_params,
            sessions: SessionTable::default(),
            adm,
            swap,
            stamp,
            n_slots: cfg.n_slots,
            cap_tokens: cfg.cap_tokens,
            prefill_chunk,
        })
    }
}

/// One request currently occupying a slot.
struct InFlight {
    session: SessionId,
    reply: Sender<Event>,
    produced: usize,
    max_tokens: usize,
}

fn run_loop(mut rig: Rig, rx: Receiver<SubmitRequest>, stats: Arc<Mutex<EngineStats>>) {
    let n = rig.n_slots;
    let mut slots: Vec<Option<InFlight>> = (0..n).map(|_| None).collect();
    let mut work: Vec<PendingWork> = (0..n)
        .map(|s| PendingWork {
            slot: SlotId(s),
            remaining_prompt: Vec::new(),
            next_pos: 0,
            decoding: false,
        })
        .collect();
    let mut sched = Scheduler {
        chunk_size: rig.prefill_chunk,
    };
    let mut graph = SlotDecodeGraph::new();

    loop {
        let idle = slots.iter().all(|s| s.is_none());
        if idle {
            // Nothing in flight: block rather than spin.
            match rx.recv() {
                Ok(req) => admit(&mut rig, &mut slots, &mut work, &stats, req),
                Err(_) => return, // all senders gone: shut down
            }
        }
        loop {
            match rx.try_recv() {
                Ok(req) => admit(&mut rig, &mut slots, &mut work, &stats, req),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if slots.iter().all(|s| s.is_none()) {
                        return;
                    }
                    break;
                }
            }
        }
        if slots.iter().all(|s| s.is_none()) {
            continue;
        }

        let batch = sched.next_batch(&mut work);
        if batch.is_empty() {
            continue;
        }
        let fwd = forward_batch_slots_graphed(
            &mut rig.gpu,
            &rig.weights,
            &rig.config,
            &batch,
            &mut rig.pool,
            &mut rig.dn_states,
            &rig.k_arenas,
            &rig.v_arenas,
            &mut rig.desc_staging,
            &rig.pbs,
            &rig.scratch,
            &rig.logits_out,
            &mut graph,
        );
        if let Err(e) = fwd {
            // A forward failure is not recoverable per-request. Report it as a
            // REJECTION carrying the reason, not as a normal Done: an
            // unsupported model (e.g. "DeltaNet layer has a non-Q8_0 weight")
            // otherwise reaches the client as a successful, empty 200, which
            // looks like the model chose to say nothing.
            let reason = e.to_string();
            for (s, f) in slots.iter_mut().enumerate() {
                if let Some(f) = f.take() {
                    let _ = send_event(
                        &f.reply,
                        Event::Rejected {
                            reason: reason.clone(),
                        },
                    );
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    work[s].remaining_prompt.clear();
                }
            }
            continue;
        }
        if rig.gpu.hip.device_synchronize().is_err() {
            continue;
        }
        if rig
            .gpu
            .sample_per_slot(
                &rig.logits_out,
                &rig.sample_params,
                n,
                rig.config.vocab_size,
                &rig.out_tokens,
            )
            .is_err()
        {
            continue;
        }
        let _ = rig.gpu.hip.device_synchronize();
        let mut ids = vec![0i32; n];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, n * 4) };
            if rig.gpu.hip.memcpy_dtoh(bytes, &rig.out_tokens.buf).is_err() {
                continue;
            }
        }

        for s in 0..n {
            let Some(f) = slots[s].as_mut() else { continue };
            // Still prefilling: these logits belong to a mid-prompt token.
            if !work[s].remaining_prompt.is_empty() {
                continue;
            }
            let tok = ids[s] as u32;
            let session = f.session;
            let hit_eos = rig.tokenizer.is_terminator(tok);
            // Emit BEFORE counting, but never emit the terminator itself --
            // otherwise `<|im_end|>` lands in the client's content.
            let gone = if hit_eos {
                false
            } else {
                send_event(&f.reply, Event::Token { id: tok }).is_err()
            };
            f.produced += 1;
            let hit_max = f.produced >= f.max_tokens;

            if gone || hit_eos || hit_max {
                let reason = if gone {
                    DoneReason::ClientGone
                } else if hit_eos {
                    DoneReason::Eos
                } else {
                    DoneReason::MaxTokens
                };
                let _ = send_event(&f.reply, Event::Done { reason });
                slots[s] = None;
                work[s].remaining_prompt.clear();
                if matches!(reason, DoneReason::ClientGone) {
                    // Nobody will follow up on a vanished client, so hand the
                    // slot back at once.
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, session);
                } else {
                    // Keep the session RESIDENT but idle. Its KV and recurrent
                    // state are exactly what a follow-up turn continuing this
                    // conversation needs; closing here is what made multi-turn
                    // reuse impossible. LRU eviction reclaims the slot when
                    // someone else needs it.
                    rig.sessions.touch(session);
                }
            } else {
                work[s].remaining_prompt.push(tok);
                if let Some(sess) = rig.sessions.get_mut(session) {
                    sess.tokens.push(tok);
                }
                rig.sessions.touch(session);
            }
        }
    }
}

/// Admit one request, evicting an idle session if that is what it takes.
fn admit(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    req: SubmitRequest,
) {
    let busy: Vec<SessionId> = slots.iter().flatten().map(|f| f.session).collect();

    // Multi-turn continuation. The client resends the whole conversation, so a
    // follow-up turn is matched by its USER turns and then built by APPENDING
    // `continuation` to the session's exact stored tokens. Appending rather
    // than re-rendering is what makes it a strict extension: the generated
    // turn began after an OpenThink opener that history rendering does not
    // replay, and re-encoding the decoded reply is not guaranteed to round
    // trip. Reuse also keeps the DeltaNet state, which is the point.
    if !req.continuation.is_empty() {
        if rig.gpu.slot_trace() {
            eprintln!(
                "[slot-trace] continuation attempt: convo={:?} suffix={} tokens",
                req.convo,
                req.continuation.len()
            );
            for (id, sess) in rig.sessions.iter() {
                eprintln!(
                    "[slot-trace]   session {} convo={:?} slot={:?} residency={:?} tokens={}",
                    id,
                    sess.convo,
                    sess.slot,
                    sess.residency,
                    sess.tokens.len()
                );
            }
        }
        if let Some(existing) = rig.sessions.find_continuation(&req.convo, &busy) {
            let base = rig
                .sessions
                .get(existing)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            // A matched session may be SWAPPED. Bring it back before use --
            // this is the half of SP6 that makes eviction worth doing, and
            // without it evicted snapshots accumulate and their sessions are
            // never seen again.
            let mut slot = rig.sessions.get(existing).and_then(|s| s.slot);
            if slot.is_none() {
                let free = rig.pool.acquire().or_else(|| {
                    rig.sessions
                        .lru_idle_victim(&busy)
                        .filter(|v| *v != existing)
                        .and_then(|v| {
                            evict(rig, v);
                            rig.pool.acquire()
                        })
                });
                if let Some(target) = free {
                    if restore(rig, existing, target) {
                        stats.lock().expect("stats").note_restore();
                        slot = Some(target);
                    } else {
                        // restore marked it Cold; its tokens survive, so the
                        // cold path below re-prefills.
                        rig.pool.release(target);
                    }
                }
            }
            if let Some(slot) = slot {
                let mut extended = base;
                extended.extend_from_slice(&req.continuation);
                if let Ok(plan) = rig.sessions.begin_turn(&mut rig.pool, existing, &extended) {
                    if let Some(sess) = rig.sessions.get_mut(existing) {
                        sess.tokens = extended.clone();
                        sess.convo = req.convo.clone();
                    }
                    if send_event(
                        &req.reply,
                        Event::Accepted {
                            session: existing.0,
                        },
                    )
                    .is_ok()
                    {
                        work[slot.0].remaining_prompt = extended[plan.reused..].to_vec();
                        work[slot.0].next_pos = plan.reused;
                        work[slot.0].decoding = false;
                        slots[slot.0] = Some(InFlight {
                            session: existing,
                            reply: req.reply,
                            produced: 0,
                            max_tokens: req.max_tokens.max(1),
                        });
                        rig.sessions.touch(existing);
                        if rig.gpu.slot_trace() {
                            eprintln!(
                                "[slot-trace] continuation HIT -- session {} reused {} of {} tokens",
                                existing.0,
                                plan.reused,
                                extended.len()
                            );
                        }
                        let mut st = stats.lock().expect("stats");
                        st.note_admitted();
                        st.note_prefix_hit();
                    }
                    return;
                }
            }
        }
        if rig.gpu.slot_trace() {
            eprintln!("[slot-trace] continuation MISS -- falling back to cold prefill");
        }
    }

    // Try to open; if the pool is full, evict the LRU idle session first.
    let id = match rig
        .sessions
        .open(&mut rig.pool, &mut rig.adm, rig.cap_tokens)
    {
        Ok(id) => id,
        Err(_) => {
            let victim = rig.sessions.lru_idle_victim(&busy);
            match victim {
                Some(v) => {
                    if !evict(rig, v) {
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                reason: "eviction failed".to_string(),
                            },
                        );
                        stats.lock().expect("stats").note_rejected();
                        return;
                    }
                    stats.lock().expect("stats").note_eviction();
                    match rig
                        .sessions
                        .open(&mut rig.pool, &mut rig.adm, rig.cap_tokens)
                    {
                        Ok(id) => id,
                        Err(e) => {
                            let _ = send_event(
                                &req.reply,
                                Event::Rejected {
                                    reason: format!("{e:?}"),
                                },
                            );
                            stats.lock().expect("stats").note_rejected();
                            return;
                        }
                    }
                }
                None => {
                    // Every resident session is generating. Reject with a
                    // reason rather than preempting one or queueing forever.
                    let _ = send_event(
                        &req.reply,
                        Event::Rejected {
                            reason: "all slots busy".to_string(),
                        },
                    );
                    stats.lock().expect("stats").note_rejected();
                    return;
                }
            }
        }
    };

    let slot = match rig.sessions.get(id).and_then(|s| s.slot) {
        Some(s) => s,
        None => {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    reason: "admitted session holds no slot".to_string(),
                },
            );
            stats.lock().expect("stats").note_rejected();
            return;
        }
    };

    // Reset the slot's recurrent state before reusing it. `seq_len = 0` clears
    // the KV, but DeltaNet state lives OUTSIDE the KV arena, so without this a
    // new conversation inherits the previous occupant's recurrent state --
    // which shows up as degenerate or echoed output on every request after the
    // first. Same trap as the swap unit: KV alone is not the whole state.
    if let Err(e) = rig.dn_states[slot.0].reset(&mut rig.gpu) {
        let _ = send_event(
            &req.reply,
            Event::Rejected {
                reason: format!("state reset failed: {e}"),
            },
        );
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        stats.lock().expect("stats").note_rejected();
        return;
    }

    // Prefix reuse: a fresh session reuses nothing, a continued one reuses its
    // common prefix. Either way the suffix is what gets prefilled.
    let plan = match rig
        .sessions
        .begin_turn(&mut rig.pool, id, &req.prompt_tokens)
    {
        Ok(p) => p,
        Err(e) => {
            let _ = send_event(&req.reply, Event::Rejected { reason: e });
            rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
            stats.lock().expect("stats").note_rejected();
            return;
        }
    };
    if let Some(sess) = rig.sessions.get_mut(id) {
        sess.tokens = req.prompt_tokens.clone();
        sess.convo = req.convo.clone();
    }

    if send_event(&req.reply, Event::Accepted { session: id.0 }).is_err() {
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        return;
    }

    work[slot.0].remaining_prompt = req.prompt_tokens[plan.reused..].to_vec();
    work[slot.0].next_pos = plan.reused;
    work[slot.0].decoding = false;
    slots[slot.0] = Some(InFlight {
        session: id,
        reply: req.reply,
        produced: 0,
        max_tokens: req.max_tokens.max(1),
    });
    stats.lock().expect("stats").note_admitted();
}

/// Capture an idle session's state, park it, and free its slot.
///
/// Returns false if anything went wrong, in which case the session is marked
/// `Cold`: its tokens survive, so it re-prefills next time. Slow, never wrong.
fn evict(rig: &mut Rig, victim: SessionId) -> bool {
    let Some(sess) = rig.sessions.get(victim) else {
        return false;
    };
    let Some(slot) = sess.slot else { return false };
    let tokens = sess.tokens.clone();

    let dn_refs = dn_buffers(&rig.dn_states[slot.0]);
    let snap = capture_slot(
        &mut rig.gpu,
        &rig.pool,
        slot,
        &rig.k_arenas,
        &rig.v_arenas,
        &dn_refs,
        &tokens,
        rig.stamp,
    );
    drop(dn_refs);

    match snap {
        Ok(snap) => match rig.swap.park(victim.0, snap) {
            Ok(()) => {
                rig.sessions.mark_swapped(&mut rig.pool, victim);
                true
            }
            Err(_) => {
                rig.sessions.mark_cold(&mut rig.pool, victim);
                true
            }
        },
        Err(_) => {
            rig.sessions.mark_cold(&mut rig.pool, victim);
            true
        }
    }
}

/// Restore a previously swapped session into `slot`. Any failure marks it
/// `Cold` so the caller re-prefills from tokens.
fn restore(rig: &mut Rig, id: SessionId, slot: SlotId) -> bool {
    match rig.swap.unpark(id.0) {
        Ok(snap) => {
            let dn_refs = dn_buffers(&rig.dn_states[slot.0]);
            let r = restore_slot(
                &mut rig.gpu,
                &mut rig.pool,
                slot,
                &rig.k_arenas,
                &rig.v_arenas,
                &dn_refs,
                &snap,
                rig.stamp,
            );
            drop(dn_refs);
            match r {
                Ok(()) => {
                    rig.sessions.mark_resident(id, slot, snap.seq_len);
                    true
                }
                Err(_) => {
                    rig.sessions.mark_cold(&mut rig.pool, id);
                    false
                }
            }
        }
        Err(_) => {
            rig.sessions.mark_cold(&mut rig.pool, id);
            false
        }
    }
}
