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
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::serve::{
    send_event, Continuation, DoneReason, EngineStats, Event, SubmitRequest,
};
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

/// Internal control plane for the engine thread. Public methods map onto these
/// messages; the GPU rig stays exclusively on that thread.
///
/// Continuous-batch scheduler integration is deferred — multi-slot is an
/// alternate single-weight daemon path, not ContinuousBatchScheduler.
enum EngineCommand {
    Submit(SubmitRequest),
    Close {
        session: u64,
        reply: Sender<Result<(), String>>,
    },
    Reset {
        reply: Sender<Result<(), String>>,
    },
}

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
    tx: Option<Sender<EngineCommand>>,
    stats: Arc<Mutex<EngineStats>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl SlotEngine {
    pub fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Submit(req))
            .map_err(|_| "engine thread is gone".to_string())
    }

    /// Close a session synchronously. If the session is mid-generation its
    /// in-flight request is cancelled (InFlight removed, reply sender dropped,
    /// work cleared), swap forgotten and SessionTable/pool/admission closed,
    /// then acknowledged. Idle close does the same. Never silently ignored.
    pub fn close(&self, session: u64) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Close {
                session,
                reply: reply_tx,
            })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    /// Drop every session and swap entry, returning the engine to a cold empty
    /// table. Rejects while any slot is still generating.
    pub fn reset(&self) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Reset { reply: reply_tx })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    pub fn stats(&self) -> EngineStats {
        *self.stats.lock().expect("stats mutex poisoned")
    }

    /// Build the rig on a new thread and start serving. Returns once the model
    /// is loaded, so a caller that gets `Ok` can submit immediately.
    pub fn spawn(cfg: EngineConfig) -> Result<SlotEngine, String> {
        Self::spawn_inner(cfg, None)
    }

    /// Start serving from a source opened and admitted by the daemon.
    ///
    /// The worker consumes this source directly; it does not reopen the model
    /// path after daemon teardown.
    pub fn spawn_with_source(cfg: EngineConfig, source: ModelSource) -> Result<SlotEngine, String> {
        Self::spawn_inner(cfg, Some(source))
    }

    fn spawn_inner(cfg: EngineConfig, source: Option<ModelSource>) -> Result<SlotEngine, String> {
        let (tx, rx) = channel::<EngineCommand>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);

        let handle = std::thread::Builder::new()
            .name("hipfire-slot-engine".to_string())
            .spawn(move || -> Result<(), String> {
                let rig = match source {
                    Some(source) => Rig::build(&cfg, Some(source)),
                    None => Rig::build(&cfg, None),
                };
                match rig {
                    Ok(rig) => {
                        let _ = ready_tx.send(Ok(()));
                        run_loop(rig, rx, stats_thread)
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        Err(e)
                    }
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
    /// Consuming shutdown that closes the channel, drains and fails any
    /// in-flight requests, joins the engine thread and returns GPU teardown
    /// or poison errors. The daemon can call this and withhold `unloaded`
    /// on failure. Graph is destroyed before buffers.
    pub fn shutdown(mut self) -> Result<(), String> {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(res) => res,
                Err(_) => Err("engine thread panicked".to_string()),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for SlotEngine {
    fn drop(&mut self) {
        // Best-effort teardown for non-shutdown paths. Closing the channel
        // signals the engine loop; join releases VRAM before Drop returns.
        // Logs but does not propagate teardown/poison errors — call
        // `shutdown()` for checked teardown.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("hipfire-slot-engine teardown: {e}"),
                Err(_) => eprintln!("hipfire-slot-engine teardown: engine thread panicked"),
            }
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
    /// Build the GPU rig.
    ///
    /// CPU arch/tensor preflight precedes this loader — `SlotBackend::cpu_preflight_source`
    /// opens the HFQ, validates `arch_id` 5|6, rejects vision tensors/models and
    /// parses config/tokenizer before this GPU path. This function assumes that
    /// preflight has passed; its `get_vram_info` + `preflight_alloc` is the
    /// actual-device VRAM admission. Keep that preflight on the live device
    /// (not a hardcoded budget).
    ///
    /// Post-weight allocation is transactional: K/V arenas, DN states, desc
    /// staging, PBS, scratch, logits/out are all owned after `Qwen35Weights`.
    /// On any `?`/error, all completed stages are freed, weights freed,
    /// caches/graph state invalidated and pool drained, so no VRAM leaks.
    fn build(cfg: &EngineConfig, source: Option<ModelSource>) -> Result<Rig, String> {
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::tokenizer::Tokenizer;
        use rdna_compute::kv_slots::preflight_alloc;

        let mut hfq = match source {
            Some(ModelSource::Hfq(hfq)) => hfq,
            Some(ModelSource::Dir(_)) => {
                return Err("SlotEngine requires an HFQ source".to_string())
            }
            None => HfqFile::open(&cfg.model_path).map_err(|e| format!("open model: {e}"))?,
        };
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
        let cap_rounded = cfg.cap_tokens.div_ceil(128) * 128;

        // Size from retained file descriptor, not a later path stat (TOCTOU).
        // When the source was admitted, the file was already opened and its
        // identity/size captured. Using `hfq.file_len()` describes the retained
        // inode, so a delete/replace of the path after admission does not change
        // the planned VRAM budget or read a different file.
        let weight_bytes = hfq.file_len();
        let kv_bytes = (n_fa_layers as u64)
            * 2
            * (cfg.n_slots as u64)
            * (cap_rounded as u64)
            * (per_pos_bytes as u64);
        let planned = weight_bytes + kv_bytes + 768 * 1024 * 1024;

        // GPU context only before preflight — no device allocations yet. Free/
        // total come from the live device so gfx1100 is not over-admitted
        // against a hardcoded R9700 budget.
        let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
        let (vram_free, vram_total) = gpu
            .hip
            .get_vram_info()
            .map_err(|e| format!("vram info: {e}"))?;
        preflight_alloc(planned, vram_free as u64, "SlotEngine")
            .map_err(|e| format!("preflight refused: {e}"))?;
        let weights: Qwen35Weights = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .map_err(|e| format!("load weights: {e}"))?;

        // ── Transactional post-weight guard ────────────────────────────────
        // Every allocation after weights is owned here. On any error, Drop
        // frees all completed stages in reverse construction order
        // (out_tokens → logits → scratch → pbs → staging → dn → arenas →
        // weights → caches/pool) so no prior stage leaks. Actual-device
        // preflight above is retained; CPU arch/tensor preflight precedes
        // this loader (see doc above).
        struct Guard {
            gpu: Option<Gpu>,
            weights: Option<Qwen35Weights>,
            k_arenas: Vec<GpuTensor>,
            v_arenas: Vec<GpuTensor>,
            dn_states: Vec<DeltaNetState>,
            desc_staging: Option<SlotDescStaging>,
            pbs: Option<PrefillBatchScratch>,
            scratch: Option<Qwen35Scratch>,
            logits_out: Option<GpuTensor>,
            out_tokens: Option<GpuTensor>,
            committed: bool,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                if self.committed {
                    return;
                }
                if let Some(mut gpu) = self.gpu.take() {
                    if let Some(t) = self.out_tokens.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(t) = self.logits_out.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(s) = self.scratch.take() {
                        let _ = s.free_gpu(&mut gpu);
                    }
                    if let Some(p) = self.pbs.take() {
                        let _ = p.free_gpu(&mut gpu);
                    }
                    if let Some(d) = self.desc_staging.take() {
                        d.free_gpu(&mut gpu);
                    }
                    for dn in self.dn_states.drain(..).rev() {
                        dn.free_gpu(&mut gpu);
                    }
                    while let Some(v) = self.v_arenas.pop() {
                        let _ = gpu.free_tensor(v);
                    }
                    while let Some(k) = self.k_arenas.pop() {
                        let _ = gpu.free_tensor(k);
                    }
                    if let Some(w) = self.weights.take() {
                        w.free_gpu(&mut gpu);
                    }
                    gpu.invalidate_weight_caches();
                    gpu.invalidate_graph_state();
                    gpu.drain_pool();
                }
            }
        }
        let mut g = Guard {
            gpu: Some(gpu),
            weights: Some(weights),
            k_arenas: Vec::with_capacity(n_fa_layers),
            v_arenas: Vec::with_capacity(n_fa_layers),
            dn_states: Vec::with_capacity(cfg.n_slots),
            desc_staging: None,
            pbs: None,
            scratch: None,
            logits_out: None,
            out_tokens: None,
            committed: false,
        };
        let pool = SlotPool::new(cfg.n_slots, cfg.cap_tokens, per_pos_bytes)
            .map_err(|e| format!("SlotPool: {e}"))?;
        let arena_bytes = pool.arena_bytes();
        for _ in 0..n_fa_layers {
            let k = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[arena_bytes], DType::Raw)
                .map_err(|e| format!("k arena: {e}"))?;
            g.k_arenas.push(k);
            let v = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[arena_bytes], DType::Raw)
                .map_err(|e| format!("v arena: {e}"))?;
            g.v_arenas.push(v);
        }
        for _ in 0..cfg.n_slots {
            let dn = DeltaNetState::new(g.gpu.as_mut().unwrap(), &config)
                .map_err(|e| format!("dn: {e}"))?;
            g.dn_states.push(dn);
        }
        let desc_staging = SlotDescStaging::new(g.gpu.as_mut().unwrap(), cfg.n_slots, max_batch)
            .map_err(|e| format!("staging: {e}"))?;
        g.desc_staging = Some(desc_staging);
        // Slots do plain prefill only — never tree-verify — so skip the GDN
        // S-tape: at max_batch=cap_tokens it is tens of GB (16k → ~21 GB).
        let pbs = PrefillBatchScratch::new_opt(g.gpu.as_mut().unwrap(), &config, max_batch, false)
            .map_err(|e| format!("pbs: {e}"))?;
        g.pbs = Some(pbs);
        let scratch =
            Qwen35Scratch::new_with_kv_max(g.gpu.as_mut().unwrap(), &config, 64, cfg.cap_tokens)
                .map_err(|e| format!("scratch: {e}"))?;
        g.scratch = Some(scratch);
        let logits_out = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots * config.vocab_size], DType::F32)
            .map_err(|e| format!("logits: {e}"))?;
        g.logits_out = Some(logits_out);
        let out_tokens = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots], DType::F32)
            .map_err(|e| format!("out_tokens: {e}"))?;
        g.out_tokens = Some(out_tokens);
        let sample_params = (0..cfg.n_slots)
            .map(|_| SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
            })
            .collect::<Vec<_>>();

        let dn_bytes: u64 = dn_buffers(&g.dn_states[0])
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
            vram_total as u64,
        );
        adm.set_host_budget(cfg.host_budget_bytes);
        let swap = SwapManager::new(cfg.swap_dir.clone(), cfg.host_budget_bytes)
            .map_err(|e| format!("SwapManager: {e}"))?;

        // Commit guard — prevent Drop cleanup, move out owned GPU state.
        g.committed = true;
        let gpu = g.gpu.take().unwrap();
        let weights = g.weights.take().unwrap();
        let k_arenas = std::mem::take(&mut g.k_arenas);
        let v_arenas = std::mem::take(&mut g.v_arenas);
        let dn_states = std::mem::take(&mut g.dn_states);
        let desc_staging = g.desc_staging.take().unwrap();
        let pbs = g.pbs.take().unwrap();
        let scratch = g.scratch.take().unwrap();
        let logits_out = g.logits_out.take().unwrap();
        let out_tokens = g.out_tokens.take().unwrap();
        // g now committed and empty; Drop is no-op.

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

    /// Release every GPU allocation owned by this rig, then invalidate caches
    /// and drain the HIP pool. Graph handles must be destroyed first so they
    /// never outlive the buffers they capture. Best-effort: keeps going after
    /// individual free failures and returns only the first error for logging.
    fn free_gpu(self, mut graph: SlotDecodeGraph) -> Result<(), String> {
        let Rig {
            mut gpu,
            weights,
            k_arenas,
            v_arenas,
            dn_states,
            desc_staging,
            pbs,
            scratch,
            logits_out,
            out_tokens,
            ..
        } = self;

        let mut first_err: Option<String> = None;
        let mut note_hip = |r: rdna_compute::HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e.to_string());
                }
            }
        };

        // Reverse of construction: graph → out_tokens → logits → scratch →
        // pbs → staging → dn → arenas → weights → caches/pool.
        graph.release(&gpu);
        note_hip(gpu.free_tensor(out_tokens));
        note_hip(gpu.free_tensor(logits_out));
        note_hip(scratch.free_gpu(&mut gpu));
        note_hip(pbs.free_gpu(&mut gpu));
        desc_staging.free_gpu(&mut gpu);
        for dn in dn_states.into_iter().rev() {
            dn.free_gpu(&mut gpu);
        }
        // Arenas were allocated per FA layer as k then v; free reverse layers
        // as v then k.
        let mut k_arenas = k_arenas;
        let mut v_arenas = v_arenas;
        while !k_arenas.is_empty() || !v_arenas.is_empty() {
            if let Some(v) = v_arenas.pop() {
                note_hip(gpu.free_tensor(v));
            }
            if let Some(k) = k_arenas.pop() {
                note_hip(gpu.free_tensor(k));
            }
        }
        weights.free_gpu(&mut gpu);
        gpu.invalidate_weight_caches();
        gpu.invalidate_graph_state();
        gpu.drain_pool();

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// One request currently occupying a slot.
struct InFlight {
    session: SessionId,
    reply: Sender<Event>,
    produced: usize,
    max_tokens: usize,
}

fn run_loop(
    mut rig: Rig,
    rx: Receiver<EngineCommand>,
    stats: Arc<Mutex<EngineStats>>,
) -> Result<(), String> {
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
    let mut poison: Option<String> = None;

    // Helper to fail every active request, close sessions, clear work, forget swap.
    // Used for HIP poison and shutdown drain. Returns the reason used.
    let fail_all_active = |rig: &mut Rig,
                           slots: &mut [Option<InFlight>],
                           work: &mut [PendingWork],
                           reason: String| {
        for (s, f) in slots.iter_mut().enumerate() {
            if let Some(f) = f.take() {
                let _ = send_event(
                    &f.reply,
                    Event::Rejected {
                        reason: reason.clone(),
                    },
                );
                rig.swap.forget(f.session.0);
                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                work[s].remaining_prompt.clear();
                work[s].decoding = false;
                work[s].next_pos = 0;
            }
        }
    };

    'serve: loop {
        let idle = slots.iter().all(|s| s.is_none());
        if idle {
            // Nothing in flight: block rather than spin.
            match rx.recv() {
                Ok(cmd) => handle_command(&mut rig, &mut slots, &mut work, &stats, cmd),
                Err(_) => break 'serve, // all senders gone: shut down after drain
            }
        }
        loop {
            match rx.try_recv() {
                Ok(cmd) => handle_command(&mut rig, &mut slots, &mut work, &stats, cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if slots.iter().all(|s| s.is_none()) {
                        break 'serve;
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
            // Retain per-request behavior only because state is provably
            // closed/reset: every active slot is taken, its session closed,
            // work cleared and swap forgotten, so no poisoned KV/DN survives.
            // If this invariant were broken, we would need to poison instead.
            let reason = e.to_string();
            for (s, f) in slots.iter_mut().enumerate() {
                if let Some(f) = f.take() {
                    let _ = send_event(
                        &f.reply,
                        Event::Rejected {
                            reason: reason.clone(),
                        },
                    );
                    rig.swap.forget(f.session.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    work[s].remaining_prompt.clear();
                    work[s].decoding = false;
                    work[s].next_pos = 0;
                }
            }
            continue;
        }
        if let Err(e) = rig.gpu.hip.device_synchronize() {
            let reason = format!("device synchronize failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        if let Err(e) = rig.gpu.sample_per_slot(
            &rig.logits_out,
            &mut rig.sample_params,
            n,
            rig.config.vocab_size,
            &rig.out_tokens,
        ) {
            let reason = format!("sample failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        if let Err(e) = rig.gpu.hip.device_synchronize() {
            let reason = format!("device synchronize failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        let mut ids = vec![0i32; n];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, n * 4) };
            if let Err(e) = rig.gpu.hip.memcpy_dtoh(bytes, &rig.out_tokens.buf) {
                let reason = format!("DtoH failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
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
            // Context-cap guard: stop before the slot's KV length would pass
            // its cap. Without this, a long multi-turn session overflows
            // set_seq_len and kills the whole batched step.
            let hit_ctx_cap = rig
                .sessions
                .get(session)
                .map(|sess| sess.tokens.len() + 1 >= rig.cap_tokens)
                .unwrap_or(false);

            if gone || hit_eos || hit_max || hit_ctx_cap {
                let reason = if gone {
                    DoneReason::ClientGone
                } else if hit_eos {
                    DoneReason::Eos
                } else {
                    DoneReason::MaxTokens
                };
                // MaxTokens: the final emitted token reached the client but has
                // not yet gone through the next forward. Keep it in the
                // authoritative stream; held seq_len remains one shorter, so
                // begin_turn prefills this token plus the next-turn suffix.
                if should_retain_terminal_tok(reason) {
                    if let Some(sess) = rig.sessions.get_mut(session) {
                        sess.tokens.push(tok);
                    }
                }
                // Terminator and undelivered (gone) tokens are counted by
                // `produced` but never reached the client.
                let generated = if hit_eos || gone {
                    f.produced - 1
                } else {
                    f.produced
                };
                let _ = send_event(&f.reply, Event::Done { reason, generated });
                slots[s] = None;
                work[s].remaining_prompt.clear();
                work[s].decoding = false;
                work[s].next_pos = 0;
                if matches!(reason, DoneReason::ClientGone) {
                    // Nobody will follow up on a vanished client, so hand the
                    // slot back at once.
                    rig.swap.forget(session.0);
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

    // Shutdown/drain: if channel closed or poisoned while slots remain, fail them.
    // This ensures no in-flight cancellation can remain reusable and teardown
    // drains every active request. Normal shutdown drain (no prior poison) is
    // not itself poison — only HIP failures poison.
    let has_active = slots.iter().any(|s| s.is_some());
    if has_active {
        let reason = poison
            .clone()
            .unwrap_or_else(|| "engine shutdown".to_string());
        fail_all_active(&mut rig, &mut slots, &mut work, reason);
    }

    // Free GPU after poisoning/shutdown drain. Graph is destroyed before buffers.
    let free_res = rig.free_gpu(graph);
    match (poison, free_res) {
        (Some(p), Ok(())) => Err(p),
        (Some(p), Err(e)) => Err(format!("{p}; teardown failed: {e}")),
        (None, Ok(())) => Ok(()),
        (None, Err(e)) => Err(e),
    }
}
fn should_retain_terminal_tok(reason: DoneReason) -> bool {
    matches!(reason, DoneReason::MaxTokens)
}

fn handle_command(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    cmd: EngineCommand,
) {
    match cmd {
        EngineCommand::Submit(req) => admit(rig, slots, work, stats, req),
        EngineCommand::Close { session, reply } => {
            let sid = SessionId(session);
            // At the engine-loop command boundary: if session is in flight,
            // cancel it — remove InFlight, drop its reply sender (Event channel),
            // clear work, forget swap, close SessionTable/pool/admission, then ack.
            // Idle close does the same forget/close. No ignored close.
            if let Some(idx) = slots
                .iter()
                .position(|s| s.as_ref().map(|f| f.session == sid).unwrap_or(false))
            {
                if let Some(inflight) = slots[idx].take() {
                    drop(inflight.reply);
                }
                work[idx].remaining_prompt.clear();
                work[idx].decoding = false;
                work[idx].next_pos = 0;
            }
            rig.swap.forget(session);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, sid);
            let _ = reply.send(Ok(()));
        }
        EngineCommand::Reset { reply } => {
            if slots.iter().any(|s| s.is_some()) {
                let _ = reply.send(Err("reset rejected: requests in flight".to_string()));
                return;
            }
            let ids: Vec<u64> = rig.sessions.iter().map(|(id, _)| id).collect();
            for id in ids {
                rig.swap.forget(id);
                rig.sessions
                    .close(&mut rig.pool, &mut rig.adm, SessionId(id));
            }
            let _ = reply.send(Ok(()));
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
    if !req.continuation.tokens().is_empty() {
        if rig.gpu.slot_trace() {
            let shape = match &req.continuation {
                Continuation::ToolResults { session, .. } => {
                    format!("tool-results of session {session}")
                }
                _ => "new user turn".to_string(),
            };
            eprintln!(
                "[slot-trace] continuation attempt ({shape}): convo={:?} suffix={} tokens",
                req.convo,
                req.continuation.tokens().len(),
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
        // The suffix shape decides which match is legal — never both. A user
        // turn searches for the session one turn behind; tool results confirm
        // the session that emitted the calls being answered. Either way the
        // suffix is appended to the session's exact stored tokens, so the KV
        // and DeltaNet state extend strictly.
        let matched = match &req.continuation {
            Continuation::Cold => None,
            Continuation::UserTurn(_) => rig.sessions.find_continuation(&req.convo, &busy),
            Continuation::ToolResults { session, .. } => {
                rig.sessions
                    .confirm_reentry(SessionId(*session), &req.convo, &busy)
            }
        };
        if let Some(existing) = matched {
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
                extended.extend_from_slice(req.continuation.tokens());
                // Prefill-side context-cap guard (mirrors decode hit_ctx_cap).
                // A prompt/extension already at the KV cap overflows set_seq_len
                // during prefill before any decode step can stop it.
                if extended.len() >= rig.cap_tokens {
                    let _ = send_event(
                        &req.reply,
                        Event::Done {
                            reason: DoneReason::MaxTokens,
                            generated: 0,
                        },
                    );
                    return;
                }
                if let Ok(plan) = rig.sessions.begin_turn(&mut rig.pool, existing, &extended) {
                    if let Some(sess) = rig.sessions.get_mut(existing) {
                        sess.tokens = extended.clone();
                        sess.convo = req.convo.clone();
                    }
                    if send_event(
                        &req.reply,
                        Event::Accepted {
                            session: existing.0,
                            reused: plan.reused,
                            prefill: extended.len() - plan.reused,
                        },
                    )
                    .is_err()
                    {
                        // Mutated session must not remain reusable — forget swap,
                        // close SessionTable/pool/admission so no poisoned KV/DN
                        // state survives. Preserve typed Continuation semantics
                        // and max-terminal-token accounting (caller retains base
                        // for retransmission).
                        rig.swap.forget(existing.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                        return;
                    }
                    work[slot.0].remaining_prompt = extended[plan.reused..].to_vec();
                    work[slot.0].next_pos = plan.reused;
                    work[slot.0].decoding = false;
                    rig.sample_params[slot.0] = SlotSampleParams {
                        temperature: req.temperature,
                        top_p: req.top_p,
                        top_k: req.top_k,
                        seed: req.seed,
                    };
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
                    return;
                }
            }
        }
        if rig.gpu.slot_trace() {
            eprintln!("[slot-trace] continuation MISS -- falling back to cold prefill");
        }
    }

    // Prefill-side context-cap guard for cold admits. Same wire reporting as
    // the decode path (DoneReason::MaxTokens → finish_reason "length").
    if req.prompt_tokens.len() >= rig.cap_tokens {
        let _ = send_event(
            &req.reply,
            Event::Done {
                reason: DoneReason::MaxTokens,
                generated: 0,
            },
        );
        return;
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

    if send_event(
        &req.reply,
        Event::Accepted {
            session: id.0,
            reused: plan.reused,
            prefill: req.prompt_tokens.len() - plan.reused,
        },
    )
    .is_err()
    {
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        return;
    }

    work[slot.0].remaining_prompt = req.prompt_tokens[plan.reused..].to_vec();
    work[slot.0].next_pos = plan.reused;
    work[slot.0].decoding = false;
    rig.sample_params[slot.0] = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
    };
    slots[slot.0] = Some(InFlight {
        session: id,
        reply: req.reply,
        produced: 0,
        max_tokens: req.max_tokens.max(1),
    });
    if rig.gpu.slot_trace() {
        eprintln!(
            "[slot-trace] cold ADMIT session {} slot={} prompt={} tokens",
            id.0,
            slot.0,
            req.prompt_tokens.len()
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure stand-in for the MaxTokens bookkeeping in `run_loop`: only
    /// MaxTokens keeps the terminal emitted tok in the session transcript.
    fn apply_terminal(tokens: &mut Vec<u32>, tok: u32, reason: DoneReason) {
        if should_retain_terminal_tok(reason) {
            tokens.push(tok);
        }
    }

    #[test]
    fn max_tokens_retains_final_emitted_tok_for_continuation() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 99, DoneReason::MaxTokens);
        assert_eq!(tokens, vec![10, 20, 99]);
    }

    #[test]
    fn eos_and_client_gone_do_not_retain_terminal_tok() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 1, DoneReason::Eos);
        apply_terminal(&mut tokens, 2, DoneReason::ClientGone);
        assert_eq!(tokens, vec![10, 20]);
    }
}
