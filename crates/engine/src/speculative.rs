//! Speculative decoding infrastructure for hipfire.
//!
//! Phase 1: holds target + draft model slots side-by-side on a single shared
//! `Gpu`. The actual speculative decode loop (draft → verify → accept) lives
//! in `spec_loop` once Phase 2 lands. For now, each slot just supports
//! independent forward passes so we can validate that loading two models at
//! once works and that both produce coherent output.
//!
//! Both slots share the same `Gpu` instance — HIP kernels run serialized on
//! the default stream, and the MQ rotation scratch buffers on `Gpu` are reused
//! across calls. This is correct as long as we never have two in-flight GEMVs
//! on different models sharing the same MQ scratch (which we won't, since
//! speculative decode serializes draft-generate then target-verify).

use crate::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use crate::hfq::HfqFile;
use crate::llama::{self, KvCache};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::tokenizer::Tokenizer;
use hip_bridge::{DeviceBuffer, HipResult};
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Task #93 Phase B seed-prediction oracle counters.
///
/// Three proxies, all derived from data the draft already computes (zero
/// extra device work). For each cycle:
///   - REJ_BOUNDARY: `drafted[accept_len + 1] == bonus_token` (PRD's "naive"
///     proxy — argmax at rejection position). Zero-by-construction when
///     `accept_len < b - 1` because the accept loop broke precisely because
///     those didn't match. Reported anyway to document the dead-end.
///   - TAIL: `drafted[b - 1] == bonus_token`. Draft's final-position argmax.
///     Gives a non-zero signal. If the usual case is "target's bonus happens
///     at position b-1 because accept_len = b-2", this proxy catches those.
///   - ANYPOS: `bonus_token ∈ drafted[1..b]`. Upper bound of any position-
///     based single-guess proxy. Useful as a ceiling.
///
/// FULLACCEPT counts cycles where `accept_len == b - 1` (full acceptance —
/// draft has no native prediction at position `b`, so REJ_BOUNDARY is
/// undefined and TAIL/ANYPOS are the only candidates there).
///
static SEED_ORACLE_TOTAL: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_REJ_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_TAIL_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ANYPOS_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_FULLACCEPT: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ACCEPT_LEN_SUM: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub struct SeedOracleStats {
    pub total: u64,
    pub rej_match: u64,
    pub tail_match: u64,
    pub anypos_match: u64,
    pub full_accept: u64,
    pub accept_len_sum: u64,
}

/// Snapshot the process-global seed-oracle counters.
pub fn read_seed_oracle_stats() -> SeedOracleStats {
    SeedOracleStats {
        total: SEED_ORACLE_TOTAL.load(Ordering::Relaxed),
        rej_match: SEED_ORACLE_REJ_MATCH.load(Ordering::Relaxed),
        tail_match: SEED_ORACLE_TAIL_MATCH.load(Ordering::Relaxed),
        anypos_match: SEED_ORACLE_ANYPOS_MATCH.load(Ordering::Relaxed),
        full_accept: SEED_ORACLE_FULLACCEPT.load(Ordering::Relaxed),
        accept_len_sum: SEED_ORACLE_ACCEPT_LEN_SUM.load(Ordering::Relaxed),
    }
}

/// Zero all seed-oracle counters. Call before a fresh generation run.
pub fn reset_seed_oracle_stats() {
    SEED_ORACLE_TOTAL.store(0, Ordering::Relaxed);
    SEED_ORACLE_REJ_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_TAIL_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_ANYPOS_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_FULLACCEPT.store(0, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.store(0, Ordering::Relaxed);
}

/// Parse HIPFIRE_DDTREE_LOGW_CUTOFF. Positive value X means "stop tree
/// expansion when next candidate's cumulative logw < -X". 0.0 / unset /
/// unparseable disables (= expand all the way to `budget`).
fn ddtree_logw_cutoff() -> f32 {
    match std::env::var("HIPFIRE_DDTREE_LOGW_CUTOFF")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        Some(x) if x > 0.0 => -x,
        _ => f32::NEG_INFINITY,
    }
}

/// DDTree meta-verifier pruner telemetry: per-cycle tree-size histogram.
/// `cycle_count` = cycles observed; `total_nodes` = sum of tree.num_nodes()
/// across cycles; `max_nodes` / `min_nodes` = range observed.
static DDTREE_META_CYCLES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_TOTAL_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MAX_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MIN_NODES: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn record_ddtree_meta_nodes(n: usize) {
    let n64 = n as u64;
    DDTREE_META_CYCLES.fetch_add(1, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.fetch_add(n64, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.fetch_max(n64, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.fetch_min(n64, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DdtreeMetaStats {
    pub cycles: u64,
    pub total_nodes: u64,
    pub max_nodes: u64,
    pub min_nodes: u64,
}

pub fn read_ddtree_meta_stats() -> DdtreeMetaStats {
    let c = DDTREE_META_CYCLES.load(Ordering::Relaxed);
    DdtreeMetaStats {
        cycles: c,
        total_nodes: DDTREE_META_TOTAL_NODES.load(Ordering::Relaxed),
        max_nodes: DDTREE_META_MAX_NODES.load(Ordering::Relaxed),
        min_nodes: if c == 0 { 0 } else { DDTREE_META_MIN_NODES.load(Ordering::Relaxed) },
    }
}

pub fn reset_ddtree_meta_stats() {
    DDTREE_META_CYCLES.store(0, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.store(u64::MAX, Ordering::Relaxed);
}

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// INT8 co-located K and V (default).
    Q8,
    /// Asym4: rotated 4-bit K + Q8 V (smaller than Q8, higher-fidelity than asym3).
    Asym4,
    /// Asym3: rotated 3-bit K + Q8 V. ~2.7× less KV BW than Q8, tightly-tuned
    /// kernel for the hot FA attention path. Good choice for long-context verify.
    Asym3,
    /// Asym2: rotated 2-bit K + Q8 V. Smallest but most lossy.
    Asym2,
}

impl Default for KvMode {
    fn default() -> Self {
        KvMode::Q8
    }
}

/// Configuration for loading a single model slot.
#[derive(Debug, Clone)]
pub struct ModelSlotConfig {
    pub max_seq: usize,
    pub kv_mode: KvMode,
    pub repeat_window: usize,
    pub state_quant: qwen35::StateQuant,
}

impl Default for ModelSlotConfig {
    fn default() -> Self {
        Self {
            max_seq: 2048,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant: qwen35::StateQuant::Q8,
        }
    }
}

/// A single loaded Qwen3.5 model with its own KV cache, DeltaNet state, and
/// forward-pass scratch. The `Gpu` is borrowed, not owned — multiple slots
/// share one `Gpu` instance.
pub struct ModelSlot {
    pub name: String,
    pub hfq: HfqFile,
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    pub scratch: Qwen35Scratch,
    pub slot_config: ModelSlotConfig,
}

impl ModelSlot {
    /// Load a model from `path` into a slot. The caller-supplied `gpu` is used
    /// for all allocations. `name` is a human-readable label used in logs.
    pub fn load(
        gpu: &mut Gpu,
        path: &Path,
        name: impl Into<String>,
        slot_config: ModelSlotConfig,
    ) -> HipResult<Self> {
        let name = name.into();
        let hfq = HfqFile::open(path).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("open {} ({}): {}", path.display(), name, e))
        })?;
        let config = qwen35::config_from_hfq(&hfq).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("invalid Qwen3.5 config in {} ({})", path.display(), name))
        })?;
        let weights = qwen35::load_weights(&hfq, &config, gpu)?;

        let n_kv_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::FullAttention)
            .count();

        // Honor the caller's requested KV cache mode. Default is Q8 for
        // backwards-compat, but DFlash verify is KV-bandwidth sensitive at
        // longer contexts — asym3/asym4 cut the verify attention cost.
        let kv_cache = match slot_config.kv_mode {
            KvMode::Q8 => KvCache::new_gpu_q8(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym4 => KvCache::new_gpu_asym4(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym3 => KvCache::new_gpu_asym3(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym2 => KvCache::new_gpu_asym2(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
        };

        let dn_state = DeltaNetState::new_with_quant(gpu, &config, slot_config.state_quant)?;
        let scratch = Qwen35Scratch::new(gpu, &config, slot_config.repeat_window)?;

        Ok(Self {
            name,
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config,
        })
    }

    /// Load the tokenizer from this slot's HFQ metadata. Each slot technically
    /// carries its own tokenizer; callers should validate that two slots'
    /// tokenizers are compatible via `Tokenizer::is_compatible_with` before
    /// sharing.
    pub fn load_tokenizer(&self) -> Option<Tokenizer> {
        Tokenizer::from_hfq_metadata(&self.hfq.metadata_json)
    }

    /// Single-token forward pass. Writes logits into `self.scratch.logits`.
    pub fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()> {
        qwen35::forward_scratch(
            gpu,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
        )
    }

    /// Reset the DeltaNet recurrent state and zero the KV write head.
    /// Does NOT shrink the KV allocation — callers track `seq_pos` separately.
    pub fn reset_state(&mut self, gpu: &mut Gpu) {
        // Use stream-ordered memset when an active_stream is set (hot path
        // inside spec_step_dflash) to avoid null-stream host stalls. ~48
        // memsets/cycle on 27B when draft rollback triggers a reset.
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.dn_state.s_matrices {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.dn_state.s_scales {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.dn_state.conv_states {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
            }
            None => {
                for s in &self.dn_state.s_matrices {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.dn_state.s_scales {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.dn_state.conv_states {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        }
    }
}

/// A pair of target + draft slots sharing one `Gpu` and one tokenizer.
///
/// Phase 1 just carries both slots. Phase 2+ adds the `spec_decode_step`
/// method for the verify-and-accept loop.
pub struct SpecPair {
    pub target: ModelSlot,
    pub draft: ModelSlot,
    pub tokenizer: Tokenizer,
}

impl SpecPair {
    /// Load target and draft from separate HFQ files on the same `Gpu`.
    /// Validates that the two models share a compatible tokenizer before
    /// returning — speculative decode requires identical vocab + token IDs.
    pub fn load(
        gpu: &mut Gpu,
        target_path: &Path,
        draft_path: &Path,
        target_cfg: ModelSlotConfig,
        draft_cfg: ModelSlotConfig,
    ) -> HipResult<Self> {
        let target = ModelSlot::load(gpu, target_path, "target", target_cfg)?;
        let draft = ModelSlot::load(gpu, draft_path, "draft", draft_cfg)?;

        let target_tok = target.load_tokenizer().ok_or_else(|| {
            hip_bridge::HipError::new(0, "target model has no tokenizer in HFQ metadata")
        })?;
        let draft_tok = draft.load_tokenizer().ok_or_else(|| {
            hip_bridge::HipError::new(0, "draft model has no tokenizer in HFQ metadata")
        })?;

        if target_tok.vocab_size() != draft_tok.vocab_size() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer mismatch: target vocab={}, draft vocab={}. \
                     Speculative decode requires identical vocabularies.",
                    target_tok.vocab_size(),
                    draft_tok.vocab_size()
                ),
            ));
        }

        // Sanity-check a round-trip on a common string — catches vocab-size
        // match but token-ID mismatch (different BPE merges producing same
        // vocab count).
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        let a = target_tok.encode(probe);
        let b = draft_tok.encode(probe);
        if a != b {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer merge rules diverge: target={:?}, draft={:?}. \
                     Speculative decode requires identical tokenization.",
                    &a, &b
                ),
            ));
        }

        Ok(Self {
            target,
            draft,
            tokenizer: target_tok,
        })
    }

    /// Run a minimal smoke test: 8 forward passes on each slot with a dummy
    /// token sequence, ensuring neither model crashes and the logits buffers
    /// contain finite values. Returns `(target_ok, draft_ok)`.
    pub fn smoke_test(&mut self, gpu: &mut Gpu) -> HipResult<(bool, bool)> {
        // Token ID 1 is a safe placeholder for both Qwen3 and Qwen3.5; the
        // smoke test only checks that the forward pass runs without crashing
        // and produces finite logits.
        let probe_token: u32 = 1;
        for pos in 0..8 {
            self.target.forward(gpu, probe_token, pos)?;
        }
        for pos in 0..8 {
            self.draft.forward(gpu, probe_token, pos)?;
        }
        let target_logits = gpu.download_f32(&self.target.scratch.logits)?;
        let draft_logits = gpu.download_f32(&self.draft.scratch.logits)?;
        let target_ok = target_logits.iter().take(1024).all(|x| x.is_finite());
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());

        // Reset both after the smoke test so the caller starts from a clean
        // state at seq_pos=0.
        self.target.reset_state(gpu);
        self.draft.reset_state(gpu);

        Ok((target_ok, draft_ok))
    }
}

/// Result of one speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of draft tokens accepted (0..=k).
    pub accepted: usize,
    /// Target's next-token prediction at the first rejection point (or after
    /// all drafted tokens if accepted == k). Appended to `committed`.
    pub bonus_token: u32,
    /// The full sequence of tokens the draft proposed this cycle.
    pub drafted: Vec<u32>,
    /// The tokens actually committed to both models: `drafted[..accepted]`
    /// followed by `bonus_token`. Always non-empty (length = accepted + 1).
    pub committed: Vec<u32>,
}

/// Backing storage for a DeltaNetState snapshot. Holds device buffers sized
/// to match the source state's tensors. Allocate once per slot, reuse across
/// all speculative cycles.
pub struct DeltaNetSnapshot {
    s_matrix_bufs: Vec<DeviceBuffer>,
    s_scale_bufs: Vec<DeviceBuffer>,
    conv_state_bufs: Vec<DeviceBuffer>,
}

impl DeltaNetSnapshot {
    /// Allocate backup buffers matching `state`'s shapes.
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState) -> HipResult<Self> {
        let mut s_matrix_bufs = Vec::with_capacity(state.s_matrices.len());
        for t in &state.s_matrices {
            s_matrix_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut s_scale_bufs = Vec::with_capacity(state.s_scales.len());
        for t in &state.s_scales {
            s_scale_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut conv_state_bufs = Vec::with_capacity(state.conv_states.len());
        for t in &state.conv_states {
            conv_state_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        Ok(Self {
            s_matrix_bufs,
            s_scale_bufs,
            conv_state_bufs,
        })
    }

    /// Copy live state → backup.
    pub fn save_from(&mut self, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        Ok(())
    }

    /// Copy backup → live state (rewinds the recurrent state to the snapshot point).
    pub fn restore_to(&self, state: &mut DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (src, dst) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        Ok(())
    }
}

/// A series of `n_slots` `DeltaNetSnapshot` slots, used by the tape-replay
/// rollback path. After each verify forward step writes its post-state into
/// the next slot, `restore_from(accept_len + 1)` jumps the live DN state
/// to exactly `start + accept_len + 1` positions of advance — no replay
/// loop needed.
///
/// VRAM cost: `n_slots × (one DeltaNetSnapshot)`. For Qwen3.5-4B and
/// `n_slots = B + 1 = 17`, that's roughly 100 MB; for 9B it scales with
/// the hybrid layer count.
pub struct DeltaNetTape {
    pub slots: Vec<DeltaNetSnapshot>,
}

/// Innovation tape for the GatedDeltaNet recurrence. During a batched verify
/// forward we capture the per-LA-layer pre-conv1d `qkv` projection and the
/// post-sigmoid `(α, β)` for every block position. On rollback we replay
/// conv1d + QK-norm + repeat-interleave + GDN for `accept_len + 1` steps
/// against the pre-verify DN snapshot — advancing both S-state AND
/// conv_state correctly, no full target re-run needed.
///
/// Why pre-conv1d qkv instead of post-conv1d (q, k, v): conv_state is a
/// recurrent buffer advanced by conv1d_silu_split. If we skipped conv1d on
/// replay the next verify would see a stale conv_state reflecting the
/// previous full-B aborted trajectory rather than the accepted prefix —
/// small numerical drift that empirically halves τ on our 4B hybrid target.
/// Running conv1d from the captured qkv advances conv_state to the right
/// place.
pub struct GdnTape {
    pub max_n: usize,
    pub qkv_dim: usize,
    pub v_dim: usize,
    pub k_dim: usize,
    pub n_v_heads: usize,
    pub n_key_heads: usize,
    pub value_head_dim: usize,
    pub key_head_dim: usize,
    /// Per-LA-layer [max_n × qkv_dim] F32 — raw qkvza projection output.
    pub qkv_bufs: Vec<GpuTensor>,
    /// Per-LA-layer [max_n × n_v_heads] F32 — post-sigmoid_alpha_gate.
    pub alpha_bufs: Vec<GpuTensor>,
    pub beta_bufs: Vec<GpuTensor>,
    /// Replay scratch (shared across layers — serial replay is fine).
    pub q_raw_scratch: GpuTensor,   // [max_n × k_dim]
    pub k_raw_scratch: GpuTensor,   // [max_n × k_dim]
    pub v_scratch: GpuTensor,       // [max_n × v_dim]
    pub q_scratch: GpuTensor,       // [max_n × v_dim] (post repeat-interleave)
    pub k_scratch: GpuTensor,       // [max_n × v_dim]
    pub attn_scratch: GpuTensor,    // [max_n × v_dim]
}

impl GdnTape {
    pub fn new_for_config(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
    ) -> HipResult<Self> {
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let n_key_heads = config.linear_num_key_heads;
        let n_la_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::LinearAttention)
            .count();

        let mut qkv_bufs = Vec::with_capacity(n_la_layers);
        let mut alpha_bufs = Vec::with_capacity(n_la_layers);
        let mut beta_bufs = Vec::with_capacity(n_la_layers);
        for _ in 0..n_la_layers {
            qkv_bufs.push(gpu.alloc_tensor(&[max_n * qkv_dim], rdna_compute::DType::F32)?);
            alpha_bufs.push(gpu.alloc_tensor(&[max_n * n_v_heads], rdna_compute::DType::F32)?);
            beta_bufs.push(gpu.alloc_tensor(&[max_n * n_v_heads], rdna_compute::DType::F32)?);
        }

        Ok(Self {
            max_n,
            qkv_dim,
            v_dim,
            k_dim,
            n_v_heads,
            n_key_heads,
            value_head_dim: config.linear_value_head_dim,
            key_head_dim: config.linear_key_head_dim,
            qkv_bufs,
            alpha_bufs,
            beta_bufs,
            q_raw_scratch: gpu.alloc_tensor(&[max_n * k_dim], rdna_compute::DType::F32)?,
            k_raw_scratch: gpu.alloc_tensor(&[max_n * k_dim], rdna_compute::DType::F32)?,
            v_scratch:     gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            q_scratch:     gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            k_scratch:     gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            attn_scratch:  gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self
            .qkv_bufs
            .into_iter()
            .chain(self.alpha_bufs.into_iter())
            .chain(self.beta_bufs.into_iter())
        {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.q_raw_scratch);
        let _ = gpu.free_tensor(self.k_raw_scratch);
        let _ = gpu.free_tensor(self.v_scratch);
        let _ = gpu.free_tensor(self.q_scratch);
        let _ = gpu.free_tensor(self.k_scratch);
        let _ = gpu.free_tensor(self.attn_scratch);
    }

    /// Replay the full LA sub-pipeline (conv1d + qk-l2norm + repeat-interleave +
    /// GDN recurrence) for `n_steps` across all LinearAttention layers. Advances
    /// both `dn_state.s_matrices`/`s_scales` AND `dn_state.conv_states` by
    /// exactly `n_steps` single-token updates. Caller must have restored the
    /// DN snapshot to the pre-verify point before calling this.
    ///
    /// Graph-capture path (OPT-IN with HIPFIRE_REPLAY_GRAPH=1): per distinct
    /// n_steps, the first call runs direct as a warmup, the second captures a
    /// hipGraph, and subsequent calls replay the graph. Eligibility:
    /// gpu.active_stream must be Some (so a verify-graph path that already
    /// created one has run first in this cycle).
    ///
    /// MEASURED NULL RESULT (2026-04-21, 27B HumanEval @ accept≈10):
    /// 77.27 → 77.41 tok/s (+0.18 %, noise). τ and mean_committed byte-exact
    /// across A/B. Replay's ~192 kernel launches per cycle add ~0.3 ms of
    /// dispatch API time out of a 130 ms cycle — graphing them saves that
    /// 0.3 ms but cycle cost lives in GDN kernel execution time (scales
    /// linearly with n_steps). Kept as opt-in infrastructure for future
    /// launch-overhead-dominated workloads (smaller models, finer kernels).
    pub fn replay_gdn(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        let graph_enabled =
            std::env::var("HIPFIRE_REPLAY_GRAPH").ok().as_deref() == Some("1");
        let can_graph = graph_enabled && gpu.active_stream.is_some();

        if can_graph && gpu.replay_has_graph(n_steps) {
            return gpu.replay_graph_launch(n_steps);
        }

        if can_graph && gpu.replay_needs_warmup(n_steps) {
            self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)?;
            gpu.replay_mark_warmup_done(n_steps);
            return Ok(());
        }

        if can_graph {
            gpu.begin_replay_graph_capture(n_steps)?;
            let r = self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps);
            if r.is_ok() {
                gpu.end_replay_graph_capture()?;
                // Same pattern as verify_graph: hipStreamBeginCapture records
                // without executing, so launch once here to apply this cycle's
                // state updates.
                gpu.replay_graph_launch(n_steps)?;
                return Ok(());
            } else {
                let _ = gpu.hip.stream_end_capture(
                    gpu.active_stream.as_ref().unwrap(),
                );
                gpu.capture_mode = false;
                gpu.capture_blobs.clear();
                return r;
            }
        }

        self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)
    }

    /// Direct kernel path — the original `replay_gdn` body, retained as a
    /// helper so both the graph-warmup first call and the non-graph fallback
    /// share one implementation.
    fn replay_gdn_inner(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        assert!(n_steps <= self.max_n, "replay_gdn: n_steps {n_steps} > max_n");
        let n_v_heads = self.n_v_heads;
        let n_key_heads = self.n_key_heads;
        let hd = self.key_head_dim;
        let v_dim = self.v_dim;
        let k_dim = self.k_dim;
        let value_head_dim = self.value_head_dim;
        let mut la_idx = 0usize;

        for (layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != qwen35::LayerType::LinearAttention {
                continue;
            }
            let conv_weight = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => &l.conv_weight,
                qwen35::LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };

            // 1. conv1d + SiLU + split — advances conv_state, writes
            //    (q_raw, k_raw, v) into scratch.
            gpu.conv1d_silu_split_f32_n(
                &self.q_raw_scratch,
                &self.k_raw_scratch,
                &self.v_scratch,
                &self.qkv_bufs[la_idx],
                conv_weight,
                &dn_state.conv_states[la_idx],
                k_dim,
                v_dim,
                n_steps,
            )?;

            // 2. L2 norm(Q) + L2 norm(K) + scale(Q).
            gpu.fused_qk_l2_norm_scale_f32_batched(
                &self.q_raw_scratch,
                &self.k_raw_scratch,
                n_key_heads,
                hd,
                1.0 / (hd as f32).sqrt(),
                config.norm_eps,
                n_steps,
            )?;

            // 3. Repeat-interleave if GQA.
            if n_key_heads < n_v_heads {
                let ratio = n_v_heads / n_key_heads;
                gpu.repeat_interleave_qk_f32_batched(
                    &self.q_raw_scratch,
                    &self.k_raw_scratch,
                    &self.q_scratch,
                    &self.k_scratch,
                    n_key_heads,
                    ratio,
                    hd,
                    n_steps,
                )?;
            } else {
                let bytes = n_steps * k_dim * 4;
                gpu.hip.memcpy_dtod_at(&self.q_scratch.buf, 0, &self.q_raw_scratch.buf, 0, bytes)?;
                gpu.hip.memcpy_dtod_at(&self.k_scratch.buf, 0, &self.k_raw_scratch.buf, 0, bytes)?;
            }

            // 4. GDN recurrence — advances S_state.
            gpu.gated_delta_net_q8_batch_seq(
                &self.q_scratch,
                &self.k_scratch,
                &self.v_scratch,
                &self.alpha_bufs[la_idx],
                &self.beta_bufs[la_idx],
                &dn_state.s_matrices[la_idx],
                &dn_state.s_scales[la_idx],
                &self.attn_scratch,
                n_steps,
                n_v_heads,
                value_head_dim,
            )?;

            la_idx += 1;
        }
        Ok(())
    }

    /// Slow-path companion to `replay_gdn` (Task #101 slow-path-kill, 2026-04-23).
    ///
    /// When the committed tree path diverges from the linearization order
    /// (`spine_accept = false` in `spec_step_ddtree_batched`), the per-tree-
    /// node qkv / alpha / beta innovations captured during tree verify at
    /// positions `[0..big_n]` need to be rearranged so that linear replay
    /// position `i+1` holds the values for accepted tree node
    /// `accepted_node_indices[i]`. Position 0 (seed) is already correct and
    /// stays put; positions 1..=accept_len are gathered from their
    /// tree-linearization slots.
    ///
    /// Uses `kv_compact_gather` (a generic slot-indexed row-gather kernel)
    /// via `gather_scratch` as staging, then memcpys back to the tape's
    /// own storage. The caller uploads `gather_indices_dev` with:
    ///   indices[0] = 0                              (seed stays at 0)
    ///   indices[i+1] = accepted_node_indices[i] + 1 (tree node → tape row)
    /// for `i ∈ [0, accept_len)`.
    pub fn gather_accepted(
        &self,
        gpu: &mut Gpu,
        gather_indices_dev: &GpuTensor,
        gather_scratch: &GpuTensor,
        n_positions: usize,
    ) -> HipResult<()> {
        let qkv_row_bytes = self.qkv_dim * 4;
        let alpha_row_bytes = self.n_v_heads * 4;
        for layer in 0..self.qkv_bufs.len() {
            // qkv
            gpu.kv_compact_gather(
                &self.qkv_bufs[layer], gather_scratch, gather_indices_dev,
                qkv_row_bytes, n_positions,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.qkv_bufs[layer].buf, 0,
                &gather_scratch.buf, 0,
                n_positions * qkv_row_bytes,
            )?;
            // alpha
            gpu.kv_compact_gather(
                &self.alpha_bufs[layer], gather_scratch, gather_indices_dev,
                alpha_row_bytes, n_positions,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.alpha_bufs[layer].buf, 0,
                &gather_scratch.buf, 0,
                n_positions * alpha_row_bytes,
            )?;
            // beta
            gpu.kv_compact_gather(
                &self.beta_bufs[layer], gather_scratch, gather_indices_dev,
                alpha_row_bytes, n_positions,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.beta_bufs[layer].buf, 0,
                &gather_scratch.buf, 0,
                n_positions * alpha_row_bytes,
            )?;
        }
        Ok(())
    }
}

impl DeltaNetTape {
    pub fn new_for(
        gpu: &mut Gpu,
        state: &DeltaNetState,
        n_slots: usize,
    ) -> HipResult<Self> {
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(DeltaNetSnapshot::new_for(gpu, state)?);
        }
        Ok(Self { slots })
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn save_at(
        &mut self,
        slot: usize,
        state: &DeltaNetState,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        self.slots[slot].save_from(state, gpu)
    }

    pub fn restore_from(
        &self,
        slot: usize,
        state: &mut DeltaNetState,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        self.slots[slot].restore_to(state, gpu)
    }
}

/// Compute the DFlash target-layer extraction indices for a model of
/// `num_target_layers` layers. Matches the `build_target_layer_ids` function in
/// the DFlash reference implementation:
///
/// ```text
/// start = 1
/// end   = num_target_layers - 3        # 29 for num_target_layers=32
/// step  = (end - start) / (num_extract - 1)
/// layers[i] = round(start + i * step)  # for i in 0..num_extract
/// ```
///
/// For Qwen3.5-9B (32 layers) and 5 extraction layers this returns
/// `[1, 8, 15, 22, 29]`, matching the hard-coded indices in the HuggingFace
/// `z-lab/Qwen3.5-9B-DFlash` config.
pub fn dflash_extract_layer_ids(num_target_layers: usize, num_extract: usize) -> Vec<usize> {
    if num_extract == 0 { return Vec::new(); }
    if num_extract == 1 { return vec![1]; }
    let start: f32 = 1.0;
    let end: f32 = (num_target_layers as i32 - 3).max(1) as f32;
    let step = (end - start) / (num_extract as f32 - 1.0);
    (0..num_extract)
        .map(|i| (start + i as f32 * step).round() as usize)
        .collect()
}

/// Ring buffer holding the most recent `max_positions` of hidden state
/// extractions from the target model's forward pass. Each of the `extract_layers`
/// entries is a `[max_positions, hidden_dim]` f32 GPU tensor. `head` is the
/// position that the NEXT write will land at (0..max_positions). `written` is
/// the total cumulative number of writes, used to tell full vs partial buffer.
///
/// For DFlash, the draft model pulls a contiguous slice ending at the most
/// recent position to use as context KV input.
/// Persistent scratch for `spec_step_ddtree_batched` — eliminates the
/// per-cycle alloc/free churn that dominated early-benchmark wall-clock
/// time. Allocated once at session start, sized for the maximum tree we
/// may see.
///
/// Contents:
/// - `attn_bias`: `[max_n × max_n]` f32 additive bias buffer. `max_n =
///   1 + tree_budget`. Per cycle the caller uploads `big_n × big_n`
///   floats into its head — unused tail space is irrelevant because the
///   FA kernel reads at `global_bid * block_cols + col` with `block_cols
///   = big_n`.
///
/// Callers pass this by `&mut` to `spec_step_ddtree_batched`. It's OK
/// to over-allocate (max_n larger than any cycle's actual tree) — the
/// per-cycle cost is only the htod of the current cycle's mask bytes.
pub struct DdtreeScratch {
    pub max_n: usize,
    pub attn_bias: GpuTensor,
    /// Per-slot parent index consumed by tree-aware LA kernels when
    /// `HIPFIRE_DDTREE_TREE_LA=1`. `[max_n]` i32, uploaded fresh each
    /// cycle via `memcpy_htod` before calling `verify_dflash_block_tree`.
    /// Allocated as Raw bytes (4 × max_n) since there's no i32 DType.
    pub parent_indices: GpuTensor,
    /// Slow-path gather scratch (Task #101 slow-path-kill, 2026-04-23).
    /// When the committed tree path diverges from the rank-0 linearization
    /// (`spine_accept = false`), we need to rearrange already-computed
    /// per-position state into committed-chain order instead of paying a
    /// full re-verify forward. Three buffers:
    ///  - `kv_gather_indices`: [max_n] i32 device buf holding absolute KV
    ///    slot indices `[start_pos + 0, start_pos + 1 + accepted[0], ...]`
    ///    that `kv_compact_gather` reads to select K/V rows per layer.
    ///  - `kv_gather_scratch_k` / `_v`: per-layer gather destination + memcpy
    ///    staging, sized to hold `max_n × widest_k_bpp` / `widest_v_bpp`
    ///    bytes for the KV quant modes this model may use.
    pub kv_gather_indices: GpuTensor,
    pub kv_gather_scratch_k: GpuTensor,
    pub kv_gather_scratch_v: GpuTensor,
    /// Separately: the GdnTape also needs a gather-then-copy-back staging
    /// buffer for qkv/alpha/beta bufs. Sized to the widest tape row
    /// (`qkv_dim * 4` bytes) × `max_n`. Reused across all LA layers.
    pub tape_gather_scratch: GpuTensor,
    /// Path B per-FA-layer pre-RoPE K capture (slow-path-kill, WIP).
    /// One F32 tensor of `[max_n × n_kv_heads × head_dim]` per FullAttention
    /// layer in `config.layer_types`. Tree verify memcpy_dtods K into the
    /// matching slot BEFORE the RoPE kernel rotates K in-place. Slow path
    /// then re-RoPEs with committed positions instead of linearization
    /// positions. Empty Vec when Path B isn't wired (default today).
    pub pre_rope_k: Vec<GpuTensor>,
}

impl DdtreeScratch {
    /// Allocate for a worst-case tree of `max_budget` non-root nodes.
    ///
    /// `n_kv_heads` / `head_dim` come from the target's Qwen35Config and
    /// size the KV-gather staging buffers for the widest-possible quant
    /// mode (Q8, asym2/3/4 all ≤ Q8 bpp in bytes-per-position on K;
    /// V is always Q8 for the asym* modes).
    ///
    /// `qkv_dim` is the per-position GdnTape qkv row width (k_dim × 2 +
    /// v_dim) — see `GdnTape::new_for_config`.
    pub fn new(
        gpu: &mut Gpu,
        max_budget: usize,
        n_kv_heads: usize,
        head_dim: usize,
        qkv_dim: usize,
        n_fa_layers: usize,
    ) -> HipResult<Self> {
        let max_n = 1 + max_budget;
        let attn_bias = gpu.alloc_tensor(
            &[max_n * max_n],
            rdna_compute::DType::F32,
        )?;
        let parent_indices = gpu.alloc_tensor(
            &[max_n * 4],
            rdna_compute::DType::Raw,
        )?;
        // Path B per-FA-layer pre-RoPE K capture. Sized once at session
        // init. Empty on n_fa_layers=0 → capture is a no-op even if the
        // env gate is set (slow-path-kill won't have data to consume).
        let mut pre_rope_k: Vec<GpuTensor> = Vec::with_capacity(n_fa_layers);
        for _ in 0..n_fa_layers {
            pre_rope_k.push(gpu.alloc_tensor(
                &[max_n * n_kv_heads * head_dim],
                rdna_compute::DType::F32,
            )?);
        }

        // Widest bytes-per-position across KV quant modes we might run under.
        // Mirrors TriAttention's `widest_bpp` sizing (see triattn.rs:784).
        let q8_bpp = n_kv_heads * (head_dim / 32) * 34;
        let asym3_k_bpp = n_kv_heads * (4 + (head_dim * 3) / 8);
        let asym4_k_bpp = n_kv_heads * (4 + head_dim / 2);
        let asym2_k_bpp = n_kv_heads * (4 + head_dim / 4);
        let widest_k_bpp = q8_bpp.max(asym3_k_bpp).max(asym4_k_bpp).max(asym2_k_bpp);
        let widest_v_bpp = q8_bpp;

        let kv_gather_indices = gpu.alloc_tensor(
            &[max_n * 4],
            rdna_compute::DType::Raw,
        )?;
        // Raw byte buffers sized to hold `max_n` full K / V rows.
        let kv_gather_scratch_k = gpu.alloc_tensor(
            &[(max_n * widest_k_bpp + 3) / 4],
            rdna_compute::DType::F32,
        )?;
        let kv_gather_scratch_v = gpu.alloc_tensor(
            &[(max_n * widest_v_bpp + 3) / 4],
            rdna_compute::DType::F32,
        )?;
        // Tape rows are F32 projections, so sized in F32 elements directly.
        let tape_gather_scratch = gpu.alloc_tensor(
            &[max_n * qkv_dim],
            rdna_compute::DType::F32,
        )?;

        Ok(Self {
            max_n,
            attn_bias,
            parent_indices,
            kv_gather_indices,
            kv_gather_scratch_k,
            kv_gather_scratch_v,
            tape_gather_scratch,
            pre_rope_k,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.attn_bias);
        let _ = gpu.free_tensor(self.parent_indices);
        let _ = gpu.free_tensor(self.kv_gather_indices);
        let _ = gpu.free_tensor(self.kv_gather_scratch_k);
        let _ = gpu.free_tensor(self.kv_gather_scratch_v);
        let _ = gpu.free_tensor(self.tape_gather_scratch);
        for t in self.pre_rope_k {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Persistent per-decode-cycle scratch for the target verify pass and the
/// draft lm_head. Prior to 2026-04-16 these were allocated fresh every cycle
/// inside `verify_dflash_block_inner` and `spec_step_dflash` — ~8 hipMalloc
/// + hipFree pairs per cycle (biggest are 16 MB logits buffers). The HIP
/// allocator is 50–200 µs per call, so per-cycle overhead was 0.5–1.5 ms
/// just in allocator churn. Preallocating once at session start removes
/// the churn with no correctness impact.
///
/// `max_n` must be ≥ max of (verify block size, tree-verify node count).
/// The demo sizes it to `max(block_size, 1 + tree_budget)` to cover both
/// the vanilla DFlash and DDTree paths.
pub struct VerifyScratch {
    pub max_n: usize,
    pub dim: usize,
    pub vocab: usize,
    pub hidden_k: usize,
    /// Post-output-norm hidden from the target forward, [max_n × dim] F32.
    /// Drives the per-position lm_head GEMM.
    pub final_hidden: GpuTensor,
    /// Scratch logits from target + draft lm_head, [max_n × vocab] F32.
    /// Reused across target verify (n=B) and draft lm_head (n=B-1).
    pub logits: GpuTensor,
    /// FWHT-rotated hidden for MQ4 lm_head path, [max_n × hidden_k] F32.
    /// Allocated unconditionally; unused on non-MQ4 targets.
    pub rot: GpuTensor,
    /// Argmax output for greedy path, [max_n] f32 (treated as i32 host-side).
    pub argmax: GpuTensor,
    /// Persistent per-layer batch scratch for `qwen35::forward_prefill_batch`.
    /// Sized to `max_n`, so `verify_dflash_block` processes each block in a
    /// single chunk without the ~25 hipMalloc/hipFree pairs the in-function
    /// allocation would incur. Present whenever the caller passes a config
    /// to `VerifyScratch::with_prefill`. Absent (None) for the legacy
    /// constructor — `forward_prefill_batch` then falls back to allocating
    /// its own scratch.
    pub prefill_batch: Option<qwen35::PrefillBatchScratch>,
}

impl VerifyScratch {
    pub fn new(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
    ) -> HipResult<Self> {
        Ok(Self {
            max_n,
            dim,
            vocab,
            hidden_k,
            final_hidden: gpu.alloc_tensor(&[max_n * dim], rdna_compute::DType::F32)?,
            logits: gpu.alloc_tensor(&[max_n * vocab], rdna_compute::DType::F32)?,
            rot: gpu.alloc_tensor(&[max_n * hidden_k], rdna_compute::DType::F32)?,
            argmax: gpu.alloc_tensor(&[max_n], rdna_compute::DType::F32)?,
            prefill_batch: None,
        })
    }

    /// Like `new`, but also allocates a persistent `PrefillBatchScratch`
    /// sized to `max_n`. Use this for DFlash verify where the same block
    /// scratch is reused every cycle — drops ~25 tensor alloc/free pairs
    /// per cycle (measured ~3-5 ms/cycle on 27B Qwen3.5 where the per-call
    /// allocation dominated verify wall-time).
    pub fn with_prefill(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
        config: &qwen35::Qwen35Config,
    ) -> HipResult<Self> {
        let mut s = Self::new(gpu, max_n, dim, vocab, hidden_k)?;
        s.prefill_batch = Some(qwen35::PrefillBatchScratch::new(gpu, config, max_n)?);
        Ok(s)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.rot);
        let _ = gpu.free_tensor(self.argmax);
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
        }
    }
}

pub struct HiddenStateRingBuffer {
    pub layer_bufs: Vec<GpuTensor>,
    pub extract_layers: Vec<usize>,
    pub max_positions: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub written: usize,
    /// Per-extract-layer staging buffer, shape `[max_batch × hidden_dim]`.
    /// Captured kernels (verify forward) write here with FIXED offsets so
    /// their captured pointers don't bake in a per-cycle `head`. After the
    /// graph returns, `commit_staging_to_ring` scatters staging → `layer_bufs`
    /// at the current head (outside the captured region, head-aware).
    pub staging_bufs: Vec<GpuTensor>,
    /// Max rows a single staging write can hold — sized to the maximum batch
    /// the caller ever passes to `write_rows_to_staging`. For DFlash verify
    /// this is `budget + 1`.
    pub max_batch: usize,
}

impl HiddenStateRingBuffer {
    /// Allocate GPU ring buffer for `num_extract` target layers.
    ///
    /// `max_batch` sizes the staging buffers used by the graph-capture path.
    /// Typical value for DFlash verify is `budget + 1`.
    pub fn new(
        gpu: &mut Gpu,
        num_target_layers: usize,
        num_extract: usize,
        hidden_dim: usize,
        max_positions: usize,
        max_batch: usize,
    ) -> HipResult<Self> {
        let extract_layers = dflash_extract_layer_ids(num_target_layers, num_extract);
        let mut layer_bufs = Vec::with_capacity(num_extract);
        let mut staging_bufs = Vec::with_capacity(num_extract);
        for _ in 0..num_extract {
            layer_bufs.push(gpu.alloc_tensor(&[max_positions * hidden_dim], rdna_compute::DType::F32)?);
            staging_bufs.push(gpu.alloc_tensor(&[max_batch * hidden_dim], rdna_compute::DType::F32)?);
        }
        Ok(Self {
            layer_bufs,
            extract_layers,
            max_positions,
            hidden_dim,
            head: 0,
            written: 0,
            staging_bufs,
            max_batch,
        })
    }

    /// If `target_layer_idx` matches one of the extraction layers, return the
    /// index into `layer_bufs`/`extract_layers` for that layer. Otherwise None.
    #[inline]
    pub fn extract_slot(&self, target_layer_idx: usize) -> Option<usize> {
        self.extract_layers.iter().position(|&l| l == target_layer_idx)
    }

    /// Copy `x` (shape `[hidden_dim]`) into the ring buffer slot for the given
    /// extraction layer at the CURRENT head position. Call once per extracted
    /// layer per forward pass, then `advance_head()` at the end of the forward
    /// to move to the next slot.
    pub fn write_at_head(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        x: &GpuTensor,
    ) -> HipResult<()> {
        let offset = self.head * self.hidden_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &self.layer_bufs[extract_idx].buf,
            offset,
            &x.buf,
            0,
            self.hidden_dim * 4,
        )
    }

    /// Advance the write head. Call once per forward pass, AFTER all layer
    /// extractions for this position have been written.
    #[inline]
    pub fn advance_head(&mut self) {
        self.head = (self.head + 1) % self.max_positions;
        self.written += 1;
    }

    /// Advance the write head by `n`. Used by the batched prefill path after
    /// writing N rows per extract layer in a single dispatch.
    #[inline]
    pub fn advance_head_by(&mut self, n: usize) {
        self.head = (self.head + n) % self.max_positions;
        self.written += n;
    }

    /// Copy `n` contiguous rows from `src` (shape `[n × hidden_dim]` row-major)
    /// into the ring buffer slot for the given extraction layer, starting at
    /// the CURRENT head position. Handles the ring-buffer wrap: if head + n
    /// exceeds max_positions, the write splits into a head→end + 0→tail pair.
    /// Call this once per extracted layer per batched forward, then advance
    /// the head by `n` via `advance_head_by(n)` at the end.
    pub fn write_rows_at_head(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;
        if head + n <= max_pos {
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                n * row_bytes,
            )?;
        } else {
            let first = max_pos - head;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                first * row_bytes,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                0,
                &src.buf,
                first * row_bytes,
                (n - first) * row_bytes,
            )?;
        }
        Ok(())
    }

    /// Write `n` contiguous rows from `src` into the staging buffer for the
    /// given extraction layer at FIXED offset 0. Safe to call inside a
    /// hipGraph stream capture: the captured memcpy node bakes in the
    /// staging pointer (which is stable across cycles), not a per-cycle head.
    ///
    /// Callers must call `commit_staging_to_ring(n)` after the forward
    /// returns (outside the captured region) to scatter staging → `layer_bufs`
    /// at the current head, then advance the head.
    pub fn write_rows_to_staging(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        debug_assert!(n <= self.max_batch,
            "write_rows_to_staging: n {} > max_batch {}", n, self.max_batch);
        let row_bytes = self.hidden_dim * 4;
        let bytes = n * row_bytes;
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(
                &self.staging_bufs[extract_idx].buf, 0,
                &src.buf, 0,
                bytes, stream,
            )
        } else {
            gpu.hip.memcpy_dtod_at(
                &self.staging_bufs[extract_idx].buf, 0,
                &src.buf, 0,
                bytes,
            )
        }
    }

    /// Scatter staging buffers into `layer_bufs` at the current head, handling
    /// ring wrap, then advance the head by `n`. Must be called AFTER the
    /// forward (outside any captured region) — uses the current `head` to
    /// compute destination offsets, which would be baked wrong in a replayed
    /// graph.
    ///
    /// When `gpu.active_stream` is Some, we first sync the stream (so the
    /// captured forward's staging writes are complete) then use sync D2D
    /// for the scatter. This matches the existing sync-memcpy semantics the
    /// rest of the engine relies on for ordering with null-stream consumers
    /// (e.g. the draft forward's D2H of hidden rows after this commit).
    pub fn commit_staging_to_ring(&mut self, gpu: &mut Gpu, n: usize) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;

        // If running under an explicit stream (graph capture path), wait
        // for the captured writes to complete before the scatter so we
        // don't read uninitialized staging.
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }

        for ei in 0..self.layer_bufs.len() {
            if head + n <= max_pos {
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf, head * row_bytes,
                    &self.staging_bufs[ei].buf, 0,
                    n * row_bytes,
                )?;
            } else {
                let first = max_pos - head;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf, head * row_bytes,
                    &self.staging_bufs[ei].buf, 0,
                    first * row_bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf, 0,
                    &self.staging_bufs[ei].buf, first * row_bytes,
                    (n - first) * row_bytes,
                )?;
            }
        }
        self.head = (head + n) % max_pos;
        self.written += n;
        Ok(())
    }

    /// Reset to empty (head=0, written=0). GPU buffers are not zeroed; stale
    /// data is simply unreadable because `written < max_positions`.
    pub fn reset(&mut self) {
        self.head = 0;
        self.written = 0;
    }
}

/// Single-pass argmax for token sampling. Not SIMD-optimized — the logit
/// vector is downloaded once per verify step so the CPU scan cost is
/// negligible relative to GEMV work.
#[inline]
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. Writes into `out` (reused across calls to
/// avoid per-position allocation in the rejection-sampling hot loop).
#[inline]
fn softmax_temp_into(logits: &[f32], temp: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let inv_t = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_t;
        if s > max { max = s; }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v * inv_t - max).exp();
        out.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for p in out.iter_mut() { *p *= inv_sum; }
}

/// Draw a categorical sample from `probs` given uniform u ∈ [0, 1).
#[inline]
fn sample_categorical(probs: &[f32], u: f32) -> u32 {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc { return i as u32; }
    }
    (probs.len() - 1) as u32
}

/// Draw from (p_target − p_draft)₊, renormalized. Used on rejection to
/// sample the "corrective" bonus token in speculative rejection sampling
/// (Chen & Leviathan 2023, algorithm 1).
#[inline]
fn sample_residual(p_target: &[f32], p_draft: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 { sum += d; }
    }
    if sum <= 0.0 {
        // Degenerate case (p_draft >= p_target everywhere). Should not
        // happen in practice if a rejection was just drawn. Fall back to
        // argmax of p_target.
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            acc += d;
            if u_scaled < acc { return i as u32; }
        }
    }
    (p_target.len() - 1) as u32
}

/// Rolling bigram n-gram cache. Keyed by the last two committed tokens
/// `(a, b)`; value is a small map from possible next-token to count.
///
/// Populated incrementally from the committed output stream. Used as a
/// "free" second opinion on top of the DFlash draft: if the cache has
/// seen a (a, b) → c transition with high enough count, and the DFlash
/// draft proposed something else at that position, the n-gram's `c`
/// often turns out to match the target's argmax.
///
/// Scales: the cache size is bounded by the number of distinct bigrams
/// in the committed output — typically a few hundred per session, so
/// no eviction policy needed.
pub struct NgramCache {
    /// `(a, b) → { next: count, ... }` with the next-token histogram.
    pub bigram: std::collections::HashMap<(u32, u32), std::collections::HashMap<u32, u32>>,
    /// Minimum count before we trust the prediction. Smaller = more
    /// aggressive (more overrides), larger = more conservative. 3 is a
    /// reasonable default on hot-loop code / repetitive text.
    pub min_count: u32,
}

impl NgramCache {
    pub fn new(min_count: u32) -> Self {
        Self {
            bigram: std::collections::HashMap::new(),
            min_count,
        }
    }

    /// Record the triple `(a, b) → c` in the cache.
    #[inline]
    pub fn observe(&mut self, a: u32, b: u32, c: u32) {
        *self
            .bigram
            .entry((a, b))
            .or_default()
            .entry(c)
            .or_insert(0) += 1;
    }

    /// Predict `c` from last-two `(a, b)` if the max-count next-token
    /// reaches `min_count`. Returns (token, count).
    #[inline]
    pub fn predict(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let map = self.bigram.get(&(a, b))?;
        let (&tok, &cnt) = map.iter().max_by_key(|(_, &c)| c)?;
        if cnt >= self.min_count {
            Some((tok, cnt))
        } else {
            None
        }
    }

    /// Record every consecutive triple in a slice of committed tokens.
    /// Caller supplies the full token stream; this walks it in-place.
    pub fn observe_many(&mut self, tokens: &[u32]) {
        if tokens.len() >= 3 {
            for w in tokens.windows(3) {
                self.observe(w[0], w[1], w[2]);
            }
        }
    }
}

/// Prompt Lookup Decoding (Saxena 2023): training-free deterministic draft
/// built from context suffix self-match. If the last N tokens of context
/// appeared earlier in context, the tokens that followed that earlier
/// occurrence are a high-quality continuation guess.
///
/// Used as the draft source in Goose bypass mode (Jin et al. 2026,
/// arXiv:2604.02047 §4.3): PLD-matched tokens have 2–18× higher acceptance
/// than bigram (TR) tokens (median 6× across 5 models × 5 benchmarks).
/// When PLD confidence is high, the spine — a deep linear chain of
/// PLD-matched tokens — is verified in one target forward pass without
/// tree construction. That's exactly what we need on Qwen3.5 hybrid
/// (24 DeltaNet + 8 FullAttention): linear verify sidesteps the
/// state-forking problem that tree verify imposes on recurrent LA layers.
pub struct PldMatcher {
    /// n-gram suffix lengths to try, longest first. Paper uses {5,4,3}.
    /// Longer matches are more selective; if the longest fails we fall
    /// back to shorter. Order matters: we return the first (longest) hit.
    pub ngram_lens: Vec<usize>,
    /// Hard cap on spine length. Paper uses 8 — sufficient for typical
    /// block sizes and avoids running off the end of a match into drift.
    pub max_extract: usize,
    /// Minimum extracted length to count as a usable spine. Very short
    /// spines aren't worth the PLD path (bigram covers 1-token lookahead
    /// at lower risk); require at least this many continuation tokens.
    pub min_extract: usize,
}

impl Default for PldMatcher {
    fn default() -> Self {
        Self { ngram_lens: vec![5, 4, 3], max_extract: 8, min_extract: 3 }
    }
}

/// Result of a successful PLD lookup.
#[derive(Debug, Clone)]
pub struct PldMatch {
    /// The extracted spine (continuation tokens after the matched suffix).
    pub tokens: Vec<u32>,
    /// The suffix length that produced this match (the longest that hit).
    pub n: usize,
    /// Number of tried n-gram lengths that agreed on `tokens[0]`. Paper
    /// §4.3 uses this as part of the bypass-mode confidence signal;
    /// higher consensus = more reliable spine. Ranges 1..=ngram_lens.len().
    pub consensus: usize,
}

impl PldMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a spine continuation for `context`. Returns `None` if no tried
    /// n-gram length produces a match of length ≥ `self.min_extract`.
    ///
    /// For each n in `self.ngram_lens`: take the last-n tokens as the
    /// suffix, search for its last occurrence earlier in context, and
    /// extract the `max_extract` tokens that followed it (stopping before
    /// the suffix itself so we don't include tokens that would be about
    /// to be re-predicted). Returns the longest-n match with a usable
    /// spine; consensus counts how many alternate n's produced the same
    /// first continuation token.
    pub fn lookup(&self, context: &[u32]) -> Option<PldMatch> {
        if self.ngram_lens.is_empty() {
            return None;
        }
        // Per-n continuation, collected to compute consensus across lengths.
        let mut firsts: Vec<u32> = Vec::with_capacity(self.ngram_lens.len());
        let mut best: Option<(usize, Vec<u32>)> = None; // (n, spine)
        for &n in &self.ngram_lens {
            if context.len() <= n {
                continue;
            }
            let suffix_start = context.len() - n;
            let suffix = &context[suffix_start..];
            let haystack = &context[..suffix_start];
            if haystack.len() < n {
                continue;
            }
            // Last occurrence (freshest) of `suffix` in `haystack`.
            let mut found: Option<usize> = None;
            for i in (0..=haystack.len() - n).rev() {
                if &haystack[i..i + n] == suffix {
                    found = Some(i);
                    break;
                }
            }
            let start = match found {
                Some(s) => s,
                None => continue,
            };
            let cont_start = start + n;
            let cont_end = (cont_start + self.max_extract).min(suffix_start);
            if cont_end <= cont_start {
                continue;
            }
            let spine: Vec<u32> = context[cont_start..cont_end].to_vec();
            if spine.len() < self.min_extract {
                continue;
            }
            firsts.push(spine[0]);
            if best.is_none() {
                best = Some((n, spine));
            }
        }

        let (n, tokens) = best?;
        let consensus = firsts.iter().filter(|&&t| t == tokens[0]).count();
        Some(PldMatch { tokens, n, consensus })
    }
}

/// Small, fast RNG for per-cycle sampling u ∈ [0, 1). Xorshift64*; deterministic
/// given the seed, cheap enough to inline into the B-rejection loop.
#[inline]
fn xorshift_next_unit(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Top 24 bits for a reasonable float mantissa; divide by 2^24.
    ((s >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Aggregated metrics for a sequence of speculative decode steps.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Total number of speculative cycles run.
    pub cycles: usize,
    /// Total number of tokens committed (sum of committed.len() across cycles).
    pub committed_tokens: usize,
    /// Total number of draft tokens accepted (sum of `accepted`).
    pub accepted_tokens: usize,
    /// Per-cycle acceptance count histogram, indexed by accepted count
    /// (0..=k). `acceptance_hist[i]` = number of cycles where exactly `i`
    /// draft tokens were accepted.
    pub acceptance_hist: Vec<usize>,
}

impl SpecStats {
    pub fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            committed_tokens: 0,
            accepted_tokens: 0,
            acceptance_hist: vec![0; k + 1],
        }
    }

    pub fn record(&mut self, step: &SpecStepResult) {
        self.cycles += 1;
        self.committed_tokens += step.committed.len();
        self.accepted_tokens += step.accepted;
        if step.accepted < self.acceptance_hist.len() {
            self.acceptance_hist[step.accepted] += 1;
        }
    }

    /// Mean accepted draft tokens per cycle. This is τ from the Leviathan paper.
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.cycles as f32
        }
    }

    /// Mean committed tokens per cycle (tau + 1 on average, since each
    /// cycle always commits one bonus token).
    pub fn mean_committed(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.committed_tokens as f32 / self.cycles as f32
        }
    }
}

/// One speculative decode step (greedy, Leviathan verify-and-accept).
/// Operates on separate `target` and `draft` `ModelSlot` handles so the
/// caller can keep them owned in top-level variables.
///
/// Preconditions:
/// - Both `target.scratch.logits` and `draft.scratch.logits` contain the
///   logits for position `pos` (from the previous commit or prompt prefill).
/// - `target_snap` / `draft_snap` are preallocated via `DeltaNetSnapshot::new_for`.
/// - `k >= 1` is the speculation count.
///
/// Postconditions:
/// - Both slots' state advances to `pos + committed.len()`, and their
///   `scratch.logits` contain logits at the new position.
/// - Returns a `SpecStepResult` describing how many draft tokens were
///   accepted, the bonus token, and the full committed sequence.
///
/// Naive sequential verification: runs the target on each drafted token one
/// at a time. Phase 5 replaces the inner loop with a single batched prefill.
pub fn spec_step_greedy(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft: &mut ModelSlot,
    pos: usize,
    k: usize,
    target_snap: &mut DeltaNetSnapshot,
    draft_snap: &mut DeltaNetSnapshot,
) -> HipResult<SpecStepResult> {
    assert!(k >= 1, "speculation count k must be ≥ 1");

    // Snapshot both models' recurrent state at position `pos` so we can
    // rewind after verification and commit the final accepted prefix.
    target_snap.save_from(&target.dn_state, gpu)?;
    draft_snap.save_from(&draft.dn_state, gpu)?;

    // Target's current logits (at position `pos`) are used to verify
    // drafted[0]. Capture before anything trashes them.
    let target_logits_at_pos: Vec<f32> = gpu.download_f32(&target.scratch.logits)?;

    // Draft k tokens. drafted[0] samples from draft's current logits (which
    // are also for position `pos`). drafted[i] samples from the logits
    // produced by draft.forward(drafted[i-1], pos+i-1).
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    {
        let first_logits = gpu.download_f32(&draft.scratch.logits)?;
        drafted.push(argmax_u32(&first_logits));
    }
    for i in 0..k {
        draft.forward(gpu, drafted[i], pos + i)?;
        if i + 1 < k {
            let logits = gpu.download_f32(&draft.scratch.logits)?;
            drafted.push(argmax_u32(&logits));
        }
    }

    // Verification: run the target on each drafted token, collect logits.
    // target_mid_logits[i] = target's prediction at position pos+i+1.
    let mut target_mid_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        target.forward(gpu, drafted[i], pos + i)?;
        target_mid_logits.push(gpu.download_f32(&target.scratch.logits)?);
    }
    // Acceptance:
    //   drafted[0] verified by target_logits_at_pos  (logits at pos)
    //   drafted[i] (i >= 1) verified by target_mid_logits[i-1] (logits at pos+i)
    let mut accepted: usize = 0;
    if !target_logits_at_pos.is_empty()
        && argmax_u32(&target_logits_at_pos) == drafted[0]
    {
        accepted = 1;
        for i in 1..k {
            if argmax_u32(&target_mid_logits[i - 1]) == drafted[i] {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Bonus token = target's prediction at position pos+accepted.
    let bonus_logits: &[f32] = if accepted == 0 {
        &target_logits_at_pos
    } else {
        &target_mid_logits[accepted - 1]
    };
    let bonus_token = argmax_u32(bonus_logits);

    // Commit = accepted draft prefix + bonus.
    let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
    committed.extend_from_slice(&drafted[..accepted]);
    committed.push(bonus_token);

    // Restore both models' state and replay the committed sequence so both
    // slots end at `pos + committed.len()` with correct logits.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    draft_snap.restore_to(&mut draft.dn_state, gpu)?;
    for (i, &tok) in committed.iter().enumerate() {
        target.forward(gpu, tok, pos + i)?;
        draft.forward(gpu, tok, pos + i)?;
    }

    Ok(SpecStepResult {
        accepted,
        bonus_token,
        drafted,
        committed,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash-specific target-side verify
// ═══════════════════════════════════════════════════════════════════════════

/// Output of a DFlash target verify step.
pub struct DflashVerifyOutput {
    /// Target argmax token at each of the B positions. argmax_per_pos[i]
    /// is what the target would greedy-decode at absolute position
    /// `start_pos + i` given the preceding context plus `draft_tokens[0..i]`.
    pub argmax_per_pos: Vec<u32>,
    /// Full logits downloaded for every position, concatenated row-major
    /// as `[B * vocab_size]`. Only populated when `want_full_logits=true`
    /// (i.e. temperature sampling). Empty otherwise — greedy decode
    /// uses GPU argmax and ships just B × 4 bytes to the host.
    pub logits_per_pos: Vec<f32>,
}

/// Run the target on `draft_tokens` (length B) positions starting at
/// `start_pos`. Advances `target.kv_cache` and `target.dn_state` by B
/// positions. Writes B hidden-state rows into `hidden_rb` (ring head
/// advances B times). Returns downloaded logits + argmax per position.
///
/// Fast path (0.1.7 batched verify): one `forward_prefill_batch` call
/// over all B tokens with hidden extraction + per-token post-output-norm
/// hidden capture. Then B sequential `weight_gemv`s against the target's
/// lm_head to get per-position logits. The batched layer-level kernels
/// amortize launch overhead across all B tokens; the lm_head still loops
/// because a batched Q8/MQ4 lm_head GEMM isn't wired yet (task #13).
///
/// Fallback: when the batched path is ineligible (non-MQ weights,
/// non-Q8/asym KV cache, N < MIN_BATCH), `forward_prefill_batch` routes
/// to the per-token loop using `forward_scratch_with_hidden`, so hidden
/// extraction still works.
pub fn verify_dflash_block(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    verify_scratch: &VerifyScratch,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu, target, draft_tokens, start_pos, hidden_rb, gdn_tape, want_full_logits, None,
        verify_scratch,
    )
}

/// Tree-verify variant of `verify_dflash_block`. Pass the linearized
/// `(positions, attn_bias)` built from a `DdTree` and this runs the whole
/// tree through a single batched forward — per-position argmax at slot i
/// corresponds to target's prediction after tree node i (slot 0 = after
/// seed / tree root).
///
/// Note: `gdn_tape` captured from a tree verify records innovations in
/// linearization order, NOT commit order. Callers that need to advance
/// GDN state to a specific committed path should either (a) re-verify
/// the committed linear prefix with `verify_dflash_block` (no tree) and
/// capture tape on that, matching `spec_step_ddtree`'s pattern, or (b)
/// implement a slot-reordering replay (not currently available).
pub fn verify_dflash_block_tree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: qwen35::TreeVerifyCtx<'_>,
    verify_scratch: &VerifyScratch,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu, target, draft_tokens, start_pos, hidden_rb, gdn_tape, want_full_logits,
        Some(tree_verify),
        verify_scratch,
    )
}

fn verify_dflash_block_inner(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: Option<qwen35::TreeVerifyCtx<'_>>,
    verify_scratch: &VerifyScratch,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let vocab = target.config.vocab_size;
    let dim = target.config.dim;

    assert!(b <= verify_scratch.max_n,
        "verify_scratch max_n {} < b {}", verify_scratch.max_n, b);
    assert_eq!(verify_scratch.dim, dim, "verify_scratch dim mismatch");
    assert_eq!(verify_scratch.vocab, vocab, "verify_scratch vocab mismatch");

    // Views into the persistent scratch — no per-cycle allocation. Sized to
    // the actual current `b` (≤ max_n) so downstream kernels see the right
    // shapes. sub_offset returns a non-owning view; do NOT free these.
    let final_hidden = verify_scratch.final_hidden.sub_offset(0, b * dim);

    // Graph-capture path eligibility. The captured forward bakes in:
    //   - N (the batch size) — via kernel grid dims
    //   - kernel selection + layer-type branches (dispatched once at capture)
    //   - weight/bias/buffer pointers (stable across cycles)
    // Per-cycle inputs (tokens, positions, kv_cache contents, dn_state contents,
    // hidden_rb staging dest) are read from device buffers whose *contents*
    // change between replays — the captured graph reads the current bytes.
    //
    // Eligibility is narrow: HFQ4G256 embedding (uploads via pbs.tokens),
    // no tree_verify (its attn_bias+positions are per-cycle), pbs is Some.
    // `gdn_tape` is safe because verify is single-chunk → tape_offset=0 always
    // → captured node's dst offset is correct across cycles.
    //
    // Default-on for eligible models (2026-04-21 smoke on 27B MQ4 Qwen3.5
    // showed +14 % tok/s 25.6→29.2, wall-per-cycle 89→80 ms via coalescing
    // verify kernels into one graph replay and saving ~1.3 ms of per-cycle
    // launch overhead). Opt out with HIPFIRE_VERIFY_GRAPH=0.
    // Tree-verify was historically excluded (tree_verify.is_none()) because
    // the tree-attention mask varies per cycle. In theory mask +
    // parent_indices live in fixed `ddtree_scratch` buffers that the caller
    // repopulates via uncaptured memcpy_htod before each graph replay, so
    // the graph's kernels would read fresh data every cycle.
    //
    // DIAGNOSTIC ONLY — known broken 2026-04-24 (commit 480e51e +
    // A/B bench ee0bedf-followup). 3-run median on 27B MQ4 asym3 b12-k2:
    //   code     τ 7.08 → 4.51 (-36 %)   tok/s 110 → 80.1 (-27 %)
    //   prose    τ 2.50 → 3.58 (+43 %)   tok/s 45.8 → 60.2 (+31 %, noisy)
    //   instr    τ 2.19 → 1.77 (-19 %)   tok/s 47.6 → 35.6 (-25 %)
    // Coherence-gate-dflash passes (no attractors), so it's a τ bug, not a
    // correctness bug. Suspect: a scalar kernarg or intra-forward memcpy
    // inside captured region bakes in first-cycle state; when tree shape
    // varies, acceptance collapses on code (high-variance trees) but
    // coincidentally holds on prose (more uniform trees). Needs root-cause
    // dive: most likely candidates are GDN tape-offset scalar kernargs or
    // the parent_indices-driven conv1d path. DO NOT ENABLE in production.
    //
    // Gate kept live so the next session can bisect without re-plumbing.
    let tree_graph_enabled = std::env::var("HIPFIRE_VERIFY_GRAPH_TREE").ok().as_deref() == Some("1");
    if tree_graph_enabled && tree_verify.is_some() {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[verify-graph-tree] WARN: HIPFIRE_VERIFY_GRAPH_TREE=1 is DIAGNOSTIC ONLY — known τ regression on code/instruct. Do not use for production benchmarks."
            );
        }
    }
    let tree_ok_for_graph = tree_verify.is_none() || tree_graph_enabled;
    let verify_graph_ok = std::env::var("HIPFIRE_VERIFY_GRAPH").ok().as_deref() != Some("0")
        && tree_ok_for_graph
        && matches!(
            target.weights.embd_format,
            crate::llama::EmbeddingFormat::HFQ4G256 | crate::llama::EmbeddingFormat::Q8_0,
        )
        && verify_scratch.prefill_batch.is_some();

    // Per-cycle timing for verify-graph A/B diagnostic
    // (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the
    // forward + lm_head; the recorded mode tag distinguishes replay vs
    // warmup-direct vs first-capture vs no-graph-eligible.
    let vg_timing = std::env::var("HIPFIRE_VERIFY_GRAPH_TIMING").ok().as_deref() == Some("1");
    let mut vg_mode = "direct";
    let vg_t0 = if vg_timing {
        gpu.hip.device_synchronize()?;
        Some(std::time::Instant::now())
    } else {
        None
    };

    let batch_result = if verify_graph_ok {
        let pbs = verify_scratch.prefill_batch.as_ref().unwrap();
        debug_assert!(b <= pbs.max_batch);
        // Pre-capture: pre-upload inputs and ensure a stream exists. memcpy_htod
        // runs on the host/null-stream side and is NOT captured.
        qwen35::upload_prefill_batch_inputs(gpu, pbs, draft_tokens, start_pos)?;
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        if gpu.verify_has_graph(b) {
            vg_mode = "replay";
            // Replay path: kernels read pbs.tokens/pbs.positions/dn_state/
            // kv_cache contents that were freshly updated above + upstream.
            gpu.verify_graph_launch(b)?;
            Ok(())
        } else if gpu.verify_needs_warmup(b) {
            vg_mode = "warmup";
            // Warmup for this b: run direct so kernel JIT and any lazy scratch
            // allocations (e.g., MQ signs/x_rot/x_q8, FP16 shadow) happen
            // outside any captured region. Capturing a JIT + scratch-malloc
            // hits "hipMalloc not permitted under stream capture" the first
            // time any kernel is compiled inline. One warmup per distinct b.
            gpu.verify_mark_warmup_done(b);
            let r = qwen35::forward_prefill_batch_single_chunk_captured(
                gpu,
                &target.weights,
                &target.config,
                draft_tokens,
                start_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
                pbs,
                Some(hidden_rb),
                Some(&final_hidden),
                gdn_tape,
                tree_verify,
            );
            if r.is_ok() {
                eprintln!("[verify-graph] warmup for B={} complete — capture next cycle at this B", b);
            }
            r
        } else {
            vg_mode = "capture";
            // Capture path: first call at this B after warmup.
            gpu.begin_verify_graph_capture(b)?;
            let r = qwen35::forward_prefill_batch_single_chunk_captured(
                gpu,
                &target.weights,
                &target.config,
                draft_tokens,
                start_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
                pbs,
                Some(hidden_rb),
                Some(&final_hidden),
                gdn_tape,
                tree_verify,
            );
            if r.is_ok() {
                let blob_count = gpu.capture_blobs.len();
                gpu.end_verify_graph_capture()?;
                // Under `hipStreamBeginCapture`, kernels + memcpys on the
                // captured stream are RECORDED, not executed. final_hidden
                // and hidden_rb staging are left stale. Launching the graph
                // once here makes this cycle's forward actually run so lm_head
                // reads fresh data. DN state double-advance (if any future
                // HIP version does execute during capture) is washed out by
                // target_snap.restore_to after verify returns. KV cache
                // double-write writes the same data to the same positions.
                gpu.verify_graph_launch(b)?;
                eprintln!(
                    "[verify-graph] captured for B={} with {} blobs (cache size: {})",
                    b, blob_count, gpu.verify_graph_count(),
                );
            } else {
                // If capture failed, tear down the partial capture so we fall
                // back to the direct path next cycle cleanly.
                let _ = gpu.hip.stream_end_capture(gpu.active_stream.as_ref().unwrap());
                gpu.capture_mode = false;
                gpu.capture_blobs.clear();
            }
            r
        }
    } else {
        qwen35::forward_prefill_batch_with_pbs(
            gpu,
            &target.weights,
            &target.config,
            draft_tokens,
            start_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(hidden_rb),
            Some(&final_hidden),
            gdn_tape,
            tree_verify,
            verify_scratch.prefill_batch.as_ref(),
        )
    };

    // Commit hidden_rb staging to the ring (outside any captured region).
    // The captured forward wrote to staging[0..b*h]; this scatter places
    // those rows at the current head and advances head by b. Under the
    // graph path we manually drive this because the non-graph chunk loop
    // (forward_prefill_batch_with_pbs) that usually calls it was bypassed.
    if verify_graph_ok && batch_result.is_ok() {
        hidden_rb.commit_staging_to_ring(gpu, b)?;
    }
    // Tree mode at topk>1 REQUIRES this sync. Without it τ degrades badly
    // (e.g. budget=60 topk=8 drops 7.0 → 3.3; 9B asym3 2026-04-14). topk=1
    // is fine without the sync (byte-exact with baseline DFlash either way).
    // Root cause suspected: siblings at the same tree depth produce
    // duplicate entries in `positions[]`, so `kv_cache_write` dispatches
    // multiple batch rows targeting the same cache slot — the async write
    // order lets a subsequent attention kernel read a partially-committed
    // slot. Fix TODO: either serialize within-kernel per-slot, or ensure
    // the "winning" sibling's write happens last. Cost ~3–5 ms per cycle
    // until fixed.
    if batch_result.is_ok() && tree_verify.is_some() {
        gpu.hip.device_synchronize()?;
    }
    batch_result?;

    // Per-position lm_head. Fast paths in priority order:
    //   Q8_0      → batched gemm_q8_0_batched (one launch + one D2H).
    //   MQ4G256   → batched rotate + gemm_hfq4g256 (one launch + one D2H).
    //   HFQ4G256  → batched gemm_hfq4g256 directly.
    //   else      → B sequential weight_gemv calls + B downloads (legacy).
    let w_out = &target.weights.output;
    let mut logits_per_pos: Vec<f32> = Vec::with_capacity(b * vocab);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);

    let try_batched = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0
        | rdna_compute::DType::HFQ4G256
        | rdna_compute::DType::MQ4G256
        | rdna_compute::DType::MQ3G256 => true,
        _ => false,
    };

    if try_batched {
        let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
        // Q8_0 gemm_q8_0_batched has a hard MAX_BATCH=16 in the kernel, so
        // tree-verify blocks exceeding 16 (budget + 1 > 16) need chunking.
        // MQ4/HFQ4 kernels have no such cap — they take the single-shot path.
        match w_out.gpu_dtype {
            rdna_compute::DType::Q8_0 => {
                const Q8_LM_MAX: usize = 16;
                let mut chunk_start = 0usize;
                while chunk_start < b {
                    let chunk_end = (chunk_start + Q8_LM_MAX).min(b);
                    let chunk_n = chunk_end - chunk_start;
                    let x_chunk = final_hidden.sub_offset(chunk_start * dim, chunk_n * dim);
                    let y_chunk = logits_batch.sub_offset(chunk_start * vocab, chunk_n * vocab);
                    gpu.gemm_q8_0_batched(
                        &w_out.buf, &x_chunk, &y_chunk, w_out.m, w_out.k, chunk_n,
                    )?;
                    chunk_start = chunk_end;
                }
            }
            rdna_compute::DType::HFQ4G256 => {
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &final_hidden, &logits_batch, w_out.m, w_out.k, b,
                )?;
            }
            rdna_compute::DType::MQ4G256 => {
                assert!(b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                    "verify_scratch.rot undersized: b*k={} > max_n*hidden_k={}",
                    b * w_out.k, verify_scratch.max_n * verify_scratch.hidden_k);
                let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
                gpu.rotate_x_mq_batched(&final_hidden, &rot, w_out.k, b)?;
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &rot, &logits_batch, w_out.m, w_out.k, b,
                )?;
            }
            rdna_compute::DType::MQ3G256 => {
                assert!(b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                    "verify_scratch.rot undersized for MQ3 lm_head: b*k={} > max_n*hidden_k={}",
                    b * w_out.k, verify_scratch.max_n * verify_scratch.hidden_k);
                let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
                gpu.rotate_x_mq_batched(&final_hidden, &rot, w_out.k, b)?;
                gpu.gemm_hfq3g256_batched_lmhead(
                    &w_out.buf, &rot, &logits_batch, w_out.m, w_out.k, b,
                )?;
            }
            _ => unreachable!(),
        }
        if want_full_logits {
            // Rejection-sampling path needs full target distribution.
            // Cost: B × vocab × 4 bytes D2H per verify (~15 MB at B=16 × 248K).
            let host_logits = gpu.download_f32(&logits_batch)?;
            for i in 0..b {
                let row = &host_logits[i * vocab..(i + 1) * vocab];
                argmax_per_pos.push(argmax_u32(row));
            }
            logits_per_pos = host_logits;
        } else {
            // GPU-side batched argmax. Writes B i32 indices; we download just
            // 4*B bytes instead of the full B×vocab logits. Saves ~15 MB of
            // PCIe D2H per verify on the 4B Q8 lm_head (~3-5 ms/iter).
            let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
            let mut host_idx = vec![0i32; b];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, b * 4)
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
            }
            for &idx in &host_idx {
                argmax_per_pos.push(idx as u32);
            }
        }
        // Greedy path doesn't need `logits_per_pos`; leave empty to avoid
        // the 15 MB D2H. If temp>0 sampling is added later, reinstate the
        // download or sample on-GPU.
    } else {
        // Fallback: B sequential GEMVs.
        for i in 0..b {
            let hidden_row = final_hidden.sub_offset(i * dim, dim);
            llama::weight_gemv(
                gpu, &target.weights.output, &hidden_row, &target.scratch.logits,
            )?;
            let row = gpu.download_f32(&target.scratch.logits)?;
            debug_assert_eq!(row.len(), vocab);
            argmax_per_pos.push(argmax_u32(&row));
            logits_per_pos.extend_from_slice(&row);
        }
    }

    if let Some(t0) = vg_t0 {
        gpu.hip.device_synchronize()?;
        eprintln!(
            "[vg-time] B={} mode={} elapsed_us={}",
            b, vg_mode, t0.elapsed().as_micros()
        );
    }

    Ok(DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    })
}

/// Download extracted target hidden states for the most recent B positions
/// from `hidden_rb` and concat them into a flat `[B × num_extract × hidden]`
/// host vector in the order expected by `dflash::draft_forward` (per-position,
/// then per-extract-layer).
///
/// Caller typically slices this by `[0..accept_len+1]` of the position
/// dimension when appending to the cumulative target_hidden buffer used
/// by subsequent draft forwards.
///
/// Partial-download path (2026-04-16): downloads only the B most recent
/// rows per layer via `memcpy_dtoh` of the exact slice needed, handling
/// the ring-buffer wrap as two segments when necessary. Prior version
/// downloaded the full `max_pos × hidden` per layer (~170 MB at ctx=2048
/// × hidden=4096 × 5 layers); this cuts per-cycle D2H to the useful
/// `B × hidden × 5 × 4` bytes (~1.3 MB). For a math prompt at ctx=1024
/// this saves ~7 ms/cycle of PCIe + sync overhead.
/// GPU-side scatter of the FIRST `n_rows` rows of the most recently written
/// `block_size` slots of `hidden_rb` into a flat dst tensor laid out as
/// `[max_ctx × num_extract × hidden]` (interleaved per-position).
///
/// Semantics mirror `download_hidden_block(b)` followed by a slice of the
/// first `rows_to_keep * num_extract * hidden` f32s — the spec_step caller
/// pattern. The ring has just been advanced by `block_size` (by a verify
/// forward); the slots written in THAT verify occupy
/// `[head − block_size, head)` (mod max_pos). We copy the first `n_rows`
/// of those to rows `[dst_row_offset, dst_row_offset + n_rows)` of dst.
///
/// For each row r in 0..n_rows:
/// - ring slot = (head − block_size + r) mod max_pos
/// - dst row   = (dst_row_offset + r)
/// - For each extract layer `ext`: D2D copy hidden×4 bytes from
///   `hidden_rb.layer_bufs[ext][slot × hidden ..]` to
///   `dst[(dst_row × num_extract + ext) × hidden ..]`.
///
/// When called after `seed_target_hidden_from_prompt` (no prior block), the
/// caller passes `block_size = n_rows = prompt_len` and `dst_row_offset = 0`.
///
/// Replaces the previous D2H-then-H2D roundtrip via `target_hidden_host`
/// Vec<f32> + `draft_forward`'s upload for the common ctx_slice=None path.
/// Eliminates 5 blocking D2H sync points per spec step (one per extract
/// layer) and the follow-on per-cycle H2D upload; remains about 80 small
/// async D2D enqueues per cycle (ne × n_rows ≤ 5 × 16), which the stream
/// dispatcher handles in ~200 µs of CPU time with zero cross-device waits.
pub fn scatter_hidden_block_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
) -> HipResult<()> {
    assert!(n_rows <= block_size, "scatter: n_rows {n_rows} > block_size {block_size}");
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let head = hidden_rb.head;
    let written = hidden_rb.written;
    assert!(block_size <= written, "scatter: block_size {block_size} > written {written}");
    let row_bytes = hidden * 4;
    let start_slot = (head + max_pos - block_size) % max_pos;

    for r in 0..n_rows {
        let slot = (start_slot + r) % max_pos;
        let dst_row = dst_row_offset + r;
        let dst_row_base_bytes = dst_row * num_extract * row_bytes;
        for ext in 0..num_extract {
            let src_offset_bytes = slot * row_bytes;
            let dst_offset_bytes = dst_row_base_bytes + ext * row_bytes;
            gpu.hip.memcpy_dtod_at(
                &dst.buf,
                dst_offset_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                src_offset_bytes,
                row_bytes,
            )?;
        }
    }
    Ok(())
}

pub fn download_hidden_block(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    b: usize,
) -> HipResult<Vec<f32>> {
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let written = hidden_rb.written;

    // Figure out which ring positions hold the most recent B writes.
    // `head` points to where the NEXT write will land. After B advances,
    // the most recent B sit at ring slots (head - B) mod max_pos ..
    // (head - 1) mod max_pos.
    assert!(b <= written, "verify must have written at least B rows to ring buffer");
    let head = hidden_rb.head;
    let start_slot = (head + max_pos - b) % max_pos;
    let row_bytes = hidden * 4;

    // Per layer, download only the B needed rows (not the full ring).
    // If start_slot + b <= max_pos: one contiguous segment.
    // Otherwise: two segments (head→end + 0→tail).
    //
    // Each layer's B-row slice lands at [ext × b × hidden] in layer_data_flat.
    let mut layer_data_flat = vec![0f32; num_extract * b * hidden];
    for ext in 0..num_extract {
        let src_buf = &hidden_rb.layer_bufs[ext].buf;
        let dst_offset_floats = ext * b * hidden;
        if start_slot + b <= max_pos {
            // Single contiguous copy.
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    b * row_bytes,
                )
            };
            gpu.hip.memcpy_dtoh_at(dst_bytes, src_buf, start_slot * row_bytes)?;
        } else {
            // Two-segment ring wrap: tail of buffer, then head.
            let first_rows = max_pos - start_slot;
            let second_rows = b - first_rows;
            let dst_first_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    first_rows * row_bytes,
                )
            };
            gpu.hip.memcpy_dtoh_at(dst_first_bytes, src_buf, start_slot * row_bytes)?;
            let dst_second_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats + first_rows * hidden) as *mut u8,
                    second_rows * row_bytes,
                )
            };
            gpu.hip.memcpy_dtoh_at(dst_second_bytes, src_buf, 0)?;
        }
    }

    // Rearrange into per-position-then-per-extract-layer order.
    // layer_data_flat is [ext × b × hidden]; we want [b × ext × hidden].
    let mut out: Vec<f32> = Vec::with_capacity(b * num_extract * hidden);
    for pi in 0..b {
        for ext in 0..num_extract {
            let src_off = (ext * b + pi) * hidden;
            out.extend_from_slice(&layer_data_flat[src_off..src_off + hidden]);
        }
    }

    debug_assert_eq!(out.len(), b * num_extract * hidden);
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash spec step — one speculative decode iteration
// ═══════════════════════════════════════════════════════════════════════════

/// One DFlash speculative iteration. Given a previously-accepted token at
/// `position - 1` (the "seed" for block_output_ids[0]) and a cumulative
/// `target_hidden_host` buffer of shape `[position × num_extract × hidden]`,
/// runs the draft to fill B-1 mask slots, verifies against the target,
/// commits the accepted prefix plus a bonus target token, and rewinds the
/// target's DeltaNet state so only `accept_len + 1` forwards are reflected.
///
/// Returns `SpecStepResult` describing accepted draft count, bonus token,
/// drafted proposals, and the full committed sequence (length accept+2:
/// `[seed_token, draft[..accept_len], posterior[accept_len]]` — note the
/// seed_token is ALSO committed here because it was the bonus token from
/// the PREVIOUS iteration and still needs the target forward at its
/// position). Callers append `committed[1..]` to the output token stream
/// (the seed was already emitted).
///
/// Side effects:
/// - Appends `accept_len + 1` positions × `num_extract × hidden` floats to
///   `target_hidden_host`.
/// - Advances target's KV cache and DeltaNet state by `accept_len + 1`
///   positions. Draft has no persistent state.
///
/// Preconditions:
/// - `target_hidden_host.len() == position × num_extract × hidden` (set up
///   by `seed_target_hidden_from_prompt`).
/// - `position ≤ draft_scratch.max_ctx_len`.
/// - `draft_cfg.block_size ≤ draft_scratch.max_block_size`.
///
/// `ctx_slice`: if `Some(N)`, the draft only sees the most recent `N` rows
/// of `target_hidden_host` (with RoPE positions `[position-N..position+B)`).
/// Use this for accept-rate bisect experiments — if training-time context
/// was shorter than inference-time, truncation may help. `None` uses the
/// full cumulative context (the default, distribution-preserving path).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    gdn_tape: Option<&mut GdnTape>,
    temp: f32,
    rng_state: &mut u64,
    block_size_override: Option<usize>,
    ngram_cache: Option<&NgramCache>,
    prev_committed: &[u32],
    cactus_delta: f32,
    pld_spine: Option<&[u32]>,
    repeat_penalty: f32,
    repeat_window: usize,
) -> HipResult<SpecStepResult> {
    // Effective block size for THIS step. Usually `draft_cfg.block_size`
    // (what the draft was trained at, 16 for Qwen3.5-*-DFlash) but a caller
    // doing adaptive-B based on rolling τ can shrink to save per-iter cost.
    //
    // When `pld_spine` is Some, shrink b to 1+pld.len() (capped at requested)
    // so we don't run off the end of the PLD continuation. PLD-supplied
    // spines are often shorter than the trained B; the paper caps at 8.
    let requested_b = block_size_override.unwrap_or(draft_cfg.block_size);
    let b = match pld_spine {
        Some(pld) => (1 + pld.len()).min(requested_b).max(2),
        None => requested_b,
    };
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;

    // Ensure active_stream is set before any draft/verify work so memset_async
    // and stream-ordered launches have a non-null stream to ride on. Without
    // this, the lm_head pre-zero memsets in dispatch.rs:4475/4545 fall through
    // to the sync hipMemset path (~46 hot calls/cycle on 27B).
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    assert!(b >= 2, "dflash block size must be ≥ 2");
    // `target_hidden_host` is only authoritative on the ctx_slice=Some path,
    // where it backs the CPU slice handed to draft_forward. On the default
    // ctx_slice=None path the data lives on GPU in draft_scratch.target_hidden
    // (populated by D2D scatter, no CPU shadow). Only enforce the length
    // invariant when we actually read it.
    if ctx_slice.is_some() {
        assert_eq!(
            target_hidden_host.len(),
            position * ne * h,
            "target_hidden_host size mismatches position"
        );
    }

    // HIPFIRE_SPEC_PHASES=1: per-cycle phase breakdown. Inserts a
    // device_synchronize at each phase boundary so the wall-clock reflects
    // ACTUAL GPU completion (not CPU enqueue of async work). Perf-heavy —
    // use only for diagnostics. When disabled, zero cost beyond a handful
    // of Instant::now() calls.
    let phase_on = std::env::var("HIPFIRE_SPEC_PHASES").ok().as_deref() == Some("1");
    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_spec_start = std::time::Instant::now();
    let mut t_phase = t_spec_start;

    // ── 1. block_output_ids seeded with prev bonus at [0], masks at [1..B] ──
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Draft state: either synthesized from a PLD spine (Goose §4.3 bypass
    // mode — deterministic, skips the DFlash forward) or produced by the
    // DFlash draft forward pass below. Declared out here so the post-draft
    // common code (ngram gating, target verify, rejection) sees the same
    // `drafted` / `draft_softmaxes` / `draft_probs_at_drafted` regardless
    // of draft source.
    let mut drafted: Vec<u32> = vec![seed_token];
    let mut draft_probs_at_drafted: Vec<f32> = Vec::new();
    let mut draft_softmaxes: Vec<Vec<f32>> = Vec::new();
    let use_temp_sampling = temp > 0.0;
    let rp_active = repeat_penalty > 1.0 && !use_temp_sampling;
    // HIPFIRE_DFLASH_NGRAM_BLOCK=1: apply llama::apply_ngram_block to every
    // host-path row in BOTH draft and target argmax paths. Bans the next
    // token after any 3/4/5/6-gram repeat (NEG_INFINITY logit). Matches the
    // production-path defense in daemon/run/infer for the AR sampler.
    // Forces the per-row host download even when RP is off (extra D2H per
    // cycle); off-by-default for that reason.
    let ngram_block_active = !use_temp_sampling
        && std::env::var("HIPFIRE_DFLASH_NGRAM_BLOCK").ok().as_deref() == Some("1");
    let host_path_active = rp_active || ngram_block_active;

    if let Some(pld) = pld_spine {
        // PLD spine path: drafted tokens come from context-suffix match.
        // At temp>0, draft "probability" at each PLD token is 1.0 — PLD is
        // context-deterministic, not a softmax. The rejection math below
        // computes residual from (target_probs − draft_probs)+ normalized,
        // and with draft one-hot at tok, the residual pulls correctly from
        // target minus just that single-position overclaim.
        for i in 0..b - 1 {
            drafted.push(pld[i]);
        }
        if use_temp_sampling {
            draft_probs_at_drafted.reserve(b - 1);
            draft_softmaxes.reserve(b - 1);
            for i in 0..b - 1 {
                let mut probs = vec![0f32; vocab];
                probs[pld[i] as usize] = 1.0;
                draft_softmaxes.push(probs);
                draft_probs_at_drafted.push(1.0);
            }
        }
    } else {
    // ── 2. noise_embedding = target.embed_tokens(block) written directly
    // into draft_scratch.x on GPU (no host round-trip). Target and draft
    // share the same Gpu, so the embedding lookup can target the draft's
    // scratch buffer. Avoids 16 × D2H + one H2D per iter (~1 ms saved).
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            crate::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("dflash: unsupported target embedding format for noise lookup"),
        }
    }

    // ── 3. Position arrays + optional context slice ─────────────────────
    // Q positions: the absolute positions of the block slots,
    //   [position + compact_offset .. position + B + compact_offset).
    // K positions by default: absolute positions of all populated target_hidden
    // rows (potentially non-contiguous after a TriAttention eviction), then
    // the same block slots.
    //
    // Pre-eviction: positions are contiguous [0..position+B), so the
    // abs_positions vec contains [0..position) and this matches the old
    // behaviour byte-for-byte.
    // Post-eviction: abs_positions contains the subset retained by the last
    // FA layer's top-B mask, paired with the correct pre-eviction absolute
    // positions so draft RoPE aligns with target.
    //
    // If `ctx_slice = Some(N)` is set, restrict the draft's context view to
    // the last `N` rows of target_hidden_host, with RoPE positions
    // [position-N..position+B). Eviction-aware abs positions are not tracked
    // on this diagnostic path — callers using it don't expect FlashCASK.
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => draft_scratch.target_hidden_abs_positions.len().min(position),
    };
    let ctx_start = position - effective_ctx_len;
    let co = target.kv_cache.compact_offset as i32;
    let positions_q: Vec<i32> =
        ((position as i32 + co)..(position as i32 + b as i32 + co)).collect();
    let positions_k: Vec<i32> = if ctx_slice.is_some() {
        // Diagnostic path: keep legacy contiguous layout. abs_positions isn't
        // tracked here and eviction isn't supported with ctx_slice anyway.
        (ctx_start as i32..(position + b) as i32).collect()
    } else {
        let mut v = Vec::with_capacity(effective_ctx_len + b);
        let th_abs = &draft_scratch.target_hidden_abs_positions;
        let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
        v.extend_from_slice(&th_abs[start_idx..]);
        for p in 0..b {
            v.push(position as i32 + p as i32 + co);
        }
        v
    };

    // Slice target_hidden_host to the last effective_ctx_len rows. When
    // ctx_slice is None, this is a no-op (ctx_start = 0). Row stride is
    // num_extract × hidden = ne * h.
    //
    // Fast path (ctx_slice == None, 2026-04-16): `draft_scratch.target_hidden`
    // is already populated via D2D scatter at the END of the previous cycle
    // (or seed_target_hidden_from_prompt for the first cycle). We pass
    // `target_hidden = None` to draft_forward so it skips the H2D upload
    // entirely — kills the per-cycle CPU roundtrip.
    //
    // ctx_slice=Some(N) still goes through the CPU shadow (target_hidden_host
    // Vec) because its moving-window semantics don't map onto the append-only
    // GPU buffer without an extra D2D shuffle. It's a diagnostic path anyway.
    let (th_arg, _th_offset): (Option<&[f32]>, usize) = if ctx_slice.is_some() {
        let th_offset = ctx_start * ne * h;
        (Some(&target_hidden_host[th_offset..]), th_offset)
    } else {
        (None, 0)
    };

    // ── 4. draft_forward ────────────────────────────────────────────────
    // noise_embedding = None: we wrote embeddings directly into
    // draft_scratch.x above via D2D (no host round-trip).
    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        th_arg,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // ── 5. Apply target.lm_head to draft hidden positions 1..B ──────────
    // Fast path: a single batched GEMM against target.weights.output over
    // (B-1) hidden rows at once. Drops lm_head from ~40 ms (B-1 serial
    // weight_gemv + downloads) to ~8 ms (one batched GEMM + one download)
    // for MQ4/HFQ4 lm_heads. Falls back to the per-row loop when the
    // output weight dtype isn't covered by the batched gemm dispatch.
    //
    // Temperature-sampling mode (temp > 0): we must DOWNLOAD the full
    // (B-1, vocab) draft logits, softmax + sample + record p_draft[token]
    // for later rejection acceptance. The greedy GPU-argmax path is kept
    // intact for temp == 0 so we don't regress that case.
    let w_out = &target.weights.output;
    let use_batched_gemm = matches!(
        w_out.gpu_dtype,
        rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256 | rdna_compute::DType::MQ3G256,
    );
    let use_q8_staged = matches!(w_out.gpu_dtype, rdna_compute::DType::Q8_0);
    if use_batched_gemm || use_q8_staged {
        // Unified batched path: one GEMM over B-1 rows, GPU-side argmax,
        // download just (B-1) × 4 bytes of indices.
        //
        // Reuses `verify_scratch.logits` and `.rot` — same buffers the target
        // verify uses. Draft calls this BEFORE verify in the cycle, so
        // there's no aliasing. The verify call overwrites these buffers
        // afterward. Avoids 2-3 hipMalloc/Free pairs per cycle.
        let batch = b - 1;
        assert!(batch <= verify_scratch.max_n,
            "verify_scratch max_n {} < draft batch {}", verify_scratch.max_n, batch);
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);

        match w_out.gpu_dtype {
            rdna_compute::DType::Q8_0 => {
                gpu.gemm_q8_0_batched(&w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)?;
            }
            rdna_compute::DType::HFQ4G256 => {
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            rdna_compute::DType::MQ4G256 => {
                assert!(batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                    "verify_scratch.rot undersized for draft lm_head");
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq4g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            rdna_compute::DType::MQ3G256 => {
                assert!(batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                    "verify_scratch.rot undersized for MQ3 draft lm_head");
                let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch)?;
                gpu.gemm_hfq3g256_batched_lmhead(
                    &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
                )?;
            }
            _ => unreachable!(),
        }

        if use_temp_sampling {
            // Full D2H of (B-1)×vocab logits, CPU softmax+sample.
            let host_logits = gpu.download_f32(&logits_batch)?;
            debug_assert_eq!(host_logits.len(), batch * vocab);
            draft_softmaxes.reserve(batch);
            for i in 0..batch {
                let row = &host_logits[i * vocab..(i + 1) * vocab];
                let mut probs = Vec::with_capacity(vocab);
                softmax_temp_into(row, temp, &mut probs);
                let u = xorshift_next_unit(rng_state);
                let t = sample_categorical(&probs, u);
                draft_probs_at_drafted.push(probs[t as usize]);
                drafted.push(t);
                draft_softmaxes.push(probs);
            }
        } else if host_path_active {
            // RP / n-gram-block path: apply per-row penalties before argmax
            // so draft and target pick from the same reshaped distribution.
            // Keeps spec-decode aligned (τ doesn't collapse from mismatched
            // argmaxes — both sides see identical -inf logits on banned toks).
            let host_logits = gpu.download_f32(&logits_batch)?;
            debug_assert_eq!(host_logits.len(), batch * vocab);
            let mut row = vec![0f32; vocab];
            for i in 0..batch {
                row.copy_from_slice(&host_logits[i * vocab..(i + 1) * vocab]);
                if rp_active {
                    llama::apply_repeat_penalty(&mut row, prev_committed, repeat_window, repeat_penalty);
                }
                if ngram_block_active {
                    llama::apply_ngram_block(&mut row, prev_committed);
                }
                drafted.push(argmax_u32(&row));
            }
        } else {
            // GPU argmax over (B-1) rows — one kernel, small D2H.
            let argmax_buf = verify_scratch.argmax.sub_offset(0, batch);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch)?;
            let mut host_idx = vec![0i32; batch];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4)
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
            }
            for &idx in &host_idx {
                drafted.push(idx as u32);
            }
        }
    } else {
        // Fallback: per-row weight_gemv loop.
        for i in 1..b {
            let hidden_row = draft_scratch.x.sub_offset(i * h, h);
            llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
            let logits = gpu.download_f32(&target.scratch.logits)?;
            debug_assert_eq!(logits.len(), vocab);
            if use_temp_sampling {
                let mut probs = Vec::with_capacity(vocab);
                softmax_temp_into(&logits, temp, &mut probs);
                let u = xorshift_next_unit(rng_state);
                let t = sample_categorical(&probs, u);
                draft_probs_at_drafted.push(probs[t as usize]);
                drafted.push(t);
                draft_softmaxes.push(probs);
            } else if host_path_active {
                let mut row = logits.clone();
                if rp_active {
                    llama::apply_repeat_penalty(&mut row, prev_committed, repeat_window, repeat_penalty);
                }
                if ngram_block_active {
                    llama::apply_ngram_block(&mut row, prev_committed);
                }
                drafted.push(argmax_u32(&row));
            } else {
                drafted.push(argmax_u32(&logits));
            }
        }
    }
    } // close else (DFlash draft path)

    for i in 1..b {
        block[i] = drafted[i];
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_draft_end = std::time::Instant::now();

    // ── 5b. N-gram override (DFlash path only) ───────────────────────────
    // When an n-gram cache is supplied, walk the block left-to-right. For
    // each position i, look up the bigram (block[i-2], block[i-1]) → t. If
    // the cache has a high-enough count for t, override block[i] with t.
    // Chained: subsequent lookups use the (possibly-overridden) prior
    // tokens. Chained overrides only "compound" when the cache captures
    // multi-step patterns (e.g. boilerplate phrases, code indentation).
    //
    // Cost: two HashMap lookups per block position = microseconds.
    //
    // Limitation: dflash's draft_forward already ran against the ORIGINAL
    // draft argmax block; overrides don't feed back into the draft. So
    // downstream positions' target-hidden cross-attention was computed
    // against the un-overridden block. In practice this doesn't matter
    // because the per-position target attention at verify time reruns
    // anyway — what matters is target's argmax at position i versus
    // block[i+1] (the override).
    // Skip bigram override when PLD is the draft source: per Goose §3,
    // PLD tokens have 2–18× higher acceptance than bigram (TR) tokens
    // (median 6×). Overriding PLD with a bigram guess strictly lowers τ.
    if pld_spine.is_none() {
        if let Some(ng) = ngram_cache {
            if prev_committed.len() >= 2 {
                let mut a = prev_committed[prev_committed.len() - 2];
                let mut bb = seed_token;
                for i in 1..b {
                    if let Some((tok, _cnt)) = ng.predict(a, bb) {
                        block[i] = tok;
                        // Also reflect the override in `drafted` so the committed
                        // sequence reported back to the caller matches what was
                        // actually verified against the target.
                        drafted[i] = tok;
                    }
                    a = bb;
                    bb = block[i];
                }
            }
        }
    }

    // ── 6. Snapshot DeltaNet pre-verify, run verify (advances state by B) ─
    //
    // If a GdnTape is supplied, the verify forward also records the
    // per-LA-layer (q, k, v, α, β) innovation tape so the rollback can
    // replay just the GDN recurrence for `accept+1` steps without
    // re-running the target.
    target_snap.save_from(&target.dn_state, gpu)?;
    // Mutable variable to allow both verify capture + rollback replay usage.
    let mut gdn_tape_opt = gdn_tape;
    // MoE targets can't populate the tape: forward_prefill_batch_with_pbs's
    // eligibility check rejects MoE (qwen35.rs `DeltaNetMoe|FullAttnMoe => false`),
    // so verify falls through to the per-token loop which doesn't write the
    // tape. With Some(tape) downstream `replay_gdn` then runs on zero-init
    // buffers, corrupting `dn_state.conv_states` and hanging the next cycle.
    // Force None so the fallback replay path (batched forward on committed
    // tokens) runs instead — correct at ~3-5 ms/cycle extra vs proper tape
    // replay. Remove once batched MoE prefill + tape recording lands.
    let target_has_moe = target.weights.layers.iter().any(|lw| matches!(
        lw,
        qwen35::LayerWeights::DeltaNetMoe(_) | qwen35::LayerWeights::FullAttnMoe(_),
    ));
    if target_has_moe {
        gdn_tape_opt = None;
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_start = std::time::Instant::now();
    let verify_out = verify_dflash_block(
        gpu, target, &block, position, hidden_rb,
        gdn_tape_opt.as_deref_mut(),
        use_temp_sampling || host_path_active,  // full target logits needed for rejection sampling, RP, or n-gram block
        verify_scratch,
    )?;

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_end = std::time::Instant::now();

    // ── 7. Acceptance ──────────────────────────────────────────────────
    //
    // Greedy path: longest prefix where block[i+1] == argmax_per_pos[i].
    //   bonus = argmax_per_pos[accept_len].
    //
    // Rejection-sampling path (temp > 0):
    //   For each i in 0..B-1:
    //     t = block[i+1] (draft sampled this at position start+i+1)
    //     p_d = draft_softmax[i][t]
    //     p_t = target_softmax[i][t]  (softmax of verify logits row i, same temp)
    //     u = rng
    //     accept if u * p_d < p_t
    //     else: rejected → bonus = sample from (p_target - p_draft)+
    //   If all accepted → bonus = sample from target_softmax[B-1].
    let mut accept_len = 0usize;
    let bonus_token;
    if use_temp_sampling {
        let tgt_logits = &verify_out.logits_per_pos;
        debug_assert_eq!(tgt_logits.len(), b * vocab);
        debug_assert_eq!(draft_softmaxes.len(), b - 1);
        let mut target_probs = Vec::with_capacity(vocab);
        let mut rejected_bonus: Option<u32> = None;
        // CACTUS (Hao & Mou 2026, arXiv:2604.04987 Corollary 5) relaxes the
        // Leviathan acceptance ratio by a KL-bounded bump √(2δ·q·(1−q)),
        // trading controlled divergence from the verifier for higher τ.
        // δ==0 reduces to vanilla SpS. Paper's strongest setting is δ=1.0.
        let use_cactus = cactus_delta > 0.0;
        for i in 0..b - 1 {
            softmax_temp_into(&tgt_logits[i * vocab..(i + 1) * vocab], temp, &mut target_probs);
            let t = block[i + 1] as usize;
            let p_d = draft_probs_at_drafted[i].max(f32::MIN_POSITIVE);
            let p_t = target_probs[t];
            // Bumped acceptance probability: γ* = min(p_t + √(2·δ·p_t·(1−p_t)), 1).
            // When δ==0 → γ* = p_t (standard Leviathan & Chen 2023).
            let accept_prob = if use_cactus {
                let bump = (2.0 * cactus_delta * p_t * (1.0 - p_t)).max(0.0).sqrt();
                (p_t + bump).min(1.0)
            } else {
                p_t
            };
            let u = xorshift_next_unit(rng_state);
            if u * p_d <= accept_prob {
                accept_len += 1;
            } else {
                // Rejected — sample bonus from the CACTUS-revised target h
                // (§2.3, Theorem 2), not raw q. h is built in-place over
                // target_probs (loop breaks right after, so no reuse):
                //   h(t)   = γ*
                //   h(i≠t) = (1−γ*)/(1−q(t)) · q(i)
                if use_cactus {
                    let qn = p_t.clamp(0.0, 1.0);
                    let gamma_star = accept_prob;
                    if qn >= 1.0 - 1e-6 {
                        // Degenerate: q is (near) one-hot on t; h is one-hot on t too.
                        for v in target_probs.iter_mut() { *v = 0.0; }
                        target_probs[t] = 1.0;
                    } else {
                        let scale = (1.0 - gamma_star) / (1.0 - qn);
                        for (j, v) in target_probs.iter_mut().enumerate() {
                            *v = if j == t { gamma_star } else { scale * *v };
                        }
                    }
                }
                let u2 = xorshift_next_unit(rng_state);
                rejected_bonus = Some(sample_residual(
                    &target_probs, &draft_softmaxes[i], u2,
                ));
                break;
            }
        }
        bonus_token = if let Some(b) = rejected_bonus {
            b
        } else {
            // All accepted: sample from target_softmax at position B-1.
            let i = b - 1;
            softmax_temp_into(&tgt_logits[i * vocab..(i + 1) * vocab], temp, &mut target_probs);
            let u = xorshift_next_unit(rng_state);
            sample_categorical(&target_probs, u)
        };
    } else {
        // Greedy path. If RP or n-gram-block is active, re-derive argmax per
        // row after applying penalties to the full target logits (requires
        // want_full_logits). `prev_committed` carries the emitted history
        // used as the penalty / block window.
        let argmax_per_pos: std::borrow::Cow<'_, [u32]> = if host_path_active {
            let tgt_logits = &verify_out.logits_per_pos;
            debug_assert_eq!(tgt_logits.len(), b * vocab);
            let mut out: Vec<u32> = Vec::with_capacity(b);
            let mut row = vec![0f32; vocab];
            for i in 0..b {
                row.copy_from_slice(&tgt_logits[i * vocab..(i + 1) * vocab]);
                if rp_active {
                    llama::apply_repeat_penalty(&mut row, prev_committed, repeat_window, repeat_penalty);
                }
                if ngram_block_active {
                    llama::apply_ngram_block(&mut row, prev_committed);
                }
                out.push(argmax_u32(&row));
            }
            std::borrow::Cow::Owned(out)
        } else {
            std::borrow::Cow::Borrowed(verify_out.argmax_per_pos.as_slice())
        };
        for i in 0..b - 1 {
            if argmax_per_pos[i] == block[i + 1] {
                accept_len += 1;
            } else {
                break;
            }
        }
        bonus_token = argmax_per_pos[accept_len];
    }

    // ── 7b. Seed-prediction oracle (Task #93 Phase B) ───────────────────
    // Three position-based proxies for the next cycle's `seed_token`
    // (= this cycle's `bonus_token`). See comment at top of file for the
    // reasoning — the PRD's "naive argmax at rejection boundary" proxy is
    // 0 % by construction (the accept loop broke precisely there), which
    // we measure as REJ_MATCH to document the dead-end. TAIL_MATCH and
    // ANYPOS_MATCH are the actually-usable ceilings.
    let rej_proxy: Option<u32> = if accept_len + 1 < b {
        Some(drafted[accept_len + 1])
    } else {
        None
    };
    let tail_proxy: u32 = drafted[b - 1];
    let anypos_hit: bool = drafted[1..b].iter().any(|&t| t == bonus_token);
    let rej_hit: bool = rej_proxy == Some(bonus_token);
    let tail_hit: bool = tail_proxy == bonus_token;
    SEED_ORACLE_TOTAL.fetch_add(1, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.fetch_add(accept_len as u64, Ordering::Relaxed);
    if rej_hit {
        SEED_ORACLE_REJ_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if tail_hit {
        SEED_ORACLE_TAIL_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if anypos_hit {
        SEED_ORACLE_ANYPOS_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if rej_proxy.is_none() {
        SEED_ORACLE_FULLACCEPT.fetch_add(1, Ordering::Relaxed);
    }
    if std::env::var("HIPFIRE_DFLASH_SEED_ORACLE").ok().as_deref() == Some("1") {
        let s = read_seed_oracle_stats();
        let denom = s.total.max(1) as f32;
        eprintln!(
            "[seed-oracle] cycle: accept_len={} b={} bonus={} rej={:?}/{} tail={}/{} anypos={} fullacc={} | cum rej={:.3} tail={:.3} anypos={:.3} mean_accept={:.2}",
            accept_len, b, bonus_token, rej_proxy, rej_hit,
            tail_proxy, tail_hit, anypos_hit, rej_proxy.is_none(),
            s.rej_match as f32 / denom,
            s.tail_match as f32 / denom,
            s.anypos_match as f32 / denom,
            s.accept_len_sum as f32 / denom,
        );
    }

    // ── 8. Committed sequence ───────────────────────────────────────────
    // committed[0] is the seed_token (already emitted by prev iter). The
    // caller's output stream appends committed[1..]. We include seed in
    // committed because target KV/state must be at position seed+accept_len+1
    // after this step.
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for i in 0..accept_len {
        committed.push(drafted[i + 1]);
    }
    committed.push(bonus_token);
    let committed_count = committed.len();
    debug_assert_eq!(committed_count, accept_len + 2);

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_accept_end = std::time::Instant::now();

    // ── 9. Append accepted target hidden rows to target_hidden_host ─────
    // Verify wrote B rows into hidden_rb. We keep the first accept_len+1
    // (= committed_count - 1) because the last committed token (bonus) is
    // ALREADY reflected in target state + will get its hidden captured on
    // the NEXT verify when it's forwarded as block[0].
    //
    // Wait: bonus_token is placed at position `position + accept_len + 1`.
    // Its hidden was captured at ring slot (verify start + accept_len),
    // which corresponds to the B-th verify forward position = position +
    // accept_len. That's the bonus position if we identify it correctly.
    //
    // Actually every verify position writes one hidden row. Position i of
    // the B-verify corresponds to absolute position `position + i`, so:
    //   block[0] hidden captured at ring slot (head - B + 0) → pos=position
    //   block[1] hidden captured at ring slot (head - B + 1) → pos=position+1
    //   ...
    //   block[accept_len] hidden captured → pos=position+accept_len (THIS is the last committed before bonus)
    //   block[accept_len+1] hidden captured → pos=position+accept_len+1 (this would be bonus; but target's prediction at that slot is what drove the bonus choice)
    //
    // The bonus token is what target WOULD predict at position+accept_len+1
    // given the B-verify input. Its hidden was NOT captured at that
    // position — the hidden at that slot is for `block[accept_len+1]`, a
    // REJECTED draft token's target forward. We can't use that hidden for
    // the committed bonus token.
    //
    // Resolution: DON'T append bonus-token hidden here. Next iter's
    // verify will forward the bonus token at its position (position +
    // committed_count - 1) as its new block[0], capturing proper hidden
    // and target state there. Committed_count - 1 rows appended here
    // covers positions [position..position + committed_count - 2] =
    // [position..position + accept_len]. Bonus at position+accept_len+1
    // sits in no-man's land — its hidden will materialize on next iter.
    //
    // This matches the reference's `target_hidden = ...[:, :accept_len+1, :]`
    // pattern which slices the verify's hidden output to accept_len+1
    // rows — NOT accept_len+2.
    let rows_to_keep = accept_len + 1;
    if ctx_slice.is_some() {
        // ctx_slice path: CPU shadow still required for the window slice.
        let hidden_block = download_hidden_block(gpu, hidden_rb, b)?;
        target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);
    } else {
        // Fast path: scatter straight from hidden_rb into draft scratch on GPU.
        // No D2H, no CPU reshape, no next-cycle H2D.
        //
        // Verify just wrote B slots to hidden_rb; we want the first
        // `rows_to_keep` (= accept+1) of those. Pass block_size=b so the
        // scatter function aligns to the verify-block origin, not the
        // ring tail.
        scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            &draft_scratch.target_hidden,
            position,
            b,
            rows_to_keep,
        )?;
        // Keep draft_forward's incremental-upload tracker in sync so any future
        // ctx_slice=Some call in the same session doesn't try to re-upload what
        // GPU already has; and so the assertion-in-draft path stays coherent.
        draft_scratch.uploaded_target_hidden_rows = position + rows_to_keep;
        // Track the absolute positions of the rows we just appended. These are
        // the logical positions `position..position+rows_to_keep` plus the
        // current target KV compact_offset (zero pre-eviction; non-zero after).
        // Used by the next cycle's `positions_k` construction.
        let co = target.kv_cache.compact_offset as i32;
        for p in 0..rows_to_keep {
            draft_scratch
                .target_hidden_abs_positions
                .push(position as i32 + p as i32 + co);
        }
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_scatter_end = std::time::Instant::now();

    // ── 10. Rewind DeltaNet + replay committed tokens ────────────────────
    // After verify, target state reflects B forwards. We need it to reflect
    // `committed_count - 1 = accept_len + 1` forwards (the seed + accepted
    // draft tokens). The bonus token is NOT replayed — it will be
    // block[0] of the next iter. This keeps the invariant that before each
    // verify, target state is at position `start` (= pre-verify position).
    target_snap.restore_to(&mut target.dn_state, gpu)?;

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_restore_end = std::time::Instant::now();
    // Tape-replay path (0.1.7 perf): if a GdnTape was captured during verify,
    // replay the GatedDeltaNet recurrence for (accept+1) steps using the
    // recorded (q, k, v, α, β) tuples — no full-target re-run needed. The
    // FullAttention layers don't need explicit rewind because the next
    // verify (starting at position + accept + 1) will overwrite their KV
    // cache slots [position + accept + 1 .. position + accept + 1 + B),
    // which subsumes the previously-written [position..position + B) range.
    //
    // Fallback (no tape): batched forward_prefill_batch over (accept+1)
    // tokens, same as the prior version — re-runs the full target but one
    // batched call instead of (accept+1) sequential decodes.
    if let Some(tape) = gdn_tape_opt.as_deref() {
        tape.replay_gdn(
            gpu, &target.weights, &target.config, &mut target.dn_state, accept_len + 1,
        )?;
    } else {
        let replay_tokens = &committed[..accept_len + 1];
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            replay_tokens,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None, None, None, None,
        )?;
    }
    // Target state is now at position + accept_len + 1. KV cache has
    // written K/V at positions [position..position+accept_len]. The bonus
    // token's K/V will be written on the next iter's verify (at position
    // `position + accept_len + 1`) as part of that iter's block[0] forward.

    if phase_on {
        gpu.hip.device_synchronize()?;
        let t_end = std::time::Instant::now();
        let us_draft   = t_draft_end.duration_since(t_spec_start).as_micros();
        let us_ngram   = t_verify_start.duration_since(t_draft_end).as_micros();
        let us_verify  = t_verify_end.duration_since(t_verify_start).as_micros();
        let us_accept  = t_accept_end.duration_since(t_verify_end).as_micros();
        let us_scatter = t_scatter_end.duration_since(t_accept_end).as_micros();
        let us_restore = t_restore_end.duration_since(t_scatter_end).as_micros();
        let us_replay  = t_end.duration_since(t_restore_end).as_micros();
        let us_total   = t_end.duration_since(t_spec_start).as_micros();
        eprintln!(
            "[phase] B={} accept={} draft={}µs ngram={}µs verify={}µs \
             cmpr={}µs scatter={}µs restore={}µs replay={}µs | total={}µs",
            b, accept_len, us_draft, us_ngram, us_verify, us_accept,
            us_scatter, us_restore, us_replay, us_total,
        );
    }
    let _ = (t_phase, t_draft_end, t_verify_start, t_verify_end,
             t_accept_end, t_scatter_end, t_restore_end);

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Run the DFlash draft forward + lm_head, return the raw per-position draft
/// logits as a host `Vec<f32>` of length `(b - 1) * vocab`.
///
/// Shared factor-out of the draft-producing half of spec_step_dflash — used by
/// spec_step_ddtree to feed Algorithm 1 with per-position top-K. The vanilla
/// DFlash path doesn't call this because it takes the argmax/softmax directly
/// on GPU (smaller D2H); the tree path needs raw logits for top-K + log-norm.
///
/// Leaves `draft_scratch.x` populated with draft hidden rows, so callers that
/// also want argmax for diagnostics can walk those rows afterward (not used
/// here). Does NOT advance the target KV cache or DeltaNet state — only the
/// draft forward runs.
#[cfg(feature = "deltanet")]
fn run_dflash_draft_for_logits(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
) -> HipResult<Vec<f32>> {
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");

    // Block: [seed, mask, mask, ...].
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Step 1: D2D embedding lookup per block slot (parallels spec_step_dflash).
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            crate::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }

    // Step 2: Positions + optional ctx_slice (identical to spec_step_dflash).
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => position,
    };
    let ctx_start = position - effective_ctx_len;
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
    let th_offset = ctx_start * ne * h;
    let th_slice: &[f32] = &target_hidden_host[th_offset..];

    // Step 3: Draft forward (fills draft_scratch.x with per-position draft
    // hidden rows).
    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        Some(th_slice),
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: Apply target.lm_head to draft hidden rows [1..B). Same batched
    // GEMM paths as spec_step_dflash. Unlike the vanilla path we download
    // the full (B-1) × vocab logits so the tree builder can compute top-K.
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;

    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            gpu.gemm_q8_0_batched(&w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            gpu.gemm_hfq4g256(&w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_hfq4g256(
                &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            // MQ3 has no scalar batched gemm (unlike MQ4), so use the
            // WMMA-residual lm_head wrapper which pre-zeros Y. Same pattern
            // as the verify path — keeps draft and verify byte-identical
            // for MQ3 targets.
            let r2 = gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ3G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    let host_logits = match gpu.download_f32(&logits_batch) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(logits_batch);
            return Err(e);
        }
    };
    let _ = gpu.free_tensor(logits_batch);
    debug_assert_eq!(host_logits.len(), batch * vocab);
    Ok(host_logits)
}

/// Like `run_dflash_draft_for_logits` but does the top-K + log-sum-exp
/// ON GPU via `topk_logsumexp_batched_f32`, returning only the top-K
/// tokens and log-probs per row. Used by `spec_step_ddtree_batched` to
/// skip the ~20 ms CPU sort and the 15 MB logits D2H.
///
/// Returns `(top_tokens, top_log_probs)` each of size `(b-1) * k` in
/// row-major order (same convention as `ddtree::topk_from_logits`).
#[allow(clippy::too_many_arguments)]
fn run_dflash_draft_for_topk_gpu(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
    k: usize,
) -> HipResult<(Vec<u32>, Vec<f32>)> {
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");
    assert!(k >= 1 && k <= 8, "topk k={} must be in [1, 8]", k);

    // Step 1-3: identical to run_dflash_draft_for_logits — embed, positions,
    // draft forward. Duplicating the small glue to avoid a refactor risk;
    // this path is shipped after. (Could factor out, but the savings is <50
    // lines and the call site is stable.)
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            crate::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            crate::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => position,
    };
    let ctx_start = position - effective_ctx_len;
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
    let th_offset = ctx_start * ne * h;
    let th_slice: &[f32] = &target_hidden_host[th_offset..];

    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        Some(th_slice),
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: lm_head → [batch × vocab] logits (GPU-resident).
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;
    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            gpu.gemm_q8_0_batched(&w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            gpu.gemm_hfq4g256(&w_out.buf, &hidden_rows, &logits_batch, w_out.m, w_out.k, batch)
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_hfq4g256(
                &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ3G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    // Step 5: GPU top-K + log-sum-exp. Writes [batch × k] indices + log-probs.
    let topk_idx_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_val_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_result = gpu.topk_logsumexp_batched_f32(
        &logits_batch, &topk_idx_gpu, &topk_val_gpu, vocab, k, batch,
    );
    let _ = gpu.free_tensor(logits_batch);
    if let Err(e) = topk_result {
        let _ = gpu.free_tensor(topk_idx_gpu);
        let _ = gpu.free_tensor(topk_val_gpu);
        return Err(e);
    }

    // Step 6: D2H just the top-K outputs (tiny — 8 × 15 × 4 = 480 bytes for k=8).
    let mut idx_host: Vec<i32> = vec![0i32; batch * k];
    let mut val_host: Vec<f32> = vec![0f32; batch * k];
    let idx_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, batch * k * 4)
    };
    let val_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(val_host.as_mut_ptr() as *mut u8, batch * k * 4)
    };
    gpu.hip.memcpy_dtoh(idx_bytes, &topk_idx_gpu.buf)?;
    gpu.hip.memcpy_dtoh(val_bytes, &topk_val_gpu.buf)?;
    let _ = gpu.free_tensor(topk_idx_gpu);
    let _ = gpu.free_tensor(topk_val_gpu);

    let top_tokens: Vec<u32> = idx_host.into_iter().map(|x| x as u32).collect();
    Ok((top_tokens, val_host))
}

/// Enumerate all root-to-leaf paths in a DdTree. Returns paths as Vec<Vec<usize>>
/// where each inner Vec is the sequence of node indices from the first
/// child-of-root (depth 1) down to a leaf. Leaves are nodes with no children
/// in the tree; if the tree is empty (N=0) this returns a single empty path.
fn enumerate_paths(tree: &crate::ddtree::DdTree) -> Vec<Vec<usize>> {
    if tree.nodes.is_empty() {
        return vec![Vec::new()];
    }
    let mut leaves: Vec<usize> = Vec::new();
    for i in 0..tree.nodes.len() {
        let slot = i + 1;
        if tree.child_maps[slot].is_empty() {
            leaves.push(i);
        }
    }
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(leaves.len());
    for &leaf_idx in &leaves {
        let mut path: Vec<usize> = Vec::new();
        let mut cur: i32 = leaf_idx as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = tree.nodes[cur as usize].parent_index;
        }
        path.reverse();
        paths.push(path);
    }
    paths
}

/// DDTree speculative step (Ringel & Romano 2026, our hybrid-arch port).
///
/// Flow per cycle:
///   1. Run DFlash draft, download raw (B-1) × vocab logits.
///   2. CPU top-K + log-norm per row → per-position (tokens, log-probs).
///   3. Algorithm 1: best-first heap builds up to `tree_budget` tree nodes.
///   4. Snapshot target state (pre-seed). Forward seed once to get posterior[0]
///      and the post-seed branch point; snapshot post-seed state.
///   5. For each root-to-leaf path in the tree, forward each node sequentially
///      through `forward_scratch`; on first visit of a node slot, record its
///      target argmax as `posterior[slot]`. Restore post-seed state between paths.
///   6. Greedy walk: follow target's argmax down the tree to the longest
///      accepted path + bonus token.
///   7. Restore to pre-seed, re-forward (seed + accepted path) with hidden
///      capture so the next cycle's DFlash draft has valid target_hidden_host.
///
/// Cost per cycle: O(N) target forwards where N is the node budget (paper
/// uses 60; we default to `draft_cfg.block_size` = 16 for a cheaper spike).
/// That's ~5× the batched-verify cost of spec_step_dflash; no batched tree
/// attention on hybrid arch would change that, but per-path verify is the
/// correctness-first path (LA state is not polluted across branches).
///
/// Temp=0 only for now — rejection-sampling / CACTUS integration is deferred
/// until the greedy signal looks promising. Paper's DDTree numbers are
/// temp=0 too, so this matches the reference setup.
#[cfg(feature = "deltanet")]
pub fn spec_step_ddtree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    assert!(b >= 2, "spec_step_ddtree: block_size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );

    // ── 1. Run DFlash draft, download raw logits ─────────────────────────
    let draft_logits = run_dflash_draft_for_logits(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        b,
    )?;

    // ── 2. Per-position top-K + log-normalize (CPU) ───────────────────────
    let (top_tokens, top_log_probs) =
        crate::ddtree::topk_from_logits(&draft_logits, b - 1, vocab, tree_topk);

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = crate::ddtree::build_ddtree_tree_with_cutoff(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        tree_budget,
        ddtree_logw_cutoff(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    // Edge case: empty tree (shouldn't happen if budget≥1 and b≥2, but guard).
    // With zero nodes there's nothing to verify — just forward seed, sample,
    // commit. Mirrors the behavior of a B=2 DFlash cycle.
    // Note: `forward_scratch_with_hidden` runs the final rmsnorm + lm_head
    // internally and leaves the next-token logits in `scratch.logits` — do
    // NOT call weight_gemv again on scratch.x (that's pre-rmsnorm hidden
    // and produces incorrect logits).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            seed_token,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        let hidden_block = download_hidden_block(gpu, hidden_rb, 1)?;
        target_hidden_host.extend_from_slice(&hidden_block[..1 * ne * h]);
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Snapshot pre-seed target state ─────────────────────────────────
    //
    // We verify each root-to-leaf path via `verify_dflash_block` starting
    // from the pre-seed state — this is the same batched target forward
    // DFlash uses for its verify, so we stay byte-exact with the non-tree
    // path. Between paths we restore pre-seed (both DN and KV cache; KV
    // overwrites happen naturally because each verify writes to the same
    // position range starting at `position`).
    target_snap.save_from(&target.dn_state, gpu)?;
    // post_seed_snap is allocated by the caller but unused in this path —
    // kept in the signature so the API stays compatible with potentially
    // sharing-the-seed-forward optimizations in a later rev. Suppress the
    // unused warning without asking the caller to annotate.
    let _ = &post_seed_snap;

    let mut posterior: Vec<u32> = vec![0; 1 + tree.num_nodes()];
    let mut posterior_set: Vec<bool> = vec![false; 1 + tree.num_nodes()];

    // ── 5. Per-path verify via verify_dflash_block ───────────────────────
    //
    // For each root-to-leaf path, run the batched target verify on
    // [seed_token, path_tokens...]. verify_dflash_block gives us argmax
    // per position via the same code path as spec_step_dflash, which
    // guarantees no numerical drift vs baseline at temp=0. Per-node
    // posterior records are first-visit-wins — all paths traversing the
    // same ancestor produce the same argmax at that ancestor's slot.
    let paths = enumerate_paths(&tree);
    for path in &paths {
        // Build verify block: [seed] + path_tokens.
        let mut verify_block: Vec<u32> = Vec::with_capacity(1 + path.len());
        verify_block.push(seed_token);
        for &ni in path {
            verify_block.push(tree.nodes[ni].token);
        }

        // Restore pre-seed state before each verify. DN state via snapshot;
        // KV cache self-overwrites at positions [position, position+N).
        target_snap.restore_to(&mut target.dn_state, gpu)?;

        // NOTE: verify_dflash_block takes &mut HiddenStateRingBuffer (not
        // Option); we pass our buffer but its writes get clobbered by the
        // step-8 replay. That's fine — we only read hidden_rb in step 9
        // after the replay. Path verifies DO advance the ring buffer head
        // but the final replay brings it right back.
        let verify_out = verify_dflash_block(
            gpu,
            target,
            &verify_block,
            position,
            hidden_rb,
            None,
            false, // want_full_logits=false — greedy only for now
            verify_scratch,
        )?;

        // verify_out.argmax_per_pos has length N = verify_block.len().
        // argmax_per_pos[i] = target's predicted NEXT token at position
        // `position + i`. That's:
        //   i=0          → prediction after seed = what should match block[1]
        //                  = posterior at root slot
        //   i=1..N-1     → prediction after node at path-position i-1
        //                  = posterior at path[i-1]'s slot
        // (We don't use argmax_per_pos[N-1] because we'd need a child of
        // the leaf, which the tree doesn't have — greedy walk stops there.)
        if !posterior_set[0] {
            posterior[0] = verify_out.argmax_per_pos[0];
            posterior_set[0] = true;
        }
        for (i, &ni) in path.iter().enumerate() {
            let slot = ni + 1;
            if !posterior_set[slot] && i + 1 < verify_out.argmax_per_pos.len() {
                posterior[slot] = verify_out.argmax_per_pos[i + 1];
                posterior_set[slot] = true;
            }
        }
    }

    // ── 6. Greedy walk: longest accepted path + bonus ─────────────────────
    let (accepted_node_indices, bonus_token) =
        crate::ddtree::follow_verified_tree(&tree, &posterior);
    let accept_len = accepted_node_indices.len();

    // ── 7. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 8. Tape-capturing verify on the committed path, then tape replay ─
    //
    // The tape records per-LA-layer (q, k, v, α, β) innovations for the
    // tokens it processes. Replaying the tape then advances DN state
    // through THOSE tokens. So the tape MUST be captured from a verify
    // whose block contains the actual committed tokens — any divergence
    // (e.g., capturing from the top-1 chain when the tree accepted a
    // rank>0 branch) feeds wrong LA updates into the next cycle's state.
    //
    // For topk=1 the tree's only path IS the top-1 chain, so committed
    // (length accept_len+1) is a prefix of the full-B DFlash block; we
    // still verify at full B here to stay batch-size-identical with the
    // DFlash baseline, then replay just the first accept_len+1 tape
    // steps. That path is byte-exact with baseline.
    //
    // For topk>1 the committed path may contain branch tokens that don't
    // appear in dflash_block's top-1 chain. In that case we fall back to
    // running the tape capture over the committed path directly — not
    // batch-size-equal to DFlash but tokens-correct. Some cross-cycle
    // numerical drift vs baseline is the tradeoff; output should remain
    // a valid target-greedy sequence.
    let topk1_is_committed_prefix = accept_len > 0 && committed[1..=accept_len].iter().enumerate()
        .all(|(d, &tok)| tok == top_tokens[d * tree_topk]);
    let tape_block: Vec<u32> = if topk1_is_committed_prefix || accept_len == 0 {
        // Safe to use full-B top-1 block (byte-exact with DFlash path).
        let mut vb: Vec<u32> = Vec::with_capacity(b);
        vb.push(seed_token);
        for d in 0..(b - 1) {
            vb.push(top_tokens[d * tree_topk]);
        }
        vb
    } else {
        // Accepted a branch — verify over the committed tokens to get
        // correct LA innovations.
        committed[..accept_len + 1].to_vec()
    };
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let _tape_verify = verify_dflash_block(
        gpu,
        target,
        &tape_block,
        position,
        hidden_rb,
        Some(gdn_tape),
        false,
        verify_scratch,
    )?;
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accept_len + 1,
    )?;
    // Target state is now at position + accept_len + 1. Bonus token's state
    // is deferred to next cycle's block[0], matching spec_step_dflash.

    // ── 9. Append (1 + accept_len) hidden rows to target_hidden_host ─────
    //
    // The tape-capturing verify wrote `tape_block.len()` rows to hidden_rb.
    // We want the FIRST (accept_len + 1) — positions [position, position +
    // accept_len] of the verified block. download_hidden_block returns the
    // most-recent N rows in order, so pulling tape_block.len() rows and
    // slicing to accept_len+1 grabs the right prefix.
    let hidden_rows_written = tape_block.len();
    let hidden_block = download_hidden_block(gpu, hidden_rb, hidden_rows_written)?;
    let rows_to_keep = accept_len + 1;
    target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Batched tree-verify counterpart of `spec_step_ddtree`. Replaces the
/// per-path DFS with a single `verify_dflash_block_tree` call using the
/// FA tree-attention mask infrastructure (commits 835aa46 / f0ee980 /
/// 704bf11). Same return value and side-effect semantics as the per-path
/// version — callers swap the two transparently.
///
/// Correctness notes:
///
/// - **FA side (tree-exact):** each tree node's Q attends only to its
///   ancestors + prompt. The mask is -inf on non-ancestor in-block keys
///   so exp-sum collapses, matching per-path DFS argmaxes exactly at
///   temp=0.
/// - **GDN side (linear-replay approximation):** in the tree forward
///   the recurrent GDN kernel advances state sequentially through the
///   linearized token order `[seed, n0, n1, ...]`. For siblings at the
///   same tree depth this cross-contaminates state — node b's S-state
///   update sees node a's innovations even though they're alternatives,
///   not sequential. At `topk=1` the tree is a pure chain so state
///   advance is identical to DFlash (byte-exact). At `topk>1` the FA
///   posteriors are still correct (ancestor attention), but the GDN
///   contribution to each node's hidden has small drift vs per-path DFS.
/// - **Tape/commit path (correct):** we do a SECOND verify on the
///   committed prefix (no tree) for tape capture. That tape is byte-
///   exact with per-path DFS's committed-prefix verify, so LA state
///   advances correctly after the cycle completes.
///
/// Target: replaces 8–16 forwards per cycle (per-path DFS) with 2
/// forwards per cycle (tree verify + linear tape capture). Converts the
/// τ wins (+40–46% on creative/essay) into wall-clock wins.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_ddtree_batched(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    scratch: &DdtreeScratch,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    assert!(b >= 2, "spec_step_ddtree_batched: block_size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );
    // Unused in the batched path (no per-path DFS), kept in signature for
    // API compatibility with `spec_step_ddtree` so callers can switch by
    // flipping a single fn pointer.
    let _ = &post_seed_snap;

    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft / topk / build /
    // pre_verify / verify. Used to diagnose where the wall-clock goes.
    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft+topk / build /
    // pre_verify / verify. The draft and top-K are fused into one GPU-
    // resident path now — no separate timer.
    let debug_tm = std::env::var("DDTREE_TIMING").is_ok();
    let t_all = std::time::Instant::now();

    // ── 1+2. GPU-resident draft + per-row top-K + log-sum-exp ────────────
    // Keeps logits on device; returns only (b-1) × k indices + log-probs
    // to the host. Replaces the prior 15 MB D2H + CPU sort pair (~34 ms)
    // with an on-device top-K (~µs) plus a ~480 byte D2H.
    let (top_tokens, top_log_probs) = run_dflash_draft_for_topk_gpu(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        b,
        tree_topk,
    )?;

    let t_draft = t_all.elapsed();
    let t_topk = t_draft; // fused with draft now

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = crate::ddtree::build_ddtree_tree_with_cutoff(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        tree_budget,
        ddtree_logw_cutoff(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    let t_build = t_all.elapsed();

    // Empty-tree shortcut (identical to spec_step_ddtree's path).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu, &target.weights, &target.config, seed_token, position,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch, hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        let hidden_block = download_hidden_block(gpu, hidden_rb, 1)?;
        target_hidden_host.extend_from_slice(&hidden_block[..ne * h]);
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Linearize the tree into (tokens, positions, mask_host, parents) ─
    let (verify_tokens, verify_positions, mask_host, parent_host) =
        crate::ddtree::linearize_tree_with_parents(&tree, seed_token, position as u32);
    let big_n = verify_tokens.len();
    debug_assert_eq!(big_n, 1 + tree.num_nodes());
    debug_assert_eq!(parent_host.len(), big_n);

    // ── 5. Upload mask to GPU into the persistent bias scratch ───────────
    //
    // Reuses `scratch.attn_bias` (sized for max_budget at init time), so
    // per cycle we only pay for the htod of the current cycle's mask. The
    // FA kernel reads at `row * block_cols + col` with block_cols = big_n;
    // unused tail space in the buffer is never accessed.
    assert!(
        big_n <= scratch.max_n,
        "tree big_n {} exceeds scratch.max_n {} (increase DdtreeScratch size)",
        big_n, scratch.max_n,
    );
    {
        let mask_bytes = unsafe {
            std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_host.len() * 4)
        };
        gpu.hip.memcpy_htod(&scratch.attn_bias.buf, mask_bytes)?;
    }

    // ── 5b. Upload parent_indices for tree-aware LA kernels ──────────────
    //
    // ON BY DEFAULT as of 2026-04-24 — Task #101 Phase 3d validation bench
    // (3-run medians on 27B MQ4 asym3 b12-k2, commit 4a3f2b3):
    //   code:     110.0 → 119.1 tok/s (+8.3 %)   τ 6.80 → 7.30 (+7 %)
    //   prose:     52.3 →  57.8 tok/s (+10.5 %)  τ 3.00 → 3.52 (+17 %)
    //   instruct:  42.1 →  47.4 tok/s (+12.6 %)  τ 2.02 → 2.47 (+22 %)
    // Coherence-gate-dflash passes on all 4 tests. Mechanism: tree-aware
    // LA kernels read parent_indices to walk ancestor chains correctly at
    // topk>1, so the fast-tape path fires on 90 %+ of cycles instead of
    // the slow-path re-verify that used to trigger on sibling pollution.
    //
    // Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected.
    let use_tree_la = std::env::var("HIPFIRE_DDTREE_TREE_LA").ok().as_deref() != Some("0");
    if use_tree_la {
        let parent_bytes = unsafe {
            std::slice::from_raw_parts(parent_host.as_ptr() as *const u8, parent_host.len() * 4)
        };
        gpu.hip.memcpy_htod(&scratch.parent_indices.buf, parent_bytes)?;
    }

    // ── 6. Snapshot pre-seed target state ─────────────────────────────────
    target_snap.save_from(&target.dn_state, gpu)?;

    // ── 7. Tree verify: single batched forward with tree-attention mask ──
    //
    // Key optimization: pass `gdn_tape` INTO the tree verify so GDN
    // innovations get captured in the linear tree-traversal order. For the
    // topk=1 (or topk>1 where the accepted path coincides with the top-1
    // linear chain) case, the committed path is a contiguous prefix of the
    // linear order — so replaying `tape[0..accept_len+1]` advances LA state
    // correctly and we save an entire forward pass. For topk>1 paths that
    // diverge from the linear prefix, we fall back to a second verify over
    // the committed tokens (step 10 below).
    //
    // argmax_per_pos[i] = target's argmax prediction at slot i in the
    // linearization, i.e. what comes AFTER the token at that slot.
    // Sub-offset view sized to the exact big_n × big_n the current tree needs.
    // scratch.attn_bias is sized for the worst case (max_n² = (1+max_budget)²),
    // but when the actual tree is smaller (e.g. topk=1 linear-chain trees
    // don't fill max_budget), forward_prefill_batch's assert rejects the
    // oversized buffer. The kernel only ever reads up to big_n² floats via
    // `tree_bias[row × block_cols + col]`, so a view is equivalent and keeps
    // the assert semantics meaningful.
    let attn_bias_view = scratch.attn_bias.sub_offset(0, big_n * big_n);
    // Parent-indices sub-view sized to big_n (one i32 per slot; stored as
    // 4 × big_n raw bytes). Only populated when HIPFIRE_DDTREE_TREE_LA=1.
    let parent_view = scratch.parent_indices.sub_offset(0, big_n * 4);
    // Path B (slow-path-kill, work-in-progress): when enabled, supply the
    // per-FA-layer pre-RoPE K capture scratch so tree verify can dump K
    // BEFORE rope_partial_interleaved mutates it. Slow path then gathers
    // accepted rows out of the scratch, re-RoPEs with committed phases,
    // and quant-writes to the committed kv slots — no full re-verify
    // forward. CONSUMER NOT YET WIRED: capture is currently a no-op
    // overhead until the slow-path branch is replaced. Keep gated until
    // the eyeball-tested smoke (see PRD trap surface) passes.
    let pre_rope_capture = if std::env::var("HIPFIRE_DDTREE_PATH_B_CAPTURE")
        .ok().as_deref() == Some("1")
        && !scratch.pre_rope_k.is_empty()
    {
        Some(scratch.pre_rope_k.as_slice())
    } else {
        None
    };
    let ctx = qwen35::TreeVerifyCtx {
        positions: &verify_positions,
        attn_bias: &attn_bias_view,
        parent_indices: if use_tree_la { Some(&parent_view) } else { None },
        pre_rope_k_capture: pre_rope_capture,
    };
    let t_pre_verify = t_all.elapsed();
    let verify_out = verify_dflash_block_tree(
        gpu, target, &verify_tokens, position, hidden_rb, Some(gdn_tape), false, ctx,
        verify_scratch,
    )?;
    let posterior = verify_out.argmax_per_pos;
    let t_post_verify = t_all.elapsed();

    // ── 8. Greedy walk: longest accepted path + bonus ─────────────────────
    let (accepted_node_indices, bonus_token) =
        crate::ddtree::follow_verified_tree(&tree, &posterior);
    let accept_len = accepted_node_indices.len();

    // ── 9. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 10. Tape/hidden path selection ────────────────────────────────────
    //
    // Fast path: accepted tree nodes occupy linear slots [0, 1, 2, ...,
    // accept_len - 1] in the tree. Their tokens are in linear-order
    // positions [1, 2, ..., accept_len] of the tape (slot 0 = seed). The
    // tree tape captures innovations at those same linear slots, so
    // `replay_gdn(accept_len + 1)` is exact with DFlash.
    //
    // Fast path is ALWAYS the case at topk=1 (tree is a chain, accepted
    // indices are [0, 1, 2, ...]). At topk>1 it holds iff the greedy walk
    // picked the rank-0 child at every accepted step (no sibling detour).
    //
    // Slow path (topk>1 detour): re-capture tape on the committed prefix
    // with a second verify, then replay as before. Costs +1 forward but
    // keeps LA state byte-correct.
    // HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when
    // committed path == linearization prefix. Diagnostic — quantifies the
    // cost of always re-running the committed tokens through a non-tree
    // verify to fix KV cache entries at committed slots (topk>1 siblings
    // at same depth otherwise race and the LAST write wins regardless of
    // which sibling was committed).
    let force_slow = std::env::var("HIPFIRE_DDTREE_FORCE_SLOW").ok().as_deref() == Some("1");
    let spine_accept = accepted_node_indices.iter().enumerate()
        .all(|(i, &ni)| ni == i);
    let fast_tape_ok = !force_slow && spine_accept;
    // Per-cycle fast/slow accounting. HIPFIRE_DDTREE_TAPE_DUMP=1 emits a
    // per-cycle line to stderr; useful to quantify how often the slow-path
    // 2nd verify fires at a given topk / workload. Aggregate stats are
    // printed by dflash_spec_demo at end-of-generation via this thread-local.
    thread_local! {
        static DDTREE_FAST_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static DDTREE_SLOW_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    if fast_tape_ok {
        DDTREE_FAST_COUNT.with(|c| c.set(c.get() + 1));
    } else if !force_slow {
        DDTREE_SLOW_COUNT.with(|c| c.set(c.get() + 1));
    }
    if std::env::var("HIPFIRE_DDTREE_TAPE_DUMP").ok().as_deref() == Some("1") {
        let fast = DDTREE_FAST_COUNT.with(|c| c.get());
        let slow = DDTREE_SLOW_COUNT.with(|c| c.get());
        eprintln!(
            "[ddtree-tape] cycle: fast_tape_ok={} accept_len={} spine_accept={} tree_la={} (cumulative fast={}/slow={})",
            fast_tape_ok, accept_len, spine_accept, use_tree_la, fast, slow,
        );
    }
    let hidden_rows_written;
    if fast_tape_ok {
        // Tape already captured in tree verify. Restore + replay directly.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
    } else if std::env::var("HIPFIRE_DDTREE_PATH_B_CAPTURE").ok().as_deref() == Some("1")
              && !scratch.pre_rope_k.is_empty()
    {
        // Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full
        // re-verify with a gather + per-commit RoPE + quant-write chain
        // that operates on the pre-RoPE K captured during tree verify
        // (qwen35.rs:3486 — Phase 1 capture). Plus the existing tape
        // gather scaffolding from ecbc49d.
        //
        // Path A failed because gathered K carried stale RoPE phase. Path
        // B fixes that by re-applying RoPE for the COMMITTED slot phases
        // before quant-writing back to the cache.
        //
        // CORRECTNESS-CRITICAL: the dflash coherence battery
        // (scripts/coherence-gate-dflash.sh) is the ONLY barrier between
        // a Path B regression and a corrupted-output release. Token
        // attractors here look like +τ/+tok-s wins on stat gates. Run
        // the eyeball check before trusting any result.
        let n_positions = accept_len + 1;
        let kv = &mut target.kv_cache;
        let n_kv_heads = kv.n_kv_heads;
        let head_dim = kv.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // ── (a) Tape gather (qkv/alpha/beta innovations into committed order)
        let tape_idx_host: Vec<i32> = std::iter::once(0i32)
            .chain(accepted_node_indices.iter().map(|&i| (i + 1) as i32))
            .collect();
        let tape_idx_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(tape_idx_host.as_ptr() as *const u8, n_positions * 4)
        };
        gpu.hip.memcpy_htod(&scratch.parent_indices.buf, tape_idx_bytes)?;
        gdn_tape.gather_accepted(
            gpu,
            &scratch.parent_indices,
            &scratch.tape_gather_scratch,
            n_positions,
        )?;

        // ── (b) Per-FA-layer K rotate + V gather + quant-write
        //
        // For K: gather pre-RoPE K rows (captured BEFORE the original
        // rope_partial in qwen35.rs:3486) by accepted indices into a
        // contiguous F32 buffer, apply RoPE with COMMITTED positions
        // [start_pos, start_pos+1, ...], then quant-write to KV cache at
        // those committed slots. The Q half of rope_partial is throwaway —
        // we feed verify_scratch.prefill_batch's fa_q_batch as a scratch
        // and ignore the rotated Q.
        //
        // For V: V doesn't carry a position-dependent rotation, so a
        // pure byte gather (raced slot → committed slot) is correct. Same
        // pattern Path A used.
        let pbs = verify_scratch.prefill_batch.as_ref()
            .expect("Path B requires VerifyScratch.prefill_batch (set during DdtreeScratch init)");

        // Tree-verify K source indices (one per accepted committed slot, in
        // pre-RoPE K scratch which has positions [0..big_n] in tree-
        // linearization order, so 0 = seed slot, i+1 = tree node i).
        let k_src_idx_host: Vec<i32> = std::iter::once(0i32)
            .chain(accepted_node_indices.iter().map(|&i| (i + 1) as i32))
            .collect();
        let k_src_idx_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(k_src_idx_host.as_ptr() as *const u8, n_positions * 4)
        };
        // Reuse parent_indices buffer for the K gather indices (it was
        // already used for the tape gather above; re-upload now).
        gpu.hip.memcpy_htod(&scratch.parent_indices.buf, k_src_idx_bytes)?;

        // Committed slot positions for RoPE + KV write: [start_pos+0..start_pos+accept_len].
        let pos_host: Vec<i32> = (0..n_positions).map(|i| (position + i) as i32).collect();
        let pos_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, n_positions * 4)
        };
        gpu.hip.memcpy_htod(&scratch.kv_gather_indices.buf, pos_bytes)?;

        // Absolute KV slots for V gather: [position+0, position+1+acc[0], ...]
        // (V is the same as Path A: byte gather from raced slots to committed).
        let v_src_abs_host: Vec<i32> = std::iter::once(position as i32)
            .chain(accepted_node_indices.iter().map(|&i| (position + 1 + i) as i32))
            .collect();
        let v_src_abs_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v_src_abs_host.as_ptr() as *const u8, n_positions * 4)
        };
        // Park the V indices in the tape_gather_scratch's first 4×n bytes —
        // tape gather is done; the buffer is free until next cycle. (Avoids
        // adding yet another tiny i32 buffer to DdtreeScratch.)
        gpu.hip.memcpy_htod(&scratch.tape_gather_scratch.buf, v_src_abs_bytes)?;

        let v_bpp = n_kv_heads * (head_dim / 32) * 34; // Q8 V (all asym* modes use Q8 V)

        let n_rot = (target.config.head_dim as f32 * target.config.partial_rotary_factor) as usize;

        for (fa_idx, layer_idx) in target.config.layer_types.iter()
            .enumerate()
            .filter_map(|(li, lt)| if *lt == qwen35::LayerType::FullAttention { Some(li) } else { None })
            .enumerate()
        {
            // 1. Gather pre-RoPE K rows by k_src_idx into pbs.fa_k_batch.
            //    Each row is n_kv_heads * head_dim F32 = kv_dim*4 bytes.
            gpu.kv_compact_gather(
                &scratch.pre_rope_k[fa_idx],
                &pbs.fa_k_batch,
                &scratch.parent_indices,
                kv_dim * 4,
                n_positions,
            )?;

            // 2. Apply RoPE in-place to gathered K with committed positions.
            //    Q is throwaway — fa_q_batch is large enough.
            gpu.rope_partial_interleaved_f32_batched(
                &pbs.fa_q_batch, &pbs.fa_k_batch, &scratch.kv_gather_indices,
                target.config.n_heads, target.config.n_kv_heads, target.config.head_dim,
                n_rot, target.config.rope_theta, n_positions,
            )?;

            // 3. V gather via the existing kv_compact_gather pattern.
            gpu.kv_compact_gather(
                &kv.v_gpu[layer_idx],
                &scratch.kv_gather_scratch_v,
                &scratch.tape_gather_scratch,
                v_bpp, n_positions,
            )?;

            // 4. Quant-write K (rotated, in pbs.fa_k_batch) + V (gathered,
            //    in scratch.kv_gather_scratch_v) to the committed KV slots.
            //    All asym* and q8 KV variants supported. F16 unquantized
            //    isn't on the batched path so we panic here — see the
            //    fa_batched_ok gate in qwen35.rs:3081.
            if kv.quant_asym3 {
                let ct = kv.givens_cos.as_ref().expect("asym3 requires Givens cos");
                let st = kv.givens_sin.as_ref().expect("asym3 requires Givens sin");
                // The batched K writer expects a contiguous K source of
                // [n × n_kv_heads × head_dim] F32 — pbs.fa_k_batch is
                // exactly that. We give it the REAL kv.v_gpu as the V
                // dst (so writer indices stay in-bounds for absolute slot
                // numbers). The V values it writes are garbage (sourced
                // from pbs.fa_v_batch, leftover from the last FA layer)
                // but we OVERWRITE every committed V slot below from a
                // proper gather of the raced-but-correctly-quantized V
                // values. So the garbage V write is a transient no-op.
                gpu.kv_cache_write_asym3_batched(
                    &kv.k_gpu[layer_idx], &kv.v_gpu[layer_idx],
                    &pbs.fa_k_batch, &pbs.fa_v_batch,
                    &scratch.kv_gather_indices,
                    ct, st, n_kv_heads, head_dim, n_positions,
                )?;
                // V byte-gather: read pre-quantized V from raced slots
                // [position+0, position+1+acc[0], ...] into a contiguous
                // scratch, then memcpy scratch → kv.v_gpu at committed
                // slots [position..position+accept_len]. Using a scratch
                // intermediate avoids same-slot src=dst memcpys (which
                // are HIP UB) when the accept chain happens to hit the
                // rank-0 prefix early.
                gpu.kv_compact_gather(
                    &kv.v_gpu[layer_idx],
                    &scratch.kv_gather_scratch_v,
                    &scratch.tape_gather_scratch,
                    v_bpp, n_positions,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &kv.v_gpu[layer_idx].buf, position * v_bpp,
                    &scratch.kv_gather_scratch_v.buf, 0,
                    n_positions * v_bpp,
                )?;
            } else {
                // TODO: asym4 / asym2 / q8 paths — same pattern as asym3
                // but with the matching kv_cache_write_*_batched call.
                // For initial Phase 2 prototype, panic so we notice if a
                // non-asym3 model accidentally enables Path B.
                panic!("Path B Phase 2 only supports asym3 KV today (got: q8={} asym4={} asym2={})",
                    kv.quant_q8, kv.quant_asym4, kv.quant_asym2);
            }
        }

        // ── (c) Replay GDN tape on the committed-order tape.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            n_positions,
        )?;
        hidden_rows_written = big_n;
    } else {
        // Default slow path: re-verify the committed prefix to get a
        // linear-order tape AND correctly RoPE'd K written to committed
        // slots. ~40-50 ms cost on 27B. Path B kill is opt-in via
        // HIPFIRE_DDTREE_PATH_B_CAPTURE=1.
        let tape_block: Vec<u32> = committed[..accept_len + 1].to_vec();
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        let _tape_verify = verify_dflash_block(
            gpu, target, &tape_block, position, hidden_rb, Some(gdn_tape), false,
            verify_scratch,
        )?;
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = tape_block.len();
    }

    // ── 11. Append (1 + accept_len) hidden rows to target_hidden_host ────
    // Default slow path's 2nd verify wrote accept_len+1 rows in committed
    // order → first N rows are correct. Fast path: rank-0 chain == linear
    // prefix → first N rows still correct. Path A slow path keeps tree-
    // verify's big_n rows in linearization order → CPU-gather committed
    // rows out of the block.
    let hidden_block = download_hidden_block(gpu, hidden_rb, hidden_rows_written)?;
    let row_stride = ne * h;
    if hidden_rows_written == big_n && !fast_tape_ok {
        target_hidden_host.extend_from_slice(&hidden_block[0..row_stride]);
        for i in 0..accept_len {
            let src_row = accepted_node_indices[i] + 1;
            let src_start = src_row * row_stride;
            target_hidden_host.extend_from_slice(&hidden_block[src_start..src_start + row_stride]);
        }
    } else {
        let rows_to_keep = accept_len + 1;
        target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * row_stride]);
    }

    if debug_tm {
        let total = t_all.elapsed();
        eprintln!(
            "[ddtree-tm] draft={:.2}ms topk={:.2}ms build={:.2}ms pre_verify={:.2}ms verify={:.2}ms total={:.2}ms  (N={} accept={})",
            t_draft.as_secs_f64() * 1000.0,
            (t_topk - t_draft).as_secs_f64() * 1000.0,
            (t_build - t_topk).as_secs_f64() * 1000.0,
            (t_pre_verify - t_build).as_secs_f64() * 1000.0,
            (t_post_verify - t_pre_verify).as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            big_n, accept_len,
        );
    }

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Auxiliary DeltaNet snapshots that Path C Phase 2 needs but Phase 1
/// does not. Pre-allocated by the caller (one of each at session start)
/// and re-used across cycles. Pass `Some(...)` to enable Phase 2 (Step 2
/// + Step 3 of the PRD); pass `None` for Phase 1 behavior.
///
/// - `parent_pre_snap`: the DN state at "after position + accepted_main −
///   1", i.e. immediately before the branch's parent is forwarded. Saved
///   by the orchestrator, restored before the branch FA forward.
/// - `main_end_snap`: the DN state at "after position + accepted_main",
///   i.e. the main path's final committed state. Saved before the branch
///   FA forward. Restored on branch reject so callers see the same
///   `target.dn_state` they would have under Phase 1.
#[cfg(feature = "deltanet")]
pub struct Phase2Snapshots<'a> {
    pub parent_pre_snap: &'a mut DeltaNetSnapshot,
    pub main_end_snap: &'a mut DeltaNetSnapshot,
}

/// Path C — main-path-first lazy FA-only re-verify (PRD orchestrator).
///
/// PRD: `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
///
/// **Phase 1** (`path_c_phase2 = None`): runs Step 1 only — linear verify
/// on the DDTree's greedy main chain, no branches. Output is bit-exact
/// with calling [`verify_dflash_block`] directly on the same chain (by
/// construction — the only forward run IS that linear verify).
///
/// **Phase 2** (`path_c_phase2 = Some(...)`): adds Steps 2+3 — at most
/// **one** lazy branch FA-only re-verify per cycle (the candidate sibling
/// at fork depth = `accepted_main` whose first token equals the main
/// verify's bonus). On branch accept the commit is extended by the
/// accepted branch tokens; on reject behavior matches Phase 1.
///
/// **No KV backup buffers needed**. Phase 2 lets the branch FA forward
/// freely overwrite KV slots past the main path's accept boundary
/// (`position + accepted_main + 1` onwards). Stale branch K/V written
/// past the new commit boundary is tolerated by the same mechanism that
/// makes [`spec_step_dflash`]'s rejected-tail K/V tolerable: the next
/// decode cycle's verify starts at the new commit boundary and overwrites
/// every slot it will subsequently read. See `spec_step_dflash` ~line
/// 3097 for the original write-up of this invariant.
///
/// Signature mirrors [`spec_step_ddtree_batched`] for drop-in dispatch
/// from the daemon, minus the `post_seed_snap` and `scratch` arguments
/// (this function uses `target_snap` + `path_c_phase2` snapshots instead).
#[cfg(feature = "deltanet")]
pub fn spec_step_ddtree_path_c(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
    path_c_phase2: Option<Phase2Snapshots<'_>>,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    assert!(b >= 2, "spec_step_ddtree_path_c: block_size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );

    // ── 1+2. GPU-resident draft + per-row top-K (identical to batched path) ──
    let (top_tokens, top_log_probs) = run_dflash_draft_for_topk_gpu(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        b,
        tree_topk,
    )?;

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    let tree = crate::ddtree::build_ddtree_tree_with_cutoff(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        tree_budget,
        ddtree_logw_cutoff(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    // ── 3b. Empty-tree shortcut (matches spec_step_ddtree_batched). Also
    //       handles the degenerate case where build returned a tree with no
    //       direct root child (would imply main_path is empty too).
    let main_path: Vec<usize> = if tree.nodes.is_empty() {
        Vec::new()
    } else {
        crate::ddtree::select_main_path(&tree)
    };
    if main_path.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu, &target.weights, &target.config, seed_token, position,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch, hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        let hidden_block = download_hidden_block(gpu, hidden_rb, 1)?;
        target_hidden_host.extend_from_slice(&hidden_block[..ne * h]);
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Build the main-path verify chain: [seed, main_path tokens…] ───
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(1 + main_path.len());
    verify_tokens.push(seed_token);
    for &ni in &main_path {
        verify_tokens.push(tree.nodes[ni].token);
    }

    // ── 5. Snapshot pre-seed target DN state for tape replay below. ──────
    target_snap.save_from(&target.dn_state, gpu)?;

    // ── 6. Linear verify on the main chain. No tree mask, no linearization
    //       phase poisoning — RoPE phases match committed slots exactly.
    //       This is the entire "Step 1" of the PRD's three-step pattern.
    let main_verify_out = verify_dflash_block(
        gpu, target, &verify_tokens, position, hidden_rb, Some(gdn_tape), false,
        verify_scratch,
    )?;
    let main_posterior = main_verify_out.argmax_per_pos;
    debug_assert_eq!(main_posterior.len(), 1 + main_path.len());

    // ── 7. Greedy walk on the main chain (see Phase 1 doc-comment). ──────
    let mut accepted_main: usize = 0;
    for j in 0..main_path.len() {
        let drafted_tok = tree.nodes[main_path[j]].token;
        if main_posterior[j] == drafted_tok {
            accepted_main = j + 1;
        } else {
            break;
        }
    }
    let bonus_token = main_posterior[accepted_main];

    // ── 8. Drive DN state to "main-end" via tape replay. Same as Phase 1.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accepted_main + 1,
    )?;

    // ── 9. Download main verify's hidden rows. download_hidden_block
    //       reads the LAST `n` rows from the ring; after main verify's
    //       writes those last `n` rows ARE the main chain's hidden states.
    //       Branch verify (if any) will overwrite this region of the ring,
    //       so we must download main rows BEFORE running it.
    let main_chain_len = 1 + main_path.len();
    let main_hidden_block = download_hidden_block(gpu, hidden_rb, main_chain_len)?;
    let row_stride = ne * h;

    // ── 10. Phase 2: lazy branch FA-only re-verify. At most one branch
    //        per cycle is structurally able to accept (PRD §"Architecture
    //        /Step 2"): the candidate is the sibling at fork depth =
    //        `accepted_main` whose first token equals `bonus_token`. All
    //        other branches' first tokens are guaranteed to mismatch the
    //        target's posterior at their fork depth (because main verify
    //        accepted main_path's child there, OR rejected with a posterior
    //        != any siblings' tokens). So one candidate at most.
    let mut accepted_branch_indices: Vec<usize> = Vec::new();
    let mut effective_bonus = bonus_token;
    let mut branch_hidden_block: Option<Vec<f32>> = None;
    // Diagnostic counters keyed by HIPFIRE_DDTREE_PATH_C_VERBOSE=1. Tracks
    // the funnel: how often does the tree even contain a sibling at the
    // right fork depth, how often does its first token match the main
    // verify's bonus, and how often does the branch FA forward then accept
    // ≥1 of its tokens. Lets us tell a "draft has no useful siblings"
    // signal apart from "draft has them but target rejects" signal.
    thread_local! {
        static PATH_C_TOTAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PATH_C_PHASE2: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PATH_C_HAS_FORK_SIBLING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PATH_C_CANDIDATE_FOUND: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PATH_C_BRANCH_ACCEPTED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static PATH_C_BRANCH_TOKENS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    PATH_C_TOTAL.with(|c| c.set(c.get() + 1));
    let verbose = std::env::var("HIPFIRE_DDTREE_PATH_C_VERBOSE").ok().as_deref() == Some("1");
    let mut diag_phase2 = false;
    let mut diag_fork_sibling = false;
    let mut diag_candidate = false;
    let mut diag_accept_branch: usize = 0;
    if let Some(snaps) = path_c_phase2 {
        diag_phase2 = true;
        PATH_C_PHASE2.with(|c| c.set(c.get() + 1));
        // Save the main-end DN state so we can roll back on branch reject.
        snaps.main_end_snap.save_from(&target.dn_state, gpu)?;

        // Find the unique candidate branch.
        let branches = crate::ddtree::enumerate_branches(&tree, &main_path, accepted_main);
        diag_fork_sibling = branches
            .iter()
            .any(|b| b.fork_depth as usize == accepted_main);
        if diag_fork_sibling {
            PATH_C_HAS_FORK_SIBLING.with(|c| c.set(c.get() + 1));
        }
        let candidate = branches.into_iter().find(|b| {
            b.fork_depth as usize == accepted_main
                && tree
                    .nodes
                    .get(b.chain[0])
                    .map(|n| n.token == bonus_token)
                    .unwrap_or(false)
        });
        diag_candidate = candidate.is_some();
        if diag_candidate {
            PATH_C_CANDIDATE_FOUND.with(|c| c.set(c.get() + 1));
        }

        if let Some(branch) = candidate {
            // Step 2.1: restore DN to the branch parent's pre-state.
            target_snap.restore_to(&mut target.dn_state, gpu)?;
            if accepted_main > 0 {
                gdn_tape.replay_gdn(
                    gpu,
                    &target.weights,
                    &target.config,
                    &mut target.dn_state,
                    accepted_main,
                )?;
            }
            // Save it: we'll restore here again on accept to replay
            // branch tape rows into the correct intermediate state.
            snaps.parent_pre_snap.save_from(&target.dn_state, gpu)?;

            // Step 2.2: run the branch FA forward. start_pos =
            // position + accepted_main is the parent's absolute slot;
            // forwarding [parent_tok, c0, c1, ...] there has the model
            // see committed RoPE phases (no linearization-slot phase
            // poisoning) and a freshly-restored DN state for LA layers.
            //
            // We pass `Some(gdn_tape)`: branch innovations are captured
            // into tape rows [0..1+chain.len()], CLOBBERING main verify's
            // captures. That's safe — main tape was already replayed in
            // step 8 to drive DN to main-end, we don't need it anymore.
            let parent_tok = if accepted_main == 0 {
                seed_token
            } else {
                tree.nodes[main_path[accepted_main - 1]].token
            };
            let mut branch_chain_tokens: Vec<u32> =
                Vec::with_capacity(1 + branch.chain.len());
            branch_chain_tokens.push(parent_tok);
            for &ni in &branch.chain {
                branch_chain_tokens.push(tree.nodes[ni].token);
            }
            let branch_start_pos = position + accepted_main;
            let branch_verify_out = verify_dflash_block(
                gpu, target, &branch_chain_tokens, branch_start_pos, hidden_rb,
                Some(gdn_tape), false, verify_scratch,
            )?;
            let branch_posterior = branch_verify_out.argmax_per_pos;
            debug_assert_eq!(branch_posterior.len(), 1 + branch.chain.len());

            // Step 2.3: greedy walk on the branch.
            //   branch_posterior[j] = target's predict at branch_start_pos+j+1
            //   given prefix [parent, c0, ..., c_{j-1}]. Accept c_j iff
            //   branch_posterior[j] == c_j.tok.
            let mut accepted_branch: usize = 0;
            for j in 0..branch.chain.len() {
                let drafted_tok = tree.nodes[branch.chain[j]].token;
                if branch_posterior[j] == drafted_tok {
                    accepted_branch = j + 1;
                } else {
                    break;
                }
            }

            diag_accept_branch = accepted_branch;
            if accepted_branch > 0 {
                PATH_C_BRANCH_ACCEPTED.with(|c| c.set(c.get() + 1));
                PATH_C_BRANCH_TOKENS.with(|c| c.set(c.get() + accepted_branch as u32));
            }

            if accepted_branch == 0 {
                // Branch reject: restore main-end DN state. KV writes the
                // branch did at slots > position+accepted_main are stale
                // but tolerated (next cycle overwrites — see fn doc-comment).
                snaps.main_end_snap.restore_to(&mut target.dn_state, gpu)?;
            } else {
                // Branch accept: extend commit by accepted_branch tokens.
                // Step 3: drive DN state to "after position + accepted_main +
                // accepted_branch" via tape replay on the branch tape we
                // just captured (rows [0..1+accepted_branch]).
                snaps.parent_pre_snap.restore_to(&mut target.dn_state, gpu)?;
                gdn_tape.replay_gdn(
                    gpu,
                    &target.weights,
                    &target.config,
                    &mut target.dn_state,
                    1 + accepted_branch,
                )?;

                // Branch hidden rows (1 + chain.len() rows for [parent,
                // c0..c_{chain.len()-1}]). Download now while they're the
                // most-recent ring writes; we'll skip parent's row at index 0
                // (it's a duplicate of the corresponding main verify row).
                let bb = download_hidden_block(gpu, hidden_rb, 1 + branch.chain.len())?;
                branch_hidden_block = Some(bb);

                // Update the bonus token to the branch's predict-after-last-
                // accepted slot (= branch_posterior[accepted_branch] —
                // same off-by-one convention as the main greedy walk).
                effective_bonus = branch_posterior[accepted_branch];

                // Capture which branch nodes we accepted so we can build
                // the final committed/drafted lists once we're out of the
                // Phase 2 block (where the borrow on `branch.chain` ends).
                accepted_branch_indices.extend_from_slice(&branch.chain[..accepted_branch]);
            }
        }
    }

    // ── 11. Build committed + drafted sequences. Includes any accepted
    //        branch tokens (empty when Phase 2 is off, no candidate, or
    //        candidate rejected).
    let total_accepted = accepted_main + accepted_branch_indices.len();
    let mut committed: Vec<u32> = Vec::with_capacity(total_accepted + 2);
    committed.push(seed_token);
    for j in 0..accepted_main {
        committed.push(tree.nodes[main_path[j]].token);
    }
    for &ni in &accepted_branch_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(effective_bonus);

    let mut drafted: Vec<u32> = Vec::with_capacity(1 + main_path.len() + accepted_branch_indices.len());
    drafted.push(seed_token);
    for &ni in &main_path {
        drafted.push(tree.nodes[ni].token);
    }
    for &ni in &accepted_branch_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 12. Append accepted hidden rows to target_hidden_host. Same row
    //        accounting as spec_step_dflash: keep `1 + accepted_count`
    //        rows (seed + accepted committed); the bonus-token row is
    //        deliberately NOT appended because its hidden was captured at
    //        a wrong-token slot (will materialize correctly on next cycle's
    //        verify). See spec_step_dflash §"step 9" comment block.
    let main_rows_to_keep = accepted_main + 1; // seed + accepted main
    target_hidden_host.extend_from_slice(&main_hidden_block[..main_rows_to_keep * row_stride]);
    if let Some(bb) = &branch_hidden_block {
        // Branch block layout: row 0 = parent (duplicate), rows
        // [1..1+accepted_branch] = c_0 .. c_{accepted_branch-1}.
        let start = row_stride;
        let end = (1 + accepted_branch_indices.len()) * row_stride;
        target_hidden_host.extend_from_slice(&bb[start..end]);
    }

    if verbose {
        let total = PATH_C_TOTAL.with(|c| c.get());
        let p2 = PATH_C_PHASE2.with(|c| c.get());
        let fs = PATH_C_HAS_FORK_SIBLING.with(|c| c.get());
        let cf = PATH_C_CANDIDATE_FOUND.with(|c| c.get());
        let ba = PATH_C_BRANCH_ACCEPTED.with(|c| c.get());
        let bt = PATH_C_BRANCH_TOKENS.with(|c| c.get());
        let mean_branch = if ba > 0 { bt as f32 / ba as f32 } else { 0.0 };
        let cand_rate = if p2 > 0 { 100.0 * cf as f32 / p2 as f32 } else { 0.0 };
        let accept_rate = if cf > 0 { 100.0 * ba as f32 / cf as f32 } else { 0.0 };
        eprintln!(
            "[path-c] cycle: phase2={} acc_main={} fork_sibling={} candidate={} accept_branch={} \
             | cumul: cycles={} phase2={} fork_sib={} cand={} ({:.1}% of phase2) \
             accept={} ({:.1}% of cand) mean_branch_tok={:.2}",
            diag_phase2, accepted_main, diag_fork_sibling, diag_candidate, diag_accept_branch,
            total, p2, fs, cf, cand_rate, ba, accept_rate, mean_branch,
        );
    }

    Ok(SpecStepResult {
        accepted: total_accepted,
        bonus_token: effective_bonus,
        drafted,
        committed,
    })
}

/// Seed `target_hidden_host` from the prompt by running the target over
/// each prompt token one at a time with hidden-state extraction enabled.
/// This is a slow but correct MVP path — the target already ran a fast
/// prefill earlier; this exists only to populate `hidden_rb` + host vec
/// with the prompt's layer-selected hidden states.
///
/// Callers with a fast-path prefill that already populates `hidden_rb`
/// should skip this and just call `download_hidden_block(hidden_rb, len)`
/// instead. For MVP we eat the redundant work because it's a one-shot
/// cost at session start.
pub fn seed_target_hidden_from_prompt(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    prompt_tokens: &[u32],
) -> HipResult<()> {
    // Reset target state to avoid double-prefill of the same context.
    target.reset_state(gpu);
    // Fast path: one batched prefill populates hidden_rb + KV + dn_state in a
    // single forward, instead of N per-token forwards. On 9B MQ4 with a
    // 6.2k-token prompt this drops prompt ingest from ~51s (121 tok/s) to
    // a few seconds, which is the primary cost of an agent's first turn.
    // `forward_prefill_batch` itself falls back to per-token internally if
    // the KV quant mode / batch size aren't on the fast path, so the call
    // is always safe — the effective cadence just varies.
    qwen35::forward_prefill_batch(
        gpu,
        &target.weights,
        &target.config,
        prompt_tokens,
        0,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        Some(hidden_rb),
        None,
        None,
        None,
    )?;
    // Gather the just-written rows from the ring buffer.
    let block = download_hidden_block(gpu, hidden_rb, prompt_tokens.len())?;
    target_hidden_host.extend_from_slice(&block);
    Ok(())
}

/// Mirror a TriAttention KV eviction into the DFlash draft's GPU-resident
/// `target_hidden` and `target_hidden_abs_positions`, so the draft's cross-
/// attention sees the same subset of context target now has.
///
/// `retain_mask` is the source-position retain selection returned by
/// `EvictionCtx::maybe_evict` (ascending, length == budget). An empty
/// `retain_mask` is a no-op — the caller should have skipped calling this
/// (CASK m-fold path returns empty because merged slots don't map cleanly
/// to a single source position).
///
/// Implementation: download the relevant `physical` rows of `target_hidden`,
/// reorder to `budget` rows on the host per `retain_mask`, upload back. Runs
/// at eviction cadence (~once per β decoded tokens) so the PCIe round-trip
/// is amortized — perf impact is small relative to the τ recovery.
///
/// Post-conditions:
/// - `draft_scratch.target_hidden_abs_positions` has exactly `budget` entries,
///   each pulled from `retain_mask[i]` of the pre-eviction abs_positions.
/// - `draft_scratch.target_hidden` GPU slots [0..budget) hold the retained
///   rows in ascending source order.
/// - `draft_scratch.uploaded_target_hidden_rows = budget` so the next
///   draft_forward sees the compacted layout as already-uploaded.
pub fn apply_eviction_retain_to_draft(
    gpu: &mut rdna_compute::Gpu,
    draft_scratch: &mut dflash::DflashScratch,
    retain_mask: &[u32],
    ne: usize,
    h: usize,
    physical: usize,
) -> HipResult<()> {
    if retain_mask.is_empty() {
        return Ok(());
    }
    let row_floats = ne * h;
    // Download only the populated prefix of target_hidden. `alloc_tensor`
    // is sized to max_ctx_len — we just need `physical` rows.
    let mut host = vec![0f32; physical * row_floats];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                host.as_mut_ptr() as *mut u8,
                host.len() * std::mem::size_of::<f32>(),
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &draft_scratch.target_hidden.buf)?;
    }
    let budget = retain_mask.len();
    let mut compacted = Vec::with_capacity(budget * row_floats);
    let mut new_abs = Vec::with_capacity(budget);
    for &src_idx in retain_mask {
        let s = src_idx as usize;
        let row = &host[s * row_floats..(s + 1) * row_floats];
        compacted.extend_from_slice(row);
        new_abs.push(
            *draft_scratch
                .target_hidden_abs_positions
                .get(s)
                .expect("retain_mask index out of range for abs_positions"),
        );
    }
    let dst_bytes = budget * row_floats * std::mem::size_of::<f32>();
    let compacted_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            compacted.as_ptr() as *const u8,
            dst_bytes,
        )
    };
    gpu.hip.memcpy_htod(&draft_scratch.target_hidden.buf, compacted_bytes)?;
    draft_scratch.target_hidden_abs_positions = new_abs;
    draft_scratch.uploaded_target_hidden_rows = budget;
    // The per-layer k_ctx/v_ctx projection cache is indexed by the
    // pre-eviction row layout. After compaction it's stale — rebuild on
    // the next draft_forward. One slow cycle per eviction is fine.
    draft_scratch.invalidate_draft_ctx_cache();
    Ok(())
}
