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

use crate::hfq::HfqFile;
use crate::llama::{self, KvCache};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use crate::tokenizer::Tokenizer;
use hip_bridge::{DeviceBuffer, HipResult};
use rdna_compute::Gpu;
use std::path::Path;

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// INT8 co-located K and V (default).
    Q8,
    /// Asymmetric: Q8 K + turbo4 V with boundary layers at Q8.
    AsymQ8Turbo4 { boundary: u8 },
    /// Q8 K + HF4-V.
    Q8kHf4v,
    /// TurboN (2/3/4) on both K and V.
    Turbo(u8),
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

        let kv_cache = match slot_config.kv_mode {
            KvMode::Q8 => KvCache::new_gpu_q8(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Q8kHf4v => KvCache::new_gpu_q8k_hf4v(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::AsymQ8Turbo4 { boundary } => KvCache::new_gpu_asym_q8k_turbo4v_boundary(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
                boundary,
                n_kv_layers,
            )?,
            KvMode::Turbo(bits) => KvCache::new_gpu_turbo(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
                bits,
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
