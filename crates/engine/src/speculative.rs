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
pub struct HiddenStateRingBuffer {
    pub layer_bufs: Vec<GpuTensor>,
    pub extract_layers: Vec<usize>,
    pub max_positions: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub written: usize,
}

impl HiddenStateRingBuffer {
    /// Allocate GPU ring buffer for `num_extract` target layers.
    pub fn new(
        gpu: &mut Gpu,
        num_target_layers: usize,
        num_extract: usize,
        hidden_dim: usize,
        max_positions: usize,
    ) -> HipResult<Self> {
        let extract_layers = dflash_extract_layer_ids(num_target_layers, num_extract);
        let mut layer_bufs = Vec::with_capacity(num_extract);
        for _ in 0..num_extract {
            layer_bufs.push(gpu.alloc_tensor(&[max_positions * hidden_dim], rdna_compute::DType::F32)?);
        }
        let _ = layer_bufs.len(); // silence unused in case of Vec field confusion
        Ok(Self {
            layer_bufs,
            extract_layers,
            max_positions,
            hidden_dim,
            head: 0,
            written: 0,
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
    /// as `[B * vocab_size]`. The spec step currently only needs argmax,
    /// but the full logits are kept so temp>0 rejection sampling (Phase
    /// 7+) can plug in without re-running the target.
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
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let vocab = target.config.vocab_size;
    let dim = target.config.dim;

    // Scratch buffer for per-token post-output-norm hidden, [B × dim].
    // Allocated fresh each verify; ~160 KB at dim=2560, B=16 — negligible
    // alloc overhead compared to the verify forward.
    let final_hidden =
        gpu.alloc_tensor(&[b * dim], rdna_compute::DType::F32)?;

    let batch_result = qwen35::forward_prefill_batch(
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
    );
    if let Err(e) = batch_result {
        let _ = gpu.free_tensor(final_hidden);
        return Err(e);
    }

    // Per-position lm_head: B sequential GEMVs into target.scratch.logits.
    let mut logits_per_pos: Vec<f32> = Vec::with_capacity(b * vocab);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);
    for i in 0..b {
        let hidden_row = final_hidden.sub_offset(i * dim, dim);
        let r = llama::weight_gemv(
            gpu, &target.weights.output, &hidden_row, &target.scratch.logits,
        );
        if let Err(e) = r {
            let _ = gpu.free_tensor(final_hidden);
            return Err(e);
        }
        let row = match gpu.download_f32(&target.scratch.logits) {
            Ok(r) => r,
            Err(e) => {
                let _ = gpu.free_tensor(final_hidden);
                return Err(e);
            }
        };
        debug_assert_eq!(row.len(), vocab);
        argmax_per_pos.push(argmax_u32(&row));
        logits_per_pos.extend_from_slice(&row);
    }

    let _ = gpu.free_tensor(final_hidden);

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
/// MVP path: downloads all `num_extract × hidden × B` floats via
/// `gpu.download_f32` per layer (fine at block size 16 + 5 layers: ~2.6 MB
/// per verify). Optimizable in 0.1.7 with a GPU-side scatter kernel.
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

    // Download every extract-layer buffer once (small — ≤ max_pos rows).
    let mut layer_data: Vec<Vec<f32>> = Vec::with_capacity(num_extract);
    for buf in &hidden_rb.layer_bufs {
        layer_data.push(gpu.download_f32(buf)?);
    }

    // Rearrange into per-position-then-per-extract-layer order.
    let mut out: Vec<f32> = Vec::with_capacity(b * num_extract * hidden);
    for pi in 0..b {
        let slot = (start_slot + pi) % max_pos;
        for ext in 0..num_extract {
            let src_off = slot * hidden;
            out.extend_from_slice(&layer_data[ext][src_off..src_off + hidden]);
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
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;

    assert!(b >= 2, "dflash block size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );

    // ── 1. block_output_ids seeded with prev bonus at [0], masks at [1..B] ──
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // ── 2. noise_embedding = target.embed_tokens(block) on host ──────────
    // B separate embedding_lookups into target.scratch.x, download each row.
    let mut noise_embedding: Vec<f32> = Vec::with_capacity(b * h);
    for &tok in &block {
        match target.weights.embd_format {
            crate::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &target.scratch.x, tok, h)?
            }
            crate::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &target.scratch.x, tok, h)?
            }
            crate::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &target.scratch.x, tok, h)?
            }
            crate::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &target.scratch.x, tok, h)?
            }
            _ => panic!("dflash: unsupported target embedding format for noise lookup"),
        }
        let row = gpu.download_f32(&target.scratch.x)?;
        debug_assert_eq!(row.len(), h);
        noise_embedding.extend_from_slice(&row);
    }

    // ── 3. Position arrays + optional context slice ─────────────────────
    // Q positions: the absolute positions of the block slots, [position..position+B).
    // K positions by default: all accepted context [0..position), then block [position..position+B).
    //
    // If `ctx_slice = Some(N)` is set, restrict the draft's context view to
    // the last `N` rows of target_hidden_host, with RoPE positions
    // [position-N..position+B). This tests whether distant context hurts
    // accept rate (e.g., if the draft was trained on shorter contexts).
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => position,
    };
    let ctx_start = position - effective_ctx_len;
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let positions_k: Vec<i32> =
        (ctx_start as i32..(position + b) as i32).collect();

    // Slice target_hidden_host to the last effective_ctx_len rows. When
    // ctx_slice is None, this is a no-op (ctx_start = 0). Row stride is
    // num_extract × hidden = ne * h.
    let th_offset = ctx_start * ne * h;
    let th_slice: &[f32] = &target_hidden_host[th_offset..];

    // ── 4. draft_forward ────────────────────────────────────────────────
    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        &noise_embedding,
        th_slice,
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
    let mut drafted: Vec<u32> = vec![seed_token]; // drafted[0] = seed (by convention; not used)
    let w_out = &target.weights.output;
    // Fast-paths on the lm_head:
    //   HFQ4G256 / MQ4G256 → single batched GEMM kernel (best: all math fused).
    //   Q8_0               → B-1 serial GEMVs with STAGED outputs + single D2H.
    //                         Saves the per-row PCIe sync that dominated the
    //                         MVP path (~15 × ~150 µs = 2.3 ms stall budget).
    //   Anything else      → per-row weight_gemv + download loop (fallback).
    let use_batched_gemm = matches!(
        w_out.gpu_dtype,
        rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256,
    );
    // Q8 lm_head is common on typed-up Qwen3.5 exports; we stage the output
    // writes into a contiguous buffer so the B-1 GEMVs don't serialize on
    // per-row D2H round-trips (even though the kernels themselves remain
    // serial — a true batched Q8 GEMM is queued for follow-up).
    let use_q8_staged = matches!(w_out.gpu_dtype, rdna_compute::DType::Q8_0);
    if use_batched_gemm {
        let batch = b - 1;
        // Draft's hidden rows 1..B live contiguously at offset [h, b*h).
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch =
            gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
        if matches!(w_out.gpu_dtype, rdna_compute::DType::MQ4G256) {
            let rotated =
                gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let rot_result = gpu.rotate_x_mq_batched(&hidden_rows, &rotated, h, batch);
            if let Err(e) = rot_result {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let gr = gpu.gemm_hfq4g256(
                &w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch,
            );
            let _ = gpu.free_tensor(rotated);
            if let Err(e) = gr {
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
        } else {
            // HFQ4G256: direct batched GEMM, no rotation.
            let gr = gpu.gemm_hfq4g256(
                &w_out.buf, &hidden_rows, &logits_batch,
                w_out.m, w_out.k, batch,
            );
            if let Err(e) = gr {
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
        }
        let logits_host = match gpu.download_f32(&logits_batch) {
            Ok(v) => v,
            Err(e) => {
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
        };
        let _ = gpu.free_tensor(logits_batch);
        debug_assert_eq!(logits_host.len(), batch * vocab);
        for i in 0..batch {
            let row = &logits_host[i * vocab..(i + 1) * vocab];
            drafted.push(argmax_u32(row));
        }
    } else if use_q8_staged {
        // Staged Q8 path: B-1 GEMVs writing into offset slots of a single
        // [B-1 × vocab] buffer, then ONE D2H download. Keeps kernels
        // async-enqueued on the stream without per-row PCIe sync points.
        let batch = b - 1;
        let logits_batch =
            gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
        for i in 0..batch {
            let hidden_row = draft_scratch.x.sub_offset((i + 1) * h, h);
            let row_out = logits_batch.sub_offset(i * vocab, vocab);
            let gr = gpu.gemv_q8_0(&w_out.buf, &hidden_row, &row_out, w_out.m, w_out.k);
            if let Err(e) = gr {
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
        }
        let logits_host = match gpu.download_f32(&logits_batch) {
            Ok(v) => v,
            Err(e) => {
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
        };
        let _ = gpu.free_tensor(logits_batch);
        debug_assert_eq!(logits_host.len(), batch * vocab);
        for i in 0..batch {
            let row = &logits_host[i * vocab..(i + 1) * vocab];
            drafted.push(argmax_u32(row));
        }
    } else {
        // Fallback: per-row weight_gemv loop (unchanged from MVP).
        for i in 1..b {
            let hidden_row = draft_scratch.x.sub_offset(i * h, h);
            llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
            let logits = gpu.download_f32(&target.scratch.logits)?;
            debug_assert_eq!(logits.len(), vocab);
            drafted.push(argmax_u32(&logits));
        }
    }
    for i in 1..b {
        block[i] = drafted[i];
    }

    // ── 6. Snapshot DeltaNet pre-verify, run verify (advances state by B) ─
    target_snap.save_from(&target.dn_state, gpu)?;
    let verify_out = verify_dflash_block(gpu, target, &block, position, hidden_rb)?;

    // ── 7. Acceptance: longest prefix where block[1..i+1] == posterior[0..i] ─
    let mut accept_len = 0usize;
    for i in 0..b - 1 {
        if verify_out.argmax_per_pos[i] == block[i + 1] {
            accept_len += 1;
        } else {
            break;
        }
    }
    let bonus_token = verify_out.argmax_per_pos[accept_len];

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
    let hidden_block = download_hidden_block(gpu, hidden_rb, b)?;
    let rows_to_keep = accept_len + 1;
    target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);

    // ── 10. Rewind DeltaNet + replay committed tokens ────────────────────
    // After verify, target state reflects B forwards. We need it to reflect
    // `committed_count - 1 = accept_len + 1` forwards (the seed + accepted
    // draft tokens). The bonus token is NOT replayed — it will be
    // block[0] of the next iter. This keeps the invariant that before each
    // verify, target state is at position `start` (= pre-verify position).
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    // Batched replay (0.1.7 perf): forward_prefill_batch over (accept+1)
    // tokens in one call, instead of (accept+1) sequential forward_scratch
    // calls. Same byte-exact effect on KV cache + DN state, but the per-layer
    // launches collapse to a single batched dispatch. Drops the replay cost
    // from ~10 ms (4 sequential decodes) to ~2-3 ms.
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
        None, None,
    )?;
    // Target state is now at position + accept_len + 1. KV cache has
    // written K/V at positions [position..position+accept_len]. The bonus
    // token's K/V will be written on the next iter's verify (at position
    // `position + accept_len + 1`) as part of that iter's block[0] forward.

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
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
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            tok,
            i,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
    }
    // Gather the just-written rows from the ring buffer.
    let block = download_hidden_block(gpu, hidden_rb, prompt_tokens.len())?;
    target_hidden_host.extend_from_slice(&block);
    Ok(())
}
