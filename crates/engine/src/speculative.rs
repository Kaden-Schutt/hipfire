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
    pub fn replay_gdn(
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
            let layer = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => l,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };

            // 1. conv1d + SiLU + split — advances conv_state, writes
            //    (q_raw, k_raw, v) into scratch.
            gpu.conv1d_silu_split_f32_n(
                &self.q_raw_scratch,
                &self.k_raw_scratch,
                &self.v_scratch,
                &self.qkv_bufs[la_idx],
                &layer.conv_weight,
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
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu, target, draft_tokens, start_pos, hidden_rb, gdn_tape, want_full_logits, None,
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
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu, target, draft_tokens, start_pos, hidden_rb, gdn_tape, want_full_logits,
        Some(tree_verify),
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
        gdn_tape,
        tree_verify,
    );
    // TODO(root-cause): tree-mode needs an explicit device sync here or
    // subsequent kernels (lm_head, argmax) see partially-written memory and
    // produce wrong argmaxes / illegal accesses. Suspected queue-ordering
    // bug in the tree-masked FA kernels — same-stream dispatch SHOULD
    // serialize but empirically doesn't on gfx1100. Cost: one pipeline
    // stall per cycle (~µs) but without it τ degrades from 2.0 to 1.1 and
    // destabilizes across runs. Fix the underlying race before shipping
    // to master.
    if batch_result.is_ok() && tree_verify.is_some() {
        if let Err(e) = gpu.hip.device_synchronize() {
            let _ = gpu.free_tensor(final_hidden);
            return Err(e);
        }
    }
    if let Err(e) = batch_result {
        let _ = gpu.free_tensor(final_hidden);
        return Err(e);
    }

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
        | rdna_compute::DType::MQ4G256 => true,
        _ => false,
    };

    if try_batched {
        let logits_batch =
            gpu.alloc_tensor(&[b * vocab], rdna_compute::DType::F32)?;
        // Q8_0 gemm_q8_0_batched has a hard MAX_BATCH=16 in the kernel, so
        // tree-verify blocks exceeding 16 (budget + 1 > 16) need chunking.
        // MQ4/HFQ4 kernels have no such cap — they take the single-shot path.
        let gemm_result = match w_out.gpu_dtype {
            rdna_compute::DType::Q8_0 => {
                const Q8_LM_MAX: usize = 16;
                let mut chunk_start = 0usize;
                let mut chunk_err: Option<hip_bridge::HipError> = None;
                while chunk_start < b {
                    let chunk_end = (chunk_start + Q8_LM_MAX).min(b);
                    let chunk_n = chunk_end - chunk_start;
                    let x_chunk = final_hidden.sub_offset(chunk_start * dim, chunk_n * dim);
                    let y_chunk = logits_batch.sub_offset(chunk_start * vocab, chunk_n * vocab);
                    let r = gpu.gemm_q8_0_batched(
                        &w_out.buf, &x_chunk, &y_chunk, w_out.m, w_out.k, chunk_n,
                    );
                    if let Err(e) = r {
                        chunk_err = Some(e);
                        break;
                    }
                    chunk_start = chunk_end;
                }
                match chunk_err {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            }
            rdna_compute::DType::HFQ4G256 => {
                gpu.gemm_hfq4g256(&w_out.buf, &final_hidden, &logits_batch, w_out.m, w_out.k, b)
            }
            rdna_compute::DType::MQ4G256 => {
                let rot = gpu.alloc_tensor(&[b * w_out.k], rdna_compute::DType::F32)?;
                let r1 = gpu.rotate_x_mq_batched(&final_hidden, &rot, w_out.k, b);
                if let Err(e) = r1 {
                    let _ = gpu.free_tensor(rot);
                    let _ = gpu.free_tensor(logits_batch);
                    let _ = gpu.free_tensor(final_hidden);
                    return Err(e);
                }
                let r2 = gpu.gemm_hfq4g256(&w_out.buf, &rot, &logits_batch, w_out.m, w_out.k, b);
                let _ = gpu.free_tensor(rot);
                r2
            }
            _ => unreachable!(),
        };
        if let Err(e) = gemm_result {
            let _ = gpu.free_tensor(logits_batch);
            let _ = gpu.free_tensor(final_hidden);
            return Err(e);
        }
        if want_full_logits {
            // Rejection-sampling path needs full target distribution.
            // Cost: B × vocab × 4 bytes D2H per verify (~15 MB at B=16 × 248K).
            let host_logits = match gpu.download_f32(&logits_batch) {
                Ok(v) => v,
                Err(e) => {
                    let _ = gpu.free_tensor(logits_batch);
                    let _ = gpu.free_tensor(final_hidden);
                    return Err(e);
                }
            };
            let _ = gpu.free_tensor(logits_batch);
            for i in 0..b {
                let row = &host_logits[i * vocab..(i + 1) * vocab];
                argmax_per_pos.push(argmax_u32(row));
            }
            logits_per_pos = host_logits;
        } else {
            // GPU-side batched argmax. Writes B i32 indices; we download just
            // 4*B bytes instead of the full B×vocab logits. Saves ~15 MB of
            // PCIe D2H per verify on the 4B Q8 lm_head (~3-5 ms/iter).
            let argmax_buf = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
            let ar = gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b);
            if let Err(e) = ar {
                let _ = gpu.free_tensor(argmax_buf);
                let _ = gpu.free_tensor(logits_batch);
                let _ = gpu.free_tensor(final_hidden);
                return Err(e);
            }
            let mut host_idx = vec![0i32; b];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, b * 4)
                };
                if let Err(e) = gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf) {
                    let _ = gpu.free_tensor(argmax_buf);
                    let _ = gpu.free_tensor(logits_batch);
                    let _ = gpu.free_tensor(final_hidden);
                    return Err(e);
                }
            }
            let _ = gpu.free_tensor(argmax_buf);
            let _ = gpu.free_tensor(logits_batch);
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
    gdn_tape: Option<&mut GdnTape>,
    temp: f32,
    rng_state: &mut u64,
    block_size_override: Option<usize>,
    ngram_cache: Option<&NgramCache>,
    prev_committed: &[u32],
    cactus_delta: f32,
    pld_spine: Option<&[u32]>,
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

    assert!(b >= 2, "dflash block size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );

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
    // noise_embedding = None: we wrote embeddings directly into
    // draft_scratch.x above via D2D (no host round-trip).
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
        rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256,
    );
    let use_q8_staged = matches!(w_out.gpu_dtype, rdna_compute::DType::Q8_0);
    if use_batched_gemm || use_q8_staged {
        // Unified batched path: one GEMM over B-1 rows, GPU-side argmax,
        // download just (B-1) × 4 bytes of indices.
        let batch = b - 1;
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch =
            gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;

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
                let r2 = gpu.gemm_hfq4g256(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
                let _ = gpu.free_tensor(rotated);
                r2
            }
            _ => unreachable!(),
        };
        if let Err(e) = gemm_result {
            let _ = gpu.free_tensor(logits_batch);
            return Err(e);
        }

        if use_temp_sampling {
            // Full D2H of (B-1)×vocab logits, CPU softmax+sample.
            let host_logits = match gpu.download_f32(&logits_batch) {
                Ok(v) => v,
                Err(e) => {
                    let _ = gpu.free_tensor(logits_batch);
                    return Err(e);
                }
            };
            let _ = gpu.free_tensor(logits_batch);
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
        } else {
            // GPU argmax over (B-1) rows — one kernel, small D2H.
            let argmax_buf = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
            let ar = gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch);
            if let Err(e) = ar {
                let _ = gpu.free_tensor(argmax_buf);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let mut host_idx = vec![0i32; batch];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4)
                };
                if let Err(e) = gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf) {
                    let _ = gpu.free_tensor(argmax_buf);
                    let _ = gpu.free_tensor(logits_batch);
                    return Err(e);
                }
            }
            let _ = gpu.free_tensor(argmax_buf);
            let _ = gpu.free_tensor(logits_batch);
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
            } else {
                drafted.push(argmax_u32(&logits));
            }
        }
    }
    } // close else (DFlash draft path)

    for i in 1..b {
        block[i] = drafted[i];
    }

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
    let verify_out = verify_dflash_block(
        gpu, target, &block, position, hidden_rb,
        gdn_tape_opt.as_deref_mut(),
        use_temp_sampling,  // need full target logits for rejection sampling
    )?;

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
        for i in 0..b - 1 {
            if verify_out.argmax_per_pos[i] == block[i + 1] {
                accept_len += 1;
            } else {
                break;
            }
        }
        bonus_token = verify_out.argmax_per_pos[accept_len];
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
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256)",
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
    let tree = crate::ddtree::build_ddtree_tree(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        tree_budget,
    );

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

    // ── 2. Per-position top-K + log-normalize ────────────────────────────
    let (top_tokens, top_log_probs) =
        crate::ddtree::topk_from_logits(&draft_logits, b - 1, vocab, tree_topk);

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    let tree = crate::ddtree::build_ddtree_tree(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        tree_budget,
    );

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

    // ── 4. Linearize the tree into (tokens, positions, mask_host) ────────
    let (verify_tokens, verify_positions, mask_host) =
        crate::ddtree::linearize_tree(&tree, seed_token, position as u32);
    let big_n = verify_tokens.len();
    debug_assert_eq!(big_n, 1 + tree.num_nodes());

    // ── 5. Upload mask to GPU as [N × N] f32 ──────────────────────────────
    // The bias buffer is read by the FA attention kernels inside step 7's
    // forward. `memcpy_htod` is synchronous on the default stream so the
    // data is in VRAM before forward kernels queue — no device sync needed.
    let attn_bias_gpu = gpu.alloc_tensor(&[big_n * big_n], rdna_compute::DType::F32)?;
    {
        let mask_bytes = unsafe {
            std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_host.len() * 4)
        };
        if let Err(e) = gpu.hip.memcpy_htod(&attn_bias_gpu.buf, mask_bytes) {
            let _ = gpu.free_tensor(attn_bias_gpu);
            return Err(e);
        }
    }

    // ── 6. Snapshot pre-seed target state ─────────────────────────────────
    if let Err(e) = target_snap.save_from(&target.dn_state, gpu) {
        let _ = gpu.free_tensor(attn_bias_gpu);
        return Err(e);
    }

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
    let ctx = qwen35::TreeVerifyCtx {
        positions: &verify_positions,
        attn_bias: &attn_bias_gpu,
    };
    let verify_out = verify_dflash_block_tree(
        gpu, target, &verify_tokens, position, hidden_rb, Some(gdn_tape), false, ctx,
    );
    let posterior = match verify_out {
        Ok(v) => v.argmax_per_pos,
        Err(e) => {
            let _ = gpu.free_tensor(attn_bias_gpu);
            return Err(e);
        }
    };
    let _ = gpu.free_tensor(attn_bias_gpu);

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
    let fast_tape_ok = accepted_node_indices.iter().enumerate()
        .all(|(i, &ni)| ni == i);
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
    } else {
        // Slow path: committed path diverges from linear order. Re-verify
        // the committed prefix to get a linear-order tape that matches.
        let tape_block: Vec<u32> = committed[..accept_len + 1].to_vec();
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        let _tape_verify = verify_dflash_block(
            gpu, target, &tape_block, position, hidden_rb, Some(gdn_tape), false,
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
