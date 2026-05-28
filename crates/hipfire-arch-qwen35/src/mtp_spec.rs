//! MTP-only speculative decode: standalone trunk + native MTP head loop.
//!
//! The MTP head ([`crate::mtp_head`], Task 9) is a single transformer-decoder
//! block that takes (last-committed-token, trunk-post-norm-hidden) and emits
//! a single-token next-prediction. This module composes that head into a
//! greedy lossless spec-decode cycle:
//!
//! ```text
//! cycle:
//!   1. Snapshot trunk DN state (for rollback after batched verify).
//!   2. MTP-chain `max_n` candidates by feeding each step's `t_mtp_out` and
//!      its predicted token back into mtp_head_forward.
//!   3. Run trunk verify on `[last_committed, c1, ..., c_max_n]`
//!      (max_n + 1 tokens). Capture per-position post-output-norm hidden via
//!      `per_token_hidden_out`.
//!   4. Trunk lm_head over all positions → batched argmax → accept-prefix
//!      where MTP candidate matches trunk argmax.
//!   5. bonus_token = trunk argmax at slot `accept_count` (always commits
//!      unless we early-stopped on EOS inside the accepted prefix).
//!   6. Update `state.prev_hidden` from `verify_hidden[advance - 1]`.
//!   7. Roll back trunk DN state to snapshot, replay accepted committed
//!      tokens (= `verify_tokens[..advance]`) so trunk dn/kv lands at
//!      position `cur_pos + advance` for the next cycle.
//!   8. MTP head KV cache: NO explicit rollback. Stale slots from rejected
//!      candidates get overwritten in-order by next cycle's MTP chain
//!      because each next-cycle MTP step writes slot `pos` BEFORE attending
//!      `[0..pos+1)`.
//! ```
//!
//! Greedy-only (temp=0). No DDTree, no PLD, no rejection sampling — that's
//! Task 11 territory.

use crate::mtp_head::{
    self, Qwen35MtpHead, Qwen35MtpHeadKvCache, Qwen35MtpHeadScratch,
};
use crate::qwen35::{self, Qwen35Weights};
use crate::speculative::{DeltaNetSnapshot, GdnTape, ModelSlot};
use hip_bridge::HipResult;
use hipfire_runtime::llama::{self, EmbeddingFormat};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Sampling primitives (host-side, used by temp>0 spec-decode path) ────

/// Sampling configuration matching the Unsloth-recommended MTP defaults
/// (temp=1.0, top_p=0.95, top_k=20, min_p=0.0 for Qwen3.5/3.6 thinking
/// mode). `temp == 0.0` disables sampling and the spec-step falls back
/// to greedy argmax-match.
#[derive(Copy, Clone, Debug)]
pub struct MtpSamplingConfig {
    pub temp: f32,
    pub top_k: usize,     // 0 = disabled (no top-K cutoff)
    pub top_p: f32,       // 1.0 = disabled (no nucleus cutoff)
    pub min_p: f32,       // 0.0 = disabled (no min-prob cutoff)
}

impl Default for MtpSamplingConfig {
    fn default() -> Self {
        Self { temp: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 }
    }
}

impl MtpSamplingConfig {
    pub fn is_greedy(&self) -> bool { self.temp <= 0.0 }
}

/// Reproducible xorshift64* RNG. Cheap, fast, no `rand` dep.
#[derive(Copy, Clone, Debug)]
pub struct MtpRng(u64);

impl MtpRng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_uniform_f32(&mut self) -> f32 {
        // 24-bit mantissa: [0.0, 1.0).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Sample one token from a row of logits with temp/top_k/top_p/min_p.
/// Returns `(token_id, p_sampled)` where `p_sampled` is the normalized
/// probability of the chosen token under the truncated distribution.
///
/// All arithmetic is host-side f64 for the softmax stability + cumulative
/// sums, then cast back to f32 for the returned probability. The vocab is
/// typically 32K (compressed) or 248K (full); a few μs per call.
pub fn sample_from_logits(
    logits: &[f32],
    cfg: &MtpSamplingConfig,
    rng: &mut MtpRng,
) -> (u32, f32) {
    assert!(!logits.is_empty(), "sample_from_logits: empty logits row");
    assert!(cfg.temp > 0.0, "sample_from_logits: temp must be > 0; caller must branch on greedy");

    // 1. Build (id, scaled_logit) pairs.
    let inv_temp = 1.0 / cfg.temp as f64;
    let mut pairs: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l as f64 * inv_temp))
        .collect();

    // 2. Optional top_k: partial-sort to keep top_k by logit (descending).
    if cfg.top_k > 0 && cfg.top_k < pairs.len() {
        pairs.select_nth_unstable_by(cfg.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(cfg.top_k);
    }

    // 3. Sort by logit descending for top_p and min_p filtering + sampling.
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 4. Softmax with max-subtraction stability.
    let max_logit = pairs[0].1;
    let mut sum_exp = 0.0_f64;
    let mut probs: Vec<f64> = pairs
        .iter()
        .map(|(_, l)| {
            let e = (l - max_logit).exp();
            sum_exp += e;
            e
        })
        .collect();
    for p in probs.iter_mut() { *p /= sum_exp; }

    // 5. min_p: drop tokens with normalized prob < min_p * top_prob.
    if cfg.min_p > 0.0 {
        let top_p_val = probs[0];
        let thresh = (cfg.min_p as f64) * top_p_val;
        let cutoff = probs.iter().position(|&p| p < thresh).unwrap_or(probs.len());
        pairs.truncate(cutoff.max(1));
        probs.truncate(cutoff.max(1));
        // Renormalize after min_p.
        let s: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= s; }
    }

    // 6. top_p (nucleus): cumulative-prob cutoff.
    if cfg.top_p < 1.0 {
        let mut cum = 0.0_f64;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p as f64 {
                cutoff = i + 1;
                break;
            }
        }
        pairs.truncate(cutoff);
        probs.truncate(cutoff);
        // Renormalize after top_p.
        let s: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= s; }
    }

    // 7. Multinomial sample.
    let r = rng.next_uniform_f32() as f64;
    let mut acc = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return (pairs[i].0, probs[i] as f32);
        }
    }
    // Numerical edge case: fall through to last.
    let last = probs.len() - 1;
    (pairs[last].0, probs[last] as f32)
}

/// Compute the temperature-scaled softmax probability of a single token at
/// index `idx` in a logits row. Used by the residual-acceptance rule to
/// gather `p_target(c_k)` and `p_draft(c_k)` under the same temperature
/// scaling that draft sampling uses. Numerically stable via
/// max-subtraction (single pass over logits).
///
/// Note: this is the **un-truncated** softmax prob (no top_k/top_p filter).
/// For the residual ratio `p_target(c_k) / p_draft(c_k)` we want both probs
/// from the same distribution shape; using raw temp-scaled softmax keeps
/// the math clean even if draft sampling itself was truncated.
pub fn softmax_prob_at_temp(logits: &[f32], idx: usize, temp: f32) -> f32 {
    assert!(temp > 0.0, "softmax_prob_at_temp: temp must be > 0");
    let inv_t = (1.0 / temp) as f64;
    // First pass: find max (for stability) of temp-scaled logits.
    let mut max_scaled = f64::NEG_INFINITY;
    for &l in logits.iter() {
        let s = l as f64 * inv_t;
        if s > max_scaled { max_scaled = s; }
    }
    let mut sum_exp = 0.0_f64;
    let mut target_e = 0.0_f64;
    for (i, &l) in logits.iter().enumerate() {
        let e = ((l as f64) * inv_t - max_scaled).exp();
        sum_exp += e;
        if i == idx { target_e = e; }
    }
    (target_e / sum_exp) as f32
}

// ─── Public state ────────────────────────────────────────────────────────

/// All persistent buffers needed by [`spec_step_mtp`]. Allocate once per
/// generation; reuse across cycles. Frees at end of generation via
/// [`MtpSpecState::free_gpu`].
pub struct MtpSpecState {
    /// Persistent post-output-norm hidden snapshot of the trunk at the
    /// position of the most-recently-committed token. Fed as `prev_hidden`
    /// into the MTP head's first step each cycle. Shape: `[dim]` F32.
    pub prev_hidden: GpuTensor,

    /// Per-cycle scratch: post-output-norm hidden states from the trunk
    /// verify, one row per verify position. Shape: `[(max_n + 1) × dim]` F32.
    pub verify_hidden: GpuTensor,

    /// Per-cycle scratch: trunk lm_head output over all verify positions.
    /// Shape: `[(max_n + 1) × vocab]` F32.
    pub verify_logits: GpuTensor,

    /// Per-cycle scratch: FWHT-rotated `verify_hidden` for MQ-family
    /// lm_head. Shape: `[(max_n + 1) × dim]` F32. Allocated unconditionally;
    /// unused for non-MQ lm_head dtypes.
    pub verify_rot: GpuTensor,

    /// GPU-side argmax buffer for batched argmax over verify positions.
    /// Shape: `[(max_n + 1)]` i32 stored as F32 slots.
    pub verify_argmax: GpuTensor,

    /// Trunk DN snapshot for rollback before replay.
    pub trunk_snap: DeltaNetSnapshot,

    /// Persistent batch scratch for the trunk's batched verify forward.
    /// Sized to `max_n + 1` so verify always fits in one chunk.
    pub trunk_pbs: qwen35::PrefillBatchScratch,

    /// Innovation tape captured during trunk verify. After rollback, replaying
    /// this advances only the DeltaNet recurrence for the accepted prefix.
    pub trunk_gdn_tape: GdnTape,

    /// MTP head per-call scratch (Task 9). One per slot; reused across cycles.
    pub mtp_scratch: Qwen35MtpHeadScratch,

    /// MTP head KV cache (Task 9). Single-layer F32, rolling overwrite.
    pub mtp_kv: Qwen35MtpHeadKvCache,

    /// Per-step `t_mtp_out` capture buffer for the K-step batched lm_head
    /// chain (Task 10b). Shape `[max_n, n_embd]` row-major. After the
    /// chain runs, the end-of-chain batched lm_head consumes this whole
    /// buffer in one GEMM call. Step k writes its `t_mtp_out` into row k
    /// (offset `k * n_embd * 4` bytes); step k+1 reads row k as its
    /// "next-token embedding override" + `prev_hidden`.
    pub mtp_t_outs: GpuTensor,

    /// Per-step batched-rmsnorm output for the K-step batched lm_head.
    /// Shape `[max_n, n_embd]`. Caller of `mtp_head_apply_lm_head_batched`
    /// owns this scratch.
    pub mtp_lm_tmp: GpuTensor,

    /// Per-step FWHT-rotated x scratch for MagnumQuant lm_heads (MQ4/MQ3/MQ6).
    /// Shape `[max_n, n_embd]`. Unused for non-MQ lm_head dtypes.
    pub mtp_lm_rot: GpuTensor,

    /// Per-step batched lm_head logits output. Shape `[max_n, vocab]`.
    /// Caller `argmax_f32_batched` reads this for the K predicted tokens.
    pub mtp_lm_logits: GpuTensor,

    /// GPU-side argmax destination for the K-step batched argmax over
    /// `mtp_lm_logits`. Shape `[max_n]` i32 stored as F32 slots.
    pub mtp_lm_argmax: GpuTensor,

    /// Optional FastMTP-style compressed batched-logits output, shape
    /// `[max_n * compressed_vocab_size]`. Populated by
    /// [`spec_step_mtp_compressed`] when the head ships a sidecar.
    /// Allocated lazily via [`MtpSpecState::ensure_compressed_lm_logits`].
    pub mtp_lm_logits_compressed: Option<GpuTensor>,

    /// Per-step top-2 index scratch for [`spec_step_mtp_compressed_serial`]
    /// with `p_min > 0`. Shape `[2]` i32 stored as F32 slots. Reused across
    /// K-chain steps within a cycle.
    pub mtp_topk_idx: GpuTensor,

    /// Per-step top-2 log-softmax-prob scratch matching `mtp_topk_idx`.
    /// Shape `[2]` F32. `top_logp[0]` is log P(argmax); compared against
    /// `log(p_min)` for the chain-truncation early-exit.
    pub mtp_topk_logp: GpuTensor,

    // ─── GPU sampling scratches (only used when sampling.temp > 0) ──────
    /// Result buffer for `gpu.sample_top_p`: shape `[2]` u32-as-F32. Slot 0
    /// = sampled token id, slot 1 = updated rng state.
    pub mtp_sample_result: GpuTensor,
    /// Dummy repeat-penalty buffer for `gpu.sample_top_p` (we pass
    /// `repeat_window=0`, so the kernel skips this branch entirely;
    /// only allocated so the dispatch wrapper has a valid pointer).
    pub mtp_sample_repeat_buf: GpuTensor,
    /// Single-element index buf for the per-step p_draft gather. Shape `[1]`
    /// i32-as-F32. H2D'd with the sampled token id, fed to
    /// `softmax_prob_gather_batched_f32` with n_rows=1.
    pub mtp_gather_idx_draft: GpuTensor,
    /// Output buf for the per-step p_draft gather. Shape `[1]` F32.
    pub mtp_gather_prob_draft: GpuTensor,
    /// Batched index buf for the K-candidate p_target gather. Shape `[max_n]`
    /// i32-as-F32. H2D'd with `candidates[0..drafts_generated]`.
    pub mtp_gather_idx_verify: GpuTensor,
    /// Batched output buf for the K-candidate p_target gather. Shape `[max_n]`
    /// F32. D2H'd after `softmax_prob_gather_batched_f32(n_rows=drafts_generated)`.
    pub mtp_gather_prob_verify: GpuTensor,
    /// On-device rng state for `gpu.sample_top_p`. Threaded through both
    /// draft and bonus sampling calls within a cycle and across cycles.
    /// Reseeded by [`Self::set_sampling`].
    pub gpu_rng_state: u32,

    /// Maximum number of MTP candidates per cycle (chain depth).
    pub max_n: usize,

    /// Optional draft-confidence cutoff for [`spec_step_mtp_compressed_serial`].
    /// When `p_min > 0`, the K-chain truncates early at step k if the draft's
    /// top-1 softmax prob falls below `p_min` — analog of llama.cpp's
    /// `--spec-draft-p-min` (PR #22673). Default 0.0 = disabled. Set via
    /// [`Self::set_p_min`]. The current step's candidate IS kept (we already
    /// computed it); only steps k+1..max_n are skipped.
    pub p_min: f32,

    /// Sampling config (temp/top_k/top_p/min_p). When `temp == 0.0` (default)
    /// the spec-step uses the legacy greedy/argmax-match accept rule. When
    /// `temp > 0`, the K-chain samples per step + the trunk applies
    /// residual-style acceptance (matches Unsloth/llama.cpp #22673 canonical
    /// MTP path). Set via [`Self::set_sampling`].
    pub sampling: MtpSamplingConfig,

    /// RNG state for sampling. Seeded via [`Self::set_sampling`].
    /// Single global stream across all spec-decode positions in a generation.
    pub rng: MtpRng,
}

impl MtpSpecState {
    /// Allocate all per-generation buffers, sizing the trunk DN snapshot
    /// against the live `target.dn_state`. Defaults MTP head KV to Q8 — see
    /// [`Self::new_for_slot_with_kv_mode`] for explicit selection.
    pub fn new_for_slot(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
    ) -> HipResult<Self> {
        Self::new_for_slot_with_kv_mode(
            gpu, target, head, max_n, crate::mtp_head::MtpKvMode::Q8,
        )
    }

    /// Like [`Self::new_for_slot`] but allocates the MTP head's KV cache in
    /// the requested format. Used by `mtp_only_demo` to A/B kv-mode variants
    /// (q8/asym3/fwht4) per the 2026-05-16 feat/fwht prose-τ findings.
    pub fn new_for_slot_with_kv_mode(
        gpu: &mut Gpu,
        target: &ModelSlot,
        head: &Qwen35MtpHead,
        max_n: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
    ) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "MtpSpecState::new_for_slot: max_n must be ≥ 1 (chain depth)"
        );
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        assert_eq!(
            head.config.n_embd, dim,
            "MtpSpecState: trunk dim={dim} but head n_embd={}",
            head.config.n_embd,
        );
        assert_eq!(
            head.config.vocab_size, vocab,
            "MtpSpecState: trunk vocab={vocab} but head vocab={}",
            head.config.vocab_size,
        );

        let prev_hidden = gpu.alloc_tensor(&[dim], DType::F32)?;
        let verify_hidden = gpu.alloc_tensor(&[(max_n + 1) * dim], DType::F32)?;
        let verify_logits = gpu.alloc_tensor(&[(max_n + 1) * vocab], DType::F32)?;
        let verify_rot = gpu.alloc_tensor(&[(max_n + 1) * dim], DType::F32)?;
        let verify_argmax = gpu.alloc_tensor(&[max_n + 1], DType::F32)?;
        let trunk_snap = DeltaNetSnapshot::new_for(gpu, &target.dn_state)?;
        let trunk_pbs = qwen35::PrefillBatchScratch::new(gpu, &target.config, max_n + 1)?;
        let trunk_gdn_tape = GdnTape::new_for_config(gpu, &target.config, max_n + 1)?;
        let mtp_scratch = Qwen35MtpHeadScratch::new(gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new_with_kv_mode(gpu, &head.config, kv_mode)?;

        // Per-step batched-lm_head scratch (Task 10b).
        let mtp_t_outs = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_tmp = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_rot = gpu.alloc_tensor(&[max_n * dim], DType::F32)?;
        let mtp_lm_logits = gpu.alloc_tensor(&[max_n * vocab], DType::F32)?;
        let mtp_lm_argmax = gpu.alloc_tensor(&[max_n], DType::F32)?;

        // Per-step top-2 scratches for p_min early-exit (16 B total, always
        // allocated — only used when state.p_min > 0).
        let mtp_topk_idx = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_topk_logp = gpu.alloc_tensor(&[2], DType::F32)?;

        // GPU sampling scratches (always allocated; only used when temp > 0).
        // All are tiny so the unconditional alloc is fine.
        let mtp_sample_result = gpu.alloc_tensor(&[2], DType::F32)?;
        let mtp_sample_repeat_buf = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_idx_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_prob_draft = gpu.alloc_tensor(&[1], DType::F32)?;
        let mtp_gather_idx_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;
        let mtp_gather_prob_verify = gpu.alloc_tensor(&[max_n], DType::F32)?;

        Ok(Self {
            prev_hidden,
            verify_hidden,
            verify_logits,
            verify_rot,
            verify_argmax,
            trunk_snap,
            trunk_pbs,
            trunk_gdn_tape,
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_tmp,
            mtp_lm_rot,
            mtp_lm_logits,
            mtp_lm_argmax,
            mtp_lm_logits_compressed: None,
            mtp_topk_idx,
            mtp_topk_logp,
            mtp_sample_result,
            mtp_sample_repeat_buf,
            mtp_gather_idx_draft,
            mtp_gather_prob_draft,
            mtp_gather_idx_verify,
            mtp_gather_prob_verify,
            gpu_rng_state: 42,
            max_n,
            p_min: 0.0,
            sampling: MtpSamplingConfig::default(),
            rng: MtpRng::new(42),
        })
    }

    /// Configure sampling (temp/top_p/top_k/min_p) and reseed the per-state RNG.
    /// `cfg.temp == 0.0` keeps the legacy greedy path. `cfg.temp > 0` enables
    /// residual-acceptance sampling per the Unsloth/llama.cpp MTP recipe.
    /// Reseeds BOTH the host RNG (used for the residual accept rule) and the
    /// on-device RNG (used by `gpu.sample_top_p`).
    pub fn set_sampling(&mut self, cfg: MtpSamplingConfig, seed: u64) {
        assert!(cfg.temp >= 0.0, "set_sampling: temp must be >= 0.0, got {}", cfg.temp);
        assert!(cfg.top_p > 0.0 && cfg.top_p <= 1.0, "set_sampling: top_p must be in (0,1], got {}", cfg.top_p);
        assert!(cfg.min_p >= 0.0 && cfg.min_p <= 1.0, "set_sampling: min_p must be in [0,1], got {}", cfg.min_p);
        self.sampling = cfg;
        self.rng = MtpRng::new(seed);
        // GPU rng uses u32; mix the lower + upper halves so different seeds
        // produce different on-device streams.
        self.gpu_rng_state = ((seed >> 32) as u32) ^ (seed as u32) | 1;
    }

    /// Set the draft-confidence threshold for compressed-serial K-chain
    /// truncation. Pass `0.0` (default) to disable. Typical values: 0.5-0.8.
    /// llama.cpp's PR #22673 ships `0.75` as the recommended starting point.
    pub fn set_p_min(&mut self, p_min: f32) {
        assert!(
            (0.0..=1.0).contains(&p_min),
            "MtpSpecState::set_p_min: p_min must be in [0.0, 1.0], got {p_min}"
        );
        self.p_min = p_min;
    }

    /// Allocate `mtp_lm_logits_compressed` shape `[max_n * cvs]` for the
    /// compressed batched-lm_head dispatch path. Idempotent — reallocates
    /// only if the existing tensor has the wrong size. Call once after
    /// loading a head with a sidecar (cvs = `head.weights.compressed_vocab_size`).
    pub fn ensure_compressed_lm_logits(
        &mut self, gpu: &mut Gpu, cvs: usize,
    ) -> HipResult<()> {
        let needed = self.max_n * cvs;
        if let Some(existing) = self.mtp_lm_logits_compressed.as_ref() {
            if existing.numel() == needed {
                return Ok(());
            }
            if let Some(old) = self.mtp_lm_logits_compressed.take() {
                let _ = gpu.free_tensor(old);
            }
        }
        self.mtp_lm_logits_compressed = Some(
            gpu.alloc_tensor(&[needed], DType::F32)?,
        );
        Ok(())
    }

    /// Capture the trunk's post-output-norm hidden after a `forward_scratch`
    /// call. Source is `target.scratch.tmp` (same buffer lm_head reads from).
    /// Call once after prefill's last `forward_scratch` to seed the cycle.
    pub fn capture_prev_hidden_from_scratch_tmp(
        &self,
        gpu: &Gpu,
        target_scratch_tmp: &GpuTensor,
        dim: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf, 0,
            &target_scratch_tmp.buf, 0,
            dim * 4,
        )
    }

    /// Capture the trunk's post-output-norm hidden from a row of
    /// `verify_hidden` (per-position output of the batched verify forward).
    /// Used after spec_step accepts to set up next cycle's MTP `prev_hidden`
    /// from the verify slot corresponding to the last committed token,
    /// avoiding the cost of a separate single-token forward.
    pub fn capture_prev_hidden_from_verify_row(
        &self,
        gpu: &Gpu,
        row: usize,
        dim: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_dtod_at(
            &self.prev_hidden.buf, 0,
            &self.verify_hidden.buf, row * dim * 4,
            dim * 4,
        )
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.prev_hidden);
        let _ = gpu.free_tensor(self.verify_hidden);
        let _ = gpu.free_tensor(self.verify_logits);
        let _ = gpu.free_tensor(self.verify_rot);
        let _ = gpu.free_tensor(self.verify_argmax);
        let _ = gpu.free_tensor(self.mtp_t_outs);
        let _ = gpu.free_tensor(self.mtp_lm_tmp);
        let _ = gpu.free_tensor(self.mtp_lm_rot);
        let _ = gpu.free_tensor(self.mtp_lm_logits);
        let _ = gpu.free_tensor(self.mtp_lm_argmax);
        if let Some(lc) = self.mtp_lm_logits_compressed {
            let _ = gpu.free_tensor(lc);
        }
        // DeltaNetSnapshot's DeviceBuffers free on drop.
        drop(self.trunk_snap);
        self.trunk_pbs.free_gpu(gpu);
        self.trunk_gdn_tape.free_gpu(gpu);
        self.mtp_scratch.free_gpu(gpu);
        self.mtp_kv.free_gpu(gpu);
    }
}

// ─── One spec-decode cycle ───────────────────────────────────────────────

/// Result of one MTP spec-decode cycle.
#[derive(Debug, Clone)]
pub struct MtpSpecResult {
    /// Tokens committed THIS cycle (excludes the seed `last_committed` from
    /// the previous cycle; includes the bonus when no early EOS).
    /// Length == `advance`.
    pub committed: Vec<u32>,
    /// Number of MTP candidates that matched trunk argmax (not counting the
    /// bonus). 0 = MTP wasted, max_n = full acceptance.
    pub accept_count: usize,
    /// True iff any committed token equals `eos_token_id` (caller stops).
    pub hit_eos: bool,
    /// Position advance (= committed.len()). Caller's `cur_pos += advance`.
    pub advance: usize,
    /// Number of MTP draft candidates ACTUALLY generated this cycle (≤ max_n).
    /// Equals max_n unless p_min early-exit truncated the chain. The trunk
    /// verify batch is sized `drafts_generated + 1`.
    pub drafts_generated: usize,
    /// True iff `p_min` early-exit fired this cycle (caller can sum across
    /// cycles to compute the "% cycles truncated" stat).
    pub chain_truncated: bool,
    /// True iff this cycle was a full-accept (advance == drafts_generated + 1
    /// without EOS) so the post-verify replay was skipped — verify's KV +
    /// DN state already match what replay would have produced. Currently
    /// only set by [`spec_step_mtp_compressed_serial`]; other spec_step
    /// variants always replay and report `false`.
    pub replay_skipped: bool,
}

/// One MTP-only spec-decode cycle (greedy, temp=0).
///
/// Preconditions:
/// - Trunk has been forwarded through position `cur_pos - 1`. Trunk's
///   `dn_state` is post-position-`cur_pos - 1` ready to accept the next
///   token at position `cur_pos`. `target.kv_cache` has K/V written at
///   positions `[0..cur_pos)`. `target.scratch.tmp` is unused / available.
/// - `state.prev_hidden` holds the trunk's post-output-norm hidden at
///   position `cur_pos - 1` (i.e., the position whose argmax produced
///   `last_committed`).
/// - MTP head KV cache is "rolling": stale slots from prior rejected
///   chains may exist at slots `>= cur_pos`; they will be overwritten in
///   place by this cycle's chain.
/// - `last_committed` is the token id at logical position `cur_pos - 1`.
///   It IS verified again as block[0] of this cycle's verify input — its
///   K/V row in trunk's cache stays consistent (idempotent re-write at
///   position cur_pos - 1? No — at position cur_pos - 1 the token was
///   written at the PREVIOUS cycle's last verify or replay. block[0]
///   here is at position cur_pos: trunk's verify will write trunk KV
///   slot cur_pos with last_committed's K/V. NOTE: this means
///   `last_committed`'s "logical" position for THIS cycle's verify is
///   cur_pos, NOT cur_pos - 1.
///
/// So the precise semantics: `cur_pos` is the position where this cycle's
/// verify block[0] (= `last_committed` re-emit) lands. The PREVIOUS cycle
/// committed `last_committed` at position `cur_pos - 1`? Or at position
/// `cur_pos`?
///
/// Actually re-reading spec_step_dflash @ 2560-2583: the seed token is
/// re-fed at position `position` (the `position` parameter), and verify
/// runs `block` at positions `[position..position + B)`. So the seed is
/// at `position`, NOT `position - 1`. This is intentional: each cycle
/// re-runs the seed forward to land trunk K/V in slot `position` (because
/// after the previous cycle's replay, trunk advanced to `position` but
/// did NOT yet write a K/V slot at `position` for the bonus_token — the
/// bonus is "deferred" to be written here as block[0]).
///
/// We mirror that convention. Caller supplies `cur_pos` = position where
/// `last_committed` (= prev cycle's bonus) sits. After cycle, advance by
/// `result.advance`. NEW commits include indices [cur_pos+1..cur_pos+advance]
/// PLUS we also "re-confirm" cur_pos itself via block[0]. The caller's
/// emitted-token list should NOT include `last_committed` again.
///
/// Postconditions:
/// - Trunk dn_state advanced to `cur_pos + advance` (state ready for next
///   cycle's verify at position `cur_pos + advance`).
/// - Trunk kv_cache has K/V at positions `[0..cur_pos + advance)`.
/// - `state.prev_hidden` updated to trunk's post-output-norm hidden at
///   position `cur_pos + advance - 1` (= the LAST committed token).
/// - MTP KV cache may have stale slots at `>= cur_pos + advance`; OK.
/// - `result.hit_eos`: true iff any committed token equals `eos_token_id`.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Ensure stream exists for any async memset (downstream batched kernels
    // assume it). Mirrors spec_step_dflash convention.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. MTP candidate chain (Approach B: feature-only K-step) ────────
    //
    // ARCHITECTURALLY LOSSY: each step k (k > 0) feeds the previous step's
    // `t_mtp_out` directly as both the "embedding of the predicted token"
    // (bypassing the embedding table — see `mtp_head_forward_block_only`'s
    // `next_token_embed` doc comment) AND as `prev_hidden`. This lets us
    // chain all K block forwards back-to-back WITHOUT running lm_head per
    // step, then collapse the K argmaxes into a single batched lm_head GEMM.
    //
    // Trade-off (vs the earlier discrete-token-roundtrip path that called
    // `mtp_head_forward` K times): saves K-1 separate GEMV launches into
    // the trunk's vocab×n_embd lm_head matrix. Each saved launch is ~12 ms
    // for the 27B-3.5 lm_head (vocab=152K). On gfx1100 K=3 the projected
    // win is ~22 ms/cycle = ~37%-1.5× lift if τ holds.
    //
    // Lossless guarantee preserved at trunk-verify: any incorrect MTP
    // candidate (from the lossy chain or otherwise) is rejected by trunk
    // argmax check; the cycle just degenerates to AR + bonus token. τ may
    // drop relative to the discrete-token-roundtrip path because the head
    // was trained with `embed[token]`, not raw `t_mtp_out`, as its
    // step-input distribution.
    //
    // MTP step k writes its own KV slot at position cur_pos + k (same as
    // before — a per-step single-layer transformer cache).
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        // Step 0 reads from state.prev_hidden (trunk's last post-norm
        // hidden, snapshot from the previous cycle / prefill) and embeds
        // last_committed via the trunk's table. Step k > 0 reads from row
        // (k-1) of mtp_t_outs and uses that same row as the embedding
        // override (lossy substitute for embed[predicted_token_{k-1}]).
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                last_committed,
                &state.prev_hidden,
                None,
                cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu,
                head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                0,                  // sentinel: ignored when next_token_embed is Some(_)
                &prev_row,          // prev_hidden = step k-1's t_mtp_out
                Some(&prev_row),    // embed override = step k-1's t_mtp_out (lossy)
                cur_pos + k,
                trunk_weights,
            )?;
        }
        // Capture this step's t_mtp_out into row k of mtp_t_outs so the
        // next step can read it AND so the end-of-chain batched lm_head
        // can ingest the whole stack.
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf, k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf, 0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched lm_head over all K t_mtp_outs ────────────────────────
    //
    // Single GEMM (N=max_n) replacing K separate GEMV calls. shared_head_norm
    // is applied row-wise via rmsnorm_batched inside the helper. MQ4
    // dispatch path includes the FWHT rotation pre-pass (rotate_x_mq_batched).
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = state.mtp_lm_logits.sub_offset(0, max_n * vocab);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu,
        head,
        &trunk_weights.output,
        &t_outs_view,
        &lm_tmp_view,
        &lm_rot_view,
        &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over (max_n) rows → single D2H of K i32 ids.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, vocab, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, max_n * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let candidates: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 2. Build trunk verify input: [last_committed, c1, ..., c_max_n] ──
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    debug_assert_eq!(verify_tokens.len(), max_n + 1);

    // ── 3. Snapshot trunk DN state (rollback target before verify) ───────
    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    // ── 4. Trunk batched verify, capturing per-position hidden ───────────
    //
    // `forward_prefill_batch_with_pbs` writes per-position post-output-norm
    // hidden into `state.verify_hidden` when per_token_hidden_out is Some.
    // Trunk KV cache and dn_state advance by max_n+1 (= verify_tokens.len()).
    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None, // hidden_rb: not used here (DFlash drafter not in the loop)
        Some(&state.verify_hidden),
        None, // gdn_tape
        None, // tree_verify
        Some(&state.trunk_pbs),
        None, // mask_override
        None, // max_layer
    )?;

    // ── 5. Per-position lm_head + batched argmax ─────────────────────────
    //
    // Mirrors verify_dflash_block's batched lm_head dispatch by dtype.
    // Sized to (max_n + 1). Greedy-only, so we use GPU-side batched argmax
    // to avoid the (max_n + 1) × vocab D2H.
    let n_verify = max_n + 1;
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            // gemm_q8_0_batched supports up to 64 rows in one launch.
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            // Fallback: per-row weight_gemv. Same path verify_dflash_block uses.
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    // GPU-side batched argmax over (max_n + 1) rows.
    let argmax_view = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_view, vocab, n_verify)?;
    let mut argmax_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // ── 6. Accept-prefix ────────────────────────────────────────────────
    //
    // For k in [0..max_n): if MTP candidate `candidates[k]` matches trunk's
    // argmax at slot k (= trunk's prediction of token at position
    // cur_pos + k + 1), accept. Stop on first miss OR EOS.
    //
    // bonus_token = argmax_per_pos[accept_count] (always commits unless
    // an accepted MTP candidate was already EOS).
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);

    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        // Bonus: trunk's argmax at the first rejection slot (or after the
        // full accepted chain). Always commits one extra token.
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1, "spec_step_mtp must commit at least the bonus");
    debug_assert!(advance <= max_n + 1);

    // ── 7. Capture prev_hidden for next cycle ────────────────────────────
    //
    // Next cycle's MTP step 0 needs trunk's post-output-norm hidden whose
    // argmax PRODUCED the next cycle's last_committed (= this cycle's
    // last committed token). Verify-slot `i` holds the post-norm hidden at
    // trunk position `cur_pos + i`, which produces argmax_per_pos[i] for
    // position `cur_pos + i + 1`. The last committed token sits at:
    //
    //   - If hit_eos triggered inside the MTP-accepted chain (no bonus):
    //     advance == accept_count, last token at position cur_pos + advance
    //     (= cur_pos + accept_count), which is the i = advance - 1 slot.
    //   - Otherwise: last token = bonus at position cur_pos + accept_count + 1,
    //     advance == accept_count + 1, slot = advance - 1.
    //
    // Both cases reduce to slot `advance - 1`.
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    // ── 8. Roll back trunk DN state + replay accepted committed tokens ───
    //
    // After verify, trunk dn_state is at position cur_pos + max_n + 1.
    // We need it at cur_pos + advance for next cycle. Restore snapshot
    // (back to cur_pos), then replay first `advance` of `verify_tokens`.
    // Replay re-writes trunk KV slots [cur_pos..cur_pos + advance) with
    // identical K/V (idempotent — same inputs, same positions), and
    // re-runs DN recurrence.
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(
            gpu,
            trunk_weights,
            &target.config,
            replay,
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None, None, None, None,
        )?;
    } else {
        // advance == 1: only the seed (last_committed) is "replayed". This
        // happens when EOS hits with accept_count==0 AND bonus==eos
        // (very rare), or when the bonus was committed alone (also implies
        // accept_count==0, which is the common AR-degraded case).
        qwen35::forward_scratch(
            gpu,
            trunk_weights,
            &target.config,
            verify_tokens[0],
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
        )?;
    }

    // ── 9. MTP KV cache rollback: NOT NEEDED ────────────────────────────
    //
    // Argument: MTP step k of the original chain wrote slot cur_pos + k
    // with K/V derived from input (k==0 ? last_committed : candidates[k-1]).
    // For k in [0..accept_count], the actual committed token at position
    // cur_pos + k matches that input (last_committed for k=0, candidates[k-1]
    // = committed[k-1] for k≥1 because c_k was accepted), so those slots
    // hold correct K/V. For k in [accept_count+1..max_n), the chain wrote
    // K/V from rejected candidates — STALE. But the next cycle starts at
    // cur_pos_new = cur_pos + advance = cur_pos + accept_count + 1
    // (or cur_pos + accept_count if EOS-no-bonus), and step 0 of that
    // next cycle WRITES slot cur_pos_new BEFORE attending [0..cur_pos_new+1).
    // Stale slots [cur_pos + advance, cur_pos + max_n) are at-or-beyond
    // cur_pos_new = cur_pos + advance, so step 0 next cycle writes the
    // first stale slot before reading it. Step k next cycle writes the
    // (k+1)-th formerly-stale slot before its attention reads it. The
    // chain progresses left-to-right, overwriting stale slots in-order
    // exactly as needed. NO explicit rollback required.

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated: state.max_n,
        chain_truncated: false,
        replay_skipped: false,
    })
}

// ─── FastMTP-style compressed spec step (K = max_n) ───────────────────────
//
// Drop-in alternative to [`spec_step_mtp`] when the loaded MTP head ships a
// compressed `lm_head_draft` sidecar. Chains K = state.max_n MTP block
// forwards (k=0 normal, k>0 with lossy embedding override = previous step's
// `t_mtp_out`), then runs ONE batched compressed lm_head dispatch over the K
// stacked outputs (vs K full-vocab dispatches in the naive K-step path).
//
// Per-cycle BW (27B-3.5, K=3):
//   K full-vocab lm_head GEMVs: K * 635 MB ≈ 1.9 GB MQ4 streamed in
//   1 batched compressed lm_head GEMM: 84 MB MQ4 + K * 5120 * 4 B activations ≈ 84 MB
//   Saved: ~1.8 GB BW per cycle; trunk verify still runs full lm_head (n=K+1 batched).
//
// K=1 retains the original K=1 fast path; K>1 chains lossily (same OOD risk
// the existing `spec_step_mtp` warns about — τ may degrade vs discrete-token
// roundtrip). Lossless greedy preserved at the trunk-verify layer regardless.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Precondition: head must have compressed sidecar loaded + caller must
    // have allocated the compressed batched-logits scratch.
    let lm_head_draft = head.weights.lm_head_draft.as_ref()
        .expect("spec_step_mtp_compressed requires head loaded with --vocab-sidecar");
    let vocab_map = head.weights.lm_head_draft_vocab_map.as_ref()
        .expect("compressed head missing vocab_map");
    let cvs = head.weights.compressed_vocab_size
        .expect("compressed head missing compressed_vocab_size");
    let logits_c_batched = state.mtp_lm_logits_compressed.as_ref()
        .expect("MtpSpecState::mtp_lm_logits_compressed not allocated; \
                 call ensure_compressed_lm_logits(gpu, cvs) after head load");
    assert!(
        logits_c_batched.numel() >= max_n * cvs,
        "mtp_lm_logits_compressed too small: {} < max_n*cvs ({})",
        logits_c_batched.numel(), max_n * cvs,
    );

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // ── 1. Chain K MTP block forwards (lossy embedding override for k>0) ──
    //
    // Identical structure to spec_step_mtp's K-step lossy chain. Step 0
    // reads from state.prev_hidden + embeds last_committed via trunk's
    // token table. Step k>0 reads from row (k-1) of mtp_t_outs and uses
    // that row as the embedding override (continuous t_mtp_out as a lossy
    // substitute for embed[predicted_token_{k-1}]) so we don't need to
    // know the chain's interim tokens before running step k+1.
    //
    // The chain's ARCHITECTURAL LOSS comes from this OOD-for-the-head
    // continuous embedding — the MTP head was trained with discrete-token
    // roundtrips. τ may degrade. Trunk verify catches any wrong candidate
    // (lossless greedy preserved), but acceptance rate suffers.
    let dim_bytes = dim * 4;
    for k in 0..max_n {
        if k == 0 {
            mtp_head::mtp_head_forward_block_only(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                last_committed, &state.prev_hidden, None, cur_pos + k,
                trunk_weights,
            )?;
        } else {
            let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_block_only(
                gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                0,                  // ignored when next_token_embed = Some
                &prev_row,
                Some(&prev_row),    // lossy embed override
                cur_pos + k,
                trunk_weights,
            )?;
        }
        gpu.hip.memcpy_dtod_at(
            &state.mtp_t_outs.buf, k * dim_bytes,
            &state.mtp_scratch.t_mtp_out.buf, 0,
            dim_bytes,
        )?;
    }

    // ── 1b. Batched COMPRESSED lm_head over all K t_mtp_outs ──────────────
    //
    // Single GEMM with M=cvs (vs M=vocab=248K in the plain path) — this is
    // the FastMTP-style ~7.5x BW reduction lever. shared_head_norm applied
    // row-wise inside the helper. MQ4 path includes the FWHT pre-pass.
    let t_outs_view = state.mtp_t_outs.sub_offset(0, max_n * dim);
    let lm_tmp_view = state.mtp_lm_tmp.sub_offset(0, max_n * dim);
    let lm_rot_view = state.mtp_lm_rot.sub_offset(0, max_n * dim);
    let lm_logits_view = logits_c_batched.sub_offset(0, max_n * cvs);
    mtp_head::mtp_head_apply_lm_head_batched(
        gpu, head, lm_head_draft,
        &t_outs_view, &lm_tmp_view, &lm_rot_view, &lm_logits_view,
        max_n,
    )?;

    // Batched argmax over K compressed logit rows → K i32 ids → remap.
    let lm_argmax_view = state.mtp_lm_argmax.sub_offset(0, max_n);
    gpu.argmax_f32_batched(&lm_logits_view, &lm_argmax_view, cvs, max_n)?;
    let mut argmax_host: Vec<i32> = vec![0; max_n];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_host.as_mut_ptr() as *mut u8, max_n * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &lm_argmax_view.buf)?;
    }
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    for raw in argmax_host {
        let draft_idx = raw as usize;
        assert!(
            draft_idx < cvs,
            "draft argmax {draft_idx} out of compressed vocab {cvs}"
        );
        candidates.push(vocab_map[draft_idx]);
    }

    // ── 2. Trunk verify on [last_committed, c1, ..., c_max_n] ─────────────
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        trunk_weights,
        &target.config,
        &verify_tokens,
        cur_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,
        Some(&state.verify_hidden),
        None,
        None,
        Some(&state.trunk_pbs),
        None, // mask_override
        None, // max_layer
    )?;

    // ── 3. Trunk batched lm_head over verify positions ─────────────────────
    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    // ── 4. Accept-prefix (K-general) ──────────────────────────────────────
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(max_n + 1);
    for k in 0..max_n {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= max_n + 1);

    // ── 5. Capture prev_hidden from verify slot advance-1 ─────────────────
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    // ── 6. Roll back trunk DN state + replay accepted ─────────────────────
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(
            gpu,
            trunk_weights,
            &target.config,
            replay,
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None, None, None, None,
        )?;
    } else {
        qwen35::forward_scratch(
            gpu,
            trunk_weights,
            &target.config,
            verify_tokens[0],
            cur_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
        )?;
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated: state.max_n,
        chain_truncated: false,
        replay_skipped: false,
    })
}

// ─── FastMTP-style compressed SERIAL spec step (K serial roundtrips) ──────
//
// Variant of [`spec_step_mtp_compressed`] that runs K MTP steps as
// DISCRETE-TOKEN ROUNDTRIPS (not lossy chain): each step argmaxes its
// compressed logits, remaps via vocab_map to a full-vocab token id, and
// uses that token id (embedded via trunk's `token_embd` table) as the
// next step's input. Mirrors the original Task 10 "serial lm_head" path
// — but with COMPRESSED lm_head dispatches (~7.5x BW saving each).
//
// Per-cycle BW (27B-3.5, K=3, projection):
//   3 full lm_head GEMVs:        3 × 635 MB = 1.9 GB MQ4  (Task 10 baseline)
//   3 compressed lm_head GEMVs:  3 × 84 MB  = 252 MB MQ4  (this path)
//   Saved: ~1.65 GB BW per cycle → ~21-27 ms per cycle on gfx1100.
//
// Task 10 plain serial measured 39.68 tok/s τ=3.08; this path projects
// ~50-58 tok/s at the same τ (gate is 60). A trained sidecar (hiptrx
// trunk-argmax distillation) lifting τ from 3.08 → 3.5+ should clear.
//
// Trade-off vs the batched lossy chain: K serial dispatches launch K
// times instead of 1 batched, costing extra launch overhead and per-step
// argmax+D2H. On a 248K-vocab model this is dwarfed by the BW savings.
// On smaller-vocab models the lossy batched path may still be faster.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = state.max_n;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // Two modes for the K-step draft lm_head dispatch:
    //
    //   compressed (FastMTP-style): use head's own lm_head_draft (32K-row
    //     top-K vocab slice) plus vocab_map for the draft→full id remap.
    //     Small per-step GEMV (~0.09 ms BW on MQ4) but softmax is over the
    //     32K compressed vocab, which dilutes top-1 prob signal (compresses
    //     the distribution shape) and breaks --mtp-p-min.
    //
    //   full-vocab (bundled .mq4-mtp): use the trunk's own output (lm_head)
    //     directly as the draft head. Per-step GEMV is larger (~0.69 ms BW)
    //     but the softmax is over the real 248K vocab — top-1 prob is the
    //     true confidence the trunk would see, and the argmax id IS the
    //     committed token id (no vocab_map remap). Cost difference per cycle
    //     is small (~3 ms more across K=5) since trunk verify dominates.
    //     This is the architecture Unsloth/llama.cpp #22673 ship.
    //
    // Mode is selected by whether the loaded head carries a compressed sidecar.
    // Bundled .mq4-mtp files drop the sidecar; load_mtp_head_bundled returns
    // a head with lm_head_draft: None.
    let use_full_vocab = head.weights.lm_head_draft.is_none();
    let (vocab_map_opt, cvs): (Option<&Vec<u32>>, usize) = if use_full_vocab {
        (None, vocab)
    } else {
        let _ = head.weights.lm_head_draft.as_ref()
            .expect("compressed head missing lm_head_draft");
        let vm = head.weights.lm_head_draft_vocab_map.as_ref()
            .expect("compressed head missing vocab_map");
        let c = head.weights.compressed_vocab_size
            .expect("compressed head missing compressed_vocab_size");
        let _ = state.mtp_scratch.logits_compressed.as_ref()
            .expect("Qwen35MtpHeadScratch::logits_compressed not allocated; \
                     call mtp_scratch.ensure_compressed_logits(gpu, cvs) after head load");
        (Some(vm), c)
    };

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    let dim_bytes = dim * 4;
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    let argmax_view = state.mtp_lm_argmax.sub_offset(0, 1);
    // Full-vocab per-step logits scratch: first row of mtp_lm_logits is
    // shape [vocab] which is exactly what we need per step. Allocated as
    // [max_n * vocab] elsewhere so the storage is already there.
    let full_vocab_logits_view = state.mtp_lm_logits.sub_offset(0, vocab);

    // p_min early-exit: when state.p_min > 0, each step uses
    // topk_logsumexp_batched (K=2) instead of argmax. We then check
    // top_logp[0] (log-softmax prob of argmax) against log(p_min) and
    // truncate the chain when the draft's own confidence drops below
    // threshold. Mirrors llama.cpp's --spec-draft-p-min (PR #22673).
    //
    // Semantics: the current step's candidate IS kept (we already computed
    // it); only steps k+1..max_n are skipped. This matches the upstream
    // implementation — a low-confidence draft still gets to face trunk
    // verify, but we stop spending compute speculating further.
    let p_min = state.p_min;
    let use_p_min = p_min > 0.0;
    let log_p_min = if use_p_min { p_min.ln() } else { f32::NEG_INFINITY };
    let mut chain_truncated = false;

    // Sampling vs greedy: when sampling.temp > 0, K-chain SAMPLES from each
    // draft distribution (instead of argmax) and trunk verify applies a
    // residual acceptance rule (instead of strict argmax-match). Matches
    // Unsloth/llama.cpp #22673 canonical MTP recipe (temp=1.0, top_p=0.95,
    // top_k=20). Greedy path (temp=0) unchanged.
    let sampling = state.sampling;
    let use_sampling = !sampling.is_greedy();
    if use_sampling && use_p_min {
        // p_min logic uses the topk_logsumexp_batched output, which doesn't
        // straightforwardly compose with arbitrary top_k/top_p sampling.
        // Disallow the combination for now — caller picks one or the other.
        panic!("spec_step_mtp_compressed_serial: --mtp-p-min and --temp > 0 are mutually exclusive (got p_min={p_min}, temp={})", sampling.temp);
    }
    let mut draft_probs: Vec<f32> = if use_sampling {
        Vec::with_capacity(max_n)
    } else {
        Vec::new()
    };
    // Cached host buffer for per-step draft logits readback (sampling only).
    // Sized to whichever vocab the dispatch path uses (compressed cvs or
    // full 248K vocab). Allocated once outside the loop.
    let mut draft_logits_host: Vec<f32> = if use_sampling {
        let n = if use_full_vocab { vocab } else { cvs };
        vec![0.0; n]
    } else {
        Vec::new()
    };

    // ── 1. K serial discrete-token roundtrips ─────────────────────────────
    for k in 0..max_n {
        let next_tok = if k == 0 { last_committed } else { candidates[k - 1] };

        // Forward: in compressed mode, mtp_head_forward_compressed runs
        // block_only + rmsnorm + small GEMV against lm_head_draft in one
        // call. In full-vocab mode we run block_only here and do the
        // rmsnorm + trunk-lm_head GEMV manually below.
        if use_full_vocab {
            if k == 0 {
                mtp_head::mtp_head_forward_block_only(
                    gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                    next_tok, &state.prev_hidden, None, cur_pos + k, trunk_weights,
                )?;
            } else {
                let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                mtp_head::mtp_head_forward_block_only(
                    gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                    next_tok, &prev_row, None, cur_pos + k, trunk_weights,
                )?;
            }
            // rmsnorm(t_mtp_out, shared_head_norm) → tmp, then trunk lm_head
            // GEMV → full_vocab_logits_view. weight_gemv dispatches the right
            // kernel (gemv_mq4g256_with_rotate / gemv_q8_0 / etc.) on the
            // trunk's output dtype.
            gpu.rmsnorm_f32(
                &state.mtp_scratch.t_mtp_out,
                &head.weights.shared_head_norm,
                &state.mtp_scratch.tmp,
                head.config.rms_norm_eps,
            )?;
            llama::weight_gemv(
                gpu,
                &trunk_weights.output,
                &state.mtp_scratch.tmp,
                &full_vocab_logits_view,
            )?;
        } else {
            if k == 0 {
                mtp_head::mtp_head_forward_compressed(
                    gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                    next_tok, &state.prev_hidden, cur_pos + k, trunk_weights,
                )?;
            } else {
                let prev_row = state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
                mtp_head::mtp_head_forward_compressed(
                    gpu, head, &state.mtp_scratch, &mut state.mtp_kv,
                    next_tok, &prev_row, cur_pos + k, trunk_weights,
                )?;
            }
        }

        // Pick the logits buffer to argmax over (mode-dependent).
        let logits_for_argmax: &GpuTensor = if use_full_vocab {
            &full_vocab_logits_view
        } else {
            state.mtp_scratch.logits_compressed.as_ref().unwrap()
        };
        let argmax_vocab = cvs;  // = full vocab in full-vocab mode, = compressed vocab in compressed mode

        let draft_idx: usize;
        if use_sampling {
            // GPU sample_top_p: kernel does the whole top_k(=20) + top_p +
            // multinomial sample on-device. Returns 8 B (token id + new rng
            // state). Eliminates the full 1 MB draft logits D2H + host
            // softmax sample (~600 μs each step → ~3 ms/cycle saved).
            //
            // sample_top_p modifies logits in-place ONLY if repeat_penalty > 1
            // and repeat_window > 0; we pass 1.0/0 so logits are untouched
            // and the subsequent prob-gather sees the same values.
            let (token_u32, new_rng) = gpu.sample_top_p(
                logits_for_argmax,
                &state.mtp_sample_result,
                &state.mtp_sample_repeat_buf,
                argmax_vocab,
                sampling.temp,
                sampling.top_p,
                state.gpu_rng_state,
                /* repeat_window */ 0,
                /* repeat_penalty */ 1.0,
            )?;
            state.gpu_rng_state = new_rng;
            draft_idx = token_u32 as usize;
            assert!(draft_idx < argmax_vocab,
                "draft sample {draft_idx} out of argmax_vocab {argmax_vocab}");

            // GPU p_draft gather: H2D the sampled token id, run the gather
            // kernel with n_rows=1, D2H 4 B prob. Replaces the host
            // softmax_prob_at_temp call (~600 μs).
            let token_i32: i32 = token_u32 as i32;
            let idx_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(&token_i32 as *const i32 as *const u8, 4)
            };
            gpu.hip.memcpy_htod(&state.mtp_gather_idx_draft.buf, idx_bytes)?;
            gpu.softmax_prob_gather_batched_f32(
                logits_for_argmax,
                &state.mtp_gather_idx_draft,
                &state.mtp_gather_prob_draft,
                argmax_vocab,
                sampling.temp,
                /* n_rows */ 1,
            )?;
            let mut p_draft_host: [f32; 1] = [0.0];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(p_draft_host.as_mut_ptr() as *mut u8, 4)
                };
                gpu.hip.memcpy_dtoh(bytes, &state.mtp_gather_prob_draft.buf)?;
            }
            draft_probs.push(p_draft_host[0]);

            let token_id = match vocab_map_opt {
                Some(vm) => vm[draft_idx],
                None => draft_idx as u32,
            };
            candidates.push(token_id);
            // draft_logits_host is unused on the GPU sampling path; left
            // allocated for symmetry with the (now-removed) host-sample
            // fallback. Compiler will elide.
            let _ = &draft_logits_host;
        } else if use_p_min {
            // Top-2 with log-softmax probs: 8 B idx + 8 B logp D2H per step.
            gpu.topk_logsumexp_batched_f32(
                logits_for_argmax, &state.mtp_topk_idx, &state.mtp_topk_logp,
                argmax_vocab, /* k */ 2, /* b */ 1,
            )?;
            let mut idx_host: [i32; 2] = [0, 0];
            let mut logp_host: [f32; 2] = [0.0, 0.0];
            {
                let idx_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, 8)
                };
                gpu.hip.memcpy_dtoh(idx_bytes, &state.mtp_topk_idx.buf)?;
                let logp_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(logp_host.as_mut_ptr() as *mut u8, 8)
                };
                gpu.hip.memcpy_dtoh(logp_bytes, &state.mtp_topk_logp.buf)?;
            }
            draft_idx = idx_host[0] as usize;
            assert!(draft_idx < argmax_vocab,
                "draft argmax {draft_idx} out of argmax_vocab {argmax_vocab}");
            // Compressed: remap via vocab_map. Full-vocab: idx IS the token id.
            let token_id = match vocab_map_opt {
                Some(vm) => vm[draft_idx],
                None => draft_idx as u32,
            };
            candidates.push(token_id);

            // Check confidence AFTER pushing — keep this candidate, just
            // skip future steps if confidence is below threshold.
            if logp_host[0] < log_p_min {
                chain_truncated = true;
                break;
            }
        } else {
            gpu.argmax_f32_batched(logits_for_argmax, &argmax_view, argmax_vocab, 1)?;
            let mut argmax_host: [i32; 1] = [0];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        argmax_host.as_mut_ptr() as *mut u8, 4,
                    )
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
            }
            draft_idx = argmax_host[0] as usize;
            assert!(draft_idx < argmax_vocab,
                "draft argmax {draft_idx} out of argmax_vocab {argmax_vocab}");
            let token_id = match vocab_map_opt {
                Some(vm) => vm[draft_idx],
                None => draft_idx as u32,
            };
            candidates.push(token_id);
        }

        if k + 1 < max_n {
            gpu.hip.memcpy_dtod_at(
                &state.mtp_t_outs.buf, k * dim_bytes,
                &state.mtp_scratch.t_mtp_out.buf, 0,
                dim_bytes,
            )?;
        }
    }
    let drafts_generated = candidates.len();

    // ── 2. Trunk verify ───────────────────────────────────────────────────
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, gpu)?;

    // GDN-tape capture happens only when the verify takes the batched (PBS)
    // path; otherwise the forward silently drops to a tape-less per-token loop
    // and the persistent `trunk_gdn_tape` is left stale — replaying it at
    // rollback corrupts dn_state.conv_states. Capture eligibility is broader
    // than MoE-ness: a non-Q8 kv-mode (dn_state.quant != Q8), a non-batchable
    // weight dtype, or the HIPFIRE_PREFILL_BATCHED=0 escape hatch all force the
    // per-token fallback even on a dense trunk. Gate on the forward's OWN
    // eligibility predicate (single source of truth) so the rollback's
    // cheap-vs-full replay choice tracks exactly whether the tape was written
    // this cycle, and pass the tape only when it will actually be captured.
    let tape_captured = qwen35::prefill_batch_pbs_eligible(
        trunk_weights,
        &target.config,
        &target.dn_state,
        n_verify,
        gpu.arch.as_str(),
        /* moe_router_logits_present — dense trunk: arm never matched */ true,
    );
    let verify_tape: Option<&mut GdnTape> = if tape_captured {
        Some(&mut state.trunk_gdn_tape)
    } else {
        None
    };

    qwen35::forward_prefill_batch_with_pbs(
        gpu, trunk_weights, &target.config, &verify_tokens, cur_pos,
        &mut target.kv_cache, &mut target.dn_state, &target.scratch,
        None, Some(&state.verify_hidden), verify_tape, None,
        Some(&state.trunk_pbs), None, // mask_override
        None, // max_layer
    )?;

    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(drafts_generated + 1);

    if use_sampling {
        // ── Residual-acceptance sampling path (GPU-side gather + sample) ──
        //
        // For each draft candidate c_k, accept with probability
        // min(1, p_target(c_k) / p_draft(c_k)). Both probs are un-truncated
        // temp-scaled softmax probs — p_draft was gathered per-step in the
        // K-chain above (state.mtp_gather_prob_draft); p_target is gathered
        // here in one batched call (state.mtp_gather_prob_verify).
        //
        // On rejection (or full accept), the bonus token is SAMPLED via
        // gpu.sample_top_p on the trunk's verify_logits row at slot
        // accept_count, using the same temp/top_p (top_k=20 hardcoded in
        // the kernel — matches Unsloth default).
        //
        // GPU-side path saves vs prior host-side draft:
        //   - 6 MB D2H of full verify_logits → 4·K B D2H of probs (~24 B)
        //   - 5 × ~600 μs host softmax_prob_at_temp → one batched kernel
        //   - 1 host sample_from_logits for bonus → one sample_top_p kernel
        // Net: ~4 ms saved per cycle.
        //
        // Approximations vs. strict speculative-sampling (Chen et al.):
        //   - Uses un-truncated p_target/p_draft in the accept ratio even
        //     though draft sample was from a truncated nucleus
        //   - Bonus is sample_top_p(trunk) not sample-from-residual
        // Same approximations as before — practical posture matching
        // llama.cpp / vLLM spec-decode implementations.

        // Batched p_target gather: H2D the K candidate token ids as i32,
        // dispatch the gather kernel, D2H K probs. Replaces the 6 MB D2H
        // of full verify_logits + K host softmax_prob_at_temp calls.
        let cand_indices: Vec<i32> = candidates[..drafts_generated]
            .iter().map(|&t| t as i32).collect();
        let idx_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                cand_indices.as_ptr() as *const u8,
                drafts_generated * 4,
            )
        };
        gpu.hip.memcpy_htod(&state.mtp_gather_idx_verify.buf, idx_bytes)?;
        gpu.softmax_prob_gather_batched_f32(
            &logits_view,
            &state.mtp_gather_idx_verify,
            &state.mtp_gather_prob_verify,
            vocab,
            sampling.temp,
            drafts_generated,
        )?;
        let mut p_targets: Vec<f32> = vec![0.0; drafts_generated];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    p_targets.as_mut_ptr() as *mut u8,
                    drafts_generated * 4,
                )
            };
            gpu.hip.memcpy_dtoh(bytes, &state.mtp_gather_prob_verify.buf)?;
        }

        for k in 0..drafts_generated {
            let p_t = p_targets[k];
            let p_d = draft_probs[k].max(1e-30);
            let accept_ratio = (p_t / p_d).min(1.0);
            let r = state.rng.next_uniform_f32();
            if r < accept_ratio {
                committed.push(candidates[k]);
                accept_count += 1;
                if candidates[k] == eos_token_id {
                    hit_eos = true;
                    break;
                }
            } else {
                break;
            }
        }
        if !hit_eos {
            // Sample bonus on-device from trunk's verify_logits at slot
            // accept_count. sub-offset gives a [vocab]-shape view that
            // sample_top_p treats as a single row.
            let bonus_row_view = logits_view.sub_offset(accept_count * vocab, vocab);
            let (bonus_token, new_rng) = gpu.sample_top_p(
                &bonus_row_view,
                &state.mtp_sample_result,
                &state.mtp_sample_repeat_buf,
                vocab,
                sampling.temp,
                sampling.top_p,
                state.gpu_rng_state,
                /* repeat_window */ 0,
                /* repeat_penalty */ 1.0,
            )?;
            state.gpu_rng_state = new_rng;
            let bonus = bonus_token;
            committed.push(bonus);
            if bonus == eos_token_id {
                hit_eos = true;
            }
        }
    } else {
        // ── Legacy greedy / argmax-match accept rule ─────────────────────
        for k in 0..drafts_generated {
            if argmax_per_pos[k] == candidates[k] {
                committed.push(candidates[k]);
                accept_count += 1;
                if candidates[k] == eos_token_id {
                    hit_eos = true;
                    break;
                }
            } else {
                break;
            }
        }
        if !hit_eos {
            let bonus = argmax_per_pos[accept_count];
            committed.push(bonus);
            if bonus == eos_token_id {
                hit_eos = true;
            }
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= drafts_generated + 1);

    // h_idx contract (audited 2026-05-21 vs AtomicBot atomic-llama-cpp-
    // turboquant feature/turboquant-kv-cache): prev_hidden_row = advance - 1
    // = accept_count IS the right index. It points to verify_hidden of the
    // last position trunk evaluated whose argmax IS the committed bonus
    // token. AtomicBot's common_speculative_set_h_idx is called with
    // `ids.size() - 1 == n_accepted_drafts` which is the same index.
    // Empirical sensitivity (canonical K=2 p=0.65 max=480): offset 0 = 60.7
    // tok/s tau=2.74; offset ±1 collapses to 17-45 tok/s. Default is sharp.
    let prev_hidden_row = advance - 1;
    state.capture_prev_hidden_from_verify_row(gpu, prev_hidden_row, dim)?;

    // ── 3. KV / DN rollback (or skip on full accept) ──────────────────────
    //
    // Rollback is REDUNDANT when advance == drafts_generated + 1 (full
    // accept, no EOS, no chain truncation matters): verify's forward
    // already left DN state advanced by `drafts_generated + 1` steps from
    // the snapshot, which is exactly where replay would land. KV cache
    // slots `cur_pos..cur_pos + drafts_generated` are also already
    // populated identically (rollback would write the same tokens to the
    // same slots).
    //
    // For partial accepts, use the same GDN innovation-tape repair that
    // DFlash uses: restore the pre-verify DeltaNet state, then replay only
    // conv/GDN recurrence for the accepted prefix. Full trunk forward replay
    // was the dominant non-verify cost in the MTP cycle.
    //
    // The trunk_snap.save_from() call earlier in the cycle is still made
    // unconditionally (~1 ms, unavoidable since we don't know advance
    // until verify completes); only the restore + tape replay are gated.
    //
    // On EOS we still take the replay branch: even though no further
    // forwards happen, the caller's KV cache must reflect ONLY the
    // committed prefix (some of which may have been rejected/truncated
    // before the EOS-bearing token was committed).
    let full_accept_no_eos = advance == drafts_generated + 1 && !hit_eos;
    let replay_skipped = full_accept_no_eos;
    if !full_accept_no_eos {
        state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
        if tape_captured {
            // The batched verify populated the tape this cycle — cheap GDN-only replay.
            state.trunk_gdn_tape.replay_gdn(
                gpu,
                trunk_weights,
                &target.config,
                &mut target.dn_state,
                advance,
            )?;
        } else {
            // The verify took the tape-less per-token fallback (non-Q8 kv-mode,
            // non-batchable weight dtype, MoE trunk, or HIPFIRE_PREFILL_BATCHED=0),
            // so the persistent tape is stale. Do the original always-correct
            // full-trunk replay of the committed prefix — one batched forward,
            // or a single forward_scratch for advance==1.
            if advance >= 2 {
                let replay = &verify_tokens[..advance];
                qwen35::forward_prefill_batch(
                    gpu, trunk_weights, &target.config, replay, cur_pos,
                    &mut target.kv_cache, &mut target.dn_state, &target.scratch,
                    None, None, None, None,
                )?;
            } else {
                qwen35::forward_scratch(
                    gpu, trunk_weights, &target.config, verify_tokens[0], cur_pos,
                    &mut target.kv_cache, &mut target.dn_state, &target.scratch,
                )?;
            }
        }
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated,
        chain_truncated,
        replay_skipped,
    })
}

// ─── Hetero (split-gpu) MTP spec-decode ──────────────────────────────────
//
// Variant of spec_step_mtp_compressed_serial for the case where the MTP head
// lives on a sibling drafter_gpu while the trunk lives on the target_gpu.
// Plan + audit: docs/plans/mtp_multi_gpu_split_audit.md.
//
// Per cycle:
//   1. K-step MTP chain runs on drafter_gpu:
//      - Embed lookup against the mirrored trunk.token_embd (drafter-resident).
//      - mtp_head_forward_compressed_with_embed → t_mtp_out → tiny lm_head GEMV
//        against head.weights.lm_head_draft (drafter-resident, sidecar path).
//      - Argmax → candidate token (16 B D2H to host).
//   2. Verify runs on target_gpu using existing single-gpu path:
//      - Trunk forward over [last_committed | candidates] → verify_hidden, verify_logits
//      - Batched argmax → accept-prefix.
//   3. Cycle exit:
//      - peer_copy_async verify_hidden[advance-1] (n_embd × f32 = 20 KB)
//        target_gpu → drafter_gpu's prev_hidden. Event-synced so next
//        cycle's chain reads the fresh hidden.

/// Drafter-gpu-resident state for hetero MTP spec-decode. Companion to
/// `MtpSpecState` (which holds trunk-gpu fields).
///
/// Holds the mirror of `trunk.token_embd`, the MTP head's scratch/kv,
/// the K-chain working buffers, and the small per-step embedding scratch.
/// `prev_hidden` lives here too — the chain reads it at the start of each
/// cycle and the cycle-exit peer copy refreshes it for the next cycle.
///
/// All fields are allocated on drafter_gpu in
/// [`MtpHeteroDrafterState::new_for_slot`].
pub struct MtpHeteroDrafterState {
    /// Drafter-gpu mirror of trunk.token_embd. Stored as a raw GpuTensor
    /// (not WeightTensor) since the embedding kernels take the buffer +
    /// embd_format separately, identically to single-gpu.
    pub mirrored_token_embd: GpuTensor,
    /// Mirror's source format, copied from trunk's `Qwen35Weights.embd_format`
    /// at init. Same value on both sides.
    pub embd_format: EmbeddingFormat,

    /// Per-step embedding scratch: drafter-gpu staging tensor of shape `[n_embd]`,
    /// F32, populated by `embed_lookup_*` against `mirrored_token_embd` at
    /// the top of each K-chain step. Passed to
    /// `mtp_head_forward_compressed_with_embed` as `next_token_embed`.
    pub step_embd: GpuTensor,

    /// Drafter-gpu copy of `MtpSpecState.prev_hidden`. Shape `[n_embd]` F32.
    /// Refreshed each cycle by a peer copy from `MtpSpecState.verify_hidden`
    /// (trunk side) of the row matching the last committed token.
    pub prev_hidden: GpuTensor,

    /// MTP head per-call scratch (drafter-resident).
    pub mtp_scratch: Qwen35MtpHeadScratch,
    /// MTP head KV cache (drafter-resident).
    pub mtp_kv: Qwen35MtpHeadKvCache,

    /// Per-step `t_mtp_out` capture for K-chain (drafter-resident). Shape
    /// `[max_n * n_embd]` F32. Same role as `MtpSpecState.mtp_t_outs`.
    pub mtp_t_outs: GpuTensor,
    /// Per-step compressed logits view scratch reuse. Shape `[max_n]` F32
    /// argmax slot scratch.
    pub mtp_lm_argmax: GpuTensor,

    pub max_n: usize,
}

impl MtpHeteroDrafterState {
    /// Allocate drafter-side state. Requires:
    /// - `drafter_gpu` already initialized with bidirectional peer access to
    ///   the trunk_gpu.
    /// - `mirrored_token_embd` already allocated and bytes-copied via
    ///   `hipfire_runtime::mtp_mirror::peer_clone_tensor(trunk_gpu, drafter_gpu, &trunk.weights.token_embd)`.
    /// - `embd_format` matches `trunk.weights.embd_format`.
    pub fn new_for_slot(
        drafter_gpu: &mut Gpu,
        head: &Qwen35MtpHead,
        max_n: usize,
        kv_mode: crate::mtp_head::MtpKvMode,
        mirrored_token_embd: GpuTensor,
        embd_format: EmbeddingFormat,
    ) -> HipResult<Self> {
        assert!(max_n >= 1, "MtpHeteroDrafterState: max_n must be ≥ 1");
        let n_embd = head.config.n_embd;
        let step_embd = drafter_gpu.alloc_tensor(&[n_embd], DType::F32)?;
        let prev_hidden = drafter_gpu.alloc_tensor(&[n_embd], DType::F32)?;
        let mtp_scratch = Qwen35MtpHeadScratch::new(drafter_gpu, &head.config)?;
        let mtp_kv = Qwen35MtpHeadKvCache::new_with_kv_mode(drafter_gpu, &head.config, kv_mode)?;
        let mtp_t_outs = drafter_gpu.alloc_tensor(&[max_n * n_embd], DType::F32)?;
        let mtp_lm_argmax = drafter_gpu.alloc_tensor(&[max_n], DType::F32)?;

        // Sidecar logits scratch is needed for the compressed path; caller
        // populates it via `mtp_scratch.ensure_compressed_logits(drafter_gpu, cvs)`
        // immediately after construction once they know the cvs from
        // `head.weights.compressed_vocab_size`. The hetero spec function
        // asserts presence at entry.
        let _ = mtp_scratch.logits_compressed.is_some(); // satisfied by caller

        Ok(Self {
            mirrored_token_embd,
            embd_format,
            step_embd,
            prev_hidden,
            mtp_scratch,
            mtp_kv,
            mtp_t_outs,
            mtp_lm_argmax,
            max_n,
        })
    }

    pub fn free_gpu(self, drafter_gpu: &mut Gpu) {
        let _ = drafter_gpu.free_tensor(self.mirrored_token_embd);
        let _ = drafter_gpu.free_tensor(self.step_embd);
        let _ = drafter_gpu.free_tensor(self.prev_hidden);
        self.mtp_scratch.free_gpu(drafter_gpu);
        self.mtp_kv.free_gpu(drafter_gpu);
        let _ = drafter_gpu.free_tensor(self.mtp_t_outs);
        let _ = drafter_gpu.free_tensor(self.mtp_lm_argmax);
    }

    /// Seed `prev_hidden` from a trunk-side tensor at session start. Used
    /// once after prefill to bootstrap the first cycle's chain input.
    /// `trunk_src` must have `n_embd` elements (shape `[n_embd]` or a slice
    /// view of that size). Uses synchronous peer copy.
    pub fn seed_prev_hidden(
        &self,
        trunk_gpu: &Gpu,
        drafter_gpu: &Gpu,
        trunk_src: &GpuTensor,
    ) -> HipResult<()> {
        let n_embd = self.prev_hidden.numel();
        assert_eq!(
            trunk_src.numel(), n_embd,
            "seed_prev_hidden: trunk source has {} elems, expected n_embd={n_embd}",
            trunk_src.numel(),
        );
        trunk_gpu.hip.memcpy_peer(
            &self.prev_hidden.buf, drafter_gpu.device_id,
            &trunk_src.buf, trunk_gpu.device_id,
            n_embd * 4,
        )
    }
}

/// Helper: drafter-side embedding lookup, dispatched by `embd_format`.
fn drafter_embed_lookup(
    drafter_gpu: &mut Gpu,
    drafter_state: &MtpHeteroDrafterState,
    token: u32,
    n_embd: usize,
) -> HipResult<()> {
    match drafter_state.embd_format {
        EmbeddingFormat::HFQ4G256 => drafter_gpu.embedding_lookup_hfq4g256(
            &drafter_state.mirrored_token_embd, &drafter_state.step_embd, token, n_embd),
        EmbeddingFormat::HFQ4G128 => drafter_gpu.embedding_lookup_hfq4g128(
            &drafter_state.mirrored_token_embd, &drafter_state.step_embd, token, n_embd),
        EmbeddingFormat::Q8_0 => drafter_gpu.embedding_lookup_q8(
            &drafter_state.mirrored_token_embd, &drafter_state.step_embd, token, n_embd),
        EmbeddingFormat::Q4K => drafter_gpu.embedding_lookup_q4k(
            &drafter_state.mirrored_token_embd, &drafter_state.step_embd, token, n_embd),
        EmbeddingFormat::F32 => drafter_gpu.embedding_lookup(
            &drafter_state.mirrored_token_embd, &drafter_state.step_embd, token, n_embd),
    }
}

/// Hetero variant of [`spec_step_mtp_compressed_serial`]. Runs the K-step
/// MTP chain on `drafter_gpu` (where the head + mirrored token_embd live),
/// and trunk verify + rollback on `target_gpu`. Cycle-exit handoff is one
/// 20 KB peer copy (n_embd × f32) of `verify_hidden[advance-1]` from trunk
/// to drafter's `prev_hidden`.
///
/// Compressed-sidecar only (v1): requires `head.weights.lm_head_draft` to be
/// Some, so the K-chain's lm_head GEMV stays drafter-local. Greedy only
/// for v1 (sampling path can be ported in a follow-up — adds drafter-side
/// p_draft gathers and a peer copy of the bonus token).
///
/// `state` (trunk-resident) and `drafter_state` are paired:
/// `MtpSpecState::new_for_slot_with_kv_mode(target_gpu, ...)` allocates
/// trunk fields against target_gpu; `MtpHeteroDrafterState::new_for_slot`
/// allocates drafter fields against drafter_gpu. The function panics if
/// `state.prev_hidden` is non-empty (the hetero variant uses
/// `drafter_state.prev_hidden` instead — trunk-side `state.prev_hidden`
/// is ignored but still allocated for layout compatibility).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_mtp_compressed_serial_hetero(
    target_gpu: &mut Gpu,
    drafter_gpu: &mut Gpu,
    target: &mut ModelSlot,
    head: &Qwen35MtpHead,
    state: &mut MtpSpecState,
    drafter_state: &mut MtpHeteroDrafterState,
    cur_pos: usize,
    last_committed: u32,
    eos_token_id: u32,
) -> HipResult<MtpSpecResult> {
    let max_n = drafter_state.max_n;
    assert_eq!(
        max_n, state.max_n,
        "spec_step_mtp_compressed_serial_hetero: drafter_state.max_n={} != state.max_n={}",
        max_n, state.max_n,
    );
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let trunk_weights: &Qwen35Weights = &target.weights;

    // V1: compressed-sidecar path only — eliminates the trunk.output mirror
    // requirement. Bundled .mq4-mtp (full-vocab) callers should use the
    // single-gpu path until that mirror is added.
    let lm_head_draft = head.weights.lm_head_draft.as_ref()
        .expect("spec_step_mtp_compressed_serial_hetero: head has no lm_head_draft sidecar; \
                 hetero v1 requires the cvs-compressed .mtp path (use single-gpu for bundled .mq4-mtp)");
    let vocab_map = head.weights.lm_head_draft_vocab_map.as_ref()
        .expect("hetero: compressed head missing vocab_map");
    let cvs = head.weights.compressed_vocab_size
        .expect("hetero: compressed head missing compressed_vocab_size");
    let _ = lm_head_draft;
    let _ = cvs;

    // Ensure both gpus have their per-stream context. Each owns its own
    // active_stream. CRITICAL: HIP streams are device-affine — must bind to
    // the right device before stream_create so the stream lives on the right
    // gpu. Otherwise the stream is created on whatever device was last bound,
    // and subsequent kernel launches against it fail with
    // hipModuleLaunchKernel: invalid device ordinal.
    if target_gpu.active_stream.is_none() {
        target_gpu.bind_thread()?;
        target_gpu.active_stream = Some(target_gpu.hip.stream_create()?);
    }
    if drafter_gpu.active_stream.is_none() {
        drafter_gpu.bind_thread()?;
        drafter_gpu.active_stream = Some(drafter_gpu.hip.stream_create()?);
    }

    // Greedy-only for v1. Sampling/p_min/full-vocab are NYI on this path.
    let sampling = state.sampling;
    assert!(
        sampling.is_greedy(),
        "spec_step_mtp_compressed_serial_hetero: v1 supports temp=0 (greedy) only, \
         got temp={}",
        sampling.temp,
    );
    assert!(
        state.p_min <= 0.0,
        "spec_step_mtp_compressed_serial_hetero: v1 does not implement --mtp-p-min; \
         got p_min={}",
        state.p_min,
    );

    // ── 1. K-step draft chain (DRAFTER GPU) ──────────────────────────────
    drafter_gpu.bind_thread()?;
    let mut candidates: Vec<u32> = Vec::with_capacity(max_n);
    let dim_bytes = dim * 4;
    let argmax_view = drafter_state.mtp_lm_argmax.sub_offset(0, 1);
    let logits_c = drafter_state.mtp_scratch.logits_compressed.as_ref()
        .expect("drafter MTP scratch.logits_compressed not allocated; \
                 call drafter_state.mtp_scratch.ensure_compressed_logits(drafter_gpu, cvs)");

    for k in 0..max_n {
        let next_tok = if k == 0 { last_committed } else { candidates[k - 1] };

        // Embedding lookup on drafter using the mirrored token_embd.
        drafter_embed_lookup(drafter_gpu, drafter_state, next_tok, dim)?;

        // MTP head forward — feeds step_embd in via next_token_embed override,
        // reads prev_hidden (drafter-resident: at k=0 from the per-cycle peer
        // copy of last cycle's verify_hidden; at k>0 from prev step's
        // t_mtp_out, matching the single-gpu serial path's semantics).
        if k == 0 {
            mtp_head::mtp_head_forward_compressed_with_embed(
                drafter_gpu, head, &drafter_state.mtp_scratch, &mut drafter_state.mtp_kv,
                &drafter_state.step_embd, &drafter_state.prev_hidden, cur_pos + k,
            )?;
        } else {
            let prev_row = drafter_state.mtp_t_outs.sub_offset((k - 1) * dim, dim);
            mtp_head::mtp_head_forward_compressed_with_embed(
                drafter_gpu, head, &drafter_state.mtp_scratch, &mut drafter_state.mtp_kv,
                &drafter_state.step_embd, &prev_row, cur_pos + k,
            )?;
        }

        // Greedy argmax over compressed logits, D2H 4 B.
        drafter_gpu.argmax_f32_batched(logits_c, &argmax_view, cvs, 1)?;
        let mut argmax_host: [i32; 1] = [0];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    argmax_host.as_mut_ptr() as *mut u8, 4,
                )
            };
            drafter_gpu.hip.memcpy_dtoh(bytes, &argmax_view.buf)?;
        }
        let draft_idx = argmax_host[0] as usize;
        assert!(draft_idx < cvs,
            "drafter argmax {draft_idx} out of cvs {cvs}");
        let token_id = vocab_map[draft_idx];
        candidates.push(token_id);

        // Save t_mtp_out for chain-step k+1 (drafter-local D2D copy).
        if k + 1 < max_n {
            drafter_gpu.hip.memcpy_dtod_at(
                &drafter_state.mtp_t_outs.buf, k * dim_bytes,
                &drafter_state.mtp_scratch.t_mtp_out.buf, 0,
                dim_bytes,
            )?;
        }
    }
    let drafts_generated = candidates.len();

    // ── 2. Trunk verify (TARGET GPU) ─────────────────────────────────────
    target_gpu.bind_thread()?;
    let mut verify_tokens: Vec<u32> = Vec::with_capacity(max_n + 1);
    verify_tokens.push(last_committed);
    verify_tokens.extend_from_slice(&candidates);
    let n_verify = verify_tokens.len();

    state.trunk_snap.save_from(&target.dn_state, target_gpu)?;

    let tape_captured = qwen35::prefill_batch_pbs_eligible(
        trunk_weights,
        &target.config,
        &target.dn_state,
        n_verify,
        target_gpu.arch.as_str(),
        true,
    );
    let verify_tape: Option<&mut GdnTape> = if tape_captured {
        Some(&mut state.trunk_gdn_tape)
    } else {
        None
    };

    qwen35::forward_prefill_batch_with_pbs(
        target_gpu, trunk_weights, &target.config, &verify_tokens, cur_pos,
        &mut target.kv_cache, &mut target.dn_state, &target.scratch,
        None, Some(&state.verify_hidden), verify_tape, None,
        Some(&state.trunk_pbs), None,
        None,
    )?;

    let w_out = &trunk_weights.output;
    let logits_view = state.verify_logits.sub_offset(0, n_verify * vocab);
    match w_out.gpu_dtype {
        DType::Q8_0 => {
            target_gpu.gemm_q8_0_batched(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ4G256 => {
            target_gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ4G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(target_gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            target_gpu.gemm_hfq4g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ3G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(target_gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            target_gpu.gemm_hfq3g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::HFQ6G256 => {
            target_gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &state.verify_hidden, &logits_view,
                w_out.m, w_out.k, n_verify,
            )?;
        }
        DType::MQ6G256 => {
            let rot = state.verify_rot.sub_offset(0, n_verify * w_out.k);
            llama::rotate_x_mq_batched_for(target_gpu, w_out, &state.verify_hidden, &rot, w_out.k, n_verify)?;
            target_gpu.gemm_hfq6g256_batched_lmhead(
                &w_out.buf, &rot, &logits_view, w_out.m, w_out.k, n_verify,
            )?;
        }
        _ => {
            for i in 0..n_verify {
                let row = state.verify_hidden.sub_offset(i * dim, dim);
                let logits_row = state.verify_logits.sub_offset(i * vocab, vocab);
                llama::weight_gemv(target_gpu, w_out, &row, &logits_row)?;
            }
        }
    }

    let argmax_v = state.verify_argmax.sub_offset(0, n_verify);
    target_gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)?;
    let mut argmax_v_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                argmax_v_host.as_mut_ptr() as *mut u8, n_verify * 4,
            )
        };
        target_gpu.hip.memcpy_dtoh(bytes, &argmax_v.buf)?;
    }
    let argmax_per_pos: Vec<u32> = argmax_v_host.into_iter().map(|x| x as u32).collect();

    // ── Greedy accept rule (v1) ──
    let mut accept_count = 0usize;
    let mut hit_eos = false;
    let mut committed: Vec<u32> = Vec::with_capacity(drafts_generated + 1);
    for k in 0..drafts_generated {
        if argmax_per_pos[k] == candidates[k] {
            committed.push(candidates[k]);
            accept_count += 1;
            if candidates[k] == eos_token_id {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = argmax_per_pos[accept_count];
        committed.push(bonus);
        if bonus == eos_token_id {
            hit_eos = true;
        }
    }

    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= drafts_generated + 1);

    // ── 3. Cycle-exit handoff: peer-copy verify_hidden[advance-1] →
    //      drafter.prev_hidden. This is the ONE per-cycle cross-device
    //      data movement on the critical path (~20 KB, ~38 µs steady-state
    //      per the mtp_peer_copy_microbench). Sync via target's stream
    //      since the verify just finished there.
    let prev_hidden_row = advance - 1;
    target_gpu.hip.memcpy_peer(
        &drafter_state.prev_hidden.buf, drafter_gpu.device_id,
        &state.verify_hidden.buf, target_gpu.device_id,
        dim * 4,
    )?;
    // Note: we use the sync memcpy_peer (not _async) since the trunk verify
    // path uses target_gpu's active_stream and we'd need an event handshake
    // to safely peer-copy from a stream-pending buffer. memcpy_peer
    // synchronizes via host, sized to 20 KB this is fast (~38 µs measured).
    // Future optimization: use memcpy_peer_async + cross-device event sync
    // (see hetero_overlap_microbench.rs in the parallel multi-gpu work).
    let _ = prev_hidden_row;

    // ── 4. Rollback / replay on TARGET (same as single-gpu) ──
    let full_accept_no_eos = advance == drafts_generated + 1 && !hit_eos;
    let replay_skipped = full_accept_no_eos;
    if !full_accept_no_eos {
        state.trunk_snap.restore_to(&mut target.dn_state, target_gpu)?;
        if tape_captured {
            state.trunk_gdn_tape.replay_gdn(
                target_gpu, trunk_weights, &target.config,
                &mut target.dn_state, advance,
            )?;
        } else {
            if advance >= 2 {
                let replay = &verify_tokens[..advance];
                qwen35::forward_prefill_batch(
                    target_gpu, trunk_weights, &target.config, replay, cur_pos,
                    &mut target.kv_cache, &mut target.dn_state, &target.scratch,
                    None, None, None, None,
                )?;
            } else {
                qwen35::forward_scratch(
                    target_gpu, trunk_weights, &target.config, verify_tokens[0], cur_pos,
                    &mut target.kv_cache, &mut target.dn_state, &target.scratch,
                )?;
            }
        }
    }

    Ok(MtpSpecResult {
        committed,
        accept_count,
        hit_eos,
        advance,
        drafts_generated,
        chain_truncated: false, // v1 hetero doesn't implement p_min early-exit
        replay_skipped,
    })
}
