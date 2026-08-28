// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 EAGLE speculative-decode step (`spec_step_gemma4_eagle`).
//!
//! Ties the validated pieces together into one draft→verify→accept→commit
//! cycle:
//!   - the EAGLE drafter (`crate::drafter::drafter_step`, per-step cosine
//!     0.994–0.9996 vs HF) drafts N tokens at a CONSTANT query position; and
//!   - the batched verify forward (`crate::forward::forward_batch_spec`,
//!     argmax-identical to B sequential `decode_step`) verifies the block in one
//!     pass and exposes per-position argmax + per-position post-`model.norm`
//!     hidden.
//!
//! Structure mirrors qwen35's `spec_step_dflash` (draft loop, greedy accept
//! indexing, bonus token, next-seed-hidden capture from the verify row) but
//! gemma4 is a PURE-ATTENTION model — there is NO DeltaNet / recurrent state, so
//! all of dflash's `DeltaNetSnapshot` / `GdnTape` / replay machinery is dropped.
//! The only spec-decode state that matters is the KV cache, and the two gemma4
//! KV caches are absolute-position (each write is keyed by `pos_buf` into a
//! `physical_cap`-sized slot), so rejected drafts' KV slots are overwritten
//! losslessly when the next round's verify re-occupies those positions from the
//! new committed length. No explicit truncation is needed.
//!
//! ## The accept invariant (greedy, temp 0)
//!
//! With greedy verify the committed sequence is provably the target's greedy
//! AR sequence: `argmax_per_pos[i]` is the target's argmax AFTER block position
//! `i`, so accepting the longest prefix where `argmax_per_pos[i] == block[i+1]`
//! and emitting `bonus = argmax_per_pos[accept_len]` yields exactly what eager
//! `decode_step` would have produced. Any divergence from eager is a bug.

use crate::config::Gemma4Config;
use crate::drafter::{drafter_step, Gemma4DrafterConfig, Gemma4DrafterScratch, Gemma4DrafterWeights};
use crate::forward::forward_batch_spec;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use crate::lowered::tensor_owner_bytes;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Result of one EAGLE spec step.
pub struct SpecStepOut {
    /// Tokens committed this round: accepted drafts ++ [bonus]. Length ≥ 1.
    pub committed: Vec<u32>,
    /// Number of drafts accepted (0..=N). `committed.len() == accept_len + 1`.
    pub accept_len: usize,
    /// The bonus token (= next round's seed token). Last element of `committed`.
    pub next_seed_token: u32,
    /// Drafts produced this round (for diagnostics). Length N.
    pub drafts: Vec<u32>,
}

/// Reusable GPU scratch for the EAGLE spec loop. Holds the seed-hidden buffer
/// (post-`model.norm`, [dim]) that seeds each round, a buffer for the drafter's
/// per-step post-projection feedback ([dim]), and the [N+1, dim] verify-hidden
/// buffer that `forward_batch_spec` fills (so the next seed hidden is captured
/// from a verify row with no extra forward).
pub struct Gemma4SpecScratch {
    /// Last committed position's post-`model.norm` hidden — seeds the round. [dim]
    pub seed_hidden: GpuTensor,
    /// Drafter per-step hidden feedback (post_proj of the previous draft). [dim]
    pub draft_hidden: GpuTensor,
    /// Verify per-position post-`model.norm` hidden ([N_max+1, dim]); the next
    /// round's seed hidden is copied out of the accepted-bonus row.
    pub verify_hidden: GpuTensor,
    dim: usize,
}

impl Gemma4SpecScratch {
    /// `max_draft` = the largest `draft_len` that will be passed to
    /// `spec_step_gemma4_eagle` (the verify block is `max_draft + 1` rows).
    pub fn new(gpu: &mut Gpu, cfg: &Gemma4Config, max_draft: usize) -> Result<Self, String> {
        let dim = cfg.dim;
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("gemma4-spec: alloc {label}: {e:?}"))
        };
        Ok(Gemma4SpecScratch {
            seed_hidden: alloc(gpu, dim, "seed_hidden")?,
            draft_hidden: alloc(gpu, dim, "draft_hidden")?,
            verify_hidden: alloc(gpu, (max_draft + 1) * dim, "verify_hidden")?,
            dim,
        })
    }

    /// Seed the cycle from the prefill's last post-`model.norm` hidden (the
    /// lm_head input of the last prefilled token). For `forward_batch` prefill,
    /// pass the per-token-hidden row of the last position; for eager prefill,
    /// pass `state.tmp` after the final `decode_step`.
    pub fn set_seed_hidden_from(&self, gpu: &Gpu, src: &GpuTensor) -> Result<(), String> {
        gpu.hip
            .memcpy_dtod_at(&self.seed_hidden.buf, 0, &src.buf, 0, self.dim * 4)
            .map_err(|e| format!("gemma4-spec: seed_hidden copy: {e:?}"))
    }
    /// Sum actual capacities of the reusable EAGLE scratch buffers.
    pub fn owner_bytes(&self) -> usize {
        tensor_owner_bytes(&self.seed_hidden)
            + tensor_owner_bytes(&self.draft_hidden)
            + tensor_owner_bytes(&self.verify_hidden)
    }


    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.seed_hidden);
        let _ = gpu.free_tensor(self.draft_hidden);
        let _ = gpu.free_tensor(self.verify_hidden);
    }
}

/// One EAGLE spec-decode step.
///
/// * `gpu` — device handle.
/// * `target_w` / `target_cfg` / `target_state` — the arch-13 target (read +
///   KV-written by verify). `target_state.n_tokens` is updated to the new
///   committed length on success.
/// * `dw` / `dcfg` / `ds` — the arch-22 drafter (weights / config / scratch),
///   read-only against the target.
/// * `sp` — reusable spec scratch. `sp.seed_hidden` MUST hold the last committed
///   position's post-`model.norm` hidden on entry (set by the previous round's
///   capture, or by `set_seed_hidden_from` after prefill). On exit it holds the
///   NEXT round's seed hidden (the accepted-bonus row's hidden).
/// * `seed_token` — the last committed token (whose KV sits at position L-1).
/// * `committed_len` — L, the number of committed tokens (= the next position).
/// * `draft_len` — N drafts to speculate (verify block is N+1 tokens). Must be
///   ≤ the `max_draft` the scratch was sized for and keep `L-1+N+1 ≤ max_seq`.
/// * `temp` — 0.0 = greedy (the only validated mode here). temp > 0 is rejected.
///
/// Returns the committed tokens (accepted drafts ++ bonus) and the next seed
/// token. Side effects: verify writes both KV caches for [L-1, L+N) and the fn
/// sets `target_state.n_tokens = L + accept_len + 1`.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_gemma4_eagle(
    gpu: &mut Gpu,
    target_w: &Gemma4Weights,
    target_cfg: &Gemma4Config,
    target_state: &mut Gemma4State,
    dw: &Gemma4DrafterWeights,
    dcfg: &Gemma4DrafterConfig,
    ds: &mut Gemma4DrafterScratch,
    sp: &mut Gemma4SpecScratch,
    seed_token: u32,
    committed_len: usize,
    draft_len: usize,
    temp: f32,
) -> Result<SpecStepOut, String> {
    if temp != 0.0 {
        return Err("gemma4-spec: only greedy (temp=0) is supported".to_string());
    }
    if draft_len == 0 {
        return Err("gemma4-spec: draft_len must be ≥ 1".to_string());
    }
    let l = committed_len;
    if l == 0 {
        return Err("gemma4-spec: committed_len must be ≥ 1 (need a seed at L-1)".to_string());
    }
    let n = draft_len;
    let dim = target_cfg.dim;
    // The drafter queries the target's shared KV at the CONSTANT position L-1
    // across the whole round (per the candidate-generator semantics).
    let query_pos = l - 1;

    // ── 1. Draft N tokens. step0: prev = seed_token, h = sp.seed_hidden. ──
    let mut drafts: Vec<u32> = Vec::with_capacity(n);
    let mut prev_token = seed_token;
    // The hidden half fed to the drafter: round-0 = the target's last-committed
    // normed hidden (sp.seed_hidden), subsequent steps = the previous step's
    // post_proj. We keep the live hidden in `sp.draft_hidden` and point at it.
    gpu.hip
        .memcpy_dtod_at(&sp.draft_hidden.buf, 0, &sp.seed_hidden.buf, 0, dim * 4)
        .map_err(|e| format!("gemma4-spec: seed→draft_hidden copy: {e:?}"))?;
    for _j in 0..n {
        let out = drafter_step(
            gpu,
            dw,
            dcfg,
            ds,
            target_w,
            target_state,
            target_cfg,
            prev_token,
            &sp.draft_hidden,
            query_pos,
        )?;
        drafts.push(out.argmax);
        prev_token = out.argmax;
        // Feed this step's post_proj as the next step's hidden half.
        gpu.hip
            .memcpy_htod(&sp.draft_hidden.buf, unsafe {
                std::slice::from_raw_parts(
                    out.post_proj_hidden.as_ptr() as *const u8,
                    out.post_proj_hidden.len() * 4,
                )
            })
            .map_err(|e| format!("gemma4-spec: draft post_proj htod: {e:?}"))?;
    }

    // ── 2. Verify. block = [seed_token, d_0..d_{N-1}] at start_pos = L-1. ──
    // forward_batch_spec writes the seed's KV at L-1 (re-write, identical) and
    // each draft d_j at L+j, and returns per-position argmax + per-position
    // post-norm hidden. Both KV caches are written for [L-1, L+N).
    let mut block: Vec<u32> = Vec::with_capacity(n + 1);
    block.push(seed_token);
    block.extend_from_slice(&drafts);
    let block_len = block.len(); // N+1

    let hidden_out = sp.verify_hidden.sub_offset(0, block_len * dim);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(block_len);
    let _last_logits = forward_batch_spec(
        target_cfg,
        target_w,
        target_state,
        gpu,
        &block,
        query_pos, // start_pos = L-1
        Some(&hidden_out),
        Some(&mut argmax_per_pos),
    )?;
    debug_assert_eq!(argmax_per_pos.len(), block_len);

    // ── 3. Greedy accept. ──
    // argmax_per_pos[i] = target's prediction AFTER block position i.
    //   accept d_i (= block[i+1]) iff argmax_per_pos[i] == block[i+1].
    //   bonus = argmax_per_pos[accept_len].
    let mut accept_len = 0usize;
    for i in 0..n {
        if argmax_per_pos[i] == block[i + 1] {
            accept_len += 1;
        } else {
            break;
        }
    }
    let bonus = argmax_per_pos[accept_len];

    // committed = block[1..1+accept_len] (accepted drafts) ++ [bonus].
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 1);
    committed.extend_from_slice(&block[1..1 + accept_len]);
    committed.push(bonus);

    // ── 4. Commit. ──
    // Accepted drafts occupy positions L..L+accept_len-1 (KV already written by
    // verify). The bonus is the NEW token at L+accept_len — its KV is NOT yet
    // written (it wasn't in the block); it becomes the next round's seed and is
    // (re)written at the top of the next verify. Rejected drafts at
    // L+accept_len..L+N-1 keep stale KV, overwritten next round.
    target_state.n_tokens = l + accept_len + 1;

    // Next seed hidden = the verify row whose argmax produced the bonus = row
    // `accept_len` of the per-position post-norm hidden buffer.
    let bonus_row = sp.verify_hidden.sub_offset(accept_len * dim, dim);
    gpu.hip
        .memcpy_dtod_at(&sp.seed_hidden.buf, 0, &bonus_row.buf, 0, dim * 4)
        .map_err(|e| format!("gemma4-spec: next seed_hidden capture: {e:?}"))?;

    Ok(SpecStepOut {
        committed,
        accept_len,
        next_seed_token: bonus,
        drafts,
    })
}
#[cfg(test)]
mod owner_tests {
    use super::*;
    use crate::config::{LayerType, RopeType};

    #[test]
    #[ignore = "requires an AMD GPU"]
    fn eager_spec_scratch_owner_bytes_includes_all_verify_buffers() {
        static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        let cfg = Gemma4Config {
            dim: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 1,
            sliding_head_dim: 4,
            sliding_n_kv_heads: 1,
            sliding_rope_theta: 10_000.0,
            sliding_rope_type: RopeType::Default,
            sliding_window: 128,
            full_head_dim: 4,
            full_n_kv_heads: 1,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            hidden_dim: 8,
            use_double_wide_mlp: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            embed_scale: 2.0,
            max_position_embeddings: 128,
            layer_types: vec![LayerType::Sliding],
            norm_plus_one: false,
        };
        let scratch = Gemma4SpecScratch::new(&mut gpu, &cfg, 3).expect("tiny spec scratch");
        assert!(scratch.owner_bytes() > 3 * cfg.dim * std::mem::size_of::<f32>());
        scratch.free(&mut gpu);
        gpu.drain_pool();
    }
}
