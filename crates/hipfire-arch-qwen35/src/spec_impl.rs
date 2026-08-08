// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 implementations of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! Provides `impl SpecTarget for ModelSlot` — the borrowed-verifier hook the
//! daemon's spec loop hands to a `Speculator`, plus [`Qwen35SpecScratch`], the
//! concrete arch-specific verify scratch a model-free speculator owns behind
//! `Box<dyn SpecScratch>`. The verify mechanics (batched forward + per-position
//! lm_head/argmax via `verify_dflash_block`, the DeltaNet snapshot/rewind
//! including the default-on Q8 error-feedback residual folded into
//! [`DeltaNetSnapshot`], and the full-accept-skip / partial-replay state fixup)
//! all live here so the speculator stays 100% arch-agnostic. The
//! `DflashSpeculator` impl itself lives in the sibling [`crate::dflash_spec`]
//! module (alongside `DflashState`, which it owns).

use crate::qwen35;

use crate::speculative::{
    apply_topp_trunc, download_hidden_block, sample_categorical,
    scatter_hidden_block_to_interleaved, verify_dflash_block, xorshift_next_unit, DeltaNetSnapshot,
    HiddenStateRingBuffer, ModelSlot, VerifyScratch,
};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Single-pass argmax over a logit row.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0
}

/// Qwen3.5 target-verify scratch for a model-free speculator. Owns the
/// per-position lm_head/argmax buffers (`VerifyScratch`), the pre-verify
/// recurrent+EF snapshot (`DeltaNetSnapshot`), and a `num_extract = 0` hidden
/// ring (zero buffers — it only satisfies `verify_dflash_block`'s required
/// `&mut` arg; nothing is written or read).
pub struct Qwen35SpecScratch {
    verify_scratch: VerifyScratch,
    hidden_rb: HiddenStateRingBuffer,
    target_snap: DeltaNetSnapshot,
}

impl SpecScratch for Qwen35SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let Qwen35SpecScratch {
            verify_scratch,
            hidden_rb,
            target_snap,
        } = *self;
        verify_scratch.free_gpu(gpu);
        // `HiddenStateRingBuffer` has no `free_gpu`; free its buffers directly
        // (both vecs empty here at num_extract=0 — no-op, robust if that changes).
        for t in hidden_rb.layer_bufs {
            let _ = gpu.free_tensor(t);
        }
        for t in hidden_rb.staging_bufs {
            let _ = gpu.free_tensor(t);
        }
        target_snap.free_gpu(gpu);
    }
}

impl SpecTarget for ModelSlot {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Reuse the canonical DeltaNet reset (zeroes s_matrices / s_scales /
        // conv_states / s_ef_residual, stream-aware) rather than re-inlining the
        // memset loop the daemon abort path currently hand-writes, then drop the
        // KV eviction offset so the next conversation rotates from absolute 0.
        self.dn_state
            .reset(gpu)
            .map_err(|e| format!("qwen35 reset_recurrent: {e}"))?;
        self.kv_cache.compact_offset = 0;
        Ok(())
    }

    fn retry_reset_eligible(&self) -> bool {
        // Covered by reset_core inventory: DeltaNet + EF residual, KV compact,
        // daemon graph invalidate + adaptive + host pos/conversation.
        true
    }
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(2);
        let dim = self.config.dim;
        let vocab = self.config.vocab_size;
        let hidden_k = dim.next_power_of_two();
        // max_n = block_size covers the largest verify block (b <= block_size).
        let verify_scratch = VerifyScratch::new(gpu, block_size, dim, vocab, hidden_k)
            .map_err(|e| format!("Qwen35SpecScratch VerifyScratch: {e}"))?;
        // num_extract = 0 (every non-DSpark path) ⇒ no hidden buffers; the forward's
        // hidden extraction is a no-op and the ring is never read (byte-identical to
        // the pre-DSpark behaviour). num_extract > 0 (a DSpark drafter configured
        // `dspark_extract_layers`) ⇒ the ring captures the per-position residual
        // hidden at those layers during `verify_block_capture_gpu`.
        let num_extract = self.dspark_extract_layers.len();
        let mut hidden_rb = HiddenStateRingBuffer::new(
            gpu,
            self.config.n_layers,
            num_extract,
            dim,
            self.ctx_capacity(),
            block_size,
        )
        .map_err(|e| format!("Qwen35SpecScratch HiddenStateRingBuffer: {e}"))?;
        // `HiddenStateRingBuffer::new` fills `extract_layers` with the evenly-spaced
        // `dflash_extract_layer_ids` default; DSpark needs the EXACT sidecar layer
        // ids. Per-layer buffer sizing depends only on the COUNT (num_extract, which
        // already matches), so overwriting the ids after construction is safe.
        if num_extract > 0 {
            hidden_rb.extract_layers = self.dspark_extract_layers.clone();
        }
        let target_snap = DeltaNetSnapshot::new_for(gpu, &self.dn_state)
            .map_err(|e| format!("Qwen35SpecScratch DeltaNetSnapshot: {e}"))?;
        // EF residual is folded into DeltaNetSnapshot (empty when EF off).
        Ok(Box::new(Qwen35SpecScratch {
            verify_scratch,
            hidden_rb,
            target_snap,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)
                .map_err(|e| format!("qwen35 spec_advance reset: {e}"))?;
        }
        // Plain target advance, chunked at PREFILL_MAX_BATCH with abort checks
        // between chunks. No hidden extraction — only KV + recurrent state move.
        if reset {
            self.reset_state(gpu)
                .map_err(|e| format!("qwen35 spec_advance reset: {e}"))?;
        }
        let chunk_max = qwen35::PREFILL_MAX_BATCH;
        let mut off = 0usize;
        let mut pos = start_pos;
        while off < tokens.len() {
            if abort() {
                let _ = self.reset_state(gpu);
                return Ok(SpecAdvance::Aborted);
            }
            let end = (off + chunk_max).min(tokens.len());
            let completed = qwen35::forward_prefill_batch_abortable(
                gpu,
                &self.weights,
                &self.config,
                &tokens[off..end],
                pos,
                &mut self.kv_cache,
                &mut self.dn_state,
                &self.scratch,
                None,
                None,
                None,
                None,
                abort,
            )
            .map_err(|e| e.to_string())?;
            if !completed {
                return Ok(SpecAdvance::Aborted);
            }
            pos += end - off;
            off = end;
        }
        // Last-position logits (the per-token forward left last-token logits in
        // scratch.logits). Hand the host row through for temp>0 first-token draws.
        let logits = gpu
            .download_f32(&self.scratch.logits)
            .map_err(|e| e.to_string())?;
        let last_argmax = argmax(&logits);
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen35SpecScratch>()
            .ok_or("verify_block: scratch is not Qwen35SpecScratch")?;
        // CONTRACT: save the pre-verify recurrent state AND s_ef residual FIRST,
        // before the forward advances them, so commit_prefix can rewind.
        s.target_snap
            .save_from(&self.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        let out = verify_dflash_block(
            gpu,
            self,
            block,
            position,
            &mut s.hidden_rb,
            None,  // gdn_tape: rewind by replay in commit_prefix, no tape
            false, // greedy: GPU argmax, no full-logit D2H
            &s.verify_scratch,
        )
        .map_err(|e| e.to_string())?;
        Ok(out.argmax_per_pos)
    }

    fn verify_block_sampled(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        rng_state: &mut u64,
    ) -> Result<Vec<u32>, String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen35SpecScratch>()
            .ok_or("verify_block_sampled: scratch is not Qwen35SpecScratch")?;
        // SAME snapshot CONTRACT as verify_block: save recurrent + s_ef residual
        // BEFORE the forward advances them, so commit_prefix can rewind a partial.
        s.target_snap
            .save_from(&self.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        // Sampled verify. Run the verify forward with want_full_logits=FALSE: it
        // leaves the per-position logits in `verify_scratch.logits` (GPU) and
        // costs only a discarded GPU argmax — NOT the B×vocab logit D2H. Then do
        // the softmax+nucleus on the GPU (`softmax_temp_topp_batched_into_f32`,
        // mirroring the DFlash FAST_SAMPLE path) so the exp + the top_p/top_k
        // histogram run on-device; the host does only the cheap nucleus-trunc +
        // categorical draw. The first cut host-softmaxed (exp+sort) over 248K×B
        // per step — that was the ~22 t/s bottleneck, not the D2H. For the
        // point-mass n-gram draft, accept_greedy_prefix(draft, picks) on these
        // SAMPLED picks is exact temp-T speculation (commit == target sample).
        let _ = verify_dflash_block(
            gpu,
            self,
            block,
            position,
            &mut s.hidden_rb,
            None,  // gdn_tape: rewind by replay in commit_prefix, no tape
            false, // logits stay on-GPU in verify_scratch.logits; no full D2H
            &s.verify_scratch,
        )
        .map_err(|e| e.to_string())?;
        let vocab = self.config.vocab_size;
        let b = block.len();
        // top_p of 0.0 means "disabled" upstream → 1.0 (no nucleus). top_k is
        // folded into the GPU kernel's tau alongside top_p. min_p is routed to AR
        // by the dispatch, so it is never set on this path.
        let top_p_eff = if top_p > 0.0 { top_p.min(1.0) } else { 1.0 };
        let logits_batch = s.verify_scratch.logits.sub_offset(0, b * vocab);
        let probs_gpu = gpu
            .alloc_tensor(&[b * vocab], DType::F32)
            .map_err(|e| e.to_string())?;
        let tau_gpu = gpu
            .alloc_tensor(&[b], DType::F32)
            .map_err(|e| e.to_string())?;
        let z_gpu = gpu
            .alloc_tensor(&[b], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.softmax_temp_topp_batched_into_f32(
            &logits_batch,
            &probs_gpu,
            &tau_gpu,
            &z_gpu,
            vocab,
            b,
            temp,
            top_p_eff,
            top_k,
            0.0, // min_p: ngram min_p parity is the follow-up; off here
        )
        .map_err(|e| e.to_string())?;
        let host_probs = gpu.download_f32(&probs_gpu).map_err(|e| e.to_string())?;
        let tau = gpu.download_f32(&tau_gpu).map_err(|e| e.to_string())?;
        let z = gpu.download_f32(&z_gpu).map_err(|e| e.to_string())?;
        let _ = gpu.free_tensor(probs_gpu);
        let _ = gpu.free_tensor(tau_gpu);
        let _ = gpu.free_tensor(z_gpu);
        let mut picks = Vec::with_capacity(b);
        for i in 0..b {
            let mut row = host_probs[i * vocab..(i + 1) * vocab].to_vec();
            // Apply the SAME nucleus cut the GPU emitted tau/Z for (identity when
            // top_p>=1 and top_k==0), then draw categorically.
            apply_topp_trunc(&mut row, tau[i], z[i]);
            let u = xorshift_next_unit(rng_state);
            picks.push(sample_categorical(&row, u));
        }
        Ok(picks)
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Full accept: verify already left state at exactly position+block.len();
        // the bonus is the next seed (not yet fed). Nothing to undo.
        let draft_len = block.len() - 1;
        if accept_len >= draft_len {
            return Ok(());
        }
        // Partial: rewind recurrent + s_ef to pre-verify, then replay the
        // committed prefix with the SAME batched forward the verify used (GDN
        // numerics must match the accepted argmax). The stale FullAttention KV at
        // [position+accept+1 .. position+block.len()) is overwritten by the next
        // verify before it can be read as context.
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen35SpecScratch>()
            .ok_or("commit_prefix: scratch is not Qwen35SpecScratch")?;
        s.target_snap
            .restore_to(&mut self.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        qwen35::forward_prefill_batch(
            gpu,
            &self.weights,
            &self.config,
            &block[..accept_len + 1],
            position,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token
    }

    fn ctx_capacity(&self) -> usize {
        self.kv_cache.physical_cap
    }

    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        Some(&mut self.kv_cache)
    }

    // ── DSpark hidden-capture hooks ─────────────────────────────────────────
    //
    // These ride the SAME DeltaNet snapshot/rewind machinery `verify_block` uses;
    // the only addition is arming the `Qwen35SpecScratch` hidden ring (built with
    // `num_extract = dspark_extract_layers.len()` by `new_spec_scratch`) and moving
    // the captured rows GPU→GPU into the caller's buffer. Empty extract layers ⇒
    // the ring is a no-op and these are never reached (the drafter routes here only
    // after `set_dflash_extract_layers`).

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        if self.dspark_extract_layers.is_empty() {
            None
        } else {
            Some(&self.dspark_extract_layers)
        }
    }

    fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        self.dspark_extract_layers = layers;
    }

    /// Greedy verify + GPU-resident hidden capture. Mirrors [`verify_block`]
    /// exactly (snapshot recurrent + s_ef, then `verify_dflash_block`), then
    /// scatters the per-position extract-layer hidden the verify just wrote into
    /// the scratch ring straight into the caller-owned `hidden_gpu` (GPU→GPU).
    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen35SpecScratch>()
            .ok_or("verify_block_capture_gpu: scratch is not Qwen35SpecScratch")?;
        // SAME snapshot CONTRACT as verify_block: save the pre-verify recurrent
        // state AND s_ef residual FIRST, so commit_prefix can rewind a partial.
        s.target_snap
            .save_from(&self.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        let out = verify_dflash_block(
            gpu,
            self,
            block,
            position,
            &mut s.hidden_rb,
            None,  // gdn_tape: rewind by replay in commit_prefix, no tape
            false, // greedy: GPU argmax, no full-logit D2H
            &s.verify_scratch,
        )
        .map_err(|e| e.to_string())?;
        // The verify forward advanced the ring head by block.len(); the most recent
        // block.len() rows are exactly this window's captures. `scatter_...` reads
        // the last `block.len()` rows relative to the current head (so it is correct
        // whether the freshly-built ring's head started at 0 or accumulated), into
        // the caller's interleaved `[block.len() × num_extract × dim]` buffer.
        let captured = !s.hidden_rb.extract_layers.is_empty();
        if captured {
            scatter_hidden_block_to_interleaved(
                gpu,
                &s.hidden_rb,
                hidden_gpu,
                0,
                block.len(),
                block.len(),
                usize::MAX, // DSpark hidden dst is not a windowed draft ring
            )
            .map_err(|e| e.to_string())?;
        }
        Ok((out.argmax_per_pos, captured))
    }

    /// Sampled (temp>0) twin of [`verify_block_capture_gpu`]. Mirrors
    /// [`verify_block_sampled`]'s draw path, then applies the identical hidden
    /// scatter. Returns per-position SAMPLED tokens + whether hidden was captured.
    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        // qwen35 DFlash's sampled verify runs the per-position draw path, which
        // does not implement the CACTUS boost — ignored here (deepseek4-only).
        _cactus_delta: f32,
        rng_state: &mut u64,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Qwen35SpecScratch>()
            .ok_or("verify_block_sampled_capture_gpu: scratch is not Qwen35SpecScratch")?;
        // SAME snapshot CONTRACT as verify_block_sampled.
        s.target_snap
            .save_from(&self.dn_state, gpu)
            .map_err(|e| e.to_string())?;
        // Sampled verify: leave the per-position logits on-GPU in
        // verify_scratch.logits (want_full_logits=false), softmax+nucleus on-device,
        // then draw categorically on the host — mirroring verify_block_sampled.
        let _ = verify_dflash_block(
            gpu,
            self,
            block,
            position,
            &mut s.hidden_rb,
            None,
            false,
            &s.verify_scratch,
        )
        .map_err(|e| e.to_string())?;
        let vocab = self.config.vocab_size;
        let b = block.len();
        let top_p_eff = if top_p > 0.0 { top_p.min(1.0) } else { 1.0 };
        let logits_batch = s.verify_scratch.logits.sub_offset(0, b * vocab);
        let probs_gpu = gpu
            .alloc_tensor(&[b * vocab], DType::F32)
            .map_err(|e| e.to_string())?;
        let tau_gpu = gpu
            .alloc_tensor(&[b], DType::F32)
            .map_err(|e| e.to_string())?;
        let z_gpu = gpu
            .alloc_tensor(&[b], DType::F32)
            .map_err(|e| e.to_string())?;
        gpu.softmax_temp_topp_batched_into_f32(
            &logits_batch,
            &probs_gpu,
            &tau_gpu,
            &z_gpu,
            vocab,
            b,
            temp,
            top_p_eff,
            top_k,
            0.0, // min_p: routed to AR by dispatch, never set on this path
        )
        .map_err(|e| e.to_string())?;
        let host_probs = gpu.download_f32(&probs_gpu).map_err(|e| e.to_string())?;
        let tau = gpu.download_f32(&tau_gpu).map_err(|e| e.to_string())?;
        let z = gpu.download_f32(&z_gpu).map_err(|e| e.to_string())?;
        let _ = gpu.free_tensor(probs_gpu);
        let _ = gpu.free_tensor(tau_gpu);
        let _ = gpu.free_tensor(z_gpu);
        let mut picks = Vec::with_capacity(b);
        for i in 0..b {
            let mut row = host_probs[i * vocab..(i + 1) * vocab].to_vec();
            apply_topp_trunc(&mut row, tau[i], z[i]);
            let u = xorshift_next_unit(rng_state);
            picks.push(sample_categorical(&row, u));
        }
        // Same GPU→GPU hidden scatter as the greedy variant.
        let captured = !s.hidden_rb.extract_layers.is_empty();
        if captured {
            scatter_hidden_block_to_interleaved(
                gpu,
                &s.hidden_rb,
                hidden_gpu,
                0,
                block.len(),
                block.len(),
                usize::MAX, // DSpark hidden dst is not a windowed draft ring
            )
            .map_err(|e| e.to_string())?;
        }
        Ok((picks, captured))
    }

    /// DSpark bootstrap: capture the seed's residual hidden at `layers` WITHOUT
    /// permanently advancing the RECURRENT DeltaNet state (the #462 crux). The
    /// seed is committed later by `verify_block` + `commit_prefix` (it is
    /// `verify_tokens[0]`), not by this capture — so snapshot recurrent + s_ef,
    /// run a 1-token capture-armed forward, download the row, then restore. The KV
    /// write at `position` is harmless (verify rewrites the same slot with the same
    /// seed); only the recurrent state needs the snapshot/restore.
    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        // Remember the extract layers so the per-window `new_spec_scratch` (called
        // AFTER this bootstrap in the initial window, and every steady-state window)
        // builds a ring that captures at exactly these ids.
        self.dspark_extract_layers = layers.to_vec();
        let dim = self.config.dim;
        let num_extract = layers.len();

        // Snapshot recurrent + default-on s_ef residual BEFORE the forward advances
        // them irreversibly (EF lives inside DeltaNetSnapshot).
        let mut snap = DeltaNetSnapshot::new_for(gpu, &self.dn_state)
            .map_err(|e| format!("capture_seed_main_hidden snapshot: {e:?}"))?;
        snap.save_from(&self.dn_state, gpu)
            .map_err(|e| format!("capture_seed_main_hidden save recurrent: {e:?}"))?;

        // Temp 1-slot ring capturing at EXACTLY `layers` (override the evenly-spaced
        // default `new` computes; buffer sizing depends only on the count).
        let mut ring = HiddenStateRingBuffer::new(
            gpu,
            self.config.n_layers,
            num_extract,
            dim,
            1, // max_positions: the single seed slot
            1, // max_batch: single-token forward
        )
        .map_err(|e| format!("capture_seed_main_hidden ring: {e:?}"))?;
        ring.extract_layers = layers.to_vec();

        // 1-token capture-armed forward at `position` (hidden_rb Some arms the
        // per-extract-layer hidden capture; same proven call as
        // `seed_target_hidden_from_prompt`).
        let host_result = qwen35::forward_prefill_batch(
            gpu,
            &self.weights,
            &self.config,
            &[seed],
            position,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
            Some(&mut ring),
            None,
            None,
            None,
        )
        .map_err(|e| format!("capture_seed_main_hidden forward: {e:?}"))
        .and_then(|()| {
            // b = 1 ⇒ yields [1 × num_extract × dim] = the concatenated main_hidden.
            download_hidden_block(gpu, &ring, 1)
                .map_err(|e| format!("capture_seed_main_hidden download: {e:?}"))
        });

        // Restore recurrent + s_ef to pre-capture (undo the irreversible advance),
        // regardless of whether the forward/download succeeded, before freeing.
        let restore_result = snap
            .restore_to(&mut self.dn_state, gpu)
            .map_err(|e| format!("capture_seed_main_hidden restore recurrent: {e:?}"));

        // Free the temp snapshot and ring buffers on every path.
        snap.free_gpu(gpu);
        for t in ring.layer_bufs {
            let _ = gpu.free_tensor(t);
        }
        for t in ring.staging_bufs {
            let _ = gpu.free_tensor(t);
        }

        restore_result?;
        host_result
    }
}
