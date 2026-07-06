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
//! lm_head/argmax via `verify_dflash_block`, the DeltaNet snapshot/rewind incl.
//! the Q8 error-feedback residual the snapshot type omits, and the
//! full-accept-skip / partial-replay state fixup) all live here so the
//! speculator stays 100% arch-agnostic. The `DflashSpeculator` impl itself lives
//! in the sibling [`crate::dflash_spec`] module (alongside `DflashState`, which
//! it owns).

use crate::qwen35::{self, DeltaNetState};
use crate::speculative::{
    apply_topp_trunc, sample_categorical, verify_dflash_block, xorshift_next_unit,
    DeltaNetSnapshot, HiddenStateRingBuffer, ModelSlot, VerifyScratch,
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

/// Copy the live `s_ef_residual` into the backup (pre-verify).
fn save_s_ef(snap: &[GpuTensor], dn: &DeltaNetState, gpu: &mut Gpu) -> Result<(), String> {
    for (dst, src) in snap.iter().zip(dn.s_ef_residual.iter()) {
        gpu.hip
            .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Copy the backup back into the live `s_ef_residual` (post-verify, pre-replay).
fn restore_s_ef(snap: &[GpuTensor], dn: &DeltaNetState, gpu: &mut Gpu) -> Result<(), String> {
    for (src, dst) in snap.iter().zip(dn.s_ef_residual.iter()) {
        gpu.hip
            .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Qwen3.5 target-verify scratch for a model-free speculator. Owns the
/// per-position lm_head/argmax buffers (`VerifyScratch`), the pre-verify
/// recurrent snapshot (`DeltaNetSnapshot` + the `s_ef_residual` backup the
/// snapshot type omits), and a `num_extract = 0` hidden ring (zero buffers —
/// it only satisfies `verify_dflash_block`'s required `&mut` arg; nothing is
/// written or read).
pub struct Qwen35SpecScratch {
    verify_scratch: VerifyScratch,
    hidden_rb: HiddenStateRingBuffer,
    target_snap: DeltaNetSnapshot,
    s_ef_snap: Vec<GpuTensor>,
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
            s_ef_snap,
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
        for t in s_ef_snap {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl SpecTarget for ModelSlot {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) {
        // Reuse the canonical DeltaNet reset (zeroes s_matrices / s_scales /
        // conv_states / s_ef_residual, stream-aware) rather than re-inlining the
        // memset loop the daemon abort path currently hand-writes, then drop the
        // KV eviction offset so the next conversation rotates from absolute 0.
        self.dn_state.reset(gpu);
        self.kv_cache.compact_offset = 0;
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
        // num_extract = 0 ⇒ no hidden buffers; the forward's hidden extraction is
        // a no-op and the ring is never read.
        let hidden_rb = HiddenStateRingBuffer::new(
            gpu,
            self.config.n_layers,
            0,
            dim,
            self.ctx_capacity(),
            block_size,
        )
        .map_err(|e| format!("Qwen35SpecScratch HiddenStateRingBuffer: {e}"))?;
        let target_snap = DeltaNetSnapshot::new_for(gpu, &self.dn_state)
            .map_err(|e| format!("Qwen35SpecScratch DeltaNetSnapshot: {e}"))?;
        // F16 backups for s_ef_residual (empty when error-feedback is off).
        let mut s_ef_snap = Vec::with_capacity(self.dn_state.s_ef_residual.len());
        for t in &self.dn_state.s_ef_residual {
            s_ef_snap.push(
                gpu.alloc_tensor(&t.shape, DType::F16)
                    .map_err(|e| format!("Qwen35SpecScratch s_ef snapshot: {e}"))?,
            );
        }
        Ok(Box::new(Qwen35SpecScratch {
            verify_scratch,
            hidden_rb,
            target_snap,
            s_ef_snap,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        // Plain target advance, chunked at PREFILL_MAX_BATCH with abort checks
        // between chunks. No hidden extraction — only KV + recurrent state move.
        if reset {
            self.reset_state(gpu);
        }
        let chunk_max = qwen35::PREFILL_MAX_BATCH;
        let mut off = 0usize;
        let mut pos = start_pos;
        while off < tokens.len() {
            if abort() {
                self.reset_state(gpu);
                return Ok(SpecAdvance::Aborted);
            }
            let end = (off + chunk_max).min(tokens.len());
            qwen35::forward_prefill_batch(
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
            )
            .map_err(|e| e.to_string())?;
            pos += end - off;
            off = end;
        }
        // Last-position argmax (the per-token forward left last-token logits in
        // scratch.logits).
        let logits = gpu
            .download_f32(&self.scratch.logits)
            .map_err(|e| e.to_string())?;
        Ok(SpecAdvance::Ready {
            last_argmax: argmax(&logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
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
        save_s_ef(&s.s_ef_snap, &self.dn_state, gpu)?;
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
        save_s_ef(&s.s_ef_snap, &self.dn_state, gpu)?;
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
        let tau_gpu = gpu.alloc_tensor(&[b], DType::F32).map_err(|e| e.to_string())?;
        let z_gpu = gpu.alloc_tensor(&[b], DType::F32).map_err(|e| e.to_string())?;
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
        restore_s_ef(&s.s_ef_snap, &self.dn_state, gpu)?;
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
}
