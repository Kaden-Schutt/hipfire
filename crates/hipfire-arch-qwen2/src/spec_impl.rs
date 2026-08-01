// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen2-family implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for Qwen2Bundle` lets the model-free `NgramSpeculator` drive
//! a Qwen2 target (e.g. VibeThinker-3B, arch_id=7) with no arch knowledge. Pure
//! GQA attention, no recurrent state, so `commit_prefix` is a no-op (the accepted-
//! prefix KV the verify wrote is already correct; the rejected tail is overwritten
//! by the next verify). Unlike llama/qwen35, Qwen2 keeps its KV in its own
//! `Qwen2State` (not the shared `llama::KvCache`), so [`SpecTarget::kv_cache_mut`]
//! stays at its `None` default — arch_id=7 has no FlashCASK eviction.
//!
//! VERIFY IS BLOCK-PARALLEL: [`SpecTarget::verify_block`] runs the whole
//! candidate block through one batched layer loop
//! ([`qwen2::forward_verify_block_batched`]) instead of a `forward_step`-per-token
//! sequential loop. The enabling piece is a new F32-KV batched-decode-with-history
//! attention kernel (`attention_decode_batched_history`): the block's post-RoPE
//! K/V are written into the F32 cache at absolute positions first, then each block
//! query row attends to the FULL prior history `[0..position+i]` (prompt +
//! accepted prefix + causal-within-block) in one launch. This sidesteps the kernel
//! gap that blocked the naive batched verify — qwen2 keeps F32 KV, and the older
//! batched attentions in `rdna-compute` were either quantized-KV (q8/asym/fwht) or
//! intra-batch only (`attention_causal_batched`, blind to prior KV — the source of
//! the token-264 attractor in commit 24a5804f's naive attempt).
//!
//! The batched verify is BYTE-IDENTICAL to the sequential one (same per-slot
//! argmax vector; see `examples/verify_block_parity.rs`), so it carries the same
//! coherence. The win scales with context: the block-parallel attention saves
//! little at short prompts (GEMMs dominate) but is +~46% decode at ~900-token
//! contexts where attention's share grows. n-gram remains opt-in
//! (`HIPFIRE_NGRAM_DRAFT=1`); force the legacy sequential verify with
//! `HIPFIRE_QWEN2_VERIFY_SEQ=1` (the byte-identical reference).
//!
//! NOTE: spec-decode on qwen2 does not yet beat plain AR on every workload — the
//! remaining ceiling is the MQ4G256/HFQ4 projection GEMMs lacking a batched
//! kernel, so each B-wide verify pays ~B× the GEMM cost via the per-row
//! `weight_gemm` fallback. That batched-GEMM work is a separate follow-up; this
//! change removes the *attention* serialization, which was the part a per-arch
//! kernel could address.

use crate::carrier::Qwen2Bundle;
use crate::qwen2;
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

/// `HIPFIRE_QWEN2_VERIFY_SEQ=1` forces the legacy sequential verify (one
/// `forward_step` per block token) instead of the block-parallel batched path.
/// Used as the byte-identical reference in the correctness test.
fn verify_sequential() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_QWEN2_VERIFY_SEQ").as_deref() == Ok("1")
    })
}

/// Qwen2 verify scratch: nothing persistent. The verify reuses the bundle's own
/// `Qwen2State` scratch (dense attention → no recurrent snapshot to carry
/// between windows).
pub struct Qwen2SpecScratch;

impl SpecScratch for Qwen2SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for Qwen2Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Pure attention: no recurrent state to zero. Rewind the KV position
        // cursor so the next prefill writes from slot 0 (O(1); KV is overwritten
        // in place). Mirrors the daemon's arch_id=7 reset handler.
        self.state.reset();
        Ok(())
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(Qwen2SpecScratch))
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
        // Pure attention: "reset" rewinds the position cursor; the per-token
        // prefill then overwrites KV at the absolute positions it writes.
        if reset {
            self.state.reset();
        }
        self.state.next_pos = start_pos;
        for &tok in tokens {
            if abort() {
                self.state.reset();
                return Ok(SpecAdvance::Aborted);
            }
            qwen2::forward_step(gpu, &self.weights, &self.config, &mut self.state, tok)
                .map_err(|e| format!("{e:?}"))?;
        }
        // forward_step leaves the last position's logits in state.logits.
        // GPU-only argmax — no host row materialised here.
        let last_argmax = gpu
            .argmax_f32(&self.state.logits, self.config.vocab_size)
            .map_err(|e| format!("{e:?}"))?;
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: None,
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // Block-parallel verify (default): one batched layer loop writes the
        // block's K/V to the F32 cache at absolute positions [position..) then
        // runs `attention_decode_batched_history` so each row attends to the FULL
        // history [0..position+i] — byte-equivalent to the sequential per-token
        // flash decode, but in one launch chain instead of B sequential decodes.
        //
        // `HIPFIRE_QWEN2_VERIFY_SEQ=1` forces the legacy sequential loop (the
        // byte-identical reference the batched path is validated against).
        if !verify_sequential() {
            return qwen2::forward_verify_block_batched(
                gpu,
                &self.weights,
                &self.config,
                &mut self.state,
                block,
                position,
            )
            .map_err(|e| format!("{e:?}"));
        }

        // Sequential verify: `forward_step(block[i])` predicts the token AFTER
        // block[i] (with block[0..i] already in the KV cache), which is exactly
        // `argmax[i]` — the verifier's pick at slot i. Each step's flash-decode
        // attention reads the FULL KV history (prompt + accepted prefix). Position
        // the cursor at `position` first so the writes land at the right absolute
        // slots (overwriting any rejected-tail KV from a prior window).
        self.state.next_pos = position;
        let mut out = Vec::with_capacity(block.len());
        for &tok in block {
            qwen2::forward_step(gpu, &self.weights, &self.config, &mut self.state, tok)
                .map_err(|e| format!("{e:?}"))?;
            out.push(
                gpu.argmax_f32(&self.state.logits, self.config.vocab_size)
                    .map_err(|e| format!("{e:?}"))?,
            );
        }
        Ok(out)
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // rejected tail is overwritten by the next verify. Nothing to rewind.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }

    // kv_cache_mut: defaulted to `None` — Qwen2State is not a `llama::KvCache`,
    // and arch_id=7 has no eviction (the daemon's eviction sites are
    // `if let Some(ev)`-gated, so this is never reached).
}
