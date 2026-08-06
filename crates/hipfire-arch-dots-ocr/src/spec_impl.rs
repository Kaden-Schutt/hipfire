// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dots.ocr implementation of the arch-generic speculative-decode seam
//! (`hipfire_runtime::spec`).
//!
//! `impl SpecTarget for DotsOcrBundle` lets the model-free `NgramSpeculator`
//! drive the dots.ocr text decoder (a plain Qwen2-1.5B) without any
//! arch knowledge. The vision tower is stateless one-shot (run once per
//! image during prefill); per-decode-step state is pure Qwen2 attention —
//! no recurrent state to snapshot.
//!
//! # Why n-gram is a good fit for dots.ocr
//!
//! dots.ocr output is highly structured layout-JSON: `<td>`, `"category":`,
//! bracket/brace patterns, repeated field names. These exact repeated token
//! sequences are what n-gram drafting specialises in. Acceptance rates on
//! structured-output benchmarks consistently exceed prose.
//!
//! # Verify strategy
//!
//! Identical to arch_id=7 (`hipfire_arch_qwen2::spec_impl`): block-parallel
//! via `qwen2::forward_verify_block_batched` by default, with
//! `HIPFIRE_QWEN2_VERIFY_SEQ=1` forcing the sequential reference. The
//! enabling piece (F32-KV batched-decode-with-history kernel) is part of
//! the qwen2 crate and works unchanged here because dots.ocr's decoder IS
//! Qwen2.
//!
//! # VL-routing note (for the orchestrator)
//!
//! dots.ocr's production entry point is `generate_vl_dots_ocr` in the
//! daemon (daemon.rs:11863), which is an image-conditioned VL path that
//! runs `vision_forward`, splices visual tokens, and performs CPU-side
//! greedy sampling. The `SpecTarget` impl here covers ONLY the Qwen2
//! text-decode step. Routing `generate_vl_dots_ocr` through the n-gram
//! spec loop is an orchestrator decision that requires the daemon to call
//! `generate_spec` (or a VL-aware variant) instead of the current bespoke
//! loop. That wiring is a separate follow-up; this commit supplies the
//! arch-generic seam the orchestrator needs.
//!
//! # commit_prefix / kv_cache_mut
//!
//! Pure attention: verify's accepted-prefix KV is already correct; the
//! rejected tail is overwritten by the next verify. `commit_prefix` is a
//! no-op. `kv_cache_mut` stays at its `None` default — `Qwen2State` is
//! not a `llama::KvCache`, so arch_id=8 has no FlashCASK eviction
//! (the daemon's eviction sites are `if let Some(ev)`-gated).

use crate::dots_ocr::{DotsOcrConfig, DotsOcrWeights};
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen2::qwen2::Qwen2State;
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::Gpu;

// ─── DotsOcrBundle ───────────────────────────────────────────────────────────

/// Bundled dots.ocr text-decoder state for the spec-decode seam.
///
/// Holds the three fields the daemon stores for arch_id=8:
/// `DotsOcrConfig`, `DotsOcrWeights`, and `Qwen2State`. The vision
/// tower (`DotsVisionWeights`) is NOT included — it is one-shot per
/// image and is freed after prefill; by the time the spec-decode loop
/// runs, only the text-decoder state is live.
pub struct DotsOcrBundle {
    pub config: DotsOcrConfig,
    pub weights: DotsOcrWeights,
    pub state: Qwen2State,
}

// ─── DotsOcrSpecScratch ───────────────────────────────────────────────────────

/// dots.ocr verify scratch: nothing persistent. Dense attention — no
/// recurrent snapshot to carry between windows. Mirrors `Qwen2SpecScratch`.
pub struct DotsOcrSpecScratch;

impl SpecScratch for DotsOcrSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

// ─── SpecTarget impl ─────────────────────────────────────────────────────────

impl SpecTarget for DotsOcrBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Pure attention: no recurrent state to zero. Rewind the KV
        // position cursor so the next prefill writes from slot 0.
        // Mirrors the Qwen2 reset path (qwen2 spec_impl.rs).
        self.state.reset();
        Ok(())
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        Ok(Box::new(DotsOcrSpecScratch))
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
                .map_err(|e| format!("dots-ocr spec_advance reset: {e}"))?;
        }
        self.state.next_pos = start_pos;
        for &tok in tokens {
            if abort() {
                return Ok(SpecAdvance::Aborted);
            }
            qwen2::forward_step(
                gpu,
                &self.weights.text,
                &self.config.text,
                &mut self.state,
                tok,
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        let last_argmax = gpu
            .argmax_f32(&self.state.logits, self.config.text.vocab_size)
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
        // Block-parallel verify via the F32-KV batched-decode-with-history
        // kernel in the qwen2 crate. dots.ocr's decoder IS Qwen2 so this
        // reuses the same path byte-for-byte with no wrapper overhead.
        //
        // `HIPFIRE_QWEN2_VERIFY_SEQ=1` falls through to the sequential
        // reference in `forward_verify_block_batched` — that env flag is
        // checked inside the qwen2 function, not here.
        qwen2::forward_verify_block_batched(
            gpu,
            &self.weights.text,
            &self.config.text,
            &mut self.state,
            block,
            position,
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct;
        // the rejected tail is overwritten by the next verify. No rewind
        // needed. Mirrors Qwen2Bundle::commit_prefix.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.text.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq
    }

    // kv_cache_mut: defaulted to `None` — Qwen2State is not a
    // `llama::KvCache`; arch_id=8 has no FlashCASK eviction.
}
