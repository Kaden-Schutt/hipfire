// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2.5-MoE (arch_id=11) implementation of the arch-generic
//! speculative-decode seam (`hipfire_runtime::spec`).
//!
//! LFM2.5-MoE is a HYBRID arch: ~18 short-conv (LIV) layers carry a recurrent
//! `[hidden,(K-1)]` rolling conv-state ring buffer each, interleaved with GQA
//! attention layers backed by the shared `llama::KvCache`. A speculative verify
//! over a B-token block advances BOTH: the conv-states roll forward by B, and
//! the attention KV is written at absolute positions `[position..position+B)`.
//!
//! So this is the *recurrent* shape of the seam (template: qwen35's DeltaNet
//! `ModelSlot` impl), NOT the stateless one (qwen2). The hard part is the
//! conv-state: on a PARTIAL accept we must roll the conv-state back to the
//! accepted prefix. We do this exactly like qwen35 snapshots its DeltaNet S/conv
//! state — by copying every conv-state ring buffer device-to-device into a
//! parallel set of snapshot buffers in [`verify_block`] BEFORE the forward
//! advances them, then restoring + replaying the accepted prefix in
//! [`commit_prefix`].
//!
//! The attention KV needs no explicit rewind: `decode_step` takes an absolute
//! `position` and writes KV there, so the accepted-prefix KV the verify wrote is
//! already correct and the rejected-tail KV is overwritten by the replay (and by
//! the next verify window). Only the conv-state — which advances *implicitly* as
//! a side effect of each conv layer and cannot be re-derived from position — must
//! be snapshotted.
//!
//! VERIFY IS SEQUENTIAL per-token (`decode_step` per block token). A batched
//! conv+attention verify kernel does not exist for this arch; the sequential
//! decode is the correct, coherence-bearing baseline (mirrors qwen2's legacy
//! sequential verify and the LFM2 decode hot path itself).

use crate::config::Lfm2MoeConfig;
use crate::forward::decode_step;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hipfire_runtime::gpu_cleanup::{BundleTeardown, GpuCleanupFailure};
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Single-pass argmax over a host logit row.
// NOTE: deliberately NOT `hipfire_runtime::llama::argmax`, and deliberately
// duplicated with the other arch crate that carries this function.
//
// The runtime's copy adds an `is_finite()` guard (its "O2b-2 finite guard")
// because it doubles as the degenerate fallback for `sample_top_p` /
// `sample_full_dist`, where a `+Inf` logit must not beat the real finite max.
//
// This copy is on the speculative-decode path and must agree bit-for-bit with
// the GPU kernel it is checked against — `kernels/src/argmax.hip:13` is a bare
// `if (data[i] > lmax)`, which *does* select `+Inf`. Adding the finite guard
// here would make draft and target disagree on `+Inf` logits and produce
// spurious spec-decode rejections.
//
// Both behaviours are correct for their own caller. Do not unify them.
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

/// Owned LFM2.5-MoE GPU bundle: the local type the arch-generic `SpecTarget`
/// seam is implemented for. Mirrors `hipfire_arch_qwen2::carrier::Qwen2Bundle`
/// (config + weights + state owned together in the arch crate) so the model-free
/// speculator can drive an LFM2 target with no arch knowledge.
///
/// The orphan rule requires `impl SpecTarget` to target a type local to THIS
/// crate, so the spec seam binds to this bundle rather than the loader-side
/// `hipfire_loader::Lfm2MoeBundle`. Integration: the loader's `Lfm2MoeCarrier`
/// should construct this type (or the loader bundle should re-export / wrap it)
/// so the spec-target guard can borrow it as `&mut dyn SpecTarget`.
pub struct Lfm2MoeBundle {
    pub config: Lfm2MoeConfig,
    pub weights: Lfm2MoeWeights,
    pub state: Lfm2MoeState,
    pub eos_tok: u32,
    /// Continuous-decode batch state (arch 11 dense, single-GPU). `Some` only
    /// when continuous batch has been staged via `batch_staging::stage_continuous_batch`;
    /// `None` is the common AR path. Lives in the bundle (not `LoadedModel`)
    /// so `LoadedModel` can become arch-free. Previously
    /// `LoadedModel.lfm2_decode_batch`.
    pub lfm2_decode_batch: Option<crate::batch::Lfm2DecodeBatchState>,
}

/// LFM2.5-MoE verify scratch: the pre-verify conv-state snapshot.
///
/// Owns one F32 GPU buffer per conv layer, each sized to match the corresponding
/// `state.conv_states[i]` (`[hidden*(K-1)]`). [`verify_block`] copies the live
/// conv-states INTO these before the block forward advances them, so a partial
/// accept can restore them in [`commit_prefix`]. The attention KV needs no
/// snapshot (absolute-position writes; see module docs), so nothing else is
/// carried between windows.
pub struct Lfm2MoeSpecScratch {
    /// `conv_snap[i]` is the saved copy of `state.conv_states[i]`.
    conv_snap: Vec<GpuTensor>,
}

impl SpecScratch for Lfm2MoeSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // GpuTensor has no Drop — free every snapshot buffer explicitly or the
        // device memory orphans (see qwen35 Qwen35SpecScratch::free).
        let Lfm2MoeSpecScratch { conv_snap } = *self;
        for t in conv_snap {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl Lfm2MoeBundle {
    /// Copy the live conv-states INTO the snapshot buffers (pre-verify).
    fn save_conv(&self, snap: &[GpuTensor], gpu: &mut Gpu) -> Result<(), String> {
        for (dst, src) in snap.iter().zip(self.state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| format!("lfm2moe: save conv snapshot: {e:?}"))?;
        }
        Ok(())
    }

    /// Copy the snapshot buffers back INTO the live conv-states (pre-replay).
    fn restore_conv(&self, snap: &[GpuTensor], gpu: &mut Gpu) -> Result<(), String> {
        for (src, dst) in snap.iter().zip(self.state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, &src.buf, src.buf.size())
                .map_err(|e| format!("lfm2moe: restore conv snapshot: {e:?}"))?;
        }
        Ok(())
    }
}

impl BundleTeardown for Lfm2MoeBundle {
    /// Checked teardown of a fully constructed bundle: every GPU owner in
    /// weights and state is freed with a checked free; on failure the exact
    /// unfreed owners are retained in the returned [`GpuCleanupFailure`] for
    /// retry — no best-effort free as a correctness mechanism.
    fn free_checked(self, gpu: &mut Gpu) -> Result<(), GpuCleanupFailure> {
        let Lfm2MoeBundle {
            config: _,
            weights,
            state,
            eos_tok: _,
            lfm2_decode_batch,
        } = self;
        let mut cf = GpuCleanupFailure::empty();
        // The batch state owns the batched KV cache, conv rings, and every
        // batched scratch buffer — all GpuTensors with no Drop. `free_gpu` is
        // infallible, so free it explicitly here (mirrors arch_model.rs and
        // qwen35's qwen35_decode_batch teardown); ignoring it would orphan
        // device memory.
        if let Some(batch) = lfm2_decode_batch {
            batch.free_gpu(gpu);
        }
        if let Err(f) = weights.free_checked(gpu) {
            cf.merge(f);
        }
        if let Err(f) = state.free_checked(gpu) {
            cf.merge(f);
        }
        if cf.is_empty() {
            Ok(())
        } else {
            Err(cf)
        }
    }
}

impl SpecTarget for Lfm2MoeBundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Zero every conv-state ring buffer + reset the token count (the daemon's
        // arch_id=11 reset path). KV is overwritten by absolute-position writes,
        // so there is no separate KV cursor to rewind here; drop the eviction
        // offset for symmetry with qwen35's reset.
        self.state
            .reset(gpu)
            .map_err(|e| format!("lfm2moe reset_recurrent: {e}"))?;
        self.state.kv.compact_offset = 0;
        Ok(())
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // One snapshot buffer per conv-state, sized to match it exactly.
        let mut conv_snap = Vec::with_capacity(self.state.conv_states.len());
        for cs in &self.state.conv_states {
            conv_snap.push(
                gpu.alloc_tensor(&cs.shape, DType::F32)
                    .map_err(|e| format!("lfm2moe: alloc conv snapshot: {e:?}"))?,
            );
        }
        Ok(Box::new(Lfm2MoeSpecScratch { conv_snap }))
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
                .map_err(|e| format!("lfm2moe spec_advance reset: {e}"))?;
        }
        // Plain target advance: feed each token at its absolute position. The
        // model owner performs the sole request reset before this hook runs.
        let mut last_logits: Vec<f32> = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            if abort() {
                return Ok(SpecAdvance::Aborted);
            }
            let pos = (start_pos + i) as u32;
            last_logits = decode_step(&self.config, &self.weights, &mut self.state, gpu, tok, pos)?;
        }
        Ok(SpecAdvance::Ready {
            last_argmax: argmax(&last_logits),
            last_logits: Some(last_logits),
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
            .downcast_mut::<Lfm2MoeSpecScratch>()
            .ok_or("verify_block: scratch is not Lfm2MoeSpecScratch")?;
        // CONTRACT: snapshot the recurrent conv-state FIRST, before the forward
        // advances it, so commit_prefix can rewind on a partial accept. (The
        // attention KV is written at absolute positions and needs no snapshot.)
        // `self`, `scratch`/`s`, and `gpu` are three disjoint objects — no
        // aliasing — so the snapshot read borrow coexists with `&mut self.state`.
        self.save_conv(&s.conv_snap, gpu)?;

        // Sequential per-token verify: decode_step(block[i]) predicts the token
        // AFTER block[i] (with block[0..i] already consumed into KV + conv-state),
        // which is exactly argmax[i] — the verifier's pick at slot i. Each step's
        // attention reads the full KV history; each conv layer rolls its state.
        let mut out = Vec::with_capacity(block.len());
        for (i, &tok) in block.iter().enumerate() {
            let pos = (position + i) as u32;
            let logits = decode_step(&self.config, &self.weights, &mut self.state, gpu, tok, pos)?;
            out.push(argmax(&logits));
        }
        Ok(out)
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Full accept: verify already left both the conv-state and the KV at
        // exactly position+block.len(); the bonus token is the next seed (not yet
        // fed). Nothing to undo.
        let draft_len = block.len() - 1;
        if accept_len >= draft_len {
            return Ok(());
        }
        // Partial accept: restore the pre-verify conv-state, then replay the
        // committed prefix block[..accept_len+1] with the SAME sequential decode
        // the verify used so the recurrent conv-state matches the accepted argmax.
        // The replay also re-writes the accepted-prefix KV at its absolute
        // positions; the stale rejected-tail KV at
        // [position+accept_len+1 .. position+block.len()) is overwritten by the
        // next verify window before it can be read as context.
        let s = scratch
            .as_any_mut()
            .downcast_mut::<Lfm2MoeSpecScratch>()
            .ok_or("commit_prefix: scratch is not Lfm2MoeSpecScratch")?;
        // `self`, `scratch`/`s`, and `gpu` are disjoint, so the snapshot read
        // borrow coexists with the conv-state write inside restore_conv. The
        // snapshot buffers stay owned by the scratch for the next window.
        self.restore_conv(&s.conv_snap, gpu)?;

        for (i, &tok) in block[..accept_len + 1].iter().enumerate() {
            let pos = (position + i) as u32;
            decode_step(&self.config, &self.weights, &mut self.state, gpu, tok, pos)?;
        }
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.eos_tok
    }

    fn ctx_capacity(&self) -> usize {
        self.state.kv.physical_cap
    }

    // kv_cache_mut: defaulted to `None`. Although LFM2 stores attention KV in the
    // shared `llama::KvCache`, FlashCASK eviction is UNSOUND on this hybrid arch:
    // evicting attention KV would desync the attention layers from the conv-state
    // layers (whose recurrent history cannot be evicted in lockstep). arch_id=11
    // therefore has no eviction; the daemon's eviction sites are
    // `if let Some(ev)`-gated so this is never reached.
}
