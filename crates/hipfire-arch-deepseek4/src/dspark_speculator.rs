// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 `Deepseek4DsparkBody` — the arch-specific seam wiring the
//! deepseek4 3-stage MoE/MLA chain into the arch-agnostic
//! [`hipfire_runtime::dspark_core::DsparkDrafter`].
//!
//! ## Stage 3: multi-slot accepted-prefix context (faithful rework)
//!
//! The generic `DsparkDrafter` in `dspark_core` drives drafting through
//! the `DsparkBody` trait and verifies through the `SpecTarget` trait.
//! This file provides:
//!
//! - `Deepseek4DsparkBody` — holds a body-local `DeepseekV4State` (only the
//!   dspark-specific fields are populated) plus shallow-cloned stage weights
//!   and `token_embd`. Its `draft_block` calls:
//!   1. `dspark_core::main_proj_ingest_batched` — projects all `ctx_len` context
//!      slots through `main_proj` + `main_norm` → `main_x_batch [ctx_len * hidden]`.
//!   2. `forward::dspark_run_body_and_hc_gate` — for each context slot writes
//!      `main_kv` into the SWA ring at its absolute position, then runs the
//!      3-stage chain → `x_head_out[block, hidden]` (B+C.begin).
//!   The markov + confidence + lm-head (part C.rest) run inside
//!   `dspark_core::run_heads` after `draft_block` returns.
//!
//! - `build_deepseek4_dspark_body` — constructs the body from the loaded
//!   sidecar weights (all shallow-cloned; the bundle owns the GPU memory).
//!
//! - `build_deepseek4_dspark_speculator` — convenience wrapper that builds
//!   `dspark_core::DsparkWeights` + calls `build_dspark_speculator`.
//!
//! ## Faithfulness vs Task-5 byte-identity
//!
//! Stage 3 intentionally changes the numeric output: the context source for
//! `dspark_run_body_and_hc_gate` is now the accepted-prefix hidden (multi-slot,
//! populated from `verify_block`'s `dspark_caps` download) instead of the
//! single-slot bootstrap-captured seed. This matches the DeepSpec reference
//! `_update` exactly. The Task-5 byte-identity no longer applies; correctness
//! is validated by coherence gate (6/6 OK) and acceptance-rate improvement.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward;
use hipfire_runtime::dspark_core::{
    build_dspark_speculator, main_proj_ingest, main_proj_ingest_batched, DsparkBody,
    DsparkConfig as CoreDsparkConfig, DsparkWeights as CoreDsparkWeights,
};
use hipfire_runtime::spec::Speculator;
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Deepseek4DsparkBody ───────────────────────────────────────────────────────

/// Arch-specific DSpark body for DeepSeek V4. Owns a body-local
/// `DeepseekV4State` (only the dspark-specific fields are used) and
/// shallow-clones of the sidecar stage weights + trunk `token_embd`.
///
/// `draft_block` runs (Stage 3):
/// 1. `main_proj_ingest_batched` → `main_x_batch [ctx_len * hidden]` (part A,
///    multi-slot: all accepted-prefix context slots projected at once).
/// 2. `dspark_run_body_and_hc_gate` → `x_head_out[block, hidden]` (B+C.begin,
///    multi-slot ring writes: one `main_kv` ring entry per context slot).
///
/// All weight tensors are **shallow clones** — the bundle keeps ownership and
/// must outlive the body. `free` releases only the draft-local state buffers
/// (`dspark_pbs`, `dspark_swa_k`).
pub struct Deepseek4DsparkBody {
    config: DeepseekV4Config,
    /// Shallow-cloned sidecar stages. The bundle's `weights.dspark.stages`
    /// retains GPU-buffer ownership.
    stages_shallow: Vec<crate::deepseek4::DeepseekV4LayerWeights>,
    /// Shallow clone of `DeepseekV4Weights::token_embd`.
    token_embd: GpuTensor,
    /// Draft-local state; only `dspark_pbs`, `dspark_main_x`, `dspark_swa_k`
    /// are populated — the trunk fields stay None.
    draft_state: DeepseekV4State,
}

impl DsparkBody for Deepseek4DsparkBody {
    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &CoreDsparkWeights,
        main_hidden: &GpuTensor, // [ctx_len * n_targets * hidden] flat
        ctx_positions: &[usize], // absolute positions; len = ctx_len
        seed: u32,
        position: usize,
        block: usize,
        x_head_out: &GpuTensor, // [block, hidden] out
    ) -> Result<(), String> {
        let hidden = self.config.hidden_size;
        let ctx_len = ctx_positions.len().max(1);

        // ── Part A: main_proj_ingest (multi-slot, Stage 3) ───────────────
        // Produce main_x_batch [ctx_len * hidden] F32 by projecting all ctx_len
        // context slots through main_proj + main_norm in one batched call.
        // For ctx_len=1 this is identical to the prior single-slot path
        // (main_proj_ingest and main_proj_ingest_batched produce the same result).
        let main_x_batch = gpu
            .alloc_tensor(&[ctx_len * hidden], DType::F32)
            .map_err(|e| format!("Deepseek4DsparkBody: alloc main_x_batch: {e:?}"))?;
        if ctx_len == 1 {
            main_proj_ingest(gpu, weights, main_hidden, &main_x_batch)?;
        } else {
            main_proj_ingest_batched(gpu, weights, main_hidden, &main_x_batch, ctx_len, hidden)?;
        }

        // ── Parts B + C.begin: 3-stage chain + HC gate ────────────────────
        // Build a temporary shallow-clone DsparkWeights (deepseek4 type) for
        // dspark_run_body_and_hc_gate. The stages are owned by the bundle;
        // we use the shallow-cloned `stages_shallow` here.
        let ds4_weights = crate::deepseek4::DsparkWeights {
            cfg: crate::deepseek4::DsparkConfig {
                block_size: weights.cfg.block_size,
                target_layer_ids: weights.cfg.target_layer_ids.clone(),
                markov_rank: weights.cfg.markov_rank,
                noise_token_id: weights.cfg.noise_token_id,
            },
            stages: self
                .stages_shallow
                .iter()
                .map(|l| l.shallow_clone())
                .collect(),
            main_proj: weights.main_proj.as_ref().map(|t| t.shallow_clone()),
            main_norm: weights.main_norm.as_ref().map(|t| t.shallow_clone()),
            markov_w1: weights.markov_w1.as_ref().map(|t| t.shallow_clone()),
            markov_w2: weights.markov_w2.as_ref().map(|t| t.shallow_clone()),
            confidence_proj: weights.confidence_proj.as_ref().map(|t| t.shallow_clone()),
            draft_head: None,
        };

        forward::dspark_run_body_and_hc_gate(
            &self.config,
            &ds4_weights,
            &mut self.draft_state,
            gpu,
            &self.token_embd,
            &main_x_batch,
            ctx_positions,
            seed,
            position as u32,
            block,
            x_head_out,
        )?;

        let _ = gpu.free_tensor(main_x_batch);
        // ds4_weights contains shallow-cloned GpuTensors; dropping them is fine
        // (no GPU free, the bundle owns the buffers).

        Ok(())
    }

    fn block_size(&self) -> usize {
        // The authoritative block_size is in CoreDsparkWeights.cfg, passed to
        // build_dspark_speculator which clamps and stores it. This method is
        // only advisory (DsparkDrafter uses its own stored block). Return the
        // V4-Flash default (5) as a safe fallback.
        5
    }

    fn persistent_verify_context_scratch(&self) -> bool {
        true
    }

    fn reset_for_retry(&mut self, gpu: &mut Gpu) {
        // Persistently allocated per-stage SWA rings survive free-on-unload
        // only; cold retry must zero them or the next draft_block reads prior
        // window main_kv history. Dropping the Option forces lazy re-alloc on
        // the next draft (identical clean slate to a fresh body).
        for ring in self.draft_state.dspark_swa_k.drain(..) {
            if let Some(t) = ring {
                let _ = gpu.free_tensor(t);
            }
        }
        // Working scratch is fully overwritten each draft_block; still drop
        // so a retry cannot observe a stale main_x / staged buffer if a later
        // path short-circuits before rewrite.
        if let Some(t) = self.draft_state.dspark_main_x.take() {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.draft_state.dspark_staged.take() {
            let _ = gpu.free_tensor(t);
        }
        // Keep dspark_pbs (pure scratch, sized to block) — next draft overwrites.
        self.draft_state.n_tokens = 0;
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // Free draft-local state buffers only (NOT the shallow-cloned weights).
        let mut state = self.draft_state;
        fn free_opt(gpu: &mut Gpu, t: &mut Option<GpuTensor>) {
            if let Some(t) = t.take() {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in state.dspark_swa_k.drain(..).flatten() {
            let _ = gpu.free_tensor(t);
        }
        free_opt(gpu, &mut state.dspark_main_x);
        free_opt(gpu, &mut state.dspark_staged);
        if let Some(pbs) = state.dspark_pbs.take() {
            pbs.free_gpu(gpu);
        }
    }
}

// ── Builder functions ─────────────────────────────────────────────────────────

/// Build the `Deepseek4DsparkBody` from the loaded sidecar and trunk weights.
/// Shallow-clones all weight tensors — `DeepseekV4Weights` retains ownership.
pub fn build_deepseek4_dspark_body(
    config: &DeepseekV4Config,
    dspark_weights: &crate::deepseek4::DsparkWeights,
    token_embd: &GpuTensor,
) -> Result<Box<dyn DsparkBody>, String> {
    let stages_shallow: Vec<_> = dspark_weights
        .stages
        .iter()
        .map(|l| l.shallow_clone())
        .collect();
    let draft_state = DeepseekV4State::new(config)
        .map_err(|e| format!("build_deepseek4_dspark_body: draft_state::new: {e}"))?;
    Ok(Box::new(Deepseek4DsparkBody {
        config: config.clone(),
        stages_shallow,
        token_embd: token_embd.shallow_clone(),
        draft_state,
    }))
}

/// Build the generic DSpark speculator for DeepSeek V4 using the shared
/// `dspark_core::DsparkDrafter`. Constructs the `Deepseek4DsparkBody` +
/// `CoreDsparkWeights` (all shallow-cloned) and calls `build_dspark_speculator`.
///
/// The bundle must outlive the returned speculator; call `spec.free(gpu)` BEFORE
/// `bundle.weights.free_gpu(gpu)` on unload.
///
/// Confidence threshold ladder: `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` env
/// > `conf_threshold` arg > 0.3. Default 0.3 (was 0.5): with the lazy verify a
/// higher drafter budget (less confidence truncation) is a net GREEDY win —
/// admitting the drafts 0.5 was cutting recovers correct tokens at ~no per-head
/// verify cost. Measured (fixed-len, warm): code 8.95→9.47 (+5.8%), fiction
/// prose 10.09→12.13 (+20%); 0.3 is the sweet spot (0.1 over-proposes → prose
/// 11.27 < 12.13, and the verify *forward* still scales with proposals). Greedy
/// output is unchanged (conf is a pure speed knob — committed tokens are the
/// target argmax regardless of how many drafts were proposed). NB temp>0 wants
/// the OPPOSITE (more truncation): at temp0.7 conf 0.5 > 0.1; but ds4 temp>0 is
/// gated off in serving, so the greedy optimum governs the default. The resolved
/// value is clamped to `[0, 1]` — it is a survival-sigmoid cutoff, so the
/// env/JSON paths (which bypass the CLI's TS validation) can't push it out of
/// range and silently degrade (`>1` ⇒ block always trims to 1 ≈ AR; `<0` ⇒
/// truncation never fires).
#[allow(clippy::too_many_arguments)]
pub fn build_deepseek4_dspark_speculator(
    config: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    block: usize,
    ctx_capacity: usize,
    conf_threshold: Option<f32>,
    supports_temp: bool,
) -> Result<Box<dyn Speculator>, String> {
    let dspark = weights
        .dspark
        .as_ref()
        .ok_or("build_deepseek4_dspark_speculator: weights.dspark is None")?;
    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("build_deepseek4_dspark_speculator: weights.token_embd is None")?;
    let lm_head = weights
        .head
        .as_ref()
        .ok_or("build_deepseek4_dspark_speculator: weights.head is None")?;
    let last_stage = dspark
        .stages
        .last()
        .ok_or("build_deepseek4_dspark_speculator: dspark has no stages")?;
    let stage_norm = last_stage
        .mtp_final_norm
        .as_ref()
        .ok_or("build_deepseek4_dspark_speculator: mtp_final_norm missing on last stage")?;

    let conf_threshold = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .or(conf_threshold)
        .unwrap_or(0.3)
        .clamp(0.0, 1.0);

    // Build arch-agnostic CoreDsparkWeights from sidecar globals (shallow clones).
    let core_weights = CoreDsparkWeights {
        cfg: CoreDsparkConfig {
            block_size: dspark.cfg.block_size,
            target_layer_ids: dspark.cfg.target_layer_ids.clone(),
            markov_rank: dspark.cfg.markov_rank,
            noise_token_id: dspark.cfg.noise_token_id,
            enable_confidence: true, // deepseek4 always has a confidence head
            confidence_uses_normed: false, // deepseek4 uses pre-norm x_head (byte-identical to task-5)
            rms_norm_eps: config.rms_norm_eps, // 1e-6 from deepseek4 config (byte-identical)
            // deepseek4's body ignores these drafter-only fields (no reduced vocab,
            // no qwen3 partial-rotary path) — defaults keep it byte-identical.
            draft_vocab_size: 0,
            partial_rotary_factor: 1.0,
            rope_theta: 1_000_000.0,
        },
        main_proj: dspark.main_proj.as_ref().map(|t| t.shallow_clone()),
        main_norm: dspark.main_norm.as_ref().map(|t| t.shallow_clone()),
        markov_w1: dspark.markov_w1.as_ref().map(|t| t.shallow_clone()),
        markov_w2: dspark.markov_w2.as_ref().map(|t| t.shallow_clone()),
        confidence_proj: dspark.confidence_proj.as_ref().map(|t| t.shallow_clone()),
        confidence_bias: None, // deepseek4 has no bias on the confidence head
        d2t: None,             // deepseek4 shares the target vocab (no d2t map)
    };

    let body = build_deepseek4_dspark_body(config, dspark, token_embd)?;

    Ok(build_dspark_speculator(
        body,
        core_weights,
        stage_norm.shallow_clone(),
        lm_head.shallow_clone(),
        block,
        ctx_capacity,
        conf_threshold,
        supports_temp,
        0.45,
    ))
}

// ── Send-bound note ────────────────────────────────────────────────────
// The pre-merge build asserted `Deepseek4DsparkBody: Send`. That bound no
// longer holds on mainline: `PrefillBatchScratch` carries
// `dspark_verify_pm4: BTreeMap<usize, ReplayController>` (mainline Redline
// verify controllers; `rdna_compute::ReplayController` is deliberately
// non-Send — it owns raw `NonNull` ROCr queue/kernarg handles). The
// `DsparkBody` trait has no `Send` supertrait and no mainline path moves
// the body across threads (`build_dspark_speculator` returns
// `Box<dyn DsparkBody>` on the serving thread), so the assertion is stale,
// not a missing safety property — do not `unsafe impl Send` to revive it.

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal config — only fields `DeepseekV4State::new` / body reset need.
    fn tiny_cfg() -> DeepseekV4Config {
        DeepseekV4Config {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            q_lora_rank: 8,
            o_lora_rank: 8,
            qk_rope_head_dim: 4,
            o_groups: 1,
            n_routed_experts: 1,
            n_shared_experts: 0,
            num_experts_per_tok: 1,
            moe_intermediate_size: 16,
            routed_scaling_factor: 1.0,
            topk_method: "noaux_tc".into(),
            scoring_func: "sqrtsoftplus".into(),
            norm_topk_prob: true,
            swiglu_limit: 10.0,
            hc_mult: 4,
            hc_sinkhorn_iters: 1,
            hc_eps: 1e-6,
            index_n_heads: 1,
            index_head_dim: 8,
            index_topk: 1,
            compress_ratios: vec![0],
            compress_rope_theta: 160000.0,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_original_max_position_embeddings: 64,
            rope_scaling_beta_fast: 32,
            rope_scaling_beta_slow: 1,
            sliding_window: 8,
            num_nextn_predict_layers: 0,
            num_hash_layers: 0,
            reap_keep: None,
            load_dspark: true,
            mq2r: false,
            mq2rxt: false,
        }
    }

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok()
    }

    /// Production behavior: a dirtied `dspark_swa_k` ring must not survive
    /// `DsparkBody::reset_for_retry` (the path `DsparkDrafter::mtp_reset` uses).
    #[test]
    fn dspark_body_reset_for_retry_drains_swa_rings() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let cfg = tiny_cfg();
        let draft_state = DeepseekV4State::new(&cfg).expect("DeepseekV4State::new");
        let emb = gpu
            .zeros(&[cfg.vocab_size * cfg.hidden_size], DType::F32)
            .expect("emb");
        let mut body = Deepseek4DsparkBody {
            config: cfg,
            stages_shallow: Vec::new(),
            token_embd: emb,
            draft_state,
        };

        // Simulate a prior draft_block that allocated a per-stage SWA ring
        // and wrote non-zero main_kv history into it.
        let ring = gpu
            .upload_f32(&[1.25f32, -0.5, 3.0, 0.25, 9.0, -2.0, 0.125, 4.0], &[8])
            .expect("ring");
        body.draft_state.dspark_swa_k.push(Some(ring));
        body.draft_state.dspark_main_x =
            Some(gpu.upload_f32(&[7.0f32; 16], &[16]).expect("main_x"));
        body.draft_state.dspark_staged = Some(gpu.upload_f32(&[2.0f32; 8], &[8]).expect("staged"));
        body.draft_state.n_tokens = 42;

        assert_eq!(body.draft_state.dspark_swa_k.len(), 1);
        assert!(body.draft_state.dspark_swa_k[0].is_some());

        body.reset_for_retry(&mut gpu);

        assert!(
            body.draft_state.dspark_swa_k.is_empty(),
            "dspark_swa_k must drain on reset_for_retry"
        );
        assert!(
            body.draft_state.dspark_main_x.is_none(),
            "dspark_main_x must drop on reset_for_retry"
        );
        assert!(
            body.draft_state.dspark_staged.is_none(),
            "dspark_staged must drop on reset_for_retry"
        );
        assert_eq!(body.draft_state.n_tokens, 0);

        let _ = gpu.free_tensor(body.token_embd);
    }

    /// Wire check: `build_dspark_speculator` → `Speculator::reset` reaches the
    /// body and frees dirtied rings without panic (same path as cold retry).
    #[test]
    fn dspark_speculator_reset_clears_body_swa_rings() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let cfg = tiny_cfg();
        let mut draft_state = DeepseekV4State::new(&cfg).expect("DeepseekV4State::new");
        draft_state
            .dspark_swa_k
            .push(Some(gpu.upload_f32(&[1.0f32; 4], &[4]).expect("ring")));
        draft_state.n_tokens = 7;
        let emb = gpu
            .zeros(&[cfg.vocab_size * cfg.hidden_size], DType::F32)
            .expect("emb");
        let body = Box::new(Deepseek4DsparkBody {
            config: cfg.clone(),
            stages_shallow: Vec::new(),
            token_embd: emb.shallow_clone(),
            draft_state,
        });

        let stage_norm = gpu.zeros(&[cfg.hidden_size], DType::F32).expect("sn");
        let lm_head = gpu
            .zeros(&[cfg.vocab_size * cfg.hidden_size], DType::F32)
            .expect("lm");
        let core_weights = CoreDsparkWeights {
            cfg: CoreDsparkConfig {
                block_size: 2,
                target_layer_ids: vec![0],
                markov_rank: 0,
                noise_token_id: 0,
                enable_confidence: false,
                confidence_uses_normed: false,
                rms_norm_eps: cfg.rms_norm_eps,
                draft_vocab_size: 0,
                partial_rotary_factor: 1.0,
                rope_theta: 1_000_000.0,
            },
            main_proj: None,
            main_norm: None,
            markov_w1: None,
            markov_w2: None,
            confidence_proj: None,
            confidence_bias: None,
            d2t: None,
        };

        let mut spec = build_dspark_speculator(
            body,
            core_weights,
            stage_norm,
            lm_head,
            2,
            8,
            0.3,
            false,
            0.45,
        );
        // Dirtied rings are inside the body; reset must free them (no panic).
        let _ = spec.reset(&mut gpu);
        spec.free(&mut gpu);
        let _ = gpu.free_tensor(emb);
    }
}
