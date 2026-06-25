// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! nemotron_h full model decode forward (N4) — composes the three validated
//! per-block structs into the flat hybrid stack, plus a host-side CPU oracle.
//!
//! ```text
//!   h = embeddings[token]
//!   for each layer L (kind from hybrid_override_pattern):
//!       hn = rmsnorm(h, layers[L].norm.weight)         # standard RMSNorm
//!       h  = h + block_L(hn)                            # pre-norm residual
//!         M → Mamba2BlockGpu::decode_step
//!         * → NemotronAttnGpu::forward(pos)
//!         - → MlpRelu2Gpu::forward
//!   h = rmsnorm(h, norm_f.weight)
//!   logits = lm_head @ h                                # [vocab]
//! ```
//! f32 / decode-only (single token). The host weights ([`NemotronWeights`]) are
//! the loader's target representation. Validated gpu-vs-cpu in
//! `examples/test_model_gpu.rs`.

use crate::attn::{gqa_attention, NemotronAttnGpu};
use crate::block::{mamba2_block_decode_step, Mamba2BlockState, Mamba2BlockWeights};
use crate::mlp::{mlp_relu2, MlpRelu2Gpu};
use crate::moe::{moe_relu2, MoeRelu2Gpu, MoeWeights};
use crate::weight::{EmbeddingTable, LinearWeight};
use crate::{BlockKind, NemotronHConfig};
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Host-side per-block weights (the loader's target). One per stack layer.
pub enum HostBlock {
    Mamba2 {
        in_proj: Vec<f32>,
        conv_weight: Vec<f32>,
        conv_bias: Vec<f32>,
        a_log: Vec<f32>,
        d: Vec<f32>,
        dt_bias: Vec<f32>,
        mixer_norm: Vec<f32>,
        out_proj: Vec<f32>,
    },
    Mlp {
        up: Vec<f32>,
        down: Vec<f32>,
    },
    Attn {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        o: Vec<f32>,
    },
    Moe(Box<MoeWeights>),
}

/// Host-side full nemotron_h weights (dequantized f32). Loader target.
pub struct NemotronWeights {
    /// `[vocab * hidden]`.
    pub embeddings: Vec<f32>,
    /// Per-layer pre-block RMSNorm weight `[hidden]`.
    pub layer_norm: Vec<Vec<f32>>,
    /// Per-layer block weights (len == num_layers, aligned with `cfg.blocks`).
    pub blocks: Vec<HostBlock>,
    /// Final RMSNorm `[hidden]`.
    pub norm_f: Vec<f32>,
    /// `[vocab * hidden]` (== embeddings if tied).
    pub lm_head: Vec<f32>,
}

enum Block {
    Mamba2(Box<crate::block_gpu::Mamba2BlockGpu>),
    Mlp(MlpRelu2Gpu),
    Attn(NemotronAttnGpu),
    Moe(Box<MoeRelu2Gpu>),
}

/// GPU-resident nemotron_h model (decode forward).
pub struct NemotronModel {
    cfg: NemotronHConfig,
    /// Whether the batched N6 prefill path can run. Both f32 and the supported
    /// HFQ linear formats use [`LinearWeight::gemm_seq`]; unsupported future
    /// quant dtypes fail there with a classifiable capability-gap error.
    batched_prefill: bool,
    embeddings: EmbeddingTable,
    layer_norm: Vec<GpuTensor>,
    layers: Vec<Block>,
    norm_f: GpuTensor,
    lm_head: LinearWeight,
    // scratch
    h: GpuTensor,
    normed: GpuTensor,
    logits: GpuTensor,
}

impl NemotronModel {
    /// Upload `w` and build the GPU model. `max_seq` is the KV-cache budget for
    /// the attention blocks.
    pub fn new(
        gpu: &mut Gpu,
        cfg: NemotronHConfig,
        w: &NemotronWeights,
        max_seq: usize,
    ) -> HipResult<Self> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let dims = cfg.mamba2_dims();
        let batched_prefill = !cfg.blocks.iter().any(|b| *b == BlockKind::Moe);

        let embeddings = EmbeddingTable::F32(gpu.upload_f32(&w.embeddings, &[vocab, hidden])?);
        let lm_head = LinearWeight::F32(gpu.upload_f32(&w.lm_head, &[vocab, hidden])?);
        let norm_f = gpu.upload_f32(&w.norm_f, &[hidden])?;

        let mut layer_norm = Vec::with_capacity(cfg.num_layers);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for (l, hb) in w.blocks.iter().enumerate() {
            layer_norm.push(gpu.upload_f32(&w.layer_norm[l], &[hidden])?);
            let block = match hb {
                HostBlock::Mamba2 {
                    in_proj,
                    conv_weight,
                    conv_bias,
                    a_log,
                    d,
                    dt_bias,
                    mixer_norm,
                    out_proj,
                } => {
                    let bw = Mamba2BlockWeights {
                        in_proj,
                        conv_weight,
                        conv_bias,
                        a_log,
                        d,
                        dt_bias,
                        norm_weight: mixer_norm,
                        out_proj,
                    };
                    Block::Mamba2(Box::new(crate::block_gpu::Mamba2BlockGpu::new(
                        gpu,
                        dims.clone(),
                        &bw,
                    )?))
                }
                HostBlock::Mlp { up, down } => Block::Mlp(MlpRelu2Gpu::new(
                    gpu,
                    hidden,
                    cfg.mlp_intermediate,
                    up,
                    down,
                )?),
                HostBlock::Attn { q, k, v, o } => Block::Attn(NemotronAttnGpu::new(
                    gpu, cfg.attn, hidden, max_seq, q, k, v, o,
                )?),
                HostBlock::Moe(w) => {
                    let moe = cfg.moe.ok_or_else(|| {
                        HipError::new(0, "nemotron MoE block present but config has no MoE shape")
                    })?;
                    Block::Moe(Box::new(MoeRelu2Gpu::new(gpu, hidden, moe, w)?))
                }
            };
            layers.push(block);
        }

        Ok(Self {
            embeddings,
            lm_head,
            norm_f,
            layer_norm,
            layers,
            h: gpu.zeros(&[hidden], DType::F32)?,
            normed: gpu.zeros(&[hidden], DType::F32)?,
            logits: gpu.zeros(&[vocab], DType::F32)?,
            batched_prefill,
            cfg,
        })
    }

    /// Build the model from a quantized HFQ container (FU4). The linear weights
    /// (`in/out/up/down`, `q/k/v/o`, `lm_head`) load as quantized
    /// [`LinearWeight`]s — mq4/hfq4/q8, with MQ4 auto-FWHT-rotated by the
    /// dispatched gemv; the recurrence + norm tensors dequantize from BF16;
    /// embeddings stay Q8 (`embedding_lookup_q8`). The Mamba `out_proj` residual
    /// rescale (`1/√num_layers`) is applied at runtime — the quantized bytes
    /// can't be pre-scaled the way the safetensors loader folds it into the
    /// weight. Returns `Err(String)` on a missing/unsupported tensor.
    pub fn from_hfq(
        gpu: &mut Gpu,
        hfq: &hipfire_runtime::hfq::HfqFile,
        cfg: NemotronHConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        use crate::loader::{load_embeddings_hfq, load_f32_hfq, load_linear_hfq};
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let dims = cfg.mamba2_dims();
        let batched_prefill = !cfg.blocks.iter().any(|b| *b == BlockKind::Moe);
        let out_proj_scale = cfg.mamba_out_proj_runtime_scale();
        let e = |x: hip_bridge::HipError| format!("nemotron hfq gpu: {x:?}");

        let embeddings =
            load_embeddings_hfq(hfq, gpu, "backbone.embeddings.weight", vocab, hidden)?;
        let lm_head = load_linear_hfq(hfq, gpu, "lm_head.weight", vocab, hidden)?;
        let norm_f = gpu
            .upload_f32(&load_f32_hfq(hfq, "backbone.norm_f.weight")?, &[hidden])
            .map_err(e)?;

        let mut layer_norm = Vec::with_capacity(cfg.num_layers);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for (l, kind) in cfg.blocks.iter().enumerate() {
            let p = format!("backbone.layers.{l}");
            layer_norm.push(
                gpu.upload_f32(&load_f32_hfq(hfq, &format!("{p}.norm.weight"))?, &[hidden])
                    .map_err(e)?,
            );
            let m = format!("{p}.mixer");
            let block = match kind {
                BlockKind::Mamba2 => {
                    let in_proj = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.in_proj.weight"),
                        dims.projection_size(),
                        hidden,
                    )?;
                    let out_proj = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.out_proj.weight"),
                        hidden,
                        dims.d_inner(),
                    )?;
                    let conv_weight = load_f32_hfq(hfq, &format!("{m}.conv1d.weight"))?;
                    let conv_bias = load_f32_hfq(hfq, &format!("{m}.conv1d.bias"))?;
                    let a_log = load_f32_hfq(hfq, &format!("{m}.A_log"))?;
                    let d = load_f32_hfq(hfq, &format!("{m}.D"))?;
                    let dt_bias = load_f32_hfq(hfq, &format!("{m}.dt_bias"))?;
                    let mixer_norm = load_f32_hfq(hfq, &format!("{m}.norm.weight"))?;
                    Block::Mamba2(Box::new(
                        crate::block_gpu::Mamba2BlockGpu::new_quant(
                            gpu,
                            dims.clone(),
                            in_proj,
                            out_proj,
                            out_proj_scale,
                            &conv_weight,
                            &conv_bias,
                            &a_log,
                            &d,
                            &dt_bias,
                            &mixer_norm,
                        )
                        .map_err(e)?,
                    ))
                }
                BlockKind::Mlp => {
                    let up = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.up_proj.weight"),
                        cfg.mlp_intermediate,
                        hidden,
                    )?;
                    let down = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.down_proj.weight"),
                        hidden,
                        cfg.mlp_intermediate,
                    )?;
                    Block::Mlp(
                        MlpRelu2Gpu::new_quant(gpu, hidden, cfg.mlp_intermediate, up, down)
                            .map_err(e)?,
                    )
                }
                BlockKind::Attention => {
                    let a = cfg.attn;
                    let q_dim = a.num_heads * a.head_dim;
                    let kv_dim = a.num_kv_heads * a.head_dim;
                    let q =
                        load_linear_hfq(hfq, gpu, &format!("{m}.q_proj.weight"), q_dim, hidden)?;
                    let k =
                        load_linear_hfq(hfq, gpu, &format!("{m}.k_proj.weight"), kv_dim, hidden)?;
                    let v =
                        load_linear_hfq(hfq, gpu, &format!("{m}.v_proj.weight"), kv_dim, hidden)?;
                    let o =
                        load_linear_hfq(hfq, gpu, &format!("{m}.o_proj.weight"), hidden, q_dim)?;
                    Block::Attn(
                        NemotronAttnGpu::new_quant(gpu, a, hidden, max_seq, q, k, v, o)
                            .map_err(e)?,
                    )
                }
                BlockKind::Moe => {
                    let moe = cfg.moe.ok_or_else(|| {
                        "nemotron hfq gpu: MoE block without MoE config".to_string()
                    })?;
                    let router = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.gate.weight"),
                        moe.n_routed_experts,
                        hidden,
                    )?;
                    let expert_bias =
                        load_f32_hfq(hfq, &format!("{m}.gate.e_score_correction_bias"))?;
                    let shared_up = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.shared_experts.up_proj.weight"),
                        moe.shared_expert_intermediate_size,
                        hidden,
                    )?;
                    let shared_down = load_linear_hfq(
                        hfq,
                        gpu,
                        &format!("{m}.shared_experts.down_proj.weight"),
                        hidden,
                        moe.shared_expert_intermediate_size,
                    )?;
                    let shared = MlpRelu2Gpu::new_quant(
                        gpu,
                        hidden,
                        moe.shared_expert_intermediate_size,
                        shared_up,
                        shared_down,
                    )
                    .map_err(e)?;
                    let mut experts = Vec::with_capacity(moe.n_routed_experts);
                    for expert_idx in 0..moe.n_routed_experts {
                        let up = load_linear_hfq(
                            hfq,
                            gpu,
                            &format!("{m}.experts.{expert_idx}.up_proj.weight"),
                            moe.intermediate_size,
                            hidden,
                        )?;
                        let down = load_linear_hfq(
                            hfq,
                            gpu,
                            &format!("{m}.experts.{expert_idx}.down_proj.weight"),
                            hidden,
                            moe.intermediate_size,
                        )?;
                        experts.push(
                            MlpRelu2Gpu::new_quant(gpu, hidden, moe.intermediate_size, up, down)
                                .map_err(e)?,
                        );
                    }
                    Block::Moe(Box::new(
                        MoeRelu2Gpu::new_quant(
                            gpu,
                            hidden,
                            moe,
                            router,
                            &expert_bias,
                            shared,
                            experts,
                        )
                        .map_err(e)?,
                    ))
                }
            };
            layers.push(block);
        }

        Ok(Self {
            embeddings,
            lm_head,
            norm_f,
            layer_norm,
            layers,
            h: gpu.zeros(&[hidden], DType::F32).map_err(e)?,
            normed: gpu.zeros(&[hidden], DType::F32).map_err(e)?,
            logits: gpu.zeros(&[vocab], DType::F32).map_err(e)?,
            batched_prefill,
            cfg,
        })
    }

    /// Decode one token at `pos`, leaving the `[vocab]` logits in `self.logits`
    /// **on the GPU** (no download/sync) for the daemon sampler.
    pub fn forward_gpu(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()> {
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;

        self.embeddings.lookup(gpu, &self.h, token, hidden)?;
        for l in 0..self.layers.len() {
            gpu.rmsnorm_f32(&self.h, &self.layer_norm[l], &self.normed, eps)?;
            match &mut self.layers[l] {
                Block::Mamba2(b) => {
                    let o = b.decode_step(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Mlp(b) => {
                    let o = b.forward(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Attn(b) => {
                    let o = b.forward(gpu, &self.normed, pos)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Moe(b) => {
                    let o = b.forward(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
            }
        }
        gpu.rmsnorm_f32(&self.h, &self.norm_f, &self.normed, eps)?;
        self.lm_head.gemv(gpu, &self.normed, &self.logits)?;
        Ok(())
    }

    /// Whether the batched N6 prefill ([`Self::prefill_batched`]) is available
    /// (f32 weights). Quantized models fall back to the per-token loop.
    pub fn can_batched_prefill(&self) -> bool {
        self.batched_prefill
    }

    /// Batched N6 prefill: process the whole prompt through the residual stream
    /// in batched form (embed → per-block `prefill` with `rmsnorm_batched`
    /// pre-norm + residual add), leaving the **last position's** `[vocab]` logits
    /// in `self.logits` and every block's recurrent/KV state at the post-prompt
    /// value — equivalent to `forward_gpu` over `tokens`, but with one launch per
    /// recurrent kernel instead of per token. Assumes fresh state (caller resets).
    /// F32 weights only; see [`Self::can_batched_prefill`].
    pub fn prefill_batched(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> HipResult<()> {
        const F32B: usize = std::mem::size_of::<f32>();
        let seq = tokens.len();
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;

        let h_seq = gpu.zeros(&[seq * hidden], DType::F32)?;
        let normed_seq = gpu.zeros(&[seq * hidden], DType::F32)?;

        // embed each token into the residual stream rows.
        for (p, &t) in tokens.iter().enumerate() {
            self.embeddings.lookup(gpu, &self.h, t, hidden)?;
            gpu.memcpy_dtod_at_auto(&h_seq.buf, p * hidden * F32B, &self.h.buf, 0, hidden * F32B)?;
        }

        for l in 0..self.layers.len() {
            gpu.rmsnorm_batched(&h_seq, &self.layer_norm[l], &normed_seq, seq, hidden, eps)?;
            let out = match &mut self.layers[l] {
                Block::Mamba2(b) => b.prefill(gpu, &normed_seq, seq)?,
                Block::Mlp(b) => b.prefill(gpu, &normed_seq, seq)?,
                Block::Attn(b) => b.prefill(gpu, &normed_seq, seq)?,
                Block::Moe(_) => {
                    return Err(HipError::unsupported(
                        "nemotron MoE batched prefill is not implemented yet",
                    ));
                }
            };
            gpu.add_inplace_f32(&h_seq, &out)?;
            let _ = gpu.free_tensor(out);
        }

        gpu.rmsnorm_batched(&h_seq, &self.norm_f, &normed_seq, seq, hidden, eps)?;
        // lm_head on the LAST position only (the next-token distribution).
        gpu.memcpy_dtod_at_auto(
            &self.normed.buf,
            0,
            &normed_seq.buf,
            (seq - 1) * hidden * F32B,
            hidden * F32B,
        )?;
        self.lm_head.gemv(gpu, &self.normed, &self.logits)?;

        let _ = gpu.free_tensor(h_seq);
        let _ = gpu.free_tensor(normed_seq);
        Ok(())
    }

    /// Like `forward_gpu`, but downloads the residual-stream hidden state after
    /// the embedding and after **each** block (43 vectors for 42 layers), plus
    /// the final `[vocab]` logits — for the HF-reference numeric bisect (FU2).
    /// `hidden[0]` = embeddings, `hidden[l+1]` = residual stream after block `l`
    /// (before `norm_f`), matching HF `output_hidden_states`.
    pub fn forward_capture(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        pos: usize,
    ) -> HipResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let mut caps: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len() + 1);

        self.embeddings.lookup(gpu, &self.h, token, hidden)?;
        gpu.hip.device_synchronize()?;
        caps.push(gpu.download_f32(&self.h)?);
        for l in 0..self.layers.len() {
            gpu.rmsnorm_f32(&self.h, &self.layer_norm[l], &self.normed, eps)?;
            match &mut self.layers[l] {
                Block::Mamba2(b) => {
                    let o = b.decode_step(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Mlp(b) => {
                    let o = b.forward(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Attn(b) => {
                    let o = b.forward(gpu, &self.normed, pos)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
                Block::Moe(b) => {
                    let o = b.forward(gpu, &self.normed)?;
                    gpu.add_inplace_f32(&self.h, o)?;
                }
            }
            gpu.hip.device_synchronize()?;
            caps.push(gpu.download_f32(&self.h)?);
        }
        gpu.rmsnorm_f32(&self.h, &self.norm_f, &self.normed, eps)?;
        self.lm_head.gemv(gpu, &self.normed, &self.logits)?;
        gpu.hip.device_synchronize()?;
        let logits = gpu.download_f32(&self.logits)?;
        Ok((caps, logits))
    }

    /// Decode one token at `pos`; returns the downloaded `[vocab]` logits.
    /// (Convenience for examples/tests; the serving path uses `forward_gpu` +
    /// the on-device `logits()` tensor.)
    pub fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<Vec<f32>> {
        self.forward_gpu(gpu, token, pos)?;
        gpu.hip.device_synchronize()?;
        gpu.download_f32(&self.logits)
    }

    /// The most recent step's logits tensor (`[vocab]`), on the GPU.
    pub fn logits_tensor(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn config(&self) -> &NemotronHConfig {
        &self.cfg
    }

    /// Zero the recurrent state (Mamba conv/SSM) for a fresh generation. The
    /// attention KV caches need no zeroing — they're overwritten per `pos` and
    /// only read over `0..=pos`.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for b in &mut self.layers {
            if let Block::Mamba2(m) = b {
                m.reset(gpu)?;
            }
        }
        Ok(())
    }

    /// Free all GPU tensors (consumes the model).
    pub fn free(self, gpu: &mut Gpu) {
        self.embeddings.free(gpu);
        self.lm_head.free(gpu);
        let _ = gpu.free_tensor(self.norm_f);
        let _ = gpu.free_tensor(self.h);
        let _ = gpu.free_tensor(self.normed);
        let _ = gpu.free_tensor(self.logits);
        for n in self.layer_norm {
            let _ = gpu.free_tensor(n);
        }
        for b in self.layers {
            match b {
                Block::Mamba2(m) => m.free(gpu),
                Block::Mlp(m) => m.free(gpu),
                Block::Attn(a) => a.free(gpu),
                Block::Moe(m) => m.free(gpu),
            }
        }
    }
}

// ── CPU oracle ──────────────────────────────────────────────────────────────

#[inline]
fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0f32 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * w[i] * inv).collect()
}

fn matvec(w: &[f32], x: &[f32], out: usize, n_in: usize) -> Vec<f32> {
    (0..out)
        .map(|i| {
            w[i * n_in..i * n_in + n_in]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Per-layer recurrent CPU state for the oracle (Mamba conv/ssm, attn KV hist).
pub enum CpuBlockState {
    Mamba2(Mamba2BlockState),
    Attn {
        k_hist: Vec<Vec<f32>>,
        v_hist: Vec<Vec<f32>>,
    },
    Mlp,
    Moe,
}

/// Build the per-layer CPU state aligned with `cfg.blocks`.
pub fn cpu_state(cfg: &NemotronHConfig) -> Vec<CpuBlockState> {
    let dims = cfg.mamba2_dims();
    cfg.blocks
        .iter()
        .map(|k| match k {
            BlockKind::Mamba2 => CpuBlockState::Mamba2(Mamba2BlockState::zeros(&dims)),
            BlockKind::Attention => CpuBlockState::Attn {
                k_hist: Vec::new(),
                v_hist: Vec::new(),
            },
            BlockKind::Mlp => CpuBlockState::Mlp,
            BlockKind::Moe => CpuBlockState::Moe,
        })
        .collect()
}

/// CPU reference forward for one token at `pos`; returns `[vocab]` logits.
pub fn forward_cpu(
    cfg: &NemotronHConfig,
    w: &NemotronWeights,
    state: &mut [CpuBlockState],
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let dims = cfg.mamba2_dims();

    let mut h = w.embeddings[token as usize * hidden..(token as usize + 1) * hidden].to_vec();

    for l in 0..cfg.blocks.len() {
        let hn = rmsnorm_cpu(&h, &w.layer_norm[l], eps);
        let out = match (&w.blocks[l], &mut state[l]) {
            (
                HostBlock::Mamba2 {
                    in_proj,
                    conv_weight,
                    conv_bias,
                    a_log,
                    d,
                    dt_bias,
                    mixer_norm,
                    out_proj,
                },
                CpuBlockState::Mamba2(st),
            ) => {
                let bw = Mamba2BlockWeights {
                    in_proj,
                    conv_weight,
                    conv_bias,
                    a_log,
                    d,
                    dt_bias,
                    norm_weight: mixer_norm,
                    out_proj,
                };
                mamba2_block_decode_step(&dims, &bw, st, &hn)
            }
            (HostBlock::Mlp { up, down }, CpuBlockState::Mlp) => {
                mlp_relu2(up, down, &hn, hidden, cfg.mlp_intermediate)
            }
            (HostBlock::Attn { q, k, v, o }, CpuBlockState::Attn { k_hist, v_hist }) => {
                let a = cfg.attn;
                let q_dim = a.num_heads * a.head_dim;
                let kv_dim = a.num_kv_heads * a.head_dim;
                let qv = matvec(q, &hn, q_dim, hidden);
                k_hist.push(matvec(k, &hn, kv_dim, hidden));
                v_hist.push(matvec(v, &hn, kv_dim, hidden));
                let att =
                    gqa_attention(&qv, k_hist, v_hist, a.num_heads, a.num_kv_heads, a.head_dim);
                matvec(o, &att, hidden, q_dim)
            }
            (HostBlock::Moe(w), CpuBlockState::Moe) => {
                let moe = cfg
                    .moe
                    .expect("nemotron MoE block present but config has no MoE shape");
                moe_relu2(&moe, w, &hn, hidden)
            }
            _ => unreachable!("block/state kind mismatch at layer {l}"),
        };
        for i in 0..hidden {
            h[i] += out[i];
        }
    }
    let _ = pos;
    let hf = rmsnorm_cpu(&h, &w.norm_f, eps);
    matvec(&w.lm_head, &hf, cfg.vocab_size, hidden)
}
