// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3-8B DSpark drafter sidecar loader + block-attention body forward.
//!
//! ## Sidecar loader
//!
//! Loads a `<stem>-dspark.hfq` sidecar (arch_id=1, 64 tensors produced by the
//! Task-6 quantiser) into:
//! - [`hipfire_runtime::dspark_core::DsparkWeights`] (globals: main_proj,
//!   main_norm, markov heads, confidence head + bias).
//! - [`Qwen3DrafterAssets`] (5-layer dense-GQA drafter body: LlamaWeights /
//!   LlamaConfig + block-sized KvCache + ForwardScratch + PrefillBatchScratch).
//!
//! ## Block-attention body forward
//!
//! [`dspark_qwen3_block_forward`] implements the 5-layer dense Qwen3 forward
//! where each layer's block queries attend **bidirectionally** over
//! `[main_x context KV ++ block KV]`.  This matches
//! `Qwen3DSparkModel._forward_backbone` in the reference:
//!   - modeling.py:373  `target_hidden_states = self.hidden_norm(self.fc(...))`
//!                      → `main_x` is computed by the caller (Task 7) before entering
//!                      this function.
//!   - modeling.py:99–116 per-layer attention: q/k/v projections, q_norm/k_norm
//!     (on concatenated K), RoPE, bidirectional GQA over [ctx++block] KV.
//!   - modeling.py:375  single `position_embeddings` call before the layer loop →
//!     all layers share the same RoPE positions (not recomputed per layer).
//!   - modeling.py:386  `self.norm(hidden_states)` → final norm applied here.
//!
//! ## Sidecar tensor layout (flat — no `model.` prefix)
//!
//! ```text
//! layers.{0..4}.self_attn.{q,k,v,o}_proj.weight   (qt=3, Q8_0/Q8F16 — 8-bit: F16 scale + 32×i8)
//! layers.{0..4}.self_attn.{q,k}_norm.weight        (qt=1, F16 → F32)
//! layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight  (qt=1)
//! layers.{0..4}.mlp.{gate,up,down}_proj.weight     (qt=3, Q8_0/Q8F16)
//! embed_tokens.weight                              (qt=1, F16 → F32)
//! main_proj.weight                                 (qt=1, F16)
//! main_norm.weight                                 (qt=1, F16 → F32)
//! markov_head.markov_w1.weight                     (qt=1, F16)
//! markov_head.markov_w2.weight                     (qt=1, F16)
//! confidence_head.proj.weight                      (qt=1, F16)
//! confidence_head.proj.bias                        (qt=1, F16 → F32 scalar)
//! norm.weight                                      (qt=1, F16 → F32)
//! lm_head.weight                                   (qt=1, F16)
//! ```
//!
//! ## Hard requirements (Task-6 review)
//! 1. `confidence_bias` loaded from `confidence_head.proj.bias` — qwen3 HAS a
//!    bias; deepseek4 sets `confidence_bias: None`.
//! 2. `dspark_enable_confidence` parsed from the sidecar metadata —
//!    `DsparkConfig::from_metadata_json` (in dspark_core) reads it; deepseek4's
//!    local `DsparkConfig` hardcodes `enable_confidence: true`.

use hipfire_runtime::dspark_core::{
    main_proj_ingest, main_proj_ingest_batched, noise_block_ids, DsparkBody, DsparkConfig,
    DsparkWeights,
};
use hipfire_runtime::hfq::{load_layer, load_weight_tensor_pread, HfqFile};
use hipfire_runtime::llama::{
    weight_gemv, ForwardScratch, KvCache, LayerWeights, LlamaConfig, LlamaWeights, ModelArch,
    PrefillBatchScratch, WeightTensor,
};
use hipfire_runtime::weight_backend::{
    dequant_f32, dequant_norm, dequant_weight_raw, load_awq_scale_for, load_embedding, read_first,
    HfqBackend,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── name resolver ─────────────────────────────────────────────────────────────
// The sidecar uses flat names (no `model.` prefix).  read_first's candidate fn
// must return just the bare name — not the `model.{name}` variant that
// flat_name_candidates would try first.
fn bare_name_candidates(name: &str) -> Vec<String> {
    vec![name.to_string()]
}

// ── Assets bundle ─────────────────────────────────────────────────────────────

/// GPU-resident assets for the 5-layer Qwen3-8B DSpark drafter body.
///
/// Produced by [`load_qwen3_dspark`] and consumed by Tasks 8–10 (body-forward,
/// window orchestration, speculator wiring).
///
/// YAGNI: only the fields definitely needed by forward + speculator are present.
pub struct Qwen3DrafterAssets {
    /// Drafter model config (n_layers=5, dim=4096, hidden=12288, n_heads=32,
    /// n_kv_heads=8, head_dim=128, has_qk_norm=true, rope_theta=1e6).
    pub config: LlamaConfig,
    /// Per-layer attention + FFN weights. Owned GPU tensors.
    pub weights: LlamaWeights,
    /// Block-only KvCache: F32, 5 layers, cap = block_size.  Reset per window.
    pub kv: KvCache,
    /// Single-token decode scratch.
    pub scratch: ForwardScratch,
    /// Block-parallel prefill scratch (block_size tokens × dim).
    pub pbs: PrefillBatchScratch,
}

// ── Public loader ─────────────────────────────────────────────────────────────

/// Load the Qwen3-8B DSpark sidecar into `(DsparkWeights, Qwen3DrafterAssets)`.
///
/// `source` is the already-opened sidecar HFQ.  The caller must call
/// `drop_mmap()` before calling this function (pread is used throughout to
/// avoid page-cache pressure on UMA).
///
/// Returns `None` when `dspark_block_size` is absent from the sidecar metadata
/// (i.e. the file is not a DSpark sidecar).  Returns `Err` on tensor load
/// failures.
pub fn load_qwen3_dspark(
    source: &HfqFile,
    gpu: &mut Gpu,
) -> Result<Option<(DsparkWeights, Qwen3DrafterAssets)>, String> {
    // 1. Parse DSpark config — includes dspark_enable_confidence (hard req #2)
    let dspark_cfg = match DsparkConfig::from_metadata_json(&source.metadata_json) {
        Some(c) => c,
        None => return Ok(None),
    };

    // 2. Derive drafter LlamaConfig from tensor shapes.
    //    The sidecar metadata only carries dspark_* keys (no model_type /
    //    hidden_size etc.), so config_from_hfq would fail on a missing
    //    `model_type` field.  Derive the config from tensor shapes instead.
    let mut cfg = config_from_sidecar_tensors(source)
        .map_err(|e| format!("qwen3_dspark: derive config: {e}"))?;
    // config_from_sidecar_tensors hardcodes rope θ=1e6 (qwen3-8B). Qwen3.5's
    // drafter uses 1e7 — take it from the sidecar metadata (defaults to 1e6 for
    // legacy sidecars, so qwen3-8B stays byte-identical).
    cfg.rope_freq_base = dspark_cfg.rope_theta;

    let q_out_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // 3. Load 5-layer drafter body
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(load_drafter_layer(source, gpu, &cfg, i, q_out_dim, kv_dim)?);
    }

    // 4. Embedding table (embed_tokens.weight, qt=1 F16 → F32 EmbeddingFormat::F32)
    let (token_embd, embd_format) = {
        let (ei, ed) = source
            .tensor_data_pread("embed_tokens.weight")
            .ok_or_else(|| "qwen3_dspark: embed_tokens.weight missing".to_string())?;
        let qt = ei.quant_type;
        load_embedding(gpu, qt, &ed, cfg.vocab_size, cfg.dim)
            .map_err(|e| format!("qwen3_dspark: embed_tokens: {e:?}"))?
    };

    // 5. Final norm (norm.weight → F32)
    let output_norm = {
        let (ni, nd) = source
            .tensor_data_pread("norm.weight")
            .ok_or_else(|| "qwen3_dspark: norm.weight missing".to_string())?;
        let qt = ni.quant_type;
        dequant_norm(gpu, qt, &nd, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: norm.weight: {e:?}"))?
    };

    // 6. lm_head.weight (qt=1 F16). Reduced-vocab drafters (EAGLE-3, e.g. qwen35
    //    ORNITH) emit a compressed draft vocab, so the lm_head has
    //    draft_vocab_size rows, not the full embed vocab. 0 ⇒ shared full vocab.
    let draft_vocab = if dspark_cfg.draft_vocab_size > 0 {
        dspark_cfg.draft_vocab_size
    } else {
        cfg.vocab_size
    };
    let lm_head = load_global_proj(source, gpu, "lm_head.weight", draft_vocab, cfg.dim)?;

    let weights = LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output: lm_head,
        layers,
        lm_head_aliases_embd: false,
    };

    // 7. DSpark globals
    //    main_proj: [dim, n_targets * dim] F16 on GPU
    let main_proj = Some(load_global_tensor(source, gpu, "main_proj.weight")?);

    //    main_norm: [dim] F32
    let main_norm = {
        let (mi, md) = source
            .tensor_data_pread("main_norm.weight")
            .ok_or_else(|| "qwen3_dspark: main_norm.weight missing".to_string())?;
        let qt = mi.quant_type;
        dequant_norm(gpu, qt, &md, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: main_norm.weight: {e:?}"))?
    };

    //    markov_w1/w2: [vocab, rank] F16
    let markov_w1 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w1.weight",
    )?);
    let markov_w2 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w2.weight",
    )?);

    //    confidence_head.proj.weight: [1, dim+rank] F16
    let confidence_proj = if dspark_cfg.enable_confidence {
        Some(load_global_tensor(
            source,
            gpu,
            "confidence_head.proj.weight",
        )?)
    } else {
        None
    };

    //    confidence_head.proj.bias: [1] F16 → F32 — hard req #1 (qwen3 has bias)
    let confidence_bias = if dspark_cfg.enable_confidence {
        let bias_gpu = {
            let (bi, bd) = source
                .tensor_data_pread("confidence_head.proj.bias")
                .ok_or_else(|| "qwen3_dspark: confidence_head.proj.bias missing".to_string())?;
            let qt = bi.quant_type;
            dequant_f32(gpu, qt, &bd, 1)
                .map_err(|e| format!("qwen3_dspark: confidence_head.proj.bias: {e:?}"))?
        };
        Some(bias_gpu)
    } else {
        None
    };

    //    d2t: reduced-vocab draft→target token map, stored by the quantizer as
    //    F32 (exact for token ids < 2^24). None ⇒ shared full vocab (qwen3-8B).
    let d2t: Option<Vec<u32>> = if dspark_cfg.draft_vocab_size > 0 {
        let (di, dd) = source
            .tensor_data_pread("d2t")
            .ok_or_else(|| "qwen3_dspark: d2t missing but draft_vocab_size>0".to_string())?;
        let dev = dequant_f32(gpu, di.quant_type, &dd, dspark_cfg.draft_vocab_size)
            .map_err(|e| format!("qwen3_dspark: d2t dequant: {e:?}"))?;
        let host = gpu
            .download_f32(&dev)
            .map_err(|e| format!("qwen3_dspark: d2t download: {e:?}"))?;
        let _ = gpu.free_tensor(dev);
        Some(host.iter().map(|&v| v as u32).collect())
    } else {
        None
    };

    // qwen3 reference modeling.py feeds once-normed hidden (self.norm(hidden))
    // to predict_confidence_step; set the flag so run_heads uses normed[i].
    // Also pin rms_norm_eps from the derived drafter config (1e-6 for qwen3).
    let mut qwen3_cfg = dspark_cfg.clone();
    qwen3_cfg.confidence_uses_normed = true;
    qwen3_cfg.rms_norm_eps = cfg.norm_eps;

    let dspark_weights = DsparkWeights {
        cfg: qwen3_cfg,
        main_proj,
        main_norm: Some(main_norm),
        markov_w1,
        markov_w2,
        confidence_proj,
        confidence_bias,
        d2t,
    };

    // 8. Allocate drafter KvCache (block-only: cap = block_size tokens)
    let block_cap = dspark_cfg.block_size;
    let kv = KvCache::new_gpu(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, block_cap)
        .map_err(|e| format!("qwen3_dspark: KvCache::new_gpu: {e:?}"))?;

    // 9. ForwardScratch (single-token decode)
    let scratch = ForwardScratch::new(gpu, &cfg)
        .map_err(|e| format!("qwen3_dspark: ForwardScratch::new: {e:?}"))?;

    // 10. PrefillBatchScratch (block-parallel forward, max_batch = block_size)
    let pbs = PrefillBatchScratch::new(gpu, &cfg, block_cap, block_cap)
        .map_err(|e| format!("qwen3_dspark: PrefillBatchScratch::new: {e:?}"))?;

    let assets = Qwen3DrafterAssets {
        config: cfg,
        weights,
        kv,
        scratch,
        pbs,
    };

    Ok(Some((dspark_weights, assets)))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Load one drafter body layer from the flat-name sidecar.
///
/// Delegates to `hipfire_runtime::hfq::load_layer` via an `HfqBackend`
/// configured with `bare_name_candidates` so it resolves `layers.N.*`
/// without the `model.` prefix that `flat_name_candidates` would prepend.
fn load_drafter_layer(
    source: &HfqFile,
    gpu: &mut Gpu,
    cfg: &LlamaConfig,
    i: usize,
    q_out_dim: usize,
    kv_dim: usize,
) -> Result<LayerWeights, String> {
    let mut b = HfqBackend {
        hfq: source,
        gpu,
        norm_bias: 0.0,
        candidates: bare_name_candidates,
        read_proj: load_weight_tensor_pread,
        layer: i,
    };
    load_layer(&mut b, cfg, q_out_dim, kv_dim, i)
        .map_err(|e| format!("qwen3_dspark layer {i}: {e:?}"))
}

/// Upload a global weight tensor as `GpuTensor` (F16 kept as F16, MQ4 as
/// Raw/Q8_0 etc.).  Used for DSpark globals consumed by dspark_core.
fn load_global_tensor(source: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let (shape, qt, bytes) = {
        let (info, bytes) = source
            .tensor_data_pread(name)
            .ok_or_else(|| format!("qwen3_dspark: {name} missing"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let qt = info.quant_type;
        // Copy shape/qt before Ref<Vec<u8>> is consumed; bytes moves here.
        (shape, qt, bytes)
    };
    let mut t = gpu
        .upload_raw(&bytes, &shape)
        .map_err(|e| format!("qwen3_dspark: upload {name}: {e:?}"))?;
    if qt == 1 {
        t.dtype = DType::F16;
    }
    Ok(t)
}

/// Load a global projection as `WeightTensor` (for lm_head.weight).
fn load_global_proj(
    source: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = read_first(source, name, bare_name_candidates)
        .ok_or_else(|| format!("qwen3_dspark: {name} missing"))?;
    let mut wt = dequant_weight_raw(gpu, info.quant_type, &data, m, k)
        .map_err(|e| format!("qwen3_dspark: {name}: {e:?}"))?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale_for(source, gpu, name, k);
    }
    Ok(wt)
}

/// Derive a `LlamaConfig` from the sidecar tensor index.
///
/// The DSpark qwen3 sidecar metadata only carries `dspark_*` keys — it has no
/// `model_type`/`hidden_size`/etc., so `config_from_hfq` fails.  We derive
/// the config from tensor shapes instead.  The qwen3-8b drafter is always a
/// dense-GQA transformer, so the derivation is exact.
fn config_from_sidecar_tensors(source: &HfqFile) -> Result<LlamaConfig, String> {
    // ── dim from embed_tokens.weight ─────────────────────────────────────────
    let embed = source
        .find_tensor_info("embed_tokens.weight")
        .ok_or_else(|| "embed_tokens.weight missing".to_string())?;
    if embed.shape.len() < 2 {
        return Err(format!(
            "embed_tokens.weight unexpected shape {:?}",
            embed.shape
        ));
    }
    let vocab_size = embed.shape[0] as usize;
    let dim = embed.shape[1] as usize;

    // ── head_dim from q_norm.weight ───────────────────────────────────────────
    let q_norm = source
        .find_tensor_info("layers.0.self_attn.q_norm.weight")
        .ok_or_else(|| "layers.0.self_attn.q_norm.weight missing".to_string())?;
    let head_dim = q_norm.shape.first().copied().unwrap_or(128) as usize;
    let has_qk_norm = true; // presence of q_norm.weight confirms it

    // ── n_heads from q_proj.weight [q_out_dim, dim] ──────────────────────────
    let wq = source
        .find_tensor_info("layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| "layers.0.self_attn.q_proj.weight missing".to_string())?;
    let q_out_dim = wq.shape[0] as usize;
    let n_heads = q_out_dim / head_dim;

    // ── n_kv_heads from k_proj.weight [kv_out_dim, dim] ──────────────────────
    let wk = source
        .find_tensor_info("layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| "layers.0.self_attn.k_proj.weight missing".to_string())?;
    let kv_out_dim = wk.shape[0] as usize;
    let n_kv_heads = kv_out_dim / head_dim;

    // ── hidden_dim from gate_proj.weight [hidden_dim, dim] ───────────────────
    let wg = source
        .find_tensor_info("layers.0.mlp.gate_proj.weight")
        .ok_or_else(|| "layers.0.mlp.gate_proj.weight missing".to_string())?;
    let hidden_dim = wg.shape[0] as usize;

    // ── n_layers: probe layers.{N}.input_layernorm.weight until absent ────────
    let mut n_layers = 0usize;
    while source
        .find_tensor_info(&format!("layers.{n_layers}.input_layernorm.weight"))
        .is_some()
    {
        n_layers += 1;
    }
    if n_layers == 0 {
        return Err("qwen3_dspark: no body layers found (layers.0.* absent)".into());
    }

    Ok(LlamaConfig {
        arch: ModelArch::Qwen3,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps: 1e-6,              // qwen3 standard
        max_seq_len: 1024,           // drafter; actual cap = block_size (set by KvCache)
        rope_freq_base: 1_000_000.0, // qwen3 rope θ = 1e6
        bos_token: 1,
        eos_token: 2,
        has_qk_norm,
    })
}

// ── Block-attention body forward ──────────────────────────────────────────────

/// GPU scratch buffers for [`dspark_qwen3_block_forward`].
///
/// Allocated once per model load (sized to `max_ctx_len + block_size`).
/// Reset is implicit: every call re-embeds `block_ids` from scratch, so no
/// state carries over.
///
/// Buffer sizing (qwen3-8b defaults: dim=4096, n_heads=32, n_kv_heads=8,
/// head_dim=128, hidden_dim=14336):
///   `q_dim = n_heads * head_dim = 4096`
///   `kv_dim = n_kv_heads * head_dim = 1024`
///   KV cache capacity = `max_ctx_len + block_size`
///
/// `max_ctx_len=1` reproduces the previous single-slot behaviour.
pub struct Qwen3DsparkScratch {
    /// Maximum context length this scratch can handle.  Calls to
    /// [`dspark_qwen3_block_forward`] must pass `ctx_positions.len() <=
    /// max_ctx_len`.
    pub max_ctx_len: usize,

    /// Q8_0 KV cache (5 drafter layers, capacity = max_ctx_len + block_size).
    /// Layout: context K/V at compact slots 0..ctx_len; block K/V at
    /// slots ctx_len..ctx_len+block.  Compact slots decouple absolute RoPE
    /// positions from KV write positions.
    pub kv: KvCache,

    /// Block-parallel scratch: x_batch[block×dim], fa_q/k/v[block×*], etc.
    /// Reuses PrefillBatchScratch so layer-loop kernels use the same buffers as
    /// `forward_prefill_chunk` (fa_q_batch, x_rot_batch, …).
    pub pbs: PrefillBatchScratch,

    /// Concatenated [ctx(ctx_len) ++ block(block)] K buffer
    /// [(max_ctx_len+block)×kv_dim] F32.
    /// Used to apply k_norm to the full combined K sequence before KV write
    /// (modeling.py:107–113 cats k_ctx+k_noise before applying k_norm).
    pub all_k: GpuTensor,

    /// Concatenated [ctx(ctx_len) ++ block(block)] V buffer
    /// [(max_ctx_len+block)×kv_dim] F32.
    /// V has no norm (modeling.py:114 just transposes), but is staged here for
    /// the batched Q8_0 KV-cache write.
    pub all_v: GpuTensor,

    /// KV positions for the combined [ctx ++ block] sequence,
    /// shape [max_ctx_len+block_size], as i32-in-F32.
    /// Set per-call to [ctx_pos[0], ..., ctx_pos[ctx_len-1],
    ///                   block_pos[0], ..., block_pos[block-1]].
    /// Used for:
    ///   1. RoPE on the concatenated K (modeling.py:116 applies RoPE to all k).
    ///   2. Q8_0 KV-cache write (kv_cache_write_q8_0_batched positions arg).
    pub positions_kv_all: GpuTensor,

    /// Block query RoPE positions [block_size] i32-in-F32.
    /// = [anchor_pos, anchor_pos+1, ..., anchor_pos+block-1].
    /// Matches Q positions from apply_rotary_pos_emb (cos[..., -q_len:, :]).
    pub positions_q_block: GpuTensor,

    /// Compact attention positions [block_size] i32-in-F32 =
    /// [ctx_len, ctx_len+1, ..., ctx_len+block-1].
    /// Passed as `positions` to `attention_q8_0_kv_batched_masked`: each block
    /// query row i uses compact slot ctx_len+i (KV was written at those slots),
    /// while context slots 0..ctx_len are always visible (they precede block_start).
    pub positions_compact: GpuTensor,

    /// Additive bias [block × block] F32 = 0.0 (bidirectional in-block mask).
    /// Combined with `block_start=ctx_len`, `block_cols=block` in the
    /// masked-attention kernel: all block queries attend to all block keys.
    /// (modeling.py:58 `self.is_causal = False`; `create_dspark_attention_mask`
    /// makes every block query see all block keys.)
    pub bias: GpuTensor,
}

impl Qwen3DsparkScratch {
    /// Allocate scratch for a drafter with the given config and `block_size`.
    ///
    /// `max_ctx_len` is the maximum number of context slots this scratch can
    /// handle.  Pass `1` for the original single-slot behaviour.  The KV cache
    /// capacity is `max_ctx_len + block_size`.
    pub fn new(
        gpu: &mut Gpu,
        config: &LlamaConfig,
        block_size: usize,
        max_ctx_len: usize,
    ) -> Result<Self, String> {
        let max_ctx_len = max_ctx_len.max(1);
        let kv_cap = max_ctx_len + block_size;
        let kv = KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_cap,
        )
        .map_err(|e| format!("Qwen3DsparkScratch: kv: {e:?}"))?;

        let pbs = PrefillBatchScratch::new(gpu, config, block_size, kv_cap)
            .map_err(|e| format!("Qwen3DsparkScratch: pbs: {e:?}"))?;

        let kv_dim = config.n_kv_heads * config.head_dim;

        let all_k = gpu
            .alloc_tensor(&[kv_cap * kv_dim], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: all_k: {e:?}"))?;
        let all_v = gpu
            .alloc_tensor(&[kv_cap * kv_dim], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: all_v: {e:?}"))?;
        let positions_kv_all = gpu
            .alloc_tensor(&[kv_cap], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_kv_all: {e:?}"))?;
        let positions_q_block = gpu
            .alloc_tensor(&[block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_q_block: {e:?}"))?;
        let positions_compact = gpu
            .alloc_tensor(&[block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_compact: {e:?}"))?;
        let bias = gpu
            .zeros(&[block_size * block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: bias: {e:?}"))?;

        Ok(Self {
            max_ctx_len,
            kv,
            pbs,
            all_k,
            all_v,
            positions_kv_all,
            positions_q_block,
            positions_compact,
            bias,
        })
    }

    /// Release all GPU allocations.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        self.pbs.free_gpu(gpu);
        for t in [
            self.all_k,
            self.all_v,
            self.positions_kv_all,
            self.positions_q_block,
            self.positions_compact,
            self.bias,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

// ── dspark_qwen3_block_forward ─────────────────────────────────────────────────

/// Qwen3-8B DSpark block-attention forward: 5-layer dense GQA over the
/// bidirectional `[context(ctx_len) ++ block(N)]` KV set.
///
/// # Numeric contract (verified against modeling.py)
///
/// ## `main_x` context: `[ctx_len, dim]`, shared across all 5 layers
///
/// Caller computes `main_x[j] = hidden_norm(fc(main_hidden[j]))` per context
/// slot (modeling.py:373, applied over the full ctx_len batch).
/// Each layer re-uses the same `main_x` to form its context K/V via this
/// layer's `k_proj`/`v_proj` (modeling.py:103–106).
/// `ctx_len=1` reproduces the single-slot forward from Task 9.
///
/// ## Per-layer op sequence (modeling.py:181–198, 99–151)
///
/// ```text
/// 1. input_layernorm(x_block)      [modeling.py:181]
/// 2. q_proj(normed_block)          [modeling.py:99]
/// 3. q_norm(q, per-head)           [modeling.py:102  — BEFORE RoPE]
/// 4. k_proj(main_x[j]) → ctx_k[j] for j in 0..ctx_len  [modeling.py:103]
/// 5. k_proj(normed_block) → blk_k [modeling.py:104]
/// 6. cat([ctx_k, blk_k]) → all_k  [modeling.py:107]
/// 7. k_norm(all_k, per-head)       [modeling.py:113 — on full (ctx_len+block) K, BEFORE RoPE]
/// 8. v_proj(main_x[j]) → ctx_v[j] for j in 0..ctx_len  [modeling.py:105]
/// 9. v_proj(normed_block) → blk_v [modeling.py:106]
/// 10. cat([ctx_v, blk_v]) → all_v  [modeling.py:110]
/// 11. RoPE(q at block_positions; all_k at [ctx_positions ++ block_positions])
///          [modeling.py:116; apply_rotary_pos_emb:34–40]
/// 12. Write all_k, all_v to Q8 KV cache at compact slots 0..ctx_len+block
/// 13. attention_q8_0_kv_batched_masked:
///          positions_compact=[ctx_len..ctx_len+block], block_start=ctx_len,
///          block_cols=block, bias=zeros → bidirectional
///          [modeling.py:58 `is_causal=False`]
/// 14. o_proj(attn_out) + residual  [modeling.py:193–194]
/// 15. post_attention_layernorm(x_block)  [modeling.py:196]
/// 16. MLP(gate/up SwiGLU) + residual    [modeling.py:197–198]
/// ```
///
/// ## RoPE position assignment
///
/// `apply_rotary_pos_emb` (modeling.py:34–40) takes `cos/sin` shaped
/// `[ctx_len+block, head_dim]` computed from
/// `full_position_ids = [ctx_positions[0], ..., ctx_positions[ctx_len-1],
///                        block_positions[0], ..., block_positions[block-1]]`.
///
/// For Q it uses the LAST `q_len=block` entries
/// (`cos[..., -q_len:, :]`) → `block_positions`.
/// For K it uses the full `ctx_len + block` entries.
///
/// `block_positions[i] = anchor_pos + i` (0-indexed), where `anchor_pos` is
/// the anchor absolute position (= ctx_positions[ctx_len-1]+1 in typical use,
/// but the caller sets both explicitly). Derived from `create_position_ids`.
///
/// ## Bidirectional mask
///
/// `attention_q8_0_kv_batched_masked` with `block_start=ctx_len`,
/// `block_cols=block`, `bias=zeros[block×block]` gives every block query full
/// visibility of all in-block keys.  Slots 0..ctx_len (context) are before
/// `block_start` → always visible.
///
/// # Arguments
///
/// * `drafter`       — 5-layer Qwen3-8B body weights (LlamaWeights).
/// * `config`        — `n_layers=5`, `has_qk_norm=true`, `rope_freq_base=1e6`.
/// * `main_x`        — `[ctx_len * dim]` F32 context rows (per-slot output of
///                     `hidden_norm(fc(main_hidden))`).
/// * `ctx_positions` — absolute RoPE positions for the `ctx_len` context rows.
///                     Length must equal `ctx_len = main_x.shape[0] / dim`.
/// * `block_ids`     — `[block]` token ids: `[seed_token, noise, noise, ...]`.
/// * `block_positions` — absolute RoPE positions for the `block` query/key rows.
///                       Length must equal `block`.
/// * `block`         — number of block slots (= block_size in practice).
/// * `scratch`       — pre-allocated [`Qwen3DsparkScratch`] with
///                     `max_ctx_len >= ctx_positions.len()`.
/// * `x_head_out`    — `[block × dim]` F32 output (pre-final-norm hidden states).
///                     Callers (e.g. `run_heads`) apply `stage_norm` exactly once.
pub fn dspark_qwen3_block_forward(
    gpu: &mut Gpu,
    drafter: &LlamaWeights,
    config: &LlamaConfig,
    main_x: &GpuTensor,
    ctx_positions: &[usize],
    block_ids: &[u32],
    block_positions: &[usize],
    block: usize,
    scratch: &Qwen3DsparkScratch,
    x_head_out: &GpuTensor,
    // <1.0 ⇒ partial-interleaved RoPE (Qwen3.5, n_rot = head_dim·factor);
    // 1.0 ⇒ full rotary (qwen3-8B, byte-identical rope_batched_f32).
    partial_rotary_factor: f32,
) -> Result<(), String> {
    let ctx_len = ctx_positions.len();
    debug_assert_eq!(block_ids.len(), block);
    debug_assert_eq!(block_positions.len(), block);
    debug_assert!(ctx_len >= 1, "ctx_len must be >= 1");
    debug_assert!(
        ctx_len <= scratch.max_ctx_len,
        "ctx_len {ctx_len} > scratch.max_ctx_len {}",
        scratch.max_ctx_len
    );
    debug_assert!(
        block <= scratch.pbs.max_batch,
        "block {block} > pbs.max_batch"
    );

    let dim = config.dim;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let kv_cap = ctx_len + block; // compact slots: 0..ctx_len=ctx, ctx_len..kv_cap=block

    // ── 0. Upload positions ────────────────────────────────────────────────────
    //
    // full_position_ids (modeling.py training):
    //   [ctx_positions[0..ctx_len], block_positions[0..block]]
    //
    // apply_rotary_pos_emb (modeling.py:34–40):
    //   K uses the full kv_cap positions.
    //   Q uses the LAST block entries (cos[..., -q_len:, :]).
    //   → positions_q_block = block_positions.

    // positions_kv_all = [ctx_positions ++ block_positions] (kv_cap entries)
    {
        let pos: Vec<i32> = ctx_positions
            .iter()
            .chain(block_positions.iter())
            .map(|&p| p as i32)
            .collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, kv_cap * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_kv_all.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_kv_all: {e:?}"))?;
    }

    // positions_q_block = block_positions (block entries: Q positions)
    {
        let pos: Vec<i32> = block_positions.iter().map(|&p| p as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, block * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_q_block.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_q_block: {e:?}"))?;
    }

    // positions_compact = [ctx_len, ctx_len+1, ..., ctx_len+block-1]
    // (compact KV-cache slots for the block queries)
    {
        let pos: Vec<i32> = (ctx_len as i32..(ctx_len + block) as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, block * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_compact.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_compact: {e:?}"))?;
    }

    // ── 1. Embed block_ids → pbs.x_batch  ─────────────────────────────────────
    //
    // Embed each token into pbs.x_batch row i.
    // drafter.embd_format is F32 (qt=1 F16 was dequantized in the loader).
    // sub_offset takes offset in ELEMENTS (not bytes); pbs.x_batch is F32.
    for (i, &tok) in block_ids.iter().enumerate() {
        let x_row = scratch.pbs.x_batch.sub_offset(i * dim, dim);
        gpu.embedding_lookup(&drafter.token_embd, &x_row, tok, dim)
            .map_err(|e| format!("dspark_qwen3: embed[{i}]: {e:?}"))?;
    }

    // ── 2. Per-layer loop ×5 ───────────────────────────────────────────────────

    for layer_idx in 0..config.n_layers {
        let layer = &drafter.layers[layer_idx];

        // ── 2a. input_layernorm(x_batch) → x_rot_batch  ───────────────────────
        // modeling.py:181  `residual = hidden_states; hidden_states = input_layernorm(hidden_states)`
        gpu.rmsnorm_batched(
            &scratch.pbs.x_batch,
            &layer.attn_norm,
            &scratch.pbs.x_rot_batch,
            block,
            dim,
            config.norm_eps,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: attn_norm: {e:?}"))?;

        // ── 2b. Q projection: wq(normed_block) → fa_q_batch  ──────────────────
        // modeling.py:99   `q = self.q_proj(hidden_states).view(...)`
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let q_row = scratch.pbs.fa_q_batch.sub_offset(i * q_dim, q_dim);
            weight_gemv(gpu, &layer.wq, &x_row, &q_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: q_proj[{i}]: {e:?}"))?;
        }

        // ── 2c. q_norm(q, per-head) — BEFORE RoPE  ────────────────────────────
        // modeling.py:102  `q = self.q_norm(q).transpose(1, 2)`
        if let Some(ref qn) = layer.q_norm {
            gpu.rmsnorm_batched(
                &scratch.pbs.fa_q_batch,
                qn,
                &scratch.pbs.fa_q_batch,
                block * config.n_heads,
                config.head_dim,
                config.norm_eps,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: q_norm: {e:?}"))?;
        }

        // ── 2d. Context K/V (ctx_len rows) + block K/V → all_k, all_v  ─────────
        // modeling.py:103  `k_ctx  = self.k_proj(target_hidden_states)` (ctx_len rows)
        // modeling.py:104  `k_noise = self.k_proj(hidden_states)`        (block rows)
        // modeling.py:107  `k = cat([k_ctx, k_noise], dim=1)` → all_k[0..kv_cap]
        // modeling.py:110  `v = cat([v_ctx, v_noise], dim=1)` → all_v[0..kv_cap]

        // Context K at slots 0..ctx_len of all_k.
        for j in 0..ctx_len {
            let mx_row = main_x.sub_offset(j * dim, dim);
            let k_row = scratch.all_k.sub_offset(j * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wk, &mx_row, &k_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_proj(ctx[{j}]): {e:?}"))?;
        }

        // Block K at slots ctx_len..ctx_len+block of all_k.
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let k_row = scratch.all_k.sub_offset((ctx_len + i) * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wk, &x_row, &k_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_proj[{i}]: {e:?}"))?;
        }

        // Context V at slots 0..ctx_len of all_v.
        for j in 0..ctx_len {
            let mx_row = main_x.sub_offset(j * dim, dim);
            let v_row = scratch.all_v.sub_offset(j * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wv, &mx_row, &v_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: v_proj(ctx[{j}]): {e:?}"))?;
        }

        // Block V at slots ctx_len..ctx_len+block of all_v.
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let v_row = scratch.all_v.sub_offset((ctx_len + i) * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wv, &x_row, &v_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: v_proj[{i}]: {e:?}"))?;
        }

        // ── 2e. k_norm(all_k) — on concatenated [ctx ++ block] K, BEFORE RoPE ─
        // modeling.py:113  `k = self.k_norm(k).transpose(1, 2)`
        // all_k is [kv_cap × kv_dim] laid out as [kv_cap*n_kv_heads] rows of
        // [head_dim] each → rmsnorm_batched treats it as that many rows.
        if let Some(ref kn) = layer.k_norm {
            gpu.rmsnorm_batched(
                &scratch.all_k,
                kn,
                &scratch.all_k,
                kv_cap * config.n_kv_heads,
                config.head_dim,
                config.norm_eps,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_norm: {e:?}"))?;
        }

        // ── 2f. RoPE on Q (block positions) and K (all kv_cap positions)  ──────
        // modeling.py:116  `q, k = apply_rotary_pos_emb(q, k, cos, sin)`
        // apply_rotary_pos_emb (modeling.py:34–40):
        //   q uses cos[..., -q_len:, :]  → block_positions (last block entries)
        //   k uses full cos              → [ctx_positions ++ block_positions]

        // Qwen3.5 rotates only n_rot = head_dim·partial_rotary_factor dims
        // (partial-interleaved/halfsplit, matching the qwen35 target forward);
        // qwen3-8B rotates the full head_dim (factor 1.0 → rope_batched_f32,
        // byte-identical). pos_offset=0: the drafter's block-only KV never compacts.
        let use_partial = partial_rotary_factor < 1.0;
        let n_rot = (config.head_dim as f32 * partial_rotary_factor) as usize;

        // RoPE on Q (only): n_heads_k=0 skips K rotation.
        if use_partial {
            gpu.rope_partial_interleaved_f32_batched(
                &scratch.pbs.fa_q_batch,
                &scratch.all_k,
                &scratch.positions_q_block,
                config.n_heads,
                0,
                config.head_dim,
                n_rot,
                config.rope_freq_base,
                block,
                0,
            )
        } else {
            gpu.rope_batched_f32(
                &scratch.pbs.fa_q_batch,
                &scratch.all_k, // dummy k (n_heads_k=0 → not modified)
                &scratch.positions_q_block,
                config.n_heads,
                0, // n_heads_k=0 → skip K
                config.head_dim,
                config.rope_freq_base,
                block,
            )
        }
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: rope Q: {e:?}"))?;

        // RoPE on K (only): n_heads_q=0 skips Q rotation.
        if use_partial {
            gpu.rope_partial_interleaved_f32_batched(
                &scratch.pbs.fa_q_batch,
                &scratch.all_k,
                &scratch.positions_kv_all,
                0,
                config.n_kv_heads,
                config.head_dim,
                n_rot,
                config.rope_freq_base,
                kv_cap,
                0,
            )
        } else {
            gpu.rope_batched_f32(
                &scratch.pbs.fa_q_batch, // dummy q (n_heads_q=0 → not modified)
                &scratch.all_k,
                &scratch.positions_kv_all,
                0, // n_heads_q=0 → skip Q
                config.n_kv_heads,
                config.head_dim,
                config.rope_freq_base,
                kv_cap, // batch = ctx_len + block
            )
        }
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: rope K: {e:?}"))?;

        // ── 2g. Write K and V to Q8 KV cache at compact slots 0..kv_cap  ───────
        // Write context K/V (slots 0..ctx_len) first, then block K/V
        // (slots ctx_len..ctx_len+block) using positions_compact.

        // Context K/V: compact slots 0..ctx_len.
        // Upload compact positions [0, 1, ..., ctx_len-1] into pbs.positions.
        {
            let ctx_compact: Vec<i32> = (0..ctx_len as i32).collect();
            let ctx_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(ctx_compact.as_ptr() as *const u8, ctx_len * 4)
            };
            gpu.hip
                .memcpy_htod_offset(&scratch.pbs.positions.buf, 0, ctx_bytes)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: htod ctx compact pos: {e:?}"))?;

            let ctx_k_slice = scratch.all_k.sub_offset(0, ctx_len * kv_dim);
            let ctx_v_slice = scratch.all_v.sub_offset(0, ctx_len * kv_dim);
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.k_gpu[layer_idx],
                &ctx_k_slice,
                &scratch.pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                ctx_len,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_k_ctx: {e:?}"))?;
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.v_gpu[layer_idx],
                &ctx_v_slice,
                &scratch.pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                ctx_len,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_v_ctx: {e:?}"))?;
        }

        // Block K/V: compact slots ctx_len..ctx_len+block.
        {
            let blk_k = scratch.all_k.sub_offset(ctx_len * kv_dim, block * kv_dim);
            let blk_v = scratch.all_v.sub_offset(ctx_len * kv_dim, block * kv_dim);
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.k_gpu[layer_idx],
                &blk_k,
                &scratch.positions_compact,
                config.n_kv_heads,
                config.head_dim,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_k_blk: {e:?}"))?;
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.v_gpu[layer_idx],
                &blk_v,
                &scratch.positions_compact,
                config.n_kv_heads,
                config.head_dim,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_v_blk: {e:?}"))?;
        }

        // ── 2h. Bidirectional masked GQA attention  ────────────────────────────
        // positions_compact = [ctx_len..ctx_len+block] (block query compact slots).
        // block_start=ctx_len, block_cols=block → all block queries see all block keys.
        // Slots 0..ctx_len (context) are before block_start → always visible.
        // modeling.py:58 `self.is_causal = False`.
        gpu.attention_q8_0_kv_batched_masked(
            &scratch.pbs.fa_q_batch,
            &scratch.kv.k_gpu[layer_idx],
            &scratch.kv.v_gpu[layer_idx],
            &scratch.pbs.fa_attn_out_batch,
            &scratch.positions_compact,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            scratch.kv.physical_cap, // max_seq = kv_cap
            kv_cap,                  // max_ctx_len = ctx_len + block (all keys visible)
            block,                   // batch_size = block query rows
            Some(&scratch.bias),     // zero bias → bidirectional in-block
            ctx_len,                 // block_start = ctx_len
            block,                   // block_cols = block
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: attn: {e:?}"))?;

        // ── 2i. o_proj(attn_out) + residual  ──────────────────────────────────
        // modeling.py:148–150  `attn_output = attn_output.reshape(...)` then `o_proj`
        // modeling.py:194      `hidden_states = residual + hidden_states`
        // Dispatch mirrors llama.rs:forward_prefill_batch_inner (lines 2761–2826):
        // Q8_0 weights use gemm_q8_0_residual_wmma (WMMA arch) or
        // gemm_q8_0_batched_chunked+add_inplace_f32 (non-WMMA); HFQ4G256 otherwise.
        let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        if wo_is_q8 && q8_wmma_arch {
            let x_n = scratch.pbs.x_batch.sub_offset(0, block * layer.wo.m);
            gpu.gemm_q8_0_residual_wmma(
                &layer.wo.buf,
                &scratch.pbs.fa_attn_out_batch,
                &x_n,
                layer.wo.m,
                layer.wo.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: o_proj (q8 wmma): {e:?}"))?;
        } else if wo_is_q8 {
            let tmp = scratch.pbs.x_rot_batch.sub_offset(0, block * layer.wo.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.wo.buf,
                &scratch.pbs.fa_attn_out_batch,
                &tmp,
                layer.wo.m,
                layer.wo.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: o_proj (q8 chunked): {e:?}"))?;
            let x_n = scratch.pbs.x_batch.sub_offset(0, block * layer.wo.m);
            gpu.add_inplace_f32(&x_n, &tmp)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: o_proj residual add: {e:?}"))?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.wo.buf,
                &scratch.pbs.fa_attn_out_batch,
                &scratch.pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: o_proj (hfq4): {e:?}"))?;
        }

        // ── 2j. post_attention_layernorm(x_batch) → x_rot_batch  ──────────────
        // modeling.py:196  `hidden_states = self.post_attention_layernorm(hidden_states)`
        gpu.rmsnorm_batched(
            &scratch.pbs.x_batch,
            &layer.ffn_norm,
            &scratch.pbs.x_rot_batch,
            block,
            dim,
            config.norm_eps,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: ffn_norm: {e:?}"))?;

        // ── 2k. MLP SwiGLU: gate/up → silu_mul → down + residual  ─────────────
        // modeling.py:197  `hidden_states = self.mlp(hidden_states)` (Qwen3MLP = SwiGLU)
        // modeling.py:198  `return residual + hidden_states`
        // Dispatch mirrors llama.rs:forward_prefill_batch_inner (lines 2838–2939):
        // Q8_0 → gemm_gate_up_q8_0_wmma (WMMA) or two gemm_q8_0_batched_chunked calls.
        let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
        if ffn_is_q8 && q8_wmma_arch {
            gpu.gemm_gate_up_q8_0_wmma(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.pbs.x_rot_batch,
                &scratch.pbs.gate_ffn_batch,
                &scratch.pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: gate_up (q8 wmma): {e:?}"))?;
        } else if ffn_is_q8 {
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_gate.buf,
                &scratch.pbs.x_rot_batch,
                &scratch.pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: gate (q8 chunked): {e:?}"))?;
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_up.buf,
                &scratch.pbs.x_rot_batch,
                &scratch.pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: up (q8 chunked): {e:?}"))?;
        } else {
            gpu.gemm_gate_up_hfq4g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &scratch.pbs.x_rot_batch,
                &scratch.pbs.gate_ffn_batch,
                &scratch.pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: gate_up (hfq4): {e:?}"))?;
        }

        gpu.silu_mul_f32(
            &scratch.pbs.gate_ffn_batch,
            &scratch.pbs.up_batch,
            &scratch.pbs.ffn_hidden_batch,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: silu_mul: {e:?}"))?;

        // Dispatch mirrors llama.rs:forward_prefill_batch_inner (lines 2947–3020):
        // Q8_0 → gemm_q8_0_residual_wmma (WMMA) or gemm_q8_0_batched_chunked+add_inplace.
        let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
        if w_down_is_q8 && q8_wmma_arch {
            let x_n = scratch.pbs.x_batch.sub_offset(0, block * layer.w_down.m);
            gpu.gemm_q8_0_residual_wmma(
                &layer.w_down.buf,
                &scratch.pbs.ffn_hidden_batch,
                &x_n,
                layer.w_down.m,
                layer.w_down.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: w_down (q8 wmma): {e:?}"))?;
        } else if w_down_is_q8 {
            let tmp = scratch
                .pbs
                .x_rot_batch
                .sub_offset(0, block * layer.w_down.m);
            gpu.gemm_q8_0_batched_chunked(
                &layer.w_down.buf,
                &scratch.pbs.ffn_hidden_batch,
                &tmp,
                layer.w_down.m,
                layer.w_down.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: w_down (q8 chunked): {e:?}"))?;
            let x_n = scratch.pbs.x_batch.sub_offset(0, block * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &tmp)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: w_down residual add: {e:?}"))?;
        } else {
            gpu.gemm_hfq4g256_residual(
                &layer.w_down.buf,
                &scratch.pbs.ffn_hidden_batch,
                &scratch.pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: w_down (hfq4): {e:?}"))?;
        }
    }

    // ── 3. Copy x_batch → x_head_out  ─────────────────────────────────────────
    // x_head_out carries the PRE-final-norm hidden states. The single final
    // RMSNorm (`stage_norm` = `output_norm`) is applied once downstream by
    // `run_heads`, matching modeling.py:386 `return self.norm(hidden_states)`
    // followed by `compute_logits(output_hidden) = lm_head(output_hidden)` —
    // no second norm between `_forward_backbone`'s return and `lm_head`.
    let n_bytes = block * dim * std::mem::size_of::<f32>();
    gpu.copy_d2d(&scratch.pbs.x_batch, x_head_out, n_bytes)
        .map_err(|e| format!("dspark_qwen3: x_batch → x_head_out copy: {e:?}"))?;

    Ok(())
}

// ── Qwen3DsparkBody impl DsparkBody ───────────────────────────────────────────

/// Arch-specific DSpark body for the 5-layer Qwen3-8B drafter.
///
/// Implements [`DsparkBody`] so that the arch-agnostic [`DsparkDrafter`]
/// (in `dspark_core`) can drive the Qwen3 block-attention forward without any
/// Qwen3-specific knowledge.
///
/// Ownership: the body owns the scratch buffers allocated at load time;
/// the weights live in [`Qwen3DrafterAssets`] which the body also owns.
pub struct Qwen3DsparkBody {
    assets: Qwen3DrafterAssets,
    scratch: Qwen3DsparkScratch,
}

impl DsparkBody for Qwen3DsparkBody {
    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &DsparkWeights,
        main_hidden: &GpuTensor, // [ctx_len * n_targets * dim] flat
        ctx_positions: &[usize], // absolute RoPE positions; len = ctx_len
        seed: u32,
        position: usize,
        block: usize,
        x_head_out: &GpuTensor, // [block, dim] out
    ) -> Result<(), String> {
        let dim = self.assets.config.dim;
        let ctx_len = ctx_positions.len().max(1);

        // ── 1. main_proj_ingest: fc(main_hidden) + main_norm → main_x  ────────
        // For ctx_len=1 use the scalar variant; for ctx_len>1 use the batched
        // variant which produces [ctx_len, dim] F32 in one call.
        let main_x = gpu
            .alloc_tensor(&[ctx_len * dim], DType::F32)
            .map_err(|e| format!("Qwen3DsparkBody: alloc main_x: {e:?}"))?;
        if ctx_len == 1 {
            main_proj_ingest(gpu, weights, main_hidden, &main_x)?;
        } else {
            main_proj_ingest_batched(gpu, weights, main_hidden, &main_x, ctx_len, dim)?;
        }

        // ── 2. block_ids = [seed, noise, noise, ...] ──────────────────────────
        let block_ids = noise_block_ids(&weights.cfg, seed);

        // ── 3. Block-attention forward → x_head_out ───────────────────────────
        // block_positions = [position, position+1, ..., position+block-1].
        // These are the block's absolute positions; the block token[0] is the
        // seed, and the drafts occupy positions [position+1 .. position+block].
        let block_positions: Vec<usize> = (0..block).map(|i| position + i).collect();
        dspark_qwen3_block_forward(
            gpu,
            &self.assets.weights,
            &self.assets.config,
            &main_x,
            ctx_positions,
            &block_ids,
            &block_positions,
            block,
            &self.scratch,
            x_head_out,
            weights.cfg.partial_rotary_factor,
        )?;

        let _ = gpu.free_tensor(main_x);
        Ok(())
    }

    fn block_size(&self) -> usize {
        // kv_cap = max_ctx_len + block_size; max_ctx_len = block_size + 1.
        // So kv_cap = 2 * block_size + 1 → block_size = (kv_cap - 1) / 2.
        // Use pbs.max_batch which was set to block_size directly at construction.
        self.scratch.pbs.max_batch
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        self.scratch.free_gpu(gpu);
        let Qwen3DrafterAssets {
            config: _,
            weights,
            kv,
            scratch,
            pbs,
        } = self.assets;
        weights.free_gpu(gpu);
        let _ = kv.free_gpu(gpu);
        scratch.free_gpu(gpu);
        pbs.free_gpu(gpu);
    }
}

/// Build the Qwen3-8B DSpark body from [`Qwen3DrafterAssets`].
///
/// Returns a `Box<dyn DsparkBody>` suitable for passing to
/// [`hipfire_runtime::dspark_core::build_dspark_speculator`].
///
/// Allocates the [`Qwen3DsparkScratch`] using `block_size` from
/// `DsparkWeights::cfg`. The scratch is sized for the multi-slot context
/// forward: `max_ctx_len = block_size + 1` so that the accepted-prefix of a
/// full-accept window (up to `block_size` accepted drafts + the seed = at most
/// `block_size + 1` slots) fits without reallocation.
pub fn build_qwen3_dspark_body(
    assets: Qwen3DrafterAssets,
    cfg: &DsparkConfig,
    gpu: &mut Gpu,
) -> Result<Box<dyn DsparkBody>, String> {
    let max_ctx_len = cfg.block_size + 1;
    let scratch = Qwen3DsparkScratch::new(gpu, &assets.config, cfg.block_size, max_ctx_len)
        .map_err(|e| format!("build_qwen3_dspark_body: scratch: {e}"))?;
    Ok(Box::new(Qwen3DsparkBody { assets, scratch }))
}
