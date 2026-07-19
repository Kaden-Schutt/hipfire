// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Numerical and state parity oracle for LFM2.5-350M dense MQ4 batched prefill.
//!
//! Cohort (exact):
//!   model     = `~/.hipfire/models/lfm2.5-350m.mq4`
//!   md5       = `cb5284b8ad5c6f9e4ca859c0aff0bcd0`
//!   arch_id   = 11 / gfx1201
//!   weights   = dense MQ4G256 projections (qt13 / 136 B group)
//!   embed/head= Q8
//!   KV        = Q8 only
//!   shape     = hidden1024, heads16/kv8/hd64, qdim1024, kvdim512, θ1e6,
//!               vocab65536, intermediate 4608 (cfg auto-adjust from HF 6656),
//!               16 layers hybrid conv/attn
//!   chunk     = default 256 (`HIPFIRE_LFM2_PREFILL_MAX_BATCH`, cap 512)
//!   flag      = `HIPFIRE_LFM2_PREFILL_BATCH=1`
//!
//! Matrix: prompt lengths `{1,2,3,127,128,255,256,257}` cover single-chunk
//! and the 257 = 256+1 two-chunk boundary.
//!
//! Frozen dense-MQ4 / A1B numeric gates (design §12):
//!   per-layer hidden row cosine              >= 0.999
//!   per-layer hidden row max-abs             <= 0.15
//!   dequantized K/V per layer/token cosine   >= 0.999
//!   dequantized K/V max-abs                  <= 0.15
//!   final conv-tail cosine per layer         >= 0.999
//!   final conv-tail max-abs                  <= 0.15
//!   final-token logits max-abs               <= 0.10
//!   KL(softmax(eager)||softmax(batched)) mean <= 5e-4
//!   same KL, max over prompts                <= 5e-3
//! plus exact discrete checks: `n_tokens`, absolute KV write positions
//! `0..N-1`, layer→conv_state_idx / layer→kv_idx bijective slot mapping,
//! and chunk coverage (`ceil(N / max_batch)` with default max_batch=256).
//! All compared values must be finite.
//!
//! # Compile dependency (core API — owned by the MQ4 core-batch lane)
//! This example compiles only after the core lane lands:
//!   * `hipfire_arch_lfm2moe::forward::forward_prefill_batch_capture`
//!     signature mirror of abandoned Q8 scaffold:
//!     `(cfg, weights, state, gpu, token_ids, start_pos, capture) -> Result<Vec<f32>, String>`
//!     where `capture[layer]` is appended with post-layer (or post-mixer if
//!     `HIPFIRE_LFM2_CAPTURE_POSTMIXER`) hidden rows laid out `[N * hidden]`.
//!   * Existing `decode_step_capture` (already exported).
//!   * Batched path gated on gfx1201 + `HIPFIRE_LFM2_PREFILL_BATCH=1` and
//!     exact 350M dense-MQ4 + Q8-KV admission.
//! No production sources are modified by this oracle.
//!
//! Usage:
//!   HIPFIRE_LFM2_PREFILL_BATCH=1 \
//!     cargo run -p hipfire-arch-lfm2moe --example prefill_batch_parity --release -- \
//!       [model.mq4]

use hipfire_arch_lfm2moe::config::{Lfm2MoeConfig, MixerKind};
use hipfire_arch_lfm2moe::forward::{decode_step_capture, forward_prefill_batch_capture};
use hipfire_arch_lfm2moe::lfm2moe::{
    Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, WeightTensor};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Exact oracle length matrix from the frozen architecture contract.
const LENGTHS: [usize; 8] = [1, 2, 3, 127, 128, 255, 256, 257];
/// Default chunk capacity — must match core `LFM2_PREFILL_MAX_BATCH`.
const DEFAULT_CHUNK_CAPACITY: usize = 256;
/// Verified artifact identity for the 350M dense MQ4 cohort.
const EXPECTED_MODEL_MD5: &str = "cb5284b8ad5c6f9e4ca859c0aff0bcd0";

// Frozen dense-MQ4 / A1B thresholds (design §12). Do not weaken.
const LOGIT_MAX_ABS_LIMIT: f32 = 0.10;
const KL_MEAN_LIMIT: f64 = 5e-4;
const KL_MAX_LIMIT: f64 = 5e-3;
const COSINE_LIMIT: f64 = 0.999;
const STATE_MAX_ABS_LIMIT: f32 = 0.15;
const MQ4_GROUP_ELEMS: usize = 256;
const MQ4_GROUP_BYTES: usize = 136; // qt13 HFQ4-layout / MQ4G256 (not Lloyd 160)

/// Frozen 350M hybrid mixer topology (conv/attn), length 16.
const MIXERS_350M: [MixerKind; 16] = [
    MixerKind::Conv,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
    MixerKind::Attention,
    MixerKind::Conv,
];

fn compute_kl(reference: &[f32], candidate: &[f32]) -> f64 {
    assert_eq!(reference.len(), candidate.len());
    let ref_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let cand_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let ref_sum: f64 = reference
        .iter()
        .map(|&value| ((value as f64) - ref_max).exp())
        .sum();
    let cand_sum: f64 = candidate
        .iter()
        .map(|&value| ((value as f64) - cand_max).exp())
        .sum();
    let log_ref_sum = ref_sum.ln() + ref_max;
    let log_cand_sum = cand_sum.ln() + cand_max;
    let mut kl = 0.0f64;
    for (&ref_value, &cand_value) in reference.iter().zip(candidate) {
        let log_p = ref_value as f64 - log_ref_sum;
        let p = log_p.exp();
        if p > 0.0 {
            let log_q = cand_value as f64 - log_cand_sum;
            kl += p * (log_p - log_q);
        }
    }
    if kl < 0.0 && kl > -1e-9 {
        0.0
    } else {
        kl
    }
}

fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f32) {
    assert_eq!(reference.len(), candidate.len());
    let mut dot = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut cand_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&a, &b) in reference.iter().zip(candidate) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite parity value (a={a}, b={b})"
        );
        dot += a as f64 * b as f64;
        ref_sq += (a as f64) * (a as f64);
        cand_sq += (b as f64) * (b as f64);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = if ref_sq == 0.0 && cand_sq == 0.0 {
        1.0
    } else {
        dot / (ref_sq.sqrt() * cand_sq.sqrt())
    };
    (cosine, max_abs)
}

fn download_prefix(gpu: &Gpu, tensor: &GpuTensor, bytes: usize) -> Vec<u8> {
    let mut host = vec![0u8; bytes];
    gpu.hip
        .memcpy_dtoh(&mut host, &tensor.buf)
        .expect("download KV prefix");
    host
}

fn dequant_q8(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 34, 0, "Q8_0 payload must be 34 B/block");
    let mut output = Vec::with_capacity((bytes.len() / 34) * 32);
    for block in bytes.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        output.extend(block[2..].iter().map(|&q| (q as i8) as f32 * scale));
    }
    output
}

fn file_md5(path: &Path) -> String {
    let output = Command::new("md5sum")
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("md5sum {}: {e}", path.display()));
    assert!(
        output.status.success(),
        "md5sum failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("empty md5sum for {}", path.display()))
        .to_owned()
}

fn mq4_row_bytes(k: usize) -> usize {
    assert!(
        k % MQ4_GROUP_ELEMS == 0,
        "MQ4G256 K={k} must be a multiple of {MQ4_GROUP_ELEMS}"
    );
    (k / MQ4_GROUP_ELEMS) * MQ4_GROUP_BYTES
}

fn require_mq4_weight(w: &WeightTensor, m: usize, k: usize, name: &str) {
    assert_eq!(
        w.gpu_dtype,
        DType::MQ4G256,
        "{name}: expected MQ4G256 (qt13/group_bytes{MQ4_GROUP_BYTES}), got {:?}",
        w.gpu_dtype
    );
    assert_eq!(w.m, m, "{name}: m mismatch");
    assert_eq!(w.k, k, "{name}: k mismatch");
    let expected_bytes = m
        .checked_mul(mq4_row_bytes(k))
        .unwrap_or_else(|| panic!("{name}: byte-size overflow"));
    assert_eq!(
        w.buf.buf.size(),
        expected_bytes,
        "{name}: MQ4G256 bytes {expected_bytes} (136 B/group), got {}",
        w.buf.buf.size()
    );
}

fn require_q8_weight(w: &WeightTensor, m: usize, k: usize, name: &str) {
    assert_eq!(
        w.gpu_dtype,
        DType::Q8_0,
        "{name}: expected Q8_0, got {:?}",
        w.gpu_dtype
    );
    assert_eq!(w.m, m, "{name}: m mismatch");
    assert_eq!(w.k, k, "{name}: k mismatch");
    let expected_bytes = m
        .checked_mul(k / 32)
        .and_then(|blocks| blocks.checked_mul(34))
        .unwrap_or_else(|| panic!("{name}: Q8 byte-size overflow"));
    assert_eq!(
        w.buf.buf.size(),
        expected_bytes,
        "{name}: Q8 bytes {expected_bytes}, got {}",
        w.buf.buf.size()
    );
}

/// Exact discrete topology: bijective layer→conv ring and layer→KV slot maps,
/// dense-MQ4 projections, Q8 embed/head, Q8 KV, frozen 350M shape.
fn assert_cohort_and_slot_mapping(cfg: &Lfm2MoeConfig, weights: &Lfm2MoeWeights, state: &Lfm2MoeState) {
    assert_eq!(cfg.hidden_size, 1024, "350M hidden");
    assert_eq!(cfg.vocab_size, 65_536, "350M vocab");
    assert_eq!(cfg.num_attention_heads, 16, "350M n_heads");
    assert_eq!(cfg.num_key_value_heads, 8, "350M n_kv");
    assert_eq!(cfg.head_dim, 64, "350M head_dim");
    assert_eq!(cfg.q_dim(), 1024);
    assert_eq!(cfg.kv_dim(), 512);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.intermediate_size, 4608);
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.num_experts, 0, "350M is dense (no MoE)");
    assert_eq!(cfg.num_dense_layers, 16);
    assert_eq!(cfg.conv_kernel_size, 3);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.layer_types.as_slice(), MIXERS_350M.as_slice());
    assert_eq!(cfg.num_attention_layers(), 6);
    assert_eq!(cfg.num_conv_layers(), 10);

    assert!(
        state.kv.quant_q8,
        "KV must be Q8 (HFQ_Q8_ONLY_POLICY); quant_q8={}",
        state.kv.quant_q8
    );
    assert!(!state.kv.quant_hfq4 && !state.kv.quant_asym4 && !state.kv.quant_asym3 && !state.kv.quant_asym2);
    assert_eq!(state.kv.n_kv_heads, cfg.num_key_value_heads);
    assert_eq!(state.kv.head_dim, cfg.head_dim);
    assert_eq!(state.kv.k_gpu.len(), cfg.num_attention_layers());
    assert_eq!(state.kv.v_gpu.len(), cfg.num_attention_layers());
    assert_eq!(state.conv_states.len(), cfg.num_conv_layers());
    assert_eq!(weights.layers.len(), cfg.num_hidden_layers);

    // Embed + tied lm_head stay Q8 even on the MQ4 cohort.
    let embed_bytes = cfg.vocab_size * (cfg.hidden_size / 32) * 34;
    assert_eq!(
        weights.embed.buf.size(),
        embed_bytes,
        "Q8 embedding byte footprint"
    );
    require_q8_weight(&weights.lm_head, cfg.vocab_size, cfg.hidden_size, "lm_head");

    let mut conv_owner = vec![None; cfg.num_conv_layers()];
    let mut kv_owner = vec![None; cfg.num_attention_layers()];
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        assert_eq!(
            cfg.mixer(layer_idx),
            MIXERS_350M[layer_idx],
            "L{layer_idx} mixer kind"
        );
        match (&layer.mixer, MIXERS_350M[layer_idx]) {
            (Mixer::Conv(conv), MixerKind::Conv) => {
                require_mq4_weight(
                    &conv.in_proj,
                    3 * cfg.hidden_size,
                    cfg.hidden_size,
                    &format!("L{layer_idx}.conv.in_proj"),
                );
                require_mq4_weight(
                    &conv.out_proj,
                    cfg.hidden_size,
                    cfg.hidden_size,
                    &format!("L{layer_idx}.conv.out_proj"),
                );
                assert!(
                    conv.conv_state_idx < cfg.num_conv_layers(),
                    "L{layer_idx} conv_state_idx {} out of range",
                    conv.conv_state_idx
                );
                if let Some(prev) = conv_owner[conv.conv_state_idx] {
                    panic!(
                        "duplicate conv_state_idx {} owned by L{prev} and L{layer_idx}",
                        conv.conv_state_idx
                    );
                }
                conv_owner[conv.conv_state_idx] = Some(layer_idx);
            }
            (Mixer::Attention(attn), MixerKind::Attention) => {
                require_mq4_weight(
                    &attn.wq,
                    cfg.q_dim(),
                    cfg.hidden_size,
                    &format!("L{layer_idx}.attention.wq"),
                );
                require_mq4_weight(
                    &attn.wk,
                    cfg.kv_dim(),
                    cfg.hidden_size,
                    &format!("L{layer_idx}.attention.wk"),
                );
                require_mq4_weight(
                    &attn.wv,
                    cfg.kv_dim(),
                    cfg.hidden_size,
                    &format!("L{layer_idx}.attention.wv"),
                );
                require_mq4_weight(
                    &attn.wo,
                    cfg.hidden_size,
                    cfg.q_dim(),
                    &format!("L{layer_idx}.attention.wo"),
                );
                assert!(
                    attn.kv_idx < cfg.num_attention_layers(),
                    "L{layer_idx} kv_idx {} out of range",
                    attn.kv_idx
                );
                if let Some(prev) = kv_owner[attn.kv_idx] {
                    panic!(
                        "duplicate kv_idx {} owned by L{prev} and L{layer_idx}",
                        attn.kv_idx
                    );
                }
                kv_owner[attn.kv_idx] = Some(layer_idx);
            }
            _ => panic!("L{layer_idx} mixer does not match frozen 350M topology"),
        }
        let Ffn::Dense(dense) = &layer.ffn else {
            panic!("L{layer_idx} must use dense FFN on 350M");
        };
        require_mq4_weight(
            &dense.w1,
            cfg.intermediate_size,
            cfg.hidden_size,
            &format!("L{layer_idx}.ffn.w1"),
        );
        require_mq4_weight(
            &dense.w3,
            cfg.intermediate_size,
            cfg.hidden_size,
            &format!("L{layer_idx}.ffn.w3"),
        );
        require_mq4_weight(
            &dense.w2,
            cfg.hidden_size,
            cfg.intermediate_size,
            &format!("L{layer_idx}.ffn.w2"),
        );
    }
    for (slot, owner) in conv_owner.iter().enumerate() {
        assert!(
            owner.is_some(),
            "conv ring slot {slot} has no owning layer (incomplete coverage)"
        );
    }
    for (slot, owner) in kv_owner.iter().enumerate() {
        assert!(
            owner.is_some(),
            "KV cache slot {slot} has no owning layer (incomplete coverage)"
        );
    }
    // Print the exact discrete maps once for the ledger (not a stage dump).
    println!("conv_slot_map={conv_owner:?}");
    println!("kv_slot_map={kv_owner:?}");
}

fn expected_chunk_count(n_tokens: usize, max_batch: usize) -> usize {
    assert!(max_batch > 0);
    n_tokens.div_ceil(max_batch)
}

fn chunk_coverage_plan(n_tokens: usize, max_batch: usize) -> Vec<usize> {
    let mut plan = Vec::new();
    let mut remaining = n_tokens;
    while remaining > 0 {
        let chunk = remaining.min(max_batch);
        plan.push(chunk);
        remaining -= chunk;
    }
    plan
}

fn resolve_chunk_capacity() -> usize {
    match std::env::var("HIPFIRE_LFM2_PREFILL_MAX_BATCH") {
        Ok(raw) => {
            let parsed: usize = raw
                .parse()
                .unwrap_or_else(|_| panic!("invalid HIPFIRE_LFM2_PREFILL_MAX_BATCH={raw}"));
            assert!(
                parsed > 0 && parsed <= 512,
                "HIPFIRE_LFM2_PREFILL_MAX_BATCH={parsed} outside (0, 512]"
            );
            parsed
        }
        Err(_) => DEFAULT_CHUNK_CAPACITY,
    }
}

fn main() {
    assert_eq!(
        std::env::var("HIPFIRE_LFM2_PREFILL_BATCH").ok().as_deref(),
        Some("1"),
        "run with HIPFIRE_LFM2_PREFILL_BATCH=1"
    );

    let model = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME is not set"))
                .join(".hipfire/models/lfm2.5-350m.mq4")
        });
    assert!(
        model.exists(),
        "model missing: {} (expected lfm2.5-350m.mq4)",
        model.display()
    );
    let md5 = file_md5(&model);
    assert_eq!(
        md5, EXPECTED_MODEL_MD5,
        "model md5 mismatch for {}: got {md5}, expected {EXPECTED_MODEL_MD5}",
        model.display()
    );
    println!("model={}", model.display());
    println!("model_md5={md5}");

    let chunk_capacity = resolve_chunk_capacity();
    println!("chunk_capacity={chunk_capacity}");
    // Matrix must exercise the single-chunk set and the 256→257 two-chunk edge.
    assert!(
        LENGTHS.contains(&1)
            && LENGTHS.contains(&256)
            && LENGTHS.contains(&257)
            && LENGTHS.iter().any(|&n| n < chunk_capacity)
            && LENGTHS.iter().any(|&n| n == chunk_capacity)
            && LENGTHS.iter().any(|&n| n > chunk_capacity),
        "LENGTHS must cover sub-chunk, exact-chunk, and multi-chunk cases for capacity {chunk_capacity}"
    );
    for &n in &LENGTHS {
        let plan = chunk_coverage_plan(n, chunk_capacity);
        assert_eq!(plan.iter().sum::<usize>(), n, "chunk plan must cover all tokens");
        assert_eq!(
            plan.len(),
            expected_chunk_count(n, chunk_capacity),
            "N={n} chunk count"
        );
        println!(
            "chunk_coverage N={n} chunks={} plan={plan:?}",
            plan.len()
        );
    }

    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "oracle requires gfx1201, got {}",
        gpu.arch
    );

    let mut hfq = HfqFile::open(&model).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    // Fixed long prompt — deterministic, long enough for N=257.
    let fixed_text = "The morning light crossed the mountains while the research team checked every instrument, recorded each observation, and compared the results with the previous expedition. ";
    let fixed_tokens = tokenizer.encode(&fixed_text.repeat(96));
    assert!(
        fixed_tokens.len() >= *LENGTHS.last().unwrap(),
        "tokenized prompt too short: {} < {}",
        fixed_tokens.len(),
        LENGTHS.last().unwrap()
    );
    // Bound every token id before any GPU mutation.
    // (Batched path also validates; mirror here so a bad prompt fails host-side.)
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    for (i, &tok) in fixed_tokens.iter().enumerate() {
        assert!(
            (tok as usize) < cfg.vocab_size,
            "token_ids[{i}]={tok} out of vocab {}",
            cfg.vocab_size
        );
    }

    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    // Probe state once for topology / Q8-KV / slot maps (freed before length loop).
    {
        let probe = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, LENGTHS[LENGTHS.len() - 1] + 8)
            .expect("probe state");
        assert_cohort_and_slot_mapping(&cfg, &weights, &probe);
        probe.free_gpu(&mut gpu);
    }

    let prompt_tokens = &fixed_tokens[..*LENGTHS.last().unwrap()];
    let mut kls = Vec::with_capacity(LENGTHS.len());

    for length in LENGTHS {
        let tokens = prompt_tokens[..length].to_vec();
        let plan = chunk_coverage_plan(length, chunk_capacity);
        let mut eager =
            Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, length + 8).expect("eager state");
        let mut batched =
            Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, length + 8).expect("batched state");

        // Exact pre-conditions.
        assert_eq!(eager.n_tokens, 0);
        assert_eq!(batched.n_tokens, 0);
        assert!(eager.kv.quant_q8 && batched.kv.quant_q8);

        let mut eager_layers = vec![Vec::new(); cfg.num_hidden_layers];
        let mut batched_layers = vec![Vec::new(); cfg.num_hidden_layers];

        // Eager: N sequential decode_step_capture (absolute positions 0..N-1).
        for (position, &token) in tokens.iter().enumerate() {
            decode_step_capture(
                &cfg,
                &weights,
                &mut eager,
                &mut gpu,
                token,
                position as u32,
                &mut eager_layers,
            )
            .unwrap_or_else(|e| panic!("eager capture step pos={position}: {e}"));
        }
        let logits_eager = gpu.download_f32(&eager.logits).expect("eager logits");

        // Batched: single forward_prefill_batch_capture over the full prompt.
        let logits_batched = forward_prefill_batch_capture(
            &cfg,
            &weights,
            &mut batched,
            &mut gpu,
            &tokens,
            0,
            &mut batched_layers,
        )
        .unwrap_or_else(|e| panic!("batched prefill N={length}: {e}"));

        // ---- exact discrete state ----
        assert_eq!(
            eager.n_tokens, length,
            "N={length}: eager n_tokens"
        );
        assert_eq!(
            batched.n_tokens, length,
            "N={length}: batched n_tokens must equal prompt length (chunk plan {plan:?})"
        );
        assert_eq!(
            eager.n_tokens, batched.n_tokens,
            "N={length}: n_tokens parity"
        );

        // Absolute KV write positions are 0..N-1 inclusive; both paths must
        // have filled exactly those slots (checked via dequant prefix length).
        let kv_positions: Vec<u32> = (0..length as u32).collect();
        assert_eq!(kv_positions.len(), length);
        assert_eq!(*kv_positions.first().unwrap_or(&0), 0);
        if length > 0 {
            assert_eq!(*kv_positions.last().unwrap(), (length as u32) - 1);
        }

        // ---- finiteness + logits ----
        assert_eq!(logits_eager.len(), cfg.vocab_size);
        assert_eq!(logits_batched.len(), cfg.vocab_size);
        assert!(
            logits_eager.iter().chain(&logits_batched).all(|v| v.is_finite()),
            "N={length}: non-finite logits"
        );
        let (logit_cos, logit_max_abs) = metrics(&logits_eager, &logits_batched);
        let kl = compute_kl(&logits_eager, &logits_batched);
        kls.push(kl);

        // ---- per-layer hidden ----
        let mut layer_min_cos = 1.0f64;
        let mut layer_max_abs = 0.0f32;
        let mut worst_layer = 0usize;
        let mut worst_token = 0usize;
        for layer in 0..cfg.num_hidden_layers {
            assert_eq!(
                eager_layers[layer].len(),
                length * cfg.hidden_size,
                "N={length} L{layer}: eager capture shape"
            );
            assert_eq!(
                batched_layers[layer].len(),
                length * cfg.hidden_size,
                "N={length} L{layer}: batched capture shape"
            );
            assert!(
                eager_layers[layer]
                    .iter()
                    .chain(&batched_layers[layer])
                    .all(|v| v.is_finite()),
                "N={length} L{layer}: non-finite hidden"
            );
            for token in 0..length {
                let begin = token * cfg.hidden_size;
                let end = begin + cfg.hidden_size;
                let (cosine, max_abs) = metrics(
                    &eager_layers[layer][begin..end],
                    &batched_layers[layer][begin..end],
                );
                if cosine < layer_min_cos {
                    layer_min_cos = cosine;
                    worst_layer = layer;
                    worst_token = token;
                }
                layer_max_abs = layer_max_abs.max(max_abs);
            }
        }

        // ---- final residual stream ----
        let eager_hidden = gpu.download_f32(&eager.h).expect("eager final hidden");
        let batched_hidden = gpu.download_f32(&batched.h).expect("batched final hidden");
        let (hidden_cos, hidden_max_abs) = metrics(&eager_hidden, &batched_hidden);

        // ---- conv tails (per conv ring slot) ----
        let mut conv_min_cos = 1.0f64;
        let mut conv_max_abs = 0.0f32;
        assert_eq!(eager.conv_states.len(), batched.conv_states.len());
        for (slot, (eager_state, batched_state)) in eager
            .conv_states
            .iter()
            .zip(&batched.conv_states)
            .enumerate()
        {
            let eager_values = gpu.download_f32(eager_state).expect("eager conv tail");
            let batched_values = gpu.download_f32(batched_state).expect("batched conv tail");
            assert_eq!(
                eager_values.len(),
                cfg.hidden_size * (cfg.conv_kernel_size - 1),
                "conv slot {slot} length"
            );
            let (cosine, max_abs) = metrics(&eager_values, &batched_values);
            conv_min_cos = conv_min_cos.min(cosine);
            conv_max_abs = conv_max_abs.max(max_abs);
        }

        // ---- dequantized Q8 KV at every absolute position 0..N-1 ----
        let bytes_per_position = cfg.num_key_value_heads * (cfg.head_dim / 32) * 34;
        let mut kv_min_cos = 1.0f64;
        let mut kv_max_abs = 0.0f32;
        assert_eq!(eager.kv.k_gpu.len(), batched.kv.k_gpu.len());
        for (slot, ((eager_k, batched_k), (eager_v, batched_v))) in eager
            .kv
            .k_gpu
            .iter()
            .zip(&batched.kv.k_gpu)
            .zip(eager.kv.v_gpu.iter().zip(&batched.kv.v_gpu))
            .enumerate()
        {
            for (which, lhs, rhs) in [("K", eager_k, batched_k), ("V", eager_v, batched_v)] {
                let eager_values = dequant_q8(&download_prefix(
                    &gpu,
                    lhs,
                    length * bytes_per_position,
                ));
                let batched_values = dequant_q8(&download_prefix(
                    &gpu,
                    rhs,
                    length * bytes_per_position,
                ));
                assert_eq!(
                    eager_values.len(),
                    length * cfg.kv_dim(),
                    "N={length} kv_slot={slot} {which}: dequant length"
                );
                assert_eq!(batched_values.len(), eager_values.len());
                // Per absolute KV write position.
                for &pos in &kv_positions {
                    let token = pos as usize;
                    let begin = token * cfg.kv_dim();
                    let end = begin + cfg.kv_dim();
                    let (cosine, max_abs) =
                        metrics(&eager_values[begin..end], &batched_values[begin..end]);
                    kv_min_cos = kv_min_cos.min(cosine);
                    kv_max_abs = kv_max_abs.max(max_abs);
                }
            }
        }

        println!(
            "N={length} chunks={} plan={plan:?} \
             logit_cos={logit_cos:.9} logit_max_abs={logit_max_abs:e} kl={kl:e} \
             layer_min_cos={layer_min_cos:.9} layer_max_abs={layer_max_abs:e} \
             worst_layer={worst_layer} worst_token={worst_token} \
             final_hidden_cos={hidden_cos:.9} final_hidden_max_abs={hidden_max_abs:e} \
             conv_min_cos={conv_min_cos:.9} conv_max_abs={conv_max_abs:e} \
             kv_min_cos={kv_min_cos:.9} kv_max_abs={kv_max_abs:e} \
             n_tokens_equal=true kv_positions=0..{last_pos} finite=true",
            plan.len(),
            last_pos = length.saturating_sub(1),
        );

        // ---- enforce frozen dense-MQ4 gates (no weakening) ----
        assert!(
            layer_min_cos >= COSINE_LIMIT && layer_max_abs <= STATE_MAX_ABS_LIMIT,
            "N={length} hidden threshold failed at layer {worst_layer} token {worst_token}: \
             cos={layer_min_cos} max_abs={layer_max_abs} \
             (need cos>={COSINE_LIMIT}, max_abs<={STATE_MAX_ABS_LIMIT})"
        );
        assert!(
            logit_max_abs <= LOGIT_MAX_ABS_LIMIT,
            "N={length} logit max_abs {logit_max_abs} exceeds {LOGIT_MAX_ABS_LIMIT}"
        );
        assert!(
            kl <= KL_MAX_LIMIT,
            "N={length} KL {kl} exceeds per-prompt max {KL_MAX_LIMIT}"
        );
        assert!(
            hidden_cos >= COSINE_LIMIT && hidden_max_abs <= STATE_MAX_ABS_LIMIT,
            "N={length} final hidden threshold failed: cos={hidden_cos} max_abs={hidden_max_abs}"
        );
        assert!(
            conv_min_cos >= COSINE_LIMIT && conv_max_abs <= STATE_MAX_ABS_LIMIT,
            "N={length} conv threshold failed: cos={conv_min_cos} max_abs={conv_max_abs}"
        );
        assert!(
            kv_min_cos >= COSINE_LIMIT && kv_max_abs <= STATE_MAX_ABS_LIMIT,
            "N={length} KV threshold failed: cos={kv_min_cos} max_abs={kv_max_abs}"
        );

        eager.free_gpu(&mut gpu);
        batched.free_gpu(&mut gpu);
    }

    let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
    let max_kl = kls.iter().copied().fold(0.0f64, f64::max);
    println!("kl_mean={mean_kl:e} kl_max={max_kl:e}");
    println!(
        "thresholds logit_max_abs<={LOGIT_MAX_ABS_LIMIT} \
         hidden/kv/conv cos>={COSINE_LIMIT} max_abs<={STATE_MAX_ABS_LIMIT} \
         kl_mean<={KL_MEAN_LIMIT} kl_max<={KL_MAX_LIMIT}"
    );
    assert!(
        mean_kl <= KL_MEAN_LIMIT,
        "mean KL {mean_kl} exceeds {KL_MEAN_LIMIT}"
    );
    assert!(
        max_kl <= KL_MAX_LIMIT,
        "max KL {max_kl} exceeds {KL_MAX_LIMIT}"
    );
    println!("LFM2_350M_MQ4_BATCHED_PREFILL_PARITY_PASS");
}
