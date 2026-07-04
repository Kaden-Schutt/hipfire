// SPDX-License-Identifier: Apache-2.0
// hipfire — tokenizer-free, multi-arch tiny-model probe harness.
//
//! Loads a tiny random-init `.hfq` for ANY supported dense/MoE arch, feeds it a
//! FIXED synthetic token-ID stream (no tokenizer), and either:
//!   - **kld**: computes KL(ref || candidate) over the per-position logits of two
//!     models (a bf16 reference vs a quantized candidate of the same arch), or
//!   - **collect**: arms the model-agnostic [`CalibCollector`], runs the forward,
//!     and drains a `<name>.hessian`/`<name>.imatrix` `.calib.hfq` (HFQM).
//!
//! This is the multi-family generalization of `fixture_golden` (qwen35-only,
//! logit-hash) and of the daemon `kld_eval`/`collect` ops (qwen35-only +
//! tokenizer-bound). Activation capture works for every arch because the shared
//! `hipfire_runtime::weights::weight_gemv` chokepoint now calls
//! `gpu.maybe_capture_activation` (no-op unless a collector is armed).
//!
//! GPU-only: every forward requires `hipfire_rdna::Gpu`. Used by the
//! `tiny_quant_probe` example and the hipfire-eval `tiny_quant` battery.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use hipfire_rdna::Gpu;

use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::weights::WeightTensor;

use hipfire_arch_gemma3 as gemma3;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
// LLaMA/Mistral (arch 0/1) live in the runtime crate (HFQ config + loader +
// forward), surfaced via the hipfire-arch-llama Architecture impl.
use hipfire_runtime::llama::{self, ForwardScratch, LlamaConfig, LlamaWeights};

/// Which arch family a fixture belongs to. Parsed from the `--arch` flag (the
/// caller always knows it — it emitted the fixture), so no arch-id sniffing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyArch {
    Qwen35,
    Qwen35Moe,
    Qwen2,
    Gemma3,
    MiniMax,
    Llama,
}

impl TinyArch {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '.'], "_")
            .as_str()
        {
            "qwen3_5" | "qwen35" | "qwen3_5_text" => Ok(Self::Qwen35),
            "qwen3_5_moe" | "qwen35moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35Moe),
            "qwen2" => Ok(Self::Qwen2),
            "gemma3" | "gemma3_text" => Ok(Self::Gemma3),
            "minimax" | "minimax_m2" => Ok(Self::MiniMax),
            "llama" | "mistral" => Ok(Self::Llama),
            other => Err(format!(
                "unknown --arch '{other}' (qwen3_5|qwen3_5_moe|qwen2|gemma3|minimax|llama)"
            )),
        }
    }

    /// The `.hfq` arch_id this family writes (for the HFQM calib metadata).
    pub fn arch_id(self) -> u32 {
        match self {
            Self::Llama => 0,
            Self::Qwen35 => 5,
            Self::Qwen35Moe => 6,
            Self::Qwen2 => 7,
            Self::MiniMax => 10,
            Self::Gemma3 => 12,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen3_5",
            Self::Qwen35Moe => "qwen3_5_moe",
            Self::Qwen2 => "qwen2",
            Self::Gemma3 => "gemma3",
            Self::MiniMax => "minimax",
            Self::Llama => "llama",
        }
    }
}

/// A loaded tiny model + everything its forward needs, behind one enum so the
/// `kld`/`collect` drivers stay arch-agnostic.
pub enum TinyModel {
    Qwen35 {
        config: qwen35::Qwen35Config,
        weights: qwen35::Qwen35Weights,
        kv: KvCache,
        dn: DeltaNetState,
        scratch: Qwen35Scratch,
    },
    Qwen2 {
        config: qwen2::Qwen2Config,
        weights: qwen2::Qwen2Weights,
        state: qwen2::Qwen2State,
    },
    Gemma3 {
        config: gemma3::Gemma3Config,
        weights: gemma3::Gemma3Weights,
        state: gemma3::Gemma3State,
    },
    MiniMax {
        config: minimax::MiniMaxConfig,
        weights: minimax::MiniMaxWeights,
        state: minimax::MiniMaxState,
    },
    Llama {
        config: LlamaConfig,
        weights: LlamaWeights,
        kv: KvCache,
        scratch: ForwardScratch,
    },
}

impl TinyModel {
    /// Load `path` as `arch`, sizing per-arch state for `max_seq` positions.
    pub fn load(
        arch: TinyArch,
        path: &Path,
        gpu: &mut Gpu,
        max_seq: usize,
    ) -> Result<Self, String> {
        let mut hfq = HfqFile::open(path).map_err(|e| format!("open {path:?}: {e:?}"))?;
        match arch {
            TinyArch::Qwen35 | TinyArch::Qwen35Moe => {
                let config =
                    qwen35::config_from_hfq(&hfq).ok_or("qwen35: config_from_hfq failed")?;
                let weights = qwen35::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("qwen35 load_weights: {e:?}"))?;
                let kv = KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq + 16,
                )
                .map_err(|e| format!("qwen35 kv: {e:?}"))?;
                let dn =
                    DeltaNetState::new(gpu, &config).map_err(|e| format!("qwen35 dn: {e:?}"))?;
                let scratch = Qwen35Scratch::new(gpu, &config, 64)
                    .map_err(|e| format!("qwen35 scratch: {e:?}"))?;
                Ok(Self::Qwen35 {
                    config,
                    weights,
                    kv,
                    dn,
                    scratch,
                })
            }
            TinyArch::Qwen2 => {
                let config = qwen2::config_from_hfq(&hfq).ok_or("qwen2: config_from_hfq failed")?;
                let weights = qwen2::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("qwen2 load_weights: {e:?}"))?;
                let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config, max_seq)
                    .map_err(|e| format!("qwen2 state: {e:?}"))?;
                Ok(Self::Qwen2 {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::Gemma3 => {
                let config =
                    gemma3::config_from_hfq(&hfq).ok_or("gemma3: config_from_hfq failed")?;
                let weights = gemma3::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("gemma3 load_weights: {e:?}"))?;
                let state = gemma3::Gemma3State::new_with_max_seq(gpu, &config, max_seq, false)
                    .map_err(|e| format!("gemma3 state: {e:?}"))?;
                Ok(Self::Gemma3 {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::MiniMax => {
                let config = minimax::MiniMaxConfig::from_hfq(&hfq)?;
                let weights = minimax::MiniMaxWeights::load(&mut hfq, &config, gpu, None)?;
                let state = minimax::MiniMaxState::new_with_max_seq(gpu, &config, max_seq)?;
                Ok(Self::MiniMax {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::Llama => {
                let config = hipfire_runtime::hfq::config_from_hfq(&hfq)
                    .ok_or("llama: config_from_hfq failed")?;
                let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, gpu)
                    .map_err(|e| format!("llama load_weights_hfq: {e:?}"))?;
                let kv = KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq + 16,
                )
                .map_err(|e| format!("llama kv: {e:?}"))?;
                let scratch = ForwardScratch::new(gpu, &config)
                    .map_err(|e| format!("llama scratch: {e:?}"))?;
                Ok(Self::Llama {
                    config,
                    weights,
                    kv,
                    scratch,
                })
            }
        }
    }

    pub fn vocab(&self) -> usize {
        match self {
            Self::Qwen35 { config, .. } => config.vocab_size,
            Self::Qwen2 { config, .. } => config.vocab_size,
            Self::Gemma3 { config, .. } => config.vocab_size,
            Self::MiniMax { config, .. } => config.vocab_size,
            Self::Llama { config, .. } => config.vocab_size,
        }
    }

    /// Forward one token, returning the host logits (`[vocab]`). `pos` is honored
    /// by qwen35/minimax; qwen2/gemma3 self-increment their own position counter,
    /// so callers MUST feed tokens strictly in order from a fresh state (which
    /// `run_logits`/`run_collect` do) — this is not a random-access scorer.
    pub fn forward_logits(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Qwen35 {
                config,
                weights,
                kv,
                dn,
                scratch,
            } => {
                qwen35::forward_scratch(gpu, weights, config, token, pos, kv, dn, scratch)
                    .map_err(|e| format!("qwen35 forward: {e:?}"))?;
                gpu.download_f32(&scratch.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Qwen2 {
                config,
                weights,
                state,
            } => {
                qwen2::forward_step(gpu, weights, config, state, token)
                    .map_err(|e| format!("qwen2 forward: {e:?}"))?;
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Gemma3 {
                config,
                weights,
                state,
            } => {
                gemma3::forward::forward_step(gpu, weights, config, state, token)
                    .map_err(|e| format!("gemma3 forward: {e:?}"))?;
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::MiniMax {
                config,
                weights,
                state,
            } => minimax::forward::decode_step(config, weights, state, gpu, token, pos as u32),
            Self::Llama {
                config,
                weights,
                kv,
                scratch,
            } => {
                // forward_scratch computes logits into scratch.logits, THEN samples.
                // We pass greedy/no-op sampling params and read the raw pre-sample
                // logits (the sampled-token return value is discarded).
                llama::forward_scratch(
                    gpu, weights, config, token, pos, kv, scratch, 0.0, 1.0, 0, 0, 1.0,
                )
                .map_err(|e| format!("llama forward: {e:?}"))?;
                gpu.download_f32(&scratch.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
        }
    }

    /// Map each captured linear's device-buffer pointer → its checkpoint tensor
    /// name (minus `.weight`), so the collector keys Hessians by the same name
    /// the quantizer matches `--hessian`/`HIPFIRE_QTIP_HESSIAN` against. MoE
    /// routed experts (indexed-GEMV, not `weight_gemv`) are intentionally absent.
    pub fn capture_names(&self) -> HashMap<usize, String> {
        match self {
            // qwen35 ships a typed walker, but it labels tensors with the
            // real-checkpoint `model.language_model.` prefix. The tiny fixtures
            // use the short `model.` prefix, and the quantizer keys the Hessian
            // sidecar by the .hfq weight name (short) — so normalize the prefix
            // here, else qtip3-sim LDLQ matches 0 tensors and silently falls back
            // to plain QTIP. (The real collect path keeps the long prefix; both
            // sides agree there.)
            Self::Qwen35 { weights, .. } => qwen35::build_capture_names(weights)
                .into_iter()
                .map(|(ptr, name)| {
                    let short = name
                        .strip_prefix("model.language_model.")
                        .map(|rest| format!("model.{rest}"))
                        .unwrap_or(name);
                    (ptr, short)
                })
                .collect(),
            Self::Qwen2 { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            Self::Gemma3 { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            Self::MiniMax { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.router, format!("{p}.block_sparse_moe.gate"));
                }
                m
            }
            Self::Llama { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
        }
    }
}

/// A fixed synthetic token-ID stream valid for any tiny fixture vocab (mod 100).
/// Mirrors `fixture_golden`'s generator so streams are comparable.
pub fn synthetic_tokens(len: usize, seed: u64) -> Vec<u32> {
    let mut st = seed ^ 0x5DEE_CE66_D8A1_0001u64;
    let mut next = || {
        st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..len).map(|_| (next() % 100) as u32).collect()
}

/// Numerically-stable log-softmax in place → returns log-probs.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - m) as f64).exp();
    }
    let lse = m as f64 + sum.ln();
    logits.iter().map(|&v| (v as f64 - lse) as f32).collect()
}

/// Result of a [`run_kld`] pass.
pub struct KldOut {
    pub mean_kld: f64,
    pub max_kld: f64,
    pub n_scored: usize,
    pub finite: bool,
}

/// Result of an autoregressive tiny forward pass.
pub struct ArHashOut {
    pub logit_hash: u64,
    pub token_hash: u64,
    pub n_steps: usize,
    pub prompt_len: usize,
    pub last_token: u32,
}

fn hash_mix(mut h: u64, x: u64) -> u64 {
    h ^= x;
    h = h.wrapping_mul(0x1000_0000_01B3);
    h ^ (h >> 32)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i as u32
}

/// Free-running greedy decode over a tiny fixture, hashing the full logit vector
/// and the generated token stream. Unlike [`run_kld`], this grows KV/state using
/// the model's own argmax outputs after a short deterministic prompt, so it is a
/// cheap tripwire for position/state/KV/long-tail decode drift.
pub fn run_ar_hash(
    arch: TinyArch,
    model_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    prompt_len: usize,
    seed: u64,
) -> Result<ArHashOut, String> {
    if len == 0 {
        return Err("ar_hash: --len must be > 0".into());
    }
    if prompt_len == 0 || prompt_len > len {
        return Err("ar_hash: --prompt-len must be in 1..=len".into());
    }

    let mut model = TinyModel::load(arch, model_path, gpu, len + 16)?;
    let vocab = model.vocab().max(1) as u32;
    let prompt = synthetic_tokens(prompt_len, seed);
    let mut next_token = prompt[0] % vocab;
    let mut logit_hash = 0xcbf2_9ce4_8422_2325u64;
    let mut token_hash = 0x9e37_79b9_7f4a_7c15u64;

    for pos in 0..len {
        let token = if pos < prompt_len {
            prompt[pos] % vocab
        } else {
            next_token
        };
        token_hash = hash_mix(token_hash, token as u64);
        let logits = model.forward_logits(gpu, token, pos)?;
        for &v in &logits {
            logit_hash = hash_mix(logit_hash, v.to_bits() as u64);
        }
        next_token = argmax(&logits) % vocab;
    }

    Ok(ArHashOut {
        logit_hash,
        token_hash,
        n_steps: len,
        prompt_len,
        last_token: next_token,
    })
}

/// Run `model` over `tokens`, returning per-position logits for pos >= warmup.
fn run_logits(
    arch: TinyArch,
    path: &Path,
    gpu: &mut Gpu,
    tokens: &[u32],
    warmup: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let mut model = TinyModel::load(arch, path, gpu, tokens.len() + 16)?;
    let mut out = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let lg = model.forward_logits(gpu, tok, pos)?;
        if pos >= warmup {
            out.push(lg);
        }
    }
    Ok(out)
}

/// KL(ref || cand) averaged over positions, feeding both models the identical
/// fixed synthetic stream (teacher-forced — inputs never depend on output, so
/// the two runs are independent and comparable position-by-position).
pub fn run_kld(
    arch: TinyArch,
    ref_path: &Path,
    cand_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    warmup: usize,
    seed: u64,
) -> Result<KldOut, String> {
    let tokens = synthetic_tokens(len, seed);
    let refs = run_logits(arch, ref_path, gpu, &tokens, warmup)?;
    let cands = run_logits(arch, cand_path, gpu, &tokens, warmup)?;
    if refs.len() != cands.len() || refs.is_empty() {
        return Err(format!(
            "kld: position mismatch ref={} cand={}",
            refs.len(),
            cands.len()
        ));
    }
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    let mut finite = true;
    for (rp, qp) in refs.iter().zip(cands.iter()) {
        let lr = log_softmax(rp);
        let lq = log_softmax(qp);
        // KL = Σ p·(log p − log q), p = exp(lr).
        let mut kl = 0.0f64;
        for (a, b) in lr.iter().zip(lq.iter()) {
            let p = (*a as f64).exp();
            if p > 0.0 {
                kl += p * (*a as f64 - *b as f64);
            }
        }
        if !kl.is_finite() {
            finite = false;
        }
        sum += kl;
        if kl > max {
            max = kl;
        }
    }
    let n = refs.len();
    Ok(KldOut {
        mean_kld: sum / n as f64,
        max_kld: max,
        n_scored: n,
        finite,
    })
}

/// Result of a [`run_collect`] pass.
pub struct CollectOut {
    pub n_tensors: usize,
    pub consistency: f32,
    pub out_path: String,
}

/// Arm the model-agnostic [`CalibCollector`], run the bf16 forward over the
/// synthetic stream (capturing per-linear input activations at the shared
/// `weight_gemv` chokepoint), and drain a `<name>.hessian`/`.imatrix`
/// `.calib.hfq` (HFQM) the quantizer can consume via `HIPFIRE_QTIP_HESSIAN`.
pub fn run_collect(
    arch: TinyArch,
    model_path: &Path,
    out_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    seed: u64,
) -> Result<CollectOut, String> {
    let tokens = synthetic_tokens(len, seed);
    let mut model = TinyModel::load(arch, model_path, gpu, tokens.len() + 16)?;

    // Routed experts (if any) would be imatrix-only, but we don't name them, so
    // a plain collector suffices — only the named dense linears are captured.
    let collector = Arc::new(CalibCollector::new());
    gpu.capture_names = model.capture_names();
    gpu.active_capture = Some(collector.clone());

    // Run the capturing forward, then ALWAYS disarm before propagating — leaving
    // `active_capture` armed would silently capture into this stale collector on
    // any later forward through the shared `&mut Gpu`.
    let fwd = (|| {
        for (pos, &tok) in tokens.iter().enumerate() {
            model.forward_logits(gpu, tok, pos)?;
        }
        Ok::<(), String>(())
    })();
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
    fwd?;

    let n_tensors = collector.len();
    if n_tensors == 0 {
        return Err(
            "collect: no tensors captured (capture_names empty or weight_gemv not hit)".into(),
        );
    }
    let meta = serde_json::json!({
        "artifact_kind": "calibration",
        "source": "tiny_quant_probe",
        "arch": arch.as_str(),
        "n_calib_tokens": tokens.len(),
    })
    .to_string();
    let consistency = collector
        .write_streaming(gpu, out_path, arch.arch_id(), &meta, &[])
        .map_err(|e| format!("collect: write {out_path:?}: {e}"))?;
    Ok(CollectOut {
        n_tensors,
        consistency,
        out_path: out_path.display().to_string(),
    })
}
