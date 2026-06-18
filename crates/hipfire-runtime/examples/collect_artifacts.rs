// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load artifact collector (Phase 3 harness).
//
//! Loads a bf16 `.hfq`, arms a unified `ActivationCapture` collector (per-tensor
//! GPTQ Hessian Σxxᵀ + imatrix diag Σx²), runs the engine forward over the
//! calibration corpus (single model load), drains, and writes the HFHS Hessian
//! sidecar. Verifies internal consistency: diag(Σxxᵀ) must equal Σx² (the two
//! reduction kernels agree on the SAME captured activations).
//!
//! This is the in-process core that the daemon `Collect` op (Phase 5) will host.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example collect_artifacts -- \
//!     --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!     --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
//!     --output /tmp/qwen3.5-0.8b-native.hessian.bin --max-tokens 512

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::calibration::{logsumexp, topk_logits, CalibCollector};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;
use std::path::Path;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

fn main() {
    let model = arg("--model", None).expect("--model required");
    let corpus = arg("--corpus", None).expect("--corpus required");
    let output = arg("--output", Some("/tmp/native.hessian.bin".into())).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("512".into()))
        .unwrap()
        .parse()
        .unwrap();

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    // Tokenize a prefix of the corpus (single calibration sequence for the harness).
    let raw = std::fs::read(&corpus).expect("read corpus");
    let take = (max_tokens * 8).min(raw.len());
    let text = String::from_utf8_lossy(&raw[..take]).to_string();
    let all: Vec<u32> = tokenizer.encode(&text);
    let n_tok = all.len().min(max_tokens);
    eprintln!("calibrating on {n_tok} tokens");

    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    // Arm calibration: ptr→name map + the collector. f32 KV + FP32 DeltaNet state
    // for faithful (lossless) activations.
    let names = qwen35::build_capture_names(&weights);
    eprintln!("capture targets: {}", names.len());
    // Diagnostic: which gpu_dtype do the linears load as → which gemv kernel.
    if let Some(qwen35::LayerWeights::DeltaNet(l0)) = weights.layers.first() {
        eprintln!(
            "DIAG layer0: wqkv={:?} wo={:?} w_down={:?}",
            l0.wqkv.gpu_dtype, l0.wo.gpu_dtype, l0.w_down.gpu_dtype
        );
    }
    let collector = std::sync::Arc::new(CalibCollector::new());
    gpu.capture_names = names;
    gpu.active_capture = Some(collector.clone());

    // MoE router histogram (cheap; reuses the engine facility). Captured during
    // the same calibration forward; folded into the artifact metadata. For MoE
    // models only (no-op when num_experts == 0). Per-subject extension: reset/take
    // at subject boundaries when the corpus exposes subject identifiers.
    let is_moe = config.num_experts > 0;
    if is_moe {
        qwen35::reset_moe_router_histogram(config.num_experts, config.num_experts_per_tok);
        eprintln!(
            "MoE: router histogram armed ({} experts, top-{})",
            config.num_experts, config.num_experts_per_tok
        );
    }

    let mut kv = KvCache::new_gpu(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        n_tok + 16,
    )
    .unwrap();
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    // KLDREF (opt-in --kldref): capture the lm-head top-K logits + logZ per
    // position (the teacher-forced bf16 reference for KLD-vs-quant eval). This is
    // an OUTPUT tap (not a linear input), so it reads scratch.logits after each
    // forward — mirrors perplexity.rs --dump-ref / the .pkld format.
    let want_kldref = std::env::args().any(|a| a == "--kldref");
    const KLDREF_TOPK: usize = 64;
    let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();

    let t0 = std::time::Instant::now();
    for (pos, &tok) in all.iter().take(n_tok).enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch,
        )
        .expect("forward");
        if want_kldref {
            let lg = gpu.download_f32(&scratch.logits).expect("logits");
            kldref.push((logsumexp(&lg), topk_logits(&lg, KLDREF_TOPK)));
        }
    }
    if want_kldref {
        eprintln!(
            "KLDREF: captured {} positions (top-{KLDREF_TOPK})",
            kldref.len()
        );
    }
    eprintln!(
        "forward over {n_tok} tokens: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Disarm before draining (avoid capturing the drain's own ops, if any).
    gpu.active_capture = None;

    // Drain the MoE router histogram (if MoE) into a JSON block for the artifact.
    let moe_meta: Option<serde_json::Value> = if is_moe {
        qwen35::take_moe_router_histogram().map(|h| {
            // Top co-occurring expert pairs (the scheduler-affinity signal),
            // summed across layers, top 64 by count.
            let mut cooc: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
            for l in &h.per_layer {
                for (&k, &v) in &l.cooccurrence {
                    *cooc.entry(k).or_insert(0) += v;
                }
            }
            let mut pairs: Vec<(u64, u64)> = cooc.into_iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            pairs.truncate(64);
            let ne = h.num_experts as u64;
            let cooc_json: Vec<serde_json::Value> = pairs
                .iter()
                .map(|(key, cnt)| serde_json::json!([key / ne, key % ne, cnt]))
                .collect();
            eprintln!(
                "MoE: routed {} tokens; nonzero experts {}/{}",
                h.routed_tokens,
                h.topk_histogram.iter().filter(|&&c| c > 0).count(),
                h.num_experts
            );
            serde_json::json!({
                "num_experts": h.num_experts,
                "k_top": h.k_top,
                "routed_tokens": h.routed_tokens,
                "routed_slots": h.routed_slots,
                "top1_histogram": h.top1_histogram,
                "topk_histogram": h.topk_histogram,
                "per_layer_topk": h.per_layer.iter().map(|l| serde_json::json!(l.topk_histogram)).collect::<Vec<_>>(),
                "top_cooccurrence": cooc_json, // [expert_a, expert_b, count]
            })
        })
    } else {
        None
    };

    // Drain the lib collector → HFQ hessian+imatrix tensors + consistency.
    let (mut tensors, max_consistency, token_counts) = collector.drain(&gpu);
    let mut per_tensor_tokens = serde_json::Map::new();
    for (name, n) in &token_counts {
        per_tensor_tokens.insert(name.clone(), serde_json::json!(n));
    }
    let f32_bytes = |v: &[f32]| -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for &x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    };
    eprintln!(
        "drained {} tensors; max diag(H)-vs-Σx² rel-err = {max_consistency:.3e} {}",
        tensors.len() / 2,
        if max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );

    // KLDREF tensors: idx [n_pos,K] + logit [n_pos,K] (F32) + logz [n_pos] (F32).
    // Indices stored as f32 (vocab < 2^24 — exact).
    let mut kldref_meta = serde_json::Value::Null;
    if !kldref.is_empty() {
        let np = kldref.len();
        let kk = kldref[0].1.len();
        let mut idx_v = Vec::with_capacity(np * kk);
        let mut lg_v = Vec::with_capacity(np * kk);
        let mut lz_v = Vec::with_capacity(np);
        for (logz, tk) in &kldref {
            lz_v.push(*logz);
            for j in 0..kk {
                let (i, l) = tk.get(j).copied().unwrap_or((0, f32::NEG_INFINITY));
                idx_v.push(i as f32);
                lg_v.push(l);
            }
        }
        for (nm, shape, data) in [
            ("lm_head.kldref_idx", vec![np as u32, kk as u32], idx_v),
            ("lm_head.kldref_logit", vec![np as u32, kk as u32], lg_v),
            ("lm_head.kldref_logz", vec![np as u32], lz_v),
        ] {
            tensors.push(HfqMemTensor {
                name: nm.to_string(),
                quant_type: 2,
                shape,
                group_size: 0,
                data: f32_bytes(&data),
            });
        }
        kldref_meta = serde_json::json!({ "n_positions": np, "top_k": kk });
    }

    // Provenance metadata (the unify-on-HFQ decision: artifacts carry their own
    // producer/corpus/token provenance, queryable via `hfq meta-get`).
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "source_model": model,
        "corpus": corpus,
        "n_calib_tokens": n_tok,
        "artifacts": ["hessian", "imatrix"],
        "per_tensor_tokens": serde_json::Value::Object(per_tensor_tokens),
    });
    if let Some(mh) = moe_meta {
        meta.as_object_mut()
            .unwrap()
            .insert("moe_router_histogram".to_string(), mh);
        meta["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("moe_router_histogram"));
    }
    if !kldref_meta.is_null() {
        meta.as_object_mut()
            .unwrap()
            .insert("kldref".to_string(), kldref_meta);
        meta["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("kldref"));
    }
    // NOTE: AWQ scales are derived at quant time from the captured imatrix (E[x²],
    // the activation side) + model weights — no separate (easily-stale) awq_scale
    // artifact is stored; the imatrix is the source of record.
    hipfire_runtime::hfq::write_hfqm_package_mem(
        Path::new(&output),
        0,
        &serde_json::to_string(&meta).unwrap(),
        &tensors,
    )
    .expect("write calib.hfq");
    eprintln!("wrote calib HFQ: {output} ({} tensors)", tensors.len());
    if max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
