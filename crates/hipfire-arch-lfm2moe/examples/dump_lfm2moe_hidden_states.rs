// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Dump per-layer hidden states from the LFM2.5-MoE forward for offline
//! comparison against the HF transformers oracle (arch bring-up validation).
//!
//! Reads a JSON array of token ids, runs `decode_step_capture` per position
//! with all layers captured, and writes a single HFHS binary of post-layer
//! (pre-final-norm) hidden states — byte-format-compatible with
//! `scripts/gen_tiny_lfm2moe.py` + `scripts/compare_hidden_states.py`.
//!
//! Usage:
//!   dump_lfm2moe_hidden_states --model <hfq> --tokens <tokens.json> --out <hfhs>
//!   [--capture-postmixer]   (capture post-conv/attn residual instead of post-layer)

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::decode_step_capture;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut tokens_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--capture-postmixer" => {
                std::env::set_var("HIPFIRE_LFM2_CAPTURE_POSTMIXER", "1");
                i += 1;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    let tokens_path = tokens_path.expect("--tokens required");
    let out_path = out_path.expect("--out required");

    // ---- read tokens (JSON array of ints) ----
    let tokens: Vec<u32> = {
        let s = std::fs::read_to_string(&tokens_path).expect("read tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse tokens json");
        v.into_iter().map(|t| t as u32).collect()
    };
    let n_ctx = tokens.len();
    eprintln!("read {n_ctx} tokens from {}", tokens_path.display());

    // ---- load model ----
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "arch=lfm2moe hidden={} layers={} (attn={} conv={}) heads={}/{} head_dim={} \
         experts={}/{} dense={} convK={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_layers(),
        cfg.num_conv_layers(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.num_dense_layers,
        cfg.conv_kernel_size,
        cfg.vocab_size,
    );
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, n_ctx + 16).expect("state");

    // ---- per-token forward with per-layer capture ----
    let mut capture: Vec<Vec<f32>> =
        vec![Vec::with_capacity(n_ctx * cfg.hidden_size); cfg.num_hidden_layers];
    let t0 = std::time::Instant::now();
    for (pos, &tok) in tokens.iter().enumerate() {
        decode_step_capture(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            tok,
            pos as u32,
            &mut capture,
        )
        .expect("decode_step_capture");
    }
    eprintln!("forward complete in {:.2}s", t0.elapsed().as_secs_f64());

    // ---- write HFHS ----
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = BufWriter::new(File::create(&out_path).expect("create out"));
    out.write_all(b"HFHS\0\0\0\0").unwrap();
    out.write_all(&(cfg.num_hidden_layers as u32).to_le_bytes())
        .unwrap();
    out.write_all(&(n_ctx as u32).to_le_bytes()).unwrap();
    out.write_all(&(cfg.hidden_size as u32).to_le_bytes())
        .unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();
    for (l, buf) in capture.iter().enumerate() {
        assert_eq!(buf.len(), n_ctx * cfg.hidden_size, "layer {l} size");
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
        out.write_all(bytes).unwrap();
        let rms =
            (buf.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / buf.len() as f64).sqrt();
        eprintln!("  layer {l}: rms={rms:.4}");
    }
    out.flush().unwrap();
    eprintln!("wrote {}", out_path.display());
}
