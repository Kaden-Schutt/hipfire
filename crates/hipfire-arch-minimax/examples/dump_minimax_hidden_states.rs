// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dump per-layer hidden states from the MiniMax-M2 forward for offline
//! comparison against the HF transformers oracle (arch bring-up validation).
//!
//! Reads chunk tokens from an HFKLDR ref, runs `decode_step_capture` per
//! position with all layers captured, and writes a single HFHS binary of
//! post-layer (pre-final-norm) hidden states — byte-format-compatible with
//! `scripts/dump_hf_hidden_states.py` + `scripts/compare_hidden_states.py`.
//!
//! Usage:
//!   dump_minimax_hidden_states --model <hfq> --ref <kldref> --out <hfhs> [--chunk N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax::forward::decode_step_capture;
    use hipfire_arch_minimax::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut chunk: usize = 0;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--ref" => {
                ref_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--chunk" => {
                chunk = argv[i + 1].parse().expect("--chunk");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    let ref_path = ref_path.expect("--ref required");
    let out_path = out_path.expect("--out required");

    // ---- read tokens from HFKLDR ref ----
    let mut ref_in = BufReader::new(File::open(&ref_path).expect("open ref"));
    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("magic");
    assert_eq!(&magic, b"HFKLDR\0\0", "bad ref magic");
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("hdr");
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    assert!(chunk < n_chunk, "chunk {chunk} >= n_chunk {n_chunk}");
    let skip = chunk * n_ctx * 4;
    if skip > 0 {
        let mut s = vec![0u8; skip];
        ref_in.read_exact(&mut s).expect("skip");
    }
    let mut tb = vec![0u8; n_ctx * 4];
    ref_in.read_exact(&mut tb).expect("tokens");
    let tokens: Vec<u32> = tb
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    eprintln!("read {} tokens from chunk {}", tokens.len(), chunk);

    // ---- load model ----
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "arch=minimax hidden={} layers={} heads={}/{} head_dim={} experts={}/{} rot={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.rotary_dim
    );
    let weights = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpu, None).expect("weights");
    let mut state = MiniMaxState::new_with_max_seq(&mut gpu, &cfg, n_ctx + 16).expect("state");

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
