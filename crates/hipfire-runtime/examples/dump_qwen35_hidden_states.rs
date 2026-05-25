// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dump per-layer hidden states from `qwen35::forward_scratch_with_hidden`
//! for offline comparison against an HF transformers oracle.
//!
//! Used by the engine-drift-floor investigation tooling — see
//! `docs/investigations/2026-05-12-engine-drift-floor/00-summary.md` for the
//! methodology and the matching HF oracle script + comparator.
//!
//! Reads chunk-N tokens from a hipfire-β kldref so the input matches the
//! eval pipeline byte-for-byte, runs `forward_scratch_with_hidden` per
//! position with ALL layers configured as extraction targets, and writes a
//! single HFHS binary containing post-layer hidden states for every layer +
//! every position.
//!
//! Reuses the existing `HiddenStateRingBuffer` from spec-decode infra
//! (which already supports per-layer extraction at arbitrary indices) —
//! this example just overrides `extract_layers` to span the full layer
//! range.
//!
//! Output format mirrors `scripts/dump_hf_hidden_states.py`:
//!   magic 8B = b"HFHS\0\0\0\0"
//!   n_layers u32
//!   n_pos u32  (= n_ctx)
//!   hidden_dim u32
//!   reserved u32 = 0
//!   body: n_layers * [n_pos, hidden_dim] f32 row-major
//!
//! Usage:
//!   dump_qwen35_hidden_states --model <hfq> --ref <kldref> \
//!                             --chunk N --out <path> \
//!                             [--kv-mode q8|asym2|asym3|asym4|fp32]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut chunk: usize = 0;
    let mut kv_mode: String = "q8".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--out" => { out_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--chunk" => { chunk = argv[i + 1].parse().expect("--chunk"); i += 2; }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4" | "fp32") {
                    eprintln!("--kv-mode must be one of: q8 asym2 asym3 asym4 fp32 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: dump_qwen35_hidden_states --model <hfq> --ref <kldref> --chunk N --out <path> [--kv-mode q8|asym3|fp32]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");
    let ref_path = ref_path.expect("--ref required");
    let out_path = out_path.expect("--out required");

    // Match eval_hipfire's env env setup so kernel paths align with the
    // floor measurements we are localizing.
    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &kv_mode);
    }

    // -------- read tokens from ref --------
    let ref_file = File::open(&ref_path).expect("open ref");
    let mut ref_in = BufReader::new(ref_file);
    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic"); std::process::exit(2);
    }
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("hdr");
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    if chunk >= n_chunk {
        eprintln!("chunk {chunk} >= n_chunk {n_chunk}"); std::process::exit(2);
    }
    let skip_bytes = chunk * n_ctx * 4;
    if skip_bytes > 0 {
        let mut skip = vec![0u8; skip_bytes];
        ref_in.read_exact(&mut skip).expect("skip");
    }
    let mut token_bytes = vec![0u8; n_ctx * 4];
    ref_in.read_exact(&mut token_bytes).expect("tokens");
    let tokens: Vec<u32> = token_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    eprintln!("read {} tokens from chunk {} of {}", tokens.len(), chunk, ref_path.display());

    // -------- load model --------
    let mut hfq = HfqFile::open(&model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("arch={} dim={} n_layers={}", gpu.arch, config.dim, config.n_layers);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");

    // -------- KV cache + DeltaNet state + scratch --------
    let kv_max = n_ctx + 16;
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv cache q8"),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv cache asym3"),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv cache asym4"),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv cache asym2"),
        "fp32" => KvCache::new_gpu(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv cache fp32"),
        other => panic!("unknown --kv-mode: {other}"),
    };
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");

    // -------- HiddenStateRingBuffer, overriding extract_layers to all layers --------
    let mut hidden_rb = HiddenStateRingBuffer::new(
        &mut gpu,
        config.n_layers,    // num_target_layers
        config.n_layers,    // num_extract == all
        config.dim,         // hidden_dim
        n_ctx,              // max_positions = exactly fits one chunk
        1,                  // max_batch = 1 (per-token forward)
    ).expect("hidden rb");
    hidden_rb.extract_layers = (0..config.n_layers).collect();
    eprintln!(
        "hidden_rb: {} layers x {} positions x {} hidden = {:.1} MB",
        hidden_rb.extract_layers.len(),
        hidden_rb.max_positions,
        hidden_rb.hidden_dim,
        (hidden_rb.extract_layers.len() * hidden_rb.max_positions * hidden_rb.hidden_dim * 4) as f64
            / (1024.0 * 1024.0)
    );

    // -------- per-token forward, head advances per call --------
    let t0 = std::time::Instant::now();
    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch_with_hidden(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_cache, &mut dn_state, &scratch,
            &mut hidden_rb,
        ).expect("forward_scratch_with_hidden");
        if pos == 0 || (pos + 1) % 256 == 0 {
            eprintln!("  pos {:4}/{}: {:.1}s elapsed", pos + 1, n_ctx, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("forward complete in {:.1}s", t0.elapsed().as_secs_f64());

    // -------- download each layer's buffer + write HFHS file --------
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    let out_file = File::create(&out_path).expect("create out");
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);
    out.write_all(b"HFHS\0\0\0\0").unwrap();
    out.write_all(&(config.n_layers as u32).to_le_bytes()).unwrap();
    out.write_all(&(n_ctx as u32).to_le_bytes()).unwrap();
    out.write_all(&(config.dim as u32).to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();

    // Ring buffer is laid out as [max_positions, hidden_dim] row-major
    // per layer. With max_positions == n_ctx and head advancing once per
    // forward, layer_bufs[k] holds positions 0..n_ctx in order.
    for (layer_idx, buf) in hidden_rb.layer_bufs.iter().enumerate() {
        let f32_data = gpu.download_f32(buf).expect("download layer");
        assert_eq!(f32_data.len(), n_ctx * config.dim,
            "layer {layer_idx} got {} elements, expected {}",
            f32_data.len(), n_ctx * config.dim);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        out.write_all(bytes).unwrap();
        if layer_idx < 3 || layer_idx == config.n_layers - 1 {
            let rms: f64 = (f32_data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                / f32_data.len() as f64).sqrt();
            eprintln!("  layer {layer_idx}: rms={rms:.4}");
        }
    }
    out.flush().unwrap();
    let size_mb = std::fs::metadata(&out_path).expect("stat").len() as f64 / (1024.0 * 1024.0);
    eprintln!("wrote {} ({:.1} MB)", out_path.display(), size_mb);
}
