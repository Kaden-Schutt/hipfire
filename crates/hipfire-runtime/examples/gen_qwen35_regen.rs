// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash data-gen: AUTOREGRESSIVE target regeneration. Seeds from the first
// `seed_len` tokens of a kldref chunk, then greedily generates with the target
// (argmax of output_norm+lm_head fed back), capturing per-layer hiddens. The
// drafter trains on the target's OWN outputs (tokens == target argmax) so the
// proxy/τ are meaningful. Writes HFHS hiddens + a .toks file of generated ids.
//
//   gen_qwen35_regen --model <hfq> --ref <kldref> --chunk C --n N --seed-len S
//                    --out <hfhs> --toks <toks> [--kv-mode q8]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{weight_gemv, KvCache};
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model = None; let mut ref_path = None; let mut out_path = None; let mut toks_path = None;
    let mut chunk = 0usize; let mut n = 512usize; let mut seed_len = 16usize; let mut kv_mode = "q8".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--out" => { out_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--toks" => { toks_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--chunk" => { chunk = argv[i + 1].parse().unwrap(); i += 2; }
            "--n" => { n = argv[i + 1].parse().unwrap(); i += 2; }
            "--seed-len" => { seed_len = argv[i + 1].parse().unwrap(); i += 2; }
            "--kv-mode" => { kv_mode = argv[i + 1].clone(); i += 2; }
            o => { eprintln!("unknown arg {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model"); let ref_path = ref_path.expect("--ref");
    let out_path = out_path.expect("--out"); let toks_path = toks_path.expect("--toks");
    unsafe { std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0"); std::env::set_var("HIPFIRE_GRAPH", "0"); std::env::set_var("HIPFIRE_KV_MODE", &kv_mode); }

    // seed tokens from the kldref chunk
    let mut rf = File::open(&ref_path).expect("ref");
    let mut magic = [0u8; 8]; rf.read_exact(&mut magic).unwrap();
    let mut hdr = [0u8; 24]; rf.read_exact(&mut hdr).unwrap();
    let n_ctx_ref = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    use std::io::{Seek, SeekFrom};
    rf.seek(SeekFrom::Start(32 + (chunk * n_ctx_ref * 4) as u64)).unwrap();
    let mut sb = vec![0u8; seed_len * 4]; rf.read_exact(&mut sb).unwrap();
    let seed: Vec<u32> = sb.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();

    let mut hfq = HfqFile::open(&model).expect("hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let eps = config.norm_eps;
    let vocab = config.vocab_size;

    let kv_max = n + 16;
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).unwrap(),
        "fp32" => KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max).unwrap(),
        o => panic!("kv-mode {o}"),
    };
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn");
    let mut hidden_rb = HiddenStateRingBuffer::new(&mut gpu, config.n_layers, config.n_layers, config.dim, n, 1).expect("rb");
    hidden_rb.extract_layers = (0..config.n_layers).collect();

    let t0 = std::time::Instant::now();
    let mut generated: Vec<u32> = Vec::with_capacity(n);
    let mut tok = seed[0];
    for pos in 0..n {
        qwen35::forward_scratch_with_hidden(&mut gpu, &weights, &config, tok, pos, &mut kv_cache, &mut dn_state, &scratch, &mut hidden_rb).expect("fwd");
        generated.push(tok);
        // logits = lm_head(output_norm(scratch.x)) -> argmax
        gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, eps).expect("norm");
        weight_gemv(&mut gpu, &weights.output, &scratch.tmp, &scratch.logits).expect("lm_head");
        let lg = gpu.download_f32(&scratch.logits).expect("dl");
        let next = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        // next token: keep seeding while within the seed prefix, else use argmax
        tok = if pos + 1 < seed_len { seed[pos + 1] } else { next };
        if pos == 0 || (pos + 1) % 128 == 0 { eprintln!("  gen {}/{} ({:.1}s)", pos + 1, n, t0.elapsed().as_secs_f32()); }
    }
    eprintln!("generated {n} tokens in {:.1}s (seed_len={seed_len})", t0.elapsed().as_secs_f32());

    // write HFHS (all layers)
    if let Some(p) = out_path.parent() { std::fs::create_dir_all(p).ok(); }
    let mut out = BufWriter::with_capacity(8 << 20, File::create(&out_path).unwrap());
    out.write_all(b"HFHS\0\0\0\0").unwrap();
    out.write_all(&(config.n_layers as u32).to_le_bytes()).unwrap();
    out.write_all(&(n as u32).to_le_bytes()).unwrap();
    out.write_all(&(config.dim as u32).to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();
    for buf in hidden_rb.layer_bufs.iter() {
        let d = gpu.download_f32(buf).unwrap();
        let raw: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
        out.write_all(raw).unwrap();
    }
    out.flush().unwrap();
    // write generated tokens
    let mut tf = BufWriter::new(File::create(&toks_path).unwrap());
    for t in &generated { tf.write_all(&t.to_le_bytes()).unwrap(); }
    tf.flush().unwrap();
    // quick coherence peek: unique-token ratio of the generated tail
    let tail = &generated[seed_len..];
    let uniq: std::collections::HashSet<u32> = tail.iter().copied().collect();
    eprintln!("wrote {} (hiddens) + {} ({} toks); gen tail unique ratio {:.3} (vocab {})",
        out_path.display(), toks_path.display(), generated.len(), uniq.len() as f32 / tail.len().max(1) as f32, vocab);
    println!("gen_qwen35_regen: OK chunk={chunk} n={n}");
}
