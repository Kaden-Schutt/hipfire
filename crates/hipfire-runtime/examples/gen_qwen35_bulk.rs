// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash bulk data-gen: ONE model load, MANY target-regenerated sequences.
// Seeds from (kldref chunk x offset) for diversity; greedy AR generation with
// per-layer hidden capture; dumps ONLY the 5 selected layers (52 MB/seq vs
// 640 MB) + a .toks file. Per-sequence GPU state is pool-scoped so memory is
// stable across thousands of sequences.
//
//   gen_qwen35_bulk --model <hfq> --ref <kldref> --out-dir <dir> --n-seqs M
//                   [--n 512] [--seed-len 32] [--start 0] [--kv-mode q8]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{weight_gemv, KvCache};
    use std::fs::File;
    use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
    use std::path::PathBuf;

    const SEL: [usize; 5] = [2, 16, 31, 46, 61];
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None; let mut ref_path = None; let mut out_dir = None;
    let mut n = 512usize; let mut seed_len = 32usize; let mut n_seqs = 64usize; let mut start = 0usize; let mut kv_mode = "q8".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--out-dir" => { out_dir = Some(argv[i + 1].clone()); i += 2; }
            "--n" => { n = argv[i + 1].parse().unwrap(); i += 2; }
            "--seed-len" => { seed_len = argv[i + 1].parse().unwrap(); i += 2; }
            "--n-seqs" => { n_seqs = argv[i + 1].parse().unwrap(); i += 2; }
            "--start" => { start = argv[i + 1].parse().unwrap(); i += 2; }
            "--kv-mode" => { kv_mode = argv[i + 1].clone(); i += 2; }
            o => { eprintln!("unknown {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model"); let ref_path = ref_path.expect("--ref"); let out_dir = out_dir.expect("--out-dir");
    unsafe { std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0"); std::env::set_var("HIPFIRE_GRAPH", "0"); std::env::set_var("HIPFIRE_KV_MODE", &kv_mode); }
    std::fs::create_dir_all(&out_dir).ok();

    // load kldref tokens (all chunks) for seeds
    let mut rf = File::open(&ref_path).expect("ref");
    let mut magic = [0u8; 8]; rf.read_exact(&mut magic).unwrap();
    let mut hdr = [0u8; 24]; rf.read_exact(&mut hdr).unwrap();
    let n_ctx_ref = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut all_toks = vec![0u8; n_chunk * n_ctx_ref * 4];
    rf.seek(SeekFrom::Start(32)).unwrap(); rf.read_exact(&mut all_toks).unwrap();
    let toks_at = |chunk: usize, off: usize, len: usize| -> Vec<u32> {
        let base = (chunk * n_ctx_ref + off) * 4;
        all_toks[base..base + len * 4].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    };

    let mut hfq = HfqFile::open(&model).expect("hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let eps = config.norm_eps;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch");

    // seed plan: distinct (chunk, offset). offsets stride so each is a different start.
    let max_off = n_ctx_ref.saturating_sub(seed_len + 1).max(1);
    let offs_per_chunk = ((n_seqs + n_chunk - 1) / n_chunk).max(1);
    let stride = (max_off / offs_per_chunk).max(1);

    gpu.pool_begin_scope();
    let t_all = std::time::Instant::now();
    for s in 0..n_seqs {
        let idx = start + s;
        let chunk = idx % n_chunk;
        let off = ((idx / n_chunk) * stride) % max_off;
        let seed = toks_at(chunk, off, seed_len);
        let ck = gpu.pool_checkpoint();
        let mut kv_cache = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, n + 16).unwrap();
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
        let mut rb = HiddenStateRingBuffer::new(&mut gpu, config.n_layers, SEL.len(), config.dim, n, 1).unwrap();
        rb.extract_layers = SEL.to_vec();
        let mut generated: Vec<u32> = Vec::with_capacity(n);
        let mut tok = seed[0];
        for pos in 0..n {
            qwen35::forward_scratch_with_hidden(&mut gpu, &weights, &config, tok, pos, &mut kv_cache, &mut dn_state, &scratch, &mut rb).expect("fwd");
            generated.push(tok);
            gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, eps).unwrap();
            weight_gemv(&mut gpu, &weights.output, &scratch.tmp, &scratch.logits).unwrap();
            let lg = gpu.download_f32(&scratch.logits).unwrap();
            let next = lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            tok = if pos + 1 < seed_len { seed[pos + 1] } else { next };
        }
        // dump 5 layers (HFHS n_layers=5, in SEL order) + toks
        let hp = format!("{out_dir}/seq{idx}.hfhs");
        let mut out = BufWriter::with_capacity(4 << 20, File::create(&hp).unwrap());
        out.write_all(b"HFHS\0\0\0\0").unwrap();
        out.write_all(&(SEL.len() as u32).to_le_bytes()).unwrap();
        out.write_all(&(n as u32).to_le_bytes()).unwrap();
        out.write_all(&(config.dim as u32).to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for buf in rb.layer_bufs.iter() {
            let d = gpu.download_f32(buf).unwrap();
            let raw: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
            out.write_all(raw).unwrap();
        }
        out.flush().unwrap();
        let mut tf = BufWriter::new(File::create(format!("{out_dir}/seq{idx}.toks")).unwrap());
        for t in &generated { tf.write_all(&t.to_le_bytes()).unwrap(); }
        tf.flush().unwrap();
        gpu.pool_release_to(ck);
        if s % 8 == 0 || s + 1 == n_seqs {
            let uniq: std::collections::HashSet<u32> = generated[seed_len..].iter().copied().collect();
            eprintln!("  seq {idx} (chunk {chunk} off {off}): uniq {:.3}  [{}/{} {:.0}s]", uniq.len() as f32 / (n - seed_len) as f32, s + 1, n_seqs, t_all.elapsed().as_secs_f32());
        }
    }
    println!("gen_qwen35_bulk: OK {n_seqs} seqs -> {out_dir} ({:.0}s)", t_all.elapsed().as_secs_f32());
}
