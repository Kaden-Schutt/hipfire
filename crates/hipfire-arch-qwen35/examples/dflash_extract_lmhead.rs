// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash FR1: extract the FAITHFUL Qwen3.6-27B lm_head (MQ4G256, qt=13) to fp32
// [vocab, dim] via weight_gemv on identity columns. MagnumQuant rotates the
// weight offline + the activation online, and FWHT is orthogonal, so
// weight_gemv(W_mq4, e_j) = FWHT(e_j)·FWHT(W_true) = W_true[:, j] exactly.
//
//   cargo run --features deltanet --example dflash_extract_lmhead -- <hfq> <out.bin>

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::weight_gemv;
    use rdna_compute::{DType, Gpu};
    use std::io::Write;

    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 3 { eprintln!("usage: dflash_extract_lmhead <hfq> <out.bin>"); std::process::exit(1); }
    let (hfq_path, out_path) = (&argv[1], &argv[2]);

    let mut hfq = HfqFile::open(std::path::Path::new(hfq_path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let (vocab, dim) = (config.vocab_size, config.dim);
    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("loading qwen3.6-27b weights (lm_head extraction)...");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let lm = &weights.output;
    eprintln!("lm_head: m(vocab)={} k(dim)={} dtype={:?} awq={}", lm.m, lm.k, lm.gpu_dtype, lm.awq_scale.is_some());
    assert_eq!(lm.m, vocab); assert_eq!(lm.k, dim);
    if lm.awq_scale.is_some() {
        eprintln!("WARNING: lm_head has an AWQ sidecar; weight_gemv ignores it -> result would be off.");
        eprintln!("  (handle AWQ per-channel scaling before trusting this extraction)");
    }

    // gather columns: col-major buffer [dim][vocab]
    let out = gpu.alloc_tensor(&[vocab], DType::F32).unwrap();
    let mut col_major = vec![0f32; dim * vocab];
    let t0 = std::time::Instant::now();
    for j in 0..dim {
        let mut e = vec![0f32; dim]; e[j] = 1.0;
        let x = gpu.upload_f32(&e, &[dim]).unwrap();
        weight_gemv(&mut gpu, lm, &x, &out).expect("weight_gemv");
        let col = gpu.download_f32(&out).unwrap();
        col_major[j * vocab..(j + 1) * vocab].copy_from_slice(&col);
        let _ = gpu.free_tensor(x);
        if j % 512 == 0 { eprintln!("  col {j}/{dim}  ({:.1}s)", t0.elapsed().as_secs_f32()); }
    }
    eprintln!("gathered {dim} cols in {:.1}s; transposing -> [vocab,dim]", t0.elapsed().as_secs_f32());

    // transpose to row-major [vocab, dim]
    let mut row = vec![0f32; vocab * dim];
    for j in 0..dim {
        let src = &col_major[j * vocab..(j + 1) * vocab];
        for i in 0..vocab { row[i * dim + j] = src[i]; }
    }
    drop(col_major);

    let mean = row.iter().sum::<f32>() / row.len() as f32;
    let absmax = row.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let finite = row.iter().all(|x| x.is_finite());
    eprintln!("lm_head fp32 [{vocab},{dim}] mean={mean:.6} absmax={absmax:.4} finite={finite}");

    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create"));
    f.write_all(b"DFHEAD\0\0").unwrap();
    f.write_all(&(vocab as u32).to_le_bytes()).unwrap();
    f.write_all(&(dim as u32).to_le_bytes()).unwrap();
    let raw: &[u8] = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
    f.write_all(raw).unwrap();
    f.flush().unwrap();
    eprintln!("wrote {out_path} ({} MB)", (16 + row.len() * 4) / 1_000_000);
    println!("dflash_extract_lmhead: OK (vocab={vocab} dim={dim} finite={finite} awq={})", lm.awq_scale.is_some());
}
