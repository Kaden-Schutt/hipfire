// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched-vs-per-token prefill parity harness for the gemma4 lowered path.
//!
//! Runs the SAME prompt through (A) per-token `forward_scratch` and
//! (B) `forward_prefill_batch` with fresh KV each, compares prefill logits
//! (max-abs diff, argmax) and an N-token greedy continuation (per-token decode
//! from both prefill states). Bypasses the daemon arch gate so it can probe
//! batched prefill on gfx12 where the daemon refuses.
//!
//! Usage:
//!   prefill_parity_gemma4 --model <hfq> [--prompt <text>] [--decode N]
//!
//! Env: HIPFIRE_GEMMA4_DUMP=1 + HIPFIRE_BATCHED_PREFILL / HIPFIRE_WMMA_PREFILL
//! affect only the lowered internals (run_prefill_gemm WMMA arm); this harness
//! calls forward_prefill_batch unconditionally for run B.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::lowered;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt =
        "The capital of France is a city with many famous museums and lovely streets".to_string();
    let mut decode_n: usize = 24;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&argv[i + 1]).expect("read prompt file");
                i += 2;
            }
            "--decode" => {
                decode_n = argv[i + 1].parse().expect("--decode");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("arch = {}", gpu.arch);
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = lowered::config_from_hfq(&hfq).expect("lowered config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = lowered::load_weights(&mut hfq, &cfg, &mut gpu).expect("weights");
    let mut ids = tok.encode(&prompt);
    if ids.first() != Some(&cfg.bos_token) {
        ids.insert(0, cfg.bos_token);
    }
    let max_seq = (ids.len() + decode_n + 16).max(cfg.sliding_window + 1);
    let scratch = lowered::Gemma4Scratch::new(&mut gpu, &cfg, max_seq).expect("scratch");
    lowered::init_scratch_constants(&mut gpu, &scratch, cfg.full_head_dim).expect("scratch consts");

    eprintln!(
        "prompt tokens = {} (chunk={})",
        ids.len(),
        scratch.max_prefill_batch
    );

    let fnv = |bytes: &[u8]| -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    let argmax = |v: &[f32]| -> (usize, f32) {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        (bi, bv)
    };

    let mut run = |label: &str, batched: bool| -> (Vec<f32>, Vec<u32>) {
        let mut kv_sliding = KvCache::new_gpu_q8_capped(
            &mut gpu,
            cfg.n_layers,
            cfg.sliding_n_kv_heads,
            cfg.sliding_head_dim,
            max_seq,
            cfg.sliding_window,
        )
        .expect("kv sliding");
        let mut kv_full = KvCache::new_gpu_asym3(
            &mut gpu,
            cfg.n_layers,
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            max_seq,
        )
        .expect("kv full");
        let t0 = std::time::Instant::now();
        if batched {
            let chunk = std::env::var("HIPFIRE_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(scratch.max_prefill_batch)
                .max(1);
            let mut off = 0usize;
            while off < ids.len() {
                let end = (off + chunk).min(ids.len());
                lowered::forward_prefill_batch(
                    &mut gpu,
                    &weights,
                    &cfg,
                    &ids[off..end],
                    off,
                    &mut kv_sliding,
                    &mut kv_full,
                    &scratch,
                )
                .expect("batched prefill");
                off = end;
            }
        } else {
            for (p, &t) in ids.iter().enumerate() {
                lowered::forward_scratch(
                    &mut gpu,
                    &weights,
                    &cfg,
                    t,
                    p,
                    &mut kv_sliding,
                    &mut kv_full,
                    &scratch,
                )
                .expect("per-token prefill");
            }
        }
        let _ = gpu.download_f32(&scratch.logits);
        eprintln!(
            "[{label}] prefill {} tok in {:.3}s",
            ids.len(),
            t0.elapsed().as_secs_f64()
        );
        let logits = gpu.download_f32(&scratch.logits).expect("logits dl");
        let (am, av) = argmax(&logits);
        let lh = fnv(unsafe {
            std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
        });
        eprintln!(
            "[{label}] prefill logits: argmax={am} ({:?}) val={av:.4} fnv=0x{lh:016x}",
            tok.decode(&[am as u32])
        );
        // Greedy continuation, per-token decode in BOTH runs (isolates prefill).
        let mut cont = Vec::new();
        let mut pos = ids.len();
        let mut next = am as u32;
        for _ in 0..decode_n {
            cont.push(next);
            lowered::forward_scratch(
                &mut gpu,
                &weights,
                &cfg,
                next,
                pos,
                &mut kv_sliding,
                &mut kv_full,
                &scratch,
            )
            .expect("decode");
            let l = gpu.download_f32(&scratch.logits).expect("dl");
            next = argmax(&l).0 as u32;
            pos += 1;
        }
        eprintln!("[{label}] cont ids:  {:?}", cont);
        eprintln!("[{label}] cont text: {:?}", tok.decode(&cont));
        kv_sliding.free_gpu(&mut gpu);
        kv_full.free_gpu(&mut gpu);
        (logits, cont)
    };

    let (la, ca) = run("per-token", false);
    let (lb, cb) = run("batched  ", true);

    let mut max_abs = 0f32;
    let mut max_i = 0usize;
    let mut n_diff = 0usize;
    for i in 0..la.len() {
        let d = (la[i] - lb[i]).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        if d > max_abs {
            max_abs = d;
            max_i = i;
        }
    }
    println!(
        "logits: n_diff={n_diff}/{} max_abs={max_abs:.6} at idx {max_i} (A={:.4} B={:.4})",
        la.len(),
        la[max_i],
        lb[max_i]
    );
    println!(
        "cont match: {}",
        if ca == cb { "IDENTICAL" } else { "DIVERGED" }
    );
    if ca != cb {
        let first = ca
            .iter()
            .zip(&cb)
            .position(|(a, b)| a != b)
            .unwrap_or(ca.len());
        println!("first cont divergence at token {first}");
        std::process::exit(2);
    }
    if n_diff == 0 {
        println!("logits BYTE-IDENTICAL");
    }
}
