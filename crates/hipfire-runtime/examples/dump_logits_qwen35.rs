// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Dump prefill logits to a binary f32 file for path-divergence comparison.
//!
//! Single-purpose tool: runs ONE prefill pass on a deterministic fake prompt
//! and writes the resulting logits vector to disk as raw f32 little-endian.
//! No warmup, no gen — the goal is byte-comparable output across two
//! processes that differ only in which dispatch path was taken (e.g.
//! HIPFIRE_FP16=0 to force the wave32 fallback on gfx906).
//!
//! Usage: dump_logits_qwen35 <model.hfq> <out.f32> [--prefill N]
//!
//! Pair with `scripts/gfx906_logit_divergence.sh` to compute max/mean
//! absolute diff across two runs.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_logits_qwen35 <model.hfq> <out.f32> [--prefill N]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_path = &args[2];

    let mut prefill_len: usize = 64;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => {
                prefill_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("dump_logits_qwen35: arch={} prefill_len={}", gpu.arch, prefill_len);

    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = (prefill_len + 16).max(512);
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        )
        .unwrap(),
        "asym4" | "turbo4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        )
        .unwrap(),
        "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        )
        .unwrap(),
        "asym2" | "turbo2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        )
        .unwrap(),
        other => panic!("unknown HIPFIRE_KV_MODE: {other}"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).unwrap();

    // Deterministic fake prompt: 0, 1, 2, ... — same as bench_qwen35_mq4.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();

    qwen35::forward_prefill_batch(
        &mut gpu,
        &weights,
        &config,
        &prompt_tokens,
        0,
        &mut kv_cache,
        &mut dn_state,
        &scratch,
        None,
        None,
        None,
        None,
    )
    .expect("prefill forward failed");
    gpu.hip.device_synchronize().expect("sync");

    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    eprintln!("logits len={} (expected ~vocab_size={})", logits.len(), config.vocab_size);

    let mut out = std::fs::File::create(out_path).expect("create out file");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            logits.as_ptr() as *const u8,
            logits.len() * std::mem::size_of::<f32>(),
        )
    };
    out.write_all(bytes).expect("write logits");
    eprintln!("wrote {} bytes to {}", bytes.len(), out_path);
}
