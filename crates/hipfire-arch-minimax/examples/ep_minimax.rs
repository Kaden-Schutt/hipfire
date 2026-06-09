// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 EP (expert-parallel) greedy decode across N GPUs — Ship 6
//! substrate-EP. The 86 GB mq2-lloyd tier does NOT fit one 32 GB card, so this
//! shards the experts across `--tp` ranks (shard-aware load: each rank uploads
//! only its owned experts) and runs the lowered decode through the EP executor
//! (Attend replicated; MoE all-reduce-EP'd via peer-direct copy+add).
//!
//! Run (hiptrx, 4× gfx1201):
//!   HIP_VISIBLE_DEVICES=0,1,2,3 cargo run --release --features deltanet \
//!       -p hipfire-arch-minimax --example ep_minimax -- \
//!       --model ~/.hipfire/models/minimax-m2.mq2lloyd --tp 4 --max 32 \
//!       --prompt "The capital of France is"

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax::forward;
    use hipfire_arch_minimax::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::{DType, GpuTensor};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 32;
    let mut tp: usize = 4;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--prompt" => { prompt = argv[i + 1].clone(); i += 2; }
            "--max" => { max = argv[i + 1].parse().expect("--max"); i += 2; }
            "--tp" => { tp = argv[i + 1].parse().expect("--tp"); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model required");

    // ── config + tokenizer (per-rank loads reopen the file) ─────────────────
    let mut hfq0 = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq0).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    let n_exp = cfg.num_local_experts;
    eprintln!(
        "minimax EP: tp={tp} hidden={} layers={} experts={}/{} vocab={}",
        cfg.hidden_size, cfg.num_hidden_layers, n_exp, cfg.num_experts_per_tok, cfg.vocab_size,
    );
    drop(hfq0);

    // ── bring up N ranks ────────────────────────────────────────────────────
    let mut gpus = Gpus::init_tp(tp, cfg.num_hidden_layers).expect("init_tp");
    let n = gpus.devices.len();
    assert_eq!(n, tp, "init_tp gave {n} devices (check HIP_VISIBLE_DEVICES)");
    for (r, d) in gpus.devices.iter().enumerate() {
        eprintln!("  rank {r}: device_id={} arch={}", d.device_id, d.arch);
    }

    // ── shard-aware replicated load (each rank uploads only its owned experts) ─
    let shard = ShardConfig::new(tp, /*tp_kv_replicate=*/ true, n_exp, ExpertAssign::Stride)
        .expect("ShardConfig");
    let mut weights_per_rank: Vec<MiniMaxWeights> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        let mut hfq = HfqFile::open(&model).expect("reopen model");
        let t = std::time::Instant::now();
        let w = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpus.devices[r], Some((&shard, r)))
            .expect("shard-aware load");
        eprintln!("  [rank {r}] loaded owned shard in {:.1}s", t.elapsed().as_secs_f64());
        weights_per_rank.push(w);
    }
    eprintln!("  all ranks loaded (stride: rank r owns experts e%{tp}==r)");

    // ── per-rank state + routed partials ────────────────────────────────────
    let prompt_ids = tok.encode(&prompt);
    let max_seq = prompt_ids.len() + max + 16;
    let mut state_per_rank: Vec<MiniMaxState> = Vec::with_capacity(n);
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        state_per_rank.push(
            MiniMaxState::new_with_max_seq(&mut gpus.devices[r], &cfg, max_seq).expect("state"),
        );
        partials.push(gpus.devices[r].zeros(&[cfg.hidden_size], DType::F32).expect("partial"));
    }
    let peer = gpus.enable_peer_all().expect("enable_peer_all");
    eprintln!("  peer_access_enabled={peer}");
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus).expect("ensure_rank_streams");

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32; let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as u32; } }
        bi
    };

    // ── EP prefill (per-token) + greedy decode ──────────────────────────────
    eprintln!("\nprompt {:?} → {} tokens", prompt, prompt_ids.len());
    let t0 = std::time::Instant::now();
    for (pos, &t) in prompt_ids.iter().enumerate() {
        forward::forward_ep(&mut gpus, &weights_per_rank, &cfg, &mut state_per_rank, &partials, t, pos as u32)
            .expect("forward_ep prefill");
    }
    gpus.devices[0].bind_thread().expect("bind0");
    let mut logits = gpus.devices[0].download_f32(&state_per_rank[0].logits).expect("dl");
    eprintln!("prefill {} tok in {:.2}s", prompt_ids.len(), t0.elapsed().as_secs_f64());

    let mut gen = Vec::new();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    let mut steady = 0usize;
    let mut steady_t = std::time::Instant::now();
    for step in 0..max {
        let next = argmax(&logits);
        gen.push(next);
        if matches!(next, 200020 | 151643 | 151645 | 2) {
            break;
        }
        if step == 2 { steady_t = std::time::Instant::now(); steady = 0; }
        forward::forward_ep(&mut gpus, &weights_per_rank, &cfg, &mut state_per_rank, &partials, next, pos as u32)
            .expect("forward_ep decode");
        gpus.devices[0].bind_thread().expect("bind0");
        logits = gpus.devices[0].download_f32(&state_per_rank[0].logits).expect("dl");
        if step >= 2 { steady += 1; }
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    let steady_tps = if steady > 0 { steady as f64 / steady_t.elapsed().as_secs_f64() } else { f64::NAN };
    eprintln!(
        "decoded {} tok in {:.2}s ({:.1} tok/s overall, {:.1} tok/s steady)",
        gen.len(), dt, gen.len() as f64 / dt, steady_tps,
    );
    println!("=== PROMPT ===\n{prompt}\n=== GENERATION (tp={tp} EP) ===\n{}", tok.decode(&gen));
    eprintln!("gen ids: {:?}", &gen[..gen.len().min(40)]);
    eprintln!("gen FNV: 0x{:016x}", fnv1a(&gen));
}
