// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 EP MoE byte-identity parity harness (Task 7, P-D decompose D1).
//!
//! Loads the model ONCE (shard-aware across `--tp` emulated ranks), then runs
//! the SAME greedy generation twice in-process — the `ep_moe_allreduce`
//! PRIMITIVE arm (flag OFF) and the decomposed `execute_steps_parallel` STEP arm
//! (flag ON) — toggled via `forward::set_minimax_moe_step_override` (the
//! `HIPFIRE_MOE_STEP` env gate is `OnceLock`-cached, so it can't be flipped
//! mid-process). Fresh per-rank state is rebuilt between runs so the two
//! generations are independent.
//!
//! Prints each run's committed-token FNV, whether they match, the first
//! divergence index, and both decoded texts for eyeball. Exit 1 on mismatch.
//!
//! Run (emulated EP-2 on one gfx1151, RCCL-free peer all-reduce):
//!   HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 cargo run --release --features deltanet \
//!       -p hipfire-arch-minimax --example moe_step_ep_parity -- \
//!       --model ~/.hipfire/models/MiniMax-M2.7.mq2 --tp 2 --max 32

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
    let mut tp: usize = 2;
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
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--tp" => {
                tp = argv[i + 1].parse().expect("--tp");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    // ── config + tokenizer ──────────────────────────────────────────────────
    let hfq0 = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq0).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    let n_exp = cfg.num_local_experts;
    eprintln!(
        "minimax EP parity: tp={tp} hidden={} layers={} experts={}/{} vocab={}",
        cfg.hidden_size, cfg.num_hidden_layers, n_exp, cfg.num_experts_per_tok, cfg.vocab_size,
    );
    drop(hfq0);

    // ── bring up N ranks + shard-aware load (weights loaded ONCE) ───────────
    let mut gpus = Gpus::init_tp(tp, cfg.num_hidden_layers).expect("init_tp");
    let n = gpus.devices.len();
    assert_eq!(
        n, tp,
        "init_tp gave {n} devices (check HIPFIRE_EMULATE_GPUS)"
    );
    for (r, d) in gpus.devices.iter().enumerate() {
        eprintln!("  rank {r}: device_id={} arch={}", d.device_id, d.arch);
    }
    let shard = ShardConfig::new(
        tp,
        /*tp_kv_replicate=*/ true,
        n_exp,
        ExpertAssign::Stride,
    )
    .expect("ShardConfig");
    let mut weights_per_rank: Vec<MiniMaxWeights> = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        let mut hfq = HfqFile::open(&model).expect("reopen model");
        let t = std::time::Instant::now();
        let w = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpus.devices[r], Some((&shard, r)))
            .expect("shard-aware load");
        eprintln!(
            "  [rank {r}] loaded owned shard in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        weights_per_rank.push(w);
    }
    let peer = gpus.enable_peer_all().expect("enable_peer_all");
    eprintln!("  peer_access_enabled={peer}");
    gpus.ensure_rank_streams().expect("ensure_rank_streams");

    let prompt_ids = tok.encode(&prompt);
    let max_seq = prompt_ids.len() + max + 16;
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i as u32;
            }
        }
        bi
    };

    // One full greedy generation with fresh per-rank state. Returns the gen ids.
    let run = |gpus: &mut Gpus, label: &str| -> Vec<u32> {
        let mut state_per_rank: Vec<MiniMaxState> = Vec::with_capacity(n);
        let mut partials: Vec<GpuTensor> = Vec::with_capacity(n);
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind");
            state_per_rank.push(
                MiniMaxState::new_with_max_seq(&mut gpus.devices[r], &cfg, max_seq).expect("state"),
            );
            partials.push(
                gpus.devices[r]
                    .zeros(&[cfg.hidden_size], DType::F32)
                    .expect("partial"),
            );
        }
        for (pos, &t) in prompt_ids.iter().enumerate() {
            forward::forward_ep(
                gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                t,
                pos as u32,
            )
            .expect("forward_ep prefill");
        }
        gpus.devices[0].bind_thread().expect("bind0");
        let mut logits = gpus.devices[0]
            .download_f32(&state_per_rank[0].logits)
            .expect("dl");
        let mut gen = Vec::new();
        let mut pos = prompt_ids.len();
        for _ in 0..max {
            let next = argmax(&logits);
            gen.push(next);
            if matches!(next, 200020 | 151643 | 151645 | 2) {
                break;
            }
            forward::forward_ep(
                gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                next,
                pos as u32,
            )
            .expect("forward_ep decode");
            gpus.devices[0].bind_thread().expect("bind0");
            logits = gpus.devices[0]
                .download_f32(&state_per_rank[0].logits)
                .expect("dl");
            pos += 1;
        }
        // state/partials drop here (weights persist for the next run). The small
        // per-run state buffers are not explicitly freed — negligible next to
        // the 79 GB weights, and the harness process exits right after.
        eprintln!(
            "[{label}] gen {} tok, FNV=0x{:016x}",
            gen.len(),
            fnv1a(&gen)
        );
        gen
    };

    // ── Run 1: primitive arm (flag OFF) ─────────────────────────────────────
    forward::set_minimax_moe_step_override(Some(false));
    let gen_off = run(&mut gpus, "PRIMITIVE (flag OFF)");

    // ── Run 2: decomposed Step arm (flag ON) ────────────────────────────────
    forward::set_minimax_moe_step_override(Some(true));
    let gen_on = run(&mut gpus, "DECOMPOSED (flag ON)");
    forward::set_minimax_moe_step_override(None);

    // ── Compare ─────────────────────────────────────────────────────────────
    let fnv_off = fnv1a(&gen_off);
    let fnv_on = fnv1a(&gen_on);
    let first_div = gen_off
        .iter()
        .zip(gen_on.iter())
        .position(|(a, b)| a != b)
        .or_else(|| {
            if gen_off.len() != gen_on.len() {
                Some(gen_off.len().min(gen_on.len()))
            } else {
                None
            }
        });

    println!("=== PROMPT ===\n{prompt}");
    println!("=== PRIMITIVE (flag OFF) ===\n{}", tok.decode(&gen_off));
    println!("=== DECOMPOSED (flag ON) ===\n{}", tok.decode(&gen_on));
    println!("--- PARITY ---");
    println!("flag OFF FNV : 0x{fnv_off:016x}  ({} tok)", gen_off.len());
    println!("flag ON  FNV : 0x{fnv_on:016x}  ({} tok)", gen_on.len());
    println!("first_div    : {first_div:?}");
    eprintln!("off ids: {:?}", &gen_off[..gen_off.len().min(40)]);
    eprintln!("on  ids: {:?}", &gen_on[..gen_on.len().min(40)]);

    if fnv_off == fnv_on && first_div.is_none() {
        println!("PARITY: BYTE-IDENTICAL ✓");
    } else {
        println!("PARITY: DIVERGED ✗");
        std::process::exit(1);
    }
}
