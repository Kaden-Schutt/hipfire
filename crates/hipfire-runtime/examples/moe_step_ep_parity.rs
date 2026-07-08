// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EP MoE byte-identity parity harness (P-D decompose D1). `--arch ds4` loads a
//! DeepSeek-V4-Flash mq2lloyd model ONCE (shard-aware across `--tp` emulated EP
//! ranks), then runs the SAME greedy generation + MTP-EP draft twice in-process:
//! the `ep_moe_allreduce` PRIMITIVE arm (flag OFF) and the decomposed
//! `execute_steps_parallel` STEP arm (flag ON), toggled via
//! `forward::set_ds4_moe_step_override` (the `HIPFIRE_MOE_STEP` env gate is
//! `OnceLock`-cached, so it can't be flipped mid-process). Fresh per-rank state
//! is rebuilt between runs so the two generations are independent.
//!
//! Prints each run's committed-token FNV, whether they match, the first
//! divergence index, both decoded texts, and the MTP-EP accept result. Exit 1 on
//! generation mismatch.
//!
//! Run (emulated EP-2 on one gfx1151, RCCL-free peer all-reduce):
//!   HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 cargo run --release \
//!       -p hipfire-runtime --example moe_step_ep_parity -- --arch ds4 \
//!       --model ~/.hipfire/models/deepseek-v4-flash.mq2lloyd --tp 2 --max 32

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

fn main() {
    use hipfire_arch_deepseek4::forward;
    use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::{DType, GpuTensor};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut arch = "ds4".to_string();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 32;
    let mut tp: usize = 2;
    let mut no_bos = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--arch" => {
                arch = argv[i + 1].clone();
                i += 2;
            }
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
            "--no-bos" => {
                no_bos = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    if arch != "ds4" {
        eprintln!("moe_step_ep_parity: only --arch ds4 is implemented (got {arch:?}). \
                   The minimax arm lives in `-p hipfire-arch-minimax --example moe_step_ep_parity`.");
        std::process::exit(1);
    }
    let model = model.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        PathBuf::from(format!("{home}/.hipfire/models/deepseek-v4-flash.mq2lloyd"))
    });

    // ── config + tokenizer ──────────────────────────────────────────────────
    let hfq0 = HfqFile::open(&model).expect("open model");
    let mut cfg = DeepseekV4::config_from_hfq(&hfq0).expect("config");
    // Emulated EP-2 loads weights ×2 into one UMA pool; the DSpark 3-stage
    // drafter sidecar (~6 GB/rank) is not exercised by this parity harness
    // (we only drive forward_ep + mtp_forward_ep), so skip it to fit the box.
    cfg.load_dspark = false;
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    let n_exp = cfg.n_routed_experts;
    eprintln!(
        "ds4 EP parity: tp={tp} hidden={} layers={} hash_layers={} experts={}/{} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        n_exp,
        cfg.num_experts_per_tok,
        cfg.vocab_size,
    );
    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tok.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup_id("<｜begin▁of▁sentence｜>");
    let eos_tok = lookup_id("<｜end▁of▁sentence｜>").unwrap_or(tok.eos_id);
    eprintln!("  bos={bos_tok:?} eos={eos_tok}");
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
    let mut weights_per_rank = Vec::with_capacity(n);
    for r in 0..n {
        gpus.devices[r].bind_thread().expect("bind");
        let mut hfq = HfqFile::open(&model).expect("reopen model");
        let t = std::time::Instant::now();
        let w = DeepseekV4::load_weights_sharded(&mut hfq, &cfg, &mut gpus.devices[r], &shard, r)
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

    let mut prompt_ids: Vec<u32> = Vec::new();
    if !no_bos {
        if let Some(b) = bos_tok {
            prompt_ids.push(b);
        }
    }
    prompt_ids.extend(tok.encode(&prompt));
    eprintln!(
        "prompt {:?} → {} tokens (bos-prepended={})",
        prompt,
        prompt_ids.len(),
        !no_bos
    );

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

    // One full greedy generation + MTP-EP draft with fresh per-rank state.
    // Returns (gen ids, MTP draft token, MTP accept?).
    let run = |gpus: &mut Gpus, label: &str| -> (Vec<u32>, Option<u32>, bool) {
        let mut state_per_rank: Vec<DeepseekV4State> = Vec::with_capacity(n);
        let mut partials: Vec<GpuTensor> = Vec::with_capacity(n);
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind");
            state_per_rank.push(DeepseekV4State::new(&cfg).expect("state"));
            partials.push(
                gpus.devices[r]
                    .zeros(&[cfg.hidden_size], DType::F32)
                    .expect("partial"),
            );
        }
        // Prefill.
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
            .download_f32(state_per_rank[0].logits.as_ref().expect("logits"))
            .expect("dl");

        // MTP-EP draft: capture h_n (full HC residual stream) per rank, then draft
        // the token AFTER t0. Compared to gen[1] below (the true next-next token).
        let t0 = argmax(&logits);
        let mut h_n_per_rank: Vec<GpuTensor> = Vec::with_capacity(n);
        for r in 0..n {
            gpus.devices[r].bind_thread().expect("bind");
            let streams = state_per_rank[r]
                .residual_streams
                .as_ref()
                .expect("residual_streams");
            let h = gpus.devices[r]
                .alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], DType::F32)
                .expect("alloc h_n");
            gpus.devices[r]
                .memcpy_dtod_auto(&h.buf, &streams.buf, cfg.hc_mult * cfg.hidden_size * 4)
                .expect("copy h_n");
            h_n_per_rank.push(h);
        }
        let mtp_logits = forward::mtp_forward_ep(
            gpus,
            &weights_per_rank,
            &cfg,
            &mut state_per_rank,
            &partials,
            &h_n_per_rank,
            t0,
            prompt_ids.len() as u32,
        )
        .expect("mtp_forward_ep");
        let mtp_draft = argmax(&mtp_logits);

        // Greedy decode.
        let mut gen = Vec::new();
        let mut pos = prompt_ids.len();
        for _ in 0..max {
            let next = argmax(&logits);
            gen.push(next);
            if next == eos_tok {
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
                .download_f32(state_per_rank[0].logits.as_ref().expect("logits"))
                .expect("dl");
            pos += 1;
        }
        let accept = gen.get(1).copied() == Some(mtp_draft);
        eprintln!(
            "[{label}] gen {} tok, FNV=0x{:016x}  MTP draft={mtp_draft} ({:?}) vs gen[1]={:?} → {}",
            gen.len(),
            fnv1a(&gen),
            tok.decode(&[mtp_draft]),
            gen.get(1),
            if accept { "ACCEPT" } else { "reject" },
        );
        (gen, Some(mtp_draft), accept)
    };

    // ── Run 1: primitive arm (flag OFF) ─────────────────────────────────────
    forward::set_ds4_moe_step_override(Some(false));
    let (gen_off, draft_off, accept_off) = run(&mut gpus, "PRIMITIVE (flag OFF)");

    // ── Run 2: decomposed Step arm (flag ON) ────────────────────────────────
    forward::set_ds4_moe_step_override(Some(true));
    let (gen_on, draft_on, accept_on) = run(&mut gpus, "DECOMPOSED (flag ON)");
    forward::set_ds4_moe_step_override(None);

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
    println!(
        "MTP draft OFF: {draft_off:?} accept={accept_off}   MTP draft ON: {draft_on:?} accept={accept_on}"
    );
    eprintln!("off ids: {:?}", &gen_off[..gen_off.len().min(40)]);
    eprintln!("on  ids: {:?}", &gen_on[..gen_on.len().min(40)]);

    if fnv_off == fnv_on && first_div.is_none() {
        println!("PARITY: BYTE-IDENTICAL ✓");
    } else {
        println!("PARITY: DIVERGED ✗");
        std::process::exit(1);
    }
}
