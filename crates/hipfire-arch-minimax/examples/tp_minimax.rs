// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 TP-of-experts parity harness — Phase 3, Task 7 (remediated).
//!
//! Runs the same prompt twice, BOTH through `forward_tp` with the I64-down
//! path and explicit named mesh-bound Tp policies:
//!   1. tp=1 (whole experts, single logical Tp rank) — `DeviceMesh::rect(&[
//!      (DimKind::Tp, 1)])` + `Gpus::from_mesh` (rank-one binding, Lane H),
//!      weights loaded with `TpExpertSlice { tp: 1, rank: 0 }`.
//!   2. tp=2 (every rank holds all experts, column/row-split by inter/tp) —
//!      via `MiniMaxWeights::load(.., tp_slice=Some(..))` + `forward_tp`
//!      with a caller-owned Tp `MoEExecutionPolicy` and a mesh-bound `Gpus`.
//!
//! The int64 down accumulator + exact i64 Tp all-reduce are partition-
//! invariant: tp=1 and tp=2 must produce **bitwise-identical** f32 logits.
//! The gate asserts token-sequence equality AND every logit element equal by
//! `to_bits()` — no tolerance, no argmax-only comparison, no F32 baseline.
//!
//! Run:
//!   HIPFIRE_DETERMINISTIC=1 HIPFIRE_EMULATE_GPUS=2 \
//!   HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 \
//!   cargo run --release -p hipfire-arch-minimax --example tp_minimax -- \
//!     --model ~/.hipfire/models/MiniMax-M2.7.mq2 --max 32

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn fnv1a_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(feature = "deltanet")]
fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax::forward;
    use hipfire_arch_minimax::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind, Gpus};
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::TpExpertSlice;
    use rdna_compute::{DType, GpuTensor};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 32;
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
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let prompt_fnv = fnv1a_bytes(prompt.as_bytes());
    eprintln!("prompt fnv1a_bytes: 0x{prompt_fnv:016x}  max: {max}");

    // ── config + tokenizer ─────────────────────────────────────────────────────
    let hfq0 = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq0).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq0.metadata_json).expect("tokenizer");
    drop(hfq0);

    eprintln!(
        "minimax: hidden={} layers={} experts={}/{} vocab={} inter={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size,
        cfg.intermediate_size,
    );

    let prompt_ids = tok.encode(&prompt);
    let max_seq = prompt_ids.len() + max + 16;
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());

    /// Load one Tp rank's weights/state/partials and run one `forward_tp`
    /// token (embed + body + final norm). `partials`/`partials_i64` are the
    /// per-rank routed partials the caller owns (zeroed for the first step).
    struct TpRun {
        gpus: Gpus,
        weights_per_rank: Vec<MiniMaxWeights>,
        state_per_rank: Vec<MiniMaxState>,
        partials: Vec<GpuTensor>,
        partials_i64: Vec<GpuTensor>,
        policy: MoEExecutionPolicy,
    }

    fn load_tp_run(cfg: &MiniMaxConfig, model: &PathBuf, max_seq: usize, tp: usize) -> TpRun {
        // Caller-owned MoE execution policy: the Tp mesh IS policy.mesh(), and
        // the `Gpus` is bound to that same mesh (`Gpus::from_mesh` — the sealed
        // executor's mesh-identity check requires it). Named rank-one Tp axes
        // bind through `Gpus::from_mesh` (Lane H); axis-less `DeviceMesh::single()`
        // remains rejected.
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, tp)]);
        let policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, mesh.clone()).expect("Tp policy");
        let mut gpus = Gpus::from_mesh(&mesh, cfg.num_hidden_layers).expect("from_mesh");
        gpus.ensure_rank_streams().expect("rank_streams");
        let _ = gpus.enable_peer_all().expect("peer_all");
        assert_eq!(
            gpus.devices.len(),
            tp,
            "from_mesh gave the wrong device count"
        );

        let mut weights_per_rank: Vec<MiniMaxWeights> = Vec::with_capacity(tp);
        let t_load = std::time::Instant::now();
        for r in 0..tp {
            gpus.devices[r].bind_thread().expect("bind");
            let mut hfq = HfqFile::open(model).expect("open model");
            let ts = TpExpertSlice { tp, rank: r };
            let w = MiniMaxWeights::load(&mut hfq, cfg, &mut gpus.devices[r], None, Some(ts))
                .expect("load");
            weights_per_rank.push(w);
        }
        eprintln!(
            "  tp={tp} all ranks loaded in {:.1}s (down row-gather included)",
            t_load.elapsed().as_secs_f64()
        );

        let mut state_per_rank: Vec<MiniMaxState> = Vec::with_capacity(tp);
        let mut partials: Vec<GpuTensor> = Vec::with_capacity(tp);
        let mut partials_i64: Vec<GpuTensor> = Vec::with_capacity(tp);
        for r in 0..tp {
            gpus.devices[r].bind_thread().expect("bind");
            state_per_rank.push(
                MiniMaxState::new_with_max_seq(&mut gpus.devices[r], cfg, max_seq).expect("state"),
            );
            partials.push(
                gpus.devices[r]
                    .zeros(&[cfg.hidden_size], DType::F32)
                    .expect("partial"),
            );
            // int64 scratch: shape [hidden] with hidden*8 bytes of capacity
            // (the runtime lowerer validates the logical shape product against
            // the i64 conversion's n and the byte capacity against n*8).
            partials_i64.push(
                gpus.devices[r]
                    .upload_raw(&vec![0u8; cfg.hidden_size * 8], &[cfg.hidden_size])
                    .expect("partial_i64"),
            );
        }
        TpRun {
            gpus,
            weights_per_rank,
            state_per_rank,
            partials,
            partials_i64,
            policy,
        }
    }

    fn tp_step(run: &mut TpRun, cfg: &MiniMaxConfig, token: u32, pos: u32) {
        forward::forward_tp(
            &mut run.gpus,
            &run.weights_per_rank,
            cfg,
            &mut run.state_per_rank,
            &run.partials,
            &run.partials_i64,
            &run.policy,
            token,
            pos,
        )
        .expect("forward_tp");
    }

    /// Greedy-generate `max` tokens from the prompt; returns the token ids and
    /// every logits vector (logits[0] = pre-first-decode output).
    fn run_greedy(
        run: &mut TpRun,
        cfg: &MiniMaxConfig,
        prompt_ids: &[u32],
        max: usize,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        for (pos, &t) in prompt_ids.iter().enumerate() {
            tp_step(run, cfg, t, pos as u32);
        }
        run.gpus.devices[0].bind_thread().expect("bind0");
        let mut logits = run.gpus.devices[0]
            .download_f32(&run.state_per_rank[0].logits)
            .expect("dl");
        let mut tokens = Vec::new();
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        let mut pos = prompt_ids.len();
        all_logits.push(logits.clone());
        for _step in 0..max {
            let next = argmax(&logits);
            tokens.push(next);
            if matches!(next, 200020 | 151643 | 151645 | 2) {
                break;
            }
            tp_step(run, cfg, next, pos as u32);
            run.gpus.devices[0].bind_thread().expect("bind0");
            logits = run.gpus.devices[0]
                .download_f32(&run.state_per_rank[0].logits)
                .expect("dl");
            all_logits.push(logits.clone());
            pos += 1;
        }
        (tokens, all_logits)
    }

    // ── tp=1 reference run — `forward_tp` under the explicit named Tp(1)
    //    mesh-bound policy, I64-down path (the historical bitwise partition-
    //    invariance gate; no F32 baseline, no tolerance). ───────────────────
    eprintln!("\n=== tp=1 run (forward_tp, Tp(1) mesh, i64 down) ===");
    let (tp1_tokens, tp1_logits_all) = {
        let mut run = load_tp_run(&cfg, &model, max_seq, 1);
        let out = run_greedy(&mut run, &cfg, &prompt_ids, max);
        // Free GPU memory before the tp=2 load (so the same physical device
        // can hold the tp=2 weights under HIPFIRE_EMULATE_GPUS=2).
        let TpRun {
            gpus,
            weights_per_rank,
            state_per_rank,
            ..
        } = run;
        let mut gpus = gpus;
        for ((state, weights), gpu0) in state_per_rank
            .into_iter()
            .zip(weights_per_rank.into_iter())
            .zip(gpus.devices.iter_mut())
        {
            state.free_gpu(gpu0);
            weights.free_gpu(gpu0);
            gpu0.bind_thread().expect("bind free");
            gpu0.drain_pool();
        }
        eprintln!("  tp=1 generated {} tokens", out.0.len());
        out
    };

    // ── tp=2 run (TP-of-experts: every rank owns all experts, inter/2 each) ───
    eprintln!("\n=== tp=2 run (forward_tp, Tp(2) mesh, i64 down) ===");
    let (tp2_tokens, tp2_logits_all) = {
        let mut run = load_tp_run(&cfg, &model, max_seq, 2);
        let out = run_greedy(&mut run, &cfg, &prompt_ids, max);
        eprintln!("  tp=2 generated {} tokens", out.0.len());
        out
    };

    // ── Bitwise parity check (token sequence + every logit by to_bits) ──────
    eprintln!("\n=== Parity check tp1 vs tp2 (bitwise) ===");
    let mut token_ok = tp1_tokens.len() == tp2_tokens.len();
    if token_ok {
        for (step, (t1, t2)) in tp1_tokens.iter().zip(tp2_tokens.iter()).enumerate() {
            if t1 != t2 {
                eprintln!("  TOKEN MISMATCH at step {step}: tp1={t1} tp2={t2}");
                token_ok = false;
            }
        }
    } else {
        eprintln!(
            "  token count mismatch: tp1={} tp2={}",
            tp1_tokens.len(),
            tp2_tokens.len()
        );
    }

    let n_steps = tp1_logits_all.len().min(tp2_logits_all.len());
    let mut first_mismatch: Option<(usize, usize)> = None;
    for step in 0..n_steps {
        let l1 = &tp1_logits_all[step];
        let l2 = &tp2_logits_all[step];
        if l1.len() != l2.len() {
            eprintln!(
                "  LOGIT LENGTH MISMATCH at step {step}: {} vs {}",
                l1.len(),
                l2.len()
            );
            first_mismatch = first_mismatch.or(Some((step, 0)));
            continue;
        }
        for (i, (a, b)) in l1.iter().zip(l2.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                if first_mismatch.is_none() {
                    eprintln!(
                        "  LOGIT BIT MISMATCH at step {step} elem {i}: tp1={a:.9} (0x{:08x}) tp2={b:.9} (0x{:08x})",
                        a.to_bits(),
                        b.to_bits()
                    );
                    first_mismatch = Some((step, i));
                }
            }
        }
    }
    if tp1_logits_all.len() != tp2_logits_all.len() {
        eprintln!(
            "  logit step count mismatch: tp1={} tp2={}",
            tp1_logits_all.len(),
            tp2_logits_all.len()
        );
    }

    eprintln!("  token sequence bitwise-exact: {token_ok}");
    eprintln!(
        "  every logit to_bits() equal: {}",
        first_mismatch.is_none()
    );
    eprintln!("  tp=1 generation:\n{}", tok.decode(&tp1_tokens));
    eprintln!("  tp=2 generation:\n{}", tok.decode(&tp2_tokens));

    assert!(
        token_ok,
        "PARITY FAIL: tp=1 and tp=2 token sequences differ (I64 TP partition invariance broken?)"
    );
    assert!(
        first_mismatch.is_none(),
        "PARITY FAIL: tp=1 and tp=2 logits differ bitwise (I64 TP partition invariance broken?)"
    );

    eprintln!("\nPARITY PASS: tp=1 == tp=2 bitwise (token sequence + every logit to_bits())");
}
