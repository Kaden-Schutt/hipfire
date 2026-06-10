// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 EAGLE speculative-decode driver — the capstone that turns the
//! validated drafter + batched verify into actual speedup.
//!
//! Loads the arch-13 TARGET HFQ + the arch-22 DRAFTER HFQ, prefills per-token
//! via `decode_step` (capturing the last position's post-`model.norm` hidden as
//! the drafter seed), then loops `spec_step_gemma4_eagle` until EOS or `--max`.
//! Reports the generated text, mean committed-per-round (τ), tok/s, and the
//! total number of draft rounds.
//!
//! Correctness mode (`--check-eager`): also runs the eager `decode_step` greedy
//! AR sequence over the same prompt and prints both token-id streams + a diff.
//! With temp 0 the spec-decode committed sequence is provably the target's
//! greedy AR sequence, so the two MUST be byte-identical.
//!
//! Measured (hiptrx gfx1201, gemma4-12B, code prompt, max 256):
//!   - Q8 target: spec == eager BYTE-IDENTICAL, deterministic, τ≈3.1–4.5 over
//!     draft-len 3..6. (Q8 spec is correctness-proven but lm_head-bound, so
//!     ~13 tok/s < the 34.8 tok/s Q8 AR baseline — the per-position full-vocab
//!     lm_head dominates each round on the bigger Q8 weights.)
//!   - MQ4-attn target (gemma4-12b-mq4attn): spec == eager BYTE-IDENTICAL for
//!     draft-len ≤ 4; **80.5 tok/s @ draft-len 3 vs 50.1 tok/s AR = +61%**,
//!     τ≈3.08. draft-len ≥ 5 currently hits a PRE-EXISTING, non-deterministic
//!     batched-MQ4 `forward_batch` fault (illegal access) on gfx12 — reproduced
//!     standalone on the UNMODIFIED `forward_batch` (`verify_batch_gemma4
//!     --bs 16`), so it is a batched-MQ4-GEMM kernel limitation, NOT a spec-loop
//!     bug. The spec loop itself is verified correct on the deterministic Q8
//!     path. See the warmup note below.
//!   - UNION target (gemma4-12b-it.union.mq4: Q8 attn + Q8 tied lm_head, MQ4
//!     FFN), 2026-06-10, after `proj_gemm_batched` Q8 routed through
//!     `gemm_q8_0_batched_chunked` (WMMA on gfx12): spec == eager
//!     BYTE-IDENTICAL for draft-len 3..=5; 86-93 tok/s @ dl3 and ~106 tok/s
//!     @ dl5 vs 51.0 tok/s AR (1.7-2.1x). Before that routing fix the scalar
//!     `gemm_q8_0_batched` cost 334 us/call x 184 attn projections per
//!     verify (~61 ms/round, 61% of GPU time per rocprofv3) -> 32 tok/s =
//!     0.63x AR. The dl >= 5 fault note above predates the fix and did NOT
//!     reproduce at dl5 (b=6) on the union target across repeated
//!     --check-eager runs; post-fix `verify_batch_gemma4 --bs 8,16,32`:
//!     B=8 PASS, B=16 PASS (the old illegal-access fault is gone), B=32
//!     argmax near-tie flip at cosine 0.9999 (numerical, no fault) — all
//!     far above the daemon's dl <= 5 (b <= 6) envelope.
//!
//! Usage:
//!   infer_gemma4_spec --model <target.hfq> --draft-model <drafter.hfq> \
//!       [--prompt <text>] [--token-ids 2,...] [--draft-len N] [--max M] \
//!       [--temperature T(0 only)] [--check-eager]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::config::Gemma4Config;
    use hipfire_arch_gemma4::drafter::{
        Gemma4DrafterConfig, Gemma4DrafterScratch, Gemma4DrafterWeights,
    };
    use hipfire_arch_gemma4::forward::{decode_step, forward_batch};
    use hipfire_arch_gemma4::gemma4::{Gemma4State, Gemma4Weights};
    use hipfire_arch_gemma4::speculative::{spec_step_gemma4_eagle, Gemma4SpecScratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut draft_model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut token_ids: Option<Vec<u32>> = None;
    let mut draft_len: usize = 4;
    let mut max: usize = 256;
    let mut temperature: f32 = 0.0;
    let mut check_eager = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--draft-model" => {
                draft_model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&argv[i + 1])
                    .expect("--prompt-file: read");
                i += 2;
            }
            "--token-ids" => {
                token_ids = Some(
                    argv[i + 1]
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<u32>().expect("--token-ids"))
                        .collect(),
                );
                i += 2;
            }
            "--draft-len" => {
                draft_len = argv[i + 1].parse().expect("--draft-len");
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--temperature" => {
                temperature = argv[i + 1].parse().expect("--temperature");
                i += 2;
            }
            "--check-eager" => {
                check_eager = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    let draft_model = draft_model.expect("--draft-model required");
    assert!(draft_len >= 1, "--draft-len must be ≥ 1");
    assert_eq!(temperature, 0.0, "only greedy (temp 0) is supported");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // ── Target ──
    let hfq = HfqFile::open(&model).expect("open target");
    let cfg = Gemma4Config::from_hfq(&hfq).expect("target config");
    eprintln!(
        "target dim={} layers={} sliding={} full={} vocab={} softcap={}",
        cfg.dim,
        cfg.n_layers,
        cfg.n_sliding_layers(),
        cfg.n_full_layers(),
        cfg.vocab_size,
        cfg.final_logit_softcapping
    );
    let tok = if token_ids.is_none() {
        Some(Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer"))
    } else {
        None
    };
    let t_load = std::time::Instant::now();
    let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("target weights");

    // ── Drafter ──
    let dhfq = HfqFile::open(&draft_model).expect("open drafter");
    let dcfg = Gemma4DrafterConfig::from_hfq(&dhfq).expect("drafter config");
    assert_eq!(
        dcfg.backbone_hidden, cfg.dim,
        "drafter backbone must equal target hidden"
    );
    eprintln!(
        "drafter hidden={} layers={} backbone={} vocab={}",
        dcfg.hidden, dcfg.n_layers, dcfg.backbone_hidden, dcfg.vocab_size
    );
    let dweights = Gemma4DrafterWeights::load(&dhfq, &dcfg, &mut gpu).expect("drafter weights");
    let mut dscratch = Gemma4DrafterScratch::new(&mut gpu, &dcfg).expect("drafter scratch");
    eprintln!("loaded target+drafter in {:.1}s", t_load.elapsed().as_secs_f64());

    // ── Prompt → tokens (prepend BOS) ──
    let mut prompt_ids = match token_ids {
        Some(ids) => ids,
        None => tok.as_ref().unwrap().encode(&prompt),
    };
    if prompt_ids.first() != Some(&cfg.bos_token) {
        prompt_ids.insert(0, cfg.bos_token);
    }
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());
    let max_seq = prompt_ids.len() + max + 16;

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (k, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = k as u32;
            }
        }
        bi
    };
    let is_eos = |t: u32| matches!(t, 1 | 106);

    // ════════════════════════════════════════════════════════════════════
    //  Optional eager reference (for the correctness gate).
    // ════════════════════════════════════════════════════════════════════
    let eager_gen: Option<Vec<u32>> = if check_eager {
        let mut st = Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("eager state");
        let mut logits = Vec::new();
        for (pos, &t) in prompt_ids.iter().enumerate() {
            logits =
                decode_step(&cfg, &weights, &mut st, &mut gpu, t, pos as u32).expect("eager prefill");
        }
        let mut gen = Vec::new();
        let mut pos = prompt_ids.len();
        for _ in 0..max {
            let next = argmax(&logits);
            if is_eos(next) {
                break;
            }
            gen.push(next);
            logits = decode_step(&cfg, &weights, &mut st, &mut gpu, next, pos as u32)
                .expect("eager decode");
            pos += 1;
        }
        Some(gen)
    } else {
        None
    };

    // ════════════════════════════════════════════════════════════════════
    //  Spec-decode run.
    // ════════════════════════════════════════════════════════════════════
    let mut state = Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("spec state");
    let mut sp = Gemma4SpecScratch::new(&mut gpu, &cfg, draft_len).expect("spec scratch");

    // ── Prefill per-token via `decode_step` (correctness-first; matches the
    //    validated drafter harness). The seed TOKEN is the last prompt token
    //    (committed at position plen-1, KV present); the first generated token
    //    is produced by the first spec round's verify (argmax_per_pos[0]),
    //    exactly as eager's first decode does — so we do NOT pre-emit it. The
    //    drafter's step-0 seed hidden = the target's post-`model.norm` hidden
    //    left in `state.tmp` after the last `decode_step` (the lm_head input).
    //
    //    NOTE: batched prefill via `forward_batch` is NOT used here. gemma4's
    //    batched MQ4 path faults at large B (b≥16) — a pre-existing
    //    `forward_batch` limitation surfaced by `verify_batch_gemma4 --bs 16+`,
    //    orthogonal to this spec wiring (the verify block stays at draft_len+1
    //    ≤ 7, well within the proven b≤8 range). Per-token prefill sidesteps it.
    let plen = prompt_ids.len();
    let t_pf = std::time::Instant::now();
    for (pos, &t) in prompt_ids.iter().enumerate() {
        decode_step(&cfg, &weights, &mut state, &mut gpu, t, pos as u32).expect("prefill");
    }
    // Seed hidden = post-`model.norm` hidden of the last prompt position.
    sp.set_seed_hidden_from(&gpu, &state.tmp).expect("seed hidden");
    eprintln!("prefill {} tok in {:.2}s", plen, t_pf.elapsed().as_secs_f64());

    // ── Warm the batched verify path on a disposable state before the spec
    //    loop. gemma4's b>1 batched MQ4 FFN GEMM lazily initializes scratch on
    //    first use and faults (illegal access) if its very first call is the one
    //    inside the loop (a pre-existing `forward_batch` fragility, unrelated to
    //    this spec wiring — see report). Priming with a throwaway `forward_batch`
    //    at b=1 THEN at the real block size (draft_len+1) initializes both the
    //    scalar and batched-WMMA scratch. (Q8 targets: cheap no-op.) Plain
    //    `forward_batch` (scalar last-row lm_head) is used — NOT
    //    `forward_batch_spec` — because the spec verify's per-position lm_head is
    //    the scalar per-row `weight_gemv` (already warm from prefill), so only
    //    the batched FFN GEMM needs priming. ──
    {
        // Prime the batched verify path (b=1 then the real block size) on a
        // disposable state. gemma4's b>1 batched MQ4 FFN GEMM lazily initializes
        // scratch on first use and can fault if its first call is the one inside
        // the loop. (Q8 targets: cheap no-op.) This is a pre-existing
        // `forward_batch` fragility, unrelated to this spec wiring — see report.
        // For MQ4-attn targets, set HIPFIRE_FP16=0 to route the batched MQ4 GEMM
        // through the deterministic scalar kernel (the gfx12 fp16-WMMA batched
        // path is non-deterministically unstable at this output width).
        let warm_b = draft_len + 1;
        let mut warm_state =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, warm_b + 4).expect("warm state");
        let _ = forward_batch(&cfg, &weights, &mut warm_state, &mut gpu, &[cfg.bos_token], 0);
        gpu.hip.device_synchronize().ok();
        let mut warm2 =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, warm_b + 4).expect("warm state 2");
        let _ = forward_batch(
            &cfg,
            &weights,
            &mut warm2,
            &mut gpu,
            &vec![cfg.bos_token; warm_b],
            0,
        );
        gpu.hip.device_synchronize().ok();
    }

    // Seed = last prompt token (at position plen-1). The first spec round's
    // verify re-writes its KV at L-1 and predicts the first generated token.
    let mut seed_token = *prompt_ids.last().unwrap();
    let mut gen: Vec<u32> = Vec::new();
    let mut rounds = 0usize;
    let mut total_accepted = 0usize; // accept_len summed (excludes per-round bonus)
    let mut stop = false;

    let t0 = std::time::Instant::now();
    while !stop && gen.len() < max {
        let committed_len = state.n_tokens; // L = positions filled = seed at L-1
        // Guard the KV/seq bound: block occupies [L-1, L-1 + draft_len + 1).
        if committed_len + draft_len + 1 >= max_seq {
            break;
        }
        let spec = spec_step_gemma4_eagle(
            &mut gpu,
            &weights,
            &cfg,
            &mut state,
            &dweights,
            &dcfg,
            &mut dscratch,
            &mut sp,
            seed_token,
            committed_len,
            draft_len,
            temperature,
        )
        .expect("spec step");
        rounds += 1;
        total_accepted += spec.accept_len;

        // Emit committed tokens (accepted drafts ++ bonus). Stop at EOS / max.
        for &t in &spec.committed {
            if is_eos(t) {
                stop = true;
                break;
            }
            gen.push(t);
            if gen.len() >= max {
                stop = true;
                break;
            }
        }
        // Next round's seed = this round's bonus (= last committed). Its hidden
        // is already staged in sp.seed_hidden by spec_step.
        seed_token = spec.next_seed_token;
    }
    let dt = t0.elapsed().as_secs_f64();

    // τ = mean tokens committed per round (accepted drafts + 1 bonus each).
    let committed_total = total_accepted + rounds; // each round commits accept_len + 1
    let tau = if rounds > 0 {
        committed_total as f64 / rounds as f64
    } else {
        0.0
    };
    let toks_per_s = if dt > 0.0 { gen.len() as f64 / dt } else { 0.0 };

    eprintln!(
        "spec-decode: {} tok in {:.2}s ({:.1} tok/s) | rounds={} τ={:.3} draft_len={}",
        gen.len(),
        dt,
        toks_per_s,
        rounds,
        tau,
        draft_len
    );

    match &tok {
        Some(t) => println!(
            "=== PROMPT ===\n{prompt}\n=== GENERATION (spec, draft_len={draft_len}) ===\n{}",
            t.decode(&gen)
        ),
        None => println!("=== GENERATION token ids ===\n{:?}", gen),
    }

    // ── Correctness gate ──
    if let Some(eager) = eager_gen {
        let identical = eager == gen;
        println!("\n=== CORRECTNESS GATE (spec vs eager greedy AR) ===");
        println!("eager  ({} tok): {:?}", eager.len(), &eager[..eager.len().min(64)]);
        println!("spec   ({} tok): {:?}", gen.len(), &gen[..gen.len().min(64)]);
        if identical {
            println!("=> IDENTICAL ✓");
        } else {
            // Find first divergence.
            let mut div = None;
            for k in 0..eager.len().min(gen.len()) {
                if eager[k] != gen[k] {
                    div = Some(k);
                    break;
                }
            }
            match div {
                Some(k) => println!(
                    "=> DIVERGE at index {k}: eager={} spec={} (eager_len={} spec_len={})",
                    eager[k],
                    gen[k],
                    eager.len(),
                    gen.len()
                ),
                None => println!(
                    "=> PREFIX-MATCH but length differs: eager_len={} spec_len={}",
                    eager.len(),
                    gen.len()
                ),
            }
            std::process::exit(1);
        }
    }

    eprintln!("token ids: {:?}", &gen[..gen.len().min(60)]);
}
