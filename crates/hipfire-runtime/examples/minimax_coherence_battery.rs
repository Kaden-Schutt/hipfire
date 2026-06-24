// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 prefill coherence battery.
//!
//! Validates that the scatter-grouped MoE prefill paths (FP16 grouped + i8
//! grouped) produce coherent output across a diverse prompt matrix — the
//! multi-prompt gate the single-prompt smoke check could not provide. Each
//! prompt is run in three modes that differ ONLY in the prefill MoE compute:
//!
//!   indexed       : HIPFIRE_MINIMAX_MOE_GROUPED=0 (per-token indexed GEMV)
//!   grouped_fp16  : scatter-grouped FP16 WMMA   (HIPFIRE_MINIMAX_MOE_I8=0)
//!   grouped_i8    : scatter-grouped i8 WMMA     (the default path on gfx1151)
//!
//! Decode (decode_step, B=1) is identical across modes, so any divergence
//! comes from the prefill MoE numerics propagating through the KV cache —
//! exactly what we want to gate. Each generation is fed through the shared
//! `hipfire-detect` bank. A cell is DEGENERATE (hard fail) iff it shows a
//! true single-token attractor (max_freq > 0.50) or a garbage/structural
//! detector (special-leak / whitespace-only / immediate-EOS / loop-guard /
//! n-gram density). The spec-decode unique-ratio sub-check is excluded — it
//! fires on legitimate greedy repetition of correct answers (the indexed
//! baseline trips it too). Decoded text is printed for human eyeball.
//!
//! Usage: minimax_coherence_battery --model <hfq> [--max N] [--modes a,b,c]
//! Exit code 1 if any (prompt, mode) cell fails.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax as minimax;
    use hipfire_detect::attractor::{AttractorFirst128, AttractorLast128};
    use hipfire_detect::eos_immediate::EosImmediate;
    use hipfire_detect::ngram::{LoopGuardMirror, NgramDensity};
    use hipfire_detect::special_leak::SpecialLeak;
    use hipfire_detect::whitespace_only::WhitespaceOnly;
    use hipfire_detect::{DetectorBank, Event};
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut max: usize = 256;
    let mut modes: Vec<String> = vec!["indexed".into(), "grouped_fp16".into(), "grouped_i8".into()];
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--modes" => {
                modes = argv[i + 1].split(',').map(|s| s.to_string()).collect();
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    // ── Prompt matrix (diverse distributions; one long prompt forces a
    //    multi-chunk grouped prefill). Raw prompts — the detectors gate the
    //    token stream, independent of chat formatting. ──
    let long_ctx = {
        let para = "Photosynthesis is the process by which green plants, algae, and \
            some bacteria convert light energy into chemical energy stored in glucose. \
            It occurs mainly in the chloroplasts, which contain the green pigment \
            chlorophyll. The process has two stages: the light-dependent reactions, \
            which capture energy from sunlight to produce ATP and NADPH, and the \
            Calvin cycle, which uses that energy to fix carbon dioxide into sugars. \
            Oxygen is released as a byproduct of splitting water molecules. ";
        let mut s = String::new();
        for _ in 0..6 {
            s.push_str(para);
        }
        s.push_str("\n\nBased on the passage above, summarize photosynthesis in one sentence:");
        s
    };
    let prompts: Vec<(&str, String)> = vec![
        ("factual", "The capital of France is".to_string()),
        (
            "reasoning",
            "If a farmer has 17 sheep and all but 9 die, how many sheep are left? \
             Answer with the number and a one-line explanation."
                .to_string(),
        ),
        (
            "code",
            "Write a Python function that returns the nth Fibonacci number.\n\ndef fib(n):"
                .to_string(),
        ),
        (
            "list",
            "List the first ten prime numbers, separated by commas:".to_string(),
        ),
        (
            "explain",
            "Explain in a short paragraph how a binary search algorithm works.".to_string(),
        ),
        ("longctx", long_ctx),
    ];

    let apply_mode = |mode: &str| {
        // Reset, then set per-mode levers (forward_batch reads these per call).
        std::env::remove_var("HIPFIRE_MINIMAX_MOE_GROUPED");
        std::env::set_var("HIPFIRE_MINIMAX_MOE_GROUPED_GATE", "8");
        // i8 is default-on, so the fp16 mode must explicitly opt OUT.
        match mode {
            "indexed" => {
                std::env::set_var("HIPFIRE_MINIMAX_MOE_GROUPED", "0");
                std::env::set_var("HIPFIRE_MINIMAX_MOE_I8", "0");
            }
            "grouped_fp16" => std::env::set_var("HIPFIRE_MINIMAX_MOE_I8", "0"),
            "grouped_i8" => std::env::set_var("HIPFIRE_MINIMAX_MOE_I8", "1"),
            other => panic!("unknown mode {other}"),
        }
    };

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &cfg, &mut gpu)
        .expect("weights");
    eprintln!(
        "loaded minimax: layers={} experts={}/{}",
        cfg.num_hidden_layers, cfg.num_local_experts, cfg.num_experts_per_tok
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
    const EOS: [u32; 4] = [200020, 151643, 151645, 2];

    // Per-(prompt, mode) set of degeneracy markers (true attractor / garbage).
    // Empty = coherent. Raw greedy prompts make the model answer correctly then
    // repeat (no chat template/EOS), which trips the spec-decode unique-ratio
    // sub-check on EVERY mode incl. indexed — so that sub-check is excluded from
    // the gate; only max_freq>0.50 and the garbage/structural detectors count.
    use std::collections::{BTreeMap, BTreeSet};
    let mut fail_sets: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    let mut summary: Vec<String> = Vec::new();

    for mode in &modes {
        apply_mode(mode);
        for (pname, ptext) in &prompts {
            let ids = tok.encode(ptext);
            let mut state =
                minimax::MiniMaxState::new_with_max_seq(&mut gpu, &cfg, ids.len() + max + 16)
                    .expect("state");

            // Prefill via forward_batch (chunk 512 → grouped when enabled).
            let mut pos = 0usize;
            let mut logits: Vec<f32> = Vec::new();
            for ck in ids.chunks(512) {
                logits =
                    minimax::forward::forward_batch(&cfg, &weights, &mut state, &mut gpu, ck, pos)
                        .unwrap_or_else(|e| panic!("[{mode}/{pname}] prefill: {e}"));
                pos += ck.len();
            }

            // Greedy decode + detector feed.
            let mut bank = DetectorBank::new();
            bank.add(Box::new(AttractorFirst128::new()));
            bank.add(Box::new(AttractorLast128::new()));
            bank.add(Box::new(NgramDensity::new()));
            bank.add(Box::new(LoopGuardMirror::new()));
            bank.add(Box::new(SpecialLeak::new()));
            bank.add(Box::new(WhitespaceOnly::new()));
            bank.add(Box::new(EosImmediate::new()));

            let mut gen: Vec<u32> = Vec::new();
            for step in 0..max {
                let next = argmax(&logits);
                if EOS.contains(&next) {
                    break;
                }
                let text = tok.decode(&[next]);
                bank.observe(&Event::Committed {
                    tok_id: next,
                    pos: step,
                    t_ms: 0,
                });
                bank.observe(&Event::Token {
                    text: &text,
                    t_ms: 0,
                    synthetic: false,
                });
                gen.push(next);
                logits = minimax::forward::decode_step(
                    &cfg, &weights, &mut state, &mut gpu, next, pos as u32,
                )
                .unwrap_or_else(|e| panic!("[{mode}/{pname}] decode: {e}"));
                pos += 1;
            }
            bank.observe(&Event::Done {
                total_tokens: gen.len(),
                total_visible_bytes: 0,
                wall_ms: 0,
                ttft_ms: 0,
            });
            let verdicts = bank.finalize();
            let fired: Vec<&'static str> = verdicts
                .iter()
                .filter(|(_, v)| v.is_fail() || v.is_warn())
                .map(|(n, _)| *n)
                .collect();

            // ── Self-computed token-loop metrics (transparent, not threshold-
            //    boundary-sensitive like the spec-decode-tuned unique-ratio). ──
            let max_freq = |toks: &[u32]| -> f64 {
                if toks.is_empty() {
                    return 0.0;
                }
                let mut counts: std::collections::HashMap<u32, usize> = Default::default();
                for &t in toks {
                    *counts.entry(t).or_insert(0) += 1;
                }
                *counts.values().max().unwrap() as f64 / toks.len() as f64
            };
            let mf_full = max_freq(&gen);

            // Real degeneracy = a TRUE single-token attractor (one token
            // dominating) or a garbage/structural detector firing. The
            // attractor unique-ratio sub-check fires on legitimate greedy
            // repetition of correct answers (present in the indexed baseline
            // too) and is reported as context, not a hard signal.
            let hard_dets = [
                "special_leak",
                "whitespace_only",
                "eos_immediate",
                "loop_guard_mirror",
                "ngram_density",
            ];
            let hard_fired: Vec<&'static str> = verdicts
                .iter()
                .filter(|(n, v)| v.is_fail() && hard_dets.contains(n))
                .map(|(n, _)| *n)
                .collect();
            let degenerate = mf_full > 0.50 || !hard_fired.is_empty();
            fail_sets.insert(
                (mode.clone(), pname.to_string()),
                if degenerate {
                    let mut s = BTreeSet::new();
                    if mf_full > 0.50 {
                        s.insert(format!("attractor(max_freq={mf_full:.2})"));
                    }
                    for h in &hard_fired {
                        s.insert(h.to_string());
                    }
                    s
                } else {
                    BTreeSet::new()
                },
            );

            // Free this prompt's KV/scratch — GpuTensor has no Drop, so without
            // this the 18 states leak and exhaust VRAM (GPU page fault).
            state.free_gpu(&mut gpu);

            let status = if degenerate { "DEGENERATE" } else { "coherent" };
            let text = tok.decode(&gen);
            let preview: String = text.chars().take(220).collect();
            println!(
                "\n=== [{mode}] {pname} ({} prompt tok → {} gen tok) : {status} (max_freq={mf_full:.2}) ===",
                ids.len(),
                gen.len()
            );
            if !fired.is_empty() {
                println!(
                    "  detectors fired (incl. greedy-verbosity soft fires): {}",
                    fired.join(", ")
                );
            }
            println!("  text: {}", preview.replace('\n', "\\n"));
            summary.push(format!(
                "{mode:<13} {pname:<10} {status} (max_freq={mf_full:.2})"
            ));
        }
    }

    println!("\n================ BATTERY SUMMARY ================");
    for s in &summary {
        println!("  {s}");
    }

    // ── Gate: a cell is DEGENERATE iff it shows a true single-token attractor
    //    (max_freq > 0.50) or a garbage/structural detector (special-leak /
    //    whitespace-only / immediate-EOS / loop-guard / n-gram density). The
    //    spec-decode unique-ratio sub-check is excluded — it fires on
    //    legitimate greedy repetition of correct answers and trips the indexed
    //    baseline too. FAIL the battery if ANY cell is degenerate. ──
    println!("\n================ COHERENCE GATE ================");
    let degenerate: Vec<String> = fail_sets
        .iter()
        .filter(|(_, s)| !s.is_empty())
        .map(|((m, p), s)| format!("{m}/{p}: {s:?}"))
        .collect();
    if degenerate.is_empty() {
        println!("  no cell is degenerate (max_freq < 0.50 everywhere, no garbage/leak/loop).");
        println!(
            "\nBATTERY: PASS ({} prompts × {} modes, no true attractors/garbage)",
            prompts.len(),
            modes.len()
        );
    } else {
        for d in &degenerate {
            println!("  DEGENERATE {d}");
        }
        println!("\nBATTERY: FAIL (degenerate output found)");
        std::process::exit(1);
    }
}
