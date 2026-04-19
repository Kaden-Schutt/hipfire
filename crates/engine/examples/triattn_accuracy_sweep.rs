//! Lightweight accuracy sweep under TriAttention eviction.
//!
//! Runs a small prompt set of reasoning/QA questions with known-good
//! expected-answer substrings. For each (prompt, budget_fraction) pair,
//! runs greedy decode with TriAttention eviction at the requested
//! budget and checks whether the expected substring appears in the
//! output. Reports pass rate per budget fraction — directional signal
//! on whether eviction preserves correctness.
//!
//! This is a cheap smoke test, not a rigorous benchmark. It uses
//! moderate-difficulty prompts that Qwen3.5-9B handles reliably at full
//! attention, so degradations under aggressive eviction show up as
//! clear test failures.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::cask::CaskCtx;
    use engine::hfq::HfqFile;
    use engine::llama::{self, KvCache};
    use engine::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch};
    use engine::tokenizer::Tokenizer;
    use engine::triattn::{EvictionCtx, TriAttnCenters};
    use rdna_compute::Gpu;
    use std::path::Path;

    enum Policy { Plain(EvictionCtx), Cask(CaskCtx) }
    impl Policy {
        fn maybe_evict(&self, gpu: &mut Gpu, kv: &mut KvCache, physical: usize)
            -> hip_bridge::HipResult<Option<engine::triattn::EvictionResult>>
        {
            match self {
                Policy::Plain(c) => c.maybe_evict(gpu, kv, physical),
                Policy::Cask(c) => c.maybe_evict(gpu, kv, physical),
            }
        }
        fn eviction_count(&self) -> usize {
            match self {
                Policy::Plain(c) => c.eviction_count.get(),
                Policy::Cask(c) => c.eviction_count(),
            }
        }
    }

    // Positional args first, then optional flags.
    let raw_args: Vec<String> = std::env::args().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut use_cask = false;
    let mut core_frac: f32 = 0.5;
    let mut fold_m: usize = 2;
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--cask" => { use_cask = true; i += 1; }
            "--core-frac" => { core_frac = raw_args[i + 1].parse().unwrap(); i += 2; }
            "--fold-m" => { fold_m = raw_args[i + 1].parse().unwrap(); i += 2; }
            _ => { positional.push(raw_args[i].clone()); i += 1; }
        }
    }
    if positional.len() < 2 {
        eprintln!("Usage: triattn_accuracy_sweep <model> <sidecar> [kv_mode=asym3] [gen=48] [--cask] [--core-frac 0.5] [--fold-m 2]");
        std::process::exit(1);
    }
    let model_path = &positional[0];
    let sidecar_path = &positional[1];
    let kv_mode: String = positional.get(2).cloned().unwrap_or_else(|| "asym3".into());
    let gen_len: usize = positional.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);

    // (prompt, expected answer substring — lowercased). Prompts chosen
    // to be small enough that the baseline passes; failures under
    // heavy eviction indicate real degradation.
    let test_cases: &[(&str, &str)] = &[
        ("Question: What is the capital of France? Answer:", "paris"),
        ("Question: What is 25 times 17? Let's compute step by step. 25 × 10 = 250. 25 × 7 = 175. 250 + 175 =", "425"),
        ("Question: Who wrote the play Hamlet? Answer:", "shakespeare"),
        ("Question: What is the chemical symbol for water? Answer:", "h2o"),
        ("Question: If x + 5 = 12, what is x? Let's solve. Subtract 5 from both sides: x =", "7"),
        ("Question: What planet is closest to the sun? Answer:", "mercury"),
        ("Question: Solve 3x = 21. Divide both sides by 3: x =", "7"),
        ("Question: In what year did World War II end? Answer:", "1945"),
        // Arithmetic word problems (GSM8K-shaped, single-number final answer).
        // Answers are distinctive enough that a substring match is reliable.
        ("Question: A train travels at 60 mph for 2.5 hours. How many miles? Let me compute: 60 × 2.5 =", "150"),
        ("Question: 15% of 80 is what? Compute: 80 × 0.15 =", "12"),
        ("Question: A square has a side length of 7 cm. Its perimeter is 4 × 7 =", "28"),
        ("Question: The average of 10, 20, and 30 is (10 + 20 + 30) / 3 = 60 / 3 =", "20"),
        ("Question: How many days are in a leap year? Answer:", "366"),
        ("Question: What is 12 squared? 12 × 12 =", "144"),
        ("Question: A shirt costs $40 with 25% off. Discount is 40 × 0.25 = $10. Final price: 40 - 10 =", "30"),
        ("Question: How many hours in 3 days? 24 × 3 =", "72"),
        ("Question: If 4 apples cost $2, how much do 10 apples cost? Price per apple: 2/4 = 0.5. Total: 10 × 0.5 =", "5"),
        ("Question: What is the 5th prime number? 2, 3, 5, 7, 11. The 5th is", "11"),
    ];

    let budget_fractions: &[f32] = &[1.00, 0.50, 0.25];

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let centers = TriAttnCenters::load(Path::new(sidecar_path)).expect("load sidecar");
    let fa_layer_ids: Vec<usize> = config.layer_types.iter().enumerate()
        .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
        .collect();
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");

    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
    let alloc_kv = |gpu: &mut Gpu, seq: usize| match kv_mode.as_str() {
        "asym3" => KvCache::new_gpu_asym3(gpu, config.n_layers, config.n_kv_heads, config.head_dim, seq).unwrap(),
        _ => KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, seq).unwrap(),
    };

    let policy_tag = if use_cask {
        format!("CASK(α={:.2}, m={})", core_frac, fold_m)
    } else {
        "TriAttention".to_string()
    };
    eprintln!("Accuracy sweep: {} prompts × {} budgets (kv={}, policy={}, gen={})",
        test_cases.len(), budget_fractions.len(), kv_mode, policy_tag, gen_len);

    // results[fraction_idx] = (pass_count, fail_count)
    let mut results: Vec<(usize, usize)> = budget_fractions.iter().map(|_| (0, 0)).collect();

    for (prompt_i, (prompt, expected)) in test_cases.iter().enumerate() {
        let ptokens = tok.encode(prompt);
        let plen = ptokens.len();

        for (fi, &frac) in budget_fractions.iter().enumerate() {
            // Budget = max(8, frac × prompt_len). For frac=1.0 we disable
            // eviction; for frac<1.0 we evict during prefill with
            // beta = budget/4 so a couple of evictions fire on a
            // small prompt.
            let budget = if frac >= 0.999 { plen + gen_len + 16 } else { (frac * plen as f32).round() as usize };
            let beta = (budget / 4).max(4);
            let alloc_seq = (budget + beta + 8).max(plen + gen_len + 16);
            let tight_seq = if frac >= 0.999 { plen + gen_len + 16 } else { budget + beta + 4 };

            let mut kv = alloc_kv(&mut gpu, tight_seq.max(plen + gen_len + 16).min(alloc_seq));
            // Simpler: allocate big enough for any scenario; eviction still
            // caps physical at budget during run.
            let _ = kv.max_seq;
            let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();

            // Re-alloc with tight sizing when eviction is active.
            if frac < 0.999 {
                kv = alloc_kv(&mut gpu, budget + beta + 4);
            }

            let ctx_opt = if frac < 0.999 {
                let base = EvictionCtx::new(
                    &mut gpu, &centers, fa_layer_ids.clone(),
                    budget, beta,
                    config.n_heads, config.n_kv_heads, config.head_dim,
                    n_rot, config.rope_theta, kv.max_seq,
                ).unwrap();
                Some(if use_cask {
                    Policy::Cask(CaskCtx::new(base, core_frac, fold_m))
                } else {
                    Policy::Plain(base)
                })
            } else { None };

            let mut physical = 0usize;
            for t in ptokens.iter() {
                qwen35::forward_scratch(&mut gpu, &weights, &config, *t, physical, &mut kv, &mut dn, &scratch).unwrap();
                physical += 1;
                if let Some(ctx) = ctx_opt.as_ref() {
                    if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
                        physical = ev.new_physical;
                    }
                }
            }
            let mut logits = gpu.download_f32(&scratch.logits).unwrap();
            let mut next = llama::argmax(&logits);
            let mut emitted = vec![next];
            for _ in 0..gen_len {
                qwen35::forward_scratch(&mut gpu, &weights, &config, next, physical, &mut kv, &mut dn, &scratch).unwrap();
                physical += 1;
                if let Some(ctx) = ctx_opt.as_ref() {
                    if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
                        physical = ev.new_physical;
                    }
                }
                logits = gpu.download_f32(&scratch.logits).unwrap();
                next = llama::argmax(&logits);
                emitted.push(next);
                if next == config.eos_token { break; }
            }
            let text = tok.decode(&emitted).to_lowercase();
            let pass = text.contains(expected);
            let ev = ctx_opt.as_ref().map(|c| c.eviction_count()).unwrap_or(0);
            if pass {
                results[fi].0 += 1;
            } else {
                results[fi].1 += 1;
            }
            eprintln!(
                "  p{prompt_i} frac={frac:.2} budget={budget:<3} ev={ev:<2} {}  [{}]",
                if pass { "✓" } else { "✗" },
                text.lines().next().unwrap_or("").trim().chars().take(90).collect::<String>(),
            );
        }
    }

    eprintln!("\n=== SUMMARY ===");
    for (fi, &frac) in budget_fractions.iter().enumerate() {
        let (pass, fail) = results[fi];
        let total = pass + fail;
        let rate = 100.0 * pass as f32 / total as f32;
        eprintln!("  frac={frac:.2}  pass {pass}/{total}  ({rate:.0}%)");
    }
}
