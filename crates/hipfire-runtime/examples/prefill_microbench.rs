//! prefill_microbench — scoring-mode speedup measurement.
//!
//! Originally the Step 0 gate for the eval_hipfire_speedup sub-plan
//! (now folded into docs/plans/issue-113-quant-quality-eval.md §5);
//! retained as a standalone tool for re-measurement when kernels change.
//!
//! Times one 2048-token chunk through two paths on a hipfire model:
//!   A) per-token forward_scratch × n_ctx  (matches eval_hipfire's current
//!      scoring inner loop)
//!   B) forward_prefill_batch(tokens, start_pos=0)  (the proposed batched
//!      path; transformer-stack only, no per_token_hidden_out capture,
//!      no lm_head fan-out — that's measured separately in Step 6)
//!
//! Reports min / mean / max wall-clock per path over `--measure-iters`
//! iterations, plus the A/B speedup ratio. Per-iter KV cache + DN state
//! are reset by re-using position 0 for forward_scratch and resetting
//! dn_state explicitly.
//!
//! Decision rule (now folded into issue-113-quant-quality-eval.md §5;
//! retained here as the original gate framing):
//!   ≥ 4× speedup → use prefill mode as canonical (current state)
//!   2-4×         → use prefill but expect more modest wall-clock wins
//!   < 2×         → DN sequentiality dominates; pursue a batched DN kernel
//!                  instead of relying on eval_hipfire-side batching
//!
//! Usage:
//!   prefill_microbench --model <path-to-hfq-model> \
//!                      [--n-ctx <2048>] \
//!                      [--kv-mode <asym3|q8|asym4|asym2>=asym3] \
//!                      [--warmup-iters <1>] \
//!                      [--measure-iters <3>]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    struct Args {
        model: PathBuf,
        n_ctx: usize,
        kv_mode: String,
        warmup_iters: usize,
        measure_iters: usize,
    }
    let mut model: Option<PathBuf> = None;
    let mut n_ctx: usize = 2048;
    let mut kv_mode = "asym3".to_string();
    let mut warmup_iters: usize = 1;
    let mut measure_iters: usize = 3;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--n-ctx" => { n_ctx = argv[i + 1].parse().expect("--n-ctx"); i += 2; }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4") {
                    eprintln!("--kv-mode must be one of: q8 asym2 asym3 asym4 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v; i += 2;
            }
            "--warmup-iters" => { warmup_iters = argv[i + 1].parse().expect("--warmup-iters"); i += 2; }
            "--measure-iters" => { measure_iters = argv[i + 1].parse().expect("--measure-iters"); i += 2; }
            "-h" | "--help" => {
                eprintln!("Usage: prefill_microbench --model <path> [--n-ctx 2048] [--kv-mode asym3] [--warmup-iters 1] [--measure-iters 3]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }
    let args = Args {
        model: model.expect("--model required"),
        n_ctx,
        kv_mode,
        warmup_iters,
        measure_iters,
    };

    // -------- eval-mode env vars (must precede Gpu::init) --------
    // Match eval_hipfire's flag forcing so the microbench measures the
    // same code-path eval_hipfire takes today. Also turn on PBS reuse so
    // the prefill scratch is allocated once and the second-and-onward
    // iterations don't pay alloc cost.
    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
        std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
    }
    eprintln!(
        "prefill_microbench: HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 \
         HIPFIRE_KV_MODE={} HIPFIRE_PREFILL_REUSE_PBS=1",
        args.kv_mode
    );

    // -------- load model --------
    let mut hfq = HfqFile::open(&args.model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("prefill_microbench: arch={} model={}", gpu.arch, args.model.display());
    if gpu.arch.starts_with("gfx12") {
        unsafe { std::env::set_var("HIPFIRE_LLOYD_GFX12", "1"); }
    }
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    // -------- KV cache + DN state + scratch --------
    let kv_max = args.n_ctx + 16;
    let mut kv_cache = match args.kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "asym2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        other => panic!("unknown --kv-mode: {other}"),
    };
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

    // Eligibility check for the batched path. If is_batchable_la rejects
    // the model's LA dtype, `forward_prefill_batch` auto-falls-back to
    // per-token internally and the "B path" would just measure the same
    // thing as the "A path". Surface this clearly so the user knows the
    // microbench can't say anything useful in that case.
    eprintln!(
        "prefill_microbench: arch={}; if forward_prefill_batch logs an \
         eligibility-fallback message during warmup, the speedup ratio \
         below is for a per-token fallback path on BOTH sides and is \
         meaningless.",
        gpu.arch
    );

    // -------- token sequence --------
    // The microbench measures kernel-level cost, which is data-independent
    // (all matmuls are bandwidth-bound, not value-dependent). Use a
    // deterministic synthetic token stream in [1, vocab) so any timing
    // run reproduces. Token 0 is reserved (often a special token).
    let tokens: Vec<u32> = (0..args.n_ctx)
        .map(|i| ((i as u32).wrapping_mul(2654435761) % (config.vocab_size as u32 - 1)) + 1)
        .collect();
    eprintln!(
        "prefill_microbench: n_ctx={} vocab_size={} (synthetic token stream)",
        args.n_ctx, config.vocab_size
    );

    // -------- timing helpers --------
    let per_token_path = |gpu: &mut rdna_compute::Gpu,
                          kv_cache: &mut KvCache,
                          dn_state: &mut DeltaNetState,
                          tokens: &[u32]| -> f64 {
        // Reset DN; KV is overwritten in place from pos 0 per token.
        dn_state.reset(gpu);
        let t0 = Instant::now();
        for pos in 0..tokens.len() {
            qwen35::forward_scratch(
                gpu, &weights, &config, tokens[pos], pos,
                kv_cache, dn_state, &scratch,
            ).expect("forward_scratch");
        }
        // One trailing sync to ensure all GPU work is done before we stop
        // the clock. Hipfire's forward_scratch already synchronises at the
        // logits download internally, but be explicit.
        gpu.hip.device_synchronize().expect("sync");
        t0.elapsed().as_secs_f64()
    };

    let prefill_path = |gpu: &mut rdna_compute::Gpu,
                        kv_cache: &mut KvCache,
                        dn_state: &mut DeltaNetState,
                        tokens: &[u32]| -> f64 {
        dn_state.reset(gpu);
        let t0 = Instant::now();
        qwen35::forward_prefill_batch(
            gpu, &weights, &config, tokens, 0,
            kv_cache, dn_state, &scratch,
            None, None, None, None,
        ).expect("forward_prefill_batch");
        gpu.hip.device_synchronize().expect("sync");
        t0.elapsed().as_secs_f64()
    };

    // -------- warmup --------
    eprintln!("prefill_microbench: warmup ({} iters each path)", args.warmup_iters);
    for _ in 0..args.warmup_iters {
        let _ = per_token_path(&mut gpu, &mut kv_cache, &mut dn_state, &tokens);
        let _ = prefill_path(&mut gpu, &mut kv_cache, &mut dn_state, &tokens);
    }

    // -------- measure (alternating, so any background drift hits both) --------
    let mut per_token_times: Vec<f64> = Vec::with_capacity(args.measure_iters);
    let mut prefill_times: Vec<f64> = Vec::with_capacity(args.measure_iters);
    for iter in 0..args.measure_iters {
        let pt = per_token_path(&mut gpu, &mut kv_cache, &mut dn_state, &tokens);
        let pb = prefill_path(&mut gpu, &mut kv_cache, &mut dn_state, &tokens);
        eprintln!(
            "  iter {}: per-token {:.3}s ({:.1} tok/s)  prefill {:.3}s ({:.1} tok/s)  speedup {:.2}×",
            iter + 1, pt, args.n_ctx as f64 / pt, pb, args.n_ctx as f64 / pb, pt / pb
        );
        per_token_times.push(pt);
        prefill_times.push(pb);
    }

    // -------- report --------
    fn stats(xs: &[f64]) -> (f64, f64, f64) {
        let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = xs.iter().copied().sum::<f64>() / xs.len() as f64;
        (min, mean, max)
    }
    let (pt_min, pt_mean, pt_max) = stats(&per_token_times);
    let (pb_min, pb_mean, pb_max) = stats(&prefill_times);
    let speedup_mean = pt_mean / pb_mean;
    let speedup_best = pt_min / pb_min;

    eprintln!();
    eprintln!("===== prefill_microbench summary (n_ctx={}, kv_mode={}) =====", args.n_ctx, args.kv_mode);
    eprintln!(
        "per-token forward_scratch × {} : min {:.3}s  mean {:.3}s  max {:.3}s  ({:.1} tok/s mean)",
        args.n_ctx, pt_min, pt_mean, pt_max, args.n_ctx as f64 / pt_mean,
    );
    eprintln!(
        "forward_prefill_batch          : min {:.3}s  mean {:.3}s  max {:.3}s  ({:.1} tok/s mean)",
        pb_min, pb_mean, pb_max, args.n_ctx as f64 / pb_mean,
    );
    eprintln!("speedup (per-token / prefill) : mean {:.2}×  best {:.2}×", speedup_mean, speedup_best);
    eprintln!();
    eprintln!("Decision rule (docs/plans/issue-113-quant-quality-eval.md §5):");
    eprintln!("  ≥ 4×    : continue with rev-2 plan as designed");
    eprintln!("  2× – 4× : continue with rescoped target");
    eprintln!("  < 2×    : halt; DN sequentiality dominates — pursue batched-DN kernel instead");
}
