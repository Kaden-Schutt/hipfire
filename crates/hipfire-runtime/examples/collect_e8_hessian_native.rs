// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — collect_e8_hessian_native: native per-(tensor,expert) GPTQ-on-E8
// Hessian collector for the qwen3_5_moe (A3B) routed experts.
//
// Drives a calibration forward over a DISJOINT corpus with
// `gpu.hessian_capture = Some(..)`. Every routed expert in the MoE CPU-top-K
// fallback (the path F32/Q8 experts take — non-indexable routed dtype) then
// accumulates its per-256-block XX^T over the RAW PRE-rotation input:
//   gate_up: post-rmsnorm hidden x  (K = hidden)
//   down:    silu(gate)*up           (K = mi)
// keyed by the FULL safetensors name == hipfire-quantize::main::hessian_key.
// After the pass, drains each (tensor,expert) BlockHessian to
//   <out-dir>/<sanitized_name>.hblk
// in the exact format hipfire-quantize::main::load_hessian_blocks reads
// (magic 0x45384831, n_blocks=K/256, K, then n_blocks*256*256 f32).
//
// PRECISION: H is precision-robust, so calibrate from a Q8/bf16 build (cheaper)
// OR the f32 oracle — whichever loads with F32/Q8 (non-indexable) experts so
// the capture-bearing CPU-top-K fallback fires. MQ4/MQ5/MQ6 experts route
// through the indexed GPU-top-K kernel which has NO host-visible per-expert x
// (capture would silently collect nothing) — so DO NOT calibrate from an MQ4+
// expert build.
//
// Usage:
//   collect_e8_hessian_native \
//       --model <oracle-or-q8.hfq> \
//       --slice <calib1.txt> [--slice <calib2.txt> ...] \
//       --out-dir /mnt/vol/e8hess \
//       [--n-ctx 512] [--max-chunks N]
//
// Build with `--features arch-qwen35,deltanet`.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features arch-qwen35,deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, HfqSource, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_runtime::model_load::Layout;
    use rdna_compute::HessianCapture;
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut slices: Vec<PathBuf> = Vec::new();
    let mut out_dir: PathBuf = PathBuf::from("/mnt/vol/e8hess");
    let mut n_ctx: usize = 512;
    let mut max_chunks: Option<usize> = None;
    // ROW_GATE: skip writing a .hblk for any (tensor,expert) whose accumulated
    // row count is below this threshold; the quantizer missing-file -> RTN
    // fallback then auto-gates under-sampled experts. Default 1024 = 4x the
    // 256 block-dim. Overridable via --row-gate or HIPFIRE_E8_ROW_GATE.
    let mut row_gate: u64 = std::env::var("HIPFIRE_E8_ROW_GATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--slice" => {
                slices.push(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--n-ctx" => {
                n_ctx = argv[i + 1].parse().expect("--n-ctx int");
                i += 2;
            }
            "--max-chunks" => {
                max_chunks = Some(argv[i + 1].parse().expect("--max-chunks int"));
                i += 2;
            }
            "--row-gate" => {
                row_gate = argv[i + 1].parse().expect("--row-gate int");
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: collect_e8_hessian_native --model <hfq> --slice <calib.txt> [--slice ...] --out-dir <dir> [--n-ctx 512] [--max-chunks N] [--row-gate 1024]");
                std::process::exit(0);
            }
            o => {
                eprintln!("unknown arg: {o}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    assert!(
        !slices.is_empty(),
        "at least one --slice <calib.txt> required"
    );

    // Force determinism knobs (mirror eval_hipfire / the HFIM collector).
    // SAFETY: single-threaded init phase.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GRAPH_MOE", "0");
    }

    // -------- load model + tokenizer --------
    let mut hfq = HfqFile::open(&model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    assert!(
        config.num_experts > 0,
        "model is not MoE (num_experts==0) — Hessian capture targets routed experts only"
    );
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "collect_e8_hessian_native: arch={} model={} n_experts={} top_k={} hidden={} mi={}",
        gpu.arch,
        model.display(),
        config.num_experts,
        config.num_experts_per_tok,
        config.dim,
        config.moe_intermediate_size,
    );
    let mut source = HfqSource::new(&mut hfq, &config);
    let layout = Layout::single(config.n_layers);
    let weights = qwen35::load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout)
        .expect("load weights");
    eprintln!(
        "loaded {} layers, vocab={}, n_ctx={}",
        weights.layers.len(),
        config.vocab_size,
        n_ctx
    );

    // -------- tokenize the calibration corpus (concat all slices) --------
    let mut text = String::new();
    for s in &slices {
        let t = std::fs::read_to_string(s).unwrap_or_else(|e| {
            eprintln!("error: read slice {}: {e}", s.display());
            std::process::exit(1);
        });
        text.push_str(&t);
        text.push('\n');
        eprintln!("  + {} ({} bytes)", s.display(), t.len());
    }
    let tokens: Vec<u32> = tokenizer.encode(&text);
    eprintln!(
        "hipfire tokenize: {} tokens from {} slice(s)",
        tokens.len(),
        slices.len()
    );

    let mut n_chunk = tokens.len() / n_ctx;
    if let Some(m) = max_chunks {
        n_chunk = n_chunk.min(m);
    }
    assert!(
        n_chunk >= 1,
        "not enough tokens for one n_ctx={n_ctx} chunk"
    );
    let tokens: Vec<u32> = tokens[..n_chunk * n_ctx].to_vec();
    eprintln!(
        "calibrating over {} chunks of n_ctx={} ({} tokens)",
        n_chunk,
        n_ctx,
        tokens.len()
    );

    // -------- KV cache + DeltaNet + scratch --------
    // q8 KV (cheap, H is precision-robust); F32 experts still route through the
    // capture-bearing CPU-top-K fallback regardless of KV mode.
    let kv_max = n_ctx + 16;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .expect("new_gpu_q8 kv");
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, kv_max).expect("scratch");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn_state");

    // -------- enable capture, run the forward, accumulate XX^T --------
    gpu.hessian_capture = Some(HessianCapture::default());
    let t0 = Instant::now();
    let mut steps = 0u64;
    for c in 0..n_chunk {
        dn_state.reset(&mut gpu);
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        for pos in 0..(n_ctx - 1) {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                chunk[pos],
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("forward_scratch");
            steps += 1;
            if let Some(cap) = gpu.hessian_capture.as_mut() {
                cap.n_tokens = steps;
            }
            if steps % 64 == 0 {
                let el = t0.elapsed().as_secs_f64();
                let n_ent = gpu
                    .hessian_capture
                    .as_ref()
                    .map(|c| c.entries.len())
                    .unwrap_or(0);
                eprint!(
                    "\r  chunk {:4}/{}  step {:7}  entries {:6}  ({:.1} tok/s)   ",
                    c + 1,
                    n_chunk,
                    steps,
                    n_ent,
                    steps as f64 / el.max(1e-9)
                );
            }
        }
    }
    eprintln!();

    // -------- drain per-(tensor,expert) .hblk --------
    let cap = gpu.hessian_capture.take().expect("capture present");
    eprintln!(
        "captured {} (tensor,expert) Hessians over {} forward steps in {:.1}s",
        cap.entries.len(),
        cap.n_tokens,
        t0.elapsed().as_secs_f64()
    );
    if cap.entries.is_empty() {
        eprintln!(
            "error: NO Hessians captured. Either the model's experts are MQ4+ \
             (indexed GPU-top-K path has no host-visible per-expert x — \
             calibrate from an F32/Q8-expert build) or no tokens routed. \
             This is a CAPTURE BUG, not a flat result."
        );
        std::process::exit(1);
    }

    std::fs::create_dir_all(&out_dir).expect("mkdir out-dir");
    let mut n_written = 0u64;
    let mut total_rows = 0u64;
    let mut gate_up_cnt = 0u64;
    let mut down_cnt = 0u64;
    let mut min_diag = f64::INFINITY;
    let mut max_diag = f64::NEG_INFINITY;
    let mut zero_diag = 0u64;
    // ROW_GATE bookkeeping: count experts cleared/gated + row-count distribution.
    let mut gated_skipped = 0u64;
    let mut cleared = 0u64;
    let mut row_counts: Vec<u64> = Vec::with_capacity(cap.entries.len());
    eprintln!(
        "ROW_GATE = {row_gate} rows (skip .hblk below this; quantizer RTN-fallback gates them)"
    );
    // Deterministic order for the log.
    let mut names: Vec<&String> = cap.entries.keys().collect();
    names.sort();
    for name in names {
        let acc = &cap.entries[name];
        let md = acc.mean_diag();
        if md < min_diag {
            min_diag = md;
        }
        if md > max_diag {
            max_diag = md;
        }
        if md <= 0.0 {
            zero_diag += 1;
        }
        total_rows += acc.n_rows;
        row_counts.push(acc.n_rows);
        // ROW_GATE: under-sampled (tensor,expert) Hessians are unreliable for LDLQ
        // (a 256-dim block needs >> 256 rows for a well-conditioned H). Skip the
        // .hblk; the quantizer falls back to RTN-E8 for any missing file.
        if acc.n_rows < row_gate {
            gated_skipped += 1;
            continue;
        }
        cleared += 1;
        if name.ends_with("gate_up_proj.weight") {
            gate_up_cnt += 1;
        }
        if name.ends_with("down_proj.weight") {
            down_cnt += 1;
        }
        acc.write_hblk(&out_dir, name).unwrap_or_else(|e| {
            eprintln!("error: write_hblk {name}: {e}");
            std::process::exit(1);
        });
        n_written += 1;
    }
    eprintln!(
        "wrote {} .hblk files to {} ({} gate_up, {} down; total_rows={}, mean_diag range [{:.4e}, {:.4e}], zero-diag entries={})",
        n_written, out_dir.display(), gate_up_cnt, down_cnt, total_rows, min_diag, max_diag, zero_diag,
    );
    // ROW_GATE distribution report.
    row_counts.sort_unstable();
    let n_seen = row_counts.len() as u64;
    let mean_rows = if n_seen > 0 {
        total_rows as f64 / n_seen as f64
    } else {
        0.0
    };
    let median_rows = if n_seen > 0 {
        row_counts[(n_seen / 2) as usize]
    } else {
        0
    };
    let min_rows = row_counts.first().copied().unwrap_or(0);
    let max_rows = row_counts.last().copied().unwrap_or(0);
    eprintln!(
        "ROW_GATE stats: {} (tensor,expert) seen; {} cleared (>= {} rows, wrote .hblk) / {} gated-to-RTN (< {} rows). rows: min={} median={} mean={:.0} max={}",
        n_seen, cleared, row_gate, gated_skipped, row_gate, min_rows, median_rows, mean_rows, max_rows,
    );
    // Spot-check: print 3 example keys so a name-mismatch can't pass silently.
    let mut ex: Vec<&String> = cap.entries.keys().collect();
    ex.sort();
    for k in ex.iter().take(3) {
        let acc = &cap.entries[*k];
        eprintln!(
            "  e.g. {k}  (K={}, n_blocks={}, rows={}, mean_diag={:.4e})",
            acc.k,
            acc.n_blocks,
            acc.n_rows,
            acc.mean_diag()
        );
    }
}
