// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! REAP importance probe for Qwen3.5-A3B (or any Qwen3.5-MoE).
//!
//! Accumulates per-(layer, expert): (count, gate_weight_sum, contribution_mass)
//! where contribution_mass = Σ gate_weight × ‖expert_output‖₂ over a corpus.
//!
//! Usage:
//!   HIPFIRE_REAP_PROBE=1 cargo run --release --features deltanet \
//!     --example reap_probe -- /workspace/q35a3b.mq4 /workspace/reap_corpus.txt \
//!     --out /workspace/reap_dump.tsv
//!
//! Corpus file: one document per line (or raw text). 30-50k tokens recommended.
//! Output TSV: layer_idx  expert_idx  count  gate_sum  contribution_mass

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;
    use std::io::{BufRead, Write};

    // Force REAP_PROBE env on in case caller forgot
    std::env::set_var("HIPFIRE_REAP_PROBE", "1");
    // Disable hipGraph (D2H syncs not allowed under capture)
    std::env::set_var("HIPFIRE_GRAPH", "0");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: reap_probe <model.mq4> <corpus.txt> [--out dump.tsv] [--max-tokens N]");
        std::process::exit(1);
    }
    let model_path = Path::new(&args[1]);
    let corpus_path = Path::new(&args[2]);

    let mut out_path = "/workspace/reap_dump.tsv".to_string();
    let mut max_tokens: usize = 50_000;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => { i += 1; if i < args.len() { out_path = args[i].clone(); } }
            "--max-tokens" => { i += 1; if i < args.len() { max_tokens = args[i].parse().unwrap_or(50_000); } }
            _ => {}
        }
        i += 1;
    }

    eprintln!("[reap_probe] model: {}", model_path.display());
    eprintln!("[reap_probe] corpus: {}", corpus_path.display());
    eprintln!("[reap_probe] max_tokens: {max_tokens}");
    eprintln!("[reap_probe] out: {out_path}");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("[reap_probe] GPU: arch={}", gpu.arch);

    let mut hfq = HfqFile::open(model_path).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    assert!(config.num_experts > 0, "this probe requires a MoE model (num_experts > 0)");
    let n_exp = config.num_experts;
    let n_layers = config.n_layers;
    eprintln!("[reap_probe] A3B config: dim={}, layers={}, experts={}, top_k={}, moe_inter={}",
        config.dim, n_layers, n_exp, config.num_experts_per_tok, config.moe_intermediate_size);

    eprintln!("[reap_probe] Loading weights ...");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    eprintln!("[reap_probe] Loaded {} layers.", weights.layers.len());

    let kv_max = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, n_layers, config.n_kv_heads, config.head_dim, kv_max,
    ).expect("kv cache alloc");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state alloc");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch alloc");

    // Read corpus
    let file = std::fs::File::open(corpus_path).expect("open corpus");
    let reader = std::io::BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap_or_default()).collect();
    eprintln!("[reap_probe] corpus: {} lines", lines.len());

    let mut total_tokens = 0usize;
    let mut doc_count = 0usize;

    'outer: for line in &lines {
        if line.trim().is_empty() { continue; }
        let tokens = tokenizer.encode(line.trim());
        if tokens.is_empty() { continue; }

        // Reset KV cache + state for each document (fresh context window)
        // Reuse the same allocations but reset position tracking
        kv_cache = KvCache::new_gpu_q8(
            &mut gpu, n_layers, config.n_kv_heads, config.head_dim, kv_max,
        ).expect("kv reset");
        dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state reset");

        let chunk_tokens: Vec<u32> = tokens.into_iter()
            .take(kv_max.saturating_sub(1))
            .collect();

        for (pos, &tok) in chunk_tokens.iter().enumerate() {
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, tok, pos,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward_scratch failed");
            total_tokens += 1;
            if total_tokens >= max_tokens {
                eprintln!("[reap_probe] reached max_tokens={max_tokens}, stopping.");
                break 'outer;
            }
        }
        doc_count += 1;
        if doc_count % 50 == 0 {
            eprintln!("[reap_probe] processed {doc_count} docs, {total_tokens} tokens so far");
        }
    }

    eprintln!("[reap_probe] corpus pass done: {total_tokens} tokens over {doc_count} docs");

    // Retrieve accumulated REAP data
    let (nl, ne, acc) = hipfire_dispatch::pipeline::reap_take_dump()
        .expect("REAP accumulator not populated — was HIPFIRE_REAP_PROBE=1 set?");

    assert!(total_tokens > 0, "no tokens processed — corpus was empty?");

    // Verify we got real data
    let total_mass: f64 = acc.iter().flat_map(|l| l.iter()).map(|row| row[2]).sum();
    let total_count: f64 = acc.iter().flat_map(|l| l.iter()).map(|row| row[0]).sum();
    eprintln!("[reap_probe] total_mass={:.4e}  total_count={:.0}", total_mass, total_count);
    assert!(total_mass > 0.0, "contribution_mass is all-zero — instrumentation did not fire");
    assert!(total_count > 0.0, "count is all-zero — no MoE layers were hit");

    // Dump TSV
    let out = std::fs::File::create(&out_path).expect("create output file");
    let mut w = std::io::BufWriter::new(out);
    writeln!(w, "layer_idx\texpert_idx\tcount\tgate_sum\tcontribution_mass").unwrap();
    let actual_layers = nl.min(n_layers);
    for li in 0..actual_layers {
        if li >= acc.len() { break; }
        for ei in 0..ne {
            if ei >= acc[li].len() { break; }
            let [count, gate_sum, mass] = acc[li][ei];
            if count > 0.0 {
                writeln!(w, "{li}\t{ei}\t{:.0}\t{:.6}\t{:.6}", count, gate_sum, mass).unwrap();
            }
        }
    }
    w.flush().unwrap();
    eprintln!("[reap_probe] dump written to {out_path}");

    // Per-layer summary: contribution_mass Gini + bottom-50%-by-mass share
    eprintln!("\n=== Per-layer REAP summary ===");
    eprintln!("{:>6}  {:>8}  {:>10}  {:>14}  {:>14}",
        "layer", "n_active", "gini_mass", "bot50%_share", "top10%_share");

    let mut global_ginis: Vec<f64> = Vec::new();
    let mut global_bot50: Vec<f64> = Vec::new();

    for li in 0..actual_layers {
        if li >= acc.len() { break; }
        let layer = &acc[li];
        // Collect all experts with count > 0
        let mut masses: Vec<f64> = layer.iter().map(|row| row[2]).filter(|&m| m > 0.0).collect();
        if masses.is_empty() { continue; }
        masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = masses.len();
        let total: f64 = masses.iter().sum();
        if total == 0.0 { continue; }

        // Gini coefficient
        let gini = {
            let mut g = 0f64;
            for (i, &m) in masses.iter().enumerate() {
                g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * m;
            }
            g / (n as f64 * total)
        };

        // Bottom-50%-by-mass share: sort ascending, find the experts whose
        // total mass covers <= 50% of the experts by count, measure their
        // fraction of total mass.
        let bottom_half_n = n / 2;
        let bot50_mass: f64 = masses[..bottom_half_n].iter().sum();
        let bot50_share = bot50_mass / total;

        // Top-10%-by-mass share
        let top10_n = (n as f64 * 0.1).ceil() as usize;
        let top10_mass: f64 = masses[n - top10_n..].iter().sum();
        let top10_share = top10_mass / total;

        global_ginis.push(gini);
        global_bot50.push(bot50_share);

        eprintln!("{:>6}  {:>8}  {:>10.4}  {:>14.4}  {:>14.4}",
            li, n, gini, bot50_share, top10_share);
    }

    if !global_ginis.is_empty() {
        let mean_gini: f64 = global_ginis.iter().sum::<f64>() / global_ginis.len() as f64;
        let mean_bot50: f64 = global_bot50.iter().sum::<f64>() / global_bot50.len() as f64;
        eprintln!("\n=== Aggregate ===");
        eprintln!("mean Gini(contribution_mass) = {mean_gini:.4}");
        eprintln!("mean bottom-50%-experts mass share = {mean_bot50:.4}  ({:.1}%)", mean_bot50 * 100.0);
        eprintln!("total tokens processed: {total_tokens}");
    }

    eprintln!("\n=== REAP PROBE COMPLETE ===");
}
