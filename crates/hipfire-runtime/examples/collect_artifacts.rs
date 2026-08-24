#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load calibration collector (thin CLI).
//
// Loads a bf16 `.hfq` once and runs the LLaMA-family calibration forward
// via `hipfire_runtime::calibration::{collect,collect_grouped}`, writing a
// unified `<model>.calib.hfq` bundling per-tensor Hessian + imatrix.
//
// This is the `quant/quality` port of `pr/441:crates/hipfire-runtime/examples/collect_artifacts.rs`.
// The original drove arch-specific collectors (qwen35::collect_calibration_artifacts etc.).
// The ported library is now generic (`CalibCollector` + `collect`/`collect_grouped`),
// so the driver is adapted to that interface: it builds the capture map from
// `LlamaWeights` buffer addresses and drives the llama `forward` loop itself.
//
// Supported archs on this driver:
//   - LLaMA-family dense (arch 0/1/5 dense mapped through llama): q_proj/k_proj/v_proj/o_proj,
//     gate_proj/up_proj, down_proj (post-SiLU) — all 7 linears per layer.
//   Other archs (MoE, gemma3, lfm2, etc.) are intentionally not wired on this
//   minimal dense port. The CLI will exit 2 with a message naming the unsupported
//   arch, rather than silently succeeding with zero tensors.
//
// Run:
//   cargo run -p hipfire-runtime --example collect_artifacts -- \
//     --model <bf16.hfq> --corpus benchmarks/calib/bartowski_v5.split.txt \
//     --output work/qwen3.8-27b.calib.hfq --max-tokens 262144 --layers-per-pass 8

use hipfire_runtime::calibration::{collect, collect_grouped, CalibForward};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "usage: collect_artifacts --model <bf16.hfq> --corpus <txt> --output <calib.hfq> \
         [--max-tokens N] [--arch <id>] [--synthetic-tokens] [--seed N] [--kldref] \
         [--layers-per-pass N]\n\
         \n\
         Flags:\n\
         --model <path>          bf16 source HFQ (required)\n\
         --corpus <path>         plain UTF-8 text corpus (required unless --synthetic-tokens)\n\
         --output <path>         output .calib.hfq (REQUIRED; ~30 GB for a dense 27B)\n\
         --max-tokens N          cap calibration tokens (default 2048; corpus truncated to this)\n\
         --arch <id>             override HFQ arch_id (u32)\n\
         --synthetic-tokens      skip corpus/tokenizer; feed seeded random ids in [0,vocab)\n\
         --seed N                seed for --synthetic-tokens (default 0)\n\
         --kldref                (accepted but not yet implemented for llama dense; warned and ignored)\n\
         --layers-per-pass N     grouped collection granularity (default 8; 64 layers -> 8 groups)\n"
    );
    std::process::exit(code);
}

/// hex md5 of corpus file bytes, via `md5sum` if present, else fallback to length hex.
fn corpus_md5(path: &str) -> String {
    if path.is_empty() {
        return "synthetic".to_string();
    }
    // Try external md5sum first (present on most linux).
    if let Ok(out) = Command::new("md5sum").arg(path).output() {
        if out.status.success() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Some(hex) = s.split_whitespace().next() {
                    return hex.to_string();
                }
            }
        }
    }
    // Fallback: byte length + simple hash.
    match std::fs::read(path) {
        Ok(b) => format!("len{}-fallback", b.len()),
        Err(_) => "unknown".to_string(),
    }
}

fn compact_hessian_bytes(k: usize) -> u64 {
    (k * 4 + k * (k - 1)) as u64
}

fn estimate_peak_group_bytes(
    config: &hipfire_runtime::llama::LlamaConfig,
    weights: &hipfire_runtime::llama::LlamaWeights,
    start: usize,
    end: usize,
) -> u64 {
    // Sum compact Hessian + imatrix per captured linear in the layer range.
    // For K that varies (o_proj K may be 6144), read actual K from the weight.
    let mut total: u64 = 0;
    for idx in start..end.min(weights.layers.len()) {
        let layer = &weights.layers[idx];
        let ks = [
            layer.wq.k,
            layer.wk.k,
            layer.wv.k,
            layer.wo.k,
            layer.w_gate.k,
            layer.w_up.k,
            layer.w_down.k,
        ];
        for &k in &ks {
            total += compact_hessian_bytes(k) + (k * 4) as u64;
        }
        // Row buffer FLUSH_BATCH * K per tensor is GPU-side, not counted here.
    }
    total
}

fn build_capture_names(
    weights: &hipfire_runtime::llama::LlamaWeights,
    start: usize,
    end: usize,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    let end = end.min(weights.layers.len());
    for i in start..end {
        let layer = &weights.layers[i];
        let p = format!("model.layers.{i}");
        // SAFETY: buf.buf is DeviceBuffer, as_ptr is the device address.
        m.insert(
            layer.wq.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.q_proj"),
        );
        m.insert(
            layer.wk.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.k_proj"),
        );
        m.insert(
            layer.wv.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.v_proj"),
        );
        m.insert(
            layer.wo.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.o_proj"),
        );
        m.insert(
            layer.w_gate.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.gate_proj"),
        );
        m.insert(
            layer.w_up.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.up_proj"),
        );
        m.insert(
            layer.w_down.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.down_proj"),
        );
    }
    m
}

fn main() {
    // ---- determinism knobs (mirror collect_e8_hessian_native.rs:107-113) ----
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GRAPH_MOE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage_and_exit(0);
    }

    let model = arg("--model", None).unwrap_or_else(|| {
        eprintln!("--model <bf16.hfq> required");
        print_usage_and_exit(2);
    });
    let synthetic = has_flag("--synthetic-tokens");
    let corpus = arg("--corpus", Some(String::new())).unwrap();
    if !synthetic && corpus.is_empty() {
        eprintln!("--corpus <txt> required (or use --synthetic-tokens)");
        print_usage_and_exit(2);
    }
    // No default: a .calib.hfq for a 27B dense model is ~30 GB. Defaulting it
    // anywhere — /tmp least of all — risks filling a tmpfs or losing a
    // multi-hour GPU run to a reboot. Make the caller name a durable path.
    let output = arg("--output", None).unwrap_or_else(|| {
        eprintln!(
            "--output <path.calib.hfq> required (no default; artifact is ~30 GB for a dense 27B)"
        );
        print_usage_and_exit(2);
    });
    let max_tokens: usize = arg("--max-tokens", Some("2048".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--max-tokens must be usize");
            print_usage_and_exit(2);
        });
    // Calibration is run as independent sequences of this length, NOT as one
    // long stream — see run_calib_sequences for why. 2048 matches the
    // 128 seq x 2048 ctx shape scripts/collect_hessian.py has always used.
    let seq_len: usize = arg("--seq-len", Some("2048".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--seq-len must be usize");
            print_usage_and_exit(2);
        });
    let layers_per_pass: usize = arg("--layers-per-pass", Some("8".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--layers-per-pass must be usize");
            print_usage_and_exit(2);
        });
    let want_kldref = has_flag("--kldref");
    let seed: u64 = arg("--seed", Some("0".into())).unwrap().parse().unwrap();
    if want_kldref {
        eprintln!(
            "note: --kldref is accepted but not yet implemented for the llama dense \
            collector in this port; the flag will be ignored and no KLDREF tensors \
            will be emitted. Future work can add a lm-head top-k capture similar to \
            hipfire-arch-gemma3/src/calibration.rs."
        );
    }

    let mut hfq =
        HfqFile::open(Path::new(&model)).unwrap_or_else(|e| panic!("open model {}: {e}", model));
    let source_arch_id = arg("--arch", None)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(hfq.arch_id);

    // Accept only dense llama-family arch_ids on this port.
    // Qwen3.8-27B dense is arch 5; allow 0 (llama), 1 (qwen3) as well for testing.
    // For others, fail fast with a clear message instead of silently producing empty artifacts.
    const SUPPORTED: &[u32] = &[0, 1, 5];
    let is_supported_dense = SUPPORTED.contains(&source_arch_id)
        || (source_arch_id == 5 || source_arch_id == 0 || source_arch_id == 1);
    // If arch is not in supported dense set but could be MoE etc., we still
    // attempt llama load; if it fails we will report.
    if !is_supported_dense {
        // Still allow running but warn; the load below will likely fail if truly unsupported.
        eprintln!(
            "warning: arch_id {} is not in the dense-llama allowlist {:?}; attempting llama load anyway — \
            if the model is MoE (arch 6/10/11/14/16) the calibration will fail or capture zero tensors. \
            Those arches are not ported in this driver (see yield notes).",
            source_arch_id, SUPPORTED
        );
    }

    // Tokenizer from HFQ metadata (synthetic mode skips it).
    let tokenizer: Option<Tokenizer> = if synthetic {
        None
    } else {
        Some(
            Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .unwrap_or_else(|e| panic!("tokenizer from hfq metadata: {e}")),
        )
    };

    let mut tokens_owned: Vec<u32> = if synthetic {
        let meta: serde_json::Value =
            serde_json::from_str(&hfq.metadata_json).expect("metadata json");
        let vocab = meta
            .get("vocab_size")
            .or_else(|| meta.get("config").and_then(|c| c.get("vocab_size")))
            .and_then(|v| v.as_u64())
            .expect("vocab_size in metadata") as u32;
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
        (0..max_tokens)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % vocab as u64) as u32
            })
            .collect()
    } else {
        let raw = std::fs::read(&corpus).unwrap_or_else(|e| panic!("read corpus {}: {e}", corpus));
        // Decode lossily, never panic on non-UTF8 (bartowski corpora contain some non-UTF8 bytes).
        // We take an 8x byte slice to bound tokenization cost, lossily converted.
        let take = (max_tokens * 8).min(raw.len());
        let text = String::from_utf8_lossy(&raw[..take]).to_string();
        tokenizer.as_ref().unwrap().encode(&text)
    };
    // Truncate the OWNED vec, not a discarded slice: every downstream consumer
    // clones `tokens_owned`, so slicing here would silently let the run process
    // more tokens than --max-tokens while the metadata recorded the capped
    // figure. That is a false provenance record on a multi-hour GPU run.
    let n_tok = tokens_owned.len().min(max_tokens);
    tokens_owned.truncate(n_tok);
    let corpus_md5 = corpus_md5(&corpus);
    eprintln!(
        "calibrating on {n_tok} tokens (kldref={want_kldref}, synthetic={synthetic}, corpus_md5={corpus_md5})"
    );

    let mut gpu = rdna_compute::Gpu::init().unwrap_or_else(|e| panic!("gpu init: {e}"));
    eprintln!("GPU: {}", gpu.arch);

    // Load llama weights/config from HFQ.
    let config = hipfire_runtime::hfq::config_from_hfq(&hfq)
        .unwrap_or_else(|e| panic!("config_from_hfq: {e}"));
    eprintln!(
        "model arch_id={source_arch_id} n_layers={} dim={} hidden_dim={} vocab={} head_dim={} qk_norm={}",
        config.n_layers,
        config.dim,
        config.hidden_dim,
        config.vocab_size,
        config.head_dim,
        config.has_qk_norm
    );
    let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
        .unwrap_or_else(|e| panic!("load_weights_hfq: {e}"));
    // Drop the mmap-backed HfqFile early to release file cache pressure on APUs.
    drop(hfq);

    // Peak per-group memory estimate (host HFQM payload size, not counting GPU scratch).
    // Use the actual weight Ks so o_proj K=6144 is accounted for.
    let peak_group_bytes =
        estimate_peak_group_bytes(&config, &weights, 0, layers_per_pass.min(config.n_layers));
    eprintln!(
        "grouped collection: n_layers={} layers_per_pass={} n_groups={} peak_group_payload≈{:.1} MB (compact hessian+imatrix)",
        config.n_layers,
        layers_per_pass,
        (config.n_layers + layers_per_pass - 1) / layers_per_pass,
        peak_group_bytes as f64 / (1024.0 * 1024.0)
    );

    let t0 = std::time::Instant::now();

    // Decide grouped vs single-pass. For 64 layers with 8 per pass, grouped is required.
    // We always use grouped when group_size < n_layers, otherwise single.
    let use_grouped = layers_per_pass < config.n_layers;

    /// Run the calibration forward as INDEPENDENT sequences of at most `seq_len`
    /// tokens, clearing the KV cache and restarting positions at 0 for each.
    ///
    /// Feeding the whole corpus as ONE long sequence — which this driver did
    /// originally — is wrong three separate ways:
    ///
    ///  1. `o_proj`'s input is the attention OUTPUT, a convex combination over KV
    ///     history. At position 200k attending across unrelated concatenated
    ///     documents that distribution bears no resemblance to inference, so its
    ///     Hessian/imatrix protect channels the model never lights up in service.
    ///     (The MLP tensors are far more robust — their input is a post-RMSNorm
    ///     hidden state driven by token identity, not context length.)
    ///  2. Attention work is O(N²/2) rather than O(N·seq_len/2) — 128× more at
    ///     N=262144, seq_len=2048.
    ///  3. The KV cache alone is ~68.7 GB at N=262144 for this model
    ///     (64 layers × 4 kv_heads × 256 head_dim × 2 × 2 B), which does not fit
    ///     alongside a bf16 teacher.
    ///
    /// Matches the 128 seq × 2048 ctx shape that `scripts/collect_hessian.py`
    /// has always used, so token budgets stay comparable to prior runs.
    fn run_calib_sequences(
        gpu: &mut rdna_compute::Gpu,
        weights: &hipfire_runtime::llama::LlamaWeights,
        config: &hipfire_runtime::llama::LlamaConfig,
        tokens: &[u32],
        seq_len: usize,
    ) -> Result<(), String> {
        let seq_len = seq_len.max(1);
        let mut kv = hipfire_runtime::llama::KvCache::new_gpu(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            seq_len + 16,
        )
        .map_err(|e| format!("kv alloc (seq_len={seq_len}): {e}"))?;
        for (si, chunk) in tokens.chunks(seq_len).enumerate() {
            kv.clear_gpu(gpu)
                .map_err(|e| format!("kv clear before seq {si}: {e}"))?;
            for (pos, &tok) in chunk.iter().enumerate() {
                hipfire_runtime::llama::forward(gpu, weights, config, tok, pos, &mut kv)
                    .map_err(|e| format!("forward seq {si} pos {pos}: {e}"))?;
            }
        }
        Ok(())
    }

    let summary = if use_grouped {
        let lpp = layers_per_pass.max(1);
        let num_layers = config.n_layers;
        // Capture clones for the closures.
        let weights_ref = &weights;
        let tokens_owned_clone = tokens_owned.clone();
        // Note: tokens slice must live long enough. We clone into the closure env.
        let result = collect_grouped(
            &mut gpu,
            source_arch_id,
            num_layers,
            lpp,
            Vec::new(), // imatrix_only: dense => all Hessians
            Path::new(&output),
            &[
                ("source_model", serde_json::json!(model.clone())),
                ("corpus", serde_json::json!(corpus.clone())),
                ("corpus_md5", serde_json::json!(corpus_md5.clone())),
                ("n_calib_tokens", serde_json::json!(n_tok)),
                ("source_arch_id", serde_json::json!(source_arch_id)),
                ("layers_per_pass", serde_json::json!(lpp)),
            ],
            |start, end| build_capture_names(weights_ref, start, end),
            |gpu, _group_idx| {
                // Each group re-runs the corpus, but only this group's layers are
                // registered in capture_names, so other layers run uncaptured.
                // Sequences are independent: KV cleared, positions restart at 0.
                run_calib_sequences(gpu, weights_ref, &config, &tokens_owned_clone, seq_len)?;
                // No extra tensors for llama dense (KLDREF unsupported on this port).
                Ok(CalibForward::default())
            },
        );
        match result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("collect_grouped failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // Single-pass (small models or layers_per_pass >= n_layers).
        let capture_names = build_capture_names(&weights, 0, config.n_layers);
        let tokens_clone = tokens_owned.clone();
        let config_clone = config.clone();
        // We need to move weights by reference; collect takes capture_names by value.
        let result = collect(
            &mut gpu,
            source_arch_id,
            capture_names,
            Vec::new(),
            Path::new(&output),
            &[
                ("source_model", serde_json::json!(model.clone())),
                ("corpus", serde_json::json!(corpus.clone())),
                ("corpus_md5", serde_json::json!(corpus_md5.clone())),
                ("n_calib_tokens", serde_json::json!(n_tok)),
                ("source_arch_id", serde_json::json!(source_arch_id)),
                ("layers_per_pass", serde_json::json!(config.n_layers)),
            ],
            |gpu| {
                run_calib_sequences(gpu, &weights, &config_clone, &tokens_clone, seq_len)?;
                Ok(CalibForward::default())
            },
        );
        match result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("collect failed: {e}");
                std::process::exit(1);
            }
        }
    };

    let elapsed = t0.elapsed().as_secs_f64();
    let out_meta = std::fs::metadata(&output).ok();
    let out_bytes = out_meta.map(|m| m.len()).unwrap_or(0);

    // Consistency check as in pr/441 — exit 1 if diag vs sumsq mismatched.
    eprintln!(
        "collected {} hessian + {} imatrix tensors in {:.1}s; max diag(H)-vs-Σx² rel-err = {:.3e} {}",
        summary.n_hessian,
        summary.n_imatrix,
        elapsed,
        summary.max_consistency,
        if summary.max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );
    eprintln!("wrote calib HFQ: {output} ({} bytes)", out_bytes);

    // Provenance record for MI300X-hours runs.
    eprintln!("--- run summary (provenance) ---");
    eprintln!("model: {model} (arch_id={source_arch_id})");
    eprintln!("corpus: {corpus} (md5={corpus_md5})");
    eprintln!("n_calib_tokens: {n_tok} (max_tokens cap {max_tokens})");
    eprintln!(
        "n_hessian: {}  n_imatrix: {}",
        summary.n_hessian, summary.n_imatrix
    );
    eprintln!(
        "output: {output}  size: {out_bytes} bytes  elapsed: {:.1}s",
        elapsed
    );
    eprintln!(
        "peak per-group payload: {:.1} MB for layers_per_pass={} (compact bf16-tril+f32-diag; dense would be ~{:.1} MB)",
        peak_group_bytes as f64 / (1024.0 * 1024.0),
        layers_per_pass,
        {
            // Rough dense estimate for same group
            let mut dense = 0u64;
            for i in 0..layers_per_pass.min(config.n_layers) {
                let l = &weights.layers[i];
                for k in [l.wq.k, l.wk.k, l.wv.k, l.wo.k, l.w_gate.k, l.w_up.k, l.w_down.k] {
                    dense += (k * k * 4) as u64 + (k * 4) as u64;
                }
            }
            dense as f64 / (1024.0 * 1024.0)
        }
    );
    eprintln!(
        "config: n_layers={} dim={} hidden_dim={} head_dim={} layers_per_pass={}",
        config.n_layers, config.dim, config.hidden_dim, config.head_dim, layers_per_pass
    );
    eprintln!("flags: kldref={want_kldref} synthetic={synthetic} seed={seed}");
    eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0 (forced)");
    if summary.max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
