//! Offline activation calibration for the gfx12 iu4 K=32 GEMM path.
//!
//! Runs FP16 forward over a small calibration corpus, snoops per-channel
//! activation distributions at every `gemm_hfq4g256_residual` call site,
//! and writes a `<model>.iu4cal` sidecar that the engine auto-loads at
//! model load time when `HIPFIRE_GFX12_IU4_CALIBRATED=1`.
//!
//! Sidecar contents (per call site, in dispatch-counter order):
//!   - `mu_a[k]`    per-input-channel activation mean
//!   - `s_a[k]`     per-input-channel activation scale (default 99-pctile-abs)
//!   - `w_mu_bias[m]` precomputed `W·mu_a` so the runtime can add it to the
//!                    GEMM output and recover `y = W·x` exactly (modulo
//!                    Q4_1 quant of the centered activation).
//!
//! Math:
//!   y = W·x = (W·diag(s)) · ((x - mu)/s) + W·mu
//!
//! At inference time:
//!   1. preshift kernel:  x_centered[t][c] = (x[t][c] - mu[c]) * inv_s[c]
//!   2. existing Q4_1 quant on x_centered
//!   3. existing iu4 GEMM with weight whose per-K=256-group scale was
//!      multiplied by `s_group[g]` at sidecar-load time
//!   4. broadcast-add `w_mu_bias[m]` to the GEMM output column
//!
//! The bias is computed CPU-side here by dequantizing the HFQ4 weight
//! row-by-row and dotting with the calibration `mu_a`. ~512 MB of CPU work
//! for a 27B model — slow but offline-only. Output sidecar is ~32 MB FP16.
//!
//! Usage:
//!   HIPFIRE_MODELS_DIR=$HOME/.hipfire/models \
//!     cargo run --release --features deltanet -p engine \
//!     --example calibrate_iu4_activations -- \
//!     --model $HOME/.hipfire/models/qwen3.5-9b.mq4
//!
//! Optional flags:
//!   --corpus <dir>         additional prompts; defaults to the embedded
//!                          coherence-style corpus
//!   --strategy {p99,mean}  s_a estimation strategy (default p99)
//!   --max-tokens <n>       cap per-prompt token budget (default 256)
//!   --out <path>           sidecar output; defaults to <model>.iu4cal
//!
//! Local note: this binary uses FP16 forward only — runs anywhere hipfire
//! runs (gfx1100, gfx1201, etc.). Test gfx1201 calibrated dispatch on
//! hiptrx; capture a calibration here and ship it with the model.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("calibrate_iu4_activations requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    real_main();
}

#[cfg(feature = "deltanet")]
fn real_main() {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use engine::hfq::HfqFile;
    use engine::quant::iu4_calibration::{
        CenteredHistogram, Iu4CalSite, Iu4Calibration, MeanAccumulator, f32_vec_to_f16,
    };
    use engine::qwen35::{
        self, DeltaNetState, LayerWeights, Qwen35Config, Qwen35Scratch, Qwen35Weights,
    };
    use engine::llama::{self, KvCache, f16_to_f32};
    use rdna_compute::Gpu;

    // ── argv parse ──────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut corpus_dir: Option<String> = None;
    let mut strategy: String = "p99".to_string();
    let mut max_tokens: usize = 256;
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--corpus" => {
                corpus_dir = args.get(i + 1).cloned();
                i += 2;
            }
            "--strategy" => {
                strategy = args.get(i + 1).cloned().unwrap_or_else(|| "p99".into());
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(256);
                i += 2;
            }
            "--out" => {
                out_path = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: calibrate_iu4_activations --model <path> [--corpus <dir>] \
                     [--strategy p99|mean] [--max-tokens 256] [--out <path>]"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
                std::process::exit(2);
            }
        }
    }
    let Some(model_path) = model_path else {
        eprintln!("--model is required (path to .mq4 / .hfq HFQ4-G256 model)");
        std::process::exit(2);
    };
    let out_path = out_path.unwrap_or_else(|| format!("{model_path}.iu4cal"));

    eprintln!("[calibrate-iu4] model:    {model_path}");
    eprintln!("[calibrate-iu4] strategy: {strategy}");
    eprintln!("[calibrate-iu4] out:      {out_path}");

    // ── corpus ──────────────────────────────────────────────────────────
    let mut prompts: Vec<String> = embedded_corpus();
    if let Some(dir) = corpus_dir.as_ref() {
        match std::fs::read_dir(dir) {
            Ok(entries) => {
                for ent in entries.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                        if let Ok(s) = std::fs::read_to_string(&p) {
                            prompts.push(s);
                        }
                    }
                }
            }
            Err(e) => eprintln!("[calibrate-iu4] failed to read corpus dir {dir}: {e}"),
        }
    }
    eprintln!("[calibrate-iu4] {} corpus prompts", prompts.len());

    // ── load model ──────────────────────────────────────────────────────
    let hfq = HfqFile::open(std::path::Path::new(&model_path))
        .expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read qwen35 config");
    eprintln!("[calibrate-iu4] config: dim={} layers={} arch_id={} n_heads={} n_kv_heads={} hidden_dim={}",
        config.dim, config.n_layers, hfq.arch_id, config.n_heads, config.n_kv_heads, config.hidden_dim);
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("[calibrate-iu4] gpu: {}", gpu.arch);
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("[calibrate-iu4] loaded {} layers", weights.layers.len());

    // ── tokenizer ───────────────────────────────────────────────────────
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    // ── KV / state / scratch ────────────────────────────────────────────
    // Use the BATCHED prefill scratch so each prompt fires through
    // gemm_hfq4g256_residual (which is what the calibration capture hook
    // is attached to). Per-token forward_scratch goes through
    // gemv_hfq4g256_residual instead and would not capture activation
    // statistics.
    let kv_seq = max_tokens.max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv cache alloc");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state alloc");
    // Force PrefillBatchScratch allocation by pretending HIPFIRE_PREFILL_REUSE_PBS
    // is set for the scratch builder. We use a per-prompt batch cap of
    // max_tokens — small enough not to OOM, large enough to do each prompt
    // in one prefill chunk.
    std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
    let pbs_max = max_tokens.max(64).min(qwen35::PREFILL_MAX_BATCH);
    std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", pbs_max.to_string());
    let scratch = Qwen35Scratch::new(&mut gpu, &config, pbs_max)
        .expect("scratch alloc");

    // ── shared capture state ────────────────────────────────────────────
    // Per-site MeanAccumulator and CenteredHistogram. Sites are discovered
    // dynamically in the first forward pass (we don't know the dispatch
    // call order a priori, just the count).
    //
    // The capture protocol is two-pass:
    //   Pass A: accumulate `sum` and `count` per site → mu = sum/count
    //   Pass B: with mu fixed, accumulate per-channel histogram of |x-mu|
    //           → s_a from p99-abs (or mean-abs)
    //
    // We record per-site (n_channels, n_output_rows) at first-touch in
    // pass A so pass B can construct CenteredHistograms with the right
    // shape.
    #[derive(Default, Clone)]
    struct SiteShape {
        n_channels: usize,
        n_output_rows: usize,
        layer_idx: u32,
        proj_id: u32,
    }
    let shapes: Arc<Mutex<HashMap<usize, SiteShape>>> = Arc::new(Mutex::new(HashMap::new()));
    let mean_accs: Arc<Mutex<HashMap<usize, MeanAccumulator>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // ── Pass A: accumulate per-channel sum/count ────────────────────────
    eprintln!("[calibrate-iu4] === pass A: per-channel mean ===");
    {
        let shapes_a = shapes.clone();
        let mean_accs_a = mean_accs.clone();
        let hook = Box::new(move |site_id: usize, x: &[f32], n: usize, k: usize, m: usize| {
            let mut sh = shapes_a.lock().unwrap();
            let entry = sh.entry(site_id).or_insert_with(|| SiteShape {
                n_channels: k,
                n_output_rows: m,
                // Layer + proj inferred from site_id assuming Qwen3.5
                // dense (wo and w_down are exactly 2 sites per layer);
                // sites alternate (0=wo of layer 0, 1=w_down of layer 0,
                // 2=wo of layer 1, ...). Storing as informational metadata.
                layer_idx: (site_id / 2) as u32,
                proj_id: (site_id % 2) as u32,
            });
            entry.n_channels = k;
            entry.n_output_rows = m;
            drop(sh);
            let mut accs = mean_accs_a.lock().unwrap();
            let acc = accs
                .entry(site_id)
                .or_insert_with(|| MeanAccumulator::new(k));
            acc.add_batch(x, n);
        });
        gpu.set_iu4_capture_hook(Some(hook));
    }
    let mut total_tokens_pass_a = 0usize;
    for (idx, prompt) in prompts.iter().enumerate() {
        let toks = tokenize_for_chat(&tokenizer, prompt, max_tokens);
        if toks.is_empty() {
            continue;
        }
        eprintln!(
            "[calibrate-iu4] pass A prompt {}/{} ({} tokens)",
            idx + 1,
            prompts.len(),
            toks.len()
        );
        gpu.reset_iu4_dispatch_counter();
        // Process each prompt as a fresh sequence (pos starts at 0). The
        // KV slots at pos 0..n get overwritten before attention reads
        // them, so stale K/V from prior prompts doesn't leak into the
        // captured activations.
        let _ = run_prefill(&mut gpu, &weights, &config, &toks, &mut kv_cache, &mut dn_state, &scratch);
        total_tokens_pass_a += toks.len();
    }
    gpu.set_iu4_capture_hook(None);
    let n_sites = shapes.lock().unwrap().len();
    eprintln!(
        "[calibrate-iu4] pass A done: {} tokens across {} sites",
        total_tokens_pass_a, n_sites
    );

    if n_sites == 0 {
        eprintln!(
            "[calibrate-iu4] FAIL: no GEMM call sites observed. \
             Empty prompts or model has no HFQ4 wo/w_down? Aborting."
        );
        std::process::exit(1);
    }

    // Compute per-site mu vectors.
    let shapes_snap: HashMap<usize, SiteShape> = shapes.lock().unwrap().clone();
    let mut mus: HashMap<usize, Vec<f32>> = HashMap::new();
    {
        let accs = mean_accs.lock().unwrap();
        for (&sid, acc) in accs.iter() {
            mus.insert(sid, acc.mean());
        }
    }

    // ── Pass B: accumulate per-channel histogram against mu ─────────────
    eprintln!("[calibrate-iu4] === pass B: per-channel histogram of |x - mu| ===");
    let hists: Arc<Mutex<HashMap<usize, CenteredHistogram>>> =
        Arc::new(Mutex::new(HashMap::new()));
    {
        let hists_b = hists.clone();
        let mus_b: HashMap<usize, Vec<f32>> = mus.clone();
        let hook = Box::new(move |site_id: usize, x: &[f32], n: usize, _k: usize, _m: usize| {
            let mut hs = hists_b.lock().unwrap();
            let h = hs.entry(site_id).or_insert_with(|| {
                CenteredHistogram::new(
                    mus_b.get(&site_id).cloned().unwrap_or_default(),
                )
            });
            h.add_batch(x, n);
        });
        gpu.set_iu4_capture_hook(Some(hook));
    }
    let mut total_tokens_pass_b = 0usize;
    for (idx, prompt) in prompts.iter().enumerate() {
        let toks = tokenize_for_chat(&tokenizer, prompt, max_tokens);
        if toks.is_empty() {
            continue;
        }
        eprintln!(
            "[calibrate-iu4] pass B prompt {}/{} ({} tokens)",
            idx + 1,
            prompts.len(),
            toks.len()
        );
        gpu.reset_iu4_dispatch_counter();
        let _ = run_prefill(&mut gpu, &weights, &config, &toks, &mut kv_cache, &mut dn_state, &scratch);
        total_tokens_pass_b += toks.len();
    }
    gpu.set_iu4_capture_hook(None);
    eprintln!("[calibrate-iu4] pass B done: {total_tokens_pass_b} tokens");

    // ── Compute s_a per strategy ────────────────────────────────────────
    let mut sas: HashMap<usize, Vec<f32>> = HashMap::new();
    {
        let hs = hists.lock().unwrap();
        for (&sid, h) in hs.iter() {
            let s = match strategy.as_str() {
                "mean" => h.mean_abs(),
                _ => h.percentile_abs(99.0),
            };
            // Floor to a small positive value to avoid divide-by-zero in
            // dead channels (e.g. embedding-zero-like distributions).
            let s_floored: Vec<f32> = s
                .iter()
                .map(|&v| if v.is_finite() && v > 1e-4 { v } else { 1e-4 })
                .collect();
            sas.insert(sid, s_floored);
        }
    }

    // ── Compute W·mu_a bias per site (CPU dequant + dot) ────────────────
    eprintln!("[calibrate-iu4] === computing W·mu_a bias (CPU dequant) ===");
    let biases = compute_w_mu_biases(&weights, &config, &shapes_snap, &mus);

    // ── Build sidecar ───────────────────────────────────────────────────
    let mut sidecar = Iu4Calibration::new();
    let mut sids: Vec<usize> = shapes_snap.keys().copied().collect();
    sids.sort();
    for sid in sids {
        let sh = shapes_snap.get(&sid).unwrap();
        let mu = mus.get(&sid).cloned().unwrap_or_else(|| vec![0.0; sh.n_channels]);
        let s = sas.get(&sid).cloned().unwrap_or_else(|| vec![1.0; sh.n_channels]);
        let bias = biases
            .get(&sid)
            .cloned()
            .unwrap_or_else(|| vec![0.0; sh.n_output_rows]);

        let mut site = Iu4CalSite::new(
            sh.layer_idx,
            sh.proj_id,
            sh.n_channels,
            sh.n_output_rows,
        );
        site.mu_a = f32_vec_to_f16(&mu);
        site.s_a = f32_vec_to_f16(&s);
        site.w_mu_bias = f32_vec_to_f16(&bias);
        sidecar.sites.push(site);
    }

    // ── Print summary ───────────────────────────────────────────────────
    let n = sidecar.n_sites();
    eprintln!("[calibrate-iu4] === sidecar summary ===");
    eprintln!("[calibrate-iu4]  sites: {n}");
    if let Some(s0) = sidecar.sites.first() {
        let mu0 = s0.mu_a.iter().take(8).map(|&b| f16_to_f32(b)).collect::<Vec<_>>();
        let s0v = s0.s_a.iter().take(8).map(|&b| f16_to_f32(b)).collect::<Vec<_>>();
        eprintln!("[calibrate-iu4]  site[0] (layer={} proj={}): k={} m={}",
            s0.layer_idx, s0.proj_id, s0.n_channels, s0.n_output_rows);
        eprintln!("[calibrate-iu4]    mu_a[0..8] = {:?}", mu0);
        eprintln!("[calibrate-iu4]    s_a[0..8]  = {:?}", s0v);
    }

    // ── Write sidecar ───────────────────────────────────────────────────
    sidecar
        .write_path(std::path::Path::new(&out_path))
        .expect("write sidecar");
    let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[calibrate-iu4] wrote {} sites, {} bytes ({:.1} MB) → {}",
        n,
        bytes,
        bytes as f64 / 1_000_000.0,
        out_path
    );

    // ── Forward-pass helper ─────────────────────────────────────────────

    fn run_prefill(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        tokens: &[u32],
        kv_cache: &mut KvCache,
        dn_state: &mut DeltaNetState,
        scratch: &Qwen35Scratch,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Run the BATCHED prefill so the dispatcher takes the
        // gemm_hfq4g256_residual path (where the calibration capture hook
        // fires). Per-token forward_scratch would route through gemv
        // helpers and miss the hook entirely.
        qwen35::forward_prefill_batch(
            gpu, weights, config, tokens, /*start_pos=*/0,
            kv_cache, dn_state, scratch,
            /*hidden_rb=*/None,
            /*per_token_hidden_out=*/None,
            /*gdn_tape=*/None,
            /*tree_verify=*/None,
        )?;
        Ok(())
    }

    fn tokenize_for_chat(
        tok: &engine::tokenizer::Tokenizer,
        text: &str,
        max_tokens: usize,
    ) -> Vec<u32> {
        // Wrap in the Qwen3.5 chat shell so we exercise activation
        // patterns the model actually sees in production. Truncate to
        // max_tokens to keep calibration cheap.
        let im_start = tok.encode("<|im_start|>");
        let im_end = tok.encode("<|im_end|>");
        let user = tok.encode("user");
        let asst = tok.encode("assistant");
        let nl = tok.encode("\n");
        let body = tok.encode(text);
        let mut out = Vec::new();
        out.extend_from_slice(&im_start);
        out.extend_from_slice(&user);
        out.extend_from_slice(&nl);
        out.extend_from_slice(&body);
        out.extend_from_slice(&im_end);
        out.extend_from_slice(&nl);
        out.extend_from_slice(&im_start);
        out.extend_from_slice(&asst);
        out.extend_from_slice(&nl);
        if out.len() > max_tokens {
            out.truncate(max_tokens);
        }
        out
    }

    fn compute_w_mu_biases(
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        shapes: &HashMap<usize, SiteShape>,
        mus: &HashMap<usize, Vec<f32>>,
    ) -> HashMap<usize, Vec<f32>> {
        let mut out: HashMap<usize, Vec<f32>> = HashMap::new();
        for (&sid, sh) in shapes.iter() {
            let mu = match mus.get(&sid) {
                Some(v) => v,
                None => continue,
            };
            // Map (layer_idx, proj_id) → WeightTensor handle.
            let layer_idx = sh.layer_idx as usize;
            let proj_id = sh.proj_id;
            if layer_idx >= weights.layers.len() {
                continue;
            }
            let weight: &engine::llama::WeightTensor = match (&weights.layers[layer_idx], proj_id) {
                (LayerWeights::FullAttn(l), 0) => &l.wo,
                (LayerWeights::FullAttn(l), 1) => &l.w_down,
                (LayerWeights::DeltaNet(l), 0) => &l.wo,
                (LayerWeights::DeltaNet(l), 1) => &l.w_down,
                _ => continue,
            };
            // Only HFQ4-G256 supported. Skip other dtypes (the calibration
            // sidecar would be unused there anyway since iu4 dispatch is
            // gated on HFQ4 weights through gemm_hfq4g256_residual).
            if weight.gpu_dtype != rdna_compute::DType::HFQ4G256
                && weight.gpu_dtype != rdna_compute::DType::MQ4G256
            {
                eprintln!(
                    "[calibrate-iu4] site {} weight dtype {:?} not HFQ4G256/MQ4G256 — bias omitted",
                    sid, weight.gpu_dtype
                );
                continue;
            }
            let m = sh.n_output_rows;
            let k = sh.n_channels;
            let bias = compute_bias_via_cpu_dequant(weight, mu, m, k);
            out.insert(sid, bias);
            eprintln!(
                "[calibrate-iu4]  site {} (layer {} proj {}) bias[0..4] = {:?}",
                sid,
                sh.layer_idx,
                sh.proj_id,
                bias_first_4(&out[&sid])
            );
        }
        out
    }

    fn bias_first_4(b: &[f32]) -> Vec<f32> {
        b.iter().take(4).copied().collect()
    }

    /// CPU-only HFQ4-G256 dequant + dot with `mu`. Reads the weight bytes
    /// straight from the host (we re-mmap via hipMemcpy d→h here; for this
    /// offline binary we accept the cost).
    fn compute_bias_via_cpu_dequant(
        weight: &engine::llama::WeightTensor,
        mu: &[f32],
        m: usize,
        k: usize,
    ) -> Vec<f32> {
        let groups_per_row = k / 256;
        let row_bytes = groups_per_row * 136;
        let total_bytes = m * row_bytes;
        // Read bytes from GPU once (we don't have a Gpu handle in scope —
        // re-acquire the singleton). Build a tiny temporary Gpu just for
        // this download isn't possible because Gpu doesn't allow multiple
        // instances safely. Instead, use the global hip runtime indirectly
        // via the existing memcpy_dtoh on the buffer's raw ptr.
        //
        // Simpler: open a hip-bridge::HipRuntime here, since we already
        // have the buffer's raw pointer + size. The runtime is reentrant.
        let rt = hip_bridge::HipRuntime::load().expect("hip rt");
        rt.set_device(0).ok();
        let mut host = vec![0u8; total_bytes];
        let buf_view = unsafe {
            hip_bridge::DeviceBuffer::from_raw(weight.buf.buf.as_ptr(), total_bytes)
        };
        rt.memcpy_dtoh(&mut host, &buf_view).expect("dtoh weight");
        std::mem::forget(buf_view);

        // Dequant + dot per row:
        //   for each row r:
        //     for each group g:
        //       scale = host[r*row_bytes + g*136 + 0..4] (FP32)
        //       zero  = host[r*row_bytes + g*136 + 4..8] (FP32)
        //       for nibble i in 0..256:
        //         w = scale * nibble + zero
        //         bias[r] += w * mu[g*256 + i]
        let mut bias = vec![0.0f32; m];
        for r in 0..m {
            let row = &host[r * row_bytes..(r + 1) * row_bytes];
            let mut acc = 0.0f64;
            for g in 0..groups_per_row {
                let off = g * 136;
                let scale = f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
                let zero = f32::from_le_bytes([row[off + 4], row[off + 5], row[off + 6], row[off + 7]]);
                let nibbles = &row[off + 8..off + 136];
                let mu_chunk = &mu[g * 256..(g + 1) * 256];
                for i in 0..256 {
                    let byte_idx = i / 2;
                    let nibble = if i % 2 == 0 {
                        nibbles[byte_idx] & 0xF
                    } else {
                        nibbles[byte_idx] >> 4
                    };
                    let w = scale * (nibble as f32) + zero;
                    acc += (w as f64) * (mu_chunk[i] as f64);
                }
            }
            bias[r] = acc as f32;
        }
        bias
    }

    /// Embedded calibration corpus. Mix of code, prose, math, multi-turn,
    /// tool-call and reasoning shapes so the captured distributions cover
    /// what production Qwen3.5 dispatch sees. Adapted from
    /// scripts/coherence-gate.sh's prompt matrix and benchmarks/prompts/.
    fn embedded_corpus() -> Vec<String> {
        vec![
            // Reasoning / counting (coherence-gate sheep prompt + variants)
            "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.".into(),
            "If a train leaves Boston at 3pm going 60 mph and another leaves New York at 4pm going 80 mph toward Boston, when do they meet? Distance = 200 miles.".into(),
            // Capital / facts
            "What is the capital of France? Answer in one short sentence.".into(),
            "What is the chemical formula for water? Answer briefly.".into(),
            "Who wrote 'Hamlet'? Answer briefly.".into(),
            "List five planets in our solar system. One sentence.".into(),
            // Code (mirrors humaneval shapes)
            "Write a one-line Python function named square that returns n*n.".into(),
            "Write a Python function called fizzbuzz that prints numbers from 1 to 100, with multiples of three printing 'Fizz' and multiples of five 'Buzz'.".into(),
            "Implement a Python function `merge_sort(arr)` that sorts a list in ascending order using merge sort. Include type hints.".into(),
            "Write a Rust function `fn fibonacci(n: u32) -> u64` that returns the nth Fibonacci number using iterative computation.".into(),
            "Write a Python function `is_prime(n: int) -> bool` that returns True if n is prime, False otherwise. Handle n < 2 correctly.".into(),
            // Prose
            "Write a short paragraph about the importance of clean water in human civilization.".into(),
            "Describe the process by which photosynthesis converts sunlight into chemical energy in plants. Two paragraphs.".into(),
            "Explain in 3 sentences why the sky appears blue during the day but red at sunset.".into(),
            "Write a brief introduction to the concept of object-oriented programming for beginners.".into(),
            // Multi-turn-ish single shots (mirroring chat shape)
            "I'm planning a trip to Japan. Can you suggest three must-visit cities and why?".into(),
            "What are some common mistakes beginners make when learning to play guitar?".into(),
            "Explain the difference between mitosis and meiosis in cell division.".into(),
            "How do I debug a Python program that's running very slowly?".into(),
            // Math
            "Solve for x: 2x + 5 = 13. Show your work.".into(),
            "What is the area of a circle with radius 7? Use π ≈ 3.14159.".into(),
            "Compute the sum of integers from 1 to 100 using Gauss's formula. Show the formula.".into(),
            // Lists / structured
            "List the first ten prime numbers as a comma-separated list.".into(),
            "Provide a JSON object with keys 'name', 'age', 'occupation' for a fictional person named Alice.".into(),
            "Translate the phrase 'good morning' into Spanish, French, German, and Japanese.".into(),
            // Mid-context retrieval shape
            "Read the following short text then answer: The cat sat on the mat. The dog barked at the cat. The mat was red. — Question: What color was the mat?".into(),
            // Code with comments (mirrors PEP-8 prompts)
            "Write a Python `lru_cache` decorator implementation from scratch with a max_size parameter, supporting hits/misses tracking. Use only the standard library.".into(),
            "Write a function in C that reverses a null-terminated string in place. Annotate each step.".into(),
            // Trivia / knowledge
            "Name three notable women scientists from the 20th century and one of their contributions.".into(),
            "What is the boiling point of water at sea level in Celsius and Fahrenheit?".into(),
            "Briefly explain Newton's three laws of motion.".into(),
        ]
    }
}
