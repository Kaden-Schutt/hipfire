#[cfg(not(feature = "deltanet"))]
fn main() -> std::process::ExitCode {
    eprintln!("test_inferenceQA requires --features deltanet");
    std::process::ExitCode::from(10)
}

// QA mirror for the inference integration harness.
// Uses subprocess isolation per case so hangs, panics, and slowdowns are attributed
// to a single case instead of collapsing the whole sweep.

#[cfg(feature = "deltanet")]
use engine::gguf::GgufFile;
#[cfg(feature = "deltanet")]
use engine::hfq::HfqFile;
#[cfg(feature = "deltanet")]
use engine::llama;
#[cfg(feature = "deltanet")]
use engine::qwen35;
#[cfg(feature = "deltanet")]
use engine::qwen35::DeltaNetState;
#[cfg(feature = "deltanet")]
use engine::tokenizer::Tokenizer;
#[cfg(feature = "deltanet")]
use std::any::Any;
#[cfg(feature = "deltanet")]
use std::env;
#[cfg(feature = "deltanet")]
use std::path::{Path, PathBuf};
#[cfg(feature = "deltanet")]
use std::process::{Command, ExitCode};
#[cfg(feature = "deltanet")]
use std::thread;
#[cfg(feature = "deltanet")]
use std::time::{Duration, Instant};

#[cfg(feature = "deltanet")]
const SKIP_EXIT: u8 = 10;
#[cfg(feature = "deltanet")]
const QWEN_GGUF_FALLBACK: &str = "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf";
#[cfg(feature = "deltanet")]
const PREFILL_MEAN_LOGIT_TOL: f64 = 0.15;
#[cfg(feature = "deltanet")]
const PREFILL_ASYM2_MEAN_LOGIT_TOL: f64 = 0.25;
#[cfg(feature = "deltanet")]
const PREFILL_SELECTED_LOGIT_TOL: f32 = 1.0;

#[cfg(feature = "deltanet")]
struct CaseDef {
    name: &'static str,
    timeout: Duration,
}

#[cfg(feature = "deltanet")]
const CASES: &[CaseDef] = &[
    CaseDef { name: "forward_finite_logits", timeout: Duration::from_secs(20) },
    CaseDef { name: "forward_scratch_matches", timeout: Duration::from_secs(20) },
    CaseDef { name: "sequence_no_hang", timeout: Duration::from_secs(20) },
    CaseDef { name: "think_token_detectable", timeout: Duration::from_secs(10) },
    CaseDef { name: "chatml_single_tokens", timeout: Duration::from_secs(10) },
    CaseDef { name: "givens4_cache_allocates", timeout: Duration::from_secs(20) },
    CaseDef { name: "givens4_forward_no_hang", timeout: Duration::from_secs(20) },
    CaseDef { name: "prefill_batch_matches_sequential", timeout: Duration::from_secs(120) },
    CaseDef { name: "decode_speed_sanity", timeout: Duration::from_secs(35) },
    CaseDef { name: "vram_leak_signal", timeout: Duration::from_secs(20) },
];

#[cfg(feature = "deltanet")]
enum CaseOutcome {
    Pass(String),
    Skip(String),
    Fail(String),
}

#[cfg(feature = "deltanet")]
struct Context {
    _hfq: HfqFile,
    model_label: String,
    tokenizer_source: String,
    config: qwen35::Qwen35Config,
    tokenizer: Tokenizer,
    gpu: rdna_compute::Gpu,
    weights: qwen35::Qwen35Weights,
}

#[cfg(feature = "deltanet")]
fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut case_name: Option<String> = None;
    let mut model_path: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--qa-case" => {
                i += 1;
                case_name = args.get(i).cloned();
            }
            "--model" => {
                i += 1;
                model_path = args.get(i).map(PathBuf::from);
            }
            other if !other.starts_with("--") && model_path.is_none() => {
                model_path = Some(PathBuf::from(other));
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.or_else(resolve_model_from_env);

    if let Some(case) = case_name {
        return run_case(&case, model_path.as_deref());
    }

    supervisor(model_path.as_deref())
}

#[cfg(feature = "deltanet")]
fn resolve_model_from_env() -> Option<PathBuf> {
    env::var("QWEN35_TEST_MODEL").ok().map(PathBuf::from)
}

#[cfg(feature = "deltanet")]
fn supervisor(model_path: Option<&Path>) -> ExitCode {
    let model_path = match model_path {
        Some(path) => path,
        None => {
            eprintln!("QA SKIP: no model supplied. Pass --model <model.hfq> or set QWEN35_TEST_MODEL.");
            return ExitCode::from(SKIP_EXIT);
        }
    };

    if !model_path.exists() {
        eprintln!("QA SKIP: model not found at {}", model_path.display());
        return ExitCode::from(SKIP_EXIT);
    }

    let exe = match env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("failed to resolve current executable: {err}");
            return ExitCode::from(1);
        }
    };

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    eprintln!("=== hipfire inference QA harness ===");
    eprintln!("Model: {}", model_path.display());

    for case in CASES {
        eprintln!("\n--- {} ---", case.name);
        let mut child = match Command::new(&exe)
            .arg("--qa-case")
            .arg(case.name)
            .arg("--model")
            .arg(model_path)
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                failed += 1;
                eprintln!("spawn failed: {err}");
                continue;
            }
        };

        let start = Instant::now();
        let code = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status.code().unwrap_or(1),
                Ok(None) => {
                    if start.elapsed() > case.timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        break 124;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(err) => {
                    eprintln!("wait failed: {err}");
                    break 1;
                }
            }
        };

        match code {
            0 => {
                passed += 1;
                eprintln!("SUPERVISOR PASS {} ({:.0}ms)", case.name, start.elapsed().as_secs_f64() * 1000.0);
            }
            x if x == SKIP_EXIT as i32 => {
                skipped += 1;
                eprintln!("SUPERVISOR SKIP {} ({:.0}ms)", case.name, start.elapsed().as_secs_f64() * 1000.0);
            }
            124 => {
                failed += 1;
                eprintln!("SUPERVISOR FAIL {} timed out after {:.1}s", case.name, case.timeout.as_secs_f64());
            }
            other => {
                failed += 1;
                eprintln!("SUPERVISOR FAIL {} rc={other}", case.name);
            }
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed:  {passed}");
    eprintln!("  Skipped: {skipped}");
    eprintln!("  Failed:  {failed}");

    if failed > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(feature = "deltanet")]
fn run_case(case_name: &str, model_path: Option<&Path>) -> ExitCode {
    let model_path = match model_path {
        Some(path) if path.exists() => path,
        Some(path) => {
            eprintln!("QA SKIP {case_name}: model not found at {}", path.display());
            return ExitCode::from(SKIP_EXIT);
        }
        None => {
            eprintln!("QA SKIP {case_name}: no model path provided");
            return ExitCode::from(SKIP_EXIT);
        }
    };

    let result = std::panic::catch_unwind(|| {
        let mut ctx = build_context(model_path)?;
        Ok::<CaseOutcome, CaseOutcome>(match case_name {
            "forward_finite_logits" => forward_finite_logits(&mut ctx),
            "forward_scratch_matches" => forward_scratch_matches(&mut ctx),
            "sequence_no_hang" => sequence_no_hang(&mut ctx),
            "think_token_detectable" => think_token_detectable(&mut ctx),
            "chatml_single_tokens" => chatml_single_tokens(&mut ctx),
            "givens4_cache_allocates" => givens4_cache_allocates(&mut ctx),
            "givens4_forward_no_hang" => givens4_forward_no_hang(&mut ctx),
            "prefill_batch_matches_sequential" => prefill_batch_matches_sequential(&mut ctx),
            "decode_speed_sanity" => decode_speed_sanity(&mut ctx),
            "vram_leak_signal" => vram_leak_signal(&mut ctx),
            other => CaseOutcome::Fail(format!("unknown case: {other}")),
        })
    });

    match result {
        Ok(Ok(CaseOutcome::Pass(msg))) => {
            eprintln!("QA PASS {case_name}: {msg}");
            ExitCode::SUCCESS
        }
        Ok(Ok(CaseOutcome::Skip(msg))) => {
            eprintln!("QA SKIP {case_name}: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Ok(Ok(CaseOutcome::Fail(msg))) | Ok(Err(CaseOutcome::Fail(msg))) => {
            eprintln!("QA FAIL {case_name}: {msg}");
            ExitCode::from(1)
        }
        Ok(Err(CaseOutcome::Skip(msg))) => {
            eprintln!("QA SKIP {case_name}: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Ok(Err(CaseOutcome::Pass(msg))) => {
            eprintln!("QA PASS {case_name}: {msg}");
            ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!("QA FAIL {case_name}: panic");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "deltanet")]
fn build_context(model_path: &Path) -> Result<Context, CaseOutcome> {
    let hfq = HfqFile::open(model_path)
        .map_err(|e| CaseOutcome::Fail(format!("failed to open HFQ: {e}")))?;
    let model_label = classify_qwen35_candidate(&hfq)?;
    let config = qwen35::config_from_hfq(&hfq)
        .ok_or_else(|| CaseOutcome::Fail("failed to parse Qwen3.5 config".to_string()))?;
    let (tokenizer, tokenizer_source) = load_tokenizer(&hfq)?;
    let mut gpu = rdna_compute::Gpu::init()
        .map_err(|e| CaseOutcome::Skip(format!("GPU init unavailable: {e}")))?;
    let weights = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        qwen35::load_weights(&hfq, &config, &mut gpu)
    }))
    .map_err(|panic| CaseOutcome::Fail(format!("weight load panicked: {}", panic_message(panic))))?
    .map_err(|e| CaseOutcome::Fail(format!("failed to load weights: {e}")))?;

    Ok(Context {
        _hfq: hfq,
        model_label,
        tokenizer_source,
        config,
        tokenizer,
        gpu,
        weights,
    })
}

#[cfg(feature = "deltanet")]
fn classify_qwen35_candidate(hfq: &HfqFile) -> Result<String, CaseOutcome> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|e| CaseOutcome::Fail(format!("bad metadata JSON: {e}")))?;
    let config = meta
        .get("config")
        .ok_or_else(|| CaseOutcome::Fail("no config in metadata".to_string()))?;
    let text_config = config.get("text_config").unwrap_or(config);
    let model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let qwen35_layout = hfq.tensor_data("model.language_model.embed_tokens.weight").is_some();
    if model_type.starts_with("qwen3_5") || model_type == "qwen3.5" || qwen35_layout {
        Ok(model_type.to_string())
    } else {
        Err(CaseOutcome::Skip(format!(
            "unsupported HFQ for qwen35 QA: model_type={model_type}"
        )))
    }
}

#[cfg(feature = "deltanet")]
fn load_tokenizer(hfq: &HfqFile) -> Result<(Tokenizer, String), CaseOutcome> {
    if let Some(tokenizer) = Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        return Ok((tokenizer, "hfq-metadata".to_string()));
    }

    let fallback = Path::new(QWEN_GGUF_FALLBACK);
    if !fallback.exists() {
        return Err(CaseOutcome::Skip(format!(
            "tokenizer metadata missing and fallback GGUF not found at {}",
            fallback.display()
        )));
    }

    let gguf = GgufFile::open(fallback)
        .map_err(|e| CaseOutcome::Skip(format!("failed to open fallback GGUF tokenizer: {e}")))?;
    let tokenizer = Tokenizer::from_gguf(&gguf)
        .ok_or_else(|| CaseOutcome::Skip("failed to parse fallback GGUF tokenizer".to_string()))?;
    Ok((tokenizer, format!("gguf:{}", fallback.display())))
}

#[cfg(feature = "deltanet")]
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(feature = "deltanet")]
fn ensure(cond: bool, msg: impl Into<String>) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(msg.into())
    }
}

#[cfg(feature = "deltanet")]
fn logit_diff_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff as f64;
    }
    (max_diff, sum_diff / a.len().max(1) as f64)
}

#[cfg(feature = "deltanet")]
fn prefill_mean_logit_tol(mode: &str) -> f64 {
    match mode {
        // The rotated 2-bit K cache is intentionally the lossiest KV mode.
        // Keep top-token and selected-logit checks strict, but allow a wider
        // full-vocab mean drift than q8/asym3/asym4.
        "asym2" | "turbo2" => PREFILL_ASYM2_MEAN_LOGIT_TOL,
        _ => PREFILL_MEAN_LOGIT_TOL,
    }
}

#[cfg(feature = "deltanet")]
fn prompt_processing_tokens(ctx: &Context) -> Result<Vec<u32>, String> {
    let prompt = "\
Write a Python function that returns the length of the longest substring without repeating characters.


Use a sliding window and include a small doctest.";
    let seed = ctx.tokenizer.encode(prompt);
    ensure(!seed.is_empty(), "prompt encoded to zero tokens")?;

    let mut tokens = seed.clone();
    while tokens.len() < 40 {
        tokens.extend(seed.iter().copied());
    }
    tokens.truncate(40);
    Ok(tokens)
}

#[cfg(feature = "deltanet")]
fn kv_cache_for_mode(
    gpu: &mut rdna_compute::Gpu,
    config: &qwen35::Qwen35Config,
    mode: &str,
    seq_len: usize,
) -> Result<llama::KvCache, String> {
    match mode {
        "q8" => llama::KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            seq_len,
        ),
        "asym4" | "turbo4" => llama::KvCache::new_gpu_asym4(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            seq_len,
        ),
        "asym3" | "turbo3" | "turbo" => llama::KvCache::new_gpu_asym3(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            seq_len,
        ),
        "asym2" | "turbo2" => llama::KvCache::new_gpu_asym2(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            seq_len,
        ),
        other => return Err(format!("unknown KV mode: {other}")),
    }
    .map_err(|e| e.to_string())
}

#[cfg(feature = "deltanet")]
fn forward_finite_logits(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let kv_seq = 128usize;
        let mut kv = llama::KvCache::new_gpu_q8(&mut ctx.gpu, ctx.config.n_layers, ctx.config.n_kv_heads, ctx.config.head_dim, kv_seq)
            .map_err(|e| e.to_string())?;
        let mut dn = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let logits = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, 1, 0, &mut kv, &mut dn)
            .map_err(|e| e.to_string())?;
        ensure(logits.len() == ctx.config.vocab_size, format!("expected vocab {} logits, got {}", ctx.config.vocab_size, logits.len()))?;
        ensure(logits[0].is_finite(), format!("logits[0] is non-finite: {}", logits[0]))?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        ensure(max > -1e10, format!("all logits are too negative: {max}"))?;
        Ok(format!("{} tokenizer={} logits[0]={:.4} max={:.4}", ctx.model_label, ctx.tokenizer_source, logits[0], max))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn forward_scratch_matches(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let kv_seq = 128usize;
        let mut kv1 = llama::KvCache::new_gpu_q8(&mut ctx.gpu, ctx.config.n_layers, ctx.config.n_kv_heads, ctx.config.head_dim, kv_seq)
            .map_err(|e| e.to_string())?;
        let mut dn1 = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let mut kv2 = llama::KvCache::new_gpu_q8(&mut ctx.gpu, ctx.config.n_layers, ctx.config.n_kv_heads, ctx.config.head_dim, kv_seq)
            .map_err(|e| e.to_string())?;
        let mut dn2 = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let scratch = qwen35::Qwen35Scratch::new(&mut ctx.gpu, &ctx.config, 64)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;

        let token = *ctx.tokenizer.encode("Hello").first().ok_or_else(|| "tokenizer returned no tokens for Hello".to_string())?;
        let logits_a = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, token, 0, &mut kv1, &mut dn1)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        qwen35::forward_scratch(&mut ctx.gpu, &ctx.weights, &ctx.config, token, 0, &mut kv2, &mut dn2, &scratch)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let logits_b = ctx.gpu.download_f32(&scratch.logits).map_err(|e| e.to_string())?;
        let max_diff = logits_a.iter().zip(logits_b.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);
        ensure(max_diff < 0.001, format!("forward vs scratch diverged: max_diff={max_diff}"))?;
        Ok(format!("max_diff={max_diff:.6}"))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn sequence_no_hang(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let kv_seq = 128usize;
        let mut kv = llama::KvCache::new_gpu_q8(&mut ctx.gpu, ctx.config.n_layers, ctx.config.n_kv_heads, ctx.config.head_dim, kv_seq)
            .map_err(|e| e.to_string())?;
        let mut dn = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let tokens = ctx.tokenizer.encode("What is the capital");
        ensure(!tokens.is_empty(), "prompt encoded to zero tokens")?;
        let t0 = Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            let _ = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, tok, pos, &mut kv, &mut dn)
                .map_err(|e: hip_bridge::HipError| e.to_string())?;
        }
        let ms = t0.elapsed().as_millis();
        let tps = tokens.len() as f64 / (ms.max(1) as f64 / 1000.0);
        Ok(format!("{} tokens in {ms}ms ({tps:.1} tok/s)", tokens.len()))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn think_token_detectable(ctx: &mut Context) -> CaseOutcome {
    let tokens = ctx.tokenizer.encode("</think>");
    if tokens.is_empty() {
        CaseOutcome::Fail(format!("</think> encoded to empty token sequence via {}", ctx.tokenizer_source))
    } else {
        CaseOutcome::Pass(format!("{} token(s): {:?}", tokens.len(), tokens))
    }
}

#[cfg(feature = "deltanet")]
fn chatml_single_tokens(ctx: &mut Context) -> CaseOutcome {
    let im_start = ctx.tokenizer.encode("<|im_start|>");
    let im_end = ctx.tokenizer.encode("<|im_end|>");
    if im_start.len() != 1 || im_end.len() != 1 {
        CaseOutcome::Fail(format!(
            "ChatML special tokens are not single tokens: im_start={:?} im_end={:?} source={}",
            im_start, im_end, ctx.tokenizer_source
        ))
    } else {
        CaseOutcome::Pass(format!("im_start={} im_end={}", im_start[0], im_end[0]))
    }
}

#[cfg(feature = "deltanet")]
fn givens4_cache_allocates(ctx: &mut Context) -> CaseOutcome {
    match llama::KvCache::new_gpu_asym3(
        &mut ctx.gpu,
        ctx.config.n_layers,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        128,
    ) {
        Ok(kv) => {
            if kv.quant_asym3 && kv.givens_cos.is_some() && kv.givens_sin.is_some() {
                CaseOutcome::Pass(format!("allocated givens4 KV for {} layers", ctx.config.n_layers))
            } else {
                CaseOutcome::Fail("givens4 KV cache flags were inconsistent".to_string())
            }
        }
        Err(err) => CaseOutcome::Fail(format!("failed to allocate givens4 KV cache: {err}")),
    }
}

#[cfg(feature = "deltanet")]
fn givens4_forward_no_hang(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let mut kv = llama::KvCache::new_gpu_asym3(
            &mut ctx.gpu,
            ctx.config.n_layers,
            ctx.config.n_kv_heads,
            ctx.config.head_dim,
            128,
        ).map_err(|e: hip_bridge::HipError| e.to_string())?;
        let mut dn = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let tokens = ctx.tokenizer.encode("Hello world");
        ensure(!tokens.is_empty(), "prompt encoded to zero tokens")?;
        let t0 = Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            let _ = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, tok, pos, &mut kv, &mut dn)
                .map_err(|e: hip_bridge::HipError| e.to_string())?;
        }
        Ok(format!("{} tokens in {}ms", tokens.len(), t0.elapsed().as_millis()))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn prefill_batch_matches_sequential(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let tokens = prompt_processing_tokens(ctx)?;
        let lengths = [2usize, 7, 17, 33];
        let kv_modes = env::var("HIPFIRE_QA_KV_MODES")
            .unwrap_or_else(|_| "q8,asym4,asym3,asym2".to_string());
        let kv_modes: Vec<&str> = kv_modes
            .split(',')
            .map(str::trim)
            .filter(|mode| !mode.is_empty())
            .collect();
        ensure(!kv_modes.is_empty(), "HIPFIRE_QA_KV_MODES produced zero modes")?;
        let mut summaries = Vec::new();

        for mode in kv_modes {
            for &n in &lengths {
                ensure(n <= tokens.len(), format!("test length {n} exceeds prompt length {}", tokens.len()))?;
                let kv_seq_len = (n + 8).max(128);
                let mut kv_seq = kv_cache_for_mode(&mut ctx.gpu, &ctx.config, mode, kv_seq_len)?;
                let mut kv_batch = kv_cache_for_mode(&mut ctx.gpu, &ctx.config, mode, kv_seq_len)?;
                let mut dn_seq = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
                    .map_err(|e: hip_bridge::HipError| e.to_string())?;
                let mut dn_batch = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
                    .map_err(|e: hip_bridge::HipError| e.to_string())?;
                let scratch = qwen35::Qwen35Scratch::new_with_kv_max(&mut ctx.gpu, &ctx.config, 64, kv_seq_len)
                    .map_err(|e: hip_bridge::HipError| e.to_string())?;

                for (pos, &tok) in tokens[..n].iter().enumerate() {
                    qwen35::forward_scratch(&mut ctx.gpu, &ctx.weights, &ctx.config, tok, pos, &mut kv_seq, &mut dn_seq, &scratch)
                        .map_err(|e: hip_bridge::HipError| e.to_string())?;
                }
                let seq_logits = ctx.gpu.download_f32(&scratch.logits).map_err(|e| e.to_string())?;

                qwen35::forward_prefill_batch(
                    &mut ctx.gpu,
                    &ctx.weights,
                    &ctx.config,
                    &tokens[..n],
                    0,
                    &mut kv_batch,
                    &mut dn_batch,
                    &scratch,
                    None,
                    None,
                    None,
                    None,
                ).map_err(|e: hip_bridge::HipError| e.to_string())?;
                let batch_logits = ctx.gpu.download_f32(&scratch.logits).map_err(|e| e.to_string())?;

                let seq_top = llama::argmax(&seq_logits);
                let batch_top = llama::argmax(&batch_logits);
                let (max_diff, mean_diff) = logit_diff_stats(&seq_logits, &batch_logits);
                let selected_diff = (seq_logits[seq_top as usize] - batch_logits[seq_top as usize]).abs();
                let mean_tol = prefill_mean_logit_tol(mode);
                ensure(
                    seq_top == batch_top,
                    format!("prefill top token diverged at mode={mode} n={n}: sequential={seq_top} batch={batch_top} max_diff={max_diff:.6} mean_diff={mean_diff:.6} selected_diff={selected_diff:.6}"),
                )?;
                ensure(
                    mean_diff < mean_tol && selected_diff < PREFILL_SELECTED_LOGIT_TOL,
                    format!("prefill logits drift too high at mode={mode} n={n}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} mean_tol={mean_tol:.3} selected_diff={selected_diff:.6}"),
                )?;

                let next = seq_top;
                qwen35::forward_scratch(&mut ctx.gpu, &ctx.weights, &ctx.config, next, n, &mut kv_seq, &mut dn_seq, &scratch)
                    .map_err(|e: hip_bridge::HipError| e.to_string())?;
                let seq_next_logits = ctx.gpu.download_f32(&scratch.logits).map_err(|e| e.to_string())?;
                qwen35::forward_scratch(&mut ctx.gpu, &ctx.weights, &ctx.config, next, n, &mut kv_batch, &mut dn_batch, &scratch)
                    .map_err(|e: hip_bridge::HipError| e.to_string())?;
                let batch_next_logits = ctx.gpu.download_f32(&scratch.logits).map_err(|e| e.to_string())?;

                let seq_next_top = llama::argmax(&seq_next_logits);
                let batch_next_top = llama::argmax(&batch_next_logits);
                let (next_max_diff, next_mean_diff) = logit_diff_stats(&seq_next_logits, &batch_next_logits);
                let next_selected_diff = (seq_next_logits[seq_next_top as usize] - batch_next_logits[seq_next_top as usize]).abs();
                ensure(
                    seq_next_top == batch_next_top,
                    format!("post-prefill decode top token diverged at mode={mode} n={n}: sequential={seq_next_top} batch={batch_next_top} max_diff={next_max_diff:.6} mean_diff={next_mean_diff:.6} selected_diff={next_selected_diff:.6}"),
                )?;
                ensure(
                    next_mean_diff < mean_tol && next_selected_diff < PREFILL_SELECTED_LOGIT_TOL,
                    format!("post-prefill decode logits drift too high at mode={mode} n={n}: max_diff={next_max_diff:.6} mean_diff={next_mean_diff:.6} mean_tol={mean_tol:.3} selected_diff={next_selected_diff:.6}"),
                )?;

                summaries.push(format!(
                    "{mode}/n={n} prefill(max={max_diff:.4},mean={mean_diff:.5},sel={selected_diff:.4}) next(max={next_max_diff:.4},mean={next_mean_diff:.5},sel={next_selected_diff:.4})"
                ));
            }
        }

        Ok(summaries.join("; "))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn decode_speed_sanity(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let kv_seq = 256usize;
        let mut kv = llama::KvCache::new_gpu_q8(&mut ctx.gpu, ctx.config.n_layers, ctx.config.n_kv_heads, ctx.config.head_dim, kv_seq)
            .map_err(|e| e.to_string())?;
        let mut dn = DeltaNetState::new(&mut ctx.gpu, &ctx.config)
            .map_err(|e: hip_bridge::HipError| e.to_string())?;
        let prompt = ctx.tokenizer.encode("The quick brown fox");
        ensure(!prompt.is_empty(), "prompt encoded to zero tokens")?;
        for (pos, &tok) in prompt.iter().enumerate() {
            let _ = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, tok, pos, &mut kv, &mut dn)
                .map_err(|e: hip_bridge::HipError| e.to_string())?;
        }
        let mut tok = 1u32;
        let mut generated = Vec::new();
        let t0 = Instant::now();
        for i in 0..20 {
            let logits = qwen35::forward(&mut ctx.gpu, &ctx.weights, &ctx.config, tok, prompt.len() + i, &mut kv, &mut dn)
                .map_err(|e: hip_bridge::HipError| e.to_string())?;
            tok = llama::argmax(&logits);
            generated.push(tok);
        }
        let ms = t0.elapsed().as_millis();
        let tps = 20.0 / (ms.max(1) as f64 / 1000.0);
        ensure(tps > 10.0, format!("decode too slow: {tps:.1} tok/s"))?;
        let preview = ctx.tokenizer.decode(&generated[..generated.len().min(8)]).replace('\n', " ");
        ensure(!preview.trim().is_empty(), "generated preview was empty")?;
        Ok(format!("{tps:.1} tok/s preview='{}'", preview.trim()))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn vram_leak_signal(ctx: &mut Context) -> CaseOutcome {
    match (|| -> Result<String, String> {
        let (free_before, _) = ctx.gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
        {
            let kv = llama::KvCache::new_gpu_q8(
                &mut ctx.gpu,
                ctx.config.n_layers,
                ctx.config.n_kv_heads,
                ctx.config.head_dim,
                512,
            ).map_err(|e| e.to_string())?;
            let (free_during, _) = ctx.gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
            let alloc_mb = (free_before - free_during) as f64 / 1e6;
            if alloc_mb <= 0.0 {
                return Err(format!("expected VRAM allocation, measured {alloc_mb:.2}MB"));
            }
            drop(kv);
        }
        let (free_after, _) = ctx.gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
        let leak_mb = (free_before as i64 - free_after as i64) as f64 / 1e6;
        Ok(format!("post-drop delta={leak_mb:.2}MB"))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}
