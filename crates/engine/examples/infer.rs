//! Unified hipfire inference — auto-detects model architecture.
//!   infer <model.hfq> [prompt...]                          # text (Qwen3 or Qwen3.5)
//!   infer <model.hfq> --image <image.png> [prompt...]      # VL mode (Qwen3.5 only)
//!   infer <model.hfq> --no-think [prompt...]               # skip thinking
//!   infer <model.hfq> --temp 0.5 [prompt...]               # override temperature
//!   infer <model.hfq> --maxgen 256 [prompt...]             # max generation tokens

use engine::hfq::{self, HfqFile};
use engine::llama::{self, KvCache, ForwardScratch};
#[cfg(feature = "deltanet")]
use engine::qwen35;
#[cfg(feature = "deltanet")]
use engine::qwen35_vl;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);
extern "C" fn handle_sigint(_: libc::c_int) { RUNNING.store(false, Ordering::SeqCst); }

const IMAGE_SIZE: usize = 448;
const IMAGE_PAD_ID: u32 = 248056;
const VISION_START_ID: u32 = 248053;
const VISION_END_ID: u32 = 248054;

fn main() {
    unsafe { libc::signal(libc::SIGINT, handle_sigint as *const () as libc::sighandler_t); }
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: infer <model.hfq> [--image <img>] [--no-think] [--temp T] [--maxgen N] [prompt...]");
        std::process::exit(1);
    }

    // Parse flags
    let no_think = args.iter().any(|a| a == "--no-think");
    let image_path = args.iter().position(|a| a == "--image").and_then(|i| args.get(i + 1).cloned());
    let temp_override: Option<f32> = args.iter().position(|a| a == "--temp").and_then(|i| args.get(i + 1)?.parse().ok());
    let max_gen: usize = args.iter().position(|a| a == "--maxgen").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(2048);
    let vl_mode = image_path.is_some();

    let mut positional = Vec::new();
    let mut skip_next = false;
    for a in args.iter().skip(1) {
        if skip_next { skip_next = false; continue; }
        if a == "--no-think" { continue; }
        if a == "--image" || a == "--temp" || a == "--maxgen" { skip_next = true; continue; }
        positional.push(a.as_str());
    }
    let model_path = positional.first().unwrap_or_else(|| {
        eprintln!("Usage: infer <model.hfq> [--image <img>] [--no-think] [--temp T] [--maxgen N] [prompt...]");
        std::process::exit(1);
    });
    let prompt_text = if positional.len() > 1 { positional[1..].join(" ") }
        else if vl_mode { "Describe this image.".to_string() }
        else { "Hello".to_string() };

    eprintln!("=== hipfire inference ===");
    eprintln!("Model: {model_path}");
    if vl_mode { eprintln!("Image: {}", image_path.as_ref().unwrap()); }
    eprintln!("Prompt: {prompt_text}");

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");

    // Detect architecture from HFQ metadata
    #[cfg(feature = "deltanet")]
    let is_qwen35 = qwen35::config_from_hfq(&hfq).is_some();
    #[cfg(not(feature = "deltanet"))]
    let is_qwen35 = false;

    if is_qwen35 {
        #[cfg(feature = "deltanet")]
        run_qwen35(&hfq, model_path, &prompt_text, image_path, vl_mode, no_think, temp_override, max_gen);
    } else {
        run_llama(&hfq, &prompt_text, temp_override, max_gen);
    }
}

// ─── Qwen3 / LLaMA path (standard attention, forward_scratch, GPU sampling) ───

fn run_llama(hfq: &HfqFile, prompt_text: &str, temp_override: Option<f32>, max_gen: usize) {
    let config = hfq::config_from_hfq(hfq).expect("failed to read model config");
    eprintln!("Arch: Qwen3/LLaMA (standard attention)");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not found in HFQ");

    let temp = temp_override.unwrap_or(0.3);
    let top_p: f32 = 0.8;

    // ChatML prompt
    let mut prompt_tokens = tokenizer.encode(prompt_text);
    if tokenizer.encode("<|im_start|>").len() == 1 {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let nl = tokenizer.encode("\n");
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&tokenizer.encode("user"));
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&prompt_tokens);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&tokenizer.encode("assistant"));
        chat.extend_from_slice(&nl);
        prompt_tokens = chat;
    }
    eprintln!("Prompt: {} tokens, temp={temp}", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading weights...");
    let weights = hfq::load_weights_hfq(hfq, &config, &mut gpu).expect("failed to load weights");

    let kv_seq = config.max_seq_len.min(4096);
    let mut kv_cache = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap();
    let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
    let mut rng_state = 42u32;

    // Batched prefill
    let t_pf = Instant::now();
    let prefill_logits = llama::prefill_forward(&mut gpu, &weights, &config, &prompt_tokens, &mut kv_cache);
    let mut next_token = if let Ok(logits) = prefill_logits {
        let ms = t_pf.elapsed().as_millis();
        eprintln!("Prefill: {}ms ({:.0} tok/s) [batched]", ms, prompt_tokens.len() as f64 / (ms as f64 / 1000.0));
        llama::argmax(&logits)
    } else {
        // Sequential fallback
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let (_, rng) = llama::forward_scratch(&mut gpu, &weights, &config, token, pos, &mut kv_cache,
                &scratch, temp.max(0.01), top_p, rng_state, 0, 1.0).expect("forward failed");
            rng_state = rng;
        }
        let ms = t_pf.elapsed().as_millis();
        eprintln!("Prefill: {}ms ({:.0} tok/s) [sequential]", ms, prompt_tokens.len() as f64 / (ms as f64 / 1000.0));
        let mut out = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out, &scratch.sample_buf.buf).unwrap();
        u32::from_ne_bytes([out[0], out[1], out[2], out[3]])
    };

    // Generate (GPU sampling — zero logits download)
    let t_gen = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut generated = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        if next_token == config.eos_token || !RUNNING.load(Ordering::Relaxed) { break; }

        token_history.push(next_token);
        let hist_start = token_history.len().saturating_sub(64);
        let hist_slice = &token_history[hist_start..];
        let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

        let pos = prompt_tokens.len() + generated.len() - 1;
        let (tok, rng) = llama::forward_scratch(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache,
            &scratch, temp.max(0.01), top_p, rng_state, hist_slice.len(), 1.1).expect("forward failed");
        next_token = tok;
        rng_state = rng;
    }

    let ms = t_gen.elapsed().as_millis();
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), ms,
        if ms > 0 { generated.len() as f64 / (ms as f64 / 1000.0) } else { 0.0 });
}

// ─── Qwen3.5 path (DeltaNet + FullAttn, thinking mode, VL support) ────────────

#[cfg(feature = "deltanet")]
fn run_qwen35(hfq: &HfqFile, model_path: &str, prompt_text: &str, image_path: Option<String>,
              vl_mode: bool, no_think: bool, temp_override: Option<f32>, max_gen: usize) {
    let text_config = qwen35::config_from_hfq(hfq).unwrap();
    eprintln!("Arch: Qwen3.5 (DeltaNet + FullAttention)");
    eprintln!("Config: dim={}, layers={}, vocab={}", text_config.dim, text_config.n_layers, text_config.vocab_size);

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not found in HFQ");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // VL: vision encode
    let visual_tokens: Option<Vec<f32>>;
    let n_visual_tokens: usize;
    if vl_mode {
        let vc = qwen35_vl::vision_config_from_hfq(hfq).expect("no vision config");
        eprintln!("Vision: hidden={}, layers={}, heads={}", vc.hidden_size, vc.num_layers, vc.num_heads);
        let img = image_path.as_ref().unwrap();
        let pixels = engine::image::load_and_preprocess(Path::new(img), IMAGE_SIZE);
        let gh = IMAGE_SIZE / vc.patch_size;
        let gw = IMAGE_SIZE / vc.patch_size;
        n_visual_tokens = (gh * gw) / (vc.spatial_merge_size * vc.spatial_merge_size);
        let patches = engine::image::extract_patches(&pixels, 3, IMAGE_SIZE, IMAGE_SIZE, vc.patch_size, vc.temporal_patch_size);
        eprintln!("Loading vision weights...");
        let vw = qwen35_vl::load_vision_weights(hfq, &vc, &mut gpu).expect("vision weights failed");
        eprintln!("Running vision encoder...");
        let t = Instant::now();
        let vt = qwen35_vl::vision_forward(&mut gpu, &vw, &vc, &patches, gh, gw).expect("vision failed");
        eprintln!("Vision encoder: {:.1}s", t.elapsed().as_secs_f32());
        drop(vw);
        visual_tokens = Some(vt);
    } else {
        visual_tokens = None;
        n_visual_tokens = 0;
    }

    // Text weights
    eprintln!("Loading text weights...");
    let weights = qwen35::load_weights(hfq, &text_config, &mut gpu).expect("text weights failed");

    let kv_seq = 4096usize;
    let mut kv_cache = KvCache::new_gpu(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap();
    let mut dn_state = qwen35::DeltaNetState::new(&mut gpu, &text_config).unwrap();

    // ChatML prompt
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let q_tokens = tokenizer.encode(prompt_text);

    let mut prompt_tokens: Vec<u32> = Vec::new();
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&tokenizer.encode("user"));
    prompt_tokens.extend_from_slice(&nl);
    if vl_mode {
        prompt_tokens.push(VISION_START_ID);
        for _ in 0..n_visual_tokens { prompt_tokens.push(IMAGE_PAD_ID); }
        prompt_tokens.push(VISION_END_ID);
        prompt_tokens.extend_from_slice(&nl);
    }
    prompt_tokens.extend_from_slice(&q_tokens);
    prompt_tokens.extend_from_slice(&im_end);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&tokenizer.encode("assistant"));
    prompt_tokens.extend_from_slice(&nl);

    eprintln!("Prompt: {} tokens{}", prompt_tokens.len(),
        if vl_mode { format!(" ({} visual + {} text)", n_visual_tokens, prompt_tokens.len() - n_visual_tokens) } else { String::new() });

    let sc = llama::SamplingConfig::vl_thinking();
    let temp = temp_override.unwrap_or(sc.think_temp);
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &text_config, sc.repeat_window).expect("scratch failed");

    // Prefill
    let t_pf = Instant::now();
    let mut visual_idx = 0usize;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        if vl_mode && token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let vt = visual_tokens.as_ref().unwrap();
            let emb = &vt[visual_idx * text_config.dim..(visual_idx + 1) * text_config.dim];
            qwen35::forward_scratch_embed(&mut gpu, &weights, &text_config, emb, pos, &mut kv_cache, &mut dn_state, &scratch).unwrap();
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, token, pos, &mut kv_cache, &mut dn_state, &scratch).unwrap();
        }
    }
    let ms = t_pf.elapsed().as_millis();
    eprintln!("Prefill: {}ms ({:.0} tok/s)", ms, prompt_tokens.len() as f64 / (ms as f64 / 1000.0));

    // Thinking mode
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let think_end_token = { let t = tokenizer.encode("</think>"); if t.len() == 1 { Some(t[0]) } else { None } };

    let prefill_len;
    let mut in_thinking;
    if !no_think {
        let think_tokens = tokenizer.encode("<think>\n");
        for (i, &t) in think_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, t, prompt_tokens.len() + i, &mut kv_cache, &mut dn_state, &scratch).unwrap();
        }
        prefill_len = prompt_tokens.len() + think_tokens.len();
        in_thinking = true;
        eprint!("<think>");
    } else {
        prefill_len = prompt_tokens.len();
        in_thinking = false;
    }

    // First token from prefill logits
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    llama::apply_ngram_block(&mut logits, &prompt_tokens);
    let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);

    let t_gen = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut generated = Vec::new();
    let mut think_count = 0usize;
    let max_think = 512;

    for _ in 0..max_gen {
        generated.push(next_token);
        token_history.push(next_token);
        if in_thinking { think_count += 1; }

        if in_thinking && (think_end_token == Some(next_token) || think_count >= max_think) {
            in_thinking = false;
            eprint!("</think>\n");
        } else {
            let text = tokenizer.decode(&[next_token]);
            if in_thinking { eprint!("{text}"); }
            else { print!("{text}"); std::io::stdout().flush().ok(); }
        }

        if next_token == text_config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }
        if !RUNNING.load(Ordering::Relaxed) { break; }

        let pos = prefill_len + generated.len() - 1;
        let t = if in_thinking { temp } else { temp };

        qwen35::forward_scratch(&mut gpu, &weights, &text_config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();

        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &token_history);
        llama::apply_repeat_penalty(&mut logits, &token_history, sc.repeat_window, sc.repeat_penalty);
        next_token = llama::sample_top_p(&logits, t, sc.top_p);
    }

    let ms = t_gen.elapsed().as_millis();
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), ms,
        if ms > 0 { generated.len() as f64 / (ms as f64 / 1000.0) } else { 0.0 });
}
