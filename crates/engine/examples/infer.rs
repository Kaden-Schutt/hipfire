//! Unified Qwen3.5 inference — text-only or vision-language.
//! Usage:
//!   infer <model.hfq> [prompt...]                          # text-only
//!   infer <model.hfq> --image <image.png> [prompt...]      # VL mode
//!   infer <model.hfq> --no-think [prompt...]               # skip thinking

use engine::hfq::HfqFile;
use engine::llama;
use engine::qwen35;
use engine::qwen35::DeltaNetState;
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
        eprintln!("Usage: infer <model.hfq> [--image <image.png>] [--no-think] [prompt...]");
        std::process::exit(1);
    }

    // Parse flags
    let no_think = args.iter().any(|a| a == "--no-think");
    let debug_cmp = args.iter().any(|a| a == "--debug-compare");
    let use_asym = args.iter().any(|a| a == "--asym");
    let max_tokens: usize = args.iter().position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
        .unwrap_or(2048);
    let turbo_bits: u8 = if args.iter().any(|a| a == "--turbo2") { 2 }
        else if args.iter().any(|a| a == "--turbo3") { 3 }
        else if args.iter().any(|a| a == "--turbo4") { 4 }
        else { 0 };
    let image_path = args.iter().position(|a| a == "--image")
        .and_then(|i| args.get(i + 1).cloned());
    let vl_mode = image_path.is_some();

    let mut positional = Vec::new();
    let mut skip_next = false;
    for a in args.iter().skip(1) {
        if skip_next { skip_next = false; continue; }
        if a == "--no-think" || a == "--debug-compare" || a == "--turbo2" || a == "--turbo3" || a == "--turbo4" || a == "--asym" { continue; }
        if a == "--image" || a == "--max-tokens" { skip_next = true; continue; }
        positional.push(a.as_str());
    }
    let model_path = positional.first().unwrap_or_else(|| {
        eprintln!("Usage: infer <model.hfq> [--image <image.png>] [--no-think] [prompt...]");
        std::process::exit(1);
    });
    let prompt_text = if positional.len() > 1 {
        positional[1..].join(" ")
    } else if vl_mode {
        "Describe this image.".to_string()
    } else {
        "Hello".to_string()
    };

    eprintln!("=== hipfire Qwen3.5 inference ===");
    eprintln!("Model: {model_path}");
    if vl_mode { eprintln!("Image: {}", image_path.as_ref().unwrap()); }
    eprintln!("Prompt: {prompt_text}");

    // Load model config + tokenizer
    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let text_config = qwen35::config_from_hfq(&hfq).expect("failed to read Qwen3.5 config");
    eprintln!("Text: dim={}, layers={}, vocab={}", text_config.dim, text_config.n_layers, text_config.vocab_size);

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not found in HFQ metadata");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // VL: load vision weights + encode image (only if --image given)
    let visual_tokens: Option<Vec<f32>>;
    let n_visual_tokens: usize;

    if vl_mode {
        let vision_config = qwen35_vl::vision_config_from_hfq(&hfq).expect("no vision config in model");
        eprintln!("Vision: hidden={}, layers={}, heads={}", vision_config.hidden_size, vision_config.num_layers, vision_config.num_heads);

        let img = image_path.as_ref().unwrap();
        let pixels = engine::image::load_and_preprocess(Path::new(img), IMAGE_SIZE);
        let grid_h = IMAGE_SIZE / vision_config.patch_size;
        let grid_w = IMAGE_SIZE / vision_config.patch_size;
        n_visual_tokens = (grid_h * grid_w) / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

        let patches = engine::image::extract_patches(
            &pixels, 3, IMAGE_SIZE, IMAGE_SIZE,
            vision_config.patch_size, vision_config.temporal_patch_size,
        );

        eprintln!("Loading vision weights...");
        let vision_weights = qwen35_vl::load_vision_weights(&hfq, &vision_config, &mut gpu)
            .expect("failed to load vision weights");

        eprintln!("Running vision encoder...");
        let t_vis = Instant::now();
        let vt = qwen35_vl::vision_forward(&mut gpu, &vision_weights, &vision_config, &patches, grid_h, grid_w)
            .expect("vision forward failed");
        eprintln!("Vision encoder: {:.1}s", t_vis.elapsed().as_secs_f32());
        drop(vision_weights); // free VRAM for text model

        visual_tokens = Some(vt);
    } else {
        visual_tokens = None;
        n_visual_tokens = 0;
    }

    // Load text weights
    eprintln!("Loading text weights...");
    let weights = qwen35::load_weights(&hfq, &text_config, &mut gpu).expect("failed to load text weights");

    let kv_seq = 4096usize;
    let mut kv_cache = if use_asym {
        llama::KvCache::new_gpu_asym_q8k_turbo4v(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap()
    } else if turbo_bits > 0 {
        eprintln!("KV cache: turbo{turbo_bits}");
        llama::KvCache::new_gpu_turbo(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq, turbo_bits).unwrap()
    } else {
        eprintln!("KV cache: Q8");
        llama::KvCache::new_gpu_q8(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap()
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &text_config).unwrap();

    if debug_cmp {
        let mut kv2 = llama::KvCache::new_gpu(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap();
        let mut dn2 = DeltaNetState::new(&mut gpu, &text_config).unwrap();
        let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &text_config, 128).unwrap();
        let test_token = tokenizer.encode("Hello")[0];
        // Run a sequence of tokens through both paths and compare at each step
        let test_seq = tokenizer.encode("What is the capital of France?");
        eprintln!("=== Debug: comparing {} tokens through both paths ===", test_seq.len());
        for (i, &tok) in test_seq.iter().enumerate() {
            let logits_a = qwen35::forward(&mut gpu, &weights, &text_config, tok, i, &mut kv_cache, &mut dn_state).unwrap();
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, tok, i, &mut kv2, &mut dn2, &scratch).unwrap();
            let logits_b = gpu.download_f32(&scratch.logits).unwrap();

            let mut max_diff = 0.0f32;
            let mut max_idx = 0;
            for j in 0..logits_a.len().min(logits_b.len()) {
                let d = (logits_a[j] - logits_b[j]).abs();
                if d > max_diff { max_diff = d; max_idx = j; }
            }
            let mut sa: Vec<(usize, f32)> = logits_a.iter().enumerate().map(|(i,&v)| (i,v)).collect();
            sa.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut sb: Vec<(usize, f32)> = logits_b.iter().enumerate().map(|(i,&v)| (i,v)).collect();
            sb.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_match = sa[0].0 == sb[0].0;
            eprintln!("  pos={i:2} tok={tok:6} max_diff={max_diff:.6} top1_match={top_match} A_top={} B_top={}", sa[0].0, sb[0].0);
            if max_diff > 1.0 {
                eprintln!("  DIVERGED at pos {i}!");
                break;
            }
        }
        std::process::exit(0);
    }

    // Build ChatML prompt
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let q_tokens = tokenizer.encode(&prompt_text);

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

    // Include <think>\n in prompt tokens (must be prefilled together, not separate)
    let mut in_thinking = false;
    if !no_think {
        prompt_tokens.extend_from_slice(&tokenizer.encode("<think>"));
        prompt_tokens.extend_from_slice(&nl);
        in_thinking = true;
    }

    eprintln!("Prompt: {} tokens{}", prompt_tokens.len(),
        if vl_mode { format!(" ({} visual + {} text)", n_visual_tokens, prompt_tokens.len() - n_visual_tokens) } else { String::new() });

    let sc = llama::SamplingConfig::text_thinking();
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &text_config, sc.repeat_window)
        .expect("failed to create scratch");

    // Prefill (zero-alloc scratch path)
    let t_pf = Instant::now();
    let mut visual_idx = 0usize;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        if vl_mode && token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let vt = visual_tokens.as_ref().unwrap();
            let emb = &vt[visual_idx * text_config.dim..(visual_idx + 1) * text_config.dim];
            qwen35::forward_scratch_embed(&mut gpu, &weights, &text_config, emb, pos, &mut kv_cache, &mut dn_state, &scratch)
                .expect("forward_scratch_embed failed");
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, token, pos, &mut kv_cache, &mut dn_state, &scratch)
                .expect("forward_scratch failed");
        }
    }
    let prefill_len = prompt_tokens.len();
    let prefill_ms = t_pf.elapsed().as_millis();
    eprintln!("Prefill: {}ms ({:.0} tok/s)", prefill_ms,
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0));
    if in_thinking { eprint!("<think>"); }

    // Thinking mode
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let think_end_seq = tokenizer.encode("</think>");
    let max_gen = max_tokens;
    let max_think = 512;

    // First token
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let temp = if in_thinking { sc.think_temp } else { sc.answer_temp };
    let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);

    let t_gen = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut generated = Vec::new();
    let mut think_count = 0usize;

    for _ in 0..max_gen {
        generated.push(next_token);
        token_history.push(next_token);
        if in_thinking { think_count += 1; }

        // Detect </think> as a token sequence (may be multi-token)
        let think_ended = in_thinking && (think_count >= max_think || {
            let gl = generated.len();
            gl >= think_end_seq.len() && generated[gl - think_end_seq.len()..] == think_end_seq[..]
        });

        if think_ended {
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
        let temp = if in_thinking { sc.think_temp } else { sc.answer_temp };

        qwen35::forward_scratch(&mut gpu, &weights, &text_config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch).expect("forward_scratch failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
        if !in_thinking {
            llama::apply_ngram_block(&mut logits, &token_history);
        }
        llama::apply_repeat_penalty(&mut logits, &token_history, sc.repeat_window, sc.repeat_penalty);
        next_token = llama::sample_top_p(&logits, temp, sc.top_p);
    }

    let gen_ms = t_gen.elapsed().as_millis();
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}

// Debug: compare forward() vs forward_scratch() logits on first token
#[allow(dead_code)]
fn debug_compare(
    gpu: &mut rdna_compute::Gpu,
    weights: &engine::qwen35::Qwen35Weights,
    config: &engine::qwen35::Qwen35Config,
    token: u32,
    kv_cache1: &mut engine::llama::KvCache,
    dn_state1: &mut engine::qwen35::DeltaNetState,
    kv_cache2: &mut engine::llama::KvCache,
    dn_state2: &mut engine::qwen35::DeltaNetState,
    scratch: &engine::qwen35::Qwen35Scratch,
) {
    // Path A: forward() 
    let logits_a = engine::qwen35::forward(gpu, weights, config, token, 0, kv_cache1, dn_state1).unwrap();
    
    // Path B: forward_scratch()
    engine::qwen35::forward_scratch(gpu, weights, config, token, 0, kv_cache2, dn_state2, scratch).unwrap();
    let logits_b = gpu.download_f32(&scratch.logits).unwrap();
    
    // Compare
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut n_diff = 0;
    for i in 0..logits_a.len().min(logits_b.len()) {
        let diff = (logits_a[i] - logits_b[i]).abs();
        if diff > 0.001 { n_diff += 1; }
        if diff > max_diff { max_diff = diff; max_idx = i; }
    }
    eprintln!("COMPARE: max_diff={max_diff:.6} at idx={max_idx}, n_diff(>0.001)={n_diff}/{}", logits_a.len());
    eprintln!("  A[{max_idx}]={:.6}  B[{max_idx}]={:.6}", logits_a[max_idx], logits_b[max_idx]);
    // Check top-5
    let mut sorted_a: Vec<(usize, f32)> = logits_a.iter().enumerate().map(|(i,&v)| (i,v)).collect();
    sorted_a.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut sorted_b: Vec<(usize, f32)> = logits_b.iter().enumerate().map(|(i,&v)| (i,v)).collect();
    sorted_b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  Top-5 A: {:?}", &sorted_a[..5].iter().map(|(i,v)| format!("{}:{:.3}", i, v)).collect::<Vec<_>>());
    eprintln!("  Top-5 B: {:?}", &sorted_b[..5].iter().map(|(i,v)| format!("{}:{:.3}", i, v)).collect::<Vec<_>>());
}
