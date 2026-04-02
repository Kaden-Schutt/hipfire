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
    let image_path = args.iter().position(|a| a == "--image")
        .and_then(|i| args.get(i + 1).cloned());
    let vl_mode = image_path.is_some();

    let mut positional = Vec::new();
    let mut skip_next = false;
    for a in args.iter().skip(1) {
        if skip_next { skip_next = false; continue; }
        if a == "--no-think" { continue; }
        if a == "--image" { skip_next = true; continue; }
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
    let mut kv_cache = llama::KvCache::new_gpu(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &text_config).unwrap();

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

    eprintln!("Prompt: {} tokens{}", prompt_tokens.len(),
        if vl_mode { format!(" ({} visual + {} text)", n_visual_tokens, prompt_tokens.len() - n_visual_tokens) } else { String::new() });

    // Scratch buffers
    let sc = llama::SamplingConfig::vl_thinking();
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &text_config, sc.repeat_window)
        .expect("failed to create scratch");

    // Prefill
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
    let prefill_ms = t_pf.elapsed().as_millis();
    eprintln!("Prefill: {}ms ({:.0} tok/s)", prefill_ms,
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0));

    // Thinking mode
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let think_end_token = { let t = tokenizer.encode("</think>"); if t.len() == 1 { Some(t[0]) } else { None } };
    let max_gen = 2048;
    let max_think = 512;

    let prefill_len;
    let mut in_thinking;
    if !no_think {
        let think_tokens = tokenizer.encode("<think>\n");
        for (i, &t) in think_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, t, prompt_tokens.len() + i, &mut kv_cache, &mut dn_state, &scratch)
                .expect("forward_scratch failed");
        }
        prefill_len = prompt_tokens.len() + think_tokens.len();
        in_thinking = true;
        eprint!("<think>");
    } else {
        prefill_len = prompt_tokens.len();
        in_thinking = false;
    }

    // First token
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    llama::apply_ngram_block(&mut logits, &prompt_tokens);
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
        let temp = if in_thinking { sc.think_temp } else { sc.answer_temp };

        qwen35::forward_scratch(&mut gpu, &weights, &text_config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch).expect("forward_scratch failed");

        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &token_history);
        llama::apply_repeat_penalty(&mut logits, &token_history, sc.repeat_window, sc.repeat_penalty);
        next_token = llama::sample_top_p(&logits, temp, sc.top_p);
    }

    let gen_ms = t_gen.elapsed().as_millis();
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}
