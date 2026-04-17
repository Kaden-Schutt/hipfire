//! Qwen3.5-VL inference: image + text question → text answer.
//! Usage: infer_vl <model.hfq> <image.png> [question...]

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
    unsafe { libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t); }
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: infer_vl <model.hfq> <image.png> [question...]");
        std::process::exit(1);
    }
    let no_think = args.iter().any(|a| a == "--no-think");
    let filtered: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let model_path = filtered.get(1).unwrap_or_else(|| { eprintln!("Usage: infer_vl <model.hfq> <image.png> [--no-think] [question...]"); std::process::exit(1); });
    let image_path = filtered.get(2).unwrap_or_else(|| { eprintln!("Usage: infer_vl <model.hfq> <image.png> [--no-think] [question...]"); std::process::exit(1); });
    let question = if filtered.len() > 3 { filtered[3..].iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ") } else { "Describe this image.".to_string() };

    eprintln!("=== hipfire Qwen3.5-VL inference ===");
    eprintln!("Model: {model_path}");
    eprintln!("Image: {image_path}");
    eprintln!("Question: {question}");

    // Load model
    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let text_config = qwen35::config_from_hfq(&hfq).expect("failed to read text config");
    let vision_config = qwen35_vl::vision_config_from_hfq(&hfq).expect("failed to read vision config");
    eprintln!("Text: dim={}, layers={}, vocab={}", text_config.dim, text_config.n_layers, text_config.vocab_size);
    eprintln!("Vision: hidden={}, layers={}, heads={}, patch={}",
        vision_config.hidden_size, vision_config.num_layers, vision_config.num_heads, vision_config.patch_size);

    // Load tokenizer
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not found in HFQ metadata");

    // Load and preprocess image
    eprintln!("Preprocessing image...");
    let (pixels, img_h, img_w) = engine::image::load_and_preprocess(Path::new(image_path), vision_config.patch_size);
    let grid_h = img_h / vision_config.patch_size;
    let grid_w = img_w / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);
    eprintln!("Image: {}x{} → {}x{} patches → {} visual tokens", img_w, img_h, grid_h, grid_w, n_visual_tokens);

    // Extract patches for vision encoder
    let patches = engine::image::extract_patches(
        &pixels, 3, img_h, img_w,
        vision_config.patch_size, vision_config.temporal_patch_size,
    );

    // Init GPU first (needed for vision weight loading)
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Load vision weights (GPU-side for fast inference)
    eprintln!("Loading vision weights...");
    let vision_weights = qwen35_vl::load_vision_weights(&hfq, &vision_config, &mut gpu)
        .expect("failed to load vision weights");

    // Run vision encoder (GPU linear layers + CPU attention)
    eprintln!("Running vision encoder...");
    let t_vis = Instant::now();
    let visual_tokens = qwen35_vl::vision_forward(&mut gpu, &vision_weights, &vision_config, &patches, grid_h, grid_w)
        .expect("vision forward failed");
    eprintln!("Vision encoder: {:.1}s", t_vis.elapsed().as_secs_f32());
    assert_eq!(visual_tokens.len(), n_visual_tokens * text_config.dim);

    eprintln!("Loading text weights...");
    let weights = qwen35::load_weights(&hfq, &text_config, &mut gpu).expect("failed to load text weights");

    let kv_seq = 2048usize;
    let mut kv_cache = llama::KvCache::new_gpu(&mut gpu, text_config.n_layers, text_config.n_kv_heads, text_config.head_dim, kv_seq).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &text_config).unwrap();

    // Build prompt with vision tokens:
    // <|im_start|>user\n<|vision_start|><|image_pad|>×N<|vision_end|>\n{question}<|im_end|>\n<|im_start|>assistant\n
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let nl = tokenizer.encode("\n");
    let q_tokens = tokenizer.encode(&question);

    let mut prompt_tokens: Vec<u32> = Vec::new();
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&user_tok);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.push(VISION_START_ID);
    for _ in 0..n_visual_tokens {
        prompt_tokens.push(IMAGE_PAD_ID);
    }
    prompt_tokens.push(VISION_END_ID);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&q_tokens);
    prompt_tokens.extend_from_slice(&im_end);
    prompt_tokens.extend_from_slice(&nl);
    prompt_tokens.extend_from_slice(&im_start);
    prompt_tokens.extend_from_slice(&asst_tok);
    prompt_tokens.extend_from_slice(&nl);

    eprintln!("Prompt: {} tokens ({} visual + {} text)", prompt_tokens.len(), n_visual_tokens, prompt_tokens.len() - n_visual_tokens);

    // Zero-alloc scratch buffers (pre-allocated once, reused every token)
    let sc = llama::SamplingConfig::vl_thinking();
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &text_config, sc.repeat_window)
        .expect("failed to create scratch");
    let mut rng_state = 42u32;

    // Prefill: process prompt tokens sequentially using scratch
    let t_pf = Instant::now();
    let mut visual_idx = 0usize;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        if token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * text_config.dim..(visual_idx + 1) * text_config.dim];
            qwen35::forward_scratch_embed(&mut gpu, &weights, &text_config, emb, pos, &mut kv_cache, &mut dn_state, &scratch)
                .expect("forward_scratch_embed failed");
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(&mut gpu, &weights, &text_config, token, pos, &mut kv_cache, &mut dn_state, &scratch)
                .expect("forward_scratch failed");
        }
    }
    let prefill_ms = t_pf.elapsed().as_millis();
    eprintln!("Prefill: {}ms ({} tokens, {:.0} tok/s)", prefill_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0));

    // Generation with zero-alloc forward + GPU sampling + dual-temp thinking
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let think_end_id = tokenizer.encode("</think>");
    let think_end_token = if think_end_id.len() == 1 { Some(think_end_id[0]) } else { None };
    let max_gen = 2048;

    // Optionally append <think>\n for thinking mode
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

    // First token: download logits from scratch, apply n-gram block, sample on CPU
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    llama::apply_ngram_block(&mut logits, &prompt_tokens);
    let temp = if in_thinking { sc.think_temp } else { sc.answer_temp };
    let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);

    let t_gen = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut generated = Vec::new();
    let max_think = 256;
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

        // Zero-alloc forward — logits land in scratch.logits on GPU
        qwen35::forward_scratch(&mut gpu, &weights, &text_config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch).expect("forward_scratch failed");

        // N-gram block (requires CPU roundtrip — TODO: GPU kernel)
        // Disabled for perf measurement — re-enable after implementing GPU n-gram kernel
        if std::env::var("NO_NGRAM").is_err() {
            logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_ngram_block(&mut logits, &token_history);
            let logits_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
            };
            gpu.hip.memcpy_htod(&scratch.logits.buf, logits_bytes).unwrap();
        }

        // GPU sampling
        let hist_start = token_history.len().saturating_sub(sc.repeat_window);
        let hist_slice = &token_history[hist_start..];
        let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
        if !hist_bytes.is_empty() {
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();
        }
        let (tok, rng) = gpu.sample_top_p(
            &scratch.logits, &scratch.sample_buf, &scratch.repeat_buf,
            text_config.vocab_size, temp, sc.top_p, rng_state,
            hist_slice.len(), sc.repeat_penalty,
        ).expect("sample_top_p failed");
        next_token = tok;
        rng_state = rng;
    }

    let gen_ms = t_gen.elapsed().as_millis();
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}
