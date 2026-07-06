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

//! Standalone multimodal bring-up driver for `hipfire-arch-gemma3-vl` (V5+).
//!
//! image(s) + prompt → SigLIP encoder → projector → splice the image embeddings
//! at the `<image>` placeholders → gemma3 text decoder → greedy continuation.
//! Bypasses the daemon (like `infer_gemma3`). Validates the whole multimodal
//! path against still images and **video slice-stacks** (an MRI series `.webm`).
//!
//! ```text
//! # single still image
//! cargo run --release --example infer_gemma3_vl -p hipfire-arch-gemma3-vl -- \
//!   --hfq ~/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq \
//!   --image benchmarks/vision/images/mri_human_brain.jpg \
//!   --prompt "Describe this brain MRI." --max-new-tokens 64
//!
//! # video slice-stack: K uniformly-sampled frames as K image blocks
//! cargo run --release --example infer_gemma3_vl -p hipfire-arch-gemma3-vl -- \
//!   --hfq ~/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq \
//!   --video "$HOME/.hipfire/datasets/MRI_BRAIN/MRI BRAIN - Set 1.webm" \
//!   --max-frames 8 --prompt "Describe this brain MRI series." --max-new-tokens 128
//! ```

use std::path::Path;

use hipfire_arch_gemma3 as g3;
use hipfire_arch_gemma3_vl as vl;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;

fn arg(flag: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
}

/// All values for a repeatable flag, in command-line order (e.g. `--image a
/// --image b` → `[a, b]`). Used for true multi-image requests.
fn args_all(flag: &str) -> Vec<String> {
    let a: Vec<String> = std::env::args().collect();
    a.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hfq_path = arg("--hfq").ok_or("--hfq required")?;
    let image_paths = args_all("--image"); // repeatable → multi-image
    let video_path = arg("--video");
    if image_paths.is_empty() == video_path.is_none() {
        return Err(
            "provide either one or more --image <path> (repeatable) or one --video <path>".into(),
        );
    }
    let max_frames: usize = arg("--max-frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let prompt = arg("--prompt").unwrap_or_else(|| "Describe this image.".to_string());
    let max_new = arg("--max-new-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    eprintln!("[1/6] opening HFQ {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path))?;
    if hfq.arch_id != 13 {
        eprintln!("  warning: arch_id={} (gemma3-vl expects 13)", hfq.arch_id);
    }
    let tok =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;

    eprintln!("[2/6] init GPU + loading multimodal weights");
    let mut gpu = Gpu::init()?;
    let loaded = vl::load_vl(&mut hfq, &mut gpu)?;
    let (text_cfg, vl_cfg, w) = (&loaded.text_cfg, &loaded.vl_cfg, &loaded.weights);
    eprintln!(
        "  text: hidden={} layers={}; vision: hidden={} layers={}; mm_tokens={}",
        text_cfg.hidden_size,
        text_cfg.num_hidden_layers,
        vl_cfg.vision.hidden_size,
        vl_cfg.vision.num_hidden_layers,
        vl_cfg.mm_tokens_per_image,
    );

    // [3/6] preprocess → one SigLIP patch tensor per image (stills = N --image
    // flags; video = K uniformly-sampled frames in slice order).
    let patch_sets: Vec<Vec<f32>> = if !image_paths.is_empty() {
        eprintln!("[3/6] preprocessing {} image(s)", image_paths.len());
        image_paths
            .iter()
            .map(|ip| {
                eprintln!("  - {ip}");
                vl::preprocess_image(Path::new(ip), &vl_cfg.vision)
            })
            .collect::<Result<_, _>>()?
    } else {
        let vp = video_path.as_ref().unwrap();
        eprintln!("[3/6] decoding video {vp} (max {max_frames} frames)");
        let frames = hipfire_media::decode_frames(Path::new(vp), max_frames)?;
        eprintln!("  sampled {} frames", frames.len());
        frames
            .iter()
            .map(|bytes| vl::preprocess_image_bytes(bytes, &vl_cfg.vision))
            .collect::<Result<_, _>>()?
    };
    let n_images = patch_sets.len();

    eprintln!("[4/6] vision encoder + projector ({n_images} image(s))");
    let th = vl_cfg.text_hidden_size;
    let mm = vl_cfg.mm_tokens_per_image;
    let mut img_embeds: Vec<f32> = Vec::with_capacity(n_images * mm * th);
    for (i, patches) in patch_sets.iter().enumerate() {
        let t_img = std::time::Instant::now();
        let vis = vl::vision_forward(&mut gpu, &w.vision, &vl_cfg.vision, patches)?;
        let img_embeds_gpu = vl::project(&mut gpu, &w.projector, vl_cfg, &vis)?;
        gpu.free_tensor(vis)?;
        img_embeds.extend(gpu.download_f32(&img_embeds_gpu)?); // [mm · th] per image
        gpu.free_tensor(img_embeds_gpu)?;
        eprintln!(
            "  image {}/{n_images} encoded in {:.2}s",
            i + 1,
            t_img.elapsed().as_secs_f64()
        );
    }

    // Build the token stream: gemma chat frame with N image blocks spliced in.
    // <bos><start_of_turn>user\n [\n\n<boi> <image>×mm <eoi>\n\n]×N {prompt}
    // <end_of_turn>\n<start_of_turn>model\n
    //
    // Each image block is wrapped in `\n\n…\n\n` — HF Gemma3's `full_image_sequence`
    // (processor replaces every `<start_of_image>` with exactly this). The `\n\n`
    // delimiters matter for multi-image: without them adjacent blocks become
    // `…<eoi><boi>…`, which is out-of-distribution and degrades the decode.
    eprintln!("[5/6] building prompt + prefilling");
    let nn = tok.encode("\n\n");
    let mut ids: Vec<u32> = Vec::new();
    if let Some(bos) = tok.special_token_id("<bos>") {
        ids.push(bos);
    }
    ids.extend(tok.encode("<start_of_turn>user\n"));
    for _ in 0..n_images {
        ids.extend(&nn);
        ids.push(vl_cfg.boi_token_index);
        ids.extend(std::iter::repeat_n(vl_cfg.image_token_index, mm));
        ids.push(vl_cfg.eoi_token_index);
        ids.extend(&nn);
    }
    ids.extend(tok.encode(&format!("{prompt}<end_of_turn>\n<start_of_turn>model\n")));

    let mut state = g3::Gemma3State::new_with_max_seq(
        &mut gpu,
        text_cfg,
        ids.len() + max_new + 16,
        hipfire_runtime::kv::KvQuantMode::Unquantized,
        4,
    )
    .map_err(|e| format!("state: {e:?}"))?;

    // Prefill: text tokens via forward_step; image placeholders consume the
    // projected embedding rows in order via forward_step_with_embed.
    let mut img_row = 0usize;
    for &id in &ids {
        if id == vl_cfg.image_token_index {
            let row = &img_embeds[img_row * th..(img_row + 1) * th];
            g3::forward_step_with_embed(&mut gpu, &w.text, text_cfg, &mut state, row)?;
            img_row += 1;
        } else {
            g3::forward_step(&mut gpu, &w.text, text_cfg, &mut state, id)?;
        }
    }
    eprintln!(
        "  prefilled {} tokens ({} image rows spliced across {} image(s))",
        ids.len(),
        img_row,
        n_images
    );

    // Decode. Default is the clean GPU-argmax greedy path (`forward_step_greedy`,
    // identical to the V5 bring-up). `--repeat-penalty > 1.0` switches to a
    // host-side argmax that penalizes recently-emitted tokens — needed only when
    // pure greedy collapses into a loop on out-of-distribution input (e.g.
    // several near-identical MRI slices → a markdown-bullet `*` attractor). The
    // daemon path has its own sampling + n-gram loop guard, so this is example-only.
    let repeat_penalty: f32 = arg("--repeat-penalty")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let repeat_window: usize = arg("--repeat-window")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    eprintln!("[6/6] decoding {max_new} tokens (repeat_penalty={repeat_penalty})");
    let eos = tok.special_token_id("<end_of_turn>");
    let mut gen: Vec<u32> = Vec::new();

    if repeat_penalty == 1.0 {
        // Clean greedy: GPU argmax + fused greedy step (the known-good V5 path).
        let mut next = gpu.argmax_f32(&state.logits, text_cfg.vocab_size)?;
        for _ in 0..max_new {
            if Some(next) == eos {
                break;
            }
            gen.push(next);
            next = g3::forward_step_greedy(&mut gpu, &w.text, text_cfg, &mut state, next)?;
        }
    } else {
        // Host argmax with repeat penalty over the recent window.
        let vocab = text_cfg.vocab_size;
        let argmax_penalized = |logits: &mut [f32], gen: &[u32]| -> u32 {
            let start = gen.len().saturating_sub(repeat_window);
            for &t in &gen[start..] {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 {
                    *l / repeat_penalty
                } else {
                    *l * repeat_penalty
                };
            }
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        let mut logits = gpu.download_f32(&state.logits)?;
        debug_assert_eq!(logits.len(), vocab);
        let mut next = argmax_penalized(&mut logits, &gen);
        for _ in 0..max_new {
            if Some(next) == eos {
                break;
            }
            gen.push(next);
            g3::forward_step(&mut gpu, &w.text, text_cfg, &mut state, next)?;
            logits = gpu.download_f32(&state.logits)?;
            next = argmax_penalized(&mut logits, &gen);
        }
    }

    println!(
        "\n=== gemma3-vl continuation ===\n{}\n==============================",
        tok.decode(&gen)
    );
    Ok(())
}
