//! Standalone multimodal bring-up driver for `hipfire-arch-gemma3-vl` (V5).
//!
//! image + prompt → SigLIP encoder → projector → splice the image embeddings at
//! the `<image>` placeholders → gemma3 text decoder → greedy continuation.
//! Bypasses the daemon (like `infer_gemma3`). Validates the whole multimodal
//! path against the brain-MRI fixture.
//!
//! ```text
//! cargo run --release --example infer_gemma3_vl -p hipfire-arch-gemma3-vl -- \
//!   --hfq ~/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq \
//!   --image benchmarks/vision/images/mri_human_brain.jpg \
//!   --prompt "Describe this brain MRI." --max-new-tokens 64
//! ```

use std::path::Path;

use hipfire_arch_gemma3 as g3;
use hipfire_arch_gemma3_vl as vl;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

fn arg(flag: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hfq_path = arg("--hfq").ok_or("--hfq required")?;
    let image_path = arg("--image").ok_or("--image required")?;
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

    eprintln!("[3/6] preprocessing image {image_path}");
    let patches = vl::preprocess_image(Path::new(&image_path), &vl_cfg.vision)?;

    eprintln!("[4/6] vision encoder + projector");
    let vis = vl::vision_forward(&mut gpu, &w.vision, &vl_cfg.vision, &patches)?;
    let img_embeds_gpu = vl::project(&mut gpu, &w.projector, vl_cfg, &vis)?;
    gpu.free_tensor(vis)?;
    let img_embeds = gpu.download_f32(&img_embeds_gpu)?; // [mm_tokens · text_hidden]
    gpu.free_tensor(img_embeds_gpu)?;
    let th = vl_cfg.text_hidden_size;
    let n_img = vl_cfg.mm_tokens_per_image;

    // Build the token stream: gemma chat frame with the image block spliced in.
    // <bos><start_of_turn>user\n <boi> [<image> × mm_tokens] <eoi> \n{prompt}
    // <end_of_turn>\n<start_of_turn>model\n
    eprintln!("[5/6] building prompt + prefilling");
    let mut ids: Vec<u32> = Vec::new();
    if let Some(bos) = tok.special_token_id("<bos>") {
        ids.push(bos);
    }
    ids.extend(tok.encode("<start_of_turn>user\n"));
    ids.push(vl_cfg.boi_token_index);
    ids.extend(std::iter::repeat(vl_cfg.image_token_index).take(n_img));
    ids.push(vl_cfg.eoi_token_index);
    ids.extend(tok.encode(&format!("\n{prompt}<end_of_turn>\n<start_of_turn>model\n")));

    let mut state = g3::Gemma3State::new_with_max_seq(&mut gpu, text_cfg, ids.len() + max_new + 16)
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
        "  prefilled {} tokens ({} image rows spliced)",
        ids.len(),
        img_row
    );

    eprintln!("[6/6] greedy-decoding {max_new} tokens");
    let eos = tok.special_token_id("<end_of_turn>");
    let mut gen: Vec<u32> = Vec::new();
    let mut next = gpu.argmax_f32(&state.logits, text_cfg.vocab_size)?;
    for _ in 0..max_new {
        if Some(next) == eos {
            break;
        }
        gen.push(next);
        next = g3::forward_step_greedy(&mut gpu, &w.text, text_cfg, &mut state, next)?;
    }

    println!(
        "\n=== gemma3-vl continuation ===\n{}\n==============================",
        tok.decode(&gen)
    );
    Ok(())
}
