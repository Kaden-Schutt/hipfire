// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! TriAttention end-to-end decode sanity check.
//!
//! 1. Load Qwen3.5-9B with Q8 KV.
//! 2. Prefill a prompt (64 tokens) via single-token forward_scratch.
//! 3. Run 30 AR tokens without eviction → reference output.
//! 4. Reset, prefill the same prompt.
//! 5. Score + top-B compact every FA layer down to budget=32
//!    (keeping half the prefill). Set `kv_cache.compact_offset`.
//! 6. Run 30 AR tokens with the compacted cache.
//! 7. Report both outputs so we can eyeball coherence.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch};
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::triattn::{self, TriAttnCenters};
    use rdna_compute::{DType, Gpu};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: triattn_decode_demo <model.mq4> <sidecar.triattn.bin> [budget=32] [prefill=64] [gen=30] [kv_mode=q8|asym3]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let sidecar_path = &args[2];
    let budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let prefill_len: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let gen_len: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);
    let kv_mode: String = args.get(6).cloned().unwrap_or_else(|| "q8".into());

    let prompt = "James Madison wrote Federalist No. 10 arguing that a large republic would curb the effects of factions better than a small one. Drawing on his reading of Montesquieu, Hume, and his own experiences in the Virginia legislature, Madison argued that factions were inevitable in a free society and that the only way to control them was to extend the sphere of the republic. By enlarging the territory and population, the influence of any single faction would be diluted because the larger the society, the more varied the interests and passions of its members. This diversity, Madison reasoned, made it harder for a single faction to gain a majority large enough to tyrannize the minority. The paper was written in November 1787 and published under the pseudonym Publius. It stands today as one of the most important works of political philosophy in American history. The core insight";

    // ── Model + centers ────────────────────────────────────────────────
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let centers = TriAttnCenters::load(Path::new(sidecar_path)).expect("load sidecar");
    assert_eq!(centers.n_heads, config.n_heads);
    assert_eq!(centers.head_dim, config.head_dim);
    assert_eq!(centers.n_layers, config.n_layers);

    let n_bands = config.head_dim / 2;
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;

    let fa_layer_ids: Vec<usize> = config.layer_types.iter().enumerate()
        .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
        .collect();
    eprintln!("FA layers: {:?}", fa_layer_ids);
    eprintln!("budget={budget} prefill={prefill_len} gen={gen_len}");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");

    let kv_seq = 512usize;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    // Pack per-layer center slices into one big GPU tensor.
    // Layout: [fa_layer × n_heads × n_bands × 3]
    let mut centers_flat = Vec::with_capacity(fa_layer_ids.len() * config.n_heads * n_bands * 3);
    for &layer_idx in &fa_layer_ids {
        for h in 0..config.n_heads {
            for f in 0..n_bands {
                let c = centers.get(layer_idx, h, f);
                centers_flat.push(c.eq_re);
                centers_flat.push(c.eq_im);
                centers_flat.push(c.e_abs_q);
            }
        }
    }
    let centers_dev = gpu.upload_f32(&centers_flat, &[centers_flat.len()]).unwrap();
    let centers_per_layer = config.n_heads * n_bands * 3;

    // ── Tokenize ───────────────────────────────────────────────────────
    let prompt_tokens = tok.encode(prompt);
    let prompt_len = prompt_tokens.len().min(prefill_len);
    eprintln!("prompt: {} tokens (using first {})", prompt_tokens.len(), prompt_len);

    // ── Run 1: reference (no eviction) ─────────────────────────────────
    let (ref_tokens, ref_text) = {
        let mut kv = match kv_mode.as_str() {
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
            ).unwrap(),
            _ => KvCache::new_gpu_q8(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
            ).unwrap(),
        };
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();

        for (p, t) in prompt_tokens.iter().take(prompt_len).enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch).unwrap();
        }
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next = llama::argmax(&logits);
        let mut emitted = Vec::new();
        emitted.push(next);
        for step in 0..gen_len {
            let pos = prompt_len + step;
            qwen35::forward_scratch(&mut gpu, &weights, &config, next, pos, &mut kv, &mut dn, &scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            emitted.push(next);
        }
        let text = tok.decode(&emitted);
        (emitted, text)
    };

    // ── Run 2: with TriAttention eviction after prefill ────────────────
    let (evict_tokens, evict_text, retain_sample) = {
        let mut kv = match kv_mode.as_str() {
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
            ).unwrap(),
            _ => KvCache::new_gpu_q8(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
            ).unwrap(),
        };
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();

        for (p, t) in prompt_tokens.iter().take(prompt_len).enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch).unwrap();
        }

        // Last-query position for scoring = absolute pos of the last token.
        let p_q = prompt_len as f32;
        let blocks_per_head = config.head_dim / 32;
        let k_bytes_per_pos = match kv_mode.as_str() {
            // asym3: [4B cnorm][(head_dim*3)/8 B packed 3-bit per head] × n_kv_heads
            "asym3" => config.n_kv_heads * (4 + (config.head_dim * 3) / 8),
            // Q8_0: 34-byte blocks × blocks_per_head × n_kv_heads
            _ => config.n_kv_heads * blocks_per_head * 34,
        };
        // V is Q8_0 in both modes (asym3 leaves V unrotated).
        let v_bytes_per_pos = config.n_kv_heads * blocks_per_head * 34;
        let k_compact_floats = (budget * k_bytes_per_pos + 3) / 4;
        let v_compact_floats = (budget * v_bytes_per_pos + 3) / 4;

        // One temp compact buffer for the gather per stream (reused across layers).
        let k_compact = gpu.zeros(&[k_compact_floats], DType::F32).unwrap();
        let v_compact = gpu.zeros(&[v_compact_floats], DType::F32).unwrap();
        let scores_buf = gpu.alloc_tensor(&[config.n_heads * prompt_len], DType::F32).unwrap();
        let retain_dev = gpu.alloc_tensor(&[budget], DType::F32).unwrap();

        let mut retain_sample: Option<Vec<u32>> = None;
        for (fa_i, &layer_idx) in fa_layer_ids.iter().enumerate() {
            // Score
            let offset = fa_i * centers_per_layer;
            let centers_layer = centers_dev.sub_offset(offset, centers_per_layer);
            match kv_mode.as_str() {
                "asym3" => gpu.triattn_score_asym3(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().unwrap(),
                    kv.givens_sin.as_ref().unwrap(),
                    &scores_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim,
                    n_rot, config.rope_theta, p_q, prompt_len,
                ).unwrap(),
                _ => gpu.triattn_score_q8(
                    &kv.k_gpu[layer_idx], &centers_layer, &scores_buf,
                    config.n_heads, config.n_kv_heads, config.head_dim,
                    n_rot, config.rope_theta, p_q, prompt_len,
                ).unwrap(),
            }
            gpu.hip.device_synchronize().unwrap();
            let scores_host = gpu.download_f32(&scores_buf).unwrap();
            let retain = triattn::compute_retain_indices(
                &scores_host[..config.n_heads * prompt_len],
                config.n_heads, prompt_len, budget,
            );
            if retain_sample.is_none() && fa_i == 0 { retain_sample = Some(retain.clone()); }

            let retain_bytes: Vec<u8> = retain.iter().flat_map(|&x| (x as i32).to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&retain_dev.buf, &retain_bytes).unwrap();

            // Compact K and V (different bytes_per_pos in asym3 since K
            // packs 3-bit + cnorm while V is still Q8_0-shaped).
            gpu.kv_compact_gather(&kv.k_gpu[layer_idx], &k_compact, &retain_dev, k_bytes_per_pos, budget).unwrap();
            gpu.kv_compact_gather(&kv.v_gpu[layer_idx], &v_compact, &retain_dev, v_bytes_per_pos, budget).unwrap();
            gpu.hip.device_synchronize().unwrap();

            gpu.hip.memcpy_dtod_at(&kv.k_gpu[layer_idx].buf, 0, &k_compact.buf, 0, budget * k_bytes_per_pos).unwrap();
            gpu.hip.memcpy_dtod_at(&kv.v_gpu[layer_idx].buf, 0, &v_compact.buf, 0, budget * v_bytes_per_pos).unwrap();
        }
        kv.compact_offset = prompt_len - budget;
        eprintln!("compact_offset set to {}", kv.compact_offset);

        // Continue AR decode. Physical cache index starts at `budget`.
        // IMPORTANT: the logits from the prefill's last token are still
        // valid (they were computed before compaction). Sample from them.
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next = llama::argmax(&logits);
        let mut emitted = Vec::new();
        emitted.push(next);
        for step in 0..gen_len {
            let pos = budget + step;
            qwen35::forward_scratch(&mut gpu, &weights, &config, next, pos, &mut kv, &mut dn, &scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            emitted.push(next);
        }
        let text = tok.decode(&emitted);
        (emitted, text, retain_sample.unwrap())
    };

    // ── Report ─────────────────────────────────────────────────────────
    eprintln!("\n=== REFERENCE (no eviction) ===");
    eprintln!("{}", ref_text);
    eprintln!("tokens: {:?}", ref_tokens);
    eprintln!("\n=== WITH TRIATTN EVICTION (budget {budget}/{prompt_len}) ===");
    eprintln!("layer0 retained positions (first 16 of {}): {:?}",
        retain_sample.len(), &retain_sample[..retain_sample.len().min(16)]);
    eprintln!("{}", evict_text);
    eprintln!("tokens: {:?}", evict_tokens);

    let first_div = ref_tokens.iter().zip(evict_tokens.iter())
        .position(|(a, b)| a != b);
    match first_div {
        Some(i) => eprintln!("\nfirst divergent token at step {i} (of {})", ref_tokens.len()),
        None => eprintln!("\nno divergence — outputs identical"),
    }
}
