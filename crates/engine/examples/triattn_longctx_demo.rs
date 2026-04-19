//! TriAttention long-context decode with *constant* KV memory footprint.
//!
//! Allocates `KvCache` at `max_seq = budget + beta + 2` — the minimum that
//! satisfies the eviction threshold. Runs a prefill longer than the cache
//! allocation, then generates `gen` tokens. Every token step (prefill and
//! decode) calls `EvictionCtx::maybe_evict`; when physical cache fills to
//! `budget + beta`, the context is compacted back down to `budget`.
//!
//! This demonstrates the headline TriAttention benefit: process arbitrary
//! prompt/generation lengths with a **fixed-size** KV cache that doesn't
//! scale with sequence length.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::llama::{self, KvCache};
    use engine::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch};
    use engine::tokenizer::Tokenizer;
    use engine::triattn::{EvictionCtx, TriAttnCenters};
    use rdna_compute::Gpu;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: triattn_longctx_demo <model> <sidecar> [budget=64] [beta=16] [gen=64] [kv_mode=asym3]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let sidecar_path = &args[2];
    let budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let beta: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let gen_len: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
    let kv_mode: String = args.get(6).cloned().unwrap_or_else(|| "asym3".into());

    // Stitch a long prompt. ~175 tokens; comfortably exceeds budget=64.
    let prompt = "James Madison wrote Federalist No. 10 arguing that a large republic would curb the effects of factions better than a small one. Drawing on his reading of Montesquieu, Hume, and his own experiences in the Virginia legislature, Madison argued that factions were inevitable in a free society and that the only way to control them was to extend the sphere of the republic. By enlarging the territory and population, the influence of any single faction would be diluted. The paper was written in November 1787 and published under the pseudonym Publius. Its core insight, often cited today, is";

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let centers = TriAttnCenters::load(Path::new(sidecar_path)).expect("load sidecar");
    let fa_layer_ids: Vec<usize> = config.layer_types.iter().enumerate()
        .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
        .collect();
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");

    let prompt_tokens = tok.encode(prompt);
    let prompt_len = prompt_tokens.len();

    // Tight allocation: just enough to hold one eviction's worth of growth.
    let kv_seq_tight = budget + beta + 2;
    eprintln!(
        "prompt={prompt_len}  gen={gen_len}  budget={budget}  beta={beta}  kv_mode={kv_mode}",
    );
    eprintln!("kv cache allocated at max_seq={kv_seq_tight} (prompt is {:.1}× larger)", prompt_len as f32 / kv_seq_tight as f32);

    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
    let alloc_kv = |gpu: &mut Gpu, seq: usize| match kv_mode.as_str() {
        "asym3" => KvCache::new_gpu_asym3(gpu, config.n_layers, config.n_kv_heads, config.head_dim, seq).unwrap(),
        _ => KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, seq).unwrap(),
    };

    // ── Reference (no eviction) — needs the full cache size ────────────
    let (ref_tokens, ref_text) = {
        let mut kv = alloc_kv(&mut gpu, prompt_len + gen_len + 32);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        for (p, t) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch).unwrap();
        }
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next = llama::argmax(&logits);
        let mut emitted = vec![next];
        for step in 0..gen_len {
            qwen35::forward_scratch(&mut gpu, &weights, &config, next, prompt_len + step, &mut kv, &mut dn, &scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            emitted.push(next);
        }
        (emitted.clone(), tok.decode(&emitted))
    };

    // ── Tight-cache path: maybe_evict after every single forward ───────
    let (evict_tokens, evict_text, evictions) = {
        let mut kv = alloc_kv(&mut gpu, kv_seq_tight);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        let ctx = EvictionCtx::new(
            &mut gpu, &centers, fa_layer_ids.clone(),
            budget, beta,
            config.n_heads, config.n_kv_heads, config.head_dim,
            n_rot, config.rope_theta, kv_seq_tight,
        ).unwrap();

        let mut physical = 0usize;
        // Prefill with in-loop eviction.
        for t in prompt_tokens.iter() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, physical, &mut kv, &mut dn, &scratch).unwrap();
            physical += 1;
            if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
                physical = ev.new_physical;
            }
        }
        eprintln!("after prefill: physical={physical}  compact_offset={}", kv.compact_offset);

        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next = llama::argmax(&logits);
        let mut emitted = vec![next];
        for _step in 0..gen_len {
            qwen35::forward_scratch(&mut gpu, &weights, &config, next, physical, &mut kv, &mut dn, &scratch).unwrap();
            physical += 1;
            if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
                physical = ev.new_physical;
            }
            logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            emitted.push(next);
        }
        let ev = ctx.eviction_count.get();
        (emitted.clone(), tok.decode(&emitted), ev)
    };

    // ── Memory footprint comparison ────────────────────────────────────
    let blocks_per_head = config.head_dim / 32;
    let v_bpp = config.n_kv_heads * blocks_per_head * 34;
    let k_bpp = if kv_mode == "asym3" {
        config.n_kv_heads * (4 + (config.head_dim * 3) / 8)
    } else { v_bpp };
    let per_pos_bytes = k_bpp + v_bpp;
    let ref_kv_bytes = (prompt_len + gen_len + 32) * per_pos_bytes * fa_layer_ids.len();
    let tight_kv_bytes = kv_seq_tight * per_pos_bytes * fa_layer_ids.len();
    eprintln!(
        "\nVRAM footprint (FA layers × positions × bytes/pos):\n  reference: {:.1} MiB ({} pos × {} B/pos × {} layers)\n  tight    : {:.1} MiB ({} pos × {} B/pos × {} layers) — {:.1}× smaller",
        ref_kv_bytes as f32 / (1024.0 * 1024.0),
        prompt_len + gen_len + 32, per_pos_bytes, fa_layer_ids.len(),
        tight_kv_bytes as f32 / (1024.0 * 1024.0),
        kv_seq_tight, per_pos_bytes, fa_layer_ids.len(),
        ref_kv_bytes as f32 / tight_kv_bytes as f32,
    );

    eprintln!("\n=== REFERENCE ({} tokens, no eviction) ===", ref_tokens.len());
    eprintln!("{}", ref_text);
    eprintln!("\n=== TIGHT CACHE + PERIODIC EVICTION ({} evictions, budget={} beta={}) ===", evictions, budget, beta);
    eprintln!("{}", evict_text);

    let div = ref_tokens.iter().zip(evict_tokens.iter()).position(|(a, b)| a != b);
    match div {
        Some(i) => eprintln!("\nfirst divergence at step {i} of {}", ref_tokens.len()),
        None => eprintln!("\nno divergence"),
    }
}
