//! Periodic TriAttention eviction during long decode.
//!
//! Prefills a prompt, then generates `gen` tokens while `EvictionCtx`
//! fires every time physical cache size hits `budget + beta`. Reports
//! eviction cadence and both the reference (no-eviction) and evicted
//! output text so coherence can be eyeballed across multiple eviction
//! cycles.

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
        eprintln!("Usage: triattn_periodic_demo <model> <sidecar> [budget=32] [beta=16] [gen=128] [kv_mode=asym3]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let sidecar_path = &args[2];
    let budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let beta: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let gen_len: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);
    let kv_mode: String = args.get(6).cloned().unwrap_or_else(|| "asym3".into());

    let prompt = "James Madison wrote Federalist No. 10 to argue that a large republic would check the dangers of majority factions. The paper is famous for its insight that";

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
    let kv_seq = (prompt_len + gen_len + 32).max(budget + beta + 16).max(256);
    eprintln!("prompt={prompt_len} gen={gen_len} budget={budget} beta={beta} kv_mode={kv_mode} kv_seq={kv_seq}");
    eprintln!("FA layers: {}", fa_layer_ids.len());
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    let alloc_kv = |gpu: &mut Gpu| match kv_mode.as_str() {
        "asym3" => KvCache::new_gpu_asym3(gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        _ => KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
    };

    // ── Reference (no eviction) ───────────────────────────────────────
    let (ref_tokens, ref_text) = {
        let mut kv = alloc_kv(&mut gpu);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        for (p, t) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch).unwrap();
        }
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next = llama::argmax(&logits);
        let mut emitted = vec![next];
        for step in 0..gen_len {
            let pos = prompt_len + step;
            qwen35::forward_scratch(&mut gpu, &weights, &config, next, pos, &mut kv, &mut dn, &scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            emitted.push(next);
        }
        (emitted.clone(), tok.decode(&emitted))
    };

    // ── Periodic TriAttention eviction ────────────────────────────────
    let (evict_tokens, evict_text, evictions) = {
        let mut kv = alloc_kv(&mut gpu);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        let ctx = EvictionCtx::new(
            &mut gpu, &centers, fa_layer_ids.clone(),
            budget, beta,
            config.n_heads, config.n_kv_heads, config.head_dim,
            n_rot, config.rope_theta, kv_seq,
        ).unwrap();

        for (p, t) in prompt_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch).unwrap();
        }
        let mut physical = prompt_len;
        if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
            physical = ev.new_physical;
        }

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

    eprintln!("\n=== REFERENCE ({} tokens generated, no eviction) ===", ref_tokens.len());
    eprintln!("{}", ref_text);
    eprintln!("\n=== WITH PERIODIC EVICTION (fired {} times, budget={} beta={}) ===", evictions, budget, beta);
    eprintln!("{}", evict_text);

    let div = ref_tokens.iter().zip(evict_tokens.iter()).position(|(a, b)| a != b);
    match div {
        Some(i) => eprintln!("\nfirst divergence at token {i} of {}", ref_tokens.len()),
        None => eprintln!("\nno divergence — outputs identical"),
    }
}
