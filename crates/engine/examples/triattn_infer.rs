//! triattn_infer — minimal command-line TriAttention / CASK inference.
//!
//! Parallel to the daemon but with TriAttention eviction baked in. Accepts
//! a prompt from stdin (if present) or `--prompt`, runs greedy decode on
//! Qwen 3.5 with periodic eviction, prints the generated continuation to
//! stdout. Use this binary (or copy its `prefill + decode` loop into your
//! own driver) when you want long-context behavior without scaling the KV
//! cache with the prompt.
//!
//! Usage:
//!   triattn_infer --model PATH --sidecar PATH [--kv-mode asym3|asym4|q8]
//!                 [--budget 512] [--beta 128] [--max-tokens 256]
//!                 [--cask] [--core-frac 0.5] [--fold-m 2]
//!                 [--prompt "..."]  (or pipe the prompt on stdin)
//!
//! `--cask` enables the CASK core-aware m-folding policy on top of
//! TriAttention scoring (arXiv:2604.10900). Only applies to --kv-mode q8
//! in v1; other modes silently fall back to plain TriAttention.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::cask::CaskCtx;
    use engine::hfq::HfqFile;
    use engine::llama::{self, KvCache};
    use engine::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch};
    use engine::tokenizer::Tokenizer;
    use engine::triattn::{EvictionCtx, TriAttnCenters};
    use rdna_compute::Gpu;
    use std::io::{Read, Write};
    use std::path::Path;

    enum Policy { Plain(EvictionCtx), Cask(CaskCtx) }
    impl Policy {
        fn maybe_evict(&self, gpu: &mut Gpu, kv: &mut KvCache, physical: usize)
            -> hip_bridge::HipResult<Option<engine::triattn::EvictionResult>>
        {
            match self {
                Policy::Plain(c) => c.maybe_evict(gpu, kv, physical),
                Policy::Cask(c) => c.maybe_evict(gpu, kv, physical),
            }
        }
        fn eviction_count(&self) -> usize {
            match self {
                Policy::Plain(c) => c.eviction_count.get(),
                Policy::Cask(c) => c.eviction_count(),
            }
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut sidecar: Option<String> = None;
    let mut kv_mode = String::from("asym3");
    let mut budget: usize = 512;
    let mut beta: usize = 128;
    let mut max_tokens: usize = 256;
    let mut prompt_arg: Option<String> = None;
    let mut use_cask = false;
    let mut core_frac: f32 = 0.5;
    let mut fold_m: usize = 2;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = Some(args[i + 1].clone()); i += 2; }
            "--sidecar" => { sidecar = Some(args[i + 1].clone()); i += 2; }
            "--kv-mode" => { kv_mode = args[i + 1].clone(); i += 2; }
            "--budget" => { budget = args[i + 1].parse().unwrap(); i += 2; }
            "--beta" => { beta = args[i + 1].parse().unwrap(); i += 2; }
            "--max-tokens" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--prompt" => { prompt_arg = Some(args[i + 1].clone()); i += 2; }
            "--cask" => { use_cask = true; i += 1; }
            "--core-frac" => { core_frac = args[i + 1].parse().unwrap(); i += 2; }
            "--fold-m" => { fold_m = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model_path = model.expect("--model required");
    let sidecar_path = sidecar.expect("--sidecar required");

    // Prompt: --prompt beats stdin. Empty stdin is fine — the model gets
    // an empty user turn and continues from the ChatML assistant header.
    let prompt = match prompt_arg {
        Some(p) => p,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).ok();
            buf.trim().to_string()
        }
    };

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let centers = TriAttnCenters::load(Path::new(&sidecar_path)).expect("load sidecar");

    let fa_layer_ids: Vec<usize> = config.layer_types.iter().enumerate()
        .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
        .collect();
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;

    let kv_seq = budget + beta + 8;

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");

    let mut kv = match kv_mode.as_str() {
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        "asym3" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        other => { eprintln!("unsupported kv mode: {other}"); std::process::exit(1); }
    };
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let base = EvictionCtx::new(
        &mut gpu, &centers, fa_layer_ids.clone(),
        budget, beta,
        config.n_heads, config.n_kv_heads, config.head_dim,
        n_rot, config.rope_theta, kv_seq,
    ).expect("build EvictionCtx");
    let ctx = if use_cask {
        Policy::Cask(CaskCtx::new(base, core_frac, fold_m))
    } else {
        Policy::Plain(base)
    };

    // ChatML wrap — matches daemon's generate() framing so outputs are
    // consistent with the normal serve path.
    let im_start = tok.encode("<|im_start|>");
    let im_end = tok.encode("<|im_end|>");
    let nl = tok.encode("\n");
    let user_tok = tok.encode("user");
    let asst_tok = tok.encode("assistant");
    let body = tok.encode(&prompt);
    let think = tok.encode("<think>");

    let mut toks: Vec<u32> = Vec::new();
    toks.extend_from_slice(&im_start);
    toks.extend_from_slice(&user_tok);
    toks.extend_from_slice(&nl);
    toks.extend_from_slice(&body);
    toks.extend_from_slice(&im_end);
    toks.extend_from_slice(&nl);
    toks.extend_from_slice(&im_start);
    toks.extend_from_slice(&asst_tok);
    toks.extend_from_slice(&nl);
    toks.extend_from_slice(&think);
    toks.extend_from_slice(&nl);

    let policy_tag = if use_cask {
        format!("CASK(core_frac={:.2}, m={})", core_frac, fold_m)
    } else {
        "TriAttention".to_string()
    };
    eprintln!(
        "triattn_infer: {} prompt tokens, budget={} beta={} kv_mode={} policy={} max_tokens={}",
        toks.len(), budget, beta, kv_mode, policy_tag, max_tokens,
    );

    // Prefill with in-loop eviction so long prompts don't exceed the cache.
    let mut physical = 0usize;
    let t0 = std::time::Instant::now();
    for t in &toks {
        qwen35::forward_scratch(&mut gpu, &weights, &config, *t, physical, &mut kv, &mut dn, &scratch).unwrap();
        physical += 1;
        if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
            physical = ev.new_physical;
        }
        #[allow(unused)]
        let _ = &ctx;
    }
    let t_prefill = t0.elapsed();
    eprintln!(
        "[triattn] prefill: {:.2}s  physical={} compact_offset={} evictions={}",
        t_prefill.as_secs_f64(), physical, kv.compact_offset, ctx.eviction_count(),
    );

    // Decode greedy.
    let t1 = std::time::Instant::now();
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next = llama::argmax(&logits);
    let mut emitted: Vec<u32> = vec![next];
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let piece = tok.decode(&[next]);
    let _ = out.write_all(piece.as_bytes());
    let _ = out.flush();

    for _ in 0..max_tokens {
        qwen35::forward_scratch(&mut gpu, &weights, &config, next, physical, &mut kv, &mut dn, &scratch).unwrap();
        physical += 1;
        if let Some(ev) = ctx.maybe_evict(&mut gpu, &mut kv, physical).unwrap() {
            physical = ev.new_physical;
        }
        #[allow(unused)]
        let _ = &ctx;
        logits = gpu.download_f32(&scratch.logits).unwrap();
        next = llama::argmax(&logits);
        if next == config.eos_token { break; }
        emitted.push(next);
        let piece = tok.decode(&[next]);
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
    }
    let t_decode = t1.elapsed();
    let _ = out.write_all(b"\n");

    let decode_tps = emitted.len() as f64 / t_decode.as_secs_f64();
    eprintln!(
        "\n[triattn] decode: {} tokens in {:.2}s = {:.1} tok/s  total_evictions={}  final_compact_offset={}",
        emitted.len(), t_decode.as_secs_f64(), decode_tps,
        ctx.eviction_count(), kv.compact_offset,
    );
}
