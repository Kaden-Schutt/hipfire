//! dflash_smoke: load a converted DFlash draft .hfq file and run a single
//! forward pass with random inputs. Verifies weights load, all kernels
//! compile + launch, and the output contains finite values.
//!
//! Usage: dflash_smoke <draft.hfq> [--block B] [--ctx L]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
    use engine::hfq::HfqFile;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dflash_smoke <draft.hfq> [--block B] [--ctx L]");
        std::process::exit(1);
    }
    let path = &args[1];
    let mut block_size: usize = 16;
    let mut ctx_len: usize = 32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--block" => {
                block_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                ctx_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    eprintln!("=== dflash_smoke ===");
    eprintln!("draft: {path}");
    eprintln!("block_size: {block_size}  ctx_len: {ctx_len}");

    let hfq = HfqFile::open(Path::new(path)).expect("open draft .hfq");
    let cfg = DflashConfig::from_hfq(&hfq).expect("parse DflashConfig");
    eprintln!(
        "config: layers={} hidden={} heads={} kv_heads={} head_dim={} block={} mask={} target_layers={:?} (of {})",
        cfg.n_layers,
        cfg.hidden,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.block_size,
        cfg.mask_token_id,
        cfg.target_layer_ids,
        cfg.num_target_layers,
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    let t0 = Instant::now();
    let weights = DflashWeights::load(&mut gpu, &hfq, &cfg).expect("load dflash weights");
    eprintln!("weights loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let mut scratch = DflashScratch::new(&mut gpu, &cfg, block_size, ctx_len).expect("alloc scratch");
    eprintln!("scratch allocated in {:.3}s", t1.elapsed().as_secs_f64());

    // Synthesize a deterministic input: seeded noise for noise_embedding and
    // target_hidden. Determinism so the smoke test's finite-value assertion
    // is reproducible across runs.
    let mut rng_state: u64 = 0xD1FEu64;
    let mut rng = || -> f32 {
        // xorshift32-ish.
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let v = (rng_state as u32) as f32 / (u32::MAX as f32);
        (v - 0.5) * 0.04
    };

    let noise_embedding: Vec<f32> = (0..block_size * cfg.hidden).map(|_| rng()).collect();
    let target_hidden: Vec<f32> = (0..ctx_len * cfg.num_extract() * cfg.hidden)
        .map(|_| rng())
        .collect();

    // Positions: assume current "start" is ctx_len. Q positions are
    // [ctx_len, ctx_len+block_size). K positions = [0, ctx_len+block_size).
    let positions_q: Vec<i32> = (ctx_len as i32..ctx_len as i32 + block_size as i32).collect();
    let positions_k: Vec<i32> = (0..(ctx_len + block_size) as i32).collect();

    let t2 = Instant::now();
    dflash::draft_forward(
        &mut gpu,
        &weights,
        &cfg,
        &noise_embedding,
        &target_hidden,
        &positions_q,
        &positions_k,
        block_size,
        ctx_len,
        &mut scratch,
    )
    .expect("draft_forward");
    // Ensure all kernels complete before we download.
    gpu.hip.device_synchronize().expect("sync");
    let elapsed_ms = t2.elapsed().as_secs_f64() * 1000.0;
    eprintln!("forward: {elapsed_ms:.2} ms");

    let out = gpu.download_f32(&scratch.x).expect("download x");
    let rows = block_size;
    let cols = cfg.hidden;
    eprintln!("output shape: [{rows}, {cols}]  (len={})", out.len());
    let finite_count = out.iter().take(1024).filter(|v| v.is_finite()).count();
    eprintln!("first-1024 finite: {finite_count}/1024");
    let (mn, mx) = out.iter().take(1024).fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
        (mn.min(v), mx.max(v))
    });
    eprintln!("first-1024 min/max: {mn:.6e} / {mx:.6e}");

    if finite_count < 1024 {
        eprintln!("FAIL: non-finite values in output");
        std::process::exit(2);
    }
    eprintln!("OK");
}
