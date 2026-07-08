// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen3_dspark_parity: GPU-vs-CPU numeric parity for the Qwen3-8B DSpark
//! drafter forward.  Exercises both ctx_len=1 (Task-9 baseline) and
//! ctx_len=3 (Stage-1 multi-slot gate).
//!
//! ## What is validated
//!
//! **ctx_len=1 (backward-compat baseline)**
//! Checks (a)–(d) against the CPU reference produced by
//! `/home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py`:
//!   (a) `main_x` = `hidden_norm(fc(main_hidden))` — cosine ≥ 0.999
//!   (b) `x_head_out` = post-final-norm block hidden `[block, dim]` — cosine ≥ 0.999
//!   (c) markov greedy token sequence — token-identical to CPU
//!   (d) confidence logits (pre-sigmoid) — cosine ≥ 0.999
//!
//! **ctx_len=3 (multi-slot Stage-1 gate)**
//! Checks (e)–(h) against the CPU reference produced by
//! `/home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py --ctx-len 3`:
//!   (e) `main_x` (3 rows) — cosine ≥ 0.999
//!   (f) `x_head_out` = post-final-norm block hidden `[block, dim]` — cosine ≥ 0.999
//!   (g) markov greedy token sequence — token-identical to CPU
//!   (h) confidence logits — cosine ≥ 0.999
//!
//! ## Inputs (ctx_len=1)
//!
//! Fixed synthetic `main_hidden[5*4096]`:
//!   `main_hidden[i] = sin(i * 0.013) * 0.5`   (same as deepseek4 parity harness)
//! Fixed `seed = 12345`, `seed_pos = 42`, `block = 7`.
//!
//! ## Inputs (ctx_len=3)
//!
//! Fixed synthetic `main_hidden[3 * 5*4096]`:
//!   `main_hidden[i] = sin(i * 0.013) * 0.5`
//! ctx_positions = [40, 41, 42] (3 accepted positions before the anchor).
//! block_positions = [43, 44, 45, 46, 47, 48, 49] (anchor_pos=43, block=7).
//! seed_tok = 12345.
//!
//! The CPU reference files are expected in
//!   `/home/bjoern/dspark-work/qwen3_parity_refs/`   (ctx_len=1)
//!   `/home/bjoern/dspark-work/qwen3_parity_refs_ctx3/`  (ctx_len=3)
//!
//! ## CPU reference
//!
//! Run BEFORE this binary:
//! ```
//! cd /home/bjoern/hipfire
//! nix develop --command bash -c '
//!   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(find /nix/store -maxdepth 1 -path "*gcc-15*-lib" | head -1)/lib:/nix/store/6v5hbaxvndmaf21rfyryxpn1xjkljrid-zlib-1.3.2/lib"
//!   export PYTHONPATH="/home/bjoern/dspark-work/DeepSpec:$PYTHONPATH"
//!   /home/bjoern/hipfire/.venv/bin/python3 /home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py
//!   /home/bjoern/hipfire/.venv/bin/python3 /home/bjoern/dspark-work/qwen3_dspark_cpu_ref.py \
//!     /home/bjoern/dspark-work/qwen3/ckpt \
//!     /home/bjoern/dspark-work/qwen3_parity_refs_ctx3 \
//!     42 12345 3
//! '
//! ```
//!
//! ## Usage
//! ```
//! source scripts/gpu-lock.sh && gpu_acquire dspark-qwen3
//! cargo build --release -p hipfire-arch-llama --example qwen3_dspark_parity
//! ./target/release/examples/qwen3_dspark_parity [path-to-qwen3-8b-dspark.hfq] [refs-dir-ctx1] [refs-dir-ctx3]
//! gpu_release
//! ```

use hipfire_arch_llama::dspark_body::{
    dspark_qwen3_block_forward, load_qwen3_dspark, Qwen3DsparkScratch,
};
use hipfire_runtime::dspark_core::{
    main_proj_ingest, main_proj_ingest_batched, noise_block_ids, run_heads,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

/// Fixed test parameters — must match qwen3_dspark_cpu_ref.py defaults.
const SEED_TOK: u32 = 12345;
const SEED_POS: usize = 42;
const BLOCK: usize = 7;
const N_TARGETS: usize = 5;

/// ctx_len=3 test parameters.
const CTX_LEN3: usize = 3;
/// ctx absolute positions for ctx_len=3: [SEED_POS-2, SEED_POS-1, SEED_POS]
const CTX_POSITIONS3: [usize; CTX_LEN3] = [40, 41, 42];
/// block (anchor) positions for ctx_len=3: anchor_pos=43, block=7
/// create_position_ids(43, 7) = [43, 44, 45, 46, 47, 48, 49]
const BLOCK_POSITIONS3: [usize; BLOCK] = [43, 44, 45, 46, 47, 48, 49];

fn main() -> Result<(), String> {
    let hfq_path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/qwen3-8b-dspark.mq4",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let refs_dir1 = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/bjoern/dspark-work/qwen3_parity_refs".into());
    let refs_dir3 = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/home/bjoern/dspark-work/qwen3_parity_refs_ctx3".into());

    eprintln!("opening {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path)).map_err(|e| format!("open: {e:?}"))?;
    hfq.drop_mmap();

    eprintln!("initialising GPU");
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU ready (arch={})", gpu.arch_caps.arch());

    // ── Load sidecar ──────────────────────────────────────────────────────────
    let (dspark_weights, assets) =
        load_qwen3_dspark(&hfq, &mut gpu)?.ok_or("load_qwen3_dspark: no dspark_* in metadata")?;

    let cfg = &dspark_weights.cfg;
    let dim = assets.config.dim;
    let vocab = assets.weights.output.m;
    eprintln!(
        "loaded: dim={dim} vocab={vocab} block_size={} markov_rank={} enable_confidence={}",
        cfg.block_size, cfg.markov_rank, cfg.enable_confidence
    );

    // Build lm_head_f16 (same for both parity cases).
    let mut lm_head_f16 = assets.weights.output.buf.shallow_clone();
    lm_head_f16.dtype = DType::F16;
    lm_head_f16.shape = vec![vocab];

    let stage_norm_ref = &assets.weights.output_norm;

    // ═══════════════════════════════════════════════════════════════════════════
    // ctx_len=1 (Task-9 baseline backward-compat check)
    // ═══════════════════════════════════════════════════════════════════════════
    eprintln!("\n=== ctx_len=1 (Task-9 baseline) ===");

    let refs1 = Path::new(&refs_dir1);
    let cpu_main_hidden1 = load_f32bin(refs1.join("main_hidden.f32bin"))?;
    let cpu_main_x1 = load_f32bin(refs1.join("main_x.f32bin"))?;
    let cpu_x_head1 = load_f32bin(refs1.join("x_head_out.f32bin"))?;
    let cpu_markov_i32_1 = load_i32bin(refs1.join("markov_tokens.i32bin"))?;
    let cpu_confidence1 = load_f32bin(refs1.join("confidence_logits.f32bin"))?;
    let cpu_markov1: Vec<u32> = cpu_markov_i32_1.iter().map(|&t| t as u32).collect();

    let concat_w = N_TARGETS * dim;
    verify_len("ctx1 main_hidden", cpu_main_hidden1.len(), concat_w)?;
    verify_len("ctx1 main_x", cpu_main_x1.len(), dim)?;
    verify_len("ctx1 x_head", cpu_x_head1.len(), BLOCK * dim)?;
    verify_len("ctx1 markov", cpu_markov1.len(), BLOCK)?;
    verify_len("ctx1 confidence", cpu_confidence1.len(), BLOCK)?;

    let main_hidden_dev1 = upload_f32(&mut gpu, &cpu_main_hidden1)?;

    // Check (a): main_proj_ingest
    let main_x_dev1 = gpu
        .alloc_tensor(&[dim], DType::F32)
        .map_err(|e| format!("alloc main_x ctx1: {e:?}"))?;
    main_proj_ingest(&mut gpu, &dspark_weights, &main_hidden_dev1, &main_x_dev1)?;
    let gpu_main_x1 = gpu
        .download_f32(&main_x_dev1)
        .map_err(|e| format!("d2h main_x ctx1: {e:?}"))?;
    let check_a = parity_stats("(a) ctx1 main_x", &gpu_main_x1, &cpu_main_x1, 0.999, None);

    // Check (b): x_head_out from dspark_qwen3_block_forward (ctx_len=1)
    let scratch1 = Qwen3DsparkScratch::new(&mut gpu, &assets.config, BLOCK, 1)
        .map_err(|e| format!("Qwen3DsparkScratch ctx1: {e}"))?;
    let x_head_dev1 = gpu
        .alloc_tensor(&[BLOCK, dim], DType::F32)
        .map_err(|e| format!("alloc x_head ctx1: {e:?}"))?;
    let block_ids1 = noise_block_ids(cfg, SEED_TOK);
    let block_positions1: Vec<usize> = (0..BLOCK).map(|i| SEED_POS + i).collect();
    dspark_qwen3_block_forward(
        &mut gpu,
        &assets.weights,
        &assets.config,
        &main_x_dev1,
        &[SEED_POS], // ctx_positions (1 slot)
        &block_ids1,
        &block_positions1,
        BLOCK,
        &scratch1,
        &x_head_dev1,
        1.0, // qwen3-8B: full rotary (byte-identical to the pre-partial-rotary signature)
    )?;
    // Apply output_norm once to compare with cpu x_head_out (= model.norm(hidden)).
    let x_head_normed_dev1 = gpu
        .alloc_tensor(&[BLOCK, dim], DType::F32)
        .map_err(|e| format!("alloc x_head_normed ctx1: {e:?}"))?;
    gpu.rmsnorm_batched(
        &x_head_dev1,
        stage_norm_ref,
        &x_head_normed_dev1,
        BLOCK,
        dim,
        assets.config.norm_eps,
    )
    .map_err(|e| format!("rmsnorm ctx1: {e:?}"))?;
    let gpu_x_head1 = gpu
        .download_f32(&x_head_normed_dev1)
        .map_err(|e| format!("d2h x_head ctx1: {e:?}"))?;
    let check_b = parity_stats(
        "(b) ctx1 x_head (normed)",
        &gpu_x_head1,
        &cpu_x_head1,
        0.999,
        None,
    );

    // Checks (c) + (d): run_heads
    let draft1 = run_heads(
        &mut gpu,
        &dspark_weights,
        stage_norm_ref,
        &lm_head_f16,
        &x_head_dev1,
        SEED_TOK,
        BLOCK,
        vocab,
    )?;
    let tokens_match1 = draft1.tokens == cpu_markov1;
    let check_d1 = parity_stats(
        "(d) ctx1 confidence",
        &draft1.confidence,
        &cpu_confidence1,
        0.999,
        None,
    );

    // Free ctx1 resources.
    let _ = gpu.free_tensor(main_hidden_dev1);
    let _ = gpu.free_tensor(main_x_dev1);
    scratch1.free_gpu(&mut gpu);
    let _ = gpu.free_tensor(x_head_dev1);
    let _ = gpu.free_tensor(x_head_normed_dev1);

    // ═══════════════════════════════════════════════════════════════════════════
    // ctx_len=3 (Stage-1 multi-slot gate)
    // ═══════════════════════════════════════════════════════════════════════════
    eprintln!("\n=== ctx_len=3 (Stage-1 multi-slot) ===");

    let refs3 = Path::new(&refs_dir3);
    let cpu_main_hidden3 = load_f32bin(refs3.join("main_hidden.f32bin"))?;
    let cpu_main_x3 = load_f32bin(refs3.join("main_x.f32bin"))?;
    let cpu_x_head3 = load_f32bin(refs3.join("x_head_out.f32bin"))?;
    let cpu_markov_i32_3 = load_i32bin(refs3.join("markov_tokens.i32bin"))?;
    let cpu_confidence3 = load_f32bin(refs3.join("confidence_logits.f32bin"))?;
    let cpu_markov3: Vec<u32> = cpu_markov_i32_3.iter().map(|&t| t as u32).collect();

    let concat_w3 = CTX_LEN3 * N_TARGETS * dim;
    verify_len("ctx3 main_hidden", cpu_main_hidden3.len(), concat_w3)?;
    verify_len("ctx3 main_x", cpu_main_x3.len(), CTX_LEN3 * dim)?;
    verify_len("ctx3 x_head", cpu_x_head3.len(), BLOCK * dim)?;
    verify_len("ctx3 markov", cpu_markov3.len(), BLOCK)?;
    verify_len("ctx3 confidence", cpu_confidence3.len(), BLOCK)?;

    let main_hidden_dev3 = upload_f32(&mut gpu, &cpu_main_hidden3)?;

    // Check (e): main_proj_ingest_batched (ctx_len=3)
    let main_x_dev3 = gpu
        .alloc_tensor(&[CTX_LEN3 * dim], DType::F32)
        .map_err(|e| format!("alloc main_x ctx3: {e:?}"))?;
    main_proj_ingest_batched(
        &mut gpu,
        &dspark_weights,
        &main_hidden_dev3,
        &main_x_dev3,
        CTX_LEN3,
        dim,
    )?;
    let gpu_main_x3 = gpu
        .download_f32(&main_x_dev3)
        .map_err(|e| format!("d2h main_x ctx3: {e:?}"))?;
    let check_e = parity_stats(
        "(e) ctx3 main_x (3 rows)",
        &gpu_main_x3,
        &cpu_main_x3,
        0.999,
        None,
    );

    // Check (f): x_head_out from dspark_qwen3_block_forward (ctx_len=3)
    let scratch3 = Qwen3DsparkScratch::new(&mut gpu, &assets.config, BLOCK, CTX_LEN3)
        .map_err(|e| format!("Qwen3DsparkScratch ctx3: {e}"))?;
    let x_head_dev3 = gpu
        .alloc_tensor(&[BLOCK, dim], DType::F32)
        .map_err(|e| format!("alloc x_head ctx3: {e:?}"))?;
    let block_ids3 = noise_block_ids(cfg, SEED_TOK);
    dspark_qwen3_block_forward(
        &mut gpu,
        &assets.weights,
        &assets.config,
        &main_x_dev3,
        &CTX_POSITIONS3,
        &block_ids3,
        &BLOCK_POSITIONS3,
        BLOCK,
        &scratch3,
        &x_head_dev3,
        1.0, // qwen3-8B: full rotary
    )?;
    // Apply output_norm once.
    let x_head_normed_dev3 = gpu
        .alloc_tensor(&[BLOCK, dim], DType::F32)
        .map_err(|e| format!("alloc x_head_normed ctx3: {e:?}"))?;
    gpu.rmsnorm_batched(
        &x_head_dev3,
        stage_norm_ref,
        &x_head_normed_dev3,
        BLOCK,
        dim,
        assets.config.norm_eps,
    )
    .map_err(|e| format!("rmsnorm ctx3: {e:?}"))?;
    let gpu_x_head3 = gpu
        .download_f32(&x_head_normed_dev3)
        .map_err(|e| format!("d2h x_head ctx3: {e:?}"))?;
    let check_f = parity_stats(
        "(f) ctx3 x_head (normed)",
        &gpu_x_head3,
        &cpu_x_head3,
        0.999,
        None,
    );

    // Checks (g) + (h): run_heads
    // For ctx_len=3 the seed token is SEED_TOK (same block_ids[0]).
    let draft3 = run_heads(
        &mut gpu,
        &dspark_weights,
        stage_norm_ref,
        &lm_head_f16,
        &x_head_dev3,
        SEED_TOK,
        BLOCK,
        vocab,
    )?;
    let tokens_match3 = draft3.tokens == cpu_markov3;
    let check_h = parity_stats(
        "(h) ctx3 confidence",
        &draft3.confidence,
        &cpu_confidence3,
        0.999,
        None,
    );

    // Free ctx3 resources.
    let _ = gpu.free_tensor(main_hidden_dev3);
    let _ = gpu.free_tensor(main_x_dev3);
    scratch3.free_gpu(&mut gpu);
    let _ = gpu.free_tensor(x_head_dev3);
    let _ = gpu.free_tensor(x_head_normed_dev3);

    // ═══════════════════════════════════════════════════════════════════════════
    // Report
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\nQwen3-8B DSpark GPU-vs-CPU parity  (seed_tok={SEED_TOK} block={BLOCK}):");
    println!(
        "  {:<40} {:>8} {:>12} {:>10}  {}",
        "check", "n", "max_abs", "cosine", "verdict"
    );
    for c in [&check_a, &check_b, &check_d1, &check_e, &check_f, &check_h] {
        println!(
            "  {:<40} {:>8} {:>12.3e} {:>10.6}  {}",
            c.name,
            c.n,
            c.max_abs,
            c.cosine,
            if c.pass { "PASS" } else { "FAIL" }
        );
    }
    // Token checks.
    let tok1_v = if tokens_match1 { "PASS" } else { "FAIL" };
    let tok3_v = if tokens_match3 { "PASS" } else { "FAIL" };
    println!(
        "  {:<40} {:>8}                           {tok1_v}",
        "(c) ctx1 markov tokens", BLOCK
    );
    if !tokens_match1 {
        let first = draft1
            .tokens
            .iter()
            .zip(cpu_markov1.iter())
            .enumerate()
            .find(|(_, (g, c))| g != c);
        if let Some((i, (g, c))) = first {
            println!("    first mismatch at slot {i}: GPU={g} CPU={c}");
        }
        println!("  GPU: {:?}", draft1.tokens);
        println!("  CPU: {cpu_markov1:?}");
    } else {
        println!("  ctx1 tokens: {:?}", draft1.tokens);
    }
    println!(
        "  {:<40} {:>8}                           {tok3_v}",
        "(g) ctx3 markov tokens", BLOCK
    );
    if !tokens_match3 {
        let first = draft3
            .tokens
            .iter()
            .zip(cpu_markov3.iter())
            .enumerate()
            .find(|(_, (g, c))| g != c);
        if let Some((i, (g, c))) = first {
            println!("    first mismatch at slot {i}: GPU={g} CPU={c}");
        }
        println!("  GPU: {:?}", draft3.tokens);
        println!("  CPU: {cpu_markov3:?}");
    } else {
        println!("  ctx3 tokens: {:?}", draft3.tokens);
    }

    let all_pass = check_a.pass
        && check_b.pass
        && tokens_match1
        && check_d1.pass
        && check_e.pass
        && check_f.pass
        && tokens_match3
        && check_h.pass;
    if all_pass {
        println!("\nPARITY PASS — ctx_len=1 (backward-compat) and ctx_len=3 (multi-slot) both match CPU reference");
        Ok(())
    } else {
        Err("PARITY FAIL — see above for first diverging check".into())
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn verify_len(label: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        Err(format!("{label}: expected {want} got {got}"))
    } else {
        Ok(())
    }
}

struct ParityCheck {
    name: &'static str,
    n: usize,
    max_abs: f32,
    cosine: f32,
    pass: bool,
}

fn parity_stats(
    name: &'static str,
    gpu: &[f32],
    cpu: &[f32],
    cosine_threshold: f32,
    max_abs_threshold: Option<f32>,
) -> ParityCheck {
    let n = gpu.len().min(cpu.len());
    let (mut dot, mut ng, mut nc, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for i in 0..n {
        let (g, c) = (gpu[i], cpu[i]);
        max_abs = max_abs.max((g - c).abs());
        dot += g as f64 * c as f64;
        ng += g as f64 * g as f64;
        nc += c as f64 * c as f64;
    }
    let cosine = if ng > 0.0 && nc > 0.0 {
        (dot / (ng.sqrt() * nc.sqrt())) as f32
    } else {
        0.0
    };
    let pass_cosine = cosine >= cosine_threshold;
    let pass_abs = max_abs_threshold.map(|t| max_abs <= t).unwrap_or(true);
    ParityCheck {
        name,
        n,
        max_abs,
        cosine,
        pass: pass_cosine && pass_abs,
    }
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[v.len()], DType::F32)
        .map_err(|e| format!("alloc: {e:?}"))?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| format!("htod: {e:?}"))?;
    Ok(t)
}

fn load_f32bin(path: impl AsRef<Path>) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: file size {} not divisible by 4",
            path.as_ref().display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut v = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
    }
    Ok(v)
}

fn load_i32bin(path: impl AsRef<Path>) -> Result<Vec<i32>, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: file size {} not divisible by 4",
            path.as_ref().display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut v = vec![0i32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
    }
    Ok(v)
}
