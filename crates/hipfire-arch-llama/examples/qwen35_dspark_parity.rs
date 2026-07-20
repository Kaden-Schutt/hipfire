// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen35_dspark_parity: GPU-vs-CPU x_head parity for the ORNITH Qwen3.5 DSpark
//! drafter forward (partial-rotary, reduced-vocab). Localizes the τ=0 bug by
//! comparing the drafter's `main_x` (fc ingest) and `x_head` (block-attention
//! forward) against the CPU reference from `ornith_dspark_cpu_ref.py`.
//!
//! Fixed input (must match the CPU ref): seed_pos=42, seed_tok=100, ctx_len=1,
//! main_hidden[i] = sin(i*0.013)*0.5 (read from the ref's dumped main_hidden.f32bin
//! to guarantee byte-identical input), block_ids=[seed, mask,...], partial-rotary
//! from the sidecar metadata.
//!
//! Usage:
//!   source scripts/gpu-lock.sh && gpu_acquire dspark-qwen35
//!   cargo build --release -p hipfire-runtime -p hipfire-arch-llama --example qwen35_dspark_parity
//!   ./target/release/examples/qwen35_dspark_parity [sidecar.mq6] [refs-dir]

use hipfire_arch_llama::dspark_body::{
    dspark_qwen3_block_forward, load_qwen3_dspark, Qwen3DsparkScratch,
};
use hipfire_runtime::dspark_core::{main_proj_ingest, noise_block_ids, run_heads};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const SEED_TOK: u32 = 100;
const SEED_POS: usize = 42;

fn main() -> Result<(), String> {
    let hfq_path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/ornith-35b-aeon-dspark.mq6",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let refs_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/home/bjoern/dspark-work/ornith_parity_refs".into());

    eprintln!("opening {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path)).map_err(|e| format!("open: {e:?}"))?;
    hfq.drop_mmap();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU ready (arch={})", gpu.arch_caps.arch());

    let (dspark_weights, assets) =
        load_qwen3_dspark(&hfq, &mut gpu)?.ok_or("load_qwen3_dspark: no dspark_* metadata")?;
    let cfg = &dspark_weights.cfg;
    let dim = assets.config.dim;
    let block = cfg.block_size;
    let n_targets = cfg.target_layer_ids.len();
    let prf = cfg.partial_rotary_factor;
    let n_rot = (assets.config.head_dim as f32 * prf) as usize;
    eprintln!(
        "loaded: dim={dim} head_dim={} block={block} n_targets={n_targets} \
         partial_rotary={prf} n_rot={n_rot} rope_theta={} rms_eps={}",
        assets.config.head_dim, cfg.rope_theta, assets.config.norm_eps
    );

    let refs = Path::new(&refs_dir);
    let cpu_main_hidden = load_f32bin(refs.join("main_hidden.f32bin"))?;
    let cpu_main_x = load_f32bin(refs.join("main_x.f32bin"))?;
    let cpu_x_head = load_f32bin(refs.join("x_head_out.f32bin"))?;
    verify_len("main_hidden", cpu_main_hidden.len(), n_targets * dim)?;
    verify_len("main_x", cpu_main_x.len(), dim)?;
    verify_len("x_head", cpu_x_head.len(), block * dim)?;

    let main_hidden_dev = upload_f32(&mut gpu, &cpu_main_hidden)?;

    // (a) main_x = hidden_norm(fc(main_hidden))
    let main_x_dev = gpu
        .alloc_tensor(&[dim], DType::F32)
        .map_err(|e| format!("alloc main_x: {e:?}"))?;
    main_proj_ingest(&mut gpu, &dspark_weights, &main_hidden_dev, &main_x_dev)?;
    let gpu_main_x = gpu
        .download_f32(&main_x_dev)
        .map_err(|e| format!("d2h main_x: {e:?}"))?;
    let check_a = parity_stats(
        "(a) main_x = hidden_norm(fc(main_hidden))",
        &gpu_main_x,
        &cpu_main_x,
        0.999,
    );

    // (b) x_head = norm(dspark_qwen3_block_forward(...))  [pre-norm out → rmsnorm]
    let scratch = Qwen3DsparkScratch::new(&mut gpu, &assets.config, block, 1)
        .map_err(|e| format!("scratch: {e}"))?;
    let x_head_dev = gpu
        .alloc_tensor(&[block, dim], DType::F32)
        .map_err(|e| format!("alloc x_head: {e:?}"))?;
    let block_ids = noise_block_ids(cfg, SEED_TOK);
    let block_positions: Vec<usize> = (0..block).map(|i| SEED_POS + i).collect();
    dspark_qwen3_block_forward(
        &mut gpu,
        &assets.weights,
        &assets.config,
        &main_x_dev,
        &[SEED_POS],
        &block_ids,
        &block_positions,
        block,
        &scratch,
        &x_head_dev,
        prf,
    )?;
    let x_head_normed = gpu
        .alloc_tensor(&[block, dim], DType::F32)
        .map_err(|e| format!("alloc x_head_normed: {e:?}"))?;
    gpu.rmsnorm_batched(
        &x_head_dev,
        &assets.weights.output_norm,
        &x_head_normed,
        block,
        dim,
        assets.config.norm_eps,
    )
    .map_err(|e| format!("rmsnorm: {e:?}"))?;
    let gpu_x_head = gpu
        .download_f32(&x_head_normed)
        .map_err(|e| format!("d2h x_head: {e:?}"))?;
    let check_b = parity_stats(
        "(b) x_head = norm(block_forward)",
        &gpu_x_head,
        &cpu_x_head,
        0.999,
    );

    // (c) heads: run_heads on the (correct) pre-norm x_head → draft tokens.
    // Dumps the pre-norm x_head so a numpy heads-reference can compare on the
    // exact same input, isolating run_heads (lm_head/markov/d2t) from the forward.
    let draft_vocab = if cfg.draft_vocab_size > 0 {
        cfg.draft_vocab_size
    } else {
        assets.weights.output.m
    };
    let mut lm_head_f16 = assets.weights.output.buf.shallow_clone();
    lm_head_f16.dtype = DType::F16;
    lm_head_f16.shape = vec![draft_vocab];
    let draft = run_heads(
        &mut gpu,
        &dspark_weights,
        &assets.weights.output_norm,
        &lm_head_f16,
        &x_head_dev,
        SEED_TOK,
        block,
        draft_vocab,
    )?;
    let x_head_prenorm = gpu
        .download_f32(&x_head_dev)
        .map_err(|e| format!("d2h x_head prenorm: {e:?}"))?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            x_head_prenorm.as_ptr() as *const u8,
            x_head_prenorm.len() * 4,
        )
    };
    std::fs::write("/tmp/hipfire_x_head_prenorm.f32bin", bytes)
        .map_err(|e| format!("write x_head prenorm: {e}"))?;
    eprintln!(
        "draft_vocab={draft_vocab}  hipfire drafts (target ids): {:?}",
        draft.tokens
    );
    eprintln!("wrote /tmp/hipfire_x_head_prenorm.f32bin (pre-norm x_head, for numpy heads check)");

    println!("\nORNITH Qwen3.5 DSpark GPU-vs-CPU x_head parity (seed_tok={SEED_TOK} block={block} n_rot={n_rot}):");
    for c in [&check_a, &check_b] {
        println!(
            "  {:<44} n={:>6} max_abs={:>11.3e} cosine={:>9.6}  {}",
            c.name,
            c.n,
            c.max_abs,
            c.cosine,
            if c.pass { "PASS" } else { "FAIL" }
        );
    }
    if check_a.pass && check_b.pass {
        println!("\nPARITY PASS — drafter forward matches the CPU reference");
        Ok(())
    } else {
        println!(
            "\nPARITY FAIL — drafter forward diverges. (a) fc-ingest, (b) block-attention forward."
        );
        Err("parity fail".into())
    }
}

// ── helpers (from qwen3_dspark_parity.rs) ────────────────────────────────────
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
    ParityCheck {
        name,
        n,
        max_abs,
        cosine,
        pass: cosine >= cosine_threshold,
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
            "{}: size {} not /4",
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
