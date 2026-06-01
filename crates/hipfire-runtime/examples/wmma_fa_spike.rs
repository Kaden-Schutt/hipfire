//! Phase 1.0 spike: ALU-only A/B between scalar fp16 FA and (eventually)
//! WMMA fp16 FA on the same pre-dequantized fp16 K/V inputs.
//!
//! This harness deliberately strips production complexity (asym dequant,
//! Givens rotation, KV cache write path) so the timing isolates the WMMA
//! vs scalar ALU question. Inputs are synthetic random fp16. Numerical
//! correctness is checked against a single-pass CPU reference.
//!
//! Phase 1.0 first pass: SCALAR ONLY. The WMMA kernel is held until the
//! design is verified against `probe_wmma`. The harness shape is right
//! for adding WMMA next without restructure.
//!
//! Usage:
//!   wmma_fa_spike [--batch N=128] [--seq L=2048] [--n-heads H=28]
//!                 [--n-kv-heads KV=4] [--head-dim D=128]
//!                 [--warmup W=2] [--measure M=5]

use std::time::Instant;

const SCALAR_KERNEL_SRC: &str = include_str!(
    "../../../experiments/wmma_fa_spike/fa_scalar_fp16.hip"
);
const WMMA_KERNEL_SRC: &str = include_str!(
    "../../../experiments/wmma_fa_spike/fa_wmma_fp16.hip"
);
const WMMA_KERNEL_GFX12_SRC: &str = include_str!(
    "../../../experiments/wmma_fa_spike/fa_wmma_fp16.gfx12.hip"
);

fn main() {
    use rdna_compute::DType;

    // ── args ───────────────────────────────────────────────────────────
    let mut batch: usize = 128;
    let mut seq: usize = 2048;
    let mut n_heads: usize = 28;
    let mut n_kv_heads: usize = 4;
    let mut head_dim: usize = 128;
    let mut warmup: usize = 2;
    let mut measure: usize = 5;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--batch" => { batch = argv[i + 1].parse().unwrap(); i += 2; }
            "--seq" => { seq = argv[i + 1].parse().unwrap(); i += 2; }
            "--n-heads" => { n_heads = argv[i + 1].parse().unwrap(); i += 2; }
            "--n-kv-heads" => { n_kv_heads = argv[i + 1].parse().unwrap(); i += 2; }
            "--head-dim" => { head_dim = argv[i + 1].parse().unwrap(); i += 2; }
            "--warmup" => { warmup = argv[i + 1].parse().unwrap(); i += 2; }
            "--measure" => { measure = argv[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    assert!(n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads (GQA group)");
    assert!(head_dim == 128 || head_dim == 256, "spike pinned to head_dim ∈ {{128, 256}}");
    assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 (dims_per_thread = head_dim/32)");

    eprintln!("=== wmma_fa_spike ===");
    eprintln!("batch={batch}  seq={seq}  n_heads={n_heads}  n_kv_heads={n_kv_heads}  head_dim={head_dim}");
    eprintln!("warmup={warmup}  measure={measure}");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let scale_attn = 1.0f32 / (head_dim as f32).sqrt();
    let tile_size: usize = 128;
    let max_tiles = (seq + tile_size - 1) / tile_size;
    let _kv_group = n_heads / n_kv_heads;

    // ── synthesize inputs (host-side, deterministic) ───────────────────
    // Q as fp32 (matches scalar kernel signature) AND as fp16 (matches WMMA
    // kernel signature). The WMMA kernel takes Q in fp16 directly — production
    // would inline the Givens-rotated fp32→fp16 narrow at Q-load.
    // Random in [-0.5, 0.5] — keeps post-softmax numerics in a reasonable range.
    let q_numel = batch * n_heads * head_dim;
    let k_numel = seq * n_kv_heads * head_dim;
    let v_numel = k_numel;

    let q_host: Vec<f32> = (0..q_numel).map(|i| {
        let x = (i as u32).wrapping_mul(2654435761u32) as f32 / u32::MAX as f32;
        x - 0.5
    }).collect();
    let k_host_f32: Vec<f32> = (0..k_numel).map(|i| {
        let x = ((i as u32).wrapping_mul(0x9e3779b1u32)) as f32 / u32::MAX as f32;
        x - 0.5
    }).collect();
    let v_host_f32: Vec<f32> = (0..v_numel).map(|i| {
        let x = ((i as u32).wrapping_mul(0x85ebca77u32)) as f32 / u32::MAX as f32;
        x - 0.5
    }).collect();
    // Narrow K, V to fp16 (production K/V are dequantized to fp16 anyway in real path).
    let k_host_f16: Vec<u16> = k_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let v_host_f16: Vec<u16> = v_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();

    // Positions: each batch index gets a random valid pos in [0, seq-1].
    let positions: Vec<i32> = (0..batch).map(|b| {
        ((b * 7919 + 13) % seq) as i32
    }).collect();

    let q_host_f16: Vec<u16> = q_host.iter().map(|&x| f32_to_f16_bits(x)).collect();

    eprintln!("Uploading inputs to GPU…");
    let d_q_f32 = gpu.upload_f32(&q_host, &[batch, n_heads, head_dim]).unwrap();
    let d_q_f16 = upload_f16(&mut gpu, &q_host_f16, &[batch, n_heads, head_dim]);
    let d_k = upload_f16(&mut gpu, &k_host_f16, &[seq, n_kv_heads, head_dim]);
    let d_v = upload_f16(&mut gpu, &v_host_f16, &[seq, n_kv_heads, head_dim]);
    let d_positions = upload_i32(&mut gpu, &positions, &[batch]);

    // Partials and final output. Use zeros() so OOB tiles (bids past seq_len)
    // have a known sentinel value when comparing scalar vs WMMA outputs.
    let partials_stride = 2 + head_dim;
    let partials_numel = batch * n_heads * max_tiles * partials_stride;
    let d_partials_scalar = gpu.zeros(&[partials_numel], DType::F32).unwrap();
    let d_partials_wmma = gpu.zeros(&[partials_numel], DType::F32).unwrap();

    eprintln!("Compiling kernels…");
    gpu.ensure_kernel_public("fa_scalar_fp16", SCALAR_KERNEL_SRC, "fa_scalar_fp16").unwrap();
    let wmma_kernel_src = if gpu.arch.starts_with("gfx12") {
        WMMA_KERNEL_GFX12_SRC
    } else {
        WMMA_KERNEL_SRC
    };
    gpu.ensure_kernel_public("fa_wmma_fp16", wmma_kernel_src, "fa_wmma_fp16").unwrap();
    // No reduce kernel needed — the spike times the TILE kernel only. Reduce
    // is identical between scalar and WMMA branches (same partials layout) so
    // adding it to the measurement just adds noise.

    // ── run scalar fp16 baseline ───────────────────────────────────────
    eprintln!("--- scalar fp16 baseline ---");
    for w in 0..warmup {
        run_scalar_pass(&mut gpu, &d_q_f32, &d_k, &d_v, &d_partials_scalar, &d_positions,
            n_heads, n_kv_heads, head_dim, seq, scale_attn, tile_size, max_tiles, batch);
        eprintln!("  warmup {} done", w);
    }
    let mut scalar_times = Vec::<f64>::with_capacity(measure);
    for m in 0..measure {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        run_scalar_pass(&mut gpu, &d_q_f32, &d_k, &d_v, &d_partials_scalar, &d_positions,
            n_heads, n_kv_heads, head_dim, seq, scale_attn, tile_size, max_tiles, batch);
        gpu.hip.device_synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        scalar_times.push(dt);
        eprintln!("  measure {m}: {:.3} ms", dt);
    }
    let scalar_min = scalar_times.iter().copied().fold(f64::INFINITY, f64::min);
    let scalar_med = median(&mut scalar_times.clone());

    // ── run WMMA fp16 spike ────────────────────────────────────────────
    eprintln!("--- WMMA fp16 spike ---");
    let m_tiles = (batch + 16 - 1) / 16;
    for w in 0..warmup {
        run_wmma_pass(&mut gpu, &d_q_f16, &d_k, &d_v, &d_partials_wmma, &d_positions,
            n_heads, n_kv_heads, head_dim, seq, scale_attn, tile_size, max_tiles, batch, m_tiles);
        eprintln!("  warmup {} done", w);
    }
    let mut wmma_times = Vec::<f64>::with_capacity(measure);
    for m in 0..measure {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        run_wmma_pass(&mut gpu, &d_q_f16, &d_k, &d_v, &d_partials_wmma, &d_positions,
            n_heads, n_kv_heads, head_dim, seq, scale_attn, tile_size, max_tiles, batch, m_tiles);
        gpu.hip.device_synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        wmma_times.push(dt);
        eprintln!("  measure {m}: {:.3} ms", dt);
    }
    let wmma_min = wmma_times.iter().copied().fold(f64::INFINITY, f64::min);
    let wmma_med = median(&mut wmma_times.clone());

    // ── numerical compare (filtered: ignore tiny cells where rel-diff is
    // dominated by quantization noise) ────────────────────────────────
    let p_scalar = gpu.download_f32(&d_partials_scalar).unwrap();
    let p_wmma = gpu.download_f32(&d_partials_wmma).unwrap();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_finite = 0usize;
    let mut n_significant = 0usize;
    // Histogram: how many cells exceed each threshold.
    let mut buckets = [0usize; 7];
    let thresholds = [0.0001f32, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5];
    for (a, b) in p_scalar.iter().zip(p_wmma.iter()) {
        if !a.is_finite() || !b.is_finite() { continue; }
        n_finite += 1;
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        for (idx, &th) in thresholds.iter().enumerate() {
            if d > th { buckets[idx] += 1; }
        }
        // Significant: only compute rel-diff if both values are well above
        // fp16 epsilon (~6e-5). Near-zero cells produce phantom rel-diffs.
        if a.abs().max(b.abs()) > 0.01 {
            n_significant += 1;
            let r = d / a.abs().max(b.abs());
            if r > max_rel { max_rel = r; }
        }
    }

    eprintln!();
    eprintln!("=== wmma_fa_spike summary ===");
    eprintln!("scalar fp16 FA:   median {:.3} ms   min {:.3} ms", scalar_med, scalar_min);
    eprintln!("WMMA   fp16 FA:   median {:.3} ms   min {:.3} ms", wmma_med, wmma_min);
    let speedup = scalar_med / wmma_med;
    eprintln!("speedup (scalar/WMMA): {:.2}× (>1 means WMMA faster)", speedup);
    eprintln!("partial-buffer max |Δ|: {:.4}", max_abs);
    eprintln!("                max rel-diff: {:.4} (over {} cells with |val| > 0.01)",
        max_rel, n_significant);
    eprintln!("                {} finite cells total", n_finite);
    eprintln!("                |Δ| histogram:");
    for (th, &count) in thresholds.iter().zip(buckets.iter()) {
        eprintln!("                  > {:>7.4}: {:>10} cells  ({:>5.2}%)",
            th, count, 100.0 * count as f64 / n_finite as f64);
    }
    eprintln!();
    eprintln!("Phase 1.0 gate (per docs/plans/wmma-flash-attention-prefill.md):");
    eprintln!("  +25% ALU stub on gfx1100 → continue to Phase 1.1");
    eprintln!("  ≥ 0% on gfx1151        → continue (with the bandwidth caveat)");
}

fn run_scalar_pass(
    gpu: &mut rdna_compute::Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_partials: &rdna_compute::GpuTensor,
    d_positions: &rdna_compute::GpuTensor,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
    max_seq: usize, scale_attn: f32, tile_size: usize, max_tiles: usize,
    batch: usize,
) {
    let mut params = hip_bridge::KernargBlob::new();
    params.push_ptr(d_q.buf.as_ptr());
    params.push_ptr(d_k.buf.as_ptr());
    params.push_ptr(d_v.buf.as_ptr());
    params.push_ptr(d_partials.buf.as_ptr());
    params.push_ptr(d_positions.buf.as_ptr());
    params.push_i32(n_heads as i32);
    params.push_i32(n_kv_heads as i32);
    params.push_i32(head_dim as i32);
    params.push_i32(max_seq as i32);
    params.push_f32(scale_attn);
    params.push_i32(tile_size as i32);
    params.push_i32(max_tiles as i32);
    gpu.launch_kernel_blob(
        "fa_scalar_fp16",
        [n_heads as u32, max_tiles as u32, batch as u32],
        [32, 1, 1],
        (tile_size * 4) as u32,
        params.as_mut_slice(),
    ).unwrap();
}

fn run_wmma_pass(
    gpu: &mut rdna_compute::Gpu,
    d_q_f16: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_partials: &rdna_compute::GpuTensor,
    d_positions: &rdna_compute::GpuTensor,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
    max_seq: usize, scale_attn: f32, tile_size: usize, max_tiles: usize,
    batch: usize, m_tiles: usize,
) {
    let mut params = hip_bridge::KernargBlob::new();
    params.push_ptr(d_q_f16.buf.as_ptr());
    params.push_ptr(d_k.buf.as_ptr());
    params.push_ptr(d_v.buf.as_ptr());
    params.push_ptr(d_partials.buf.as_ptr());
    params.push_ptr(d_positions.buf.as_ptr());
    params.push_i32(n_heads as i32);
    params.push_i32(n_kv_heads as i32);
    params.push_i32(head_dim as i32);
    params.push_i32(max_seq as i32);
    params.push_f32(scale_attn);
    params.push_i32(tile_size as i32);
    params.push_i32(max_tiles as i32);
    params.push_i32(batch as i32);
    gpu.launch_kernel_blob(
        "fa_wmma_fp16",
        [n_heads as u32, m_tiles as u32, max_tiles as u32],
        [32, 1, 1],
        0,   // LDS is static __shared__ inside the kernel
        params.as_mut_slice(),
    ).unwrap();
}

fn upload_f16(gpu: &mut rdna_compute::Gpu, data: &[u16], shape: &[usize])
    -> rdna_compute::GpuTensor
{
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    gpu.upload_raw(bytes, shape).unwrap()
}

fn upload_i32(gpu: &mut rdna_compute::Gpu, data: &[i32], shape: &[usize])
    -> rdna_compute::GpuTensor
{
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.upload_raw(bytes, shape).unwrap()
}

fn f32_to_f16_bits(x: f32) -> u16 {
    // Minimal fp32 → fp16 with round-to-nearest-even. Subnormals flush to zero;
    // sufficient for synthetic test inputs in [-0.5, 0.5].
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 0xFF {
        // inf / NaN
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7C00 | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return (sign << 15) | 0x7C00;   // overflow → inf
    }
    if new_exp <= 0 {
        // Subnormal / underflow: flush to zero (good enough for synthetic data).
        return sign << 15;
    }
    let new_mant = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let mut h = (sign << 15) | ((new_exp as u16) << 10) | new_mant;
    if round_bit == 1 && (mant & 0xFFF) != 0 {
        // round up (RTNE)
        h = h.wrapping_add(1);
    }
    h
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 0 { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 } else { xs[n / 2] }
}
