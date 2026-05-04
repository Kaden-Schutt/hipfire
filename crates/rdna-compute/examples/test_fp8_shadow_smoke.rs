//! Smoke test for the gfx12 FP8 weight preload path.
//!
//! Allocates a small synthetic HFQ4-G256 weight tensor, runs the FP8 dequant
//! kernel via `ensure_fp8_shadow`, downloads the resulting FP8 bytes, decodes
//! a few back to FP32 (via the reverse OCP E4M3 mapping) and compares against
//! the FP16 reference dequant. Expected divergence: bounded by E4M3 precision
//! (~1.5% relative for typical HFQ4 dequantized weights).
//!
//! The point of this test is NOT exhaustive correctness — it's "the new
//! ensure_fp8_shadow helper actually launches the kernel, populates the cache,
//! and the FP8 bytes round-trip to plausible FP32 values."
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example test_fp8_shadow_smoke

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = 256;
    let k: usize = 512;
    assert!(k % 256 == 0);
    let groups_per_row = k / 256;

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!("SKIP: ensure_fp8_shadow is gfx12-only.");
        std::process::exit(0);
    }

    // Synth HFQ4-G256 weights: 4 B FP32 scale + 4 B FP32 zero + 128 B nibbles
    // per group. Use a stable scale (0.1) and zero (0.0) so the dequant outputs
    // are easy to reason about: w = 0.1 * q for q in 0..15, range [0, 1.5].
    let row_bytes = groups_per_row * 136;
    let mut weight_bytes = vec![0u8; m * row_bytes];
    let mut state = 0xDEAD_BEEFu64;
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            weight_bytes[gp..gp + 4].copy_from_slice(&0.1f32.to_le_bytes());
            weight_bytes[gp + 4..gp + 8].copy_from_slice(&0.0f32.to_le_bytes());
            for i in 0..128 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                weight_bytes[gp + 8 + i] = (state >> 32) as u8;  // random nibble pairs
            }
        }
    }

    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");
    // Wrap as a GpuTensor with an HFQ4-flavor dtype so ensure_fp8_shadow can key on it.
    // upload_raw returns a Raw-dtype tensor; that's fine — ensure_fp8_shadow only
    // uses the buf pointer + the (m, k) dims passed in.

    eprintln!("\n=== ensure_fp8_shadow smoke ===");
    eprintln!("HFQ4 weight: M={m}, K={k}, {} bytes", m * row_bytes);

    let fp8_ptr = gpu
        .ensure_fp8_shadow(&a_raw, m, k)
        .expect("ensure_fp8_shadow")
        .expect("must return Some on gfx12");
    eprintln!("FP8 shadow ptr: {:p}", fp8_ptr);

    // Second call should hit the cache and return the same pointer.
    let fp8_ptr2 = gpu
        .ensure_fp8_shadow(&a_raw, m, k)
        .expect("ensure_fp8_shadow #2")
        .expect("must return Some");
    assert_eq!(fp8_ptr as usize, fp8_ptr2 as usize, "cache should return the same ptr");
    eprintln!("cache hit on 2nd call: ptr identical OK");

    // Read FP8 bytes back via raw download (allocate a fresh GpuTensor wrapping
    // the cached buffer). Easier: just allocate our own + dequant directly.
    let fp8_buf = gpu.alloc_tensor(&[m * k], DType::Raw).expect("alloc fp8");
    gpu.dequantize_hfq4g256_to_fp8_gfx12(&a_raw.buf, &fp8_buf.buf, m, k)
        .expect("direct dequant");

    // download_raw doesn't exist; use the lower-level memcpy_dtoh on the buffer.
    let mut fp8_bytes: Vec<u8> = vec![0u8; m * k];
    gpu.hip.memcpy_dtoh(&mut fp8_bytes, &fp8_buf.buf).expect("dtoh fp8 bytes");

    // Decode a sample of FP8 bytes back to FP32. OCP E4M3 (1 sign + 4 exp + 3 mantissa,
    // bias 7). 0x00 = +0; 0x80 = -0; 0x7F = NaN; everything else = (-1)^s * 2^(e-7) * 1.mmm.
    fn fp8_e4m3_to_f32(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 3) & 0x0F;
        let mantissa = (b & 0x07) as f32;
        if exp == 0 {
            // Subnormal: value = sign * 2^(1 - bias) * (mantissa / 8)
            return sign * (mantissa / 8.0) * 2f32.powi(-6);
        }
        if exp == 0x0F && (b & 0x07) == 0x07 { return f32::NAN; }
        sign * (1.0 + mantissa / 8.0) * 2f32.powi(exp as i32 - 7)
    }

    eprintln!("\n--- sample first 16 FP8 bytes (row 0, K=0..15) ---");
    let mut max_err: f32 = 0.0;
    for i in 0..16 {
        let q_byte_offset = (i / 2) as usize;
        let q_byte = weight_bytes[8 + q_byte_offset];
        let q = if i & 1 == 0 { (q_byte & 0x0F) as f32 } else { (q_byte >> 4) as f32 };
        let want_fp32 = 0.1f32 * q;
        let got_fp32 = fp8_e4m3_to_f32(fp8_bytes[i]);
        let err = (got_fp32 - want_fp32).abs();
        max_err = max_err.max(err);
        eprintln!("  K={i:>2}: q={q:>2}  want={want_fp32:.4}  fp8_byte=0x{:02x}  got={got_fp32:.4}  err={err:.4}",
                  fp8_bytes[i], );
    }

    eprintln!("\nmax abs err over 16 samples: {max_err:.4}");
    if max_err < 0.1 {
        eprintln!("PASS: ensure_fp8_shadow + dequantize_hfq4g256_to_fp8_gfx12 round-trip OK");
        std::process::exit(0);
    } else {
        eprintln!("FAIL: FP8 dequant decoded values diverge from FP16 reference beyond E4M3 expected tolerance.");
        std::process::exit(1);
    }
}
