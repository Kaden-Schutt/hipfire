//! CPU-only self-test: fwht_shfl_forward_512 shuffle path == scalar reference.
//!
//! Runs entirely on the CPU by simulating the warp-shuffle butterfly using the
//! same mathematical structure, then compares against the scalar FWHT-512
//! reference.  No GPU is required.
//!
//! The scalar reference (`fwht_forward_512_ref`) matches the device
//! `fwht_forward_512` in turbo_common.h: signs1 → 9-pass butterfly → scale →
//! signs2.  The shuffle simulation (`fwht_shfl_forward_512_sim`) mirrors the
//! device `fwht_shfl_forward_512`: 4 register-local passes followed by 5
//! ds_swizzle passes.  Both operate on the same 32-lane × 16-element layout.
//!
//! Spec validation: §1.5 of the fwht3-hd512 implementation spec.
//! Gate: max abs error < 1e-5.
//!
//! Run:
//!   cargo run -p rdna-compute --example test_fwht512_scalar_parity

use rdna_compute::gen_fwht_signs;

const N: usize = 512;
const LANES: usize = 32;
const ELEMS_PER_LANE: usize = N / LANES; // = 16

// ---- Scalar reference (matches fwht_forward_512 in turbo_common.h) --------

fn fwht_forward_512_ref(x: &mut [f32; N], signs1: &[f32; N], signs2: &[f32; N]) {
    for i in 0..N { x[i] *= signs1[i]; }
    let mut stride = 1;
    while stride < N {
        let mut i = 0;
        while i < N {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 1.0_f32 / (N as f32).sqrt(); // 0.0441941738…
    for i in 0..N { x[i] *= scale * signs2[i]; }
}

// ---- Shuffle simulation (matches fwht_shfl_forward_512 in turbo_common.h) -

/// Simulate ds_swizzle HADAMARD_BFLY(v, pattern, stride, tid) across all 32 lanes.
/// For each lane tid: fetch partner = v[tid XOR stride]; then
///   if tid & stride != 0 { v[tid] = partner - v[tid] }
///   else                  { v[tid] = v[tid] + partner }
fn hadamard_bfly_all(v: &mut [f32; LANES], xor_stride: usize) {
    let old = *v;
    for tid in 0..LANES {
        let partner = old[tid ^ xor_stride];
        if tid & xor_stride != 0 {
            v[tid] = partner - old[tid];
        } else {
            v[tid] = old[tid] + partner;
        }
    }
}

fn fwht_shfl_forward_512_sim(x: &mut [f32; N], signs1: &[f32; N], signs2: &[f32; N]) {
    // Repack into 16-register layout: regs[elem][lane]
    // Lane tid owns dims tid*16 .. tid*16+15
    let mut regs: [[f32; LANES]; ELEMS_PER_LANE] = [[0.0; LANES]; ELEMS_PER_LANE];
    for tid in 0..LANES {
        let d0 = tid * ELEMS_PER_LANE;
        for i in 0..ELEMS_PER_LANE {
            regs[i][tid] = x[d0 + i] * signs1[d0 + i];
        }
    }

    // Passes 1-4: register-local (within each lane's 16 elements)
    for tid in 0..LANES {
        let v = &mut regs;
        // Pass 1: stride 1 — pairs (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
        for k in [0, 2, 4, 6, 8, 10, 12, 14] {
            let a = v[k][tid]; let b = v[k+1][tid];
            v[k][tid] = a + b; v[k+1][tid] = a - b;
        }
        // Pass 2: stride 2 — pairs (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
        for k in [0, 1, 4, 5, 8, 9, 12, 13] {
            let b_idx = k + 2;
            let a = v[k][tid]; let b = v[b_idx][tid];
            v[k][tid] = a + b; v[b_idx][tid] = a - b;
        }
        // Pass 3: stride 4 — pairs (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)
        for k in [0, 1, 2, 3, 8, 9, 10, 11] {
            let b_idx = k + 4;
            let a = v[k][tid]; let b = v[b_idx][tid];
            v[k][tid] = a + b; v[b_idx][tid] = a - b;
        }
        // Pass 4: stride 8 — pairs (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
        for k in 0..8 {
            let b_idx = k + 8;
            let a = v[k][tid]; let b = v[b_idx][tid];
            v[k][tid] = a + b; v[b_idx][tid] = a - b;
        }
    }

    // Passes 5-9: lane-level ds_swizzle butterflies (xor-strides 1,2,4,8,16)
    for r in 0..ELEMS_PER_LANE {
        hadamard_bfly_all(&mut regs[r], 1);
        hadamard_bfly_all(&mut regs[r], 2);
        hadamard_bfly_all(&mut regs[r], 4);
        hadamard_bfly_all(&mut regs[r], 8);
        hadamard_bfly_all(&mut regs[r], 16);
    }

    // Scale by 1/sqrt(512) and apply signs2; repack back into linear order
    let scale = 1.0_f32 / (N as f32).sqrt();
    for tid in 0..LANES {
        let d0 = tid * ELEMS_PER_LANE;
        for i in 0..ELEMS_PER_LANE {
            x[d0 + i] = regs[i][tid] * scale * signs2[d0 + i];
        }
    }
}

fn main() {
    // Use the canonical gen_fwht_signs generator with same seeds as existing tests.
    let signs1_vec = gen_fwht_signs(43, N);
    let signs2_vec = gen_fwht_signs(1043, N);
    let mut s1 = [0.0f32; N];
    let mut s2 = [0.0f32; N];
    s1.copy_from_slice(&signs1_vec);
    s2.copy_from_slice(&signs2_vec);

    // Test with a gradient input (same as test_fwht_128_gpu_vs_cpu)
    let base: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001 - 0.25).collect();
    let mut ref_x = [0.0f32; N];
    let mut sim_x = [0.0f32; N];
    ref_x.copy_from_slice(&base);
    sim_x.copy_from_slice(&base);

    fwht_forward_512_ref(&mut ref_x, &s1, &s2);
    fwht_shfl_forward_512_sim(&mut sim_x, &s1, &s2);

    let max_err = ref_x
        .iter()
        .zip(sim_x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("FWHT-512 scalar-ref vs shuffle-sim: max abs error = {:e}", max_err);
    println!("First 8 elements:");
    for i in 0..8 {
        println!(
            "  [{i:3}]  ref={:.7}  sim={:.7}  diff={:.2e}",
            ref_x[i], sim_x[i], (ref_x[i] - sim_x[i]).abs()
        );
    }

    // Also test with random-ish input using a different sign seed
    let signs1b = gen_fwht_signs(7, N);
    let signs2b = gen_fwht_signs(42, N);
    let mut s1b = [0.0f32; N];
    let mut s2b = [0.0f32; N];
    s1b.copy_from_slice(&signs1b);
    s2b.copy_from_slice(&signs2b);

    let base2: Vec<f32> = (0..N)
        .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5)
        .collect();
    let mut ref_x2 = [0.0f32; N];
    let mut sim_x2 = [0.0f32; N];
    ref_x2.copy_from_slice(&base2);
    sim_x2.copy_from_slice(&base2);

    fwht_forward_512_ref(&mut ref_x2, &s1b, &s2b);
    fwht_shfl_forward_512_sim(&mut sim_x2, &s1b, &s2b);

    let max_err2 = ref_x2
        .iter()
        .zip(sim_x2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nRandom-ish input: max abs error = {:e}", max_err2);

    let threshold = 1e-5_f32;
    let combined_max = max_err.max(max_err2);
    if combined_max >= threshold {
        eprintln!("\nFAIL: max abs error {:e} >= {:e}", combined_max, threshold);
        std::process::exit(1);
    }
    println!("\nPASS: fwht_shfl_forward_512 shuffle simulation agrees with scalar reference within 1e-5");
}
