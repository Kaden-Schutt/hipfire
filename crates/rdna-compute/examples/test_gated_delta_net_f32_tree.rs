// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tree-aware gated_delta_net_f32 correctness test.
//!
//! Verifies: for spine topology (parent_indices = [-1, 0, 1, 2, ...]), the
//! FP32 tree-GDN kernel produces outputs matching the linear FP32 GDN kernel
//! (`gated_delta_net_f32`) called N times in sequence (n_tokens=1 each) on
//! the rolling F32 S state.
//!
//! Tolerance (not byte-exact): the linear FP32 kernel uses a 128-thread/head
//! decomposition while the tree kernel uses 32 threads × 32 tiles, so the
//! dot-product reduction order differs → small ULP-level differences are
//! expected and correct. A real layout/offset bug in the new F32 tape
//! read/write would produce diffs orders of magnitude larger than the
//! reduction-order floor, so a tight absolute tolerance catches it.
//!
//! Build: `cargo run --release --features deltanet \
//!   --example test_gated_delta_net_f32_tree -p rdna-compute`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_gated_delta_net_f32_tree requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::DType;
    const HD: usize = 128;
    const N_HEADS: usize = 4; // smaller than prod to keep test fast
    const N_TOKENS: usize = 5;
    // Reduction-order floor is ~1e-5 on these ~0.25-scale inputs over a
    // 5-step recurrence; a layout bug lands well above this.
    const TOL: f32 = 1e-3;

    let mut gpu = init_gpu();

    // Deterministic inputs (same generators as the Q8 tree test).
    let q: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 3)).collect();
    let k: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 5)).collect();
    let v: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 7)).collect();
    let gate: Vec<f32> = (0..N_TOKENS * N_HEADS).map(|i| sin_det(i, 11) * 0.1 - 0.5).collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_HEADS).map(|i| sigmoid(sin_det(i, 13))).collect();

    // Initial S state: deterministic F32 values (no quant).
    let s_f32_init: Vec<f32> = (0..N_HEADS * HD * HD)
        .map(|i| (((i * 5381) % 251) as f32 - 125.0) * 0.004)
        .collect();

    // ---- Reference: N successive linear FP32 calls with n_tokens=1 ----
    let out_ref = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();
    let sf_ref = gpu.upload_f32(&s_f32_init, &[N_HEADS * HD * HD]).unwrap();
    for t in 0..N_TOKENS {
        let q1 = gpu.upload_f32(&q[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]).unwrap();
        let k1 = gpu.upload_f32(&k[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]).unwrap();
        let v1 = gpu.upload_f32(&v[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]).unwrap();
        let g1 = gpu.upload_f32(&gate[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS]).unwrap();
        let b1 = gpu.upload_f32(&beta[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS]).unwrap();
        let o1 = gpu.zeros(&[1, N_HEADS * HD], DType::F32).unwrap();
        // sf_ref is advanced IN PLACE each call (rolling state).
        gpu.gated_delta_net_f32(&q1, &k1, &v1, &g1, &b1, &sf_ref, &o1, 1, N_HEADS, HD).unwrap();
        let row_bytes = N_HEADS * HD * 4;
        gpu.hip.memcpy_dtod_at(&out_ref.buf, t * row_bytes, &o1.buf, 0, row_bytes).unwrap();
        for t1 in [q1, k1, v1, g1, b1, o1] {
            gpu.free_tensor(t1).unwrap();
        }
    }
    let out_ref_host = gpu.download_f32(&out_ref).unwrap();

    // ---- Tree kernel with spine parents ----
    let parents: Vec<i32> = (0..N_TOKENS as i32).map(|t| t - 1).collect();
    let q_gpu = gpu.upload_f32(&q, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let gate_gpu = gpu.upload_f32(&gate, &[N_TOKENS, N_HEADS]).unwrap();
    let beta_gpu = gpu.upload_f32(&beta, &[N_TOKENS, N_HEADS]).unwrap();
    let sf_init_tree = gpu.upload_f32(&s_f32_init, &[N_HEADS * HD * HD]).unwrap();
    let tape_f32 = gpu.zeros(&[N_TOKENS * N_HEADS * HD * HD], DType::F32).unwrap();
    let parents_gpu = upload_i32(&mut gpu, &parents);
    let out_tree = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();

    gpu.gated_delta_net_f32_tree_batch_seq(
        &q_gpu, &k_gpu, &v_gpu, &gate_gpu, &beta_gpu,
        &sf_init_tree, &tape_f32, &parents_gpu,
        &out_tree,
        N_TOKENS, N_HEADS, HD,
    ).unwrap();

    let out_tree_host = gpu.download_f32(&out_tree).unwrap();

    // Compare with tolerance.
    let mut max_diff: f32 = 0.0;
    let mut fails = 0usize;
    for (i, (x, y)) in out_ref_host.iter().zip(out_tree_host.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > TOL {
            if fails < 5 {
                eprintln!("  spine[{i}]: ref={x} tree={y} diff={d}");
            }
            fails += 1;
        }
    }
    println!(
        "spine output: max_diff={max_diff:.3e} tol={TOL:.0e} ({}/{} over tol)",
        fails,
        out_ref_host.len()
    );
    if fails > 0 {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    println!("PASS");
}

#[cfg(feature = "deltanet")]
fn init_gpu() -> rdna_compute::Gpu {
    rdna_compute::Gpu::init().expect("GPU init")
}

#[cfg(feature = "deltanet")]
fn sin_det(i: usize, mul: usize) -> f32 {
    ((((i * mul * 2654435761) % 10007) as f32 / 10007.0) - 0.5) * 0.25
}

#[cfg(feature = "deltanet")]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "deltanet")]
fn upload_i32(gpu: &mut rdna_compute::Gpu, data: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[data.len() * 4], rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
