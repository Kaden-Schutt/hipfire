// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tree-aware gated_delta_net_q8 correctness test.
//!
//! Verifies: for spine topology (parent_indices = [-1, 0, 1, 2, ...]),
//! tree-GDN produces outputs byte-exact with calling linear GDN N times
//! in sequence (n_tokens=1 each call) on the rolling s_q8 state.
//!
//! The linear batch (n_tokens>1) kernel keeps S in F32 LDS across the
//! whole batch and requants once at the end — that's qualitatively
//! different, so it's NOT our reference. The reference is the per-token
//! linear decode path, which matches the tree-GDN spine semantics.
//!
//! Build: `cargo run --release --features deltanet \
//!   --example test_gated_delta_net_tree -p rdna-compute`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_gated_delta_net_tree requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::{DType, Gpu, GpuTensor};
    const HD: usize = 128;
    const N_HEADS: usize = 4;          // smaller than prod (16) to keep test fast
    const N_TOKENS: usize = 5;

    let mut gpu = Gpu::init().expect("GPU init");

    // Deterministic inputs.
    let q: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 3)).collect();
    let k: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 5)).collect();
    let v: Vec<f32> = (0..N_TOKENS * N_HEADS * HD).map(|i| sin_det(i, 7)).collect();
    let gate: Vec<f32> = (0..N_TOKENS * N_HEADS).map(|i| sin_det(i, 11) * 0.1 - 0.5).collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_HEADS).map(|i| sigmoid(sin_det(i, 13))).collect();

    // Initial S state: random-ish Q8 values + scales.
    let s_q8_init: Vec<i8> = (0..N_HEADS * HD * HD)
        .map(|i| (((i * 5381) % 251) as i32 - 125) as i8)
        .collect();
    let s_scales_init: Vec<f32> = (0..N_HEADS * HD)
        .map(|i| 0.005 + ((i * 97) % 13) as f32 * 0.002)
        .collect();

    // ---- Reference: N successive linear calls with n_tokens=1 ----
    let q_gpu    = upload_f32(&mut gpu, &q,    &[N_TOKENS, N_HEADS * HD]);
    let k_gpu    = upload_f32(&mut gpu, &k,    &[N_TOKENS, N_HEADS * HD]);
    let v_gpu    = upload_f32(&mut gpu, &v,    &[N_TOKENS, N_HEADS * HD]);
    let gate_gpu = upload_f32(&mut gpu, &gate, &[N_TOKENS, N_HEADS]);
    let beta_gpu = upload_f32(&mut gpu, &beta, &[N_TOKENS, N_HEADS]);

    let sq_ref = upload_i8(&mut gpu, &s_q8_init, &[N_HEADS * HD * HD]);
    let sc_ref = upload_f32(&mut gpu, &s_scales_init, &[N_HEADS * HD]);
    let out_ref = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();

    // Slice the input/output tensors per-token by calling the existing
    // batch_seq kernel with n_tokens=1 at sliding pointer offsets. Since
    // we can't offset tensor pointers via the high-level API, we upload
    // each token's slice into a 1-token tensor and copy the output back.
    for t in 0..N_TOKENS {
        let q1 = upload_f32(&mut gpu, &q[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]);
        let k1 = upload_f32(&mut gpu, &k[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]);
        let v1 = upload_f32(&mut gpu, &v[t * N_HEADS * HD..(t + 1) * N_HEADS * HD], &[1, N_HEADS * HD]);
        let g1 = upload_f32(&mut gpu, &gate[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS]);
        let b1 = upload_f32(&mut gpu, &beta[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS]);
        let o1 = gpu.zeros(&[1, N_HEADS * HD], DType::F32).unwrap();
        gpu.gated_delta_net_q8_batch_seq(&q1, &k1, &v1, &g1, &b1, &sq_ref, &sc_ref, &o1, 1, N_HEADS, HD, None).unwrap();
        // Scatter o1 back into out_ref[t].
        let row_bytes = N_HEADS * HD * 4;
        gpu.hip.memcpy_dtod_at(&out_ref.buf, t * row_bytes, &o1.buf, 0, row_bytes).unwrap();
        gpu.free_tensor(q1).unwrap();
        gpu.free_tensor(k1).unwrap();
        gpu.free_tensor(v1).unwrap();
        gpu.free_tensor(g1).unwrap();
        gpu.free_tensor(b1).unwrap();
        gpu.free_tensor(o1).unwrap();
    }
    let out_ref_host = gpu.download_f32(&out_ref).unwrap();

    // ---- Tree kernel with spine parents ----
    let parents: Vec<i32> = (0..N_TOKENS as i32).map(|t| t - 1).collect();

    let sq_init_tree = upload_i8(&mut gpu, &s_q8_init, &[N_HEADS * HD * HD]);
    let sc_init_tree = upload_f32(&mut gpu, &s_scales_init, &[N_HEADS * HD]);
    let tape_q8 = gpu.alloc_tensor(&[N_TOKENS * N_HEADS * HD * HD], DType::Raw).unwrap();
    let tape_sc = gpu.zeros(&[N_TOKENS * N_HEADS * HD], DType::F32).unwrap();
    let parents_gpu = upload_i32(&mut gpu, &parents);
    let out_tree = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();

    gpu.gated_delta_net_q8_tree_batch_seq(
        &q_gpu, &k_gpu, &v_gpu, &gate_gpu, &beta_gpu,
        &sq_init_tree, &sc_init_tree,
        &tape_q8, &tape_sc, &parents_gpu,
        &out_tree,
        N_TOKENS, N_HEADS, HD,
    ).unwrap();

    let out_tree_host = gpu.download_f32(&out_tree).unwrap();

    // Compare outputs byte-exact.
    let fails = cmp_exact("spine output", &out_ref_host, &out_tree_host);
    if fails > 0 {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    println!("PASS");
}

#[cfg(feature = "deltanet")]
fn sin_det(i: usize, mul: usize) -> f32 {
    ((((i * mul * 2654435761) % 10007) as f32 / 10007.0) - 0.5) * 0.25
}

#[cfg(feature = "deltanet")]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(feature = "deltanet")]
fn upload_f32(gpu: &mut rdna_compute::Gpu, data: &[f32], shape: &[usize]) -> rdna_compute::GpuTensor {
    gpu.upload_f32(data, shape).unwrap()
}

#[cfg(feature = "deltanet")]
fn upload_i8(gpu: &mut rdna_compute::Gpu, data: &[i8], shape: &[usize]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(shape, rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

#[cfg(feature = "deltanet")]
fn upload_i32(gpu: &mut rdna_compute::Gpu, data: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[data.len() * 4], rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

#[cfg(feature = "deltanet")]
fn cmp_exact(label: &str, a: &[f32], b: &[f32]) -> usize {
    let mut n = 0;
    let mut max_diff: f32 = 0.0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            let d = (x - y).abs();
            if d > max_diff { max_diff = d; }
            if n < 5 { eprintln!("  {label}[{i}]: ref={x} tree={y} diff={d}"); }
            n += 1;
        }
    }
    if n > 0 {
        eprintln!("{label}: FAIL {n}/{} (max_diff={max_diff})", a.len());
        1
    } else {
        println!("{label}: byte-exact");
        0
    }
}
