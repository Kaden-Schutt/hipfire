// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tree-aware conv1d_silu_split kernel correctness test.
//!
//! Verifies:
//! 1. Spine parent_indices = [-1, 0, 1, 2, ...] produces byte-exact output
//!    vs the linear conv1d_silu_split_f32_n kernel.
//! 2. n=1 with parent=[-1] matches single-token decode semantics.
//! 3. Sibling topology (two tokens sharing the same parent) matches the
//!    analytic parent-chain CPU reference within 1e-5.
//!
//! Build: `cargo run --release --features deltanet --example test_conv1d_tree -p rdna-compute`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_conv1d_tree requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::{DType, Gpu};

    let mut gpu = Gpu::init().expect("GPU init failed");

    let k_dim: usize = 32;
    let v_dim: usize = 64;
    let n_ch: usize = 2 * k_dim + v_dim;

    let weight: Vec<f32> = (0..n_ch * 4)
        .map(|i| (((i * 7919) % 101) as f32 - 50.0) * 0.01)
        .collect();
    let state: Vec<f32> = (0..n_ch * 3)
        .map(|i| (((i * 2027) % 73) as f32 - 36.0) * 0.01)
        .collect();

    let w = gpu.upload_f32(&weight, &[n_ch, 4]).unwrap();
    let mut fails = 0;

    // ---------- Case 1: spine (byte-exact vs linear kernel) ---------------
    let n: usize = 6;
    let input: Vec<f32> = (0..n * n_ch)
        .map(|i| (((i * 104729) % 211) as f32 - 105.0) * 0.01)
        .collect();
    let parents: Vec<i32> = (0..n as i32).map(|t| t - 1).collect();
    fails += run_case(&mut gpu, &input, &weight, &state, &parents, k_dim, v_dim, n, "spine", true);

    // ---------- Case 2: n=1 decode ----------------------------------------
    let input1: Vec<f32> = input[..n_ch].to_vec();
    fails += run_case(&mut gpu, &input1, &weight, &state, &vec![-1i32], k_dim, v_dim, 1, "decode", true);

    // ---------- Case 3: siblings (approx vs CPU reference) ---------------
    let n_s: usize = 4;
    let input_s: Vec<f32> = input[..n_s * n_ch].to_vec();
    let parents_s: Vec<i32> = vec![-1, 0, 0, 1];
    fails += run_case(&mut gpu, &input_s, &weight, &state, &parents_s, k_dim, v_dim, n_s, "siblings", false);

    let _ = w;
    if fails > 0 {
        eprintln!("FAIL: {fails} subtests failed");
        std::process::exit(1);
    }
    println!("PASS");
}

#[cfg(feature = "deltanet")]
fn run_case(
    gpu: &mut rdna_compute::Gpu,
    input: &[f32],
    weight: &[f32],
    state: &[f32],
    parents: &[i32],
    k_dim: usize,
    v_dim: usize,
    n: usize,
    label: &str,
    compare_linear: bool,
) -> usize {
    use rdna_compute::DType;
    let n_ch = 2 * k_dim + v_dim;
    let s_lin = gpu.upload_f32(state, &[n_ch, 3]).unwrap();
    let s_tree = gpu.upload_f32(state, &[n_ch, 3]).unwrap();
    let w_gpu = gpu.upload_f32(weight, &[n_ch, 4]).unwrap();
    let x = gpu.upload_f32(input, &[n, n_ch]).unwrap();
    let p = alloc_i32(gpu, parents);

    let q_l = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let k_l = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let v_l = gpu.zeros(&[n, v_dim], DType::F32).unwrap();
    let q_t = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let k_t = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let v_t = gpu.zeros(&[n, v_dim], DType::F32).unwrap();

    gpu.conv1d_silu_split_tree_f32_n(&q_t, &k_t, &v_t, &x, &w_gpu, &s_tree, &p, k_dim, v_dim, n).unwrap();
    let qt = gpu.download_f32(&q_t).unwrap();
    let kt = gpu.download_f32(&k_t).unwrap();
    let vt = gpu.download_f32(&v_t).unwrap();

    let mut fails = 0;
    if compare_linear {
        gpu.conv1d_silu_split_f32_n(&q_l, &k_l, &v_l, &x, &w_gpu, &s_lin, k_dim, v_dim, n).unwrap();
        fails += cmp_exact(&format!("{label} q"), &gpu.download_f32(&q_l).unwrap(), &qt);
        fails += cmp_exact(&format!("{label} k"), &gpu.download_f32(&k_l).unwrap(), &kt);
        fails += cmp_exact(&format!("{label} v"), &gpu.download_f32(&v_l).unwrap(), &vt);
    } else {
        let q_ref = cpu_ref(input, weight, state, parents, k_dim, v_dim, n, 0);
        let k_ref = cpu_ref(input, weight, state, parents, k_dim, v_dim, n, 1);
        let v_ref = cpu_ref(input, weight, state, parents, k_dim, v_dim, n, 2);
        fails += cmp_approx(&format!("{label} q"), &qt, &q_ref, 1e-5);
        fails += cmp_approx(&format!("{label} k"), &kt, &k_ref, 1e-5);
        fails += cmp_approx(&format!("{label} v"), &vt, &v_ref, 1e-5);
    }
    fails
}

#[cfg(feature = "deltanet")]
fn alloc_i32(gpu: &mut rdna_compute::Gpu, data: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[data.len() * 4], rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

#[cfg(feature = "deltanet")]
fn cmp_exact(label: &str, a: &[f32], b: &[f32]) -> usize {
    let mut n = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if n < 3 { eprintln!("  {label}[{i}]: lin={x} tree={y}"); }
            n += 1;
        }
    }
    if n > 0 { eprintln!("{label}: FAIL {n}/{}", a.len()); 1 } else { println!("{label}: byte-exact"); 0 }
}

#[cfg(feature = "deltanet")]
fn cmp_approx(label: &str, a: &[f32], b: &[f32], tol: f32) -> usize {
    let mut max_err: f32 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) { max_err = max_err.max((x - y).abs()); }
    if max_err > tol { eprintln!("{label}: FAIL max_err={max_err} tol={tol}"); 1 }
    else { println!("{label}: approx-ok (max_err={max_err:.2e})"); 0 }
}

#[cfg(feature = "deltanet")]
fn cpu_ref(
    input: &[f32], weight: &[f32], state: &[f32], parents: &[i32],
    k_dim: usize, v_dim: usize, n: usize, target: u8,
) -> Vec<f32> {
    let n_ch = 2 * k_dim + v_dim;
    let out_dim = if target == 2 { v_dim } else { k_dim };
    let mut out = vec![0.0f32; n * out_dim];
    for t in 0..n {
        for c in 0..n_ch {
            let w0 = weight[c * 4];
            let w1 = weight[c * 4 + 1];
            let w2 = weight[c * 4 + 2];
            let w3 = weight[c * 4 + 3];
            let s0 = state[c * 3];
            let s1 = state[c * 3 + 1];
            let s2 = state[c * 3 + 2];
            let p1 = parents[t];
            let p2 = if p1 >= 0 { parents[p1 as usize] } else { p1 - 1 };
            let p3 = if p2 >= 0 { parents[p2 as usize] } else { p2 - 1 };
            let pick = |p: i32| -> f32 {
                if p >= 0 { input[(p as usize) * n_ch + c] }
                else if p == -1 { s0 } else if p == -2 { s1 } else { s2 }
            };
            let y = w3 * input[t * n_ch + c] + w2 * pick(p1) + w1 * pick(p2) + w0 * pick(p3);
            let r = y / (1.0 + (-y).exp());
            let (write, idx) = if c < k_dim { (target == 0, t * k_dim + c) }
                else if c < 2 * k_dim { (target == 1, t * k_dim + (c - k_dim)) }
                else { (target == 2, t * v_dim + (c - 2 * k_dim)) };
            if write { out[idx] = r; }
        }
    }
    out
}
