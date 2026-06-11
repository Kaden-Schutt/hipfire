// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness check for `conv1d_silu_split_f32_n`.
//!
//! On gfx1151, n_tokens >= 4 routes through the parallel-token prefill kernel
//! plus a state-update kernel. This test compares Q/K/V outputs and final
//! rolling state against a CPU sequential reference.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_conv1d_gfx1151 requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::{DType, Gpu};

    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let k_dim = 128usize;
    let v_dim = 256usize;
    let n_ch = 2 * k_dim + v_dim;
    let weight: Vec<f32> = (0..n_ch * 4)
        .map(|i| (((i * 7919 + 17) % 127) as f32 - 63.0) * 0.00625)
        .collect();
    let state: Vec<f32> = (0..n_ch * 3)
        .map(|i| (((i * 2027 + 31) % 113) as f32 - 56.0) * 0.0078125)
        .collect();

    for &n_tokens in &[1usize, 2, 3, 4, 6, 17, 64] {
        eprintln!("\n=== n_tokens={n_tokens} ===");
        let input: Vec<f32> = (0..n_tokens * n_ch)
            .map(|i| (((i * 104_729 + 19) % 257) as f32 - 128.0) * 0.004)
            .collect();
        let (q_ref, k_ref, v_ref, state_ref) =
            cpu_ref(&input, &weight, &state, k_dim, v_dim, n_tokens);

        let d_input = gpu.upload_f32(&input, &[n_tokens, n_ch]).unwrap();
        let d_weight = gpu.upload_f32(&weight, &[n_ch, 4]).unwrap();
        let d_state = gpu.upload_f32(&state, &[n_ch, 3]).unwrap();
        let d_q = gpu.zeros(&[n_tokens, k_dim], DType::F32).unwrap();
        let d_k = gpu.zeros(&[n_tokens, k_dim], DType::F32).unwrap();
        let d_v = gpu.zeros(&[n_tokens, v_dim], DType::F32).unwrap();

        gpu.conv1d_silu_split_f32_n(
            &d_q, &d_k, &d_v, &d_input, &d_weight, &d_state, k_dim, v_dim, n_tokens,
        )
        .unwrap();

        check_close("q", &gpu.download_f32(&d_q).unwrap(), &q_ref, 1.0e-5);
        check_close("k", &gpu.download_f32(&d_k).unwrap(), &k_ref, 1.0e-5);
        check_close("v", &gpu.download_f32(&d_v).unwrap(), &v_ref, 1.0e-5);
        check_close(
            "state",
            &gpu.download_f32(&d_state).unwrap(),
            &state_ref,
            0.0,
        );

        gpu.free_tensor(d_input).unwrap();
        gpu.free_tensor(d_weight).unwrap();
        gpu.free_tensor(d_state).unwrap();
        gpu.free_tensor(d_q).unwrap();
        gpu.free_tensor(d_k).unwrap();
        gpu.free_tensor(d_v).unwrap();
    }
}

#[cfg(feature = "deltanet")]
fn cpu_ref(
    input: &[f32],
    weight: &[f32],
    state_in: &[f32],
    k_dim: usize,
    v_dim: usize,
    n_tokens: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_ch = 2 * k_dim + v_dim;
    let mut state = state_in.to_vec();
    let mut q = vec![0.0f32; n_tokens * k_dim];
    let mut k = vec![0.0f32; n_tokens * k_dim];
    let mut v = vec![0.0f32; n_tokens * v_dim];

    for t in 0..n_tokens {
        for c in 0..n_ch {
            let x = input[t * n_ch + c];
            let s0 = state[c * 3];
            let s1 = state[c * 3 + 1];
            let s2 = state[c * 3 + 2];
            let y = weight[c * 4 + 3] * x
                + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1
                + weight[c * 4] * s2;
            let result = y / (1.0 + (-y).exp());

            state[c * 3 + 2] = s1;
            state[c * 3 + 1] = s0;
            state[c * 3] = x;

            if c < k_dim {
                q[t * k_dim + c] = result;
            } else if c < 2 * k_dim {
                k[t * k_dim + (c - k_dim)] = result;
            } else {
                v[t * v_dim + (c - 2 * k_dim)] = result;
            }
        }
    }

    (q, k, v, state)
}

#[cfg(feature = "deltanet")]
fn check_close(label: &str, got: &[f32], expected: &[f32], tol: f32) {
    let mut max_abs = 0.0f32;
    let mut mismatches = 0usize;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs = (g - e).abs();
        max_abs = max_abs.max(abs);
        if abs > tol {
            if mismatches < 3 {
                eprintln!("  {label}[{i}]: got={g:.8e} expected={e:.8e} abs={abs:.3e}");
            }
            mismatches += 1;
        }
    }
    eprintln!(
        "{label}: max_abs={max_abs:.6e} mismatches={mismatches}/{}",
        got.len()
    );
    assert_eq!(
        mismatches, 0,
        "{label}: max_abs {max_abs:.6e} exceeds {tol:.1e}"
    );
}
