// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness check for `conv1d_silu_split_f32_n`.
//!
//! On gfx1151, n_tokens >= 64 routes through the parallel-token prefill kernel
//! plus a state-update kernel. This test compares Q/K/V outputs and final
//! rolling state against a CPU sequential reference.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_conv1d_gfx1151 requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_rdna::{DType, Gpu};

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

    eprintln!("\n=== decode ===");
    let input: Vec<f32> = (0..n_ch)
        .map(|i| (((i * 65_537 + 23) % 251) as f32 - 125.0) * 0.0035)
        .collect();
    let (decode_ref, decode_state_ref) = cpu_decode_ref(&input, &weight, &state);

    let d_input = gpu.upload_f32(&input, &[n_ch]).unwrap();
    let d_weight = gpu.upload_f32(&weight, &[n_ch, 4]).unwrap();
    let d_state = gpu.upload_f32(&state, &[n_ch, 3]).unwrap();
    let d_out = gpu.zeros(&[n_ch], DType::F32).unwrap();

    gpu.conv1d_decode_f32(&d_out, &d_input, &d_weight, &d_state, n_ch)
        .unwrap();

    check_close(
        "decode_out",
        &gpu.download_f32(&d_out).unwrap(),
        &decode_ref,
        1.0e-5,
    );
    check_close(
        "decode_state",
        &gpu.download_f32(&d_state).unwrap(),
        &decode_state_ref,
        0.0,
    );

    gpu.free_tensor(d_input).unwrap();
    gpu.free_tensor(d_weight).unwrap();
    gpu.free_tensor(d_state).unwrap();
    gpu.free_tensor(d_out).unwrap();

    eprintln!("\n=== fused decode gate+conv ===");
    let n_heads = v_dim / 64;
    let beta: Vec<f32> = (0..n_heads)
        .map(|i| (((i * 4099 + 7) % 197) as f32 - 98.0) * 0.01)
        .collect();
    let alpha: Vec<f32> = (0..n_heads)
        .map(|i| (((i * 6151 + 11) % 211) as f32 - 105.0) * 0.01)
        .collect();
    let dt_bias: Vec<f32> = (0..n_heads)
        .map(|i| (((i * 17 + 3) % 31) as f32 - 15.0) * 0.0025)
        .collect();
    let a_log: Vec<f32> = (0..n_heads)
        .map(|i| (((i * 19 + 5) % 29) as f32 - 14.0) * 0.003)
        .collect();
    let (beta_ref, alpha_ref) = cpu_gate_ref(&beta, &alpha, &dt_bias, &a_log);
    let (q_ref, k_ref, v_ref, state_ref) =
        cpu_fused_decode_conv_ref(&input, &weight, &state, k_dim, v_dim);

    let d_beta = gpu.upload_f32(&beta, &[n_heads]).unwrap();
    let d_alpha = gpu.upload_f32(&alpha, &[n_heads]).unwrap();
    let d_dt_bias = gpu.upload_f32(&dt_bias, &[n_heads]).unwrap();
    let d_a_log = gpu.upload_f32(&a_log, &[n_heads]).unwrap();
    let d_input = gpu.upload_f32(&input, &[n_ch]).unwrap();
    let d_weight = gpu.upload_f32(&weight, &[n_ch, 4]).unwrap();
    let d_state = gpu.upload_f32(&state, &[n_ch, 3]).unwrap();
    let d_q = gpu.zeros(&[k_dim], DType::F32).unwrap();
    let d_k = gpu.zeros(&[k_dim], DType::F32).unwrap();
    let d_v = gpu.zeros(&[v_dim], DType::F32).unwrap();

    gpu.fused_sigmoid_alpha_gate_conv1d_silu_split_f32(
        &d_beta, &d_alpha, &d_dt_bias, &d_a_log, &d_q, &d_k, &d_v, &d_input, &d_weight, &d_state,
        n_heads, k_dim, v_dim,
    )
    .unwrap();

    check_close(
        "fused_beta",
        &gpu.download_f32(&d_beta).unwrap(),
        &beta_ref,
        1.0e-6,
    );
    check_close(
        "fused_alpha",
        &gpu.download_f32(&d_alpha).unwrap(),
        &alpha_ref,
        1.0e-6,
    );
    check_close("fused_q", &gpu.download_f32(&d_q).unwrap(), &q_ref, 1.0e-5);
    check_close("fused_k", &gpu.download_f32(&d_k).unwrap(), &k_ref, 1.0e-5);
    check_close("fused_v", &gpu.download_f32(&d_v).unwrap(), &v_ref, 1.0e-5);
    check_close(
        "fused_state",
        &gpu.download_f32(&d_state).unwrap(),
        &state_ref,
        0.0,
    );

    for t in [
        d_beta, d_alpha, d_dt_bias, d_a_log, d_input, d_weight, d_state, d_q, d_k, d_v,
    ] {
        gpu.free_tensor(t).unwrap();
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
fn cpu_decode_ref(input: &[f32], weight: &[f32], state_in: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n_ch = input.len();
    let mut state = state_in.to_vec();
    let mut out = vec![0.0f32; n_ch];

    for c in 0..n_ch {
        let x = input[c];
        let s0 = state[c * 3];
        let s1 = state[c * 3 + 1];
        let s2 = state[c * 3 + 2];
        out[c] = weight[c * 4 + 3] * x
            + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1
            + weight[c * 4] * s2;

        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3] = x;
    }

    (out, state)
}

#[cfg(feature = "deltanet")]
fn cpu_fused_decode_conv_ref(
    input: &[f32],
    weight: &[f32],
    state_in: &[f32],
    k_dim: usize,
    v_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_ch = 2 * k_dim + v_dim;
    let mut state = state_in.to_vec();
    let mut q = vec![0.0f32; k_dim];
    let mut k = vec![0.0f32; k_dim];
    let mut v = vec![0.0f32; v_dim];

    for c in 0..n_ch {
        let x = input[c];
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
            q[c] = result;
        } else if c < 2 * k_dim {
            k[c - k_dim] = result;
        } else {
            v[c - 2 * k_dim] = result;
        }
    }

    (q, k, v, state)
}

#[cfg(feature = "deltanet")]
fn cpu_gate_ref(
    beta: &[f32],
    alpha: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut beta_out = beta.to_vec();
    let mut alpha_out = alpha.to_vec();

    for i in 0..beta.len() {
        beta_out[i] = 1.0 / (1.0 + (-beta_out[i]).exp());
        let biased = alpha_out[i] + dt_bias[i];
        let sp = if biased > 20.0 {
            biased
        } else if biased < -20.0 {
            biased.exp()
        } else {
            (1.0 + biased.exp()).ln()
        };
        alpha_out[i] = sp * (-a_log[i].exp());
    }

    (beta_out, alpha_out)
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
