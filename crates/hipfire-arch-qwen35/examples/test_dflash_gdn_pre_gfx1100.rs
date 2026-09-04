// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S5-gdn-pre-tape-fusion parity gate (gfx1100-only).
//!
//! Byte-for-byte oracle for the two fused GDN pre-kernels against the exact
//! old launch sequences, on synthetic DeltaNet shapes (2 key heads, 6 value
//! heads, head_dim 128, ratio 3):
//!
//! - capture: `fused_sigmoid_alpha_gate_f32_batched` + 3 tape memcpys +
//!   `conv1d_silu_split_f32_n` + `fused_qk_l2_norm_scale_interleave_f32_batched`
//!   versus one `dflash_gdn_pre_capture_gfx1100`. Compares beta/alpha, tape
//!   rows, q_raw/k_raw/v/q/k, and conv_state.
//! - end-to-end: `gated_delta_net_q8_batch_seq` (with EF residual) on both
//!   arms' outputs from identical S state; compares attn_out, s_matrices,
//!   s_scales, and EF residual.
//! - replay: old `conv1d + in-place QK norm + repeat_interleave` versus one
//!   `dflash_gdn_pre_replay_gfx1100` from the captured tape, starting from a
//!   common restored conv state; compares q_raw/k_raw (normed, old in-place
//!   postcondition), v/q/k, conv_state, untouched alpha/beta, plus the same
//!   GDN end-to-end comparison.
//!
//! Every compared buffer is pre-poisoned, so an unwritten element fails the
//! gate. Any mismatch aborts with a nonzero exit. Non-gfx1100 exits 0 with a
//! skip note (the launchers only fuse on exact gfx1100).

use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_KEY: usize = 2;
const N_V: usize = 6;
const RATIO: usize = 3;
const K_DIM: usize = N_KEY * HD;
const V_DIM: usize = N_V * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const N_CH: usize = QKV_DIM;
const MAX_N: usize = 24;
const S_SIZE: usize = N_V * HD * HD;
const EPS: f32 = 1e-6;

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; v.len() * 4];
    for (i, f) in v.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&f.to_ne_bytes());
    }
    b
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random fill (LCG), scaled per buffer kind so sigmoid,
/// softplus, conv, and norm all see non-degenerate magnitudes.
fn fill_lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            (u as f32) * scale
        })
        .collect()
}

fn upload(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    gpu.hip
        .memcpy_htod(&t.buf, &f32s_to_bytes(v))
        .expect("upload");
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    let n = t.byte_size();
    gpu.hip
        .memcpy_htod(&t.buf, &vec![0xABu8; n])
        .expect("poison");
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("download");
    b
}

fn check_eq(name: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    if a != b {
        let mut first = 0;
        while first < a.len() && a[first] == b[first] {
            first += 1;
        }
        let af = bytes_to_f32s(&a[first..(first + 4).min(a.len())]);
        let bf = bytes_to_f32s(&b[first..(first + 4).min(b.len())]);
        panic!(
            "{name}: byte mismatch at byte {first} ({} total): old={af:?} new={bf:?}",
            a.len()
        );
    }
    eprintln!("  ok {name} ({} bytes identical)", a.len());
}

struct Arm {
    beta: GpuTensor,
    alpha: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    conv_state: GpuTensor,
    tape_qkv: GpuTensor,
    tape_alpha: GpuTensor,
    tape_beta: GpuTensor,
    attn: GpuTensor,
    s: GpuTensor,
    scales: GpuTensor,
    ef: GpuTensor,
}

impl Arm {
    fn alloc(gpu: &mut Gpu, s_dtype_size_note: bool) -> Self {
        let _ = s_dtype_size_note;
        // Q8 S state is raw bytes (s_size); mirror weights.rs allocation.
        let s_buf = gpu.hip.malloc(S_SIZE).expect("s alloc");
        gpu.hip.memset(&s_buf, 0, S_SIZE).expect("s zero");
        let s = GpuTensor {
            buf: s_buf,
            shape: vec![S_SIZE],
            dtype: DType::F32,
        };
        Self {
            beta: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).expect("beta"),
            alpha: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).expect("alpha"),
            q_raw: gpu.alloc_tensor(&[MAX_N * K_DIM], DType::F32).expect("q_raw"),
            k_raw: gpu.alloc_tensor(&[MAX_N * K_DIM], DType::F32).expect("k_raw"),
            v: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("v"),
            q: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("q"),
            k: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("k"),
            conv_state: gpu
                .alloc_tensor(&[N_CH * 3], DType::F32)
                .expect("conv_state"),
            tape_qkv: gpu
                .alloc_tensor(&[MAX_N * QKV_DIM], DType::F32)
                .expect("tape_qkv"),
            tape_alpha: gpu
                .alloc_tensor(&[MAX_N * N_V], DType::F32)
                .expect("tape_alpha"),
            tape_beta: gpu
                .alloc_tensor(&[MAX_N * N_V], DType::F32)
                .expect("tape_beta"),
            attn: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("attn"),
            s,
            scales: gpu
                .zeros(&[N_V * HD], DType::F32)
                .expect("scales"),
            ef: gpu.zeros(&[S_SIZE], DType::F16).expect("ef"),
        }
    }

    fn poison_all(&self, gpu: &Gpu) {
        poison(gpu, &self.beta);
        poison(gpu, &self.alpha);
        poison(gpu, &self.q_raw);
        poison(gpu, &self.k_raw);
        poison(gpu, &self.v);
        poison(gpu, &self.q);
        poison(gpu, &self.k);
        poison(gpu, &self.conv_state);
        poison(gpu, &self.tape_qkv);
        poison(gpu, &self.tape_alpha);
        poison(gpu, &self.tape_beta);
        poison(gpu, &self.attn);
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch_caps.is_gfx1100() {
        eprintln!("SKIP: test_dflash_gdn_pre_gfx1100 requires exact gfx1100");
        return;
    }
    eprintln!("=== dflash_gdn_pre parity (gfx1100) ===");

    // Shared inputs, uploaded identically into both arms.
    let qkv_in = gpu
        .alloc_tensor(&[MAX_N * QKV_DIM], DType::F32)
        .expect("qkv_in");
    let dt_bias = gpu.alloc_tensor(&[N_V], DType::F32).expect("dt_bias");
    let a_log = gpu.alloc_tensor(&[N_V], DType::F32).expect("a_log");
    let conv_w = gpu
        .alloc_tensor(&[N_CH * 4], DType::F32)
        .expect("conv_w");
    upload(&gpu, &qkv_in, &fill_lcg(MAX_N * QKV_DIM, 0x1234, 0.6));
    upload(&gpu, &dt_bias, &fill_lcg(N_V, 0xB1A5, 1.0));
    upload(&gpu, &a_log, &fill_lcg(N_V, 0xA106, 0.5));
    upload(&gpu, &conv_w, &fill_lcg(N_CH * 4, 0xC0DE, 0.25));

    let q_scale = 1.0 / (HD as f32).sqrt();

    for n in [1usize, 2, 8, 16] {
        for tape_offset in [0usize, 2] {
            assert!(tape_offset + n <= MAX_N);
            eprintln!("--- capture n={n} tape_offset={tape_offset} ---");
            let mut old = Arm::alloc(&mut gpu, true);
            let mut new = Arm::alloc(&mut gpu, true);
            old.poison_all(&gpu);
            new.poison_all(&gpu);

            // Identical live inputs in both arms (first n rows matter).
            let beta_in = fill_lcg(MAX_N * N_V, 0xBE7A, 2.0);
            let alpha_in = fill_lcg(MAX_N * N_V, 0xA1FA, 2.0);
            let conv_init = fill_lcg(N_CH * 3, 0x57A7, 0.2);
            for arm in [&old, &new] {
                upload(&gpu, &arm.beta, &beta_in);
                upload(&gpu, &arm.alpha, &alpha_in);
                upload(&gpu, &arm.conv_state, &conv_init);
            }

            // Old path, verbatim hook order.
            gpu.fused_sigmoid_alpha_gate_f32_batched(
                &old.beta, &old.alpha, &dt_bias, &a_log, N_V, n,
            )
            .expect("old sigmoid");
            gpu.memcpy_dtod_at_auto(
                &old.tape_qkv.buf,
                tape_offset * QKV_DIM * 4,
                &qkv_in.buf,
                0,
                n * QKV_DIM * 4,
            )
            .expect("old tape qkv");
            gpu.memcpy_dtod_at_auto(
                &old.tape_alpha.buf,
                tape_offset * N_V * 4,
                &old.alpha.buf,
                0,
                n * N_V * 4,
            )
            .expect("old tape alpha");
            gpu.memcpy_dtod_at_auto(
                &old.tape_beta.buf,
                tape_offset * N_V * 4,
                &old.beta.buf,
                0,
                n * N_V * 4,
            )
            .expect("old tape beta");
            gpu.conv1d_silu_split_f32_n(
                &old.q_raw, &old.k_raw, &old.v, &qkv_in, &conv_w, &old.conv_state,
                K_DIM, V_DIM, n,
            )
            .expect("old conv");
            gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                &old.q_raw, &old.k_raw, &old.q, &old.k, N_KEY, RATIO, HD, q_scale, EPS, n,
            )
            .expect("old qk");

            // New path: single launch.
            let fused = gpu
                .dflash_gdn_pre_capture_gfx1100(
                    &new.beta, &new.alpha, &dt_bias, &a_log, &qkv_in, &conv_w,
                    &new.conv_state, &new.q_raw, &new.k_raw, &new.v, &new.q, &new.k,
                    &new.tape_qkv, &new.tape_alpha, &new.tape_beta, N_V, N_KEY, HD,
                    K_DIM, V_DIM, QKV_DIM, n, tape_offset, q_scale, EPS,
                )
                .expect("new capture");
            assert!(fused, "capture must take the fused route on gfx1100");

            for (name, o, w) in [
                ("beta", &old.beta, &new.beta),
                ("alpha", &old.alpha, &new.alpha),
                ("tape_qkv", &old.tape_qkv, &new.tape_qkv),
                ("tape_alpha", &old.tape_alpha, &new.tape_alpha),
                ("tape_beta", &old.tape_beta, &new.tape_beta),
                ("q_raw", &old.q_raw, &new.q_raw),
                ("k_raw", &old.k_raw, &new.k_raw),
                ("v", &old.v, &new.v),
                ("q", &old.q, &new.q),
                ("k", &old.k, &new.k),
                ("conv_state", &old.conv_state, &new.conv_state),
            ] {
                check_eq(name, &download(&gpu, o), &download(&gpu, w));
            }

            // End-to-end through the untouched Q8 recurrence owner.
            for arm in [&old, &new] {
                gpu.gated_delta_net_q8_batch_seq(
                    &arm.q, &arm.k, &arm.v, &arm.alpha, &arm.beta, &arm.s,
                    &arm.scales, &arm.attn, n, N_V, HD, Some(&arm.ef),
                )
                .expect("gdn q8");
            }
            for (name, o, w) in [
                ("gdn_attn", &old.attn, &new.attn),
                ("gdn_s", &old.s, &new.s),
                ("gdn_scales", &old.scales, &new.scales),
                ("gdn_ef", &old.ef, &new.ef),
            ] {
                check_eq(name, &download(&gpu, o), &download(&gpu, w));
            }

            // Replay from the captured tape (offset 0 captures only here).
            if tape_offset == 0 {
                for n_steps in [1usize, 2, 7, 16] {
                    if n_steps > n {
                        continue;
                    }
                    eprintln!("--- replay n_steps={n_steps} (from n={n} tape) ---");
                    // Restore semantics: both arms restart conv from the same state.
                    let restored = fill_lcg(N_CH * 3, 0x5EED, 0.2);
                    upload(&gpu, &old.conv_state, &restored);
                    upload(&gpu, &new.conv_state, &restored);
                    // Re-poison replay scratch (q_raw/k_raw/v/q/k/attn) only.
                    for t in [&old.q_raw, &old.k_raw, &old.v, &old.q, &old.k, &old.attn] {
                        poison(&gpu, t);
                    }
                    for t in [&new.q_raw, &new.k_raw, &new.v, &new.q, &new.k, &new.attn] {
                        poison(&gpu, t);
                    }
                    // Fresh S state per arm.
                    for arm in [&old, &new] {
                        gpu.hip.memset(&arm.s.buf, 0, S_SIZE).expect("s rezero");
                        upload(&gpu, &arm.scales, &vec![0f32; N_V * HD]);
                    }
                    // Old replay path, verbatim replay_gdn_inner steps 1-3.
                    gpu.conv1d_silu_split_f32_n(
                        &old.q_raw, &old.k_raw, &old.v, &old.tape_qkv, &conv_w,
                        &old.conv_state, K_DIM, V_DIM, n_steps,
                    )
                    .expect("old replay conv");
                    gpu.fused_qk_l2_norm_scale_f32_batched(
                        &old.q_raw, &old.k_raw, N_KEY, HD, q_scale, EPS, n_steps,
                    )
                    .expect("old replay norm");
                    gpu.repeat_interleave_qk_f32_batched(
                        &old.q_raw, &old.k_raw, &old.q, &old.k, N_KEY, RATIO, HD,
                        n_steps,
                    )
                    .expect("old replay repeat");

                    // New replay path: single launch (alpha/beta bufs pass
                    // through untouched — GDN reads tape directly).
                    let fused = gpu
                        .dflash_gdn_pre_replay_gfx1100(
                            &new.tape_qkv, &conv_w, &new.conv_state, &new.q_raw,
                            &new.k_raw, &new.v, &new.q, &new.k, N_V, N_KEY, HD,
                            K_DIM, V_DIM, QKV_DIM, n_steps, q_scale, EPS,
                        )
                        .expect("new replay");
                    assert!(fused, "replay must take the fused route on gfx1100");

                    for (name, o, w) in [
                        ("replay_q_raw", &old.q_raw, &new.q_raw),
                        ("replay_k_raw", &old.k_raw, &new.k_raw),
                        ("replay_v", &old.v, &new.v),
                        ("replay_q", &old.q, &new.q),
                        ("replay_k", &old.k, &new.k),
                        ("replay_conv_state", &old.conv_state, &new.conv_state),
                    ] {
                        check_eq(name, &download(&gpu, o), &download(&gpu, w));
                    }
                    for arm in [&old, &new] {
                        gpu.gated_delta_net_q8_batch_seq(
                            &arm.q, &arm.k, &arm.v, &arm.tape_alpha, &arm.tape_beta,
                            &arm.s, &arm.scales, &arm.attn, n_steps, N_V, HD,
                            Some(&arm.ef),
                        )
                        .expect("replay gdn q8");
                    }
                    for (name, o, w) in [
                        ("replay_gdn_attn", &old.attn, &new.attn),
                        ("replay_gdn_s", &old.s, &new.s),
                        ("replay_gdn_scales", &old.scales, &new.scales),
                        ("replay_gdn_ef", &old.ef, &new.ef),
                    ] {
                        check_eq(name, &download(&gpu, o), &download(&gpu, w));
                    }
                }
            }

            for arm in [old, new] {
                let _ = gpu.free_tensor(arm.beta);
                let _ = gpu.free_tensor(arm.alpha);
                let _ = gpu.free_tensor(arm.q_raw);
                let _ = gpu.free_tensor(arm.k_raw);
                let _ = gpu.free_tensor(arm.v);
                let _ = gpu.free_tensor(arm.q);
                let _ = gpu.free_tensor(arm.k);
                let _ = gpu.free_tensor(arm.conv_state);
                let _ = gpu.free_tensor(arm.tape_qkv);
                let _ = gpu.free_tensor(arm.tape_alpha);
                let _ = gpu.free_tensor(arm.tape_beta);
                let _ = gpu.free_tensor(arm.attn);
                let _ = gpu.free_tensor(arm.s);
                let _ = gpu.free_tensor(arm.scales);
                let _ = gpu.free_tensor(arm.ef);
            }
        }
    }

    // Ineligible shape stays on the old path without launching.
    let scratch = gpu.alloc_tensor(&[8], DType::F32).expect("scratch");
    let ineligible = gpu
        .dflash_gdn_pre_replay_gfx1100(
            &scratch, &scratch, &scratch, &scratch, &scratch, &scratch, &scratch,
            &scratch, N_V, N_KEY, HD, K_DIM, V_DIM, QKV_DIM, 17, q_scale, EPS,
        )
        .expect("ineligible call");
    assert!(!ineligible, "n_steps=17 must decline the fused route");
    eprintln!("  ok ineligible-shape decline");
    let _ = gpu.free_tensor(scratch);

    eprintln!("PASS: dflash_gdn_pre parity (capture + replay + GDN, all byte-identical)");
}
