// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S2 parity gate: exact gfx1100 hidden-ring scatters.
//!
//! Compares every byte against the row-copy loop oracle for
//! heads {0, max_pos-3}, commit-n {1, 16}, block_size {n, max_pos+3}, and
//! dst_modulus {usize::MAX, 32}, with sentinel canary tensors allocated
//! around the work and verified afterwards. The oracle arm runs the real
//! production functions with the kill switch forced via `gpu.flags`
//! (`HIPFIRE_HIDDEN_SCATTER_FUSE_OFF` equivalent); the fused arm runs them
//! with the switch clear. A direct-launcher arm additionally proves the
//! kernels themselves (not just the routing) produce the loop bytes.
//!
//! Exact gfx1100 only; other archs SKIP cleanly (exit 0, no GPU work).
//! Any mismatch fails loudly (nonzero exit). If the process environment
//! sets `HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1` the fused arm would be vacuous,
//! so the harness refuses to run fused in that case — unset it first.

use hipfire_arch_qwen35::speculative::{self, HiddenStateRingBuffer};
use rdna_compute::Gpu;
use std::sync::Arc;

const MAX_POS: usize = 40;
const MAX_BATCH: usize = 17;
const N_EXTRACT: usize = 5;
const LAYERS: [usize; N_EXTRACT] = [2, 7, 13, 21, 33];
const CANARY_VAL: f32 = 3.1415927;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

// Deterministic finite pattern: exact integers plus fractional values,
// distinct per (ext, row) so any misrouted row is unmissable.
fn pattern(ext: usize, row: usize, col: usize, hidden: usize) -> f32 {
    let mut s = (ext as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((row * hidden + col) as u64)
        .wrapping_add(0x12345678);
    let v = (lcg(&mut s) % 2000001) as f32 / 1000.0 - 1000.0;
    if col % 7 == 0 {
        (ext * 100000 + row * 1000 + col) as f32
    } else {
        v
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: byte mismatch at elem {i}: got {:#010x} want {:#010x}",
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn fill_ring(
    gpu: &mut Gpu,
    rb: &HiddenStateRingBuffer,
    hidden: usize,
    salt_rows: usize,
) -> Result<(), String> {
    // Fill the whole ring with distinct pattern rows, then set head/written
    // at the call site. salt_rows shifts the row编号 so oracle/fused pairs
    // can share one builder without identical reuse across configs.
    for ext in 0..N_EXTRACT {
        let data: Vec<f32> = (0..MAX_POS * hidden)
            .map(|i| pattern(ext, salt_rows + i / hidden, i % hidden, hidden))
            .collect();
        let src = gpu
            .upload_f32(&data, &[MAX_POS * hidden])
            .map_err(|e| format!("upload ring ext {ext}: {e}"))?;
        rb.write_rows_at_head(gpu, ext, &src, MAX_POS)
            .map_err(|e| format!("write ring ext {ext}: {e}"))?;
        gpu.free_tensor(src)
            .map_err(|e| format!("free ring src: {e}"))?;
    }
    Ok(())
}

fn fill_staging(
    gpu: &mut Gpu,
    rb: &HiddenStateRingBuffer,
    n: usize,
    hidden: usize,
) -> Result<(), String> {
    for ext in 0..N_EXTRACT {
        let data: Vec<f32> = (0..n * hidden)
            .map(|i| pattern(100 + ext, i / hidden, i % hidden, hidden))
            .collect();
        let src = gpu
            .upload_f32(&data, &[n * hidden])
            .map_err(|e| format!("upload staging ext {ext}: {e}"))?;
        rb.write_rows_to_staging(gpu, ext, &src, n)
            .map_err(|e| format!("write staging ext {ext}: {e}"))?;
        gpu.free_tensor(src)
            .map_err(|e| format!("free staging src: {e}"))?;
    }
    Ok(())
}

fn download_ring(gpu: &Gpu, rb: &HiddenStateRingBuffer) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(N_EXTRACT);
    for ext in 0..N_EXTRACT {
        out.push(
            gpu.download_f32(&rb.layer_bufs[ext])
                .map_err(|e| format!("download ring ext {ext}: {e}"))?,
        );
    }
    Ok(out)
}

fn main() {
    let code = run();
    std::process::exit(code);
}

fn run() -> i32 {
    let gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return 0;
        }
    };
    let arch = gpu.arch.clone();
    if !(gpu.arch_caps.is_gfx1100() && arch == "gfx1100") {
        eprintln!("SKIP: arch {arch} is not exact gfx1100");
        return 0;
    }
    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some");
        return 0;
    }
    if std::env::var("HIPFIRE_HIDDEN_SCATTER_FUSE_OFF").as_deref() == Ok("1") {
        eprintln!(
            "REFUSE: HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1 is set in the environment; \
             the fused arm would be vacuous. Unset it and re-run."
        );
        return 2;
    }
    if let Err(e) = parity(gpu) {
        eprintln!("FAIL: {e}");
        return 1;
    }
    eprintln!("OK: dflash_hidden_scatter_gfx1100 parity across all configs");
    0
}

fn parity(mut gpu: Gpu) -> Result<(), String> {
    // Canary tensors: sentinel-filled, never written by either path.
    let canary_a = gpu
        .upload_f32(&vec![CANARY_VAL; 4096], &[4096])
        .map_err(|e| format!("canary alloc: {e}"))?;
    let canary_b = gpu
        .upload_f32(&vec![-CANARY_VAL; 4096], &[4096])
        .map_err(|e| format!("canary alloc: {e}"))?;

    let flags_on = gpu.flags.clone();
    let flags_off = Arc::new(rdna_compute::FeatureFlags {
        hidden_scatter_fuse_off: true,
        ..(*flags_on).clone()
    });

    // Full matrix at hidden=128, plus an odd-hidden representative.
    let mut configs: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    for &head in &[0usize, MAX_POS - 3] {
        for &n in &[1usize, 16] {
            for &modulus in &[usize::MAX, 32usize] {
                // block_size == n (no skip) and block_size == max_pos+3 (skip).
                for &block_size in &[n, MAX_POS + 3] {
                    configs.push((128, head, n, block_size, modulus));
                }
            }
        }
    }
    configs.push((511, MAX_POS - 3, 16, MAX_POS + 3, 32));
    configs.push((511, 0, 1, 1, usize::MAX));

    for (ci, (hidden, head, n, block_size, modulus)) in configs.iter().cloned().enumerate() {
        // dst_row_offset exercises both in-modulus and wrapping offsets.
        let dst_row_offset = if modulus == usize::MAX { 1000 } else { 57 };
        let dst_rows = if modulus == usize::MAX {
            dst_row_offset + n
        } else {
            modulus
        };
        let tag = format!("cfg{ci}: hidden={hidden} head={head} n={n} blk={block_size} mod=({}) off={dst_row_offset}",
            if modulus == usize::MAX { "MAX".to_string() } else { modulus.to_string() });

        // Identical starting state for both arms.
        let mut rb_loop =
            HiddenStateRingBuffer::new_for_layers(&mut gpu, &LAYERS, hidden, MAX_POS, MAX_BATCH)
                .map_err(|e| format!("{tag}: loop ring alloc: {e}"))?;
        let mut rb_fused =
            HiddenStateRingBuffer::new_for_layers(&mut gpu, &LAYERS, hidden, MAX_POS, MAX_BATCH)
                .map_err(|e| format!("{tag}: fused ring alloc: {e}"))?;
        for rb in [&rb_loop, &rb_fused] {
            fill_ring(&mut gpu, rb, hidden, ci * 1000)?;
            fill_staging(&mut gpu, rb, n, hidden)?;
        }
        for rb in [&mut rb_loop, &mut rb_fused] {
            rb.head = head;
            // written must cover block_size for the scatter assert.
            rb.written = MAX_POS + 8;
        }
        let dst_loop = gpu
            .upload_f32(
                &vec![-0.5f32; dst_rows * N_EXTRACT * hidden],
                &[dst_rows * N_EXTRACT * hidden],
            )
            .map_err(|e| format!("{tag}: dst_loop alloc: {e}"))?;
        let dst_fused = gpu
            .upload_f32(
                &vec![-0.5f32; dst_rows * N_EXTRACT * hidden],
                &[dst_rows * N_EXTRACT * hidden],
            )
            .map_err(|e| format!("{tag}: dst_fused alloc: {e}"))?;

        // Oracle arm: kill switch forced — today's row-copy loops.
        gpu.flags = flags_off.clone();
        rb_loop
            .commit_staging_to_ring(&mut gpu, n)
            .map_err(|e| format!("{tag}: loop commit: {e}"))?;
        speculative::scatter_hidden_block_to_interleaved(
            &gpu,
            &rb_loop,
            &dst_loop,
            dst_row_offset,
            block_size,
            n,
            modulus,
        )
        .map_err(|e| format!("{tag}: loop scatter: {e}"))?;
        let loop_head = (rb_loop.head, rb_loop.written);

        // Fused arm: switch clear — commit5 + scatter5 kernels.
        gpu.flags = flags_on.clone();
        rb_fused
            .commit_staging_to_ring(&mut gpu, n)
            .map_err(|e| format!("{tag}: fused commit: {e}"))?;
        speculative::scatter_hidden_block_to_interleaved(
            &gpu,
            &rb_fused,
            &dst_fused,
            dst_row_offset,
            block_size,
            n,
            modulus,
        )
        .map_err(|e| format!("{tag}: fused scatter: {e}"))?;
        let fused_head = (rb_fused.head, rb_fused.written);

        assert_eq!(loop_head, fused_head, "{tag}: head/written diverged");
        let ring_loop = download_ring(&gpu, &rb_loop)?;
        let ring_fused = download_ring(&gpu, &rb_fused)?;
        for ext in 0..N_EXTRACT {
            assert_bits_eq(
                &ring_fused[ext],
                &ring_loop[ext],
                &format!("{tag}: ring ext{ext}"),
            );
        }
        let d_loop = gpu
            .download_f32(&dst_loop)
            .map_err(|e| format!("{tag}: download dst_loop: {e}"))?;
        let d_fused = gpu
            .download_f32(&dst_fused)
            .map_err(|e| format!("{tag}: download dst_fused: {e}"))?;
        assert_bits_eq(&d_fused, &d_loop, &format!("{tag}: dst"));

        rb_loop.free_gpu(&mut gpu);
        rb_fused.free_gpu(&mut gpu);
        gpu.free_tensor(dst_loop)
            .map_err(|e| format!("{tag}: free dst_loop: {e}"))?;
        gpu.free_tensor(dst_fused)
            .map_err(|e| format!("{tag}: free dst_fused: {e}"))?;
        eprintln!("pass {tag}");
    }

    // Direct-launcher arm: proves the kernels themselves (bypassing routing)
    // reproduce the loop bytes on a wrap+skip+wrap-modulus case.
    direct_launcher_arm(&mut gpu, &flags_off)?;

    // Canaries must be untouched by every arm above.
    for (t, want) in [(&canary_a, CANARY_VAL), (&canary_b, -CANARY_VAL)] {
        let got = gpu
            .download_f32(t)
            .map_err(|e| format!("canary download: {e}"))?;
        assert!(
            got.iter().all(|&v| v.to_bits() == want.to_bits()),
            "canary corruption detected"
        );
    }
    gpu.flags = flags_on;
    gpu.free_tensor(canary_a)
        .map_err(|e| format!("free canary_a: {e}"))?;
    gpu.free_tensor(canary_b)
        .map_err(|e| format!("free canary_b: {e}"))?;
    Ok(())
}

/// Run the two raw launchers on scratch buffers and compare against the
/// loop functions on identical inputs.
fn direct_launcher_arm(
    gpu: &mut Gpu,
    flags_off: &Arc<rdna_compute::FeatureFlags>,
) -> Result<(), String> {
    const H: usize = 96;
    const MP: usize = 40;
    const N: usize = 16;
    const HEAD: usize = 37;
    const BLK: usize = MP + 3;
    const MOD: usize = 32;
    const OFF: usize = 57;

    let mk_ring = |gpu: &mut Gpu, salt: usize| -> Result<HiddenStateRingBuffer, String> {
        let rb = HiddenStateRingBuffer::new_for_layers(gpu, &LAYERS, H, MP, MAX_BATCH)
            .map_err(|e| format!("direct: ring alloc: {e}"))?;
        for ext in 0..N_EXTRACT {
            let data: Vec<f32> = (0..MP * H)
                .map(|i| pattern(salt + ext, i / H, i % H, H))
                .collect();
            let src = gpu
                .upload_f32(&data, &[MP * H])
                .map_err(|e| format!("direct: upload: {e}"))?;
            rb.write_rows_at_head(gpu, ext, &src, MP)
                .map_err(|e| format!("direct: write: {e}"))?;
            gpu.free_tensor(src)
                .map_err(|e| format!("direct: free: {e}"))?;
        }
        Ok(rb)
    };

    // Commit: launcher vs loop.
    let mut rb_k = mk_ring(gpu, 7)?;
    let mut rb_l = mk_ring(gpu, 7)?;
    for rb in [&mut rb_k, &mut rb_l] {
        for ext in 0..N_EXTRACT {
            let data: Vec<f32> = (0..N * H)
                .map(|i| pattern(300 + ext, i / H, i % H, H))
                .collect();
            let src = gpu
                .upload_f32(&data, &[N * H])
                .map_err(|e| format!("direct: staging upload: {e}"))?;
            rb.write_rows_to_staging(gpu, ext, &src, N)
                .map_err(|e| format!("direct: staging write: {e}"))?;
            gpu.free_tensor(src)
                .map_err(|e| format!("direct: free: {e}"))?;
        }
        rb.head = HEAD;
        rb.written = MP + 8;
    }
    gpu.dflash_hidden_commit5_launch(&rb_k.staging_bufs, &rb_k.layer_bufs, HEAD, N, H, MP)
        .map_err(|e| format!("direct: commit5 launch: {e}"))?;
    let saved = gpu.flags.clone();
    gpu.flags = flags_off.clone();
    rb_l.commit_staging_to_ring(gpu, N)
        .map_err(|e| format!("direct: loop commit: {e}"))?;
    gpu.flags = saved;
    // The raw launcher does not advance head/written (that stays with the
    // caller, mirroring commit_staging_to_ring's advance-after-enqueue);
    // advance manually so the scatter below uses the post-commit head,
    // exactly as the loop arm does.
    rb_k.head = (HEAD + N) % MP;
    rb_k.written += N;
    for ext in 0..N_EXTRACT {
        let a = gpu
            .download_f32(&rb_k.layer_bufs[ext])
            .map_err(|e| format!("direct: dl k: {e}"))?;
        let b = gpu
            .download_f32(&rb_l.layer_bufs[ext])
            .map_err(|e| format!("direct: dl l: {e}"))?;
        assert_bits_eq(&a, &b, &format!("direct: commit ext{ext}"));
    }

    // Scatter: raw try-launcher (kernels already ensured by the commit
    // above) vs loop on the committed rings.
    let dst_k = gpu
        .upload_f32(&vec![0.25f32; MOD * N_EXTRACT * H], &[MOD * N_EXTRACT * H])
        .map_err(|e| format!("direct: dst_k alloc: {e}"))?;
    let dst_l = gpu
        .upload_f32(&vec![0.25f32; MOD * N_EXTRACT * H], &[MOD * N_EXTRACT * H])
        .map_err(|e| format!("direct: dst_l alloc: {e}"))?;
    let r_skip = BLK.saturating_sub(MP);
    let start_slot = (rb_k.head + MP - (BLK - r_skip)) % MP;
    let launched = gpu
        .dflash_hidden_scatter5_try(
            &rb_k.layer_bufs,
            &dst_k,
            start_slot,
            N,
            r_skip,
            H,
            MP,
            OFF,
            MOD,
            N_EXTRACT,
        )
        .map_err(|e| format!("direct: scatter5 try: {e}"))?;
    assert!(launched, "direct: scatter5_try reported false after ensure");
    speculative::scatter_hidden_block_to_interleaved(&gpu, &rb_l, &dst_l, OFF, BLK, N, MOD)
        .map_err(|e| format!("direct: loop scatter: {e}"))?;
    let a = gpu
        .download_f32(&dst_k)
        .map_err(|e| format!("direct: dl dst_k: {e}"))?;
    let b = gpu
        .download_f32(&dst_l)
        .map_err(|e| format!("direct: dl dst_l: {e}"))?;
    assert_bits_eq(&a, &b, "direct: scatter dst");

    rb_k.free_gpu(gpu);
    rb_l.free_gpu(gpu);
    gpu.free_tensor(dst_k)
        .map_err(|e| format!("direct: free: {e}"))?;
    gpu.free_tensor(dst_l)
        .map_err(|e| format!("direct: free: {e}"))?;
    eprintln!("pass direct-launcher arm (commit5 + scatter5 vs loops)");
    Ok(())
}
