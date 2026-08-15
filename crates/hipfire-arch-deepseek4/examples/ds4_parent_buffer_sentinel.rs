// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Sentinel test: separate SWA / compressed buffers cannot cross-resolve.
//!
//! Reference `model.py` keeps one contiguous `kv_cache` with
//! `compressor.kv_cache = kv_cache[:, win:]` and a unified index space
//! (`offset = seqlen` prefill / `win` decode). Parent keeps SWA and
//! compressed as **separate** tensors + `offset = 0` into compressed.
//!
//! This harness proves by construction on hardware:
//! 1. Gather from `main_kv` with index `j` always returns compressed row `j`
//!    (never SWA row `j`), including when `j` is a legal SWA column.
//! 2. Joint softmax kernel reads SWA columns only from the SWA buffer and
//!    top-k columns only from the gathered top-k buffer.
//!
//! No checkpoint load. gfx942 only.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_buffer_sentinel
//! ```
use rdna_compute::{DType, Gpu, GpuTensor};
use std::process::ExitCode;

const HEAD_DIM: usize = 512;
const N_HEADS: usize = 1;
const SWA_WINDOW: usize = 16;
const TOPK_WINDOW: usize = 8;
const N_COMPRESSED: usize = 8;
const BATCH: usize = 2;

/// SWA column c carries value `SWA_BASE + c` on every dim.
const SWA_BASE: f32 = 1000.0;
/// Compressed row j carries value `COMP_BASE + j` on every dim.
const COMP_BASE: f32 = 2000.0;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    println!("=== ds4_parent_buffer_sentinel ===");
    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err("gfx942 required".to_owned());
    }

    // ── 1. Gather sentinel ──────────────────────────────────────────────
    let mut main_kv = vec![0.0f32; N_COMPRESSED * HEAD_DIM];
    for j in 0..N_COMPRESSED {
        for d in 0..HEAD_DIM {
            main_kv[j * HEAD_DIM + d] = COMP_BASE + j as f32;
        }
    }
    let topk_idx: Vec<i32> = {
        let row = [0i32, 5, 7, -1, 3, 5, 1, 2];
        let mut v = Vec::with_capacity(BATCH * TOPK_WINDOW);
        for _ in 0..BATCH {
            v.extend_from_slice(&row);
        }
        v
    };
    let main_kv_t = upload_f32(&mut gpu, &main_kv, &[N_COMPRESSED, HEAD_DIM])?;
    let topk_idx_t = upload_i32(&mut gpu, &topk_idx, &[BATCH, TOPK_WINDOW])?;
    let topk_out = gpu
        .zeros(&[BATCH, HEAD_DIM, TOPK_WINDOW], DType::F32)
        .map_err(|e| format!("topk_out alloc: {e:?}"))?;

    gpu.deepseek4_topk_kv_gather_batched_f32(
        &main_kv_t,
        &topk_idx_t,
        &topk_out,
        TOPK_WINDOW as i32,
        HEAD_DIM as i32,
        N_COMPRESSED as i32,
        TOPK_WINDOW as i32,
        /*col_offset=*/ 0,
        /*scale=*/ 1.0,
        BATCH as i32,
    )
    .map_err(|e| format!("gather: {e:?}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    let gathered = download_f32(&gpu, &topk_out, BATCH * HEAD_DIM * TOPK_WINDOW)?;
    println!("gather: topk_idx row = {:?}", &topk_idx[..TOPK_WINDOW]);
    for (k, &idx) in topk_idx[..TOPK_WINDOW].iter().enumerate() {
        let got = gathered[k]; // b=0, d=0, col k  → layout [B, D, K]
        let expect = if idx >= 0 {
            COMP_BASE + idx as f32
        } else {
            0.0
        };
        if (got - expect).abs() > 1e-5 {
            return Err(format!(
                "gather col {k} idx={idx}: got {got} expect {expect} \
                 (SWA bleed would look like {})",
                SWA_BASE + idx.max(0) as f32
            ));
        }
        println!("  col {k:>2} idx={idx:>2} → {got:.1} (expect {expect:.1})");
    }
    let idx5_got = gathered[1]; // col 1 = idx 5
    assert!((idx5_got - (COMP_BASE + 5.0)).abs() < 1e-5);
    assert!((idx5_got - (SWA_BASE + 5.0)).abs() > 100.0);
    println!("PASS gather: index 5 → compressed row 5 (2005), not SWA col 5 (1005)");

    // ── 2. Joint-softmax buffer separation ──────────────────────────────
    let mut swa = vec![0.0f32; BATCH * HEAD_DIM * SWA_WINDOW];
    for b in 0..BATCH {
        for d in 0..HEAD_DIM {
            for c in 0..SWA_WINDOW {
                swa[((b * HEAD_DIM + d) * SWA_WINDOW) + c] = SWA_BASE + c as f32;
            }
        }
    }
    let mut topk = vec![0.0f32; BATCH * HEAD_DIM * TOPK_WINDOW];
    for b in 0..BATCH {
        for d in 0..HEAD_DIM {
            for k in 0..TOPK_WINDOW {
                let idx = topk_idx[b * TOPK_WINDOW + k];
                let v = if idx >= 0 {
                    COMP_BASE + idx as f32
                } else {
                    0.0
                };
                topk[((b * HEAD_DIM + d) * TOPK_WINDOW) + k] = v;
            }
        }
    }
    let q = vec![1.0f32; BATCH * N_HEADS * HEAD_DIM];
    let sink = vec![-1.0e9f32; N_HEADS];
    let n_valid = vec![SWA_WINDOW as i32; BATCH];
    let n_active = vec![4i32; BATCH];

    let q_t = upload_f32(&mut gpu, &q, &[BATCH, N_HEADS, HEAD_DIM])?;
    let swa_t = upload_f32(&mut gpu, &swa, &[BATCH, HEAD_DIM, SWA_WINDOW])?;
    let topk_t = upload_f32(&mut gpu, &topk, &[BATCH, HEAD_DIM, TOPK_WINDOW])?;
    let sink_t = upload_f32(&mut gpu, &sink, &[N_HEADS])?;
    let n_valid_t = upload_i32(&mut gpu, &n_valid, &[BATCH])?;
    let n_active_t = upload_i32(&mut gpu, &n_active, &[BATCH])?;
    let out_t = gpu
        .zeros(&[BATCH, N_HEADS, HEAD_DIM], DType::F32)
        .map_err(|e| format!("out alloc: {e:?}"))?;

    gpu.deepseek4_attn_swa_topk_batched_f32(
        &q_t,
        &swa_t,
        &swa_t,
        &topk_t,
        &topk_t,
        &sink_t,
        &n_valid_t,
        &n_active_t,
        &out_t,
        N_HEADS as i32,
        HEAD_DIM as i32,
        SWA_WINDOW as i32,
        TOPK_WINDOW as i32,
        BATCH as i32,
    )
    .map_err(|e| format!("joint attn: {e:?}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    let out = download_f32(&gpu, &out_t, BATCH * N_HEADS * HEAD_DIM)?;
    let o0 = out[0];
    println!(
        "joint attn out[0]={o0:.4}  (expect ~{} COMP-max; SWA-max={}; SWA-col7={})",
        COMP_BASE + 7.0,
        SWA_BASE + (SWA_WINDOW as f32 - 1.0),
        SWA_BASE + 7.0
    );
    if (o0 - (COMP_BASE + 7.0)).abs() > 1.0 {
        return Err(format!(
            "joint attn output {o0} not in COMP band — buffer cross-read suspected"
        ));
    }
    if (o0 - (SWA_BASE + 7.0)).abs() < 50.0 {
        return Err(format!(
            "joint attn output {o0} looks like SWA col 7 — compressed resolved into SWA"
        ));
    }
    println!("PASS joint: output in COMP band ({o0:.1}); buffers do not cross-resolve");

    // ── 3. n_active=0 → pure SWA mass ───────────────────────────────────
    let n_active0 = vec![0i32; BATCH];
    let n_active0_t = upload_i32(&mut gpu, &n_active0, &[BATCH])?;
    let out0_t = gpu
        .zeros(&[BATCH, N_HEADS, HEAD_DIM], DType::F32)
        .map_err(|e| format!("out0 alloc: {e:?}"))?;
    gpu.deepseek4_attn_swa_topk_batched_f32(
        &q_t,
        &swa_t,
        &swa_t,
        &topk_t,
        &topk_t,
        &sink_t,
        &n_valid_t,
        &n_active0_t,
        &out0_t,
        N_HEADS as i32,
        HEAD_DIM as i32,
        SWA_WINDOW as i32,
        TOPK_WINDOW as i32,
        BATCH as i32,
    )
    .map_err(|e| format!("joint attn n_active=0: {e:?}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let out0 = download_f32(&gpu, &out0_t, BATCH * N_HEADS * HEAD_DIM)?;
    let s0 = out0[0];
    let swa_hi = SWA_BASE + (SWA_WINDOW as f32 - 1.0);
    println!("joint attn n_active=0 out[0]={s0:.4}  (expect ~{swa_hi})");
    if (s0 - swa_hi).abs() > 1.0 {
        return Err(format!(
            "n_active=0 output {s0} not in SWA band — topk leaked into SWA-only path"
        ));
    }
    println!("PASS n_active=0: output in SWA band only");

    println!();
    println!(
        "VERDICT: index j is unambiguous — SWA col j and compressed row j live in \
         separate tensors; joint kernel never unifies them. offset=0 into \
         main_kv_cache is structurally safe against SWA/compressed aliasing."
    );
    Ok(())
}

fn upload_f32(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(data, shape)
        .map_err(|e| format!("upload_f32: {e:?}"))
}

fn upload_i32(gpu: &mut Gpu, data: &[i32], shape: &[usize]) -> Result<GpuTensor, String> {
    let mut bytes = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let t = gpu
        .alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("alloc i32: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&t.buf, &bytes)
        .map_err(|e| format!("memcpy i32: {e:?}"))?;
    Ok(t)
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!(
            "download too small have {} need {nbytes}",
            t.buf.size()
        ));
    }
    let mut data = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("download: {e:?}"))?;
    Ok(data)
}
