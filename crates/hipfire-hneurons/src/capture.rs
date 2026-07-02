// SPDX-License-Identifier: Apache-2.0
//! CETT capture session + the arch-agnostic FFN forward tap.
//!
//! A dense-transformer forward calls [`maybe_capture_ffn`] right after its
//! `down_proj`, passing the down_proj INPUT (the FFN neuron activations) and its
//! OUTPUT. When a capture session is active, the hook accumulates each RESPONSE
//! token's per-neuron CETT into a GPU-resident per-layer sum via
//! [`Gpu::cett_accumulate_layer`] — NO per-layer host download (which used to
//! force ~2 device syncs per layer and serialize the capture forward; ~90% of
//! capture time). Only the reduced `[layers × intermediate]` sum is downloaded
//! once, at [`finish_capture`]. Placement is one line per arch (like the steer
//! hook). Process-global (single-threaded daemon dispatch), gated by a fast
//! `ACTIVE` atomic so a normal forward pays only one relaxed load.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor, OwnedTensor};

struct CaptureAcc {
    /// `[num_layers * intermediate]` column norms `‖W_down[:,j]‖`, uploaded once.
    col_norm_gpu: OwnedTensor,
    /// `[num_layers * intermediate]` running CETT sums (RAII: freed on drop).
    sums_gpu: OwnedTensor,
    num_layers: usize,
    intermediate: usize,
    hidden: usize,
    /// First GLOBAL token position belonging to the response.
    response_start: usize,
    /// Number of response tokens folded so far (same for every layer).
    count: usize,
}

// The accumulator holds `OwnedTensor` (not `Sync`), and the whole capture —
// begin → tapped forward → finish — runs inline on ONE daemon dispatch thread,
// so it lives in thread-local storage. Only the `ACTIVE` gate is a shared atomic
// so a normal (non-capturing) forward pays a single relaxed load.
thread_local! {
    static CAPTURE: RefCell<Option<CaptureAcc>> = const { RefCell::new(None) };
}
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Whether a CETT capture session is active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Begin a CETT capture session. `col_norms[layer]` is `‖W_down[:,j]‖` per
/// neuron; `response_start` is the first response token position (prompt
/// positions before it are ignored); `hidden` is the residual width. Uploads the
/// column norms and allocates a zeroed GPU sum accumulator.
pub fn begin_capture(
    gpu: &mut Gpu,
    col_norms: Vec<Vec<f32>>,
    response_start: usize,
    hidden: usize,
) -> HipResult<()> {
    let num_layers = col_norms.len();
    let intermediate = col_norms.first().map_or(0, |c| c.len());
    let n = (num_layers * intermediate).max(1);
    // Flatten row-major and upload once.
    let mut flat: Vec<f32> = Vec::with_capacity(n);
    for row in &col_norms {
        flat.extend_from_slice(row);
    }
    flat.resize(n, 0.0);
    let col_norm_gpu = gpu.alloc_owned(&[n], DType::F32)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
    gpu.hip.memcpy_htod(&col_norm_gpu.buf, bytes)?;
    let sums_gpu = gpu.alloc_owned(&[n], DType::F32)?;
    gpu.hip.memset(&sums_gpu.buf, 0, sums_gpu.buf.size())?;
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(CaptureAcc {
            col_norm_gpu,
            sums_gpu,
            num_layers,
            intermediate,
            hidden,
            response_start,
            count: 0,
        });
    });
    ACTIVE.store(true, Ordering::Release);
    Ok(())
}

/// End the session and return the per-layer mean CETT feature
/// (`[layers][neurons]`), or `None` if no capture was active. Downloads the GPU
/// sum accumulator once and divides by the response-token count.
pub fn finish_capture(gpu: &mut Gpu) -> HipResult<Option<Vec<Vec<f32>>>> {
    // Take the acc out of TLS (clears the session) so the RefCell borrow isn't
    // held across the download and the OwnedTensors drop at end of scope.
    let acc = CAPTURE.with(|c| c.borrow_mut().take());
    ACTIVE.store(false, Ordering::Release);
    let Some(acc) = acc else {
        return Ok(None);
    };
    let flat = gpu.download_f32(&acc.sums_gpu)?;
    let (nl, i) = (acc.num_layers, acc.intermediate);
    let denom = acc.count.max(1) as f32;
    let mut feature = Vec::with_capacity(nl);
    for l in 0..nl {
        let base = l * i;
        let mut row = Vec::with_capacity(i);
        for j in 0..i {
            row.push(flat[base + j] / denom);
        }
        feature.push(row);
    }
    // `acc` drops here → OwnedTensors return to the pool (deferred mailbox).
    Ok(Some(feature))
}

/// Tear down any active capture session (drops the GPU accumulator).
pub fn clear() {
    CAPTURE.with(|c| *c.borrow_mut() = None);
    ACTIVE.store(false, Ordering::Release);
}

/// Block-FFN forward tap: call right after `down_proj`, with the down_proj INPUT
/// (`ffn_hidden`, `[num_positions × intermediate]`) and OUTPUT (`down_out`,
/// `[num_positions × hidden]`). `batch_start` is the GLOBAL position of the first
/// row so a chunked prefill gates the response region correctly. Accumulates
/// per-neuron CETT on the GPU; no host bounce. No-op unless a session is active.
pub fn maybe_capture_ffn(
    gpu: &mut Gpu,
    ffn_hidden: &GpuTensor,
    down_out: &GpuTensor,
    layer_idx: usize,
    batch_start: usize,
    num_positions: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    CAPTURE.with(|c| -> HipResult<()> {
        let mut slot = c.borrow_mut();
        let Some(acc) = slot.as_mut() else {
            return Ok(());
        };
        // Skip the chunk entirely if it's all prompt.
        if batch_start + num_positions <= acc.response_start {
            return Ok(());
        }
        let inter = acc.intermediate;
        let hidden = acc.hidden;
        // Local index of the first response token in this chunk.
        let local_resp_start = acc
            .response_start
            .saturating_sub(batch_start)
            .min(num_positions);
        // Non-owning views into this layer's slice of the [nl*inter] buffers.
        let col_norm = acc.col_norm_gpu.sub_offset(layer_idx * inter, inter);
        let sums = acc.sums_gpu.sub_offset(layer_idx * inter, inter);
        let out_norm = gpu.alloc_owned(&[num_positions.max(1)], DType::F32)?;
        gpu.cett_accumulate_layer(
            ffn_hidden,
            &col_norm,
            down_out,
            &out_norm,
            &sums,
            num_positions,
            inter,
            hidden,
            local_resp_start,
        )?;
        // Count response tokens once per chunk (all layers see the same tokens).
        if layer_idx == 0 {
            acc.count += num_positions - local_resp_start;
        }
        Ok(())
    })
}
