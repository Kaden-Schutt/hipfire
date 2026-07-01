// SPDX-License-Identifier: Apache-2.0
//! CETT capture session + the arch-agnostic FFN forward tap.
//!
//! A dense-transformer forward calls [`maybe_capture_ffn`] right after its
//! `down_proj`, passing the down_proj INPUT (the FFN neuron activations) and its
//! OUTPUT. When a capture session is active, the hook folds each RESPONSE token's
//! per-neuron CETT into the running per-layer feature. Placement is one line per
//! arch (like the steer hook); the logic here is model-agnostic. The session is
//! process-global (the daemon dispatch is single-threaded), gated by a fast
//! `ACTIVE` atomic so a normal forward pays only one relaxed load.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

use crate::{cett, CettFeatures};

struct CaptureAcc {
    /// `col_norms[layer][neuron]` = `‖W_down[:,j]‖` (precomputed once per model).
    col_norms: Vec<Vec<f32>>,
    /// First GLOBAL token position that belongs to the response; earlier
    /// (prompt) positions are not folded.
    response_start: usize,
    intermediate: usize,
    hidden: usize,
    features: CettFeatures,
}

enum Session {
    Inactive,
    Capturing(CaptureAcc),
}

static SESSION: OnceLock<RwLock<Session>> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);

fn session() -> &'static RwLock<Session> {
    SESSION.get_or_init(|| RwLock::new(Session::Inactive))
}

fn set_session(s: Session) {
    let active = !matches!(s, Session::Inactive);
    *session().write().unwrap() = s;
    ACTIVE.store(active, Ordering::Release);
}

/// Whether a CETT capture session is active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Begin a CETT capture session. `col_norms[layer]` is `‖W_down[:,j]‖` per neuron
/// for that layer; `response_start` is the first token position belonging to the
/// response (prompt positions before it are ignored); `hidden` is the residual
/// width (for the `down_proj` output slicing).
pub fn begin_capture(col_norms: Vec<Vec<f32>>, response_start: usize, hidden: usize) {
    let num_layers = col_norms.len();
    let intermediate = col_norms.first().map_or(0, |c| c.len());
    set_session(Session::Capturing(CaptureAcc {
        col_norms,
        response_start,
        intermediate,
        hidden,
        features: CettFeatures::new(num_layers, intermediate),
    }));
}

/// End the session and return the per-layer mean CETT feature (`[layers][neurons]`),
/// or `None` if no capture was active.
pub fn finish_capture() -> Option<Vec<Vec<f32>>> {
    let out = match &*session().read().unwrap() {
        Session::Capturing(acc) => Some(acc.features.finish()),
        _ => None,
    };
    if out.is_some() {
        set_session(Session::Inactive);
    }
    out
}

/// Tear down any active capture session.
pub fn clear() {
    set_session(Session::Inactive);
}

/// Block-FFN forward tap: call right after `down_proj`, with the down_proj INPUT
/// (`ffn_hidden`, `[num_positions × intermediate]` neuron activations) and OUTPUT
/// (`down_out`, `[num_positions × hidden]`). `batch_start` is the GLOBAL position
/// of the first row so a chunked prefill gates the response region correctly.
/// No-op unless a capture session is active.
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
    match &mut *session().write().unwrap() {
        Session::Inactive => {}
        Session::Capturing(acc) => {
            // Skip the whole chunk (no host bounce) if it's entirely prompt.
            if batch_start + num_positions <= acc.response_start {
                return Ok(());
            }
            let (inter, hidden) = (acc.intermediate, acc.hidden);
            let act = gpu.download_f32(ffn_hidden)?;
            let out = gpu.download_f32(down_out)?;
            for local in 0..num_positions {
                if batch_start + local < acc.response_start {
                    continue;
                }
                let a = &act[local * inter..local * inter + inter];
                let o = &out[local * hidden..local * hidden + hidden];
                let out_norm = o.iter().map(|x| x * x).sum::<f32>().sqrt();
                let c = cett(a, &acc.col_norms[layer_idx], out_norm);
                acc.features.add_token(layer_idx, &c);
            }
        }
    }
    Ok(())
}
