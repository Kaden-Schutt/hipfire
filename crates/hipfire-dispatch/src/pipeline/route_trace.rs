// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Opt-in capture of the routed-expert SELECTION, so "prefill and decode
//! sometimes choose different experts" can be a measured rate instead of an
//! estimate.
//!
//! # Why the selection and not the router logits
//!
//! It is tempting to capture the logits and re-run top-k on the host. That
//! measures a different thing: the device kernels
//! (`moe_router_softmax_topk_k8_*` / `moe_topk_renorm_k8_batched`) have their
//! own ordering and tie-breaking, and a host re-implementation would attribute
//! its own disagreements with them to the routes. What matters is what the
//! model actually computed with, so this records the `topk_indices` buffer the
//! expert GEMVs are about to index with.
//!
//! # Why it is a file and not a return value
//!
//! The two routes being compared are a batched prefill and a per-token loop;
//! they are called from different places, produce a different number of
//! records per call, and neither has a natural channel back to the caller that
//! wants the comparison. A trace file keyed by call order is the cheapest
//! thing that lets one gate read both.
//!
//! Off unless `HIPFIRE_ESCHA_ROUTE_TRACE` names a path. When off, the only
//! cost is one relaxed atomic load per MoE layer.
//!
//! # Record format
//!
//! Little-endian, appended in CALL ORDER:
//!
//! ```text
//! u32 n_tokens        // rows in this record (1 for decode, chunk size for prefill)
//! u32 k               // experts per token
//! i32 ids[n_tokens*k] // token-major: row t is ids[t*k .. (t+1)*k]
//! ```
//!
//! A reader reconstructs `(token, layer)` from call order: every MoE layer
//! emits exactly one record per forward, so with `L` MoE layers the j-th
//! record of a per-token run is `(token = j / L, layer = j % L)` and the j-th
//! record of a batched run covering one chunk is `(layer = j % L)` with its
//! `n_tokens` rows being that chunk's tokens in order. Readers MUST check the
//! totals agree rather than assuming — see the gate.

use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::sync::OnceLock;

use rdna_compute::{Gpu, GpuTensor};

fn sink() -> &'static Mutex<Option<File>> {
    static SINK: OnceLock<Mutex<Option<File>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let f = hipfire_config::developer_var("HIPFIRE_ESCHA_ROUTE_TRACE")
            .ok()
            .and_then(|path| match File::create(&path) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("[escha-route-trace] cannot create {path}: {e}");
                    None
                }
            });
        Mutex::new(f)
    })
}

/// Redirect the trace to a new file, closing the previous one.
///
/// Exists because the comparison this module serves needs TWO traces from ONE
/// process (a batched arm and a per-token arm over the same prompt, with the
/// model loaded once — it is 37.5 GB). Re-reading the env var would not work:
/// the sink is initialised on first use, which is inside the first MoE layer,
/// long before the second arm starts.
pub fn reopen(path: &str) {
    let f = match File::create(path) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("[escha-route-trace] cannot create {path}: {e}");
            None
        }
    };
    if let Ok(mut slot) = sink().lock() {
        *slot = f;
    }
}

/// True when tracing is on. Callers should check this before doing anything
/// expensive (the capture below forces a device sync).
pub fn enabled() -> bool {
    sink().lock().map(|s| s.is_some()).unwrap_or(false)
}

/// Append one layer's selection.
///
/// `topk_indices` is the device buffer the routed GEMVs index with: `i32`
/// expert ids stored in an F32 tensor (the same 4-bytes-per-element
/// reinterpretation `moe_topk_renorm_k8_batched` writes). Only the first
/// `n_tokens * k` entries are read — the buffer itself is usually the larger
/// `[max_batch x k]` scratch.
///
/// Synchronises the device: this is a diagnostic, and a capture that raced the
/// kernel that produced the values would measure nothing.
pub fn record(gpu: &Gpu, topk_indices: &GpuTensor, n_tokens: usize, k: usize) {
    let want = n_tokens * k;
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let mut bytes = vec![0u8; want * 4];
    if gpu.hip.memcpy_dtoh(&mut bytes, &topk_indices.buf).is_err() {
        return;
    }
    let Ok(mut slot) = sink().lock() else { return };
    let Some(f) = slot.as_mut() else { return };
    let _ = f.write_all(&(n_tokens as u32).to_le_bytes());
    let _ = f.write_all(&(k as u32).to_le_bytes());
    let _ = f.write_all(&bytes);
}
