// SPDX-License-Identifier: Apache-2.0
//! H-Neuron intervention session + the arch-agnostic FFN gain tap.
//!
//! A dense-transformer forward calls [`maybe_intervene_ffn`] right BEFORE its
//! `down_proj`, passing the down_proj INPUT (the FFN neuron activations). When an
//! intervention session is active, the hook multiplies each H-Neuron's activation
//! by a scalar `gain` (1.0 for every other neuron) — the dose-response knob for
//! the bidirectional gain sweep: `gain < 1` down-weights and `gain > 1`
//! up-weights the hallucination-associated neurons, `gain == 1` is the identity
//! control. Unlike CETT capture (response tokens only), intervention is applied
//! to ALL positions — prefill AND decode — since the neurons must be perturbed
//! wherever they fire. Process-global (single-threaded daemon dispatch), gated by
//! a fast `ACTIVE` atomic so a normal forward pays only one relaxed load.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor, OwnedTensor};

struct Intervention {
    /// `[num_layers * intermediate]` per-neuron gain (H-Neuron → `gain`, else
    /// 1.0), uploaded once (RAII: freed on drop).
    gain_gpu: OwnedTensor,
    num_layers: usize,
    intermediate: usize,
}

// Holds an `OwnedTensor` (not `Sync`); the whole session runs inline on one
// daemon dispatch thread, so it lives in TLS. Only `ACTIVE` is a shared atomic.
thread_local! {
    static INTERVENE: RefCell<Option<Intervention>> = const { RefCell::new(None) };
}
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Whether an H-Neuron intervention session is active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// Begin an intervention session. `hneurons` are FLAT feature indices
/// (`layer * intermediate + neuron`) into the `[num_layers][intermediate]` grid —
/// the positive-weight set from the L1 probe. Every listed neuron's activation is
/// scaled by `gain`; all others by 1.0. Builds the per-neuron gain mask host-side
/// and uploads it once.
pub fn begin_intervention(
    gpu: &mut Gpu,
    num_layers: usize,
    intermediate: usize,
    hneurons: &[usize],
    gain: f32,
) -> HipResult<()> {
    let n = (num_layers * intermediate).max(1);
    let mut mask = vec![1.0f32; n];
    for &idx in hneurons {
        if idx < n {
            mask[idx] = gain;
        }
    }
    let gain_gpu = gpu.alloc_owned(&[n], DType::F32)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u8, mask.len() * 4) };
    gpu.hip.memcpy_htod(&gain_gpu.buf, bytes)?;
    INTERVENE.with(|c| {
        *c.borrow_mut() = Some(Intervention {
            gain_gpu,
            num_layers,
            intermediate,
        });
    });
    ACTIVE.store(true, Ordering::Release);
    Ok(())
}

/// Tear down any active intervention session (drops the GPU gain mask).
pub fn clear() {
    INTERVENE.with(|c| *c.borrow_mut() = None);
    ACTIVE.store(false, Ordering::Release);
}

/// Block-FFN forward tap: call right BEFORE `down_proj` with the down_proj INPUT
/// (`ffn_hidden`, `[num_positions × intermediate]`). Scales each H-Neuron's
/// activation in place by the session gain. No-op unless a session is active.
pub fn maybe_intervene_ffn(
    gpu: &mut Gpu,
    ffn_hidden: &GpuTensor,
    layer_idx: usize,
    num_positions: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    INTERVENE.with(|c| -> HipResult<()> {
        let mut slot = c.borrow_mut();
        let Some(iv) = slot.as_mut() else {
            return Ok(());
        };
        if layer_idx >= iv.num_layers {
            return Ok(());
        }
        let inter = iv.intermediate;
        // Non-owning view into this layer's slice of the [nl*inter] gain buffer.
        let gain = iv.gain_gpu.sub_offset(layer_idx * inter, inter);
        gpu.hneuron_gain_layer(ffn_hidden, &gain, num_positions, inter)
    })
}
