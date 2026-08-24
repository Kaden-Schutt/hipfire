// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Adaptive KV reset must invalidate captured execution state.
//!
//! Dropped when `KvCache` moved to `saddle-core` (wave 1, C1) and never
//! restored; recovered verbatim from `8510ca5f2`. It lives here rather than
//! beside the other nine recovered tests because it drives
//! `hipfire_runtime::kv_adaptive`, which sits one layer above `saddle-core`.

use hipfire_runtime::kv_adaptive;
use rdna_compute::Gpu;
use saddle_core::kv::*;

fn flag_standin(mode: KvMode, v_mode: VMode, n_kv_heads: usize, head_dim: usize) -> KvCache {
    let (q8, a4, a3, a2, fwht) = KvCache::vmm_mode_flags(mode);
    KvCache {
        k_gpu: vec![],
        v_gpu: vec![],
        k_scales: vec![],
        v_scales: vec![],
        kv_dim: n_kv_heads * head_dim,
        max_seq: 128,
        physical_cap: 128,
        n_kv_heads,
        head_dim,
        quantized: true,
        quant_q8: q8,
        quant_int8: false,
        quant_hfq4: false,
        quant_asym4: a4,
        quant_asym3: a3,
        quant_asym2: a2,
        quant_fwht: fwht,
        boundary_layers: 0,
        givens_cos: None,
        givens_sin: None,
        layer_is_boundary: vec![],
        compact_offset: 0,
        v_mode,
    }
}

#[test]
fn adaptive_reset_invalidates_captured_execution_state() {
    let Ok(mut gpu) = Gpu::init() else {
        eprintln!("skip: no GPU");
        return;
    };
    let mut cache = flag_standin(KvMode::Fwht2, VMode::Lloyd2, 4, 256);
    let mut adaptive = kv_adaptive::KvAdaptive::from_preset(
        kv_adaptive::Preset::Aggressive,
        128,
        4,
        256,
    );
    adaptive.cur_k = kv_adaptive::KMode::Fwht2;
    adaptive.cur_v = VMode::Lloyd2;
    adaptive.next_step = adaptive.steps.len();
    gpu.graphs.ar_forward_replay_enabled = true;
    gpu.graphs.ar_forward_kernel_dirty = false;

    adaptive.reset_with_cache(&mut gpu, &mut cache);

    assert_eq!(cache.current_kv_mode().unwrap(), KvMode::Fwht4);
    assert_eq!(cache.v_mode, VMode::Q8);
    assert!(!gpu.graphs.ar_forward_replay_enabled);
    assert!(gpu.graphs.ar_forward_kernel_dirty);
}
