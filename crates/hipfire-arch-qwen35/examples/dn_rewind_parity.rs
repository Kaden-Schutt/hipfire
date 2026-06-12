// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dn_rewind_parity: byte-parity probe for DeltaNetSnapshot restore + GdnTape replay.
//!
//! Verifies that `DeltaNetSnapshot::save_from` + `restore_to` + `GdnTape::replay_gdn`
//! produces bit-exact DN state (s_matrices, s_scales, s_ef_residual) compared with
//! the reference committed state produced by direct forward passes.
//!
//! Protocol:
//!   1. Forward tokens 0..K via forward_prefill_batch (no tape) → advance state to step K.
//!   2. Snapshot at K via DeltaNetSnapshot::save_from.
//!   3. Forward tokens K..N via forward_prefill_batch WITH gdn_tape → committed state.
//!   4. Download committed state as reference (s_matrices, s_scales, s_ef_residual).
//!   5. restore_to(snapshot) + replay_gdn(N-K steps).
//!   6. Download restored state, compare byte-for-byte.
//!   7. Print per-component max-abs-diff and PASS/FAIL.
//!
//! Control: WS_NO_EF_SNAP=1 skips the s_ef_residual save/restore in the snapshot
//!   (simulates the pre-fix state) so the before/after contrast is visible.
//!   With WS_NO_EF_SNAP=1 s_ef_residual should show nonzero diff; without it, zero.
//!
//! Usage:
//!   HIPFIRE_DN_STATE_EF=1 HIPFIRE_VERIFY_GRAPH=0 \
//!     ./target/release/examples/dn_rewind_parity /workspace/qwen3.6-27b.mq4-awq-barto

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35;
    use hipfire_arch_qwen35::speculative::{DeltaNetSnapshot, GdnTape, KvMode, ModelSlot, ModelSlotConfig};
    use hipfire_runtime::hfq::HfqFile;
    use rdna_compute::Gpu;
    use std::path::Path;

    const STEP_K: usize = 8;
    const STEP_N: usize = 24;
    let n_replay = STEP_N - STEP_K;

    // Fixed prompt tokens — just the BOS + a short run of token IDs.
    // We need at least N tokens; using a deterministic sequence.
    let prompt: Vec<u32> = (0..STEP_N as u32).map(|i| 1 + (i % 1000)).collect();

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/workspace/qwen3.6-27b.mq4-awq-barto".to_string());

    let no_ef_snap = std::env::var("WS_NO_EF_SNAP").map(|v| v == "1").unwrap_or(false);
    if no_ef_snap {
        eprintln!("[probe] WS_NO_EF_SNAP=1: s_ef_residual save/restore disabled (simulates pre-fix)");
    }

    eprintln!("[probe] model: {model_path}");
    eprintln!("[probe] STEP_K={STEP_K} STEP_N={STEP_N} n_replay={n_replay}");

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("[probe] gpu: {}", gpu.arch);

    // Load model via ModelSlot
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = 256;
    slot_cfg.kv_mode = KvMode::Q8;
    slot_cfg.state_quant = qwen35::StateQuant::Q8;
    let mut slot = ModelSlot::load(&mut gpu, Path::new(&model_path), "target", slot_cfg)
        .expect("ModelSlot::load");

    let n_la = slot.dn_state.s_matrices.len();
    eprintln!("[probe] n_la_layers={n_la} ef_residual_len={}", slot.dn_state.s_ef_residual.len());

    let ef_active = !slot.dn_state.s_ef_residual.is_empty();
    if !ef_active {
        eprintln!("[probe] WARNING: s_ef_residual is EMPTY — EF is off (HIPFIRE_DN_STATE_EF=0?). EF residual test will be vacuous.");
    } else {
        eprintln!("[probe] EF residual is ACTIVE ({n_la} layers). Fix is load-bearing.");
    }

    // Allocate GdnTape sized for n_replay steps (K..N)
    let mut tape = GdnTape::new_for_config(&mut gpu, &slot.config, n_replay)
        .expect("GdnTape::new_for_config");

    // ── Phase 1: forward tokens 0..K (no tape) ──────────────────────────
    eprintln!("[probe] Phase 1: forward tokens 0..{STEP_K} (no tape)...");
    qwen35::forward_prefill_batch(
        &mut gpu,
        &slot.weights,
        &slot.config,
        &prompt[..STEP_K],
        0,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
        None,
        None,
        None,
        None,
    ).expect("phase1 forward");

    // ── Phase 2: snapshot at K ───────────────────────────────────────────
    eprintln!("[probe] Phase 2: snapshot at step {STEP_K}...");
    let mut snapshot = DeltaNetSnapshot::new_for(&mut gpu, &slot.dn_state)
        .expect("DeltaNetSnapshot::new_for");
    snapshot.save_from(&slot.dn_state, &mut gpu)
        .expect("save_from");

    // ── Phase 3: forward tokens K..N with tape ───────────────────────────
    eprintln!("[probe] Phase 3: forward tokens {STEP_K}..{STEP_N} with GdnTape...");
    qwen35::forward_prefill_batch(
        &mut gpu,
        &slot.weights,
        &slot.config,
        &prompt[STEP_K..STEP_N],
        STEP_K,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
        None,
        None,
        Some(&mut tape),
        None,
    ).expect("phase3 forward with tape");

    // ── Phase 4: download committed reference state ──────────────────────
    eprintln!("[probe] Phase 4: downloading committed state...");
    let ref_s_matrices: Vec<Vec<u8>> = slot.dn_state.s_matrices.iter().map(|t| {
        let mut v = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("download s_matrix");
        v
    }).collect();
    let ref_s_scales: Vec<Vec<f32>> = slot.dn_state.s_scales.iter().map(|t| {
        let n = t.buf.size() / 4;
        let mut v = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("download s_scale");
        v.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect::<Vec<f32>>()
    }).collect();
    let ref_ef_residual: Vec<Vec<u8>> = slot.dn_state.s_ef_residual.iter().map(|t| {
        let mut v = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("download ef_residual");
        v
    }).collect();

    // ── Phase 5: WS_NO_EF_SNAP control — optionally zero out snapshot EF ─
    // (This simulates the pre-fix state where s_ef_residual_bufs were
    // uninitialized/stale, so restore_to would write zeros back.)
    if no_ef_snap {
        eprintln!("[probe] WS_NO_EF_SNAP: zeroing snapshot s_ef_residual_bufs to simulate pre-fix...");
        // We simulate "stale" by memset-ing the snapshot EF bufs to 0 via
        // the live state: temporarily zero the live ef_residual, then
        // re-save (this overwrites the snapshot EF bufs with zeros).
        // Then we restore the live EF to what it was.
        // Easier: just zero the live EF before restore+replay — the net
        // effect is the same as if the snapshot never saved it.
        // We restore and then zero the EF buffers before replay.
        snapshot.restore_to(&mut slot.dn_state, &mut gpu)
            .expect("restore_to (no_ef_snap)");
        // Zero the EF residuals to simulate stale snapshot
        for t in &slot.dn_state.s_ef_residual {
            gpu.hip.memset(&t.buf, 0, t.buf.size()).expect("zero ef_residual");
        }
    } else {
        // ── Phase 5 (normal): restore snapshot ──────────────────────────
        eprintln!("[probe] Phase 5: restore_to from snapshot...");
        snapshot.restore_to(&mut slot.dn_state, &mut gpu)
            .expect("restore_to");
    }

    // ── Phase 6: replay_gdn ──────────────────────────────────────────────
    eprintln!("[probe] Phase 6: replay_gdn for {n_replay} steps...");
    tape.replay_gdn(
        &mut gpu,
        &slot.weights,
        &slot.config,
        &mut slot.dn_state,
        n_replay,
    ).expect("replay_gdn");

    // ── Phase 7: download restored state and compare ─────────────────────
    eprintln!("[probe] Phase 7: comparing restored vs committed state...");

    let mut all_pass = true;
    let mut s_matrix_max = 0u8;
    let mut s_scale_max_abs = 0f32;
    let mut ef_residual_max = 0u8;

    // Compare s_matrices (raw bytes — Q8 quantized)
    for (i, (t, ref_bytes)) in slot.dn_state.s_matrices.iter().zip(ref_s_matrices.iter()).enumerate() {
        let mut actual = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut actual, &t.buf).expect("download restored s_matrix");
        let max_diff = actual.iter().zip(ref_bytes.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        if max_diff > s_matrix_max { s_matrix_max = max_diff; }
        if max_diff != 0 {
            all_pass = false;
            eprintln!("  MISMATCH s_matrices[{i}]: max_byte_diff={max_diff}");
        }
    }

    // Compare s_scales (f32)
    for (i, (t, ref_scales)) in slot.dn_state.s_scales.iter().zip(ref_s_scales.iter()).enumerate() {
        let mut raw = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut raw, &t.buf).expect("download restored s_scale");
        let actual_scales: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
            .collect();
        let max_diff = actual_scales.iter().zip(ref_scales.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_diff > s_scale_max_abs { s_scale_max_abs = max_diff; }
        if max_diff != 0.0 {
            all_pass = false;
            eprintln!("  MISMATCH s_scales[{i}]: max_abs_diff={max_diff:.6e}");
        }
    }

    // Compare s_ef_residual (f16 as raw bytes — byte-exact comparison)
    if ef_active {
        for (i, (t, ref_bytes)) in slot.dn_state.s_ef_residual.iter().zip(ref_ef_residual.iter()).enumerate() {
            let mut actual = vec![0u8; t.buf.size()];
            gpu.hip.memcpy_dtoh(&mut actual, &t.buf).expect("download restored ef_residual");
            let max_diff = actual.iter().zip(ref_bytes.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            if max_diff > ef_residual_max { ef_residual_max = max_diff; }
            if max_diff != 0 {
                all_pass = false;
                eprintln!("  MISMATCH s_ef_residual[{i}]: max_byte_diff={max_diff}");
            }
        }
    }

    eprintln!("");
    eprintln!("┌─────────────────────────────────────────────────────────────┐");
    eprintln!("│ dn_rewind_parity results                                    │");
    eprintln!("├─────────────────────────────────────────────────────────────┤");
    eprintln!("│ s_matrices   max_byte_diff : {s_matrix_max:<5} {}",
        if s_matrix_max == 0 { "PASS (byte-exact)       │" } else { "FAIL (nonzero diff)    │" });
    eprintln!("│ s_scales     max_abs_diff  : {s_scale_max_abs:<10.4e} {}",
        if s_scale_max_abs == 0.0 { "PASS (bit-exact)       │" } else { "FAIL (nonzero diff)    │" });
    if ef_active {
        eprintln!("│ s_ef_residual max_byte_diff: {ef_residual_max:<5} {}",
            if ef_residual_max == 0 { "PASS (byte-exact)       │" } else { "FAIL (nonzero diff)    │" });
    } else {
        eprintln!("│ s_ef_residual             : VACUOUS (EF off)             │");
    }
    eprintln!("├─────────────────────────────────────────────────────────────┤");
    if all_pass {
        eprintln!("│ OVERALL: PASS — restore+replay is byte-exact              │");
    } else {
        eprintln!("│ OVERALL: FAIL — restore+replay diverges from committed    │");
    }
    eprintln!("└─────────────────────────────────────────────────────────────┘");

    if !all_pass {
        std::process::exit(1);
    }
}
