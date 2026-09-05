// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 gate: `DeltaNetSnapshot` bulk-copy save/restore correctness.
//!
//! Builds synthetic 12-layer DeltaNet states (all four families, EF on and
//! off) and proves the contract transitively through the live state (backup
//! buffers are private, so every backup assertion is proved by a restore):
//!
//! - save(P) -> poison(Q) -> restore -> live==P proves backup held P.
//! - A second poison(Q2) -> restore -> live==P proves restore leaves backup
//!   unchanged.
//! - The same double-restore chain after `save_from_async_on` (+ stream sync)
//!   proves the async path.
//! - A same-shape, different-allocation state rides the stale-fingerprint
//!   memcpy fallback, then a matching restore rewinds through the tables —
//!   proving fallback/fast-path interop.
//! - A canary buffer (never a copy destination) must survive every op.
//!
//! Buffer sizes cover multi-chunk items and tails: S = 200000 B (3x65536 +
//! 3392), scales = 4096 B (1 item), conv = 100000 B (65536 + 34464), EF =
//! 400000 B (6x65536 + 6784). EF-on is 14 items/layer (168 total), EF-off is
//! 7/layer (84 total).
//!
//! Run: `cargo run --release -p hipfire-arch-qwen35 --example
//! test_dflash_snapshot_bulk_gfx1100`. Passes on any arch (off-gfx1100 the
//! snapshot rides the memcpy loops and `bulk_n_items()` is `None`); on
//! gfx1100 the tables must arm with the exact item counts.

use hipfire_arch_qwen35::qwen35::{DeltaNetState, StateQuant};
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use rdna_compute::{DType, Gpu, GpuTensor};

const N_LAYERS: usize = 12;
const S_BYTES: usize = 200_000;
const SCALE_BYTES: usize = 4_096;
const CONV_BYTES: usize = 100_000;
const EF_BYTES: usize = 400_000;
// Items per layer: S 4 + scales 1 + conv 2 (+ EF 7 when on).
const ITEMS_PER_LAYER_EF_ON: u32 = 14;
const ITEMS_PER_LAYER_EF_OFF: u32 = 7;

/// Deterministic poison bytes, seeded per (family, layer, stream-id).
fn pattern(fam: u8, layer: usize, len: usize, seed: u64) -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64
        .wrapping_add(seed)
        .wrapping_add((fam as u64) << 56)
        .wrapping_add((layer as u64) << 32);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as u8
        })
        .collect()
}

fn alloc_fam(gpu: &mut Gpu, bytes: usize, dtype: DType, fam: u8, seed: u64) -> Vec<GpuTensor> {
    let mut out = Vec::with_capacity(N_LAYERS);
    for layer in 0..N_LAYERS {
        let buf = gpu.hip.malloc(bytes).expect("malloc family tensor");
        gpu.hip
            .memcpy_htod(&buf, &pattern(fam, layer, bytes, seed))
            .expect("fill family tensor");
        out.push(GpuTensor {
            buf,
            shape: vec![bytes],
            dtype,
        });
    }
    out
}

fn make_state(gpu: &mut Gpu, ef_on: bool, seed: u64) -> DeltaNetState {
    DeltaNetState {
        s_matrices: alloc_fam(gpu, S_BYTES, DType::F32, 0, seed),
        s_scales: alloc_fam(gpu, SCALE_BYTES, DType::F32, 1, seed),
        conv_states: alloc_fam(gpu, CONV_BYTES, DType::F32, 2, seed),
        s_ef_residual: if ef_on {
            alloc_fam(gpu, EF_BYTES, DType::F16, 3, seed)
        } else {
            Vec::new()
        },
        quant: StateQuant::Q8,
    }
}

/// Assert every live tensor in every family equals its `seed` pattern.
fn expect_live(gpu: &Gpu, tag: &str, what: &str, state: &DeltaNetState, seed: u64) {
    for (fam, fam_id) in [
        (&state.s_matrices, 0u8),
        (&state.s_scales, 1),
        (&state.conv_states, 2),
        (&state.s_ef_residual, 3),
    ] {
        assert_eq!(
            fam.len(),
            if fam_id == 3 && state.s_ef_residual.is_empty() {
                0
            } else {
                N_LAYERS
            },
            "{tag} {what}: family {fam_id} layer count"
        );
        for (layer, t) in fam.iter().enumerate() {
            let mut host = vec![0u8; t.buf.size()];
            gpu.hip.memcpy_dtoh(&mut host, &t.buf).expect("dtoh");
            assert_eq!(
                host,
                pattern(fam_id, layer, host.len(), seed),
                "{tag} {what}: family {fam_id} layer {layer} mismatch"
            );
        }
    }
}

fn poison_state(gpu: &mut Gpu, state: &DeltaNetState, seed: u64) {
    for (fam, fam_id) in [
        (&state.s_matrices, 0u8),
        (&state.s_scales, 1),
        (&state.conv_states, 2),
        (&state.s_ef_residual, 3),
    ] {
        for (layer, t) in fam.iter().enumerate() {
            gpu.hip
                .memcpy_htod(&t.buf, &pattern(fam_id, layer, t.buf.size(), seed))
                .expect("poison");
        }
    }
}

fn run_case(gpu: &mut Gpu, gfx1100: bool, ef_on: bool) {
    let tag = if ef_on { "EF-on" } else { "EF-off" };
    let mut state = make_state(gpu, ef_on, 0x11);
    let mut snap = DeltaNetSnapshot::new_for(gpu, &state).expect("new_for");
    assert_eq!(snap.s_ef_len(), if ef_on { N_LAYERS } else { 0 });
    let expect_items = N_LAYERS as u32
        * if ef_on {
            ITEMS_PER_LAYER_EF_ON
        } else {
            ITEMS_PER_LAYER_EF_OFF
        };
    match snap.bulk_n_items() {
        Some(n) => {
            assert!(gfx1100, "{tag}: tables armed off gfx1100");
            assert_eq!(n, expect_items, "{tag}: item count");
        }
        None => assert!(
            !gfx1100,
            "{tag}: tables disarmed on gfx1100 (n_items would be {expect_items})"
        ),
    }
    eprintln!(
        "{tag}: bulk_n_items={:?} (expect {expect_items} on gfx1100)",
        snap.bulk_n_items()
    );

    // Canary: never a copy destination; must survive every op byte-identical.
    let canary = gpu.hip.malloc(4096).expect("canary malloc");
    gpu.hip
        .memcpy_htod(&canary, &vec![0xA5u8; 4096])
        .expect("canary fill");
    let check_canary = |gpu: &Gpu, where_: &str| {
        let mut host = vec![0u8; 4096];
        gpu.hip
            .memcpy_dtoh(&mut host, &canary)
            .expect("canary read");
        assert_eq!(
            host,
            vec![0xA5u8; 4096],
            "{tag}: canary clobbered ({where_})"
        );
    };

    // 1. save(P) -> poison(Q) -> restore -> live==P: backup held P, and the
    // poison proves restore rewound rather than no-op'd.
    snap.save_from(&state, gpu).expect("save_from");
    expect_live(gpu, tag, "live-after-save", &state, 0x11);
    poison_state(gpu, &state, 0x22);
    snap.restore_to(&mut state, gpu).expect("restore_to");
    expect_live(gpu, tag, "live-after-restore", &state, 0x11);
    check_canary(gpu, "save/restore");

    // 2. restore leaves backup unchanged: poison again, restore again.
    poison_state(gpu, &state, 0x33);
    snap.restore_to(&mut state, gpu).expect("second restore_to");
    expect_live(gpu, tag, "live-after-second-restore", &state, 0x11);
    check_canary(gpu, "second-restore");

    // 3. async save on a fresh stream, then poison + sync restore.
    poison_state(gpu, &state, 0x44);
    let stream = gpu.hip.stream_create().expect("stream_create");
    snap.save_from_async_on(&state, gpu, &stream)
        .expect("save_from_async_on");
    gpu.hip.stream_synchronize(&stream).expect("stream sync");
    poison_state(gpu, &state, 0x55);
    snap.restore_to(&mut state, gpu)
        .expect("restore after async");
    expect_live(gpu, tag, "live-after-async-restore", &state, 0x44);
    check_canary(gpu, "async-save");

    // 4. stale fingerprint: same shapes, different allocations -> memcpy
    // fallback tracks the alien state; a matching restore then rewinds the
    // original live state to it (fallback/fast interop).
    let state2 = make_state(gpu, ef_on, 0x99);
    snap.save_from(&state2, gpu).expect("alien save_from");
    snap.restore_to(&mut state, gpu)
        .expect("restore after alien");
    expect_live(gpu, tag, "live-after-alien-restore", &state, 0x99);
    check_canary(gpu, "alien");
    state2.free_gpu(gpu);

    let _ = gpu.hip.free(canary);
    snap.free_gpu(gpu);
    state.free_gpu(gpu);
    eprintln!("{tag}: PASS");
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let gfx1100 = gpu.arch_caps.is_gfx1100();
    eprintln!("arch={} gfx1100={gfx1100}", gpu.arch);
    run_case(&mut gpu, gfx1100, true);
    run_case(&mut gpu, gfx1100, false);
    println!("S1 bulk snapshot gate: PASS (EF on/off, gfx1100={gfx1100})");
}
