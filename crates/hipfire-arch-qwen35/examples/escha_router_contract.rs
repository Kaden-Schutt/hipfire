// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4b: hipfire's arch-6 router must select the same experts Escha does.
//!
//! Escha rounds router logits to f16 BEFORE top-k (`ref.py`: the logits are
//! computed as f16 then widened to f32 to select). Selecting on unrounded f32
//! logits is a different function, and the rounding manufactures exact ties
//! that f32 never produces.
//!
//! Asserts the SET, not the order: the combine is a sum over slots, so intra-k
//! order cannot change the output. On the fixture, token 3 has two experts on
//! identical f16 logits (1.80078), and which one lands in which slot is
//! implementation-defined.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_router_contract -- /data/hipfire-models/escha-35b.hfq
use std::collections::HashSet;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hipfire-quantize/tests/data/escha")
        .join(name)
}

fn read_f16_as_f32(name: &str) -> Vec<f32> {
    let raw = std::fs::read(fixture(name)).expect("run fetch-goldens.sh first");
    raw.chunks_exact(2)
        .map(|c| hipfire_quantize::float16::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn main() {
    let hfq = std::env::args().nth(1).expect("usage: <model.hfq>");
    let x = read_f16_as_f32("moeblk_x.f16"); // [8, 2048]
    let raw_ids = std::fs::read(fixture("moeblk_ids.i64")).unwrap();
    let want_ids: Vec<i64> = raw_ids
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect(); // [8, 8]

    // Call hipfire's real router for layer 0 on each token.
    let got = hipfire_arch_qwen35::escha_router_topk_for_test(&hfq, 0, &x, 8, 2048, 8)
        .expect("router");

    let mut bad = 0usize;
    for t in 0..8 {
        let want: HashSet<i64> = want_ids[t * 8..(t + 1) * 8].iter().copied().collect();
        let mine: HashSet<i64> = got[t * 8..(t + 1) * 8].iter().map(|&v| v as i64).collect();
        if want != mine {
            bad += 1;
            println!("token {t}: escha={:?}", &want_ids[t * 8..(t + 1) * 8]);
            println!("         hipfire={:?}", &got[t * 8..(t + 1) * 8]);
        }
    }
    println!("tokens with a differing top-8 SET: {bad}/8");
    assert_eq!(
        bad, 0,
        "arch-6 router selects different experts than escha. Most likely cause: \
         it is not rounding logits to f16 before top-k."
    );
    println!("G4b PASS");
}
