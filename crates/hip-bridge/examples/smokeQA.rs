#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for the hip-bridge smoke test.

use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            println!("hip-bridge smoke QA: PASS - {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("hip-bridge smoke QA: SKIP - {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("hip-bridge smoke QA: FAIL - {msg}");
            ExitCode::from(1)
        }
    }
}

enum Outcome {
    Skip(String),
    Fail(String),
}

fn run() -> Result<String, Outcome> {
    let hip = hip_bridge::HipRuntime::load()
        .map_err(|e| Outcome::Skip(format!("HIP runtime unavailable: {e}")))?;

    let count = hip
        .device_count()
        .map_err(|e| Outcome::Fail(format!("device_count failed: {e}")))?;
    if count <= 0 {
        return Err(Outcome::Skip("no GPU devices found".to_string()));
    }

    hip.set_device(0)
        .map_err(|e| Outcome::Fail(format!("set_device failed: {e}")))?;

    let size = 4096usize;
    let buf = hip
        .malloc(size)
        .map_err(|e| Outcome::Fail(format!("malloc failed: {e}")))?;

    let src: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    hip.memcpy_htod(&buf, &src)
        .map_err(|e| Outcome::Fail(format!("H2D copy failed: {e}")))?;

    let mut dst = vec![0u8; size];
    hip.memcpy_dtoh(&mut dst, &buf)
        .map_err(|e| Outcome::Fail(format!("D2H copy failed: {e}")))?;

    if src != dst {
        let mismatch = src.iter().zip(&dst).position(|(a, b)| a != b).unwrap_or(0);
        let _ = hip.free(buf);
        return Err(Outcome::Fail(format!(
            "data mismatch at byte {mismatch}: src={} dst={}",
            src[mismatch], dst[mismatch]
        )));
    }

    hip.free(buf)
        .map_err(|e| Outcome::Fail(format!("free failed: {e}")))?;
    Ok(format!(
        "{} devices visible, {} bytes round-tripped",
        count, size
    ))
}
