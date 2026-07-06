// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-kernel profile of qwen2 (arch_id=7) block-parallel spec-decode verify.
//!
//! Confirms WHERE the time goes in `forward_verify_block_batched` — projection
//! GEMM (per-row MQ4G256 GEMV fallback) vs attention vs argmax vs memcpy — to
//! justify (or refute) a batched MQ4G256 GEMM. Loads VibeThinker-3B, prefills a
//! long prompt, warms, then times one B=8 verify under `profile::start/stop` and
//! prints aggregated µs by kernel.
//!
//! Usage:
//!   cargo run --release --example profile_verify -p hipfire-arch-qwen2 -- \
//!       --hfq ~/.hipfire/models/vibethinker-3b.mq4.hfq

use std::collections::BTreeMap;
use std::path::Path;

use hipfire_arch_qwen2::qwen2;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut hfq_path: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq_path = it.next(),
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let hfq_path = hfq_path.ok_or("--hfq is required")?;

    let mut hfq = HfqFile::open(Path::new(&hfq_path))?;
    let cfg = qwen2::config_from_hfq(&hfq)?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let mut gpu = Gpu::init()?;
    let weights = qwen2::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = qwen2::Qwen2State::new_with_max_seq(&mut gpu, &cfg, 1024)?;

    let para = "The history of computing spans many centuries and cultures. \
        Early mechanical calculators gave way to electromechanical relays, then \
        vacuum tubes, then transistors, and finally integrated circuits. Each \
        leap reduced size and cost while increasing speed and reliability. \
        Today's processors contain billions of transistors on a single die. ";
    let prompt = format!("{para}{para}{para}Summarize the key transitions:");
    let prompt_ids = tok.encode(&prompt);
    let position = prompt_ids.len();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let prefill = |gpu: &mut Gpu, state: &mut qwen2::Qwen2State| -> Result<(), String> {
        state.reset();
        for &t in &prompt_ids {
            qwen2::forward_step(gpu, &weights, &cfg, state, t).map_err(|e| format!("{e:?}"))?;
        }
        Ok(())
    };

    // A realistic B=8 candidate block: greedy continuation off the prompt.
    prefill(&mut gpu, &mut state)?;
    let mut block: Vec<u32> = vec![gpu.argmax_f32(&state.logits, cfg.vocab_size)?];
    let mut nxt = block[0];
    for _ in 1..8 {
        nxt = qwen2::forward_step_greedy(&mut gpu, &weights, &cfg, &mut state, nxt)?;
        block.push(nxt);
    }
    eprintln!("block (B={}): {block:?}", block.len());

    // Warm: a few throwaway verifies (JIT + DPM + cache).
    for _ in 0..3 {
        prefill(&mut gpu, &mut state)?;
        let _ = qwen2::forward_verify_block_batched(
            &mut gpu, &weights, &cfg, &mut state, &block, position,
        )
        .map_err(|e| format!("{e:?}"))?;
    }

    // Timed: one B=8 verify under the per-kernel profiler.
    prefill(&mut gpu, &mut state)?;
    rdna_compute::profile::start();
    let _ =
        qwen2::forward_verify_block_batched(&mut gpu, &weights, &cfg, &mut state, &block, position)
            .map_err(|e| format!("{e:?}"))?;
    let entries = rdna_compute::profile::stop().unwrap_or_default();

    // Aggregate by kernel name.
    let mut by_kernel: BTreeMap<&str, (f64, usize, usize)> = BTreeMap::new();
    let mut total = 0.0f64;
    for e in &entries {
        let slot = by_kernel.entry(e.kernel).or_insert((0.0, 0, 0));
        slot.0 += e.time_us;
        slot.1 += 1;
        slot.2 += e.bytes;
        total += e.time_us;
    }
    let mut rows: Vec<(&str, f64, usize, usize)> =
        by_kernel.iter().map(|(k, v)| (*k, v.0, v.1, v.2)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!(
        "\n=== qwen2 batched verify (B={}) per-kernel profile ===",
        block.len()
    );
    eprintln!(
        "{:<44} {:>10} {:>6} {:>9} {:>8}",
        "kernel", "us_total", "calls", "%cycle", "GB/s"
    );
    for (k, us, calls, bytes) in &rows {
        let pct = if total > 0.0 { us / total * 100.0 } else { 0.0 };
        let gbs = if *us > 0.0 {
            *bytes as f64 / (*us * 1e3)
        } else {
            0.0
        };
        eprintln!("{k:<44} {us:>10.1} {calls:>6} {pct:>8.1}% {gbs:>8.1}");
    }
    eprintln!("{:<44} {total:>10.1} us total", "TOTAL");
    Ok(())
}
