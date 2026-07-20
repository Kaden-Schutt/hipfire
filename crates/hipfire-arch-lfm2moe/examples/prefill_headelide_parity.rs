// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Exact output/state parity check for LFM2.5 Phase 0 prefill head elision.
//!
//! Usage: prefill_headelide_parity [model.hfq]

use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
use hipfire_arch_lfm2moe::forward::{decode_step, decode_step_prefill};
use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hipfire_arch_lfm2moe::redline_plan::{DecodeExecutionMode, RetainedFixtureEvidence};
use hipfire_runtime::hfq::HfqFile;
use std::path::PathBuf;

const N_TOKENS: usize = 48;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let model = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME is not set"))
            .join(".hipfire/models/lfm2.5-350m.q8")
    });
    if args.next().is_some() {
        eprintln!("usage: prefill_headelide_parity [model.hfq]");
        std::process::exit(2);
    }

    let tokens: Vec<u32> = (0..N_TOKENS).map(|i| 10 + (i % 1000) as u32).collect();

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    let max_seq = N_TOKENS + 16;

    // Path A: eager head/logits on every prompt token.
    let mut state_eager =
        Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("eager state");
    let mut logits_eager = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        logits_eager = decode_step(
            &cfg,
            &weights,
            &mut state_eager,
            &mut gpu,
            token,
            position as u32,
            RetainedFixtureEvidence::ABSENT,
            DecodeExecutionMode::Prefill,
        )
        .expect("eager decode_step");
    }
    let n_tokens_eager = state_eager.n_tokens;
    let conv_eager: Vec<Vec<f32>> = state_eager
        .conv_states
        .iter()
        .map(|state| gpu.download_f32(state).expect("download eager conv state"))
        .collect();

    // Path B: skip the head on every non-final prompt token, then emit final logits.
    let mut state_elided =
        Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("elided state");
    for (position, &token) in tokens[..N_TOKENS - 1].iter().enumerate() {
        decode_step_prefill(
            &cfg,
            &weights,
            &mut state_elided,
            &mut gpu,
            token,
            position as u32,
        )
        .expect("head-elided decode_step_prefill");
    }
    let last_position = N_TOKENS - 1;
    let logits_elided = decode_step(
        &cfg,
        &weights,
        &mut state_elided,
        &mut gpu,
        tokens[last_position],
        last_position as u32,
        RetainedFixtureEvidence::ABSENT,
        DecodeExecutionMode::Prefill,
    )
    .expect("final elided-path decode_step");
    let n_tokens_elided = state_elided.n_tokens;
    let conv_elided: Vec<Vec<f32>> = state_elided
        .conv_states
        .iter()
        .map(|state| gpu.download_f32(state).expect("download elided conv state"))
        .collect();

    let max_abs_logit_diff = if logits_eager.len() == logits_elided.len() {
        logits_eager
            .iter()
            .zip(&logits_elided)
            .map(|(eager, elided)| (eager - elided).abs())
            .fold(0.0_f32, f32::max)
    } else {
        f32::INFINITY
    };
    let logits_bit_identical = logits_eager.len() == logits_elided.len()
        && logits_eager
            .iter()
            .zip(&logits_elided)
            .all(|(eager, elided)| eager.to_bits() == elided.to_bits());
    let logits_finite = logits_eager
        .iter()
        .chain(&logits_elided)
        .all(|value| value.is_finite());
    let n_tokens_equal = n_tokens_eager == n_tokens_elided;

    let mut conv_bit_identical = conv_eager.len() == conv_elided.len();
    let mut conv_max_diffs = Vec::with_capacity(conv_eager.len().max(conv_elided.len()));
    for layer in 0..conv_eager.len().max(conv_elided.len()) {
        let max_abs_conv_diff = match (conv_eager.get(layer), conv_elided.get(layer)) {
            (Some(eager), Some(elided)) if eager.len() == elided.len() => eager
                .iter()
                .zip(elided)
                .map(|(eager, elided)| (eager - elided).abs())
                .fold(0.0_f32, f32::max),
            _ => f32::INFINITY,
        };
        let layer_identical = matches!(
            (conv_eager.get(layer), conv_elided.get(layer)),
            (Some(eager), Some(elided))
                if eager.len() == elided.len()
                    && eager
                        .iter()
                        .zip(elided)
                        .all(|(eager, elided)| eager.to_bits() == elided.to_bits())
        );
        conv_bit_identical &= layer_identical;
        conv_max_diffs.push(max_abs_conv_diff);
    }
    let conv_finite = conv_eager
        .iter()
        .chain(&conv_elided)
        .flatten()
        .all(|value| value.is_finite());

    println!("model={}", model.display());
    println!("prompt_tokens={N_TOKENS}");
    println!("max_abs_logit_diff={max_abs_logit_diff:e}");
    println!("logits_bit_identical={logits_bit_identical}");
    println!("logits_finite={logits_finite}");
    println!("n_tokens_eager={n_tokens_eager}");
    println!("n_tokens_elided={n_tokens_elided}");
    println!("n_tokens_equal={n_tokens_equal}");
    for (layer, max_abs_conv_diff) in conv_max_diffs.iter().enumerate() {
        println!("conv_layer_{layer}_max_abs_conv_diff={max_abs_conv_diff:e}");
    }
    println!("conv_bit_identical={conv_bit_identical}");
    println!("conv_finite={conv_finite}");

    if logits_bit_identical && logits_finite && n_tokens_equal && conv_bit_identical && conv_finite
    {
        println!("PREFILL_HEAD_ELIDE_PARITY_PASS");
    } else {
        eprintln!("PREFILL_HEAD_ELIDE_PARITY_FAIL");
        std::process::exit(1);
    }
}
