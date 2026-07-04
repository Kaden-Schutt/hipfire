// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2 DFlash seed smoke.
//!
//! Loads an LFM2 target HFQ plus an arch=20 DFlash sidecar, captures target
//! hidden rows with `prefill_batch_with_hidden`, runs one DFlash draft forward,
//! then applies the LFM2 target lm_head to draft rows 1..B.
//!
//! This is a pre-admission smoke: it proves the target-hidden producer and
//! draft/lm_head consumer contract, not full speculative accept/rollback.
//!
//! Usage:
//!   lfm2_dflash_seed_smoke --model <lfm2.hfq> --draft <lfm2.dflash.hfq>
//!     [--prompt <text>] [--block B] [--synthetic-noise] [--synthetic-hidden]
//!     [--probe-before-target]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::dflash::{
        lfm2_dflash_sync_gemm, lfm2_dflash_use_f16_weights, run_dflash_draft_for_logits,
        spec_step_dflash, validate_dflash_contract, verify_dflash_tokens, Lfm2DflashTargetSnapshot,
    };
    use hipfire_arch_lfm2moe::forward::{prefill_batch_with_hidden, Lfm2HiddenCapture};
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;
    use std::time::Instant;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut draft: Option<PathBuf> = None;
    let mut prompt = "Write a tiny merge sort in Rust.".to_string();
    let mut block_override: Option<usize> = None;
    let mut synthetic_noise = false;
    let mut synthetic_hidden = false;
    let mut probe_before_target = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--draft" => {
                draft = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--block" => {
                block_override = Some(argv[i + 1].parse().expect("--block expects usize"));
                i += 2;
            }
            "--synthetic-noise" => {
                synthetic_noise = true;
                i += 1;
            }
            "--synthetic-hidden" => {
                synthetic_hidden = true;
                i += 1;
            }
            "--probe-before-target" => {
                probe_before_target = true;
                i += 1;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let model = model.expect("--model required");
    let draft = draft.expect("--draft required");

    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    let mut target_hfq = HfqFile::open(&model).expect("open target hfq");
    let target_cfg = Lfm2MoeConfig::from_hfq(&target_hfq).expect("target config");
    let tokenizer = Tokenizer::from_hfq_metadata(&target_hfq.metadata_json).expect("tokenizer");

    let draft_hfq = HfqFile::open(&draft).expect("open draft hfq");
    let draft_cfg = DflashConfig::from_hfq(&draft_hfq).expect("draft config");
    if let Err(e) = validate_dflash_contract(&target_cfg, &draft_cfg) {
        eprintln!("FAIL: {e}");
        std::process::exit(2);
    }

    let block_size = block_override.unwrap_or(draft_cfg.block_size);
    if block_size < 2 {
        eprintln!("FAIL: block_size must be >= 2");
        std::process::exit(2);
    }

    let prompt_ids = tokenizer.encode(&prompt);
    if prompt_ids.is_empty() {
        eprintln!("FAIL: prompt tokenized to zero tokens");
        std::process::exit(2);
    }
    eprintln!(
        "target: hidden={} layers={} vocab={} prompt_tokens={} draft_layers={:?} block={}",
        target_cfg.hidden_size,
        target_cfg.num_hidden_layers,
        target_cfg.vocab_size,
        prompt_ids.len(),
        draft_cfg.target_layer_ids,
        block_size,
    );

    if probe_before_target {
        run_synthetic_dflash_probe(
            &mut gpu,
            &draft_hfq,
            &draft_cfg,
            block_size,
            prompt_ids.len(),
            "pre-target",
        )
        .expect("pre-target synthetic dflash probe");
    }

    let target_weights =
        Lfm2MoeWeights::load(&mut target_hfq, &target_cfg, &mut gpu).expect("target weights");

    let max_seq = prompt_ids.len() + block_size + 8;
    let mut target_state =
        Lfm2MoeState::new_with_max_seq(&mut gpu, &target_cfg, max_seq).expect("target state");
    let mut capture = Lfm2HiddenCapture::new(
        target_cfg.num_hidden_layers,
        target_cfg.hidden_size,
        draft_cfg.target_layer_ids.clone(),
    )
    .expect("capture config");

    let t_prefill = Instant::now();
    let target_logits = prefill_batch_with_hidden(
        &target_cfg,
        &target_weights,
        &mut target_state,
        &mut gpu,
        &prompt_ids,
        &mut capture,
    )
    .expect("target prefill/capture");
    gpu.hip.device_synchronize().expect("prefill sync");
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let seed_token = argmax(&target_logits) as u32;
    let (hidden_finite, hidden_min, hidden_max, hidden_mean_abs) = hidden_stats(capture.rows());
    eprintln!(
        "captured positions={} floats={} finite={}/{} min={:.6e} max={:.6e} mean_abs={:.6e} seed_token={} decoded={:?} prefill_ms={:.2}",
        capture.position_count(),
        capture.rows().len(),
        hidden_finite,
        capture.rows().len(),
        hidden_min,
        hidden_max,
        hidden_mean_abs,
        seed_token,
        tokenizer.decode(&[seed_token]),
        prefill_ms,
    );
    if hidden_finite != capture.rows().len() {
        eprintln!("FAIL: non-finite captured target hidden rows");
        std::process::exit(2);
    }

    let t_load = Instant::now();
    let draft_weights = DflashWeights::load_with_f16(
        &mut gpu,
        &draft_hfq,
        &draft_cfg,
        lfm2_dflash_use_f16_weights(),
    )
    .expect("draft weights");
    let mut draft_scratch = DflashScratch::new_with_mq_and_sync(
        &mut gpu,
        &draft_cfg,
        block_size,
        prompt_ids.len(),
        draft_weights.has_mq,
        lfm2_dflash_sync_gemm(),
    )
    .expect("draft scratch");
    eprintln!(
        "draft load mode: f16_weights={} sync_gemm={}",
        lfm2_dflash_use_f16_weights(),
        draft_scratch.sync_gemm
    );
    eprintln!(
        "draft loaded+scratch in {:.2}s",
        t_load.elapsed().as_secs_f64()
    );

    eprintln!(
        "noise source: {}",
        if synthetic_noise {
            "synthetic host noise"
        } else {
            "target embeddings [seed, mask, ...]"
        }
    );
    let noise_embedding: Option<Vec<f32>> = if synthetic_noise {
        let mut rng_state: u64 = 0x1F_2D_3C_4Bu64;
        let mut rng = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let v = (rng_state as u32) as f32 / (u32::MAX as f32);
            (v - 0.5) * 0.04
        };
        Some(
            (0..block_size * target_cfg.hidden_size)
                .map(|_| rng())
                .collect(),
        )
    } else {
        None
    };

    let target_hidden_synthetic: Option<Vec<f32>> = if synthetic_hidden {
        let mut rng_state: u64 = 0xD1FEu64;
        let mut rng = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let v = (rng_state as u32) as f32 / (u32::MAX as f32);
            (v - 0.5) * 0.04
        };
        let len = prompt_ids.len() * draft_cfg.num_extract() * draft_cfg.hidden;
        eprintln!("target_hidden source: synthetic host rows ({len} floats)");
        Some((0..len).map(|_| rng()).collect())
    } else {
        eprintln!("target_hidden source: captured LFM2 rows");
        None
    };
    let target_hidden_arg = target_hidden_synthetic
        .as_deref()
        .unwrap_or_else(|| capture.rows());

    let t_bridge = Instant::now();
    eprintln!(
        "draft bridge start: ctx={} block={} target_hidden_floats={}",
        prompt_ids.len(),
        block_size,
        target_hidden_arg.len()
    );
    let bridge = run_dflash_draft_for_logits(
        &mut gpu,
        &target_weights,
        &target_cfg,
        &draft_weights,
        &draft_cfg,
        &mut draft_scratch,
        target_hidden_arg,
        prompt_ids.len(),
        seed_token,
        None,
        block_size,
        noise_embedding.as_deref(),
    )
    .expect("lfm2 dflash draft bridge");
    gpu.hip.device_synchronize().expect("draft bridge sync");
    eprintln!("draft bridge complete");
    let bridge_ms = t_bridge.elapsed().as_secs_f64() * 1000.0;

    let finite = bridge.logits.iter().filter(|v| v.is_finite()).count();
    let first_tokens: Vec<u32> = (0..bridge.batch.min(8))
        .map(|row| {
            let start = row * bridge.vocab_size;
            argmax(&bridge.logits[start..start + bridge.vocab_size]) as u32
        })
        .collect();
    eprintln!(
        "draft_bridge_ms={:.2} logits_shape=[{},{}] finite={}/{}",
        bridge_ms,
        bridge.batch,
        bridge.vocab_size,
        finite,
        bridge.logits.len(),
    );
    eprintln!("draft argmax first rows: {:?}", first_tokens);

    if finite != bridge.logits.len() {
        eprintln!("FAIL: non-finite draft logits");
        std::process::exit(2);
    }

    let draft_tokens: Vec<u32> = (0..bridge.batch)
        .map(|row| {
            let start = row * bridge.vocab_size;
            argmax(&bridge.logits[start..start + bridge.vocab_size]) as u32
        })
        .collect();
    let mut verify_tokens = Vec::with_capacity(1 + draft_tokens.len());
    verify_tokens.push(seed_token);
    verify_tokens.extend_from_slice(&draft_tokens);

    let mut target_snap =
        Lfm2DflashTargetSnapshot::new_for(&mut gpu, &target_state, verify_tokens.len())
            .expect("target snapshot alloc");
    target_snap
        .save_from(&mut gpu, &target_state)
        .expect("target snapshot save");
    let t_verify = Instant::now();
    eprintln!(
        "target verify start: start_pos={} block={}",
        prompt_ids.len(),
        verify_tokens.len()
    );
    let verify = verify_dflash_tokens(
        &mut gpu,
        &target_weights,
        &target_cfg,
        &mut target_state,
        &draft_cfg,
        &verify_tokens,
        prompt_ids.len(),
    )
    .expect("lfm2 dflash target verify");
    gpu.hip.device_synchronize().expect("target verify sync");
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
    let verify_finite = verify
        .logits_per_pos
        .iter()
        .filter(|v| v.is_finite())
        .count();
    let hidden_finite = verify
        .target_hidden_rows
        .iter()
        .filter(|v| v.is_finite())
        .count();
    eprintln!(
        "target_verify_ms={:.2} logits_shape=[{},{}] finite={}/{} hidden_finite={}/{}",
        verify_ms,
        verify.batch,
        verify.vocab_size,
        verify_finite,
        verify.logits_per_pos.len(),
        hidden_finite,
        verify.target_hidden_rows.len(),
    );
    eprintln!(
        "target argmax first rows: {:?}",
        &verify.argmax_per_pos[..verify.argmax_per_pos.len().min(8)]
    );
    if verify_finite != verify.logits_per_pos.len()
        || hidden_finite != verify.target_hidden_rows.len()
    {
        eprintln!("FAIL: non-finite target verify output");
        std::process::exit(2);
    }
    target_snap
        .restore_to(&mut gpu, &mut target_state)
        .expect("target snapshot restore");
    target_snap.free_gpu(&mut gpu);
    if target_state.n_tokens != prompt_ids.len() {
        eprintln!(
            "FAIL: target restore left n_tokens={} expected={}",
            target_state.n_tokens,
            prompt_ids.len()
        );
        std::process::exit(2);
    }
    eprintln!(
        "target snapshot restored to n_tokens={}",
        target_state.n_tokens
    );

    let mut spec_target_hidden = target_hidden_arg.to_vec();
    let mut spec_snap = Lfm2DflashTargetSnapshot::new_for(&mut gpu, &target_state, block_size)
        .expect("spec snapshot alloc");
    let t_spec = Instant::now();
    let spec = spec_step_dflash(
        &mut gpu,
        &target_weights,
        &target_cfg,
        &mut target_state,
        &draft_weights,
        &draft_cfg,
        &mut draft_scratch,
        &mut spec_target_hidden,
        &mut spec_snap,
        prompt_ids.len(),
        seed_token,
        None,
        Some(block_size),
    )
    .expect("lfm2 dflash spec step");
    gpu.hip.device_synchronize().expect("spec step sync");
    let spec_ms = t_spec.elapsed().as_secs_f64() * 1000.0;
    let expected_tokens = prompt_ids.len() + spec.advance;
    let expected_hidden = expected_tokens * draft_cfg.num_extract() * draft_cfg.hidden;
    eprintln!(
        "spec_step_ms={:.2} accepted={} advance={} bonus={} committed={:?} drafted_first={:?} target_argmax_first={:?}",
        spec_ms,
        spec.accepted,
        spec.advance,
        spec.bonus_token,
        spec.committed,
        &spec.drafted[..spec.drafted.len().min(8)],
        &spec.target_argmax_per_pos[..spec.target_argmax_per_pos.len().min(8)]
    );
    if target_state.n_tokens != expected_tokens {
        eprintln!(
            "FAIL: spec step left n_tokens={} expected={}",
            target_state.n_tokens, expected_tokens
        );
        std::process::exit(2);
    }
    if spec_target_hidden.len() != expected_hidden {
        eprintln!(
            "FAIL: spec hidden floats={} expected={}",
            spec_target_hidden.len(),
            expected_hidden
        );
        std::process::exit(2);
    }
    spec_snap.free_gpu(&mut gpu);
    println!("LFM2 DFLASH SEED SMOKE PASS");
}

#[cfg(feature = "deltanet")]
fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

#[cfg(feature = "deltanet")]
fn hidden_stats(v: &[f32]) -> (usize, f32, f32, f64) {
    let mut finite = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum_abs = 0.0f64;
    for &x in v {
        if x.is_finite() {
            finite += 1;
            min_v = min_v.min(x);
            max_v = max_v.max(x);
            sum_abs += (x as f64).abs();
        }
    }
    let mean_abs = if finite == 0 {
        f64::NAN
    } else {
        sum_abs / finite as f64
    };
    (finite, min_v, max_v, mean_abs)
}

#[cfg(feature = "deltanet")]
fn run_synthetic_dflash_probe(
    gpu: &mut hipfire_rdna::Gpu,
    hfq: &hipfire_runtime::hfq::HfqFile,
    cfg: &hipfire_runtime::dflash::DflashConfig,
    block_size: usize,
    ctx_len: usize,
    label: &str,
) -> hip_bridge::HipResult<()> {
    use hipfire_runtime::dflash::{self, DflashScratch, DflashWeights};
    use std::time::Instant;

    eprintln!(
        "standalone draft probe ({label}) start: ctx={} block={} target_hidden_floats={}",
        ctx_len,
        block_size,
        ctx_len * cfg.num_extract() * cfg.hidden
    );

    let weights = DflashWeights::load_with_f16(
        gpu,
        hfq,
        cfg,
        hipfire_arch_lfm2moe::dflash::lfm2_dflash_use_f16_weights(),
    )?;
    let mut scratch = DflashScratch::new_with_mq_and_sync(
        gpu,
        cfg,
        block_size,
        ctx_len,
        weights.has_mq,
        hipfire_arch_lfm2moe::dflash::lfm2_dflash_sync_gemm(),
    )?;

    let mut rng_state: u64 = 0xD1FEu64;
    let mut rng = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let v = (rng_state as u32) as f32 / (u32::MAX as f32);
        (v - 0.5) * 0.04
    };

    let noise_embedding: Vec<f32> = (0..block_size * cfg.hidden).map(|_| rng()).collect();
    let target_hidden: Vec<f32> = (0..ctx_len * cfg.num_extract() * cfg.hidden)
        .map(|_| rng())
        .collect();
    let positions_q: Vec<i32> = (ctx_len as i32..ctx_len as i32 + block_size as i32).collect();
    let positions_k: Vec<i32> = (0..(ctx_len + block_size) as i32).collect();

    let t = Instant::now();
    dflash::draft_forward(
        gpu,
        &weights,
        cfg,
        Some(&noise_embedding),
        Some(&target_hidden),
        &positions_q,
        &positions_k,
        block_size,
        ctx_len,
        &mut scratch,
    )?;
    gpu.hip.device_synchronize()?;
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;

    let out = gpu.download_f32(&scratch.x)?;
    let finite = out.iter().take(1024).filter(|v| v.is_finite()).count();
    let (mn, mx) = out
        .iter()
        .take(1024)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    eprintln!(
        "standalone draft probe ({label}) ok: forward_ms={elapsed_ms:.2} first1024_finite={finite}/1024 min={mn:.6e} max={mx:.6e}",
    );

    scratch.free_gpu(gpu);
    weights.free_gpu(gpu);
    Ok(())
}
