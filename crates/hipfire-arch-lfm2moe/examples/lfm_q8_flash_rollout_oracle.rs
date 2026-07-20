// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-350M shared Q8 flash rollout oracle (model-decision parity).
//!
//! Builds two independent reset-identical decode states from the same config
//! and weights:
//!   * raw   — `HIPFIRE_ATTN_FLASH=never` frozen at state construction
//!   * flash — `HIPFIRE_ATTN_FLASH=always` frozen at state construction
//!
//! Feeds the exact same prompt tokens into both via public `decode_step`
//! (`RetainedFixtureEvidence::ABSENT`, `DecodeExecutionMode::Oracle`) and
//! requires finite logits + identical argmax at every position. Reports
//! per-position and global logit max-abs / RMS plus raw top-1 margin.
//! Does not invent a post-hoc logit tolerance gate.
//!
//! Forced production-lowered route (graphs/spec/replay off, Q8 KV, max_seq 2048):
//!   HIPFIRE_LFM2_GRAPH=0
//!   HIPFIRE_FORWARD_LOWERED=1
//!   HIPFIRE_LFM2_350M_MQ4_DECODE_FUSION=0
//!
//! Exact cohort:
//!   model  = `/home/kaden/.hipfire/models/lfm2.5-350m.mq4`
//!   md5    = `cb5284b8ad5c6f9e4ca859c0aff0bcd0`
//!   prompt = `What is the capital of France? Reply in one short sentence.`
//!   md5    = `837ead29f20dcf48a9207e607c7394f2`
//!
//! Run:
//!   flock /tmp/hipfire-gpu.lock cargo run -p hipfire-arch-lfm2moe --release \
//!       --features deltanet --example lfm_q8_flash_rollout_oracle -- \
//!       --model /home/kaden/.hipfire/models/lfm2.5-350m.mq4

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
    std::process::exit(1);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::decode_step;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_arch_lfm2moe::redline_plan::{DecodeExecutionMode, RetainedFixtureEvidence};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    const DEFAULT_MODEL: &str = "/home/kaden/.hipfire/models/lfm2.5-350m.mq4";
    const EXPECTED_MODEL_MD5: &str = "cb5284b8ad5c6f9e4ca859c0aff0bcd0";
    const PROMPT: &str = "What is the capital of France? Reply in one short sentence.";
    const PROMPT_MD5: &str = "837ead29f20dcf48a9207e607c7394f2";
    const MAX_SEQ: usize = 2048;

    // Production lowered route; graphs/spec/replay off. Force HIP replay before
    // any GPU/model init so cohort capture cannot pick auto/PM4 defaults.
    std::env::set_var("HIPFIRE_REPLAY_BACKEND", "hip");
    std::env::set_var("HIPFIRE_LFM2_GRAPH", "0");
    std::env::set_var("HIPFIRE_FORWARD_LOWERED", "1");
    std::env::set_var("HIPFIRE_LFM2_350M_MQ4_DECODE_FUSION", "0");

    let argv: Vec<String> = std::env::args().collect();
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                if i + 1 >= argv.len() {
                    eprintln!("--model requires a path");
                    std::process::exit(2);
                }
                model = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    // Exact model identity before GPU/model init side effects deepen.
    let model_md5 = {
        use std::process::Command;
        let output = Command::new("md5sum")
            .arg(&model)
            .output()
            .unwrap_or_else(|e| panic!("md5sum {}: {e}", model.display()));
        assert!(
            output.status.success(),
            "md5sum failed for {}: {}",
            model.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .split_whitespace()
            .next()
            .unwrap_or_else(|| panic!("empty md5sum for {}", model.display()))
            .to_owned()
    };
    assert_eq!(
        model_md5,
        EXPECTED_MODEL_MD5,
        "model md5 mismatch for {}: got {model_md5}, expected {EXPECTED_MODEL_MD5}",
        model.display()
    );

    // Prompt identity: md5sum-compatible digest over exact UTF-8 prompt bytes.
    let prompt_md5 = {
        use std::io::Write;
        use std::process::Command;
        let mut child = Command::new("md5sum")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn md5sum for prompt");
        {
            let mut stdin = child.stdin.take().expect("md5sum stdin");
            stdin
                .write_all(PROMPT.as_bytes())
                .expect("write prompt bytes to md5sum");
        }
        let out = child.wait_with_output().expect("md5sum prompt wait");
        assert!(
            out.status.success(),
            "md5sum prompt failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        stdout
            .split_whitespace()
            .next()
            .expect("empty md5sum for prompt")
            .to_owned()
    };
    assert_eq!(
        prompt_md5, PROMPT_MD5,
        "prompt md5 mismatch: got {prompt_md5}, expected {PROMPT_MD5}"
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "oracle requires exact gpu.arch gfx1201, got {}",
        gpu.arch
    );
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    println!("=== LFM Q8 flash rollout oracle ===");
    println!("model_path={}", model.display());
    println!("model_md5={model_md5}");
    println!("expected_model_md5={EXPECTED_MODEL_MD5}");
    println!("prompt_md5={prompt_md5}");
    println!("expected_prompt_md5={PROMPT_MD5}");
    println!("gpu.arch={}", gpu.arch);
    println!("HIPFIRE_REPLAY_BACKEND=hip");
    println!(
        "dims hidden={} layers={} heads={}/{} head_dim={} vocab={} max_seq={MAX_SEQ}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size
    );
    println!("route HIPFIRE_LFM2_GRAPH=0 HIPFIRE_FORWARD_LOWERED=1 HIPFIRE_LFM2_350M_MQ4_DECODE_FUSION=0");
    println!("kv=Q8 graphs=off retained=ABSENT mode=Oracle");

    // Freeze flash_mode at construction; restore prior env afterward.
    let prior_flash = std::env::var("HIPFIRE_ATTN_FLASH").ok();
    std::env::set_var("HIPFIRE_ATTN_FLASH", "never");
    let mut state_raw =
        Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, MAX_SEQ).expect("raw state (flash=never)");
    std::env::set_var("HIPFIRE_ATTN_FLASH", "always");
    let mut state_flash = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, MAX_SEQ)
        .expect("flash state (flash=always)");
    match &prior_flash {
        Some(v) => std::env::set_var("HIPFIRE_ATTN_FLASH", v),
        None => std::env::remove_var("HIPFIRE_ATTN_FLASH"),
    }

    state_raw.reset(&mut gpu).expect("reset raw");
    state_flash.reset(&mut gpu).expect("reset flash");

    let prompt_ids = tok.encode(PROMPT);
    println!("prompt={PROMPT:?}");
    println!("prompt_md5_committed={PROMPT_MD5}");
    println!("n_tokens={}", prompt_ids.len());
    print!("token_ids=[");
    for (i, &t) in prompt_ids.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{t}");
    }
    println!("]");

    let argmax_and_margin = |v: &[f32]| -> (usize, f32, f32) {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                second = best_v;
                best_v = x;
                best_i = i;
            } else if x > second {
                second = x;
            }
        }
        let margin = if second.is_finite() {
            best_v - second
        } else {
            f32::INFINITY
        };
        (best_i, best_v, margin)
    };

    let mut global_max_abs = 0f64;
    let mut global_sum_sq = 0f64;
    let mut global_count = 0usize;
    let mut min_raw_top1_margin = f32::INFINITY;
    let mut fail = false;

    for (pos, &token) in prompt_ids.iter().enumerate() {
        let logits_raw = decode_step(
            &cfg,
            &weights,
            &mut state_raw,
            &mut gpu,
            token,
            pos as u32,
            RetainedFixtureEvidence::ABSENT,
            DecodeExecutionMode::Oracle,
        )
        .unwrap_or_else(|e| panic!("raw decode_step pos={pos}: {e}"));
        let logits_flash = decode_step(
            &cfg,
            &weights,
            &mut state_flash,
            &mut gpu,
            token,
            pos as u32,
            RetainedFixtureEvidence::ABSENT,
            DecodeExecutionMode::Oracle,
        )
        .unwrap_or_else(|e| panic!("flash decode_step pos={pos}: {e}"));

        if logits_raw.len() != logits_flash.len() {
            eprintln!(
                "FAIL pos={pos}: length mismatch raw={} flash={}",
                logits_raw.len(),
                logits_flash.len()
            );
            fail = true;
            break;
        }
        if logits_raw.len() != cfg.vocab_size {
            eprintln!(
                "FAIL pos={pos}: unexpected logits len {} (vocab {})",
                logits_raw.len(),
                cfg.vocab_size
            );
            fail = true;
            break;
        }

        let mut nonfinite = false;
        let mut max_abs = 0f64;
        let mut sum_sq = 0f64;
        for (&r, &f) in logits_raw.iter().zip(logits_flash.iter()) {
            if !r.is_finite() || !f.is_finite() {
                nonfinite = true;
            }
            let d = (r as f64) - (f as f64);
            max_abs = max_abs.max(d.abs());
            sum_sq += d * d;
        }
        let n = logits_raw.len();
        let rms = (sum_sq / n as f64).sqrt();
        global_max_abs = global_max_abs.max(max_abs);
        global_sum_sq += sum_sq;
        global_count += n;

        let (am_raw, _raw_top, raw_margin) = argmax_and_margin(&logits_raw);
        let (am_flash, _flash_top, _flash_margin) = argmax_and_margin(&logits_flash);
        min_raw_top1_margin = min_raw_top1_margin.min(raw_margin);

        let argmax_ok = am_raw == am_flash;
        if nonfinite || !argmax_ok {
            fail = true;
        }

        println!(
            "pos={pos} token={token} argmax_raw={am_raw} argmax_flash={am_flash} \
             raw_top1_margin={raw_margin:.6e} max_abs={max_abs:.6e} rms={rms:.6e}{}",
            if nonfinite {
                " NONFINITE"
            } else if !argmax_ok {
                " ARGMAX_MISMATCH"
            } else {
                ""
            }
        );
    }

    let global_rms = if global_count > 0 {
        (global_sum_sq / global_count as f64).sqrt()
    } else {
        0.0
    };

    println!("--- summary ---");
    println!("positions={}", prompt_ids.len());
    println!("global_max_abs={global_max_abs:.6e}");
    println!("global_rms={global_rms:.6e}");
    println!("min_raw_top1_margin={min_raw_top1_margin:.6e}");
    if fail {
        println!("LFM Q8 FLASH ROLLOUT ORACLE FAIL");
    } else {
        println!("LFM Q8 FLASH ROLLOUT ORACLE PASS");
    }

    state_raw.free_gpu(&mut gpu);
    state_flash.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);

    if fail {
        std::process::exit(1);
    }
}
