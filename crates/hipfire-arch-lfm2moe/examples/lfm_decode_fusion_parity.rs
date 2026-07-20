// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Exact gfx1201 LFM2.5-350M MQ4 decode-fusion behavior-equivalence arm.
//!
//! Single-process arm driven by `scripts/lfm_decode_fusion_parity.py`. Loads
//! through [`Lfm2MoeBundle::new`], resets with [`Lfm2MoeBundle::redline_reset_state`],
//! then feeds the committed token vector one decode position at a time via
//! production `decode_step` (`DecodeExecutionMode::Oracle`). Emits one JSON
//! document on stdout with fixture identity, architecture, route flags, and
//! per-position finite logits / argmax / compact exact logits / `n_tokens` /
//! conv tails / dequantized just-written Q8 KV.
//!
//! Fusion flag is process-global `LazyLock` — baseline (`=0`) and candidate
//! (`=1`) MUST be separate fresh processes. The active-route marker is emitted
//! on stderr by the production admission path:
//!   `[lfm2moe] exact gfx1201 350m decode fusion active: shared RMSNorm+FWHT`
//!
//! Forced production lowered route (set by the Python driver; mirrored here):
//!   HIPFIRE_LFM2_GRAPH=0
//!   HIPFIRE_FORWARD_LOWERED=1
//!   HIPFIRE_REPLAY_BACKEND=hip
//!   HIPFIRE_LFM2_GFX1201_DECODE_FUSION={0|1}
//!
//! Exact cohort:
//!   model   = `/home/kaden/.hipfire/models/lfm2.5-350m.mq4`
//!   md5     = `cb5284b8ad5c6f9e4ca859c0aff0bcd0`
//!   arch_id = 11 / gfx1201
//!   tokens  = `[1,17,42,256,1024,4096,8191,7,511,2048,63,30000]`
//!
//! Run (via harness):
//!   python3 scripts/lfm_decode_fusion_parity.py
//!
//! Direct arm:
//!   flock /tmp/hipfire-gpu.lock cargo run -p hipfire-arch-lfm2moe --release \
//!       --features deltanet --example lfm_decode_fusion_parity -- \
//!       --model /home/kaden/.hipfire/models/lfm2.5-350m.mq4

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
    std::process::exit(1);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::validate_lfm_retained_fixture;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_arch_lfm2moe::redline_plan::{authenticate_retained_artifact, DecodeExecutionMode};
    use hipfire_arch_lfm2moe::{Lfm2MoeBundle, ARCH_ID};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::f16_to_f32;
    use serde_json::{json, Value};
    use std::path::PathBuf;

    const DEFAULT_MODEL: &str = "/home/kaden/.hipfire/models/lfm2.5-350m.mq4";
    const EXPECTED_MODEL_MD5: &str = "cb5284b8ad5c6f9e4ca859c0aff0bcd0";
    const ARCH_NAME: &str = "gfx1201";
    const MAX_SEQ: usize = 2048;
    /// Committed twelve-step decode vector (design § correctness gates).
    const TOKENS: [u32; 12] = [1, 17, 42, 256, 1024, 4096, 8191, 7, 511, 2048, 63, 30000];
    /// Production one-shot admission marker (stderr). Python requires it only
    /// on the fusion=1 arm.
    const FUSION_MARKER: &str =
        "[lfm2moe] exact gfx1201 350m decode fusion active: shared RMSNorm+FWHT";

    // Force the exact production lowered route before any LazyLock/OnceLock
    // readers fire. Fusion itself is left to the process environment so the
    // Python driver can A/B two fresh processes.
    std::env::set_var("HIPFIRE_REPLAY_BACKEND", "hip");
    std::env::set_var("HIPFIRE_LFM2_GRAPH", "0");
    std::env::set_var("HIPFIRE_FORWARD_LOWERED", "1");

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

    let fusion_env = std::env::var("HIPFIRE_LFM2_GFX1201_DECODE_FUSION")
        .ok()
        .unwrap_or_else(|| "0".to_owned());
    let fusion_requested = fusion_env == "1";

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

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    assert_eq!(
        gpu.arch.as_str(),
        ARCH_NAME,
        "oracle requires exact gpu.arch {ARCH_NAME}, got {}",
        gpu.arch
    );

    let mut hfq = HfqFile::open(&model).expect("open model");
    let retained_artifact =
        authenticate_retained_artifact(&mut hfq).expect("authenticate exact retained artifact");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    let state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, MAX_SEQ).expect("state");

    validate_lfm_retained_fixture(&cfg, &weights, &state, ARCH_ID)
        .expect("exact loaded fixture must validate");

    let n_attn = cfg.num_attention_layers();
    let n_conv = cfg.num_conv_layers();
    let hidden = cfg.hidden_size;
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let n_kv_heads = cfg.num_key_value_heads;
    let vocab = cfg.vocab_size;
    let conv_tail_elems = hidden * (cfg.conv_kernel_size - 1);
    let bytes_per_kv_pos = n_kv_heads * (head_dim / 32) * 34;

    let mut bundle = Lfm2MoeBundle::new(cfg, weights, state, 0, ARCH_ID, retained_artifact);
    assert!(
        bundle.retained_fixture_evidence(),
        "bundle construction must cache verified fixture evidence"
    );
    bundle
        .redline_reset_state(&mut gpu)
        .expect("redline_reset_state");
    assert_eq!(bundle.n_tokens(), 0, "reset must zero n_tokens");

    let b64 = |bytes: &[u8]| -> String {
        // Minimal base64 (no external crate): RFC 4648 alphabet.
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
        let mut i = 0;
        while i + 3 <= bytes.len() {
            let n =
                ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            out.push(T[((n >> 6) & 63) as usize] as char);
            out.push(T[(n & 63) as usize] as char);
            i += 3;
        }
        if i < bytes.len() {
            let rem = bytes.len() - i;
            let b0 = bytes[i] as u32;
            let b1 = if rem > 1 { bytes[i + 1] as u32 } else { 0 };
            let n = (b0 << 16) | (b1 << 8);
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            if rem == 1 {
                out.push('=');
                out.push('=');
            } else {
                out.push(T[((n >> 6) & 63) as usize] as char);
                out.push('=');
            }
        }
        out
    };

    let f32_le_bytes = |vals: &[f32]| -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    };

    let dequant_q8 = |bytes: &[u8]| -> Vec<f32> {
        assert_eq!(bytes.len() % 34, 0, "Q8_0 payload must be 34 B/block");
        let mut output = Vec::with_capacity((bytes.len() / 34) * 32);
        for block in bytes.chunks_exact(34) {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            output.extend(block[2..].iter().map(|&q| (q as i8) as f32 * scale));
        }
        output
    };

    let argmax = |v: &[f32]| -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        best_i
    };

    let mut positions: Vec<Value> = Vec::with_capacity(TOKENS.len());

    for (pos, &token) in TOKENS.iter().enumerate() {
        let logits = bundle
            .decode_step(&mut gpu, token, pos as u32, DecodeExecutionMode::Oracle)
            .unwrap_or_else(|e| panic!("decode_step pos={pos} token={token}: {e}"));
        assert_eq!(
            logits.len(),
            vocab,
            "pos={pos}: logits len {} != vocab {vocab}",
            logits.len()
        );

        let all_finite = logits.iter().all(|x| x.is_finite());
        let am = argmax(&logits);
        let n_tokens = bundle.n_tokens();
        assert_eq!(
            n_tokens,
            pos + 1,
            "pos={pos}: n_tokens {n_tokens} != consumed {}",
            pos + 1
        );

        let snap = bundle
            .redline_snapshot(&gpu)
            .unwrap_or_else(|e| panic!("redline_snapshot pos={pos}: {e}"));
        assert_eq!(snap.n_tokens, n_tokens);

        // Conv tails: recurrent is the concatenation of every conv-state
        // buffer (each hidden*(K-1) f32 LE).
        let conv_bytes_per = conv_tail_elems * 4;
        assert_eq!(
            snap.recurrent.len(),
            n_conv * conv_bytes_per,
            "pos={pos}: recurrent bytes"
        );
        let mut conv_tails = Vec::with_capacity(n_conv);
        let mut conv_finite = true;
        for slot in 0..n_conv {
            let begin = slot * conv_bytes_per;
            let end = begin + conv_bytes_per;
            let raw = &snap.recurrent[begin..end];
            let mut vals = Vec::with_capacity(conv_tail_elems);
            for chunk in raw.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !v.is_finite() {
                    conv_finite = false;
                }
                vals.push(v);
            }
            conv_tails.push(json!({
                "slot": slot,
                "elems": conv_tail_elems,
                "f32_le_b64": b64(&f32_le_bytes(&vals)),
            }));
        }

        // Q8 KV layout in the snapshot: all K tensors (n_attn) then all V
        // tensors (n_attn). Each tensor is physical_cap * bytes_per_kv_pos
        // bytes; just-written position is at absolute index `pos`.
        let physical_cap = bundle.max_seq(); // matches construction (no smaller cap)
                                             // Prefer the live tensor size from the snapshot to stay exact.
        assert!(
            snap.kv.len() % (2 * n_attn) == 0 || n_attn == 0,
            "pos={pos}: kv byte packing"
        );
        let per_tensor = if n_attn == 0 {
            0
        } else {
            // k_scales/v_scales empty for Q8 → exactly 2 * n_attn tensors.
            snap.kv.len() / (2 * n_attn)
        };
        assert!(
            per_tensor >= (pos + 1) * bytes_per_kv_pos,
            "pos={pos}: per-tensor bytes {per_tensor} < written span"
        );
        let _ = physical_cap; // documented; size comes from snapshot

        let mut kv_written = Vec::with_capacity(n_attn * 2);
        let mut kv_finite = true;
        for which in 0..2 {
            // 0 = K, 1 = V
            for slot in 0..n_attn {
                let tensor_idx = which * n_attn + slot;
                let base = tensor_idx * per_tensor;
                let off = base + pos * bytes_per_kv_pos;
                let raw = &snap.kv[off..off + bytes_per_kv_pos];
                let deq = dequant_q8(raw);
                assert_eq!(
                    deq.len(),
                    kv_dim,
                    "pos={pos} slot={slot} which={which}: dequant len"
                );
                if deq.iter().any(|x| !x.is_finite()) {
                    kv_finite = false;
                }
                kv_written.push(json!({
                    "kind": if which == 0 { "K" } else { "V" },
                    "slot": slot,
                    "position": pos,
                    "elems": deq.len(),
                    "f32_le_b64": b64(&f32_le_bytes(&deq)),
                }));
            }
        }

        positions.push(json!({
            "pos": pos,
            "token": token,
            "n_tokens": n_tokens,
            "argmax": am,
            "logits_len": logits.len(),
            "logits_finite": all_finite,
            "logits_f32_le_b64": b64(&f32_le_bytes(&logits)),
            "conv_finite": conv_finite,
            "conv_tails": conv_tails,
            "kv_finite": kv_finite,
            "kv_written": kv_written,
        }));
    }

    let report = json!({
        "oracle": "lfm_decode_fusion_parity",
        "model_path": model.display().to_string(),
        "model_md5": model_md5,
        "expected_model_md5": EXPECTED_MODEL_MD5,
        "arch": gpu.arch,
        "arch_id": ARCH_ID,
        "expected_arch": ARCH_NAME,
        "expected_arch_id": ARCH_ID,
        "retained_fixture_evidence": bundle.retained_fixture_evidence(),
        "fusion_env": fusion_env,
        "fusion_requested": fusion_requested,
        // API does not expose a direct "marker observed" boolean; the production
        // path prints FUSION_MARKER on stderr exactly once when fusion admits.
        // The Python harness is the authority for marker presence.
        "fusion_marker_text": FUSION_MARKER,
        "route": {
            "HIPFIRE_LFM2_GRAPH": "0",
            "HIPFIRE_FORWARD_LOWERED": "1",
            "HIPFIRE_REPLAY_BACKEND": "hip",
            "HIPFIRE_LFM2_GFX1201_DECODE_FUSION": fusion_env,
        },
        "dims": {
            "hidden": hidden,
            "layers": bundle.model_dimensions().1,
            "vocab": vocab,
            "n_attn": n_attn,
            "n_conv": n_conv,
            "kv_dim": kv_dim,
            "head_dim": head_dim,
            "n_kv_heads": n_kv_heads,
            "bytes_per_kv_pos": bytes_per_kv_pos,
            "conv_tail_elems": conv_tail_elems,
            "max_seq": MAX_SEQ,
        },
        "tokens": TOKENS.as_slice(),
        "positions": positions,
    });

    println!(
        "{}",
        serde_json::to_string(&report).expect("serialize report")
    );

    bundle.free_gpu(&mut gpu);
}
