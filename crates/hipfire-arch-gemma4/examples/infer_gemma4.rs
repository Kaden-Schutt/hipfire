// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal Gemma 4 greedy inference — real-model e2e coherence check.
//! Loads an HFQ through the carrier bundle loader; `--route auto` follows the
//! architecture policy and `--route eager|lowered` enables explicit diagnostics.
//!
//! - Prepends the BOS token (2) if absent.
//! - Greedy argmax + a repetition penalty (default 1.3) over the recent window.
//! - Stops on EOS {1, 106} (`<end_of_turn>`).
//! - The embed √dim scale + final logit softcap are applied inside the forward.
//! Usage: infer_gemma4 --model <hfq> [--route auto|eager|lowered] [--prompt <text>] [--max N] [--rep-pen R]
#[cfg(feature = "deltanet")]
use hipfire_arch_gemma4::Gemma4Route;

#[cfg(feature = "deltanet")]
fn parse_route(value: &str) -> Result<Gemma4Route, &'static str> {
    match value {
        "auto" => Ok(Gemma4Route::Auto),
        "eager" => Ok(Gemma4Route::Eager),
        "lowered" => Ok(Gemma4Route::Lowered),
        _ => Err("infer_gemma4: --route must be one of auto|eager|lowered"),
    }
}
#[cfg(feature = "deltanet")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FullKvMode {
    Asym3,
    Fwht3,
}

#[cfg(feature = "deltanet")]
fn full_kv_mode(value: &str) -> FullKvMode {
    if value == "fwht3" {
        FullKvMode::Fwht3
    } else {
        FullKvMode::Asym3
    }
}

#[cfg(feature = "deltanet")]
fn ensure_bos(prompt_ids: &mut Vec<u32>, bos_token: u32) {
    if prompt_ids.first() != Some(&bos_token) {
        prompt_ids.insert(0, bos_token);
    }
}

#[cfg(all(test, feature = "deltanet"))]
mod tests {
    use super::{ensure_bos, full_kv_mode, parse_route, FullKvMode};
    use hipfire_arch_gemma4::Gemma4Route;

    #[test]
    fn non_default_bos_is_prepended_once_after_route_config_is_known() {
        let mut ids = vec![7, 8];
        ensure_bos(&mut ids, 3);
        assert_eq!(ids, vec![3, 7, 8]);
        ensure_bos(&mut ids, 3);
        assert_eq!(ids, vec![3, 7, 8]);
    }

    #[test]
    fn route_parser_defaults_to_carrier_auto() {
        assert_eq!(parse_route("auto"), Ok(Gemma4Route::Auto));
        assert_eq!(parse_route("eager"), Ok(Gemma4Route::Eager));
        assert_eq!(parse_route("lowered"), Ok(Gemma4Route::Lowered));
    }

    #[test]
    fn route_parser_rejects_unknown_route() {
        assert_eq!(
            parse_route("step").unwrap_err(),
            "infer_gemma4: --route must be one of auto|eager|lowered"
        );
    }

    #[test]
    fn kv_mode_selects_fwht3_full_cache() {
        assert_eq!(full_kv_mode("fwht3"), FullKvMode::Fwht3);
        assert_eq!(full_kv_mode(""), FullKvMode::Asym3);
    }
}

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
struct DecodeResult {
    gen: Vec<u32>,
    logits: Vec<f32>,
}

#[cfg(feature = "deltanet")]
fn route_name(route: Gemma4Route) -> &'static str {
    match route {
        Gemma4Route::Auto => "auto",
        Gemma4Route::Eager => "eager",
        Gemma4Route::Lowered => "lowered",
    }
}

#[cfg(feature = "deltanet")]
fn argmax_with_penalty(v: &mut [f32], history: &[u32], pen: f32) -> u32 {
    if pen != 1.0 {
        for &t in history {
            let idx = t as usize;
            if idx < v.len() {
                let x = v[idx];
                v[idx] = if x > 0.0 { x / pen } else { x * pen };
            }
        }
    }
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

#[cfg(feature = "deltanet")]
fn run_eager(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut hipfire_arch_gemma4::Gemma4EagerBundle,
    prompt_ids: &[u32],
    max: usize,
    rep_pen: f32,
) -> Result<DecodeResult, String> {
    use hipfire_arch_gemma4::forward::{decode_step, decode_step_with_graph};

    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_ids.iter().enumerate() {
        logits = decode_step(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            token,
            pos as u32,
        )
        .map_err(|error| format!("eager prefill: {error}"))?;
    }
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut gen = Vec::with_capacity(max);
    let mut history = prompt_ids.to_vec();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    for _ in 0..max {
        let next = argmax_with_penalty(&mut logits, &history, rep_pen);
        if matches!(next, 1 | 106) {
            break;
        }
        gen.push(next);
        history.push(next);
        logits = decode_step_with_graph(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            next,
            pos as u32,
        )
        .map_err(|error| format!("eager decode: {error}"))?;
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    eprintln!(
        "decoded {} tok in {:.2}s ({:.1} tok/s)",
        gen.len(),
        dt,
        gen.len() as f64 / dt
    );
    Ok(DecodeResult { gen, logits })
}

#[cfg(feature = "deltanet")]
fn run_lowered(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut hipfire_arch_gemma4::Gemma4LoweredBundle,
    prompt_ids: &[u32],
    max: usize,
    rep_pen: f32,
) -> Result<DecodeResult, String> {
    use hipfire_arch_gemma4::lowered;

    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_ids.iter().enumerate() {
        lowered::forward_scratch(
            gpu,
            &bundle.weights,
            &bundle.config,
            token,
            pos,
            &mut bundle.kv_sliding,
            &mut bundle.kv_full,
            &bundle.scratch,
        )
        .map_err(|error| format!("lowered prefill: {error:?}"))?;
        logits = gpu
            .download_f32(&bundle.scratch.logits)
            .map_err(|error| format!("lowered prefill logits: {error:?}"))?;
    }
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut gen = Vec::with_capacity(max);
    let mut history = prompt_ids.to_vec();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    for _ in 0..max {
        let next = argmax_with_penalty(&mut logits, &history, rep_pen);
        if matches!(next, 1 | 106) {
            break;
        }
        gen.push(next);
        history.push(next);
        lowered::forward_scratch(
            gpu,
            &bundle.weights,
            &bundle.config,
            next,
            pos,
            &mut bundle.kv_sliding,
            &mut bundle.kv_full,
            &bundle.scratch,
        )
        .map_err(|error| format!("lowered decode: {error:?}"))?;
        logits = gpu
            .download_f32(&bundle.scratch.logits)
            .map_err(|error| format!("lowered decode logits: {error:?}"))?;
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    eprintln!(
        "decoded {} tok in {:.2}s ({:.1} tok/s)",
        gen.len(),
        dt,
        gen.len() as f64 / dt
    );
    Ok(DecodeResult { gen, logits })
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::{load_gemma4_bundle_with_route, Gemma4Bundle};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 8192;
    let mut rep_pen: f32 = 1.3;
    let mut kv_mode = String::new();
    let mut route = Gemma4Route::Auto;
    let mut token_ids: Option<Vec<u32>> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--route" => {
                route = match parse_route(&argv[i + 1]) {
                    Ok(route) => route,
                    Err(error) => {
                        eprintln!("{error}");
                        std::process::exit(2);
                    }
                };
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&argv[i + 1]).expect("--prompt-file: read");
                i += 2;
            }
            // Bypass the tokenizer: feed comma-separated input token ids (e.g.
            // HF-tokenized). Output token ids are printed for external decode.
            "--token-ids" => {
                token_ids = Some(
                    argv[i + 1]
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<u32>().expect("--token-ids"))
                        .collect(),
                );
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--rep-pen" => {
                rep_pen = argv[i + 1].parse().expect("--rep-pen");
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = argv[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(&model).expect("open model");
    // Tokenizer only needed for text encode/decode; --token-ids bypasses it.
    let tok = if token_ids.is_none() {
        Some(Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer"))
    } else {
        None
    };

    let mut prompt_ids = match token_ids {
        Some(ids) => ids,
        None => tok.as_ref().unwrap().encode(&prompt),
    };
    // Leave room for the bundle-specific BOS token inserted after loading.
    let max_seq = prompt_ids.len().saturating_add(max).saturating_add(17);

    let cask = CaskConfig::default();
    let kv_mode_override = match full_kv_mode(&kv_mode) {
        FullKvMode::Fwht3 => Some("fwht3"),
        FullKvMode::Asym3 => None,
    };
    let model_path = model.to_str().expect("model path must be UTF-8");
    let mut ctx = LoadCtx {
        path: model_path,
        max_seq,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        kv_mode_override,
        kv_backend: KvBackend::Contiguous,
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_bands: None,
        mtp_mode: "auto",
        mtp_k: 3,
        spec: SpecLoadCfg::default(),
        kv_physical_cap: None,
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
    };
    let t_load = std::time::Instant::now();
    let mut bundle = match load_gemma4_bundle_with_route(ModelSource::Hfq(hfq), &mut ctx, route) {
        Ok(bundle) => bundle,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    eprintln!("loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());

    let bos_token = match &bundle {
        Gemma4Bundle::Eager(bundle) => bundle.config.bos_token,
        Gemma4Bundle::Lowered(bundle) => bundle.config.bos_token,
    };
    ensure_bos(&mut prompt_ids, bos_token);
    let actual_route = match &bundle {
        Gemma4Bundle::Eager(bundle) => {
            eprintln!(
                "gemma4 dim={} layers={} sliding={} full={} vocab={} softcap={}",
                bundle.config.dim,
                bundle.config.n_layers,
                bundle.config.n_sliding_layers(),
                bundle.config.n_full_layers(),
                bundle.config.vocab_size,
                bundle.config.final_logit_softcapping
            );
            "eager"
        }
        Gemma4Bundle::Lowered(bundle) => {
            let n_sliding = bundle
                .config
                .layer_types
                .iter()
                .filter(|layer| matches!(layer, hipfire_arch_gemma4::lowered::LayerType::Sliding))
                .count();
            let n_full = bundle
                .config
                .layer_types
                .iter()
                .filter(|layer| matches!(layer, hipfire_arch_gemma4::lowered::LayerType::Full))
                .count();
            eprintln!(
                "gemma4 dim={} layers={} sliding={} full={} vocab={} softcap={}",
                bundle.config.dim,
                bundle.config.n_layers,
                n_sliding,
                n_full,
                bundle.config.vocab_size,
                bundle.config.final_logit_softcapping
            );
            "lowered"
        }
    };
    eprintln!(
        "gemma4 route={} requested={}",
        actual_route,
        route_name(route)
    );
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());

    let result = match &mut bundle {
        Gemma4Bundle::Eager(bundle) => run_eager(&mut gpu, bundle, &prompt_ids, max, rep_pen),
        Gemma4Bundle::Lowered(bundle) => run_lowered(&mut gpu, bundle, &prompt_ids, max, rep_pen),
    }
    .unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });

    match &tok {
        Some(t) => println!(
            "=== PROMPT ===\n{prompt}\n=== GENERATION ===\n{}",
            t.decode(&result.gen)
        ),
        None => println!("=== GENERATION token ids ===\n{:?}", result.gen),
    }
    eprintln!("token ids: {:?}", &result.gen[..result.gen.len().min(60)]);
    let mut final_logit_hash = 0xcbf29ce484222325u64;
    for value in &result.logits {
        for byte in value.to_bits().to_le_bytes() {
            final_logit_hash ^= u64::from(byte);
            final_logit_hash = final_logit_hash.wrapping_mul(0x100000001b3);
        }
    }
    eprintln!("final logits fnv1a64: 0x{final_logit_hash:016x}");
}
