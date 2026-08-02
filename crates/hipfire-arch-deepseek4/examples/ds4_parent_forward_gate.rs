// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 5/6 of the DS4 parent-checkpoint calibration path: full 43-layer
//! forward of the ORIGINAL mixed-precision parent checkpoint over a pinned
//! token sequence, producing saved reference logits (`.plog`) and a
//! provenance manifest for Gate 6.
//!
//! Usage:
//! ```text
//! ds4_parent_forward_gate --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!                         [--tokens 32 | --token-ids tokens.bin] \
//!                         [--plog OUT.plog] [--manifest PATH] \
//!                         [--skip-shard-hashes]
//! ```
//!
//! Prefer `--token-ids` (flat u32 LE from `ds4_tokenize_corpus`) for any
//! promoted artifact. The PRNG path (`--tokens` alone) is smoke-only.
//!
//! Cross-process determinism: run
//! `examples/ds4_parent_forward_gate_determinism.sh` (two separate processes,
//! compare `logits_sha256`).
//!
//! Must run on gfx942 (mi300x).

use hipfire_arch_deepseek4::parent::attention::{PARENT_DIM, PARENT_HEAD_DIM};
use hipfire_arch_deepseek4::parent::compressor::compressor_prefill_n_out;
use hipfire_arch_deepseek4::parent::forward::PARENT_HC_MULT;
use hipfire_arch_deepseek4::parent::head::{
    parent_logits_to_plog, PARENT_HC_DIM, PARENT_VOCAB,
};
use hipfire_arch_deepseek4::parent::inventory::ParentInventory;
use hipfire_arch_deepseek4::parent::manifest::{
    sha256_bytes, sha256_file, CaptureBoundary, CaptureInfo, CorpusInfo, ModelInfo, ModelQuantInfo,
    OutputInfo, OutputKind, ParentManifest, ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_arch_deepseek4::parent::model::{
    assert_compress_events, parent_model_forward, parent_model_forward_traced, LayerHcNormStats,
    ParentModelScratch,
};
use hipfire_arch_deepseek4::parent::plog::PlogWriter;
use hipfire_arch_deepseek4::parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
/// Gate 5 bar: fixed 32-token sequence.
const DEFAULT_TOKENS: usize = 32;
/// Gate 1 main-tower projection total (MTP excluded), exact byte count from
/// the handoff / inventory gate (two independent paths agreed bit-for-bit).
const GATE1_TOTAL_BYTES: u64 = 161_872_686_172;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Deterministic PRNG seed for the fixed 32-token sequence (printed).
const TOKEN_SEED: u64 = 0xD5_46_A7_E4_04_6A_75;

/// Layer-to-layer HC L2 ratio outside this band is flagged.
/// Residual stacks with healthy residual connections stay near 1; geometric
/// blow-up or collapse is the defect this catches. Band is deliberately
/// wide enough for real residual dynamics but tight enough to reject a
/// 2×-per-layer explosion over 43 layers.
const NORM_RATIO_LO: f64 = 0.25;
const NORM_RATIO_HI: f64 = 4.0;

/// Short real prompt for the coherence eyeball (decoded via tokenizer).
const COHERENCE_PROMPT: &str = "The capital of France is";

fn main() -> ExitCode {
    match run() {
        Ok(pass) => {
            if pass {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let model_path = Path::new(&args.model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }

    // ── Token sequence (prefer pinned --token-ids) ──────────────────────
    let (token_ids, token_source_desc, promoted_input) = if let Some(tid) = &args.token_ids {
        let ids = read_token_ids_file(tid)?;
        if ids.is_empty() {
            return Err(format!(
                "deepseek4 parent: --token-ids {} is empty",
                tid.display()
            ));
        }
        // Optional --tokens truncates a longer file (never pads).
        let ids = if args.tokens_explicit && args.tokens < ids.len() {
            ids[..args.tokens].to_vec()
        } else if args.tokens_explicit && args.tokens > ids.len() {
            return Err(format!(
                "deepseek4 parent: --tokens {} exceeds token-ids length {}",
                args.tokens,
                ids.len()
            ));
        } else {
            ids
        };
        let desc = format!("token-ids file {}", tid.display());
        (ids, desc, true)
    } else {
        let n = args.tokens;
        if n == 0 {
            return Err("deepseek4 parent: --tokens must be > 0".into());
        }
        let ids = select_token_ids(TOKEN_SEED, n);
        let desc = format!(
            "PRNG smoke sequence seed={TOKEN_SEED:#x} (NOT for promoted artifacts)"
        );
        (ids, desc, false)
    };
    let n_tokens = token_ids.len();
    if n_tokens == 0 {
        return Err("deepseek4 parent: token sequence is empty".into());
    }

    println!("=== ds4_parent_forward_gate (Gate 5/6) ===");
    println!("model: {}", model_path.display());
    println!("tokens: {n_tokens}");
    println!("token_source: {token_source_desc}");
    println!("promoted_input: {promoted_input}");
    println!("skip_shard_hashes: {}", args.skip_shard_hashes);
    if args.skip_shard_hashes && promoted_input {
        eprintln!(
            "WARN: --skip-shard-hashes with --token-ids: promoted artifacts require real shard hashes"
        );
    }
    if let Some(p) = args.plog.as_ref() {
        println!("plog: {}", p.display());
    }
    if let Some(m) = args.manifest.as_ref() {
        println!("manifest: {}", m.display());
    }
    println!();

    // ── 1. Admit + inventory + full load ────────────────────────────────
    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;

    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err(
            "deepseek4 parent: gfx942 required (parent calibration is fail-closed)".to_owned(),
        );
    }
    println!("gpu: gfx942");

    let admit_t0 = Instant::now();
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK ({:.1} ms): layers={} hash_layers={} n_routed={} topk={}",
        admit_t0.elapsed().as_secs_f64() * 1000.0,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );
    if cfg.num_hidden_layers != 43 {
        return Err(format!(
            "deepseek4 parent: Gate 5 expects 43 layers, config has {}",
            cfg.num_hidden_layers
        ));
    }

    let inv = ParentInventory::build(&source, &cfg)?;
    println!("inventory entries={}", inv.entries.len());

    let plan = ParentLoadPlan {
        layers: 0..cfg.num_hidden_layers,
        load_experts: true,
    };
    println!(
        "load plan: layers={:?} load_experts={}  (expect ~150.8 GiB / ~41 s)",
        plan.layers, plan.load_experts
    );
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    let res = weights.residency();
    println!(
        "loaded layers={:?} experts={} in {load_s:.3} s",
        weights.layer_range, weights.experts_loaded
    );
    println!(
        "residency: total={:.3} GiB ({} bytes)  dense_bf16={:.3} expert={:.3} \
         bf16={:.3} f32={:.3} i64={:.3}",
        res.total_bytes() as f64 / GIB,
        res.total_bytes(),
        res.dense_bf16_bytes as f64 / GIB,
        res.expert_compressed_bytes as f64 / GIB,
        res.bf16_bytes as f64 / GIB,
        res.f32_bytes as f64 / GIB,
        res.i64_bytes as f64 / GIB,
    );

    let mut checks: Vec<CheckRow> = Vec::new();

    let resid_ok = res.total_bytes() == GATE1_TOTAL_BYTES;
    checks.push(CheckRow {
        name: "residency_vs_gate1".into(),
        pass: resid_ok,
        detail: format!(
            "got {} want {GATE1_TOTAL_BYTES} (delta {:+})",
            res.total_bytes(),
            res.total_bytes() as i64 - GATE1_TOTAL_BYTES as i64
        ),
    });
    if !resid_ok {
        eprintln!(
            "WARN: residency {} != Gate 1 projection {GATE1_TOTAL_BYTES}",
            res.total_bytes()
        );
    }
    if weights.layers.len() != cfg.num_hidden_layers {
        return Err(format!(
            "deepseek4 parent: loaded {} layers, expected {}",
            weights.layers.len(),
            cfg.num_hidden_layers
        ));
    }
    for (i, layer) in weights.layers.iter().enumerate() {
        if layer.experts.len() != cfg.n_routed_experts {
            return Err(format!(
                "deepseek4 parent: layer {i} experts.len() {} != {}",
                layer.experts.len(),
                cfg.n_routed_experts
            ));
        }
    }

    // ── 2. Token ids (already loaded above) ─────────────────────────────
    // Print a short prefix/suffix for provenance; full dump is noisy at 1024.
    let preview_n = n_tokens.min(16);
    print!("token_ids first[{preview_n}] = [");
    for (i, &t) in token_ids.iter().take(preview_n).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{t}");
    }
    println!("]");
    if n_tokens > preview_n {
        print!("token_ids last[{preview_n}] = [");
        let start = n_tokens - preview_n;
        for (i, &t) in token_ids[start..].iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{t}");
        }
        println!("]");
    }
    let token_ids_sha = sha256_bytes(u32_slice_as_le_bytes(&token_ids));
    println!("token_ids_sha256 = {token_ids_sha}");

    // Tokenizer (for coherence eyeball + optional decode of argmax).
    let tok_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_tokenizer_json(&tok_path)
        .map_err(|e| format!("deepseek4 parent: tokenizer load: {e}"))?
        .ok_or_else(|| {
            format!(
                "deepseek4 parent: missing tokenizer.json at {}",
                tok_path.display()
            )
        })?;
    println!(
        "tokenizer loaded: bos_id={} eos_id={}",
        tokenizer.bos_id, tokenizer.eos_id
    );

    // ── 3. Scratch + logits tile ────────────────────────────────────────
    let mut scratch = ParentModelScratch::new(&mut gpu, &cfg, n_tokens)?;
    println!(
        "ParentModelScratch::bytes() = {} ({:.3} MiB)  max_rows={} n_layers={}",
        scratch.bytes(),
        scratch.bytes() as f64 / (1024.0 * 1024.0),
        scratch.max_rows(),
        scratch.n_layers(),
    );
    let logits = zeros_f32(&mut gpu, &[n_tokens, PARENT_VOCAB])?;
    let logits_bytes = n_tokens.saturating_mul(PARENT_VOCAB).saturating_mul(4);
    println!(
        "logits tile: [{n_tokens}, {PARENT_VOCAB}] F32 = {} ({:.3} MiB)",
        logits_bytes,
        logits_bytes as f64 / (1024.0 * 1024.0)
    );

    // ── 4. Traced full forward ──────────────────────────────────────────
    let mut layer_norms: Vec<LayerHcNormStats> = Vec::new();
    let mut compress_events: Vec<(usize, usize)> = Vec::new();
    let fwd_t0 = Instant::now();
    let fwd_result = parent_model_forward_traced(
        &mut gpu,
        backend,
        &weights,
        &cfg,
        &mut scratch,
        &token_ids,
        /* start_pos */ 0,
        &logits,
        &mut layer_norms,
        &mut compress_events,
    );
    // Drain device errors so a mid-stack refusal surfaces cleanly.
    let _ = gpu.hip.device_synchronize();
    let fwd_s = fwd_t0.elapsed().as_secs_f64();

    let mut blocked_on_sibling = false;
    let mut sibling_detail = String::new();
    match fwd_result {
        Ok(()) => {
            println!("parent_model_forward_traced wall = {fwd_s:.3} s");
            checks.push(CheckRow {
                name: "full_43_layer_forward".into(),
                pass: true,
                detail: format!("{fwd_s:.3} s, {} layers traced", layer_norms.len()),
            });
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("compress_ratio") || msg.contains("compressor/indexer") {
                blocked_on_sibling = true;
                sibling_detail = msg.clone();
                eprintln!();
                eprintln!("BLOCKED-ON-SIBLING: full 43-layer forward refused:");
                eprintln!("  {msg}");
                eprintln!(
                    "  AttnCompIdx is lifting compress_ratio!=0; ratio-0 layers alone \
                     cannot close Gate 5. Continuing with partial diagnostics where possible."
                );
                checks.push(CheckRow {
                    name: "full_43_layer_forward".into(),
                    pass: false,
                    detail: format!("blocked-on-sibling: {msg}"),
                });
            } else {
                return Err(msg);
            }
        }
    }

    // ── 4b. Per-layer compress-event table ──────────────────────────────
    println!();
    println!("=== per-layer compress events ===");
    println!(
        "  {:>5}  {:>5}  {:>8}  {:>8}  {}",
        "layer", "ratio", "observed", "expect", "status"
    );
    if compress_events.is_empty() {
        println!("  (none — forward did not complete)");
        checks.push(CheckRow {
            name: "compress_events".into(),
            pass: false,
            detail: "empty".into(),
        });
    } else {
        let mut n_ratio0 = 0usize;
        let mut n_ratio4 = 0usize;
        let mut n_ratio128 = 0usize;
        let mut n_zero_fire = 0usize;
        for (i, &(ratio, observed)) in compress_events.iter().enumerate() {
            let expect = if ratio == 0 {
                0
            } else {
                compressor_prefill_n_out(n_tokens, ratio)
            };
            let status = if ratio == 0 {
                n_ratio0 += 1;
                if observed == 0 {
                    "ok"
                } else {
                    "BAD"
                }
            } else {
                if ratio == 4 {
                    n_ratio4 += 1;
                } else if ratio == 128 {
                    n_ratio128 += 1;
                }
                if expect > 0 && observed == 0 {
                    n_zero_fire += 1;
                    "FAIL-ZERO"
                } else if observed == expect {
                    "ok"
                } else {
                    "MISMATCH"
                }
            };
            // Always print ratio-128 rows + any non-ok row; summarize the rest.
            if ratio == 128 || status != "ok" || n_tokens <= 64 {
                println!(
                    "  {i:>5}  {ratio:>5}  {observed:>8}  {expect:>8}  {status}"
                );
            }
        }
        println!(
            "  summary: layers={}  ratio0={n_ratio0}  ratio4={n_ratio4}  \
             ratio128={n_ratio128}  zero_fire={n_zero_fire}",
            compress_events.len()
        );
        // Full table dump for ratio-4 summary stats too.
        let r4_obs: Vec<usize> = compress_events
            .iter()
            .filter_map(|&(r, o)| if r == 4 { Some(o) } else { None })
            .collect();
        let r128_obs: Vec<usize> = compress_events
            .iter()
            .filter_map(|&(r, o)| if r == 128 { Some(o) } else { None })
            .collect();
        if !r4_obs.is_empty() {
            let min = *r4_obs.iter().min().unwrap();
            let max = *r4_obs.iter().max().unwrap();
            println!(
                "  ratio-4 observed: n={} min={min} max={max} expect={}",
                r4_obs.len(),
                compressor_prefill_n_out(n_tokens, 4)
            );
        }
        if !r128_obs.is_empty() {
            let min = *r128_obs.iter().min().unwrap();
            let max = *r128_obs.iter().max().unwrap();
            println!(
                "  ratio-128 observed: n={} min={min} max={max} expect={}",
                r128_obs.len(),
                compressor_prefill_n_out(n_tokens, 128)
            );
        }
        match assert_compress_events(&compress_events, n_tokens) {
            Ok(()) => {
                checks.push(CheckRow {
                    name: "compress_events".into(),
                    pass: true,
                    detail: format!(
                        "{} layers; no silent zero-fire on ratio>0",
                        compress_events.len()
                    ),
                });
            }
            Err(e) => {
                eprintln!("{e}");
                checks.push(CheckRow {
                    name: "compress_events".into(),
                    pass: false,
                    detail: e,
                });
            }
        }
    }

    // If blocked, report which layers are ratio-0 (runnable without sibling).
    if blocked_on_sibling {
        let ratio0: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.compress_ratio(i) == 0)
            .collect();
        println!();
        println!(
            "ratio-0 layers available without sibling: {ratio0:?} ({} of {})",
            ratio0.len(),
            cfg.num_hidden_layers
        );
        let _ = &mut scratch;
        let _ = &logits;
    }

    // ── 5. Logits stats ─────────────────────────────────────────────────
    let logits_host = if !blocked_on_sibling || !layer_norms.is_empty() {
        // Only meaningful if head ran; head only runs after all layers.
        if blocked_on_sibling {
            Vec::new()
        } else {
            download_f32(&gpu, &logits, n_tokens * PARENT_VOCAB)?
        }
    } else {
        Vec::new()
    };

    let mut logits_l2 = 0.0f64;
    let mut logits_mean = 0.0f64;
    let mut logits_std = 0.0f64;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut argmax_ids: Vec<u32> = Vec::new();
    let mut argmax_vals: Vec<f32> = Vec::new();

    if !logits_host.is_empty() {
        let (nan, inf, _zero, l2, _amax) = finite_stats(&logits_host);
        n_nan = nan;
        n_inf = inf;
        logits_l2 = l2 as f64;
        let (mean, std) = mean_std(&logits_host);
        logits_mean = mean;
        logits_std = std;
        println!();
        println!("=== logits ===");
        println!(
            "L2={logits_l2:.6e}  mean={logits_mean:.6e}  std={logits_std:.6e}  \
             nan={n_nan} inf={n_inf}  nelems={}",
            logits_host.len()
        );
        // Cap argmax dump: full dump at ≤64 tokens, else first/last 8.
        let dump_positions: Vec<usize> = if n_tokens <= 64 {
            (0..n_tokens).collect()
        } else {
            let mut v: Vec<usize> = (0..8).collect();
            v.extend((n_tokens - 8)..n_tokens);
            v
        };
        println!("argmax per position ({} shown of {n_tokens}):", dump_positions.len());
        for &r in &dump_positions {
            let row = &logits_host[r * PARENT_VOCAB..(r + 1) * PARENT_VOCAB];
            let (idx, val) = argmax(row);
            // Still fill all argmax_ids for decode.
            println!("  pos {r:>4}: token={idx}  logit={val:.4}");
        }
        for r in 0..n_tokens {
            let row = &logits_host[r * PARENT_VOCAB..(r + 1) * PARENT_VOCAB];
            let (idx, val) = argmax(row);
            argmax_ids.push(idx as u32);
            argmax_vals.push(val);
        }
        let finite_ok = n_nan == 0 && n_inf == 0;
        checks.push(CheckRow {
            name: "logits_finite".into(),
            pass: finite_ok,
            detail: format!("nan={n_nan} inf={n_inf}"),
        });
        checks.push(CheckRow {
            name: "logits_nonzero_l2".into(),
            pass: logits_l2.is_finite() && logits_l2 > 0.0,
            detail: format!("L2={logits_l2:.6e}"),
        });
    } else {
        checks.push(CheckRow {
            name: "logits_finite".into(),
            pass: false,
            detail: "no logits (forward incomplete)".into(),
        });
    }

    // ── 6. Per-layer HC L2 + stability ──────────────────────────────────
    println!();
    println!("=== per-layer HC residual norms (post hc_post_ffn) ===");
    println!(
        "  statistic for stack_stability = MEDIAN per-row L2; aggregate/p90/max are diagnostic only.\n           Why not aggregate: a single massive-activation row (L37→L38 case: median 404→413 while          aggregate 14222→116670, ratio 8.2x) dominates the flat L2 and false-fails a healthy stack.          See combfix/MEDIAN_TRAJ_ARTIFACT_REPORT.txt."
    );
    if layer_norms.is_empty() {
        println!("  (none — forward did not complete any layer)");
        checks.push(CheckRow {
            name: "layer_norm_trace".into(),
            pass: false,
            detail: "empty".into(),
        });
    } else {
        for (i, s) in layer_norms.iter().enumerate() {
            // Absolute layer order for the layers that ran.
            println!(
                "  layer {i:>2}: median={:.6}  p90={:.6}  max={:.6}  aggregate={:.6}",
                s.median, s.p90, s.max, s.aggregate
            );
        }
        // Verdict keys on median only — aggregate is not more conservative, it is wrong
        // under massive activations (L37→L38: med 404→413 vs agg 14222→116670).
        let median_series: Vec<f32> = layer_norms.iter().map(|s| s.median).collect();
        let (stable, stab_detail) = stability_verdict(&median_series);
        println!();
        println!("=== stability ===");
        println!("{stab_detail}");
        println!(
            "verdict: {}",
            if stable {
                "STABLE — no geometric blow-up or collapse across the stack"
            } else {
                "UNSTABLE — monotonic trend or ratio outside sane band"
            }
        );
        checks.push(CheckRow {
            name: "stack_stability".into(),
            pass: stable && layer_norms.len() == cfg.num_hidden_layers,
            detail: if layer_norms.len() != cfg.num_hidden_layers {
                format!(
                    "only {}/{} layers traced; {}",
                    layer_norms.len(),
                    cfg.num_hidden_layers,
                    stab_detail
                )
            } else {
                stab_detail
            },
        });
    }

    // ── 7. Determinism (same process, twice) ────────────────────────────
    println!();
    println!("=== determinism (same process, second forward) ===");
    let mut det_pass = false;
    let mut det_detail = String::new();
    if blocked_on_sibling {
        det_detail = "skipped — forward blocked on sibling".into();
        println!("{det_detail}");
    } else {
        // Second forward into a fresh logits tile; compare bit-identical.
        let logits2 = zeros_f32(&mut gpu, &[n_tokens, PARENT_VOCAB])?;
        let t1 = Instant::now();
        parent_model_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut scratch,
            &token_ids,
            0,
            &logits2,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("deepseek4 parent: sync det2: {e:?}"))?;
        let det_s = t1.elapsed().as_secs_f64();
        let host2 = download_f32(&gpu, &logits2, n_tokens * PARENT_VOCAB)?;
        let mut n_mismatch = 0usize;
        let mut first_mismatch = None;
        for (i, (a, b)) in logits_host.iter().zip(host2.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                n_mismatch += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, *a, *b));
                }
            }
        }
        det_pass = n_mismatch == 0;
        det_detail = if det_pass {
            format!("bit-identical ({:.3} s second forward)", det_s)
        } else {
            format!(
                "{n_mismatch} mismatches; first={first_mismatch:?}"
            )
        };
        println!("{det_detail}");

        // Cross-process hash: hash the logits; a fresh process can re-run and
        // compare. We report the hash here so the handoff can pin it.
        let logits_sha = sha256_bytes(f32_slice_as_le_bytes(&logits_host));
        println!("logits_sha256 (in-process reference) = {logits_sha}");
        println!(
            "cross-process: re-run this binary on the same input and compare logits_sha256; \
             bit-identical within process is required, cross-process is the handoff pin"
        );
        det_detail = format!("{det_detail}; logits_sha256={logits_sha}");
    }
    checks.push(CheckRow {
        name: "determinism_in_process".into(),
        pass: det_pass,
        detail: det_detail,
    });

    // ── 8. Coherence eyeball ────────────────────────────────────────────
    println!();
    println!("=== coherence eyeball ===");
    let mut coherence_text = String::new();
    let mut coherence_ok = false;
    if blocked_on_sibling || logits_host.is_empty() {
        println!("skipped — no full-model logits");
        checks.push(CheckRow {
            name: "coherence_eyeball".into(),
            pass: false,
            detail: "skipped (forward incomplete)".into(),
        });
    } else {
        // Decode argmax continuation of the fixed sequence.
        let argmax_text = tokenizer.decode(&argmax_ids);
        println!("argmax decode of fixed {n_tokens}-token input continuation:");
        println!("  {argmax_text:?}");

        // Short real prompt → encode → forward → greedy 16-token continuation.
        // We only have a prefill path here; do a single forward on the prompt
        // tokens and take the last-position argmax as the next token, then
        // stop (full autoregressive decode would need start_pos wiring and
        // is out of scope for the gate bar). Print last-pos top-5 instead.
        let prompt_ids = tokenizer.encode(COHERENCE_PROMPT);
        println!(
            "prompt = {COHERENCE_PROMPT:?} → {} tokens: {:?}",
            prompt_ids.len(),
            prompt_ids
        );
        if prompt_ids.is_empty() {
            println!("WARN: prompt encoded to 0 tokens");
        } else if prompt_ids.len() > n_tokens {
            println!(
                "WARN: prompt has {} tokens > scratch max_rows {n_tokens}; \
                 truncating to fit",
                prompt_ids.len()
            );
        }
        let p_rows = prompt_ids.len().min(n_tokens).max(1);
        let p_ids: Vec<u32> = prompt_ids.iter().copied().take(p_rows).collect();
        // Reuse logits tile (sized for n_tokens >= p_rows).
        let p_logits = zeros_f32(&mut gpu, &[p_rows, PARENT_VOCAB])?;
        match parent_model_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut scratch,
            &p_ids,
            0,
            &p_logits,
        ) {
            Ok(()) => {
                let ph = download_f32(&gpu, &p_logits, p_rows * PARENT_VOCAB)?;
                let last = &ph[(p_rows - 1) * PARENT_VOCAB..p_rows * PARENT_VOCAB];
                let top = top_k(last, 5);
                println!("last-position top-5:");
                for (rank, (id, val)) in top.iter().enumerate() {
                    let piece = tokenizer.decode(&[*id as u32]);
                    println!("  #{rank} id={id} logit={val:.4} piece={piece:?}");
                }
                let (aid, _aval) = argmax(last);
                let cont = tokenizer.decode(&[aid as u32]);
                // Coherence criterion: the REAL prompt's next-token prediction,
                // not the fixed PRNG sequence's argmax (that input is random
                // token ids and is not expected to decode as language).
                let cont_l = cont.to_lowercase();
                let top0_piece = tokenizer.decode(&[top[0].0 as u32]);
                let top0_l = top0_piece.to_lowercase();
                let looks_paris = cont_l.contains("paris")
                    || cont_l.contains("巴黎")
                    || top0_l.contains("paris")
                    || top0_l.contains("巴黎")
                    || top.iter().take(3).any(|(id, _)| {
                        let p = tokenizer.decode(&[*id as u32]).to_lowercase();
                        p.contains("paris") || p.contains("巴黎")
                    });
                coherence_ok = looks_paris;
                coherence_text = format!(
                    "prompt={COHERENCE_PROMPT:?} + greedy_next={cont:?}  \
                     top0={top0_piece:?}  looks_like_language={coherence_ok}  \
                     fixed_argmax_decode={argmax_text:?}"
                );
                println!(
                    "coherence criterion (real prompt next-token): looks_like_language = {coherence_ok}"
                );
                println!(
                    "human read: {}",
                    if coherence_ok {
                        "next-token for 'The capital of France is' is Paris-like — coherent"
                    } else {
                        "next-token is NOT Paris-like — inspect top-5 above"
                    }
                );
                // Still report the fixed-sequence argmax printable fraction as
                // diagnostic only (random token ids → not expected to be English).
                let printable = argmax_text
                    .chars()
                    .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
                    .count();
                let frac = if argmax_text.is_empty() {
                    0.0
                } else {
                    printable as f64 / argmax_text.chars().count().max(1) as f64
                };
                println!(
                    "fixed-seq argmax printable fraction = {frac:.3} (diagnostic only; random ids)"
                );
            }
            Err(e) => {
                coherence_text = format!("coherence forward failed: {e}");
                println!("{coherence_text}");
            }
        }
        checks.push(CheckRow {
            name: "coherence_eyeball".into(),
            pass: coherence_ok,
            detail: coherence_text.clone(),
        });
    }

    // ── 9. .plog + manifest ─────────────────────────────────────────────
    println!();
    println!("=== plog + manifest ===");
    let mut plog_ok = false;
    let mut plog_detail = String::new();
    if let Some(plog_path) = args.plog.as_ref() {
        if blocked_on_sibling || logits_host.is_empty() {
            plog_detail = "skipped — no full-model logits to write".into();
            println!("{plog_detail}");
            checks.push(CheckRow {
                name: "plog_write".into(),
                pass: false,
                detail: plog_detail.clone(),
            });
        } else {
            if let Some(parent) = plog_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        format!(
                            "deepseek4 parent: create plog dir {}: {e}",
                            parent.display()
                        )
                    })?;
                }
            }
            let mut w = PlogWriter::create(plog_path, n_tokens, PARENT_VOCAB)?;
            parent_logits_to_plog(&gpu, &logits, n_tokens, PARENT_VOCAB, &mut w)?;
            w.finish()?;
            let plog_sha = sha256_file(plog_path)?;
            let plog_bytes = std::fs::metadata(plog_path)
                .map_err(|e| format!("deepseek4 parent: plog metadata: {e}"))?
                .len();
            let expect_bytes =
                8u64 + 4 + 4 + 8 + (n_tokens as u64) * (PARENT_VOCAB as u64) * 4;
            plog_ok = plog_bytes == expect_bytes;
            plog_detail = format!(
                "path={} bytes={plog_bytes} (expect {expect_bytes}) sha256={plog_sha}",
                plog_path.display()
            );
            println!("{plog_detail}");
            checks.push(CheckRow {
                name: "plog_write".into(),
                pass: plog_ok,
                detail: plog_detail.clone(),
            });

            // Manifest sidecar.
            let manifest_path = args.manifest.clone().unwrap_or_else(|| {
                let mut p = plog_path.clone();
                p.set_extension("manifest.json");
                p
            });
            let (producer, engine) = ParentManifest::probe_environment("gfx942")?;
            let source_info = build_source_info(model_path, args.skip_shard_hashes)?;
            let corpus = CorpusInfo {
                token_ids_sha256: token_ids_sha.clone(),
                n_tokens,
                description: format!(
                    "Gate 6 parent baseline: {n_tokens} tokens from {token_source_desc}; \
                     promoted_input={promoted_input}"
                ),
            };
            let outputs = vec![OutputInfo {
                path: plog_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("out.plog")
                    .to_string(),
                sha256: plog_sha.clone(),
                bytes: plog_bytes,
                kind: OutputKind::Logits,
            }];
            let manifest = ParentManifest {
                schema: MANIFEST_SCHEMA.to_string(),
                produced_utc: utc_now_rfc3339(),
                producer,
                engine,
                source: source_info,
                model: ModelInfo {
                    model_type: cfg.model_type.clone(),
                    num_hidden_layers: cfg.num_hidden_layers,
                    mtp_loaded: false,
                    rope_convention: "yarn".to_string(),
                    quant: ModelQuantInfo {
                        quant_method: cfg.quant_method.clone(),
                        fmt: cfg.fmt.clone(),
                        scale_fmt: cfg.scale_fmt.clone(),
                        expert_dtype: cfg.expert_dtype.clone(),
                        weight_block_size: cfg.weight_block_size,
                    },
                },
                corpus: Some(corpus),
                capture: CaptureInfo {
                    boundary: CaptureBoundary::PostDynamicFp8,
                    tensors: vec![],
                },
                outputs,
            };
            manifest.validate()?;
            manifest.write_to(&manifest_path)?;
            println!(
                "manifest wrote {}  validate=OK  boundary=PostDynamicFp8",
                manifest_path.display()
            );
            checks.push(CheckRow {
                name: "manifest_validate".into(),
                pass: true,
                detail: format!("path={}", manifest_path.display()),
            });
        }
    } else {
        println!("no --plog given; skipping plog/manifest emission");
        // Still emit a no-output manifest if --manifest alone was given.
        if let Some(manifest_path) = args.manifest.as_ref() {
            let (producer, engine) = ParentManifest::probe_environment("gfx942")?;
            let source_info = build_source_info(model_path, args.skip_shard_hashes)?;
            let corpus = CorpusInfo {
                token_ids_sha256: token_ids_sha.clone(),
                n_tokens,
                description: format!(
                    "Gate 6 parent baseline: {n_tokens} tokens from {token_source_desc}; \
                     promoted_input={promoted_input} (no plog emitted)"
                ),
            };
            // validate() forbids outputs without corpus and forbids corpus-
            // free outputs; a corpus with empty outputs is fine.
            let manifest = ParentManifest {
                schema: MANIFEST_SCHEMA.to_string(),
                produced_utc: utc_now_rfc3339(),
                producer,
                engine,
                source: source_info,
                model: ModelInfo {
                    model_type: cfg.model_type.clone(),
                    num_hidden_layers: cfg.num_hidden_layers,
                    mtp_loaded: false,
                    rope_convention: "yarn".to_string(),
                    quant: ModelQuantInfo {
                        quant_method: cfg.quant_method.clone(),
                        fmt: cfg.fmt.clone(),
                        scale_fmt: cfg.scale_fmt.clone(),
                        expert_dtype: cfg.expert_dtype.clone(),
                        weight_block_size: cfg.weight_block_size,
                    },
                },
                corpus: Some(corpus),
                capture: CaptureInfo {
                    boundary: CaptureBoundary::PostDynamicFp8,
                    tensors: vec![],
                },
                outputs: vec![],
            };
            manifest.validate()?;
            manifest.write_to(manifest_path)?;
            println!(
                "manifest (no plog) wrote {}  validate=OK",
                manifest_path.display()
            );
            checks.push(CheckRow {
                name: "manifest_validate".into(),
                pass: true,
                detail: format!("path={} (no plog)", manifest_path.display()),
            });
        }
    }

    // ── 10. Summary table ───────────────────────────────────────────────
    println!();
    println!("=== wall clock ===");
    println!("load:    {load_s:.3} s");
    println!("forward: {fwd_s:.3} s");
    println!("total:   {:.3} s", load_s + fwd_s);
    println!(
        "scratch: {:.3} MiB  logits_tile: {:.3} MiB  weights: {:.3} GiB",
        scratch.bytes() as f64 / (1024.0 * 1024.0),
        logits_bytes as f64 / (1024.0 * 1024.0),
        res.total_bytes() as f64 / GIB,
    );
    if blocked_on_sibling {
        println!();
        println!("sibling_block: {sibling_detail}");
    }

    println!();
    println!("=== PASS/FAIL ===");
    let mut all_pass = true;
    let mut w_name = 0usize;
    for c in &checks {
        w_name = w_name.max(c.name.len());
    }
    for c in &checks {
        let mark = if c.pass { "PASS" } else { "FAIL" };
        if !c.pass {
            all_pass = false;
        }
        println!("  {mark}  {:<w_name$}  {}", c.name, c.detail);
    }
    println!();
    if blocked_on_sibling {
        println!(
            "GATE 5: BLOCKED-ON-SIBLING (compressor/indexer wiring not landed). \
             Everything else above is complete; re-run after AttnCompIdx lands."
        );
        // Non-zero exit: gate is not green.
        return Ok(false);
    }
    if all_pass {
        println!("GATE 5/6: PASS");
    } else {
        println!("GATE 5/6: FAIL");
    }
    Ok(all_pass)
}


// ── Stability ───────────────────────────────────────────────────────────────

/// Report layer-to-layer norm ratios and decide stable / unstable.
fn stability_verdict(norms: &[f32]) -> (bool, String) {
    if norms.is_empty() {
        return (false, "no layer norms".into());
    }
    if norms.len() == 1 {
        let n = norms[0];
        let ok = n.is_finite() && n > 0.0;
        return (
            ok,
            format!("single layer L2={n:.6} ({})", if ok { "ok" } else { "bad" }),
        );
    }
    let mut ratios = Vec::with_capacity(norms.len() - 1);
    let mut out_of_band = 0usize;
    let mut non_finite = 0usize;
    let mut lines = Vec::new();
    for w in norms.windows(2) {
        let (a, b) = (w[0] as f64, w[1] as f64);
        if !a.is_finite() || !b.is_finite() || a <= 0.0 {
            non_finite += 1;
            ratios.push(f64::NAN);
            lines.push(format!("{a:.6} -> {b:.6}  ratio=NaN/inf"));
            continue;
        }
        let r = b / a;
        ratios.push(r);
        let flag = if r < NORM_RATIO_LO || r > NORM_RATIO_HI {
            out_of_band += 1;
            "  <-- OUT OF BAND"
        } else {
            ""
        };
        lines.push(format!("{a:.6} -> {b:.6}  ratio={r:.4}{flag}"));
    }
    // Monotonic trend: all ratios > 1.05 or all < 0.95 over the full stack.
    let finite_ratios: Vec<f64> = ratios.iter().copied().filter(|r| r.is_finite()).collect();
    let mut mono_up = !finite_ratios.is_empty();
    let mut mono_down = !finite_ratios.is_empty();
    for &r in &finite_ratios {
        if r <= 1.05 {
            mono_up = false;
        }
        if r >= 0.95 {
            mono_down = false;
        }
    }
    let mono = mono_up || mono_down;
    let mono_s = if mono_up {
        "MONOTONIC GROWTH"
    } else if mono_down {
        "MONOTONIC DECAY"
    } else {
        "no monotonic trend"
    };

    // Geometric mean ratio.
    let geo = if finite_ratios.is_empty() {
        f64::NAN
    } else {
        let log_sum: f64 = finite_ratios.iter().map(|r| r.ln()).sum();
        (log_sum / finite_ratios.len() as f64).exp()
    };

    let stable = non_finite == 0 && out_of_band == 0 && !mono;
    let summary = format!(
        "layers={}  ratios_out_of_band[{NORM_RATIO_LO}..{NORM_RATIO_HI}]={out_of_band}  \
         non_finite={non_finite}  {mono_s}  geo_mean_ratio={geo:.4}",
        norms.len()
    );
    // Keep the per-step lines available but don't drown the summary; print
    // first/mid/last few if long.
    let preview = if lines.len() <= 8 {
        lines.join(" | ")
    } else {
        let head = lines[..3].join(" | ");
        let mid = &lines[lines.len() / 2];
        let tail = lines[lines.len() - 3..].join(" | ");
        format!("{head} | … | {mid} | … | {tail}")
    };
    (stable, format!("{summary}\n  steps: {preview}"))
}

// ── Check row ───────────────────────────────────────────────────────────────

struct CheckRow {
    name: String,
    pass: bool,
    detail: String,
}

// ── Stats helpers ───────────────────────────────────────────────────────────

fn finite_stats(v: &[f32]) -> (usize, usize, usize, f32, f32) {
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_zero = 0usize;
    let mut acc = 0.0f64;
    let mut amax = 0.0f32;
    for &x in v {
        if x.is_nan() {
            n_nan += 1;
        } else if x.is_infinite() {
            n_inf += 1;
        } else {
            acc += (x as f64) * (x as f64);
            amax = amax.max(x.abs());
            if x == 0.0 {
                n_zero += 1;
            }
        }
    }
    (n_nan, n_inf, n_zero, acc.sqrt() as f32, amax)
}

fn mean_std(v: &[f32]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i, best_v)
}

fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.into_iter().take(k).map(|i| (i, row[i])).collect()
}

// ── Token selection ─────────────────────────────────────────────────────────

/// SplitMix64-derived deterministic token ids in `[0, VOCAB)`.
fn select_token_ids(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // splitmix64
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push((z % (PARENT_VOCAB as u64)) as u32);
    }
    out
}

/// Load flat u32 LE token-ids file produced by `ds4_tokenize_corpus`.
fn read_token_ids_file(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "deepseek4 parent: read token-ids {}: {e}",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "deepseek4 parent: token-ids file {} has {} bytes (not a multiple of 4)",
            path.display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// ── IO helpers ──────────────────────────────────────────────────────────────

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("deepseek4 parent: alloc {shape:?}: {e:?}"))?;
    let nelems: usize = shape.iter().product();
    let zeros = vec![0u8; nelems.saturating_mul(4)];
    gpu.hip
        .memcpy_htod(&t.buf, &zeros)
        .map_err(|e| format!("deepseek4 parent: zero-fill: {e:?}"))?;
    Ok(t)
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "deepseek4 parent: download_f32 expects F32 (got {:?})",
            t.dtype
        ));
    }
    let nbytes = nelems
        .checked_mul(4)
        .ok_or_else(|| "deepseek4 parent: download_f32 overflow".to_string())?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: download_f32 buffer too small ({} < {nbytes})",
            t.buf.size()
        ));
    }
    let mut host = vec![0.0f32; nelems];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download_f32: {e:?}"))?;
    Ok(host)
}

fn f32_slice_as_le_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn u32_slice_as_le_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    tokens: usize,
    /// True when the user passed `--tokens` explicitly (allows truncating
    /// a longer `--token-ids` file).
    tokens_explicit: bool,
    token_ids: Option<PathBuf>,
    plog: Option<PathBuf>,
    manifest: Option<PathBuf>,
    skip_shard_hashes: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut tokens = DEFAULT_TOKENS;
    let mut tokens_explicit = false;
    let mut token_ids: Option<PathBuf> = None;
    let mut plog: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut skip_shard_hashes = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "flag --model missing value".to_string())?,
                );
            }
            "--tokens" => {
                let v = args
                    .next()
                    .ok_or_else(|| "flag --tokens missing value".to_string())?;
                tokens = v.parse().map_err(|e| format!("--tokens: {e}"))?;
                if tokens == 0 {
                    return Err("--tokens must be > 0".into());
                }
                tokens_explicit = true;
            }
            "--token-ids" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --token-ids missing value".to_string())?;
                token_ids = Some(PathBuf::from(p));
            }
            "--plog" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --plog missing value".to_string())?;
                plog = Some(PathBuf::from(p));
            }
            "--manifest" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --manifest missing value".to_string())?;
                manifest = Some(PathBuf::from(p));
            }
            "--skip-shard-hashes" => skip_shard_hashes = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_forward_gate --model <dir> \
                     [--tokens 32 | --token-ids FILE] [--plog OUT.plog] \
                     [--manifest out/manifest.json] [--skip-shard-hashes]\n\
                     Prefer --token-ids (flat u32 LE from ds4_tokenize_corpus) \
                     for any promoted artifact."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    Ok(Args {
        model,
        tokens,
        tokens_explicit,
        token_ids,
        plog,
        manifest,
        skip_shard_hashes,
    })
}

// ── Manifest source pinning ─────────────────────────────────────────────────

fn build_source_info(root: &Path, skip_shard_hashes: bool) -> Result<SourceInfo, String> {
    let root_str = root
        .to_str()
        .ok_or_else(|| "deepseek4 parent: model root is not valid UTF-8".to_string())?
        .to_string();

    let config_path = root.join("config.json");
    let index_path = root.join("model.safetensors.index.json");
    let tokenizer_path = root.join("tokenizer.json");

    let config_sha256 = sha256_file(&config_path)?;
    let index_sha256 = if index_path.is_file() {
        sha256_file(&index_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing index file {}",
            index_path.display()
        ));
    };
    let tokenizer_sha256 = if tokenizer_path.is_file() {
        sha256_file(&tokenizer_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing tokenizer.json at {}",
            tokenizer_path.display()
        ));
    };

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("deepseek4 parent: read_dir {}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err("deepseek4 parent: no .safetensors shards found".into());
    }

    let hash_t0 = Instant::now();
    let mut shards = Vec::with_capacity(shard_paths.len());
    for (i, p) in shard_paths.iter().enumerate() {
        let meta = std::fs::metadata(p)
            .map_err(|e| format!("deepseek4 parent: metadata {}: {e}", p.display()))?;
        let bytes = meta.len();
        let file = p
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("deepseek4 parent: bad shard name {}", p.display()))?
            .to_string();
        let sha256 = if skip_shard_hashes {
            format!("SKIPPED_SHARD_HASH_{i:02}")
        } else {
            let t0 = Instant::now();
            let h = sha256_file(p)?;
            println!(
                "  hashed {file} ({bytes} bytes) in {:.1} s → {}",
                t0.elapsed().as_secs_f64(),
                &h[..16.min(h.len())]
            );
            h
        };
        shards.push(ShardInfo {
            file,
            sha256,
            bytes,
        });
    }
    if !skip_shard_hashes {
        println!(
            "hashed {} shards in {:.1} s",
            shards.len(),
            hash_t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "SKIPPED hashing {} shards (--skip-shard-hashes); placeholders are not a pin",
            shards.len()
        );
    }

    Ok(SourceInfo {
        root: root_str,
        index_sha256,
        shards,
        config_sha256,
        tokenizer_sha256,
    })
}

fn utc_now_rfc3339() -> String {
    if let Ok(out) = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

// Silence unused-import warnings for constants referenced only in docs /
// diagnostic paths that the compiler may not see as used on all branches.
#[allow(dead_code)]
fn _keep_shape_consts() {
    let _ = (PARENT_DIM, PARENT_HEAD_DIM, PARENT_HC_DIM, PARENT_HC_MULT);
}
