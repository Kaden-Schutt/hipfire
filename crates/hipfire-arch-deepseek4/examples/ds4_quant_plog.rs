// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Quantized-side logit capture for Gate 6 parent-calibration KLD.
//!
//! Loads an MQ2R / MQ2-Lloyd HFQ model, runs a token sequence from a flat
//! `u32` LE file (byte-identical to the parent run), and writes logits in the
//! parent `HFPLOG01` (`.plog`) format plus a provenance `ParentManifest`.
//!
//! Usage:
//! ```text
//! ds4_quant_plog --model <path.hfq> --expect-sha256 <hex> \
//!                --token-ids tokens.bin --plog OUT.plog \
//!                [--route-scale S] [--manifest PATH]
//! ```
//!
//! `--expect-sha256` is mandatory: the model file is hashed before load and
//! the run refuses on mismatch. Never re-tokenize here — consume the sibling's
//! `tokens.bin` so parent and quant see identical ids.
//!
//! Forward shape: the quantized path does not expose a single multi-token
//! prefill that returns logits at every position (batched prefill returns
//! only the last-position logits). This binary therefore runs sequential
//! `decode_step` over the full sequence at absolute positions
//! `0..n_tokens-1` with `start_pos` semantics matching a fresh prefill
//! (`state` starts empty, positions are contiguous from 0). That matches the
//! parent single-prefill positional convention; chunked batched-prefill is
//! intentionally not used because it would silently drop intermediate rows.

use hipfire_arch_deepseek4::forward::decode_step;
use hipfire_arch_deepseek4::parent::manifest::{
    sha256_bytes, sha256_file, CaptureBoundary, CaptureInfo, CorpusInfo, ModelInfo, ModelQuantInfo,
    OutputInfo, OutputKind, ParentManifest, ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_arch_deepseek4::parent::plog::PlogWriter;
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4Config};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const COHERENCE_PROMPT: &str = "The capital of France is";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let model_path = Path::new(&args.model);
    if !model_path.is_file() {
        return Err(format!(
            "deepseek4 parent: --model must be an HFQ file, got {}",
            model_path.display()
        ));
    }
    let token_ids_path = Path::new(&args.token_ids);
    if !token_ids_path.is_file() {
        return Err(format!(
            "deepseek4 parent: --token-ids not found: {}",
            token_ids_path.display()
        ));
    }
    let plog_path = Path::new(&args.plog);
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

    println!("=== ds4_quant_plog (Gate 6 quantized capture) ===");
    println!("model: {}", model_path.display());
    println!("expect-sha256: {}", args.expect_sha256);
    println!("token-ids: {}", token_ids_path.display());
    println!("plog: {}", plog_path.display());
    if let Some(m) = args.manifest.as_ref() {
        println!("manifest: {}", m.display());
    }
    println!();

    // ── 1. Mandatory model sha256 pin ───────────────────────────────────
    println!("=== model sha256 (mandatory) ===");
    let hash_t0 = Instant::now();
    let model_sha = sha256_file(model_path)?;
    let hash_s = hash_t0.elapsed().as_secs_f64();
    let model_bytes = std::fs::metadata(model_path)
        .map_err(|e| format!("deepseek4 parent: model metadata: {e}"))?
        .len();
    println!(
        "got:  {model_sha}  ({model_bytes} bytes in {hash_s:.1} s)"
    );
    println!("want: {}", args.expect_sha256);
    if !eq_hex_ci(&model_sha, &args.expect_sha256) {
        return Err(format!(
            "deepseek4 parent: model sha256 mismatch — refusing to load \
             (got {model_sha}, want {}). Filename is not provenance.",
            args.expect_sha256
        ));
    }
    println!("sha256: OK");
    println!();

    // ── 2. Token ids (byte-identical to parent; never re-tokenize) ─────
    let token_ids = read_token_ids(token_ids_path)?;
    let n_tokens = token_ids.len();
    if n_tokens == 0 {
        return Err("deepseek4 parent: --token-ids is empty".into());
    }
    let token_ids_sha = sha256_bytes(u32_slice_as_le_bytes(&token_ids));
    println!("=== tokens ===");
    println!("n_tokens: {n_tokens}");
    println!("token_ids_sha256: {token_ids_sha}");
    print!("token_ids[0..{}] = [", n_tokens.min(8));
    for (i, &t) in token_ids.iter().take(8).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{t}");
    }
    if n_tokens > 8 {
        print!(", ...");
    }
    println!("]");
    println!();

    // ── 3. route_scale override (process-level, logged) ─────────────────
    let route_scale_override = args.route_scale;
    if let Some(s) = route_scale_override {
        // Must be set before any forward that reads the LazyLock.
        // SAFETY: single-threaded main; no concurrent env readers yet.
        unsafe {
            std::env::set_var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE", s.to_string());
        }
        println!("route_scale: override HIPFIRE_DEEPSEEK4_ROUTE_SCALE={s}");
    } else {
        println!(
            "route_scale: unset — will use cfg.routed_scaling_factor from the model"
        );
    }
    println!();

    // ── 4. Load model ───────────────────────────────────────────────────
    println!("=== load ===");
    let mut hfq = HfqFile::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: open HFQ {}: {e:?}",
            model_path.display()
        )
    })?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    // Capture path does not need the DSpark sidecar.
    cfg.load_dspark = false;
    let cfg_route = cfg.routed_scaling_factor;
    let effective_route = route_scale_override.unwrap_or(cfg_route);
    if route_scale_override.is_none() {
        println!(
            "route_scale: using model cfg.routed_scaling_factor={cfg_route} (no override)"
        );
    } else {
        println!(
            "route_scale: effective={effective_route} (cfg.routed_scaling_factor={cfg_route})"
        );
    }
    println!(
        "config: layers={} hidden={} vocab={} window={} mq2r={} experts={} topk={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.sliding_window,
        cfg.mq2r,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );

    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("deepseek4 parent: tokenizer from HFQ: {e:?}"))?;
    println!(
        "tokenizer: bos_id={} eos_id={} vocab~{}",
        tokenizer.bos_id,
        tokenizer.eos_id,
        cfg.vocab_size
    );

    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    println!("gpu: {}", gpu.arch);
    if !gpu.arch.contains("gfx942") && std::env::var_os("HIPFIRE_DS4_QUANT_PLOG_ALLOW_NON_GFX942").is_none()
    {
        return Err(format!(
            "deepseek4 parent: gfx942 required for Gate 6 capture (got {}); \
             set HIPFIRE_DS4_QUANT_PLOG_ALLOW_NON_GFX942=1 to override for dry-run",
            gpu.arch
        ));
    }

    let load_t0 = Instant::now();
    let mut state = DeepseekV4::new_state(&mut gpu, &cfg)?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    // Drop the file handle; weights are on device.
    drop(hfq);
    let load_s = load_t0.elapsed().as_secs_f64();
    println!("loaded in {load_s:.3} s");
    println!();

    // ── 5. Sequential decode_step capture (positions 0..n-1) ────────────
    //
    // Parent runs one prefill over all tokens at start_pos=0 and emits a
    // logit row after every position. Quantized batched prefill only returns
    // the last row, so we mirror the parent shape with per-token decode_step
    // at absolute positions 0,1,...,n_tokens-1 on a fresh state. RoPE /
    // compressor / SWA see the same absolute positions as a single prefill.
    println!("=== forward (sequential decode_step, start_pos=0 contiguous) ===");
    println!(
        "shape note: quantized path cannot emit all-position logits from \
         forward_prefill_batch_chunked (last-row only); using decode_step \
         loop so row t = logits after position t, matching parent HFPLOG01."
    );

    let mut w = PlogWriter::create(plog_path, n_tokens, cfg.vocab_size)?;
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    let mut n_finite = 0u64;
    let mut n_nan = 0u64;
    let mut n_inf = 0u64;
    let mut last_row: Vec<f32> = Vec::new();

    let fwd_t0 = Instant::now();
    for (pos, &tok) in token_ids.iter().enumerate() {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .map_err(|e| format!("deepseek4 parent: decode_step pos={pos} tok={tok}: {e}"))?;
        if logits.len() != cfg.vocab_size {
            return Err(format!(
                "deepseek4 parent: logits len {} != vocab {}",
                logits.len(),
                cfg.vocab_size
            ));
        }
        for &v in &logits {
            if v.is_nan() {
                n_nan += 1;
            } else if v.is_infinite() {
                n_inf += 1;
            } else {
                let x = v as f64;
                sum += x;
                sum_sq += x * x;
                n_finite += 1;
            }
        }
        w.push_row(&logits)?;
        if pos + 1 == n_tokens {
            last_row = logits;
        }
        if pos == 0 || (pos + 1) % 64 == 0 || pos + 1 == n_tokens {
            let elapsed = fwd_t0.elapsed().as_secs_f64();
            let rate = (pos + 1) as f64 / elapsed.max(1e-9);
            println!(
                "  pos={:5}/{}  elapsed={:.1}s  ({:.2} tok/s)",
                pos + 1,
                n_tokens,
                elapsed,
                rate
            );
            let _ = std::io::stdout().flush();
        }
    }
    w.finish()?;
    let fwd_s = fwd_t0.elapsed().as_secs_f64();
    println!("forward done in {fwd_s:.3} s ({:.2} tok/s)", n_tokens as f64 / fwd_s.max(1e-9));

    let mean = if n_finite > 0 {
        sum / n_finite as f64
    } else {
        f64::NAN
    };
    let var = if n_finite > 1 {
        (sum_sq / n_finite as f64) - mean * mean
    } else {
        0.0
    };
    let std = var.max(0.0).sqrt();
    let l2 = sum_sq.sqrt();
    println!("logits: L2={l2:.6}  mean={mean:.6}  std={std:.6}");
    println!("logits: finite={n_finite}  NaN={n_nan}  Inf={n_inf}");
    if n_nan > 0 || n_inf > 0 {
        eprintln!("WARN: non-finite logits present (NaN={n_nan} Inf={n_inf})");
    }
    if !last_row.is_empty() {
        let top = top_k(&last_row, 5);
        println!("last-pos top-5 (id, logit):");
        for (id, v) in &top {
            let piece = tokenizer.decode(&[*id as u32]);
            println!("  {id:>6}  {v:9.4}  {piece:?}");
        }
    }
    println!();

    // ── 6. plog provenance ──────────────────────────────────────────────
    let plog_sha = sha256_file(plog_path)?;
    let plog_bytes = std::fs::metadata(plog_path)
        .map_err(|e| format!("deepseek4 parent: plog metadata: {e}"))?
        .len();
    let expect_bytes =
        8u64 + 4 + 4 + 8 + (n_tokens as u64) * (cfg.vocab_size as u64) * 4;
    println!("=== plog ===");
    println!(
        "path={} bytes={plog_bytes} (expect {expect_bytes}) sha256={plog_sha}",
        plog_path.display()
    );
    if plog_bytes != expect_bytes {
        return Err(format!(
            "deepseek4 parent: plog size {plog_bytes} != expect {expect_bytes}"
        ));
    }
    println!();

    // ── 7. Manifest ─────────────────────────────────────────────────────
    // probe_environment shells out to `git` with the process cwd. Launch
    // scripts / nohup often start with cwd=/, which is not a repo — pin to
    // the worktree that built this binary (or CARGO_MANIFEST_DIR parent).
    chdir_to_git_worktree()?;
    let manifest_path = args.manifest.clone().unwrap_or_else(|| {
        let mut p = plog_path.to_path_buf();
        p.set_extension("manifest.json");
        p
    });
    let arch_label = gpu.arch.clone();
    let (producer, engine) = ParentManifest::probe_environment(&arch_label)?;
    let model_file = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.hfq")
        .to_string();
    let root_str = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_str()
        .unwrap_or(".")
        .to_string();
    // Single-file HFQ: pin the file itself as the sole "shard". index/config/
    // tokenizer digests are the model sha (HFQ embeds metadata) so validate()
    // sees non-empty provenance without inventing multi-shard structure.
    let source = SourceInfo {
        root: root_str,
        index_sha256: model_sha.clone(),
        shards: vec![ShardInfo {
            file: model_file,
            sha256: model_sha.clone(),
            bytes: model_bytes,
        }],
        config_sha256: model_sha.clone(),
        tokenizer_sha256: model_sha.clone(),
    };
    let quant_label = if cfg.mq2r {
        "mq2r"
    } else {
        "mq2lloyd"
    };
    let route_desc = if let Some(s) = route_scale_override {
        format!("route_scale_override={s} (HIPFIRE_DEEPSEEK4_ROUTE_SCALE); cfg.routed_scaling_factor={cfg_route}")
    } else {
        format!("route_scale=cfg.routed_scaling_factor={cfg_route} (no override)")
    };
    let corpus = CorpusInfo {
        token_ids_sha256: token_ids_sha.clone(),
        n_tokens,
        description: format!(
            "Quantized HFPLOG01 capture; token-ids file {} sha256={token_ids_sha}; \
             sequential decode_step positions 0..{}; {route_desc}; quant={quant_label}",
            token_ids_path.display(),
            n_tokens.saturating_sub(1),
        ),
    };
    let plog_name = plog_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("out.plog")
        .to_string();
    let outputs = vec![OutputInfo {
        path: plog_name,
        sha256: plog_sha.clone(),
        bytes: plog_bytes,
        kind: OutputKind::Logits,
    }];
    let manifest = ParentManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        produced_utc: utc_now_rfc3339(),
        producer,
        engine,
        source,
        model: ModelInfo {
            model_type: "deepseek_v4".to_string(),
            num_hidden_layers: cfg.num_hidden_layers,
            mtp_loaded: false,
            rope_convention: "yarn".to_string(),
            quant: ModelQuantInfo {
                quant_method: quant_label.to_string(),
                fmt: if cfg.mq2r {
                    "mfp4g32e8soa+mq2lloyd_experts".to_string()
                } else {
                    "q8_0_shared+mq2lloyd_experts".to_string()
                },
                scale_fmt: "ue8m0".to_string(),
                expert_dtype: "mq2lloyd".to_string(),
                weight_block_size: [128, 128],
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
    println!("=== manifest ===");
    println!(
        "wrote {}  validate=OK  boundary=PostDynamicFp8  effective_route_scale={effective_route}",
        manifest_path.display()
    );
    println!();

    // ── 8. Coherence eyeball (short real prompt, same binary) ───────────
    println!("=== coherence eyeball ===");
    println!("prompt: {COHERENCE_PROMPT:?}");
    // Fresh state so the long capture KV does not bleed in.
    state = DeepseekV4::new_state(&mut gpu, &cfg)?;
    let prompt_ids = tokenizer.encode(COHERENCE_PROMPT);
    if prompt_ids.is_empty() {
        return Err("deepseek4 parent: coherence prompt encoded to zero tokens".into());
    }
    println!("prompt_ids ({}): {:?}", prompt_ids.len(), prompt_ids);
    let mut coh_last: Vec<f32> = Vec::new();
    let coh_t0 = Instant::now();
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        coh_last = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
            .map_err(|e| format!("deepseek4 parent: coherence decode_step: {e}"))?;
    }
    let coh_s = coh_t0.elapsed().as_secs_f64();
    let top = top_k(&coh_last, 5);
    println!("top-5 after prompt ({coh_s:.3} s):");
    let mut reads_as_language = false;
    for (rank, (id, v)) in top.iter().enumerate() {
        let piece = tokenizer.decode(&[*id as u32]);
        println!("  #{rank}  id={id}  logit={v:.4}  piece={piece:?}");
        // Crude coherence: top-1 piece contains a letter and is not empty/garbage.
        if rank == 0 {
            let t = piece.trim();
            reads_as_language = !t.is_empty()
                && t.chars().any(|c| c.is_ascii_alphabetic())
                && t.chars().count() < 32;
        }
    }
    // Also emit a few greedy continuations for the eyeball.
    let mut cont = Vec::new();
    let mut pos = prompt_ids.len() as u32;
    let mut next = top[0].0 as u32;
    for _ in 0..8 {
        cont.push(next);
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos)
            .map_err(|e| format!("deepseek4 parent: coherence cont: {e}"))?;
        next = argmax(&logits).0 as u32;
        pos += 1;
    }
    let cont_text = tokenizer.decode(&cont);
    println!("greedy continuation (8 tok): {cont_text:?}");
    let full = format!("{COHERENCE_PROMPT}{cont_text}");
    println!("joined: {full:?}");
    if reads_as_language {
        println!("coherence: reads as language (top-1 alphabetic piece)");
    } else {
        println!(
            "coherence: DOES NOT read as language — finite garbage is a real outcome, not success"
        );
    }
    println!();

    // ── 9. Summary ──────────────────────────────────────────────────────
    println!("=== summary ===");
    println!("model_sha256:        {model_sha}  (OK)");
    println!("load_s:              {load_s:.3}");
    println!("forward_s:           {fwd_s:.3}");
    println!("logits_L2:           {l2:.6}");
    println!("logits_mean:         {mean:.6}");
    println!("logits_std:          {std:.6}");
    println!("logits_nan:          {n_nan}");
    println!("logits_inf:          {n_inf}");
    println!("plog:                {}", plog_path.display());
    println!("plog_bytes:          {plog_bytes}");
    println!("plog_sha256:         {plog_sha}");
    println!("manifest:            {}", manifest_path.display());
    println!("manifest_validate:   OK");
    println!("effective_route_scale: {effective_route}");
    println!("cfg_routed_scaling_factor: {cfg_route}");
    println!("token_ids_sha256:    {token_ids_sha}");
    println!("coherence_language:  {reads_as_language}");
    Ok(())
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    expect_sha256: String,
    token_ids: String,
    plog: PathBuf,
    route_scale: Option<f32>,
    manifest: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut expect_sha256: Option<String> = None;
    let mut token_ids: Option<String> = None;
    let mut plog: Option<PathBuf> = None;
    let mut route_scale: Option<f32> = None;
    let mut manifest: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(args.next().ok_or("deepseek4 parent: --model needs a value")?);
            }
            "--expect-sha256" => {
                expect_sha256 = Some(
                    args.next()
                        .ok_or("deepseek4 parent: --expect-sha256 needs a value")?,
                );
            }
            "--token-ids" => {
                token_ids = Some(
                    args.next()
                        .ok_or("deepseek4 parent: --token-ids needs a value")?,
                );
            }
            "--plog" => {
                plog = Some(
                    PathBuf::from(
                        args.next()
                            .ok_or("deepseek4 parent: --plog needs a value")?,
                    ),
                );
            }
            "--route-scale" => {
                let v = args
                    .next()
                    .ok_or("deepseek4 parent: --route-scale needs a value")?;
                let s: f32 = v.parse().map_err(|e| {
                    format!("deepseek4 parent: --route-scale parse {v:?}: {e}")
                })?;
                if !s.is_finite() || s <= 0.0 {
                    return Err(format!(
                        "deepseek4 parent: --route-scale must be finite and > 0, got {s}"
                    ));
                }
                route_scale = Some(s);
            }
            "--manifest" => {
                manifest = Some(PathBuf::from(
                    args.next()
                        .ok_or("deepseek4 parent: --manifest needs a value")?,
                ));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_quant_plog --model <path.hfq> --expect-sha256 <hex> \\\n\
                     \t\t--token-ids tokens.bin --plog OUT.plog \\\n\
                     \t\t[--route-scale S] [--manifest PATH]"
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("deepseek4 parent: unknown flag {other}"));
            }
        }
    }

    let model = model.ok_or(
        "deepseek4 parent: missing --model\n\
         usage: ds4_quant_plog --model <path.hfq> --expect-sha256 <hex> \
         --token-ids tokens.bin --plog OUT.plog [--route-scale S] [--manifest PATH]",
    )?;
    let expect_sha256 = expect_sha256.ok_or(
        "deepseek4 parent: missing --expect-sha256 (mandatory; refuse-on-mismatch)",
    )?;
    if expect_sha256.len() != 64 || !expect_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "deepseek4 parent: --expect-sha256 must be 64 hex chars, got {:?}",
            expect_sha256
        ));
    }
    let token_ids = token_ids.ok_or("deepseek4 parent: missing --token-ids")?;
    let plog = plog.ok_or("deepseek4 parent: missing --plog")?;

    Ok(Args {
        model,
        expect_sha256,
        token_ids,
        plog,
        route_scale,
        manifest,
    })
}

// ── helpers ─────────────────────────────────────────────────────────────────
fn chdir_to_git_worktree() -> Result<(), String> {
    // Prefer walking up from the running binary (…/target/release/examples/X
    // → repo root). Fall back to CARGO_MANIFEST_DIR when present (cargo run).
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(mut p) = exe.parent().map(|p| p.to_path_buf()) {
            for _ in 0..8 {
                candidates.push(p.clone());
                if !p.pop() {
                    break;
                }
            }
        }
    }
    if let Ok(m) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut p = PathBuf::from(m);
        candidates.push(p.clone());
        if p.pop() {
            candidates.push(p.clone());
            if p.pop() {
                candidates.push(p);
            }
        }
    }
    for c in &candidates {
        if c.join(".git").exists() {
            std::env::set_current_dir(c).map_err(|e| {
                format!(
                    "deepseek4 parent: chdir to git worktree {}: {e}",
                    c.display()
                )
            })?;
            return Ok(());
        }
    }
    // Leave cwd alone and let probe_environment surface a clear error.
    Ok(())
}


fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "deepseek4 parent: read tokens {}: {e}",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "deepseek4 parent: tokens {} size {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        out.push(u32::from_le_bytes(b));
    }
    Ok(out)
}

fn u32_slice_as_le_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn eq_hex_ci(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .all(|(x, y)| x.to_ascii_lowercase() == y.to_ascii_lowercase())
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
    idx.into_iter()
        .take(k.min(row.len()))
        .map(|i| (i, row[i]))
        .collect()
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

// Keep cfg type referenced for clarity in docs / future fields.
#[allow(dead_code)]
fn _keep_cfg(cfg: &DeepseekV4Config) {
    let _ = cfg.vocab_size;
}
