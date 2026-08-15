// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Multi-length quantized logit capture: load HFQ once, run several
//! token-id prefixes (each a strict prefix of the next), write one
//! `.plog` per length. Avoids re-hashing/re-loading the ~82 GB model
//! between lengths for the length-sweep measurement.
//!
//! Usage:
//! ```text
//! ds4_quant_plog_multi --model <path.hfq> --expect-sha256 <hex> \
//!     --token-ids L=tokens_L.bin --plog L=out_L.plog \
//!     [--token-ids M=tokens_M.bin --plog M=out_M.plog ...] \
//!     [--route-scale S]
//! ```
//! Pair count must match; pairs are run shortest-first. Between lengths the
//! decode state is fully reset (`reset` + `zero_decode_caches`).

use hipfire_arch_deepseek4::forward::decode_step;
use hipfire_ds4_parent::manifest::{
    sha256_bytes, sha256_file, CaptureBoundary, CaptureInfo, CorpusInfo, ModelInfo, ModelQuantInfo,
    OutputInfo, OutputKind, ParentManifest, ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_ds4_parent::plog::PlogWriter;
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4Config};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Pair {
    label: String,
    token_ids: PathBuf,
    plog: PathBuf,
}

fn parse_labeled(flag: &str, raw: &str) -> Result<(String, PathBuf), String> {
    let (lab, path) = raw
        .split_once('=')
        .ok_or_else(|| format!("deepseek4 parent: {flag} wants LABEL=PATH, got {raw}"))?;
    if lab.is_empty() || path.is_empty() {
        return Err(format!("deepseek4 parent: empty label/path in {flag} {raw}"));
    }
    Ok((lab.to_string(), PathBuf::from(path)))
}

fn run() -> Result<(), String> {
    let mut model: Option<PathBuf> = None;
    let mut expect_sha256: Option<String> = None;
    let mut route_scale: Option<f32> = None;
    let mut token_specs: Vec<(String, PathBuf)> = Vec::new();
    let mut plog_specs: Vec<(String, PathBuf)> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(PathBuf::from(
                    args.next()
                        .ok_or("deepseek4 parent: --model needs a value")?,
                ));
            }
            "--expect-sha256" => {
                expect_sha256 = Some(
                    args.next()
                        .ok_or("deepseek4 parent: --expect-sha256 needs a value")?,
                );
            }
            "--token-ids" => {
                let raw = args
                    .next()
                    .ok_or("deepseek4 parent: --token-ids needs LABEL=PATH")?;
                token_specs.push(parse_labeled("--token-ids", &raw)?);
            }
            "--plog" => {
                let raw = args
                    .next()
                    .ok_or("deepseek4 parent: --plog needs LABEL=PATH")?;
                plog_specs.push(parse_labeled("--plog", &raw)?);
            }
            "--route-scale" => {
                let s: f32 = args
                    .next()
                    .ok_or("deepseek4 parent: --route-scale needs a value")?
                    .parse()
                    .map_err(|e| format!("deepseek4 parent: --route-scale: {e}"))?;
                route_scale = Some(s);
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: ds4_quant_plog_multi --model <path.hfq> --expect-sha256 <hex> \\\n\
                     \t--token-ids L=tokens_L.bin --plog L=out_L.plog \\\n\
                     \t[--token-ids M=... --plog M=...] [--route-scale S]"
                );
                return Ok(());
            }
            other => return Err(format!("deepseek4 parent: unknown flag {other}")),
        }
    }

    let model_path = model.ok_or("missing --model")?;
    let expect_sha256 = expect_sha256.ok_or("missing --expect-sha256")?;
    if token_specs.is_empty() {
        return Err("deepseek4 parent: need at least one --token-ids LABEL=PATH".into());
    }
    if token_specs.len() != plog_specs.len() {
        return Err(format!(
            "deepseek4 parent: --token-ids count {} != --plog count {}",
            token_specs.len(),
            plog_specs.len()
        ));
    }
    // Match by label; preserve token_specs order then sort by n_tokens after load.
    let mut pairs: Vec<Pair> = Vec::with_capacity(token_specs.len());
    for (lab, tok) in &token_specs {
        let plog = plog_specs
            .iter()
            .find(|(l, _)| l == lab)
            .map(|(_, p)| p.clone())
            .ok_or_else(|| format!("deepseek4 parent: no --plog for label {lab}"))?;
        pairs.push(Pair {
            label: lab.clone(),
            token_ids: tok.clone(),
            plog,
        });
    }

    println!("=== ds4_quant_plog_multi (length sweep capture) ===");
    println!("model: {}", model_path.display());
    println!("expect-sha256: {expect_sha256}");
    println!("pairs: {}", pairs.len());
    for p in &pairs {
        println!(
            "  [{}] tokens={} -> plog={}",
            p.label,
            p.token_ids.display(),
            p.plog.display()
        );
    }
    println!();

    // ── 1. Mandatory model sha256 pin ───────────────────────────────────
    println!("=== model sha256 (mandatory) ===");
    let hash_t0 = Instant::now();
    let model_sha = sha256_file(&model_path)?;
    let hash_s = hash_t0.elapsed().as_secs_f64();
    let model_bytes = std::fs::metadata(&model_path)
        .map_err(|e| format!("deepseek4 parent: model metadata: {e}"))?
        .len();
    println!("got:  {model_sha}  ({model_bytes} bytes in {hash_s:.1} s)");
    println!("want: {expect_sha256}");
    if !eq_hex_ci(&model_sha, &expect_sha256) {
        return Err(format!(
            "deepseek4 parent: model sha256 mismatch — refusing to load \
             (got {model_sha}, want {expect_sha256}). Filename is not provenance."
        ));
    }
    println!("sha256: OK");
    println!();

    // ── 2. Load all token files; sort shortest-first ────────────────────
    let mut loaded: Vec<(Pair, Vec<u32>, String)> = Vec::new();
    for p in pairs {
        let ids = read_token_ids(&p.token_ids)?;
        if ids.is_empty() {
            return Err(format!(
                "deepseek4 parent: empty token-ids {}",
                p.token_ids.display()
            ));
        }
        let sha = sha256_bytes(u32_slice_as_le_bytes(&ids));
        println!(
            "tokens[{}]: n={} sha256={} path={}",
            p.label,
            ids.len(),
            sha,
            p.token_ids.display()
        );
        loaded.push((p, ids, sha));
    }
    // Strict-prefix check: each longer sequence must start with the shorter.
    loaded.sort_by_key(|(_, ids, _)| ids.len());
    for w in loaded.windows(2) {
        let (ref a, ref a_ids, _) = w[0];
        let (ref b, ref b_ids, _) = w[1];
        if !b_ids.starts_with(a_ids) {
            return Err(format!(
                "deepseek4 parent: tokens[{}] (n={}) is NOT a prefix of tokens[{}] (n={}) — \
                 length sweep would be confounded",
                a.label,
                a_ids.len(),
                b.label,
                b_ids.len()
            ));
        }
        println!(
            "prefix OK: [{}] n={} ⊂ [{}] n={}",
            a.label,
            a_ids.len(),
            b.label,
            b_ids.len()
        );
    }
    println!();

    // ── 3. route_scale ──────────────────────────────────────────────────
    if let Some(s) = route_scale {
        // SAFETY: single-threaded main; set before any forward.
        unsafe {
            std::env::set_var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE", s.to_string());
        }
        println!("route_scale: override HIPFIRE_DEEPSEEK4_ROUTE_SCALE={s}");
    } else {
        println!("route_scale: unset — will use cfg.routed_scaling_factor from the model");
    }

    // ── 4. Load model once ──────────────────────────────────────────────
    println!("=== load ===");
    let mut hfq = HfqFile::open(&model_path).map_err(|e| {
        format!(
            "deepseek4 parent: open HFQ {}: {e:?}",
            model_path.display()
        )
    })?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)
        .map_err(|e| format!("deepseek4 parent: config_from_hfq: {e}"))?;
    let cfg_route = cfg.routed_scaling_factor;
    let effective_route = route_scale.unwrap_or(cfg_route);
    println!(
        "route_scale: effective={effective_route} (cfg.routed_scaling_factor={cfg_route})"
    );
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
    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    println!("gpu: {}", gpu.arch);
    if !gpu.arch.contains("gfx942")
        && std::env::var_os("HIPFIRE_DS4_QUANT_PLOG_ALLOW_NON_GFX942").is_none()
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
    drop(hfq);
    let load_s = load_t0.elapsed().as_secs_f64();
    println!("loaded in {load_s:.3} s");
    println!();

    let quant_label = if cfg.mq2r { "mq2r" } else { "mq2lloyd" };
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

    // ── 5. Per-length sequential decode ─────────────────────────────────
    let wall0 = Instant::now();
    for (pair, token_ids, token_ids_sha) in &loaded {
        let n_tokens = token_ids.len();
        println!(
            "=== forward [{}] n={n_tokens} (sequential decode_step) ===",
            pair.label
        );

        // Fresh KV / compressor state for every length.
        state.reset();
        state.zero_decode_caches(&mut gpu);

        if let Some(parent) = pair.plog.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "deepseek4 parent: create plog dir {}: {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let mut w = PlogWriter::create(&pair.plog, n_tokens, cfg.vocab_size)?;
        let mut sum_sq = 0.0f64;
        let mut n_finite = 0u64;
        let mut n_nan = 0u64;
        let mut n_inf = 0u64;
        let mut last_row: Vec<f32> = Vec::new();
        let fwd_t0 = Instant::now();
        for (pos, &tok) in token_ids.iter().enumerate() {
            let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)
                .map_err(|e| {
                    format!(
                        "deepseek4 parent: decode_step label={} pos={pos} tok={tok}: {e}",
                        pair.label
                    )
                })?;
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
                    sum_sq += (v as f64) * (v as f64);
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
                    "  [{}] pos={:5}/{}  elapsed={:.1}s  ({:.2} tok/s)",
                    pair.label,
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
        let l2 = sum_sq.sqrt();
        println!(
            "[{}] forward done in {fwd_s:.3} s ({:.2} tok/s)  L2={l2:.6}  finite={n_finite} NaN={n_nan} Inf={n_inf}",
            pair.label,
            n_tokens as f64 / fwd_s.max(1e-9)
        );
        if n_nan > 0 || n_inf > 0 {
            eprintln!(
                "WARN: [{}] non-finite logits NaN={n_nan} Inf={n_inf}",
                pair.label
            );
        }
        if !last_row.is_empty() {
            let (id, v) = argmax(&last_row);
            let piece = tokenizer.decode(&[id as u32]);
            println!("[{}] last-pos top-1 id={id} logit={v:.4} piece={piece:?}", pair.label);
        }

        let plog_sha = sha256_file(&pair.plog)?;
        let plog_bytes = std::fs::metadata(&pair.plog)
            .map_err(|e| format!("deepseek4 parent: plog metadata: {e}"))?
            .len();
        let expect_bytes =
            8u64 + 4 + 4 + 8 + (n_tokens as u64) * (cfg.vocab_size as u64) * 4;
        println!(
            "[{}] plog path={} bytes={plog_bytes} (expect {expect_bytes}) sha256={plog_sha}",
            pair.label,
            pair.plog.display()
        );
        if plog_bytes != expect_bytes {
            return Err(format!(
                "deepseek4 parent: [{}] plog size {plog_bytes} != expect {expect_bytes}",
                pair.label
            ));
        }

        // Manifest per length (lightweight provenance).
        chdir_to_git_worktree()?;
        let mut manifest_path = pair.plog.clone();
        manifest_path.set_extension("manifest.json");
        let arch_label = gpu.arch.clone();
        let (producer, engine) = ParentManifest::probe_environment(&arch_label)?;
        let route_desc = if let Some(s) = route_scale {
            format!(
                "route_scale_override={s} (HIPFIRE_DEEPSEEK4_ROUTE_SCALE); cfg.routed_scaling_factor={cfg_route}"
            )
        } else {
            format!("route_scale=cfg.routed_scaling_factor={cfg_route} (no override)")
        };
        let corpus = CorpusInfo {
            token_ids_sha256: token_ids_sha.clone(),
            n_tokens,
            description: format!(
                "Multi-length quant HFPLOG01; label={}; token-ids {} sha256={token_ids_sha}; \
                 sequential decode_step 0..{}; {route_desc}; quant={quant_label}; multi-load",
                pair.label,
                pair.token_ids.display(),
                n_tokens.saturating_sub(1),
            ),
        };
        let plog_name = pair
            .plog
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
        let source = SourceInfo {
            root: root_str.clone(),
            index_sha256: model_sha.clone(),
            shards: vec![ShardInfo {
                file: model_file.clone(),
                sha256: model_sha.clone(),
                bytes: model_bytes,
            }],
            config_sha256: model_sha.clone(),
            tokenizer_sha256: model_sha.clone(),
        };
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
        println!(
            "[{}] manifest {} validate=OK boundary=PostDynamicFp8 route={effective_route}",
            pair.label,
            manifest_path.display()
        );
        println!();
    }

    println!("=== multi-length summary ===");
    println!("quant={quant_label}  model_sha={model_sha}");
    println!("load_s={load_s:.3}  hash_s={hash_s:.1}  wall_s={:.1}", wall0.elapsed().as_secs_f64());
    println!("effective_route_scale={effective_route}");
    for (pair, ids, sha) in &loaded {
        println!(
            "  [{}] n={} tok_sha={sha} plog={}",
            pair.label,
            ids.len(),
            pair.plog.display()
        );
    }
    let _ = tokenizer; // silence if unused on some cfgs
    Ok(())
}

// ── helpers (mirrors ds4_quant_plog) ────────────────────────────────────────

fn chdir_to_git_worktree() -> Result<(), String> {
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
