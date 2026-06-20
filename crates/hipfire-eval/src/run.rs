// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Top-level run dispatch: env entry point, model-glob expansion, and the
//! plan / sweep / status / fetch subcommand handlers.
//!
//! `run_from_env` reads argv+env into an `EvalConfig` and dispatches to the
//! right mode (dry-run plan, multi-SKU sweep, status, dataset fetch, or a real
//! `run_eval`). The core battery executor `run_eval` itself stays at the crate
//! root. Extracted verbatim from the former `hipfire-eval/src/lib.rs` monolith
//! (no behavior change).

use crate::*;

pub fn run_from_env() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprint!("{}", usage());
        return Ok(());
    }
    if args.iter().any(|a| a == "--version") {
        print!("{}", version_report());
        return Ok(());
    }
    let config = parse_args_from(args)?;

    if config.status {
        print_eval_status(&config);
        return Ok(());
    }
    if config.fetch {
        return run_fetch(&config);
    }
    if let Some(spec) = config.models_spec.clone() {
        return run_sweep(&config, &spec);
    }
    if config.dry_run {
        let (ctx, datasets) = dry_run_inputs(&config)?;
        let plan = plan_model(&config, &ctx, &datasets)?;
        print_plans(std::slice::from_ref(&plan));
        return Ok(());
    }
    run_eval(config)
}

/// Expand a `--models` spec (comma-separated globs / prefixes / paths) against
/// the model directory (HIPFIRE_MODELS_DIR or ~/.hipfire/models). A token with
/// `/` or an existing `.hfq` path is taken literally; otherwise it matches model
/// filenames by simple `*` glob, falling back to substring match when it has no
/// `*` (so `qwen3.5` matches every `*qwen3.5*.hfq`).
fn expand_models(spec: &str) -> Result<Vec<String>, String> {
    let models_dir = std::env::var_os("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hipfire")
                .join("models")
        });
    let entries: Vec<(String, PathBuf)> = std::fs::read_dir(&models_dir)
        .map_err(|e| format!("read models dir {}: {e}", models_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "hfq"))
        .filter_map(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| (n.to_string(), p.clone()))
        })
        .collect();

    let mut out: Vec<String> = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if tok.contains('/') || (tok.ends_with(".hfq") && Path::new(tok).exists()) {
            out.push(tok.to_string());
            continue;
        }
        let mut hit = false;
        for (name, path) in &entries {
            let stem = name.trim_end_matches(".hfq");
            let matched = if tok.contains('*') {
                glob_match(tok, name) || glob_match(tok, stem)
            } else {
                name.contains(tok)
            };
            if matched {
                out.push(path.display().to_string());
                hit = true;
            }
        }
        if !hit {
            eprintln!(
                "[eval] --models: no match for '{tok}' in {}",
                models_dir.display()
            );
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        return Err(format!(
            "--models '{spec}' matched no .hfq files in {}",
            models_dir.display()
        ));
    }
    Ok(out)
}

/// Minimal `*`-glob (any-run wildcard; no `?`/classes), enough for model specs.
fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut pos = 0usize;
    // First segment must be a prefix (unless pattern starts with '*').
    if let Some(first) = parts.first() {
        if !text[pos..].starts_with(first) {
            return false;
        }
        pos += first.len();
    }
    for (idx, seg) in parts.iter().enumerate().skip(1) {
        if seg.is_empty() {
            continue;
        }
        if idx == parts.len() - 1 && !pattern.ends_with('*') {
            // Last segment must match the suffix.
            return text[pos..].ends_with(seg);
        }
        match text[pos..].find(seg) {
            Some(off) => pos += off + seg.len(),
            None => return false,
        }
    }
    true
}

// ── Sweep / dry-run / status / fetch machinery ───────────────────────────────

struct CellPlan {
    battery: BatteryId,
    state: &'static str, // CACHED | READY | BLOCKED
    reason: Option<String>,
}

struct ModelPlan {
    model: String,
    cells: Vec<CellPlan>,
}

fn eval_context(config: &EvalConfig) -> EvalContext {
    EvalContext::new_with_overrides(HostProfileOverrides {
        memory_class: config.host_memory_class.clone(),
        memory_width_bits: config.host_memory_width_bits,
        memory_bandwidth_gbps: config.host_memory_bandwidth_gbps,
    })
}

/// Artifact that would BLOCK a battery from running (the cases the user cares
/// about: missing model / corpus / dataset). `None` ⇒ would run. Deliberately
/// does not try to predict executor-binary availability (too coupled); those
/// surface as normal skips at run time.
fn cell_block_reason(
    battery: BatteryId,
    config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
) -> Option<String> {
    if !config.model.is_empty() && !Path::new(&config.model).exists() {
        return Some(format!("model not found: {}", config.model));
    }
    match battery {
        BatteryId::Perplexity => {
            let corpus_rel = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CORPUS").unwrap_or_else(|_| {
                "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt".into()
            });
            let corpus = repo_root()
                .map(|r| r.join(&corpus_rel))
                .unwrap_or_else(|| PathBuf::from(&corpus_rel));
            if !corpus.exists() {
                return Some(format!(
                    "corpus missing ({}) — set HIPFIRE_EVAL_PERPLEXITY_CORPUS or stage under ~/.hipfire/datasets/",
                    corpus.display()
                ));
            }
            None
        }
        BatteryId::Barrage => datasets
            .iter()
            .find(|d| d.status == EvalStatus::Skip)
            .map(|d| {
                format!(
                    "dataset {} unavailable ({}) — run `hipfire eval --fetch`",
                    d.suite.as_str(),
                    d.reason.as_deref().unwrap_or("not cached")
                )
            }),
        _ => None,
    }
}

/// `ctx` (hardware/build) and `datasets` (shared HF datasets) are
/// model-independent — the caller builds them ONCE and passes them in, so a
/// many-model dry-run does one host-profiling pass, not one per SKU.
fn plan_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Result<ModelPlan, String> {
    let mut cells = Vec::new();
    for &battery in &config.batteries {
        let key = result_cache_key(battery, config, ctx, datasets)?;
        let path = result_cache_path(config, &key);
        let cached = config.cache_mode == EvalCacheMode::Use && path.exists();
        let (state, reason) = if cached {
            ("CACHED", None)
        } else if let Some(r) = cell_block_reason(battery, config, datasets) {
            ("BLOCKED", Some(r))
        } else {
            ("READY", None)
        };
        cells.push(CellPlan {
            battery,
            state,
            reason,
        });
    }
    Ok(ModelPlan {
        model: config.model.clone(),
        cells,
    })
}

/// Build the shared dry-run context + dataset manifest once (no network:
/// datasets are resolved with fetch disabled).
fn dry_run_inputs(config: &EvalConfig) -> Result<(EvalContext, Vec<DatasetManifestEntry>), String> {
    let ctx = eval_context(config);
    let mut probe = config.clone();
    probe.fetch_datasets = false;
    let datasets = resolve_datasets(&probe)?;
    Ok((ctx, datasets))
}

fn print_plans(plans: &[ModelPlan]) {
    let (mut cached, mut ready, mut blocked) = (0u32, 0u32, 0u32);
    let mut missing: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    println!("=== dry-run plan (CACHED = skip, READY = would run, BLOCKED = missing artifact) ===");
    for p in plans {
        println!("\n{}", model_artifact_stem(&p.model));
        for c in &p.cells {
            match c.state {
                "CACHED" => cached += 1,
                "READY" => ready += 1,
                _ => {
                    blocked += 1;
                    if let Some(r) = &c.reason {
                        *missing.entry(r.clone()).or_insert(0) += 1;
                    }
                }
            }
            match &c.reason {
                Some(r) => println!("  {:<8} {:<14} {}", c.state, c.battery.as_str(), r),
                None => println!("  {:<8} {}", c.state, c.battery.as_str()),
            }
        }
    }
    println!("\n=== totals: {cached} cached (skip), {ready} would run, {blocked} blocked ===");
    if !missing.is_empty() {
        println!("--- missing artifacts (resolve, then re-run) ---");
        for (reason, n) in &missing {
            println!("  [{n}x] {reason}");
        }
    }
}

fn sweep_base_dir(config: &EvalConfig) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("sweeps")
        .join(format!("{ts}-{}", config.tier.as_str()))
}

fn run_sweep(config: &EvalConfig, spec: &str) -> Result<(), String> {
    let models = expand_models(spec)?;
    let base = sweep_base_dir(config);
    fs::create_dir_all(&base).map_err(|e| format!("create {}: {e}", base.display()))?;
    eprintln!(
        "[eval] sweep: {} model(s){} -> {}",
        models.len(),
        if config.dry_run { " (dry-run)" } else { "" },
        base.display()
    );

    // Hardware/build ctx + shared dataset manifest are model-independent: build
    // ONCE for the whole dry-run (one host-profiling pass, not one per SKU).
    let dry_inputs = if config.dry_run {
        Some(dry_run_inputs(config)?)
    } else {
        None
    };

    let mut plans = Vec::new();
    let mut outcomes: Vec<(String, PathBuf, String)> = Vec::new();
    for m in &models {
        let mut cfg = config.clone();
        cfg.model = m.clone();
        cfg.models_spec = None;
        cfg.out_dir = base.join(model_artifact_stem(m));
        if cfg.draft.is_none() && matches!(cfg.dflash, DflashMode::Auto | DflashMode::On) {
            cfg.draft =
                discover_dflash_draft_for_model(Path::new(m)).map(|p| p.display().to_string());
        }
        if let Some((ctx, datasets)) = &dry_inputs {
            match plan_model(&cfg, ctx, datasets) {
                Ok(p) => plans.push(p),
                Err(e) => eprintln!("[eval] plan {m}: {e}"),
            }
            continue;
        }
        eprintln!("[eval] === {} ===", model_artifact_stem(m));
        // Fail-isolation: a model that errors or panics (e.g. OOM) is recorded
        // and the sweep continues to the next SKU.
        let run_cfg = cfg.clone();
        let outcome =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_eval(run_cfg))) {
                Ok(Ok(())) => "ok".to_string(),
                Ok(Err(e)) => {
                    eprintln!("[eval] {m}: FAILED: {e}");
                    format!("failed: {e}")
                }
                Err(_) => {
                    eprintln!("[eval] {m}: PANIC — skipped, continuing sweep");
                    "panic".to_string()
                }
            };
        outcomes.push((m.clone(), cfg.out_dir.clone(), outcome));
    }

    if config.dry_run {
        print_plans(&plans);
        return Ok(());
    }
    write_sweep_rollup(&base, &outcomes)?;
    Ok(())
}

/// Aggregate each model's results.jsonl into a cross-model rollup (CSV + md).
fn write_sweep_rollup(base: &Path, outcomes: &[(String, PathBuf, String)]) -> Result<(), String> {
    let mut csv = String::from("model,outcome,pass,skip,fail,ppl,gen_tok_s\n");
    let mut md = String::from("| model | outcome | pass | skip | fail | ppl | gen tok/s |\n|---|---|---|---|---|---|---|\n");
    for (model, out_dir, outcome) in outcomes {
        let (mut pass, mut skip, mut fail) = (0u32, 0u32, 0u32);
        let (mut ppl, mut tok_s) = (String::new(), String::new());
        if let Ok(text) = std::fs::read_to_string(out_dir.join("results.jsonl")) {
            for line in text.lines() {
                let Ok(v) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                match v.get("status").and_then(|s| s.as_str()) {
                    Some("pass") => pass += 1,
                    Some("skip") => skip += 1,
                    Some("fail") => fail += 1,
                    _ => {}
                }
                let metrics = v.get("metrics");
                if ppl.is_empty() {
                    if let Some(p) = metrics.and_then(|m| m.get("ppl")).and_then(|p| p.as_f64()) {
                        ppl = format!("{p:.4}");
                    }
                }
                if tok_s.is_empty() {
                    if let Some(t) = metrics
                        .and_then(|m| m.get("gen_tok_s").or_else(|| m.get("tok_s")))
                        .and_then(|t| t.as_f64())
                    {
                        tok_s = format!("{t:.1}");
                    }
                }
            }
        }
        let stem = model_artifact_stem(model);
        csv.push_str(&format!(
            "{stem},{outcome},{pass},{skip},{fail},{ppl},{tok_s}\n"
        ));
        md.push_str(&format!(
            "| {stem} | {outcome} | {pass} | {skip} | {fail} | {ppl} | {tok_s} |\n"
        ));
    }
    let csv_path = base.join("rollup.csv");
    let md_path = base.join("rollup.md");
    fs::write(&csv_path, &csv).map_err(|e| format!("write {}: {e}", csv_path.display()))?;
    fs::write(&md_path, &md).map_err(|e| format!("write {}: {e}", md_path.display()))?;
    println!("\n{md}");
    println!("rollup: {}", csv_path.display());
    Ok(())
}

fn print_eval_status(config: &EvalConfig) {
    let ctx = eval_context(config);
    let cache_entries = std::fs::read_dir(&config.result_cache)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .flat_map(|e| std::fs::read_dir(e.path()).into_iter().flatten())
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .count()
        })
        .unwrap_or(0);
    let models_dir = std::env::var_os("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hipfire")
                .join("models")
        });
    let model_count = std::fs::read_dir(&models_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "hfq"))
                .count()
        })
        .unwrap_or(0);
    println!("hipfire-eval status");
    println!("  hardware_bucket: {}", ctx.host_profile.hardware_bucket);
    println!(
        "  arch / rocm:     {} / {}",
        ctx.arch.as_deref().unwrap_or("?"),
        ctx.rocm.as_deref().unwrap_or("?")
    );
    println!(
        "  binary / commit: {} / {}{}",
        ctx.binary_hash.as_deref().unwrap_or("?"),
        ctx.commit_sha.as_deref().unwrap_or("?"),
        if ctx.git_dirty.unwrap_or(false) {
            " (dirty)"
        } else {
            ""
        }
    );
    println!(
        "  result cache:    {} ({cache_entries} entries)",
        config.result_cache.display()
    );
    println!("  dataset cache:   {}", config.dataset_cache.display());
    println!(
        "  models dir:      {} ({model_count} .hfq)",
        models_dir.display()
    );
}

fn run_fetch(config: &EvalConfig) -> Result<(), String> {
    let mut cfg = config.clone();
    cfg.fetch_datasets = true;
    if cfg.suites.is_empty() {
        cfg.suites = default_suites(EvalTier::Extensive);
    }
    eprintln!(
        "[eval] fetching datasets for suites: {} -> {}",
        cfg.suites
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(","),
        cfg.dataset_cache.display()
    );
    let datasets = resolve_datasets(&cfg)?;
    for d in &datasets {
        println!(
            "  {:<14} {:<12} {}",
            d.suite.as_str(),
            format!("{:?}", d.status),
            d.reason.as_deref().unwrap_or(&d.source)
        );
    }
    println!(
        "Note: KLD references are GPU-generated, not fetched — make them with `collect_artifacts --kldref` or `perplexity --dump-ref` into ~/.hipfire/datasets/kldref/."
    );
    Ok(())
}
