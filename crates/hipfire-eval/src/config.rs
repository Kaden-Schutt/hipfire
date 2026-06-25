// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CLI argument parsing and run defaults.
//!
//! `parse_args_from` builds an `EvalConfig` from argv; `usage`/`version_report`
//! render the help/version text; the `default_*` helpers supply per-tier battery
//! and suite sets plus cache/output directories. The `EvalConfig` struct and the
//! tier/battery/suite enums themselves stay at the crate root as shared
//! vocabulary. Extracted verbatim from the former `hipfire-eval/src/lib.rs`
//! monolith (no behavior change).

use crate::*;

pub fn parse_args_from<I, S>(args: I) -> Result<EvalConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut model: Option<String> = None;
    let mut draft: Option<String> = None;
    let mut baseline: Option<String> = None;
    let mut reference: Option<String> = None;
    let mut tier = EvalTier::Fast;
    let mut batteries: Option<Vec<BatteryId>> = None;
    let mut suites: Vec<SuiteId> = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut kv_mode: Option<String> = None;
    let mut max_tokens = 64usize;
    let mut dflash = DflashMode::Off;
    let mut profile = ProfileMode::Off;
    let mut quality_max_chunks: Option<usize> = None;
    let mut kldref: Option<PathBuf> = None;
    let mut quality_json: Option<PathBuf> = None;
    let mut performance_json: Option<PathBuf> = None;
    let mut evidence_json: Vec<PathBuf> = Vec::new();
    let mut evidence_dirs: Vec<PathBuf> = Vec::new();
    let mut candidate_variant: Option<String> = None;
    let mut baseline_variant: Option<String> = None;
    let mut reference_variant: Option<String> = None;
    let mut performance_candidate_variant: Option<String> = None;
    let mut performance_baseline_variant: Option<String> = None;
    let mut performance_reference_variant: Option<String> = None;
    let mut executor = EvalExecutorMode::Auto;
    let mut fetch_datasets = false;
    let mut offline = false;
    let mut dataset_cache: Option<PathBuf> = None;
    let mut result_cache: Option<PathBuf> = None;
    let mut cache_mode = EvalCacheMode::Use;
    let mut runs = 1usize;
    let mut runs_explicit = false;
    let mut warmup_runs = 0usize;
    let mut benchmark = false;
    let mut host_memory_class: Option<String> = None;
    let mut host_memory_width_bits: Option<u32> = None;
    let mut host_memory_bandwidth_gbps: Option<f64> = None;
    let mut fail_on_admission = false;
    let mut models_spec: Option<String> = None;
    let mut dry_run = false;
    let mut status = false;
    let mut fetch = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--model" => {
                model = Some(take_value(&argv, i, "--model")?);
                i += 2;
            }
            "--models" => {
                models_spec = Some(take_value(&argv, i, "--models")?);
                i += 2;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            "--status" => {
                status = true;
                i += 1;
            }
            "--fetch" => {
                fetch = true;
                i += 1;
            }
            "--draft" => {
                draft = Some(take_value(&argv, i, "--draft")?);
                i += 2;
            }
            "--baseline" => {
                baseline = Some(take_value(&argv, i, "--baseline")?);
                i += 2;
            }
            "--reference" => {
                reference = Some(take_value(&argv, i, "--reference")?);
                i += 2;
            }
            "--tier" => {
                tier = EvalTier::parse(&take_value(&argv, i, "--tier")?)?;
                i += 2;
            }
            "--battery" | "--batteries" => {
                batteries = Some(parse_csv(
                    &take_value(&argv, i, "--battery")?,
                    BatteryId::parse,
                )?);
                i += 2;
            }
            "--suite" | "--suites" => {
                suites.extend(parse_csv(
                    &take_value(&argv, i, "--suite")?,
                    SuiteId::parse,
                )?);
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(take_value(&argv, i, "--out")?));
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = Some(take_value(&argv, i, "--kv-mode")?);
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = parse_usize(&take_value(&argv, i, "--max-tokens")?, "--max-tokens")?;
                i += 2;
            }
            "--dflash" => {
                dflash = DflashMode::parse(&take_value(&argv, i, "--dflash")?)?;
                i += 2;
            }
            "--profile" => {
                profile = ProfileMode::parse(&take_value(&argv, i, "--profile")?)?;
                i += 2;
            }
            "--quality-max-chunks" => {
                quality_max_chunks = Some(parse_usize(
                    &take_value(&argv, i, "--quality-max-chunks")?,
                    "--quality-max-chunks",
                )?);
                i += 2;
            }
            "--kldref" | "--kld-ref" => {
                kldref = Some(PathBuf::from(take_value(&argv, i, "--kldref")?));
                i += 2;
            }
            "--quality-json" => {
                quality_json = Some(PathBuf::from(take_value(&argv, i, "--quality-json")?));
                i += 2;
            }
            "--performance-json" => {
                performance_json = Some(PathBuf::from(take_value(&argv, i, "--performance-json")?));
                i += 2;
            }
            "--evidence-json" => {
                evidence_json.push(PathBuf::from(take_value(&argv, i, "--evidence-json")?));
                i += 2;
            }
            "--evidence-dir" => {
                evidence_dirs.push(PathBuf::from(take_value(&argv, i, "--evidence-dir")?));
                i += 2;
            }
            "--candidate-variant" => {
                candidate_variant = Some(take_value(&argv, i, "--candidate-variant")?);
                i += 2;
            }
            "--baseline-variant" => {
                baseline_variant = Some(take_value(&argv, i, "--baseline-variant")?);
                i += 2;
            }
            "--reference-variant" => {
                reference_variant = Some(take_value(&argv, i, "--reference-variant")?);
                i += 2;
            }
            "--performance-candidate-variant" => {
                performance_candidate_variant =
                    Some(take_value(&argv, i, "--performance-candidate-variant")?);
                i += 2;
            }
            "--performance-baseline-variant" => {
                performance_baseline_variant =
                    Some(take_value(&argv, i, "--performance-baseline-variant")?);
                i += 2;
            }
            "--performance-reference-variant" => {
                performance_reference_variant =
                    Some(take_value(&argv, i, "--performance-reference-variant")?);
                i += 2;
            }
            "--executor" => {
                executor = EvalExecutorMode::parse(&take_value(&argv, i, "--executor")?)?;
                i += 2;
            }
            "--fetch-datasets" => {
                fetch_datasets = true;
                i += 1;
            }
            "--offline" => {
                offline = true;
                i += 1;
            }
            "--dataset-cache" => {
                dataset_cache = Some(PathBuf::from(take_value(&argv, i, "--dataset-cache")?));
                i += 2;
            }
            "--result-cache" | "--cache-dir" => {
                result_cache = Some(PathBuf::from(take_value(&argv, i, argv[i].as_str())?));
                i += 2;
            }
            "--force" => {
                cache_mode = EvalCacheMode::Force;
                i += 1;
            }
            "--regenerate" => {
                cache_mode = EvalCacheMode::Regenerate;
                i += 1;
            }
            "--no-cache" => {
                cache_mode = EvalCacheMode::Off;
                i += 1;
            }
            "--runs" => {
                runs = parse_usize(&take_value(&argv, i, "--runs")?, "--runs")?;
                runs_explicit = true;
                i += 2;
            }
            "--warmup-runs" => {
                warmup_runs =
                    parse_usize(&take_value(&argv, i, "--warmup-runs")?, "--warmup-runs")?;
                i += 2;
            }
            "--benchmark" => {
                benchmark = true;
                i += 1;
            }
            "--host-memory-class" => {
                host_memory_class = Some(take_value(&argv, i, "--host-memory-class")?);
                i += 2;
            }
            "--host-memory-width-bits" => {
                host_memory_width_bits = Some(parse_u32(
                    &take_value(&argv, i, "--host-memory-width-bits")?,
                    "--host-memory-width-bits",
                )?);
                i += 2;
            }
            "--host-memory-bandwidth-gbps" => {
                host_memory_bandwidth_gbps = Some(parse_f64(
                    &take_value(&argv, i, "--host-memory-bandwidth-gbps")?,
                    "--host-memory-bandwidth-gbps",
                )?);
                i += 2;
            }
            "--fail-on-admission" => {
                fail_on_admission = true;
                i += 1;
            }
            other => return Err(format!("unknown arg: {other}\n\n{}", usage())),
        }
    }

    if fetch_datasets && offline {
        return Err("--fetch-datasets and --offline are mutually exclusive".to_string());
    }
    if benchmark && !runs_explicit {
        runs = 5;
    }
    if runs == 0 {
        return Err("--runs must be at least 1".to_string());
    }
    // --model is required for a single run, but --models (sweep), --status, and
    // --fetch supply or don't need it; use a placeholder that run_from_env
    // replaces per sweep iteration. The tiny_quant battery emits + quantizes its
    // own fixtures, so it needs no --model either.
    let tiny_quant_only = batteries
        .as_ref()
        .is_some_and(|b| !b.is_empty() && b.iter().all(|x| *x == BatteryId::TinyQuant));
    let model = match model {
        Some(m) => m,
        None if models_spec.is_some() || status || fetch || tiny_quant_only => String::new(),
        None => return Err(format!("error: --model is required\n\n{}", usage())),
    };
    let batteries = batteries.unwrap_or_else(|| default_batteries(tier));
    if suites.is_empty() && batteries.contains(&BatteryId::Barrage) {
        suites = default_suites(tier);
    }
    suites.sort();
    suites.dedup();
    if draft.is_none() && matches!(dflash, DflashMode::Auto | DflashMode::On) {
        draft = discover_dflash_draft_for_model(Path::new(&model))
            .map(|path| path.display().to_string());
    }
    let out_dir = out_dir.unwrap_or_else(|| default_output_dir(&model, tier));
    let dataset_cache = dataset_cache.unwrap_or_else(default_dataset_cache);
    let result_cache = result_cache.unwrap_or_else(default_result_cache);

    Ok(EvalConfig {
        model,
        draft,
        baseline,
        reference,
        tier,
        batteries,
        suites,
        out_dir,
        kv_mode,
        max_tokens,
        dflash,
        profile,
        quality_max_chunks,
        kldref,
        quality_json,
        performance_json,
        evidence_json,
        evidence_dirs,
        candidate_variant,
        baseline_variant,
        reference_variant,
        performance_candidate_variant,
        performance_baseline_variant,
        performance_reference_variant,
        executor,
        fetch_datasets,
        offline,
        dataset_cache,
        result_cache,
        cache_mode,
        runs,
        warmup_runs,
        benchmark,
        host_memory_class,
        host_memory_width_bits,
        host_memory_bandwidth_gbps,
        fail_on_admission,
        models_spec,
        dry_run,
        status,
        fetch,
    })
}

pub fn usage() -> String {
    "Usage:\n  hipfire-eval --model <model> [--tier fast|medium|long|extensive]\n\n\
     Options:\n\
       --version                print Hipfire eval runner version/git metadata\n\
       --models <glob|csv>      sweep many SKUs from the model dir (e.g. 'qwen3.5,qwen3.6' or 'qwen3.5-9b-*'); per-model out dirs + a cross-model rollup\n\
       --dry-run                plan only: resolve models/batteries/cache/artifacts and report (no tests run, nothing fetched/generated)\n\
       --status                 print cache/dataset/hardware status and exit\n\
       --fetch                  ensure datasets are present (HF fetch), then exit\n\
       --battery <a,b>          smoke,coherence,quality,retrieval,speed,dflash,pflash,agentic,runtime,prompt_shape,structured,barrage,longctx,vision,cask,profile,perplexity,calibrate\n\
       --suite <a,b>            gpqa,lm_eval_micro,humaneval,deep_swe,swe_bench,ruler,nolima,needle_chain,niah,sequential_niah\n\
       --baseline <model>       baseline quantized model for candidate comparison\n\
       --reference <model>      higher precision reference model or fixture\n\
       --out <dir>              output directory\n\
       --draft <path>           DFlash draft artifact\n\
       --dflash <off|auto|on>   DFlash mode (default: off)\n\
       --kv-mode <mode>         KV mode metadata to record\n\
       --max-tokens <N>         short decode cap for execution batteries (default: 64)\n\
       --profile <off|passive>  profiling mode (default: off)\n\
       --quality-max-chunks <N> quality canary chunk cap\n\
       --kldref <path>          HFQM .kldref.hfq override for quality battery\n\
       --quality-json <path>    ingest kld_reduce.py result-data.json for quality battery\n\
       --performance-json <path> ingest benchmark/perf JSON for speed battery\n\
       --evidence-json <path>   ingest profiler/runtime evidence JSON; repeatable\n\
       --evidence-dir <dir>     ingest standard runtime evidence JSON files from a directory; repeatable\n\
       --candidate-variant <v>  quality-json variant for --model (default: model stem)\n\
       --baseline-variant <v>   quality-json variant for --baseline (default: baseline stem)\n\
       --reference-variant <v>  quality-json variant for --reference (default: reference stem)\n\
       --performance-candidate-variant <v> performance-json variant for --model\n\
       --performance-baseline-variant <v>  performance-json variant for --baseline\n\
       --performance-reference-variant <v> performance-json variant for --reference\n\
       --executor <auto|none|examples|daemon|direct|mock> execution backend (default: auto; daemon uses the JSONL adapter; examples/direct run Hipfire example binaries; mock is no-GPU test-only)\n\
       --fetch-datasets         opt in to Hugging Face dataset fetches\n\
       --offline                forbid network fetches\n\
       --dataset-cache <dir>    dataset cache root (default: ~/.hipfire/datasets)\n\
       --result-cache <dir>     result cache root (default: ~/.hipfire/eval-results/cache)\n\
       --force                  ignore cache hits for this run, but write new cache entries\n\
       --regenerate             delete and replace matching cache entries before running\n\
       --no-cache               disable result cache reads and writes\n\
       --runs <N>               repeat each scored battery N times (default: 1)\n\
       --warmup-runs <N>        run and discard N warmup battery passes before scored repeats\n\
       --benchmark              shorthand for --runs 5 unless --runs is provided; emits aggregate rows\n\
       --host-memory-class <s>  override uncertain host memory class (e.g. gddr6, lpddr5x)\n\
       --host-memory-width-bits <N> override uncertain memory bus width/channel width\n\
       --host-memory-bandwidth-gbps <N> override computed peak memory bandwidth\n\
       --fail-on-admission      exit non-zero after writing artifacts unless admission verdict is promote\n"
        .to_string()
}

pub fn version_report() -> String {
    let context = EvalContext::new();
    let mut lines = vec![
        format!("hipfire-eval {}", env!("CARGO_PKG_VERSION")),
        format!("hipfire_version {}", env!("CARGO_PKG_VERSION")),
        format!(
            "git_commit {}",
            context.commit_sha.unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_branch {}",
            context.git_branch.unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_describe {}",
            context
                .git_describe
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_dirty {}",
            context
                .git_dirty
                .map(|dirty| dirty.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "binary_hash {}",
            context.binary_hash.unwrap_or_else(|| "unknown".to_string())
        ),
    ];
    if let Some(arch) = context.arch {
        lines.push(format!("arch {arch}"));
    }
    if let Some(rocm) = context.rocm {
        lines.push(format!("rocm {rocm}"));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn take_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1)
        .filter(|v| !v.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_csv<T>(raw: &str, parse: fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    raw.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse(s.trim()))
        .collect()
}

fn parse_usize(raw: &str, flag: &str) -> Result<usize, String> {
    raw.parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn parse_u32(raw: &str, flag: &str) -> Result<u32, String> {
    raw.parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn parse_f64(raw: &str, flag: &str) -> Result<f64, String> {
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("{flag} must be a positive number"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{flag} must be a positive finite number"))
    }
}

pub fn default_batteries(tier: EvalTier) -> Vec<BatteryId> {
    let mut out = vec![
        BatteryId::Smoke,
        BatteryId::Coherence,
        BatteryId::Quality,
        BatteryId::Retrieval,
        BatteryId::Speed,
        BatteryId::Dflash,
        BatteryId::Agentic,
        BatteryId::PromptShape,
        BatteryId::Structured,
    ];
    if matches!(
        tier,
        EvalTier::Medium | EvalTier::Long | EvalTier::Extensive
    ) {
        // Perplexity (PPL, + KLD when a reference resolves) joins medium: the
        // run itself is fast when the corpus/kldref are present, and it skips
        // with an actionable reason when they are not.
        out.extend([
            BatteryId::Barrage,
            BatteryId::Longctx,
            BatteryId::Perplexity,
        ]);
    }
    if matches!(tier, EvalTier::Long | EvalTier::Extensive) {
        out.push(BatteryId::Profile);
    }
    if matches!(tier, EvalTier::Extensive) {
        out.extend([BatteryId::Vision, BatteryId::Cask]);
    }
    out
}

pub fn default_suites(tier: EvalTier) -> Vec<SuiteId> {
    match tier {
        EvalTier::Fast => vec![SuiteId::Gpqa],
        EvalTier::Medium => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
        ],
        EvalTier::Long => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
            SuiteId::Ruler,
            SuiteId::NoLiMa,
            SuiteId::Niah,
        ],
        EvalTier::Extensive => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::Ruler,
            SuiteId::NoLiMa,
            SuiteId::Niah,
            SuiteId::NeedleChain,
            SuiteId::SequentialNiah,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
        ],
    }
}

pub fn default_output_dir(model: &str, tier: EvalTier) -> PathBuf {
    let stem = model_artifact_stem(model);
    let leaf = format!("{}-{}-{}", utc_stamp_compact(), stem, tier.as_str());
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("runs")
        .join(leaf)
}

fn default_dataset_cache() -> PathBuf {
    // Reusable top-level location (shared across eval runs / tools), overridable
    // with --dataset-cache. (Was ~/.hipfire/eval/datasets.)
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("datasets")
}

pub(crate) fn default_result_cache() -> PathBuf {
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("cache")
}
