// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Eval dataset resolution, fetching, and prompt-artifact materialization.
//!
//! Resolves the dataset manifest for the selected suites (GPQA / HumanEval /
//! lm-eval-micro / builtin barrage), fetches/caches them, parses their items,
//! and writes per-item prompt artifacts with provenance. Extracted verbatim
//! from the former `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::*;

pub(crate) fn resolve_datasets(config: &EvalConfig) -> Result<Vec<DatasetManifestEntry>, String> {
    let mut entries = Vec::new();
    for suite in &config.suites {
        if matches!(
            *suite,
            SuiteId::LmEvalMicro | SuiteId::DeepSwe | SuiteId::SweBench
        ) {
            entries.push(builtin_dataset_entry(*suite));
            continue;
        }
        let cache_path = config.dataset_cache.join(suite.as_str());
        if let Some(reason) = dataset_unavailable_reason(*suite, &cache_path) {
            if config.fetch_datasets {
                match fetch_dataset(*suite, &cache_path) {
                    Ok(fetched) => entries.push(DatasetManifestEntry {
                        suite: *suite,
                        source: fetched.source,
                        repo_id: suite.hf_repo_id().map(str::to_string),
                        revision: fetched.revision,
                        files: fetched.files,
                        digest: directory_hash(&cache_path),
                        license: suite.license().map(str::to_string),
                        cache_path: cache_path.display().to_string(),
                        selected_item_ids: selected_item_ids(*suite),
                        status: EvalStatus::Pass,
                        reason: None,
                    }),
                    Err(reason) => entries.push(dataset_skip(*suite, &cache_path, reason)),
                }
                continue;
            }

            let reason = if config.offline && !cache_path.exists() {
                "dataset not cached and --offline forbids fetch".to_string()
            } else if config.offline {
                format!("{reason}; --offline forbids fetch")
            } else {
                format!("{reason}; rerun with --fetch-datasets to opt in")
            };
            entries.push(dataset_skip(*suite, &cache_path, reason));
            continue;
        }

        if cache_path.exists() {
            entries.push(DatasetManifestEntry {
                suite: *suite,
                source: "local_cache".to_string(),
                repo_id: suite.hf_repo_id().map(str::to_string),
                revision: suite.hf_revision().map(str::to_string),
                files: list_files(&cache_path),
                digest: directory_hash(&cache_path),
                license: suite.license().map(str::to_string),
                cache_path: cache_path.display().to_string(),
                selected_item_ids: selected_item_ids(*suite),
                status: EvalStatus::Pass,
                reason: None,
            });
            continue;
        }
    }
    Ok(entries)
}

pub(crate) fn builtin_dataset_entry(suite: SuiteId) -> DatasetManifestEntry {
    let selected_item_ids = selected_item_ids(suite);
    let files = match suite {
        SuiteId::LmEvalMicro => vec!["builtin:lm_eval_micro:v1".to_string()],
        SuiteId::DeepSwe => vec!["builtin:deep_swe_micro:v1".to_string()],
        SuiteId::SweBench => vec!["builtin:swe_bench_micro:v1".to_string()],
        _ => Vec::new(),
    };
    let digest = match suite {
        SuiteId::LmEvalMicro => Some(stable_hash_bytes(
            lm_eval_micro_items()
                .iter()
                .flat_map(|item| {
                    item.item_id
                        .as_bytes()
                        .iter()
                        .copied()
                        .chain([0])
                        .chain(item.prompt.as_bytes().iter().copied())
                        .chain([0xff])
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        SuiteId::DeepSwe | SuiteId::SweBench => Some(stable_hash_bytes(
            builtin_barrage_items(suite)
                .iter()
                .flat_map(|item| {
                    item.item_id
                        .as_bytes()
                        .iter()
                        .copied()
                        .chain([0])
                        .chain(item.prompt.as_bytes().iter().copied())
                        .chain([0xff])
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )),
        _ => None,
    };
    DatasetManifestEntry {
        suite,
        source: "builtin".to_string(),
        repo_id: None,
        revision: Some("hipfire-native-v1".to_string()),
        files,
        digest,
        license: Some("hipfire-native".to_string()),
        cache_path: format!("builtin:{}", suite.as_str()),
        selected_item_ids,
        status: EvalStatus::Pass,
        reason: None,
    }
}

pub(crate) fn dataset_unavailable_reason(suite: SuiteId, cache_path: &Path) -> Option<String> {
    match suite {
        SuiteId::Gpqa => {
            if !cache_path.exists() {
                return Some("dataset not cached".to_string());
            }
            if gpqa_csv_paths(cache_path).is_empty() {
                if cache_path.join("dataset.zip").exists() {
                    Some(
                        "GPQA cache contains encrypted dataset.zip but no extracted gpqa_*.csv files"
                            .to_string(),
                    )
                } else {
                    Some("GPQA cache has no gpqa_*.csv files".to_string())
                }
            } else {
                None
            }
        }
        SuiteId::HumanEval => {
            if !cache_path.exists() {
                return Some("dataset not cached".to_string());
            }
            if humaneval_jsonl_paths(cache_path).is_empty() {
                Some("HumanEval cache has no HumanEval*.jsonl files".to_string())
            } else {
                None
            }
        }
        _ => {
            if cache_path.exists() {
                None
            } else {
                Some("dataset not cached".to_string())
            }
        }
    }
}

pub(crate) fn dataset_skip(
    suite: SuiteId,
    cache_path: &Path,
    reason: String,
) -> DatasetManifestEntry {
    DatasetManifestEntry {
        suite,
        source: "unavailable".to_string(),
        repo_id: suite.hf_repo_id().map(str::to_string),
        revision: suite.hf_revision().map(str::to_string),
        files: Vec::new(),
        digest: None,
        license: suite.license().map(str::to_string),
        cache_path: cache_path.display().to_string(),
        selected_item_ids: selected_item_ids(suite),
        status: EvalStatus::Skip,
        reason: Some(reason),
    }
}

pub(crate) struct FetchedDataset {
    pub(crate) source: String,
    pub(crate) revision: Option<String>,
    pub(crate) files: Vec<String>,
}

pub(crate) fn fetch_dataset(suite: SuiteId, cache_path: &Path) -> Result<FetchedDataset, String> {
    if let Ok(root) = std::env::var("HIPFIRE_EVAL_DATASET_MIRROR") {
        let mirror_path = Path::new(&root).join(suite.as_str());
        if mirror_path.exists() {
            copy_dir_recursive(&mirror_path, cache_path).map_err(|e| {
                format!(
                    "copy dataset mirror {} to {}: {e}",
                    mirror_path.display(),
                    cache_path.display()
                )
            })?;
            return Ok(FetchedDataset {
                source: "local_mirror".to_string(),
                revision: suite.hf_revision().map(str::to_string),
                files: list_files(cache_path),
            });
        }
    }

    let repo_id = suite
        .hf_repo_id()
        .ok_or_else(|| format!("suite {} has no native HF fetch recipe yet", suite.as_str()))?;
    fs::create_dir_all(cache_path).map_err(|e| format!("create dataset cache: {e}"))?;
    let revision = suite.hf_revision();
    let script = format!(
        "from huggingface_hub import snapshot_download\nsnapshot_download(repo_id={repo_id:?}, repo_type='dataset', revision={revision:?}, local_dir={cache:?}, local_dir_use_symlinks=False)",
        repo_id = repo_id,
        revision = revision,
        cache = cache_path.display().to_string(),
    );
    let out = Command::new("python3")
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("python3/huggingface_hub unavailable: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(FetchedDataset {
        source: "huggingface".to_string(),
        revision: revision.map(str::to_string),
        files: list_files(cache_path),
    })
}

pub(crate) fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

pub(crate) fn selected_item_ids(suite: SuiteId) -> Vec<String> {
    match suite {
        SuiteId::Gpqa => vec!["gpqa_diamond:0".to_string(), "gpqa_main:0".to_string()],
        SuiteId::LmEvalMicro => vec![
            "arc_easy:0".to_string(),
            "hellaswag:0".to_string(),
            "mmlu_stem:0".to_string(),
        ],
        SuiteId::HumanEval => vec!["HumanEval/0".to_string(), "HumanEval/53".to_string()],
        SuiteId::DeepSwe => vec!["deep_swe_verified:0".to_string()],
        SuiteId::SweBench => vec!["swe_bench_lite:0".to_string()],
        SuiteId::Ruler => vec!["ruler_niah_4k:0".to_string()],
        SuiteId::NoLiMa => vec!["nolima_4k:0".to_string()],
        SuiteId::NeedleChain => vec!["needle_chain_4k:0".to_string()],
        SuiteId::Niah => vec!["niah_4k:0".to_string()],
        SuiteId::SequentialNiah => vec!["sequential_niah_4k:0".to_string()],
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GpqaItem {
    pub(crate) item_id: String,
    pub(crate) dataset_file: String,
    pub(crate) prompt: String,
    pub(crate) correct_answer: String,
    pub(crate) answer_label: String,
    pub(crate) choices: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub(crate) struct HumanEvalItem {
    pub(crate) item_id: String,
    pub(crate) task_id: String,
    pub(crate) dataset_file: String,
    pub(crate) prompt: String,
    pub(crate) canonical_solution_hash: Option<String>,
    pub(crate) test_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct LmEvalMicroItem {
    pub(crate) item_id: String,
    pub(crate) task: String,
    pub(crate) prompt: String,
    pub(crate) answer_label: String,
    pub(crate) answer_hash: String,
    pub(crate) choices_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct BuiltinBarrageItem {
    pub(crate) item_id: String,
    pub(crate) suite: SuiteId,
    pub(crate) task: String,
    pub(crate) prompt: String,
    pub(crate) answer_label: String,
    pub(crate) answer_hash: String,
    pub(crate) choices_count: usize,
    pub(crate) dataset_file: String,
    pub(crate) prompt_format: String,
    pub(crate) scoring_mode: String,
}

pub(crate) fn lm_eval_micro_items() -> Vec<LmEvalMicroItem> {
    [
        (
            "arc_easy:0",
            "arc_easy",
            "Question: Which object is designed to measure temperature?\n\nA. Barometer\nB. Thermometer\nC. Compass\nD. Stopwatch\n\nAnswer with only the letter A, B, C, or D.\n",
            "B",
            "Thermometer",
        ),
        (
            "hellaswag:0",
            "hellaswag",
            "Choose the most plausible continuation.\n\nA person opens an umbrella while walking outside because\n\nA. it has started raining.\nB. the oven is preheating.\nC. the book needs a bookmark.\nD. the train is underwater.\n\nAnswer with only the letter A, B, C, or D.\n",
            "A",
            "it has started raining.",
        ),
        (
            "mmlu_stem:0",
            "mmlu_stem",
            "Question: A triangle has angles 30 degrees and 60 degrees. What is the third angle?\n\nA. 30 degrees\nB. 60 degrees\nC. 90 degrees\nD. 120 degrees\n\nAnswer with only the letter A, B, C, or D.\n",
            "C",
            "90 degrees",
        ),
    ]
    .into_iter()
    .map(|(item_id, task, prompt, answer_label, answer)| LmEvalMicroItem {
        item_id: item_id.to_string(),
        task: task.to_string(),
        prompt: prompt.to_string(),
        answer_label: answer_label.to_string(),
        answer_hash: stable_hash_bytes(answer.as_bytes()),
        choices_count: 4,
    })
    .collect()
}

pub(crate) fn lm_eval_micro_materialized_items(
    item_ids: &[String],
) -> Result<Vec<LmEvalMicroItem>, String> {
    let items = lm_eval_micro_items();
    let mut out = Vec::new();
    for id in item_ids {
        let item = items
            .iter()
            .find(|item| &item.item_id == id)
            .cloned()
            .ok_or_else(|| format!("lm_eval_micro item {id} not found"))?;
        out.push(item);
    }
    Ok(out)
}

pub(crate) fn builtin_barrage_items(suite: SuiteId) -> Vec<BuiltinBarrageItem> {
    let rows = match suite {
        SuiteId::DeepSwe => vec![(
            "deep_swe_verified:0",
            "deep_swe_patch_reasoning",
            "A regression report says that `hipfire-eval --suite gpqa --offline` should never try to fetch Hugging Face data. The current parser accepts both `--fetch-datasets` and `--offline`, then later attempts a dataset download.\n\nWhich minimal patch best preserves the intended contract?\n\nA. Ignore `--offline` whenever `--fetch-datasets` is also present.\nB. Reject `--fetch-datasets` and `--offline` together during CLI parsing before any dataset resolution.\nC. Fetch the dataset first, then mark the row skipped if network fails.\nD. Remove the GPQA suite from all tiers.\n\nAnswer with only the letter A, B, C, or D.\n",
            "B",
            "Reject mutually exclusive fetch/offline flags during CLI parsing.",
            "deep_swe_micro_zero_shot_v1",
        )],
        SuiteId::SweBench => vec![(
            "swe_bench_lite:0",
            "swe_bench_bug_localization",
            "A failing test reports: `summary.md does not mention admission verdict reject after --fail-on-admission writes artifacts`. The code already builds `admission.json` correctly, but the Markdown summary only prints pass/fail/skip counts.\n\nWhich change most directly fixes the user-visible bug?\n\nA. Delete `admission.json` so the summary cannot disagree with it.\nB. Change the pass/fail/skip counters to include skipped rows twice.\nC. Add the admission verdict and findings section to `summary.md` using the same admission artifact built for JSON output.\nD. Make `--fail-on-admission` exit before writing artifacts.\n\nAnswer with only the letter A, B, C, or D.\n",
            "C",
            "Add the admission verdict and findings section to the Markdown summary.",
            "swe_bench_micro_zero_shot_v1",
        )],
        _ => Vec::new(),
    };
    rows.into_iter()
        .map(
            |(item_id, task, prompt, answer_label, answer, prompt_format)| BuiltinBarrageItem {
                item_id: item_id.to_string(),
                suite,
                task: task.to_string(),
                prompt: prompt.to_string(),
                answer_label: answer_label.to_string(),
                answer_hash: stable_hash_bytes(answer.as_bytes()),
                choices_count: 4,
                dataset_file: format!("builtin:{}:v1", suite.as_str()),
                prompt_format: prompt_format.to_string(),
                scoring_mode: "exact_letter".to_string(),
            },
        )
        .collect()
}

pub(crate) fn builtin_barrage_materialized_items(
    suite: SuiteId,
    item_ids: &[String],
) -> Result<Vec<BuiltinBarrageItem>, String> {
    let items = builtin_barrage_items(suite);
    let mut out = Vec::new();
    for id in item_ids {
        let item = items
            .iter()
            .find(|item| &item.item_id == id)
            .cloned()
            .ok_or_else(|| format!("{} item {id} not found", suite.as_str()))?;
        out.push(item);
    }
    Ok(out)
}

pub(crate) fn gpqa_csv_paths(cache_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_gpqa_csv_paths(cache_path, 0, &mut out);
    out.sort();
    out
}

pub(crate) fn collect_gpqa_csv_paths(path: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_gpqa_csv_paths(&p, depth + 1, out);
        } else if p
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| matches!(name, "gpqa_diamond.csv" | "gpqa_main.csv"))
        {
            out.push(p);
        }
    }
}

pub(crate) fn gpqa_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<GpqaItem>, String> {
    let mut out = Vec::new();
    for id in item_ids {
        let Some((subset, row_idx)) = id.split_once(':') else {
            continue;
        };
        let row_idx: usize = row_idx
            .parse()
            .map_err(|_| format!("invalid GPQA item id row index: {id}"))?;
        let csv_path = gpqa_csv_paths(cache_path)
            .into_iter()
            .find(|p| p.file_stem().and_then(OsStr::to_str) == Some(subset))
            .ok_or_else(|| format!("GPQA subset CSV not found for {subset}"))?;
        out.push(read_gpqa_item(&csv_path, subset, row_idx)?);
    }
    Ok(out)
}

pub(crate) fn read_gpqa_item(
    path: &Path,
    subset: &str,
    row_idx: usize,
) -> Result<GpqaItem, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("open GPQA CSV {}: {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("read GPQA CSV headers: {e}"))?
        .clone();
    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("GPQA CSV missing header {name:?}"))
    };
    let q_col = find("Question")?;
    let correct_col = find("Correct Answer")?;
    let i1_col = find("Incorrect Answer 1")?;
    let i2_col = find("Incorrect Answer 2")?;
    let i3_col = find("Incorrect Answer 3")?;
    let rec_col = headers.iter().position(|h| h == "Record ID");

    for (idx, row) in reader.records().enumerate() {
        let row = row.map_err(|e| format!("read GPQA CSV row: {e}"))?;
        if idx != row_idx {
            continue;
        }
        let question = row.get(q_col).unwrap_or("").trim().to_string();
        let correct_answer = row.get(correct_col).unwrap_or("").trim().to_string();
        let incorrect = [
            row.get(i1_col).unwrap_or("").trim().to_string(),
            row.get(i2_col).unwrap_or("").trim().to_string(),
            row.get(i3_col).unwrap_or("").trim().to_string(),
        ];
        if question.is_empty()
            || correct_answer.is_empty()
            || incorrect.iter().any(String::is_empty)
        {
            return Err(format!(
                "GPQA row {subset}:{row_idx} has empty question/choice"
            ));
        }
        let record_suffix = rec_col
            .and_then(|c| row.get(c))
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!(":{s}"))
            .unwrap_or_default();
        let item_id = format!("{subset}:{row_idx}{record_suffix}");
        return Ok(build_gpqa_item(
            item_id,
            path.file_name()
                .and_then(OsStr::to_str)
                .unwrap_or(subset)
                .to_string(),
            question,
            correct_answer,
            incorrect,
        ));
    }
    Err(format!("GPQA row {subset}:{row_idx} not found"))
}

pub(crate) fn build_gpqa_item(
    item_id: String,
    dataset_file: String,
    question: String,
    correct_answer: String,
    incorrect: [String; 3],
) -> GpqaItem {
    let mut raw_choices = vec![
        (true, correct_answer.clone()),
        (false, incorrect[0].clone()),
        (false, incorrect[1].clone()),
        (false, incorrect[2].clone()),
    ];
    let rotate = (stable_hash_bytes(item_id.as_bytes())
        .bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize)))
        % raw_choices.len();
    raw_choices.rotate_left(rotate);

    let labels = ["A", "B", "C", "D"];
    let mut choices = Vec::new();
    let mut answer_label = "A".to_string();
    for (idx, (is_correct, answer)) in raw_choices.into_iter().enumerate() {
        let label = labels[idx].to_string();
        if is_correct {
            answer_label = label.clone();
        }
        choices.push((label, answer));
    }

    let mut prompt = String::new();
    prompt.push_str("Answer the following graduate-level science multiple-choice question.\n");
    prompt.push_str("Return only the letter of the correct answer.\n\n");
    prompt.push_str("Question:\n");
    prompt.push_str(question.trim());
    prompt.push_str("\n\nChoices:\n");
    for (label, answer) in &choices {
        prompt.push_str(label);
        prompt.push_str(". ");
        prompt.push_str(answer.trim());
        prompt.push('\n');
    }
    prompt.push_str("\nAnswer:");

    GpqaItem {
        item_id,
        dataset_file,
        prompt,
        correct_answer,
        answer_label,
        choices,
    }
}

pub(crate) fn write_gpqa_prompt_artifact(
    dir: &Path,
    _config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
) -> Result<Option<(String, usize)>, String> {
    let mut rows = Vec::new();
    for d in datasets {
        if d.suite != SuiteId::Gpqa || d.status != EvalStatus::Pass {
            continue;
        }
        match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
            Ok(items) => {
                for item in items {
                    rows.push(with_dataset_provenance(
                        json!({
                            "schema": 1,
                            "suite": "gpqa",
                            "item_id": item.item_id,
                            "status": "pass",
                            "dataset_file": item.dataset_file,
                            "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                            "prompt_format": "gpqa_zero_shot_v1",
                            "answer_label": item.answer_label,
                            "answer_hash": stable_hash_bytes(item.correct_answer.as_bytes()),
                            "choices_count": item.choices.len(),
                        }),
                        d,
                    ));
                }
            }
            Err(reason) => {
                for id in &d.selected_item_ids {
                    rows.push(with_dataset_provenance(
                        json!({
                            "schema": 1,
                            "suite": "gpqa",
                            "item_id": id,
                            "status": "skip",
                            "reason": reason.clone(),
                        }),
                        d,
                    ));
                }
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let rel = "artifacts/gpqa_prompts.jsonl";
    let path = dir.join("gpqa_prompts.jsonl");
    let mut f = File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    for row in &rows {
        serde_json::to_writer(&mut f, row)
            .map_err(|e| format!("serialize GPQA prompt row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(Some((rel.to_string(), rows.len())))
}

pub(crate) fn write_barrage_prompt_artifact(
    dir: &Path,
    datasets: &[DatasetManifestEntry],
) -> Result<Option<(String, usize)>, String> {
    let rows = barrage_prompt_artifact_rows(datasets);
    if rows.is_empty() {
        return Ok(None);
    }
    let rel = "artifacts/barrage_prompts.jsonl";
    let path = dir.join("barrage_prompts.jsonl");
    let mut f = File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    for row in &rows {
        serde_json::to_writer(&mut f, row)
            .map_err(|e| format!("serialize barrage prompt row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(Some((rel.to_string(), rows.len())))
}

pub(crate) fn with_dataset_provenance(mut row: Value, dataset: &DatasetManifestEntry) -> Value {
    if let Value::Object(ref mut object) = row {
        object.insert("dataset_source".to_string(), json!(dataset.source));
        object.insert("dataset_repo_id".to_string(), json!(dataset.repo_id));
        object.insert("dataset_revision".to_string(), json!(dataset.revision));
        object.insert("dataset_digest".to_string(), json!(dataset.digest));
        object.insert("dataset_license".to_string(), json!(dataset.license));
        object.insert("dataset_cache_path".to_string(), json!(dataset.cache_path));
    }
    row
}

pub(crate) fn barrage_prompt_artifact_rows(datasets: &[DatasetManifestEntry]) -> Vec<Value> {
    let mut rows = Vec::new();
    for d in datasets {
        match d.suite {
            SuiteId::Gpqa if d.status == EvalStatus::Pass => {
                match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(json!({
                                "schema": 1,
                                "suite": "gpqa",
                                "item_id": item.item_id,
                                "status": "pass",
                                "dataset_file": item.dataset_file,
                                "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                "prompt_format": "gpqa_zero_shot_v1",
                                "answer_label": item.answer_label,
                                "answer_hash": stable_hash_bytes(item.correct_answer.as_bytes()),
                                "choices_count": item.choices.len(),
                            }), d)
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "gpqa",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::LmEvalMicro if d.status == EvalStatus::Pass => {
                match lm_eval_micro_materialized_items(&d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "lm_eval_micro",
                                    "item_id": item.item_id,
                                    "task": item.task,
                                    "status": "pass",
                                    "dataset_file": "builtin:lm_eval_micro:v1",
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": "lm_eval_micro_zero_shot_v1",
                                    "answer_label": item.answer_label,
                                    "answer_hash": item.answer_hash,
                                    "choices_count": item.choices_count,
                                }),
                                d,
                            )
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "lm_eval_micro",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::HumanEval if d.status == EvalStatus::Pass => {
                match humaneval_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            let mut row = with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "humaneval",
                                    "item_id": item.item_id,
                                    "task_id": item.task_id,
                                    "status": "pass",
                                    "dataset_file": item.dataset_file,
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": "humaneval_completion_v1",
                                    "scoring_mode": "execution_only",
                                }),
                                d,
                            );
                            if let Value::Object(ref mut object) = row {
                                if let Some(hash) = item.canonical_solution_hash {
                                    object
                                        .insert("canonical_solution_hash".to_string(), json!(hash));
                                }
                                if let Some(hash) = item.test_hash {
                                    object.insert("test_hash".to_string(), json!(hash));
                                }
                            }
                            row
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": "humaneval",
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            SuiteId::DeepSwe | SuiteId::SweBench if d.status == EvalStatus::Pass => {
                match builtin_barrage_materialized_items(d.suite, &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().map(|item| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": item.suite.as_str(),
                                    "item_id": item.item_id,
                                    "task": item.task,
                                    "status": "pass",
                                    "dataset_file": item.dataset_file,
                                    "prompt_hash": stable_hash_bytes(item.prompt.as_bytes()),
                                    "prompt_format": item.prompt_format,
                                    "answer_label": item.answer_label,
                                    "answer_hash": item.answer_hash,
                                    "choices_count": item.choices_count,
                                    "scoring_mode": item.scoring_mode,
                                }),
                                d,
                            )
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().map(|id| {
                            with_dataset_provenance(
                                json!({
                                    "schema": 1,
                                    "suite": d.suite.as_str(),
                                    "item_id": id,
                                    "status": "skip",
                                    "reason": reason,
                                }),
                                d,
                            )
                        }));
                    }
                }
            }
            _ => {}
        }
    }
    rows
}

pub(crate) fn humaneval_jsonl_paths(cache_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_humaneval_jsonl_paths(cache_path, 0, &mut out);
    out.sort();
    out
}

pub(crate) fn collect_humaneval_jsonl_paths(path: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_humaneval_jsonl_paths(&p, depth + 1, out);
        } else if p.file_name().and_then(OsStr::to_str).is_some_and(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with(".jsonl") && lower.contains("humaneval")
        }) {
            out.push(p);
        }
    }
}

pub(crate) fn humaneval_materialized_items(
    cache_path: &Path,
    item_ids: &[String],
) -> Result<Vec<HumanEvalItem>, String> {
    let paths = humaneval_jsonl_paths(cache_path);
    if paths.is_empty() {
        return Err("HumanEval JSONL not found".to_string());
    }
    let mut out = Vec::new();
    for id in item_ids {
        let mut found = None;
        for path in &paths {
            if let Some(item) = read_humaneval_item_by_task_id(path, id)? {
                found = Some(item);
                break;
            }
            let row_idx = humaneval_item_row_index(id)?;
            if let Some(item) = read_humaneval_item_by_row(path, row_idx)? {
                found = Some(item);
                break;
            }
        }
        out.push(found.ok_or_else(|| format!("HumanEval row {id} not found"))?);
    }
    Ok(out)
}

pub(crate) fn humaneval_item_row_index(id: &str) -> Result<usize, String> {
    id.rsplit_once('/')
        .map(|(_, idx)| idx)
        .unwrap_or(id)
        .parse()
        .map_err(|_| format!("invalid HumanEval item id row index: {id}"))
}

pub(crate) fn read_humaneval_item_by_task_id(
    path: &Path,
    task_id: &str,
) -> Result<Option<HumanEvalItem>, String> {
    let body = fs::read_to_string(path)
        .map_err(|e| format!("read HumanEval JSONL {}: {e}", path.display()))?;
    for (idx, line) in body.lines().enumerate() {
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("parse HumanEval JSONL row {idx}: {e}"))?;
        if value
            .get("task_id")
            .and_then(Value::as_str)
            .is_some_and(|candidate| candidate == task_id)
        {
            return parse_humaneval_item(path, idx, value).map(Some);
        }
    }
    Ok(None)
}

pub(crate) fn read_humaneval_item_by_row(
    path: &Path,
    row_idx: usize,
) -> Result<Option<HumanEvalItem>, String> {
    let body = fs::read_to_string(path)
        .map_err(|e| format!("read HumanEval JSONL {}: {e}", path.display()))?;
    for (idx, line) in body.lines().enumerate() {
        if idx != row_idx {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("parse HumanEval JSONL row {row_idx}: {e}"))?;
        return parse_humaneval_item(path, row_idx, value).map(Some);
    }
    Ok(None)
}

pub(crate) fn parse_humaneval_item(
    path: &Path,
    row_idx: usize,
    value: Value,
) -> Result<HumanEvalItem, String> {
    let task_id = value
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or("HumanEval/unknown")
        .to_string();
    let prompt = value
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("HumanEval row {row_idx} missing prompt"))?
        .to_string();
    if prompt.trim().is_empty() {
        return Err(format!("HumanEval row {row_idx} has empty prompt"));
    }
    let canonical_solution_hash = value
        .get("canonical_solution")
        .and_then(Value::as_str)
        .map(|s| stable_hash_bytes(s.as_bytes()));
    let test_hash = value
        .get("test")
        .and_then(Value::as_str)
        .map(|s| stable_hash_bytes(s.as_bytes()));
    Ok(HumanEvalItem {
        item_id: task_id.clone(),
        task_id,
        dataset_file: path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("HumanEval.jsonl")
            .to_string(),
        prompt,
        canonical_solution_hash,
        test_hash,
    })
}
