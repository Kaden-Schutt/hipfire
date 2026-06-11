use std::path::{Path, PathBuf};

use crate::config::models_dir;

/// Extension preferred order for fuzzy scan (most-to-least preferred quant).
static QUANT_PREFERENCE: &[&str] = &["-mq4", "-hf4", "-mq3", "-mq2lloyd", "-mq6", "-hf6", "-q8"];

/// Resolve a model identifier to an absolute file path.
///
/// Resolution order (mirrors Bun CLI findModel):
/// 1. Direct file path — if the string exists on disk, use it as-is.
/// 2. `~/.hipfire/models/<arg>` — if that exists.
/// 3. `~/.hipfire/models/<arg>.hfq` — bare name + extension.
/// 4. User aliases from `~/.hipfire/models.json`.
/// 5. Fuzzy scan of `~/.hipfire/models/` — walks one level, ranks by quant preference.
pub fn find_model(arg: &str) -> Option<PathBuf> {
    // 1. Direct path
    let direct = PathBuf::from(arg);
    if direct.exists() {
        return Some(direct);
    }

    let mdir = models_dir();

    // 2. Direct in models dir
    let in_models = mdir.join(arg);
    if in_models.exists() {
        return Some(in_models);
    }

    // 3. With .hfq extension
    let with_ext = mdir.join(format!("{arg}.hfq"));
    if with_ext.exists() {
        return Some(with_ext);
    }

    // 4. User aliases
    let aliases_path = dirs::home_dir()?.join(".hipfire").join("models.json");
    if let Ok(s) = std::fs::read_to_string(&aliases_path) {
        if let Ok(map) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(path_str) = map.get(arg).and_then(|v| v.as_str()) {
                let p = PathBuf::from(path_str);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    // 5. Fuzzy scan — find all .hfq files whose name contains the tag stem
    let tag_stem = normalize_tag_stem(arg);
    let mut candidates = scan_models_dir(&mdir, &tag_stem);
    candidates.sort_by_key(|p| quant_rank(p));
    candidates.into_iter().next()
}

/// Strip known version/size suffixes to get a fuzzy match stem.
fn normalize_tag_stem(tag: &str) -> String {
    tag.replace(':', "-").to_lowercase()
}

fn quant_rank(path: &Path) -> usize {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    QUANT_PREFERENCE
        .iter()
        .position(|q| name.contains(q))
        .unwrap_or(QUANT_PREFERENCE.len())
}

fn scan_models_dir(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        if name.ends_with(".hfq") && !is_role_sidecar(&name) && name.contains(stem) {
            out.push(path.clone());
        }
        // One level deep
        if path.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&path) {
                for se in sub.flatten() {
                    let sp = se.path();
                    let sn = sp
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase();
                    if sn.ends_with(".hfq") && !is_role_sidecar(&sn) && sn.contains(stem) {
                        out.push(sp);
                    }
                }
            }
        }
    }
    out
}

fn is_role_sidecar(name: &str) -> bool {
    name.ends_with(".triattn.hfq")
        || name.ends_with(".dflash.hfq")
        || name.ends_with(".mtp.hfq")
        || name.ends_with(".hfqm2.hfq")
        || name.ends_with(".hfqm2.hfq.tmp")
}

/// List all non-sidecar .hfq files in the models directory.
pub fn list_local_models() -> Vec<PathBuf> {
    let mdir = models_dir();
    let Ok(entries) = std::fs::read_dir(&mdir) else {
        return vec![];
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            n.ends_with(".hfq") && !is_role_sidecar(&n)
        })
        .collect();
    out.sort();
    out
}

/// Derive a display name (tag) from a model file path.
pub fn model_display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .trim_end_matches(".hfq")
        .trim_end_matches(".tmp")
        .to_string()
}
