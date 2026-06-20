// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    time::Duration,
};

use serde::Deserialize;
use serde_json::Value;

use super::{config::ConfigState, HipfirePaths};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub size_gb: f64,
    #[serde(default)]
    pub min_vram_gb: f64,
    #[serde(default)]
    pub desc: String,
    #[serde(default)]
    pub triattn: Option<SidecarEntry>,
    #[serde(default)]
    pub mtp: Option<SidecarEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SidecarEntry {
    #[serde(default)]
    pub file: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    models: BTreeMap<String, ModelEntry>,
    #[serde(default)]
    aliases: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct RegistryState {
    pub models: Vec<ModelRow>,
    pub aliases: BTreeMap<String, String>,
    pub local_files: Vec<LocalModel>,
    pub selected: usize,
    pub expanded_groups: BTreeSet<String>,
    pub loaded_path: Option<PathBuf>,
    pub warning: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModelRow {
    pub tag: String,
    pub entry: ModelEntry,
    pub downloaded: bool,
    pub has_triattn: bool,
    pub has_mtp: bool,
    pub has_draft: bool,
    pub has_template: bool,
}

#[derive(Clone, Debug)]
pub struct LocalModel {
    pub file: String,
    pub size: String,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub enum ModelListItem {
    Group {
        name: String,
        count: usize,
        downloaded: usize,
        expanded: bool,
    },
    Model {
        model_index: usize,
    },
}

#[derive(Clone, Debug)]
pub enum RegistryAction {
    ToggledGroup { name: String, expanded: bool },
    SelectedModel { tag: String },
}

impl RegistryState {
    pub fn load(paths: &HipfirePaths, config: &ConfigState) -> Self {
        match load_remote_registry(config) {
            Ok(registry) => registry,
            Err(err) => {
                let mut fallback = Self::load_local(paths);
                fallback.warning = Some(err);
                fallback
            }
        }
    }

    fn load_local(paths: &HipfirePaths) -> Self {
        let local_files = list_local_models(paths);
        let local_names = local_files
            .iter()
            .map(|m| m.file.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let loaded_path = None;
        let registry = RegistryFile::default();

        let registry_file_keys = registry
            .models
            .values()
            .map(|entry| normalized_file_key(&entry.file))
            .collect::<BTreeSet<_>>();

        let mut models = registry
            .models
            .into_iter()
            .map(|(tag, entry)| {
                let downloaded = local_names.contains(entry.file.as_str());
                let has_triattn = entry
                    .triattn
                    .as_ref()
                    .map(|s| local_names.contains(s.file.as_str()))
                    .unwrap_or(false);
                let has_mtp = entry
                    .mtp
                    .as_ref()
                    .map(|s| local_names.contains(s.file.as_str()))
                    .unwrap_or(false);
                ModelRow {
                    tag,
                    entry,
                    downloaded,
                    has_triattn,
                    has_mtp,
                    has_draft: false,
                    has_template: false,
                }
            })
            .collect::<Vec<_>>();
        models.extend(
            local_files
                .iter()
                .filter(|local| is_selectable_model_file(&local.file))
                .filter(|local| !registry_file_keys.contains(&normalized_file_key(&local.file)))
                .map(|local| ModelRow {
                    tag: local.file.clone(),
                    entry: ModelEntry {
                        file: local.file.clone(),
                        size_gb: local.bytes as f64 / 1_000_000_000.0,
                        desc: "local model file outside registry".into(),
                        ..ModelEntry::default()
                    },
                    downloaded: true,
                    has_triattn: false,
                    has_mtp: false,
                    has_draft: false,
                    has_template: false,
                }),
        );
        models.sort_by(|a, b| a.tag.cmp(&b.tag));

        Self {
            models,
            aliases: registry.aliases,
            local_files,
            selected: 0,
            expanded_groups: BTreeSet::new(),
            loaded_path,
            warning: None,
        }
    }

    pub fn downloaded_count(&self) -> usize {
        self.models.iter().filter(|m| m.downloaded).count()
    }

    pub fn visible_items(&self) -> Vec<ModelListItem> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, row) in self.models.iter().enumerate() {
            groups.entry(group_key(&row.tag)).or_default().push(idx);
        }

        let mut out = Vec::new();
        for (name, indices) in groups {
            let expanded = self.expanded_groups.contains(&name);
            let downloaded = indices
                .iter()
                .filter(|idx| self.models[**idx].downloaded)
                .count();
            out.push(ModelListItem::Group {
                name: name.clone(),
                count: indices.len(),
                downloaded,
                expanded,
            });
            if expanded {
                out.extend(
                    indices
                        .into_iter()
                        .map(|model_index| ModelListItem::Model { model_index }),
                );
            }
        }
        out
    }

    pub fn visible_len(&self) -> usize {
        self.visible_items().len()
    }

    pub fn clamp_selected(&mut self) {
        let len = self.visible_len().max(1);
        if self.selected >= len {
            self.selected = len - 1;
        }
    }

    pub fn activate_selected(&mut self) -> Option<RegistryAction> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, expanded, .. } => {
                if expanded {
                    self.expanded_groups.remove(&name);
                    self.clamp_selected();
                    Some(RegistryAction::ToggledGroup {
                        name,
                        expanded: false,
                    })
                } else {
                    self.expanded_groups.insert(name.clone());
                    Some(RegistryAction::ToggledGroup {
                        name,
                        expanded: true,
                    })
                }
            }
            ModelListItem::Model { model_index } => {
                self.models
                    .get(model_index)
                    .map(|row| RegistryAction::SelectedModel {
                        tag: row.tag.clone(),
                    })
            }
        }
    }

    pub fn expand_selected_group(&mut self) -> Option<String> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, .. } => {
                self.expanded_groups.insert(name.clone());
                Some(name)
            }
            ModelListItem::Model { .. } => None,
        }
    }

    pub fn collapse_selected_group(&mut self) -> Option<String> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, .. } => {
                self.expanded_groups.remove(&name);
                self.clamp_selected();
                Some(name)
            }
            ModelListItem::Model { model_index } => {
                let name = group_key(&self.models.get(model_index)?.tag);
                self.expanded_groups.remove(&name);
                self.selected = self
                    .visible_items()
                    .iter()
                    .position(
                        |item| matches!(item, ModelListItem::Group { name: n, .. } if n == &name),
                    )
                    .unwrap_or(0);
                Some(name)
            }
        }
    }
}

fn load_remote_registry(config: &ConfigState) -> Result<RegistryState, String> {
    let url = format!(
        "http://{}:{}/operator/models/registry",
        config.probe_host(),
        config.port
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(650))
        .build();
    let body = agent
        .get(&url)
        .call()
        .map_err(|err| format!("registry unavailable: {err}"))?
        .into_string()
        .map_err(|err| format!("registry read error: {err}"))?;
    let payload = serde_json::from_str::<Value>(&body)
        .map_err(|err| format!("registry parse error: {err}"))?;
    let models = payload
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "registry endpoint returned unexpected JSON".to_string())?;
    let mut local_files = Vec::new();
    let mut rows = Vec::new();
    for model in models {
        let id = model
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let file = model
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or(id.as_str())
            .to_string();
        let bytes = model.get("bytes").and_then(Value::as_u64).unwrap_or(0);
        local_files.push(LocalModel {
            file: file.clone(),
            size: format_bytes(bytes),
            bytes,
        });
        let has_triattn = non_empty_array(model.get("triattn"));
        let has_mtp = false;
        let has_draft = non_empty_array(model.get("drafts"));
        let has_template = non_empty_array(model.get("chat_templates"));
        let mut features = Vec::new();
        if has_draft {
            features.push("draft");
        }
        if has_template {
            features.push("template");
        }
        rows.push(ModelRow {
            tag: id,
            entry: ModelEntry {
                file,
                size_gb: bytes as f64 / 1_000_000_000.0,
                desc: if features.is_empty() {
                    "daemon model registry".into()
                } else {
                    format!("daemon model registry ({})", features.join(", "))
                },
                ..ModelEntry::default()
            },
            downloaded: true,
            has_triattn,
            has_mtp,
            has_draft,
            has_template,
        });
    }
    rows.sort_by(|a, b| a.tag.cmp(&b.tag));
    local_files.sort_by(|a, b| a.file.cmp(&b.file));
    Ok(RegistryState {
        models: rows,
        aliases: BTreeMap::new(),
        local_files,
        selected: 0,
        expanded_groups: BTreeSet::new(),
        loaded_path: None,
        warning: None,
    })
}

fn non_empty_array(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
}

fn group_key(tag: &str) -> String {
    if let Some((family, _)) = tag.split_once(':') {
        return family.to_string();
    }
    if let Some((family, _)) = tag.split_once('-') {
        return family.to_string();
    }
    tag.to_string()
}

fn normalized_file_key(file: &str) -> String {
    file.to_string()
}

fn is_selectable_model_file(file: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    if lower.ends_with(".bin")
        || lower.ends_with(".mtp")
        || lower.ends_with("-mtp")
        || lower.contains(".triattn.")
    {
        return false;
    }
    [
        "-hf4.hfq",
        "-hf6.hfq",
        "-mq2.hfq",
        "-mq3.hfq",
        "-mq4.hfq",
        "-mq6.hfq",
        "-q8.hfq",
        "-q8f16.hfq",
        "-hfp4.hfq",
        "-mfp4.hfq",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn list_local_models(paths: &HipfirePaths) -> Vec<LocalModel> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&paths.models) else {
        return out;
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let file = entry.file_name().to_string_lossy().to_string();
        out.push(LocalModel {
            file,
            size: format_bytes(meta.len()),
            bytes: meta.len(),
        });
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GB {
        format!("{:.1} GiB", bytes as f64 / GB)
    } else if bytes as f64 >= MB {
        format!("{:.0} MiB", bytes as f64 / MB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::{group_key, is_selectable_model_file, normalized_file_key};

    #[test]
    fn groups_registry_tags_and_local_files_by_family() {
        assert_eq!(group_key("qwopus:27b-mq6"), "qwopus");
        assert_eq!(group_key("qwopus-27b-mq6.hfq"), "qwopus");
        assert_eq!(group_key("qwen3.5:9b"), "qwen3.5");
        assert_eq!(group_key("deepseek-v4-flash"), "deepseek");
    }

    #[test]
    fn filters_model_files_without_showing_sidecars() {
        assert!(is_selectable_model_file("qwopus-27b-mq6.hfq"));
        assert!(is_selectable_model_file("qwen3.5-9b-hfp4.hfq"));
        assert!(is_selectable_model_file("lfm2.5-350m-q8.hfq"));
        assert!(!is_selectable_model_file("qwen3.5-27b-mq4.triattn.hfq"));
        assert!(!is_selectable_model_file("qwen3.6-27b-mq4.mtp.hfq"));
    }

    #[test]
    fn keeps_canonical_hfq_names_for_duplicate_detection() {
        assert_eq!(
            normalized_file_key("qwen3.5-27b-mq4.dflash.hfq"),
            "qwen3.5-27b-mq4.dflash.hfq"
        );
        assert_eq!(
            normalized_file_key("qwen3.5-9b-hf4.hfq"),
            "qwen3.5-9b-hf4.hfq"
        );
    }
}
