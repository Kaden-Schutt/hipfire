// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

use serde::Deserialize;

use super::HipfirePaths;

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
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut warning = None;
        let local_files = list_local_models(paths);
        let local_names = local_files
            .iter()
            .map(|m| m.file.as_str())
            .collect::<std::collections::BTreeSet<_>>();

        let (loaded_path, registry) = match paths.registry_path() {
            Some(path) => match fs::read_to_string(path) {
                Ok(raw) => match serde_json::from_str::<RegistryFile>(&raw) {
                    Ok(registry) => (Some(path.to_path_buf()), registry),
                    Err(err) => {
                        warning = Some(format!("registry parse error: {err}"));
                        (Some(path.to_path_buf()), RegistryFile::default())
                    }
                },
                Err(err) => {
                    warning = Some(format!("registry read error: {err}"));
                    (Some(path.to_path_buf()), RegistryFile::default())
                }
            },
            None => {
                warning = Some("registry.json not found".into());
                (None, RegistryFile::default())
            }
        };

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
            warning,
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
    file.replace(".q4.hfq", ".hf4")
        .replace(".hfq6.hfq", ".hf6")
        .replace("-hfq4.hfq", ".hf4")
        .replace(".hfq", ".hf4")
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
        ".hf4", ".hf6", ".hfq", ".mq2", ".mq3", ".mq4", ".mq6", ".q8", ".q8f16", ".hfp4", ".mfp4",
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
        assert_eq!(group_key("qwopus-27b.mq6"), "qwopus");
        assert_eq!(group_key("qwen3.5:9b"), "qwen3.5");
        assert_eq!(group_key("deepseek-v4-flash"), "deepseek");
    }

    #[test]
    fn filters_model_files_without_showing_sidecars() {
        assert!(is_selectable_model_file("qwopus-27b.mq6"));
        assert!(is_selectable_model_file("qwen3.5-9b.hfp4"));
        assert!(is_selectable_model_file("lfm2.5-350m.q8"));
        assert!(!is_selectable_model_file("qwen3.5-27b.mq4.triattn.bin"));
        assert!(!is_selectable_model_file("qwen3.6-27b.mq4-mtp"));
    }

    #[test]
    fn normalizes_legacy_hfq_names_for_duplicate_detection() {
        assert_eq!(
            normalized_file_key("qwen35-27b-dflash-mq4.hfq"),
            "qwen35-27b-dflash-mq4.hf4"
        );
        assert_eq!(normalized_file_key("qwen3.5-9b.q4.hfq"), "qwen3.5-9b.hf4");
    }
}
