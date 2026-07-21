// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Session UI state persisted across runs — the active tab and which Models
//! groups were expanded — so the TUI reopens where the user left off.
//!
//! Non-critical: a missing or corrupt file degrades to defaults, never an
//! error (UI state is a convenience, not load-bearing). Writes are atomic
//! (temp file in the same dir + rename), mirroring the config writer.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UiState {
    /// Tab title (e.g. "Models"); unknown/empty falls back to Home on load.
    pub active_tab: String,
    /// Model-family group headers the user had expanded in the Models tab.
    pub expanded_groups: Vec<String>,
}

impl UiState {
    /// Load persisted UI state. Missing or unparsable file -> defaults (never an
    /// error — see module docs).
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist atomically: write a sibling temp file, then rename over the
    /// target (rename is atomic on the same filesystem). Best-effort — callers
    /// treat a failed save as non-fatal.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("ui_state.json");
        let tmp = path.with_file_name(format!("{name}.tmp"));
        fs::write(&tmp, body.as_bytes())?;
        fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_default() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-uistate-missing-{}", std::process::id()));
        let p = dir.join("nope.json");
        let s = UiState::load(&p);
        assert_eq!(s.active_tab, "");
        assert!(s.expanded_groups.is_empty());
    }

    #[test]
    fn round_trip_save_load() {
        let dir = std::env::temp_dir().join(format!("hipfire-uistate-rt-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("ui_state.json");
        let s = UiState {
            active_tab: "Models".into(),
            expanded_groups: vec!["qwen".into(), "lfm".into()],
        };
        s.save(&p).expect("save");
        let back = UiState::load(&p);
        assert_eq!(back.active_tab, "Models");
        assert_eq!(
            back.expanded_groups,
            vec!["qwen".to_string(), "lfm".to_string()]
        );
        // No temp file left behind.
        assert!(!p.with_file_name("ui_state.json.tmp").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_file_is_default_not_error() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-uistate-corrupt-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("ui_state.json");
        fs::write(&p, b"{not valid json").unwrap();
        let s = UiState::load(&p);
        assert_eq!(s.active_tab, "");
        let _ = fs::remove_dir_all(&dir);
    }
}
