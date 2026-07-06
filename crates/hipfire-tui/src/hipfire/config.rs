// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

use serde_json::Value;

use super::HipfirePaths;

#[derive(Clone, Debug)]
pub struct ConfigState {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub values: BTreeMap<String, String>,
    /// Keys explicitly present in `~/.hipfire/config.json` — i.e. user overrides,
    /// as opposed to inherited/hardcoded defaults. `values` always carries every
    /// default key merged with the disk overlay, so this set is the only way to
    /// tell an override apart from a default (drives reset gating + the 5c
    /// override marker).
    pub overrides: BTreeSet<String>,
    pub per_model_count: usize,
    pub loaded_from_disk: bool,
    pub warning: Option<String>,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut values = defaults();
        let mut overrides = BTreeSet::new();
        let mut loaded_from_disk = false;
        let mut warning = None;

        match fs::read_to_string(&paths.config) {
            Ok(raw) => match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => {
                    loaded_from_disk = true;
                    for (k, v) in map {
                        overrides.insert(k.clone());
                        values.insert(k, value_to_string(&v));
                    }
                }
                Ok(_) => warning = Some("config.json is not an object; using defaults".into()),
                Err(err) => warning = Some(format!("config parse error: {err}")),
            },
            Err(_) => {}
        }

        let per_model_count = fs::read_to_string(&paths.per_model_config)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().map(|m| m.len()))
            .unwrap_or(0);

        let host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "0.0.0.0".into());
        let port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let default_model = values
            .get("default_model")
            .cloned()
            .unwrap_or_else(|| "qwen3.5:9b".into());

        Self {
            host,
            port,
            default_model,
            values,
            overrides,
            per_model_count,
            loaded_from_disk,
            warning,
        }
    }

    /// True when `key` is an explicit user override in config.json (vs an
    /// inherited/hardcoded default). Drives reset gating ("already default") and
    /// the override marker in the Settings view.
    pub fn is_override(&self, key: &str) -> bool {
        self.overrides.contains(key)
    }

    /// The hardcoded default map, for cross-checking the knob explainer defaults
    /// (5e) against the real defaults so the two can't drift.
    #[cfg(test)]
    pub(crate) fn defaults_for_test() -> BTreeMap<String, String> {
        defaults()
    }

    pub fn probe_host(&self) -> String {
        match self.host.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".into(),
            "::" => "::1".into(),
            other => other.to_string(),
        }
    }

    /// Config keys behind each easy-mode row, in display order. `None` marks a
    /// row that is composite/non-editable inline (e.g. the `host:port` serve
    /// row, or the Model row which is set from the Models tab). The TUI editor
    /// uses this to resolve a selected easy row to a writable config key.
    pub fn easy_keys(&self) -> Vec<Option<&'static str>> {
        vec![
            None,            // Model (set via Models tab)
            Some("max_seq"), // Context
            Some("dflash_mode"),
            Some("prefill_compression"), // Prefill (pflash)
            Some("kv_cache"),
            Some("thinking"),
            Some("thinking_budget"),
            None, // Serve host:port (composite)
        ]
    }

    /// True when prefill compression is requested (auto/always) but no
    /// `prefill_drafter` is set — pflash then silently no-ops. Mirrors the bun
    /// CLI warning (`prefill_compression=… but prefill_drafter is unset`). Surfaced
    /// honestly in the easy row + as a toast when the user enables compression.
    ///
    /// Scope: this reflects the GLOBAL config (`~/.hipfire/config.json`), which is
    /// what the Settings tab edits — like every other row here. A per-model
    /// override (`hipfire config <tag> set prefill_drafter …`, counted in
    /// `per_model_count`) is resolved by the daemon at serve time and is NOT
    /// layered in here; the daemon's own gate is the final word. Empty-string =
    /// disabled matches bun exactly (`"" disables`).
    pub fn pflash_needs_drafter(&self) -> bool {
        let compression = self
            .values
            .get("prefill_compression")
            .map(String::as_str)
            .unwrap_or("off");
        let drafter_empty = self
            .values
            .get("prefill_drafter")
            .map(|s| s.is_empty())
            .unwrap_or(true);
        compression != "off" && drafter_empty
    }

    /// Per-easy-row override status, parallel to [`easy_rows`]/[`easy_keys`].
    /// Unlike `easy_keys` (which is `None` for the non-inline-editable Model /
    /// Serve composite rows), this resolves override status for EVERY row from
    /// its underlying config key(s), so the 5c override marker is consistent with
    /// the advanced view — a user-set `default_model` / `host` / `port` shows as
    /// changed in easy mode too, not just advanced.
    pub fn easy_override_state(&self) -> Vec<bool> {
        vec![
            self.is_override("default_model"),                       // Model
            self.is_override("max_seq"),                             // Context
            self.is_override("dflash_mode"),                         // Spec decode
            self.is_override("prefill_compression"),                 // Prefill
            self.is_override("kv_cache"),                            // KV cache
            self.is_override("thinking"),                            // Thinking
            self.is_override("thinking_budget"),                     // Reasoning budget
            self.is_override("host") || self.is_override("port"),    // Serve
        ]
    }

    /// Help-lookup key for each easy row, parallel to [`easy_rows`]/[`easy_keys`].
    /// Unlike `easy_keys`, this maps the composite Model/Serve rows to a
    /// representative key (default_model / host) so the 5e explainer pane has
    /// something to show for every row.
    pub fn easy_help_keys(&self) -> Vec<&'static str> {
        vec![
            "default_model",       // Model
            "max_seq",             // Context
            "dflash_mode",         // Spec decode
            "prefill_compression", // Prefill
            "kv_cache",            // KV cache
            "thinking",            // Thinking
            "thinking_budget",     // Reasoning budget
            "host",                // Serve
        ]
    }

    pub fn easy_rows(&self) -> Vec<(&'static str, String, &'static str)> {
        vec![
            (
                "Model",
                self.default_model.clone(),
                "Default model pre-warmed by serve and used by chat.",
            ),
            (
                "Context",
                self.values
                    .get("max_seq")
                    .cloned()
                    .unwrap_or_else(|| "32768".into()),
                "KV cache capacity allocated at load.",
            ),
            (
                "Spec decode",
                self.values
                    .get("dflash_mode")
                    .cloned()
                    .unwrap_or_else(|| "off".into()),
                "DFlash mode. Keep off unless intentionally testing drafts.",
            ),
            (
                "Prefill",
                {
                    let c = self
                        .values
                        .get("prefill_compression")
                        .cloned()
                        .unwrap_or_else(|| "off".into());
                    // Honest: compression on without a drafter is a no-op.
                    if self.pflash_needs_drafter() {
                        format!("{c} (needs drafter)")
                    } else {
                        c
                    }
                },
                "Prefill KV compression (pflash). Needs a prefill_drafter (.hfq) to engage.",
            ),
            (
                "KV cache",
                self.values
                    .get("kv_cache")
                    .cloned()
                    .unwrap_or_else(|| "auto".into()),
                "Precision/memory tradeoff for attention cache.",
            ),
            (
                "Thinking",
                self.values
                    .get("thinking")
                    .cloned()
                    .unwrap_or_else(|| "on".into()),
                "Whether reasoning models emit a hidden think block.",
            ),
            (
                "Reasoning budget",
                self.values
                    .get("thinking_budget")
                    .cloned()
                    .unwrap_or_else(|| "med".into()),
                "Named <think> token cap (low/med/high/xhigh/max/uncapped).",
            ),
            (
                "Serve",
                format!("{}:{}", self.host, self.port),
                "OpenAI-compatible endpoint used by chat and API clients.",
            ),
        ]
    }
}

fn defaults() -> BTreeMap<String, String> {
    [
        ("kv_cache", "auto"),
        ("kv_adaptive", "off"),
        ("flash_mode", "auto"),
        ("default_model", "qwen3.5:9b"),
        ("temperature", "0.3"),
        ("top_p", "0.8"),
        ("repeat_penalty", "1.05"),
        ("max_tokens", "4096"),
        ("max_seq", "32768"),
        ("thinking", "on"),
        ("thinking_budget", "med"),
        ("max_think_tokens", "2048"),
        ("max_total_think_tokens", "0"),
        ("host", "0.0.0.0"),
        ("port", "11435"),
        ("idle_timeout", "300"),
        ("dflash_mode", "off"),
        ("dflash_adaptive_b", "true"),
        ("dflash_ngram_block", "auto"),
        ("cask", "false"),
        ("cask_budget", "512"),
        ("cask_beta", "128"),
        ("cask_auto_attach", "true"),
        ("prompt_normalize", "true"),
        ("mmq_screen", "auto"),
        ("prefill_compression", "off"),
        // pflash (prefill KV compression). Compression no-ops without a drafter.
        ("prefill_drafter", ""),
        ("prefill_threshold", "32768"),
        ("mtp_mode", "auto"),
        ("mtp_k", "3"),
        // Mirror bun CONFIG_DEFAULTS for the remaining surfaced keys.
        ("chat_template", ""),
        ("default_chatml", "true"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        _ => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ConfigState seeded from an explicit value map (defaults overlaid).
    fn state_with(pairs: &[(&str, &str)]) -> ConfigState {
        let mut values = defaults();
        let mut overrides = BTreeSet::new();
        for (k, v) in pairs {
            overrides.insert((*k).to_string());
            values.insert((*k).to_string(), (*v).to_string());
        }
        ConfigState {
            host: "0.0.0.0".into(),
            port: 11435,
            default_model: "qwen3.5:9b".into(),
            values,
            overrides,
            per_model_count: 0,
            loaded_from_disk: true,
            warning: None,
        }
    }

    #[test]
    fn pflash_needs_drafter_logic() {
        // 5d honest state: compression on + no drafter -> warns; with a drafter,
        // or compression off, -> fine.
        assert!(!state_with(&[]).pflash_needs_drafter(), "off by default = fine");
        assert!(
            state_with(&[("prefill_compression", "auto")]).pflash_needs_drafter(),
            "auto without drafter needs one"
        );
        assert!(
            state_with(&[("prefill_compression", "always")]).pflash_needs_drafter(),
            "always without drafter needs one"
        );
        assert!(
            !state_with(&[("prefill_compression", "auto"), ("prefill_drafter", "/d.hfq")])
                .pflash_needs_drafter(),
            "auto WITH a drafter is fine"
        );
    }

    #[test]
    fn easy_rows_surface_pflash_with_honest_hint() {
        // The Prefill easy row exists and flags the missing drafter inline.
        let st = state_with(&[("prefill_compression", "auto")]);
        let prefill = st
            .easy_rows()
            .into_iter()
            .find(|(label, _, _)| *label == "Prefill")
            .expect("Prefill easy row present");
        assert!(prefill.1.contains("needs drafter"), "no-op state shown: {}", prefill.1);

        // With a drafter set, the hint goes away.
        let st2 = state_with(&[("prefill_compression", "auto"), ("prefill_drafter", "/d.hfq")]);
        let prefill2 = st2
            .easy_rows()
            .into_iter()
            .find(|(label, _, _)| *label == "Prefill")
            .unwrap();
        assert!(!prefill2.1.contains("needs drafter"), "hint cleared: {}", prefill2.1);
    }

    #[test]
    fn easy_view_vectors_stay_parallel() {
        // All four easy vectors must stay equal length (5d adds a row, 5e adds
        // help keys) or the marker/help attaches to the wrong row.
        let st = state_with(&[]);
        let n = st.easy_rows().len();
        assert_eq!(st.easy_keys().len(), n);
        assert_eq!(st.easy_override_state().len(), n);
        assert_eq!(st.easy_help_keys().len(), n);
    }

    #[test]
    fn easy_help_keys_have_explainers() {
        // 5e: every easy row's help key must resolve to a curated explainer.
        for key in state_with(&[]).easy_help_keys() {
            assert!(
                crate::hipfire::knobs::knob_info(key).is_some(),
                "easy help key {key} has no explainer"
            );
        }
    }
}
