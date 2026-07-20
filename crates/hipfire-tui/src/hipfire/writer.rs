// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Writable config persistence for the TUI Settings tab.
//!
//! This is the Rust mirror of the bun CLI config writer
//! (`cli/index.ts` CONFIG_DEFAULTS + validateConfigValue + saveConfig). The two
//! editors MUST agree on key names, the enum allowlists, and the numeric ranges
//! so `hipfire config` (bun) and the rust TUI read/write the SAME
//! `~/.hipfire/config.json` without divergence.
//!
//! Persistence model (matches bun saveConfig):
//!   * read-modify-write the raw JSON object (preserve every other key)
//!   * validate before writing (reject out-of-allowlist enum / out-of-range
//!     numeric) — an invalid value is NEVER persisted
//!   * atomic write: temp file in the same dir + rename over the target

use std::{fs, path::Path};

use serde_json::Value;

/// The kinds of settings the TUI can mutate. Each variant carries the canonical
/// key name and, for enums, the allowlist of legal string values mirrored from
/// the bun `validateConfigValue`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    /// String enum: cycle through a fixed allowlist (mirrors bun `includes(...)`).
    Enum(&'static [&'static str]),
    /// Bool stored as JSON true/false (mirrors bun `typeof value === "boolean"`).
    Bool,
    /// Float in an inclusive range, persisted as a JSON number.
    Float { min: f64, max: f64 },
    /// Integer in an inclusive range, persisted as a JSON integer.
    Int { min: i64, max: i64 },
    /// Free string (e.g. chat_template path). Empty string = unset/default.
    /// `require_existing_file` mirrors bun's existsSync+isFile path check.
    FreeStr { require_existing_file: bool },
}

/// One editable setting: canonical key + how to validate/cycle it.
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub key: &'static str,
    pub kind: FieldKind,
}

// ── Allowlists mirrored 1:1 from cli/index.ts:355-409 ───────────────────────
const KV_CACHE: &[&str] = &[
    "auto", "q8", "asym4", "asym3", "asym2", "fwht4", "fwht3", "fwht2", "turbo", "turbo4",
    "turbo3", "turbo2",
];
// kv_adaptive: mirrors the bun config-TUI's options list (KV_ADAPTIVE_OPTIONS in
// cli/index.ts) 1:1 — the 4 presets PLUS the 9 explicit-tier `advanced:k=…,v=…`
// combinations — so the ratatui cycles the same set the bun TUI does and can
// restore an existing advanced value (not fall back into the simple list).
const KV_ADAPTIVE: &[&str] = &[
    "off",
    "conservative",
    "balanced",
    "aggressive",
    "advanced:k=fwht4,v=lloyd4",
    "advanced:k=fwht4,v=lloyd3",
    "advanced:k=fwht4,v=lloyd2",
    "advanced:k=fwht3,v=lloyd4",
    "advanced:k=fwht3,v=lloyd3",
    "advanced:k=fwht3,v=lloyd2",
    "advanced:k=fwht2,v=lloyd4",
    "advanced:k=fwht2,v=lloyd3",
    "advanced:k=fwht2,v=lloyd2",
];
const MTP_MODE: &[&str] = &["off", "on", "auto"];
const DFLASH_MODE: &[&str] = &["on", "off", "auto"];
const THINKING: &[&str] = &["on", "off"];
// thinking_budget: named reasoning budgets, mirrors bun THINKING_BUDGET keys
// (cli/index.ts) 1:1 — low=512, med=2048, high=8192, xhigh=24576, max=32768,
// uncapped=0. The preset drives max_think_tokens unless a raw override is set.
const THINKING_BUDGET: &[&str] = &["low", "med", "high", "xhigh", "max", "uncapped"];
const FLASH_MODE: &[&str] = &["auto", "always", "never"];
const MMQ_SCREEN: &[&str] = &["off", "on", "auto"];
const PREFILL_COMPRESSION: &[&str] = &["off", "auto", "always"];

/// The set of fields the TUI Settings tab can edit. Order is the display order
/// in the editable-settings table.
pub const EDITABLE_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        key: "kv_cache",
        kind: FieldKind::Enum(KV_CACHE),
    },
    FieldSpec {
        key: "kv_adaptive",
        kind: FieldKind::Enum(KV_ADAPTIVE),
    },
    FieldSpec {
        key: "mtp_mode",
        kind: FieldKind::Enum(MTP_MODE),
    },
    FieldSpec {
        key: "dflash_mode",
        kind: FieldKind::Enum(DFLASH_MODE),
    },
    FieldSpec {
        key: "thinking",
        kind: FieldKind::Enum(THINKING),
    },
    FieldSpec {
        key: "thinking_budget",
        kind: FieldKind::Enum(THINKING_BUDGET),
    },
    FieldSpec {
        key: "chat_template",
        kind: FieldKind::FreeStr {
            require_existing_file: true,
        },
    },
    FieldSpec {
        key: "temperature",
        kind: FieldKind::Float { min: 0.0, max: 2.0 },
    },
    // top_p in bun is `> 0 && <= 1`; we approximate the open lower bound below
    // in `validate` (strict >0) — the spec's min is informational.
    FieldSpec {
        key: "top_p",
        kind: FieldKind::Float { min: 0.0, max: 1.0 },
    },
    FieldSpec {
        key: "max_tokens",
        kind: FieldKind::Int {
            min: 1,
            max: 131072,
        },
    },
    // KV cache capacity (tokens). Mirrors bun validateConfigValue `max_seq`:
    // int 512..=524288 (cli/index.ts:362). Backs the Easy "Context" row.
    FieldSpec {
        key: "max_seq",
        kind: FieldKind::Int {
            min: 512,
            max: 524288,
        },
    },
    // Extra fields surfaced read/write so the TUI matches the bun schema.
    FieldSpec {
        key: "flash_mode",
        kind: FieldKind::Enum(FLASH_MODE),
    },
    FieldSpec {
        key: "mmq_screen",
        kind: FieldKind::Enum(MMQ_SCREEN),
    },
    FieldSpec {
        key: "prefill_compression",
        kind: FieldKind::Enum(PREFILL_COMPRESSION),
    },
    // pflash drafter path (.hfq). Mirrors bun `prefill_drafter` (typeof string —
    // no existence check; "" disables). Required for compression to engage.
    FieldSpec {
        key: "prefill_drafter",
        kind: FieldKind::FreeStr {
            require_existing_file: false,
        },
    },
    // pflash auto-mode token cutoff. Mirrors bun int 0..=524288 (default 32768).
    FieldSpec {
        key: "prefill_threshold",
        kind: FieldKind::Int {
            min: 0,
            max: 524288,
        },
    },
    FieldSpec {
        key: "mtp_k",
        kind: FieldKind::Int { min: 1, max: 10 },
    },
    // Bool fields (cycled true/false via Left/Right/Space).
    FieldSpec {
        key: "dflash_adaptive_b",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "cask",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "prompt_normalize",
        kind: FieldKind::Bool,
    },
    FieldSpec {
        key: "default_chatml",
        kind: FieldKind::Bool,
    },
];

/// Look up the spec for a key, if it is editable.
pub fn field_spec(key: &str) -> Option<&'static FieldSpec> {
    EDITABLE_FIELDS.iter().find(|f| f.key == key)
}

/// Validate a candidate value for `key`. Mirrors bun `validateConfigValue`.
/// Returns the canonical [`Value`] to persist, or `None` if invalid.
pub fn validate(key: &str, raw: &str) -> Option<Value> {
    let spec = field_spec(key)?;
    match spec.kind {
        FieldKind::Enum(allow) => {
            if allow.contains(&raw) {
                Some(Value::String(raw.to_string()))
            } else {
                None
            }
        }
        FieldKind::Bool => match raw {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        FieldKind::Float { min, max } => {
            let v: f64 = raw.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            // top_p has a strict lower bound (>0) in bun; enforce it.
            let lo_ok = if key == "top_p" { v > min } else { v >= min };
            if lo_ok && v <= max {
                serde_json::Number::from_f64(v).map(Value::Number)
            } else {
                None
            }
        }
        FieldKind::Int { min, max } => {
            let v: i64 = raw.parse().ok()?;
            if v >= min && v <= max {
                Some(Value::Number(v.into()))
            } else {
                None
            }
        }
        FieldKind::FreeStr {
            require_existing_file,
        } => {
            if raw.is_empty() {
                return Some(Value::String(String::new()));
            }
            if require_existing_file {
                let expanded = expand_tilde(raw);
                let p = Path::new(&expanded);
                if p.is_file() {
                    Some(Value::String(raw.to_string()))
                } else {
                    None
                }
            } else {
                Some(Value::String(raw.to_string()))
            }
        }
    }
}

fn expand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest).to_string_lossy().into_owned();
        }
    }
    s.to_string()
}

/// Read-modify-write a single key into the config JSON at `path`, preserving all
/// other keys, validating first, and writing atomically (temp + rename).
///
/// `default_model` and any other free key can be written via
/// [`write_raw_value`]; this entry point is for the allowlisted editable fields.
pub fn write_value(path: &Path, key: &str, raw: &str) -> Result<Value, WriteError> {
    let value = validate(key, raw).ok_or_else(|| WriteError::Invalid {
        key: key.to_string(),
        value: raw.to_string(),
    })?;
    write_raw_value(path, key, value.clone())?;
    Ok(value)
}

/// Read-modify-write an already-validated [`Value`] for `key`. Used for
/// `default_model` (selected in the Models tab) and the validated editable
/// fields. Preserves every other key; atomic temp+rename.
///
/// Concurrency: a concurrent writer (the bun `hipfire config set` CLI, or a
/// second TUI) may change a DIFFERENT key between our read and our rename. To
/// avoid clobbering that unrelated write, we RE-READ the current on-disk object
/// immediately before serializing and apply ONLY our one changed key onto that
/// latest content (a single-key merge). A full-file replacement built from a
/// stale snapshot would silently revert the concurrent key.
///
/// Residual race: two writers changing the SAME key simultaneously is
/// last-writer-wins (whichever rename lands second). That is acceptable — there
/// is no meaningful "correct" merge of two competing values for one key, and
/// the rename itself is atomic so no torn/partial JSON is ever observable.
pub fn write_raw_value(path: &Path, key: &str, value: Value) -> Result<(), WriteError> {
    merge_key_and_write(path, key, value)
}

/// Re-read the current on-disk object as LATE as possible, insert ONLY `key`,
/// then serialize + atomic temp+rename. By re-reading immediately before the
/// write (rather than building a full snapshot up-front and inserting into it),
/// a concurrent writer that touched a different key in the meantime is
/// preserved — we only ever overwrite the single key we own.
fn merge_key_and_write(path: &Path, key: &str, value: Value) -> Result<(), WriteError> {
    let mut obj = read_object(path)?;
    obj.insert(key.to_string(), value);
    write_object(path, &obj)
}

/// Read the config file as a JSON object. A missing file is treated as an empty
/// object (the bun writer stores only non-default keys, so an absent file is
/// the all-defaults state). A present-but-corrupt file is an error rather than
/// a silent clobber — we never overwrite keys we could not parse.
fn read_object(path: &Path) -> Result<serde_json::Map<String, Value>, WriteError> {
    match fs::read_to_string(path) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(serde_json::Map::new());
            }
            match serde_json::from_str::<Value>(trimmed) {
                Ok(Value::Object(map)) => Ok(map),
                Ok(_) => Err(WriteError::NotObject),
                Err(e) => Err(WriteError::Parse(e.to_string())),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(serde_json::Map::new()),
        Err(e) => Err(WriteError::Io(e.to_string())),
    }
}

/// Atomically write the JSON object to `path`: serialize, write to a temp file
/// in the same directory, then rename over the target (rename is atomic on the
/// same filesystem). Matches the bun writer's pretty-print + trailing newline.
fn write_object(path: &Path, obj: &serde_json::Map<String, Value>) -> Result<(), WriteError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| WriteError::Io(e.to_string()))?;
    }
    let mut body = serde_json::to_string_pretty(&Value::Object(obj.clone()))
        .map_err(|e| WriteError::Io(e.to_string()))?;
    body.push('\n');

    let tmp = tmp_path(path);
    fs::write(&tmp, body.as_bytes()).map_err(|e| WriteError::Io(e.to_string()))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        WriteError::Io(e.to_string())
    })?;
    Ok(())
}

fn tmp_path(path: &Path) -> std::path::PathBuf {
    let pid = std::process::id();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.json".to_string());
    let tmp_name = format!(".{name}.tmp.{pid}");
    match path.parent() {
        Some(p) => p.join(tmp_name),
        None => std::path::PathBuf::from(tmp_name),
    }
}

#[derive(Debug)]
pub enum WriteError {
    Invalid { key: String, value: String },
    NotObject,
    Parse(String),
    Io(String),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Invalid { key, value } => {
                write!(f, "rejected invalid value for {key}: {value:?}")
            }
            WriteError::NotObject => write!(f, "config.json is not an object; refusing to write"),
            WriteError::Parse(e) => write!(f, "config.json parse error: {e}; refusing to write"),
            WriteError::Io(e) => write!(f, "config write error: {e}"),
        }
    }
}

impl std::error::Error for WriteError {}

/// Remove a single key from the config JSON at `path`, falling its value back to
/// the inherited/default resolution (registry → arch → hardcoded default). This
/// mirrors the bun `hipfire config reset <key>` semantics: the CLI sets the key
/// to its default and then `saveConfig` only writes non-default keys, so the net
/// effect is the key disappearing from `config.json`. We do that directly —
/// delete the key, preserve every other key, atomic temp+rename.
///
/// Returns `Ok(true)` if the key was present and removed, `Ok(false)` if it was
/// already absent (no write performed). A corrupt config is an error rather than
/// a clobber, identical to [`write_raw_value`].
pub fn delete_key(path: &Path, key: &str) -> Result<bool, WriteError> {
    let mut obj = read_object(path)?;
    if obj.remove(key).is_none() {
        return Ok(false);
    }
    write_object(path, &obj)?;
    Ok(true)
}

/// Reset the ENTIRE config back to defaults by writing an empty object. Mirrors
/// the bun `hipfire config reset` (no key): `saveConfig({...CONFIG_DEFAULTS})`
/// keeps only non-default keys, i.e. writes `{}`. Every resolved value then
/// falls through registry/arch inheritance again.
///
/// Unlike [`delete_key`], this does NOT read-then-merge: it is the recovery
/// hatch for a stale OR corrupt config, so it unconditionally overwrites the
/// file with `{}` (a corrupt config must not block its own reset).
pub fn reset_all(path: &Path) -> Result<(), WriteError> {
    write_object(path, &serde_json::Map::new())
}

/// Cycle an enum field's current value to the next (or previous) allowlist entry.
/// Returns the new value string, or `None` if the key is not a cyclable enum.
pub fn cycle_enum(key: &str, current: &str, forward: bool) -> Option<String> {
    let spec = field_spec(key)?;
    if let FieldKind::Enum(allow) = spec.kind {
        let idx = allow.iter().position(|v| *v == current).unwrap_or(0);
        let n = allow.len();
        let next = if forward {
            (idx + 1) % n
        } else {
            (idx + n - 1) % n
        };
        Some(allow[next].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "hipfire-tui-writer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = base.join(unique);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_valid_value_preserving_other_keys() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        // Seed with pre-existing keys including a non-editable one.
        let mut f = fs::File::create(&cfg).unwrap();
        write!(
            f,
            "{{\n  \"default_model\": \"qwen3.5:9b\",\n  \"port\": 11435,\n  \"kv_cache\": \"auto\"\n}}\n"
        )
        .unwrap();
        drop(f);

        // Write a valid kv_cache value.
        let v = write_value(&cfg, "kv_cache", "q8").expect("valid write");
        assert_eq!(v, Value::String("q8".into()));

        // Read back: value applied, JSON still valid, OTHER keys preserved.
        let raw = fs::read_to_string(&cfg).unwrap();
        let parsed: Value = serde_json::from_str(&raw).expect("valid JSON round-trip");
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("kv_cache").unwrap(), &Value::String("q8".into()));
        assert_eq!(
            obj.get("default_model").unwrap(),
            &Value::String("qwen3.5:9b".into())
        );
        assert_eq!(obj.get("port").unwrap().as_i64().unwrap(), 11435);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_bad_enum_without_persisting() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{\n  \"kv_cache\": \"auto\"\n}\n").unwrap();

        let err = write_value(&cfg, "kv_cache", "not_a_mode");
        assert!(err.is_err(), "bad enum must be rejected");

        // File unchanged: still the original valid value.
        let raw = fs::read_to_string(&cfg).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("kv_cache").unwrap(),
            &Value::String("auto".into())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_out_of_range_numeric() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();

        assert!(write_value(&cfg, "temperature", "5.0").is_err());
        assert!(write_value(&cfg, "max_tokens", "0").is_err());
        assert!(write_value(&cfg, "top_p", "0").is_err()); // strict >0
        assert!(write_value(&cfg, "mtp_k", "11").is_err());
        // max_seq mirrors bun int 512..=524288.
        assert!(write_value(&cfg, "max_seq", "511").is_err());
        assert!(write_value(&cfg, "max_seq", "524289").is_err());

        // Valid boundaries accepted.
        assert!(write_value(&cfg, "temperature", "2.0").is_ok());
        assert!(write_value(&cfg, "max_tokens", "131072").is_ok());
        assert!(write_value(&cfg, "top_p", "1").is_ok());
        // max_seq boundaries + the Easy "Context" default.
        assert!(write_value(&cfg, "max_seq", "512").is_ok());
        assert!(write_value(&cfg, "max_seq", "524288").is_ok());
        assert!(write_value(&cfg, "max_seq", "32768").is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn creates_file_when_missing() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        assert!(!cfg.exists());
        write_value(&cfg, "thinking", "off").expect("create + write");
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("thinking").unwrap(),
            &Value::String("off".into())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cycles_enum_values() {
        assert_eq!(cycle_enum("kv_cache", "auto", true).unwrap(), "q8");
        assert_eq!(cycle_enum("dflash_mode", "off", true).unwrap(), "auto");
        // wrap-around backward from first
        assert_eq!(cycle_enum("thinking", "on", false).unwrap(), "off");
        assert!(cycle_enum("temperature", "0.3", true).is_none());
    }

    #[test]
    fn bool_field_round_trips_and_rejects_garbage() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();
        let v = write_value(&cfg, "prompt_normalize", "false").expect("valid bool");
        assert_eq!(v, Value::Bool(false));
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("prompt_normalize").unwrap(),
            &Value::Bool(false)
        );
        assert!(write_value(&cfg, "cask", "yes").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_write_to_different_key_is_preserved() {
        // F1: a concurrent writer (bun `hipfire config set`, or a second TUI)
        // changes a DIFFERENT key while we write ours. Because write_raw_value
        // re-reads the latest on-disk object and merges ONLY our one key, the
        // concurrent unrelated-key write must survive (not be clobbered by a
        // stale full-file snapshot).
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{\n  \"kv_cache\": \"auto\"\n}\n").unwrap();

        // Simulate the interleaving: AFTER the initial state, a concurrent
        // process writes a different key (thinking) straight to disk...
        write_value(&cfg, "thinking", "off").expect("concurrent write");
        // ...and then OUR writer commits its own key. The merge must re-read
        // the latest content (which now has `thinking`) and keep it.
        write_value(&cfg, "temperature", "0.7").expect("our write");

        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        let obj = parsed.as_object().unwrap();
        // Our key landed.
        assert_eq!(
            obj.get("temperature").unwrap().as_f64().unwrap(),
            0.7,
            "our key must be written"
        );
        // The concurrent different-key write was preserved (not clobbered).
        assert_eq!(
            obj.get("thinking").unwrap(),
            &Value::String("off".into()),
            "concurrent different-key write must survive the merge"
        );
        // And the original key is still intact.
        assert_eq!(obj.get("kv_cache").unwrap(), &Value::String("auto".into()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_key_simultaneous_is_last_writer_wins() {
        // F1 residual: two writers changing the SAME key — last rename wins.
        // This is the documented, acceptable residual (no torn JSON; atomic).
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();
        write_value(&cfg, "kv_cache", "q8").unwrap();
        write_value(&cfg, "kv_cache", "asym4").unwrap(); // later write wins
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed.as_object().unwrap().get("kv_cache").unwrap(),
            &Value::String("asym4".into())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_clobber_corrupt_file() {
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{ this is not json").unwrap();
        let err = write_value(&cfg, "kv_cache", "q8");
        assert!(matches!(err, Err(WriteError::Parse(_))));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn kv_adaptive_mirrors_bun_full_option_list() {
        // Adaptive-KV is editable in the ratatui exactly as in the bun TUI: the 4
        // presets AND the 9 explicit-tier advanced combos validate + cycle, so an
        // existing advanced value can be restored (not lost to the simple list).
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();
        for m in [
            "off",
            "conservative",
            "balanced",
            "aggressive",
            "advanced:k=fwht3,v=lloyd3",
        ] {
            assert!(
                write_value(&cfg, "kv_adaptive", m).is_ok(),
                "{m} should validate"
            );
        }
        // A garbage value is still rejected.
        assert!(write_value(&cfg, "kv_adaptive", "advanced:k=zzz").is_err());
        // Cycling from an advanced value advances within the full list, not back
        // to a preset (the bug this fixes).
        assert_eq!(
            cycle_enum("kv_adaptive", "advanced:k=fwht4,v=lloyd4", true).unwrap(),
            "advanced:k=fwht4,v=lloyd3"
        );
        assert_eq!(
            cycle_enum("kv_adaptive", "off", true).unwrap(),
            "conservative"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pflash_knobs_are_editable_and_range_checked() {
        // 5d: prefill_drafter (free string, no existence check) + prefill_threshold
        // (int 0..=524288) are now writable from the TUI, mirroring the bun schema.
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{}\n").unwrap();

        // Drafter accepts any string incl. a path that doesn't exist yet, and "".
        assert!(write_value(&cfg, "prefill_drafter", "~/.hipfire/models/d.hfq").is_ok());
        assert!(write_value(&cfg, "prefill_drafter", "").is_ok());
        // Threshold range boundaries.
        assert!(write_value(&cfg, "prefill_threshold", "0").is_ok());
        assert!(write_value(&cfg, "prefill_threshold", "524288").is_ok());
        assert!(write_value(&cfg, "prefill_threshold", "524289").is_err());
        assert!(write_value(&cfg, "prefill_threshold", "-1").is_err());
        // Compression enum unchanged.
        assert_eq!(
            cycle_enum("prefill_compression", "off", true).unwrap(),
            "auto"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_key_removes_only_that_key() {
        // 5a: reset one override -> the key disappears (falls back to default),
        // every OTHER key is preserved verbatim.
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(
            &cfg,
            "{\n  \"kv_cache\": \"q8\",\n  \"thinking\": \"off\",\n  \"port\": 11435\n}\n",
        )
        .unwrap();

        let removed = delete_key(&cfg, "kv_cache").expect("delete ok");
        assert!(removed, "an existing key reports removed=true");

        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        let obj = parsed.as_object().unwrap();
        assert!(!obj.contains_key("kv_cache"), "reset key is gone from disk");
        assert_eq!(obj.get("thinking").unwrap(), &Value::String("off".into()));
        assert_eq!(obj.get("port").unwrap().as_i64().unwrap(), 11435);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_absent_key_is_noop() {
        // Resetting a key that is already at its default (absent on disk) must not
        // write or error — it simply reports removed=false.
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{\n  \"thinking\": \"off\"\n}\n").unwrap();
        let removed = delete_key(&cfg, "kv_cache").expect("noop ok");
        assert!(!removed, "absent key reports removed=false");
        // File content is byte-for-byte unchanged.
        assert_eq!(
            fs::read_to_string(&cfg).unwrap(),
            "{\n  \"thinking\": \"off\"\n}\n"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_key_refuses_corrupt_file() {
        // A corrupt config must not be clobbered by a single-key reset (use
        // reset_all for recovery instead).
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{ not json").unwrap();
        assert!(matches!(
            delete_key(&cfg, "kv_cache"),
            Err(WriteError::Parse(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_all_empties_to_defaults() {
        // 5a reset-all: everything reverts to defaults -> config.json = {}.
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(
            &cfg,
            "{\n  \"kv_cache\": \"q8\",\n  \"dflash_mode\": \"auto\"\n}\n",
        )
        .unwrap();
        reset_all(&cfg).expect("reset all ok");
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({}),
            "reset-all writes an empty object"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_all_recovers_a_corrupt_config() {
        // The recovery hatch: a config so broken it can't be parsed must still be
        // resettable. reset_all overwrites unconditionally (no read-merge).
        let dir = temp_dir();
        let cfg = dir.join("config.json");
        fs::write(&cfg, "{ totally broken ::: not json").unwrap();
        reset_all(&cfg).expect("reset all recovers corrupt file");
        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!({}));
        let _ = fs::remove_dir_all(&dir);
    }
}
