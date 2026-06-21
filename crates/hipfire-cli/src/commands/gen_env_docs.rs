// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire gen-env-docs` (hidden) — scan the tracked source tree for
//! environment-variable usage and render the committed env-var docs:
//! `docs/env-vars.md` (a Markdown table) and
//! `crates/hipfire-runtime/src/env_docs.rs` (the compiled `EnvVarDoc` registry).
//!
//! This is the Rust-native replacement for the former `scripts/gen-env-docs.py`
//! + `scripts/check-env-docs.py`. With `--check` it regenerates to memory and
//! diffs against the committed files (plus verifies that every `HIPFIRE_*` var
//! named in the top-level docs is covered), exiting non-zero on drift — the
//! freshness gate `tests/no-gpu-ci.sh` runs.
//!
//! Two deliberate improvements over the Python it replaces:
//!   * `source:` fields are **repo-relative**, not absolute machine paths.
//!   * Pipe-escaping on read-back is idempotent (the Python re-escaped `|`
//!     every run, so its output never stabilized — there was no working
//!     content gate). The `.rs` output is also piped through `rustfmt` so it
//!     stops fighting `cargo fmt`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use regex::Regex;

#[derive(Debug, clap::Args)]
pub struct GenEnvDocsArgs {
    /// Markdown reference written/checked (repo-relative).
    #[arg(long, default_value = "docs/env-vars.md")]
    pub doc: String,
    /// Generated Rust registry module written/checked (repo-relative).
    #[arg(long, default_value = "crates/hipfire-runtime/src/env_docs.rs")]
    pub rust_module: String,
    /// Verify the committed files match the current source without writing;
    /// exit non-zero on any drift (for CI).
    #[arg(long)]
    pub check: bool,
}

/// One discovered env-var documentation entry.
#[derive(Clone, Debug)]
struct EnvDoc {
    name: String,
    description: String,
    /// Repo-relative `path:line`.
    source: String,
}

const EXCLUDE_PREFIXES: &[&str] = &["third_party/", "target/"];

const UNHELPFUL_DESCRIPTIONS: &[&str] = &[
    "Behavioral use is defined in source; add a dedicated env-doc entry.",
    "`HIPFIRE_`",
    "HIPFIRE_*",
    "HIPFIRE",
];

const SCORE_TOKENS: &[&str] = &[
    "default",
    "defaults",
    "set",
    "enable",
    "disable",
    "opt",
    "file",
    "path",
    "directory",
    "mode",
    "timeout",
    "rate",
    "batch",
    "budget",
    "token",
    "kv",
    "draft",
    "loop",
    "spec",
    "ddtree",
    "gpu",
    "dump",
    "log",
];

pub fn run(args: GenEnvDocsArgs) -> anyhow::Result<()> {
    let root = repo_root()?;
    let docs = collect_env_data(&root)?;

    let markdown = render_markdown(&docs);
    let rust_unformatted = render_rust_module(&docs);
    let rust = rustfmt(&rust_unformatted)?;

    let doc_path = root.join(&args.doc);
    let rust_path = root.join(&args.rust_module);

    if args.check {
        let mut stale = Vec::new();
        check_file(&doc_path, markdown.as_bytes(), &mut stale);
        check_file(&rust_path, rust.as_bytes(), &mut stale);
        let missing = coverage_gaps(&root, &markdown);

        if !stale.is_empty() || !missing.is_empty() {
            let mut msg = String::new();
            if !stale.is_empty() {
                msg.push_str(&format!(
                    "env docs are stale ({} file(s)): {}\n",
                    stale.len(),
                    stale.join(", ")
                ));
            }
            for (doc, name) in &missing {
                msg.push_str(&format!(
                    "docs/env-vars.md missing `{name}` referenced by {doc}\n"
                ));
            }
            msg.push_str("regenerate with `cargo run -p hipfire-cli -- gen-env-docs` and commit.");
            anyhow::bail!("{msg}");
        }
        eprintln!("gen-env-docs: env docs are up to date");
        return Ok(());
    }

    if let Some(parent) = doc_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&doc_path, markdown.as_bytes())?;
    eprintln!("gen-env-docs: wrote {}", doc_path.display());
    std::fs::write(&rust_path, rust.as_bytes())?;
    eprintln!("gen-env-docs: wrote {}", rust_path.display());
    Ok(())
}

fn repo_root() -> anyhow::Result<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Ok(PathBuf::from(s));
        }
    }
    Ok(std::env::current_dir()?)
}

/// Curated entries for vars that only appear in docs, never in scanned source.
fn doc_only_entries() -> Vec<EnvDoc> {
    vec![
        EnvDoc {
            name: "HIPFIRE_LOCAL".to_string(),
            description: "Force local-spawn behavior and skip serve HTTP in documented workflows"
                .to_string(),
            source: "README.md:962".to_string(),
        },
        EnvDoc {
            name: "HIPFIRE_PYTHON".to_string(),
            description:
                "Python interpreter used by the no-GPU CI shell gate for Python tooling and tests"
                    .to_string(),
            source: ".github/CONTRIBUTING.md:86".to_string(),
        },
    ]
}

/// `git ls-files`, filtered to `.rs`/`.ts`, excluding vendored/build trees.
fn tracked_sources(root: &Path) -> anyhow::Result<Vec<String>> {
    let out = Command::new("git")
        .args(["ls-files"])
        .current_dir(root)
        .output()?;
    if !out.status.success() {
        anyhow::bail!("git ls-files failed in {}", root.display());
    }
    let mut files = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if !(line.ends_with(".rs") || line.ends_with(".ts")) {
            continue;
        }
        if EXCLUDE_PREFIXES.iter().any(|p| line.starts_with(p)) {
            continue;
        }
        files.push(line.to_string());
    }
    Ok(files)
}

struct EnvUsage {
    name: String,
    /// Repo-relative `path:line`.
    source: String,
    line_idx: usize, // 0-based
}

fn env_read_re() -> Regex {
    Regex::new(concat!(
        r#"std::env::var(?:_os)?\(\s*["']([A-Z][A-Z0-9_]+)["']\s*\)"#,
        r#"|env::var(?:_os)?\(\s*["']([A-Z][A-Z0-9_]+)["']\s*\)"#,
        r#"|std::env::set_var\(\s*["']([A-Z][A-Z0-9_]+)["']\s*,"#,
        r#"|env::set_var\(\s*["']([A-Z][A-Z0-9_]+)["']\s*,"#,
        r#"|process\.env\.([A-Z][A-Z0-9_]+)"#,
        r#"|process\.env\[["']([A-Z][A-Z0-9_]+)["']\]"#,
    ))
    .expect("env read regex")
}

fn extract_env_usages(rel: &str, lines: &[String], re: &Regex) -> Vec<EnvUsage> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        for caps in re.captures_iter(line) {
            for g in caps.iter().skip(1).flatten() {
                let name = g.as_str();
                if !name.is_empty() {
                    out.push(EnvUsage {
                        name: name.to_string(),
                        source: format!("{rel}:{}", i + 1),
                        line_idx: i,
                    });
                    break;
                }
            }
        }
    }
    out
}

fn collect_env_data(root: &Path) -> anyhow::Result<Vec<EnvDoc>> {
    let read_re = env_read_re();
    let mut order: Vec<String> = Vec::new();
    let mut usages: HashMap<String, Vec<EnvUsage>> = HashMap::new();
    let mut raw_lines: HashMap<String, Vec<String>> = HashMap::new();

    for rel in tracked_sources(root)? {
        let Ok(text) = std::fs::read_to_string(root.join(&rel)) else {
            continue;
        };
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        for usage in extract_env_usages(&rel, &lines, &read_re) {
            if !usages.contains_key(&usage.name) {
                order.push(usage.name.clone());
            }
            usages.entry(usage.name.clone()).or_default().push(usage);
        }
        raw_lines.insert(rel, lines);
    }

    let existing = collect_existing_descs(root);

    let mut docs: Vec<EnvDoc> = Vec::new();
    for name in &order {
        let usage_list = &usages[name];
        let mut best: Option<EnvDoc> = None;
        for usage in usage_list {
            let lines = &raw_lines[source_path(&usage.source)];
            let line = &lines[usage.line_idx];
            let cands = extract_comment_descriptions(usage.line_idx, lines);
            let desc = infer_default(
                existing.get(name).map(String::as_str),
                name,
                &cands,
                line,
                lines,
                usage.line_idx,
                &usage.source,
            );
            // Match gen-env-docs.py: adopt the first usage, then re-adopt on
            // every *helpful* usage — so a var documented in multiple files is
            // attributed to the last helpful usage in scan (git ls-files) order.
            if best.is_none() || is_helpful_description(&desc) {
                best = Some(EnvDoc {
                    name: name.clone(),
                    description: desc.clone(),
                    source: usage.source.clone(),
                });
            }
        }
        let best = best.unwrap_or_else(|| EnvDoc {
            name: name.clone(),
            description: infer_name_from_var(name, &usage_list[0].source),
            source: usage_list[0].source.clone(),
        });
        docs.push(best);
    }

    // Curated doc-only entries override/extend scanned ones.
    for entry in doc_only_entries() {
        if let Some(slot) = docs.iter_mut().find(|d| d.name == entry.name) {
            *slot = entry;
        } else {
            docs.push(entry);
        }
    }

    docs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(docs)
}

fn source_path(source: &str) -> &str {
    source.rsplit_once(':').map(|(p, _)| p).unwrap_or(source)
}

// ─── description heuristics (ported from gen-env-docs.py) ───────────────────

fn normalize_comment_line(line: &str) -> String {
    let mut s = line.trim().to_string();
    for prefix in ["//!", "///", "//", "#"] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim_start().to_string();
            break;
        }
    }
    s = s.trim().to_string();
    // Drop a leading `inline-code` label, e.g. "`HIPFIRE_X` — desc".
    if s.starts_with('`') && s[1..].contains('`') {
        let rest = &s[1..];
        if let Some(end) = rest.find('`') {
            if end > 0 {
                s = rest[end + 1..]
                    .trim_start_matches([' ', ':', '—', '-'])
                    .to_string();
            }
        }
    }
    collapse_ws(&s)
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_wrapping_quotes(text: &str) -> String {
    let b = text.as_bytes();
    if b.len() >= 2
        && ((text.starts_with('"') && text.ends_with('"'))
            || (text.starts_with('\'') && text.ends_with('\'')))
    {
        return text[1..text.len() - 1].to_string();
    }
    text.to_string()
}

fn normalize_description(text: &str) -> String {
    let trimmed = text.trim().trim_matches([' ', '.']);
    collapse_ws(trimmed).replace('`', "\"")
}

fn is_helpful_description(value: &str) -> bool {
    let text = normalize_description(value);
    if text.is_empty() {
        return false;
    }
    if UNHELPFUL_DESCRIPTIONS.contains(&text.as_str()) {
        return false;
    }
    if text.to_lowercase().contains("behavioral use is defined") {
        return false;
    }
    if text.len() < 18 {
        return false;
    }
    if text.chars().all(|c| {
        c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-' || c == '_' || c.is_whitespace()
    }) {
        return false;
    }
    if text.split_whitespace().count() <= 2 {
        return false;
    }
    true
}

fn score_comment(text: &str, var: &str) -> usize {
    let lower = text.to_lowercase();
    let mut score = text.len();
    if text.contains(var) {
        score += 40;
    }
    if SCORE_TOKENS.iter().any(|tok| lower.contains(tok)) {
        score += 20;
    }
    score
}

fn comment_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(?://!|///|//|#)\s?(.*)$").unwrap())
}

fn extract_comment_descriptions(line_idx: usize, lines: &[String]) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    if line_idx >= lines.len() {
        return candidates;
    }

    let raw_line = &lines[line_idx];
    if let Some(pos) = raw_line.find("//") {
        let c = normalize_comment_line(&raw_line[pos + 2..]);
        if !c.is_empty() {
            candidates.push(c);
        }
    }

    // Backward: contiguous doc comments directly above the usage.
    for back in 1..18 {
        if back > line_idx {
            break;
        }
        let txt = lines[line_idx - back].trim();
        if txt.is_empty() {
            break;
        }
        let Some(caps) = comment_re().captures(txt) else {
            break;
        };
        let clean = normalize_comment_line(&caps[1]);
        if !clean.is_empty() {
            candidates.push(clean);
        }
    }

    // Forward: a small tail for docs written after the code.
    for fwd in 1..8 {
        let cur = line_idx + fwd;
        if cur >= lines.len() {
            break;
        }
        let txt = lines[cur].trim();
        if txt.is_empty() {
            break;
        }
        let Some(caps) = comment_re().captures(txt) else {
            continue;
        };
        let clean = normalize_comment_line(&caps[1]);
        if !clean.is_empty() {
            candidates.push(clean);
        }
    }

    // Dedupe (normalized), preserving order.
    let mut seen = std::collections::HashSet::new();
    candidates
        .into_iter()
        .map(|c| normalize_description(&c))
        .filter(|c| !c.is_empty() && seen.insert(c.clone()))
        .collect()
}

fn usage_hints() -> &'static Vec<(Regex, &'static str)> {
    use std::sync::OnceLock;
    static HINTS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    HINTS.get_or_init(|| {
        let mk = |p: &str| Regex::new(&format!("(?i){p}")).unwrap();
        vec![
            (
                mk(r#"as_deref\(\)\s*==\s*Some\("1"\)"#),
                "Enabled when set to 1.",
            ),
            (
                mk(r#"as_deref\(\)\s*!=\s*Some\("0"\)"#),
                "Enabled by default; set to 0 to disable.",
            ),
            (
                mk(r#"as_deref\(\)\s*==\s*Some\("0"\)"#),
                "Disabled when set to 0.",
            ),
            (
                mk(r#"as_deref\(\)\.unwrap_or\("([^"]+)"\)"#),
                "Defaults to {} when unset.",
            ),
            (
                mk(r#"unwrap_or_else\(\|.*\|\s*("[^"]+")"#),
                "Defaults to {} when unset.",
            ),
            (
                mk(r#"parse::<\w+>\(\)\.unwrap_or\(([^\)]+)\)"#),
                "Parsed with fallback default {}.",
            ),
            (
                mk(r#"match\s+std::env::var"#),
                "Selects behavior from recognized values.",
            ),
            (
                mk(r#"\bparse::<(?:u\d+|usize|bool|f\d+|String|u8|u16|u32|u64)>"#),
                "Parsed into numeric or typed runtime setting.",
            ),
            (
                mk(r#"\bSome\(("[^"]+"|'[^']+')\)|\bOk\(("[^"]+"|'[^']+')\)"#),
                "Environment toggle value controls runtime behavior.",
            ),
            (
                mk(r#"as_deref\(\)\.is_ok\(\)"#),
                "Optional toggle; presence may enable feature behavior.",
            ),
            (
                mk(r#"Some\("true"\)|Some\("1"\)|Some\("yes"\)|Some\("on"\)"#),
                "Boolean-style toggle env var.",
            ),
        ]
    })
}

fn fill_template(template: &str, value: Option<&str>) -> String {
    match value {
        Some(v) if template.contains("{}") => template.replacen("{}", v, 1),
        _ => template.to_string(),
    }
}

fn infer_from_expression(
    line: &str,
    var: &str,
    line_ctx: &[String],
    line_idx: usize,
) -> Option<String> {
    let line_text = line.trim();
    if !line_text.is_empty() {
        for (re, template) in usage_hints() {
            if let Some(caps) = re.captures(line_text) {
                let first = caps
                    .iter()
                    .skip(1)
                    .flatten()
                    .map(|m| strip_wrapping_quotes(m.as_str()))
                    .next();
                return Some(normalize_description(&fill_template(
                    template,
                    first.as_deref(),
                )));
            }
        }

        let lower = line_text.to_lowercase();
        if lower.contains("parse().ok()") || lower.contains("parse::<") {
            let parse_type = if lower.contains("parse::<usize>()") {
                "usize/integer"
            } else if lower.contains("parse::<u32>()") {
                "u32"
            } else if lower.contains("parse::<u16>()") {
                "u16"
            } else if lower.contains("parse::<bool>()") {
                "boolean"
            } else if lower.contains("parse::<f32>()") || lower.contains("parse::<f64>()") {
                "floating-point"
            } else {
                "value"
            };
            return Some(format!(
                "Parsed as {parse_type} configuration from environment value."
            ));
        }
        if lower.contains("match std::env::var") || lower.contains("match env::var") {
            return Some(format!(
                "Reads `{var}` and branches runtime behavior by recognized values."
            ));
        }
        if lower.contains(".set_var(") && lower.contains(&var.to_lowercase()) {
            return Some(format!(
                "Sets `{var}` for runtime or child process configuration."
            ));
        }
    }

    let window_start = line_idx.saturating_sub(4);
    let window_end = (line_idx + 4).min(line_ctx.len());
    let local_text = line_ctx[window_start..window_end].join(" ").to_lowercase();
    if local_text.contains("parse::<") && local_text.contains("unwrap_or") {
        return Some(format!("Parses `{var}` with fallback defaults."));
    }
    if local_text.contains("match") && local_text.contains("as_deref") {
        return Some(format!(
            "Interprets `{var}` from environment to select behavior."
        ));
    }
    if local_text.contains("set_var") {
        return Some(format!(
            "Used to configure runtime execution by explicitly setting `{var}`."
        ));
    }
    None
}

fn infer_name_from_var(var: &str, source: &str) -> String {
    let mut label = var.strip_prefix("HIPFIRE_").unwrap_or(var).to_string();
    label = label.replace(['_', '-'], " ").to_lowercase();
    label = label.replace("ddtree", "DDTree");
    label = label
        .replace("mtp", "MTP")
        .replace("kv", "KV")
        .replace("q8", "Q8")
        .replace("q4", "Q4");
    label = collapse_ws(&label);
    if !label.is_empty() {
        format!("Runtime variable controlling {label} in hipfire.")
    } else {
        let file = Path::new(source)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        format!("Runtime control variable `{var}` defined in {file}.")
    }
}

/// Existing descriptions parsed from the committed Markdown table, so curated
/// wording is preserved across regenerations. Pipe-escapes are collapsed so the
/// round-trip is idempotent (the Python re-escaped `|` every run).
fn collect_existing_descs(root: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let doc = root.join("docs/env-vars.md");
    let Ok(text) = std::fs::read_to_string(&doc) else {
        return out;
    };
    let row = Regex::new(r"^\| `([A-Z0-9_]+)` \| (.*?) \|").unwrap();
    let esc_pipe = Regex::new(r"\\+\|").unwrap();
    for line in text.lines() {
        if let Some(caps) = row.captures(line) {
            let name = caps[1].to_string();
            let desc = esc_pipe.replace_all(caps[2].trim(), "|").to_string();
            out.insert(name, desc);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn infer_default(
    existing_desc: Option<&str>,
    var: &str,
    cands: &[String],
    line: &str,
    lines: &[String],
    line_idx: usize,
    usage_source: &str,
) -> String {
    if let Some(d) = existing_desc {
        if is_helpful_description(d) {
            return normalize_description(d);
        }
    }

    let mut ranked: Vec<(usize, &String)> = cands
        .iter()
        .filter(|c| is_helpful_description(c))
        .map(|c| (score_comment(c, var), c))
        .collect();
    if !ranked.is_empty() {
        // Highest score wins; ties resolve to first-seen (stable sort).
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        return ranked[0].1.clone();
    }

    if let Some(inferred) = infer_from_expression(line, var, lines, line_idx) {
        return normalize_description(&inferred);
    }

    infer_name_from_var(var, usage_source)
}

// ─── rendering ──────────────────────────────────────────────────────────────

fn escape_rust_str(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn render_markdown(docs: &[EnvDoc]) -> String {
    let total = docs.len();
    let hipfire = docs
        .iter()
        .filter(|d| d.name.starts_with("HIPFIRE_"))
        .count();
    let non = total - hipfire;
    let mut lines = vec![
        "# hipfire environment variables — canonical reference".to_string(),
        String::new(),
        "Generated automatically from source and inline comments by \
         `hipfire gen-env-docs`."
            .to_string(),
        String::new(),
        "| Variable | Description | Defined at |".to_string(),
        "|---|---|---|".to_string(),
    ];
    for doc in docs {
        let desc = doc.description.replace('|', "\\|");
        lines.push(format!("| `{}` | {desc} | `{}` |", doc.name, doc.source));
    }
    lines.push(String::new());
    lines.push(format!("- Total env vars: **{total}**"));
    lines.push(format!("- `HIPFIRE_*` vars: **{hipfire}**"));
    lines.push(format!("- non-`HIPFIRE_*` vars: **{non}**"));
    lines.join("\n") + "\n"
}

fn render_rust_module(docs: &[EnvDoc]) -> String {
    let mut s = String::new();
    s.push_str("#![allow(dead_code)]\n\n");
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str("//\n");
    s.push_str("// Generated automatically from source env usage by `hipfire gen-env-docs`.\n");
    s.push_str("// Do not hand-edit. Re-run `cargo run -p hipfire-cli -- gen-env-docs`.\n\n");
    s.push_str("/// Canonical environment-variable documentation registry.\n");
    s.push_str("///\n");
    s.push_str("/// Each entry is sourced from inline comments or generated defaults.\n");
    s.push_str("pub struct EnvVarDoc {\n");
    s.push_str("    pub name: &'static str,\n");
    s.push_str("    pub description: &'static str,\n");
    s.push_str("    pub source: &'static str,\n");
    s.push_str("}\n\n");
    s.push_str("impl EnvVarDoc {\n");
    s.push_str("    pub const fn name(&self) -> &'static str {\n");
    s.push_str("        self.name\n");
    s.push_str("    }\n");
    s.push_str("}\n\n");

    for doc in docs {
        s.push_str(&format!("/// `{}` — {}\n", doc.name, doc.description));
        s.push_str(&format!(
            "pub const ENV_{}: EnvVarDoc = EnvVarDoc {{\n",
            doc.name
        ));
        s.push_str(&format!("    name: \"{}\",\n", doc.name));
        s.push_str(&format!(
            "    description: \"{}\",\n",
            escape_rust_str(&doc.description)
        ));
        s.push_str(&format!("    source: \"{}\",\n", doc.source));
        s.push_str("};\n\n");
    }

    s.push_str("/// All documented environment variables in deterministic order.\n");
    s.push_str("pub const ALL_ENV_VARS: &[EnvVarDoc] = &[\n");
    for doc in docs {
        s.push_str(&format!("    ENV_{},\n", doc.name));
    }
    s.push_str("];\n");
    s
}

/// Pipe the generated module through `rustfmt` so the committed file is
/// fmt-stable (this is the fix for the perpetual `cargo fmt` churn).
fn rustfmt(src: &str) -> anyhow::Result<String> {
    use std::io::Write;
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2021", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn rustfmt (is it installed?): {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(src.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        anyhow::bail!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?)
}

fn check_file(path: &Path, expected: &[u8], stale: &mut Vec<String>) {
    let matches = std::fs::read(path)
        .map(|got| got == expected)
        .unwrap_or(false);
    if !matches {
        stale.push(path.display().to_string());
    }
}

/// Top-level docs that must not name a `HIPFIRE_*` var absent from the table.
fn coverage_gaps(root: &Path, markdown: &str) -> Vec<(String, String)> {
    let reference_docs = ["AGENTS.md", "README.md", ".github/CONTRIBUTING.md"];
    let var_re = Regex::new(r"\bHIPFIRE_[A-Z0-9_]+\b").unwrap();
    let mut missing = Vec::new();
    for rel in reference_docs {
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        let mut names: Vec<String> = var_re
            .find_iter(&text)
            .map(|m| m.as_str().to_string())
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            if !markdown.contains(&name) {
                missing.push((rel.to_string(), name));
            }
        }
    }
    missing
}
