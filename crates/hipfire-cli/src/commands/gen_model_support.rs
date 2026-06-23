// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire gen-model-support` (hidden) — render the model-support matrix from
//! its single source of truth `docs/model-support.toml` into the two committed
//! artifacts:
//!   * `crates/hipfire-model/src/model_support_generated.rs` — the compiled
//!     `ARCH_ROWS` / `QUANT_TABLE` / `GATE_TABLE` that `arch_features` and
//!     serving admission consume.
//!   * the matrix chart spliced into `MODEL-SUPPORT.md` between the generated
//!     markers (the surrounding hand-written prose is preserved).
//!
//! With `--check` it regenerates to memory and diffs against the committed files,
//! exiting non-zero on drift — the freshness gate `tests/no-gpu-ci.sh` runs.
//! This kills the hand-written `arch_features` ↔ MODEL-SUPPORT.md drift
//! structurally (same governance pattern as `gen-env-docs`).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

#[derive(Debug, clap::Args)]
pub struct GenModelSupportArgs {
    /// Canonical source matrix (repo-relative).
    #[arg(long, default_value = "docs/model-support.toml")]
    pub source: String,
    /// Generated Rust tables module (repo-relative).
    #[arg(
        long,
        default_value = "crates/hipfire-model/src/model_support_generated.rs"
    )]
    pub rust_module: String,
    /// Markdown doc whose generated section is rewritten (repo-relative).
    #[arg(long, default_value = "MODEL-SUPPORT.md")]
    pub doc: String,
    /// Verify committed artifacts match the source without writing; exit
    /// non-zero on drift (for CI).
    #[arg(long)]
    pub check: bool,
}

// ── Source schema (docs/model-support.toml) ────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Spec {
    #[serde(default)]
    arch: Vec<ArchEntry>,
    #[serde(default)]
    quant: Vec<QuantEntry>,
    #[serde(default)]
    gate: Vec<GateEntry>,
}

#[derive(Debug, Deserialize)]
struct ArchEntry {
    ids: Vec<u32>,
    label: String,
    prefill: String,
    dflash: String,
    mtp: String,
    kv: String,
    vision: String,
}

#[derive(Debug, Deserialize)]
struct QuantEntry {
    name: String,
    label: String,
    weight_bits: u32,
    act_bits: u32,
    status: String,
}

#[derive(Debug, Deserialize)]
struct GateEntry {
    arch: u32,
    quant: String,
    feature: String,
    support: String,
    note: String,
}

const BEGIN_MARK: &str = "<!-- BEGIN GENERATED model-support (source: docs/model-support.toml — run `cargo run -p hipfire-cli -- gen-model-support`) -->";
const END_MARK: &str = "<!-- END GENERATED model-support -->";

pub fn run(args: GenModelSupportArgs) -> anyhow::Result<()> {
    let root = repo_root()?;
    let source_path = root.join(&args.source);
    let raw = std::fs::read_to_string(&source_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", source_path.display()))?;
    let spec: Spec = toml::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", source_path.display()))?;
    validate(&spec)?;

    let rust = rustfmt(&render_rust(&spec))?;
    let chart = render_chart(&spec);

    let rust_path = root.join(&args.rust_module);
    let doc_path = root.join(&args.doc);
    let existing_doc = std::fs::read_to_string(&doc_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", doc_path.display()))?;
    let new_doc = splice_section(&existing_doc, &chart).ok_or_else(|| {
        anyhow::anyhow!(
            "{} is missing the generated markers ({BEGIN_MARK} … {END_MARK})",
            doc_path.display()
        )
    })?;

    if args.check {
        let mut stale = Vec::new();
        check_file(&rust_path, rust.as_bytes(), &mut stale);
        check_file(&doc_path, new_doc.as_bytes(), &mut stale);
        if !stale.is_empty() {
            anyhow::bail!(
                "model-support artifacts are stale ({} file(s)): {}\n\
                 regenerate with `cargo run -p hipfire-cli -- gen-model-support` and commit.",
                stale.len(),
                stale.join(", ")
            );
        }
        eprintln!("gen-model-support: artifacts are up to date");
        return Ok(());
    }

    std::fs::write(&rust_path, rust.as_bytes())?;
    eprintln!("gen-model-support: wrote {}", rust_path.display());
    std::fs::write(&doc_path, new_doc.as_bytes())?;
    eprintln!("gen-model-support: wrote {}", doc_path.display());
    Ok(())
}

/// Reject malformed support marks early with a clear message (otherwise a typo
/// would silently render as Unknown).
fn validate(spec: &Spec) -> anyhow::Result<()> {
    let ok = |m: &str| matches!(m, "full" | "partial" | "none");
    for a in &spec.arch {
        for (field, v) in [
            ("prefill", &a.prefill),
            ("dflash", &a.dflash),
            ("mtp", &a.mtp),
            ("vision", &a.vision),
        ] {
            if !ok(v) {
                anyhow::bail!(
                    "arch {:?} field `{field}` = {v:?}; expected full|partial|none",
                    a.ids
                );
            }
        }
    }
    for g in &spec.gate {
        if !ok(&g.support) {
            anyhow::bail!(
                "gate arch={} quant={} support={:?}; expected full|partial|none",
                g.arch,
                g.quant,
                g.support
            );
        }
    }
    Ok(())
}

fn support_variant(m: &str) -> &'static str {
    match m {
        "full" => "FeatureSupport::Full",
        "partial" => "FeatureSupport::Partial",
        "none" => "FeatureSupport::None",
        _ => "FeatureSupport::Unknown",
    }
}

fn support_glyph(m: &str) -> &'static str {
    match m {
        "full" => "✅",
        "partial" => "🟡",
        "none" => "❌",
        _ => "?",
    }
}

fn render_rust(spec: &Spec) -> String {
    let mut s = String::new();
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str("// @generated by `hipfire gen-model-support` from docs/model-support.toml.\n");
    s.push_str("// DO NOT EDIT BY HAND — edit the .toml and regenerate.\n");
    s.push_str("#![allow(dead_code)]\n\n");
    s.push_str("use crate::{ArchFeatures, FeatureSupport};\n\n");

    // Arch table.
    s.push_str("/// One row of the per-arch capability matrix.\n");
    s.push_str("pub struct ArchRow {\n    pub ids: &'static [u32],\n    pub features: ArchFeatures,\n}\n\n");
    s.push_str("/// Per-arch capabilities, keyed by HFQ arch_id (see `arch_features`).\n");
    s.push_str("pub const ARCH_ROWS: &[ArchRow] = &[\n");
    for a in &spec.arch {
        let ids = a
            .ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            "    ArchRow {{ ids: &[{ids}], features: ArchFeatures {{ label: {:?}, prefill: {}, dflash: {}, mtp: {}, kv: {:?}, vision: {} }} }},\n",
            a.label,
            support_variant(&a.prefill),
            support_variant(&a.dflash),
            support_variant(&a.mtp),
            a.kv,
            support_variant(&a.vision),
        ));
    }
    s.push_str("];\n\n");

    // Quant table.
    s.push_str("/// A quant format and its weight/activation bit-width + maturity.\n");
    s.push_str("pub struct QuantInfo {\n    pub name: &'static str,\n    pub label: &'static str,\n    pub weight_bits: u32,\n    pub act_bits: u32,\n    pub status: &'static str,\n}\n\n");
    s.push_str("pub const QUANT_TABLE: &[QuantInfo] = &[\n");
    for q in &spec.quant {
        s.push_str(&format!(
            "    QuantInfo {{ name: {:?}, label: {:?}, weight_bits: {}, act_bits: {}, status: {:?} }},\n",
            q.name, q.label, q.weight_bits, q.act_bits, q.status,
        ));
    }
    s.push_str("];\n\n");

    // Gate table.
    s.push_str("/// An intentional (arch × quant × feature) override of the arch-level\n");
    s.push_str("/// capability — the per-quant axis the arch matrix can't express.\n");
    s.push_str("pub struct GateRow {\n    pub arch: u32,\n    pub quant: &'static str,\n    pub feature: &'static str,\n    pub support: FeatureSupport,\n    pub note: &'static str,\n}\n\n");
    s.push_str("pub const GATE_TABLE: &[GateRow] = &[\n");
    for g in &spec.gate {
        s.push_str(&format!(
            "    GateRow {{ arch: {}, quant: {:?}, feature: {:?}, support: {}, note: {:?} }},\n",
            g.arch,
            g.quant,
            g.feature,
            support_variant(&g.support),
            g.note,
        ));
    }
    s.push_str("];\n");
    s
}

fn render_chart(spec: &Spec) -> String {
    let mut s = String::new();
    s.push_str("### Capability matrix (generated)\n\n");
    s.push_str("Machine-readable subset consumed by `arch_features` / admission. Edit `docs/model-support.toml`.\n\n");
    s.push_str(
        "| Arch (arch_id) | Batched prefill | DFlash spec | MTP spec | KV quant | Vision |\n",
    );
    s.push_str("|---|---|---|---|---|---|\n");
    for a in &spec.arch {
        let ids = a
            .ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            "| {} ({}) | {} | {} | {} | {} | {} |\n",
            a.label,
            ids,
            support_glyph(&a.prefill),
            support_glyph(&a.dflash),
            support_glyph(&a.mtp),
            a.kv,
            support_glyph(&a.vision),
        ));
    }

    s.push_str("\n### Quant formats (generated)\n\n");
    s.push_str("| Quant | Weight bits | Act bits | Status |\n");
    s.push_str("|---|---|---|---|\n");
    for q in &spec.quant {
        s.push_str(&format!(
            "| {} ({}) | {} | {} | {} |\n",
            q.name, q.label, q.weight_bits, q.act_bits, q.status,
        ));
    }

    s.push_str("\n### Intentional gates (generated)\n\n");
    s.push_str("Per-quant overrides of an arch capability (admission consults these before green-lighting).\n\n");
    s.push_str("| Arch | Quant | Feature | Support | Note |\n");
    s.push_str("|---|---|---|---|---|\n");
    for g in &spec.gate {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            g.arch,
            g.quant,
            g.feature,
            support_glyph(&g.support),
            g.note,
        ));
    }
    s
}

/// Replace the content between the generated markers (exclusive), preserving the
/// markers and all surrounding prose. `None` if either marker is absent.
fn splice_section(existing: &str, inner: &str) -> Option<String> {
    let begin = existing.find(BEGIN_MARK)?;
    let after_begin = begin + BEGIN_MARK.len();
    let end_rel = existing[after_begin..].find(END_MARK)?;
    let end = after_begin + end_rel;
    Some(format!(
        "{}\n\n{}\n{}",
        &existing[..after_begin],
        inner.trim_end(),
        &existing[end..]
    ))
}

fn check_file(path: &Path, expected: &[u8], stale: &mut Vec<String>) {
    let matches = std::fs::read(path)
        .map(|got| got == expected)
        .unwrap_or(false);
    if !matches {
        stale.push(path.display().to_string());
    }
}

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
