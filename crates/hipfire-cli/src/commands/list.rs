// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire list` — local models plus, per model, the on-disk artifacts (chat
//! template, MTP / DFlash / TriAttention sidecars, Hessian/calibration) and the
//! per-arch feature status. The feature columns mirror MODEL-SUPPORT.md (the
//! canonical support matrix); this command reports what's *present on disk* and
//! the *arch capability*, not a live load (no GPU touched).

use std::path::Path;

use crate::model::list_local_models;
use hipfire_model::{model_display_name, read_hfq_metadata};

/// Per-arch feature status, keyed by HFQ arch_id, mirroring MODEL-SUPPORT.md.
/// Marks: "y" full · "~" partial/limited · "-" not implemented/applicable.
struct ArchFeat {
    label: &'static str,
    prefill: &'static str, // batched prefill
    dflash: &'static str,
    mtp: &'static str,
    kv: &'static str, // KV-quant menu
    vision: &'static str,
}

fn arch_feat(arch_id: u32) -> ArchFeat {
    // Source of truth: MODEL-SUPPORT.md feature matrix. Keep in sync when that
    // matrix changes.
    match arch_id {
        5 | 6 => ArchFeat { label: "qwen3.5", prefill: "y", dflash: "y", mtp: "y", kv: "full", vision: "~" },
        9 => ArchFeat { label: "deepseek4", prefill: "y", dflash: "-", mtp: "~", kv: "fp32", vision: "-" },
        10 => ArchFeat { label: "minimax", prefill: "~", dflash: "-", mtp: "~", kv: "fp32", vision: "-" },
        11 => ArchFeat { label: "lfm2-moe", prefill: "~", dflash: "-", mtp: "-", kv: "fp32", vision: "-" },
        12 => ArchFeat { label: "gemma3", prefill: "y", dflash: "-", mtp: "-", kv: "fp32+q8", vision: "-" },
        13 => ArchFeat { label: "gemma3-vl", prefill: "y", dflash: "-", mtp: "-", kv: "fp32+q8", vision: "y" },
        7 => ArchFeat { label: "qwen2", prefill: "y", dflash: "-", mtp: "-", kv: "fp32", vision: "-" },
        8 => ArchFeat { label: "dots-ocr", prefill: "y", dflash: "-", mtp: "-", kv: "fp32", vision: "y" },
        0 | 1 => ArchFeat { label: "llama", prefill: "~", dflash: "-", mtp: "-", kv: "fp32", vision: "-" },
        _ => ArchFeat { label: "unknown", prefill: "?", dflash: "?", mtp: "?", kv: "?", vision: "?" },
    }
}

/// Quant/format token of the display name. Scans `-`-delimited segments for a
/// known format prefix (mq4, oq4, qtip3, q8, bf16, …) so calibration modifiers
/// that trail the format (e.g. `oq4-ldlq`) don't mask it; falls back to the last
/// segment. A bundled `+feature` suffix is stripped (shown in ARTIFACTS instead).
fn quant_token(display: &str) -> String {
    const FORMATS: &[&str] = &[
        "bf16", "fp16", "f16", "q8", "mq2", "mq3", "mq4", "mq6", "mq8", "oq4", "oq8", "qtip2",
        "qtip3", "iu8", "w4a8", "w8a8",
    ];
    let mut best: Option<&str> = None;
    for seg in display.split('-') {
        let head = seg.split('+').next().unwrap_or(seg);
        let low = head.to_ascii_lowercase();
        if FORMATS.iter().any(|f| low.starts_with(f)) {
            best = Some(head);
        }
    }
    best.map(str::to_string).unwrap_or_else(|| {
        display
            .rsplit('-')
            .next()
            .unwrap_or(display)
            .split('+')
            .next()
            .unwrap_or(display)
            .to_string()
    })
}

/// Sidecar/template presence for one primary model path.
struct Artifacts {
    template: bool,
    mtp: bool,     // sidecar file or bundled (+mtp)
    dflash: bool,  // sidecar or bundled
    triattn: bool, // sidecar
    hessian: bool, // <base>.calib.hfq
}

fn detect_artifacts(path: &Path, display: &str) -> Artifacts {
    // Sidecar siblings: <base>.<role>.hfq (base = path with .hfq stripped).
    let full = path.to_string_lossy();
    let base = full.strip_suffix(".hfq").unwrap_or(&full);
    let sib = |role: &str| Path::new(&format!("{base}.{role}.hfq")).exists();
    // Bundled features ride the `+feature` filename tokens (mq4+mtp, …).
    let bundled: Vec<&str> = display.split('+').skip(1).collect();
    let has_bundled = |f: &str| bundled.iter().any(|b| b.contains(f));

    // Chat template lives in tokenizer_config.chat_template (HF convention);
    // some artifacts stash it top-level. Present + non-empty either place.
    let template = read_hfq_metadata(path)
        .ok()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(&m.metadata_json).ok())
        .map(|v| {
            let nonempty = |t: Option<&serde_json::Value>| {
                t.and_then(|x| x.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false)
            };
            nonempty(v.get("chat_template"))
                || nonempty(v.get("tokenizer_config").and_then(|tc| tc.get("chat_template")))
        })
        .unwrap_or(false);

    Artifacts {
        template,
        mtp: sib("mtp") || has_bundled("mtp"),
        dflash: sib("dflash") || has_bundled("dflash"),
        triattn: sib("triattn"),
        hessian: sib("calib"),
    }
}

/// Render a boolean as a tick/cross glyph.
fn yn(v: bool) -> &'static str {
    if v {
        "✓"
    } else {
        "·"
    }
}

/// Render a tri-state arch-feature mark ("y"/"~"/"-"/"?") as a glyph.
fn tri(m: &str) -> &'static str {
    match m {
        "y" => "✓",
        "~" => "~",
        "?" => "?",
        _ => "·",
    }
}

struct Row {
    name: String,
    quant: String,
    arch: String,
    art: Artifacts,
    feat: Option<ArchFeat>,
}

pub fn run() {
    let models = list_local_models();
    if models.is_empty() {
        println!("No models found in ~/.hipfire/models/");
        return;
    }

    let mut rows: Vec<Row> = Vec::with_capacity(models.len());
    for p in &models {
        let name = model_display_name(p);
        let arch_id = read_hfq_metadata(p).ok().map(|m| m.arch_id);
        let art = detect_artifacts(p, &name);
        rows.push(Row {
            quant: quant_token(&name),
            arch: arch_id.map_or_else(|| "?".to_string(), |id| format!("{} ({id})", arch_feat(id).label)),
            feat: arch_id.map(arch_feat),
            name,
            art,
        });
    }

    let name_w = rows.iter().map(|r| r.name.len()).max().unwrap_or(5).max(5);
    let quant_w = rows.iter().map(|r| r.quant.len()).max().unwrap_or(5).max(5);
    let arch_w = rows.iter().map(|r| r.arch.len()).max().unwrap_or(4).max(4);

    // Group banner over the tick columns.
    println!(
        "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:^17}   {:^21}",
        "", "", "", "─ on disk ─", "─ arch features ─"
    );
    println!(
        "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
        "MODEL", "QUANT", "ARCH", "tpl", "mtp", "dfl", "tri", "hess", "pfil", "dfl", "mtp", "kv", "vis"
    );
    for r in &rows {
        let f = r.feat.as_ref();
        println!(
            "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
            r.name,
            r.quant,
            r.arch,
            yn(r.art.template),
            yn(r.art.mtp),
            yn(r.art.dflash),
            yn(r.art.triattn),
            yn(r.art.hessian),
            f.map_or("?", |f| tri(f.prefill)),
            f.map_or("?", |f| tri(f.dflash)),
            f.map_or("?", |f| tri(f.mtp)),
            f.map_or("?", |f| f.kv),
            f.map_or("?", |f| tri(f.vision)),
        );
    }
    println!("\non disk: tpl=chat template · mtp/dfl/tri=draft/TriAttn sidecar (or bundled) · hess=Hessian/calib");
    println!("arch features (per MODEL-SUPPORT.md): ✓=full · ~=partial · ·=none · pfil=batched prefill · kv=quant menu");
}
