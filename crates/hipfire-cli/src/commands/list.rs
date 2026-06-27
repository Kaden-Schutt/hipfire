// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire list` — local models as a capability/artifact matrix. Pure
//! presentation: inventory comes from the canonical local LLM registry, while
//! capability detection lives in `hipfire_model::model_card` (GPU-free).

use std::path::Path;

use hipfire_config::models_dir;
use hipfire_model::{
    build_local_llm_registry, model_card, ArchFeatures, FeatureSupport, LlmModelRegistryEntry,
    ModelCard, Sidecars,
};

/// Tick/cross glyph for an on-disk artifact.
fn yn(v: bool) -> &'static str {
    if v {
        "✓"
    } else {
        "·"
    }
}

/// Glyph for a tri-state arch-feature support level.
fn tri(s: FeatureSupport) -> &'static str {
    match s {
        FeatureSupport::Full => "✓",
        FeatureSupport::Partial => "~",
        FeatureSupport::None => "·",
        FeatureSupport::Unknown => "?",
    }
}

fn arch_cell(card: &ModelCard) -> String {
    match card.arch_id {
        Some(id) => format!("{} ({id})", card.features.label),
        None => "?".to_string(),
    }
}

fn nonempty_cell(values: &[String]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values.join(",")
    }
}

fn optional_cell(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or("-")
        .to_string()
}

struct ListRow {
    model: String,
    size: String,
    tags: String,
    features: String,
    quant: String,
    artifact_arch: String,
    card: ModelCard,
}

fn card_from_registry_entry(entry: &LlmModelRegistryEntry) -> ModelCard {
    let mut card = model_card(Path::new(&entry.path));
    card.quant = entry.quant.clone();
    card.sidecars.template |= !entry.chat_templates.is_empty();
    card.sidecars.dflash |= !entry.drafts.is_empty();
    card.sidecars.triattn |= !entry.triattn.is_empty();
    card
}

fn row_from_registry_entry(entry: &LlmModelRegistryEntry) -> ListRow {
    ListRow {
        model: entry.model.clone(),
        size: optional_cell(entry.size.as_deref()),
        tags: nonempty_cell(&entry.tags),
        features: nonempty_cell(&entry.features),
        quant: entry.quant.clone(),
        artifact_arch: optional_cell(entry.arch.as_deref()),
        card: card_from_registry_entry(entry),
    }
}

pub fn run() {
    let registry = build_local_llm_registry();
    if registry.models.is_empty() {
        println!("No valid HFQ models found in {}", models_dir().display());
        return;
    }

    let rows: Vec<ListRow> = registry
        .models
        .iter()
        .map(row_from_registry_entry)
        .collect();
    let arch_cells: Vec<String> = rows.iter().map(|row| arch_cell(&row.card)).collect();

    let model_w = rows
        .iter()
        .map(|row| row.model.len())
        .max()
        .unwrap_or(5)
        .max(5);
    let size_w = rows
        .iter()
        .map(|row| row.size.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let tags_w = rows
        .iter()
        .map(|row| row.tags.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let feat_w = rows
        .iter()
        .map(|row| row.features.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let quant_w = rows
        .iter()
        .map(|row| row.quant.len())
        .max()
        .unwrap_or(5)
        .max(5);
    let art_w = rows
        .iter()
        .map(|row| row.artifact_arch.len())
        .max()
        .unwrap_or(3)
        .max(3);
    let arch_w = arch_cells.iter().map(|a| a.len()).max().unwrap_or(4).max(4);

    // Grouped banner over the tick columns.
    println!(
        "{:<model_w$}  {:<size_w$}  {:<tags_w$}  {:<feat_w$}  {:<quant_w$}  {:<art_w$}  {:<arch_w$}  {:^17}   {:^21}",
        "", "", "", "", "", "", "", "─ on disk ─", "─ arch features ─"
    );
    println!(
        "{:<model_w$}  {:<size_w$}  {:<tags_w$}  {:<feat_w$}  {:<quant_w$}  {:<art_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
        "MODEL",
        "SIZE",
        "TAGS",
        "FEAT",
        "QUANT",
        "ART",
        "ARCH",
        "tpl",
        "mtp",
        "dfl",
        "tri",
        "hess",
        "pfil",
        "dfl",
        "mtp",
        "kv",
        "vis"
    );
    for (row, arch) in rows.iter().zip(&arch_cells) {
        let card = &row.card;
        let Sidecars {
            template,
            mtp,
            dflash,
            triattn,
            hessian,
        } = card.sidecars;
        let ArchFeatures {
            prefill,
            dflash: f_dflash,
            mtp: f_mtp,
            kv,
            vision,
            ..
        } = card.features;
        println!(
            "{:<model_w$}  {:<size_w$}  {:<tags_w$}  {:<feat_w$}  {:<quant_w$}  {:<art_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
            row.model,
            row.size,
            row.tags,
            row.features,
            row.quant,
            row.artifact_arch,
            arch,
            yn(template),
            yn(mtp),
            yn(dflash),
            yn(triattn),
            yn(hessian),
            tri(prefill),
            tri(f_dflash),
            tri(f_mtp),
            kv,
            tri(vision),
        );
    }
    println!("\nmodel columns: size/quant come from HFQ metadata or index; tags/features/artifact arch come from HFQ metadata");
    println!(
        "on disk: tpl=chat template · mtp/dfl/tri=draft/TriAttn sidecar (or bundled) · hess=Hessian/calib"
    );
    println!(
        "arch features (per MODEL-SUPPORT.md): ✓=full · ~=partial · ·=none · pfil=batched prefill · kv=quant menu"
    );
}
