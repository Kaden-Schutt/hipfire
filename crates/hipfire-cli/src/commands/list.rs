// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire list` — local models as a capability/artifact matrix. Pure
//! presentation: all detection lives in `hipfire_model::model_card` (shared with
//! serving admission); this command only renders the cards (no GPU touched).

use crate::model::list_local_models;
use hipfire_model::{model_card, ArchFeatures, FeatureSupport, ModelCard, Sidecars};

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

pub fn run() {
    let models = list_local_models();
    if models.is_empty() {
        println!("No models found in ~/.hipfire/models/");
        return;
    }

    let cards: Vec<ModelCard> = models.iter().map(|p| model_card(p)).collect();
    let arch_cells: Vec<String> = cards.iter().map(arch_cell).collect();

    let name_w = cards.iter().map(|c| c.name.len()).max().unwrap_or(5).max(5);
    let quant_w = cards
        .iter()
        .map(|c| c.quant.len())
        .max()
        .unwrap_or(5)
        .max(5);
    let arch_w = arch_cells.iter().map(|a| a.len()).max().unwrap_or(4).max(4);

    // Grouped banner over the tick columns.
    println!(
        "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:^17}   {:^21}",
        "", "", "", "─ on disk ─", "─ arch features ─"
    );
    println!(
        "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
        "MODEL", "QUANT", "ARCH", "tpl", "mtp", "dfl", "tri", "hess", "pfil", "dfl", "mtp", "kv", "vis"
    );
    for (card, arch) in cards.iter().zip(&arch_cells) {
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
            "{:<name_w$}  {:<quant_w$}  {:<arch_w$}  {:>3} {:>3} {:>3} {:>3} {:>4}   {:>4} {:>4} {:>3} {:>7} {:>3}",
            card.name,
            card.quant,
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
    println!("\non disk: tpl=chat template · mtp/dfl/tri=draft/TriAttn sidecar (or bundled) · hess=Hessian/calib");
    println!("arch features (per MODEL-SUPPORT.md): ✓=full · ~=partial · ·=none · pfil=batched prefill · kv=quant menu");
}
