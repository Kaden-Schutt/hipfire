//! Per-layer per-expert hit-count tracking for MoE expert-offload Phase 0.
//!
//! Gated on `HIPFIRE_MOE_EXPERT_HEATMAP=1`. When active, every routed-expert
//! decision (top-K index per token per layer) increments a counter cell in
//! a `[n_layers, n_experts]` matrix. On unload, the matrix is dumped to
//! disk for offline analysis (hit-rate at LRU sizes 8/16/32/64/128).
//!
//! See `docs/plans/moe-egpu-offload.prd` § Phase 0 for the decision gate
//! that this profile feeds (proceed if 32-cache covers ≥80% of decisions).

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

pub struct MoEHeatmap {
    counts: Vec<u64>,
    n_layers: usize,
    n_experts: usize,
    tokens_seen: u64,
    routed_decisions: u64,
    model_name: String,
}

static HEATMAP: OnceLock<Mutex<Option<MoEHeatmap>>> = OnceLock::new();

pub fn enabled() -> bool {
    matches!(std::env::var("HIPFIRE_MOE_EXPERT_HEATMAP").as_deref(), Ok("1"))
}

/// Initialize the singleton. No-op if env var unset. Replaces any prior
/// state — the typical path is daemon load → init → record many → unload
/// → dump_and_clear.
pub fn init(n_layers: usize, n_experts: usize, model_name: String) {
    if !enabled() {
        return;
    }
    let cell = HEATMAP.get_or_init(|| Mutex::new(None));
    let mut g = cell.lock().expect("moe_heatmap mutex poisoned");
    *g = Some(MoEHeatmap {
        counts: vec![0; n_layers * n_experts],
        n_layers,
        n_experts,
        tokens_seen: 0,
        routed_decisions: 0,
        model_name,
    });
    eprintln!(
        "[moe-heatmap] enabled: n_layers={n_layers} n_experts={n_experts}"
    );
}

pub fn is_active() -> bool {
    HEATMAP
        .get()
        .and_then(|c| c.lock().ok().map(|g| g.is_some()))
        .unwrap_or(false)
}

/// Record a single-token decode decision: K expert indices for one layer.
pub fn record_decode(layer_idx: usize, indices: &[i32]) {
    let Some(cell) = HEATMAP.get() else { return };
    let Ok(mut g) = cell.lock() else { return };
    let Some(h) = g.as_mut() else { return };
    if layer_idx >= h.n_layers {
        return;
    }
    let base = layer_idx * h.n_experts;
    for &idx in indices {
        if idx < 0 {
            continue;
        }
        let i = idx as usize;
        if i < h.n_experts {
            h.counts[base + i] += 1;
            h.routed_decisions += 1;
        }
    }
    if layer_idx == 0 {
        h.tokens_seen += 1;
    }
}

/// Record a prefill batch: `[n_tokens × k_top]` row-major i32 indices.
pub fn record_prefill(layer_idx: usize, indices: &[i32], n_tokens: usize, k_top: usize) {
    let Some(cell) = HEATMAP.get() else { return };
    let Ok(mut g) = cell.lock() else { return };
    let Some(h) = g.as_mut() else { return };
    if layer_idx >= h.n_layers {
        return;
    }
    if indices.len() < n_tokens * k_top {
        return;
    }
    let base = layer_idx * h.n_experts;
    for token in 0..n_tokens {
        for k in 0..k_top {
            let idx = indices[token * k_top + k];
            if idx < 0 {
                continue;
            }
            let i = idx as usize;
            if i < h.n_experts {
                h.counts[base + i] += 1;
                h.routed_decisions += 1;
            }
        }
    }
    if layer_idx == 0 {
        h.tokens_seen += n_tokens as u64;
    }
}

/// Dump the matrix to disk and clear state. Returns the written path.
pub fn dump_and_clear() -> Option<PathBuf> {
    let cell = HEATMAP.get()?;
    let mut g = cell.lock().ok()?;
    let h = g.take()?;

    let dir = std::env::var("HIPFIRE_MOE_HEATMAP_DIR")
        .unwrap_or_else(|_| "/tmp/hipfire-moe-debug/dumps".to_string());
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[moe-heatmap] failed to create {dir}: {e}");
        return None;
    }

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let model_safe: String = h
        .model_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let path = std::path::Path::new(&dir).join(format!("heatmap-{model_safe}-{ts}.csv"));

    let mut buf = String::new();
    buf.push_str(&format!(
        "# n_layers={} n_experts={} tokens_seen={} routed_decisions={} model={}\n",
        h.n_layers, h.n_experts, h.tokens_seen, h.routed_decisions, h.model_name
    ));
    buf.push_str("layer,expert,count\n");
    for layer in 0..h.n_layers {
        let base = layer * h.n_experts;
        for expert in 0..h.n_experts {
            let c = h.counts[base + expert];
            if c > 0 {
                buf.push_str(&format!("{layer},{expert},{c}\n"));
            }
        }
    }
    if let Err(e) = std::fs::write(&path, buf) {
        eprintln!("[moe-heatmap] failed to write {path:?}: {e}");
        return None;
    }
    eprintln!(
        "[moe-heatmap] wrote {} ({} tokens, {} routed decisions)",
        path.display(),
        h.tokens_seen,
        h.routed_decisions
    );
    Some(path)
}
