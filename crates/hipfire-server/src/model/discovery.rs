use std::path::{Path, PathBuf};

use hipfire_config::{hipfire_dir, models_dir};
use hipfire_model::{
    build_local_llm_registry, find_model_in_roots, list_local_models_in, LlmModelRegistry,
};

/// Resolve a model identifier to an absolute file path for **network callers**.
///
/// Unlike the local CLI resolver, this is confined to a fixed set of read-only
/// roots — `~/.hipfire/models` plus the admin-configured `network_dir` (e.g. an
/// NFS share) when set. It never honors an arbitrary absolute path or a
/// `..`-escaping identifier from the request, and canonicalizes every result to
/// confirm it stays inside a root. Within each root it resolves, in order:
/// exact name, `<arg>.hfq`, normalized variants, then a quant-ranked fuzzy
/// scan; admin `~/.hipfire/models.json` aliases are honored only when their
/// target also stays inside a root.
pub fn find_model(arg: &str, network_dir: Option<&Path>) -> Option<PathBuf> {
    let mut roots = vec![models_dir()];
    if let Some(dir) = network_dir {
        roots.push(dir.to_path_buf());
    }
    let aliases = hipfire_dir().join("models.json");
    find_model_in_roots(arg, &roots, Some(&aliases))
}

/// List all non-sidecar .hfq files in the models directory.
pub fn list_local_models() -> Vec<PathBuf> {
    list_local_models_in(&models_dir())
}

pub fn local_llm_registry() -> LlmModelRegistry {
    build_local_llm_registry()
}
