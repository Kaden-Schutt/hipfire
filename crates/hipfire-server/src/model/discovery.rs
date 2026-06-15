use std::path::PathBuf;

use crate::config::{hipfire_dir, models_dir};
use hipfire_model::find_model_in;
pub use hipfire_model::{list_local_models_in, model_display_name};

/// Resolve a model identifier to an absolute file path.
///
/// Resolution order (mirrors Bun CLI findModel):
/// 1. Direct file path — if the string exists on disk, use it as-is.
/// 2. `~/.hipfire/models/<arg>` — if that exists.
/// 3. `~/.hipfire/models/<arg>.hfq` — bare name + extension.
/// 4. User aliases from `~/.hipfire/models.json`.
/// 5. Fuzzy scan of `~/.hipfire/models/` — walks one level, ranks by quant preference.
pub fn find_model(arg: &str) -> Option<PathBuf> {
    let mdir = models_dir();
    let aliases = hipfire_dir().join("models.json");
    find_model_in(arg, &mdir, Some(&aliases))
}

/// List all non-sidecar .hfq files in the models directory.
pub fn list_local_models() -> Vec<PathBuf> {
    list_local_models_in(&models_dir())
}
