use std::path::PathBuf;

use hipfire_config::{hipfire_dir, models_dir};
pub use hipfire_model::model_display_name;
use hipfire_model::{find_model_in, list_local_models_in};

pub fn find_model(arg: &str) -> Option<PathBuf> {
    let mdir = models_dir();
    let aliases = hipfire_dir().join("models.json");
    find_model_in(arg, &mdir, Some(&aliases))
}

pub fn list_local_models() -> Vec<PathBuf> {
    list_local_models_in(&models_dir())
}
