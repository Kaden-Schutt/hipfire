use std::path::PathBuf;

use hipfire_config::{hipfire_dir, models_dir};
use hipfire_model::find_model_in;

pub fn find_model(arg: &str) -> Option<PathBuf> {
    let mdir = models_dir();
    let aliases = hipfire_dir().join("models.json");
    find_model_in(arg, &mdir, Some(&aliases))
}
