//! Compatibility facade for model-source contracts.
//!
//! Core model-source traits and artifact identity helpers live in
//! `hipfire-model`. Runtime still owns concrete HFQ/safetensors openers in
//! this slice, so `open_model` remains here until those loaders move.

pub use hipfire_model::{
    detect_model_artifact_format, is_role_sidecar_name, model_display_name, normalize_tag_stem,
    quant_preference_rank, ModelArtifactFormat, ModelSource, QuantConfig, TensorInfo,
    QUANT_PREFERENCE,
};

/// Open a model from a path, auto-detecting the format.
/// - If path is a directory with config.json: opens as SafetensorsSource
/// - If path ends in .hfq: opens as HfqFile
/// - Otherwise: tries HfqFile first, then directory
pub fn open_model(path: &std::path::Path) -> Result<Box<dyn ModelSource>, String> {
    if path.is_dir() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            let source = crate::safetensors_source::SafetensorsSource::open(path)
                .map_err(|e| format!("safetensors open failed: {e}"))?;
            Ok(Box::new(source))
        } else {
            Err(format!("{}: directory has no config.json", path.display()))
        }
    } else {
        let hfq = crate::hfq::HfqFile::open(path).map_err(|e| format!("{e}"))?;
        Ok(Box::new(hfq))
    }
}
