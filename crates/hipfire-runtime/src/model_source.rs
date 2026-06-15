//! Compatibility facade for model-source contracts.
//!
//! Core model-source traits, artifact identity helpers, and opening policy
//! live in `hipfire-model`. Runtime still owns concrete HFQ/safetensors
//! openers in this slice, so `open_model` remains here until those loaders
//! move.

pub use hipfire_model::{
    detect_model_artifact_format, is_role_sidecar_name, model_display_name, normalize_tag_stem,
    open_model_source_with, quant_preference_rank, ModelArtifactFormat, ModelSource, QuantConfig,
    TensorInfo, QUANT_PREFERENCE,
};

/// Open a model from a path, auto-detecting the format.
/// - If path is a directory with config.json: opens as SafetensorsSource
/// - If path ends in .hfq: opens as HfqFile
/// - Otherwise: tries HfqFile first, then directory
pub fn open_model(path: &std::path::Path) -> Result<Box<dyn ModelSource>, String> {
    open_model_source_with(
        path,
        |path| {
            crate::hfq::HfqFile::open(path)
                .map(|source| Box::new(source) as Box<dyn ModelSource>)
                .map_err(|e| format!("{e}"))
        },
        |path| {
            crate::safetensors_source::SafetensorsSource::open(path)
                .map(|source| Box::new(source) as Box<dyn ModelSource>)
                .map_err(|e| format!("{e}"))
        },
    )
}
