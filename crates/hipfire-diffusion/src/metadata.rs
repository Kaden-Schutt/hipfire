// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Diffusion `.hfq` metadata types — the serde-serialized container schema
//! (`DiffusionHfqMetadata` and its component structs) plus the `inspect_hfq`
//! summary type. Plain data; re-exported at the crate root (3.8 Part 2 split).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionHfqMetadata {
    pub artifact_kind: String,
    pub schema_version: u32,
    pub pipeline: DiffusionPipelineMetadata,
    #[serde(default)]
    pub tokenizer: DiffusionTokenizerMetadata,
    #[serde(default)]
    pub tokenizer_2: Option<DiffusionTokenizerMetadata>,
    #[serde(default)]
    pub batch: DiffusionBatchMetadata,
    #[serde(default)]
    pub quantization: DiffusionQuantizationMetadata,
    #[serde(default)]
    pub components: BTreeMap<String, DiffusionComponentMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionPipelineMetadata {
    pub class_name: String,
    pub source: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub latent_channels: Option<u32>,
    #[serde(default)]
    pub latent_height: Option<u32>,
    #[serde(default)]
    pub latent_width: Option<u32>,
    #[serde(default)]
    pub supported_widths: Vec<u32>,
    #[serde(default)]
    pub supported_heights: Vec<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionTokenizerMetadata {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub max_length: Option<u32>,
    #[serde(default)]
    pub entries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionBatchMetadata {
    pub max_batch: u32,
    pub batched_runtime: bool,
}

impl Default for DiffusionBatchMetadata {
    fn default() -> Self {
        Self {
            max_batch: 1,
            batched_runtime: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionQuantizationMetadata {
    pub weight_format: String,
    pub activation_format: String,
    pub tensor_roles_version: u32,
}

impl Default for DiffusionQuantizationMetadata {
    fn default() -> Self {
        Self {
            weight_format: "source".to_string(),
            activation_format: "fp16".to_string(),
            tensor_roles_version: 1,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionComponentMetadata {
    #[serde(default)]
    pub class_name: Option<String>,
    #[serde(default)]
    pub config_entry: Option<String>,
    #[serde(default)]
    pub weight_entries: Vec<String>,
    #[serde(default)]
    pub tensor_roles: Vec<DiffusionTensorRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffusionTensorRole {
    pub role: String,
    pub entry: String,
    pub dtype: String,
    #[serde(default)]
    pub quant_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionModelSummary {
    pub path: PathBuf,
    pub title: String,
    pub model_name: String,
    pub pipeline_class: String,
    pub max_batch: u32,
    pub weight_format: String,
}
