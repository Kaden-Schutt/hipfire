// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Layered config resolution with provenance.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::{ConfigField, Requirement};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigLayerKind {
    CompiledDefault,
    Global,
    Profile,
    Host,
    Node,
    Pool,
    Model,
    ModelHost,
    Environment,
    Cli,
    Request,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigLayer {
    pub kind: ConfigLayerKind,
    pub id: Option<String>,
    pub values: BTreeMap<String, Value>,
}

impl ConfigLayer {
    pub fn new(kind: ConfigLayerKind) -> Self {
        Self {
            kind,
            id: None,
            values: BTreeMap::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_value(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    pub fn from_json_object(
        kind: ConfigLayerKind,
        id: Option<impl Into<String>>,
        value: &Value,
    ) -> Option<Self> {
        let object = value.as_object()?;
        let mut layer = ConfigLayer::new(kind);
        layer.id = id.map(Into::into);
        layer.values = object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Some(layer)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct ConfigValueSource {
    pub kind: ConfigLayerKind,
    pub id: Option<String>,
}

impl ConfigValueSource {
    fn from_layer(layer: &ConfigLayer) -> Self {
        Self {
            kind: layer.kind,
            id: layer.id.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedConfigValue {
    pub key: String,
    pub value: Option<Value>,
    pub source: Option<ConfigValueSource>,
    pub overrode: Vec<ConfigValueSource>,
    pub missing_required: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnknownConfigKey {
    pub key: String,
    pub source: ConfigValueSource,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigResolution {
    pub values: Vec<ResolvedConfigValue>,
    pub unknown_keys: Vec<UnknownConfigKey>,
}

pub fn resolve_config_layers(fields: &[ConfigField], layers: &[ConfigLayer]) -> ConfigResolution {
    let known = fields
        .iter()
        .map(|field| field.key)
        .collect::<BTreeSet<_>>();
    let mut unknown_keys = Vec::new();
    for layer in layers {
        for key in layer.values.keys() {
            if !known.contains(key.as_str()) {
                unknown_keys.push(UnknownConfigKey {
                    key: key.clone(),
                    source: ConfigValueSource::from_layer(layer),
                });
            }
        }
    }

    let mut values = fields
        .iter()
        .map(|field| resolve_field(field, layers))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.key.cmp(&b.key));
    unknown_keys.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.source.kind.cmp(&b.source.kind))
            .then_with(|| a.source.id.cmp(&b.source.id))
    });

    ConfigResolution {
        values,
        unknown_keys,
    }
}

pub fn config_layers_from_document(raw: &Value, model_tag: Option<&str>) -> Vec<ConfigLayer> {
    let Some(object) = raw.as_object() else {
        return Vec::new();
    };

    let mut global = ConfigLayer::new(ConfigLayerKind::Global);
    for (key, value) in object {
        if key != "model_overrides" {
            global.values.insert(key.clone(), value.clone());
        }
    }

    let mut layers = Vec::new();
    if !global.values.is_empty() {
        layers.push(global);
    }

    if let Some(tag) = model_tag {
        if let Some(model_values) = object
            .get("model_overrides")
            .and_then(Value::as_object)
            .and_then(|overrides| overrides.get(tag))
        {
            if let Some(layer) =
                ConfigLayer::from_json_object(ConfigLayerKind::Model, Some(tag), model_values)
            {
                if !layer.values.is_empty() {
                    layers.push(layer);
                }
            }
        }
    }

    layers
}

fn resolve_field(field: &ConfigField, layers: &[ConfigLayer]) -> ResolvedConfigValue {
    let mut value = field.default.map(parse_default_value);
    let mut source = value.as_ref().map(|_| ConfigValueSource {
        kind: ConfigLayerKind::CompiledDefault,
        id: None,
    });
    let mut overrode = Vec::new();

    for layer in layers {
        if let Some(next) = layer.values.get(field.key) {
            if let Some(prev) = source.take() {
                overrode.push(prev);
            }
            value = Some(next.clone());
            source = Some(ConfigValueSource::from_layer(layer));
        }
    }

    let missing_required = matches!(field.requirement, Requirement::Required) && source.is_none();

    ResolvedConfigValue {
        key: field.key.to_string(),
        value,
        source,
        overrode,
        missing_required,
    }
}

fn parse_default_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{resolve_config_layers, ConfigLayer, ConfigLayerKind};
    use crate::schema::config_schema;

    fn value<'a>(
        resolution: &'a super::ConfigResolution,
        key: &str,
    ) -> &'a super::ResolvedConfigValue {
        resolution
            .values
            .iter()
            .find(|value| value.key == key)
            .expect("resolved key")
    }

    #[test]
    fn resolves_defaults_without_layers() {
        let resolution = resolve_config_layers(config_schema(), &[]);
        let max_tokens = value(&resolution, "max_tokens");

        assert_eq!(max_tokens.value, Some(json!(512)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::CompiledDefault)
        );
        assert!(max_tokens.overrode.is_empty());
        assert!(resolution.unknown_keys.is_empty());
    }

    #[test]
    fn higher_layers_override_lower_layers_with_provenance() {
        let global = ConfigLayer::new(ConfigLayerKind::Global).with_value("max_tokens", 256);
        let model = ConfigLayer::new(ConfigLayerKind::Model)
            .with_id("qwen3.5:9b")
            .with_value("max_tokens", 1024);
        let request = ConfigLayer::new(ConfigLayerKind::Request).with_value("max_tokens", 64);

        let resolution = resolve_config_layers(config_schema(), &[global, model, request]);
        let max_tokens = value(&resolution, "max_tokens");

        assert_eq!(max_tokens.value, Some(json!(64)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Request)
        );
        assert_eq!(
            max_tokens
                .overrode
                .iter()
                .map(|source| source.kind)
                .collect::<Vec<_>>(),
            vec![
                ConfigLayerKind::CompiledDefault,
                ConfigLayerKind::Global,
                ConfigLayerKind::Model,
            ]
        );
    }

    #[test]
    fn reports_unknown_layer_keys() {
        let layer = ConfigLayer::new(ConfigLayerKind::Host)
            .with_id("strix-halo-01")
            .with_value("vision.max_cores", 6);

        let resolution = resolve_config_layers(config_schema(), &[layer]);

        assert_eq!(resolution.unknown_keys.len(), 1);
        assert_eq!(resolution.unknown_keys[0].key, "vision.max_cores");
        assert_eq!(
            resolution.unknown_keys[0].source.kind,
            ConfigLayerKind::Host
        );
        assert_eq!(
            resolution.unknown_keys[0].source.id.as_deref(),
            Some("strix-halo-01")
        );
    }

    #[test]
    fn builds_global_and_model_layers_from_config_document() {
        let raw = json!({
            "max_tokens": 256,
            "temperature": 0.4,
            "model_overrides": {
                "qwen3.5:9b": {
                    "temperature": 0.1,
                    "kv_cache": "q8"
                },
                "other": {
                    "temperature": 0.8
                }
            }
        });

        let layers = super::config_layers_from_document(&raw, Some("qwen3.5:9b"));
        let resolution = resolve_config_layers(config_schema(), &layers);

        let temperature = value(&resolution, "temperature");
        assert_eq!(temperature.value, Some(json!(0.1)));
        assert_eq!(
            temperature.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Model)
        );
        assert_eq!(
            temperature
                .source
                .as_ref()
                .and_then(|source| source.id.as_deref()),
            Some("qwen3.5:9b")
        );

        let max_tokens = value(&resolution, "max_tokens");
        assert_eq!(max_tokens.value, Some(json!(256)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Global)
        );

        let model_overrides = value(&resolution, "model_overrides");
        assert_eq!(model_overrides.value, Some(json!({})));
        assert_eq!(
            model_overrides.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::CompiledDefault)
        );
    }
}
