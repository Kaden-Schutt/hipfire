use std::path::{Path, PathBuf};

/// One selective-requant edit (applied in SP2+; parsed & validated now).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantOverride {
    pub layer: usize,
    pub role: Role,
    /// Only meaningful for `Role::RoutedExperts`; empty ⇒ whole role at this layer.
    pub experts: Vec<u32>,
    /// Optional exact tensor-name allowlist. Empty means the whole role (subject
    /// to `experts`); non-empty narrows the override to these names only.
    pub tensors: Vec<String>,
    pub tier: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    RoutedExperts,
    SharedExpert,
    Attention,
    Router,
    LmHead,
    Embed,
}

impl Role {
    pub fn parse(s: &str) -> Result<Role, String> {
        Ok(match s {
            "routed_experts" => Role::RoutedExperts,
            "shared_expert" => Role::SharedExpert,
            "attention" => Role::Attention,
            "router" => Role::Router,
            "lm_head" => Role::LmHead,
            "embed" => Role::Embed,
            other => return Err(format!("reap: unknown role '{other}'")),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ReapPlan {
    pub model_arch: Option<String>,
    pub num_layers: usize,
    pub original_experts: usize,
    /// `keep[l][slot]` = original expert index in compact slot `slot`.
    /// `None` ⇒ no pruning (keep all `original_experts`).
    pub keep: Option<Vec<Vec<u32>>>,
    pub quant_overrides: Vec<QuantOverride>,
    pub dir: PathBuf,
}

impl ReapPlan {
    /// Returns 0 if keep is Some with an empty outer vec (cannot arise from load()).
    pub fn kept_per_layer(&self) -> usize {
        match &self.keep {
            Some(k) => k.first().map(|r| r.len()).unwrap_or(0),
            None => self.original_experts,
        }
    }

    /// Load `<dir>/reap_plan.json`, validating against the model's layer/expert
    /// counts (passed BEFORE any n_routed_experts override).
    pub fn load(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        let v = Self::read_plan_value(dir)?;

        // Cross-check the json's counts against the model's BEFORE parsing the
        // body, so a count mismatch is reported as the primary error.
        let original_experts = v["original_experts"]
            .as_u64()
            .unwrap_or(orig_experts_expected as u64) as usize;
        if original_experts != orig_experts_expected {
            return Err(format!(
                "reap: original_experts {original_experts} != model n_routed_experts {orig_experts_expected}"
            ));
        }
        let num_layers = v["num_layers"]
            .as_u64()
            .unwrap_or(num_layers_expected as u64) as usize;
        if num_layers != num_layers_expected {
            return Err(format!(
                "reap: num_layers {num_layers} != model num_hidden_layers {num_layers_expected}"
            ));
        }

        // `parse_value` defaults absent counts to 0; restore the model-derived
        // counts so a plan that omits them still validates its keep/overrides
        // against the model's dimensions (matching the pre-refactor behavior).
        let mut v = v;
        if v["original_experts"].as_u64().is_none() {
            v["original_experts"] = serde_json::json!(orig_experts_expected);
        }
        if v["num_layers"].as_u64().is_none() {
            v["num_layers"] = serde_json::json!(num_layers_expected);
        }
        Self::parse_value(&v, dir)
    }

    /// Load `<dir>/reap_plan.json` WITHOUT validating against the model's
    /// layer/expert counts. The counts (`num_layers`/`original_experts`) are
    /// taken from the json itself. Used by the quantizer, which reads the plan
    /// before it knows the model's counts. All per-override and per-keep
    /// validations (non-integer experts, experts-on-non-routed-role,
    /// index-vs-original bounds, layer-count consistency) are still enforced.
    pub fn load_unchecked(dir: &str) -> Result<Self, String> {
        let v = Self::read_plan_value(dir)?;
        Self::parse_value(&v, dir)
    }

    /// Read & json-parse `<dir>/reap_plan.json`.
    fn read_plan_value(dir: &str) -> Result<serde_json::Value, String> {
        let path = Path::new(dir).join("reap_plan.json");
        let txt =
            std::fs::read_to_string(&path).map_err(|e| format!("reap: read {path:?}: {e}"))?;
        serde_json::from_str(&txt).map_err(|e| format!("reap: parse {path:?}: {e}"))
    }

    /// Parse a `reap_plan.json` value into a `ReapPlan`, taking `num_layers`
    /// and `original_experts` FROM the json. Shared by `load` (which adds the
    /// `== expected` cross-checks first) and `load_unchecked`. Defaults to 0
    /// when a count is absent.
    fn parse_value(v: &serde_json::Value, dir: &str) -> Result<Self, String> {
        let original_experts = v["original_experts"].as_u64().unwrap_or(0) as usize;
        let num_layers = v["num_layers"].as_u64().unwrap_or(0) as usize;

        let keep = match v["keep"]["per_layer"].as_array() {
            None => None,
            Some(arr) => {
                if arr.len() != num_layers {
                    return Err(format!(
                        "reap: keep.per_layer has {} layers, model has {num_layers}",
                        arr.len()
                    ));
                }
                let kept = arr
                    .first()
                    .and_then(|r| r.as_array())
                    .map(|r| r.len())
                    .unwrap_or(0);
                // A keep map present but empty (0 kept experts) would override
                // n_routed_experts to 0 downstream → a model with no routed
                // experts (router selects from an empty set). Reject early.
                if kept == 0 {
                    return Err(
                        "reap: keep.per_layer present but keeps 0 experts (empty first layer)"
                            .to_string(),
                    );
                }
                let mut out = Vec::with_capacity(arr.len());
                for (l, row) in arr.iter().enumerate() {
                    let r = row
                        .as_array()
                        .ok_or_else(|| format!("reap: keep layer {l} not an array"))?;
                    if r.len() != kept {
                        return Err(format!(
                            "reap: keep layer {l} has {} entries, expected {kept}",
                            r.len()
                        ));
                    }
                    let mut v32 = Vec::with_capacity(kept);
                    let mut seen = std::collections::HashSet::with_capacity(kept);
                    for x in r {
                        let idx = x
                            .as_u64()
                            .ok_or_else(|| format!("reap: keep layer {l} non-integer index"))?
                            as u32;
                        if idx as usize >= original_experts {
                            return Err(format!(
                                "reap: keep layer {l} index {idx} >= original_experts {original_experts}"
                            ));
                        }
                        // A duplicate kept index would gather the same expert into
                        // two compact slots → wrong router fan-out / double-count.
                        if !seen.insert(idx) {
                            return Err(format!("reap: keep layer {l} has duplicate index {idx}"));
                        }
                        v32.push(idx);
                    }
                    out.push(v32);
                }
                Some(out)
            }
        };

        let mut quant_overrides = Vec::new();
        if let Some(arr) = v["quant_overrides"].as_array() {
            for (i, o) in arr.iter().enumerate() {
                let layer = o["layer"]
                    .as_u64()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing layer"))?
                    as usize;
                if layer >= num_layers {
                    return Err(format!(
                        "reap: quant_override[{i}] layer {layer} >= num_layers {num_layers}"
                    ));
                }
                let role = Role::parse(
                    o["role"]
                        .as_str()
                        .ok_or_else(|| format!("reap: quant_override[{i}] missing role"))?,
                )?;
                let experts: Vec<u32> = if let Some(a) = o["experts"].as_array() {
                    a.iter().enumerate().map(|(j, x)| {
                        let n = x.as_u64()
                            .ok_or_else(|| format!("reap: quant_override[{i}] experts[{j}] not an integer"))? as u32;
                        if n as usize >= original_experts {
                            return Err(format!("reap: quant_override[{i}] expert {n} >= original_experts {original_experts}"));
                        }
                        Ok(n)
                    }).collect::<Result<Vec<_>, String>>()?
                } else {
                    Vec::new()
                };
                if !experts.is_empty() && role != Role::RoutedExperts {
                    return Err(format!(
                        "reap: quant_override[{i}] lists experts but role is not routed_experts"
                    ));
                }
                let tensors: Vec<String> = if let Some(a) = o["tensors"].as_array() {
                    let layer_tok = format!("layers.{layer}.");
                    let global_role = matches!(role, Role::LmHead | Role::Embed);
                    let mut seen = std::collections::HashSet::with_capacity(a.len());
                    a.iter()
                        .enumerate()
                        .map(|(j, x)| {
                            let name = x.as_str().ok_or_else(|| {
                                format!("reap: quant_override[{i}] tensors[{j}] not a string")
                            })?;
                            if name.is_empty() {
                                return Err(format!(
                                    "reap: quant_override[{i}] tensors[{j}] is empty"
                                ));
                            }
                            if !global_role && !name.contains(&layer_tok) {
                                return Err(format!(
                                    "reap: quant_override[{i}] tensor '{name}' does not belong to layer {layer}"
                                ));
                            }
                            if !seen.insert(name) {
                                return Err(format!(
                                    "reap: quant_override[{i}] has duplicate tensor '{name}'"
                                ));
                            }
                            Ok(name.to_string())
                        })
                        .collect::<Result<Vec<_>, String>>()?
                } else {
                    Vec::new()
                };
                let tier = o["tier"]
                    .as_str()
                    .ok_or_else(|| format!("reap: quant_override[{i}] missing tier"))?
                    .to_string();
                quant_overrides.push(QuantOverride {
                    layer,
                    role,
                    experts,
                    tensors,
                    tier,
                });
            }
        }

        Ok(ReapPlan {
            model_arch: v["model_arch"].as_str().map(|s| s.to_string()),
            num_layers,
            original_experts,
            keep,
            quant_overrides,
            dir: PathBuf::from(dir),
        })
    }

    /// Load `<dir>/reap_plan.json` if present, else fall back to a legacy
    /// `<dir>/keep_by_layer.json` (keep-only; no overrides). Lets old
    /// deepseek4 sidecars keep working through the generic loader.
    pub fn load_any(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        if Path::new(dir).join("reap_plan.json").exists() {
            return Self::load(dir, num_layers_expected, orig_experts_expected);
        }
        Self::load_legacy_keepmap(dir, num_layers_expected, orig_experts_expected)
    }

    /// Resolve a REAP plan from the immutable process policy for an arch's
    /// config loader.
    ///
    /// Reads `HIPFIRE_REAP_PLAN` (and `legacy_alias_env` if given — e.g. ds4's
    /// `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP`). On a hit it rejects a dense
    /// (`orig_experts == 0`) checkpoint, loads the plan via [`Self::load_any`]
    /// validated against the ORIGINAL counts, logs the active prune, and returns
    /// `Ok(Some(plan))`. No configured path means `Ok(None)` (no pruning). The caller
    /// then overrides its own routed-expert count field to `plan.kept_per_layer()`
    /// and stores the plan — that assignment differs per arch (`num_experts` /
    /// `n_routed_experts` / `num_local_experts`) so it stays at the call site.
    /// Consolidates the env-read + dense-guard + load + log boilerplate that was
    /// duplicated across the four arch loaders (review #10). `arch` names the
    /// caller in the error/log messages.
    pub fn from_config(
        arch: &str,
        legacy_alias_env: Option<&str>,
        num_layers: usize,
        orig_experts: usize,
    ) -> Result<Option<Self>, String> {
        let dir = hipfire_config::developer_var("HIPFIRE_REAP_PLAN")
            // TOML policy first (beta's developer_var); raw env as the final
            // fallback so bare-env invocations and test seams keep working.
            .or_else(|_| std::env::var("HIPFIRE_REAP_PLAN"))
            .ok()
            .or_else(|| {
                legacy_alias_env.and_then(|name| {
                    hipfire_config::developer_var(name)
                        .or_else(|_| std::env::var(name))
                        .ok()
                })
            });
        let Some(dir) = dir else { return Ok(None) };
        // MoE-only feature: a dense (orig_experts == 0) checkpoint has no routed
        // experts to prune. Refuse rather than silently divide-by-zero / mislead.
        if orig_experts == 0 {
            return Err(format!(
                "{arch}: HIPFIRE_REAP_PLAN={dir} set but this is a dense \
                 (0 routed experts) checkpoint"
            ));
        }
        let plan = Self::load_any(&dir, num_layers, orig_experts)?;
        eprintln!(
            "{arch}: REAP plan ACTIVE — keeping {} of {orig_experts} routed experts/layer; dir {dir}",
            plan.kept_per_layer(),
        );
        Ok(Some(plan))
    }

    /// Load the legacy `<dir>/keep_by_layer.json` schema (top-level
    /// `kept_per_layer:u64`, optional `original_experts:u64`, and
    /// `keep: [[u32; kept]; num_layers]`) and produce a keep-only
    /// `ReapPlan`. Validation mirrors the old `ReapKeepMap::load`: the
    /// `original_experts` must match the model, layer count must match,
    /// each row length must equal `kept_per_layer`, and every index must
    /// be `< original_experts`.
    pub fn load_legacy_keepmap(
        dir: &str,
        num_layers_expected: usize,
        orig_experts_expected: usize,
    ) -> Result<Self, String> {
        let path = Path::new(dir).join("keep_by_layer.json");
        let txt = std::fs::read_to_string(&path)
            .map_err(|e| format!("reap: legacy keep-map read {path:?}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| format!("reap: legacy keep-map parse {path:?}: {e}"))?;

        let kept = v["kept_per_layer"]
            .as_u64()
            .ok_or("reap: legacy keep-map missing kept_per_layer")? as usize;
        if kept == 0 {
            return Err("reap: legacy keep-map keeps 0 experts (kept_per_layer == 0)".to_string());
        }
        let original_experts = v["original_experts"]
            .as_u64()
            .unwrap_or(orig_experts_expected as u64) as usize;
        if original_experts != orig_experts_expected {
            return Err(format!(
                "reap: legacy keep-map original_experts {original_experts} != model n_routed_experts {orig_experts_expected}"
            ));
        }
        let keep_arr = v["keep"]
            .as_array()
            .ok_or("reap: legacy keep-map missing `keep` array")?;
        if keep_arr.len() != num_layers_expected {
            return Err(format!(
                "reap: legacy keep-map has {} layers, model has {num_layers_expected}",
                keep_arr.len()
            ));
        }
        let mut keep = Vec::with_capacity(keep_arr.len());
        for (l, row) in keep_arr.iter().enumerate() {
            let r = row
                .as_array()
                .ok_or_else(|| format!("reap: legacy keep layer {l} not an array"))?;
            if r.len() != kept {
                return Err(format!(
                    "reap: legacy keep layer {l} has {} entries, expected {kept}",
                    r.len()
                ));
            }
            let mut v32 = Vec::with_capacity(kept);
            let mut seen = std::collections::HashSet::with_capacity(kept);
            for x in r {
                let idx = x
                    .as_u64()
                    .ok_or_else(|| format!("reap: legacy keep layer {l} non-integer index"))?
                    as u32;
                if idx as usize >= original_experts {
                    return Err(format!(
                        "reap: legacy keep layer {l} index {idx} >= original_experts {original_experts}"
                    ));
                }
                if !seen.insert(idx) {
                    return Err(format!(
                        "reap: legacy keep layer {l} has duplicate index {idx}"
                    ));
                }
                v32.push(idx);
            }
            keep.push(v32);
        }

        Ok(ReapPlan {
            model_arch: None,
            num_layers: num_layers_expected,
            original_experts,
            keep: Some(keep),
            quant_overrides: Vec::new(),
            dir: PathBuf::from(dir),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_plan(json: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(d.path().join("reap_plan.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        d
    }

    fn write_legacy(json: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(d.path().join("keep_by_layer.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        d
    }

    #[test]
    fn loads_legacy_keepmap() {
        let d = write_legacy(r#"{"kept_per_layer":2,"original_experts":4,"keep":[[0,3],[1,2]]}"#);
        let p = ReapPlan::load_any(d.path().to_str().unwrap(), 2, 4).unwrap();
        assert_eq!(p.kept_per_layer(), 2);
        assert_eq!(p.keep.as_ref().unwrap()[0], vec![0, 3]);
    }

    #[test]
    fn keep_all_when_keep_absent() {
        let d = write_plan(r#"{"original_experts":8,"num_layers":2}"#);
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 8).unwrap();
        assert!(p.keep.is_none());
        assert_eq!(p.kept_per_layer(), 8);
    }

    #[test]
    fn parses_keep_and_overrides() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":2,
                "keep":{"per_layer":[[0,2,3],[1,2,3]]},
                "quant_overrides":[{"layer":1,"role":"routed_experts","experts":[2],
                "tensors":["layers.1.ffn.experts.2.w1.weight"],"tier":"mq3lloyd"}]}"#,
        );
        let p = ReapPlan::load(d.path().to_str().unwrap(), 2, 4).unwrap();
        assert_eq!(p.kept_per_layer(), 3);
        assert_eq!(p.keep.as_ref().unwrap()[0], vec![0, 2, 3]);
        assert_eq!(p.quant_overrides.len(), 1);
        assert_eq!(p.quant_overrides[0].tier, "mq3lloyd");
        assert_eq!(
            p.quant_overrides[0].tensors,
            vec!["layers.1.ffn.experts.2.w1.weight"]
        );
    }

    #[test]
    fn rejects_out_of_range_index() {
        let d = write_plan(r#"{"original_experts":4,"num_layers":1,"keep":{"per_layer":[[0,9]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("index 9 >= original_experts 4"), "got: {err}");
    }

    #[test]
    fn rejects_experts_on_non_routed_role() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"attention","experts":[1],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("not routed_experts"), "got: {err}");
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let d = write_plan(r#"{"original_experts":4,"num_layers":3,"keep":{"per_layer":[[0,1]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 3, 4).unwrap_err();
        assert!(err.contains("keep.per_layer has 1 layers"), "got: {err}");
    }

    #[test]
    fn rejects_non_integer_override_expert() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"routed_experts","experts":[1,"bad"],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("not an integer"), "got: {err}");
    }

    #[test]
    fn rejects_exact_tensor_from_wrong_layer() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":2,
                "quant_overrides":[{"layer":0,"role":"attention",
                "tensors":["layers.1.attn.wq_b.weight"],"tier":"mfp4e8soa"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 2, 4).unwrap_err();
        assert!(err.contains("does not belong to layer 0"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_exact_tensor() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"attention",
                "tensors":["layers.0.attn.wq_b.weight","layers.0.attn.wq_b.weight"],
                "tier":"mfp4e8soa"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("duplicate tensor"), "got: {err}");
    }

    #[test]
    fn global_head_allowlist_does_not_require_layer_token() {
        let d = write_plan(
            r#"{"original_experts":256,"num_layers":43,
                "quant_overrides":[{"layer":0,"role":"lm_head",
                "tensors":["head.weight"],"tier":"mfp4e8soa"}]}"#,
        );
        let plan = ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap();
        assert_eq!(plan.quant_overrides[0].tensors, ["head.weight"]);
    }

    #[test]
    fn load_unchecked_parses_without_model_counts() {
        let d = write_plan(
            r#"{"original_experts":256,"num_layers":43,
                "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7,12],"tier":"mq3lloyd"},
                                   {"layer":41,"role":"attention","tier":"q8"}]}"#,
        );
        let p = ReapPlan::load_unchecked(d.path().to_str().unwrap()).unwrap();
        assert_eq!(p.original_experts, 256);
        assert_eq!(p.num_layers, 43);
        assert_eq!(p.quant_overrides.len(), 2);
        assert_eq!(p.quant_overrides[0].tier, "mq3lloyd");
        assert_eq!(p.quant_overrides[0].experts, vec![7, 12]);
    }

    #[test]
    fn rejects_empty_keep_map() {
        // A keep map that keeps 0 experts would override n_routed_experts to 0.
        let d = write_plan(r#"{"original_experts":4,"num_layers":1,"keep":{"per_layer":[[]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("keeps 0 experts"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_keep_index() {
        let d = write_plan(r#"{"original_experts":4,"num_layers":1,"keep":{"per_layer":[[1,1]]}}"#);
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("duplicate index 1"), "got: {err}");
    }

    #[test]
    fn rejects_empty_legacy_keep_map() {
        let d = write_legacy(r#"{"kept_per_layer":0,"original_experts":4,"keep":[[]]}"#);
        let err = ReapPlan::load_any(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("keeps 0 experts"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_legacy_keep_index() {
        let d = write_legacy(r#"{"kept_per_layer":2,"original_experts":4,"keep":[[2,2]]}"#);
        let err = ReapPlan::load_any(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains("duplicate index 2"), "got: {err}");
    }

    #[test]
    fn rejects_out_of_range_override_expert() {
        let d = write_plan(
            r#"{"original_experts":4,"num_layers":1,
                "quant_overrides":[{"layer":0,"role":"routed_experts","experts":[9],"tier":"q8"}]}"#,
        );
        let err = ReapPlan::load(d.path().to_str().unwrap(), 1, 4).unwrap_err();
        assert!(err.contains(">= original_experts 4"), "got: {err}");
    }
}
