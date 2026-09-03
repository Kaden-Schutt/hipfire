//! Converter for EschaLabs Escha-W2 checkpoints (`quant_method` = `escha` /
//! `eschamoe`) into `.hfq`.
//!
//! Code streams are copied byte-for-byte; `memcmp` on the round-trip is a
//! post-condition. See docs/plans/escha-w2-port-design.md.

use crate::hfq::QuantType;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Leaf {
    Code,
    Rin,
    Rout,
    SIn,
    SOut,
    Config,
    Bias,
    Int8,
    Int8Scale,
    Passthrough,
    UnknownEscha,
}

/// The complete escha leaf namespace. Anything else beginning `escha_` is a
/// format mismatch from a newer exporter and must stop conversion.
///
/// `ignore` in `quantization_config` means "not escha-coded", NOT "not
/// quantized" — both Escha releases list `embed_tokens` and `lm_head` there
/// and still ship them as `weight_int8`. Classification therefore keys off
/// the tensor suffix actually present, never off the config's `ignore` list.
pub(crate) fn classify_leaf(name: &str) -> Leaf {
    let suffix = name.rsplit('.').next().unwrap_or("");
    match suffix {
        "escha_code" => Leaf::Code,
        "escha_rin" => Leaf::Rin,
        "escha_rout" => Leaf::Rout,
        "escha_s_in" => Leaf::SIn,
        "escha_s_out" => Leaf::SOut,
        "escha_config" => Leaf::Config,
        "bias" => Leaf::Bias,
        "weight_int8" => Leaf::Int8,
        "weight_scale" => Leaf::Int8Scale,
        s if s.starts_with("escha_") => Leaf::UnknownEscha,
        _ => Leaf::Passthrough,
    }
}

/// `K` from the code tensor's own shape: the last dimension is `16 * K`.
///
/// This is the ONLY source of truth. `escha_config` is optional — an export
/// made without the end-to-end fine-tune ships none — and `layer_meta.bits`
/// is self-inconsistent across releases (the same projection is recorded as
/// bits 3.0 in one release and bits 2.0 in the other, both with K=3).
pub(crate) fn k_from_code_shape(shape: &[usize]) -> Result<usize, String> {
    let last = *shape.last().ok_or("escha_code has no dimensions")?;
    if last % 16 != 0 {
        return Err(format!(
            "escha_code last dim {last} is not a multiple of 16"
        ));
    }
    let k = last / 16;
    if k != 2 && k != 3 {
        return Err(format!(
            "unsupported escha code rate K={k} (expected 2 or 3)"
        ));
    }
    Ok(k)
}

pub(crate) fn quant_type_for_k(k: usize) -> Result<QuantType, String> {
    match k {
        2 => Ok(QuantType::ESCHA2T16),
        3 => Ok(QuantType::ESCHA3T16),
        _ => Err(format!("unsupported escha code rate K={k}")),
    }
}

/// Required: code, rin, rout. Optional: s_in, s_out, config, bias. Missing
/// any of the required three is a hard error — Escha's own test for this is
/// `rejects_incomplete_linear`, whose docstring requires failing loudly at
/// load rather than decoding into noise.
pub(crate) fn check_linear_complete(proj: &str, present: &[Leaf]) -> Result<(), String> {
    for req in [Leaf::Code, Leaf::Rin, Leaf::Rout] {
        if !present.contains(&req) {
            return Err(format!(
                "incomplete escha linear '{proj}': missing {req:?}; \
                 refusing to decode into noise"
            ));
        }
    }
    Ok(())
}

use crate::hfq::{write_hfq, HfqTensor};
use hipfire_quantize::escha_ref::fold_scales;
use hipfire_quantize::safetensors_file::SafetensorsFile;
use std::collections::BTreeMap;

/// Convert an Escha-W2 checkpoint directory into a single `.hfq`.
///
/// `arch` is 6 for `eschamoe` (MoE) and 5 for `escha` (dense).
pub(crate) fn convert_escha(src_dir: &Path, out: &Path) -> Result<(), String> {
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(src_dir.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let qc = &cfg["quantization_config"];
    let method = qc["quant_method"].as_str().unwrap_or_default();
    let version = qc["format_version"].as_str().unwrap_or_default();
    if version != "2.0" {
        return Err(format!(
            "unsupported escha format_version {version:?}; expected \"2.0\""
        ));
    }
    let arch: u32 = match method {
        "eschamoe" => 6,
        "escha" => 5,
        other => return Err(format!("not an escha checkpoint: quant_method {other:?}")),
    };

    // Tensors can straddle shards (the 27B's mlp.up_proj has its escha_code in
    // shard 2 while its metadata sits in shard 1), so resolve through every
    // shard rather than per-file.
    let mut shards = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(src_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    for p in &paths {
        shards.push(SafetensorsFile::open(p).map_err(|e| e.to_string())?);
    }
    let find = |name: &str| shards.iter().find_map(|s| s.tensor_data(name));

    // Group leaves by projection prefix so completeness can be checked.
    let mut by_proj: BTreeMap<String, Vec<(String, Leaf)>> = BTreeMap::new();
    let mut passthrough: Vec<String> = Vec::new();
    for s in &shards {
        for name in s.tensor_names() {
            let leaf = classify_leaf(name);
            match leaf {
                Leaf::UnknownEscha => {
                    return Err(format!(
                        "unknown escha tensor '{name}': this build implements \
                         escha_code/rin/rout/s_in/s_out/config only. A newer \
                         exporter shipped a leaf we do not decode; refusing."
                    ))
                }
                Leaf::Passthrough | Leaf::Int8 | Leaf::Int8Scale => {
                    passthrough.push(name.to_string())
                }
                _ => {
                    let prefix = name
                        .rsplit_once('.')
                        .ok_or_else(|| format!("{name}: escha leaf name has no '.' separator"))?
                        .0
                        .to_string();
                    by_proj
                        .entry(prefix)
                        .or_default()
                        .push((name.to_string(), leaf));
                }
            }
        }
    }

    let mut tensors: Vec<HfqTensor> = Vec::new();
    for (proj, leaves) in &by_proj {
        let kinds: Vec<Leaf> = leaves.iter().map(|(_, l)| *l).collect();
        check_linear_complete(proj, &kinds)?;

        let (meta, data) = find(&format!("{proj}.escha_code"))
            .ok_or_else(|| format!("{proj}: escha_code vanished between passes"))?;
        let k = k_from_code_shape(&meta.shape)?;
        let qt = quant_type_for_k(k)?;

        // Verbatim: the code stream is copied byte-for-byte. memcmp on the
        // round-trip is the post-condition (G1).
        tensors.push(HfqTensor {
            name: format!("{proj}.escha_code"),
            quant_type: qt,
            shape: meta.shape.iter().map(|&d| d as u32).collect(),
            group_size: 16,
            data: data.to_vec(),
            spilled_len: 0,
        });

        // Fold the optional end-to-end scales into rin/rout — one f32 pair per
        // projection, per row when the tensor is E-stacked.
        let (rin_m, rin_d) = find(&format!("{proj}.escha_rin"))
            .ok_or_else(|| format!("{proj}: escha_rin vanished between passes"))?;
        let (rout_m, rout_d) = find(&format!("{proj}.escha_rout"))
            .ok_or_else(|| format!("{proj}: escha_rout vanished between passes"))?;
        let s_in = find(&format!("{proj}.escha_s_in")).map(|(_, d)| as_f32(d));
        let s_out = find(&format!("{proj}.escha_s_out")).map(|(_, d)| as_f32(d));
        let (ri, ro) = fold_scales(
            &as_u16(rin_d),
            &as_u16(rout_d),
            s_in.as_deref(),
            s_out.as_deref(),
        );
        tensors.push(f32_tensor(
            &format!("{proj}.escha_rin_eff"),
            &rin_m.shape,
            ri,
        ));
        tensors.push(f32_tensor(
            &format!("{proj}.escha_rout_eff"),
            &rout_m.shape,
            ro,
        ));

        if let Some((bm, bd)) = find(&format!("{proj}.bias")) {
            tensors.push(HfqTensor {
                name: format!("{proj}.bias"),
                quant_type: QuantType::F16,
                shape: bm.shape.iter().map(|&d| d as u32).collect(),
                group_size: 0,
                data: bd.to_vec(),
                spilled_len: 0,
            });
        }
    }

    for name in &passthrough {
        match classify_leaf(name) {
            // Consumed alongside its weight_int8 sibling.
            Leaf::Int8Scale => continue,
            Leaf::Int8 => {
                let prefix = name
                    .rsplit_once('.')
                    .ok_or_else(|| format!("{name}: escha leaf name has no '.' separator"))?
                    .0;
                let (m, d) = find(name)
                    .ok_or_else(|| format!("{name}: weight_int8 vanished between passes"))?;
                let (_, sd) = find(&format!("{prefix}.weight_scale")).ok_or_else(|| {
                    format!("{name}: weight_int8 without a matching weight_scale")
                })?;
                let oc = m.shape[0];
                let ic = m.shape[1];
                let w8: Vec<i8> = d.iter().map(|&b| b as i8).collect();
                let q8 = int8_rows_to_q8_0(&w8, &as_u16(sd), oc, ic)?;
                tensors.push(HfqTensor {
                    name: format!("{prefix}.weight"),
                    quant_type: QuantType::Q8F16,
                    shape: vec![oc as u32, ic as u32],
                    group_size: 32,
                    data: q8,
                    spilled_len: 0,
                });
            }
            _ => {
                let (m, d) = find(name)
                    .ok_or_else(|| format!("{name}: passthrough tensor vanished between passes"))?;
                tensors.push(HfqTensor {
                    name: name.clone(),
                    quant_type: match m.dtype.as_str() {
                        "F16" => QuantType::F16,
                        "F32" => QuantType::F32,
                        "BF16" => QuantType::BF16,
                        other => return Err(format!("{name}: unhandled dtype {other}")),
                    },
                    shape: m.shape.iter().map(|&d| d as u32).collect(),
                    group_size: 0,
                    data: d.to_vec(),
                    spilled_len: 0,
                });
            }
        }
    }

    let metadata = build_metadata(src_dir, &cfg, version, method)?;
    write_hfq(out, arch, &metadata, &tensors, None).map_err(|e| e.to_string())
}

/// Build the HFQ `metadata_json` envelope.
///
/// The envelope shape is NOT free-form. It mirrors `pipeline.rs`'s envelope
/// key-for-key and only ADDS the `escha` provenance key:
///
/// * `config` — the parsed config.json verbatim. `config_from_metadata_json`
///   (hipfire-arch-qwen35) requires it to reconstruct the arch config at load
///   time, and it self-detects the nested `text_config`/`vision_config` these
///   VL-shaped checkpoints carry — do not flatten or pre-process it here.
/// * `tokenizer` — tokenizer.json verbatim as a STRING, not a nested object.
///   That is what `Tokenizer::from_hfq_metadata` expects; anything else and it
///   returns `MetadataMissing { field: "tokenizer | gguf_meta" }`. vocab.json
///   and merges.txt are NOT carried separately — tokenizer.json already holds
///   the BPE vocab and merge table, and no reader looks for the sidecars.
/// * `tokenizer_config` — carries `chat_template`, the ONLY key
///   `HfqFile::chat_template()` reads and hence the only source
///   `resolve_chat_template` has for arch 5/6.
/// * `generation_config` — authoritative bos/eos ids. Escha's is an ARRAY eos
///   `[248046, 248044]`, so `from_hfq_metadata` keeps its heuristic eos; the
///   bos scalar 248044 still overrides.
fn build_metadata(
    src_dir: &Path,
    cfg: &serde_json::Value,
    version: &str,
    method: &str,
) -> Result<String, String> {
    let read_json = |name: &str| -> Option<serde_json::Value> {
        std::fs::read_to_string(src_dir.join(name))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    };

    let tokenizer_str = std::fs::read_to_string(src_dir.join("tokenizer.json")).ok();
    if tokenizer_str.is_none() {
        return Err(format!(
            "escha: no tokenizer.json in {} — the .hfq would convert cleanly and \
             then be unservable (Tokenizer::from_hfq_metadata would fail). \
             Refusing to write a model that cannot be driven.",
            src_dir.display()
        ));
    }

    // Some checkpoints ship the Jinja template in a `chat_template.jinja`
    // sidecar rather than inside tokenizer_config.json. Fold it in only when
    // tokenizer_config lacks one — an existing template wins, same rule as
    // `pipeline.rs`.
    let tokenizer_config = {
        let mut tc = read_json("tokenizer_config.json");
        let jinja_path = src_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let has_template = tc
                .as_ref()
                .and_then(|v| v.get("chat_template"))
                .map(|v| !v.is_null())
                .unwrap_or(false);
            if !has_template {
                if let Ok(jinja) = std::fs::read_to_string(&jinja_path) {
                    let n = jinja.len();
                    let obj = tc.get_or_insert_with(|| serde_json::json!({}));
                    if let Some(map) = obj.as_object_mut() {
                        map.insert(
                            "chat_template".to_string(),
                            serde_json::Value::String(jinja),
                        );
                        eprintln!(
                            "  embedded chat_template.jinja into tokenizer_config ({n} bytes)"
                        );
                    }
                }
            }
        }
        tc
    };
    if tokenizer_config
        .as_ref()
        .and_then(|v| v.get("chat_template"))
        .and_then(|v| if v.is_null() { None } else { Some(v) })
        .is_none()
    {
        eprintln!(
            "escha: warning: no chat_template in tokenizer_config.json and no \
             chat_template.jinja sidecar — the daemon will fall back to a \
             hand-rolled frame for this instruct model"
        );
    }

    let metadata = serde_json::json!({
        "config": cfg,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
        "generation_config": read_json("generation_config.json"),
        "escha": { "format_version": version, "quant_method": method },
    });
    serde_json::to_string(&metadata).map_err(|e| format!("serialize metadata: {e}"))
}

/// Escha's int8 is per-output-ROW; hipfire's `Q8_0` is per-32-element block
/// (34 bytes: f16 scale then 32 int8). Replicating the row scale into every
/// block of that row passes the int8 bytes through unchanged, so the
/// reconstruction is bit-identical to Escha's `w8a16`. Cost is 2 bytes per 32
/// elements — 6.25% — for scales that are all equal within a row.
///
/// Do NOT recompute per-block scales from the dequantised values. That is a
/// second quantisation and adds avoidable error.
pub(crate) fn int8_rows_to_q8_0(
    w8: &[i8],
    scale_f16: &[u16],
    oc: usize,
    ic: usize,
) -> Result<Vec<u8>, String> {
    if w8.len() != oc * ic {
        return Err(format!(
            "int8 tensor is {} bytes, expected {oc}x{ic}",
            w8.len()
        ));
    }
    if scale_f16.len() != oc {
        return Err(format!("expected {oc} row scales, got {}", scale_f16.len()));
    }
    if ic % 32 != 0 {
        return Err(format!("Q8_0 needs a multiple of 32 per row, got ic={ic}"));
    }
    let mut out = Vec::with_capacity(oc * (ic / 32) * 34);
    for o in 0..oc {
        let s = scale_f16[o].to_le_bytes();
        for blk in 0..ic / 32 {
            out.extend_from_slice(&s);
            let base = o * ic + blk * 32;
            out.extend(w8[base..base + 32].iter().map(|&v| v as u8));
        }
    }
    Ok(out)
}

fn as_u16(d: &[u8]) -> Vec<u16> {
    d.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn as_f32(d: &[u8]) -> Vec<f32> {
    d.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_tensor(name: &str, shape: &[usize], v: Vec<f32>) -> HfqTensor {
    HfqTensor {
        name: name.to_string(),
        quant_type: QuantType::F32,
        shape: shape.iter().map(|&d| d as u32).collect(),
        group_size: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        spilled_len: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_quantize::escha_ref::f16_rne;

    #[test]
    fn k_comes_from_the_code_shape_not_metadata() {
        // gate_up: [E, in/16, out/16, 16K] with K=2 -> last dim 32
        assert_eq!(k_from_code_shape(&[256, 128, 64, 32]), Ok(2));
        // down: K=3 -> last dim 48
        assert_eq!(k_from_code_shape(&[256, 32, 128, 48]), Ok(3));
        // dense exports have no E axis
        assert_eq!(k_from_code_shape(&[320, 1088, 32]), Ok(2));
        assert!(k_from_code_shape(&[256, 128, 64, 33]).is_err());
    }

    #[test]
    fn quant_type_follows_k() {
        assert_eq!(quant_type_for_k(2), Ok(QuantType::ESCHA2T16));
        assert_eq!(quant_type_for_k(3), Ok(QuantType::ESCHA3T16));
        assert!(quant_type_for_k(4).is_err());
    }

    /// `ignore` means "not escha-coded", NOT "not quantized" — both models
    /// list embed_tokens and lm_head there and ship them as weight_int8.
    /// Classification must key off the tensor suffix actually present.
    #[test]
    fn classify_leaf_keys_off_the_suffix() {
        assert_eq!(
            classify_leaf("l.0.mlp.experts.gate_up_proj.escha_code"),
            Leaf::Code
        );
        assert_eq!(
            classify_leaf("l.0.mlp.experts.gate_up_proj.escha_rin"),
            Leaf::Rin
        );
        assert_eq!(
            classify_leaf("l.0.mlp.experts.gate_up_proj.escha_s_out"),
            Leaf::SOut
        );
        assert_eq!(classify_leaf("lm_head.weight_int8"), Leaf::Int8);
        assert_eq!(classify_leaf("lm_head.weight_scale"), Leaf::Int8Scale);
        assert_eq!(
            classify_leaf("l.0.input_layernorm.weight"),
            Leaf::Passthrough
        );
    }

    /// A future export carrying a rotation variant this version does not
    /// implement must stop conversion, not decode under the wrong rotation.
    #[test]
    fn unknown_escha_leaf_is_rejected() {
        assert_eq!(
            classify_leaf("l.0.mlp.gate_proj.escha_rotation_theta"),
            Leaf::UnknownEscha
        );
    }

    /// Required: code, rin, rout. Missing any is "incomplete escha linear" —
    /// fail loudly at load, never a partial decode.
    #[test]
    fn incomplete_linear_is_rejected() {
        let mut present = vec![Leaf::Code, Leaf::Rin, Leaf::Rout];
        assert!(check_linear_complete("proj", &present).is_ok());
        present.pop();
        let err = check_linear_complete("proj", &present).unwrap_err();
        assert!(err.contains("incomplete escha linear"), "{err}");
    }

    /// Optional: s_in, s_out, config, bias. An export without the end-to-end
    /// stage ships none of them and must still convert.
    #[test]
    fn optional_leaves_may_all_be_absent() {
        assert!(check_linear_complete("proj", &[Leaf::Code, Leaf::Rin, Leaf::Rout]).is_ok());
    }

    /// The row scale must be replicated into every block with the int8 bytes
    /// untouched — that is what makes the repack bit-exact. Recomputing block
    /// scales would be a second quantisation.
    #[test]
    fn int8_repack_replicates_the_row_scale() {
        let oc = 2;
        let ic = 64; // two Q8_0 blocks per row
        let w8: Vec<i8> = (0..(oc * ic)).map(|i| (i % 127) as i8).collect();
        let scale = vec![f16_rne(0.5), f16_rne(2.0)];
        let q8 = int8_rows_to_q8_0(&w8, &scale, oc, ic).unwrap();
        assert_eq!(q8.len(), oc * (ic / 32) * 34);
        // Both blocks of row 0 carry row 0's scale, unchanged.
        assert_eq!(&q8[0..2], &scale[0].to_le_bytes());
        assert_eq!(&q8[34..36], &scale[0].to_le_bytes());
        // Row 1's blocks carry row 1's scale.
        assert_eq!(&q8[68..70], &scale[1].to_le_bytes());
        // Payload bytes are passed through verbatim.
        assert_eq!(q8[2] as i8, w8[0]);
        assert_eq!(q8[36] as i8, w8[32]);
    }

    #[test]
    fn int8_repack_rejects_a_ragged_row() {
        assert!(int8_rows_to_q8_0(&[0i8; 20], &[0u16], 1, 20).is_err());
    }
}

/// `convert_escha` integration test: a minimal synthetic checkpoint directory
/// (config.json + one safetensors shard) through the real converter, then a
/// real `HfqFile` read-back. This lives in-module (not under `tests/`) for
/// the same reason as `reap_overlay::integ`: `hipfire-quantize` is a binary
/// crate with no library target, so a `tests/` integration target can't reach
/// `convert_escha`, which is crate-private. `hipfire-runtime` (CPU-only HFQ
/// container reader) is a dev-dependency.
///
/// This is the test that Finding 1 (missing top-level `config` key) proves
/// would have caught the regression: it asserts the round-tripped metadata
/// carries `config` with the fields written into config.json.
#[cfg(test)]
mod convert_escha_tests {
    use super::*;
    use hipfire_quantize::escha_ref::f16_rne;
    use hipfire_runtime::hfq::HfqFile;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;
    use std::collections::HashMap;

    /// escha_code payload: deterministic, recognizable bytes so the
    /// verbatim-repack assertion can compare against a value computed
    /// independently of what `build_fixture` wrote.
    fn code_bytes() -> Vec<u8> {
        (0..(4 * 32)).map(|i| (i % 251) as u8).collect()
    }

    /// Write a minimal Escha-W2 checkpoint directory: config.json with a
    /// `quantization_config` block plus a couple of recognisable top-level
    /// config fields, and one safetensors shard with one complete escha
    /// linear (escha_code last dim 16*K=32 => K=2, plus escha_rin/rout) and
    /// one int8 passthrough pair (weight_int8 + weight_scale).
    fn build_fixture(dir: &Path) {
        let cfg = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "quantization_config": {
                "quant_method": "eschamoe",
                "format_version": "2.0",
            },
        });
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();

        // A real escha checkpoint always ships these; without them the .hfq
        // converts cleanly and is then unservable, which is the regression
        // `tokenizer_and_chat_template_land_in_metadata` pins.
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},"added_tokens":[]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token":false,"chat_template":"ESCHA_TEMPLATE"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"bos_token_id":1,"eos_token_id":[1,0]}"#,
        )
        .unwrap();

        let code = code_bytes();

        let rin: Vec<u16> = (0..4).map(|i| f16_rne(1.0 + i as f32 * 0.1)).collect();
        let rout: Vec<u16> = (0..4).map(|i| f16_rne(2.0 + i as f32 * 0.1)).collect();
        let rin_bytes: Vec<u8> = rin.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rout_bytes: Vec<u8> = rout.iter().flat_map(|v| v.to_le_bytes()).collect();

        // int8 pair: oc=2, ic=32 (one Q8_0 block per row).
        let w8: Vec<u8> = (0..(2 * 32)).map(|i| (i as i8) as u8).collect();
        let scale: Vec<u16> = vec![f16_rne(0.5), f16_rne(1.5)];
        let scale_bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert(
            "layers.0.mlp.up_proj.escha_code".to_string(),
            TensorView::new(Dtype::U8, vec![4, 32], &code).unwrap(),
        );
        tensors.insert(
            "layers.0.mlp.up_proj.escha_rin".to_string(),
            TensorView::new(Dtype::F16, vec![4], &rin_bytes).unwrap(),
        );
        tensors.insert(
            "layers.0.mlp.up_proj.escha_rout".to_string(),
            TensorView::new(Dtype::F16, vec![4], &rout_bytes).unwrap(),
        );
        tensors.insert(
            "lm_head.weight_int8".to_string(),
            TensorView::new(Dtype::I8, vec![2, 32], &w8).unwrap(),
        );
        tensors.insert(
            "lm_head.weight_scale".to_string(),
            TensorView::new(Dtype::F16, vec![2], &scale_bytes).unwrap(),
        );

        let bytes = safetensors::serialize(&tensors, None).unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    /// Unique scratch dir under the system temp root, cleaned up on drop.
    struct TempCheckpointDir(std::path::PathBuf);
    impl TempCheckpointDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "hipfire_escha_convert_test_{tag}_{}_{nanos}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempCheckpointDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn convert_escha_embeds_config_and_repacks_code_verbatim() {
        let src = TempCheckpointDir::new("ok");
        build_fixture(&src.0);
        let out = src.0.join("out.hfq");

        convert_escha(&src.0, &out).expect("conversion of a well-formed fixture must succeed");

        let hf = HfqFile::open(&out).expect("convert_escha's output must parse as a valid .hfq");

        // The regression Finding 1 caught: metadata must carry a top-level
        // `config` key, or every arch loader's `config_from_metadata_json`
        // fails before a tensor is read.
        let meta: serde_json::Value = serde_json::from_str(&hf.metadata_json)
            .expect("metadata_json must itself be valid JSON");
        let config = meta
            .get("config")
            .expect("metadata must carry a top-level `config` key");
        assert_eq!(config["hidden_size"], 64);
        assert_eq!(config["num_hidden_layers"], 2);
        assert_eq!(config["num_attention_heads"], 4);
        assert_eq!(config["vocab_size"], 100);

        // The verbatim-repack contract (G1): escha_code bytes in the output
        // are byte-identical to the input.
        let (_, out_code) = hf
            .tensor_data("layers.0.mlp.up_proj.escha_code")
            .expect("escha_code tensor must survive conversion");
        assert_eq!(out_code, code_bytes().as_slice());
    }

    /// The converter originally emitted only `config` + `escha`, so the .hfq
    /// carried no tokenizer, no chat template and no generation_config: it
    /// converted cleanly, passed G1, and could not be served at all. Pin the
    /// exact keys the readers use — `Tokenizer::from_hfq_metadata` wants
    /// `tokenizer` as a STRING, and `HfqFile::chat_template()` reads ONLY
    /// `tokenizer_config.chat_template`.
    #[test]
    fn tokenizer_and_chat_template_land_in_metadata() {
        let src = TempCheckpointDir::new("tok");
        build_fixture(&src.0);
        let out = src.0.join("out.hfq");
        convert_escha(&src.0, &out).expect("conversion must succeed");
        let hf = HfqFile::open(&out).expect("output must parse");

        let meta: serde_json::Value = serde_json::from_str(&hf.metadata_json).unwrap();
        let tok = meta["tokenizer"]
            .as_str()
            .expect("`tokenizer` must be a STRING holding tokenizer.json verbatim");
        assert!(tok.contains("\"vocab\""), "tokenizer.json must be verbatim");
        assert_eq!(meta["tokenizer_config"]["add_bos_token"], false);
        assert_eq!(meta["generation_config"]["bos_token_id"], 1);
        // The provenance key must survive alongside the new siblings.
        assert_eq!(meta["escha"]["quant_method"], "eschamoe");

        // The reader paths themselves, not just the raw keys.
        assert_eq!(
            hf.chat_template().as_deref(),
            Some("ESCHA_TEMPLATE"),
            "resolve_chat_template (arch 5|6) reads this and nothing else"
        );
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hf.metadata_json)
            .expect("the embedded metadata must build a working Tokenizer");
    }

    /// A checkpoint with no tokenizer.json cannot produce a servable model.
    /// Fail at convert time rather than shipping a 12 GB file that only fails
    /// when someone tries to run it — same fail-closed rule this converter
    /// applies to unknown escha leaves and incomplete linears.
    #[test]
    fn missing_tokenizer_is_rejected() {
        let src = TempCheckpointDir::new("notok");
        build_fixture(&src.0);
        std::fs::remove_file(src.0.join("tokenizer.json")).unwrap();
        let err = convert_escha(&src.0, &src.0.join("out.hfq")).unwrap_err();
        assert!(err.contains("no tokenizer.json"), "{err}");
    }
}
