// SPDX-License-Identifier: Apache-2.0
// hipfire — tiny random-init model fixtures for fast kernel/plumbing gating.
//
// Emits a HF-format (safetensors + config.json) random-init model in a
// hipfire-supported architecture's exact tensor layout, at "tiny" dims
// (<10M params) that still preserve the structural features gating needs
// (≥1 of each layer type, etc.). The output flows through the normal
// `--input` quantize path, so it exercises the arch-specific name-mapper too.
//
// The manifest here is the single source of truth re-used from what the
// ingest path expects; as new archs gain support, add a `tiny_*` builder.
// See TODO.md "Tiny random-init fixtures + golden-output tripwire".

// The fixture manifest vocabulary (TensorSpec/Init/Dt) lives in hipfire-arch-api so
// each family's `-spec` crate can DECLARE its ToyModel fixture with only that dep;
// this crate keeps the writer (seeded RNG → safetensors + shared tokenizer).
use hipfire_arch_api::{Dt, Init, TensorSpec};
use hipfire_primitives::conv::f32_to_bf16_bits as bf16_bits;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// Deterministic splitmix64 → reproducible fixtures across machines.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [-1, 1).
    fn next_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

/// Tiny DFlash draft sidecar fixture. This is not a quality model; it is a
/// runtime/training artifact shape that can flow through `dflash_convert`.
struct DflashTiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    target_layer_ids: Vec<usize>,
    num_target_layers: usize,
}

impl DflashTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            block_size: 16,
            target_layer_ids: vec![0, 1],
            num_target_layers: 4,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "dflash",
            "num_hidden_layers": self.layers,
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "block_size": self.block_size,
            "num_target_layers": self.num_target_layers,
            "dflash_config": {
                "mask_token_id": self.vocab - 1,
                "target_layer_ids": self.target_layer_ids,
            },
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let num_extract = self.target_layer_ids.len();
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "fc.weight",
            vec![h, num_extract * h],
            Init::Uniform(0.03),
        ));
        t.push(TensorSpec::f16(
            "hidden_norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::f16("norm.weight", vec![h], Init::NormOnes));
        for i in 0..self.layers {
            let p = format!("layers.{i}");
            let sa = format!("{p}.self_attn");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.03),
            ));
        }
        t
    }
}

/// Generate little-endian bytes for one tensor at its declared dtype.
fn gen_bytes(spec: &TensorSpec, rng: &mut SplitMix64) -> Vec<u8> {
    let n: usize = spec.shape.iter().product();
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let v = match spec.init {
            Init::Uniform(s) => rng.next_unit() * s,
            Init::NormOnes => 1.0 + rng.next_unit() * 0.02,
            Init::ALog => -2.0 + rng.next_unit() * 0.5, // exp(A_log) small & positive
            Init::Zeros => 0.0,
        };
        let bits = match spec.dt {
            Dt::Bf16 => bf16_bits(v),
            Dt::F16 => crate::f32_to_f16(v),
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// Write a safetensors file: [u64 LE header len][JSON header][concatenated data].
fn write_safetensors(
    path: &Path,
    specs: &[TensorSpec],
    rng: &mut SplitMix64,
) -> Result<(), String> {
    let mut datas: Vec<Vec<u8>> = Vec::with_capacity(specs.len());
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for spec in specs {
        let bytes = gen_bytes(spec, rng);
        let end = offset + bytes.len();
        header.insert(
            spec.name.clone(),
            serde_json::json!({
                "dtype": spec.dt.st_name(),
                "shape": spec.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
        datas.push(bytes);
    }
    let header_json = serde_json::to_string(&header).map_err(|e| e.to_string())?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    f.write_all(&(header_json.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(header_json.as_bytes())
        .map_err(|e| e.to_string())?;
    for d in &datas {
        f.write_all(d).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Emit a tiny random-init fixture for `arch` into `out_dir` (created if absent).
/// Writes `config.json` + `model.safetensors`. Reproducible for a given `seed`.
/// GPT-2 byte→unicode mapping (the fixed table every byte-level BPE uses). MUST
/// match `hipfire_model::tokenizer::byte_to_gpt2_char` exactly, or the loader's
/// `build_byte_to_id` rejects the vocab. Printable bytes map to themselves; the
/// rest map to U+0100.. in byte order. Validated by `tests::tiny_tokenizer_loads`.
fn gpt2_byte_chars() -> [char; 256] {
    let mut out = ['?'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    out
}

/// A minimal, arch-agnostic byte-level BPE `tokenizer.json` for the tiny fixtures.
/// Every model's real tokenizer is fused to its trained weights and can't be
/// swapped — but a random-init fixture's tokenizer is arbitrary, so ALL fixtures
/// share this one: 256 single-byte tokens (no merges) + `<|endoftext|>`, with a
/// `ByteLevel` pre-tokenizer/decoder so hipfire detects it as byte-level BPE. This
/// makes each tiny `.hfq` a COMPLETE model the real `serving-core` loader accepts,
/// so quant-testing runs on the production load+forward path (no bespoke harness).
fn byte_level_tokenizer_json() -> serde_json::Value {
    let chars = gpt2_byte_chars();
    let mut vocab = serde_json::Map::new();
    for (i, c) in chars.iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::Value::from(i as u64));
    }
    let eot = 256u64;
    vocab.insert("<|endoftext|>".to_string(), serde_json::Value::from(eot));
    serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "added_tokens": [{
            "id": eot, "content": "<|endoftext|>", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true,
        }],
    })
}

/// Fetch a migrated family's fixture from the offline arch registry (its `-spec`
/// crate's `ToyModel`). Returns the same `(config, specs)` shape the local `*Tiny`
/// arms produce, so `emit_fixture` writes it through the identical path. The config
/// round-trips through a `Value` so the pretty-printed bytes stay byte-identical to
/// what the old in-crate arm wrote.
fn toy_fixture_from_registry(
    arch_id: u16,
    seed: u64,
) -> Result<(serde_json::Value, Vec<TensorSpec>), String> {
    use hipfire_arch_api::{ArchId, ArchRegistry};
    let f = ArchRegistry::build()
        .get(ArchId(arch_id))
        .and_then(|a| a.caps.toy_model)
        .ok_or_else(|| format!("--emit-fixture: arch_id {arch_id} declares no ToyModel"))?
        .fixture(seed);
    let config = serde_json::from_str(&f.config_json)
        .map_err(|e| format!("parse toy config for arch {arch_id}: {e}"))?;
    Ok((config, f.tensors))
}

pub fn emit_fixture(arch: &str, out_dir: &Path, seed: u64) -> Result<(), String> {
    let arch_norm = arch.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir:?}: {e}"))?;

    let (config, specs) = match arch_norm.as_str() {
        // Every model family's fixture now lives in its `-spec` crate's ToyModel,
        // dispatched by the registered ArchId. `dflash` is a draft sidecar (no arch
        // id / model family), so it keeps its local `DflashTiny` builder.
        "qwen3_5" | "qwen35" | "qwen3_5_text" => toy_fixture_from_registry(5, seed)?,
        "qwen3_5_moe" | "qwen35moe" | "qwen3_5_moe_text" => toy_fixture_from_registry(6, seed)?,
        "qwen2" => toy_fixture_from_registry(1, seed)?,
        "gemma3" | "gemma3_text" => toy_fixture_from_registry(12, seed)?,
        "minimax" | "minimax_m2" => toy_fixture_from_registry(10, seed)?,
        "mamba2" | "mamba_2" => toy_fixture_from_registry(15, seed)?,
        "llama" | "mistral" => toy_fixture_from_registry(0, seed)?,
        "dflash" | "dflash_draft" | "tiny_dflash" => {
            let m = DflashTiny::preset();
            (m.config_json(), m.manifest())
        }
        other => {
            return Err(format!(
                "--emit-fixture: unsupported arch '{other}'. Supported: qwen3_5 \
                 (arch 5 dense), qwen3_5_moe (arch 6 MoE), qwen2 (arch 7, quantize \
                 with --arch-id 7), gemma3 (arch 12), minimax (arch 10), mamba2 \
                 (arch 15), llama (arch 0), dflash (draft sidecar). Add a tiny \
                 preset per arch as support lands."
            ));
        }
    };

    let mut rng = SplitMix64(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    write_safetensors(&out_dir.join("model.safetensors"), &specs, &mut rng)?;
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write config.json: {e}"))?;

    // Shared byte-level tokenizer → the quantizer embeds it into the .hfq metadata
    // ("tokenizer"), making the fixture a COMPLETE model the real serving-core
    // loader accepts (so quant tests run on the production path, not a bypass).
    std::fs::write(
        out_dir.join("tokenizer.json"),
        serde_json::to_string(&byte_level_tokenizer_json()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write tokenizer.json: {e}"))?;

    let n_params: usize = specs
        .iter()
        .map(|s| s.shape.iter().product::<usize>())
        .sum();
    eprintln!(
        "emit-fixture: wrote {arch_norm} fixture to {out_dir:?} \
         ({} tensors, {:.2}M params, seed {seed:#x})",
        specs.len(),
        n_params as f64 / 1e6,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_tokenizer_loads() {
        // Emit a fixture and load its tokenizer.json through hipfire's REAL
        // tokenizer. `from_hf_json` runs `build_byte_to_id`, which errors if any
        // byte 0..=255 is missing from the vocab — so this fails loudly if our
        // `gpt2_byte_chars` ever drifts from `byte_to_gpt2_char`.
        let dir = tempfile::tempdir().unwrap();
        emit_fixture("llama", dir.path(), 42).unwrap();
        let tok_path = dir.path().join("tokenizer.json");
        assert!(tok_path.exists(), "emit_fixture must write tokenizer.json");
        let tok = hipfire_model::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .expect("tokenizer.json parses")
            .expect("tokenizer present");
        // Every byte encodes to an id within the fixture's vocab (4096), so the
        // synthetic prompt used for KLD indexes valid embedding rows.
        let ids = tok.encode("hello, hipfire! 42");
        assert!(!ids.is_empty());
        assert!(
            ids.iter().all(|&id| id < 4096),
            "ids must fit vocab: {ids:?}"
        );
    }

    #[test]
    fn bf16_roundtrip_basic() {
        // 1.0 and 0.0 are exact in bf16.
        assert_eq!(bf16_bits(0.0), 0x0000);
        assert_eq!(bf16_bits(1.0), 0x3F80);
    }

    // Per-family manifests now live in each `-spec` crate's ToyModel; these are
    // integration checks that the registry dispatch yields the right structure.
    // (llama's own invariants moved to hipfire-arch-llama-spec.)

    /// Total param count for a manifest, for the <10M tiny budget assert.
    fn n_params(specs: &[TensorSpec]) -> usize {
        specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum()
    }

    #[test]
    fn qwen35_manifest_has_both_layer_types_and_is_tiny() {
        let specs = toy_fixture_from_registry(5, 42).unwrap().1;
        let has = |sub: &str| specs.iter().any(|s| s.name.contains(sub));
        // Dense Qwen3.5 interleaves linear-attn (DeltaNet) and full-attn layers.
        assert!(has("linear_attn"), "has a linear-attention layer");
        assert!(has("self_attn.q_proj"), "has a full-attention layer");
        assert!(
            n_params(&specs) < 10_000_000,
            "fixture must stay <10M params"
        );
        // in_proj_qkv = 2*key + value head dims.
        let qkv = specs
            .iter()
            .find(|s| s.name.ends_with("in_proj_qkv.weight"))
            .unwrap();
        assert_eq!(qkv.shape[0], 2 * 128 * 2 + 2 * 128);
    }

    #[test]
    fn qwen35_moe_manifest_has_experts_router_shared_and_is_tiny() {
        let specs = toy_fixture_from_registry(6, 42).unwrap().1;
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("mlp.gate.weight"), "router");
        assert!(has("mlp.experts.gate_up_proj"), "stacked experts");
        assert!(has("mlp.experts.down_proj"));
        assert!(has("mlp.shared_expert.gate_proj.weight"), "shared expert");
        assert!(has("mlp.shared_expert_gate.weight"));
        // stacked-3D expert tensor: [num_experts, 2*moe_inter, hidden].
        let gu = specs
            .iter()
            .find(|s| s.name.ends_with("experts.gate_up_proj"))
            .unwrap();
        assert_eq!(gu.shape, vec![8, 2 * 128, 256]);
        assert!(
            n_params(&specs) < 10_000_000,
            "moe fixture must stay <10M params"
        );
    }

    #[test]
    fn qwen2_manifest_has_qkv_bias_and_is_tiny() {
        let specs = toy_fixture_from_registry(1, 42).unwrap().1;
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("self_attn.q_proj.bias"), "qwen2 must carry q bias");
        assert!(has("self_attn.k_proj.bias"));
        assert!(has("self_attn.v_proj.bias"));
        assert!(has("mlp.gate_proj.weight"), "dense SwiGLU");
        assert!(!has("lm_head.weight"), "tied ⇒ no separate lm_head");
        assert!(
            n_params(&specs) < 10_000_000,
            "qwen2 fixture must stay <10M params"
        );
    }

    #[test]
    fn gemma3_manifest_has_four_norms_and_qk_norm() {
        let specs = toy_fixture_from_registry(12, 42).unwrap().1;
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("self_attn.q_norm.weight"), "per-head QK-norm");
        assert!(
            has("pre_feedforward_layernorm.weight"),
            "gemma 4-norm layout"
        );
        assert!(has("post_feedforward_layernorm.weight"));
        assert!(
            n_params(&specs) < 10_000_000,
            "gemma3 fixture must stay <10M params"
        );
    }

    #[test]
    fn minimax_manifest_has_split_experts_router_bias_and_untied_head() {
        let specs = toy_fixture_from_registry(10, 42).unwrap().1;
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(has("lm_head.weight"), "minimax is untied");
        assert!(has("block_sparse_moe.gate.weight"), "router");
        assert!(
            has("block_sparse_moe.e_score_correction_bias"),
            "routing bias"
        );
        assert!(has("block_sparse_moe.experts.0.w1.weight"), "split experts");
        assert!(has("block_sparse_moe.experts.0.w2.weight"));
        assert!(has("block_sparse_moe.experts.0.w3.weight"));
        // All experts identical shape (packed-layout uniform-stride requirement).
        let w1: Vec<_> = specs
            .iter()
            .filter(|s| s.name.ends_with(".w1.weight"))
            .collect();
        assert!(w1.windows(2).all(|w| w[0].shape == w[1].shape));
        assert!(
            n_params(&specs) < 10_000_000,
            "minimax fixture must stay <10M params"
        );
    }

    #[test]
    fn mamba2_manifest_has_state_spaces_names_and_is_tiny() {
        let specs = toy_fixture_from_registry(15, 42).unwrap().1;
        let has = |suf: &str| specs.iter().any(|s| s.name.ends_with(suf));
        assert!(
            has("backbone.embedding.weight"),
            "state-spaces embedding name"
        );
        assert!(has("mixer.in_proj.weight"));
        assert!(has("mixer.conv1d.weight"));
        assert!(has("mixer.A_log"));
        assert!(has("mixer.D"));
        assert!(has("mixer.dt_bias"));
        assert!(has("mixer.norm.weight"));
        assert!(has("mixer.out_proj.weight"));
        assert!(
            n_params(&specs) < 10_000_000,
            "mamba2 fixture must stay <10M params"
        );
    }

    #[test]
    fn emit_new_families_are_deterministic() {
        let base = std::env::temp_dir().join(format!("hipfire-fx-fam-{}", std::process::id()));
        for arch in ["qwen2", "gemma3", "minimax", "mamba2", "llama"] {
            let dir = base.join(arch);
            emit_fixture(arch, &dir, 7).unwrap();
            let a = std::fs::read(dir.join("model.safetensors")).unwrap();
            emit_fixture(arch, &dir, 7).unwrap();
            let b = std::fs::read(dir.join("model.safetensors")).unwrap();
            assert_eq!(
                a, b,
                "{arch}: same seed must produce byte-identical safetensors"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn emit_is_deterministic_for_seed() {
        let dir = std::env::temp_dir().join(format!("hipfire-fixture-test-{}", std::process::id()));
        emit_fixture("qwen3_5", &dir, 42).unwrap();
        let a = std::fs::read(dir.join("model.safetensors")).unwrap();
        emit_fixture("qwen3_5", &dir, 42).unwrap();
        let b = std::fs::read(dir.join("model.safetensors")).unwrap();
        assert_eq!(a, b, "same seed must produce byte-identical safetensors");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
