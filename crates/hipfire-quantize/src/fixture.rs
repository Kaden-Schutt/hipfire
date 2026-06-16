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

/// f32 → bf16 bits, round-to-nearest-even.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return 0x7FC0; // NaN
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// How a tensor's elements are initialized.
#[derive(Clone, Copy)]
enum Init {
    /// Zero-mean uniform in [-scale, scale] — generic projections.
    Uniform(f32),
    /// RMSNorm weights: ~1.0 + small jitter.
    NormOnes,
    /// Mamba/DeltaNet A_log: small negative so decay stays well-conditioned.
    ALog,
    /// Bias-like: zeros.
    Zeros,
}

/// One tensor in the manifest: name, shape, init policy.
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    init: Init,
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: Vec<usize>, init: Init) -> Self {
        Self {
            name: name.into(),
            shape,
            init,
        }
    }
}

/// Tiny Qwen3.5 (arch 5) dense text config. Mirrors the real text_config
/// fields the ingest/arch-detect path reads, at fixture dims.
struct Qwen35Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    full_attn_interval: usize,
    // linear-attn (DeltaNet)
    l_key_heads: usize,
    l_key_head_dim: usize,
    l_val_heads: usize,
    l_val_head_dim: usize,
    conv_kernel: usize,
    // MoE (arch 6). `experts == 0` ⇒ dense (arch 5).
    experts: usize,
    experts_per_tok: usize,
    moe_inter: usize,
    shared_inter: usize,
}

impl Qwen35Tiny {
    /// ~3.9M params: 4 layers (3 linear-attn + 1 full-attn), tiny vocab.
    /// head_dim is pinned to 128 — the gated DeltaNet kernels are specialized
    /// for HD=128 (and full-attn supports it), so smaller HDs hard-error.
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            full_attn_interval: 4,
            l_key_heads: 2,
            l_key_head_dim: 128,
            l_val_heads: 2,
            l_val_head_dim: 128,
            conv_kernel: 4,
            experts: 0,
            experts_per_tok: 0,
            moe_inter: 0,
            shared_inter: 0,
        }
    }

    /// ~6M params: arch-6 MoE. Same hybrid attention as the dense preset, but
    /// every layer's FFN is MoE (8 experts top-2 + an always-on shared expert),
    /// matching the A3B layout (all layers MoE; attention type still varies).
    fn moe_preset() -> Self {
        Self {
            experts: 8,
            experts_per_tok: 2,
            moe_inter: 128,
            shared_inter: 128,
            ..Self::preset()
        }
    }

    fn is_moe(&self) -> bool {
        self.experts > 0
    }

    /// `full_attention` every `full_attn_interval`-th layer (positions
    /// interval-1, 2*interval-1, …), else `linear_attention` — matches the
    /// real checkpoint's layer_types pattern.
    fn layer_types(&self) -> Vec<&'static str> {
        (0..self.layers)
            .map(|i| {
                if (i + 1) % self.full_attn_interval == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    }

    fn config_json(&self) -> serde_json::Value {
        let mut c = serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_output_gate": true,
            "full_attention_interval": self.full_attn_interval,
            "layer_types": self.layer_types(),
            "linear_num_key_heads": self.l_key_heads,
            "linear_key_head_dim": self.l_key_head_dim,
            "linear_num_value_heads": self.l_val_heads,
            "linear_value_head_dim": self.l_val_head_dim,
            "linear_conv_kernel_dim": self.conv_kernel,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        });
        if self.is_moe() {
            let o = c.as_object_mut().unwrap();
            o.insert("model_type".into(), "qwen3_5_moe_text".into());
            o.insert("num_experts".into(), self.experts.into());
            o.insert("num_experts_per_tok".into(), self.experts_per_tok.into());
            o.insert("moe_intermediate_size".into(), self.moe_inter.into());
            o.insert(
                "shared_expert_intermediate_size".into(),
                self.shared_inter.into(),
            );
            o.insert("norm_topk_prob".into(), true.into());
            o.insert("decoder_sparse_step".into(), 1.into());
            o.insert("mlp_only_layers".into(), serde_json::json!([]));
        }
        c
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let mut t = Vec::new();
        // Globals (tie_word_embeddings ⇒ no separate lm_head).
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::new(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));

        let qkv =
            self.l_key_heads * self.l_key_head_dim * 2 + self.l_val_heads * self.l_val_head_dim;
        let v_dim = self.l_val_heads * self.l_val_head_dim;
        let attn_q = self.n_heads * self.head_dim * 2; // attn_output_gate ⇒ 2× wide
        let kv_dim = self.n_kv_heads * self.head_dim;
        let o_in = self.n_heads * self.head_dim;

        for (i, kind) in self.layer_types().into_iter().enumerate() {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::new(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            if self.is_moe() {
                // MoE FFN: router + stacked-3D routed experts + always-on shared expert.
                let mi = self.moe_inter;
                let si = self.shared_inter;
                t.push(TensorSpec::new(
                    format!("{p}.mlp.gate.weight"),
                    vec![self.experts, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.gate_up_proj"),
                    vec![self.experts, 2 * mi, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.down_proj"),
                    vec![self.experts, h, mi],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.gate_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.up_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.down_proj.weight"),
                    vec![h, si],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    vec![1, h],
                    Init::Uniform(0.05),
                ));
            } else {
                // Dense MLP (SwiGLU).
                t.push(TensorSpec::new(
                    format!("{p}.mlp.gate_proj.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.up_proj.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.down_proj.weight"),
                    vec![h, self.inter],
                    Init::Uniform(0.05),
                ));
            }

            if kind == "linear_attention" {
                let la = format!("{p}.linear_attn");
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_qkv.weight"),
                    vec![qkv, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_z.weight"),
                    vec![v_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_a.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_b.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.A_log"),
                    vec![self.l_val_heads],
                    Init::ALog,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.dt_bias"),
                    vec![self.l_val_heads],
                    Init::Zeros,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.conv1d.weight"),
                    vec![qkv, 1, self.conv_kernel],
                    Init::Uniform(0.1),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.norm.weight"),
                    vec![self.l_val_head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.out_proj.weight"),
                    vec![h, v_dim],
                    Init::Uniform(0.05),
                ));
            } else {
                let sa = format!("{p}.self_attn");
                t.push(TensorSpec::new(
                    format!("{sa}.q_proj.weight"),
                    vec![attn_q, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.v_proj.weight"),
                    vec![kv_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.o_proj.weight"),
                    vec![h, o_in],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.q_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
            }
        }
        t
    }
}

/// Generate bf16 little-endian bytes for one tensor.
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
        out.extend_from_slice(&bf16_bits(v).to_le_bytes());
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
                "dtype": "BF16",
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
pub fn emit_fixture(arch: &str, out_dir: &Path, seed: u64) -> Result<(), String> {
    let arch_norm = arch.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir:?}: {e}"))?;

    let (config, specs) = match arch_norm.as_str() {
        "qwen3_5" | "qwen35" | "qwen3_5_text" => {
            let m = Qwen35Tiny::preset();
            (m.config_json(), m.manifest())
        }
        "qwen3_5_moe" | "qwen35moe" | "qwen3_5_moe_text" => {
            let m = Qwen35Tiny::moe_preset();
            (m.config_json(), m.manifest())
        }
        other => {
            return Err(format!(
                "--emit-fixture: unsupported arch '{other}'. Supported: qwen3_5 \
                 (arch 5 dense), qwen3_5_moe (arch 6 MoE). Add a tiny preset per \
                 arch as support lands."
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
    fn bf16_roundtrip_basic() {
        // 1.0 and 0.0 are exact in bf16.
        assert_eq!(bf16_bits(0.0), 0x0000);
        assert_eq!(bf16_bits(1.0), 0x3F80);
    }

    #[test]
    fn manifest_has_both_layer_types_and_is_tiny() {
        let m = Qwen35Tiny::preset();
        let lt = m.layer_types();
        assert!(lt.contains(&"linear_attention"));
        assert!(lt.contains(&"full_attention"));
        let specs = m.manifest();
        let n: usize = specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n < 10_000_000, "fixture must stay <10M params, got {n}");
        // in_proj_qkv = 2*key + value head dims.
        let qkv = specs
            .iter()
            .find(|s| s.name.ends_with("in_proj_qkv.weight"))
            .unwrap();
        assert_eq!(qkv.shape[0], 2 * 128 * 2 + 2 * 128);
    }

    #[test]
    fn moe_manifest_has_experts_router_shared_and_is_tiny() {
        let m = Qwen35Tiny::moe_preset();
        assert!(m.is_moe());
        let specs = m.manifest();
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
        let n: usize = specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n < 10_000_000, "moe fixture must stay <10M params, got {n}");
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
