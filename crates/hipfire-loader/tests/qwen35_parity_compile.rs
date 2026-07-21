// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![cfg(feature = "test-utils")]

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{
    DeltaNetState, LayerWeights, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use hipfire_arch_qwen35::test_utils::legacy_delta_net_sequence;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::weight_backend::embedding_format_dtype;
use rdna_compute::Gpu;
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// The legacy loader is exposed only through the test-utils feature; the
/// ignored tests below compare it with the production manifest-backed paths.
#[test]
fn loader_can_compile_qwen35_legacy_parity_seam() {
    let _ = legacy_delta_net_sequence
        as fn(
            &mut Gpu,
            &Qwen35Weights,
            &Qwen35Config,
            u32,
            usize,
            &mut KvCache,
            &mut DeltaNetState,
            &Qwen35Scratch,
        ) -> hip_bridge::HipResult<()>;
}

type WeightMeta = (String, rdna_compute::DType, usize, usize, usize, bool, bool);

fn collect_meta(weights: &Qwen35Weights, config: &Qwen35Config) -> Vec<WeightMeta> {
    fn add(out: &mut Vec<WeightMeta>, name: String, w: &hipfire_runtime::llama::WeightTensor) {
        out.push((
            name,
            w.gpu_dtype,
            w.m,
            w.k,
            w.row_stride,
            w.paro.is_some(),
            w.awq_scale.is_some(),
        ));
    }
    let mut out = Vec::new();
    out.push((
        "token_embd".into(),
        semantic_embedding_dtype(weights.token_embd.dtype, weights.embd_format),
        config.vocab_size,
        config.dim,
        0,
        false,
        false,
    ));
    add(&mut out, "output".into(), &weights.output);
    for (layer, value) in weights.layers.iter().enumerate() {
        let mut add_layer = |prefix: &str, w: &hipfire_runtime::llama::WeightTensor| {
            add(&mut out, format!("{layer}.{prefix}"), w)
        };
        match value {
            LayerWeights::DeltaNet(l) => {
                add_layer("wqkv", &l.wqkv);
                add_layer("wz", &l.wz);
                add_layer("w_alpha", &l.w_alpha);
                add_layer("w_beta", &l.w_beta);
                add_layer("wo", &l.wo);
                add_layer("w_gate", &l.w_gate);
                add_layer("w_up", &l.w_up);
                add_layer("w_down", &l.w_down);
            }
            LayerWeights::FullAttn(l) => {
                add_layer("wq", &l.wq);
                add_layer("wk", &l.wk);
                add_layer("wv", &l.wv);
                add_layer("wo", &l.wo);
                add_layer("w_gate", &l.w_gate);
                add_layer("w_up", &l.w_up);
                add_layer("w_down", &l.w_down);
            }
            LayerWeights::DeltaNetMoe(l) => {
                add_layer("wqkv", &l.wqkv);
                add_layer("wz", &l.wz);
                for (expert, e) in l.ffn.experts.iter().enumerate() {
                    add_layer(&format!("expert.{expert}.gate_up"), &e.gate_up);
                    add_layer(&format!("expert.{expert}.down"), &e.down);
                }
            }
            LayerWeights::FullAttnMoe(l) => {
                add_layer("wq", &l.wq);
                add_layer("wk", &l.wk);
                add_layer("wv", &l.wv);
                add_layer("wo", &l.wo);
                for (expert, e) in l.ffn.experts.iter().enumerate() {
                    add_layer(&format!("expert.{expert}.gate_up"), &e.gate_up);
                    add_layer(&format!("expert.{expert}.down"), &e.down);
                }
            }
        }
    }
    out
}

/// Legacy `load_embedding` uploads raw bytes, so its `GpuTensor` metadata is
/// `Raw`; the manifest path tags the same bytes with the semantic Q8_0 dtype.
/// Compare the embedding format, not the transport tag, before progressing to
/// deterministic logits.
fn semantic_embedding_dtype(
    stored: rdna_compute::DType,
    format: hipfire_runtime::llama::EmbeddingFormat,
) -> rdna_compute::DType {
    if stored == rdna_compute::DType::Raw {
        embedding_format_dtype(format)
    } else {
        stored
    }
}

#[test]
fn embedding_metadata_normalizes_raw_transport_to_q8_semantics() {
    assert_eq!(
        semantic_embedding_dtype(
            rdna_compute::DType::Raw,
            hipfire_runtime::llama::EmbeddingFormat::Q8_0,
        ),
        rdna_compute::DType::Q8_0
    );
    assert_eq!(
        semantic_embedding_dtype(
            rdna_compute::DType::Q8_0,
            hipfire_runtime::llama::EmbeddingFormat::Q8_0,
        ),
        rdna_compute::DType::Q8_0
    );
}

fn one_logit_step(gpu: &mut Gpu, weights: &Qwen35Weights, config: &Qwen35Config) -> Vec<f32> {
    let mut kv = KvCache::new_gpu_asym3_capped(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        64,
        64,
    )
    .expect("kv");
    let mut dn = DeltaNetState::new_with_quant(gpu, config, StateQuant::Q8).expect("dn");
    let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, 64).expect("scratch");
    qwen35::forward_scratch(gpu, weights, config, 1, 0, &mut kv, &mut dn, &scratch)
        .expect("deterministic one-token forward");
    let logits = gpu.download_f32(&scratch.logits).expect("logits");
    scratch.free_gpu(gpu);
    dn.free_gpu(gpu);
    kv.free_gpu(gpu);
    logits
}

#[test]
#[ignore = "requires AMD GPU and HIPFIRE_QWEN35_HFQ fixture; run under scripts/gpu-lock.sh"]
fn hfq_legacy_vs_manifest_typed_metadata_and_deterministic_logits() {
    let _lock = GPU_LOCK.lock().unwrap();
    let Ok(path) = std::env::var("HIPFIRE_QWEN35_HFQ") else {
        return;
    };
    let mut legacy_hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path)).unwrap();
    let config = qwen35::config_from_hfq(&legacy_hfq).unwrap();
    let mut legacy_gpu = Gpu::init().unwrap();
    let mut legacy_src = qwen35::HfqSource::new(&mut legacy_hfq, &config);
    let legacy = qwen35::load_weights(
        &mut legacy_src,
        std::slice::from_mut(&mut legacy_gpu),
        &qwen35::Layout::single(config.n_layers),
    )
    .unwrap();
    let legacy_meta = collect_meta(&legacy, &config);
    let legacy_logits = one_logit_step(&mut legacy_gpu, &legacy, &config);
    legacy.free_gpu(&mut legacy_gpu);

    let manifest_hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path)).unwrap();
    let mut manifest_gpu = Gpu::init().unwrap();
    let manifest =
        hipfire_arch_qwen35::load_qwen35_hfq_weights(&manifest_hfq, &config, &mut manifest_gpu)
            .unwrap();
    assert_eq!(legacy_meta, collect_meta(&manifest, &config));
    assert_eq!(
        legacy_logits,
        one_logit_step(&mut manifest_gpu, &manifest, &config)
    );
    manifest.free_gpu(&mut manifest_gpu);
}

#[test]
#[ignore = "requires AMD GPU and HIPFIRE_QWEN35_PARO fixture; run under scripts/gpu-lock.sh"]
fn paro_legacy_vs_manifest_typed_metadata_and_deterministic_logits() {
    let _lock = GPU_LOCK.lock().unwrap();
    let Ok(path) = std::env::var("HIPFIRE_QWEN35_PARO") else {
        return;
    };
    let legacy_source =
        hipfire_runtime::safetensors_source::SafetensorsSource::open(std::path::Path::new(&path))
            .unwrap();
    let config = qwen35::config_from_safetensors(&legacy_source).unwrap();
    let mut legacy_gpu = Gpu::init().unwrap();
    let mut legacy_source_ref = qwen35::ParoSource::new(&legacy_source, &config).unwrap();
    let legacy = qwen35::load_weights(
        &mut legacy_source_ref,
        std::slice::from_mut(&mut legacy_gpu),
        &qwen35::Layout::single(config.n_layers),
    )
    .unwrap();
    let legacy_meta = collect_meta(&legacy, &config);
    let legacy_logits = one_logit_step(&mut legacy_gpu, &legacy, &config);
    legacy.free_gpu(&mut legacy_gpu);

    let manifest_source =
        hipfire_runtime::safetensors_source::SafetensorsSource::open(std::path::Path::new(&path))
            .unwrap();
    let mut manifest_gpu = Gpu::init().unwrap();
    // Paro MoE/A3B is intentionally excluded from the manifest path until
    // its separate gate/up payloads and layer-shared `paro_shared` ownership
    // are represented. Dense Paro remains the manifest parity path.
    let production = if config.num_experts > 0 {
        let mut paro_source = qwen35::ParoSource::new(&manifest_source, &config).unwrap();
        qwen35::load_weights(
            &mut paro_source,
            std::slice::from_mut(&mut manifest_gpu),
            &qwen35::Layout::single(config.n_layers),
        )
        .unwrap()
    } else {
        hipfire_arch_qwen35::load_qwen35_paro_weights(&manifest_source, &config, &mut manifest_gpu)
            .unwrap()
    };
    assert_eq!(legacy_meta, collect_meta(&production, &config));
    assert_eq!(
        legacy_logits,
        one_logit_step(&mut manifest_gpu, &production, &config)
    );
    production.free_gpu(&mut manifest_gpu);
}
