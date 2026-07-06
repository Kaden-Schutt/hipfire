// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Phase-3 store→forward bridge: load a REAL llama-family `.mq4` model's **whole**
//! weight set through the generic `fulfill_manifest` + `WeightStore` (a per-arch
//! `source` closure replaces the bespoke imperative loader), assemble a
//! `LlamaWeights` entirely from the store, and prove a real forward through it is
//! **logit-identical** to bespoke `Llama::load_weights`.
//!
//! The `source` closure mirrors the HFQ loader's per-tensor rule exactly:
//! `quant_type 1` (F16) and `2` (F32) → dequant to F32; every other (quantized)
//! type → raw bytes verbatim + its real `DType`. So norms/embed/lm_head (F16→F32
//! or raw-quant) and the projection matrices all land byte-for-byte as the
//! bespoke loader would store them. A byte+dtype spot-check on the projections
//! runs first; the full-model logit parity is the end-to-end proof (any wrong
//! byte / dtype / shape / embd_format would diverge the logits).
//!
//! Run: cargo run -p hipfire-runtime --release --example llama_store_load \
//!         [~/.hipfire/models/qwen3-0.6b-llama.mq4]

use hipfire_arch_llama::Llama;
use hipfire_hardware::DeviceMesh;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{
    self, f16_to_f32, EmbeddingFormat, KvCache, LayerWeights, LlamaWeights, WeightTensor,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::WeightEntry;
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;

/// HFQ `quant_type` byte → the `DType` the bespoke loader assigns for a
/// *quantized* (raw-uploaded) tensor. F16(1)/F32(2) are handled separately
/// (dequant to F32), so they are absent here.
fn qtype_to_dtype(q: u8) -> Option<DType> {
    Some(match q {
        0 => DType::Q4F16G64,
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        _ => return None,
    })
}

/// Logical manifest name (+ layer) → on-disk HFQ tensor name (HF/safetensors
/// convention, the `.mq4` / `load_weights_hfq` path).
fn on_disk(name: &str, layer: Option<usize>) -> Option<String> {
    Some(match (name, layer) {
        ("token_embd", None) => "model.embed_tokens.weight".to_string(),
        ("output_norm", None) => "model.norm.weight".to_string(),
        ("lm_head", None) => "lm_head.weight".to_string(),
        (n, Some(l)) => {
            let p = format!("model.layers.{l}");
            match n {
                "wq" => format!("{p}.self_attn.q_proj.weight"),
                "wk" => format!("{p}.self_attn.k_proj.weight"),
                "wv" => format!("{p}.self_attn.v_proj.weight"),
                "wo" => format!("{p}.self_attn.o_proj.weight"),
                "q_norm" => format!("{p}.self_attn.q_norm.weight"),
                "k_norm" => format!("{p}.self_attn.k_norm.weight"),
                "attn_norm" => format!("{p}.input_layernorm.weight"),
                "ffn_norm" => format!("{p}.post_attention_layernorm.weight"),
                "ffn_gate" => format!("{p}.mlp.gate_proj.weight"),
                "ffn_up" => format!("{p}.mlp.up_proj.weight"),
                "ffn_down" => format!("{p}.mlp.down_proj.weight"),
                _ => return None,
            }
        }
        _ => return None,
    })
}

/// F16 little-endian bytes → F32 bytes (the loader's `load_f16_tensor` dequant).
fn f16_bytes_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let f32: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    f32.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn readback(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let n = t.buf.size();
    let mut b = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("memcpy_dtoh");
    b
}

/// Move a resident tensor out of the store (panics if missing / an alias).
fn take_gpu(store: &mut WeightStore, name: &str, layer: Option<usize>) -> GpuTensor {
    match store.take(name, layer, 0) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("store missing resident {name}[{layer:?}]"),
    }
}

/// Logical `(m, k)` of a per-layer projection, from its manifest entry.
fn mk(manifest: &[WeightEntry], name: &str, l: usize) -> (usize, usize) {
    let e = manifest
        .iter()
        .find(|e| e.name == name && e.layer == Some(l))
        .unwrap_or_else(|| panic!("no manifest entry {name}[{l}]"));
    (e.logical_shape[0], e.logical_shape[1])
}

/// A `WeightTensor` wrapping a store buffer (its dtype is the store dtype).
fn wt(buf: GpuTensor, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        gpu_dtype: buf.dtype,
        buf,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

/// One greedy forward at pos 0 for token 1 → the logit vector.
fn forward_logits(gpu: &mut Gpu, w: &LlamaWeights, cfg: &llama::LlamaConfig) -> Vec<f32> {
    let mut kv =
        KvCache::new_gpu_q8(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 256).expect("kv");
    let scratch = <Llama as Architecture>::new_state(gpu, cfg).expect("scratch");
    llama::forward_scratch_embed(gpu, w, cfg, 1u32, 0, &scratch).expect("embed");
    llama::forward_scratch_compute(gpu, w, cfg, 0, &mut kv, &scratch).expect("compute");
    gpu.download_f32(&scratch.logits).expect("logits")
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/bjoern/.hipfire/models/qwen3-0.6b-llama.mq4".to_string());
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let cfg = Llama::config_from_hfq(&hfq).expect("config_from_hfq");
    let n_layers = cfg.n_layers;

    // Whole-model manifest.
    let manifest: Vec<WeightEntry> = Llama::weight_manifest(&cfg);
    let has_lm_head = hfq.tensor_data("lm_head.weight").is_some();

    // Pre-read every tensor's (bytes, dtype) mirroring the HFQ loader's rule:
    // F16/F32 → dequant to F32; else raw + real quant dtype. (Immutable HFQ
    // borrows finish here, before the &mut bespoke load below.)
    let mut src: HashMap<(String, Option<usize>), (Vec<u8>, DType)> = HashMap::new();
    for e in &manifest {
        // Tied lm_head: no lm_head.weight → bespoke dequants token_embd to F32.
        if e.name == "lm_head" && !has_lm_head {
            let (_, bytes) = hfq.tensor_data("model.embed_tokens.weight").unwrap();
            src.insert(
                (e.name.clone(), e.layer),
                (f16_bytes_to_f32_bytes(&bytes), DType::F32),
            );
            continue;
        }
        let name = on_disk(&e.name, e.layer)
            .unwrap_or_else(|| panic!("no on-disk name for {}[{:?}]", e.name, e.layer));
        let (info, bytes) = hfq
            .tensor_data(&name)
            .unwrap_or_else(|| panic!("HFQ missing tensor {name}"));
        let (blob, dt) = match info.quant_type {
            1 => (f16_bytes_to_f32_bytes(&bytes), DType::F32), // F16 → F32 dequant
            2 => (bytes.to_vec(), DType::F32),                 // already F32
            q => (
                bytes.to_vec(),
                qtype_to_dtype(q).unwrap_or_else(|| panic!("{name}: unhandled quant_type {q}")),
            ),
        };
        src.insert((e.name.clone(), e.layer), (blob, dt));
    }

    // Bespoke reference load + its forward logits.
    let mut gpu = Gpu::init().expect("Gpu::init");
    let bespoke_w = Llama::load_weights(&mut hfq, &cfg, &mut gpu).expect("bespoke load_weights");
    let ref_logits = forward_logits(&mut gpu, &bespoke_w, &cfg);
    let mut gpus = Gpus::single(gpu, n_layers);

    // Whole-model store-backed load.
    let mut store = fulfill_manifest(&manifest, &DeviceMesh::single(), n_layers, &gpus, |e| {
        src.get(&(e.name.clone(), e.layer))
            .cloned()
            .ok_or_else(|| format!("no source bytes for {}[{:?}]", e.name, e.layer))
    })
    .expect("fulfill_manifest");

    // Byte+dtype spot-check on the projections vs bespoke (localises a failure
    // to the source before the assembly/forward).
    let mut spot = 0usize;
    for l in 0..n_layers {
        for (lname, field) in [
            ("wq", &bespoke_w.layers[l].wq),
            ("wk", &bespoke_w.layers[l].wk),
            ("wv", &bespoke_w.layers[l].wv),
            ("wo", &bespoke_w.layers[l].wo),
            ("ffn_gate", &bespoke_w.layers[l].w_gate),
            ("ffn_up", &bespoke_w.layers[l].w_up),
            ("ffn_down", &bespoke_w.layers[l].w_down),
        ] {
            let st = match store.get(lname, Some(l), 0) {
                Some(WeightHandle::Resident(t)) => t,
                _ => panic!("store missing {lname}[{l}]"),
            };
            assert_eq!(st.dtype, field.gpu_dtype, "{lname}[{l}] dtype");
            assert!(
                readback(&gpus.devices[0], st) == readback(&gpus.devices[0], &field.buf),
                "{lname}[{l}] bytes differ from bespoke"
            );
            spot += 1;
        }
    }
    println!("llama_store_load: {spot} projection tensors byte+dtype identical to bespoke");

    // Assemble a LlamaWeights ENTIRELY from the store.
    let token_embd = take_gpu(&mut store, "token_embd", None);
    let embd_format = match token_embd.dtype {
        DType::Q8_0 => EmbeddingFormat::Q8_0,
        DType::Q4K => EmbeddingFormat::Q4K,
        DType::HFQ4G256 => EmbeddingFormat::HFQ4G256,
        DType::HFQ4G128 => EmbeddingFormat::HFQ4G128,
        DType::F32 => EmbeddingFormat::F32,
        other => panic!("unexpected token_embd dtype {other:?}"),
    };
    let output_norm = take_gpu(&mut store, "output_norm", None);
    let lm = take_gpu(&mut store, "lm_head", None);
    let output = wt(lm, cfg.vocab_size, cfg.dim);

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (wqm, wqk) = mk(&manifest, "wq", l);
        let (wkm, wkk) = mk(&manifest, "wk", l);
        let (wvm, wvk) = mk(&manifest, "wv", l);
        let (wom, wok) = mk(&manifest, "wo", l);
        let (gm, gk) = mk(&manifest, "ffn_gate", l);
        let (um, uk) = mk(&manifest, "ffn_up", l);
        let (dm, dk) = mk(&manifest, "ffn_down", l);
        layers.push(LayerWeights {
            attn_norm: take_gpu(&mut store, "attn_norm", Some(l)),
            wq: wt(take_gpu(&mut store, "wq", Some(l)), wqm, wqk),
            wk: wt(take_gpu(&mut store, "wk", Some(l)), wkm, wkk),
            wv: wt(take_gpu(&mut store, "wv", Some(l)), wvm, wvk),
            wo: wt(take_gpu(&mut store, "wo", Some(l)), wom, wok),
            q_norm: cfg
                .has_qk_norm
                .then(|| take_gpu(&mut store, "q_norm", Some(l))),
            k_norm: cfg
                .has_qk_norm
                .then(|| take_gpu(&mut store, "k_norm", Some(l))),
            ffn_norm: take_gpu(&mut store, "ffn_norm", Some(l)),
            w_gate: wt(take_gpu(&mut store, "ffn_gate", Some(l)), gm, gk),
            w_up: wt(take_gpu(&mut store, "ffn_up", Some(l)), um, uk),
            w_down: wt(take_gpu(&mut store, "ffn_down", Some(l)), dm, dk),
        });
    }
    let stored_w = LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        // The store's lm_head is its own resident buffer (reuploaded from embed
        // bytes in the tied case), never a view of token_embd — matching the llama
        // loader convention (model_load.rs: "llama always returns false").
        lm_head_aliases_embd: false,
    };

    // Forward through the store-assembled model and compare to bespoke.
    let asm_logits = forward_logits(&mut gpus.devices[0], &stored_w, &cfg);
    assert_eq!(asm_logits.len(), ref_logits.len(), "logit-length mismatch");
    let max_abs = asm_logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (asm_argmax, ref_argmax) = (argmax(&asm_logits), argmax(&ref_logits));
    assert!(
        asm_logits.iter().all(|x| x.is_finite()),
        "store-assembled forward produced non-finite logits"
    );
    assert_eq!(
        asm_argmax, ref_argmax,
        "argmax diverged: store {asm_argmax} vs bespoke {ref_argmax}"
    );
    assert!(
        max_abs == 0.0,
        "store-assembled logits differ from bespoke (max |Δ| = {max_abs})"
    );
    println!(
        "llama_store_load: WHOLE-MODEL store→forward OK — {} tensors fulfilled, forward \
         logit-IDENTICAL to bespoke (max |Δ|=0, argmax token {asm_argmax})",
        manifest.len()
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}
