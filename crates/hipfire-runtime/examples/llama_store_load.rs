// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Phase-3 walking skeleton: load a REAL llama-family `.mq4` model's quantized
//! projection weights through the generic `fulfill_manifest` + `WeightStore`,
//! and prove each placed tensor is **byte- and dtype-identical** to the bespoke
//! `Llama::load_weights` output. This is the "wire `WeightStore` into a real
//! load" bridge (device-mesh §4): the manifest + a per-arch `source` closure
//! replace the bespoke imperative loader for these tensors.
//!
//! Scope — the quantized attention/MLP projections (`wq/wk/wv/wo/ffn_gate/
//! ffn_up/ffn_down`), which the HFQ loader uploads **raw, verbatim** (no
//! transform), so a raw-byte fulfill matches byte-for-byte. Norms / embedding /
//! lm_head undergo an F16→F32 host dequant in the loader; reproducing that in
//! the `source` closure is a follow-up (the store already carries the real
//! dtype, so it is a source-side change only).
//!
//! Then the **consumption** half: assemble a `LlamaWeights` whose projections
//! are the store buffers (norms/embed/lm_head from the bespoke load) and run a
//! real forward through it — proving the store tensors' metadata (dtype/shape)
//! is end-to-end usable by the kernels, i.e. the store *feeds the forward*.
//!
//! Run: cargo run -p hipfire-runtime --release --example llama_store_load \
//!         [~/.hipfire/models/qwen3-0.6b-llama.mq4]

use hipfire_arch_llama::Llama;
use hipfire_hardware::DeviceMesh;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache, LayerWeights, LlamaWeights, WeightTensor};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;

/// Move a resident projection out of the store (for arch-weight assembly).
fn take_resident(store: &mut WeightStore, name: &str, l: usize) -> GpuTensor {
    match store.take(name, Some(l), 0) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("store missing {name}[{l}] for assembly"),
    }
}

/// Build a `WeightTensor` around a store-loaded buffer, reusing the bespoke
/// tensor's metadata (m/k/row_stride/paro/awq — not captured by the manifest).
/// Only `buf` (and its already-verified dtype) come from the store, so the
/// forward genuinely consumes the store buffer.
fn wt_from(buf: GpuTensor, b: WeightTensor) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype: b.gpu_dtype,
        m: b.m,
        k: b.k,
        row_stride: b.row_stride,
        paro: b.paro,
        awq_scale: b.awq_scale,
    }
}

/// HFQ `quant_type` byte → the `DType` the bespoke loader assigns
/// (`WeightTensor.gpu_dtype`). Covers the quantized projection formats; norms /
/// F16 tensors are out of scope here (they dequant to F32).
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

/// Logical manifest name → on-disk HFQ tensor name, for the quantized
/// projections only (`None` for everything else — norms, embed, lm_head).
fn on_disk(name: &str, layer: usize) -> Option<String> {
    let p = format!("model.layers.{layer}");
    Some(match name {
        "wq" => format!("{p}.self_attn.q_proj.weight"),
        "wk" => format!("{p}.self_attn.k_proj.weight"),
        "wv" => format!("{p}.self_attn.v_proj.weight"),
        "wo" => format!("{p}.self_attn.o_proj.weight"),
        "ffn_gate" => format!("{p}.mlp.gate_proj.weight"),
        "ffn_up" => format!("{p}.mlp.up_proj.weight"),
        "ffn_down" => format!("{p}.mlp.down_proj.weight"),
        _ => return None,
    })
}

/// The bespoke `WeightTensor` for a logical projection name.
fn bespoke<'a>(w: &'a LlamaWeights, name: &str, layer: usize) -> Option<&'a WeightTensor> {
    let l = w.layers.get(layer)?;
    Some(match name {
        "wq" => &l.wq,
        "wk" => &l.wk,
        "wv" => &l.wv,
        "wo" => &l.wo,
        "ffn_gate" => &l.w_gate,
        "ffn_up" => &l.w_up,
        "ffn_down" => &l.w_down,
        _ => return None,
    })
}

fn readback(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let n = t.buf.size();
    let mut b = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("memcpy_dtoh");
    b
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/bjoern/.hipfire/models/qwen3-0.6b-llama.mq4".to_string());
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let cfg = Llama::config_from_hfq(&hfq).expect("config_from_hfq");
    let n_layers = cfg.n_layers;

    // The manifest, filtered to the quantized projections this skeleton covers.
    let manifest: Vec<_> = Llama::weight_manifest(&cfg)
        .into_iter()
        .filter(|e| e.layer.is_some() && on_disk(&e.name, e.layer.unwrap()).is_some())
        .collect();

    // Pre-read each projection's raw bytes + real dtype (immutable HFQ borrows
    // finish here, before the &mut load below). Keyed (logical_name, layer).
    let mut src: HashMap<(String, usize), (Vec<u8>, DType)> = HashMap::new();
    for e in &manifest {
        let layer = e.layer.unwrap();
        let name = on_disk(&e.name, layer).unwrap();
        let (info, bytes) = hfq
            .tensor_data(&name)
            .unwrap_or_else(|| panic!("HFQ missing tensor {name}"));
        let dt = qtype_to_dtype(info.quant_type)
            .unwrap_or_else(|| panic!("{name}: unhandled quant_type {}", info.quant_type));
        src.insert((e.name.clone(), layer), (bytes.to_vec(), dt));
    }

    // Bespoke load (the reference), then wrap the device into a 1×1 Gpus.
    let mut gpu = Gpu::init().expect("Gpu::init");
    let bespoke_w = Llama::load_weights(&mut hfq, &cfg, &mut gpu).expect("bespoke load_weights");
    let mut gpus = Gpus::single(gpu, n_layers);

    // Store-backed load via the generic fulfill_manifest + our source closure.
    let mut store = fulfill_manifest(&manifest, &DeviceMesh::single(), n_layers, &gpus, |e| {
        src.get(&(e.name.clone(), e.layer.unwrap()))
            .cloned()
            .ok_or_else(|| format!("no source bytes for {}[{:?}]", e.name, e.layer))
    })
    .expect("fulfill_manifest");

    // Compare every fulfilled tensor to the bespoke WeightTensor: real dtype +
    // exact device bytes.
    let mut n_ok = 0usize;
    let mut example_dtype = None;
    for e in &manifest {
        let layer = e.layer.unwrap();
        let store_t = match store.get(&e.name, e.layer, 0) {
            Some(WeightHandle::Resident(t)) => t,
            _ => panic!("store missing {}[{layer}]", e.name),
        };
        let bt = bespoke(&bespoke_w, &e.name, layer).unwrap();
        assert_eq!(
            store_t.dtype, bt.gpu_dtype,
            "{}[{layer}] dtype mismatch: store {:?} vs bespoke {:?}",
            e.name, store_t.dtype, bt.gpu_dtype
        );
        let sb = readback(&gpus.devices[0], store_t);
        let bb = readback(&gpus.devices[0], &bt.buf);
        assert_eq!(
            sb.len(),
            bb.len(),
            "{}[{layer}] byte-length mismatch",
            e.name
        );
        assert!(
            sb == bb,
            "{}[{layer}] bytes differ from bespoke loader",
            e.name
        );
        example_dtype.get_or_insert(store_t.dtype);
        n_ok += 1;
    }

    println!(
        "llama_store_load: OK — {n_ok} quantized projection tensors across {n_layers} layers \
         byte+dtype identical to bespoke Llama::load_weights (e.g. {:?})",
        example_dtype.unwrap()
    );

    // ── Phase-3 CONSUMPTION: assemble a LlamaWeights whose projections come
    // from the store, then run a REAL forward through it. Byte-identity is
    // already proven, so this proves the store tensors' metadata (dtype/shape)
    // is end-to-end usable by the kernels — the store *feeds the forward*.
    // (Norms / embed / lm_head come from the bespoke load: their F16→F32-dequant
    // source is a documented follow-up; here we exercise the store buffers.)
    let LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
    } = bespoke_w;
    let mut new_layers = Vec::with_capacity(layers.len());
    for (l, bl) in layers.into_iter().enumerate() {
        let LayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        } = bl;
        new_layers.push(LayerWeights {
            attn_norm,
            wq: wt_from(take_resident(&mut store, "wq", l), wq),
            wk: wt_from(take_resident(&mut store, "wk", l), wk),
            wv: wt_from(take_resident(&mut store, "wv", l), wv),
            wo: wt_from(take_resident(&mut store, "wo", l), wo),
            q_norm,
            k_norm,
            ffn_norm,
            w_gate: wt_from(take_resident(&mut store, "ffn_gate", l), w_gate),
            w_up: wt_from(take_resident(&mut store, "ffn_up", l), w_up),
            w_down: wt_from(take_resident(&mut store, "ffn_down", l), w_down),
        });
    }
    let merged = LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers: new_layers,
    };

    let gpu = &mut gpus.devices[0];
    let mut kv = KvCache::new_gpu_q8(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 256)
        .expect("kv cache");
    let scratch = <Llama as Architecture>::new_state(gpu, &cfg).expect("scratch");
    llama::forward_scratch_embed(gpu, &merged, &cfg, 1u32, 0, &scratch).expect("forward embed");
    llama::forward_scratch_compute(gpu, &merged, &cfg, 0, &mut kv, &scratch)
        .expect("forward compute");
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");

    assert!(
        logits.iter().all(|x| x.is_finite()),
        "store-assembled forward produced non-finite logits"
    );
    let argmax = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0;
    assert!(
        argmax < cfg.vocab_size,
        "argmax {argmax} out of vocab {}",
        cfg.vocab_size
    );
    println!(
        "llama_store_load: forward on store-assembled weights OK — {} finite logits, argmax token \
         {argmax} (the real forward consumed the store's projection buffers)",
        logits.len()
    );
}
