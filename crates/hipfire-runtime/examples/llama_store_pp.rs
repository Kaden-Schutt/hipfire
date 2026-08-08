// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Phase-3 PP-2 store placement: fulfill the WHOLE llama manifest on an emulated
//! 2-stage pipeline mesh (`HIPFIRE_EMULATE_GPUS=2`), assert every tensor lands on
//! the mesh-correct pipeline stage (embed→stage 0, final-norm+lm_head→last stage,
//! layers banded by `stage_for_layer`), and prove the placed weights are
//! forward-usable by gathering them into a single `LlamaWeights` and running a
//! forward that is **logit-identical** to bespoke.
//!
//! Then the *execution* half: a REAL banded pipeline forward — stage 0 runs
//! embed + `forward_scratch_band(0..k)` on device 0, hands the residual to
//! device 1 via `Gpus::boundary_copy`, and stage 1 runs `forward_scratch_band(k..n)`
//! + `forward_scratch_head` on device 1. Each band touches only its stage's
//! layers, which live on that stage's device. The result is logit-identical to
//! bespoke — full pipeline-parallel load + execute on a real model.
//!
//! Run: cargo run -p hipfire-runtime --release --example llama_store_pp \
//!         [~/.hipfire/models/qwen3-0.6b-llama.mq4]

use hipfire_arch_llama::Llama;
use hipfire_hardware::DimKind;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::config::resolve_mesh;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{
    self, f16_to_f32, EmbeddingFormat, KvCache, LayerWeights, LlamaWeights, WeightTensor,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{placement_devices, WeightEntry};
use hipfire_runtime::weight_store::{fulfill_manifest, WeightHandle, WeightStore};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::path::Path;

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

fn f16_bytes_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let f32: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    f32.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn take_gpu(store: &mut WeightStore, name: &str, layer: Option<usize>, device: usize) -> GpuTensor {
    match store.take(name, layer, device) {
        Some(WeightHandle::Resident(t)) => t,
        _ => panic!("store missing resident {name}[{layer:?}] on device {device}"),
    }
}

fn mk(manifest: &[WeightEntry], name: &str, l: usize) -> (usize, usize) {
    let e = manifest
        .iter()
        .find(|e| e.name == name && e.layer == Some(l))
        .unwrap();
    (e.logical_shape[0], e.logical_shape[1])
}

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

fn forward_logits(gpu: &mut Gpu, w: &LlamaWeights, cfg: &llama::LlamaConfig) -> Vec<f32> {
    let mut kv =
        KvCache::new_gpu_q8(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 256).expect("kv");
    let scratch = <Llama as Architecture>::new_state(gpu, cfg).expect("scratch");
    llama::forward_scratch_embed(gpu, w, cfg, 1u32, 0, &scratch).expect("embed");
    llama::forward_scratch_compute(gpu, w, cfg, 0, &mut kv, &scratch).expect("compute");
    gpu.download_f32(&scratch.logits).expect("logits")
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

fn main() {
    // Emulate a 2-GPU box on one physical card (both stages alias device 0).
    // Must be set before the first RuntimeConfig read / device bring-up.
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/bjoern/.hipfire/models/qwen3-0.6b-llama.mq4".to_string());
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let cfg = Llama::config_from_hfq(&hfq).expect("config_from_hfq");
    let n_layers = cfg.n_layers;
    let manifest: Vec<WeightEntry> = Llama::weight_manifest(&cfg);
    let has_lm_head = hfq.tensor_data("lm_head.weight").is_some();

    // Universal source (same rule as the HFQ loader): F16/F32 → F32; else raw.
    let mut src: HashMap<(String, Option<usize>), (Vec<u8>, DType)> = HashMap::new();
    for e in &manifest {
        if e.name == "lm_head" && !has_lm_head {
            let (_, bytes) = hfq.tensor_data("model.embed_tokens.weight").unwrap();
            src.insert(
                (e.name.clone(), e.layer),
                (f16_bytes_to_f32_bytes(&bytes), DType::F32),
            );
            continue;
        }
        let name = on_disk(&e.name, e.layer).unwrap();
        let (info, bytes) = hfq.tensor_data(&name).unwrap();
        let (blob, dt) = match info.quant_type {
            1 => (f16_bytes_to_f32_bytes(&bytes), DType::F32),
            2 => (bytes.to_vec(), DType::F32),
            q => (bytes.to_vec(), qtype_to_dtype(q).unwrap()),
        };
        src.insert((e.name.clone(), e.layer), (blob, dt));
    }

    // 2-stage pipeline: init_uniform(2) aliases both stages onto device 0.
    let mut gpus = Gpus::init_uniform(2, n_layers).expect("init_uniform(2) — emulated");
    let bespoke_w =
        Llama::load_weights(&mut hfq, &cfg, &mut gpus.devices[0]).expect("bespoke load_weights");
    let ref_logits = forward_logits(&mut gpus.devices[0], &bespoke_w, &cfg);

    let mesh = resolve_mesh(2, 1, 1, Some(2));
    assert!(mesh.has_axis(DimKind::Pp), "expected a Pp mesh");
    assert_eq!(mesh.n_devices(), 2);

    // Fulfill the WHOLE model across the pipeline stages.
    let mut store = fulfill_manifest(&manifest, &mesh, n_layers, &gpus, |e| {
        src.get(&(e.name.clone(), e.layer))
            .cloned()
            .ok_or_else(|| format!("no source for {}[{:?}]", e.name, e.layer))
    })
    .expect("fulfill_manifest (PP-2)");

    // Placement: every tensor landed on exactly the mesh-correct stage device.
    let mut s0 = 0usize;
    let mut s1 = 0usize;
    for e in &manifest {
        let expected = placement_devices(e, &mesh, n_layers);
        assert_eq!(
            store.devices_for(&e.name, e.layer),
            expected,
            "{}[{:?}] placed on wrong stage",
            e.name,
            e.layer
        );
        match expected.as_slice() {
            [0] => s0 += 1,
            [1] => s1 += 1,
            other => panic!("unexpected PP placement {other:?} for {}", e.name),
        }
    }
    // Sanity: embed on stage 0, output-norm + lm_head on the last stage, layers
    // split by the uniform band.
    assert_eq!(
        placement_devices(
            manifest.iter().find(|e| e.name == "token_embd").unwrap(),
            &mesh,
            n_layers
        ),
        vec![0]
    );
    assert_eq!(
        placement_devices(
            manifest.iter().find(|e| e.name == "output_norm").unwrap(),
            &mesh,
            n_layers
        ),
        vec![1]
    );
    assert_eq!(
        placement_devices(
            manifest.iter().find(|e| e.name == "lm_head").unwrap(),
            &mesh,
            n_layers
        ),
        vec![1]
    );
    println!(
        "llama_store_pp: PP-2 placement OK — {s0} tensors on stage 0, {s1} on stage 1 \
         (embed→0, output-norm+lm_head→1, {n_layers} layers banded by stage_for_layer)"
    );

    // Gather the pipeline-placed weights into one LlamaWeights (taking each from
    // the stage device it landed on) and run a forward — proves the placed
    // tensors are forward-usable and byte-correct on their stages.
    let dev_of = |name: &str, layer: Option<usize>| -> usize {
        let e = manifest
            .iter()
            .find(|e| e.name == name && e.layer == layer)
            .unwrap();
        placement_devices(e, &mesh, n_layers)[0]
    };

    let token_embd = take_gpu(&mut store, "token_embd", None, dev_of("token_embd", None));
    let embd_format = match token_embd.dtype {
        DType::Q8_0 => EmbeddingFormat::Q8_0,
        DType::Q4K => EmbeddingFormat::Q4K,
        DType::HFQ4G256 => EmbeddingFormat::HFQ4G256,
        DType::HFQ4G128 => EmbeddingFormat::HFQ4G128,
        DType::F32 => EmbeddingFormat::F32,
        other => panic!("unexpected token_embd dtype {other:?}"),
    };
    let output_norm = take_gpu(&mut store, "output_norm", None, dev_of("output_norm", None));
    let lm = take_gpu(&mut store, "lm_head", None, dev_of("lm_head", None));
    let output = wt(lm, cfg.vocab_size, cfg.dim);

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let d = dev_of("wq", Some(l)); // all of layer l's weights share its stage
        let (wqm, wqk) = mk(&manifest, "wq", l);
        let (wkm, wkk) = mk(&manifest, "wk", l);
        let (wvm, wvk) = mk(&manifest, "wv", l);
        let (wom, wok) = mk(&manifest, "wo", l);
        let (gm, gk) = mk(&manifest, "ffn_gate", l);
        let (um, uk) = mk(&manifest, "ffn_up", l);
        let (dm, dk) = mk(&manifest, "ffn_down", l);
        layers.push(LayerWeights {
            attn_norm: take_gpu(&mut store, "attn_norm", Some(l), d),
            wq: wt(take_gpu(&mut store, "wq", Some(l), d), wqm, wqk),
            wk: wt(take_gpu(&mut store, "wk", Some(l), d), wkm, wkk),
            wv: wt(take_gpu(&mut store, "wv", Some(l), d), wvm, wvk),
            wo: wt(take_gpu(&mut store, "wo", Some(l), d), wom, wok),
            q_norm: cfg
                .has_qk_norm
                .then(|| take_gpu(&mut store, "q_norm", Some(l), d)),
            k_norm: cfg
                .has_qk_norm
                .then(|| take_gpu(&mut store, "k_norm", Some(l), d)),
            ffn_norm: take_gpu(&mut store, "ffn_norm", Some(l), d),
            w_gate: wt(take_gpu(&mut store, "ffn_gate", Some(l), d), gm, gk),
            w_up: wt(take_gpu(&mut store, "ffn_up", Some(l), d), um, uk),
            w_down: wt(take_gpu(&mut store, "ffn_down", Some(l), d), dm, dk),
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

    // (a) Sanity: one gathered forward on device 0 (all layers are physically
    // co-resident under emulation) — logit-identical to bespoke.
    let asm_logits = forward_logits(&mut gpus.devices[0], &stored_w, &cfg);
    assert_eq!(
        argmax(&asm_logits),
        argmax(&ref_logits),
        "gathered argmax diverged"
    );
    assert!(
        asm_logits.iter().zip(&ref_logits).all(|(a, b)| a == b),
        "gathered logits differ from bespoke"
    );
    println!("llama_store_pp: gathered forward logit-identical to bespoke");

    // (b) REAL banded pipeline forward. Stage 0 runs embed + band(0..k) on device
    // 0 using its own weights, hands the residual `scratch.x` to device 1 via
    // `boundary_copy`, and stage 1 runs band(k..n) + head on device 1 — each
    // band touches only its stage's layers, which live on that stage's device.
    let k = (0..n_layers)
        .find(|&l| mesh.stage_for_layer(l, n_layers) == 1)
        .unwrap_or(n_layers);

    let scratch0 =
        <Llama as Architecture>::new_state(&mut gpus.devices[0], &cfg).expect("scratch0");
    let scratch1 =
        <Llama as Architecture>::new_state(&mut gpus.devices[1], &cfg).expect("scratch1");
    let mut kv0 = KvCache::new_gpu_q8(
        &mut gpus.devices[0],
        n_layers,
        cfg.n_kv_heads,
        cfg.head_dim,
        256,
    )
    .expect("kv0");
    let mut kv1 = KvCache::new_gpu_q8(
        &mut gpus.devices[1],
        n_layers,
        cfg.n_kv_heads,
        cfg.head_dim,
        256,
    )
    .expect("kv1");
    let _ = gpus.enable_peer_all().expect("enable_peer_all");

    llama::forward_scratch_embed(&mut gpus.devices[0], &stored_w, &cfg, 1u32, 0, &scratch0)
        .expect("embed (stage 0)");
    llama::forward_scratch_band(
        &mut gpus.devices[0],
        &stored_w,
        &cfg,
        0..k,
        0,
        &mut kv0,
        &scratch0,
    )
    .expect("band (stage 0)");
    // Residual hand-off, stage 0 → stage 1.
    let evt = gpus
        .boundary_copy(0, 1, &scratch0.x.buf, &scratch1.x.buf, cfg.dim * 4)
        .expect("boundary_copy");
    gpus.wait_boundary(evt).expect("wait_boundary");
    llama::forward_scratch_band(
        &mut gpus.devices[1],
        &stored_w,
        &cfg,
        k..n_layers,
        0,
        &mut kv1,
        &scratch1,
    )
    .expect("band (stage 1)");
    llama::forward_scratch_head(&mut gpus.devices[1], &stored_w, &cfg, &scratch1)
        .expect("head (stage 1)");
    let pp_logits = gpus.devices[1]
        .download_f32(&scratch1.logits)
        .expect("pp logits");

    let max_abs = pp_logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        pp_logits.iter().all(|x| x.is_finite()),
        "banded PP non-finite logits"
    );
    assert_eq!(
        argmax(&pp_logits),
        argmax(&ref_logits),
        "banded PP argmax diverged"
    );
    assert!(
        max_abs == 0.0,
        "banded PP logits differ from bespoke (max |Δ|={max_abs})"
    );
    println!(
        "llama_store_pp: REAL banded PP forward OK — stage0 layers 0..{k} on dev0 → boundary_copy \
         → stage1 layers {k}..{n_layers}+head on dev1, logit-IDENTICAL to bespoke (max |Δ|=0, \
         argmax token {})",
        argmax(&pp_logits)
    );
}
