// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Diagnostic-only harness for the `forward_batch_slots` golden-test failure
// (SP3 Task 3's `test_forward_slots_golden`, n_slots=1 slot=0 step=0).
// Not part of any gate. For each `max_layer` in 1..=n_layers, runs a FRESH
// reference `forward_prefill_batch_with_pbs_opts` call and a FRESH candidate
// `forward_batch_slots_with_max_layer` call — both bounded to that same
// `max_layer`, both replaying the identical 5-token stream from
// start_pos=0 on brand-new KV cache / DeltaNet state (fresh state is
// required: reusing state across iterations would double-apply the
// DeltaNet recurrence for already-processed layers) — and diffs
// `pbs.x_batch[0..n*dim]` to find the first layer where the two diverge.
//
// Usage:
//   cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 \
//     --example bisect_forward_slots -- <model.hf4>

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{forward_batch_slots_with_max_layer, SlotDescStaging};
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch,
    };
    use hipfire_arch_qwen35::slot_batch::SlotBatch;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bisect_forward_slots <model.hf4>");
        std::process::exit(1);
    });

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse Qwen3.5 config");
    let n_fa_layers = config.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count();
    let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34;
    let dim = config.dim;

    const PROMPT_LEN: usize = 5;
    const CAP_TOKENS: usize = 64;
    let tokens: Vec<u32> = (0..PROMPT_LEN).map(|i| ((i as u32) * 131 % 900) + 1).collect();

    let weight_bytes = std::fs::metadata(&model_path).expect("stat model file").len();
    let planned = weight_bytes + 2 * 1024 * 1024 * 1024u64; // weights + generous flat slop
    preflight_alloc(planned, R9700_VRAM_BYTES, "bisect_forward_slots").expect("preflight refused");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    println!("model: {} layers, dim={dim}, prompt_len={PROMPT_LEN}", config.n_layers);
    println!(
        "{:>3} {:>16} {:>14} {:>14} {:>10}",
        "L", "type", "max|ref|", "max|ref-cand|", "rel"
    );

    let kv_seq = (PROMPT_LEN + 16).max(CAP_TOKENS).max(512);

    for max_layer in 1..=config.n_layers {
        // ---- reference: fresh KvCache + DeltaNetState + scratch every time ----
        let mut ref_kv =
            KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
                .expect("ref KvCache");
        let mut ref_dn = DeltaNetState::new(&mut gpu, &config).expect("ref DeltaNetState");
        let ref_scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq)
            .expect("ref Qwen35Scratch");
        let ref_pbs = PrefillBatchScratch::new(&mut gpu, &config, PROMPT_LEN).expect("ref PrefillBatchScratch");

        qwen35::forward_prefill_batch_with_pbs_opts(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut ref_kv,
            &mut ref_dn,
            &ref_scratch,
            None,
            None,
            None,
            None,
            Some(&ref_pbs),
            None,
            Some(max_layer),
            false,
        )
        .expect("reference forward (bounded)");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_x = gpu.download_f32(&ref_pbs.x_batch.sub_offset(0, PROMPT_LEN * dim)).expect("dl ref x");

        ref_kv.free_gpu(&mut gpu).expect("free ref_kv");
        ref_dn.free_gpu(&mut gpu);
        ref_scratch.free_gpu(&mut gpu);
        ref_pbs.free_gpu(&mut gpu);

        // ---- candidate: fresh SlotPool/arenas/DeltaNetState/scratch every time ----
        let mut pool = SlotPool::new(1, CAP_TOKENS, per_pos_bytes).expect("SlotPool::new");
        let slot0 = pool.acquire().expect("acquire slot 0");
        assert_eq!(slot0.0, 0);
        let arena_bytes = pool.arena_bytes();
        let k_arenas: Vec<GpuTensor> =
            (0..n_fa_layers).map(|_| gpu.zeros(&[arena_bytes], DType::Raw).expect("k_arena")).collect();
        let v_arenas: Vec<GpuTensor> =
            (0..n_fa_layers).map(|_| gpu.zeros(&[arena_bytes], DType::Raw).expect("v_arena")).collect();
        let mut dn_states = vec![DeltaNetState::new(&mut gpu, &config).expect("cand DeltaNetState")];
        let mut desc_staging = SlotDescStaging::new(&mut gpu, 1, PROMPT_LEN).expect("SlotDescStaging");
        let cand_pbs = PrefillBatchScratch::new(&mut gpu, &config, PROMPT_LEN).expect("cand PrefillBatchScratch");
        let cand_scratch =
            Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CAP_TOKENS).expect("cand Qwen35Scratch");
        let logits_out = gpu.zeros(&[config.vocab_size], DType::F32).expect("logits_out");
        let batch = SlotBatch::build(&[(SlotId(0), &tokens[..], 0usize)]);

        forward_batch_slots_with_max_layer(
            &mut gpu,
            &weights,
            &config,
            &batch,
            &mut pool,
            &mut dn_states,
            &k_arenas,
            &v_arenas,
            &mut desc_staging,
            &cand_pbs,
            &cand_scratch,
            &logits_out,
            Some(max_layer),
        )
        .expect("candidate forward (bounded)");
        gpu.hip.device_synchronize().expect("sync cand");
        let cand_x = gpu.download_f32(&cand_pbs.x_batch.sub_offset(0, PROMPT_LEN * dim)).expect("dl cand x");

        for t in k_arenas {
            gpu.free_tensor(t).expect("free k_arena");
        }
        for t in v_arenas {
            gpu.free_tensor(t).expect("free v_arena");
        }
        for dn in dn_states {
            dn.free_gpu(&mut gpu);
        }
        desc_staging.free_gpu(&mut gpu);
        cand_pbs.free_gpu(&mut gpu);
        cand_scratch.free_gpu(&mut gpu);
        gpu.free_tensor(logits_out).expect("free logits_out");

        let max_ref = ref_x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let max_diff = ref_x
            .iter()
            .zip(cand_x.iter())
            .fold(0.0f32, |a, (&r, &c)| a.max((r - c).abs()));
        let rel = max_diff / max_ref.max(1e-6);
        let lt = config.layer_types[max_layer - 1];
        println!("{max_layer:>3} {lt:>16?} {max_ref:>14.6} {max_diff:>14.6} {rel:>10.4}");
        if rel > 0.02 {
            println!("  >>> first material divergence at layer {max_layer} ({lt:?})");
            break;
        }
    }

    println!("done");
}
