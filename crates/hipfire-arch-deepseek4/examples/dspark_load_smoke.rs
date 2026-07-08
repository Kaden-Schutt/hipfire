//! dspark_load_smoke: load ONLY the DSpark sidecar HFQ through
//! `DeepseekV4::load_dspark` and verify every stage + DSpark head tensor lands
//! on the GPU, then free. Exercises the loader in isolation (~5.6 GB) without
//! the 83 GB trunk. No inference.
//!
//! Usage: dspark_load_smoke [path-to-dspark.mq2lloyd]

use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/deepseek-v4-flash-dspark.mq2lloyd",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    eprintln!("opening {path}");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    hfq.drop_mmap();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;

    let dspark = DeepseekV4::load_dspark(&hfq, &mut gpu, &cfg)?
        .ok_or_else(|| "load_dspark returned None — no dspark_* config in metadata".to_string())?;

    println!("OK: {} stages loaded", dspark.stages.len());
    println!(
        "  cfg: block_size={} target_layers={:?} markov_rank={} noise_tok={}",
        dspark.cfg.block_size,
        dspark.cfg.target_layer_ids,
        dspark.cfg.markov_rank,
        dspark.cfg.noise_token_id,
    );
    let mut all_ok = true;
    for (i, s) in dspark.stages.iter().enumerate() {
        let dense_ok = s.attn_norm.is_some()
            && s.wq_a.is_some()
            && s.wq_b.is_some()
            && s.wkv.is_some()
            && s.wo_a.is_some()
            && s.wo_b.is_some()
            && s.attn_sink.is_some()
            && s.hc_attn_fn.is_some()
            && s.hc_ffn_fn.is_some()
            && s.gate_weight.is_some()
            && s.shared_w1.is_some();
        // Routed experts use the fused gate_up + w2 layout (w1/w3 fold into
        // expert_gate_up_blob), so check those, not expert_w1_blob.
        let experts_ok = s.expert_gate_up_blob.is_some() && s.expert_w2_blob.is_some();
        let is_last = i == dspark.stages.len() - 1;
        let head_ok = !is_last
            || (s.mtp_hc_head_fn.is_some()
                && s.mtp_hc_head_base.is_some()
                && s.mtp_final_norm.is_some());
        println!("  stage {i}: dense_ok={dense_ok} experts_ok={experts_ok} head_ok={head_ok}");
        all_ok &= dense_ok && experts_ok && head_ok;
    }
    let globals_ok = dspark.main_proj.is_some()
        && dspark.main_norm.is_some()
        && dspark.markov_w1.is_some()
        && dspark.markov_w2.is_some()
        && dspark.confidence_proj.is_some();
    println!(
        "  globals: main_proj={} main_norm={} markov_w1={} markov_w2={} confidence={}",
        dspark.main_proj.is_some(),
        dspark.main_norm.is_some(),
        dspark.markov_w1.is_some(),
        dspark.markov_w2.is_some(),
        dspark.confidence_proj.is_some(),
    );
    all_ok &= globals_ok;

    dspark.free_gpu(&mut gpu);
    println!("freed OK");

    if all_ok {
        println!("SMOKE PASS");
        Ok(())
    } else {
        Err("SMOKE FAIL — some tensors missing".into())
    }
}
