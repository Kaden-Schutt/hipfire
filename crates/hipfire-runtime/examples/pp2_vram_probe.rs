//! Real-measurement VRAM probe for the multi-GPU PP=2 path. Loads the
//! given model with `load_weights_multi`, builds `Qwen35ScratchSet` +
//! `KvCache::new_gpu_asym3_capped_multi` + `DeltaNetState` (Q8), and
//! reports per-card VRAM deltas at each stage. Output is a markdown
//! table row suitable for `docs/multi-gpu.md`.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p hipfire-runtime \
//!         --release --features deltanet --example pp2_vram_probe -- \
//!         ~/.hipfire/models/qwen3.5-0.8b.mq4 [max_seq=4096]

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35ScratchSet, StateQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use std::path::Path;

fn used_gb(gpus: &Gpus, baseline_free: &[(usize, usize)]) -> Vec<f64> {
    (0..gpus.devices.len())
        .map(|i| {
            let (free, _) = gpus.devices[i].hip.get_vram_info().unwrap_or((0, 0));
            (baseline_free[i].0 as f64 - free as f64) / 1e9
        })
        .collect()
}

fn fmt_per_card(label: &str, used: &[f64]) {
    print!("{label:>16}:");
    for (i, u) in used.iter().enumerate() {
        print!("  dev{i}={u:6.3} GiB");
    }
    println!();
}

fn main() {
    let path = std::env::args().nth(1).expect("Usage: ... <model.mq4> [max_seq]");
    let max_seq: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config_from_hfq");
    eprintln!(
        "{}: layers={}, dim={}, hidden={}, vocab={}, kv_heads={}, head_dim={}, max_seq={max_seq}",
        Path::new(&path).file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        config.n_layers,
        config.dim,
        config.hidden_dim,
        config.vocab_size,
        config.n_kv_heads,
        config.head_dim,
    );

    let mut gpus = Gpus::init_uniform(2, config.n_layers).expect("init_uniform");
    let baseline_free: Vec<(usize, usize)> = (0..gpus.devices.len())
        .map(|i| gpus.devices[i].hip.get_vram_info().unwrap_or((0, 0)))
        .collect();
    println!("\n── per-card VRAM deltas (PP=2 split) ─────────────────────");
    let mut after = used_gb(&gpus, &baseline_free);
    fmt_per_card("baseline", &after);

    // Stage 1: weights via load_weights_multi
    let weights = qwen35::load_weights_multi(&hfq, &config, &mut gpus).expect("load_weights_multi");
    let after_weights = used_gb(&gpus, &baseline_free);
    let weights_delta: Vec<f64> = after_weights
        .iter()
        .zip(after.iter())
        .map(|(a, b)| a - b)
        .collect();
    fmt_per_card("after weights", &after_weights);
    fmt_per_card("Δ weights", &weights_delta);
    after = after_weights;

    // Stage 2: ScratchSet
    let scratch_set =
        Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 64, max_seq).expect("scratch");
    let after_scratch = used_gb(&gpus, &baseline_free);
    let scratch_delta: Vec<f64> = after_scratch
        .iter()
        .zip(after.iter())
        .map(|(a, b)| a - b)
        .collect();
    fmt_per_card("after scratch", &after_scratch);
    fmt_per_card("Δ scratch", &scratch_delta);
    after = after_scratch;

    // Stage 3: KvCache asym3 capped
    let kv = KvCache::new_gpu_asym3_capped_multi(
        &mut gpus,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_seq,
        max_seq,
    )
    .expect("kv asym3");
    let after_kv = used_gb(&gpus, &baseline_free);
    let kv_delta: Vec<f64> = after_kv.iter().zip(after.iter()).map(|(a, b)| a - b).collect();
    fmt_per_card("after KV", &after_kv);
    fmt_per_card("Δ KV", &kv_delta);
    after = after_kv;

    // Stage 4: DeltaNetState
    let (dn, la_to_device) = DeltaNetState::new_with_quant_multi(&mut gpus, &config, StateQuant::Q8)
        .expect("dn multi");
    let after_dn = used_gb(&gpus, &baseline_free);
    let dn_delta: Vec<f64> = after_dn.iter().zip(after.iter()).map(|(a, b)| a - b).collect();
    fmt_per_card("after DN state", &after_dn);
    fmt_per_card("Δ DN state", &dn_delta);

    // Aggregate and emit a markdown row matching docs/multi-gpu.md schema:
    //   model | quant | n_layers | dim | KV mode | ctx |
    //   weights | KV | scratch | total | per-card PP=2 | fits 24 GB?
    let weights_max = weights_delta.iter().cloned().fold(0.0_f64, f64::max);
    let kv_max = kv_delta.iter().cloned().fold(0.0_f64, f64::max);
    let scratch_dn_max = scratch_delta
        .iter()
        .zip(dn_delta.iter())
        .map(|(s, d)| s + d)
        .fold(0.0_f64, f64::max);
    let total_max = after_dn.iter().cloned().fold(0.0_f64, f64::max);
    let total_sum: f64 = after_dn.iter().sum();

    let quant_tag = if path.contains(".mq4") { "mq4" } else if path.contains(".mq3") { "mq3" } else { "?" };
    let model_tag = Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?");

    println!("\n── markdown row ──────────────────────────────────────────");
    println!(
        "| {} | {} | {} | {} | asym3 | {} | {:.1} GB | {} MB | {} MB | {:.1} GB | {:.1} GB | {} |",
        model_tag,
        quant_tag,
        config.n_layers,
        config.dim,
        max_seq,
        total_sum,
        (kv_max * 1000.0).round() as i64,
        (scratch_dn_max * 1000.0).round() as i64,
        total_sum,
        total_max,
        if total_max < 24.0 { "yes" } else { "tight" },
    );
    println!(
        "(weights/card max={:.2} GB, kv/card max={:.3} GB, scratch+dn/card max={:.3} GB)",
        weights_max, kv_max, scratch_dn_max,
    );

    // Cleanup
    dn.free_gpu_multi(&mut gpus, &la_to_device);
    kv.free_gpu_multi(&mut gpus);
    scratch_set.free_gpu_multi(&mut gpus);
    weights.free_gpu_multi(&mut gpus);
    eprintln!("(freed)");
}
