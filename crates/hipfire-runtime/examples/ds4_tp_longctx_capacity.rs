// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 TP compressed-cache capacity probe.
//!
//! Loads the production EP/TP route once, grows the request-owned F32
//! compressor caches through a list of capacities, and records per-rank VMM
//! reserve/mapping plus physical VRAM. It intentionally performs no prefill;
//! use `scripts/serve_harness.py` for coherent long-context generation.

use hipfire_arch_deepseek4::forward::ensure_request_capacity;
use hipfire_arch_deepseek4::DeepseekV4State;
use hipfire_loader::{load_model_ep, EpArch};
use rdna_compute::Gpu;

const DEFAULT_TOKENS: &[usize] = &[20_480, 81_920, 1_048_576];

struct Args {
    model: String,
    tp: usize,
    tokens: Vec<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut tp = 3usize;
    let mut tokens = DEFAULT_TOKENS.to_vec();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        let value = |index: usize, flag: &str| {
            argv.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match argv[i].as_str() {
            "--model" => {
                model = Some(value(i, "--model")?);
                i += 2;
            }
            "--tp" => {
                tp = value(i, "--tp")?
                    .parse()
                    .map_err(|_| "invalid --tp".to_string())?;
                i += 2;
            }
            "--tokens" => {
                tokens = value(i, "--tokens")?
                    .split(',')
                    .map(|raw| {
                        raw.parse::<usize>()
                            .map_err(|_| format!("invalid --tokens entry {raw:?}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            flag => return Err(format!("unknown argument {flag}")),
        }
    }
    if tp == 0 || tokens.is_empty() || tokens.contains(&0) {
        return Err("--tp and every --tokens entry must be nonzero".to_string());
    }
    if !tokens.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("--tokens entries must be strictly increasing".to_string());
    }
    Ok(Args {
        model: model.ok_or_else(|| "--model is required".to_string())?,
        tp,
        tokens,
    })
}

#[derive(Clone, Copy)]
struct CacheSummary {
    tensors: usize,
    vmm_tensors: usize,
    dense_tensors: usize,
    logical_bytes: usize,
    mapped_bytes: usize,
    pointer_hash: u64,
}

fn mix_hash(hash: &mut u64, value: usize) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn cache_summary(state: &DeepseekV4State, gpu: &Gpu) -> CacheSummary {
    let mut summary = CacheSummary {
        tensors: 0,
        vmm_tensors: 0,
        dense_tensors: 0,
        logical_bytes: 0,
        mapped_bytes: 0,
        pointer_hash: 0xcbf29ce484222325,
    };
    for layer in &state._indexer {
        for tensor in [&layer.main_kv_cache, &layer.indexer_kv_cache]
            .into_iter()
            .flatten()
        {
            summary.tensors += 1;
            summary.logical_bytes += tensor.byte_size();
            mix_hash(&mut summary.pointer_hash, tensor.buf.as_ptr() as usize);
            if let Some(mapped) = gpu.vmm_mapped_bytes(tensor) {
                summary.vmm_tensors += 1;
                summary.mapped_bytes += mapped;
            } else {
                summary.dense_tensors += 1;
                summary.mapped_bytes += tensor.buf.size();
            }
        }
    }
    summary
}

fn report_rank(stage: &str, rank: usize, state: &DeepseekV4State, gpu: &Gpu, pbs_rows: usize) {
    let summary = cache_summary(state, gpu);
    let (free_bytes, total_bytes) = gpu.hip.get_vram_info().expect("hipMemGetInfo");
    let used_bytes = total_bytes.saturating_sub(free_bytes);
    println!(
        "RANK stage={stage} rank={rank} device={} arch={} prepared_tokens={} active_rows={} pbs_rows={} cache_tensors={} vmm_tensors={} dense_tensors={} logical_bytes={} mapped_bytes={} pointer_hash=0x{:016x} used_bytes={} free_bytes={} total_bytes={}",
        gpu.device_id,
        gpu.arch,
        state.compressor_capacity.prepared_tokens(),
        state.compressor_capacity.active_rows(),
        pbs_rows,
        summary.tensors,
        summary.vmm_tensors,
        summary.dense_tensors,
        summary.logical_bytes,
        summary.mapped_bytes,
        summary.pointer_hash,
        used_bytes,
        free_bytes,
        total_bytes,
    );
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let max_seq = *args.tokens.last().expect("nonempty tokens");
    println!(
        "CAPACITY_PROBE model={} tp={} max_seq={} tokens={:?}",
        args.model, args.tp, max_seq, args.tokens
    );
    let mut loaded = load_model_ep(&args.model, max_seq, args.tp)?;
    let ep = loaded
        .ep
        .as_mut()
        .ok_or_else(|| "loader did not produce EP state".to_string())?;
    let EpArch::Ds4 {
        config,
        state,
        prefill,
        ..
    } = &mut ep.inner
    else {
        return Err("loader returned a non-DeepSeek EP route".to_string());
    };
    if ep.gpus.devices.len() != args.tp
        || state.len() != args.tp
        || prefill.len() != args.tp
        || ep
            .gpus
            .devices
            .iter()
            .any(|gpu| !gpu.arch_caps.is_gfx1201())
    {
        return Err(format!(
            "probe requires exact gfx1201 TP{} (devices={}, state={}, prefill={})",
            args.tp,
            ep.gpus.devices.len(),
            state.len(),
            prefill.len(),
        ));
    }

    for rank in 0..args.tp {
        report_rank(
            "loaded",
            rank,
            &state[rank],
            &ep.gpus.devices[rank],
            prefill[rank].idx_score_capacity,
        );
    }

    for required_tokens in args.tokens {
        let mut errors = Vec::new();
        for rank in 0..args.tp {
            ep.gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("bind rank {rank}: {error:?}"))?;
            if let Err(error) = ensure_request_capacity(
                config,
                &mut state[rank],
                &mut ep.gpus.devices[rank],
                &mut prefill[rank],
                required_tokens,
            ) {
                errors.push(format!("rank {rank}: {error}"));
                break;
            }
        }
        let status = if errors.is_empty() {
            "pass"
        } else {
            "rejected"
        };
        println!(
            "CAPACITY_RESULT tokens={required_tokens} status={status} errors={:?}",
            errors
        );
        for rank in 0..args.tp {
            report_rank(
                &format!("tokens_{required_tokens}"),
                rank,
                &state[rank],
                &ep.gpus.devices[rank],
                prefill[rank].idx_score_capacity,
            );
        }
        if !errors.is_empty() {
            break;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CAPACITY_PROBE status=fail error={error:?}");
        std::process::exit(2);
    }
}
