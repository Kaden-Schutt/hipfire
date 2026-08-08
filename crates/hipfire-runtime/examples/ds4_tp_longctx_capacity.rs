// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 TP compressed-cache capacity probe.
//!
//! Loads the production EP/TP route once, grows the request-owned F32
//! compressor caches through a list of capacities, and records per-rank VMM
//! reserve/mapping plus physical VRAM. It intentionally performs no prefill;
//! use `scripts/serve_harness.py` for coherent long-context generation.

use hipfire_arch_deepseek4::forward::{ensure_request_capacity, forward_ep_prefill_batch_chunked};
use hipfire_arch_deepseek4::DeepseekV4State;
use hipfire_loader::{load_model_ep, EpArch};
use rdna_compute::Gpu;

const DEFAULT_TOKENS: &[usize] = &[20_480, 81_920, 1_048_576];

struct Args {
    model: String,
    tp: usize,
    tokens: Vec<usize>,
    identity_tokens: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut tp = 3usize;
    let mut tokens = DEFAULT_TOKENS.to_vec();
    let mut identity_tokens = None;
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
            "--identity-tokens" => {
                identity_tokens = Some(
                    value(i, "--identity-tokens")?
                        .parse()
                        .map_err(|_| "invalid --identity-tokens".to_string())?,
                );
                i += 2;
            }
            flag => return Err(format!("unknown argument {flag}")),
        }
    }
    if tp == 0
        || tokens.is_empty()
        || tokens.contains(&0)
        || identity_tokens.is_some_and(|value| value == 0)
    {
        return Err(
            "--tp, --identity-tokens, and every --tokens entry must be nonzero".to_string(),
        );
    }
    if !tokens.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("--tokens entries must be strictly increasing".to_string());
    }
    Ok(Args {
        model: model.ok_or_else(|| "--model is required".to_string())?,
        tp,
        tokens,
        identity_tokens,
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

fn cache_bits(
    gpu: &mut Gpu,
    tensor: &rdna_compute::GpuTensor,
    elements: usize,
) -> Result<Vec<u32>, String> {
    let view = tensor.sub_offset(0, elements);
    gpu.download_f32(&view)
        .map(|values| values.into_iter().map(f32::to_bits).collect())
        .map_err(|error| format!("download compressor cache: {error:?}"))
}

fn prove_cache_identity(
    states: &[DeepseekV4State],
    gpus: &mut [Gpu],
    identity_tokens: usize,
) -> Result<(), String> {
    if states.len() != gpus.len() || states.len() < 2 {
        return Err(format!(
            "cache identity requires matching multi-rank state/GPU slices (state={}, gpus={})",
            states.len(),
            gpus.len(),
        ));
    }

    let mut tensors = 0usize;
    let mut compared_elements = 0usize;
    let mut rank_hashes = vec![0xcbf29ce484222325u64; states.len()];
    for layer_idx in 0..states[0]._indexer.len() {
        let ratio = states[0]._indexer[layer_idx].compress_ratio as usize;
        if ratio == 0 {
            continue;
        }
        let filled_rows = identity_tokens / ratio;
        if filled_rows == 0 {
            continue;
        }
        for (cache_name, row_elems, is_indexer) in [
            (
                "main",
                states[0]._indexer[layer_idx]
                    .main_kv_cache
                    .as_ref()
                    .map(|t| t.shape[1]),
                false,
            ),
            (
                "indexer",
                states[0]._indexer[layer_idx]
                    .indexer_kv_cache
                    .as_ref()
                    .map(|t| t.shape[1]),
                true,
            ),
        ] {
            let Some(row_elems) = row_elems else {
                if is_indexer && ratio != 4 {
                    continue;
                }
                return Err(format!(
                    "rank 0 missing {cache_name} compressor cache at layer {layer_idx}"
                ));
            };
            let elements = filled_rows
                .checked_mul(row_elems)
                .ok_or_else(|| "compressor identity element overflow".to_string())?;
            let mut reference: Option<Vec<u32>> = None;
            for rank in 0..states.len() {
                gpus[rank]
                    .bind_thread()
                    .map_err(|error| format!("identity bind rank {rank}: {error:?}"))?;
                let layer = &states[rank]._indexer[layer_idx];
                let tensor = if is_indexer {
                    layer.indexer_kv_cache.as_ref()
                } else {
                    layer.main_kv_cache.as_ref()
                }
                .ok_or_else(|| {
                    format!(
                        "rank {rank} missing {cache_name} compressor cache at layer {layer_idx}"
                    )
                })?;
                if tensor.shape.get(1).copied() != Some(row_elems)
                    || tensor.shape.first().copied().unwrap_or(0) < filled_rows
                {
                    return Err(format!(
                        "rank {rank} {cache_name} compressor cache shape mismatch at layer {layer_idx}: shape={:?}, required_rows={filled_rows}, row_elems={row_elems}",
                        tensor.shape,
                    ));
                }
                let bits = cache_bits(&mut gpus[rank], tensor, elements)?;
                for &value in &bits {
                    mix_hash(&mut rank_hashes[rank], value as usize);
                }
                if let Some(expected) = reference.as_ref() {
                    if &bits != expected {
                        let first = bits
                            .iter()
                            .zip(expected)
                            .position(|(actual, wanted)| actual != wanted)
                            .unwrap_or(0);
                        return Err(format!(
                            "compressor cache differs: rank={rank} layer={layer_idx} cache={cache_name} element={first} expected=0x{:08x} actual=0x{:08x}",
                            expected[first], bits[first],
                        ));
                    }
                } else {
                    reference = Some(bits);
                }
            }
            tensors += 1;
            compared_elements = compared_elements.saturating_add(elements);
        }
    }

    if !rank_hashes.windows(2).all(|pair| pair[0] == pair[1]) {
        return Err(format!(
            "compressor cache aggregate hash mismatch after element parity: {rank_hashes:x?}"
        ));
    }
    println!(
        "CACHE_IDENTITY status=pass tokens={identity_tokens} ranks={} tensors={} compared_elements={} raw_bits_per_rank={} aggregate_hash=0x{:016x}",
        states.len(),
        tensors,
        compared_elements,
        compared_elements,
        rank_hashes[0],
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let max_seq =
        (*args.tokens.last().expect("nonempty tokens")).max(args.identity_tokens.unwrap_or(0));
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
        weights,
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

    if let Some(identity_tokens) = args.identity_tokens {
        let vocab = config.vocab_size.max(1);
        let identity_input: Vec<u32> = (0..identity_tokens)
            .map(|position| ((position.wrapping_mul(7_919) + 101) % vocab) as u32)
            .collect();
        forward_ep_prefill_batch_chunked(
            &mut ep.gpus,
            weights,
            config,
            state,
            prefill,
            &identity_input,
            0,
        )?;
        prove_cache_identity(state, &mut ep.gpus.devices, identity_tokens)?;
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
