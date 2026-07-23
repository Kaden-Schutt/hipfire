// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline fixed-slot sampled generator for Qwen3.5/A3B.
//!
//! Input JSONL accepts one of:
//!   {"id": ..., "prompt": "..."}
//!   {"id": ..., "messages": [{"role":"user","content":"..."}, ...]}
//!   {"id": ..., "tokens": [1,2,3]}
//!
//! Example:
//!   cargo run --release -p hipfire-runtime --example qwen35_batch_generate -- \
//!     model.mq4 --input prompts.jsonl --output completions.jsonl \
//!     --batch 16 --max-seq 4096 --max-new 512 --temperature 1 \
//!     --top-p .95 --top-k 20
//!
//! Four independent workers can read the same input by setting
//! `HIP_VISIBLE_DEVICES` per process and passing distinct
//! `--shard-index 0..3 --shard-count 4` pairs. Output rows retain the original
//! input index and can be concatenated, then sorted by `index` if needed.

use hipfire_arch_qwen35::qwen35::{self, Qwen35DecodeBatchState, Qwen35Scratch, Qwen35Weights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    config: Option<PathBuf>,
    device: Option<String>,
    batch: usize,
    max_seq: usize,
    max_new: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_window: usize,
    seed: u32,
    raw_prompt: bool,
    shard_index: usize,
    shard_count: usize,
    shadow_iterations: Option<usize>,
    batched_seed: bool,
    wave_refill: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut it = std::env::args().skip(1);
        let model = it.next().map(PathBuf::from).ok_or_else(|| {
            "usage: qwen35_batch_generate MODEL --input IN.jsonl --output OUT.jsonl [options]"
                .to_string()
        })?;
        let mut input = None;
        let mut output = None;
        let mut args = Self {
            model,
            input: PathBuf::new(),
            output: PathBuf::new(),
            config: None,
            device: None,
            batch: 16,
            max_seq: 4096,
            max_new: 512,
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(20),
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            repeat_window: 128,
            seed: 0x1357_9bdf,
            raw_prompt: false,
            shard_index: 0,
            shard_count: 1,
            shadow_iterations: None,
            batched_seed: true,
            wave_refill: false,
        };
        while let Some(flag) = it.next() {
            let value = |it: &mut std::iter::Skip<std::env::Args>, flag: &str| {
                it.next().ok_or_else(|| format!("missing value for {flag}"))
            };
            match flag.as_str() {
                "--input" => input = Some(PathBuf::from(value(&mut it, &flag)?)),
                "--output" => output = Some(PathBuf::from(value(&mut it, &flag)?)),
                "--config" => args.config = Some(PathBuf::from(value(&mut it, &flag)?)),
                "--device" => args.device = Some(value(&mut it, &flag)?),
                "--batch" => args.batch = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                "--max-seq" => args.max_seq = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                "--max-new" => args.max_new = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                "--temperature" => {
                    args.temperature = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--top-p" => args.top_p = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                "--top-k" => {
                    let k: u32 = value(&mut it, &flag)?.parse().map_err(|_| flag)?;
                    args.top_k = (k > 0).then_some(k);
                }
                "--min-p" => {
                    let p: f32 = value(&mut it, &flag)?.parse().map_err(|_| flag)?;
                    args.min_p = (p > 0.0).then_some(p);
                }
                "--repeat-penalty" => {
                    args.repeat_penalty = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--presence-penalty" => {
                    args.presence_penalty = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--frequency-penalty" => {
                    args.frequency_penalty = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--repeat-window" => {
                    args.repeat_window = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--seed" => args.seed = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                "--shard-index" => {
                    args.shard_index = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--shard-count" => {
                    args.shard_count = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                }
                "--shadow-iterations" => {
                    args.shadow_iterations = Some(value(&mut it, &flag)?.parse().map_err(|_| flag)?)
                }
                "--raw-prompt" => args.raw_prompt = true,
                "--sequential-seed" => args.batched_seed = false,
                "--wave-refill" => args.wave_refill = true,
                "--help" | "-h" => {
                    return Err(
                        "usage: qwen35_batch_generate MODEL --input IN.jsonl --output OUT.jsonl \
                         [--batch 16] [--max-seq 4096] [--max-new 512] \
                         [--temperature 1] [--top-p .95] [--top-k 20] [--seed N] \
                         [--min-p 0] [--repeat-penalty 1] \
                         [--presence-penalty 1.5] [--frequency-penalty 0] \
                         [--repeat-window 128] \
                         [--config CONFIG.toml] [--device PHYSICAL_ID] \
                         [--shadow-iterations N] \
                         [--shard-index I --shard-count N] [--raw-prompt] \
                         [--sequential-seed] [--wave-refill]"
                            .to_string(),
                    )
                }
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        args.input = input.ok_or_else(|| "--input is required".to_string())?;
        args.output = output.ok_or_else(|| "--output is required".to_string())?;
        if args.batch == 0 || args.max_seq < 2 || args.max_new == 0 {
            return Err("batch/max-new must be non-zero and max-seq >= 2".to_string());
        }
        if !(0.0..=1.0).contains(&args.top_p)
            || args.top_p == 0.0
            || args
                .min_p
                .is_some_and(|value| !(0.0..=1.0).contains(&value))
            || args.repeat_penalty < 1.0
            || args.presence_penalty < 0.0
            || args.frequency_penalty < 0.0
            || args.repeat_window == 0
            || args.repeat_window > 128
        {
            return Err(
                "invalid sampling parameter range (repeat-window must be 1..=128)".to_string(),
            );
        }
        if args.shard_count == 0 || args.shard_index >= args.shard_count {
            return Err("shard-count must be non-zero and shard-index < shard-count".to_string());
        }
        if args.shadow_iterations == Some(0) {
            return Err("--shadow-iterations must be non-zero".to_string());
        }
        Ok(args)
    }
}

fn install_startup_config(args: &Args) -> Result<(), String> {
    let (path, global) = if let Some(path) = args.config.as_deref() {
        (
            path.to_owned(),
            hipfire_config::load_toml_layer(path)
                .map_err(|e| format!("load {}: {e}", path.display()))?,
        )
    } else {
        let loaded = hipfire_config::load_global(&hipfire_config::ConfigPaths::discover())
            .map_err(|e| format!("load local config: {e}"))?;
        (loaded.path, loaded.layer)
    };
    let mut layers = vec![hipfire_config::NamedLayer {
        source: hipfire_config::ConfigSource::GlobalUser { path },
        layer: global,
    }];
    let environment =
        hipfire_config::load_env_layer().map_err(|e| format!("load environment: {e}"))?;
    if !environment.values.is_empty() {
        layers.push(hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: environment,
        });
    }
    if let Some(device) = &args.device {
        let mut one_shot = hipfire_config::ConfigLayer::default();
        one_shot
            .set_cli("hardware.devices", device)
            .map_err(|e| format!("invalid --device: {e}"))?;
        layers.push(hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::OneShot {
                argument: format!("--device {device}"),
            },
            layer: one_shot,
        });
    }
    let resolved = hipfire_config::resolve(layers).map_err(|e| format!("resolve config: {e}"))?;
    let process = hipfire_config::ProcessConfig::from_resolved(&resolved)
        .map_err(|e| format!("build process config: {e}"))?;
    hipfire_config::apply_device_visibility(&process)
        .map_err(|e| format!("apply device visibility: {e}"))?;
    let runtime = hipfire_runtime::config::RuntimeConfig::from_process_config(&process);
    hipfire_config::install_process_config(process)
        .map_err(|_| "process configuration was already initialized".to_string())?;
    hipfire_runtime::config::init_with(runtime)
        .map_err(|_| "runtime process configuration was already initialized".to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct InputRow {
    #[serde(default)]
    id: Value,
    prompt: Option<String>,
    messages: Option<Vec<Message>>,
    tokens: Option<Vec<u32>>,
    max_new_tokens: Option<usize>,
}

struct Job {
    index: usize,
    id: Value,
    prompt_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    max_new: usize,
}

struct Slot {
    job: Option<Job>,
    next_token: u32,
    next_pos: usize,
    rng_state: u32,
}

#[derive(PartialEq)]
struct BatchShadowSnapshot {
    logits: Vec<u8>,
    kv: Vec<u8>,
    recurrent: Vec<u8>,
}

impl BatchShadowSnapshot {
    fn json(&self) -> Value {
        json!({
            "logits_bytes": self.logits.len(),
            "logits_hash": format!("{:016x}", shadow_hash(&self.logits)),
            "kv_bytes": self.kv.len(),
            "kv_hash": format!("{:016x}", shadow_hash(&self.kv)),
            "recurrent_bytes": self.recurrent.len(),
            "recurrent_hash": format!("{:016x}", shadow_hash(&self.recurrent)),
        })
    }
}

fn shadow_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn append_shadow_buffer(
    gpu: &Gpu,
    output: &mut Vec<u8>,
    buffer: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    let start = output.len();
    output.resize(start + buffer.size(), 0);
    gpu.hip
        .memcpy_dtoh(&mut output[start..], buffer)
        .map_err(|error| error.to_string())
}

fn batch_shadow_snapshot(
    gpu: &Gpu,
    state: &Qwen35DecodeBatchState,
) -> Result<BatchShadowSnapshot, String> {
    let mut logits = Vec::new();
    append_shadow_buffer(gpu, &mut logits, &state.logits.buf)?;
    let mut kv = Vec::new();
    for tensor in state
        .kv_cache
        .k_gpu
        .iter()
        .chain(state.kv_cache.v_gpu.iter())
        .chain(state.kv_cache.k_scales.iter())
        .chain(state.kv_cache.v_scales.iter())
    {
        append_shadow_buffer(gpu, &mut kv, &tensor.buf)?;
    }
    let mut recurrent = Vec::new();
    for tensor in state
        .dn_state
        .s_matrices
        .iter()
        .chain(state.dn_state.s_scales.iter())
        .chain(state.dn_state.conv_states.iter())
        .chain(state.dn_state.s_ef_residual.iter())
    {
        append_shadow_buffer(gpu, &mut recurrent, &tensor.buf)?;
    }
    Ok(BatchShadowSnapshot {
        logits,
        kv,
        recurrent,
    })
}

fn batch_prefix_hashes(
    gpu: &Gpu,
    state: &Qwen35DecodeBatchState,
) -> Result<BTreeMap<String, String>, String> {
    let mut hashes = BTreeMap::new();
    macro_rules! hash_tensor {
        ($name:expr, $tensor:expr) => {{
            let mut bytes = Vec::new();
            append_shadow_buffer(gpu, &mut bytes, &$tensor.buf)?;
            hashes.insert($name.to_string(), format!("{:016x}", shadow_hash(&bytes)));
        }};
    }
    for (name, tensor) in [
        ("x_batch", &state.pbs.x_batch),
        ("x_rot_batch", &state.pbs.x_rot_batch),
        ("x_norm_batch", &state.pbs.x_norm_batch),
        ("dn_qkv_batch", &state.pbs.dn_qkv_batch),
        ("dn_z_batch", &state.pbs.dn_z_batch),
        ("dn_alpha_batch", &state.pbs.dn_alpha_batch),
        ("dn_beta_batch", &state.pbs.dn_beta_batch),
        ("dn_q_raw_batch", &state.pbs.dn_q_raw_batch),
        ("dn_k_raw_batch", &state.pbs.dn_k_raw_batch),
        ("dn_v_batch", &state.pbs.dn_v_batch),
        ("dn_q_batch", &state.pbs.dn_q_batch),
        ("dn_k_batch", &state.pbs.dn_k_batch),
        ("dn_attn_out_batch", &state.pbs.dn_attn_out_batch),
        ("dn_normed_batch", &state.pbs.dn_normed_batch),
        ("gate_ffn_batch", &state.pbs.gate_ffn_batch),
        ("up_batch", &state.pbs.up_batch),
        ("ffn_hidden_batch", &state.pbs.ffn_hidden_batch),
        ("dn_normed_rot_batch", &state.pbs.dn_normed_rot_batch),
        ("positions", &state.pbs.positions),
        ("tokens", &state.pbs.tokens),
        ("fa_q_full_batch", &state.pbs.fa_q_full_batch),
        ("fa_q_batch", &state.pbs.fa_q_batch),
        ("fa_gate_batch", &state.pbs.fa_gate_batch),
        ("fa_k_batch", &state.pbs.fa_k_batch),
        ("fa_v_batch", &state.pbs.fa_v_batch),
        ("fa_attn_out_batch", &state.pbs.fa_attn_out_batch),
        ("fa_attn_out_rot_batch", &state.pbs.fa_attn_out_rot_batch),
        ("final_hidden", &state.final_hidden),
        ("logits", &state.logits),
        ("lm_rot", &state.lm_rot),
    ] {
        hash_tensor!(name, tensor);
    }
    for (name, tensor) in [
        (
            "moe_router_logits_batch",
            state.pbs.moe_router_logits_batch.as_ref(),
        ),
        (
            "moe_shared_scalar_batch",
            state.pbs.moe_shared_scalar_batch.as_ref(),
        ),
        (
            "moe_shared_gate_batch",
            state.pbs.moe_shared_gate_batch.as_ref(),
        ),
        (
            "moe_shared_up_batch",
            state.pbs.moe_shared_up_batch.as_ref(),
        ),
        (
            "moe_shared_rot_batch",
            state.pbs.moe_shared_rot_batch.as_ref(),
        ),
        (
            "moe_topk_indices_batch",
            state.pbs.moe_topk_indices_batch.as_ref(),
        ),
        (
            "moe_topk_weights_batch",
            state.pbs.moe_topk_weights_batch.as_ref(),
        ),
        ("moe_gate_batch", state.pbs.moe_gate_batch.as_ref()),
        ("moe_up_batch", state.pbs.moe_up_batch.as_ref()),
        ("moe_rot_batch", state.pbs.moe_rot_batch.as_ref()),
        (
            "moe_down_expanded_batch",
            state.pbs.moe_down_expanded_batch.as_ref(),
        ),
        (
            "moe_expert_token_counts",
            state.pbs.moe_expert_token_counts.as_ref(),
        ),
        ("moe_expert_offsets", state.pbs.moe_expert_offsets.as_ref()),
        (
            "moe_sorted_slot_index",
            state.pbs.moe_sorted_slot_index.as_ref(),
        ),
        ("moe_inverse_perm", state.pbs.moe_inverse_perm.as_ref()),
        (
            "moe_expert_tile_ids",
            state.pbs.moe_expert_tile_ids.as_ref(),
        ),
        (
            "moe_y_gate_up_grouped",
            state.pbs.moe_y_gate_up_grouped.as_ref(),
        ),
        ("moe_y_down_grouped", state.pbs.moe_y_down_grouped.as_ref()),
    ] {
        if let Some(tensor) = tensor {
            hash_tensor!(name, tensor);
        }
    }
    let model = batch_shadow_snapshot(gpu, state)?;
    hashes.insert(
        "kv_state".to_string(),
        format!("{:016x}", shadow_hash(&model.kv)),
    );
    hashes.insert(
        "recurrent_state".to_string(),
        format!("{:016x}", shadow_hash(&model.recurrent)),
    );
    Ok(hashes)
}

fn reset_shadow_state(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    slots: &[Slot],
) -> Result<(), String> {
    state
        .reset(gpu)
        .map_err(|error| format!("reset batch shadow state: {error}"))?;
    scratch
        .clear_gpu(gpu)
        .map_err(|error| format!("reset shared shadow scratch: {error}"))?;
    for (lane, slot) in slots.iter().enumerate() {
        let job = slot
            .job
            .as_ref()
            .ok_or_else(|| format!("shadow lane {lane} has no seeded job"))?;
        state
            .prefill_lane(gpu, weights, config, scratch, lane, &job.prompt_tokens)
            .map_err(|error| format!("re-prime shadow lane {lane}: {error}"))?;
    }
    // The conversion caches key on source pointer. Shadow restoration changes
    // the pointee contents while retaining that pointer, so force the direct
    // oracle to emit the same conversion launches captured by the tape.
    gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("synchronize restored shadow state: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn run_batch_prefix_arm(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    slots: &[Slot],
    tokens: &[u32],
    positions: &[usize],
    prefix: usize,
    recorded: bool,
    arm_checkpoint: u32,
    decode_checkpoint: u32,
) -> Result<BTreeMap<String, String>, String> {
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
    reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(decode_checkpoint);
    let position =
        qwen35::prepare_decode_batch_inputs(gpu, weights, config, tokens, positions, state)
            .map_err(|error| format!("prepare batch prefix inputs: {error}"))?;
    if recorded {
        gpu.replay_recorded_hip_prefix(prefix)
            .map_err(|error| format!("execute recorded HIP prefix {prefix}: {error}"))?;
    } else {
        gpu.replay.begin_diagnostic_launch_prefix(prefix);
        let result = qwen35::forward_decode_batch_prepared(
            gpu, weights, config, tokens, positions, position, state, scratch,
        );
        let admitted = gpu.replay.finish_diagnostic_launch_prefix();
        if let Err(error) = result {
            if error.code != rdna_compute::replay::DIAGNOSTIC_PREFIX_COMPLETE_CODE
                && !error.message.contains("diagnostic launch prefix complete")
            {
                return Err(format!("execute ordinary HIP prefix {prefix}: {error}"));
            }
        }
        if admitted != prefix {
            return Err(format!(
                "ordinary HIP prefix admitted {admitted} launches, expected {prefix}"
            ));
        }
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("synchronize batch prefix {prefix}: {error}"))?;
    batch_prefix_hashes(gpu, state)
}

#[allow(clippy::too_many_arguments)]
fn run_batch_retained_prefix_arm(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    slots: &[Slot],
    tokens: &[u32],
    positions: &[usize],
    prefix: usize,
    arm_checkpoint: u32,
    decode_checkpoint: u32,
) -> Result<BTreeMap<String, String>, String> {
    if gpu.replay.uses_pm4_transport() {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, prefix)
            .map_err(|reason| format!("prepare PM4 prefix {prefix}: {reason}"))?;
    } else {
        gpu.replay
            .prepare_linear_aql_prefix(gpu.device_id as usize, prefix)
            .map_err(|reason| format!("prepare AQL prefix {prefix}: {reason}"))?;
    }
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
    reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(decode_checkpoint);
    let position =
        qwen35::prepare_decode_batch_inputs(gpu, weights, config, tokens, positions, state)
            .map_err(|error| format!("prepare retained prefix inputs: {error}"))?;
    if gpu.replay.uses_pm4_transport() {
        unsafe { gpu.replay.replay_pm4(position) }
            .map_err(|reason| format!("execute PM4 prefix {prefix}: {reason}"))?;
    } else {
        unsafe { gpu.replay.replay_linear_aql(position) }
            .map_err(|reason| format!("execute AQL prefix {prefix}: {reason}"))?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("synchronize retained prefix {prefix}: {error}"))?;
    batch_prefix_hashes(gpu, state)
}

fn advance_shadow_tokens(tokens: &mut [u32], observation: usize, step: usize, vocab: usize) {
    for (lane, token) in tokens.iter_mut().enumerate() {
        let delta = 17 + observation * 29 + step * 13 + lane * 3;
        *token = ((*token as usize + delta) % vocab) as u32;
    }
}

fn run_batch_shadow_gate(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    slots: &[Slot],
    iterations: usize,
) -> Result<Value, String> {
    if gpu.replay.request() != rdna_compute::replay::ReplayBackendRequest::Shadow {
        return Err(
            "--shadow-iterations requires replay.backend = \"shadow\" and manual capture"
                .to_string(),
        );
    }
    let initial_tokens: Vec<u32> = slots.iter().map(|slot| slot.next_token).collect();
    let initial_positions: Vec<usize> = slots.iter().map(|slot| slot.next_pos).collect();

    // Capture exactly one ordinary HIP batch forward. Shadow mode can prepare
    // and execute this tape explicitly, but can never change the model route.
    let capture_position = qwen35::prepare_decode_batch_inputs(
        gpu,
        weights,
        config,
        &initial_tokens,
        &initial_positions,
        state,
    )
    .map_err(|error| format!("prepare batch shadow capture inputs: {error}"))?;
    gpu.replay
        .begin_capture()
        .map_err(|reason| format!("begin batch shadow capture: {reason}"))?;
    gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
    qwen35::forward_decode_batch_prepared(
        gpu,
        weights,
        config,
        &initial_tokens,
        &initial_positions,
        capture_position,
        state,
        scratch,
    )
    .map_err(|error| format!("capture ordinary HIP batch forward: {error}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("synchronize batch capture: {error}"))?;
    let capture = gpu
        .replay
        .finish_capture()
        .map_err(|reason| format!("finish batch shadow capture: {reason}"))?;
    let contracts = gpu
        .replay
        .probe_aql_contracts(gpu.device_id as usize)
        .map_err(|reason| format!("probe batch replay ABI: {reason}"))?;
    if gpu.replay.uses_pm4_transport() {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, capture.launch_count)
            .map_err(|reason| format!("prepare batch PM4 shadow: {reason}"))?;
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map_err(|reason| format!("prepare batch AQL shadow: {reason}"))?;
    }
    let identity = gpu
        .replay
        .prepared_route_identity()
        .ok_or_else(|| "batch shadow prepare produced no retained identity".to_string())?;
    let captured_launches = gpu
        .replay
        .recorded_launches()
        .iter()
        .take(32)
        .enumerate()
        .map(|(index, launch)| {
            json!({
                "index": index,
                "kernel": launch.kernel,
                "grid": launch.grid,
                "block": launch.block,
                "kernarg_bytes": launch.kernarg.len(),
                "first_u64": launch.kernarg
                    .chunks_exact(8)
                    .take(8)
                    .map(|bytes| format!(
                        "{:016x}",
                        u64::from_ne_bytes(bytes.try_into().expect("eight-byte kernarg chunk"))
                    ))
                    .collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();

    let mut observations = Vec::with_capacity(2);
    for observation in 0..2 {
        let arm_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
        reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
        let decode_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();

        let mut direct_tokens = initial_tokens.clone();
        let mut direct_positions = initial_positions.clone();
        if observation != 0 {
            advance_shadow_tokens(&mut direct_tokens, observation, 997, config.vocab_size);
        }
        let direct_started = Instant::now();
        for step in 0..iterations {
            qwen35::forward_decode_batch(
                gpu,
                weights,
                config,
                &direct_tokens,
                &direct_positions,
                state,
                scratch,
            )
            .map_err(|error| format!("ordinary HIP shadow arm: {error}"))?;
            if step + 1 != iterations {
                advance_shadow_tokens(&mut direct_tokens, observation, step, config.vocab_size);
                for position in &mut direct_positions {
                    *position += 1;
                }
            }
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("synchronize ordinary HIP shadow arm: {error}"))?;
        let direct_host_us = direct_started.elapsed().as_secs_f64() * 1_000_000.0;
        let direct = batch_shadow_snapshot(gpu, state)?;

        // Before blaming either retained transport, replay the captured launch
        // blobs through ordinary HIP. This isolates capture fidelity from AQL
        // or PM4 lowering: if this arm differs, the tape itself is not yet a
        // valid shadow oracle.
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
        reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(decode_checkpoint);
        let mut recorded_tokens = initial_tokens.clone();
        let mut recorded_positions = initial_positions.clone();
        if observation != 0 {
            advance_shadow_tokens(&mut recorded_tokens, observation, 997, config.vocab_size);
        }
        let recorded_started = Instant::now();
        for step in 0..iterations {
            qwen35::prepare_decode_batch_inputs(
                gpu,
                weights,
                config,
                &recorded_tokens,
                &recorded_positions,
                state,
            )
            .map_err(|error| format!("prepare recorded HIP batch inputs: {error}"))?;
            gpu.replay_recorded_hip_prefix(capture.launch_count)
                .map_err(|error| format!("execute recorded HIP shadow: {error}"))?;
            if step + 1 != iterations {
                advance_shadow_tokens(&mut recorded_tokens, observation, step, config.vocab_size);
                for position in &mut recorded_positions {
                    *position += 1;
                }
            }
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("synchronize recorded HIP shadow arm: {error}"))?;
        let recorded_host_us = recorded_started.elapsed().as_secs_f64() * 1_000_000.0;
        let recorded = batch_shadow_snapshot(gpu, state)?;

        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
        reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(decode_checkpoint);
        let mut replay_tokens = initial_tokens.clone();
        let mut replay_positions = initial_positions.clone();
        if observation != 0 {
            advance_shadow_tokens(&mut replay_tokens, observation, 997, config.vocab_size);
        }
        let replay_started = Instant::now();
        let mut replay_gpu_us = 0.0;
        for step in 0..iterations {
            let position = qwen35::prepare_decode_batch_inputs(
                gpu,
                weights,
                config,
                &replay_tokens,
                &replay_positions,
                state,
            )
            .map_err(|error| format!("prepare retained batch inputs: {error}"))?;
            replay_gpu_us += if gpu.replay.uses_pm4_transport() {
                unsafe { gpu.replay.replay_pm4(position) }
                    .map_err(|reason| format!("execute batch PM4 shadow: {reason}"))?
                    .span_microseconds()
            } else {
                unsafe { gpu.replay.replay_linear_aql(position) }
                    .map_err(|reason| format!("execute batch AQL shadow: {reason}"))?
                    .span_microseconds()
            };
            if step + 1 != iterations {
                advance_shadow_tokens(&mut replay_tokens, observation, step, config.vocab_size);
                for position in &mut replay_positions {
                    *position += 1;
                }
            }
        }
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("synchronize retained shadow arm: {error}"))?;
        let replay_host_us = replay_started.elapsed().as_secs_f64() * 1_000_000.0;
        let replay = batch_shadow_snapshot(gpu, state)?;
        let logits_equal = direct.logits == replay.logits;
        let kv_equal = direct.kv == replay.kv;
        let recurrent_equal = direct.recurrent == replay.recurrent;
        let recorded_logits_equal = direct.logits == recorded.logits;
        let recorded_kv_equal = direct.kv == recorded.kv;
        let recorded_recurrent_equal = direct.recurrent == recorded.recurrent;
        observations.push(json!({
            "observation": observation,
            "iterations": iterations,
            "bit_exact": logits_equal && kv_equal && recurrent_equal,
            "recorded_hip_bit_exact":
                recorded_logits_equal && recorded_kv_equal && recorded_recurrent_equal,
            "recorded_hip_logits_equal": recorded_logits_equal,
            "recorded_hip_kv_equal": recorded_kv_equal,
            "recorded_hip_recurrent_equal": recorded_recurrent_equal,
            "logits_equal": logits_equal,
            "kv_equal": kv_equal,
            "recurrent_equal": recurrent_equal,
            "direct_host_us": direct_host_us,
            "recorded_hip_host_us": recorded_host_us,
            "replay_host_us": replay_host_us,
            "replay_gpu_us": replay_gpu_us,
            "direct": direct.json(),
            "recorded_hip": recorded.json(),
            "replay": replay.json(),
        }));
    }
    let bit_exact = observations
        .iter()
        .all(|row| row["bit_exact"].as_bool() == Some(true));
    let recorded_hip_bit_exact = observations
        .iter()
        .all(|row| row["recorded_hip_bit_exact"].as_bool() == Some(true));
    let prefix_localization = if !recorded_hip_bit_exact {
        let arm_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
        reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
        let decode_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        let mut rows = Vec::new();
        let mut last_equal = 0usize;
        let mut first_different = None;
        let mut prefix = 1usize;
        while prefix <= capture.launch_count {
            let direct = run_batch_prefix_arm(
                gpu,
                weights,
                config,
                scratch,
                state,
                slots,
                &initial_tokens,
                &initial_positions,
                prefix,
                false,
                arm_checkpoint,
                decode_checkpoint,
            )?;
            let recorded = run_batch_prefix_arm(
                gpu,
                weights,
                config,
                scratch,
                state,
                slots,
                &initial_tokens,
                &initial_positions,
                prefix,
                true,
                arm_checkpoint,
                decode_checkpoint,
            )?;
            let differing = direct
                .iter()
                .filter_map(|(name, hash)| (recorded.get(name) != Some(hash)).then(|| name.clone()))
                .collect::<Vec<_>>();
            let equal = differing.is_empty();
            rows.push(json!({"prefix": prefix, "equal": equal, "differing": differing}));
            if !equal {
                first_different = Some(prefix);
                break;
            }
            last_equal = prefix;
            if prefix == capture.launch_count {
                break;
            }
            prefix = prefix.saturating_mul(2).min(capture.launch_count);
        }
        if let Some(mut upper) = first_different {
            let mut lower = last_equal;
            while lower + 1 < upper {
                let mid = lower + (upper - lower) / 2;
                let direct = run_batch_prefix_arm(
                    gpu,
                    weights,
                    config,
                    scratch,
                    state,
                    slots,
                    &initial_tokens,
                    &initial_positions,
                    mid,
                    false,
                    arm_checkpoint,
                    decode_checkpoint,
                )?;
                let recorded = run_batch_prefix_arm(
                    gpu,
                    weights,
                    config,
                    scratch,
                    state,
                    slots,
                    &initial_tokens,
                    &initial_positions,
                    mid,
                    true,
                    arm_checkpoint,
                    decode_checkpoint,
                )?;
                let differing = direct
                    .iter()
                    .filter_map(|(name, hash)| {
                        (recorded.get(name) != Some(hash)).then(|| name.clone())
                    })
                    .collect::<Vec<_>>();
                let equal = differing.is_empty();
                rows.push(json!({"prefix": mid, "equal": equal, "differing": differing}));
                if equal {
                    lower = mid;
                } else {
                    upper = mid;
                }
            }
            last_equal = lower;
            first_different = Some(upper);
        }
        let first_kernel = first_different.and_then(|divergence| {
            gpu.replay
                .recorded_launches()
                .get(divergence - 1)
                .map(|launch| launch.kernel.clone())
        });
        json!({
            "stage": "ordinary_hip_to_captured_hip",
            "last_equal_prefix": last_equal,
            "first_different_prefix": first_different,
            "first_different_kernel": first_kernel,
            "probes": rows,
        })
    } else if !bit_exact {
        let arm_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(arm_checkpoint);
        reset_shadow_state(gpu, weights, config, scratch, state, slots)?;
        let decode_checkpoint = rdna_compute::norm::gdn_requant_frame_checkpoint();
        let mut rows = Vec::new();
        let mut last_equal = 1usize;
        let mut first_different = None;
        let mut prefix = 2usize.min(capture.launch_count);
        while prefix <= capture.launch_count {
            let recorded = run_batch_prefix_arm(
                gpu,
                weights,
                config,
                scratch,
                state,
                slots,
                &initial_tokens,
                &initial_positions,
                prefix,
                true,
                arm_checkpoint,
                decode_checkpoint,
            )?;
            let retained = run_batch_retained_prefix_arm(
                gpu,
                weights,
                config,
                scratch,
                state,
                slots,
                &initial_tokens,
                &initial_positions,
                prefix,
                arm_checkpoint,
                decode_checkpoint,
            )?;
            let differing = recorded
                .iter()
                .filter_map(|(name, hash)| (retained.get(name) != Some(hash)).then(|| name.clone()))
                .collect::<Vec<_>>();
            let equal = differing.is_empty();
            rows.push(json!({"prefix": prefix, "equal": equal, "differing": differing}));
            if !equal {
                first_different = Some(prefix);
                break;
            }
            last_equal = prefix;
            if prefix == capture.launch_count {
                break;
            }
            prefix = prefix.saturating_mul(2).min(capture.launch_count);
        }
        if let Some(mut upper) = first_different {
            let mut lower = last_equal;
            while lower + 1 < upper {
                let mid = lower + (upper - lower) / 2;
                let recorded = run_batch_prefix_arm(
                    gpu,
                    weights,
                    config,
                    scratch,
                    state,
                    slots,
                    &initial_tokens,
                    &initial_positions,
                    mid,
                    true,
                    arm_checkpoint,
                    decode_checkpoint,
                )?;
                let retained = run_batch_retained_prefix_arm(
                    gpu,
                    weights,
                    config,
                    scratch,
                    state,
                    slots,
                    &initial_tokens,
                    &initial_positions,
                    mid,
                    arm_checkpoint,
                    decode_checkpoint,
                )?;
                let differing = recorded
                    .iter()
                    .filter_map(|(name, hash)| {
                        (retained.get(name) != Some(hash)).then(|| name.clone())
                    })
                    .collect::<Vec<_>>();
                let equal = differing.is_empty();
                rows.push(json!({"prefix": mid, "equal": equal, "differing": differing}));
                if equal {
                    lower = mid;
                } else {
                    upper = mid;
                }
            }
            last_equal = lower;
            first_different = Some(upper);
        }
        let first_kernel = first_different.and_then(|divergence| {
            gpu.replay
                .recorded_launches()
                .get(divergence - 1)
                .map(|launch| launch.kernel.clone())
        });
        json!({
            "stage": if gpu.replay.uses_pm4_transport() {
                "captured_hip_to_pm4"
            } else {
                "captured_hip_to_aql"
            },
            "last_equal_prefix": last_equal,
            "first_different_prefix": first_different,
            "first_different_kernel": first_kernel,
            "probes": rows,
        })
    } else {
        Value::Null
    };
    Ok(json!({
        "type": "qwen35_batch_shadow_result",
        "backend": if gpu.replay.uses_pm4_transport() { "pm4_ib" } else { "aql_packets" },
        "bit_exact": bit_exact,
        "batch": slots.len(),
        "iterations": iterations,
        "capture": {
            "dispatches": capture.launch_count,
            "unique_kernels": capture.unique_kernel_count,
            "sequence_hash": format!("{:016x}", capture.sequence_hash),
            "abi_contracts": contracts.len(),
            "batch_buffers": {
                "x_batch": format!("{:016x}", state.pbs.x_batch.buf.as_ptr() as usize),
                "x_rot_batch": format!("{:016x}", state.pbs.x_rot_batch.buf.as_ptr() as usize),
                "x_norm_batch": format!("{:016x}", state.pbs.x_norm_batch.buf.as_ptr() as usize),
                "positions": format!("{:016x}", state.pbs.positions.buf.as_ptr() as usize),
                "tokens": format!("{:016x}", state.pbs.tokens.buf.as_ptr() as usize),
                "final_hidden": format!("{:016x}", state.final_hidden.buf.as_ptr() as usize),
                "logits": format!("{:016x}", state.logits.buf.as_ptr() as usize),
                "lm_rot": format!("{:016x}", state.lm_rot.buf.as_ptr() as usize),
            },
            "first_launches": captured_launches,
        },
        "retained": {
            "dispatches": identity.dispatch_count,
            "packets": identity.packet_count,
            "queue_id": identity.queue_id,
            "command_dwords": identity.command_dwords,
        },
        "prefix_localization": prefix_localization,
        "observations": observations,
    }))
}

fn repeat_suffix(prompt: &[u32], output: &[u32], capacity: usize) -> Vec<u32> {
    let total = prompt.len() + output.len();
    let start = total.saturating_sub(capacity);
    let mut suffix = Vec::with_capacity(total - start);
    if start < prompt.len() {
        suffix.extend_from_slice(&prompt[start..]);
        suffix.extend_from_slice(output);
    } else {
        suffix.extend_from_slice(&output[start - prompt.len()..]);
    }
    suffix
}

fn job_seed(seed: u32, index: usize) -> u32 {
    let mut state = seed ^ (index as u32).wrapping_mul(0x9e37_79b9);
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    if state == 0 {
        0xa341_316c
    } else {
        state
    }
}

fn chatml_tokens(tokenizer: &Tokenizer, messages: &[Message]) -> Vec<u32> {
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let mut out = Vec::new();
    for message in messages {
        out.extend_from_slice(&im_start);
        out.extend_from_slice(&tokenizer.encode(&message.role));
        out.extend_from_slice(&nl);
        out.extend_from_slice(&tokenizer.encode(&message.content));
        out.extend_from_slice(&im_end);
        out.extend_from_slice(&nl);
    }
    out.extend_from_slice(&im_start);
    out.extend_from_slice(&tokenizer.encode("assistant"));
    out.extend_from_slice(&nl);
    out
}

fn row_tokens(row: &InputRow, tokenizer: &Tokenizer, raw_prompt: bool) -> Result<Vec<u32>, String> {
    let tokens = if let Some(tokens) = &row.tokens {
        tokens.clone()
    } else if let Some(messages) = &row.messages {
        chatml_tokens(tokenizer, messages)
    } else if let Some(prompt) = &row.prompt {
        if raw_prompt {
            tokenizer.encode(prompt)
        } else {
            chatml_tokens(
                tokenizer,
                &[Message {
                    role: "user".to_string(),
                    content: prompt.clone(),
                }],
            )
        }
    } else {
        return Err("row needs one of prompt, messages, or tokens".to_string());
    };
    if tokens.is_empty() {
        return Err("tokenized prompt is empty".to_string());
    }
    Ok(tokens)
}

fn load_jobs(args: &Args, tokenizer: &Tokenizer) -> Result<VecDeque<Job>, String> {
    let file = File::open(&args.input).map_err(|e| format!("open input: {e}"))?;
    let mut jobs = VecDeque::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read input line {}: {e}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        if index % args.shard_count != args.shard_index {
            continue;
        }
        let row: InputRow = if line.trim_start().starts_with('"') {
            let prompt: String = serde_json::from_str(&line)
                .map_err(|e| format!("parse input line {}: {e}", index + 1))?;
            InputRow {
                id: Value::Null,
                prompt: Some(prompt),
                messages: None,
                tokens: None,
                max_new_tokens: None,
            }
        } else {
            serde_json::from_str(&line)
                .map_err(|e| format!("parse input line {}: {e}", index + 1))?
        };
        let prompt_tokens = row_tokens(&row, tokenizer, args.raw_prompt)?;
        let max_new = row.max_new_tokens.unwrap_or(args.max_new);
        if prompt_tokens.len() + max_new > args.max_seq {
            return Err(format!(
                "input row {} needs {} prompt + {} output tokens, exceeding --max-seq {}",
                index + 1,
                prompt_tokens.len(),
                max_new,
                args.max_seq
            ));
        }
        jobs.push_back(Job {
            index,
            id: row.id,
            prompt_tokens,
            output_tokens: Vec::with_capacity(max_new),
            max_new,
        });
    }
    Ok(jobs)
}

#[allow(clippy::too_many_arguments)]
fn seed_lane(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    lane: usize,
    job: Job,
    args: &Args,
) -> Result<Slot, String> {
    state
        .reset_lane(gpu, config, lane)
        .map_err(|e| format!("reset lane {lane}: {e}"))?;
    state
        .prefill_lane(gpu, weights, config, scratch, lane, &job.prompt_tokens)
        .map_err(|e| format!("prefill lane {lane}: {e}"))?;
    let next_pos = job.prompt_tokens.len();
    let rng_state = job_seed(args.seed, job.index);
    let history = repeat_suffix(
        &job.prompt_tokens,
        &job.output_tokens,
        args.repeat_window.min(state.sample_repeat_capacity),
    );
    let (next_token, rng_state) = state
        .sample_lane_product(
            gpu,
            config,
            lane,
            &history,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_p,
            rng_state,
            args.repeat_penalty,
            args.presence_penalty,
            args.frequency_penalty,
        )
        .map_err(|e| format!("sample refill lane {lane}: {e}"))?;
    Ok(Slot {
        job: Some(job),
        next_token,
        next_pos,
        rng_state,
    })
}

/// Seed a complete fixed-shape batch through the independent decode path.
///
/// For equal-length prompts, stepping all lanes by prompt position turns
/// `batch * prompt_len` serial prefill work into `prompt_len` full-batch
/// launches. The state transition is the same one used for autoregressive
/// decode: each lane owns disjoint KV and recurrent-state storage, and logits
/// after the last prompt position are ready for the first sampled token.
#[allow(clippy::too_many_arguments)]
fn seed_equal_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &Qwen35Scratch,
    state: &mut Qwen35DecodeBatchState,
    jobs: Vec<Job>,
    args: &Args,
) -> Result<Vec<Slot>, String> {
    let prompt_len = jobs
        .first()
        .map(|job| job.prompt_tokens.len())
        .ok_or_else(|| "cannot seed an empty batch".to_string())?;
    if prompt_len == 0
        || jobs.len() != state.max_batch
        || jobs.iter().any(|job| job.prompt_tokens.len() != prompt_len)
    {
        return Err("batched seed requires a full batch of equal-length prompts".to_string());
    }

    let mut tokens = vec![0u32; jobs.len()];
    let mut positions = vec![0usize; jobs.len()];
    for position in 0..prompt_len {
        for (lane, job) in jobs.iter().enumerate() {
            tokens[lane] = job.prompt_tokens[position];
            positions[lane] = position;
        }
        qwen35::forward_decode_batch(gpu, weights, config, &tokens, &positions, state, scratch)
            .map_err(|e| format!("batched seed position {position}: {e}"))?;
    }

    let repeat_capacity = args.repeat_window.min(state.sample_repeat_capacity);
    let mut repeat_tokens = vec![0u32; jobs.len() * state.sample_repeat_capacity];
    let mut repeat_lengths = vec![0u32; jobs.len()];
    let rng_states: Vec<u32> = jobs
        .iter()
        .map(|job| job_seed(args.seed, job.index))
        .collect();
    for (lane, job) in jobs.iter().enumerate() {
        let history = repeat_suffix(&job.prompt_tokens, &job.output_tokens, repeat_capacity);
        repeat_lengths[lane] = history.len() as u32;
        let start = lane * state.sample_repeat_capacity;
        repeat_tokens[start..start + history.len()].copy_from_slice(&history);
    }
    let sampled = state
        .sample_product(
            gpu,
            config,
            jobs.len(),
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_p,
            args.repeat_penalty,
            args.presence_penalty,
            args.frequency_penalty,
        )
        .map_err(|e| format!("sample batched seed: {e}"))?;

    Ok(jobs
        .into_iter()
        .zip(sampled)
        .map(|(job, (next_token, rng_state))| Slot {
            job: Some(job),
            next_token,
            next_pos: prompt_len,
            rng_state,
        })
        .collect())
}

fn equal_prompt_batch(jobs: &VecDeque<Job>, batch: usize) -> bool {
    jobs.len() >= batch
        && jobs
            .front()
            .map(|first| {
                jobs.iter()
                    .take(batch)
                    .all(|job| job.prompt_tokens.len() == first.prompt_tokens.len())
            })
            .unwrap_or(false)
}

fn idle_slot(eos: u32, max_seq: usize) -> Slot {
    Slot {
        job: None,
        next_token: eos,
        next_pos: max_seq.saturating_sub(1),
        rng_state: 0,
    }
}

fn write_completion(
    writer: &mut BufWriter<File>,
    tokenizer: &Tokenizer,
    job: Job,
    finish_reason: &str,
    args: &Args,
) -> Result<(), String> {
    let completion = tokenizer.decode(&job.output_tokens);
    let row = json!({
        "index": job.index,
        "id": job.id,
        "completion": completion,
        "completion_tokens": job.output_tokens,
        "finish_reason": finish_reason,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p.unwrap_or(0.0),
            "repeat_penalty": args.repeat_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "repeat_window": args.repeat_window,
        }
    });
    serde_json::to_writer(&mut *writer, &row).map_err(|e| format!("write output: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write output newline: {e}"))
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    install_startup_config(&args)?;
    let mut hfq = HfqFile::open(Path::new(&args.model)).map_err(|e| format!("open model: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("read config: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("load tokenizer: {e}"))?;
    let mut jobs = load_jobs(&args, &tokenizer)?;
    if jobs.is_empty() {
        File::create(&args.output).map_err(|e| format!("create output: {e}"))?;
        return Ok(());
    }

    eprintln!(
        "batch-generator: jobs={} batch={} shard={}/{} max_seq={} max_new={} temp={} top_p={} top_k={:?} min_p={} repeat={} presence={} frequency={} window={}",
        jobs.len(),
        args.batch,
        args.shard_index,
        args.shard_count,
        args.max_seq,
        args.max_new,
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p.unwrap_or(0.0),
        args.repeat_penalty,
        args.presence_penalty,
        args.frequency_penalty,
        args.repeat_window,
    );
    let mut gpu = Gpu::init().map_err(|e| format!("GPU init: {e}"))?;
    let weights = {
        let mut source = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout)
    }
    .map_err(|e| format!("load weights: {e}"))?;
    let scratch =
        Qwen35Scratch::new(&mut gpu, &config, 256).map_err(|e| format!("allocate scratch: {e}"))?;
    let active_n = args.batch.min(jobs.len());
    let mut state = Qwen35DecodeBatchState::new(&mut gpu, &config, active_n, args.max_seq)
        .map_err(|e| format!("allocate batch state: {e}"))?;
    let output = File::create(&args.output).map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(output);
    let eos = config.eos_token;
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end = (im_end.len() == 1).then_some(im_end[0]);
    let wall_start = Instant::now();
    let seed_start = Instant::now();
    let equal_prompt_len = equal_prompt_batch(&jobs, active_n);
    let use_batched_seed =
        args.batched_seed && args.shadow_iterations.is_none() && equal_prompt_len;
    let mut slots = if use_batched_seed {
        let seed_jobs: Vec<Job> = jobs.drain(..active_n).collect();
        seed_equal_batch(
            &mut gpu, &weights, &config, &scratch, &mut state, seed_jobs, &args,
        )?
    } else {
        let mut slots = Vec::with_capacity(active_n);
        for lane in 0..active_n {
            let job = jobs.pop_front().expect("active_n bounded by jobs");
            slots.push(seed_lane(
                &mut gpu, &weights, &config, &scratch, &mut state, lane, job, &args,
            )?);
        }
        slots
    };
    let seed_time = seed_start.elapsed();
    let wave_refill = args.wave_refill && use_batched_seed;
    eprintln!(
        "seed: route={} refill={} lanes={} elapsed={:.3}s",
        if use_batched_seed {
            "independent-batch"
        } else {
            "sequential"
        },
        if wave_refill { "wave" } else { "continuous" },
        active_n,
        seed_time.as_secs_f64(),
    );
    if let Some(iterations) = args.shadow_iterations {
        let report = run_batch_shadow_gate(
            &mut gpu, &weights, &config, &scratch, &mut state, &slots, iterations,
        )?;
        serde_json::to_writer_pretty(&mut writer, &report)
            .map_err(|error| format!("write batch shadow report: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("write batch shadow newline: {error}"))?;
        writer
            .flush()
            .map_err(|error| format!("flush batch shadow report: {error}"))?;
        eprintln!(
            "shadow: backend={} batch={} iterations={} exact={}",
            report["backend"].as_str().unwrap_or("unknown"),
            active_n,
            iterations,
            report["bit_exact"].as_bool().unwrap_or(false),
        );
        return if report["bit_exact"].as_bool() == Some(true) {
            Ok(())
        } else {
            Err("batch retained-launch shadow parity failed".to_string())
        };
    }
    let mut model_time = Duration::ZERO;
    let mut generated = 0usize;
    let mut batched_generated = 0usize;
    let mut completed = 0usize;
    let mut refill_time = Duration::ZERO;
    let mut output_time = Duration::ZERO;
    loop {
        // Consume the samples produced from the previous logits. If a lane
        // finishes, refill it before the next fixed-shape model launch.
        for lane in 0..active_n {
            loop {
                let tok = slots[lane].next_token;
                let Some(job) = slots[lane].job.as_mut() else {
                    break;
                };
                job.output_tokens.push(tok);
                generated += 1;
                let eos_hit = tok == eos || im_end == Some(tok);
                let length_hit = job.output_tokens.len() >= job.max_new;
                if !eos_hit && !length_hit {
                    break;
                }

                let finished = slots[lane].job.take().unwrap();
                let output_start = Instant::now();
                write_completion(
                    &mut writer,
                    &tokenizer,
                    finished,
                    if eos_hit { "stop" } else { "length" },
                    &args,
                )?;
                output_time += output_start.elapsed();
                completed += 1;
                if completed % 32 == 0 {
                    writer.flush().map_err(|e| format!("flush output: {e}"))?;
                    eprintln!("completed={completed} queued={}", jobs.len());
                }
                if let Some(replacement) = jobs.pop_front() {
                    if wave_refill {
                        jobs.push_front(replacement);
                        slots[lane] = idle_slot(eos, args.max_seq);
                        break;
                    }
                    let refill_start = Instant::now();
                    slots[lane] = seed_lane(
                        &mut gpu,
                        &weights,
                        &config,
                        &scratch,
                        &mut state,
                        lane,
                        replacement,
                        &args,
                    )?;
                    refill_time += refill_start.elapsed();
                    continue;
                }
                // Keep the fixed batch dense while other real lanes finish.
                // Reusing EOS at one valid position is harmless; this lane's
                // logits are ignored from here onward.
                slots[lane].next_token = eos;
                slots[lane].next_pos = slots[lane].next_pos.min(args.max_seq - 1);
                break;
            }
        }

        if slots.iter().all(|slot| slot.job.is_none()) {
            if jobs.is_empty() {
                break;
            }

            let refill_start = Instant::now();
            state
                .reset(&mut gpu)
                .map_err(|e| format!("reset batch for next wave: {e}"))?;
            let batched_wave = args.batched_seed && equal_prompt_batch(&jobs, active_n);
            slots = if batched_wave {
                let seed_jobs: Vec<Job> = jobs.drain(..active_n).collect();
                seed_equal_batch(
                    &mut gpu, &weights, &config, &scratch, &mut state, seed_jobs, &args,
                )?
            } else {
                let mut next_slots = Vec::with_capacity(active_n);
                for lane in 0..active_n {
                    if let Some(job) = jobs.pop_front() {
                        next_slots.push(seed_lane(
                            &mut gpu, &weights, &config, &scratch, &mut state, lane, job, &args,
                        )?);
                    } else {
                        next_slots.push(idle_slot(eos, args.max_seq));
                    }
                }
                next_slots
            };
            let elapsed = refill_start.elapsed();
            refill_time += elapsed;
            eprintln!(
                "wave: route={} lanes={} queued={} elapsed={:.3}s",
                if batched_wave {
                    "independent-batch"
                } else {
                    "sequential-tail"
                },
                slots.iter().filter(|slot| slot.job.is_some()).count(),
                jobs.len(),
                elapsed.as_secs_f64(),
            );
            continue;
        }

        let tokens: Vec<u32> = slots.iter().map(|slot| slot.next_token).collect();
        let positions: Vec<usize> = slots.iter().map(|slot| slot.next_pos).collect();
        let real_lanes = slots.iter().filter(|slot| slot.job.is_some()).count();
        let t = Instant::now();
        qwen35::forward_decode_batch(
            &mut gpu, &weights, &config, &tokens, &positions, &mut state, &scratch,
        )
        .map_err(|e| format!("batched forward: {e}"))?;
        let repeat_capacity = args.repeat_window.min(state.sample_repeat_capacity);
        let mut repeat_tokens = vec![0u32; active_n * state.sample_repeat_capacity];
        let mut repeat_lengths = vec![0u32; active_n];
        let rng_states: Vec<u32> = slots.iter().map(|slot| slot.rng_state).collect();
        for (lane, slot) in slots.iter().enumerate() {
            let Some(job) = &slot.job else {
                continue;
            };
            let history = repeat_suffix(&job.prompt_tokens, &job.output_tokens, repeat_capacity);
            repeat_lengths[lane] = history.len() as u32;
            let start = lane * state.sample_repeat_capacity;
            repeat_tokens[start..start + history.len()].copy_from_slice(&history);
        }
        let sampled = state
            .sample_product(
                &mut gpu,
                &config,
                active_n,
                &repeat_tokens,
                &repeat_lengths,
                &rng_states,
                args.temperature,
                args.top_p,
                args.top_k,
                args.min_p,
                args.repeat_penalty,
                args.presence_penalty,
                args.frequency_penalty,
            )
            .map_err(|e| format!("batched sample: {e}"))?;
        model_time += t.elapsed();
        batched_generated += real_lanes;
        for lane in 0..active_n {
            if slots[lane].job.is_some() {
                slots[lane].next_token = sampled[lane].0;
                slots[lane].rng_state = sampled[lane].1;
                slots[lane].next_pos += 1;
            }
        }
    }
    writer.flush().map_err(|e| format!("flush output: {e}"))?;
    let wall = wall_start.elapsed();
    let replay_observation = gpu.replay.replay_observation();
    let replay_identity = gpu.replay.prepared_route_identity();
    eprintln!(
        "done: completions={} tokens={} batched_tokens={} decode_model={:.1} tok/s wall={:.1} tok/s model_time={:.3}s wall={:.3}s",
        completed,
        generated,
        batched_generated,
        if model_time.is_zero() {
            0.0
        } else {
            batched_generated as f64 / model_time.as_secs_f64()
        },
        generated as f64 / wall.as_secs_f64().max(1e-9),
        model_time.as_secs_f64(),
        wall.as_secs_f64(),
    );
    eprintln!(
        "phases: seed={:.3}s refill={:.3}s output={:.3}s unaccounted={:.3}s",
        seed_time.as_secs_f64(),
        refill_time.as_secs_f64(),
        output_time.as_secs_f64(),
        wall.saturating_sub(seed_time + refill_time + output_time + model_time)
            .as_secs_f64(),
    );
    eprintln!(
        "redline: request={:?} state={:?} transport={} retained_replays={} first_position={:?} last_position={:?} dispatches={:?} packets={:?} command_dwords={:?}",
        gpu.replay.request(),
        gpu.replay.state(),
        gpu.replay.transport_name(),
        replay_observation.count,
        replay_observation.first_position,
        replay_observation.last_position,
        replay_identity.map(|identity| identity.dispatch_count),
        replay_identity.and_then(|identity| identity.packet_count),
        replay_identity.and_then(|identity| identity.command_dwords),
    );
    Ok(())
}
