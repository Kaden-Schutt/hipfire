// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent sampled-batch EP benchmark for Qwen3.x A3B.
//!
//! This is intentionally separate from `qwen35_batch_generate`: the existing
//! single-GPU DP/Redline route remains the certified product baseline while
//! this harness characterizes EP2, then EP2x2 as two independent processes.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("qwen35_batch_ep_bench requires --features deltanet");
}

#[cfg(feature = "deltanet")]
mod enabled {
    use hipfire_arch_qwen35::qwen35::{self, Qwen35DecodeBatchState, Qwen35Scratch, Qwen35Weights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::{DType, GpuTensor};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    struct Args {
        model: PathBuf,
        config: Option<PathBuf>,
        tp: usize,
        batch: usize,
        max_seq: usize,
        steps: usize,
        warmup: usize,
        prompt: String,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        seed: u32,
    }

    impl Args {
        fn parse() -> Result<Self, String> {
            let mut it = std::env::args().skip(1);
            let model = it.next().map(PathBuf::from).ok_or_else(|| {
                "usage: qwen35_batch_ep_bench MODEL [--tp 2] [--batch 128]".to_string()
            })?;
            let mut args = Self {
                model,
                config: None,
                tp: 2,
                batch: 128,
                max_seq: 4096,
                steps: 32,
                warmup: 4,
                prompt: "Explain why retained GPU command submission helps language-model decode."
                    .to_string(),
                temperature: 1.0,
                top_p: 0.95,
                top_k: Some(20),
                seed: 0x1357_9bdf,
            };
            while let Some(flag) = it.next() {
                let value = |it: &mut std::iter::Skip<std::env::Args>, flag: &str| {
                    it.next().ok_or_else(|| format!("missing value for {flag}"))
                };
                match flag.as_str() {
                    "--config" => args.config = Some(PathBuf::from(value(&mut it, &flag)?)),
                    "--tp" => args.tp = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--batch" => args.batch = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--max-seq" => {
                        args.max_seq = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                    }
                    "--steps" => args.steps = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--warmup" => args.warmup = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--prompt" => args.prompt = value(&mut it, &flag)?,
                    "--temperature" => {
                        args.temperature = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                    }
                    "--top-p" => args.top_p = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--top-k" => {
                        let top_k = value(&mut it, &flag)?.parse().map_err(|_| flag)?;
                        args.top_k = (top_k != 0).then_some(top_k);
                    }
                    "--seed" => args.seed = value(&mut it, &flag)?.parse().map_err(|_| flag)?,
                    "--help" | "-h" => {
                        return Err("usage: qwen35_batch_ep_bench MODEL \
                             [--config CONFIG.toml] [--tp 2] [--batch 128] \
                             [--max-seq 4096] [--steps 32] [--warmup 4] \
                             [--prompt TEXT] [--temperature 1] [--top-p .95] \
                             [--top-k 20] [--seed N]"
                            .to_string());
                    }
                    _ => return Err(format!("unknown argument: {flag}")),
                }
            }
            if args.tp == 0
                || args.batch == 0
                || args.steps == 0
                || args.warmup >= args.steps
                || args.max_seq < 2
                || !(0.0..=1.0).contains(&args.top_p)
                || args.top_p == 0.0
            {
                return Err("invalid EP benchmark shape or sampling parameters".to_string());
            }
            Ok(args)
        }
    }

    fn install_startup_config(path: Option<&Path>) -> Result<(), String> {
        let (path, global) = if let Some(path) = path {
            (
                path.to_owned(),
                hipfire_config::load_toml_layer(path)
                    .map_err(|error| format!("load {}: {error}", path.display()))?,
            )
        } else {
            let loaded = hipfire_config::load_global(&hipfire_config::ConfigPaths::discover())
                .map_err(|error| format!("load local config: {error}"))?;
            (loaded.path, loaded.layer)
        };
        let mut layers = vec![hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::GlobalUser { path },
            layer: global,
        }];
        let environment = hipfire_config::load_env_layer()
            .map_err(|error| format!("load environment: {error}"))?;
        if !environment.values.is_empty() {
            layers.push(hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::LegacyEnv {
                    name: "HIPFIRE_*".into(),
                },
                layer: environment,
            });
        }
        let resolved =
            hipfire_config::resolve(layers).map_err(|error| format!("resolve config: {error}"))?;
        let process = hipfire_config::ProcessConfig::from_resolved(&resolved)
            .map_err(|error| format!("build process config: {error}"))?;
        hipfire_config::apply_device_visibility(&process)
            .map_err(|error| format!("apply device visibility: {error}"))?;
        let runtime = hipfire_runtime::config::RuntimeConfig::from_process_config(&process);
        hipfire_config::install_process_config(process)
            .map_err(|_| "process configuration was already initialized".to_string())?;
        hipfire_runtime::config::init_with(runtime)
            .map_err(|_| "runtime process configuration was already initialized".to_string())
    }

    fn fnv1a_u32(values: &[u32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in values {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        }
        hash
    }

    fn load_ep_weights(
        model: &Path,
        config: &qwen35::Qwen35Config,
        gpus: &mut Gpus,
        tp: usize,
    ) -> Result<Vec<Qwen35Weights>, String> {
        let shard = ShardConfig::new(tp, true, config.num_experts, ExpertAssign::Stride)
            .map_err(|error| format!("build EP shard: {error}"))?;
        let mut weights = Vec::with_capacity(tp);
        for rank in 0..tp {
            gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("bind rank {rank}: {error}"))?;
            let mut hfq = HfqFile::open(model)
                .map_err(|error| format!("open model for rank {rank}: {error}"))?;
            eprintln!("rank {rank}: loading owned expert shard");
            qwen35::set_ep_expert_shard(Some((shard.clone(), rank)));
            let loaded = {
                let mut source = qwen35::HfqSource::new(&mut hfq, config);
                let layout = qwen35::Layout::single(config.n_layers);
                qwen35::load_weights(
                    &mut source,
                    std::slice::from_mut(&mut gpus.devices[rank]),
                    &layout,
                )
            };
            qwen35::set_ep_expert_shard(None);
            weights
                .push(loaded.map_err(|error| format!("load rank {rank} expert shard: {error}"))?);
        }
        Ok(weights)
    }

    fn repeat_inputs(histories: &[Vec<u32>], capacity: usize) -> (Vec<u32>, Vec<u32>) {
        let mut repeat_tokens = vec![0_u32; histories.len() * capacity];
        let mut repeat_lengths = vec![0_u32; histories.len()];
        for (lane, history) in histories.iter().enumerate() {
            let len = history.len().min(capacity);
            let suffix = &history[history.len() - len..];
            let start = lane * capacity;
            repeat_tokens[start..start + len].copy_from_slice(suffix);
            repeat_lengths[lane] = len as u32;
        }
        (repeat_tokens, repeat_lengths)
    }

    fn sample(
        gpu: &mut rdna_compute::Gpu,
        state: &Qwen35DecodeBatchState,
        config: &qwen35::Qwen35Config,
        histories: &[Vec<u32>],
        rng_states: &[u32],
        args: &Args,
    ) -> Result<Vec<(u32, u32)>, String> {
        let (repeat_tokens, repeat_lengths) =
            repeat_inputs(histories, state.sample_repeat_capacity);
        state
            .sample_product(
                gpu,
                config,
                args.batch,
                &repeat_tokens,
                &repeat_lengths,
                rng_states,
                args.temperature,
                args.top_p,
                args.top_k,
                None,
                1.0,
                1.5,
                0.0,
            )
            .map_err(|error| format!("sample batch: {error}"))
    }

    pub fn run() -> Result<(), String> {
        let args = Args::parse()?;
        install_startup_config(args.config.as_deref())?;

        let hfq = HfqFile::open(&args.model)
            .map_err(|error| format!("open {}: {error}", args.model.display()))?;
        let config =
            qwen35::config_from_hfq(&hfq).map_err(|error| format!("read config: {error}"))?;
        if config.num_experts == 0 {
            return Err("EP benchmark requires an MoE model".to_string());
        }
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|error| format!("load tokenizer: {error}"))?;
        drop(hfq);
        let prompt_tokens = tokenizer.encode(&args.prompt);
        if prompt_tokens.is_empty() || prompt_tokens.len() + args.steps >= args.max_seq {
            return Err("prompt is empty or leaves insufficient decode capacity".to_string());
        }

        eprintln!(
            "EP batch: tp={} batch={} steps={} warmup={} prompt_tokens={} max_seq={}",
            args.tp,
            args.batch,
            args.steps,
            args.warmup,
            prompt_tokens.len(),
            args.max_seq,
        );
        let mut gpus =
            Gpus::init_tp(args.tp, config.n_layers).map_err(|error| format!("init EP: {error}"))?;
        if gpus.devices.len() != args.tp {
            return Err(format!(
                "EP init exposed {} ranks, expected {}",
                gpus.devices.len(),
                args.tp
            ));
        }
        for (rank, gpu) in gpus.devices.iter().enumerate() {
            eprintln!(
                "rank {rank}: logical_device={} arch={}",
                gpu.device_id, gpu.arch
            );
        }

        let weights = load_ep_weights(&args.model, &config, &mut gpus, args.tp)?;
        let mut states = Vec::with_capacity(args.tp);
        let mut scratch = Vec::with_capacity(args.tp);
        let mut partials: Vec<GpuTensor> = Vec::with_capacity(args.tp);
        for rank in 0..args.tp {
            gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("bind rank {rank}: {error}"))?;
            states.push(
                Qwen35DecodeBatchState::new(
                    &mut gpus.devices[rank],
                    &config,
                    args.batch,
                    args.max_seq,
                )
                .map_err(|error| format!("allocate rank {rank} batch state: {error}"))?,
            );
            scratch.push(
                Qwen35Scratch::new(&mut gpus.devices[rank], &config, 64)
                    .map_err(|error| format!("allocate rank {rank} scratch: {error}"))?,
            );
            partials.push(
                gpus.devices[rank]
                    .zeros(&[args.batch * config.dim], DType::F32)
                    .map_err(|error| format!("allocate rank {rank} routed partial: {error}"))?,
            );
        }
        let peer_access = gpus
            .enable_peer_all()
            .map_err(|error| format!("enable EP peer access: {error}"))?;
        hipfire_runtime::ep::ensure_rank_streams(&mut gpus)
            .map_err(|error| format!("create rank streams: {error}"))?;
        eprintln!("peer_access_enabled={peer_access}");

        // Seed every lane through the same prompt. Per-lane RNG states make
        // the first sampled token diverge before the measured window begins.
        let mut histories = vec![Vec::<u32>::new(); args.batch];
        for (position, &token) in prompt_tokens.iter().enumerate() {
            let tokens = vec![token; args.batch];
            let positions = vec![position; args.batch];
            qwen35::forward_decode_batch_ep(
                &mut gpus,
                &weights,
                &config,
                &tokens,
                &positions,
                &mut states,
                &scratch,
                &partials,
            )
            .map_err(|error| format!("EP prompt step {position}: {error}"))?;
            for history in &mut histories {
                history.push(token);
            }
        }

        let mut rng_states: Vec<u32> = (0..args.batch)
            .map(|lane| args.seed ^ (lane as u32).wrapping_mul(0x9e37_79b9))
            .collect();
        gpus.devices[0]
            .bind_thread()
            .map_err(|error| format!("bind sampling rank: {error}"))?;
        let initial = sample(
            &mut gpus.devices[0],
            &states[0],
            &config,
            &histories,
            &rng_states,
            &args,
        )?;
        let mut tokens: Vec<u32> = initial.iter().map(|sample| sample.0).collect();
        rng_states = initial.iter().map(|sample| sample.1).collect();

        let mut step_ms = Vec::with_capacity(args.steps);
        let mut output_tokens = Vec::with_capacity(args.batch * args.steps);
        for step in 0..args.steps {
            let position = prompt_tokens.len() + step;
            let positions = vec![position; args.batch];
            let started = Instant::now();
            qwen35::forward_decode_batch_ep(
                &mut gpus,
                &weights,
                &config,
                &tokens,
                &positions,
                &mut states,
                &scratch,
                &partials,
            )
            .map_err(|error| format!("EP decode step {step}: {error}"))?;
            for (lane, &token) in tokens.iter().enumerate() {
                histories[lane].push(token);
                output_tokens.push(token);
            }
            gpus.devices[0]
                .bind_thread()
                .map_err(|error| format!("bind sampling rank: {error}"))?;
            let sampled = sample(
                &mut gpus.devices[0],
                &states[0],
                &config,
                &histories,
                &rng_states,
                &args,
            )?;
            tokens = sampled.iter().map(|sample| sample.0).collect();
            rng_states = sampled.iter().map(|sample| sample.1).collect();
            step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        for rank in 0..args.tp {
            gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("bind final rank {rank}: {error}"))?;
            gpus.devices[rank]
                .hip
                .device_synchronize()
                .map_err(|error| format!("synchronize final rank {rank}: {error}"))?;
        }
        let mut rank_residuals = Vec::with_capacity(args.tp);
        for rank in 0..args.tp {
            gpus.devices[rank]
                .bind_thread()
                .map_err(|error| format!("bind residual rank {rank}: {error}"))?;
            let residual = states[rank]
                .pbs
                .x_batch
                .sub_offset(0, args.batch * config.dim);
            rank_residuals.push(
                gpus.devices[rank]
                    .download_f32(&residual)
                    .map_err(|error| format!("download rank {rank} residual: {error}"))?,
            );
        }
        let residual_exact = rank_residuals
            .iter()
            .skip(1)
            .all(|residual| residual == &rank_residuals[0]);
        let settled = &step_ms[args.warmup..];
        let settled_ms = settled.iter().sum::<f64>() / settled.len() as f64;
        let model_tok_s = args.batch as f64 * 1000.0 / settled_ms;
        let report = json!({
            "tp": args.tp,
            "batch": args.batch,
            "steps": args.steps,
            "warmup": args.warmup,
            "prompt_tokens": prompt_tokens.len(),
            "peer_access_enabled": peer_access,
            "rank_residual_bit_exact": residual_exact,
            "settled_step_ms": settled_ms,
            "model_tok_s": model_tok_s,
            "token_hash": format!("{:016x}", fnv1a_u32(&output_tokens)),
            "step_ms": step_ms,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("serialize report: {error}"))?
        );
        Ok(())
    }
}

#[cfg(feature = "deltanet")]
fn main() {
    if let Err(error) = enabled::run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
