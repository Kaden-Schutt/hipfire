// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline fixed-slot sampled generator for Qwen3.5/A3B.
//!
//! Correctness/performance oracle for the independent continuous-batch decode
//! path. Input JSONL accepts one of:
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

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("qwen35_batch_generate requires --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "deltanet")]
fn run() -> Result<(), String> {
    use hipfire_arch_qwen35::qwen35::{self, Qwen35DecodeBatchState, Qwen35Scratch, Qwen35Weights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::Gpu;
    use serde::Deserialize;
    use serde_json::{json, Value};
    use std::collections::{HashSet, VecDeque};
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
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
        batched_seed: bool,
        wave_refill: bool,
        resume: bool,
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
                batched_seed: true,
                wave_refill: false,
                resume: false,
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
                    "--max-seq" => {
                        args.max_seq = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                    }
                    "--max-new" => {
                        args.max_new = value(&mut it, &flag)?.parse().map_err(|_| flag)?
                    }
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
                    "--raw-prompt" => args.raw_prompt = true,
                    "--sequential-seed" => args.batched_seed = false,
                    "--wave-refill" => args.wave_refill = true,
                    "--resume" => args.resume = true,
                    "--help" | "-h" => return Err(
                        "usage: qwen35_batch_generate MODEL --input IN.jsonl --output OUT.jsonl \
                             [--batch 16] [--max-seq 4096] [--max-new 512] \
                             [--temperature 1] [--top-p .95] [--top-k 20] [--seed N] \
                             [--min-p 0] [--repeat-penalty 1] \
                             [--presence-penalty 1.5] [--frequency-penalty 0] \
                             [--repeat-window 128] \
                             [--config CONFIG.toml] [--device PHYSICAL_ID] \
                             [--shard-index I --shard-count N] [--raw-prompt] \
                             [--sequential-seed] [--wave-refill] [--resume]"
                            .to_string(),
                    ),
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
                return Err(
                    "shard-count must be non-zero and shard-index < shard-count".to_string()
                );
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
        let resolved =
            hipfire_config::resolve(layers).map_err(|e| format!("resolve config: {e}"))?;
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

    fn row_tokens(
        row: &InputRow,
        tokenizer: &Tokenizer,
        raw_prompt: bool,
    ) -> Result<Vec<u32>, String> {
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

    fn expected_sampling(args: &Args) -> Value {
        json!({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p.unwrap_or(0.0),
            "repeat_penalty": args.repeat_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "repeat_window": args.repeat_window,
        })
    }

    fn load_completed_indices(args: &Args) -> Result<HashSet<usize>, String> {
        if !args.resume || !args.output.exists() {
            return Ok(HashSet::new());
        }
        let bytes = std::fs::read(&args.output).map_err(|e| format!("read resume output: {e}"))?;
        let expected = expected_sampling(args);
        let mut completed = HashSet::new();
        let mut offset = 0usize;
        for (line_index, raw_line) in bytes.split_inclusive(|byte| *byte == b'\n').enumerate() {
            let terminated = raw_line.ends_with(b"\n");
            let line = raw_line.strip_suffix(b"\n").unwrap_or(raw_line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.iter().all(u8::is_ascii_whitespace) {
                offset += raw_line.len();
                continue;
            }
            let row: Value = match serde_json::from_slice(line) {
                Ok(row) => row,
                Err(error) if !terminated && offset + raw_line.len() == bytes.len() => {
                    let output = OpenOptions::new()
                        .write(true)
                        .open(&args.output)
                        .map_err(|e| format!("open partial resume output for repair: {e}"))?;
                    output
                        .set_len(offset as u64)
                        .and_then(|_| output.sync_all())
                        .map_err(|e| format!("truncate partial resume output: {e}"))?;
                    eprintln!(
                        "resume: discarded incomplete final record from {} at byte {} ({error})",
                        args.output.display(),
                        offset
                    );
                    break;
                }
                Err(error) => {
                    return Err(format!(
                        "parse resume output {} line {}: {error}",
                        args.output.display(),
                        line_index + 1
                    ));
                }
            };
            let index = row
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    format!(
                        "resume output {} line {} has no valid index",
                        args.output.display(),
                        line_index + 1
                    )
                })?;
            if index % args.shard_count != args.shard_index {
                return Err(format!(
                    "resume output {} line {} index {} belongs to shard {}/{}",
                    args.output.display(),
                    line_index + 1,
                    index,
                    index % args.shard_count,
                    args.shard_count
                ));
            }
            if row.get("sampling") != Some(&expected) {
                return Err(format!(
                    "resume output {} line {} has different sampling parameters",
                    args.output.display(),
                    line_index + 1
                ));
            }
            if !completed.insert(index) {
                return Err(format!(
                    "resume output {} contains duplicate index {index}",
                    args.output.display()
                ));
            }
            offset += raw_line.len();
        }
        Ok(completed)
    }

    fn open_output(args: &Args) -> Result<File, String> {
        if !args.resume {
            return File::create(&args.output).map_err(|e| format!("create output: {e}"));
        }
        let mut output = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&args.output)
            .map_err(|e| format!("open resume output: {e}"))?;
        let length = output
            .metadata()
            .map_err(|e| format!("stat resume output: {e}"))?
            .len();
        if length > 0 {
            output
                .seek(SeekFrom::End(-1))
                .map_err(|e| format!("seek resume output: {e}"))?;
            let mut final_byte = [0u8; 1];
            output
                .read_exact(&mut final_byte)
                .map_err(|e| format!("read resume output tail: {e}"))?;
            if final_byte[0] != b'\n' {
                output
                    .write_all(b"\n")
                    .and_then(|_| output.flush())
                    .map_err(|e| format!("terminate resume output: {e}"))?;
            }
        }
        Ok(output)
    }

    fn load_jobs(
        args: &Args,
        tokenizer: &Tokenizer,
        completed: &HashSet<usize>,
    ) -> Result<VecDeque<Job>, String> {
        let file = File::open(&args.input).map_err(|e| format!("open input: {e}"))?;
        let mut jobs = VecDeque::new();
        let mut unmatched = completed.clone();
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("read input line {}: {e}", index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            if index % args.shard_count != args.shard_index {
                continue;
            }
            if unmatched.remove(&index) {
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
        if !unmatched.is_empty() {
            let mut indices: Vec<_> = unmatched.into_iter().collect();
            indices.sort_unstable();
            return Err(format!(
                "resume output contains {} indices absent from this input shard; first={:?}",
                indices.len(),
                &indices[..indices.len().min(8)]
            ));
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

    let args = Args::parse()?;
    install_startup_config(&args)?;
    let mut hfq = HfqFile::open(Path::new(&args.model)).map_err(|e| format!("open model: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("read config: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("load tokenizer: {e}"))?;
    let completed_before_start = load_completed_indices(&args)?;
    let mut jobs = load_jobs(&args, &tokenizer, &completed_before_start)?;
    if jobs.is_empty() {
        open_output(&args)?;
        eprintln!(
            "batch-generator: already complete, resumed={}",
            completed_before_start.len()
        );
        return Ok(());
    }

    eprintln!(
        "batch-generator: jobs={} resumed={} batch={} shard={}/{} max_seq={} max_new={} temp={} top_p={} top_k={:?} min_p={} repeat={} presence={} frequency={} window={}",
        jobs.len(),
        completed_before_start.len(),
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
    let mut state = Qwen35DecodeBatchState::new(
        &mut gpu,
        &config,
        active_n,
        args.max_seq,
        args.repeat_window,
    )
    .map_err(|e| format!("allocate batch state: {e}"))?;
    let output = open_output(&args)?;
    let mut writer = BufWriter::new(output);
    let eos = config.eos_token;
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end = (im_end.len() == 1).then_some(im_end[0]);
    let wall_start = Instant::now();
    let seed_start = Instant::now();
    let equal_prompt_len = equal_prompt_batch(&jobs, active_n);
    let use_batched_seed = args.batched_seed && equal_prompt_len;
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
    let mut model_time = Duration::ZERO;
    let mut generated = 0usize;
    let mut batched_generated = 0usize;
    let mut completed = completed_before_start.len();
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
    Ok(())
}
