// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Interactive REPL for hipfire — like `ollama run`.
//! Usage: hipfire-run <model.hfq> [--system "prompt"] [--kv givens4|givens2]
//!        hipfire-run <model.hfq> --prompt-file prompt.txt [--max-tokens N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("Build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    fn print_usage() {
        println!(
            "Usage: run <model.hfq> [options]\n\
             \n\
             Options:\n\
               --draft-model <path>       DFlash/MTP draft model path\n\
               --system, -s <prompt>      system prompt\n\
               --kv <q8|givens4|givens2>  KV cache mode (default: q8)\n\
               --temp <float>             sampling temperature (default: 0.3)\n\
               --max-seq <n>              context length (default: 4096)\n\
               --prompt-file <path>       run one prompt non-interactively\n\
               --max-tokens, --max <n>    max generated tokens for --prompt-file\n\
               --session-reset-smoke      run the session reset smoke test\n\
               --fp32-state               use FP32 DeltaNet state\n\
               --q8-state                 use Q8 DeltaNet state (default)\n\
               --q4-state                 use Q4 DeltaNet state\n\
               --speculative              enable speculative draft path\n\
               --spec-k <n>               speculative draft count (default: 4)\n\
               --no-penalty               disable repetition penalty\n\
               --help, -h                 print this help"
        );
    }

    fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
        hfq.tensors()
            .iter()
            .map(|t| {
                t.shape
                    .iter()
                    .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
            })
            .sum()
    }

    fn warn_tiny_model_state(path: &str, q: qwen35::StateQuant) {
        const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
        if q == qwen35::StateQuant::FP32 {
            return;
        }
        if let Ok(hfq) = HfqFile::open(Path::new(path)) {
            let params = hfq_parameter_count(&hfq);
            if params < TINY_MODEL_PARAMS {
                eprintln!(
                    "warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {:?})",
                    params as f64 / 1.0e9,
                    q
                );
            }
        }
    }

    fn is_bf16_artifact_path(path: &str) -> bool {
        Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| {
                s.to_ascii_lowercase()
                    .split(|c| matches!(c, '-' | '_' | '.'))
                    .any(|part| part == "bf16")
            })
            .unwrap_or(false)
    }

    fn write_json_pretty(path: &Path, value: &serde_json::Value) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, format!("{body}\n"))
    }

    fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
        std::process::Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|stdout| stdout.trim().to_string())
            .filter(|stdout| !stdout.is_empty())
    }

    fn git_dirty() -> Option<bool> {
        command_stdout("git", &["status", "--porcelain"]).map(|stdout| !stdout.is_empty())
    }

    fn command_digest(tool: &str, path: &Path) -> Option<String> {
        std::process::Command::new(tool)
            .arg(path)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|stdout| stdout.split_whitespace().next().map(str::to_string))
    }

    fn hipfire_runtime_context() -> serde_json::Value {
        use serde_json::json;

        let binary_path = std::env::current_exe().ok();
        json!({
            "schema": 1,
            "runner": "hipfire-runtime/examples/run",
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": command_stdout("git", &["rev-parse", "HEAD"]),
            "git_branch": command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_describe": command_stdout("git", &["describe", "--always", "--dirty", "--tags"]),
            "git_dirty": git_dirty(),
            "binary_path": binary_path.as_ref().map(|path| path.display().to_string()),
            "binary_hash": binary_path.as_deref().and_then(|path| command_digest("sha256sum", path)),
        })
    }

    fn write_oneshot_evidence(
        dir: &Path,
        prompt_file: &str,
        prompt_tokens: usize,
        emitted_tokens: usize,
        prefill_forward_calls: usize,
        decode_forward_calls: usize,
        prefill_secs: f64,
        decode_secs: f64,
        ttft_ms: f64,
        vram_used_mb: u64,
        vram_total_mb: u64,
    ) -> std::io::Result<()> {
        use serde_json::json;

        let prefill_tok_s = prompt_tokens as f64 / prefill_secs.max(1e-9);
        let decode_tok_s = emitted_tokens as f64 / decode_secs.max(1e-9);
        let runtime_context = hipfire_runtime_context();
        let base = json!({
            "case_id": "run_oneshot",
            "prompt_path": prompt_file,
            "hipfire_runtime_context": runtime_context,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "decode_tokens_emitted": emitted_tokens,
            },
        });

        write_json_pretty(
            &dir.join("performance.json"),
            &json!({
                "schema": 1,
                "kind": "performance",
                "hipfire_runtime_context": runtime_context,
                "records": [{
                    "case_id": "run_oneshot",
                    "prompt_path": prompt_file,
                    "hipfire_runtime_context": runtime_context,
                    "metrics": {
                        "tok_s": decode_tok_s,
                        "decode_tok_s": decode_tok_s,
                        "prefill_tok_s": prefill_tok_s,
                        "ttft_ms": ttft_ms,
                        "prompt_tokens": prompt_tokens,
                        "decode_tokens_emitted": emitted_tokens,
                    }
                }]
            }),
        )?;
        write_json_pretty(
            &dir.join("phase_timings.json"),
            &json!({
                "schema": 1,
                "kind": "phase_timings",
                "hipfire_runtime_context": runtime_context,
                "records": [{
                    "case_id": "run_oneshot",
                    "prompt_path": prompt_file,
                    "hipfire_runtime_context": runtime_context,
                    "metrics": {
                        "prefill_ms": prefill_secs * 1000.0,
                        "decode_ms": decode_secs * 1000.0,
                        "ttft_ms": ttft_ms,
                    }
                }]
            }),
        )?;
        write_json_pretty(
            &dir.join("memory.json"),
            &json!({
                "schema": 1,
                "kind": "memory",
                "hipfire_runtime_context": runtime_context,
                "records": [{
                    "case_id": "run_oneshot",
                    "prompt_path": prompt_file,
                    "hipfire_runtime_context": runtime_context,
                    "metrics": {
                        "vram_peak_bytes": (vram_used_mb as f64) * 1024.0 * 1024.0,
                        "vram_used_mb": vram_used_mb,
                        "vram_total_mb": vram_total_mb,
                    }
                }]
            }),
        )?;
        write_json_pretty(
            &dir.join("launch_counts.json"),
            &json!({
                "schema": 1,
                "kind": "launch_counts",
                "hipfire_runtime_context": runtime_context,
                "records": [{
                    "case_id": "run_oneshot",
                    "prompt_path": prompt_file,
                    "hipfire_runtime_context": runtime_context,
                    "metrics": {
                        "kernel_launches": prefill_forward_calls + decode_forward_calls,
                        "graph_launches": 0,
                        "memcpy_ops": emitted_tokens,
                        "model_forward_calls": prefill_forward_calls + decode_forward_calls,
                        "prefill_forward_calls": prefill_forward_calls,
                        "decode_forward_calls": decode_forward_calls,
                        "counting_scope": "model_forward_call_proxy"
                    },
                    "notes": "examples/run currently exposes model forward-call counts, not raw HIP kernel launch counters"
                }]
            }),
        )?;
        write_json_pretty(
            &dir.join("run_oneshot.json"),
            &json!({
                "schema": 1,
                "kind": "run_oneshot",
                "hipfire_runtime_context": runtime_context,
                "records": [base]
            }),
        )
    }

    fn histogram_object_u64(values: &[u64]) -> serde_json::Value {
        use serde_json::json;

        let mut object = serde_json::Map::new();
        for (idx, &count) in values.iter().enumerate() {
            if count > 0 {
                object.insert(idx.to_string(), json!(count));
            }
        }
        serde_json::Value::Object(object)
    }

    fn histogram_object_f64(values: &[f64]) -> serde_json::Value {
        use serde_json::json;

        let mut object = serde_json::Map::new();
        for (idx, &sum) in values.iter().enumerate() {
            if sum != 0.0 {
                object.insert(idx.to_string(), json!(sum));
            }
        }
        serde_json::Value::Object(object)
    }

    fn histogram_entropy(values: &[u64]) -> f64 {
        let total: u64 = values.iter().sum();
        if total == 0 {
            return 0.0;
        }
        values
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total as f64;
                -p * p.ln()
            })
            .sum()
    }

    fn write_moe_router_evidence(
        dir: &Path,
        prompt_file: &str,
        hist: qwen35::MoeRouterHistogram,
    ) -> std::io::Result<()> {
        use serde_json::json;

        if hist.routed_slots == 0 {
            return Ok(());
        }
        let runtime_context = hipfire_runtime_context();
        write_json_pretty(
            &dir.join("moe_router_histogram.json"),
            &json!({
                "schema": 1,
                "kind": "moe_router_histogram",
                "hipfire_runtime_context": runtime_context,
                "records": [{
                    "case_id": "run_oneshot",
                    "prompt_path": prompt_file,
                    "hipfire_runtime_context": runtime_context,
                    "metrics": {
                        "expert_hits": histogram_object_u64(&hist.topk_histogram),
                        "router_top1_histogram": histogram_object_u64(&hist.top1_histogram),
                        "router_topk_histogram": histogram_object_u64(&hist.topk_histogram),
                        "router_weight_sums": histogram_object_f64(&hist.weight_sums),
                        "router_entropy": histogram_entropy(&hist.topk_histogram),
                        "router_dropped_tokens": hist.dropped_indices,
                        "routed_tokens": hist.routed_tokens,
                        "routed_slots": hist.routed_slots,
                        "num_experts": hist.num_experts,
                        "k_top": hist.k_top,
                        "collection_scope": "qwen35_moe_decode_and_prefill_forward_calls",
                    }
                }]
            }),
        )
    }

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Parse flags
    let mut system_prompt: Option<String> = None;
    let mut kv_mode_str: String = "q8".to_string();
    let mut temp: f32 = 0.3;
    let mut max_seq: usize = 4096;
    let mut state_quant = qwen35::StateQuant::Q8;
    let mut draft_model: Option<String> = None;
    let mut speculative = false;
    let mut spec_k: usize = 4;
    let mut no_penalty = false;
    let mut prompt_file: Option<String> = None;
    let mut evidence_dir: Option<String> = std::env::var("HIPFIRE_EVAL_EVIDENCE_DIR").ok();
    let mut oneshot_max_tokens: usize = 64;
    let mut session_reset_smoke = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--system" | "-s" => {
                i += 1;
                system_prompt = Some(args[i].clone());
            }
            "--kv" => {
                i += 1;
                kv_mode_str = args[i].clone();
            }
            "--fp32-state" => {
                state_quant = qwen35::StateQuant::FP32;
            }
            "--q8-state" => {
                state_quant = qwen35::StateQuant::Q8;
            }
            "--q4-state" => {
                state_quant = qwen35::StateQuant::Q4;
            }
            "--temp" => {
                i += 1;
                temp = args[i].parse().unwrap_or(0.3);
            }
            "--max-seq" => {
                i += 1;
                max_seq = args[i].parse().unwrap_or(4096);
            }
            "--prompt-file" => {
                i += 1;
                prompt_file = Some(args[i].clone());
            }
            "--session-reset-smoke" => {
                session_reset_smoke = true;
            }
            "--evidence-dir" => {
                i += 1;
                evidence_dir = Some(args[i].clone());
            }
            "--max-tokens" | "--max" => {
                i += 1;
                oneshot_max_tokens = args[i].parse().unwrap_or(64);
            }
            "--draft-model" => {
                i += 1;
                draft_model = Some(args[i].clone());
            }
            "--speculative" => {
                speculative = true;
            }
            "--spec-k" => {
                i += 1;
                spec_k = args[i].parse().unwrap_or(4).max(1);
            }
            "--no-penalty" => {
                no_penalty = true;
            }
            _ => {}
        }
        i += 1;
    }
    if is_bf16_artifact_path(model_path) {
        if kv_mode_str != "fp32" {
            eprintln!("BF16 artifact detected: forcing KV cache to fp32");
        }
        kv_mode_str = "fp32".to_string();
        state_quant = qwen35::StateQuant::FP32;
    }

    // Load model
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading {}...", model_path);

    use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
    fn parse_kv_mode(mode: &str) -> KvMode {
        match mode {
            "fp32" | "f32" => KvMode::Fp32,
            "q8" | "" => KvMode::Q8,
            "asym4" | "turbo4" => KvMode::Asym4,
            "asym3" | "turbo3" | "turbo" => KvMode::Asym3,
            "asym2" | "turbo2" => KvMode::Asym2,
            "fwht4" => KvMode::Fwht4,
            "fwht3" => KvMode::Fwht3,
            "fwht2" => KvMode::Fwht2,
            other => {
                panic!("unknown --kv {other}; expected fp32|q8|asym4|asym3|asym2|fwht4|fwht3|fwht2")
            }
        }
    }
    match state_quant {
        qwen35::StateQuant::FP32 => eprintln!("DeltaNet state: FP32"),
        qwen35::StateQuant::Q8 => eprintln!("DeltaNet state: Q8"),
        qwen35::StateQuant::Q4 => eprintln!("DeltaNet state: Q4 (half VRAM vs Q8)"),
    }
    warn_tiny_model_state(model_path, state_quant);
    let target_kv_mode = parse_kv_mode(&kv_mode_str);
    eprintln!("KV cache: {kv_mode_str} ({target_kv_mode:?})");
    let target_cfg = ModelSlotConfig {
        max_seq,
        kv_mode: target_kv_mode,
        repeat_window: 128,
        state_quant,
    };
    let mut target_slot = ModelSlot::load(&mut gpu, Path::new(model_path), "target", target_cfg)
        .expect("failed to load target model");
    let tokenizer = target_slot
        .load_tokenizer()
        .expect("failed to load tokenizer");

    // Optional draft model slot (Phase 1 of speculative decode). Validated for
    // tokenizer compatibility, smoke-tested, then parked. The REPL still runs
    // the target model alone until Phase 2 wires in the verify-and-accept loop.
    let mut draft_slot: Option<hipfire_arch_qwen35::speculative::ModelSlot> = None;
    if let Some(ref dpath) = draft_model {
        use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
        let vram_before = gpu.hip.get_vram_info().map(|(f, _)| f).unwrap_or(0);

        let draft_cfg = ModelSlotConfig {
            max_seq,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant,
        };

        eprintln!("Loading draft {}...", dpath);
        let mut slot = ModelSlot::load(&mut gpu, Path::new(dpath), "draft", draft_cfg)
            .expect("failed to load draft model");

        // Tokenizer compatibility check (vocab size + probe round-trip).
        let draft_tok = slot
            .load_tokenizer()
            .expect("draft has no tokenizer in HFQ metadata");
        assert_eq!(
            tokenizer.vocab_size(), draft_tok.vocab_size(),
            "tokenizer mismatch: target vocab={} draft vocab={} — speculative decode requires identical vocabularies",
            tokenizer.vocab_size(), draft_tok.vocab_size()
        );
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        assert_eq!(
            tokenizer.encode(probe),
            draft_tok.encode(probe),
            "tokenizer merge rules diverge between target and draft"
        );

        // Smoke test: 8 forward passes with a placeholder token. Must produce finite logits.
        for pos in 0..8 {
            slot.forward(&mut gpu, 1u32, pos)
                .expect("draft smoke-test forward failed");
        }
        let draft_logits = gpu.download_f32(&slot.scratch.logits).unwrap();
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());
        assert!(draft_ok, "draft smoke test produced non-finite logits");
        slot.reset_state(&mut gpu);

        let vram_after = gpu.hip.get_vram_info().map(|(f, _)| f).unwrap_or(0);
        let draft_mb = (vram_before.saturating_sub(vram_after)) as f64 / 1e6;
        eprintln!(
            "Draft: {} layers, dim={}, vocab={} — VRAM {:.0} MB, smoke test OK",
            slot.config.n_layers, slot.config.dim, slot.config.vocab_size, draft_mb
        );
        draft_slot = Some(slot);
    }

    // Speculative decode mode requires a draft model.
    let spec_active = speculative && draft_slot.is_some();
    if speculative && draft_slot.is_none() {
        eprintln!("--speculative ignored: no --draft-model provided");
    }
    // Snapshots for DeltaNet state rollback during verify-and-accept. Allocated
    // once and reused across REPL turns. Only materialized in spec mode.
    let mut target_snap: Option<hipfire_arch_qwen35::speculative::DeltaNetSnapshot> = None;
    let mut draft_snap: Option<hipfire_arch_qwen35::speculative::DeltaNetSnapshot> = None;
    if spec_active {
        use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
        target_snap = Some(DeltaNetSnapshot::new_for(&mut gpu, &target_slot.dn_state).unwrap());
        if let Some(ref d) = draft_slot {
            draft_snap = Some(DeltaNetSnapshot::new_for(&mut gpu, &d.dn_state).unwrap());
        }
        eprintln!(
            "Speculative decode: greedy, K={}, draft={}",
            spec_k,
            draft_slot.as_ref().map(|d| d.name.as_str()).unwrap_or("?")
        );
    }

    eprintln!(
        "Model: {} layers, dim={}, vocab={}",
        target_slot.config.n_layers, target_slot.config.dim, target_slot.config.vocab_size
    );
    eprintln!(
        "GPU: {} ({:.1} GB VRAM)",
        gpu.arch,
        gpu.hip
            .get_vram_info()
            .map(|(_, t)| t as f64 / 1e9)
            .unwrap_or(0.0)
    );
    if let Some(ref s) = system_prompt {
        eprintln!(
            "System: {}",
            if s.len() > 60 {
                format!("{}...", &s[..60])
            } else {
                s.clone()
            }
        );
    }
    eprintln!("Type /help for commands. Ctrl+C to quit.\n");

    // ChatML token IDs
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let sc = llama::SamplingConfig::text_thinking();

    if session_reset_smoke {
        use serde_json::json;

        let prompt_path = prompt_file
            .clone()
            .unwrap_or_else(|| "benchmarks/prompts/trains-meet.txt".to_string());
        let raw_prompt = std::fs::read_to_string(&prompt_path).unwrap_or_else(|e| {
            eprintln!("--session-reset-smoke: read {prompt_path}: {e}");
            std::process::exit(1);
        });
        let make_turn = |raw: &str| -> Vec<u32> {
            let input_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(raw.trim_end());
            let q_tokens = tokenizer.encode(&input_norm);
            let mut tokens = Vec::new();
            if let Some(ref sys) = system_prompt {
                let sys_tok = tokenizer.encode("system");
                let sys_content = tokenizer.encode(sys);
                tokens.extend_from_slice(&im_start);
                tokens.extend_from_slice(&sys_tok);
                tokens.extend_from_slice(&nl);
                tokens.extend_from_slice(&sys_content);
                tokens.extend_from_slice(&im_end);
                tokens.extend_from_slice(&nl);
            }
            tokens.extend_from_slice(&im_start);
            tokens.extend_from_slice(&user_tok);
            tokens.extend_from_slice(&nl);
            tokens.extend_from_slice(&q_tokens);
            tokens.extend_from_slice(&im_end);
            tokens.extend_from_slice(&nl);
            tokens.extend_from_slice(&im_start);
            tokens.extend_from_slice(&asst_tok);
            tokens.extend_from_slice(&nl);
            tokens
        };
        fn logits_hash(logits: &[f32]) -> String {
            let mut state = 0xcbf29ce484222325u64;
            for value in logits.iter().take(4096) {
                state ^= value.to_bits() as u64;
                state = state.wrapping_mul(0x100000001b3);
            }
            format!("fnv64:{state:016x}")
        }
        fn forward_prompt(
            gpu: &mut rdna_compute::Gpu,
            slot: &mut hipfire_arch_qwen35::speculative::ModelSlot,
            tokens: &[u32],
            start_pos: usize,
            top_p: f32,
        ) -> (u32, String) {
            for (i, &tok) in tokens.iter().enumerate() {
                slot.forward(gpu, tok, start_pos + i).unwrap();
            }
            let logits = gpu.download_f32(&slot.scratch.logits).unwrap();
            let token = llama::sample_top_p(&logits, 0.0, top_p);
            (token, logits_hash(&logits))
        }

        let recall_tokens = make_turn(&raw_prompt);
        let distractor_tokens = make_turn(
            "Remember this unrelated code word for the next turn: orchid. Reply with only OK.",
        );
        let started = Instant::now();
        target_slot.reset_state(&mut gpu);
        let (fresh_token, fresh_hash) =
            forward_prompt(&mut gpu, &mut target_slot, &recall_tokens, 0, sc.top_p);
        let dirty_start = recall_tokens.len();
        let _ = forward_prompt(
            &mut gpu,
            &mut target_slot,
            &distractor_tokens,
            dirty_start,
            sc.top_p,
        );
        let dirty_recall_start = dirty_start + distractor_tokens.len();
        let (dirty_token, dirty_hash) = forward_prompt(
            &mut gpu,
            &mut target_slot,
            &recall_tokens,
            dirty_recall_start,
            sc.top_p,
        );
        target_slot.reset_state(&mut gpu);
        let _ = gpu.hip.device_synchronize();
        let (reset_token, reset_hash) =
            forward_prompt(&mut gpu, &mut target_slot, &recall_tokens, 0, sc.top_p);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let pass = fresh_token == reset_token && fresh_hash == reset_hash;
        let report = json!({
            "schema": 1,
            "kind": "session_reset_smoke",
            "case_id": "multi_turn_reset_recall",
            "prompt_path": prompt_path,
            "hipfire_runtime_context": hipfire_runtime_context(),
            "status": if pass { "pass" } else { "fail" },
            "metrics": {
                "executor": "direct",
                "session_turns": 3,
                "reset_count": 1,
                "kv_reset": true,
                "dn_state_reset": true,
                "fresh_next_token": fresh_token,
                "dirty_next_token": dirty_token,
                "reset_next_token": reset_token,
                "fresh_logits_hash": fresh_hash,
                "dirty_logits_hash": dirty_hash,
                "reset_logits_hash": reset_hash,
                "recall_prompt_tokens": recall_tokens.len(),
                "distractor_prompt_tokens": distractor_tokens.len(),
                "elapsed_ms": elapsed_ms,
            }
        });
        if let Some(dir) = evidence_dir.as_deref() {
            if let Err(err) = write_json_pretty(&Path::new(dir).join("session_reset.json"), &report)
            {
                eprintln!("warning: failed to write session reset evidence to {dir}: {err}");
            }
        }
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        if pass {
            return;
        }
        std::process::exit(2);
    }

    let mut seq_pos: usize = 0;
    let mut conversation_tokens: Vec<u32> = Vec::new();
    let mut total_tokens: usize = 0;
    // Aggregate speculative decode stats across REPL turns (only populated when
    // --speculative is active). Shown via /stats.
    let mut spec_stats = hipfire_arch_qwen35::speculative::SpecStats::new(spec_k);

    if let Some(path) = prompt_file {
        let raw_prompt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("--prompt-file: read {path}: {e}");
            std::process::exit(1);
        });
        let input_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(raw_prompt.trim_end());
        let q_tokens = tokenizer.encode(&input_norm);
        let mut new_tokens: Vec<u32> = Vec::new();
        let collect_moe_router = evidence_dir.is_some() && target_slot.config.num_experts > 0;
        if collect_moe_router {
            qwen35::reset_moe_router_histogram(
                target_slot.config.num_experts,
                target_slot.config.num_experts_per_tok,
            );
        }
        if let Some(ref sys) = system_prompt {
            let sys_tok = tokenizer.encode("system");
            let sys_content = tokenizer.encode(sys);
            new_tokens.extend_from_slice(&im_start);
            new_tokens.extend_from_slice(&sys_tok);
            new_tokens.extend_from_slice(&nl);
            new_tokens.extend_from_slice(&sys_content);
            new_tokens.extend_from_slice(&im_end);
            new_tokens.extend_from_slice(&nl);
        }
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&user_tok);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&q_tokens);
        new_tokens.extend_from_slice(&im_end);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&asst_tok);
        new_tokens.extend_from_slice(&nl);

        let t_prefill = Instant::now();
        for (i, &tok) in new_tokens.iter().enumerate() {
            target_slot.forward(&mut gpu, tok, i).unwrap();
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        let mut logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
        let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);
        let mut emitted: Vec<u32> = Vec::new();
        let mut conversation_tokens = new_tokens.clone();
        let t_decode = Instant::now();
        let mut ttft_secs: Option<f64> = None;
        let mut decode_forward_calls = 0usize;
        while emitted.len() < oneshot_max_tokens {
            emitted.push(next_token);
            conversation_tokens.push(next_token);
            if ttft_secs.is_none() {
                ttft_secs = Some(t_decode.elapsed().as_secs_f64());
            }
            if next_token == target_slot.config.eos_token
                || im_end_token == Some(next_token)
                || tokenizer.is_terminator(next_token)
            {
                break;
            }
            let pos = new_tokens.len() + emitted.len() - 1;
            if pos >= max_seq {
                break;
            }
            target_slot.forward(&mut gpu, next_token, pos).unwrap();
            decode_forward_calls += 1;
            logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
            if !no_penalty {
                llama::apply_ngram_block(&mut logits, &conversation_tokens);
                llama::apply_repeat_penalty(
                    &mut logits,
                    &conversation_tokens,
                    sc.repeat_window,
                    sc.repeat_penalty,
                );
            }
            next_token = llama::sample_top_p(&logits, temp, sc.top_p);
        }
        let decode_secs = t_decode.elapsed().as_secs_f64();
        let text = tokenizer.decode(&emitted);
        println!("{text}");
        let (vram_free_bytes, vram_total_bytes) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        let vram_used_mb =
            ((vram_total_bytes.saturating_sub(vram_free_bytes)) as f64 / (1024.0 * 1024.0)) as u64;
        let vram_total_mb = (vram_total_bytes as f64 / (1024.0 * 1024.0)) as u64;
        let ttft_ms = (prefill_secs + ttft_secs.unwrap_or(0.0)) * 1000.0;
        if let Some(dir) = evidence_dir.as_deref() {
            if let Err(err) = write_oneshot_evidence(
                Path::new(dir),
                &path,
                new_tokens.len(),
                emitted.len(),
                new_tokens.len(),
                decode_forward_calls,
                prefill_secs,
                decode_secs,
                ttft_ms,
                vram_used_mb,
                vram_total_mb,
            ) {
                eprintln!("warning: failed to write --evidence-dir {dir}: {err}");
            }
            if collect_moe_router {
                if let Some(hist) = qwen35::take_moe_router_histogram() {
                    if let Err(err) = write_moe_router_evidence(Path::new(dir), &path, hist) {
                        eprintln!(
                            "warning: failed to write MoE router evidence to --evidence-dir {dir}: {err}"
                        );
                    }
                }
            }
        }
        eprintln!("=== BENCH METRICS ===");
        eprintln!("prompt_tokens: {}", new_tokens.len());
        eprintln!("prefill_secs: {:.4}", prefill_secs);
        eprintln!(
            "prefill_tok_s: {:.2}",
            new_tokens.len() as f64 / prefill_secs.max(1e-9)
        );
        eprintln!("ttft_ms: {:.2}", ttft_ms);
        eprintln!("decode_tokens_emitted: {}", emitted.len());
        eprintln!("decode_secs: {:.4}", decode_secs);
        eprintln!(
            "decode_tok_s: {:.2}",
            emitted.len() as f64 / decode_secs.max(1e-9)
        );
        eprintln!("decode_tau: 1.0000");
        eprintln!("decode_accept_rate: 1.0000");
        eprintln!("vram_used_mb: {}", vram_used_mb);
        eprintln!("vram_total_mb: {}", vram_total_mb);
        eprintln!("=====================");
        return;
    }

    // REPL
    let stdin = std::io::stdin();
    loop {
        // Prompt
        print!(">>> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 {
            break;
        } // EOF
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let input_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(input);
        let input: &str = &input_norm;
        if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
            tokenizer.dump_prompt_heat(input);
        }

        // Commands
        match input {
            "/quit" | "/exit" | "/q" => break,
            "/reset" | "/clear" => {
                seq_pos = 0;
                conversation_tokens.clear();
                total_tokens = 0;
                target_slot.reset_state(&mut gpu);
                if let Some(ref mut d) = draft_slot {
                    d.reset_state(&mut gpu);
                }
                eprintln!("Conversation reset.\n");
                continue;
            }
            "/help" | "/?" => {
                eprintln!("Commands:");
                eprintln!("  /reset  — clear conversation history");
                eprintln!("  /quit   — exit");
                eprintln!("  /stats  — show token counts and speed");
                eprintln!("  /help   — this message\n");
                continue;
            }
            "/stats" => {
                eprintln!("Position: {}/{} tokens used", seq_pos, max_seq);
                eprintln!("Total generated: {} tokens", total_tokens);
                if spec_active && spec_stats.cycles > 0 {
                    eprintln!(
                        "Speculative: {} cycles, tau={:.2} (accepted/cycle), committed/cycle={:.2}",
                        spec_stats.cycles,
                        spec_stats.tau(),
                        spec_stats.mean_committed()
                    );
                    eprint!("  acceptance histogram: ");
                    for (i, &c) in spec_stats.acceptance_hist.iter().enumerate() {
                        eprint!("a{}={} ", i, c);
                    }
                    eprintln!();
                }
                eprintln!();
                continue;
            }
            _ => {}
        }

        // Capacity guard
        let prompt_est = tokenizer.encode(input).len() + 20;
        if seq_pos + prompt_est + 512 > max_seq {
            eprintln!("[context full — auto-resetting]\n");
            seq_pos = 0;
            conversation_tokens.clear();
            target_slot.reset_state(&mut gpu);
            if let Some(ref mut d) = draft_slot {
                d.reset_state(&mut gpu);
            }
        }

        // Build ChatML tokens for this turn
        let q_tokens = tokenizer.encode(input);
        let mut new_tokens: Vec<u32> = Vec::new();

        // System prompt on first turn
        if seq_pos == 0 {
            if let Some(ref sys) = system_prompt {
                let sys_tok = tokenizer.encode("system");
                let sys_content = tokenizer.encode(sys);
                new_tokens.extend_from_slice(&im_start);
                new_tokens.extend_from_slice(&sys_tok);
                new_tokens.extend_from_slice(&nl);
                new_tokens.extend_from_slice(&sys_content);
                new_tokens.extend_from_slice(&im_end);
                new_tokens.extend_from_slice(&nl);
            }
        }
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&user_tok);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&q_tokens);
        new_tokens.extend_from_slice(&im_end);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&asst_tok);
        new_tokens.extend_from_slice(&nl);

        // Prefill: run the prompt through BOTH models so their state is
        // aligned at the same position. In non-spec mode the draft model is
        // still fed the prompt so that /toggle-mid-session works cleanly,
        // though the draft's state is unused until speculative is enabled.
        let t0 = Instant::now();
        for (i, &tok) in new_tokens.iter().enumerate() {
            target_slot.forward(&mut gpu, tok, seq_pos + i).unwrap();
            if spec_active {
                if let Some(ref mut d) = draft_slot {
                    d.forward(&mut gpu, tok, seq_pos + i).unwrap();
                }
            }
        }
        seq_pos += new_tokens.len();
        conversation_tokens.extend_from_slice(&new_tokens);

        let mut generated = 0usize;
        let mut in_thinking = false;
        let mut thinking_shown = false;
        // Capture EOS token IDs as plain values so the emit_token closure
        // doesn't borrow from `target_slot` (which would conflict with the
        // later &mut target_slot passed into spec_step_greedy).
        let eos_token = target_slot.config.eos_token;
        let im_end_token_val = im_end_token;

        // Helper closure: prints a token and returns true if generation should stop.
        let emit_token = |tok: u32,
                          conversation_tokens: &mut Vec<u32>,
                          in_thinking: &mut bool,
                          thinking_shown: &mut bool,
                          generated: &mut usize|
         -> bool {
            *generated += 1;
            conversation_tokens.push(tok);
            let text = tokenizer.decode(&[tok]);
            if text.contains("<think>") {
                *in_thinking = true;
                if !*thinking_shown {
                    eprint!("\x1b[2m");
                    *thinking_shown = true;
                }
            }
            if *in_thinking {
                eprint!("{}", text);
                if text.contains("</think>") {
                    *in_thinking = false;
                    eprint!("\x1b[0m\n");
                }
            } else {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            }
            tok == eos_token || im_end_token_val == Some(tok) || tokenizer.is_terminator(tok)
        };

        if spec_active {
            // Speculative decode loop. Each cycle drafts spec_k tokens, the
            // target verifies them sequentially (Phase 2 naive path), and the
            // accepted prefix + bonus is committed to both models.
            let ts = target_snap.as_mut().unwrap();
            let ds = draft_snap.as_mut().unwrap();
            let draft_ref = draft_slot.as_mut().unwrap();
            'outer: loop {
                let pos = seq_pos + generated;
                if pos + spec_k + 1 >= max_seq {
                    break;
                }

                let step = hipfire_arch_qwen35::speculative::spec_step_greedy(
                    &mut gpu,
                    &mut target_slot,
                    draft_ref,
                    pos,
                    spec_k,
                    ts,
                    ds,
                )
                .unwrap();
                spec_stats.record(&step);

                for tok in &step.committed {
                    let stop = emit_token(
                        *tok,
                        &mut conversation_tokens,
                        &mut in_thinking,
                        &mut thinking_shown,
                        &mut generated,
                    );
                    if stop {
                        break 'outer;
                    }
                    if generated >= 2048 {
                        break 'outer;
                    }
                }
            }
        } else {
            // Target-only generation path (baseline, unchanged behavior).
            let mut logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
            let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);
            loop {
                let stop = emit_token(
                    next_token,
                    &mut conversation_tokens,
                    &mut in_thinking,
                    &mut thinking_shown,
                    &mut generated,
                );
                if stop {
                    break;
                }
                if generated >= 2048 {
                    break;
                }

                let pos = seq_pos + generated - 1;
                if pos >= max_seq {
                    break;
                }
                target_slot.forward(&mut gpu, next_token, pos).unwrap();
                logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
                if !no_penalty {
                    llama::apply_ngram_block(&mut logits, &conversation_tokens);
                    llama::apply_repeat_penalty(
                        &mut logits,
                        &conversation_tokens,
                        sc.repeat_window,
                        sc.repeat_penalty,
                    );
                }
                next_token = llama::sample_top_p(&logits, temp, sc.top_p);
            }
        }

        seq_pos += generated;
        total_tokens += generated;
        conversation_tokens.extend_from_slice(&im_end);
        conversation_tokens.extend_from_slice(&nl);

        let elapsed = t0.elapsed();
        let tok_s = generated as f64 / elapsed.as_secs_f64();
        if spec_active && spec_stats.cycles > 0 {
            eprintln!(
                "\n\x1b[2m({} tokens, {:.1} tok/s | spec: {} cycles, tau={:.2})\x1b[0m\n",
                generated,
                tok_s,
                spec_stats.cycles,
                spec_stats.tau()
            );
        } else {
            eprintln!(
                "\n\x1b[2m({} tokens, {:.1} tok/s)\x1b[0m\n",
                generated, tok_s
            );
        }
    }

    eprintln!("Bye!");
}
