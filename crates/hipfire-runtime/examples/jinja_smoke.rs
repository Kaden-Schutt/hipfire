//! Stage 0/2-partial smoke harness — render via JinjaChatFrame and
//! run end-to-end inference on Qwen3.5/3.6 (dense or A3B). Validates
//! that the multi-turn JinjaChatFrame produces prompts that lead to
//! coherent generation across the full thinking-on / thinking-off /
//! tool-using axes.
//!
//! Usage:
//!   jinja_smoke --model PATH --scenario A|B|C --output JSONL [--max-gen N]
//!
//! Scenarios:
//!   A — Deliberately confusing prompt, thinking ENABLED
//!   B — Deliberately difficult prompt, thinking DISABLED
//!   C — Multi-turn (5 messages) with tool calls, thinking ENABLED
//!
//! Output: one JSONL record per run with rendered_prompt summary,
//! generation summary, token counts, prefill_ms, decode tok/s, and
//! coherence flags (max_freq + unique_ratio in first 256 / last 256).
//!
//! GPU device is selected via ROCR_VISIBLE_DEVICES on the parent
//! process — this binary itself doesn't manage device selection.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role, ToolCall};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum Scenario {
    A, // confusing prompt, thinking ON
    B, // difficult prompt, thinking OFF
    C, // multi-turn 5-msg with tool calls, thinking ON
}

fn parse_scenario(s: &str) -> Option<Scenario> {
    match s.to_ascii_uppercase().as_str() {
        "A" => Some(Scenario::A),
        "B" => Some(Scenario::B),
        "C" => Some(Scenario::C),
        _ => None,
    }
}

fn build_messages(scenario: Scenario) -> (Vec<Message>, Vec<serde_json::Value>) {
    match scenario {
        Scenario::A => (
            vec![
                Message {
                    role: Role::System,
                    content: "You are a careful reasoner. If the user's question is contradictory, ambiguous, or self-referential, identify the contradiction explicitly before answering.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "I am lying. The previous statement is true. Which of the two is the lie? Explain by listing the assumptions you have to make to answer, then commit to one.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
            ],
            vec![],
        ),
        Scenario::B => (
            vec![
                Message {
                    role: Role::System,
                    content: "You are a senior systems engineer. Answer concisely with concrete code where applicable.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "Write a Rust async function that fans out 1000 concurrent HTTPS GETs against an arbitrary URL list, caps in-flight at 32, retries on 5xx with exponential backoff capped at 30s, and aggregates response sizes. No external crates beyond `tokio`, `reqwest`, `futures`. Provide complete code that compiles.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
            ],
            vec![],
        ),
        Scenario::C => {
            let tools = vec![
                json!({
                    "type": "function",
                    "function": {
                        "name": "list_landmarks",
                        "description": "List notable landmarks in the given city",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }
                }),
                json!({
                    "type": "function",
                    "function": {
                        "name": "get_population",
                        "description": "Return current population estimate for a city in millions",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }
                }),
            ];
            let messages = vec![
                Message {
                    role: Role::System,
                    content: "You are a travel concierge. Use the available tools to answer user questions about cities.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "What's the capital of France?".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: "The capital of France is Paris.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: "List 3 famous landmarks there using the list_landmarks tool, and also fetch its population.".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: "I'll fetch both.".into(),
                    tool_calls: vec![
                        ToolCall {
                            name: "list_landmarks".into(),
                            arguments: json!({ "city": "Paris" }),
                        },
                    ],
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: "Eiffel Tower\nLouvre Museum\nNotre-Dame Cathedral".into(),
                    tool_call_id: Some("call_landmarks_paris".into()),
                    tool_calls: vec![],
                },
            ];
            (messages, tools)
        }
    }
}

fn enable_thinking_for(s: Scenario) -> bool {
    match s {
        Scenario::A | Scenario::C => true,
        Scenario::B => false,
    }
}

/// Window stats for coherence scoring.
fn window_stats(toks: &[u32], window: usize) -> (f32, f32) {
    let slice: &[u32] = if toks.len() <= window { toks } else { &toks[toks.len() - window..] };
    if slice.is_empty() { return (0.0, 0.0); }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in slice { *counts.entry(t).or_insert(0) += 1; }
    let max_freq = (*counts.values().max().unwrap_or(&0) as f32) / (slice.len() as f32);
    let unique_ratio = (counts.len() as f32) / (slice.len() as f32);
    (max_freq, unique_ratio)
}

fn parse_args() -> (String, Scenario, String, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut scenario: Option<Scenario> = None;
    let mut output: Option<String> = None;
    let mut max_gen: usize = 4096;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args.get(i+1).cloned(); i += 2; }
            "--scenario" => { scenario = args.get(i+1).and_then(|s| parse_scenario(s)); i += 2; }
            "--output" => { output = args.get(i+1).cloned(); i += 2; }
            "--max-gen" => { max_gen = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(4096); i += 2; }
            _ => { i += 1; }
        }
    }
    let model = model.expect("--model PATH required");
    let scenario = scenario.expect("--scenario A|B|C required");
    let output = output.expect("--output JSONL required");
    (model, scenario, output, max_gen)
}

fn main() {
    let (model_path, scenario, output_path, max_gen) = parse_args();

    eprintln!("=== jinja_smoke ===");
    eprintln!("model: {}", model_path);
    eprintln!("scenario: {:?}", scenario);
    eprintln!("max_gen: {}", max_gen);

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let template = hfq.chat_template().expect("model lacks chat_template");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not in HFQ metadata");

    let (messages, tools) = build_messages(scenario);
    let enable_thinking = enable_thinking_for(scenario);

    let frame = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: None,
        user: "",
        enable_thinking,
        bos_token: None,
    };

    let t_render = Instant::now();
    let rendered = match frame.render_messages(
        &messages,
        if tools.is_empty() { None } else { Some(&tools) },
        None,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("RENDER FAILED: {e}");
            let rec = json!({
                "model": model_path, "scenario": format!("{:?}", scenario),
                "ok": false, "stage": "render", "error": e,
            });
            fs::write(&output_path, format!("{}\n", rec)).expect("write output");
            std::process::exit(1);
        }
    };
    let render_ms = t_render.elapsed().as_millis();
    eprintln!("render_ms: {} | rendered_len: {} bytes", render_ms, rendered.len());

    let prompt_tokens = tokenizer.encode(&rendered);
    eprintln!("prompt_tokens: {}", prompt_tokens.len());

    // Load model + run inference. Reuses arch-qwen35 primitives the
    // daemon's AR path uses; the smoke is bit-equivalent to a daemon
    // chat-completion call where the prompt is pre-rendered.
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    // KV cache sized for prompt + max_gen + headroom. Use asym3 (KV
    // 5.5x compressed) for the bigger models to keep each smoke under
    // a single 32 GB R9700 budget. asym3 is the daemon default for 27B
    // and A3B per the existing config.
    let kv_seq = (prompt_tokens.len() + max_gen + 64).max(2048);
    let mut kv_cache = KvCache::new_gpu_asym3(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv cache");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    // Sequential prefill, AR loop (greedy from the deterministic CPU
    // sampler — matches daemon temp=0.0 path for reproducibility).
    let t_prefill = Instant::now();
    let mut logits = vec![0.0f32; config.vocab_size];
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv_cache, &mut dn_state, &scratch)
            .expect("prefill forward");
        if pos == prompt_tokens.len() - 1 {
            logits = gpu.download_f32(&scratch.logits).expect("download logits");
        }
    }
    let prefill_ms = t_prefill.elapsed().as_millis();
    eprintln!("prefill: {}ms ({:.0} tok/s)", prefill_ms,
        prompt_tokens.len() as f64 * 1000.0 / (prefill_ms as f64).max(1.0));

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let endoftext = tokenizer.encode("<|endoftext|>");
    let endoftext_token = if endoftext.len() == 1 { Some(endoftext[0]) } else { None };

    let sc = llama::SamplingConfig::text_thinking();
    let temp = if enable_thinking { sc.think_temp } else { sc.answer_temp };

    let t_decode = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut token_history: Vec<u32> = prompt_tokens.clone();

    // Apply CPU-side repeat-penalty pre-pass (matches the daemon's
    // sampler::sample path). Without this, top-p sampling at temp 0.3
    // collapses to a greedy attractor on hard prompts and produces
    // visible loops in the last 256 tokens. text_thinking() defaults
    // to penalty=1.15, window=128.
    let sample_one = |logits: &mut [f32], history: &[u32]| -> u32 {
        if sc.repeat_penalty != 1.0 && sc.repeat_window > 0 {
            llama::apply_repeat_penalty(logits, history, sc.repeat_window, sc.repeat_penalty);
        }
        llama::sample_top_p(logits, temp, sc.top_p)
    };

    let mut next_token = sample_one(&mut logits, &token_history);

    for _ in 0..max_gen {
        generated.push(next_token);
        token_history.push(next_token);
        if Some(next_token) == im_end_token { break; }
        if Some(next_token) == endoftext_token { break; }
        if next_token == config.eos_token { break; }

        let pos = prompt_tokens.len() + generated.len() - 1;
        qwen35::forward_scratch(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &mut dn_state, &scratch)
            .expect("decode forward");
        logits = gpu.download_f32(&scratch.logits).expect("download logits");
        next_token = sample_one(&mut logits, &token_history);
    }
    let decode_ms = t_decode.elapsed().as_millis();
    let decode_tok_s = (generated.len() as f64) * 1000.0 / (decode_ms as f64).max(1.0);

    let gen_text = tokenizer.decode(&generated);
    let (max_freq_first, unique_first) = window_stats(&generated[..generated.len().min(256)], 256);
    let (max_freq_last, unique_last) = window_stats(&generated, 256);

    let rendered_first = &rendered[..rendered.len().min(400)];
    let rendered_last_start = rendered.len().saturating_sub(200);
    let rendered_last = &rendered[rendered_last_start..];

    let gen_first = &gen_text.as_str()[..gen_text.len().min(500)];
    let gen_last_start = gen_text.len().saturating_sub(300);
    let gen_last = &gen_text.as_str()[gen_last_start..];

    let rec = json!({
        "model": model_path,
        "scenario": format!("{:?}", scenario),
        "ok": true,
        "enable_thinking": enable_thinking,
        "render_ms": render_ms,
        "rendered_len_bytes": rendered.len(),
        "rendered_first_400": rendered_first,
        "rendered_last_200": rendered_last,
        "prompt_tokens": prompt_tokens.len(),
        "max_gen": max_gen,
        "generated_tokens": generated.len(),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_s,
        "gen_first_500": gen_first,
        "gen_last_300": gen_last,
        "max_freq_first_256": max_freq_first,
        "unique_ratio_first_256": unique_first,
        "max_freq_last_256": max_freq_last,
        "unique_ratio_last_256": unique_last,
        "stopped_im_end": Some(generated.last().copied()) == im_end_token.map(Some),
        "stopped_endoftext": Some(generated.last().copied()) == endoftext_token.map(Some),
    });

    let mut out = fs::File::create(&output_path).expect("create output");
    writeln!(out, "{}", rec).expect("write output");

    eprintln!("=== summary ===");
    eprintln!("generated: {} tokens, {:.0} tok/s", generated.len(), decode_tok_s);
    eprintln!("max_freq first/last: {:.3} / {:.3}", max_freq_first, max_freq_last);
    eprintln!("unique_ratio first/last: {:.3} / {:.3}", unique_first, unique_last);
    eprintln!("output: {}", output_path);
}
