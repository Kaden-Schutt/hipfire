//! Render-time microbench, with three timing paths to expose the cost
//! split between Environment setup, per-render template execution, and
//! the existing hand-rolled `ChatFrame::Plain` build path.
//!
//! Usage:
//!   render_bench <model.mq4> [iterations]
//!
//! Default 1000 iterations. Reports mean / p50 / p99 / min / max in
//! microseconds for each path.
//!
//! What this answers: "does Jinja rendering hurt perf vs ChatFrame::Plain
//! in production?" Production daemons would build the minijinja Environment
//! once at model load and reuse it per request — Path J2 measures that. The
//! current `JinjaChatFrame::render` impl rebuilds Env+parse each call
//! (Path J1), which is correct for Stage 0 dead-code but a real cost we
//! must cache before wiring into the daemon (tracked: Stage 2 follow-up).

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{
    AssistantPrefix, ChatFrame, JinjaChatFrame,
};
use hipfire_runtime::tokenizer::Tokenizer;
use minijinja::{context, Environment, Error, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

fn percentile(sorted_us: &[u64], pct: f64) -> u64 {
    if sorted_us.is_empty() { return 0; }
    let i = ((sorted_us.len() as f64) * pct).floor() as usize;
    sorted_us[i.min(sorted_us.len() - 1)]
}

fn report(label: &str, samples_us: &mut Vec<u64>) {
    samples_us.sort();
    let n = samples_us.len();
    let mean = samples_us.iter().sum::<u64>() / (n as u64);
    println!(
        "  {:<48} n={} mean={}us p50={}us p99={}us min={}us max={}us",
        label, n, mean,
        percentile(samples_us, 0.50),
        percentile(samples_us, 0.99),
        samples_us.first().copied().unwrap_or(0),
        samples_us.last().copied().unwrap_or(0),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: render_bench <model.mq4> [iters]");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let template = hfq.chat_template().expect("model lacks chat_template");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer not in HFQ metadata");

    let system = "You are a careful coding assistant. Respond concisely with concrete code where applicable.";
    let user = "Write a Rust async function that fans out 1000 concurrent HTTPS GETs against an arbitrary URL list, caps in-flight at 32, retries on 5xx with exponential backoff capped at 30s, and aggregates response sizes.";

    println!("=== render_bench ({} iterations) ===", iters);
    println!("model:    {}", model_path);
    println!("template: {} bytes", template.len());
    println!();

    // ── Path J0: Environment setup + template parse, ONCE.
    //
    // Measure how long the one-time setup takes. In production the
    // daemon would do this at load_model time and store the Env in
    // LoadedModel; per-request render reuses it.
    let t = Instant::now();
    let mut env_cached = Environment::new();
    env_cached.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env_cached.set_unknown_method_callback(unknown_method_callback);
    env_cached.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env_cached.add_template("chat", &template).expect("template parse");
    let env_setup_us = t.elapsed().as_micros() as u64;
    println!("Path J0 (one-time Env setup + template parse): {} us", env_setup_us);
    println!();

    // Build the bos_token + messages once; identical across all paths.
    let bos_token: String = String::from_utf8_lossy(
        &tokenizer.decode_bytes(&[tokenizer.bos_id])
    ).to_string();
    let empty_list: Vec<serde_json::Value> = Vec::new();
    let empty_map = serde_json::Map::<String, serde_json::Value>::new();
    let messages_single = vec![
        json!({ "role": "system", "content": system, "tool_calls": [] }),
        json!({ "role": "user", "content": user, "tool_calls": [] }),
    ];

    // ── Path J1: full JinjaChatFrame::render — rebuilds Env each call.
    // This is the CURRENT cost of `JinjaChatFrame::render` as committed.
    let frame_jinja = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: Some(system),
        user,
        enable_thinking: true,
        bos_token: None,
    };
    let mut j1: Vec<u64> = Vec::with_capacity(iters);
    let mut last_render_len = 0;
    for _ in 0..iters {
        let t = Instant::now();
        let s = frame_jinja.render().expect("J1 render");
        j1.push(t.elapsed().as_micros() as u64);
        last_render_len = s.len();
    }

    // ── Path J2: cached Env + template render only — the production cost.
    // Equivalent to what the daemon should do with a cached renderer.
    let tmpl_cached = env_cached.get_template("chat").expect("template lookup");
    let mut j2: Vec<u64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let ctx = context! {
            messages => Value::from_serialize(&messages_single),
            add_generation_prompt => true,
            enable_thinking => true,
            bos_token => bos_token.clone(),
            tools => Value::from_serialize(&empty_list),
            documents => Value::from_serialize(&empty_list),
            tool_call_kwargs => Value::from_serialize(&empty_map),
        };
        let _s = tmpl_cached.render(ctx).expect("J2 render");
        j2.push(t.elapsed().as_micros() as u64);
    }

    // ── Path P: hand-rolled ChatFrame::Plain — the legacy baseline.
    let frame_plain = ChatFrame {
        tokenizer: &tokenizer,
        system: Some(system),
        user,
        assistant_prefix: AssistantPrefix::OpenThink,
        raw: false,
    };
    let mut p: Vec<u64> = Vec::with_capacity(iters);
    let mut last_plain_len = 0;
    for _ in 0..iters {
        let t = Instant::now();
        let toks = frame_plain.build();
        p.push(t.elapsed().as_micros() as u64);
        last_plain_len = toks.len();
    }

    // ── Path T: tokenize the J1/J2 rendered string. Both J paths
    // need this on top, since they produce a String the daemon must
    // encode; Path P already returns tokens directly.
    let representative = frame_jinja.render().expect("render for tokenize");
    let mut t_us: Vec<u64> = Vec::with_capacity(iters);
    let mut last_tok_n = 0;
    for _ in 0..iters {
        let t = Instant::now();
        let toks = tokenizer.encode(&representative);
        t_us.push(t.elapsed().as_micros() as u64);
        last_tok_n = toks.len();
    }

    println!("Output sizes:");
    println!("  Jinja rendered string:   {} bytes  ({} tokens after encode)", last_render_len, last_tok_n);
    println!("  Plain build tokens:      {} tokens", last_plain_len);
    println!();
    println!("Per-call timings (microseconds):");
    report("J1 — full JinjaChatFrame::render (no cache)", &mut j1);
    report("J2 — cached Env + tmpl.render (production)", &mut j2);
    report("P  — ChatFrame::Plain build", &mut p);
    report("T  — tokenizer.encode (rendered string)", &mut t_us);
    println!();

    // Effective end-to-end "ready-for-forward" cost: bytes -> tokens.
    let mean_j2 = j2.iter().sum::<u64>() / (j2.len() as u64);
    let mean_t = t_us.iter().sum::<u64>() / (t_us.len() as u64);
    let mean_p = p.iter().sum::<u64>() / (p.len() as u64);
    let mean_j1 = j1.iter().sum::<u64>() / (j1.len() as u64);

    println!("Apples-to-apples (bytes -> ready tokens):");
    println!("  Jinja prod path:    {} us = {} (J2) + {} (T)", mean_j2 + mean_t, mean_j2, mean_t);
    println!("  Plain baseline:     {} us (P, returns tokens directly)", mean_p);
    println!("  Delta:              {} us per request", (mean_j2 + mean_t).saturating_sub(mean_p));
    println!();
    println!("If JinjaChatFrame::render were called as-is (J1, no caching):");
    println!("  Jinja no-cache path: {} us = {} (J1) + {} (T)", mean_j1 + mean_t, mean_j1, mean_t);
    println!("  Delta vs Plain:      {} us per request", (mean_j1 + mean_t).saturating_sub(mean_p));
}
