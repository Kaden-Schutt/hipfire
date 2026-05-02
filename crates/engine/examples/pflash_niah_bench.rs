//! Full-prefill NIAH baseline harness (PFlash Phase 0).
//!
//! Loads a Qwen3.5-family target model (4B/9B/27B; dense or hybrid),
//! ingests a NIAH fixture from `benchmarks/longctx/niah/niah_<N>k.jsonl`,
//! runs `qwen35::forward_prefill_batch` to populate the KV cache, then
//! decodes up to `max_gen` tokens via `qwen35::forward_scratch`. Reports
//! TTFT broken into tokenize / prefill / first decode step / total.
//! Records source prompt md5, binary md5, model md5, and token counts.
//!
//! PASS = the expected substring appears in the decoded answer.
//!
//! Usage:
//!   cargo run --release --features deltanet --example pflash_niah_bench -- \
//!     <model.hfq> <fixture.jsonl> [--maxgen 64] [--q8kv|--asym3]
//!
//! Defaults: --maxgen 64, --asym3 (best for long-ctx K).

use engine::hfq::HfqFile;
use engine::llama::{self, KvCache};
use engine::qwen35::{self, DeltaNetState};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn md5_hex(bytes: &[u8]) -> String {
    use std::process::Command;
    let mut child = Command::new("md5sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn md5sum");
    let mut stdin = child.stdin.take().unwrap();
    use std::io::Write;
    stdin.write_all(bytes).expect("write stdin");
    drop(stdin);
    let out = child.wait_with_output().expect("md5sum wait");
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().next().unwrap_or("").to_string()
}

fn md5_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_default();
    md5_hex(&bytes)
}

fn parse_jsonl_record(text: &str) -> (String, String, String) {
    fn extract(text: &str, key: &str) -> String {
        let needle = format!("\"{key}\":");
        let i = text.find(&needle).unwrap_or_else(|| panic!("missing {key}"));
        let rest = &text[i + needle.len()..];
        let start = rest.find('"').unwrap_or_else(|| panic!("expected string for {key}")) + 1;
        let mut out = String::new();
        let bytes = rest.as_bytes();
        let mut j = start;
        while j < bytes.len() {
            let b = bytes[j];
            if b == b'\\' && j + 1 < bytes.len() {
                let esc = bytes[j + 1];
                match esc {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    _ => out.push(esc as char),
                }
                j += 2;
            } else if b == b'"' {
                break;
            } else {
                out.push(b as char);
                j += 1;
            }
        }
        out
    }
    let filler = extract(text, "filler_text");
    let question = extract(text, "question");
    let expected = extract(text, "expected_answer_substring");
    (filler, question, expected)
}

fn wrap_chatml(tokenizer: &engine::tokenizer::Tokenizer, prompt: &str) -> Vec<u32> {
    let body = tokenizer.encode(prompt);
    let im_start = tokenizer.encode("<|im_start|>");
    if im_start.len() != 1 {
        return body;
    }
    let im_end = tokenizer.encode("<|im_end|>");
    let user = tokenizer.encode("user");
    let asst = tokenizer.encode("assistant");
    let nl = tokenizer.encode("\n");
    let think_end = tokenizer.encode("</think>");
    let mut out = Vec::with_capacity(body.len() + 32);
    out.extend_from_slice(&im_start);
    out.extend_from_slice(&user);
    out.extend_from_slice(&nl);
    out.extend_from_slice(&body);
    out.extend_from_slice(&im_end);
    out.extend_from_slice(&nl);
    out.extend_from_slice(&im_start);
    out.extend_from_slice(&asst);
    out.extend_from_slice(&nl);
    // Force think-off: skip <think>, jump straight to </think>\n
    out.extend_from_slice(&think_end);
    out.extend_from_slice(&nl);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: pflash_niah_bench <model.hfq> <fixture.jsonl> [--maxgen N] [--q8kv|--asym3]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let fixture_path = &args[2];
    let max_gen: usize = args.iter().position(|a| a == "--maxgen")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let use_q8 = args.iter().any(|a| a == "--q8kv");
    let kv_label = if use_q8 { "q8" } else { "asym3" };

    eprintln!("=== PFlash NIAH baseline (full prefill) ===");
    eprintln!("model:   {model_path}");
    eprintln!("fixture: {fixture_path}");
    eprintln!("maxgen:  {max_gen}");
    eprintln!("kv mode: {kv_label}");

    // Binary md5 — required by PRD §6 / §5.3.3 report fields. Reads the
    // running executable from /proc/self/exe so reruns of the same binary
    // produce stable hashes regardless of cwd.
    let bin_md5 = md5_file(Path::new("/proc/self/exe"));
    eprintln!("binary md5:  {bin_md5}");

    let raw = fs::read_to_string(fixture_path).expect("read fixture");
    let raw_md5 = md5_hex(raw.as_bytes());
    eprintln!("fixture md5: {raw_md5}");
    let (filler, question, expected) = parse_jsonl_record(&raw);
    let prompt_text = format!("{filler}\n\n{question}");
    let prompt_md5 = md5_hex(prompt_text.as_bytes());
    eprintln!("prompt md5:  {prompt_md5}");
    eprintln!("expected:    {expected:?}");

    let model_md5 = md5_file(Path::new(model_path));
    eprintln!("model md5:   {model_md5}");

    let t_load_start = Instant::now();
    let hfq = HfqFile::open(Path::new(model_path)).expect("open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("qwen35 config");
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("loaded in {:.1}s | dim={} layers={} heads={} kv_heads={}",
        t_load_start.elapsed().as_secs_f64(),
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);

    let t_tok = Instant::now();
    let tokens = wrap_chatml(&tokenizer, &prompt_text);
    let tok_ms = t_tok.elapsed().as_millis();
    eprintln!("tokenize:    {tok_ms} ms ({} tokens)", tokens.len());
    let tokens_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let tokens_md5 = md5_hex(&tokens_bytes);
    eprintln!("tokens md5:  {tokens_md5}");

    let kv_seq = (tokens.len() + max_gen + 256).next_power_of_two().max(2048);
    let mut kv = if use_q8 {
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
            .expect("kv q8")
    } else {
        KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq)
            .expect("kv asym3")
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn_state");
    let scratch = qwen35::Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).expect("scratch");

    let t_pre = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &tokens, 0, &mut kv, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("forward_prefill_batch");
    let prefill_ms = t_pre.elapsed().as_millis();
    let prefill_tok_s = if prefill_ms > 0 { tokens.len() as f64 / (prefill_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("prefill:     {prefill_ms} ms ({prefill_tok_s:.0} tok/s)");

    // First decoded token comes directly from prefill logits — no
    // additional GPU work, but isolate the argmax + download as part of
    // "first decode step" timing so the tokenize/prefill/first-decode/total
    // breakdown matches the PRD §6 contract.
    let t_first_dec = Instant::now();
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let first_token = llama::argmax(&logits);
    let first_decode_ms = t_first_dec.elapsed().as_millis();
    eprintln!("first dec:   {first_decode_ms} ms (token from prefill logits)");

    // Sustained decode loop. `decode_steps` counts ONLY actual
    // forward_scratch calls; the first token (already accounted for above)
    // is not in the denominator. This avoids inflating decode tok/s by
    // counting the prefill-derived token as decode work.
    let t_dec = Instant::now();
    let mut next_token = first_token;
    let mut generated: Vec<u32> = vec![first_token];
    let mut decode_steps: usize = 0;
    for _ in 1..max_gen {
        if next_token == config.eos_token {
            break;
        }
        let pos = tokens.len() + generated.len() - 1;
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos, &mut kv, &mut dn_state, &scratch,
        ).expect("forward_scratch");
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        next_token = llama::argmax(&logits);
        generated.push(next_token);
        decode_steps += 1;
    }
    let decode_ms = t_dec.elapsed().as_millis();
    let decode_tok_s = if decode_ms > 0 && decode_steps > 0 {
        decode_steps as f64 / (decode_ms as f64 / 1000.0)
    } else { 0.0 };
    let answer = tokenizer.decode(&generated);
    eprintln!("decode:      {decode_ms} ms ({decode_steps} forward_scratch calls, {decode_tok_s:.1} tok/s)");

    let ttft_ms = tok_ms + prefill_ms + first_decode_ms;
    let total_ms = ttft_ms + decode_ms;
    eprintln!("--- TTFT ---");
    eprintln!("tokenize:    {tok_ms} ms");
    eprintln!("prefill:     {prefill_ms} ms");
    eprintln!("first dec:   {first_decode_ms} ms");
    eprintln!("ttft:        {ttft_ms} ms");
    eprintln!("decode rest: {decode_ms} ms");
    eprintln!("total:       {total_ms} ms");

    let pass = answer.contains(&expected);
    eprintln!("--- ANSWER ---");
    eprintln!("{answer}");
    eprintln!("--- VERDICT ---");
    if pass {
        eprintln!("PASS: expected substring found in answer");
        std::process::exit(0);
    } else {
        eprintln!("FAIL: expected {expected:?} not in answer");
        std::process::exit(1);
    }
}
