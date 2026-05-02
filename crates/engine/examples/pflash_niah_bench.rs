//! NIAH harness with optional PFlash compression (PRD §6 Phase 5 gate).
//!
//! Loads a Qwen3.5-family target model (4B/9B/27B; dense or hybrid),
//! ingests a NIAH fixture from `benchmarks/longctx/niah/niah_<N>k.jsonl`,
//! optionally runs PFlash compression via a matched-tokenizer drafter
//! (e.g. qwen3.5-0.8b → qwen3.5-27b), then prefills + decodes through
//! the target. Reports TTFT broken into tokenize / compress / prefill /
//! first decode / total, plus source/kept token counts when PFlash ran.
//! Records source prompt md5, binary md5, model md5.
//!
//! PASS = the expected substring appears in the decoded answer (so a
//! PFlash-on PASS proves the needle survived the compression).
//!
//! Usage:
//!   cargo run --release --features deltanet --example pflash_niah_bench -- \
//!     <model.hfq> <fixture.jsonl> [--maxgen N] [--q8kv|--asym3] \
//!     [--pflash <drafter.hfq> [--keep-ratio K] [--block-size B] \
//!      [--sink-tokens N] [--recent-tokens N]]
//!
//! Defaults: --maxgen 64, --asym3 (best for long-ctx K), no PFlash.
//! When --pflash is given: keep-ratio 0.30, block-size 64, sink 16, recent 32.

use engine::hfq::HfqFile;
use engine::llama::{self, KvCache};
use engine::pflash::{self, BypassReason, PflashConfig, PflashDecision, PflashMode, PflashState, RequestKind};
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

fn extract_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let i = text.find(&needle)?;
    let rest = &text[i + needle.len()..];
    // Skip whitespace then expect '"'
    let start = rest.find('"')? + 1;
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
    Some(out)
}

/// Parse a JSON array of strings: `["foo","bar"]`. Stops at the matching `]`.
/// Used for `expected_answer_substrings` in multi-needle fixtures. Tolerates
/// `\"` and `\\` escapes consistent with `extract_string_field`.
fn extract_string_array(text: &str, key: &str) -> Option<Vec<String>> {
    let needle = format!("\"{key}\":");
    let i = text.find(&needle)?;
    let rest = &text[i + needle.len()..];
    let lb = rest.find('[')?;
    let bytes = rest.as_bytes();
    let mut j = lb + 1;
    let mut out = Vec::new();
    while j < bytes.len() {
        // Skip whitespace + commas
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b',' || bytes[j] == b'\n' || bytes[j] == b'\t') {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] == b']' { break; }
        if bytes[j] != b'"' { return None; }
        j += 1;
        let mut s = String::new();
        while j < bytes.len() {
            let b = bytes[j];
            if b == b'\\' && j + 1 < bytes.len() {
                let esc = bytes[j + 1];
                match esc {
                    b'n' => s.push('\n'),
                    b't' => s.push('\t'),
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    _ => s.push(esc as char),
                }
                j += 2;
            } else if b == b'"' {
                j += 1;
                break;
            } else {
                s.push(b as char);
                j += 1;
            }
        }
        out.push(s);
    }
    Some(out)
}

fn extract_usize_field(text: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\":");
    let i = text.find(&needle)?;
    let rest = &text[i + needle.len()..];
    rest.trim_start()
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|s| s.parse().ok())
}

/// Parse a fixture record into `(filler, question, expected_substrings, min_recovered)`.
/// Backward compatible: a single-needle record yields a 1-element vec and
/// `min_recovered=1`. Multi-needle records carry `expected_answer_substrings`
/// (plural) and `min_recovered` directly.
fn parse_jsonl_record(text: &str) -> (String, String, Vec<String>, usize) {
    let filler = extract_string_field(text, "filler_text")
        .expect("missing filler_text");
    let question = extract_string_field(text, "question")
        .expect("missing question");
    if let Some(arr) = extract_string_array(text, "expected_answer_substrings") {
        let min_recovered = extract_usize_field(text, "min_recovered").unwrap_or(arr.len());
        (filler, question, arr, min_recovered)
    } else {
        let single = extract_string_field(text, "expected_answer_substring")
            .expect("missing expected_answer_substring (and no plural array)");
        (filler, question, vec![single], 1)
    }
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

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// Companion-file path for a pre-tokenized fixture: replaces the trailing
/// `.jsonl` with `.tok.jsonl`. The pretok file mirrors the source fields
/// plus a tokens array and a tokenizer-signature marker. Stored next to
/// the source fixture so re-runs without `--write-pretok` find it.
fn pretok_companion_path(fixture: &Path) -> std::path::PathBuf {
    let stem = fixture.with_extension("");
    let mut s = stem.into_os_string();
    s.push(".tok.jsonl");
    std::path::PathBuf::from(s)
}

/// JSON-encode a u32 array as `[1,2,3,...]` without pulling in a full
/// JSON library. Pre-tokenized fixtures only need a numeric array.
fn encode_token_array(tokens: &[u32]) -> String {
    let mut s = String::with_capacity(tokens.len() * 6 + 2);
    s.push('[');
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&t.to_string());
    }
    s.push(']');
    s
}

/// Read a u32 array out of a pretok JSONL line. Strict but minimal: the
/// writer is `encode_token_array`, so we only need to parse `[N,N,...,N]`.
fn parse_token_array(text: &str) -> Vec<u32> {
    let needle = "\"tokens\":";
    let i = text.find(needle).expect("missing tokens field in pretok jsonl");
    let rest = &text[i + needle.len()..];
    let lb = rest.find('[').expect("expected [ for tokens array");
    let rb = rest.find(']').expect("expected ] for tokens array");
    rest[lb + 1..rb]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().expect("token id parse"))
        .collect()
}

fn parse_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let i = text.find(&needle)?;
    let rest = &text[i + needle.len()..];
    let start = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut j = start;
    let mut out = String::new();
    while j < bytes.len() {
        let b = bytes[j];
        if b == b'"' { break; }
        out.push(b as char);
        j += 1;
    }
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: pflash_niah_bench <model.hfq> <fixture.jsonl> \
                   [--maxgen N] [--q8kv|--asym3] [--pretok|--write-pretok] \
                   [--pflash <drafter.hfq> [--keep-ratio K] [--block-size B] \
                   [--sink-tokens N] [--recent-tokens N]]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let fixture_path = &args[2];
    let max_gen: usize = parse_arg(&args, "--maxgen").unwrap_or(64);
    let use_q8 = args.iter().any(|a| a == "--q8kv");
    let kv_label = if use_q8 { "q8" } else { "asym3" };
    let drafter_path: Option<String> = args.iter().position(|a| a == "--pflash")
        .and_then(|i| args.get(i + 1)).cloned();
    let keep_ratio: f32 = parse_arg(&args, "--keep-ratio").unwrap_or(0.30);
    let block_size: usize = parse_arg(&args, "--block-size").unwrap_or(64);
    let sink_tokens: usize = parse_arg(&args, "--sink-tokens").unwrap_or(16);
    let recent_tokens: usize = parse_arg(&args, "--recent-tokens").unwrap_or(32);
    let use_pretok = args.iter().any(|a| a == "--pretok");
    let write_pretok = args.iter().any(|a| a == "--write-pretok");
    if use_pretok && write_pretok {
        eprintln!("FAIL: --pretok and --write-pretok are mutually exclusive \
                   (one reads cached tokens, the other generates them; combining \
                   them would re-author the cache against itself)");
        std::process::exit(2);
    }

    let mode_label = if drafter_path.is_some() { "PFlash compressed" } else { "full prefill" };
    eprintln!("=== PFlash NIAH ({mode_label}) ===");
    eprintln!("model:   {model_path}");
    eprintln!("fixture: {fixture_path}");
    eprintln!("maxgen:  {max_gen}");
    eprintln!("kv mode: {kv_label}");
    if let Some(d) = &drafter_path {
        eprintln!("drafter: {d}");
        eprintln!("pflash:  keep_ratio={keep_ratio} block={block_size} sink={sink_tokens} recent={recent_tokens}");
    }

    // Binary md5 — required by PRD §6 / §5.3.3 report fields. Reads the
    // running executable from /proc/self/exe so reruns of the same binary
    // produce stable hashes regardless of cwd.
    let bin_md5 = md5_file(Path::new("/proc/self/exe"));
    eprintln!("binary md5:  {bin_md5}");

    let pretok_path = pretok_companion_path(Path::new(fixture_path));
    let pretok_available = use_pretok && pretok_path.exists();
    if use_pretok && !pretok_available {
        eprintln!("FAIL: --pretok requested but {} does not exist (run --write-pretok first)",
            pretok_path.display());
        std::process::exit(2);
    }

    // Compute the SOURCE fixture md5 even in pretok mode so we can verify
    // the pretok artifact wasn't authored against a different (now-edited)
    // source. Without this, regenerating niah_8k.jsonl with a new needle
    // would silently keep using the old niah_8k.tok.jsonl tokens because
    // the tokenizer signature still matches.
    let source_raw = fs::read_to_string(fixture_path).expect("read source fixture");
    let source_raw_md5 = md5_hex(source_raw.as_bytes());
    eprintln!("source fixture md5: {source_raw_md5}");

    let (raw, raw_md5, filler, question, expected_substrings, min_recovered, pretok_tokens, pretok_sig) =
    if pretok_available {
        let raw = fs::read_to_string(&pretok_path).expect("read pretok fixture");
        let raw_md5 = md5_hex(raw.as_bytes());
        eprintln!("pretok fixture: {} ({raw_md5})", pretok_path.display());
        // Stale-pretok guard: the pretok records the source md5 it was
        // authored against; if the source has changed since, fail loudly
        // rather than encode-with-old / verdict-against-new.
        let recorded_source_md5 = parse_string_field(&raw, "source_fixture_md5")
            .expect("missing source_fixture_md5 in pretok jsonl (was it written by an old --write-pretok?)");
        if recorded_source_md5 != source_raw_md5 {
            eprintln!("FAIL: pretok source_fixture_md5 {recorded_source_md5} != \
                       current source fixture md5 {source_raw_md5}; \
                       re-run --write-pretok against the current source");
            std::process::exit(2);
        }
        let question = parse_string_field(&raw, "question").expect("question");
        // Plural form preferred (multi-needle); fall back to singular.
        let (expected_arr, min_rec) = if let Some(arr) = extract_string_array(&raw, "expected_answer_substrings") {
            let mr = extract_usize_field(&raw, "min_recovered").unwrap_or(arr.len());
            (arr, mr)
        } else {
            let single = parse_string_field(&raw, "expected_answer_substring").expect("expected");
            (vec![single], 1usize)
        };
        let sig = parse_string_field(&raw, "tokenizer_signature").expect("tokenizer_signature");
        let toks = parse_token_array(&raw);
        eprintln!("pretok tokens: {} (sig {sig}, source md5 verified)", toks.len());
        (raw, raw_md5, String::new(), question, expected_arr, min_rec, Some(toks), Some(sig))
    } else {
        eprintln!("fixture md5: {source_raw_md5}");
        let (filler, question, expected, min_rec) = parse_jsonl_record(&source_raw);
        (source_raw.clone(), source_raw_md5.clone(), filler, question, expected, min_rec, None, None)
    };
    let prompt_text = if pretok_tokens.is_some() {
        // pretok mode never reconstructs a prompt string -- the tokens
        // _are_ the prompt. Hash the question alone for traceability.
        question.clone()
    } else {
        format!("{filler}\n\n{question}")
    };
    let prompt_md5 = md5_hex(prompt_text.as_bytes());
    eprintln!("prompt md5:  {prompt_md5}");
    eprintln!("expected ({}/{} req): {expected_substrings:?}", min_recovered, expected_substrings.len());
    let _ = raw; // keep raw alive for any later debug; main path doesn't need it.

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
    let source_tokens: Vec<u32> = if let Some(pretok) = pretok_tokens {
        // Use the §5.3 compat signature (excludes audio/TTS padding) so a
        // pretok authored with qwen3.5-0.8b's tokenizer is accepted by
        // qwen3.5-27b's tokenizer. Strict signature() differs across family
        // sizes only in those padding slots, which never appear in encoded
        // text, so two §5.3-compatible tokenizers produce identical
        // encodings for the source -- the tokens are safe to consume.
        let actual_sig = pflash::tokenizer_compat_signature(&tokenizer).to_string();
        let recorded = pretok_sig.unwrap_or_default();
        if actual_sig != recorded {
            eprintln!("FAIL: pretok tokenizer compat signature {recorded} != model tokenizer compat signature {actual_sig}; \
                       re-run --write-pretok with a §5.3-compatible tokenizer");
            std::process::exit(2);
        }
        eprintln!("tokenize:    skipped (pretok mode, compat signature matches)");
        pretok
    } else {
        wrap_chatml(&tokenizer, &prompt_text)
    };
    let tok_ms = if pretok_available { 0 } else { t_tok.elapsed().as_millis() };
    if !pretok_available {
        eprintln!("tokenize:    {tok_ms} ms ({} tokens)", source_tokens.len());
    }
    let source_bytes: Vec<u8> = source_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let source_md5 = md5_hex(&source_bytes);
    eprintln!("source tokens md5: {source_md5}");

    if write_pretok {
        // Persist a pre-tokenized companion right after encode so future
        // bench runs can use --pretok and skip the slow O(N²) encoder.
        // Single-needle fixtures keep the singular expected_answer_substring
        // for backward compat; multi-needle fixtures persist the plural
        // array + min_recovered.
        // Record the §5.3 compat signature (not strict signature()) so the
        // pretok travels safely across same-family members of different
        // sizes (e.g. authored with 0.8B's tokenizer, consumed by 27B's
        // tokenizer); see pflash::tokenizer_compat_signature for rationale.
        let sig = pflash::tokenizer_compat_signature(&tokenizer).to_string();
        let mut line = String::with_capacity(source_tokens.len() * 6 + 256);
        line.push('{');
        line.push_str(&format!("\"source_fixture\":\"{}\",", fixture_path.replace('"', "")));
        line.push_str(&format!("\"source_fixture_md5\":\"{raw_md5}\","));
        line.push_str(&format!("\"tokenizer_signature\":\"{sig}\","));
        line.push_str(&format!("\"question\":\"{}\",", question.replace('"', "\\\"")));
        if expected_substrings.len() == 1 && min_recovered == 1 {
            line.push_str(&format!("\"expected_answer_substring\":\"{}\",",
                expected_substrings[0].replace('"', "\\\"")));
        } else {
            line.push_str("\"expected_answer_substrings\":[");
            for (i, s) in expected_substrings.iter().enumerate() {
                if i > 0 { line.push(','); }
                line.push_str(&format!("\"{}\"", s.replace('"', "\\\"")));
            }
            line.push_str("],");
            line.push_str(&format!("\"min_recovered\":{},", min_recovered));
        }
        line.push_str(&format!("\"tokens_count\":{},", source_tokens.len()));
        line.push_str(&format!("\"tokens_md5\":\"{source_md5}\","));
        line.push_str("\"tokens\":");
        line.push_str(&encode_token_array(&source_tokens));
        line.push_str("}\n");
        fs::write(&pretok_path, line).expect("write pretok jsonl");
        eprintln!("wrote pretok: {} ({} tokens)", pretok_path.display(), source_tokens.len());
    }

    // ── PFlash compression (optional) ────────────────────────────────────
    // Drafter is loaded transiently after target weights, before target KV
    // alloc, then unloaded so its VRAM goes back to the pool for the
    // target's KV cache. This matches pflash_compress_demo's load order.
    let mut compress_ms: u128 = 0;
    let mut score_ms: u128 = 0;
    let mut select_ms: u128 = 0;
    let mut gather_ms: u128 = 0;
    let tokens: Vec<u32> = if let Some(drafter_path_str) = &drafter_path {
        let pflash_cfg = PflashConfig {
            mode: PflashMode::Always,
            keep_ratio,
            block_size,
            sink_tokens,
            recent_tokens,
            min_keep_tokens: 0,
            drafter_path: Some(drafter_path_str.clone()),
            ..Default::default()
        };
        let mut state = PflashState::new(&pflash_cfg);
        let drafter_max_kv = source_tokens.len() + 64;
        let t_load_drafter = Instant::now();
        pflash::load_drafter(
            &mut state, &mut gpu, Path::new(drafter_path_str), &tokenizer, drafter_max_kv,
        ).expect("load_drafter");
        eprintln!("drafter loaded: {:.1}s | tokenizer_compat={}",
            t_load_drafter.elapsed().as_secs_f64(), state.tokenizer_compat);
        if !state.tokenizer_compat {
            eprintln!("FAIL: drafter tokenizer incompatible with target -- cannot compress safely");
            state.unload_drafter(&mut gpu);
            std::process::exit(2);
        }

        let t_compress = Instant::now();
        let decision = pflash::maybe_compress_prompt(
            &mut gpu, &mut state, &pflash_cfg, &source_tokens, RequestKind::Text, &[],
        ).expect("maybe_compress_prompt");
        compress_ms = t_compress.elapsed().as_millis();

        let kept = match decision {
            PflashDecision::Compressed(cp) => {
                score_ms = cp.timings.score_ms as u128;
                select_ms = cp.timings.select_ms as u128;
                gather_ms = cp.timings.gather_ms as u128;
                eprintln!("compress:    {compress_ms} ms (score={score_ms}ms select={select_ms}ms gather={gather_ms}ms)");
                eprintln!("compressed:  {} -> {} tokens (ratio {:.3}, alpha implicit)",
                    cp.source_tokens, cp.kept_tokens,
                    cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32);
                eprintln!("source_md5:    {}", cp.source_md5);
                eprintln!("compressed_md5:{}", cp.compressed_md5);
                eprintln!("kept_spans:  {} ranges (first={:?} last={:?})",
                    cp.kept_spans.len(), cp.kept_spans.first(), cp.kept_spans.last());
                cp.token_ids
            }
            PflashDecision::Bypass { reason: BypassReason::BelowThreshold { source_tokens: st, threshold } } => {
                eprintln!("bypass(BelowThreshold): {st} tokens, threshold {threshold}");
                eprintln!("(compression would keep entire prompt -- running full prefill)");
                source_tokens.clone()
            }
            PflashDecision::Bypass { reason } => {
                eprintln!("FAIL: unexpected pflash bypass: {reason:?}");
                state.unload_drafter(&mut gpu);
                std::process::exit(2);
            }
        };

        // Free drafter VRAM before allocating target KV.
        state.unload_drafter(&mut gpu);
        kept
    } else {
        source_tokens.clone()
    };

    let tokens_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let tokens_md5 = md5_hex(&tokens_bytes);
    eprintln!("prefill tokens md5: {tokens_md5} ({} tokens)", tokens.len());

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

    // HIP kernel launches are async. Without an explicit synchronize,
    // `t_pre.elapsed()` would only measure host-side launch time and the
    // first D2H download in the "first dec" bucket would absorb the real
    // prefill compute. Mirror the bench_qwen35_mq4 pattern: sync inside
    // the prefill timer, then time download+argmax separately as the
    // first-decode-step bucket.
    let t_pre = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &tokens, 0, &mut kv, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("forward_prefill_batch");
    gpu.hip.device_synchronize().expect("sync after prefill");
    let prefill_ms = t_pre.elapsed().as_millis();
    let prefill_tok_s = if prefill_ms > 0 { tokens.len() as f64 / (prefill_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("prefill:     {prefill_ms} ms ({prefill_tok_s:.0} tok/s)");

    // First decoded token comes directly from prefill logits. With the
    // sync above, this bucket truly measures only the D2H download +
    // host-side argmax, not pending prefill kernels.
    let t_first_dec = Instant::now();
    let logits = gpu.download_f32(&scratch.logits).expect("download logits");
    let first_token = llama::argmax(&logits);
    let first_decode_ms = t_first_dec.elapsed().as_millis();
    eprintln!("first dec:   {first_decode_ms} ms (download + argmax of prefill logits)");

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

    let ttft_ms = tok_ms + compress_ms + prefill_ms + first_decode_ms;
    let total_ms = ttft_ms + decode_ms;
    eprintln!("--- TTFT ---");
    eprintln!("tokenize:    {tok_ms} ms");
    if drafter_path.is_some() {
        eprintln!("compress:    {compress_ms} ms (score={score_ms}ms select={select_ms}ms gather={gather_ms}ms)");
    }
    eprintln!("prefill:     {prefill_ms} ms");
    eprintln!("first dec:   {first_decode_ms} ms");
    eprintln!("ttft:        {ttft_ms} ms");
    eprintln!("decode rest: {decode_ms} ms");
    eprintln!("total:       {total_ms} ms");

    let recovered: Vec<&String> = expected_substrings.iter()
        .filter(|s| answer.contains(s.as_str()))
        .collect();
    let pass = recovered.len() >= min_recovered;
    eprintln!("--- ANSWER ---");
    eprintln!("{answer}");
    eprintln!("--- VERDICT ---");
    eprintln!("recovered: {} / {} (min_recovered={})", recovered.len(), expected_substrings.len(), min_recovered);
    for s in &expected_substrings {
        let mark = if answer.contains(s.as_str()) { "+" } else { "-" };
        eprintln!("  [{mark}] {s:?}");
    }
    if pass {
        eprintln!("PASS: {} substring(s) found, min_recovered={}", recovered.len(), min_recovered);
        std::process::exit(0);
    } else {
        eprintln!("FAIL: {} of {} substrings recovered, need {}", recovered.len(), expected_substrings.len(), min_recovered);
        std::process::exit(1);
    }
}
