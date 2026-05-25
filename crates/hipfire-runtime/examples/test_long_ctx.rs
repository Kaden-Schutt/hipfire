// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qualitative multi-turn long-context test for MQ4 models.
//!
//! Loads a model once, runs N turns against a persistent KV cache
//! (daemon-style: prefill only the new tokens each turn via
//! `forward_prefill_batch`). Streams decoded text as it generates and
//! reports per-turn prefill/decode token throughput.
//!
//! Usage:
//!   test_long_ctx <model.mq4> [--turns-file <path>] [--max-gen N]
//!
//! If --turns-file is omitted, a built-in 3-turn fixture runs that
//! exercises a long single-turn prompt followed by two short follow-ups.
//! Turns file format: one turn per line; blank lines are skipped.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache, SamplingConfig};
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test_long_ctx <model.mq4> [--turns-file <path>] [--max-gen N] [--no-think]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let mut turns_file: Option<String> = None;
    let mut max_gen: usize = 512;
    let mut no_think = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--turns-file" => { turns_file = Some(args[i + 1].clone()); i += 2; }
            "--max-gen"    => { max_gen = args[i + 1].parse().unwrap_or(512); i += 2; }
            "--no-think"   => { no_think = true; i += 1; }
            _ => { i += 1; }
        }
    }

    // ── Turn content ────────────────────────────────────────────────────
    // Built-in fixture: turn 1 is a ~900-token fictional scenario that
    // forces retrieval from context (no memorization shortcut). Turns 2
    // and 3 are short follow-ups that reference things from turn 1, so
    // coherence across the multi-turn KV cache shows up immediately.
    let turns: Vec<String> = if let Some(path) = turns_file {
        std::fs::read_to_string(&path).expect("read turns file")
            .lines().filter(|l| !l.trim().is_empty()).map(|s| s.to_string()).collect()
    } else {
        vec![
            // Turn 1: long scenario + specific question.
            "I'm going to describe a small fictional village, then ask you a question about it. \
Read carefully before answering.\n\n\
The village of Sallisaw sits at the fork of two rivers: the Green River flowing from the \
northeast and the Whitewater flowing from the southwest. The two rivers merge into the Broad \
River which heads due south to the sea, two days' travel away by barge. Sallisaw has about \
four hundred residents. The main crops are winter barley and summer beans, rotated year to \
year on the terraces above the eastern bank of the Green.\n\n\
The village council has five members: Elder Margery, who owns the mill at the forks; Kenton \
the blacksmith, whose forge anchors the north road; Reesha, a retired barge pilot who still \
knows every shoal from here to the coast; Annoth, the priest of the small temple to the \
water spirits; and Yarrow, a farmer who speaks for the eastern terraces. Elder Margery has \
been on the council for twenty-two years; everyone else joined in the last six.\n\n\
Three months ago a fire broke out in the thatch market on the east bank. It destroyed \
eleven stalls and the grain-weigher's office before villagers formed a bucket line from the \
Green and stopped it. The cause was never determined — Kenton suspected a spark from one \
of the smokehouses, but Annoth argued it was a slight against the river spirits because the \
council had quietly redirected a seasonal creek away from the temple's garden to irrigate \
new bean plots. No one was hurt but the grain records for that season were lost.\n\n\
Since the fire, the council has been arguing about whether to rebuild the market on the \
same spot (which is flat and near the docks but burned once) or to move it across the forks \
to the western bank (safer from runoff fires but farther from the barge landing). Reesha \
and Yarrow favor the eastern rebuild; Kenton and Annoth favor the western site; Elder \
Margery refuses to break the tie until she hears from the barge guild in Portsmouth, whose \
reply is two weeks late.\n\n\
Question: based on this description, which council member would you expect to be the \
strongest advocate for restoring the redirected creek back to the temple's garden, and why? \
Give your reasoning in one short paragraph.".to_string(),

            // Turn 2: specific follow-up that requires remembering turn 1.
            "Good. Now: which council member seems least connected to the fire debate, and \
why might that be a political weakness for them?".to_string(),

            // Turn 3: a reasoning follow-up that requires remembering both prior turns.
            "One more: if Elder Margery is still waiting on the barge guild reply when another \
fire breaks out, whose position on the market rebuild gets strongest, and what changes about \
it?".to_string(),
        ]
    };

    eprintln!("=== test_long_ctx: {} ===", model_path);
    eprintln!("turns: {}", turns.len());

    // ── Load model ──────────────────────────────────────────────────────
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    // Size the KV cache for the sum of all turn prompts + generations, with slack.
    let max_seq = 8192usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq,
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Use Qwen3.5-recommended sampling: temp=0.7, top_p=0.8. The 0.3
    // "thinking" temp is known to fall into repetition traps on long
    // creative generations.
    let sc = SamplingConfig {
        think_temp: 0.7,
        answer_temp: 0.7,
        top_p: 0.8,
        repeat_penalty: 1.15,
        repeat_window: 128,
    };
    let temp = sc.think_temp;
    let top_p = sc.top_p;

    // Special tokens
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let think_open = tokenizer.encode("<think>");
    let think_end_seq = tokenizer.encode("</think>");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    // Hard cap on thinking tokens — matches infer.rs. Without this, a
    // thinking model happily spirals inside <think>...</think> without ever
    // emitting the close tag, especially at low sampling temperature.
    const MAX_THINK_TOKENS: usize = 512;

    let mut seq_pos = 0usize;
    let mut history: Vec<u32> = Vec::new();
    let mut rng_state: u64 = 0xDEADBEEF_CAFEBABE;

    println!("\n================================================================");
    println!("MODEL: {}", model_path);
    println!("MAX_SEQ: {}  MAX_GEN/TURN: {}  SAMPLING: temp={} top_p={}",
             max_seq, max_gen, temp, top_p);
    println!("================================================================\n");

    for (turn_idx, turn_text) in turns.iter().enumerate() {
        println!("────────────────────────────────────────────────────────────────");
        println!(">>> TURN {} USER:", turn_idx + 1);
        println!("{}", turn_text);
        println!("────────────────────────────────────────────────────────────────");

        // Build new tokens for this turn: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n
        let body = tokenizer.encode(turn_text);
        let mut new_tokens: Vec<u32> = Vec::new();
        if turn_idx == 0 {
            // First turn: no system prompt needed for this test
        }
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&user_tok);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&body);
        new_tokens.extend_from_slice(&im_end);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&asst_tok);
        new_tokens.extend_from_slice(&nl);
        if !no_think {
            new_tokens.extend_from_slice(&think_open);
            new_tokens.extend_from_slice(&nl);
        }

        // ── Prefill ──
        let t_pf = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &new_tokens, seq_pos,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("prefill");
        let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
        let prefill_tok_s = new_tokens.len() as f64 / (prefill_ms / 1000.0);
        seq_pos += new_tokens.len();
        history.extend_from_slice(&new_tokens);

        eprintln!("\n[turn {} prefill: {} tokens in {:.1} ms = {:.0} tok/s] (ctx={})",
                 turn_idx + 1, new_tokens.len(), prefill_ms, prefill_tok_s, seq_pos);

        // ── Decode ──
        println!("\n<<< TURN {} ASSISTANT:", turn_idx + 1);
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next_token = llama::sample_top_p(&logits, temp, top_p);

        let t_gen = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted_bytes = 0usize;
        let mut in_thinking = !no_think;
        let mut think_count = 0usize;

        for _ in 0..max_gen {
            generated.push(next_token);
            history.push(next_token);
            if in_thinking { think_count += 1; }

            // Detect </think> as a TOKEN SEQUENCE (can be multi-token).
            // Same pattern as infer.rs.
            let think_ended = in_thinking && (
                think_count >= MAX_THINK_TOKENS ||
                (generated.len() >= think_end_seq.len()
                 && generated[generated.len() - think_end_seq.len()..] == think_end_seq[..])
            );
            if think_ended {
                in_thinking = false;
            }

            // Stream decoded text, only emitting complete UTF-8.
            let all_bytes = tokenizer.decode_bytes(&generated);
            let new_bytes = &all_bytes[emitted_bytes..];
            let vl = match std::str::from_utf8(new_bytes) {
                Ok(_) => new_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            if vl > 0 {
                let text = std::str::from_utf8(&new_bytes[..vl]).unwrap();
                print!("{}", text);
                let _ = std::io::stdout().flush();
                emitted_bytes += vl;
            }

            // Write this token's K/V to the cache FIRST, then check for
            // termination. Breaking before forward_scratch leaves a gap at
            // the im_end position which corrupts the next turn's context.
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, next_token, seq_pos,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward");
            seq_pos += 1;

            if next_token == config.eos_token { break; }
            if Some(next_token) == im_end_token { break; }

            logits = gpu.download_f32(&scratch.logits).unwrap();
            // Anti-repeat pipeline. apply_ngram_block is DISABLED across
            // the full conversation history because it will aggressively
            // block legitimate tokens that happen to follow n-grams from
            // the user's earlier turns (e.g. if the user asked "rephrase
            // your answer" and we already used those words, the block
            // will stop us from generating anything sensible).
            // Only apply it over the current turn's own tokens.
            let turn_start = history.len() - generated.len();
            llama::apply_ngram_block(&mut logits, &history[turn_start..]);
            llama::apply_repeat_penalty(
                &mut logits,
                &history[turn_start..],
                sc.repeat_window,
                sc.repeat_penalty,
            );

            // If we JUST forced out of thinking because of MAX_THINK_TOKENS,
            // inject the </think> token sequence into history so the model
            // actually sees it.
            if think_count == MAX_THINK_TOKENS && in_thinking == false && !generated.ends_with(&think_end_seq) {
                // Append </think>\n as the next token(s). Simplest: force
                // the next token to be the first token of </think>.
                // (Multi-token </think> is handled by forward_scratching
                // subsequent tokens below if needed.)
                next_token = think_end_seq[0];
                continue;
            }

            let t = if in_thinking { sc.think_temp } else { sc.answer_temp };
            next_token = llama::sample_top_p(&logits, t, top_p);

            if seq_pos + 4 >= max_seq {
                eprintln!("\n[ctx exhausted at seq_pos={}]", seq_pos);
                break;
            }
        }
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let gen_tok_s = generated.len() as f64 / (gen_ms / 1000.0);

        // If generation stopped because of max_gen (not EOS / im_end), the
        // turn is unterminated and the next turn's prefix will be glued onto
        // an unclosed assistant block — which confuses the model. Force-close
        // the turn by writing the closing tokens to the KV cache so the next
        // turn starts from a clean boundary.
        let last = generated.last().copied();
        let needs_close = last != Some(config.eos_token) && last != im_end_token;
        if needs_close {
            let mut closing: Vec<u32> = Vec::new();
            if in_thinking {
                // Still in thinking — close the thought first.
                closing.extend_from_slice(&think_end_seq);
                closing.extend_from_slice(&nl);
            }
            // Then close the assistant turn.
            closing.extend_from_slice(&im_end);
            closing.extend_from_slice(&nl);
            for &tok in &closing {
                qwen35::forward_scratch(
                    &mut gpu, &weights, &config, tok, seq_pos,
                    &mut kv_cache, &mut dn_state, &scratch,
                ).expect("forward close");
                seq_pos += 1;
                history.push(tok);
            }
            eprintln!("\n[turn {} force-closed ({} tokens appended)]",
                     turn_idx + 1, closing.len());
        }

        println!();
        eprintln!("[turn {} decode: {} tokens in {:.1} ms = {:.0} tok/s] (ctx={})\n",
                 turn_idx + 1, generated.len(), gen_ms, gen_tok_s, seq_pos);

        let _ = rng_state; // unused; present for future sample API
    }

    println!("\n================================================================");
    println!("DONE. Total ctx: {} / {} tokens", seq_pos, max_seq);
    println!("================================================================");
}
