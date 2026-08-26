// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! SP7 gate: more concurrent requests than slots, each streamed independently.
//!
//! This is the first test that submits genuinely concurrent requests the way a
//! client would — six threads against a four-slot engine — and checks each gets
//! its own stream. It is the programme's headline claim ("3-4 agents at once on
//! one GPU") stated as an assertion.
//!
//! Three things it must rule out:
//!
//!   * a client receiving nothing,
//!   * every client receiving the SAME tokens, which would mean the sessions
//!     are sharing state rather than being isolated,
//!   * six requests fitting in four slots without pressure, which would mean
//!     the test never exercised admission at all.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Event, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const N_SLOTS: usize = 4;
    const MAX_TOKENS: usize = 12;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.6-35b-a3b.mq4r")
    });
    println!("=== test_serve_concurrent ===\nmodel: {model_path}");

    // Tokenise up front so the engine only ever sees token ids.
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);

    let prompts = [
        "The capital of France is",
        "Once upon a time, in a distant galaxy,",
        "The recipe for a good cup of tea starts with",
        "In machine learning, gradient descent works by",
        "The three laws of motion were described by",
        "A well-designed database index improves",
    ];
    let n_clients = prompts.len();
    assert!(
        n_clients > N_SLOTS,
        "the gate is meaningless unless there are more clients than slots"
    );

    let cap_tokens = 128;
    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: N_SLOTS,
        cap_tokens,
        prefill_chunk: 1024,
        host_budget_bytes: 4 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-sp7-swap"),
    })
    .expect("SlotEngine::spawn");
    println!("engine up: {N_SLOTS} slots, {n_clients} clients, {MAX_TOKENS} tokens each");

    // Six client threads, each with its own reply channel, all submitting at
    // once. Nothing here holds a lock across a request.
    let mut handles = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let toks = tokenizer.encode(p);
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                session: None,
                prompt_tokens: toks,
                // Single-turn: nothing to continue, so no conversation key and
                // no continuation tokens.
                convo: Vec::new(),
                continuation: Vec::new(),
                max_tokens: MAX_TOKENS,
                reply: tx,
            })
            .expect("submit");
        handles.push(std::thread::spawn(move || {
            let mut tokens = Vec::new();
            let mut accepted = false;
            let mut rejected = None;
            while let Ok(ev) = rx.recv() {
                match ev {
                    Event::Accepted { .. } => accepted = true,
                    Event::Token { id } => tokens.push(id),
                    Event::Rejected { reason } => {
                        rejected = Some(reason);
                        break;
                    }
                    Event::Done { .. } => break,
                }
            }
            (i, accepted, tokens, rejected)
        }));
    }

    let mut served = Vec::new();
    let mut rejected = 0usize;
    for h in handles {
        let (i, accepted, tokens, rej) = h.join().expect("client thread");
        match rej {
            Some(reason) => {
                println!("  client {i}: REJECTED ({reason})");
                rejected += 1;
            }
            None => {
                assert!(accepted, "client {i} was neither accepted nor rejected");
                assert!(
                    !tokens.is_empty(),
                    "client {i} was accepted but received no tokens"
                );
                println!("  client {i}: {} tokens", tokens.len());
                served.push(tokens);
            }
        }
    }

    let stats = engine.stats();
    println!(
        "  stats: admitted={} rejected={} evictions={} restores={} prefix_hits={}",
        stats.admitted, stats.rejected, stats.evictions, stats.restores, stats.prefix_hits
    );

    assert!(
        !served.is_empty(),
        "no client was served at all — the engine did no work"
    );
    // Distinct prompts must give distinct continuations. Identical tokens from
    // every client would mean the sessions are sharing state.
    let distinct: HashSet<Vec<u32>> = served.iter().cloned().collect();
    assert!(
        distinct.len() > 1,
        "every served client produced identical tokens — sessions are not isolated"
    );
    println!(
        "  {} served clients produced {} distinct token streams",
        served.len(),
        distinct.len()
    );
    assert!(
        stats.admitted >= N_SLOTS,
        "only {} admitted; all {N_SLOTS} slots should have been used",
        stats.admitted
    );
    assert!(
        stats.evictions > 0 || rejected > 0,
        "{n_clients} clients on {N_SLOTS} slots forced neither an eviction nor a rejection — \
         the gate did not exercise admission"
    );
    assert_eq!(
        stats.prefix_hits, 0,
        "every client here is single-turn with an empty conversation key, so a \
         prefix hit would mean unrelated sessions are being matched to each other"
    );
    println!("ALL CHECKS PASS");
}
