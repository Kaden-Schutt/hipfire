// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek-V4 and small-architecture generation bodies, lifted verbatim
//! from `crates/hipfire-daemon/src/main.rs` (wave 5 / D3).
//! See `lib.rs` for the layering rationale.

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_gemma4 as gemma4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_muse_glimmer as glimmer;
use hipfire_arch_qwen2::qwen2;
use std::any::Any;
use std::path::PathBuf;
use std::sync::OnceLock;

use hipfire_loader::{AsstTurnCache, LoadedModel};
use hipfire_runtime::prompt_frame::ThinkMode;
use std::io::Write;
use std::time::Instant;

use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;

use crate::common::*;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_runtime::emit_text::{
    ThinkOutputRouter, ThinkRouteEvent, ToolOutputRouter, ToolRouteError, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig};
use hipfire_runtime::spec::{
    accept_greedy_prefix, ClientEvent, EvictRetain, FinishSummary, SpecRequestConfig, SpecTarget,
    Speculator, StopReason,
};

pub fn glimmer_turn_key(fp: u64, ordinal: usize) -> u64 {
    fp.wrapping_add((ordinal as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_mul(0x9E3779B97F4A7C15)
}

/// Post-parse generation / active-request errors.
///
/// Attempt ID is always read from TLS [`active_attempt_id`] — callers cannot
/// pass, hard-code, or choose zero independently.
pub fn emit_active_attempt_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    write_error_envelope(
        stdout,
        id,
        message,
        class,
        retryable,
        rolled_back,
        active_attempt_id(),
    );
}

/// Speculative wire terminal after `Deepseek4Emit::finish` + length known.
/// Length always suppresses call release and cache; malformed is error XOR done.
#[derive(Debug, Clone, PartialEq)]
pub enum Ds4SpecWireTerminal {
    Malformed(Ds4MalformedTerminalAction),
    Done {
        finish_reason: &'static str,
        release_tool_calls: bool,
        store_cache: bool,
    },
}

pub fn ds4_spec_wire_terminal(
    finish_reason: &str,
    tool_calls: usize,
    hit_length_cap: bool,
) -> Ds4SpecWireTerminal {
    if finish_reason == "malformed_protocol" {
        return Ds4SpecWireTerminal::Malformed(ds4_malformed_terminal_action(
            "unclosed DSML tool_calls block at end of output",
        ));
    }
    if hit_length_cap {
        return Ds4SpecWireTerminal::Done {
            finish_reason: "length",
            release_tool_calls: false,
            store_cache: false,
        };
    }
    if tool_calls > 0 {
        Ds4SpecWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: true,
            store_cache: true,
        }
    } else {
        Ds4SpecWireTerminal::Done {
            finish_reason: "stop",
            release_tool_calls: false,
            store_cache: true,
        }
    }
}

/// Derive cache action from the speculative wire terminal + finish payload.
/// Length/malformed never store; safe stop/tool_calls store when authorized.
pub fn ds4_cache_action(
    terminal: &Ds4SpecWireTerminal,
    finish: &FinishSummary,
    visible_for_cache: &str,
) -> Ds4CacheAction {
    match terminal {
        Ds4SpecWireTerminal::Malformed(_) => Ds4CacheAction {
            store: false,
            fingerprint_text: String::new(),
            tool_calls: Vec::new(),
        },
        Ds4SpecWireTerminal::Done {
            store_cache,
            release_tool_calls,
            ..
        } => {
            let tool_calls = if *release_tool_calls {
                finish_summary_held_tool_calls(finish)
            } else {
                Vec::new()
            };
            Ds4CacheAction {
                store: *store_cache,
                fingerprint_text: normalize_asst_turn_for_fingerprint(visible_for_cache),
                tool_calls,
            }
        }
    }
}

/// Full single-GPU DS4 AR abort reset while the decode `state` borrow is live
/// (disjoint host fields on `m` + explicit `state`). Attests via device sync,
/// then emits the correlated cancel lifecycle (or unattested fail-closed error).
///
/// Callers must pass host fields reborrowed from `LoadedModel` separately from
/// `state` (which is already borrowed from `m.state` as `ModelState::Deepseek4`).
pub fn ds4_ar_client_abort(
    stdout: &mut impl std::io::Write,
    id: &str,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    speculator: &mut Option<Box<dyn Speculator>>,
    gpu: &mut rdna_compute::Gpu,
    state: &mut deepseek4::DeepseekV4State,
    completion_tokens: usize,
) {
    *seq_pos = 0;
    conversation_tokens.clear();
    state.reset();
    state.zero_decode_caches(gpu);
    free_checkpoints(prefill_checkpoints, gpu);
    free_checkpoints(dflash_checkpoints, gpu);
    let spec_reset = if let Some(s) = speculator.as_mut() {
        s.reset(gpu).map_err(|e| format!("spec.reset: {e}"))
    } else {
        Ok(())
    };
    fail_closed_invalidate_graphs_and_replay(gpu);
    let epilogue = fail_closed_epilogue_after_sync(spec_reset, fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    emit_spec_cancel_after_rollback(stdout, id, completion_tokens, &epilogue);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerSpecMode {
    Off,
    Greedy,
    ChainSampled,
}

/// Glimmer speculation admission. Greedy is the validated path; the sampled
/// chain-sample path mirrors the arch 5/6 `chain_sample_route` conditions.
pub fn glimmer_spec_admission(
    has_drafter: bool,
    max_tokens: usize,
    temp: f32,
    min_p: Option<f32>,
    fast_sample_on: bool,
    temp_spec_env_off: bool,
    batched_logits_available: bool,
) -> GlimmerSpecMode {
    if !has_drafter || max_tokens <= 1 {
        return GlimmerSpecMode::Off;
    }
    if temp <= 0.01 {
        return GlimmerSpecMode::Greedy;
    }
    if temp > 1e-6
        && fast_sample_on
        && !temp_spec_env_off
        && min_p.map_or(true, |p| p <= 0.0)
        && batched_logits_available
    {
        return GlimmerSpecMode::ChainSampled;
    }
    GlimmerSpecMode::Off
}

#[cfg(test)]
mod deepseek4_reasoning_prefix_tests {
    use super::{
        deepseek4_reasoning_prefix, ThinkMode, DEEPSEEK4_REASONING_HIGH_PREFIX,
        DEEPSEEK4_REASONING_MAX_PREFIX,
    };

    #[test]
    fn parent_effort_prefixes_are_distinct_and_low_is_empty() {
        assert_eq!(deepseek4_reasoning_prefix(ThinkMode::NonThink), "");
        assert_eq!(deepseek4_reasoning_prefix(ThinkMode::Low), "");
        assert_eq!(
            deepseek4_reasoning_prefix(ThinkMode::High),
            DEEPSEEK4_REASONING_HIGH_PREFIX
        );
        assert_eq!(
            deepseek4_reasoning_prefix(ThinkMode::Max),
            DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert_ne!(
            DEEPSEEK4_REASONING_HIGH_PREFIX,
            DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert!(DEEPSEEK4_REASONING_HIGH_PREFIX.ends_with("\n\n"));
        assert!(DEEPSEEK4_REASONING_MAX_PREFIX.ends_with("\n\n"));
    }
}

/// Typed daemon error envelope (Increment B / Task 13).
///
/// Wire: `{type:error, id?, message, class, retryable, rolled_back, attempt_id}`.
/// Classification MUST use `class`/`retryable`/`rolled_back` — never message prefix.
///
/// Prefer [`emit_active_attempt_error`] (post-activation) or
/// [`emit_active_attempt_error`] (pre-activation protocol reject). This low-level
/// writer is reserved for paths that must echo a validated non-TLS attempt
/// (reset ack failures after a parsed reset attempt_id).
pub fn write_error_envelope(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
    attempt_id: u64,
) {
    let mut envelope = serde_json::json!({
        "type": "error",
        "message": message,
        "class": class,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": attempt_id,
    });
    if let Some(id) = id {
        envelope["id"] = serde_json::Value::String(id.to_owned());
    }
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// deepseek4 MTP spec-decode through the unified `generate_spec` (Phase 4 T4c-2).
///
/// This is the ds4 sibling of `generate_dflash`: it owns the arch-specific
/// prologue (DSML prompt render + `plan_cache(CachePolicy::deepseek4())` + the
/// DSA decode-cache miss teardown) and epilogue (the ds4 `done` envelope), and
/// drives the shared decode core via `m.speculator` (a `Deepseek4MtpDrafter`),
/// the `Deepseek4Bundle` target (via `spec_target_guard`), and `Deepseek4Emit`.
/// Greedy-only — the dispatch routes here only at `temp <= 1e-6`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn generate_deepseek4_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    // Sampling: temp<=1e-6 → greedy (argmax-accept, byte-identical to before);
    // temp>0 → DSpark sampled verify. cactus_delta is the opt-in acceptance boost.
    temp: f32,
    top_p: f32,
    top_k: usize,
    cactus_delta: f32,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    // Zero-budget reject before DSML render, decode-cache teardown, graph
    // invalidate, decoded_vocab build, or set_sampling. Inner generate_spec
    // keeps the same guard as defense-in-depth.
    if max_tokens == 0 {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // eos token (for the DSML prompt build) from the bundle — immutable peek.
    let Some(b) = m.deepseek4() else {
        emit_error_with_id(stdout, id, "deepseek4 bundle missing on arch_id=9 spec");
        return;
    };
    let eos_tok = b.eos_tok;

    // DSML prompt render (same builder the bespoke loop uses).
    let prompt_ids = {
        let tokenizer = match m.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                emit_error_with_id(stdout, id, "tokenizer not loaded");
                return;
            }
        };
        build_deepseek4_dsml_prompt(
            tokenizer,
            system_prompt,
            tools,
            messages_history,
            prompt,
            think_mode,
            eos_tok,
            &mut m.asst_turn_cache,
        )
    };
    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }

    // spec_k: env chain → 2 (the deepseek4-specific default; see generate_deepseek4).
    let spec_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| Some(hipfire_runtime::config::get().mtp_k))
        .unwrap_or(2);

    // Prefix-cache plan (ds4 policy: forced-cold on partial, ring-safety length
    // guard, step-back exact). Pure decision; the GPU teardown is applied below.
    let plan = hipfire_runtime::cache_plan::plan_cache(
        &prompt_ids,
        &m.conversation_tokens,
        &hipfire_runtime::cache_plan::CachePolicy::deepseek4(),
        &[],
        false,
    );
    let cached_tokens = plan.cached_tokens;
    let suffix: Vec<u32> = prompt_ids[plan.start_pos..].to_vec();

    // Capacity guard (KV sized for physical_cap; overrun is a serve-killing
    // panic). Mirrors the bespoke ds4 pre-prefill guard — generate_spec's own
    // guard checks ctx_capacity (max_position_embeddings), which for ds4 can far
    // exceed physical_cap, so keep this explicit one.
    if plan
        .start_pos
        .saturating_add(suffix.len())
        .saturating_add(max_tokens)
        > m.physical_cap
    {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                plan.start_pos + suffix.len(),
                max_tokens,
                m.physical_cap
            ),
        );
        return;
    }

    let required_tokens = plan
        .start_pos
        .saturating_add(suffix.len())
        .saturating_add(max_tokens);
    if let Some(bundle) = m.deepseek4_mut() {
        if let Err(error) = deepseek4::forward::ensure_compressor_capacity(
            &bundle.config,
            &mut bundle.state,
            gpu,
            required_tokens,
        ) {
            emit_error_with_id(
                stdout,
                id,
                format!("deepseek4 context-capacity preflight failed: {error}"),
            );
            return;
        }
    }

    // DSA decode-cache miss teardown (the part NOT done by the drafter's
    // cache-miss `state.reset()` or generate_spec's seq_pos/conversation clear):
    // zero the position-indexed rings + invalidate the captured decode graph so a
    // fresh conversation reproduces a freshly-launched daemon's clean state.
    if !plan.cache_hit {
        if let Some(b) = m.deepseek4_mut() {
            b.state.zero_decode_caches(gpu);
        }
        gpu.invalidate_graph_state();
    }

    // The ds4 emitter builds its in-step tool-call grammar from the raw tool
    // JSON inside `make_spec_emitter`, but that grammar masks per-token over the
    // decoded vocab — which the neutral `SpecEmitCtx` can't lazily derive (it has
    // no `&mut m`). Build/cache the vocab Arc here when tools are present and
    // hand it down; mirrors the lazy cache the old `build_deepseek4_spec_grammar`
    // did internally.
    let decoded_vocab: Option<std::sync::Arc<Vec<String>>> =
        if tools.map_or(false, |t| !t.is_empty()) {
            if m.decoded_vocab.is_none() {
                let tok = m.tokenizer.as_ref().expect("tokenizer present");
                let n = tok.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tok.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
        } else {
            None
        };

    // Configure the speculator's sampling before the loop — mirrors generate_dflash's
    // configure_request before generate_spec. Greedy (temp<=1e-6) leaves it at argmax-accept
    // (byte-identical to the prior hardcoded-greedy path); temp>0 drives the DSpark
    // sampled verify, and cactus_delta>0 applies the opt-in acceptance boost.
    if let Some(spec) = m.speculator.as_mut() {
        spec.configure_request(SpecRequestConfig {
            temp,
            top_p,
            top_k,
            min_p: 0.0,
            cactus_delta,
            rng_seed: SpecRequestConfig::default().rng_seed,
            allow_ngram_modifier: false,
        });
    }

    // Open the wire contract before any token can reach the client. The CLI's
    // stream latch (`StreamContractError::PreStartEvent`) fail-closes on any
    // event that precedes `gen_start`, so without this every DS4 request over
    // `hipfire serve` dies with "stream must begin with gen_start; got token
    // before contract latch". Mirrors generate_dflash's emit before
    // generate_spec.
    //
    // `ds4_gen_start_contract_version()` is None for arch 9 — DS4 does not
    // advertise semantic contract v2, so the client keeps whole-output tool
    // extraction. That helper exists precisely for this call site.
    //
    // started_in_think follows the DS4 frame mapping documented on ThinkMode
    // (prompt_frame.rs): NonThink renders `<｜Assistant｜></think>`, so the model
    // begins in visible-answer mode; High/Max render the `<think>` open-token,
    // so it begins inside the reasoning span.
    emit_gen_start(
        stdout,
        id,
        !matches!(think_mode, ThinkMode::NonThink),
        ds4_gen_start_contract_version(),
    );
    let prompt_tokens_total = prompt_ids.len();
    let run = match crate::qwen::generate_spec(
        m,
        gpu,
        stdout,
        id,
        prompt_ids,
        suffix,
        plan.start_pos,
        plan.cache_hit,
        None, // resume_from — ds4 has no DeltaNet checkpoints
        max_tokens,
        SpecEmitRequest {
            im_end: None,
            tools: tools.map(|t| t.to_vec()),
            stop: Vec::new(),
            max_think: 0,
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            think_mode,
            decoded_vocab,
        },
        temp, // temp>0 → DSpark sampled verify (routed here only when supports_temp_verify)
    ) {
        Some(r) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        None => return,
    };

    // ── ds4 done envelope ────────────────────────────────────────
    let tok_s = if run.decode_s > 0.0 {
        run.generated as f64 / run.decode_s
    } else {
        0.0
    };
    let prefill_tok_s = if run.prefill_s > 0.0 {
        run.prefill_tokens_len as f64 / run.prefill_s
    } else {
        0.0
    };
    // accept_pct denominator is windows × k (the bespoke loop added `spec_k` per
    // window to `spec_drafts_offered`, not the actual n_proposed).
    let accept_pct = if run.spec_cycles > 0 && spec_k > 0 {
        run.spec_accepted as f64 / (run.spec_cycles * spec_k) as f64 * 100.0
    } else {
        0.0
    };
    // Fail-closed DSML / length-suppresses-calls: Deepseek4Emit holds ToolCalls
    // on FinishSummary; generate_spec does not render them. Classify length vs
    // stop vs malformed here, then release held calls only when tool-safe.
    // Prefer generate_spec's production epilogue when the turn ended fail-closed
    // (grammar / open-think / malformed) so rolled_back is truthful.
    if let Some(ep) = run.fail_closed_rollback.as_ref() {
        let detail = if run.grammar_violated {
            "grammar violation during speculative decode"
        } else if run.finish.open_think || run.finish.finish_reason == "open_think" {
            "open think span at end of generation (validation)"
        } else if run.finish.finish_reason == "malformed_protocol" {
            "unclosed DSML tool_calls block at end of output"
        } else {
            "fail-closed speculative decode"
        };
        let mut action = ds4_malformed_terminal_action(detail);
        action.rolled_back = ep.rolled_back;
        if let Some(ctx) = ep.context.as_ref() {
            if !ep.rolled_back {
                action.message = format!("{} ({ctx})", action.message);
            }
        }
        emit_ds4_malformed_action(stdout, id, &action);
        return;
    }
    // Semantic stop (StopSequence/EOS/ThinkCap) or decoded_eot at cap is not
    // length — preserves stop/tool_calls when generated == max_tokens.
    let hit_length_cap = qwen_dflash_hit_length_cap(
        run.generated,
        max_tokens,
        run.finish.decoded_eot,
        run.semantic_stop.is_some(),
    );
    match ds4_spec_wire_terminal(
        run.finish.finish_reason,
        run.finish.tool_calls,
        hit_length_cap,
    ) {
        Ds4SpecWireTerminal::Malformed(action) => {
            // No epilogue on SpecRun — still fail-closed without claiming rollback.
            emit_ds4_malformed_action(stdout, id, &action);
            return;
        }
        Ds4SpecWireTerminal::Done {
            finish_reason,
            release_tool_calls,
            store_cache,
        } => {
            // τ / drafter / pending done fixed before handshake so commit_ready
            // carries the exact eventual done payload.
            let tau = if run.spec_cycles > 0 {
                run.spec_accepted as f64 / run.spec_cycles as f64
            } else {
                0.0
            };
            let drafter = m.speculator.as_ref().map(|s| s.name()).unwrap_or("none");
            let mut pending_done = serde_json::json!({
                "type": "done",
                "id": id,
                "tokens": run.generated,
                "tok_s": tok_s,
                "prompt_tokens": prompt_tokens_total,
                "prefill_tokens": run.prefill_tokens_len,
                "cached_tokens": cached_tokens,
                "prefill_ms": (run.prefill_s * 1000.0) as u128,
                "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
                "decode_tok_s": (tok_s * 10.0).round() / 10.0,
                "total_ms": (run.total_s * 1000.0) as u128,
                "finish_reason": finish_reason,
                "drafter": drafter,
                "tau": tau,
                "spec_k": spec_k,
                "spec_windows": run.spec_cycles,
                "spec_accept_pct": accept_pct,
                "attempt_id": active_attempt_id(),
            });
            // Stage held ToolCalls into commit_ready/done; no post-commit event.
            let held_calls = finish_summary_held_tool_calls(&run.finish);
            stage_terminal_tool_calls(&mut pending_done, finish_reason, &held_calls);
            // Two-phase commit before cache / done.
            let decision = await_client_terminal_commit(stdout, id, &pending_done);
            let effects = ds4_client_commit_effects(decision, release_tool_calls, store_cache);
            if !effects.emit_done {
                let ep = production_fail_closed_rollback(m, gpu, None, None);
                emit_spec_cancel_after_rollback(stdout, id, run.generated, &ep);
                return;
            }
            // Honor computed store_cache: safe stop/tool_calls store the
            // verbatim raw streamed_tokens body for DSML asst-turn replay.
            // Length sets store_cache=false; malformed returned above; Abort
            // suppressed via effects.store_cache.
            let terminal = Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls: effects.release_tool_calls,
                store_cache: effects.store_cache,
            };
            let action = ds4_cache_action(&terminal, &run.finish, run.finish.visible_text.as_str());
            if action.store {
                if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    eprintln!(
                        "[asst-cache store spec] content.len={} tool_calls={} tokens={}",
                        action.fingerprint_text.len(),
                        action.tool_calls.len(),
                        run.streamed_tokens.len(),
                    );
                }
                let _ = ds4_apply_cache_action(
                    |fp, seq| {
                        if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                            .ok()
                            .as_deref()
                            == Some("1")
                        {
                            eprintln!(
                                "[asst-cache store spec] fp={:#018x} tokens={}",
                                fp,
                                seq.len()
                            );
                        }
                        m.asst_turn_cache.insert(
                            fp,
                            hipfire_runtime::prompt_frame::CachedAssistantTurn {
                                reasoning: None,
                                tools: Vec::new(),
                                content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                                    token_ids: seq,
                                    text: String::new(),
                                }),
                            },
                        );
                    },
                    &action,
                    run.streamed_tokens.clone(),
                );
            }
            emit_staged_terminal_done(stdout, &pending_done);
            // Per-request debug summary (stderr → serve.log): active drafter, τ, tok/s.
            eprintln!(
                "[req {id}] drafter={drafter} tau={tau:.2} tok/s={tok_s:.1} decode ({} tok, {} windows, accept={accept_pct:.0}%)",
                run.generated, run.spec_cycles
            );
        }
    }
}
pub fn generate_deepseek4(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    if m.deepseek4_heterogeneous_mut().is_some() {
        generate_deepseek4_heterogeneous(
            m,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            think_mode,
            tools,
            messages_history,
        );
        return;
    }
    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    // The single-GPU ds4 bundle (config/weights/state/eos/pbs) lives in `Deepseek4Bundle`.
    // `pbs` is now inside the bundle so we downcast once and split fields disjointly.
    let Some(b) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
    }) else {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"deepseek4_config missing on arch_id=9 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    };
    // Split the bundle's fields disjointly so pbs (+state) and config/weights/eos do not overlap.
    let hipfire_arch_deepseek4::Deepseek4Bundle {
        config: cfg,
        weights,
        state,
        pbs,
        eos_tok,
        ..
    } = &mut *b;
    let pbs = pbs
        .as_mut()
        .expect("deepseek4_pbs missing on arch_id=9 generate");
    let cfg = &*cfg;
    let weights = &*weights;
    let eos_tok = *eos_tok;

    let prompt_ids = build_deepseek4_dsml_prompt(
        tokenizer,
        system_prompt,
        tools,
        messages_history,
        prompt,
        think_mode,
        eos_tok,
        &mut m.asst_turn_cache,
    );

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    if std::env::var("HIPFIRE_DEEPSEEK4_DUMP_PROMPT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let rendered = tokenizer.decode(&prompt_ids);
        let path = format!(
            "/tmp/hipfire-prompt-{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        let _ = std::fs::write(
            &path,
            format!("# tokens: {}\n{}\n", prompt_ids.len(), rendered),
        );
        eprintln!("[v4f prompt dump] tokens={} → {}", prompt_ids.len(), path);
    }

    // This function is now AR-only: the MTP spec-decode path (greedy, temp≈0,
    // speculator present) is dispatched to `generate_deepseek4_spec` (T4c-2),
    // which owns the spec_requested/spec_mode/spec_k resolution. Reaching here
    // means either temp>0 (sample) or no speculator (fallback) — either way the
    // plain AR sampler below honours the requested `temp`/`top_p`.

    let t0 = Instant::now();

    // ── Prefix-cache LCP detection ──────────────────────────────────
    //
    // Reasonix's prompt-caching model (`tmp/reasonix_arch.md` Pillar 1):
    // construct prompts as `immutable_prefix + append_only_log` so the
    // backend's prefix cache hits on every turn. Reasonix is a CLIENT
    // that targets DeepSeek's server-side cache; for LOCAL inference we
    // implement the server side here.
    //
    // Compare the freshly-tokenized prompt against the tokens we know
    // are already resident in the V4F KV / SWA / compressed-KV rings
    // from the prior request (`m.conversation_tokens`). If the new
    // prompt FULLY EXTENDS the prior conversation — i.e., starts with
    // the entire `conversation_tokens` — we can skip prefill for those
    // tokens and only prefill the suffix.
    //
    // SWA-safety analysis for partial LCP (lcp < prior.len()):
    //
    // Suppose prior wrote positions [0..prior_max_pos], turn 2's suffix
    // writes [lcp..prompt_ids.len()-1]. After turn 2's prefill the new
    // max position is `prompt_ids.len() - 1`. The model's first decode
    // attends to a window of `min(prompt_ids.len(), 128)` positions
    // ending at `prompt_ids.len() - 1`. Each window position maps to a
    // unique ring slot via `pos % 128`. For correctness, every slot in
    // that window must currently hold K_rotated for the matching
    // position:
    //
    //   * For positions in `[0..lcp-1]` — turn 1 wrote them, content
    //     matches by LCP definition. Untouched since.
    //   * For positions in `[lcp..prompt_ids.len()-1]` — turn 2's suffix
    //     prefill just wrote them. Content matches the new prompt.
    //
    // Stale-slot risk: if turn 1 had written a slot at some position
    // `P_late ∈ [lcp..prior_max_pos]` AND turn 2 doesn't overwrite that
    // slot, the slot holds K_rotated for P_late, not the new prompt's
    // token at that position. The window read returns wrong content.
    //
    // Turn 2's suffix prefill covers positions [lcp..prompt_ids.len()-1].
    // To overwrite every slot turn 1 wrote in `[lcp..prior_max_pos]`,
    // we need `prompt_ids.len() - 1 ≥ prior_max_pos`, i.e.
    // `prompt_ids.len() ≥ prior.len()`. Equivalently: the new prompt
    // must be at least as long as the cached conversation.
    //
    // We additionally guard `lcp == prior.len() && prompt_ids.len() ==
    // prior.len()` (full match, nothing to do) with a noop check
    // downstream (suffix_tokens is empty).
    //
    // After the daemon's `reset` handler clears `m.conversation_tokens`
    // (legacy stateless path), `prior` is empty and `lcp = 0` → full
    // prefill. For prefix-cache mode the serve stops calling reset for
    // V4F and lets this LCP detection drive cache-hit accounting.
    let lcp: usize = {
        let prior = &m.conversation_tokens;
        if prior.is_empty() || prompt_ids.len() < prior.len() {
            0
        } else {
            let mut n = 0usize;
            while n < prior.len() && prior[n] == prompt_ids[n] {
                n += 1;
            }
            // Edge case: new prompt is byte-identical to the cached
            // conversation. Suffix would be empty and
            // `forward_prefill_batch_chunked` errors on that. Step the
            // LCP back one so we always prefill ≥ 1 token (and the
            // post-prefill logits are well-defined for the first
            // decode step). Costs us one token of cache credit on
            // exact-repeat prompts — rare in practice.
            if n == prompt_ids.len() && n > 0 {
                n - 1
            } else {
                n
            }
        }
    };

    // DSA compressor-ring safety on a PARTIAL prefix-cache hit.
    //
    // The DSA decode caches (SWA ring, compressor/indexer ring state, full +
    // compressed KV) are *position-indexed* and were left by the prior turn at
    // ITS end position. A FULL hit (`lcp == prior length`) resumes exactly where
    // the prior turn left those rings, so the incremental prefill is correct —
    // this is the normal "growing conversation" path and stays fast.
    //
    // A PARTIAL hit (`0 < lcp < prior length`) resumes the suffix prefill from
    // `start_pos = lcp`, but the compressor ring still holds the prior turn's
    // *end* window, not `lcp`'s. The first compressed block committed after the
    // resume point then pools a STALE overlap window — and with ratio-4 overlap
    // that window reaches back over the just-cached tail, corrupting far-context
    // recall (the cwd/tool-path "lossiness" symptom). The ring can't be cheaply
    // repopulated (a position's hidden state depends on its SWA window, which
    // chains all the way back to token 0), so the correct, robust fix is to fall
    // back to a cold rebuild for partial hits only. Full hits are unaffected.
    let lcp = if lcp > 0 && lcp < m.conversation_tokens.len() {
        0
    } else {
        lcp
    };

    // Select/map the complete request's compressed-cache bucket before any
    // cache reset, prefill, HipGraph capture, or retained-PM4 replay. DS4's
    // VMM owner addresses remain stable; crossing a geometry bucket re-arms
    // replay automatically so the new tape is captured once at that shape.
    let required_tokens = prompt_ids.len().saturating_add(max_tokens);
    if let Err(error) =
        deepseek4::forward::ensure_request_capacity(cfg, state, gpu, pbs, required_tokens)
    {
        emit_error_with_id(
            stdout,
            id,
            format!("deepseek4 context-capacity preflight failed: {error}"),
        );
        return;
    }

    if lcp == 0 {
        // Cache miss — start a fresh conversation in V4F's state.
        state.reset();
        // reset() only rewinds n_tokens; the position-indexed decode caches
        // (SWA ring, compressed/full KV, indexer scratch) still hold the prior
        // turn's residue, which bleeds into this fresh conversation's forward
        // and makes greedy output drift turn-to-turn (the "recall/tool-calls
        // unreliable" symptom). Zero them so a fresh conversation reproduces a
        // freshly-launched daemon's clean, deterministic state.
        state.zero_decode_caches(gpu);
        m.conversation_tokens.clear();
        // Tear down the captured V4F decode hipGraph alongside the
        // state, same rationale as the daemon's `"reset"` handler:
        // a fresh-context turn invalidates every device-buffer pointer
        // and host scalar the captured graph baked in at capture time
        // (state.attn_state_buf slot/n_valid/k_active values derived
        // from the prior n_tokens, compressor ring/commit slots, etc.).
        // Without this, the warmup-then-replay state machine fires
        // warmup on the first decode (because `state.reset()` clears
        // `ar_forward_warmed_up`), then immediately replays the STALE
        // graph on the second decode and crashes with the same
        // "download logits (graph path): illegal memory access" we
        // saw on multi-turn pi sessions before the explicit-reset fix.
        gpu.invalidate_graph_state();
    }
    let start_pos: u32 = lcp as u32;

    // Slice off the suffix — the only tokens we actually need to prefill.
    // For lcp=0 this is the full prompt; for a full cache hit on a turn
    // that adds N new tokens this is just those N.
    let suffix_tokens: &[u32] = &prompt_ids[lcp..];

    // O2b-2 capacity guard (ds4 single-GPU): after any cache reset above, the
    // KV ends at start_pos + suffix_tokens.len() (== prompt_ids.len()) and
    // decode appends max_tokens. forward_prefill_batch_chunked writes into a KV
    // sized for m.physical_cap; overrunning it is a KV-overrun panic that takes
    // down serve. Emit a clean error and return BEFORE prefill.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if (start_pos as usize)
        .saturating_add(suffix_tokens.len())
        .saturating_add(max_tokens)
        > m.physical_cap
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds checkpoint context capacity: prompt={} + max_tokens={} > capacity={}"}}"#,
            id,
            start_pos as usize + suffix_tokens.len(),
            max_tokens,
            m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    // Prefill: batched chunked through PBS. (The MTP-fill prefill variant moved
    // to the drafter's `mtp_prefill`, driven by `generate_deepseek4_spec`.)
    let prefill_result = deepseek4::forward::forward_prefill_batch_chunked(
        cfg,
        weights,
        state,
        gpu,
        suffix_tokens,
        start_pos,
        pbs,
    );
    let last_logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("deepseek4prefill failed: {e:?}"));
            return;
        }
    };
    // `forward_prefill_batch_chunked` does NOT advance `state.n_tokens`.
    // Callers are responsible for it (mirrors deepseek4_chat's explicit
    // `state.n_tokens = pos as u64;` at deepseek4_chat.rs:324). Without this,
    // the next decode_step queries the SWA cache at the BOS position
    // instead of the next-prediction position and the model emits
    // attractor garbage at greedy temp=0.
    state.n_tokens = (start_pos as usize + suffix_tokens.len()) as u64;
    // Keep `m.conversation_tokens` in lockstep with what's actually
    // resident in the KV/SWA/compressed-KV rings:
    //   - On a CACHE MISS (lcp==0): replace with prompt_ids (we just
    //     full-prefilled the whole prompt).
    //   - On a CACHE HIT (lcp>0): truncate the prior tracker down to
    //     `lcp` before appending the suffix. For partial LCP this
    //     matters — tokens in the prior tracker beyond `lcp` came
    //     from a previous turn's decode but the slots they lived in
    //     have just been overwritten by the suffix prefill. Leaving
    //     them in the tracker would let the NEXT request's LCP
    //     comparison run off the end of what's actually cached and
    //     make divergent assumptions about ring contents.
    if lcp == 0 {
        m.conversation_tokens.clear();
        m.conversation_tokens.extend_from_slice(&prompt_ids);
    } else {
        m.conversation_tokens.truncate(lcp);
        m.conversation_tokens.extend_from_slice(suffix_tokens);
    }
    let cached_tokens: usize = lcp;

    // Sync to ensure all prefill kernels have completed before stopping
    // the timer (head's download_f32 already syncs but defensive).
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = t0.elapsed().as_millis();

    // Prefill-complete abort observation (full route reset, not cursor-only).
    if check_abort(id) {
        ds4_ar_client_abort(
            stdout,
            id,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.speculator,
            gpu,
            state,
            0,
        );
        return;
    }

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let pos_after_prefill = state.n_tokens as u32;

    // Sampler. HF DeepSeek-V4-Flash card recommends temp=1.0, top_p=1.0
    // for local deployment; we honor that as the default. Pure greedy
    // (temp <= 1e-6) is supported but enters block-level attractors on
    // structured prompts.
    let top_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let seed: u64 = std::env::var("HIPFIRE_DEEPSEEK4_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15)
        });
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    // Track whether the decode loop saw a complete
    // `<｜DSML｜tool_calls>` block close. Drives `finish_reason` in the
    // `done` envelope below.
    let mut tool_calls_parsed_count: usize;
    // Plain autoregressive decode. The MTP spec-decode path now routes through
    // `generate_deepseek4_spec` + the unified `generate_spec` (T4c-2); this
    // function is the AR sampler path (temp>0) and the no-speculator fallback.
    // The bare block scopes the parser/matcher/sampler locals (was the old
    // `else` arm of the now-deleted `if spec_mode` spec loop).
    {
        // Plain decode loop. Sampler honours `temp` + `top_p` from the
        // request; HF default is temp=1.0, top_p=1.0 (multinomial across
        // the full vocab, no nucleus cut). Greedy (temp <= 1e-6) is
        // dangerous — see fn doc.
        //
        // Tokens are fed through a DSML stream parser that recognises
        // `<think>…</think>` reasoning blocks and
        // `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` tool-call blocks. The
        // parser emits:
        //   - StreamEvent::Token(text)       → JSONL `{type:"token"}`
        //   - StreamEvent::Reasoning(text)   → JSONL `{type:"reasoning"}`
        //   - StreamEvent::ToolCalls(calls)  → JSONL `{type:"tool_calls"}`
        // Markers split across token boundaries are buffered until they
        // resolve. The CLI / HTTP layer maps these to OpenAI SSE chunks.
        // Prime the parser's initial state to match the bootstrap tag
        // we appended to `prompt_ids`. In High/Max think modes the
        // prompt ends with `<think>` and the model's first generated
        // token is the body of that thinking block — without
        // `new_in_think()` the parser would sit in `Normal` and
        // misclassify every reasoning token as plain content,
        // including the trailing `</think>` which then leaks into
        // `message.content`. NonThink mode appends `</think>` (closing
        // a zero-length think block) so the response starts in Normal.
        let mut parser = match think_mode {
            ThinkMode::Low | ThinkMode::High | ThinkMode::Max => {
                deepseek4::dsml::StreamParser::new_in_think()
            }
            ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
        };

        // Grammar-guided decoding setup. When tools are present, we mask
        // the logits against a small state machine that mirrors the DSML
        // format — inside a `<｜DSML｜tool_calls>` block the model can
        // only emit token IDs whose decoded text is a prefix of a legal
        // continuation (e.g. `<｜DSML｜invoke name="` or a schema-defined
        // tool name). In free-emission states (`Out`, `InParamBody`,
        // and any time tools is None / empty) the mask is all-true and
        // the mask compute is skipped.
        //
        // Why this exists: V4F MQ2-Lloyd has damaged logit precision on
        // format-structural tokens — even with the byte-identical HF
        // system prompt at temp=1.0 it deterministically emits invented
        // variants like `<｜DSML｜tool_cbl>`, `<｜DSML｜calling>`,
        // `</｜DSML｜paper>` that no parser can recover. The mask makes
        // those tokens unreachable at the sampler level.
        let tool_schemas: Vec<deepseek4::grammar::ToolSchema> = tools
            .map(|arr| {
                arr.iter()
                    .map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let parameters = func.get("parameters");
                        let params: Vec<String> = parameters
                            .and_then(|p| p.get("properties"))
                            .and_then(|p| p.as_object())
                            .map(|m| m.keys().cloned().collect())
                            .unwrap_or_default();
                        let required: Vec<String> = parameters
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        deepseek4::grammar::ToolSchema {
                            name,
                            params,
                            required,
                        }
                    })
                    .filter(|s: &deepseek4::grammar::ToolSchema| !s.name.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let grammar_active = !tool_schemas.is_empty();
        let mut matcher = deepseek4::grammar::Matcher::new(tool_schemas);
        // Precompute (or fetch the cached) decoded vocab. `tokenizer.decode`
        // per id over ~129k ids is allocator-heavy enough that doing it
        // per-request adds tens of ms of pure overhead to every tool-
        // using V4F turn. The cache lives on `LoadedModel.decoded_vocab`
        // as an `Arc<Vec<String>>` and is cleared on model unload.
        //
        // Borrow note: `m.decoded_vocab` is a disjoint field from
        // `m.state` (whose `ModelState::Deepseek4` bundle `state` holds `&mut`
        // to) and from `m.tokenizer` (which `tokenizer` holds `&` to), so the
        // assignment compiles under Rust's split-borrows.
        let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
            if m.decoded_vocab.is_none() {
                let n = tokenizer.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
        } else {
            None
        };
        let empty_vocab: Vec<String> = Vec::new();
        let decoded_vocab: &[String] = decoded_vocab_arc
            .as_deref()
            .map(|v| v.as_slice())
            .unwrap_or(&empty_vocab);
        let mut grammar_mask: Vec<bool> = vec![true; decoded_vocab.len()];

        // Open the wire contract before the first sample. Same fail-closed CLI
        // latch as the spec path: any event ahead of `gen_start` aborts the
        // stream with "stream must begin with gen_start; got token before
        // contract latch". Placed after prefill and grammar setup but before
        // the first `sample_token`, so no token can outrun it.
        emit_gen_start(
            stdout,
            id,
            !matches!(think_mode, ThinkMode::NonThink),
            ds4_gen_start_contract_version(),
        );

        // Apply mask to the prefill-returned logits before the first
        // sample (matcher is in `Out` here so this is a no-op, but the
        // codepath stays uniform).
        let mut first_logits = last_logits;
        if grammar_active && !matcher.is_free() {
            matcher.token_mask(&decoded_vocab, &mut grammar_mask);
            deepseek4::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut first_logits);
        }
        let mut next_tok: u32 =
            deepseek4::sampling::sample_token(&first_logits, temp, top_k, top_p, &mut rng);
        let mut pos = pos_after_prefill;
        // Token-cache capture for the prefix-cache replay path. We
        // mirror the parser events into local accumulators so that —
        // after decode completes — we can fingerprint the just-emitted
        // assistant turn by (content_text, tool_calls) and store the
        // exact token IDs that the model emitted at
        // `conversation_tokens[decode_start..decode_end]`.
        //
        // Why mirror rather than re-parse: the streamed events from
        // `parser.feed` carry the parser's reconstructed structure
        // (Reasoning fragments split off from Token, ToolCalls
        // assembled from `<｜DSML｜tool_calls>` blocks). Replaying that
        // here once captures all the logical structure without a
        // second tokenizer pass.
        let decode_start_tokens_idx = m.conversation_tokens.len();
        let mut emit_text_buf = String::new();
        let mut emit_tool_calls_buf: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
        let mut dsml_malformed: Option<String> = None;
        use hipfire_arch_deepseek4::dsml::StreamEvent;
        // Reasoning is not fingerprinted (history re-opens think from think_mode).
        // ToolCalls are buffered turn-wide and released only on a tool-safe terminal.
        let mut absorb_event = |ev: &StreamEvent| {
            ds4_absorb_stream_event(
                ev,
                &mut emit_text_buf,
                &mut emit_tool_calls_buf,
                &mut dsml_malformed,
            );
        };

        while generated_count < max_tokens && next_tok != eos_tok {
            if check_abort(id) {
                ds4_ar_client_abort(
                    stdout,
                    id,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.speculator,
                    gpu,
                    state,
                    generated_count,
                );
                return;
            }
            let frag = tokenizer.decode(&[next_tok]);
            for ev in parser.feed(&frag) {
                absorb_event(&ev);
                emit_stream_event(stdout, id, ev);
            }
            emit_committed_event(
                stdout,
                id,
                next_tok,
                generated_count,
                decode_t0.elapsed().as_millis() as u64,
            );
            let _ = stdout.flush();
            m.conversation_tokens.push(next_tok);
            if grammar_active {
                matcher.advance(&frag);
            }
            generated_count += 1;
            match deepseek4::forward::decode_step_with_graph(
                cfg, weights, state, gpu, next_tok, pos,
            ) {
                Ok(mut logits) => {
                    if grammar_active && !matcher.is_free() {
                        matcher.token_mask(&decoded_vocab, &mut grammar_mask);
                        deepseek4::grammar::Matcher::apply_mask_to_logits(
                            &grammar_mask,
                            &mut logits,
                        );
                    }
                    next_tok =
                        deepseek4::sampling::sample_token(&logits, temp, top_k, top_p, &mut rng);
                    pos += 1;
                }
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("deepseek4decode failed: {e:?}"));
                    let _ = stdout.flush();
                    return;
                }
            }
        }
        // Flush any buffered partial markers / content.
        for ev in parser.finish() {
            absorb_event(&ev);
            emit_stream_event(stdout, id, ev);
        }
        let _ = stdout.flush();

        // Cache the just-emitted token sequence under its (content,
        // tool_calls) fingerprint so the next request's V4F history
        // render can replay verbatim and avoid BPE re-encode drift.
        // Trim leading EOS/zero residue defensively (the loop never
        // pushes EOS, but a future model that emits EOS mid-stream
        // shouldn't end up with EOS landing in the cached tokens).
        drop(absorb_event); // release the &mut emit_*_buf borrow
                            // Final mid-turn abort observation before terminal classification.
        if check_abort(id) {
            ds4_ar_client_abort(
                stdout,
                id,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.speculator,
                gpu,
                state,
                generated_count,
            );
            return;
        }
        // Pure turn-wide terminal: later Malformed discards every buffered call.
        let terminal = ds4_ar_ep_finish_route(
            dsml_malformed,
            emit_tool_calls_buf,
            generated_count >= max_tokens,
        );
        let (finish_reason, wire_tool_calls, store_cache) = match &terminal {
            Ds4ArEpRouteTerminal::Malformed(action) => {
                emit_ds4_malformed_action(stdout, id, action);
                return;
            }
            Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            } => (*finish_reason, wire_tool_calls.clone(), *store_cache),
        };
        // Timing + pending done fixed before handshake so commit_ready carries
        // the exact eventual done payload. Abort rolls back + emits one
        // cancellation lifecycle (or fail-closed error if unattested).
        m.seq_pos = state.n_tokens as usize;
        let _ = gpu.hip.device_synchronize();
        let decode_ms = decode_t0.elapsed().as_millis().max(1);
        let total_ms = t0.elapsed().as_millis().max(1);
        let tok_s = if generated_count > 0 && decode_ms > 0 {
            (generated_count as f64 * 1000.0) / decode_ms as f64
        } else {
            0.0
        };
        let prompt_tokens_total = prompt_ids.len();
        let prefill_tokens_actual = suffix_tokens.len();
        let prefill_tok_s = if prefill_ms > 0 {
            prefill_tokens_actual as f64 * 1000.0 / prefill_ms as f64
        } else {
            0.0
        };
        let mut pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": generated_count,
            "tok_s": tok_s,
            "prompt_tokens": prompt_tokens_total,
            "prefill_tokens": prefill_tokens_actual,
            "cached_tokens": cached_tokens,
            "prefill_ms": prefill_ms,
            "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
            "decode_tok_s": (tok_s * 10.0).round() / 10.0,
            "total_ms": total_ms,
            "finish_reason": finish_reason,
            "drafter": "ar",
            "attempt_id": active_attempt_id(),
        });
        // Stage canonical calls before handshake; no post-commit tool_calls event.
        stage_terminal_tool_calls(&mut pending_done, finish_reason, &wire_tool_calls);
        // Two-phase commit before cache / normal done.
        let decision = await_client_terminal_commit(stdout, id, &pending_done);
        let effects = ds4_client_commit_effects(decision, !wire_tool_calls.is_empty(), store_cache);
        if !effects.emit_done {
            ds4_ar_client_abort(
                stdout,
                id,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.speculator,
                gpu,
                state,
                generated_count,
            );
            return;
        }
        // Snapshot count retained for debug visibility only.
        tool_calls_parsed_count = if effects.release_tool_calls {
            wire_tool_calls.len()
        } else {
            0
        };
        // Skip caching when the turn produced no replay-able payload —
        // empty trimmed content AND no tool_calls. Length terminals also
        // suppress cache. Shared via ds4_ar_ep_cache_action / apply.
        let mut action = ds4_ar_ep_cache_action(&terminal, &emit_text_buf);
        if !effects.store_cache {
            action.store = false;
        }
        if action.store
            && generated_count > 0
            && m.conversation_tokens.len() > decode_start_tokens_idx
        {
            let cached_seq: Vec<u32> = m.conversation_tokens[decode_start_tokens_idx..].to_vec();
            if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[asst-cache store] content.len={} tool_calls={} tokens={}",
                    emit_text_buf.len(),
                    wire_tool_calls.len(),
                    cached_seq.len(),
                );
            }
            let _ = ds4_apply_cache_action(
                |fp, seq| {
                    if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        eprintln!("[asst-cache store] fp={:#018x} tokens={}", fp, seq.len());
                    }
                    m.asst_turn_cache.insert(
                        fp,
                        hipfire_runtime::prompt_frame::CachedAssistantTurn {
                            reasoning: None,
                            tools: Vec::new(),
                            content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                                token_ids: seq,
                                text: String::new(),
                            }),
                        },
                    );
                },
                &action,
                cached_seq,
            );
        }
        emit_staged_terminal_done(stdout, &pending_done);
        // Per-request debug summary: this request ran autoregressive (no drafter),
        // e.g. spec disabled or a temp path the loaded drafter can't verify. Making
        // the AR fall-through visible is the point — a "stall" is often just AR.
        let _ = tool_calls_parsed_count;
        eprintln!("[req {id}] drafter=ar tau=1.00 tok/s={tok_s:.1} decode ({generated_count} tok, autoregressive)");
    }
}
pub fn ds4_heterogeneous_client_abort(
    model: &mut hipfire_arch_deepseek4::heterogeneous::DeepseekV4HeterogeneousModel,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    stdout: &mut impl std::io::Write,
    id: &str,
    completion_tokens: usize,
) {
    *seq_pos = 0;
    conversation_tokens.clear();
    let reset = model.reset_for_request_attested();
    match reset {
        Ok(()) => {
            eprintln!(
                "[req {id}] drafter=ar-heterogeneous abort=client rollback=attested post_join=true completion_tokens={completion_tokens}"
            );
            let (aborted, done) =
                ds4_ep_abort_wire_events(id, completion_tokens, active_attempt_id());
            let _ = writeln!(stdout, "{aborted}");
            let _ = writeln!(stdout, "{done}");
            let _ = stdout.flush();
        }
        Err(error) => emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("client cancelled; heterogeneous rollback failed: {error}"),
            "runtime",
            false,
            false,
        ),
    }
}
#[allow(clippy::too_many_arguments)]
pub fn generate_deepseek4_heterogeneous(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    if tools.is_some_and(|items| !items.is_empty()) {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tools are not yet admitted on the DeepSeek V4 heterogeneous G4 route",
            "unsupported",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let Some(tokenizer) = m.tokenizer.as_ref() else {
        emit_error_with_id(stdout, id, "tokenizer not loaded");
        return;
    };
    let eos_tok = match m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_loader::Deepseek4HeterogeneousBundle>()
    }) {
        Some(bundle) => bundle.eos_tok,
        _ => {
            emit_error_with_id(stdout, id, "deepseek4 heterogeneous state missing");
            return;
        }
    };
    let prompt_ids = build_deepseek4_dsml_prompt(
        tokenizer,
        system_prompt,
        None,
        messages_history,
        prompt,
        think_mode,
        eos_tok,
        &mut m.asst_turn_cache,
    );
    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }
    if prompt_ids.len().saturating_add(max_tokens) > m.physical_cap {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "prompt exceeds checkpoint context capacity: prompt={} + max_tokens={} > capacity={}",
                prompt_ids.len(),
                max_tokens,
                m.physical_cap
            ),
        );
        return;
    }

    let total_t0 = Instant::now();
    let prefill_t0 = Instant::now();
    let (mut logits, prefill_ms) = {
        let Some(bundle) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any)
                .downcast_mut::<hipfire_loader::Deepseek4HeterogeneousBundle>()
        }) else {
            emit_error_with_id(stdout, id, "deepseek4 heterogeneous state missing");
            return;
        };
        if let Err(error) = bundle.model.reset_for_request() {
            emit_error_with_id(
                stdout,
                id,
                format!("deepseek4 heterogeneous reset failed: {error}"),
            );
            return;
        }
        let mut logits = Vec::new();
        for (position, &token) in prompt_ids.iter().enumerate() {
            match bundle
                .model
                .decode_step_with_abort(token, position as u32, &|| check_abort(id))
            {
                Ok(Some(next)) => logits = next,
                Ok(None) => {
                    ds4_heterogeneous_client_abort(
                        &mut bundle.model,
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        stdout,
                        id,
                        0,
                    );
                    return;
                }
                Err(error) => {
                    emit_error_with_id(
                        stdout,
                        id,
                        format!("deepseek4 heterogeneous prefill failed: {error}"),
                    );
                    return;
                }
            }
        }
        if let Some(state) = bundle.model.state.as_mut() {
            state.n_tokens = prompt_ids.len() as u64;
        }
        let _ = bundle.model.dense_gpu.hip.device_synchronize();
        (logits, prefill_t0.elapsed().as_millis())
    };

    emit_gen_start(
        stdout,
        id,
        !matches!(think_mode, ThinkMode::NonThink),
        ds4_gen_start_contract_version(),
    );
    let top_k = std::env::var("HIPFIRE_DEEPSEEK4_TOP_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut rng = deepseek4::sampling::Xorshift::new(0x1357_9bdf);
    let mut parser = match think_mode {
        ThinkMode::Low | ThinkMode::High | ThinkMode::Max => {
            deepseek4::dsml::StreamParser::new_in_think()
        }
        ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
    };
    let mut next_tok = deepseek4::sampling::sample_token(&logits, temp, top_k, top_p, &mut rng);
    let decode_t0 = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_ids.len() as u32;
    let mut emitted_tokens = Vec::with_capacity(max_tokens);
    let mut emit_text_buf = String::new();
    let mut emit_tool_calls_buf = Vec::new();
    let mut dsml_malformed = None;

    while generated < max_tokens && next_tok != eos_tok {
        let fragment = tokenizer.decode(&[next_tok]);
        for event in parser.feed(&fragment) {
            ds4_absorb_stream_event(
                &event,
                &mut emit_text_buf,
                &mut emit_tool_calls_buf,
                &mut dsml_malformed,
            );
            emit_stream_event(stdout, id, event);
        }
        emit_committed_event(
            stdout,
            id,
            next_tok,
            generated,
            decode_t0.elapsed().as_millis() as u64,
        );
        let _ = stdout.flush();
        emitted_tokens.push(next_tok);
        generated += 1;
        if generated >= max_tokens {
            break;
        }
        let Some(bundle) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any)
                .downcast_mut::<hipfire_loader::Deepseek4HeterogeneousBundle>()
        }) else {
            emit_error_with_id(stdout, id, "deepseek4 heterogeneous state disappeared");
            return;
        };
        match bundle
            .model
            .decode_step_with_abort(next_tok, pos, &|| check_abort(id))
        {
            Ok(Some(next)) => logits = next,
            Ok(None) => {
                ds4_heterogeneous_client_abort(
                    &mut bundle.model,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    stdout,
                    id,
                    generated,
                );
                return;
            }
            Err(error) => {
                emit_error_with_id(
                    stdout,
                    id,
                    format!("deepseek4 heterogeneous decode failed: {error}"),
                );
                return;
            }
        }
        pos = pos.saturating_add(1);
        if let Some(state) = bundle.model.state.as_mut() {
            state.n_tokens = pos as u64;
        }
        next_tok = deepseek4::sampling::sample_token(&logits, temp, top_k, top_p, &mut rng);
    }
    for event in parser.finish() {
        ds4_absorb_stream_event(
            &event,
            &mut emit_text_buf,
            &mut emit_tool_calls_buf,
            &mut dsml_malformed,
        );
        emit_stream_event(stdout, id, event);
    }
    let terminal =
        ds4_ar_ep_finish_route(dsml_malformed, emit_tool_calls_buf, generated >= max_tokens);
    let (finish_reason, wire_tool_calls) = match terminal {
        Ds4ArEpRouteTerminal::Malformed(action) => {
            emit_ds4_malformed_action(stdout, id, &action);
            return;
        }
        Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            ..
        } => (finish_reason, wire_tool_calls),
    };
    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let tok_s = generated as f64 * 1000.0 / decode_ms as f64;
    let mut pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": tok_s,
        "prompt_tokens": prompt_ids.len(),
        "prefill_tokens": prompt_ids.len(),
        "cached_tokens": 0,
        "prefill_ms": prefill_ms,
        "total_ms": total_t0.elapsed().as_millis().max(1),
        "finish_reason": finish_reason,
        "drafter": "ar-heterogeneous",
        "attempt_id": active_attempt_id(),
    });
    stage_terminal_tool_calls(&mut pending_done, finish_reason, &wire_tool_calls);
    let decision = await_client_terminal_commit(stdout, id, &pending_done);
    let effects = ds4_client_commit_effects(decision, !wire_tool_calls.is_empty(), true);
    if !effects.emit_done {
        let Some(bundle) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any)
                .downcast_mut::<hipfire_loader::Deepseek4HeterogeneousBundle>()
        }) else {
            emit_error_with_id(
                stdout,
                id,
                "deepseek4 heterogeneous state disappeared on abort",
            );
            return;
        };
        ds4_heterogeneous_client_abort(
            &mut bundle.model,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            stdout,
            id,
            generated,
        );
        return;
    }
    m.seq_pos = pos as usize;
    m.conversation_tokens.clear();
    m.conversation_tokens.extend_from_slice(&prompt_ids);
    m.conversation_tokens.extend_from_slice(&emitted_tokens);
    emit_staged_terminal_done(stdout, &pending_done);
    eprintln!(
        "[req {id}] drafter=ar-heterogeneous tau=1.00 tok/s={tok_s:.1} decode ({generated} tok)"
    );
}
/// Gemma 4 dense text (arch_id=13) eager AR path.
///
/// Ported from `origin/feat/gemma4-union` and rehomed onto the beta
/// `ModelState::Gemma4(bundle)` design: reads `bundle.config` /
/// `bundle.weights` / `bundle.state` / `bundle.eos_tok` instead of the PR's
/// `m.gemma4_*` Option fields. Same shape as `generate_lfm2moe` (prefill loop,
/// decode loop, JSONL `token` / `done` events) with four gemma4-specifics:
///
///   1. Prompt build goes through the model's chat template (Jinja), rendered
///      with an EXPLICIT `bos_token: Some("<bos>")` — gemma4's tokenizer
///      decodes `<bos>` to id 2 but the raw-encode fallback doesn't prepend
///      it, so the BOS guard below handles the fallback path.
///   2. Prefill is per-token eager `gemma4::forward::decode_step`; decode is
///      `gemma4::forward::decode_step_with_graph` (hipGraph, default OFF).
///   3. Sampling is host-side from the returned logits vector via
///      `deepseek4::sampling::sample_token`.
///   4. Stop set: `<eos>` (config.eos_token, id 1) UNION `bundle.eos_tok`
///      (loader-resolved `<end_of_turn>` / `<turn|>`) UNION documented 106 —
///      the HF `eos_token_id` list is `[1, 106]`; parsing as a scalar drops
///      106 and decode loops `<turn|>` forever.
///
/// Prompt-cache: arch 13 is intentionally ABSENT from the `cache_capable`
/// allowlist (see load handler) — this path has no LCP prefix-cache block and
/// always cold-prefills the full Jinja render. Enabling cache would corrupt
/// KV slot offsets after turn 1.
///
/// EAGLE spec-decode (arch-22 drafter via `params.drafter`) is wired when
/// `bundle.eagle.is_some()` and `temp <= 1e-6` — greedy-only accept rule,
/// same contract as DFlash. `HIPFIRE_GEMMA4_EAGLE=0` opts out.
#[inline]
fn gemma4_prefill_batch_for_arch(gpu_arch: &str, requested: Option<usize>) -> usize {
    requested
        .unwrap_or(if matches!(gpu_arch, "gfx1100" | "gfx1201") {
            64
        } else {
            1
        })
        .clamp(1, 64)
}

struct Gemma4LogitTraceConfig {
    dir: PathBuf,
    top_k: usize,
    max_steps: usize,
    full_steps: Vec<usize>,
}

fn gemma4_logit_trace_config() -> Option<&'static Gemma4LogitTraceConfig> {
    static CONFIG: OnceLock<Option<Gemma4LogitTraceConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let dir = std::env::var_os("HIPFIRE_GEMMA4_LOGIT_TRACE_DIR")?;
            let top_k = std::env::var("HIPFIRE_GEMMA4_LOGIT_TRACE_TOPK")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16)
                .clamp(1, 128);
            let max_steps = std::env::var("HIPFIRE_GEMMA4_LOGIT_TRACE_MAX_STEPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8192);
            let full_steps = std::env::var("HIPFIRE_GEMMA4_LOGIT_TRACE_FULL_STEPS")
                .unwrap_or_default()
                .split(',')
                .filter_map(|value| value.trim().parse().ok())
                .collect();
            Some(Gemma4LogitTraceConfig {
                dir: PathBuf::from(dir),
                top_k,
                max_steps,
                full_steps,
            })
        })
        .as_ref()
}

fn gemma4_trace_file_stem(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let hash = id.bytes().fold(0xcbf29ce484222325_u64, |state, byte| {
        (state ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{sanitized}-{hash:016x}-attempt-{}", active_attempt_id())
}

fn trace_gemma4_logits(
    gpu: &rdna_compute::Gpu,
    request_id: &str,
    step: usize,
    sampled_token: u32,
    logits: &[f32],
) {
    let Some(config) = gemma4_logit_trace_config() else {
        return;
    };
    if step >= config.max_steps || logits.is_empty() {
        return;
    }

    let mut top: Vec<(usize, f32)> = Vec::with_capacity(config.top_k);
    let non_finite_logits = logits.iter().filter(|value| !value.is_finite()).count();
    for (token, &value) in logits.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let insert_at = top.partition_point(|&(_, current)| current >= value);
        if insert_at < config.top_k {
            top.insert(insert_at, (token, value));
            if top.len() > config.top_k {
                top.pop();
            }
        }
    }
    let margin = match top.as_slice() {
        [(_, first), (_, second), ..] => Some(first - second),
        _ => None,
    };
    let stem = gemma4_trace_file_stem(request_id);
    if let Err(error) = std::fs::create_dir_all(&config.dir) {
        eprintln!("[gemma4-logits] create trace directory failed: {error}");
        return;
    }
    let record = serde_json::json!({
        "schema_version": 1,
        "request_id": request_id,
        "step": step,
        "vocab": logits.len(),
        "non_finite_logits": non_finite_logits,
        "top1_token": top.first().map(|entry| entry.0),
        "top1_top2_margin": margin,
        "sampled_token": sampled_token,
        "top_k": top.iter().map(|&(token, value)| {
            if value.is_finite() {
                serde_json::json!({"token": token, "logit": value})
            } else {
                serde_json::json!({
                    "token": token,
                    "logit_special": if value.is_sign_positive() { "+inf" } else { "-inf" },
                })
            }
        }).collect::<Vec<_>>(),
        "routes": {
            "batched_embedding_prefill": gpu.flags.gemma4_batched_embedding_prefill,
            "ple_branch_batched_prefill": gpu.flags.gemma4_ple_branch_batched_prefill,
            "ple_activation_fused_prefill": gpu.flags.gemma4_ple_activation_fused_prefill,
        },
    });
    let jsonl_path = config.dir.join(format!("{stem}.jsonl"));
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true);
    if step == 0 {
        options.truncate(true);
    } else {
        options.append(true);
    }
    match options.open(&jsonl_path) {
        Ok(mut file) => {
            if let Err(error) = writeln!(file, "{record}") {
                eprintln!(
                    "[gemma4-logits] write {} failed: {error}",
                    jsonl_path.display()
                );
            }
        }
        Err(error) => eprintln!(
            "[gemma4-logits] open {} failed: {error}",
            jsonl_path.display()
        ),
    }

    if config.full_steps.contains(&step) {
        let binary_path = config.dir.join(format!("{stem}.step-{step}.f32le"));
        let temporary_path = binary_path.with_extension("f32le.tmp");
        match std::fs::File::create(&temporary_path) {
            Ok(mut file) => {
                let mut complete = true;
                for value in logits {
                    if let Err(error) = file.write_all(&value.to_le_bytes()) {
                        eprintln!(
                            "[gemma4-logits] write {} failed: {error}",
                            temporary_path.display()
                        );
                        complete = false;
                        break;
                    }
                }
                if complete {
                    if let Err(error) = std::fs::rename(&temporary_path, &binary_path) {
                        eprintln!(
                            "[gemma4-logits] rename {} failed: {error}",
                            binary_path.display()
                        );
                    }
                }
            }
            Err(error) => eprintln!(
                "[gemma4-logits] create {} failed: {error}",
                temporary_path.display()
            ),
        }
    }
}

#[cfg(test)]
mod gemma4_prefill_batch_tests {
    use super::gemma4_prefill_batch_for_arch;

    #[test]
    fn validated_arches_default_to_full_wmma_batch_tile_group() {
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1100", None), 64);
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1201", None), 64);
    }

    #[test]
    fn unvalidated_arches_keep_the_sequential_default() {
        for arch in ["gfx1151", "gfx1200", "gfx942"] {
            assert_eq!(gemma4_prefill_batch_for_arch(arch, None), 1);
        }
    }

    #[test]
    fn explicit_override_wins_and_is_clamped() {
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1100", Some(8)), 8);
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1201", Some(32)), 32);
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1100", Some(0)), 1);
        assert_eq!(gemma4_prefill_batch_for_arch("gfx1100", Some(128)), 64);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_gemma4(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    enable_thinking: bool,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    // `Some(k)` emits OpenAI logprobs with k candidates per token. `None` is the
    // default and produces exactly the envelope this path emitted before.
    logprobs_top_k: Option<usize>,
) {
    // Gemma4 thinking is explicit boolean (default off); max_think_tokens is orthogonal cap.

    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded");
        return;
    }
    // Open the stream contract BEFORE any GPU work or token emission: the CLI's
    // StreamContractGate fail-closes on any event preceding `gen_start`, so
    // without this the first `token` is rejected, the client aborts, and the
    // HTTP handler waits forever. Same fix as DS4 (e99583afa) and lfm2moe.
    let gen_contract = gen_start_contract_version_for_arch(m.arch_id);
    emit_gen_start(stdout, id, false, gen_contract);
    let Some(bundle) = m
        .state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_loader::Gemma4Bundle>())
    else {
        emit_error_with_id(
            stdout,
            id,
            "gemma4 bundle missing on arch_id=13 generate (eager dense only;              EAGLE/lowered not yet wired)",
        );
        return;
    };

    let bos_tok = bundle.config.bos_token;
    let cfg_eos_tok = bundle.config.eos_token;

    // ── Prompt build (same two-path branch as the lfm2moe AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        let mut ids: Vec<u32> = if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking,
                bos_token: Some("<bos>"),
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    reasoning_content: None,
                                    name: None,
                                    rendered_name: None,
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                            synthesized = v;
                            &synthesized
                        }
                    };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => tokenizer.encode(&rendered),
                Err(e) => {
                    eprintln!(
                        "[daemon] jinja render failed in gemma4 path ({e}) —                          falling back to BOS + raw prompt"
                    );
                    tokenizer.encode(prompt)
                }
            }
        } else {
            tokenizer.encode(prompt)
        };
        if ids.first() != Some(&bos_tok) {
            ids.insert(0, bos_tok);
        }
        ids
    };

    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }

    // Stop set (see doc comment item 4). `bundle.eos_tok` already resolved
    // `<end_of_turn>` / `<turn|>` at load; union with config.eos_token
    // (`<eos>`, id 1) and the documented 106 for robustness across exports.
    let stop_set: Vec<u32> = {
        let mut s = vec![cfg_eos_tok, bundle.eos_tok];
        if !s.contains(&106) {
            s.push(106);
        }
        s.dedup();
        s
    };

    // Capacity guard. No eviction on arch_id=13 — reset the KV cursors when
    // the requested run would overflow the physical cache.
    let overflow = { bundle.state.n_tokens + prompt_ids.len() + max_tokens > bundle.state.max_seq };
    if overflow {
        let (n, cap) = (bundle.state.n_tokens, bundle.state.max_seq);
        eprintln!("[daemon] arch_id=13 context full ({n}/{cap}) — resetting Gemma4State");
        bundle.state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }
    // Hard refusal: even from a cold cache the prompt alone must fit (KV
    // writes at pos >= max_seq would be out of bounds).
    let cache_cap = bundle.state.max_seq;
    if prompt_ids.len() >= cache_cap {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "gemma4 prompt is {} tokens but max_seq is {cache_cap}",
                prompt_ids.len()
            ),
        );
        return;
    }

    let t0 = Instant::now();

    // ── Prefill. `forward_batch` preserves the eager result/KV contract while
    // reading projection weights once for a small token block. Keep the path
    // opt-in until long-context parity is certified on each supported board.
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let requested_prefill_batch = std::env::var("HIPFIRE_GEMMA4_PREFILL_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let resolved_prefill_batch =
            gemma4_prefill_batch_for_arch(&gpu.arch, requested_prefill_batch);
        let prefill_batch = if resolved_prefill_batch > 1
            && !gemma4::forward::supports_batched_prefill(&bundle.weights)
        {
            eprintln!(
                "[daemon] Gemma4 batched prefill is unavailable for this weight format; using eager prefill"
            );
            1
        } else {
            resolved_prefill_batch
        };
        let mut offset = 0usize;
        while offset < prompt_ids.len() {
            let width = prefill_batch.min(prompt_ids.len() - offset);
            let start_pos = bundle.state.n_tokens;
            let result = if width == 1 {
                gemma4::forward::decode_step(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    prompt_ids[offset],
                    start_pos as u32,
                )
            } else {
                gemma4::forward::forward_batch(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    &prompt_ids[offset..offset + width],
                    start_pos,
                )
            };
            match result {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("gemma4 prefill failed: {e:?}"));
                    return;
                }
            }
            offset += width;
        }
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();
    let mut gemma_router = GemmaThoughtRouter::new(enable_thinking, max_think_tokens);

    // ── EAGLE spec-decode fast path (arch-22 drafter loaded; greedy only) ──
    //
    // Mirrors the DFlash dispatch contract: the accept rule is greedy-argmax,
    // so the committed stream is PROVABLY the target's greedy AR sequence —
    // the same tokens the AR loop below would emit at temp 0. That invariant
    // is the gate `infer_gemma4_spec --check-eager` validates byte-for-byte
    // (spec == eager on hiptrx/gfx1201, Q8 and MQ4-attn targets, dl ≤ 4).
    // temp > 0 falls through to the AR sampling loop, exactly like DFlash.
    // GATED OFF: EAGLE currently violates its own correctness contract.
    //
    // The accept rule is greedy-argmax, so at temp 0 the committed stream is
    // supposed to be PROVABLY the target's greedy AR sequence — byte-identical
    // to the AR loop below. Measured on gfx1201, 12B-it, temp 0, 48 tokens,
    // same prompt, on TWO different quantizations:
    //
    //   uniform MQ4   AR    52.81 tok/s  "To understand quantum computing, you
    //                                     first have to understand how a normal
    //                                     computer works..."
    //                 EAGLE 44.28 tok/s  "To explain it, we have to look at the
    //                                     \"rules\" of the world we see..."
    //                 tau 1.531, rounds 32 — NOT byte-identical, and 0.84x
    //
    //   Q8            diverges the same way at tau 1.912, 0.73x
    //
    // Divergence begins at the first committed token, which points at the seed
    // hidden / first-round verify rather than at acceptance rate. Reproducing
    // across two quants rules out a quant-specific kernel issue.
    //
    // Until parity is proven it must not run: wrong tokens that look fluent are
    // worse than a refusal, and it is slower anyway. Opt in for debugging with
    // HIPFIRE_GEMMA4_EAGLE=1; HIPFIRE_GEMMA4_EAGLE=0 remains an explicit off.
    //
    // Separately: a K-map-promoted target cannot run this path at all —
    // `proj_gemm_batched` handles only Q8_0 and MQ4G256/HFQ4G256, so the
    // mode-3 `Promote6` on v_proj yields MQ6G256 and the verify fails loud with
    // "dtype MQ6G256 has no batched proj kernel". A uniform (--no-kmap) target
    // avoids that, which is how the numbers above were taken.
    let eagle_active = bundle.eagle.is_some()
        && temp <= 1e-6
        && std::env::var("HIPFIRE_GEMMA4_EAGLE").ok().as_deref() == Some("1");
    if eagle_active {
        let draft_len = bundle.eagle.as_ref().unwrap().draft_len;
        // Seed hidden = post-`model.norm` hidden of the last prompt position
        // (left in `state.tmp` by the final prefill `decode_step` — the
        // lm_head input). The seed TOKEN is the last prompt token; its KV is
        // re-written (identically) at the top of the first verify. The first
        // generated token comes out of round 1's verify (argmax_per_pos[0]),
        // exactly as the AR loop's first sample from `last_logits` would —
        // so `last_logits` is intentionally unused on this path.
        {
            let eagle = bundle.eagle.as_ref().unwrap();
            if let Err(e) = eagle
                .spec_scratch
                .set_seed_hidden_from(gpu, &bundle.state.tmp)
            {
                emit_error_with_id(stdout, id, format!("gemma4 eagle seed hidden: {e}"));
                return;
            }
        }
        let prefill_end = bundle.state.n_tokens;
        let mut seed_token = *prompt_ids.last().unwrap();
        let mut generated_count = 0usize;
        let mut rounds = 0usize;
        let mut total_accepted = 0usize;
        let mut stop = false;
        let mut ttft_ms: Option<f64> = None;
        let decode_t0 = Instant::now();
        while !stop && generated_count < max_tokens {
            let committed_len = bundle.state.n_tokens;
            // KV/seq bound: the verify block occupies [L-1, L-1+draft_len+1).
            if committed_len + draft_len + 1 >= cache_cap {
                break;
            }
            let spec = {
                // Split borrows: config/weights immutably, state+eagle mutably.
                // Use raw pointers to avoid borrow-checker overlap on `bundle`.
                let bundle_ptr = bundle as *mut hipfire_loader::Gemma4Bundle;
                unsafe {
                    let cfg = &(*bundle_ptr).config;
                    let weights = &(*bundle_ptr).weights;
                    let state = &mut (*bundle_ptr).state;
                    let eagle = (*bundle_ptr).eagle.as_mut().unwrap();
                    gemma4::speculative::spec_step_gemma4_eagle(
                        gpu,
                        weights,
                        cfg,
                        state,
                        &eagle.drafter_weights,
                        &eagle.drafter_config,
                        &mut eagle.drafter_scratch,
                        &mut eagle.spec_scratch,
                        seed_token,
                        committed_len,
                        draft_len,
                        0.0,
                    )
                }
            };
            let spec = match spec {
                Ok(s) => s,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("gemma4 eagle spec step failed: {e}"));
                    return;
                }
            };
            rounds += 1;
            total_accepted += spec.accept_len;
            // Emit the committed tokens (accepted drafts ++ bonus); stop at
            // EOS / max_tokens — identical to what the AR loop would commit.
            for &t in &spec.committed {
                if stop_set.contains(&t) {
                    stop = true;
                    break;
                }
                if ttft_ms.is_none() {
                    ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let frag = {
                    let tokenizer = m.tokenizer.as_ref().unwrap();
                    tokenizer.decode(&[t])
                };
                let (emits, _) = gemma_router.push(&frag);
                for ev in emits {
                    match ev {
                        GemmaEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                        GemmaEmit::Token(text) => emit_visible_token(stdout, id, &text),
                    }
                }
                m.conversation_tokens.push(t);
                generated_count += 1;
                if generated_count >= max_tokens {
                    stop = true;
                    break;
                }
            }
            // Next round's seed = this round's bonus; its hidden is already
            // staged in spec_scratch.seed_hidden by spec_step.
            seed_token = spec.next_seed_token;
            // If we stopped on EOS mid-block, spec.n_tokens already counts the
            // committed-but-not-emitted tail — the cursor settle below re-anchors
            // to emitted extent.
            if stop {
                break;
            }
        }

        // ── Cursor settle. `spec_step` leaves n_tokens = L+accept_len+1 with
        // the final bonus's KV slot unwritten, and an EOS mid-block leaves
        // committed-but-not-emitted tokens counted. Re-anchor the cursor to
        // the EMITTED extent and re-forward the last emitted token at its
        // slot so every position < n_tokens carries valid KV — the same
        // invariant the AR loop leaves behind. (KV writes are absolute-
        // position keyed and deterministic, so the re-write is identical to
        // a fresh forward — the same property the verify itself relies on
        // when it re-writes the seed's KV each round.) ──
        if generated_count > 0 {
            let last_tok = *m.conversation_tokens.last().unwrap();
            let last_pos = (prefill_end + generated_count - 1) as u32;
            let bundle_ptr = bundle as *mut hipfire_loader::Gemma4Bundle;
            let settle_res = unsafe {
                let cfg = &(*bundle_ptr).config;
                let weights = &(*bundle_ptr).weights;
                let state = &mut (*bundle_ptr).state;
                gemma4::forward::decode_step(cfg, weights, state, gpu, last_tok, last_pos)
            };
            if let Err(e) = settle_res {
                eprintln!("[daemon] gemma4 eagle cursor settle failed: {e:?}");
                bundle.state.n_tokens = prefill_end + generated_count;
            }
        } else {
            bundle.state.n_tokens = prefill_end;
        }
        m.seq_pos = bundle.state.n_tokens;

        let decode_ms = decode_t0.elapsed().as_millis().max(1);
        let total_ms = t0.elapsed().as_millis().max(1);
        let tok_s = if generated_count > 0 {
            (generated_count as f64 * 1000.0) / decode_ms as f64
        } else {
            0.0
        };
        let prefill_tok_s = prompt_ids.len() as f64 * 1000.0 / prefill_ms.max(1) as f64;
        let ttft_ms = ttft_ms.unwrap_or(total_ms as f64);
        // τ = mean tokens committed per round (accepted drafts + 1 bonus).
        let tau = if rounds > 0 {
            (total_accepted + rounds) as f64 / rounds as f64
        } else {
            0.0
        };
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_tokens":{},"prefill_ms":{},"prefill_tok_s":{:.2},"decode_tok_s":{:.2},"ttft_ms":{:.3},"total_ms":{},"spec":"gemma4_eagle","rounds":{},"tau":{:.3},"draft_len":{},"attempt_id":{}}}"#,
            id,
            generated_count,
            tok_s,
            prompt_ids.len(),
            prefill_ms,
            prefill_tok_s,
            tok_s,
            ttft_ms,
            total_ms,
            rounds,
            tau,
            draft_len,
            active_attempt_id(),
        );
        let _ = stdout.flush();
        return;
    }

    // ── Decode loop. Sample host-side from the running logits vector. ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let mut ttft_ms: Option<f64> = None;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        trace_gemma4_logits(gpu, id, generated_count, next_tok, &last_logits);
        if stop_set.contains(&next_tok) {
            break;
        }

        if ttft_ms.is_none() {
            ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        let (emits, _) = gemma_router.push(&frag);
        for ev in emits {
            match ev {
                GemmaEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                GemmaEmit::Token(text) => {
                    let mut envelope = serde_json::json!({
                        "type": "token",
                        "id": id,
                        "text": text,
                        "attempt_id": active_attempt_id(),
                    });
                    if let Some((lp, top)) = crate::common::token_logprob_fields(
                        &last_logits,
                        next_tok,
                        logprobs_top_k,
                        m.tokenizer.as_ref().unwrap(),
                    ) {
                        envelope["logprob"] = serde_json::json!(lp);
                        envelope["top_logprobs"] = top;
                    }
                    let _ = writeln!(stdout, "{}", envelope);
                    let _ = stdout.flush();
                }
            }
        }
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // KV-capacity guard: the next forward writes KV at slot n_tokens;
        // forwarding at pos >= max_seq would write out of bounds. Stop
        // cleanly here — the just-emitted token is still valid.
        if bundle.state.n_tokens >= cache_cap {
            break;
        }

        let position = bundle.state.n_tokens as u32;
        let step = gemma4::forward::decode_step_with_graph(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            next_tok,
            position,
        );
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("gemma4 decode failed: {e:?}"));
                return;
            }
        }
    }
    for ev in gemma_router.flush() {
        match ev {
            GemmaEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
            GemmaEmit::Token(text) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap(),
                    active_attempt_id()
                );
                let _ = stdout.flush();
            }
        }
    }

    m.seq_pos = bundle.state.n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let prefill_tok_s = prompt_ids.len() as f64 * 1000.0 / prefill_ms.max(1) as f64;
    let ttft_ms = ttft_ms.unwrap_or(total_ms as f64);
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_tokens":{},"prefill_ms":{},"prefill_tok_s":{:.2},"decode_tok_s":{:.2},"ttft_ms":{:.3},"total_ms":{},"attempt_id":{}}}"#,
        id,
        generated_count,
        tok_s,
        prompt_ids.len(),
        prefill_ms,
        prefill_tok_s,
        tok_s,
        ttft_ms,
        total_ms,
        active_attempt_id(),
    );
    let _ = stdout.flush();
}
/// Muse Glimmer dense text (arch_id=14) eager AR path.
///
/// Mirrors `generate_gemma4` shape (prefill loop / decode loop, JSONL
/// `token` / `done` events) with Glimmer specifics:
///
///   1. Prompt build goes through `JinjaChatFrame` when `HIPFIRE_JINJA_CHAT=1`
///      and the model carries a chat_template, falling back to
///      `tokenizer.encode` (BOS-prepended) otherwise.
///   2. `glimmer::forward::decode_step` returns the full logits `Vec<f32>`
///      (the state does NOT stash a greedy next-token), so sampling runs
///      host-side via `deepseek4::sampling::sample_token` on that vector.
///   3. Stop set: config `eos_token` (scalar 200001) UNION any
///      tokenizer-resolved end-of-turn token (e.g. `<end_of_turn>` /
///      `<|im_end|>` if present). Glimmer's `eos_token_id` is a scalar
///      (200001), unlike Gemma4's HF list [1, 106] which required a
///      special-cased 106 fallback — do not copy that workaround.
///   4. No LCP prompt-cache block (same reason as gemma4): this path
///      always cold-prefills the full Jinja render. Arch 14 is therefore
///      intentionally ABSENT from the `cache_capable` allowlist.
///   5. No hipGraph-captured decode path (per spec).
#[allow(clippy::too_many_arguments)]
/// Tap layers whose residual stream feeds the DFlash drafter's `encoder.fc`.
///
/// HF's `output_hidden_states` tuple is 1-indexed with entry 0 being the
/// EMBEDDING output, so a config `target_layer_ids = [1,13,25,37,49]` most
/// likely names `hidden_states[i]` — the output of decoder layers
/// `[0,12,24,36,48]`. We capture after `glimmer_layer_decode(l)`, where `l` is a
/// 0-based decoder index, so the two conventions differ by exactly one.
///
/// This is invisible in every check except the acceptance rate: a one-layer
/// shift still produces well-formed hidden states of the right magnitude, so
/// nothing errors and nothing looks wrong — the drafter simply never agrees.
/// `HIPFIRE_GLIMMER_TAP_LAYERS=0,12,24,36,48` selects the other convention so
/// the two can be compared on hardware instead of argued about.
pub fn glimmer_tap_layers(n_layers: usize) -> Vec<usize> {
    match std::env::var("HIPFIRE_GLIMMER_TAP_LAYERS") {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .filter(|l| *l < n_layers)
            .collect(),
        _ => vec![1, 13, 25, 37, 49],
    }
}

/// Harmony channel state for Muse Glimmer (arch 14).
/// Model generates ` to=<recipient><|message|>...<|eom|>` per channel.
/// `<|eom|>` ending a `self` (reasoning) channel -> keep decoding into answer;
/// ending a `user` (final) channel -> stop. `<|eot|>` / eos always stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerChannel {
    AwaitingHeader,
    Reasoning,
    Answer,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlimmerEmit {
    Reasoning(String),
    Token(String),
    Tool(String),
}

/// Map the request's think budget onto Muse Glimmer's `Reasoning strength:` control.
///
/// The model card names exactly four levels — `low | medium | high | xhigh` — and specifies
/// them as a SYSTEM-PROMPT directive, which the Onyx template renders through its
/// `reasoning_strength` context variable (defaulting to `high` when unset). It is the model's
/// only supported effort dial: there is no non-thinking mode.
///
/// `max_think_tokens` is the engine-side CAP (`1` = the caller asked for no thinking, `0` =
/// uncapped). The cap alone cannot stop the model from opening a self channel — it can only
/// force-close one mid-flight, which wastes the tokens it already spent. Asking for a lower
/// strength up front is what actually shortens the reasoning, so the two are wired together.
pub fn glimmer_reasoning_strength(
    reasoning_effort: Option<&str>,
    max_think_tokens: usize,
) -> &'static str {
    let _ = max_think_tokens;
    match reasoning_effort {
        Some(s) if s.eq_ignore_ascii_case("low") => "low",
        Some(s) if s.eq_ignore_ascii_case("medium") || s.eq_ignore_ascii_case("med") => "medium",
        Some(s) if s.eq_ignore_ascii_case("high") => "high",
        Some(s) if s.eq_ignore_ascii_case("xhigh") || s.eq_ignore_ascii_case("max") => "xhigh",
        _ => "high",
    }
}

pub struct GlimmerHarmonyRouter {
    pub state: GlimmerChannel,
    pub pending: String,
    pub reasoning_tokens: usize,
    pub max_think_tokens: usize,
    pub just_forced: bool,
}

impl GlimmerHarmonyRouter {
    pub fn new(max_think_tokens: usize) -> Self {
        Self {
            state: GlimmerChannel::AwaitingHeader,
            pending: String::new(),
            reasoning_tokens: 0,
            max_think_tokens,
            just_forced: false,
        }
    }

    pub fn thinking_enabled(&self) -> bool {
        self.max_think_tokens != 1
    }
    pub fn just_forced(&self) -> bool {
        self.just_forced
    }
    pub fn push(&mut self, frag: &str) -> (Vec<GlimmerEmit>, bool) {
        if frag.is_empty() {
            return (Vec::new(), false);
        }
        self.pending.push_str(frag);
        let mut out = Vec::new();
        let mut should_stop = false;
        let entered_as_reasoning = self.state == GlimmerChannel::Reasoning;
        loop {
            match self.state {
                GlimmerChannel::AwaitingHeader => {
                    if let Some(at) = self.pending.find("<|message|>") {
                        let header_end = at + "<|message|>".len();
                        let header_str = self.pending[..header_end].to_string();
                        let recipient = if let Some(to_pos) = header_str.rfind(" to=") {
                            header_str[to_pos + 4..at].trim().to_string()
                        } else {
                            "user".to_string()
                        };
                        // `to=self` is ALWAYS the reasoning channel. Muse Glimmer has no
                        // "no-thinking" mode: the Onyx template unconditionally emits
                        // `Reasoning strength: <low|medium|high|xhigh>.` into the system block,
                        // so the model always opens a self channel. Effort is controlled by the
                        // strength (see `glimmer_reasoning_strength`), and `max_think_tokens`
                        // is a CAP that force-closes an over-long span — neither turns the
                        // channel off.
                        //
                        // Classifying `to=self` as Answer when thinking was "off" published the
                        // model's private reasoning as visible content AND made the turn
                        // unreproducible by the template, so the recorder refused it and the
                        // prefix cache could never store a thinking-off turn.
                        let new_state = if recipient == "self" {
                            GlimmerChannel::Reasoning
                        } else if recipient == "user" {
                            GlimmerChannel::Answer
                        } else {
                            GlimmerChannel::Tool
                        };
                        self.pending.drain(..header_end);
                        self.state = new_state;
                        continue;
                    } else {
                        let hold = glimmer_longest_marker_suffix(&self.pending);
                        if hold == 0 || hold >= self.pending.len() {
                            break;
                        }
                        break;
                    }
                }
                GlimmerChannel::Reasoning | GlimmerChannel::Answer | GlimmerChannel::Tool => {
                    let eom = self.pending.find("<|eom|>");
                    let eot = self.pending.find("<|eot|>");
                    let eos = self.pending.find("<|end_of_text|>");
                    let mut best: Option<(usize, &'static str)> = None;
                    for (pos, name) in
                        [(eom, "<|eom|>"), (eot, "<|eot|>"), (eos, "<|end_of_text|>")]
                    {
                        if let Some(p) = pos {
                            if best.map_or(true, |(bp, _)| p < bp) {
                                best = Some((p, name));
                            }
                        }
                    }
                    if let Some((at, marker)) = best {
                        if at > 0 {
                            let text = self.pending[..at].to_string();
                            self.pending.drain(..at);
                            if !text.is_empty() {
                                let ev = if self.state == GlimmerChannel::Reasoning {
                                    GlimmerEmit::Reasoning(text)
                                } else if self.state == GlimmerChannel::Tool {
                                    GlimmerEmit::Tool(text)
                                } else {
                                    GlimmerEmit::Token(text)
                                };
                                out.push(ev);
                            }
                            continue;
                        }
                        self.pending.drain(..marker.len());
                        if marker == "<|eom|>" {
                            if self.state == GlimmerChannel::Reasoning {
                                self.state = GlimmerChannel::AwaitingHeader;
                                continue;
                            } else if self.state == GlimmerChannel::Tool {
                                self.state = GlimmerChannel::AwaitingHeader;
                                continue;
                            } else if self.just_forced {
                                // This eom is the stale reasoning terminator after a forced close
                                self.just_forced = false;
                                self.state = GlimmerChannel::AwaitingHeader;
                                continue;
                            } else {
                                should_stop = true;
                                break;
                            }
                        } else {
                            should_stop = true;
                            break;
                        }
                    } else {
                        let hold = glimmer_longest_marker_suffix(&self.pending);
                        let emit_len = self.pending.len().saturating_sub(hold);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_string();
                            self.pending.drain(..emit_len);
                            if !text.is_empty() {
                                let ev = if self.state == GlimmerChannel::Reasoning {
                                    GlimmerEmit::Reasoning(text)
                                } else if self.state == GlimmerChannel::Tool {
                                    GlimmerEmit::Tool(text)
                                } else {
                                    GlimmerEmit::Token(text)
                                };
                                out.push(ev);
                            }
                        }
                        break;
                    }
                }
            }
        }
        let emitted_reasoning = out.iter().any(|e| matches!(e, GlimmerEmit::Reasoning(_)));
        let should_count = entered_as_reasoning || emitted_reasoning;
        if should_count
            && self.max_think_tokens != 0
            && self.max_think_tokens != 1
            && self.reasoning_tokens < self.max_think_tokens
        {
            self.reasoning_tokens += 1;
            if self.reasoning_tokens >= self.max_think_tokens
                && self.state == GlimmerChannel::Reasoning
            {
                if !self.pending.is_empty() {
                    let tail = std::mem::take(&mut self.pending);
                    if !is_marker_prefix(&tail) && !tail.is_empty() {
                        out.push(GlimmerEmit::Reasoning(tail));
                    }
                }
                self.state = GlimmerChannel::Answer;
                self.just_forced = true;
            }
        }
        (out, should_stop)
    }
    pub fn flush(&mut self) -> Vec<GlimmerEmit> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        if self.state == GlimmerChannel::AwaitingHeader {
            self.pending.clear();
            return Vec::new();
        }
        let text = std::mem::take(&mut self.pending);
        if is_marker_prefix(&text) {
            return Vec::new();
        }
        vec![if self.state == GlimmerChannel::Reasoning {
            GlimmerEmit::Reasoning(text)
        } else if self.state == GlimmerChannel::Tool {
            GlimmerEmit::Tool(text)
        } else {
            GlimmerEmit::Token(text)
        }]
    }
}

/// Gemma thought-channel router (minimal, follows Glimmer patterns and checked-in Gemma template).
///
/// Gemma4 template uses `<|channel>thought\n` ... `\n<channel|>` to wrap reasoning.
/// When `enable_thinking` is false the Jinja prompt already emits a closed empty
/// `thought` channel (`<|channel>thought\n<channel|>`), so the router starts in
/// Answer mode. When true it starts awaiting the thought opening. Content inside
/// the thought channel is emitted as semantic `reasoning` events;
/// content outside is visible answer tokens. `max_think_tokens` is an orthogonal
/// force-close cap (0 = uncapped) that, when reached, flushes pending reasoning
/// and transitions to Answer, mirroring Glimmer's `max_think_tokens` handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaChannel {
    AwaitingThought,
    Reasoning,
    Answer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmaEmit {
    Reasoning(String),
    Token(String),
}

pub struct GemmaThoughtRouter {
    pub state: GemmaChannel,
    pub pending: String,
    pub reasoning_tokens: usize,
    pub max_think_tokens: usize,
    pub enable_thinking: bool,
    pub just_forced: bool,
}

impl GemmaThoughtRouter {
    pub fn new(enable_thinking: bool, max_think_tokens: usize) -> Self {
        let state = if enable_thinking {
            GemmaChannel::AwaitingThought
        } else {
            GemmaChannel::Answer
        };
        Self {
            state,
            pending: String::new(),
            reasoning_tokens: 0,
            max_think_tokens,
            enable_thinking,
            just_forced: false,
        }
    }

    pub fn push(&mut self, frag: &str) -> (Vec<GemmaEmit>, bool) {
        if frag.is_empty() {
            return (Vec::new(), false);
        }
        self.pending.push_str(frag);
        let mut out = Vec::new();
        let entered_as_reasoning = self.state == GemmaChannel::Reasoning;
        loop {
            match self.state {
                GemmaChannel::AwaitingThought => {
                    if let Some(pos) = self.pending.find("<|channel>thought") {
                        let mut header_end = pos + "<|channel>thought".len();
                        if self.pending[header_end..].starts_with('\n') {
                            header_end += 1;
                        }
                        if pos > 0 {
                            let pre = self.pending[..pos].to_string();
                            self.pending.drain(..pos);
                            header_end -= pos;
                            if !pre.is_empty() && !gemma_is_marker_prefix(&pre) {
                                out.push(GemmaEmit::Token(pre));
                            }
                            continue;
                        }
                        self.pending.drain(..header_end);
                        self.state = GemmaChannel::Reasoning;
                        continue;
                    } else {
                        let hold = gemma_longest_marker_suffix(&self.pending);
                        if hold == 0 || hold >= self.pending.len() {
                            break;
                        }
                        let emit_len = self.pending.len() - hold;
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_string();
                            self.pending.drain(..emit_len);
                            if !text.is_empty() && !gemma_is_marker_prefix(&text) {
                                out.push(GemmaEmit::Token(text));
                            }
                        }
                        break;
                    }
                }
                GemmaChannel::Reasoning => {
                    if let Some(pos) = self.pending.find("<channel|>") {
                        if pos > 0 {
                            let text = self.pending[..pos].to_string();
                            self.pending.drain(..pos);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Reasoning(text));
                            }
                            continue;
                        }
                        let end = "<channel|>".len();
                        self.pending.drain(..end);
                        if self.pending.starts_with('\n') {
                            self.pending.drain(..1);
                        }
                        self.state = GemmaChannel::Answer;
                        continue;
                    } else {
                        let hold = gemma_longest_marker_suffix(&self.pending);
                        let emit_len = self.pending.len().saturating_sub(hold);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_string();
                            self.pending.drain(..emit_len);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Reasoning(text));
                            }
                        }
                        break;
                    }
                }
                GemmaChannel::Answer => {
                    if let Some(pos) = self.pending.find("<turn|>") {
                        if pos > 0 {
                            let text = self.pending[..pos].to_string();
                            self.pending.drain(..pos);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Token(text));
                            }
                            continue;
                        }
                        self.pending.drain(.."<turn|>".len());
                        if !self.pending.is_empty() && !gemma_is_marker_prefix(&self.pending) {
                            let tail = std::mem::take(&mut self.pending);
                            out.push(GemmaEmit::Token(tail));
                        }
                        break;
                    }
                    let hold = gemma_longest_marker_suffix(&self.pending);
                    let emit_len = self.pending.len().saturating_sub(hold);
                    if emit_len > 0 {
                        let text = self.pending[..emit_len].to_string();
                        self.pending.drain(..emit_len);
                        if !text.is_empty() {
                            out.push(GemmaEmit::Token(text));
                        }
                    }
                    break;
                }
            }
        }
        let emitted_reasoning = out.iter().any(|e| matches!(e, GemmaEmit::Reasoning(_)));
        let should_count = entered_as_reasoning || emitted_reasoning;
        if should_count
            && self.max_think_tokens != 0
            && self.reasoning_tokens < self.max_think_tokens
        {
            self.reasoning_tokens += 1;
            if self.reasoning_tokens >= self.max_think_tokens
                && self.state == GemmaChannel::Reasoning
            {
                if !self.pending.is_empty() {
                    let tail = std::mem::take(&mut self.pending);
                    if !gemma_is_marker_prefix(&tail) && !tail.is_empty() {
                        out.push(GemmaEmit::Reasoning(tail));
                    }
                }
                self.state = GemmaChannel::Answer;
                self.just_forced = true;
            }
        }
        (out, false)
    }

    pub fn flush(&mut self) -> Vec<GemmaEmit> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        if self.state == GemmaChannel::AwaitingThought {
            self.pending.clear();
            return Vec::new();
        }
        let text = std::mem::take(&mut self.pending);
        if gemma_is_marker_prefix(&text) {
            return Vec::new();
        }
        vec![if self.state == GemmaChannel::Reasoning {
            GemmaEmit::Reasoning(text)
        } else {
            GemmaEmit::Token(text)
        }]
    }
}

pub fn gemma_is_marker_prefix(s: &str) -> bool {
    const MARKERS: &[&str] = &[
        "<|channel>thought",
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
    ];
    MARKERS.iter().any(|m| m.starts_with(s) || s.starts_with(m))
}

pub fn gemma_longest_marker_suffix(s: &str) -> usize {
    const MARKERS: &[&str] = &[
        "<|channel>thought",
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
    ];
    for len in (1..=s.len()).rev() {
        if !s.is_char_boundary(s.len() - len) {
            continue;
        }
        let suffix = &s[s.len() - len..];
        if MARKERS.iter().any(|m| m.starts_with(suffix)) {
            return len;
        }
    }
    0
}

pub fn is_marker_prefix(s: &str) -> bool {
    const MARKERS: &[&str] = &[
        "<|start|>",
        "<|message|>",
        "<|eom|>",
        "<|eot|>",
        "<|end_of_text|>",
    ];
    MARKERS.iter().any(|m| m.starts_with(s) || s.starts_with(m))
}

/// Length in BYTES of the longest suffix of `s` that could be the start of a
/// Harmony marker, so a marker split across two decode chunks is not emitted
/// as visible text.
///
/// Must respect char boundaries: an earlier version sliced `&s[s.len()-len..]`
/// for every len and panicked with "byte index N is not a char boundary" the
/// first time the model emitted a multibyte character — `×` in an arithmetic
/// reasoning span was enough to kill the daemon. Markers are pure ASCII, so a
/// boundary that falls inside a multibyte char can never begin one; skip it.
pub fn glimmer_longest_marker_suffix(s: &str) -> usize {
    const MARKERS: &[&str] = &[
        "<|start|>",
        "<|message|>",
        "<|eom|>",
        "<|eot|>",
        "<|end_of_text|>",
    ];
    let max = MARKERS.iter().map(|m| m.len()).max().unwrap_or(0);
    let upper = s.len().min(max.saturating_sub(1));
    for len in (1..=upper).rev() {
        let start = s.len() - len;
        if !s.is_char_boundary(start) {
            continue;
        }
        let suffix = &s[start..];
        if MARKERS.iter().any(|m| m.starts_with(suffix)) {
            return len;
        }
    }
    0
}

pub const GLIMMER_START_ID: u32 = 200022;
pub const GLIMMER_MESSAGE_ID: u32 = 200023;
pub const GLIMMER_EOM_ID: u32 = 200007;
pub const GLIMMER_EOT_ID: u32 = 200008;
pub const GLIMMER_END_OF_TEXT_ID: u32 = 200001;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlimmerRecordedRecipient {
    Self_,
    User,
    Tool(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlimmerOpenBody {
    pub recipient: GlimmerRecordedRecipient,
    pub token_ids: Vec<u32>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlimmerRecordRefusal {
    UnexpectedMarker,
    NonCanonicalHeader,
    ForcedReasoningClose,
    NonCanonicalBodyOrder,
    NonCanonicalTerminator,
    Incomplete,
    ToolCallMismatch,
    EmptySelfBody,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlimmerRecorderState {
    Header { first: bool, decoded: String },
    AwaitingStart,
    Body(GlimmerOpenBody),
    Terminal,
    Refused(GlimmerRecordRefusal),
}

pub struct GlimmerChannelRecorder {
    pub state: GlimmerRecorderState,
    pub reasoning: Option<hipfire_runtime::prompt_frame::CachedAssistantBody>,
    pub tools: Vec<hipfire_runtime::prompt_frame::CachedAssistantToolBody>,
    pub content: Option<hipfire_runtime::prompt_frame::CachedAssistantBody>,
}

impl GlimmerChannelRecorder {
    pub fn new() -> Self {
        Self {
            state: GlimmerRecorderState::Header {
                first: true,
                decoded: String::new(),
            },
            reasoning: None,
            tools: Vec::new(),
            content: None,
        }
    }
    pub fn push(&mut self, token_id: u32, decoded_frag: &str) {
        if matches!(
            self.state,
            GlimmerRecorderState::Refused(_) | GlimmerRecorderState::Terminal
        ) {
            return;
        }
        match token_id {
            GLIMMER_START_ID => {
                match &mut self.state {
                    GlimmerRecorderState::AwaitingStart => {
                        self.state = GlimmerRecorderState::Header {
                            first: false,
                            decoded: String::new(),
                        };
                    }
                    GlimmerRecorderState::Header { first, decoded } => {
                        if *first {
                            decoded.push_str(decoded_frag);
                        } else {
                            self.state = GlimmerRecorderState::Refused(
                                GlimmerRecordRefusal::UnexpectedMarker,
                            );
                        }
                    }
                    GlimmerRecorderState::Body(_) => {
                        self.state =
                            GlimmerRecorderState::Refused(GlimmerRecordRefusal::UnexpectedMarker);
                    }
                    _ => {}
                }
                return;
            }
            GLIMMER_MESSAGE_ID => {
                let (first, decoded) = match &self.state {
                    GlimmerRecorderState::Header { first, decoded } => (*first, decoded.clone()),
                    _ => {
                        self.state =
                            GlimmerRecorderState::Refused(GlimmerRecordRefusal::UnexpectedMarker);
                        return;
                    }
                };
                // The FIRST header continues the prompt: `add_generation_prompt` already
                // emitted `<|start|>assistant`, so the model's own first emission is just
                // ` to=self` / ` to=user` and the recorder never sees the word `assistant`.
                // Every LATER header is emitted in full by the model
                // (`<|start|>assistant to=user<|message|>`) and must carry it.
                //
                // Requiring it unconditionally refused EVERY turn with `NonCanonicalHeader`,
                // which silently disabled the whole verbatim-splice cache in production while
                // the unit tests passed — they fed `"assistant to=self"` as the first header,
                // a string the real model never emits.
                if !first && !decoded.contains("assistant") {
                    self.state =
                        GlimmerRecorderState::Refused(GlimmerRecordRefusal::NonCanonicalHeader);
                    return;
                }
                let recipient_str = if let Some(to_pos) = decoded.rfind(" to=") {
                    decoded[to_pos + 4..].trim().to_string()
                } else {
                    "user".to_string()
                };
                let recipient = if recipient_str == "self" {
                    // No thinking-disabled refusal: `to=self` is always the reasoning channel
                    // (see `GlimmerHarmonyRouter`), so a self body is always reproducible by the
                    // template's `reasoning_content` envelope regardless of the think budget.
                    if self.reasoning.is_some() {
                        self.state = GlimmerRecorderState::Refused(
                            GlimmerRecordRefusal::NonCanonicalBodyOrder,
                        );
                        return;
                    }
                    GlimmerRecordedRecipient::Self_
                } else if recipient_str == "user" {
                    if self.content.is_some() || !self.tools.is_empty() {
                        self.state = GlimmerRecorderState::Refused(
                            GlimmerRecordRefusal::NonCanonicalBodyOrder,
                        );
                        return;
                    }
                    GlimmerRecordedRecipient::User
                } else if !recipient_str.is_empty() {
                    if self.content.is_some() {
                        self.state = GlimmerRecorderState::Refused(
                            GlimmerRecordRefusal::NonCanonicalBodyOrder,
                        );
                        return;
                    }
                    GlimmerRecordedRecipient::Tool(recipient_str)
                } else {
                    self.state =
                        GlimmerRecorderState::Refused(GlimmerRecordRefusal::NonCanonicalHeader);
                    return;
                };
                self.state = GlimmerRecorderState::Body(GlimmerOpenBody {
                    recipient,
                    token_ids: Vec::new(),
                    text: String::new(),
                });
                return;
            }
            GLIMMER_EOM_ID => {
                let body =
                    match std::mem::replace(&mut self.state, GlimmerRecorderState::AwaitingStart) {
                        GlimmerRecorderState::Body(b) => b,
                        _ => {
                            self.state = GlimmerRecorderState::Refused(
                                GlimmerRecordRefusal::UnexpectedMarker,
                            );
                            return;
                        }
                    };
                match body.recipient {
                    GlimmerRecordedRecipient::Self_ => {
                        if body.text.is_empty() && body.token_ids.is_empty() {
                            self.state =
                                GlimmerRecorderState::Refused(GlimmerRecordRefusal::EmptySelfBody);
                            return;
                        }
                        if self.reasoning.is_some() {
                            self.state = GlimmerRecorderState::Refused(
                                GlimmerRecordRefusal::NonCanonicalBodyOrder,
                            );
                            return;
                        }
                        self.reasoning = Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                            token_ids: body.token_ids,
                            text: body.text,
                        });
                        self.state = GlimmerRecorderState::AwaitingStart;
                    }
                    GlimmerRecordedRecipient::Tool(recipient) => {
                        self.tools
                            .push(hipfire_runtime::prompt_frame::CachedAssistantToolBody {
                                recipient,
                                token_ids: body.token_ids,
                            });
                        self.state = GlimmerRecorderState::AwaitingStart;
                    }
                    GlimmerRecordedRecipient::User => {
                        self.state = GlimmerRecorderState::Refused(
                            GlimmerRecordRefusal::NonCanonicalTerminator,
                        );
                        return;
                    }
                }
                return;
            }
            GLIMMER_EOT_ID => {
                let body = match std::mem::replace(&mut self.state, GlimmerRecorderState::Terminal)
                {
                    GlimmerRecorderState::Body(b) => b,
                    _ => {
                        self.state =
                            GlimmerRecorderState::Refused(GlimmerRecordRefusal::UnexpectedMarker);
                        return;
                    }
                };
                match body.recipient {
                    GlimmerRecordedRecipient::User => {
                        if self.content.is_some() {
                            self.state = GlimmerRecorderState::Refused(
                                GlimmerRecordRefusal::NonCanonicalBodyOrder,
                            );
                            return;
                        }
                        self.content = Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                            token_ids: body.token_ids,
                            text: body.text,
                        });
                        self.state = GlimmerRecorderState::Terminal;
                    }
                    GlimmerRecordedRecipient::Tool(recipient) => {
                        self.tools
                            .push(hipfire_runtime::prompt_frame::CachedAssistantToolBody {
                                recipient,
                                token_ids: body.token_ids,
                            });
                        self.state = GlimmerRecorderState::Terminal;
                    }
                    GlimmerRecordedRecipient::Self_ => {
                        self.state = GlimmerRecorderState::Refused(
                            GlimmerRecordRefusal::NonCanonicalTerminator,
                        );
                        return;
                    }
                }
                return;
            }
            GLIMMER_END_OF_TEXT_ID => {
                self.state = GlimmerRecorderState::Refused(GlimmerRecordRefusal::UnexpectedMarker);
                return;
            }
            _ => {}
        }
        match &mut self.state {
            GlimmerRecorderState::Body(body) => {
                body.token_ids.push(token_id);
                body.text.push_str(decoded_frag);
            }
            GlimmerRecorderState::Header { decoded, .. } => {
                decoded.push_str(decoded_frag);
            }
            GlimmerRecorderState::AwaitingStart => {
                self.state = GlimmerRecorderState::Refused(GlimmerRecordRefusal::UnexpectedMarker);
            }
            _ => {}
        }
    }
    pub fn mark_forced_reasoning_close(&mut self) {
        if matches!(self.state, GlimmerRecorderState::Refused(_)) {
            return;
        }
        self.state = GlimmerRecorderState::Refused(GlimmerRecordRefusal::ForcedReasoningClose);
    }
    pub fn into_cached_turn(
        mut self,
        parsed_tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
    ) -> Result<hipfire_runtime::prompt_frame::CachedAssistantTurn, GlimmerRecordRefusal> {
        if let GlimmerRecorderState::Refused(r) = self.state {
            return Err(r);
        }
        match std::mem::replace(&mut self.state, GlimmerRecorderState::Terminal) {
            GlimmerRecorderState::Body(body) => match body.recipient {
                GlimmerRecordedRecipient::Self_ => return Err(GlimmerRecordRefusal::Incomplete),
                GlimmerRecordedRecipient::Tool(recipient) => {
                    self.tools
                        .push(hipfire_runtime::prompt_frame::CachedAssistantToolBody {
                            recipient,
                            token_ids: body.token_ids,
                        });
                }
                GlimmerRecordedRecipient::User => {
                    if self.content.is_some() {
                        return Err(GlimmerRecordRefusal::NonCanonicalBodyOrder);
                    }
                    self.content = Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                        token_ids: body.token_ids,
                        text: body.text,
                    });
                }
            },
            GlimmerRecorderState::Header { .. } | GlimmerRecorderState::AwaitingStart => {
                return Err(GlimmerRecordRefusal::Incomplete);
            }
            GlimmerRecorderState::Terminal => {
                self.state = GlimmerRecorderState::Terminal;
            }
            GlimmerRecorderState::Refused(r) => return Err(r),
        }
        if !self.tools.is_empty() && self.content.is_some() {
            return Err(GlimmerRecordRefusal::NonCanonicalBodyOrder);
        }
        if self.tools.len() != parsed_tool_calls.len() {
            if !(self.tools.is_empty() && parsed_tool_calls.is_empty()) {
                return Err(GlimmerRecordRefusal::ToolCallMismatch);
            }
        }
        for (i, t) in self.tools.iter().enumerate() {
            if i < parsed_tool_calls.len() && t.recipient != parsed_tool_calls[i].name {
                return Err(GlimmerRecordRefusal::ToolCallMismatch);
            }
        }
        Ok(hipfire_runtime::prompt_frame::CachedAssistantTurn {
            reasoning: self.reasoning,
            tools: self.tools,
            content: self.content,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerMirrorAction {
    Aligned,
    TruncateMirror(usize),
    RollbackCursor(usize),
}

pub fn glimmer_mirror_action(mirror_len: usize, n_tokens: usize) -> GlimmerMirrorAction {
    if mirror_len == n_tokens {
        GlimmerMirrorAction::Aligned
    } else if mirror_len > n_tokens {
        GlimmerMirrorAction::TruncateMirror(n_tokens)
    } else {
        GlimmerMirrorAction::RollbackCursor(mirror_len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerMirrorReconcile {
    Aligned,
    Truncated { from: usize, to: usize },
    RolledBack { mirror_len: usize, n_tokens: usize },
    Reset,
}

pub fn reconcile_glimmer_mirror(
    bundle: &mut hipfire_loader::MuseGlimmerBundle,
    conversation_tokens: &mut Vec<u32>,
    seq_pos: &mut usize,
) -> GlimmerMirrorReconcile {
    let mirror_len = conversation_tokens.len();
    let n_tokens = bundle.state.n_tokens;
    let action = glimmer_mirror_action(mirror_len, n_tokens);
    let result = match action {
        GlimmerMirrorAction::Aligned => {
            *seq_pos = n_tokens;
            GlimmerMirrorReconcile::Aligned
        }
        GlimmerMirrorAction::TruncateMirror(to) => {
            // Mirror ahead of target: capture is already correct; truncate mirror only.
            let from = mirror_len;
            conversation_tokens.truncate(to);
            *seq_pos = to;
            GlimmerMirrorReconcile::Truncated { from, to }
        }
        GlimmerMirrorAction::RollbackCursor(new_len) => {
            // Mirror behind target: rewind session (KV + capture + drafter) to mirror.
            match bundle.rewind_session_to(new_len) {
                Ok(()) => {
                    *seq_pos = new_len;
                    GlimmerMirrorReconcile::RolledBack {
                        mirror_len: new_len,
                        n_tokens,
                    }
                }
                Err(_) => {
                    bundle.reset_session_state();
                    conversation_tokens.clear();
                    *seq_pos = 0;
                    GlimmerMirrorReconcile::Reset
                }
            }
        }
    };
    debug_assert_eq!(conversation_tokens.len(), bundle.state.n_tokens);
    debug_assert_eq!(*seq_pos, bundle.state.n_tokens);
    result
}

pub fn normalize_glimmer_tool_arguments(
    value: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match value {
        serde_json::Value::Object(map) => Ok(serde_json::Value::Object(map.clone())),
        serde_json::Value::Null => Ok(serde_json::Value::Object(Default::default())),
        serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(serde_json::Value::Object(map)) => Ok(serde_json::Value::Object(map)),
            Ok(_) => Err(format!(
                "tool arguments string must parse to object, got: {}",
                s
            )),
            Err(e) => Err(format!("tool arguments string invalid JSON: {}: {}", s, e)),
        },
        _ => Err(format!("tool arguments must be object, got: {}", value)),
    }
}

pub fn prepare_glimmer_onyx_history(
    messages: &[hipfire_runtime::prompt_frame::Message],
) -> Result<Vec<hipfire_runtime::prompt_frame::Message>, String> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_tool_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (idx, msg) in messages.iter().enumerate() {
        let mut cloned = msg.clone();
        if cloned.role == hipfire_runtime::prompt_frame::Role::Assistant
            && !cloned.tool_calls.is_empty()
        {
            pending_tool_map.clear();
            for tc in &mut cloned.tool_calls {
                let normalized = normalize_glimmer_tool_arguments(&tc.arguments)?;
                tc.arguments = normalized;
                tc.rendered_body = None;
                if let Some(id) = &tc.id {
                    if !id.is_empty() {
                        if pending_tool_map.contains_key(id) {
                            return Err(format!(
                                "duplicate tool_call id {} at assistant turn {}",
                                id, idx
                            ));
                        }
                        pending_tool_map.insert(id.clone(), tc.name.clone());
                    }
                }
            }
        } else if cloned.role == hipfire_runtime::prompt_frame::Role::Tool {
            let tcid = cloned.tool_call_id.clone().unwrap_or_default();
            let mut resolved: Option<String> = None;
            if !tcid.is_empty() && pending_tool_map.contains_key(&tcid) {
                resolved = pending_tool_map.get(&tcid).cloned();
            } else if !tcid.is_empty() {
                for prev in messages[..idx].iter().rev() {
                    if prev.role == hipfire_runtime::prompt_frame::Role::Assistant {
                        for tc in &prev.tool_calls {
                            if tc.id.as_deref() == Some(&tcid) {
                                resolved = Some(tc.name.clone());
                                break;
                            }
                        }
                        if resolved.is_some() {
                            break;
                        }
                    }
                }
            }
            if let Some(name) = resolved {
                cloned.rendered_name = Some(name);
            } else if cloned.name.is_some() {
                cloned.rendered_name = cloned.name.clone();
            } else if !tcid.is_empty() {
                return Err(format!(
                    "unresolved tool_call_id {} at tool turn {}",
                    tcid, idx
                ));
            }
        }
        out.push(cloned);
    }
    Ok(out)
}

pub fn glimmer_store_cached_turn(
    cache: &mut hipfire_loader::AsstTurnCache,
    recorder: GlimmerChannelRecorder,
    parsed_tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
    ordinal: usize,
) -> bool {
    let turn = match recorder.into_cached_turn(parsed_tool_calls) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[glimmer-cache store skip] recorder refusal: {:?}", e);
            return false;
        }
    };
    if let Some(content) = &turn.content {
        if !turn.tools.is_empty() {
            eprintln!("[glimmer-cache store skip] both content and tools present");
            return false;
        }
        let normalized =
            hipfire_runtime::tokenizer::maybe_normalize_prompt(&content.text).into_owned();
        let fp_raw = asst_turn_fingerprint(&normalized, &[]);
        let fp = glimmer_turn_key(fp_raw, ordinal);
        cache.insert(fp, turn);
        true
    } else if !turn.tools.is_empty() {
        let fp_raw = asst_turn_fingerprint("", parsed_tool_calls);
        let fp = glimmer_turn_key(fp_raw, ordinal);
        cache.insert(fp, turn);
        true
    } else {
        eprintln!("[glimmer-cache store skip] empty turn");
        false
    }
}

#[derive(Debug, Clone)]
pub struct GlimmerRouteError {
    pub detail: String,
}

pub fn parse_glimmer_atem(
    body: &str,
) -> Result<Vec<hipfire_runtime::prompt_frame::ToolCall>, GlimmerRouteError> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("<atem:invoke") {
        let after_start = &rest[start..];
        let name_start = after_start.find("name=\"").ok_or(GlimmerRouteError {
            detail: "missing invoke name".into(),
        })? + 6;
        let name_end = after_start[name_start..]
            .find('"')
            .ok_or(GlimmerRouteError {
                detail: "unterminated name".into(),
            })?
            + name_start;
        let name = after_start[name_start..name_end].to_string();
        let invoke_close = after_start
            .find("</atem:invoke>")
            .ok_or(GlimmerRouteError {
                detail: "missing invoke close".into(),
            })?;
        let invoke_body = &after_start[..invoke_close];
        let mut args = serde_json::Map::new();
        let mut param_rest = invoke_body;
        while let Some(p_start) = param_rest.find("<atem:parameter") {
            let param_after = &param_rest[p_start..];
            let p_name_start = param_after.find("name=\"").ok_or(GlimmerRouteError {
                detail: "missing param name".into(),
            })? + 6;
            let p_name_end = param_after[p_name_start..]
                .find('"')
                .ok_or(GlimmerRouteError {
                    detail: "unterminated param name".into(),
                })?
                + p_name_start;
            let p_name = param_after[p_name_start..p_name_end].to_string();
            let tag_end = param_after.find('>').ok_or(GlimmerRouteError {
                detail: "missing param tag end".into(),
            })? + 1;
            let close_tag = "</atem:parameter>";
            let inner_start = tag_end;
            let inner_end = param_after.find(close_tag).ok_or(GlimmerRouteError {
                detail: "missing param close".into(),
            })?;
            let raw = param_after[inner_start..inner_end].trim();
            let value = if raw == "true" {
                serde_json::Value::Bool(true)
            } else if raw == "false" {
                serde_json::Value::Bool(false)
            } else if raw == "null" {
                serde_json::Value::Null
            } else if raw.starts_with("{") || raw.starts_with("[") {
                serde_json::from_str(raw).unwrap_or(serde_json::Value::String(raw.to_string()))
            } else if let Ok(n) = raw.parse::<i64>() {
                serde_json::Value::Number(n.into())
            } else if let Ok(f) = raw.parse::<f64>() {
                serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap())
            } else {
                serde_json::Value::String(raw.to_string())
            };
            args.insert(p_name, value);
            param_rest = &param_after[inner_end + close_tag.len()..];
        }
        calls.push(hipfire_runtime::prompt_frame::ToolCall {
            id: None,
            name,
            arguments: serde_json::Value::Object(args),
            rendered_body: None,
        });
        rest = &rest[start + invoke_close + "</atem:invoke>".len()..];
    }
    Ok(calls)
}

/// Run the engine's two-phase terminal handshake for arch 14 and release the turn.
///
/// The wire contract is `commit_ready` -> client `commit` -> a byte-identical `done`
/// (`crates/hipfire-client/src/lib.rs:742-776`, and `committed_done_payload_mismatch` at
/// `lib.rs:1173-1194` compares the two payloads modulo `type`). Writing `done` straight out
/// skips the handshake: the client treats the unexpected terminal as a protocol error, aborts,
/// and the abort drain then reports
/// `cancellation drain rejected non-aborted done (saw_aborted=false, finish_reason=...)`.
/// That is a hung request from the user's side, so this helper is mandatory on every arch-14
/// terminal.
///
/// Returns `true` when the client committed — only then may the caller publish side effects
/// (assistant-turn cache store). On abort the caller MUST fail-closed reset, because the mirror
/// and KV are mid-turn and no later turn may reuse them.
pub fn glimmer_commit_terminal(
    stdout: &mut std::io::Stdout,
    id: &str,
    pending_done: &serde_json::Value,
    generated: usize,
) -> bool {
    match await_client_terminal_commit(stdout, id, pending_done) {
        ClientTerminalDecision::Commit => {
            emit_staged_terminal_done(stdout, pending_done);
            true
        }
        ClientTerminalDecision::Abort => {
            let attempt_id = active_attempt_id();
            let _ = writeln!(
                stdout,
                "{}",
                hipfire_runtime::semantic::wire_aborted(id, "client_cancelled", attempt_id)
            );
            let _ = writeln!(
                stdout,
                "{}",
                hipfire_runtime::semantic::wire_aborted_done(id, generated, attempt_id)
            );
            let _ = stdout.flush();
            false
        }
    }
}

/// Request-local profitability controller for Muse Glimmer speculative serve.
///
/// Pure integer state machine: no GPU, no allocation, no float in the decision path.
/// Caller feeds only eligible full windows (invalid/terminal/truncated already filtered).
/// First eligible window is an untimed warmup AR B=1; every subsequent disjoint group of
/// four eligible windows requests one measured B=1 probe. Retirement is sticky for this
/// object; the next request constructs a fresh controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerProfitProbeKind {
    None,
    /// Untimed useful B=1 after the first eligible window; excluded from evaluation.
    Warmup,
    /// Timed B=1 after four measured eligible windows.
    Measured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerProfitGuardStatus {
    Disabled,
    Warming,
    Monitoring,
    Retired,
}

/// Frozen policy constants (not knobs).
pub const GLIMMER_PROFIT_WARMUP_WINDOWS: u32 = 1;
pub const GLIMMER_PROFIT_WINDOWS_PER_EVAL: u32 = 4;
pub const GLIMMER_PROFIT_BAD_NUM: u128 = 105; // ratio >= 1.05
pub const GLIMMER_PROFIT_GOOD_NUM: u128 = 98; // ratio <= 0.98
pub const GLIMMER_PROFIT_RATIO_DEN: u128 = 100;
pub const GLIMMER_PROFIT_BAD_TO_RETIRE: u32 = 2;

#[derive(Debug, Clone)]
pub struct GlimmerSpecProfitGuard {
    pub enabled: bool,
    pub retired: bool,
    pub warmup_done: bool,
    /// Eligible full windows observed (warmup + measured), excluding ignored zeros.
    pub eligible_windows: u32,
    /// Windows accumulated in the current post-warmup evaluation group (0..4).
    pub group_windows: u32,
    pub sum_spec_ns: u128,
    pub sum_productive: u128,
    pub pending_probe: GlimmerProfitProbeKind,
    /// Completed measured evaluations.
    pub evaluations: u32,
    pub bad_evaluations: u32,
    /// Last measured evaluation inputs (telemetry only).
    pub last_spec_ns: u128,
    pub last_productive: u128,
    pub last_ar_probe_ns: u128,
    /// Last ratio as milli-units S*1000/(A*P) when defined; else 0. Telemetry only.
    pub last_ratio_milli: u128,
    /// Evaluation index (1-based) at which retirement latched; 0 if not retired.
    pub retire_evaluation: u32,
    /// `eligible_windows` snapshot at retirement; 0 if not retired.
    pub retire_cycle: u32,
}

impl GlimmerSpecProfitGuard {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            retired: false,
            warmup_done: false,
            eligible_windows: 0,
            group_windows: 0,
            sum_spec_ns: 0,
            sum_productive: 0,
            pending_probe: GlimmerProfitProbeKind::None,
            evaluations: 0,
            bad_evaluations: 0,
            last_spec_ns: 0,
            last_productive: 0,
            last_ar_probe_ns: 0,
            last_ratio_milli: 0,
            retire_evaluation: 0,
            retire_cycle: 0,
        }
    }

    pub fn status(&self) -> GlimmerProfitGuardStatus {
        if !self.enabled {
            GlimmerProfitGuardStatus::Disabled
        } else if self.retired {
            GlimmerProfitGuardStatus::Retired
        } else if !self.warmup_done {
            GlimmerProfitGuardStatus::Warming
        } else {
            GlimmerProfitGuardStatus::Monitoring
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_retired(&self) -> bool {
        self.retired
    }

    pub fn evaluations(&self) -> u32 {
        self.evaluations
    }

    pub fn bad_evaluations(&self) -> u32 {
        self.bad_evaluations
    }

    pub fn eligible_windows(&self) -> u32 {
        self.eligible_windows
    }

    pub fn pending_probe(&self) -> GlimmerProfitProbeKind {
        self.pending_probe
    }

    pub fn last_spec_ns(&self) -> u128 {
        self.last_spec_ns
    }

    pub fn last_productive(&self) -> u128 {
        self.last_productive
    }

    pub fn last_ar_probe_ns(&self) -> u128 {
        self.last_ar_probe_ns
    }

    pub fn last_ratio_milli(&self) -> u128 {
        self.last_ratio_milli
    }

    pub fn retire_evaluation(&self) -> u32 {
        self.retire_evaluation
    }

    pub fn retire_cycle(&self) -> u32 {
        self.retire_cycle
    }

    /// Observe one eligible full speculative window.
    ///
    /// `spec_ns` is the synchronized draft+verify(+commit tail) cost; `productive_rows`
    /// is keep_rows (== accepted+1). Zero time or zero rows are ignored and never become
    /// evidence. While a probe is outstanding, further windows are ignored until
    /// `observe_probe` clears the latch.
    pub fn observe_full_window(
        &mut self,
        spec_ns: u128,
        productive_rows: usize,
    ) -> GlimmerProfitProbeKind {
        if !self.enabled || self.retired {
            return GlimmerProfitProbeKind::None;
        }
        if !matches!(self.pending_probe, GlimmerProfitProbeKind::None) {
            return GlimmerProfitProbeKind::None;
        }
        if spec_ns == 0 || productive_rows == 0 {
            return GlimmerProfitProbeKind::None;
        }

        self.eligible_windows = self.eligible_windows.saturating_add(1);

        if !self.warmup_done {
            // First eligible window: request untimed useful AR B=1; exclude from S/P.
            let _ = GLIMMER_PROFIT_WARMUP_WINDOWS; // frozen = 1
            self.warmup_done = true;
            self.pending_probe = GlimmerProfitProbeKind::Warmup;
            return GlimmerProfitProbeKind::Warmup;
        }

        self.sum_spec_ns = self.sum_spec_ns.saturating_add(spec_ns);
        self.sum_productive = self.sum_productive.saturating_add(productive_rows as u128);
        self.group_windows = self.group_windows.saturating_add(1);

        if self.group_windows >= GLIMMER_PROFIT_WINDOWS_PER_EVAL {
            self.pending_probe = GlimmerProfitProbeKind::Measured;
            return GlimmerProfitProbeKind::Measured;
        }
        GlimmerProfitProbeKind::None
    }

    /// Record the AR B=1 probe that followed a prior `observe_full_window` request.
    ///
    /// Warmup probes are discarded (no evaluation). Measured probes compare
    /// `S*100` against `A*P*{105,98}` with saturating/checked integer arithmetic only.
    /// Zero probe time or zero accumulated rows/time is not evidence.
    pub fn observe_probe(&mut self, ar_probe_ns: u128) {
        if !self.enabled || self.retired {
            return;
        }
        match self.pending_probe {
            GlimmerProfitProbeKind::None => {}
            GlimmerProfitProbeKind::Warmup => {
                self.pending_probe = GlimmerProfitProbeKind::None;
            }
            GlimmerProfitProbeKind::Measured => {
                self.pending_probe = GlimmerProfitProbeKind::None;
                let s = self.sum_spec_ns;
                let p = self.sum_productive;
                // Clear the group whether or not this sample is evidence so cadence
                // cannot stall on a zero probe.
                self.sum_spec_ns = 0;
                self.sum_productive = 0;
                self.group_windows = 0;

                if ar_probe_ns == 0 || p == 0 || s == 0 {
                    return;
                }

                self.last_spec_ns = s;
                self.last_productive = p;
                self.last_ar_probe_ns = ar_probe_ns;
                // Telemetry-only milli-ratio; never used for the decision.
                if let Some(den) = ar_probe_ns.checked_mul(p) {
                    if den > 0 {
                        self.last_ratio_milli = s.saturating_mul(1000) / den;
                    }
                }

                self.evaluations = self.evaluations.saturating_add(1);

                // bad iff S*100 >= A*P*105; good iff S*100 <= A*P*98; else deadband.
                let left = match s.checked_mul(GLIMMER_PROFIT_RATIO_DEN) {
                    Some(v) => v,
                    None => u128::MAX, // overflow => treat as extremely unprofitable
                };
                let ap = match ar_probe_ns.checked_mul(p) {
                    Some(v) => v,
                    None => {
                        // Denominator overflow: cannot form a finite ratio safely; skip.
                        return;
                    }
                };
                let bad_right = match ap.checked_mul(GLIMMER_PROFIT_BAD_NUM) {
                    Some(v) => v,
                    None => u128::MAX,
                };
                let good_right = match ap.checked_mul(GLIMMER_PROFIT_GOOD_NUM) {
                    Some(v) => v,
                    None => u128::MAX,
                };

                if left >= bad_right {
                    self.bad_evaluations = self.bad_evaluations.saturating_add(1);
                    if self.bad_evaluations >= GLIMMER_PROFIT_BAD_TO_RETIRE {
                        self.retired = true;
                        self.retire_evaluation = self.evaluations;
                        self.retire_cycle = self.eligible_windows;
                    }
                } else if left <= good_right {
                    self.bad_evaluations = 0;
                }
                // deadband: retain bad_evaluations
            }
        }
    }
}

/// Capture-aware one-token target step used by AR, profit probes, and
/// `HIPFIRE_GLIMMER_SPEC_FALLBACK=1` recovery. Mutates only target KV/capture/logits
/// and RNG — never Harmony routing, mirrors, or generation counters.
pub fn glimmer_ar_forward_pending(
    bundle: &mut hipfire_loader::MuseGlimmerBundle,
    gpu: &mut rdna_compute::Gpu,
    token: u32,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    gpu_sample: bool,
    gpu_rng: u32,
    host_rng: &mut deepseek4::sampling::Xorshift,
) -> Result<(u32, u32), String> {
    let position = bundle.state.n_tokens as u32;
    let top_k_host = top_k.map(|k| k as usize).unwrap_or(0);
    let has_drafter = bundle.drafter.is_some();
    let device_capture = has_drafter && bundle.device_hidden_capture_enabled();
    if has_drafter {
        if device_capture {
            if gpu_sample {
                glimmer::forward::decode_step_sampled_with_device_capture(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    token,
                    position,
                    temp,
                    top_p,
                    top_k,
                    gpu_rng,
                )
            } else {
                glimmer::forward::decode_step_with_device_capture(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    token,
                    position,
                )
                .map(|logits| {
                    let tok = deepseek4::sampling::sample_token(
                        &logits, temp, top_k_host, top_p, host_rng,
                    );
                    (tok, gpu_rng)
                })
            }
        } else if gpu_sample {
            let tap = glimmer_tap_layers(bundle.config.n_layers);
            glimmer::forward::decode_step_sampled_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                token,
                position,
                &tap,
                &mut bundle.target_hidden_host,
                temp,
                top_p,
                top_k,
                gpu_rng,
            )
        } else {
            let tap = glimmer_tap_layers(bundle.config.n_layers);
            glimmer::forward::decode_step_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                token,
                position,
                &tap,
                &mut bundle.target_hidden_host,
            )
            .map(|logits| {
                let tok =
                    deepseek4::sampling::sample_token(&logits, temp, top_k_host, top_p, host_rng);
                (tok, gpu_rng)
            })
        }
    } else if gpu_sample {
        glimmer::forward::decode_step_sampled(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            token,
            position,
            temp,
            top_p,
            top_k,
            gpu_rng,
        )
    } else {
        glimmer::forward::decode_step(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            gpu,
            token,
            position,
        )
        .map(|logits| {
            let tok = deepseek4::sampling::sample_token(&logits, temp, top_k_host, top_p, host_rng);
            (tok, gpu_rng)
        })
    }
}

/// Pure post-window ledger: bonus already routed, not yet in KV/capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlimmerProfitLedger {
    pub mirror_len: usize,
    pub state_n_tokens: usize,
}

pub fn glimmer_profit_ledger_post_window(commit_end: usize) -> GlimmerProfitLedger {
    GlimmerProfitLedger {
        mirror_len: commit_end.saturating_add(1),
        state_n_tokens: commit_end,
    }
}

/// Decode the already-emitted pending bonus once: state advances; mirror unchanged.
pub fn glimmer_profit_ledger_after_bonus_decode(led: GlimmerProfitLedger) -> GlimmerProfitLedger {
    GlimmerProfitLedger {
        mirror_len: led.mirror_len,
        state_n_tokens: led.state_n_tokens.saturating_add(1),
    }
}

/// Continue-spec routes the returned prediction once (mirror only). Retire/AR tail
/// leave the prediction unpushed until the common AR path chooses.
pub fn glimmer_profit_ledger_route_prediction(led: GlimmerProfitLedger) -> GlimmerProfitLedger {
    GlimmerProfitLedger {
        mirror_len: led.mirror_len.saturating_add(1),
        state_n_tokens: led.state_n_tokens,
    }
}

pub fn glimmer_profit_guard_status_label(g: &GlimmerSpecProfitGuard) -> &'static str {
    if !g.enabled() {
        "disabled"
    } else if g.is_retired() {
        "retired"
    } else if g.evaluations() > 0 {
        "continue"
    } else {
        match g.status() {
            GlimmerProfitGuardStatus::Warming => "warming",
            GlimmerProfitGuardStatus::Monitoring => "monitoring",
            GlimmerProfitGuardStatus::Disabled => "disabled",
            GlimmerProfitGuardStatus::Retired => "retired",
        }
    }
}

pub fn generate_muse_glimmer(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    top_k: usize,
    min_p: Option<f32>,
    max_tokens: usize,
    max_think_tokens: usize,
    think_mode: hipfire_runtime::prompt_frame::ThinkMode,
    reasoning_effort: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    let _ = think_mode;
    // max_think_tokens semantics: 1=no-think (ignored for Glimmer; strength from effort), 0=uncapped, N=cap N reasoning tokens then force-close.
    // Open the stream contract FIRST — before any validation, refusal, or GPU work.
    //
    // The HTTP CLI's StreamContractGate requires `gen_start` to be the first event of a
    // request and DROPS anything that arrives before it. Every `emit_error_with_id` below
    // (no tokenizer, missing bundle, history-prep failure, empty prompt, prompt longer than
    // max_seq) used to fire before the latch, so the client never saw the refusal: it waited
    // out its full timeout, then sent `abort` to a request the daemon had already abandoned,
    // and serve's single lane stayed occupied so every later request answered
    // `serve queue wait exceeded 30000ms`. One oversized prompt took the endpoint down.
    //
    // Reproduced with an ~8400-token prompt against max_seq=5120: the refusal is correct and
    // instant, but the client hung 300s and three follow-up requests 503'd.
    //
    // `gen_start` then `error` with no tokens in between is a legal sequence; latching early
    // costs nothing and makes every arch-14 failure routable.
    let gen_contract = gen_start_contract_version_for_arch(m.arch_id);
    emit_gen_start(stdout, id, false, gen_contract);

    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded");
        return;
    }
    // Contract: MuseGlimmerBundle(bundle) — see cross-agent contract.
    // If the loader has not yet published this variant, this match will fail
    // to compile, which is intentional (loud Err rather than silent drop).
    let Some(bundle) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_loader::MuseGlimmerBundle>()
    }) else {
        emit_error_with_id(
            stdout,
            id,
            "muse_glimmer bundle missing on arch_id=14 generate (expected MuseGlimmerBundle)",
        );
        return;
    };

    let bos_tok = bundle.config.bos_token;
    let cfg_eos_tok = bundle.config.eos_token;

    // Splice telemetry. Without it, a cache MISS is indistinguishable from a cache that was
    // never consulted — and a silently-inert verbatim splice still produces plausible output
    // (the render just falls back to retokenizing), so only the LCP would ever betray it.
    let glimmer_replay_hits = std::cell::Cell::new(0usize);
    let glimmer_replay_lookups = std::cell::Cell::new(0usize);

    // ── Prompt build (same two-path branch as the gemma4 AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        let mut ids: Vec<u32> = if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: true,
                bos_token: Some("<bos>"),
                reasoning_strength: Some(glimmer_reasoning_strength(
                    reasoning_effort,
                    max_think_tokens,
                )),
                reasoning_effort: None,
            };
            if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let raw_slice: &[hipfire_runtime::prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::System,
                                content: sys.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                        }
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::User,
                            content: prompt.to_string(),
                            reasoning_content: None,
                            name: None,
                            rendered_name: None,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            tool_plan: String::new(),
                        });
                        synthesized = v;
                        &synthesized
                    }
                };
                let prepared = match prepare_glimmer_onyx_history(raw_slice) {
                    Ok(p) => p,
                    Err(e) => {
                        emit_error_with_id(stdout, id, format!("glimmer history prep failed: {e}"));
                        return;
                    }
                };
                let mut prepared_owned = prepared;
                {
                    let mut ordinal = 0usize;
                    for msg in &mut prepared_owned {
                        if msg.role == hipfire_runtime::prompt_frame::Role::Assistant {
                            let has_client_reasoning = msg
                                .reasoning_content
                                .as_ref()
                                .map(|s| !s.trim().is_empty())
                                .unwrap_or(false);
                            if !has_client_reasoning {
                                let fp_raw = asst_turn_fingerprint(
                                    &normalize_asst_turn_for_fingerprint(&msg.content),
                                    &msg.tool_calls,
                                );
                                let fp = glimmer_turn_key(fp_raw, ordinal);
                                if let Some(cached) = m.asst_turn_cache.get(&fp) {
                                    if let Some(reasoning) = &cached.reasoning {
                                        msg.reasoning_content = Some(reasoning.text.clone());
                                    }
                                }
                            }
                            ordinal += 1;
                        }
                    }
                }
                if messages_history.is_some() {
                    let mut lookup_ordinal = 0usize;
                    let mut cache_lookup = |msg: &hipfire_runtime::prompt_frame::Message| -> Option<
                        hipfire_runtime::prompt_frame::CachedAssistantTurn,
                    > {
                        if msg.role != hipfire_runtime::prompt_frame::Role::Assistant {
                            return None;
                        }
                        let fp_raw = asst_turn_fingerprint(
                            &normalize_asst_turn_for_fingerprint(&msg.content),
                            &msg.tool_calls,
                        );
                        let fp = glimmer_turn_key(fp_raw, lookup_ordinal);
                        lookup_ordinal += 1;
                        glimmer_replay_lookups.set(glimmer_replay_lookups.get() + 1);
                        let hit = m.asst_turn_cache.get(&fp).cloned();
                        if hit.is_some() {
                            glimmer_replay_hits.set(glimmer_replay_hits.get() + 1);
                        }
                        hit
                    };
                    match hipfire_runtime::prompt_frame::build_cached_history_jinja(
                        &frame,
                        &prepared_owned,
                        tools,
                        &mut cache_lookup,
                    ) {
                        Ok(cached_ids) => cached_ids,
                        Err(e) => {
                            eprintln!("[daemon] glimmer build_cached_history_jinja failed ({e}) — falling back to plain render");
                            match frame.render_messages(&prepared_owned, tools, None) {
                                Ok(rendered) => tokenizer.encode(&rendered),
                                Err(e2) => {
                                    eprintln!("[daemon] jinja render failed in muse_glimmer path ({e2}) — falling back to BOS + raw prompt");
                                    tokenizer.encode(prompt)
                                }
                            }
                        }
                    }
                } else {
                    match frame.render_messages(&prepared_owned, tools, None) {
                        Ok(rendered) => tokenizer.encode(&rendered),
                        Err(e) => {
                            eprintln!("[daemon] jinja render failed in muse_glimmer path ({e}) — falling back to BOS + raw prompt");
                            tokenizer.encode(prompt)
                        }
                    }
                }
            } else {
                match frame.render() {
                    Ok(rendered) => tokenizer.encode(&rendered),
                    Err(e) => {
                        eprintln!("[daemon] jinja render failed in muse_glimmer path ({e}) — falling back to BOS + raw prompt");
                        tokenizer.encode(prompt)
                    }
                }
            }
        } else {
            tokenizer.encode(prompt)
        };
        if ids.first() != Some(&bos_tok) {
            ids.insert(0, bos_tok);
        }
        ids
    };

    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }

    // Stop set: scalar eos_token (200001) plus any tokenizer-resolved
    // end-of-turn token. Do NOT copy Gemma4's 106 fallback — Glimmer's
    // HF eos_token_id is a scalar, not a list. Check tokenizer for
    // end-of-turn / im_end style tokens and include if present.
    let stop_set: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let mut s = vec![cfg_eos_tok];
        // Bundle eos_tok may be same as cfg_eos_tok; dedup below.
        if !s.contains(&bundle.eos_tok) {
            s.push(bundle.eos_tok);
        }
        // Candidate end-of-turn / chat-boundary tokens. Use special_token_id
        // (covers vocab-registered specials) plus a single-id encode probe
        // for tokenizers that expose the string via encode.
        for candidate in [
            // Muse Glimmer turn boundaries: `<|end_of_text|>` (200001) and `<|eot|>` (200008)
            // always stop. `<|eom|>` does NOT always stop — it ends the CURRENT CHANNEL
            // (self->keep decoding into answer, user->stop), so it is handled by the
            // Harmony router, not the static stop set.
            "<|eot|>",
            "<|end_of_text|>",
            "<end_of_turn>",
            "<|end_of_turn|>",
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<eos>",
        ] {
            if let Some(tid) = tokenizer.special_token_id(candidate) {
                if !s.contains(&tid) {
                    s.push(tid);
                }
            } else {
                let ids = tokenizer.encode(candidate);
                if ids.len() == 1 && !s.contains(&ids[0]) {
                    // Only accept single-id encodings to avoid polluting stop
                    // set with subword fragments.
                    // For <|endoftext|> this will typically be multi-token and
                    // thus ignored; special_token_id above is the reliable path
                    // for that token.
                    s.push(ids[0]);
                }
            }
        }
        s.dedup();
        s
    };

    // Capacity guard. No eviction on arch_id=14 — reset the KV cursors when
    // the requested run would overflow the physical cache.
    let overflow = { bundle.state.n_tokens + prompt_ids.len() + max_tokens > bundle.state.max_seq };
    if overflow {
        let (n, cap) = (bundle.state.n_tokens, bundle.state.max_seq);
        eprintln!("[daemon] arch_id=14 context full ({n}/{cap}) — resetting GlimmerState");
        bundle.reset_session_state();
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        m.asst_turn_cache.clear();
    }
    // Hard refusal: even from a cold cache the prompt alone must fit (KV
    // writes at pos >= max_seq would be out of bounds).
    let cache_cap = bundle.state.max_seq;
    if prompt_ids.len() >= cache_cap {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "muse_glimmer prompt is {} tokens but max_seq is {cache_cap}",
                prompt_ids.len()
            ),
        );
        return;
    }

    // ── Prefix cache (LCP) — strict forward extension only.
    // Candidate: `lcp == prior_len && 0 < lcp < prompt_ids.len()`.
    // Real hit also requires `bundle.can_rewind_session_to(lcp)`.
    // On hit: rewind_session_to(lcp) BEFORE mirror/seq truncation.
    // On miss / cache-disabled / rewind error: reset_session_state + full prefill at 0.
    let (prefill_ids, prefill_start_pos): (Vec<u32>, u32) = {
        let cache_disabled = std::env::var("HIPFIRE_GLIMMER_PROMPT_CACHE")
            .ok()
            .as_deref()
            == Some("0");
        let trace = std::env::var("HIPFIRE_GLIMMER_CACHE_TRACE").ok().as_deref() == Some("1");
        if cache_disabled {
            // Opting out of the cache does NOT restore the CLI's per-request
            // reset — arch 14 is in the cache_capable allowlist either way, so
            // the CLI keeps the session warm. Cold-reset here or the opt-out
            // path becomes the very double-write the cache was added to avoid.
            if bundle.state.n_tokens > 0 || !m.conversation_tokens.is_empty() {
                bundle.reset_session_state();
                m.conversation_tokens.clear();
                m.seq_pos = 0;
                m.asst_turn_cache.clear();
            }
            (prompt_ids.clone(), 0u32)
        } else {
            let prior_len = m.conversation_tokens.len();
            let max_match = prior_len.min(prompt_ids.len());
            let mut lcp = 0usize;
            while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
                lcp += 1;
            }
            // STRICT FORWARD EXTENSION ONLY. Candidate iff the whole mirror is a
            // prefix of the new render: `lcp == prior_len && 0 < lcp < prompt_ids.len()`.
            let candidate = lcp == prior_len && lcp > 0 && lcp < prompt_ids.len();
            let can_rewind = candidate && bundle.can_rewind_session_to(lcp);
            let cache_hit = can_rewind;
            if trace {
                let reject_reason = if !candidate {
                    "not_strict_forward_extension"
                } else if !can_rewind {
                    "capture_state_cannot_rewind"
                } else {
                    ""
                };
                eprintln!(
                    "[glimmer-cache] prior_len={} n_tokens={} prompt_len={} lcp={} candidate={} hit={} reason={} replay={}/{}",
                    prior_len,
                    bundle.state.n_tokens,
                    prompt_ids.len(),
                    lcp,
                    candidate,
                    cache_hit,
                    reject_reason,
                    glimmer_replay_hits.get(),
                    glimmer_replay_lookups.get()
                );
            }
            if cache_hit {
                // Rewind session (KV + capture + drafter) BEFORE mirror/seq truncate.
                if let Err(e) = bundle.rewind_session_to(lcp) {
                    if trace {
                        eprintln!(
                            "[glimmer-cache] rewind_session_to({lcp}) failed: {e} — converting to MISS"
                        );
                    }
                    bundle.reset_session_state();
                    m.conversation_tokens.clear();
                    m.seq_pos = 0;
                    m.asst_turn_cache.clear();
                    (prompt_ids.clone(), 0u32)
                } else {
                    m.conversation_tokens.truncate(lcp);
                    m.seq_pos = lcp;
                    (prompt_ids[lcp..].to_vec(), lcp as u32)
                }
            } else {
                // MISS with a hot cache MUST cold-reset. Covers both misses:
                // lcp == 0 (fully divergent) and lcp == prompt_ids.len()
                // (identical prompt re-sent), plus candidate rejected by capture.
                if bundle.state.n_tokens > 0 || prior_len > 0 {
                    bundle.reset_session_state();
                    m.conversation_tokens.clear();
                    m.seq_pos = 0;
                    m.asst_turn_cache.clear();
                }
                (prompt_ids.clone(), 0u32)
            }
        }
    };

    let t0 = Instant::now();

    // ── Prefill: chunked batched forward. Device capture when drafter +
    // device backend enabled; else host prefill (with target layers only when
    // a drafter exists). Device mode never mutates target_hidden_host.
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let capture = bundle.drafter.is_some();
        let device_capture = capture && bundle.device_hidden_capture_enabled();
        let start_pos = prefill_start_pos;
        let prefill_result = if device_capture {
            glimmer::forward::prefill_with_device_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                &prefill_ids,
                start_pos,
            )
        } else {
            let tap_layers: Vec<usize> = glimmer_tap_layers(bundle.config.n_layers);
            let target_layers: &[usize] = if capture { &tap_layers } else { &[] };
            glimmer::forward::prefill_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                &prefill_ids,
                start_pos,
                target_layers,
                &mut bundle.target_hidden_host,
            )
        };
        match prefill_result {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("muse_glimmer prefill failed: {e:?}"));
                glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                if device_capture {
                    // Device prefill mutates target KV + capture transaction; length-only
                    // reconcile is not enough — leave zero reusable session state.
                    bundle.reset_session_state();
                    m.conversation_tokens.clear();
                    m.asst_turn_cache.clear();
                    m.seq_pos = 0;
                } else {
                    let _ = reconcile_glimmer_mirror(
                        bundle,
                        &mut m.conversation_tokens,
                        &mut m.seq_pos,
                    );
                }
                return;
            }
        }
    }
    for &tok in &prefill_ids {
        m.conversation_tokens.push(tok);
    }
    // Everything up to here had its KV written by the batched prefill path, so
    // it is the furthest a later turn may reuse. Tokens appended below by the
    // decode loop are written by a different kernel and must not be reused.
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop. Sample host-side from the running logits vector. ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let mut harmony_router = GlimmerHarmonyRouter::new(max_think_tokens);
    let mut glimmer_recorder = GlimmerChannelRecorder::new();
    let mut glimmer_emitted_ids: Vec<u32> = Vec::new();
    let mut glimmer_visible_acc: String = String::new();
    let mut glimmer_tool_acc: String = String::new();
    // Speculative path: greedy is the validated 3.17x path; sampled chain-sample
    // (temp>0) is gated on dflash_fast_sample + batched logits + !min_p.
    // The drafter presence is gated by HIPFIRE_DFLASH_DRAFT / dflash_mode=off
    // (daemon.rs:13118), so AR is untouched when absent.
    let batched_logits_available =
        hipfire_arch_muse_glimmer::forward::glimmer_batched_logits_available(
            &bundle.config,
            &bundle.weights,
        );
    let fast_sample_on = hipfire_runtime::config::get().dflash_fast_sample;
    let temp_spec_env_off = std::env::var("HIPFIRE_DFLASH_TEMP_SPEC").ok().as_deref() == Some("0");
    let spec_mode = glimmer_spec_admission(
        bundle.drafter.is_some(),
        max_tokens,
        temp,
        min_p,
        fast_sample_on,
        temp_spec_env_off,
        batched_logits_available,
    );
    let use_spec = spec_mode != GlimmerSpecMode::Off;
    // Sampled chain needs its own xorshift state seeded from the same source as
    // the AR rng so a fixed request seed stays reproducible. Do not use 0x13579BDF.
    let mut chain_rng: u64 = seed;
    // Logits buffer hoisted outside the window loop and reused — never reallocated per window.
    let mut logits_buf: Vec<f32> = Vec::new();
    // Shared by native AR, profit probes, and post-retirement AR tail.
    let top_k_opt = if top_k > 0 { Some(top_k as u32) } else { None };
    let gpu_sample = !matches!(
        std::env::var("HIPFIRE_GLIMMER_GPU_SAMPLE").ok().as_deref(),
        Some("0")
    );
    let mut gpu_rng: u32 = (rng.next_u64() as u32) | 1;
    // Common AR tail seed. Native AR initializes from prefill logits; retirement
    // hands the untouched greedy target prediction; completed spec leaves None.
    let mut ar_pending: Option<u32> = None;
    // Spec counters lifted out of the branch so the terminal can report them.
    // Without this the whole DFlash run was invisible on the wire: the daemon
    // logged `[glimmer-spec] done: windows 30 ... tau 6.967` to stderr while
    // `timings.dflash` / `.tau` / `.cycles` came back null, so serve_harness
    // correctly refused to credit a dflash run it could not observe.
    let mut spec_windows: usize = 0;
    let mut spec_accepted: usize = 0;
    // Request-local profit guard: default on for greedy spec; kill-switch or
    // timing-distorting diag/audit modes disable it for this request only.
    let profit_guard_env_off = std::env::var("HIPFIRE_GLIMMER_SPEC_PROFIT_GUARD")
        .ok()
        .as_deref()
        == Some("0");
    let profit_guard_diag_off = std::env::var("HIPFIRE_GLIMMER_SPEC_DIAG").ok().as_deref()
        == Some("1")
        || std::env::var("HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT")
            .ok()
            .as_deref()
            == Some("1");
    if use_spec && profit_guard_diag_off && !profit_guard_env_off {
        eprintln!(
            "[glimmer-profit] guard disabled: HIPFIRE_GLIMMER_SPEC_DIAG=1 or HIPFIRE_GLIMMER_DEVICE_CAPTURE_AUDIT=1 distorts timing"
        );
    }
    let mut profit_guard =
        GlimmerSpecProfitGuard::new(use_spec && !profit_guard_env_off && !profit_guard_diag_off);
    if use_spec {
        // Off-by-one note: target_layer_ids [1,13,25,37,49] are 0-based layer
        // indices (dflash_extract_layer_ids(52,5) with start=1.0, end=49.0,
        // step=12.0 → [1,13,25,37,49]). Verified by reproducing the function
        // at speculative.rs:1374 (python: dflash_extract_layer_ids(52,5) == that list).
        // If they were 1-based, extraction would be off by one layer and
        // acceptance would collapse — this choice is logged.
        eprintln!(
            "[glimmer-spec] DFlash drafter active: block={} mask={} layers={:?} (0-based)",
            bundle.drafter.as_ref().unwrap().config.block_size,
            bundle.drafter.as_ref().unwrap().config.mask_token_id,
            bundle.drafter.as_ref().unwrap().config.target_layer_ids
        );
        // Seed is the last prefill token's greedy pick (argmax of last_logits).
        // For the first window we need a committed seed token — use the last
        // prompt token (already in KV) as seed, and last_logits as its logits.
        let mut last_pick = {
            let mut best = 0u32;
            let mut bestv = f32::NEG_INFINITY;
            for (i, &v) in last_logits.iter().enumerate() {
                if v > bestv {
                    bestv = v;
                    best = i as u32;
                }
            }
            best
        };
        // If prefill already placed the prompt's last token at n_tokens-1,
        // the seed position is n_tokens (next write slot). Use that.
        let mut cur_pos = bundle.state.n_tokens as u32;
        // Emit the seed (first generated token) before the first draft window.
        // In AR, last_pick is the first emitted token; in spec, the block's
        // seed is that token, and drafts are for the following positions.
        // Without this, the first window would emit picks[1] (second token)
        // and the output would start mid-sequence (e.g. ":" instead of "def").
        // Harmony: route seed through channel router so header ` to=self<|message|>` is stripped.
        // Static stop (and harmony-forced stop) must not enter the speculative loop and must not
        // commit the stop token into KV/capture — fall through to shared post-spec finalization.
        let mut skip_spec_loop = false;
        if !stop_set.contains(&last_pick) && generated_count < max_tokens {
            let frag = m.tokenizer.as_ref().unwrap().decode(&[last_pick]);
            // Feed through harmony router; header fragments produce no visible event
            let (events, should_stop_seed) = harmony_router.push(&frag);
            for ev in events {
                match ev {
                    GlimmerEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                    GlimmerEmit::Token(text) => {
                        glimmer_visible_acc.push_str(&text);
                        emit_visible_token(stdout, id, &text)
                    }
                    GlimmerEmit::Tool(text) => {
                        glimmer_tool_acc.push_str(&text);
                    }
                }
            }
            if harmony_router.just_forced() {
                glimmer_recorder.mark_forced_reasoning_close();
            }
            glimmer_recorder.push(last_pick, &frag);
            // Seed was routed through harmony; if it indicated stop, we still count it but don't enter spec loop
            m.conversation_tokens.push(last_pick);
            glimmer_emitted_ids.push(last_pick);
            generated_count += 1;
            if should_stop_seed {
                let tau = 0.0;
                eprintln!("[glimmer-spec] seed is stop, tau {:.3}", tau);
                skip_spec_loop = true;
            }
            // Write seed's KV at cur_pos (already done by prefill? No, prefill's last token was prompt, not seed)
            // The seed needs to be written at cur_pos before drafting.
            // Do a single decode_step for the seed to advance KV and get its logits for next window.
            // For now, we will let the first verify block handle the seed's KV write at cur_pos,
            // but we have already emitted the seed, so cur_pos should advance by 1 for the next drafts.
            // Actually the seed's KV will be written by the first verify's block[0] at cur_pos,
            // so we should NOT advance cur_pos yet — the block's seed is at cur_pos.
            // We keep cur_pos as is for the first window's verify.
        } else if stop_set.contains(&last_pick) {
            // Prefill logits already selected a static stop: do not verify/commit it.
            let tau = 0.0;
            eprintln!("[glimmer-spec] seed is stop, tau {:.3}", tau);
            skip_spec_loop = true;
        }
        // Track tau: sum accepted per window / windows
        let mut total_proposed: usize = 0;
        let mut total_accepted: usize = 0;
        let mut windows: usize = 0;
        if !skip_spec_loop {
            loop {
                let t_window = std::time::Instant::now();
                let do_window_timing =
                    std::env::var("HIPFIRE_GLIMMER_TIMING").ok().as_deref() == Some("1");
                if generated_count >= max_tokens {
                    break;
                }
                if bundle.state.n_tokens >= cache_cap {
                    break;
                }
                // --- Draft: B-1 tokens via glimmer_drafter_forward (raw noise) ---
                // Noise is [seed, MASK*(B-1)] embedded raw (no embed_norm) — the
                // embed_norm contract at forward.rs:84 / drafter.rs:1. Build the
                // raw embeddings on host for the stub drafter, then call the
                // drafter. The stub validates sizes/mask and returns Ok; drafts are
                // synthesized as mask_id (low tau honest) to prove the drafter was
                // consulted — a perturbed draft (e.g. corrupt one token) will trip
                // the byte-identical check, proving the comparison can fail.
                let (block_size, mask_id, hidden) = {
                    let d = bundle.drafter.as_ref().unwrap();
                    (d.config.block_size, d.config.mask_token_id, d.config.hidden)
                };
                if block_size < 2
                    || block_size > hipfire_arch_muse_glimmer::glimmer::GLIMMER_MAX_SPEC_BLOCK
                {
                    break;
                }
                // Noise embedding: row 0 is the seed (last committed token), rows
                // 1..B-1 are the mask token. These are RAW target embeddings — no
                // embed_norm — per modeling_muse_glimmer.py:439, which keeps the norm
                // out of the embedding matrix precisely because the DFlash path must
                // embed without it. The target AR path DOES apply it.
                //
                // This buffer is the drafter's entire input. Allocating it and leaving
                // it zero-filled is what produced tau 1.000 with token-0 drafts: the
                // head ran on zeros, so every row decoded to the same argmax.
                let mut noise_embedding = vec![0f32; block_size * hidden];
                for i in 0..block_size {
                    let t = if i == 0 { last_pick } else { mask_id };
                    match glimmer::forward::embed_raw(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        gpu,
                        t,
                    ) {
                        Ok(v) => {
                            noise_embedding[i * hidden..(i + 1) * hidden]
                                .copy_from_slice(&v[..hidden]);
                        }
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("glimmer drafter noise embed row {i}: {e}"),
                            );
                            let _ = reconcile_glimmer_mirror(
                                bundle,
                                &mut m.conversation_tokens,
                                &mut m.seq_pos,
                            );
                            return;
                        }
                    }
                }
                let t_after_noise = t_window.elapsed();
                let device_capture = bundle.device_hidden_capture_enabled();
                // Freeze effective ctx_cap to min(configured, drafter scratch, device log).
                let configured_ctx_cap = std::env::var("HIPFIRE_GLIMMER_CTX_CAP")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .filter(|v| *v > 0)
                    .unwrap_or(glimmer::drafter::GLIMMER_DRAFTER_CTX_CAP_DEFAULT);
                let scratch_cap = bundle
                    .drafter
                    .as_ref()
                    .map(|d| d.scratch.ctx_capacity())
                    .unwrap_or(configured_ctx_cap);
                let mut ctx_cap = configured_ctx_cap.min(scratch_cap);
                if device_capture {
                    if let Some(log) = bundle.state.target_hidden_log() {
                        ctx_cap = ctx_cap.min(log.capacity_rows());
                    }
                }
                // Target history must end exactly at cur_pos — gap is fatal.
                let history_end: Result<usize, String> = if device_capture {
                    bundle
                        .device_hidden_log_end()
                        .ok_or_else(|| "glimmer-spec: device hidden log missing".to_string())
                } else {
                    match bundle.capture_row_elems() {
                        None => Err("glimmer-spec: host capture row_elems missing".to_string()),
                        Some(row_elems) => {
                            if row_elems == 0 || bundle.target_hidden_host.len() % row_elems != 0 {
                                Err(format!(
                                    "glimmer-spec: host capture misaligned len={} row_elems={}",
                                    bundle.target_hidden_host.len(),
                                    row_elems
                                ))
                            } else {
                                Ok(bundle.target_hidden_host.len() / row_elems)
                            }
                        }
                    }
                };
                let history_end = match history_end {
                    Ok(h) => h,
                    Err(e) => {
                        emit_error_with_id(stdout, id, e.clone());
                        glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                        // Capture/log integrity errors (incl. poisoned transaction) leave no reusable state.
                        bundle.reset_session_state();
                        m.conversation_tokens.clear();
                        m.asst_turn_cache.clear();
                        m.seq_pos = 0;
                        return;
                    }
                };
                if history_end != cur_pos as usize {
                    emit_error_with_id(
                    stdout,
                    id,
                    format!(
                        "glimmer-spec: target history end {history_end} != cur_pos {cur_pos} (capture hole)"
                    ),
                );
                    glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                    bundle.reset_session_state();
                    m.conversation_tokens.clear();
                    m.asst_turn_cache.clear();
                    m.seq_pos = 0;
                    return;
                }
                let ctx_len = (cur_pos as usize).min(ctx_cap);
                let cur_start = cur_pos as usize - ctx_len;
                // Single position span: ctx rows then block rows (Q sits on the tail).
                let positions: Vec<i32> =
                    (cur_start as i32..cur_pos as i32 + block_size as i32).collect();
                debug_assert_eq!(
                    positions.len(),
                    ctx_len + block_size,
                    "positions.len() must equal ctx_len + block_size"
                );
                // Device: forward from resident log. Host: suffix Vec + host call.
                let drafter_ok = if device_capture {
                    let log = match bundle.state.target_hidden_log() {
                        Some(l) => l,
                        None => {
                            emit_error_with_id(
                                stdout,
                                id,
                                "glimmer-spec: device hidden log missing at draft",
                            );
                            glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                            bundle.reset_session_state();
                            m.conversation_tokens.clear();
                            m.asst_turn_cache.clear();
                            m.seq_pos = 0;
                            return;
                        }
                    };
                    let drafter = bundle.drafter.as_mut().unwrap();
                    glimmer::drafter::glimmer_drafter_forward_device(
                        gpu,
                        &drafter.config,
                        &drafter.weights,
                        &mut drafter.scratch,
                        &noise_embedding,
                        log,
                        &positions,
                        block_size,
                        ctx_len,
                    )
                } else {
                    let drafter = bundle.drafter.as_mut().unwrap();
                    let ne = drafter.config.num_extract();
                    let hidden = drafter.config.hidden;
                    let row_elems = ne * hidden;
                    let n_rows = bundle.target_hidden_host.len() / row_elems;
                    let target_hidden: Vec<f32> = if ctx_len == 0 {
                        Vec::new()
                    } else {
                        let start = (n_rows - ctx_len) * row_elems; // SUFFIX — most recent rows
                        bundle.target_hidden_host[start..].to_vec()
                    };
                    glimmer::drafter::glimmer_drafter_forward(
                        gpu,
                        &drafter.config,
                        &drafter.weights,
                        &mut drafter.scratch,
                        &noise_embedding,
                        &target_hidden,
                        &positions,
                        block_size,
                        ctx_len,
                    )
                };
                if let Err(e) = drafter_ok {
                    // Bring-up: requested drafter must be loud and fatal, not silent fallback
                    let fatal = std::env::var("HIPFIRE_GLIMMER_SPEC_FALLBACK")
                        .ok()
                        .as_deref()
                        != Some("1");
                    if fatal {
                        emit_error_with_id(stdout, id, format!("glimmer drafter forward failed (requested via HIPFIRE_DFLASH_DRAFT): {e}"));
                        let es = format!("{e}");
                        if es.contains("poisoned") || es.contains("session requires reset") {
                            glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                            bundle.reset_session_state();
                            m.conversation_tokens.clear();
                            m.asst_turn_cache.clear();
                            m.seq_pos = 0;
                        } else {
                            let _ = reconcile_glimmer_mirror(
                                bundle,
                                &mut m.conversation_tokens,
                                &mut m.seq_pos,
                            );
                        }
                        return;
                    }
                    eprintln!("[glimmer-spec] drafter forward failed: {e} — falling back to AR for this window (HIPFIRE_GLIMMER_SPEC_FALLBACK=1)");
                    // One-token-ahead AR fallback: `last_pick` is already emitted and pending at
                    // `cur_pos`. Decode/capture that pending token (no re-emit/re-count), then
                    // sample a NEW prediction from the returned logits and make it the next pending seed.
                    if bundle.state.n_tokens >= cache_cap {
                        break;
                    }
                    if generated_count >= max_tokens {
                        break;
                    }
                    let pending = last_pick;
                    // Force greedy recovery parameters — speculative path is greedy.
                    let fb = glimmer_ar_forward_pending(
                        bundle, gpu, pending, 0.0, 1.0, None, gpu_sample, gpu_rng, &mut rng,
                    );
                    match fb {
                        Ok((next_tok, new_rng)) => {
                            gpu_rng = new_rng;
                            // Advance only for the decoded pending token.
                            cur_pos = bundle.state.n_tokens as u32;
                            if stop_set.contains(&next_tok) {
                                // Static stop becomes the pending seed but must not be verified/committed.
                                last_pick = next_tok;
                                break;
                            }
                            if generated_count >= max_tokens {
                                last_pick = next_tok;
                                break;
                            }
                            let frag = m.tokenizer.as_ref().unwrap().decode(&[next_tok]);
                            let (events, should_stop_fb) = harmony_router.push(&frag);
                            for ev in events {
                                match ev {
                                    GlimmerEmit::Reasoning(text) => {
                                        emit_reasoning_token(stdout, id, &text)
                                    }
                                    GlimmerEmit::Token(text) => {
                                        glimmer_visible_acc.push_str(&text);
                                        emit_visible_token(stdout, id, &text)
                                    }
                                    GlimmerEmit::Tool(text) => {
                                        glimmer_tool_acc.push_str(&text);
                                    }
                                }
                            }
                            if harmony_router.just_forced() {
                                glimmer_recorder.mark_forced_reasoning_close();
                            }
                            glimmer_recorder.push(next_tok, &frag);
                            m.conversation_tokens.push(next_tok);
                            glimmer_emitted_ids.push(next_tok);
                            generated_count += 1;
                            last_pick = next_tok;
                            if should_stop_fb {
                                break;
                            }
                        }
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("muse_glimmer decode failed: {e:?}"),
                            );
                            glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                            // Capture-aware: fail closed — no length-only reconcile after target decode failure.
                            bundle.reset_session_state();
                            m.conversation_tokens.clear();
                            m.asst_turn_cache.clear();
                            m.seq_pos = 0;
                            return;
                        }
                    }
                    continue;
                }
                let t_after_drafter = t_window.elapsed();
                // Bring-up diagnostic: HIPFIRE_GLIMMER_SPEC_DIAG=1 — device mode
                // does not require/download host hidden; print backend + logical length.
                if std::env::var("HIPFIRE_GLIMMER_SPEC_DIAG").ok().as_deref() == Some("1")
                    && windows < 2
                {
                    let l2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
                    let nz = |v: &[f32]| -> usize { v.iter().filter(|x| **x != 0.0).count() };
                    eprintln!(
                    "[glimmer-diag] ctx_len {} ctx_cap {} cur_pos {} positions len {} head {:?} tail {:?}",
                    ctx_len,
                    ctx_cap,
                    cur_pos,
                    positions.len(),
                    positions.get(..4.min(positions.len())),
                    if positions.len() > 4 {
                        &positions[positions.len() - 4..]
                    } else {
                        &positions[..]
                    },
                );
                    if device_capture {
                        eprintln!(
                        "[glimmer-diag] target_hidden backend=device logical_end {} capacity {}",
                        history_end,
                        bundle
                            .state
                            .target_hidden_log()
                            .map(|l| l.capacity_rows())
                            .unwrap_or(0)
                    );
                    } else {
                        let row_elems = bundle.capture_row_elems().unwrap_or(1);
                        let th_len = bundle.target_hidden_host.len();
                        eprintln!(
                            "[glimmer-diag] target_hidden backend=host len {} rows {} (no D2H)",
                            th_len,
                            th_len / row_elems.max(1)
                        );
                    }
                    eprintln!(
                        "[glimmer-diag] noise_embedding: len {} l2 {:.4} nonzero {}",
                        noise_embedding.len(),
                        l2(&noise_embedding),
                        nz(&noise_embedding)
                    );
                    let drafter = bundle.drafter.as_ref().unwrap();
                    if let Ok(dx) = gpu.download_f32(&drafter.scratch.x) {
                        for r in [0usize, 1, 2] {
                            let s = r * hidden;
                            if s + hidden <= dx.len() {
                                let row = &dx[s..s + hidden];
                                eprintln!(
                                "[glimmer-diag] draft hidden row {r}: l2 {:.4} nonzero {} head {:?}",
                                l2(row),
                                nz(row),
                                &row[..4]
                            );
                            }
                        }
                    }
                }
                // Real drafts: run TARGET lm_head over drafter hidden rows 1..B-1.
                let drafter = bundle.drafter.as_mut().unwrap();
                let hidden_for_draft = {
                    let n = (block_size - 1) * hidden;
                    let t = gpu
                        .alloc_tensor(&[n], rdna_compute::DType::F32)
                        .map_err(|e| format!("drafter hidden alloc: {e:?}"))
                        .unwrap();
                    gpu.hip
                        .memcpy_dtod_at(&t.buf, 0, &drafter.scratch.x.buf, hidden * 4, n * 4)
                        .map_err(|e| format!("drafter hidden copy: {e:?}"))
                        .unwrap();
                    t
                };
                let drafts = match hipfire_arch_muse_glimmer::forward::glimmer_lm_head_picks(
                    gpu,
                    &bundle.config,
                    &bundle.weights,
                    &hidden_for_draft,
                    block_size - 1,
                    &bundle.state.logits,
                    &bundle.state.logits_batch,
                    &bundle.state.argmax_batch,
                    None,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        gpu.free_tensor(hidden_for_draft).ok();
                        emit_error_with_id(stdout, id, format!("glimmer drafter lm_head: {e}"));
                        let _ = reconcile_glimmer_mirror(
                            bundle,
                            &mut m.conversation_tokens,
                            &mut m.seq_pos,
                        );
                        return;
                    }
                };
                gpu.free_tensor(hidden_for_draft).ok();
                let t_after_draft_lm = t_window.elapsed();
                total_proposed += drafts.len();
                // Log first draft for visibility
                if windows < 2 {
                    eprintln!(
                        "[glimmer-spec] drafts sample: {:?}",
                        &drafts[..drafts.len().min(4)]
                    );
                }
                // --- Verify: block = [seed, drafts...] at cur_pos ---
                // Do NOT commit/truncate/rollback before routing emitted tokens.
                let window_start_cur_pos = cur_pos as usize;
                let mut block = Vec::with_capacity(block_size);
                block.push(last_pick);
                block.extend_from_slice(&drafts);
                // For ChainSampled, pass Some(&mut logits_buf) so the verify path
                // copies the per-position target logits (row-major [pos][vocab])
                // needed by naive_sample_chain. Greedy passes None and pays nothing.
                let picks = if spec_mode == GlimmerSpecMode::ChainSampled {
                    if device_capture {
                        match glimmer::forward::verify_block_with_device_capture(
                            &bundle.config,
                            &bundle.weights,
                            &mut bundle.state,
                            gpu,
                            &block,
                            cur_pos,
                            Some(&mut logits_buf),
                        ) {
                            Ok(p) => p,
                            Err(e) => {
                                emit_error_with_id(
                                    stdout,
                                    id,
                                    format!("muse_glimmer verify failed: {e:?}"),
                                );
                                glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                                bundle.reset_session_state();
                                m.conversation_tokens.clear();
                                m.asst_turn_cache.clear();
                                m.seq_pos = 0;
                                return;
                            }
                        }
                    } else {
                        match glimmer::forward::verify_block_with_capture(
                            &bundle.config,
                            &bundle.weights,
                            &mut bundle.state,
                            gpu,
                            &block,
                            cur_pos,
                            &glimmer_tap_layers(bundle.config.n_layers),
                            &mut bundle.target_hidden_host,
                            Some(&mut logits_buf),
                        ) {
                            Ok(p) => p,
                            Err(e) => {
                                emit_error_with_id(
                                    stdout,
                                    id,
                                    format!("muse_glimmer verify failed: {e:?}"),
                                );
                                let _ = reconcile_glimmer_mirror(
                                    bundle,
                                    &mut m.conversation_tokens,
                                    &mut m.seq_pos,
                                );
                                return;
                            }
                        }
                    }
                } else if device_capture {
                    match glimmer::forward::verify_block_with_device_capture(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        gpu,
                        &block,
                        cur_pos,
                        None,
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("muse_glimmer verify failed: {e:?}"),
                            );
                            glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                            bundle.reset_session_state();
                            m.conversation_tokens.clear();
                            m.asst_turn_cache.clear();
                            m.seq_pos = 0;
                            return;
                        }
                    }
                } else {
                    match glimmer::forward::verify_block_with_capture(
                        &bundle.config,
                        &bundle.weights,
                        &mut bundle.state,
                        gpu,
                        &block,
                        cur_pos,
                        &glimmer_tap_layers(bundle.config.n_layers),
                        &mut bundle.target_hidden_host,
                        None,
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("muse_glimmer verify failed: {e:?}"),
                            );
                            let _ = reconcile_glimmer_mirror(
                                bundle,
                                &mut m.conversation_tokens,
                                &mut m.seq_pos,
                            );
                            return;
                        }
                    }
                };
                let t_after_verify = t_window.elapsed();
                // Accept via shared variables — sampled reuses the same emit/keep_rows/rollback code.
                let (accept, bonus) = if spec_mode == GlimmerSpecMode::ChainSampled {
                    hipfire_runtime::ddtree::naive_sample_chain(
                        &logits_buf,
                        &drafts,
                        bundle.config.vocab_size,
                        temp,
                        top_p,
                        top_k,
                        &mut chain_rng,
                    )
                } else {
                    let ga = accept_greedy_prefix(&drafts, &picks, None);
                    let a = ga.accepted;
                    let b = if a < drafts.len() {
                        picks[a]
                    } else {
                        picks[picks.len() - 1]
                    };
                    (a, b)
                };
                total_accepted += accept;
                windows += 1;
                // Honest tau logging per window (parent re-runs perf gate)
                if windows % 16 == 0 || accept + 1 < block_size {
                    eprintln!(
                        "[glimmer-spec] window {}: proposed {} accepted {} tau {:.2} cur_pos {}",
                        windows,
                        drafts.len(),
                        accept,
                        if windows > 0 {
                            total_accepted as f32 / windows as f32
                        } else {
                            0.0
                        },
                        cur_pos
                    );
                }
                if do_window_timing {
                    let t_total = t_window.elapsed();
                    eprintln!(
                    "[glimmer-window-timing] win {} noise={:.1}ms drafter={:.1}ms draft_lm={:.1}ms verify={:.1}ms total={:.1}ms ctx_len={} B={}",
                    windows,
                    t_after_noise.as_secs_f64() * 1000.0,
                    (t_after_drafter - t_after_noise).as_secs_f64() * 1000.0,
                    (t_after_draft_lm - t_after_drafter).as_secs_f64() * 1000.0,
                    (t_after_verify - t_after_draft_lm).as_secs_f64() * 1000.0,
                    t_total.as_secs_f64() * 1000.0,
                    ctx_len,
                    block_size
                );
                }
                // Build to_emit (accepted drafts + bonus), route/push until stop/max,
                // THEN commit capture + rewind. Uncaptured bonus is never in keep_rows.
                let mut to_emit: Vec<u32> = Vec::with_capacity(accept + 1);
                to_emit.extend_from_slice(&drafts[..accept]);
                to_emit.push(bonus);
                // Proven-able-to-fail check: corrupt one draft and show identity trips.
                // Only meaningful for Greedy (sampled uses naive_sample_chain, not
                // accept_greedy_prefix). Gate so sampled does not emit a spurious
                // FAILED when the greedy comparison is not in use.
                if spec_mode == GlimmerSpecMode::Greedy
                    && std::env::var("HIPFIRE_GLIMMER_SPEC_PERTURB")
                        .ok()
                        .as_deref()
                        == Some("1")
                    && !drafts.is_empty()
                {
                    let mut perturbed = drafts.clone();
                    perturbed[0] = picks[0];
                    let ga2 = accept_greedy_prefix(&perturbed, &picks, None);
                    if ga2.accepted == accept {
                        eprintln!("[glimmer-spec] PERTURB CHECK FAILED: forcing draft[0]=target_pick[0] did not change accept {} (picks[0]={}) — comparison not sensitive or drafter never ran", accept, picks[0]);
                    } else {
                        eprintln!("[glimmer-spec] PERTURB CHECK OK: forcing draft[0]=target_pick[0] changed accept {} -> {} — comparison can fail, drafter ran (windows={})", accept, ga2.accepted, windows);
                    }
                }
                // Emit via Harmony router: header stripped, reasoning vs token, eom handling
                let mut should_stop_block = false;
                for &tok in &to_emit {
                    if generated_count >= max_tokens {
                        break;
                    }
                    if stop_set.contains(&tok) {
                        let frag = m.tokenizer.as_ref().unwrap().decode(&[tok]);
                        let (events, _) = harmony_router.push(&frag);
                        for ev in events {
                            match ev {
                                GlimmerEmit::Reasoning(text) => {
                                    emit_reasoning_token(stdout, id, &text)
                                }
                                GlimmerEmit::Token(text) => {
                                    glimmer_visible_acc.push_str(&text);
                                    emit_visible_token(stdout, id, &text)
                                }
                                GlimmerEmit::Tool(text) => {
                                    glimmer_tool_acc.push_str(&text);
                                }
                            }
                        }
                        if harmony_router.just_forced() {
                            glimmer_recorder.mark_forced_reasoning_close();
                        }
                        glimmer_recorder.push(tok, &frag);
                        last_pick = tok;
                        m.conversation_tokens.push(tok);
                        glimmer_emitted_ids.push(tok);
                        generated_count += 1;
                        should_stop_block = true;
                        break;
                    }
                    let frag = m.tokenizer.as_ref().unwrap().decode(&[tok]);
                    let (events, should_stop) = harmony_router.push(&frag);
                    for ev in events {
                        match ev {
                            GlimmerEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                            GlimmerEmit::Token(text) => {
                                glimmer_visible_acc.push_str(&text);
                                emit_visible_token(stdout, id, &text)
                            }
                            GlimmerEmit::Tool(text) => {
                                glimmer_tool_acc.push_str(&text);
                            }
                        }
                    }
                    if harmony_router.just_forced() {
                        glimmer_recorder.mark_forced_reasoning_close();
                    }
                    glimmer_recorder.push(tok, &frag);
                    m.conversation_tokens.push(tok);
                    glimmer_emitted_ids.push(tok);
                    generated_count += 1;
                    if should_stop {
                        last_pick = tok;
                        should_stop_block = true;
                        break;
                    }
                }
                // keep_rows = seed + actually retained accepted input rows; bonus uncaptured.
                let keep_rows = (accept + 1).min(
                    m.conversation_tokens
                        .len()
                        .saturating_sub(window_start_cur_pos),
                );
                // Wire routing above is excluded from S.
                let pre_route_ns = t_after_verify.as_nanos();
                let t_commit = std::time::Instant::now();
                // Device: commit staged verify prefix. Host: rewind truncates host atomically.
                if device_capture {
                    if let Err(e) = glimmer::forward::commit_device_verify_capture(
                        &mut bundle.state,
                        gpu,
                        keep_rows,
                    ) {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("muse_glimmer commit_device_verify_capture failed: {e}"),
                        );
                        glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                        bundle.reset_session_state();
                        m.conversation_tokens.clear();
                        m.asst_turn_cache.clear();
                        m.seq_pos = 0;
                        return;
                    }
                }
                let commit_end = window_start_cur_pos + keep_rows;
                if let Err(e) = bundle.rewind_session_to(commit_end) {
                    emit_error_with_id(
                        stdout,
                        id,
                        format!(
                            "muse_glimmer rewind_session_to({commit_end}) after verify failed: {e}"
                        ),
                    );
                    glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                    bundle.reset_session_state();
                    m.conversation_tokens.clear();
                    m.asst_turn_cache.clear();
                    m.seq_pos = 0;
                    return;
                }
                cur_pos = commit_end as u32;
                // Only after transaction resolution may break on stop/max.
                if should_stop_block {
                    break;
                }
                // Full non-terminal window eligible for profit evidence: keep_rows
                // matches accepted+1 and one useful B=1 step still fits.
                let full_window = keep_rows == accept + 1
                    && keep_rows > 0
                    && generated_count < max_tokens
                    && (bundle.state.n_tokens as usize) < cache_cap;
                let mut probe_kind = GlimmerProfitProbeKind::None;
                if full_window && profit_guard.enabled() && !profit_guard.is_retired() {
                    if let Err(e) = gpu.hip.device_synchronize() {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("muse_glimmer profit guard synchronization failed: {e:?}"),
                        );
                        glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                        bundle.reset_session_state();
                        m.conversation_tokens.clear();
                        m.asst_turn_cache.clear();
                        m.seq_pos = 0;
                        return;
                    }
                    // t_after_verify covers draft+verify from t_window start.
                    // t_commit covers commit/rewind through completion of any
                    // asynchronous capture copy; wire routing remains excluded.
                    let commit_ns = t_commit.elapsed().as_nanos();
                    let spec_ns = pre_route_ns.saturating_add(commit_ns);
                    probe_kind = profit_guard.observe_full_window(spec_ns, keep_rows);
                }
                if matches!(
                    probe_kind,
                    GlimmerProfitProbeKind::Warmup | GlimmerProfitProbeKind::Measured
                ) {
                    // Already-emitted bonus at commit_end: decode once without re-route/re-count.
                    let measured = matches!(probe_kind, GlimmerProfitProbeKind::Measured);
                    let probe_t0 = std::time::Instant::now();
                    let probe = glimmer_ar_forward_pending(
                        bundle, gpu, bonus, 0.0, 1.0, None, gpu_sample, gpu_rng, &mut rng,
                    );
                    match probe {
                        Ok((pred, new_rng)) => {
                            gpu_rng = new_rng;
                            if let Err(e) = gpu.hip.device_synchronize() {
                                emit_error_with_id(
                                    stdout,
                                    id,
                                    format!(
                                        "muse_glimmer profit probe synchronization failed: {e:?}"
                                    ),
                                );
                                glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                                bundle.reset_session_state();
                                m.conversation_tokens.clear();
                                m.asst_turn_cache.clear();
                                m.seq_pos = 0;
                                return;
                            }
                            let probe_ns = probe_t0.elapsed().as_nanos();
                            // Warmup still reports elapsed; controller discards it.
                            profit_guard.observe_probe(probe_ns);
                            cur_pos = bundle.state.n_tokens as u32;
                            if measured {
                                let ratio = profit_guard.last_ratio_milli() as f64 / 1000.0;
                                let action = if profit_guard.is_retired() {
                                    "retire"
                                } else {
                                    "continue"
                                };
                                eprintln!(
                                    "[glimmer-profit] action={action} ctx={} windows=4 productive={} spec_ms={:.3} ar_b1_ms={:.3} ratio={:.4} bad_rounds={}/2",
                                    bundle.state.n_tokens,
                                    profit_guard.last_productive(),
                                    profit_guard.last_spec_ns() as f64 / 1_000_000.0,
                                    profit_guard.last_ar_probe_ns() as f64 / 1_000_000.0,
                                    ratio,
                                    profit_guard.bad_evaluations(),
                                );
                            }
                            if stop_set.contains(&pred) {
                                // Let the common AR terminal path route/record/flush
                                // the untouched stop prediction exactly once.
                                ar_pending = Some(pred);
                                break;
                            }
                            if profit_guard.is_retired() {
                                ar_pending = Some(pred);
                                break;
                            }
                            // Continue spec: route the returned prediction once and
                            // restore the one-token-ahead seed invariant.
                            if generated_count >= max_tokens {
                                break;
                            }
                            let frag = m.tokenizer.as_ref().unwrap().decode(&[pred]);
                            let (events, should_stop_pred) = harmony_router.push(&frag);
                            for ev in events {
                                match ev {
                                    GlimmerEmit::Reasoning(text) => {
                                        emit_reasoning_token(stdout, id, &text)
                                    }
                                    GlimmerEmit::Token(text) => {
                                        glimmer_visible_acc.push_str(&text);
                                        emit_visible_token(stdout, id, &text)
                                    }
                                    GlimmerEmit::Tool(text) => {
                                        glimmer_tool_acc.push_str(&text);
                                    }
                                }
                            }
                            if harmony_router.just_forced() {
                                glimmer_recorder.mark_forced_reasoning_close();
                            }
                            glimmer_recorder.push(pred, &frag);
                            m.conversation_tokens.push(pred);
                            glimmer_emitted_ids.push(pred);
                            generated_count += 1;
                            last_pick = pred;
                            if should_stop_pred {
                                break;
                            }
                        }
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("muse_glimmer profit probe decode failed: {e:?}"),
                            );
                            glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                            bundle.reset_session_state();
                            m.conversation_tokens.clear();
                            m.asst_turn_cache.clear();
                            m.seq_pos = 0;
                            return;
                        }
                    }
                } else {
                    // No probe this window: bonus remains the uncommitted next seed.
                    last_pick = bonus;
                }
                if generated_count >= max_tokens {
                    break;
                }
            }
        } // !skip_spec_loop
        spec_windows = windows;
        spec_accepted = total_accepted;
        if windows > 0 {
            let tau = if windows > 0 {
                (total_accepted as f32 + windows as f32) / windows as f32
            } else {
                0.0
            };
            eprintln!("[glimmer-spec] done: windows {} proposed {} accepted {} tau {:.3} (bonus per window counted)", windows, total_proposed, total_accepted, tau);
        }
    } else {
        // Native AR: host-draw first pending from prefill logits; remaining
        // steps share the common capture-aware helper and requested sampling.
        ar_pending = Some(deepseek4::sampling::sample_token(
            &last_logits,
            temp,
            top_k,
            top_p,
            &mut rng,
        ));
    }

    // Common AR tail: native AR and post-retirement handoff only.
    // Retirement forces greedy; native AR keeps the request's sampling params.
    if let Some(mut next_tok) = ar_pending.take() {
        // Greedy spec forces greedy AR tail; sampled spec and native AR use the request's real sampling.
        let ar_temp = if spec_mode == GlimmerSpecMode::Greedy {
            0.0
        } else {
            temp
        };
        let ar_top_p = if spec_mode == GlimmerSpecMode::Greedy {
            1.0
        } else {
            top_p
        };
        let ar_top_k_opt = if spec_mode == GlimmerSpecMode::Greedy {
            None
        } else {
            top_k_opt
        };

        loop {
            if generated_count >= max_tokens {
                break;
            }
            // eos/eot always stop; eom is handled via router text marker
            if stop_set.contains(&next_tok) {
                // Flush any pending via router marker path then stop
                let frag = m.tokenizer.as_ref().unwrap().decode(&[next_tok]);
                let (events, _) = harmony_router.push(&frag);
                for ev in events {
                    match ev {
                        GlimmerEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                        GlimmerEmit::Token(text) => {
                            glimmer_visible_acc.push_str(&text);
                            emit_visible_token(stdout, id, &text)
                        }
                        GlimmerEmit::Tool(text) => {
                            glimmer_tool_acc.push_str(&text);
                        }
                    }
                }
                if harmony_router.just_forced() {
                    glimmer_recorder.mark_forced_reasoning_close();
                }
                glimmer_recorder.push(next_tok, &frag);
                // Push stop token to KV? No — mirror prior break-before-push for eos.
                break;
            }

            let frag = {
                let tokenizer = m.tokenizer.as_ref().unwrap();
                tokenizer.decode(&[next_tok])
            };
            let (events, should_stop) = harmony_router.push(&frag);
            for ev in events {
                match ev {
                    GlimmerEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                    GlimmerEmit::Token(text) => {
                        glimmer_visible_acc.push_str(&text);
                        emit_visible_token(stdout, id, &text)
                    }
                    GlimmerEmit::Tool(text) => {
                        glimmer_tool_acc.push_str(&text);
                    }
                }
            }
            if harmony_router.just_forced() {
                glimmer_recorder.mark_forced_reasoning_close();
            }
            glimmer_recorder.push(next_tok, &frag);
            if should_stop {
                break;
            }

            if bundle.state.n_tokens >= cache_cap {
                break;
            }

            match glimmer_ar_forward_pending(
                bundle,
                gpu,
                next_tok,
                ar_temp,
                ar_top_p,
                ar_top_k_opt,
                gpu_sample,
                gpu_rng,
                &mut rng,
            ) {
                Ok((sampled, new_rng)) => {
                    gpu_rng = new_rng;
                    m.conversation_tokens.push(next_tok);
                    glimmer_emitted_ids.push(next_tok);
                    generated_count += 1;
                    next_tok = sampled;
                }
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("muse_glimmer decode failed: {e:?}"));
                    glimmer::forward::abort_device_capture_stage(&mut bundle.state);
                    let _ = reconcile_glimmer_mirror(
                        bundle,
                        &mut m.conversation_tokens,
                        &mut m.seq_pos,
                    );
                    return;
                }
            }
        }
        // Flush any trailing incomplete channel text
        for ev in harmony_router.flush() {
            match ev {
                GlimmerEmit::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                GlimmerEmit::Token(text) => emit_visible_token(stdout, id, &text),
                GlimmerEmit::Tool(text) => {
                    glimmer_tool_acc.push_str(&text);
                }
            }
        }
    }

    // A parse failure here used to vanish into `unwrap_or_default()`: the model
    // emits a whole turn into the tool channel, ATEM refuses it, and the caller
    // gets an empty content with no tool_calls and no reason why. Keep the
    // empty-vec fallback (a malformed call must not fabricate one) but say so.
    let parsed_tool_calls = match parse_glimmer_atem(&glimmer_tool_acc) {
        Ok(calls) => {
            if calls.is_empty() && !glimmer_tool_acc.trim().is_empty() {
                eprintln!(
                    "[glimmer-atem] tool channel produced {} bytes that parsed to ZERO calls; first 600 bytes: {:?}",
                    glimmer_tool_acc.len(),
                    &glimmer_tool_acc[..glimmer_tool_acc.len().min(600)]
                );
            }
            calls
        }
        Err(e) => {
            if !glimmer_tool_acc.trim().is_empty() {
                eprintln!(
                    "[glimmer-atem] tool channel produced {} bytes that failed to parse: {e:?}; first 600 bytes: {:?}",
                    glimmer_tool_acc.len(),
                    &glimmer_tool_acc[..glimmer_tool_acc.len().min(600)]
                );
            }
            Vec::new()
        }
    };
    // Degenerate-attractor guard. A reasoning-only turn legitimately leaves both
    // accumulators empty, so emptiness is not the signal; an identical token
    // repeated for the whole turn is. That is what a zeroed/NaN logits vector
    // looks like downstream (argmax -> id 0, forever), and it otherwise reaches
    // the caller as a silent empty completion.
    if generated_count >= 8 {
        if let Some(&first) = glimmer_emitted_ids.first() {
            if glimmer_emitted_ids.iter().all(|&t| t == first) {
                eprintln!(
                    "[glimmer-degenerate] {generated_count} tokens all decoded to the same id {first}; \
                     logits are almost certainly degenerate (zeroed or NaN)"
                );
            }
        }
    }
    let _ = reconcile_glimmer_mirror(bundle, &mut m.conversation_tokens, &mut m.seq_pos);

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let hit_length_cap = generated_count >= max_tokens;
    let finish_reason = if hit_length_cap {
        "length"
    } else if !parsed_tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let mut pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    // Report the speculation run on the terminal. `tau` counts the free bonus
    // token per window, matching the `[glimmer-spec] done:` line and every
    // other arch's definition, so cross-arch tau comparisons stay meaningful.
    if use_spec && spec_windows > 0 {
        let tau = (spec_accepted as f64 + spec_windows as f64) / spec_windows as f64;
        pending_done["dflash"] = serde_json::Value::Bool(true);
        pending_done["cycles"] = serde_json::json!(spec_windows);
        pending_done["tau"] = serde_json::json!((tau * 1000.0).round() / 1000.0);
        let mode_str = match spec_mode {
            GlimmerSpecMode::Greedy => "greedy",
            GlimmerSpecMode::ChainSampled => "chain_sampled",
            GlimmerSpecMode::Off => "off",
        };
        pending_done["dflash_mode"] = serde_json::Value::String(mode_str.into());
    } else if use_spec {
        // Spec was admitted but produced zero windows (e.g. immediate stop);
        // still report the mode so the wire is unambiguous.
        let mode_str = match spec_mode {
            GlimmerSpecMode::Greedy => "greedy",
            GlimmerSpecMode::ChainSampled => "chain_sampled",
            GlimmerSpecMode::Off => "off",
        };
        pending_done["dflash_mode"] = serde_json::Value::String(mode_str.into());
    }
    if use_spec {
        pending_done["dflash_profit_guard"] =
            serde_json::Value::String(glimmer_profit_guard_status_label(&profit_guard).into());
        pending_done["dflash_retired"] = serde_json::Value::Bool(profit_guard.is_retired());
        pending_done["dflash_guard_evaluations"] = serde_json::json!(profit_guard.evaluations());
        if profit_guard.evaluations() > 0 {
            let ratio = profit_guard.last_ratio_milli() as f64 / 1000.0;
            pending_done["dflash_profit_ratio"] =
                serde_json::json!((ratio * 1000.0).round() / 1000.0);
            pending_done["dflash_ar_b1_ms"] = serde_json::json!(
                ((profit_guard.last_ar_probe_ns() as f64 / 1_000_000.0) * 1000.0).round() / 1000.0
            );
        }
        if profit_guard.is_retired() {
            pending_done["dflash_retire_cycle"] = serde_json::json!(profit_guard.retire_cycle());
        }
    }

    stage_terminal_tool_calls(&mut pending_done, finish_reason, &parsed_tool_calls);
    // Cache store is a COMMITTED side effect: publish it only after the client commits.
    if glimmer_commit_terminal(stdout, id, &pending_done, generated_count) {
        if !hit_length_cap && !glimmer_emitted_ids.is_empty() {
            let ordinal = messages_history
                .map(|h| {
                    h.iter()
                        .filter(|m| {
                            matches!(m.role, hipfire_runtime::prompt_frame::Role::Assistant)
                        })
                        .count()
                })
                .unwrap_or(0);
            let _ = glimmer_store_cached_turn(
                &mut m.asst_turn_cache,
                glimmer_recorder,
                &parsed_tool_calls,
                ordinal,
            );
        }
    } else {
        // Fail closed: the mirror and KV are mid-turn, so no later turn may reuse them.
        bundle.reset_session_state();
        m.conversation_tokens.clear();
        m.asst_turn_cache.clear();
        m.seq_pos = 0;
    }
}
pub fn generate_lfm2moe(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if m.lfm2moe().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "lfm2moe_config missing on arch_id=11 generate",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the minimax AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
        // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
        // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
        // the .hfq carries no template, e.g. an older A1B convert).
        // Jinja default-ON (flipped 2026-06-09): render through the model's chat
        // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
        // ChatML/Plain). Falls back to Plain automatically when no template resolves.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    reasoning_content: None,
                                    name: None,
                                    rendered_name: None,
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                            synthesized = v;
                            &synthesized
                        }
                    };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => tokenizer.encode(&rendered),
                Err(e) => {
                    eprintln!("[daemon] jinja render failed in lfm2moe path ({e}) — falling back to Plain");
                    hipfire_runtime::prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        raw: false,
                    }
                    .build()
                }
            }
        } else {
            hipfire_runtime::prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                raw: false,
            }
            .build()
        }
    };

    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "empty prompt after tokenize",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.lfm2moe().unwrap().eos_tok;

    // LFM2.5 emits MULTIPLE EOS-class tokens inconsistently: the chat turn-end
    // `<|im_end|>` (which `eos_tok` resolves to) but ALSO the document-end
    // `<|endoftext|>` (and rarely `</s>`). The single `next_tok == eos_tok` stop
    // in the decode loop misses the others, so the model LEAKS a literal
    // `<|endoftext|>` into the visible answer (observed: "...Paris.<|endoftext|>")
    // and wastefully keeps generating. This id-set is the FAST path: it catches
    // any EOS-class token whose literal string round-trips through `encode` to a
    // single id (true for `<|im_end|>`). NOTE it does NOT round-trip for
    // `<|endoftext|>` (encode yields subwords, not the special id), so the
    // reliable catch for that one is the string-level guard in the decode loop
    // below, which matches on the DECODED frag.
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
            }
        }
        v
    };

    // Capacity guard BEFORE gen_start — an oversized prompt is a validation error
    // and must not emit gen_start (client expects no stream on validation failure).
    // saturating_add: an adversarially huge max_tokens must not wrap usize
    // and slip under the cap.
    let cap = m.lfm2moe().unwrap().state.max_seq;
    if prompt_ids.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_ids.len(), max_tokens, cap),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    // Open the stream contract BEFORE any GPU work or token emission.
    // The HTTP CLI's StreamContractGate requires the first event to be
    // `gen_start` with the exact (id, attempt_id). Without this, the
    // first `token` is rejected as "stream must begin with gen_start",
    // the client sends `abort`, and the daemon hangs awaiting `commit`
    // while the HTTP handler waits for a stream that will never deliver.
    // This was the root cause of the 3-minute hang on native `hipfire serve`
    // for LFM2.5-230M/350M (direct `infer_lfm2moe` bypasses the gate and was
    // coherent). Mirrors the DS4 fix `e99583afa` and Qwen's `emit_gen_start`.
    let gen_contract = gen_start_contract_version_for_arch(m.arch_id);
    emit_gen_start(stdout, id, false, gen_contract);

    // Cross-conversation reset (FIX: LFM turn-to-turn KV accumulation). The
    // prior design only reset on capacity overflow, so every request APPENDED to
    // the KV at the growing `n_tokens` and the model attended to all prior
    // requests' tokens at offset RoPE positions — benign for a couple of short
    // turns (RoPE decay + the prompt re-establishes context) but it bloats the KV
    // and degrades quality over a real multi-request serve session, only
    // recovering at overflow. LFM2.5's hybrid conv+GQA state can't be cheaply
    // rewound to an arbitrary prefix (the conv window chains back to token 0, like
    // ds4's SWA ring), so partial prefix-reuse is unsafe → cold-rebuild every
    // turn. This is NOT a perf regression: the path already re-prefills the full
    // prompt each turn; it now does so from position 0 with no stale KV. A
    // continuing conversation re-prefills its whole history from the prompt, so
    // multi-turn is preserved (validated: Bjorn/axolotl recall).
    let _ = m.lfm2moe_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. The LAST decode_step's logits
    // are the predictions for the first generated token. Abort is checked
    // before each token so a client cancel during a long prompt (thousands of
    // tokens) does not run the full prefill before honoring the cancel. ──
    let mut last_logits: Vec<f32> = Vec::new();
    let mut prefill_aborted = false;
    {
        let b = m.lfm2moe_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            if check_abort(id) {
                prefill_aborted = true;
                break;
            }
            match lfm2moe::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm2moe prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    if prefill_aborted || check_abort(id) {
        let ep = production_fail_closed_rollback(m, gpu, None, None);
        emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
        return;
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop. Sample host-side from the running logits vector.
    // Abort is checked at the top of every iteration so a mid-decode
    // client cancel stops the loop immediately and emits `aborted`+`done`.
    // Without this, the loop would run for the full `max_tokens` of wasted
    // work after the HTTP client has already timed out and sent `abort`.
    // ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if check_abort(id) {
            let ep = production_fail_closed_rollback(m, gpu, None, None);
            emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
            return;
        }
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            break;
        }

        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        // String-level EOS-class guard. The id-based `stop_toks` above misses
        // `<|endoftext|>` because encoding the literal STRING doesn't round-trip
        // to the special-token id (it yields subwords), so the real token id is
        // never in the set. The daemon decodes one token at a time, so the
        // leaking turn-end token arrives as its own frag — catch it on the
        // decoded text and stop WITHOUT emitting (was: "...Paris.<|endoftext|>").
        if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
            break;
        }
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // Check abort before the next GPU decode_step to avoid launching
        // more work after the client has already cancelled.
        if check_abort(id) {
            let ep = production_fail_closed_rollback(m, gpu, None, None);
            emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
            return;
        }

        let step = {
            let b = m.lfm2moe_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            lfm2moe::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe decode failed: {e:?}"));
                return;
            }
        }
    }

    // Abort latched between loop exit and commit must not proceed to the
    // two-phase commit handshake — emit the attested `aborted` terminal
    // instead so the client's drain loop terminates.
    if check_abort(id) {
        let ep = production_fail_closed_rollback(m, gpu, None, None);
        emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
        return;
    }

    m.seq_pos = m.lfm2moe().unwrap().state.n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {
            let ep = production_fail_closed_rollback(m, gpu, None, None);
            emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
        }
    }
}
/// MiniMax-M2 (arch_id=10) generate path — minimal AR bring-up.
///
/// Mirrors `generate_lfm2moe`'s shape (prefill loop / chunked batched prefill,
/// per-token decode loop, JSONL `token` / `done` events) with two differences:
///
///   1. Prompt build goes through `JinjaChatFrame` when `HIPFIRE_JINJA_CHAT=1`
///      and the model carries a chat_template (so MiniMax-M2's own ChatML-ish
///      template + `tools` / `messages` reach the upstream Jinja branches),
///      falling back to the hand-rolled `ChatFrame::Plain` scaffold otherwise.
///   2. `minimax::forward::decode_step` returns the full logits `Vec<f32>`
///      (the state does NOT stash a greedy next-token), so sampling runs
///      host-side via `deepseek4::sampling::sample_token` on that vector.
///
/// Out of scope for the scaffold (and intentionally NOT wired): spec-decode,
/// MTP, grammar-constrained decoding, tool-call parsing/execution, repeat
/// penalty, multi-GPU, eviction/prefix-cache. Correctness first.
#[allow(clippy::too_many_arguments)]
pub fn generate_minimax(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if m.minimax().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "minimax_config missing on arch_id=10 generate",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the lfm2moe AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with
    // the MiniMax `<think>` generation-primer, so we only re-emit the opener
    // (below) when the model truly begins inside the reasoning block. A jinja
    // render failure that falls back to the Plain frame leaves it false.
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // MiniMax-M2 (arch 10) and LFM2.5 (arch 11) REQUIRE their embedded Jinja
        // chat_template — their structural tokens are NOT ChatML. MiniMax frames
        // turns with `]~b]ai` / `[e~[` and primes the assistant with `<think>\n`;
        // LFM2 needs its `<|startoftext|>` BOS. The hand-rolled Plain ChatML frame
        // emits `<|im_start|>`/`<|im_end|>` which these models never trained on,
        // producing an off-distribution prompt that (a) decodes incoherently and
        // (b) never matches across turns so the LCP prompt-cache is dead. Force
        // jinja on for both (falls back to Plain only when the .hfq carries no
        // template).
        // Jinja default-ON (flipped 2026-06-09); opt out with HIPFIRE_JINJA_CHAT=0.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    reasoning_content: None,
                                    name: None,
                                    rendered_name: None,
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                            synthesized = v;
                            &synthesized
                        }
                    };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => {
                    primed_think = rendered.trim_end().ends_with("<think>");
                    tokenizer.encode(&rendered)
                }
                Err(e) => {
                    eprintln!("[daemon] jinja render failed in minimax path ({e}) — falling back to Plain");
                    hipfire_runtime::prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        raw: false,
                    }
                    .build()
                }
            }
        } else {
            hipfire_runtime::prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                raw: false,
            }
            .build()
        }
    };

    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "empty prompt after tokenize",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.minimax().unwrap().eos_tok;

    // Capacity guard. No eviction on arch_id=10 — reset the KV cursor when the
    // FULL rendered conversation + generation would overflow. `prompt_ids` is
    // the full Jinja-rendered conversation; the LCP below reuses the warm prefix.
    let overflow = {
        let state = &m.minimax().unwrap().state;
        prompt_ids.len().saturating_add(max_tokens) > state.max_seq
    };
    if overflow {
        let (n, cap) = {
            let state = &m.minimax().unwrap().state;
            (state.n_tokens, state.max_seq)
        };
        eprintln!("[daemon] arch_id=10 context full ({n}/{cap}) — resetting MiniMaxState",);
        m.minimax_mut().unwrap().state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();

        // O2b-2 capacity guard (minimax single): the reset above recovers a
        // grown multi-turn conversation, but a SINGLE prompt larger than the
        // whole context can never fit — prefilling it would write past the KV
        // (sized for state.max_seq) and panic, taking down serve. After the
        // reset, if prompt + generation still overflows, emit a clean error.
        let cap = m.minimax().unwrap().state.max_seq;
        // saturating_add: an adversarially huge max_tokens must not wrap usize
        // and slip under the cap.
        if prompt_ids.len().saturating_add(max_tokens) > cap {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_ids.len(), max_tokens, cap),
                "context_length",
                false,
                false
            );
            let _ = stdout.flush();
            return;
        }
    }

    // ── Prefix cache (LCP) with PARTIAL reuse. `prompt_ids` is the full
    // Jinja-rendered conversation (the trained chat template). MiniMax-M2 is an
    // INTERLEAVED-THINKING model: its chat_template renders a prior turn's
    // `<think>…</think>` reasoning into history ONLY while no newer user message
    // follows (`loop.index0 > last_user_index`). Once the next user turn
    // arrives the canonical render DROPS that reasoning, so every position after
    // the most-recent assistant opener shifts and turn N+1 diverges from turn
    // N's KV at that opener — i.e. `lcp < prior_len`, never a pure forward
    // extension. We therefore support PARTIAL reuse: rewind `n_tokens` to `lcp`
    // and re-prefill the (reasoning-free, hence shorter) suffix. MiniMax is
    // standard attention with no compound recurrent/compressed state, so KV
    // positions ≥ lcp are simply overwritten by the new prefill and the stale
    // tail is never attended. The reused prefix GROWS with the conversation
    // (all older turns, reasoning already stripped, stay matched), so
    // steady-state per-turn prefill is just {last visible answer} + {new user}.
    let prefill_ids: Vec<u32> =
        {
            let prior_len = m.conversation_tokens.len();
            let max_match = prior_len.min(prompt_ids.len());
            let mut lcp = 0usize;
            while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
                lcp += 1;
            }
            // A usable common prefix that leaves at least one fresh token to prefill
            // (the render always appends a new `]~b]ai\n<think>\n` primer, so
            // lcp == rendered_len cannot occur on a normal turn). `partial` is the
            // interleaved-thinking divergence (lcp < prior_len); lcp == prior_len is
            // the degenerate pure-extension case (rewind is then a no-op).
            let cache_hit = lcp > 0 && lcp < prompt_ids.len();
            let partial = lcp < prior_len;
            if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                "[minimax-cache] prior_len={} rendered_len={} lcp={} hit={} partial={} n_tokens={}",
                prior_len, prompt_ids.len(), lcp, cache_hit, cache_hit && partial,
                m.minimax().unwrap().state.n_tokens,
            );
            }
            if cache_hit {
                // Rewind KV + token history to the common prefix. When lcp ==
                // prior_len this is a no-op; when lcp < prior_len it discards the
                // stale reasoning+answer tail. The prefill loop below reads
                // `state.n_tokens` as its base position, so n_tokens is the only
                // KV state the rewind must touch (plus the mirror token history).
                m.minimax_mut().unwrap().state.n_tokens = lcp;
                m.conversation_tokens.truncate(lcp);
                m.seq_pos = lcp;
                prompt_ids[lcp..].to_vec()
            } else {
                if prior_len > 0 {
                    m.minimax_mut().unwrap().state.reset();
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                }
                prompt_ids.clone()
            }
        };

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token, or chunked batched prefill.
    // Disjoint field borrows of `m` (config / weights / state) let us also
    // push to `m.conversation_tokens` in the same scope. The LAST forward's
    // logits are the predictions for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.minimax_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        // Batched prefill: process the prompt in chunks of <=64 tokens through
        // the batched verify forward (one weight read per chunk vs one
        // decode_step per token) → much lower TTFT. Validated byte-identical to
        // the sequential path (cosine 1.0). DEFAULT ON when every layer's expert
        // dtypes have batched kernels; the pre-check routes unsupported tiers
        // (MQ3-Lloyd etc.) to the sequential path to avoid a mid-pass error.
        // Force off with HIPFIRE_MINIMAX_BATCH_PREFILL=0.
        let batch_prefill = std::env::var_os("HIPFIRE_MINIMAX_BATCH_PREFILL")
            .map_or(true, |v| v != "0")
            && minimax::forward::forward_batch_supported(weights);
        if batch_prefill && !prefill_ids.is_empty() {
            let mut pos = state.n_tokens;
            // 512-token prefill chunks let forward_batch's scatter-grouped MoE
            // (>=256 rows/chunk) read each expert weight once per chunk via
            // WMMA instead of once per routed token — ~2.2x prefill on
            // MiniMax-M2 (256 experts/top-8). Shorter prompts fall to the
            // indexed path inside forward_batch (below the grouped gate).
            for chunk in prefill_ids.chunks(512) {
                match minimax::forward::forward_batch(cfg, weights, state, gpu, chunk, pos) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("minimax batch prefill failed: {e:?}"),
                        );
                        return;
                    }
                }
                pos += chunk.len();
            }
        } else {
            let mut position = state.n_tokens as u32;
            for &tok in &prefill_ids {
                match minimax::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(stdout, id, format!("minimax prefill failed: {e:?}"));
                        return;
                    }
                }
                position += 1;
            }
        }
    }
    for &tok in &prefill_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // MiniMax-M2's chat template unconditionally primes the assistant turn
    // with `<think>\n` (chat_template.jinja generation-prompt block), so the
    // model's GENERATED tokens begin *inside* the reasoning block and it only
    // ever emits the closing `</think>`. Every downstream `<think>` consumer —
    // the serve reasoning_content/content split, the run/chat-path stripper,
    // and the history `stripThinkingInline` — keys on a LEADING `<think>` and
    // so never engages, leaking the chain-of-thought into `message.content`.
    // The primer is already in the KV from prefill; re-emit it into the token
    // stream (display-only, not pushed to state) so the assistant message is a
    // well-formed `<think>...</think>...` block for every consumer.
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type": "token", "id": id, "text": "<think>\n", "attempt_id": active_attempt_id()})
        );
        let _ = stdout.flush();
    }

    // ── Decode loop. Sample host-side from the running logits vector.
    // `temp <= 0` makes sample_token greedy; otherwise top_p nucleus.
    // Seed the PRNG from wall-clock nanos so successive same-prompt runs
    // don't lock-step (greedy is still deterministic). ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        // Sample next token from the most recent logits.
        let next_tok = deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if next_tok == eos_tok {
            break;
        }

        // Emit the text fragment. Build through serde_json so a user-supplied
        // `id` or arbitrary-UTF-8 fragment can't corrupt the JSONL line.
        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // Advance one step on the freshly sampled token.
        let step = {
            let b = m.minimax_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            // hipGraph decode (opt-in via HIPFIRE_MINIMAX_GRAPH=1, default eager
            // — measured only +1.0% on gfx1151). First call warms up eager, then
            // captures + replays.
            minimax::forward::decode_step_with_graph(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("minimax decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.minimax().unwrap().state.n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}
/// Cohere2-MoE / North-Mini-Code (arch_id=12) generate path. Mirrors
/// `generate_minimax`, plus: BATCHED `forward_batch` prefill (≈9×, see the
/// prefill block) and a Cohere agentic-marker state machine in the decode loop
/// that suppresses the special markers, routes `<|START_THINKING|>` content to a
/// reasoning channel, emits `<|START_TEXT|>` content as the visible answer, and
/// parses `<|START_ACTION|>` blocks into `tool_calls` events. Decode is plain
/// per-token `decode_step` (no hipGraph variant yet). Out of scope (NOT wired):
/// spec-decode, MTP, grammar-constrained decoding, tool EXECUTION, repeat
/// penalty, multi-GPU.
#[allow(clippy::too_many_arguments)]
pub fn generate_cohere2moe(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) {
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if m.cohere2moe().is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "cohere2moe_config missing on arch_id=12 generate",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the minimax / lfm2moe AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with a
    // `<think>` generation-primer, so we only re-emit the opener (below) when
    // the model truly begins inside the reasoning block. (A jinja render failure
    // now returns an error rather than falling back, so we never reach decode
    // with an off-distribution Plain frame.)
    // Set on the sole surviving (successful-render) path; the failure paths in
    // the block below return, so this needs no dead initializer.
    let primed_think: bool;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // Cohere2-MoE / North-Mini-Code REQUIRES its embedded Jinja chat_template
        // — its structural tokens are NOT ChatML (turns end with
        // `<|END_OF_TURN_TOKEN|>`). The hand-rolled Plain ChatML frame emits
        // `<|im_start|>`/`<|im_end|>` which this model never trained on,
        // producing an off-distribution prompt that (a) decodes incoherently and
        // (b) never matches across turns so the LCP prompt-cache is dead. Force
        // jinja on (falls back to Plain only when the .hfq carries no template).
        // Jinja default-ON; opt out with HIPFIRE_JINJA_CHAT=0.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    reasoning_content: None,
                                    name: None,
                                    rendered_name: None,
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                reasoning_content: None,
                                name: None,
                                rendered_name: None,
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
                            });
                            synthesized = v;
                            &synthesized
                        }
                    };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => {
                    primed_think = rendered.trim_end().ends_with("<think>");
                    if std::env::var("HIPFIRE_C2M_DUMP_PROMPT").ok().as_deref() == Some("1") {
                        let ids = tokenizer.encode(&rendered);
                        eprintln!(
                            "[c2m prompt dump] rendered chars={} tokens={}\n>>> HEAD(400):\n{}\n>>> TAIL(800):\n{}\n<<< end",
                            rendered.len(), ids.len(),
                            &rendered[..rendered.len().min(400)],
                            &rendered[rendered.len().saturating_sub(800)..],
                        );
                    }
                    tokenizer.encode(&rendered)
                }
                Err(e) => {
                    // North-Mini-Code's turns are NOT ChatML; a Plain frame emits
                    // <|im_start|>/<|im_end|> the model never trained on → off-
                    // distribution garbage that reads as a model-quality bug. Refuse
                    // rather than silently serve it.
                    eprintln!(
                        "[daemon] cohere2moe jinja render failed ({e}) — refusing ChatML fallback"
                    );
                    emit_error_with_id(stdout, id, format!("cohere2moe jinja render failed: {e}"));
                    return;
                }
            }
        } else {
            // cohere2moe REQUIRES its embedded jinja template (see above). When it
            // is unavailable — HIPFIRE_JINJA_CHAT=0, or the .hfq carries no template
            // — the only frame we could build is Plain ChatML, off-distribution for
            // North. Refuse loudly instead of serving garbage.
            let why = if !jinja_enabled {
                "HIPFIRE_JINJA_CHAT=0 disables the required template"
            } else {
                "model .hfq carries no chat_template"
            };
            eprintln!("[daemon] cohere2moe cannot build a valid prompt frame ({why}) — refusing ChatML fallback");
            emit_error_with_id(
                stdout,
                id,
                format!("cohere2moe requires its jinja chat template ({why})"),
            );
            return;
        }
    };

    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "empty prompt after tokenize",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.cohere2moe().unwrap().eos_tok;

    // Capacity guard. No eviction on arch_id=12 — reset the KV cursor when the
    // FULL rendered conversation + generation would overflow. `prompt_ids` is
    // the full Jinja-rendered conversation; the LCP below reuses the warm prefix.
    // The KV cache holds `max_seq` positions. A prompt that itself exceeds that
    // cannot be prefilled without writing PAST the cache — which previously
    // produced silent GPU-memory corruption (degenerate/garbage output): the
    // guard reset the cursor but then prefilled the oversized prompt anyway.
    // Fix: prompt too long → clean error + free KV; prompt fits but generation
    // would overflow → cap the token budget to the remaining slots so decode
    // stops at capacity instead of writing OOB.
    let max_seq = m.cohere2moe().unwrap().state.max_seq;
    if prompt_ids.len() >= max_seq {
        eprintln!(
            "[daemon] arch_id=12 prompt {} >= max_seq {} — refusing (would OOB the KV cache)",
            prompt_ids.len(),
            max_seq,
        );
        emit_error_with_id(
            stdout,
            id,
            format!(
                "cohere2moe: prompt is {} tokens but KV capacity (max_seq) is {} — load with a larger max_seq or shorten the prompt",
                prompt_ids.len(),
                max_seq
            ),
        );
        let _ = m.cohere2moe_mut().unwrap().state.reset(gpu);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        return;
    }
    // Cap generation so prefill(prompt) + decode(max_tokens) never exceeds the
    // cache. `max_tokens` is shadowed for the decode loop below.
    let max_tokens = max_tokens.min(max_seq - prompt_ids.len());

    // ── Prefix cache (LCP) with PARTIAL reuse. `prompt_ids` is the full
    // Jinja-rendered conversation (the trained chat template). Cohere2-MoE is
    // standard attention with no compound recurrent/compressed state, so KV
    // positions ≥ lcp are simply overwritten by the new prefill and the stale
    // tail is never attended. We rewind `n_tokens` to `lcp` and re-prefill the
    // suffix; the reused prefix GROWS with the conversation.
    let prefill_ids: Vec<u32> = {
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(prompt_ids.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        // A usable common prefix that leaves at least one fresh token to prefill.
        // `partial` is the divergence case (lcp < prior_len); lcp == prior_len is
        // the degenerate pure-extension case (rewind is then a no-op).
        let cache_hit = lcp > 0 && lcp < prompt_ids.len();
        let partial = lcp < prior_len;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cohere2moe-cache] prior_len={} rendered_len={} lcp={} hit={} partial={} n_tokens={}",
                prior_len, prompt_ids.len(), lcp, cache_hit, cache_hit && partial,
                m.cohere2moe().unwrap().state.n_tokens,
            );
        }
        if cache_hit {
            // Rewind KV + token history to the common prefix. When lcp ==
            // prior_len this is a no-op; when lcp < prior_len it discards the
            // stale tail. The prefill loop below reads `state.n_tokens` as its
            // base position, so n_tokens is the only KV state the rewind must
            // touch (plus the mirror token history).
            m.cohere2moe_mut().unwrap().state.n_tokens = lcp;
            m.conversation_tokens.truncate(lcp);
            m.seq_pos = lcp;
            prompt_ids[lcp..].to_vec()
        } else {
            if prior_len > 0 {
                let _ = m.cohere2moe_mut().unwrap().state.reset(gpu);
                m.seq_pos = 0;
                m.conversation_tokens.clear();
            }
            prompt_ids.clone()
        }
    };

    let t0 = Instant::now();

    // ── Prefill: BATCHED `forward_batch` (~9× the per-token path) when the
    // expert tier supports the indexed-MoE GEMV (MQ4/MQ6), chunked at 256 (the
    // WMMA Q8-projection + grouped-MoE sweet spot). `forward_batch` writes KV at
    // [start_pos..start_pos+b] and its attention covers the cached prefix
    // [0..start_pos], so it composes with the prompt cache (start_pos = lcp).
    // Q8/F16 expert tiers (no indexed kernel) fall back to per-token decode_step.
    // The LAST forward's logits predict the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.cohere2moe_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        if cohere2moe::forward::forward_batch_supported(weights) && prefill_ids.len() > 1 {
            let mut i = 0;
            while i < prefill_ids.len() {
                let end = (i + 256).min(prefill_ids.len());
                let start_pos = state.n_tokens;
                match cohere2moe::forward::forward_batch(
                    cfg,
                    weights,
                    state,
                    gpu,
                    &prefill_ids[i..end],
                    start_pos,
                ) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("cohere2moe batched prefill failed: {e:?}"),
                        );
                        return;
                    }
                }
                i = end;
            }
        } else {
            let mut position = state.n_tokens as u32;
            for &tok in &prefill_ids {
                match cohere2moe::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(stdout, id, format!("cohere2moe prefill failed: {e:?}"));
                        return;
                    }
                }
                position += 1;
            }
        }
    }
    for &tok in &prefill_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // Re-emit a leading `<think>\n` opener into the token stream (display-only,
    // not pushed to state) when the rendered prompt primed the assistant turn
    // inside a reasoning block, so downstream `<think>` consumers see a
    // well-formed block. No-op for templates that don't prime think.
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type": "token", "id": id, "text": "<think>\n", "attempt_id": active_attempt_id()})
        );
        let _ = stdout.flush();
    }

    // ── Decode loop. Sample host-side from the running logits vector.
    // `temp <= 0` makes sample_token greedy; otherwise top_p nucleus.
    // Seed the PRNG from wall-clock nanos so successive same-prompt runs
    // don't lock-step (greedy is still deterministic). ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    // Cohere agentic markers are SPECIAL TOKENS. Resolve their ids once (scoped
    // borrow so the loop can still mutate `m`). This model emits <|START_TEXT|>
    // for its response (NOT the template's <|START_RESPONSE|>, which isn't a
    // real special token here — verified empirically).
    let (mk_think0, mk_think1, mk_text0, mk_text1, mk_act0, mk_act1) = {
        let tk = m.tokenizer.as_ref().unwrap();
        // `encode` SPLITS these added tokens (they round-trip on DECODE only),
        // so resolve content→id via special_token_id, with North-Mini-Code's
        // fixed marker ids as the fallback.
        let mark = |s: &str, fb: u32| -> u32 { tk.special_token_id(s).unwrap_or(fb) };
        (
            mark("<|START_THINKING|>", 255010),
            mark("<|END_THINKING|>", 255011),
            mark("<|START_TEXT|>", 255012),
            mark("<|END_TEXT|>", 255013),
            mark("<|START_ACTION|>", 255014),
            mark("<|END_ACTION|>", 255015),
        )
    };
    #[derive(PartialEq, Clone, Copy)]
    enum Sec {
        Pre,
        Think,
        Text,
        Action,
    }
    let mut sec = Sec::Pre;
    let mut action_buf = String::new();

    // Degenerate-output guards. The long-context forward can collapse into a
    // single-token attractor — observed in a Pi session where a ~30K-token
    // prompt drove the model to emit `<PAD>` until the client aborted ~9.5 min
    // later. These stop the decode promptly instead of hanging. They do NOT mask
    // the underlying forward bug: a fired guard MEANS the forward produced
    // garbage (long-context attention/KV) and is logged loudly so it's caught.
    let pad_tok = m.tokenizer.as_ref().unwrap().special_token_id("<PAD>");
    let mut last_tok: u32 = u32::MAX;
    let mut repeat_run: usize = 0;
    const REPEAT_GUARD: usize = 24; // consecutive-identical-token attractor

    // Empty-turn guard: North sometimes emits <|END_THINKING|> then
    // <|END_OF_TURN_TOKEN|> with no response or action, surfacing as a turn with
    // reasoning only and empty visible content (the model ends after <think>
    // without returning a result). Track whether anything visible (a text token
    // or a tool_call) was produced; if EOS arrives before that, mask it and
    // re-sample so the model is forced into <|START_TEXT|>/<|START_ACTION|> and
    // actually returns content. Opt out with HIPFIRE_C2M_EMPTY_TURN_GUARD=0.
    let empty_turn_guard = std::env::var("HIPFIRE_C2M_EMPTY_TURN_GUARD")
        .ok()
        .as_deref()
        != Some("0");
    let mut emitted_visible = false;
    let mut eos_suppressions = 0usize;
    const MAX_EOS_SUPPRESS: usize = 3;

    // Think-budget force-close (mechanism #2): a heavy reasoner can out-think its
    // token budget and return reasoning only (it hits max_tokens still inside
    // <think>). Reserve room for an answer; honor an explicit max_think_tokens
    // when the client sets one. When thinking reaches the budget with nothing
    // visible yet, inject <|END_THINKING|> + <|START_TEXT|> so the model closes
    // thinking and answers within the reserve instead of hitting the cap empty.
    let think_reserve = (max_tokens / 4).clamp(64, 512).min(max_tokens / 2);
    let think_budget = if max_think_tokens > 1 {
        max_think_tokens.min(max_tokens.saturating_sub(think_reserve))
    } else {
        max_tokens.saturating_sub(think_reserve)
    };
    let mut think_count = 0usize;
    let mut think_force_closed = false;
    let mut forced_toks: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

    // Tool-calling robustness against non-Cohere harnesses: known tool names (to
    // snap a verbose/hallucinated name back to the real tool) and a buffer of the
    // visible output (to recover a tool call the model wrote as TEXT instead of
    // via <|START_ACTION|>). Handles both {function:{name}} (OpenAI) and {name}.
    let known_tools: Vec<String> = tools
        .map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .or_else(|| t.get("name"))
                        .and_then(|n| n.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();
    // name -> [parameter names], for snapping glitched argument keys.
    let tool_params: Vec<(String, Vec<String>)> = tools
        .map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    let name = f.get("name").and_then(|n| n.as_str())?.to_string();
                    let params = f
                        .get("parameters")
                        .and_then(|p| p.get("properties"))
                        .and_then(|p| p.as_object())
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    Some((name, params))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut tool_calls_emitted = false;
    let mut held_tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
    let mut vis_buf = String::new();

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        if empty_turn_guard
            && forced_toks.is_empty()
            && !think_force_closed
            && !emitted_visible
            && sec == Sec::Think
            && think_count >= think_budget
        {
            eprintln!(
                "[cohere2moe] think-budget guard: force-closing thinking at {think_count} think-tok \
                 (budget {think_budget}, max_tokens {max_tokens}) — forcing an answer"
            );
            forced_toks.push_back(mk_think1);
            forced_toks.push_back(mk_text0);
            think_force_closed = true;
        }
        let mut next_tok = match forced_toks.pop_front() {
            Some(t) => t,
            None => deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng),
        };
        if next_tok == eos_tok {
            if empty_turn_guard && !emitted_visible && eos_suppressions < MAX_EOS_SUPPRESS {
                // Reasoning-only turn in progress — the model is ending after
                // <think> with nothing visible. FORCE a <|START_TEXT|> continuation
                // rather than re-sampling the EOS-masked distribution: re-sampling
                // drew from the low-probability tail (where garbage / other-language
                // tokens live), emitting a garbage first token that then derailed the
                // turn (the "veen hier" / "ماعopilot" glitches). Injecting the marker
                // puts the model in the response section so its NEXT token is sampled
                // normally (same as the think-budget force-close, which is coherent).
                // Close thinking first if we somehow took EOS while still inside it.
                eos_suppressions += 1;
                eprintln!(
                    "[cohere2moe] empty-turn guard: forcing START_TEXT (EOS after thinking, no \
                     visible output, #{eos_suppressions}/{MAX_EOS_SUPPRESS}) at gen {generated_count}"
                );
                if sec == Sec::Think {
                    forced_toks.push_back(mk_think1);
                }
                forced_toks.push_back(mk_text0);
                next_tok = forced_toks.pop_front().unwrap();
            } else {
                break;
            }
        }
        // Degenerate-output guards (see above). `<PAD>` is never a valid
        // generation token, and any token repeating REPEAT_GUARD× in a row is an
        // attractor. Either means the forward collapsed — stop before emitting
        // or decoding further, and log so the real bug isn't silently swallowed.
        if Some(next_tok) == pad_tok {
            eprintln!(
                "[cohere2moe] DEGENERATE OUTPUT: <PAD> (id {next_tok}) emitted at gen {generated_count}, \
                 ctx={} — forward collapse (long-context attention/KV bug); stopping",
                m.cohere2moe().map(|b| b.state.n_tokens).unwrap_or(0)
            );
            break;
        }
        if next_tok == last_tok {
            repeat_run += 1;
            if repeat_run >= REPEAT_GUARD {
                eprintln!(
                    "[cohere2moe] DEGENERATE OUTPUT: token {next_tok} repeated {repeat_run}× at gen {generated_count} \
                     (attractor) — forward collapse; stopping"
                );
                break;
            }
        } else {
            last_tok = next_tok;
            repeat_run = 1;
        }
        // Probe-mode token-id stream (HIPFIRE_EMIT_TOKEN_IDS=1). Was wired into
        // the qwen35/deepseek4 generate loops but NOT here, so coherence_probe
        // and token-id detectors silently saw nothing for north-mini-code.
        emit_committed_event(
            stdout,
            id,
            next_tok,
            generated_count,
            decode_t0.elapsed().as_millis() as u64,
        );
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // Agentic-marker state machine — markers themselves are never emitted.
        if next_tok == mk_think0 {
            sec = Sec::Think;
        } else if next_tok == mk_text0 {
            sec = Sec::Text;
        } else if next_tok == mk_act0 {
            sec = Sec::Action;
            action_buf.clear();
        } else if next_tok == mk_think1 || next_tok == mk_text1 {
            sec = Sec::Pre;
        } else if next_tok == mk_act1 {
            // End of an action block → parse the JSON array into tool_calls,
            // snapping any verbose/hallucinated tool name to a real tool.
            let mut calls = cohere2moe::spec_emit::parse_cohere_action(&action_buf);
            cohere2moe::spec_emit::snap_call_names(&mut calls, &known_tools, &tool_params);
            if !calls.is_empty() {
                // Hold until commit_ready — no mid-stream / post-commit tool event.
                for c in calls {
                    let name = c
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = c
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));
                    if !name.is_empty() {
                        held_tool_calls.push(hipfire_runtime::prompt_frame::ToolCall {
                            id: None,
                            name,
                            arguments,
                            rendered_body: None,
                        });
                    }
                }
                emitted_visible = true;
                tool_calls_emitted = true;
            }
            sec = Sec::Pre;
        } else {
            // Build the fragment through serde_json so arbitrary UTF-8 can't
            // corrupt the JSONL line.
            let frag = {
                let tokenizer = m.tokenizer.as_ref().unwrap();
                tokenizer.decode(&[next_tok])
            };
            // Defense-in-depth: never emit a Cohere structural marker into
            // visible output / the action buffer. The ID state machine above
            // handles the 6 THINKING/TEXT/ACTION markers; this catches any OTHER
            // special token the model might emit (START_OF_TURN_TOKEN,
            // CHATBOT_TOKEN, START_TOOL_RESULT, …) — each decodes to a full
            // `<|MARKER|>`. The token is still fed to decode_step below; only its
            // emit is dropped, so a state-machine miss can never leak a marker.
            let is_marker = frag.len() > 4
                && frag.starts_with("<|")
                && frag.ends_with("|>")
                && frag[2..frag.len() - 2]
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_');
            if is_marker {
                // suppressed
            } else {
                match sec {
                    Sec::Action => action_buf.push_str(&frag),
                    Sec::Think => {
                        // Reasoning channel: tagged so clients can fold it; the CLI
                        // (ignoring unknown fields) shows it inline.
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type": "token", "id": id, "text": frag, "reasoning": true, "attempt_id": active_attempt_id()})
                        );
                        let _ = stdout.flush();
                        think_count += 1;
                    }
                    Sec::Text | Sec::Pre => {
                        vis_buf.push_str(&frag);
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type": "token", "id": id, "text": frag, "attempt_id": active_attempt_id()})
                        );
                        let _ = stdout.flush();
                        emitted_visible = true;
                    }
                }
            }
        }

        // Advance one step on the freshly sampled token (plain eager decode —
        // no hipGraph variant on this arch yet).
        let step = {
            let b = m.cohere2moe_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            cohere2moe::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("cohere2moe decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.cohere2moe().unwrap().state.n_tokens;

    // Tool-call-as-text recovery: if the model never emitted a <|START_ACTION|>
    // block but wrote a tool-call JSON array (`[{tool_name, parameters}]`) as
    // visible text — which happens when a non-Cohere harness primes it with a
    // generic tool-call format — parse it out and emit it as a real tool_calls
    // event so the harness can actually execute it.
    // Recover tools written as text only when no action-block tools were held.
    // All authoritative calls stage onto commit_ready/done (no tool event).
    if !tool_calls_emitted {
        let mut recovered = cohere2moe::spec_emit::parse_cohere_action(&vis_buf);
        if !recovered.is_empty() {
            cohere2moe::spec_emit::snap_call_names(&mut recovered, &known_tools, &tool_params);
            eprintln!(
                "[cohere2moe] recovered {} tool_call(s) written as text (model skipped <|START_ACTION|>)",
                recovered.len()
            );
            for v in recovered {
                let name = v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = v
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                if !name.is_empty() {
                    held_tool_calls.push(hipfire_runtime::prompt_frame::ToolCall {
                        id: None,
                        name,
                        arguments,
                        rendered_body: None,
                    });
                }
            }
        }
    }

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    // Prefill metrics so the CLI bench measures the REAL production prefill
    // (batched `forward_batch`) straight off the generate response — mirrors
    // the qwen35 `done` fields (`ttft_ms` ≈ prefill_ms, `decode_tok_s` = tok_s).
    // Prior to this, cohere2moe omitted these, so `hipfire bench` fell back to
    // the synthetic per-token `bench_prefill` path and under-reported prefill ~6×.
    let prefill_tokens = prefill_ids.len();
    let prefill_tok_s = if prefill_ms > 0 {
        (prefill_tokens as f64 * 1000.0) / prefill_ms as f64
    } else {
        0.0
    };
    let finish_reason = if !held_tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let mut pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (tok_s * 100.0).round() / 100.0,
        "ttft_ms": prefill_ms,
        "total_ms": total_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    stage_terminal_tool_calls(&mut pending_done, finish_reason, &held_tool_calls);
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}
/// Qwen2 generate path (arch_id=7, hipfire-arch-qwen2).
///
/// Phase-1 bring-up scope: encode prompt → prefill → greedy decode loop
/// → stream `{"type":"token",...}` events → `{"type":"done",...}`.
///
/// Deliberately bypasses qwen35/llama machinery — no PFlash, no DFlash,
/// no eviction, no ChatML scaffolding, no tool-use, no `<think>` /
/// `max_think_tokens`, no repeat penalty, no top-p sampling. These
/// land as the surrounding daemon features mature for the Qwen2 path.
/// `temp` is currently honored only as a "≤ 1e-6 means greedy"
/// signal; anything else falls back to greedy too (no sampler wired).
///
/// Conversation state on the daemon side advances via
/// `m.seq_pos` (mirrors the qwen35/llama bookkeeping) plus
/// `state.next_pos` inside `Qwen2State`. On context overflow we hard
/// reset (no CASK eviction on arch_id=7) — same fallback the
/// llama path uses.
pub fn generate_qwen2(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    _system_prompt: Option<&str>,
    _temp: f32,
    _top_p: f32,
    max_tokens: usize,
    _repeat_penalty: f32,
    _repeat_window: usize,
) {
    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "tokenizer not loaded",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let state_ref = match m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen2::Qwen2Bundle>()
    }) {
        Some(b) => b,
        _ => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "qwen2 state missing on arch_id=7 generate",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let cfg = &state_ref.config;
    let weights = &state_ref.weights;
    let state = &mut state_ref.state;

    // Apply the model's chat template (if present) so instruct-tuned qwen2
    // models receive properly-framed ChatML input. Falls back to raw encoding
    // when no template is loaded (e.g. base/pre-train weights).
    let prompt_ids = if let Some(template) = m.chat_template.as_ref() {
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: _system_prompt,
            user: prompt,
            enable_thinking: true,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        match frame.render() {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!("[daemon] qwen2 jinja render failed ({e}) — falling back to raw encode");
                tokenizer.encode(prompt)
            }
        }
    } else {
        tokenizer.encode(prompt)
    };
    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "empty prompt after tokenize",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // Capacity guard. No eviction on arch_id=7 yet — reset state when
    // the requested run would overflow the KV budget.
    if state
        .next_pos
        .saturating_add(prompt_ids.len())
        .saturating_add(max_tokens)
        > state.max_seq
    {
        eprintln!(
            "[daemon] arch_id=7 context full ({}/{}) — resetting Qwen2State.next_pos",
            state.next_pos, state.max_seq,
        );
        state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();

        // O2b-2 capacity guard (qwen2 single): the reset above (next_pos=0)
        // recovers a grown multi-turn conversation, but a SINGLE prompt larger
        // than the whole context still overflows — prefilling it writes past
        // the KV (sized for state.max_seq) and panics, taking down serve. After
        // the reset, if prompt + generation still overflows, emit a clean error.
        // saturating_add: an adversarially huge max_tokens must not wrap usize
        // and slip under the cap.
        if prompt_ids.len().saturating_add(max_tokens) > state.max_seq {
            let cap = state.max_seq;
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_ids.len(), max_tokens, cap),
                "context_length",
                false,
                false
            );
            let _ = stdout.flush();
            return;
        }
    }

    let t0 = Instant::now();

    // Prefill: forward_step per prompt token. The last call leaves
    // logits in state.logits — these are the predictions for the
    // first generated token.
    for &tok in &prompt_ids {
        if let Err(e) = qwen2::forward_step(gpu, weights, cfg, state, tok) {
            emit_error_with_id(stdout, id, format!("qwen2 prefill failed: {e:?}"));
            let _ = stdout.flush();
            return;
        }
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // Decode loop. Greedy argmax for now (see fn doc for sampling
    // scope). The first generated token is argmax of the prefill's
    // final logits; each subsequent token requires another
    // forward_step.
    let mut generated_count: usize = 0;
    let eos_set: &[u32] = &cfg.eos_token_ids;
    let decode_t0 = Instant::now();
    let mut next_tok = match gpu.argmax_f32(&state.logits, cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("argmax failed: {e:?}"));
            let _ = stdout.flush();
            return;
        }
    };

    loop {
        if generated_count >= max_tokens {
            break;
        }
        if eos_set.contains(&next_tok) {
            break;
        }
        // Emit text fragment for this token. Tokenizer.decode handles
        // BPE byte-fragment reassembly; for special tokens that decode
        // to an empty string we still advance the loop. Build through
        // serde_json so `id` (user-supplied) and `frag` (arbitrary
        // UTF-8 with possible `"` / `\` / control chars) can't corrupt
        // the JSONL line.
        let frag = tokenizer.decode(&[next_tok]);
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        match qwen2::forward_step_greedy(gpu, weights, cfg, state, next_tok) {
            Ok(t) => next_tok = t,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("forward_step_greedy failed: {e:?}"));
                let _ = stdout.flush();
                return;
            }
        }
    }

    // Daemon bookkeeping: seq_pos matches Qwen2State's internal cursor.
    m.seq_pos = state.next_pos;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 && decode_ms > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}
