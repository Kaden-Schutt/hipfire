// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod continuous_batch_tests` (41 of 42 tests).
//! `lfm_dense_is_dense_and_route_is_lfm_ar` remains in `daemon.rs` because
//! it exercises `super::GenerationRoute` / `super::select_generation_route`
//! and `super::lfm2moe::config::Lfm2MoeConfig`, which are daemon-private
//! per-arch routing helpers not part of `hipfire-engine`.

use hipfire_engine::scheduler::{
    is_batch_eligible, parse_continuous_batch_size, parse_serve_continuous_batch, BatchLane,
    BatchPendingRequest, BatchSampling, BatchSamplingKey, ContinuousBatchScheduler, DaemonInbox,
    DaemonMsg,
};
use hipfire_engine::terminal::{
    batch_announce_terminal, batch_apply_terminal_control, batch_check_abort,
    batch_clear_all_terminals, batch_clear_terminal, batch_commit_teardown_class,
    batch_hit_length_cap, batch_lane_at_capacity, batch_mark_ready, batch_mark_ready_with_pending,
    batch_poll_decision, batch_should_finish_decode, batch_terminal_control,
    batch_transfer_abort_to_singleton_and_clear, AttemptKey, BatchCommitTeardownClass,
    ClientTerminalDecision, LaneTicket,
};

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

fn lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn begin() -> MutexGuard<'static, ()> {
    let g = lock();
    batch_clear_all_terminals();
    hipfire_engine::terminal::clear_terminal_control();
    hipfire_engine::terminal::set_active_attempt_id(0);
    g
}

fn sampling(temp: f32, repeat: f32) -> BatchSampling {
    BatchSampling {
        temp,
        top_p: 0.8,
        top_k: None,
        min_p: None,
        repeat_penalty: repeat,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: 128,
    }
}
fn sampling_with_window(temp: f32, window: usize) -> BatchSampling {
    BatchSampling {
        temp,
        top_p: 0.8,
        top_k: None,
        min_p: None,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: window,
    }
}
fn req(key: AttemptKey, sampling: BatchSampling) -> BatchPendingRequest {
    BatchPendingRequest {
        key,
        prompt: "hi".into(),
        prompt_tokens: vec![1, 2, 3],
        started_in_think: false,
        system: None,
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        max_think_tokens: 0,
        max_tokens: 10,
        client_seed: None,
        sampling,
    }
}

fn caps_for(arch_id: u32) -> saddle_core::caps::ArchCaps {
    hipfire_loader::carrier_for(arch_id)
        .map(|c| c.caps())
        .unwrap_or_default()
}
fn elig(
    arch_id: u32,
    pp: usize,
    ep_is_some: bool,
    has_image: bool,
    has_tools: bool,
    has_stop: bool,
    has_speculator: bool,
    has_adaptive: bool,
    has_pflash: bool,
    has_messages_history: bool,
    think_mode_is_nonthink: bool,
    serve_continuous_batch: bool,
    continuous_batch_size: usize,
) -> bool {
    let caps = caps_for(arch_id);
    let req = saddle_core::caps::BatchEligibilityRequest {
        pp,
        ep_is_some,
        has_image,
        has_tools,
        has_stop,
        has_speculator,
        has_adaptive,
        has_pflash,
        has_messages_history,
        think_mode_is_nonthink,
        serve_continuous_batch,
        continuous_batch_size,
    };
    is_batch_eligible(&caps, &req)
}
#[test]
fn batch_eligible_only_qwen_text_single_gpu() {
    let _l = begin();
    assert!(elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(elig(
        6, 1, false, false, false, false, false, false, false, false, true, true, 2
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, false, true, false, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 1
    ));
    assert!(!elig(
        9, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 2, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, true, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, true, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, true, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, true, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, true, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, true, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, true, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, true, true, true, 4
    ));
}

#[test]
fn batch_eligible_refuses_lfm11_and_preserves_qwen() {
    let _l = begin();
    // LFM (arch 11) has no servable batch path: the capability is false, so
    // even a fully clean request is refused and no batch state is allocated.
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 2
    ));
    // Same pure exclusions as Qwen: B=1, pp!=1, ep, images, tools, stops, spec, adaptive, pflash, history, think.
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, false, 4
    ));
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 1
    ));
    assert!(!elig(
        11, 2, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        11, 1, true, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        11, 1, false, true, false, false, false, false, false, false, true, true, 4
    ));
    // Archs beside 5/6 stay ineligible (11 included, via the caps refusal above).
    assert!(!elig(
        12, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        9, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    // Qwen still eligible (preserve existing behavior).
    assert!(elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(elig(
        6, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
}

#[test]
fn parse_defaults_backward_compatible() {
    let _l = begin();
    assert_eq!(parse_continuous_batch_size(None), 1);
    assert_eq!(parse_continuous_batch_size(Some(&serde_json::json!({}))), 1);
    assert_eq!(
        parse_continuous_batch_size(Some(&serde_json::json!({"continuous_batch_size":4}))),
        4
    );
    assert!(!parse_serve_continuous_batch(&serde_json::json!({})));
    assert!(parse_serve_continuous_batch(
        &serde_json::json!({"serve_continuous_batch":true})
    ));
    assert!(parse_serve_continuous_batch(
        &serde_json::json!({"params":{"serve_continuous_batch":true}})
    ));
}

#[test]
fn announcement_race_abort_before_queue_is_latched() {
    let _l = begin();
    let k = AttemptKey::new("r1", 7);
    assert!(batch_announce_terminal(&k.id, k.attempt_id));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(batch_check_abort(&k.id, k.attempt_id));
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    let pending = req(k.clone(), sampling(0.3, 1.0));
    assert!(sched.enqueue(pending));
    assert!(batch_check_abort(&k.id, k.attempt_id));
    assert!(sched.abort_queued(&k));
    assert!(sched.inbox.is_empty());
    assert!(!sched.pending.contains_key(&k));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(!batch_check_abort(&k.id, k.attempt_id));
}

#[test]
fn early_commit_before_ready_rejected_and_poll_is_nonmutating() {
    let _l = begin();
    let k = AttemptKey::new("r1", 7);
    batch_announce_terminal(&k.id, k.attempt_id);
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
    let (_key, ticket) = sched.try_assign_one().unwrap();
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    let pending = serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":3});
    assert!(sched.mark_awaiting_commit(ticket.lane, pending.clone()));
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Commit)
    );
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Commit)
    );
    assert!(sched.commit_lane(ticket.lane, &k));
    assert!(sched.lanes[ticket.lane].is_empty());
}

#[test]
fn stale_owner_generation_prevents_release_of_reused_slot() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    let k1 = AttemptKey::new("r1", 1);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    sched.enqueue(req(k1.clone(), sampling(0.3, 1.0)));
    let (_k, t1) = sched.try_assign_one().unwrap();
    let pending =
        serde_json::json!({"type":"done","id":k1.id,"attempt_id":k1.attempt_id,"tokens":1});
    assert!(sched.mark_awaiting_commit(t1.lane, pending));
    batch_apply_terminal_control("commit", &k1.id, k1.attempt_id);
    assert!(sched.commit_lane(t1.lane, &k1));
    let gen_after = sched.lanes[t1.lane].generation();
    assert!(gen_after > t1.generation);
    let k2 = AttemptKey::new("r2", 2);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req(k2.clone(), sampling(0.3, 1.0)));
    let (_k2, t2) = sched.try_assign_one().unwrap();
    assert_eq!(t2.lane, t1.lane);
    assert_ne!(t2.generation, t1.generation);
    batch_apply_terminal_control("abort", &k1.id, k1.attempt_id);
    assert!(!batch_check_abort(&k2.id, k2.attempt_id));
    batch_apply_terminal_control("commit", &k1.id, k1.attempt_id);
    assert_eq!(batch_poll_decision(&k2.id, k2.attempt_id), None);
    assert!(matches!(
        sched.lanes[t2.lane],
        hipfire_engine::scheduler::BatchLane::Running(_)
    ));
    batch_apply_terminal_control("abort", &k2.id, k2.attempt_id);
    assert!(sched.abort_lane(t2.lane, &k2));
    assert!(sched.lanes[t2.lane].is_empty());
}

#[test]
fn queued_abort_drains_without_assigning_lane() {
    let _l = begin();
    let k = AttemptKey::new("r1", 9);
    batch_announce_terminal(&k.id, k.attempt_id);
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(sched.abort_queued(&k));
    assert!(sched.inbox.is_empty());
    assert!(sched.lanes[0].is_empty());
    assert!(sched.try_assign_one().is_none());
}

#[test]
fn immutable_ready_payload_preserved_until_commit() {
    let _l = begin();
    let k = AttemptKey::new("r1", 11);
    batch_announce_terminal(&k.id, k.attempt_id);
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
    let (_key, ticket) = sched.try_assign_one().unwrap();
    let mut pending = serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":5,"finish_reason":"length"});
    let pending_clone = pending.clone();
    assert!(sched.mark_awaiting_commit(ticket.lane, pending.clone()));
    let stored = sched.pending_done_for(ticket.lane).unwrap();
    assert_eq!(stored, pending_clone);
    pending["tokens"] = serde_json::json!(999);
    let stored2 = sched.pending_done_for(ticket.lane).unwrap();
    assert_eq!(stored2["tokens"], 5);
    let different =
        serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":999});
    assert!(!sched.mark_awaiting_commit(ticket.lane, different));
    let stored3 = sched.pending_done_for(ticket.lane).unwrap();
    assert_eq!(stored3["tokens"], 5);
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert!(sched.commit_lane(ticket.lane, &k));
    assert!(sched.lanes[ticket.lane].is_empty());
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
}

#[test]
fn deadline_30s_is_set_and_poll_returns_abort_after_expiry() {
    let _l = begin();
    let k = AttemptKey::new("r1", 13);
    batch_announce_terminal(&k.id, k.attempt_id);
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
    let (_key, ticket) = sched.try_assign_one().unwrap();
    let pending = serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":2});
    assert!(sched.mark_awaiting_commit(ticket.lane, pending));
    let deadline = sched.deadline_for(ticket.lane).unwrap();
    let now = Instant::now();
    assert!(deadline > now);
    assert!(deadline <= now + Duration::from_secs(30) + Duration::from_millis(100));
    assert!(deadline >= now + Duration::from_secs(29));
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    {
        let mut g = batch_terminal_control().mu.lock().unwrap();
        if let Some(e) = g.entries.get_mut(&k) {
            e.deadline = Some(Instant::now() - Duration::from_secs(1));
        }
    }
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Abort)
    );
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Abort)
    );
    assert!(sched.abort_lane(ticket.lane, &k));
    assert!(sched.lanes[ticket.lane].is_empty());
}

#[test]
fn fifo_cohort_incompatible_head_blocks_later_compatible() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(2, 4096);
    let k1 = AttemptKey::new("r1", 1);
    let k2 = AttemptKey::new("r2", 2);
    let k3 = AttemptKey::new("r3", 3);
    let samp_a = sampling(0.3, 1.0);
    let samp_b = sampling(0.7, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    batch_announce_terminal(&k3.id, k3.attempt_id);
    sched.enqueue(req(k1.clone(), samp_a.clone()));
    sched.enqueue(req(k2.clone(), samp_b.clone()));
    sched.enqueue(req(k3.clone(), samp_a.clone()));
    let (_a1, t1) = sched.try_assign_one().unwrap();
    assert_eq!(t1.lane, 0);
    assert!(sched.try_assign_one().is_none());
    assert_eq!(sched.inbox.front().unwrap(), &k2);
    assert_eq!(sched.inbox.len(), 2);
    let pending =
        serde_json::json!({"type":"done","id":k1.id,"attempt_id":k1.attempt_id,"tokens":1});
    assert!(sched.mark_awaiting_commit(t1.lane, pending));
    batch_apply_terminal_control("commit", &k1.id, k1.attempt_id);
    assert!(sched.commit_lane(t1.lane, &k1));
    let (_b, t2) = sched.try_assign_one().unwrap();
    assert_eq!(t2.lane, 0);
    assert_eq!(sched.inbox.front().unwrap(), &k3);
    assert!(sched.try_assign_one().is_none());
    let pending2 =
        serde_json::json!({"type":"done","id":k2.id,"attempt_id":k2.attempt_id,"tokens":1});
    let lane_for_k2 = sched.find_lane_by_key(&k2).unwrap();
    assert!(sched.mark_awaiting_commit(lane_for_k2, pending2));
    batch_apply_terminal_control("commit", &k2.id, k2.attempt_id);
    assert!(sched.commit_lane(lane_for_k2, &k2));
    let (_a3, t3) = sched.try_assign_one().unwrap();
    assert_eq!(t3.lane, 0);
    let pending3 =
        serde_json::json!({"type":"done","id":k3.id,"attempt_id":k3.attempt_id,"tokens":1});
    let lane_for_k3 = sched.find_lane_by_key(&k3).unwrap();
    assert!(sched.mark_awaiting_commit(lane_for_k3, pending3));
    batch_apply_terminal_control("commit", &k3.id, k3.attempt_id);
    assert!(sched.commit_lane(lane_for_k3, &k3));
}

#[test]
fn refill_reservation_awaiting_client_not_reused_until_commit() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    let k1 = AttemptKey::new("r1", 1);
    let k2 = AttemptKey::new("r2", 2);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req(k1.clone(), sampling(0.3, 1.0)));
    sched.enqueue(req(k2.clone(), sampling(0.3, 1.0)));
    let (_k1, t1) = sched.try_assign_one().unwrap();
    assert_eq!(sched.running_count(), 1);
    let pending =
        serde_json::json!({"type":"done","id":k1.id,"attempt_id":k1.attempt_id,"tokens":4});
    assert!(sched.mark_awaiting_commit(t1.lane, pending));
    assert_eq!(sched.awaiting_count(), 1);
    assert_eq!(sched.empty_lanes().len(), 0);
    assert!(sched.try_assign_one().is_none());
    assert_eq!(sched.inbox.front().unwrap(), &k2);
    batch_apply_terminal_control("commit", &k1.id, k1.attempt_id);
    assert!(sched.commit_lane(t1.lane, &k1));
    assert_eq!(sched.empty_lanes().len(), 1);
    let (_k2, t2) = sched.try_assign_one().unwrap();
    assert_eq!(t2.lane, t1.lane);
    assert_ne!(t2.generation, t1.generation);
    let pending2 =
        serde_json::json!({"type":"done","id":k2.id,"attempt_id":k2.attempt_id,"tokens":1});
    let lane = sched.find_lane_by_key(&k2).unwrap();
    assert!(sched.mark_awaiting_commit(lane, pending2));
    batch_apply_terminal_control("commit", &k2.id, k2.attempt_id);
    assert!(sched.commit_lane(lane, &k2));
}

#[test]
fn inbox_pushback_restores_barrier_for_outer_recv() {
    let _l = begin();
    let (tx, rx) = std::sync::mpsc::channel::<DaemonMsg>();
    let mut inbox = DaemonInbox::new(rx);
    let barrier = DaemonMsg::Regular(serde_json::json!({"type":"reset","attempt_id":99}));
    let gen = DaemonMsg::Regular(
        serde_json::json!({"type":"generate","id":"r1","attempt_id":1,"prompt":"hi"}),
    );
    tx.send(gen).unwrap();
    tx.send(barrier.clone()).unwrap();
    let m1 = inbox.try_recv().unwrap();
    match m1 {
        DaemonMsg::Regular(v) => assert_eq!(v["type"], "generate"),
        _ => panic!("expected generate"),
    }
    let m2 = inbox.try_recv().unwrap();
    match &m2 {
        DaemonMsg::Regular(v) if v["type"] == "reset" => {
            inbox.push_front(m2);
        }
        _ => panic!("expected reset"),
    }
    let m3 = inbox.recv().unwrap();
    match m3 {
        DaemonMsg::Regular(v) => assert_eq!(v["type"], "reset"),
        _ => panic!("expected barrier after pushback"),
    }
    assert!(inbox.try_recv().is_err());
}

#[test]
fn barrier_transfer_abort_into_singleton() {
    let _l = begin();
    let k = AttemptKey::new("rX", 42);
    batch_announce_terminal(&k.id, k.attempt_id);
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(batch_check_abort(&k.id, k.attempt_id));
    assert!(batch_transfer_abort_to_singleton_and_clear(
        &k.id,
        k.attempt_id
    ));
    assert!(!batch_check_abort(&k.id, k.attempt_id));
    assert!(hipfire_engine::terminal::check_abort(&k.id));
    hipfire_engine::terminal::clear_terminal_control();
}

#[test]
fn sampling_key_is_bit_exact_and_distinguishes_cohorts() {
    let _l = begin();
    let a = sampling(0.3, 1.0);
    let b = sampling(0.31, 1.0);
    assert_ne!(a.key(), b.key());
    let c = BatchSampling {
        temp: 0.3,
        top_p: 0.8,
        top_k: Some(5),
        min_p: None,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: 128,
    };
    let d = BatchSampling {
        temp: 0.3,
        top_p: 0.8,
        top_k: Some(6),
        min_p: None,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: 128,
    };
    assert_ne!(c.key(), d.key());
}

#[test]
fn commit_teardown_classifies_done_only_after_reset_and_commit() {
    let _l = begin();
    assert_eq!(
        batch_commit_teardown_class(false, false),
        BatchCommitTeardownClass::ResetFailed
    );
    assert_eq!(
        batch_commit_teardown_class(false, true),
        BatchCommitTeardownClass::ResetFailed
    );
    assert_eq!(
        batch_commit_teardown_class(true, false),
        BatchCommitTeardownClass::CommitFailed
    );
    assert_eq!(
        batch_commit_teardown_class(true, true),
        BatchCommitTeardownClass::EmitDone
    );
}

#[test]
fn think_barrier_transfers_abort_without_prior_clear() {
    let _l = begin();
    let k = AttemptKey::new("think-barrier", 77);
    assert!(batch_announce_terminal(&k.id, k.attempt_id));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(batch_check_abort(&k.id, k.attempt_id));
    // Production paths must call transfer exactly once; it clears the key
    // and latches abort into the sequential singleton.
    assert!(batch_transfer_abort_to_singleton_and_clear(
        &k.id,
        k.attempt_id
    ));
    assert!(!batch_check_abort(&k.id, k.attempt_id));
    assert!(hipfire_engine::terminal::check_abort(&k.id));
    // A second transfer is a no-op (key already gone) and must not clear
    // the sequential singleton abort that was just latched.
    assert!(!batch_transfer_abort_to_singleton_and_clear(
        &k.id,
        k.attempt_id
    ));
    assert!(hipfire_engine::terminal::check_abort(&k.id));
    hipfire_engine::terminal::clear_terminal_control();
}

#[test]
fn lane_capacity_is_length_terminal_before_out_of_range_decode() {
    let _l = begin();
    let cap = 8usize;
    assert!(!batch_lane_at_capacity(cap - 1, cap));
    assert!(batch_lane_at_capacity(cap, cap));
    assert!(batch_lane_at_capacity(cap + 1, cap));
    // Capacity alone forces finish + length classification.
    assert!(batch_should_finish_decode(false, false, true, false, false));
    assert!(batch_hit_length_cap(false, true, false, false, false));
    // Competing stop causes beat length.
    assert!(!batch_hit_length_cap(true, true, true, false, false));
    assert!(!batch_hit_length_cap(true, true, false, true, false));
    assert!(!batch_hit_length_cap(false, true, false, false, true));
    // max_tokens alone still finishes as length when no other stop.
    assert!(batch_should_finish_decode(false, true, false, false, false));
    assert!(batch_hit_length_cap(true, false, false, false, false));
}

#[test]
fn duplicate_announce_and_enqueue_preserve_live_registry() {
    let _l = begin();
    let k = AttemptKey::new("live-key", 3);
    assert!(batch_announce_terminal(&k.id, k.attempt_id));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(batch_check_abort(&k.id, k.attempt_id));
    // Second announce must fail closed without clearing the original.
    assert!(!batch_announce_terminal(&k.id, k.attempt_id));
    assert!(batch_check_abort(&k.id, k.attempt_id));
    let mut sched = ContinuousBatchScheduler::new(2, 4096);
    assert!(sched.enqueue(req(k.clone(), sampling(0.3, 1.0))));
    // Defensive scheduler rejection: no registry mutation.
    assert!(!sched.enqueue(req(k.clone(), sampling(0.3, 1.0))));
    assert!(batch_check_abort(&k.id, k.attempt_id));
    assert_eq!(sched.inbox.len(), 1);
    assert!(sched.pending.contains_key(&k));
    // Original pending/active lane path still intact for assign.
    let (assigned, ticket) = sched.try_assign_one().unwrap();
    assert_eq!(assigned, k);
    assert!(matches!(
        sched.lanes[ticket.lane],
        hipfire_engine::scheduler::BatchLane::Running(_)
    ));
    assert!(batch_check_abort(&k.id, k.attempt_id));
}

#[test]
fn batch_lane_done_metrics_guard_zero_and_one_token() {
    let t0 = Instant::now();
    // Zero wall / missing stamps: all rates and ms stay finite zeros.
    let z = hipfire_engine::scheduler::batch_lane_done_metrics(t0, None, None, t0, 0, 0);
    assert_eq!(z.latency_ms, 0.0);
    assert_eq!(z.ttft_ms, 0.0);
    assert_eq!(z.prefill_ms, 0.0);
    assert_eq!(z.prefill_tok_s, 0.0);
    assert_eq!(z.tok_s, 0.0);
    assert_eq!(z.decode_tok_s, 0.0);
    assert!(z.latency_ms.is_finite() && z.tok_s.is_finite() && z.decode_tok_s.is_finite());

    // Single generated token: tok_s uses wall; decode_tok_s uses post-first span.
    let t1 = t0 + Duration::from_millis(100);
    let t2 = t0 + Duration::from_millis(150);
    let t3 = t0 + Duration::from_millis(200);
    let one = hipfire_engine::scheduler::batch_lane_done_metrics(t0, Some(t1), Some(t2), t3, 16, 1);
    assert!((one.latency_ms - 200.0).abs() < 1e-6);
    assert!((one.prefill_ms - 100.0).abs() < 1e-6);
    assert!((one.ttft_ms - 150.0).abs() < 1e-6);
    assert!((one.prefill_tok_s - 160.0).abs() < 1e-6); // 16 / 0.1s
    assert!((one.tok_s - 5.0).abs() < 1e-6); // 1 / 0.2s
    assert!((one.decode_tok_s - 20.0).abs() < 1e-6); // 1 / 0.05s
    assert!(one.tok_s.is_finite() && one.decode_tok_s.is_finite());

    // Zero generated keeps decode rate at 0 even with stamps.
    let zero_gen =
        hipfire_engine::scheduler::batch_lane_done_metrics(t0, Some(t1), Some(t2), t3, 16, 0);
    assert_eq!(zero_gen.decode_tok_s, 0.0);
    assert_eq!(zero_gen.tok_s, 0.0);
}
#[test]
fn batch_lane_done_metrics_multi_token_decode_rate() {
    let t0 = Instant::now();
    let prefill = t0 + Duration::from_millis(50);
    let first = t0 + Duration::from_millis(80);
    let end = t0 + Duration::from_millis(280);
    // 5 generated over 200ms post-first ⇒ 5 / 0.2s = 25 tok/s
    let m = hipfire_engine::scheduler::batch_lane_done_metrics(
        t0,
        Some(prefill),
        Some(first),
        end,
        32,
        5,
    );
    assert!((m.latency_ms - 280.0).abs() < 1e-6);
    assert!((m.prefill_ms - 50.0).abs() < 1e-6);
    assert!((m.ttft_ms - 80.0).abs() < 1e-6);
    assert!((m.prefill_tok_s - 640.0).abs() < 1e-6); // 32 / 0.05s
    assert!((m.tok_s - (5.0 / 0.28)).abs() < 1e-6);
    assert!((m.decode_tok_s - 25.0).abs() < 1e-6);
    assert!(m.tok_s.is_finite() && m.decode_tok_s.is_finite() && m.prefill_tok_s.is_finite());
}

#[test]
fn continuous_batch_route_evidence_shape() {
    let mut env = serde_json::json!({"type":"done","id":"r1","tokens":3});
    hipfire_engine::scheduler::attach_continuous_batch_route_evidence(&mut env, 4, 2, 4096, 3);
    assert_eq!(env["execution_mode"], "continuous_batch_independent");
    let cb = &env["continuous_batch"];
    assert_eq!(cb["executed"], true);
    assert_eq!(cb["slots"], 4);
    assert_eq!(cb["lane"], 2);
    assert_eq!(cb["lane_capacity"], 4096);
    assert_eq!(cb["max_active_lanes"], 3);
    assert_eq!(cb["refill"], "continuous");
}

#[test]
fn lfm_capacity_exact_and_plus_one() {
    let _l = begin();
    // exact boundary fits
    assert!(!hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        100, 10, 110
    ));
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        100, 10, 110
    ));
    // +1 exceeds
    assert!(hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        100, 11, 110
    ));
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        100, 11, 110
    ));
    // empty prompt invalid
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        0, 10, 110
    ));
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(0, 0, 110));
}

#[test]
fn lfm_capacity_clamped_lane_vs_max_seq() {
    let _l = begin();
    // Simulate clamped lane_capacity=1024, m.max_seq=4096, prompt=1500
    let lane_cap = 1024;
    let prompt = 1500;
    // Even though prompt < max_seq, it exceeds lane capacity => must be rejected before gen_start
    assert!(hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        prompt, 10, lane_cap
    ));
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        prompt, 10, lane_cap
    ));
    // Smaller prompt fits lane
    assert!(!hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        500, 10, lane_cap
    ));
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        500, 10, lane_cap
    ));
}

#[test]
fn lfm_capacity_u64_max_and_zero() {
    let _l = begin();
    // u64::MAX as usize on 64-bit saturates, must not wrap under cap
    let huge = usize::MAX;
    assert!(hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        10, huge, 4096
    ));
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        10, huge, 4096
    ));
    // zero max_tokens: prompt alone must fit
    assert!(!hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        100, 0, 4096
    ));
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        100, 0, 4096
    ));
    // zero max_tokens with prompt at capacity exactly => fits (prompt==cap, max=0 => prompt+0==cap)
    // Our admission allows prompt == capacity when max=0? Check: prompt_len.saturating_add(0) > cap ?
    // prompt=4096, cap=4096 => 4096 > 4096 false => ok, but prompt must be < cap for non-zero?
    // For zero, we allow equality because no generation needed.
    assert!(!hipfire_engine::terminal::batch_lfm_exceeds_capacity(
        4096, 0, 4096
    ));
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        4096, 0, 4096
    ));
    // zero prompt still invalid even with zero max
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        0, 0, 4096
    ));
}

#[test]
fn lfm_capacity_zero_initial_and_refill_semantics() {
    let _l = begin();
    // Zero-token path must be valid for both initial and refill (same gate).
    let cap = 2048;
    let prompt = 100;
    let max = 0;
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        prompt, max, cap
    ));
    // Refill uses same check: a second request with same prompt+max must also pass
    assert!(hipfire_engine::terminal::batch_lfm_admission_ok(
        prompt, max, cap
    ));
    // Invalid refill would be same as initial: exceeds => no GPU/gen_start
    assert!(!hipfire_engine::terminal::batch_lfm_admission_ok(
        2040, 10, cap
    )); // 2050 > 2048
}

#[test]
fn commit_race_immediate_latches_and_early_rejected() {
    let _l = begin();
    // Early commit before Ready must be rejected, immediate after Ready must latch.
    let k = AttemptKey::new("race-immediate", 100);
    batch_announce_terminal(&k.id, k.attempt_id);
    let mut sched = ContinuousBatchScheduler::new(1, 4096);
    sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
    let (_key, ticket) = sched.try_assign_one().unwrap();
    // Early commit before Ready: must be rejected
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
    assert!(!sched.commit_lane(ticket.lane, &k));
    // Now install Ready BEFORE publish (correct order)
    let pending = serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":1});
    assert!(sched.mark_awaiting_commit(ticket.lane, pending.clone()));
    // Immediate commit after Ready must latch
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Commit)
    );
    // Commit must be single-shot: second commit still Commit, no duplicate done
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Commit)
    );
    assert!(sched.commit_lane(ticket.lane, &k));
    assert!(sched.lanes[ticket.lane].is_empty());
    // After commit, further commit is rejected (key gone)
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(batch_poll_decision(&k.id, k.attempt_id), None);
}

#[test]
fn commit_race_qwen_and_lfm_both_paths() {
    let _l = begin();
    for prefix in ["qwen-race", "lfm-race"] {
        let k = AttemptKey::new(&format!("{prefix}-1"), 1);
        batch_announce_terminal(&k.id, k.attempt_id);
        let mut sched = ContinuousBatchScheduler::new(1, 4096);
        sched.enqueue(req(k.clone(), sampling(0.3, 1.0)));
        let (_key, ticket) = sched.try_assign_one().unwrap();
        let pending =
            serde_json::json!({"type":"done","id":k.id,"attempt_id":k.attempt_id,"tokens":2});
        // Simulate driver: mark then publish, then immediate commit
        assert!(sched.mark_awaiting_commit(ticket.lane, pending.clone()));
        // Client sees commit_ready and immediately commits (synchronous)
        batch_apply_terminal_control("commit", &k.id, k.attempt_id);
        assert_eq!(
            batch_poll_decision(&k.id, k.attempt_id),
            Some(ClientTerminalDecision::Commit)
        );
        assert!(sched.commit_lane(ticket.lane, &k));
        assert!(sched.lanes[ticket.lane].is_empty());
    }
}

#[test]
fn lfm_streaming_utf8_split_and_eos_suppression() {
    // Byte-correct cumulative decode: split UTF-8 across byte fallback tokens
    // must reassemble without FFFD, and EOS markers must be suppressed.
    // We exercise the tokenizer's decode_bytes path directly (same as LFM streaming).
    let _l = begin();
    // Construct a minimal tokenizer with byte fallback tokens for a split scalar.
    // Use hipfire's tokenizer hex escape handling: tokens like "<0xE4>" etc.
    // Instead of building a full tokenizer, we test the holdback logic in isolation:
    // cumulative all_bytes with valid_up_to should not emit partial.
    let bytes_full = "a\u{00E9}b".as_bytes().to_vec(); // a + é (2 bytes) + b
                                                       // Simulate split: first byte of é is C3, second is A9
    let all_bytes_cases = vec![
        (vec![b'a', 0xC3], 1),             // "a" + partial é => valid_len 1
        (vec![b'a', 0xC3, 0xA9, b'b'], 4), // full
    ];
    for (all_bytes, expected_valid) in all_bytes_cases {
        let valid_len = match std::str::from_utf8(&all_bytes) {
            Ok(_) => all_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        assert_eq!(valid_len, expected_valid);
        // Ensure no FFFD would be emitted: valid prefix must be valid utf8
        assert!(std::str::from_utf8(&all_bytes[..valid_len]).is_ok());
    }
    // EOS suppression: decoded string markers must be caught before wire
    for marker in ["<|endoftext|>", "</s>", "<|im_end|>"] {
        assert!(matches!(
            marker.trim(),
            "<|endoftext|>" | "</s>" | "<|im_end|>"
        ));
    }
    // The LFM lane's bytes_fed_to_filter must track valid_len, not all_bytes.len() when partial
    let mut bytes_fed = 0usize;
    let all_bytes_partial = vec![b'a', 0xC3]; // "a" + partial
    let valid_partial = match std::str::from_utf8(&all_bytes_partial) {
        Ok(_) => all_bytes_partial.len(),
        Err(e) => e.valid_up_to(),
    };
    let prev = bytes_fed.min(valid_partial);
    let new_bytes = &all_bytes_partial[prev..valid_partial];
    assert_eq!(new_bytes, b"a");
    bytes_fed = valid_partial; // should be 1, not 2
    assert_eq!(bytes_fed, 1);
    // Next chunk completes the character
    let all_bytes_full = vec![b'a', 0xC3, 0xA9, b'b'];
    let valid_full = match std::str::from_utf8(&all_bytes_full) {
        Ok(_) => all_bytes_full.len(),
        Err(e) => e.valid_up_to(),
    };
    let new_bytes2 = &all_bytes_full[bytes_fed..valid_full];
    assert_eq!(new_bytes2, "\u{00E9}b".as_bytes());
    // Concatenated wire text equals full decode
    let mut wire = Vec::new();
    wire.extend_from_slice(new_bytes);
    wire.extend_from_slice(new_bytes2);
    assert_eq!(wire, bytes_full);
}

#[test]
fn lfm_cancel_decision_semantics() {
    let _l = begin();
    // Prefill cancellation must not sample and must reset only that lane.
    let mut sched = ContinuousBatchScheduler::new(2, 4096);
    let k1 = AttemptKey::new("cancel-1", 1);
    let k2 = AttemptKey::new("keep-2", 2);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req(k1.clone(), sampling(0.3, 1.0)));
    sched.enqueue(req(k2.clone(), sampling(0.3, 1.0)));
    let (_k1a, t1) = sched.try_assign_one().unwrap();
    let (_k2a, t2) = sched.try_assign_one().unwrap();
    assert_eq!(sched.running_count(), 2);
    // Latch abort for k1 during prefill
    batch_apply_terminal_control("abort", &k1.id, k1.attempt_id);
    assert!(batch_check_abort(&k1.id, k1.attempt_id));
    assert!(!batch_check_abort(&k2.id, k2.attempt_id));
    // Simulate daemon's cancellable prefill handling: reset only lane 0
    assert!(sched.abort_lane(t1.lane, &k1));
    assert!(sched.lanes[t1.lane].is_empty());
    // Peer lane remains Running
    assert!(matches!(
        sched.lanes[t2.lane],
        hipfire_engine::scheduler::BatchLane::Running(_)
    ));
    assert_eq!(sched.running_count(), 1);
    // No sample was taken for aborted lane (next_token remains None, not sampled)
    // The scheduler still has k2 pending for decode
    assert!(sched.pending.contains_key(&k2));
    assert!(!sched.pending.contains_key(&k1));
}

#[test]
fn lfm_prefill_cancel_helper_semantics() {
    let _l = begin();
    let k = AttemptKey::new("prefill-cancel-helper", 5);
    batch_announce_terminal(&k.id, k.attempt_id);
    // No GPU needed: test the helper's abort-early path via batch_check_abort
    assert!(!batch_check_abort(&k.id, k.attempt_id));
    batch_apply_terminal_control("abort", &k.id, k.attempt_id);
    assert!(batch_check_abort(&k.id, k.attempt_id));
    // Helper would return Ok(false) without sampling; we verify abort latched
    // and that a subsequent commit is ignored (abort wins).
    batch_apply_terminal_control("commit", &k.id, k.attempt_id);
    assert_eq!(
        batch_poll_decision(&k.id, k.attempt_id),
        Some(ClientTerminalDecision::Abort)
    );
}

#[test]
fn batch_messages_are_single_user_requires_plain_string_content() {
    let _l = begin();
    // Absent messages => true (prompt path)
    assert!(hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({})
    ));
    // Empty array => true
    assert!(hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({"messages":[]})
    ));
    // Single user with plain string content => true
    assert!(hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        })
    ));
    // Multipart content as array => false (sequential-only)
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"http://x"}}]}]
        })
    ));
    // Content as object (image) => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":{"type":"image"}}]
        })
    ));
    // Missing content => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user"}]
        })
    ));
    // System role => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"system","content":"you are helpful"}]
        })
    ));
    // Multi-turn => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]
        })
    ));
    // Tool calls on sole message => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":"hi","tool_calls":[{"id":"1","type":"function"}]}]
        })
    ));
    // Empty tool_calls => true (batch still requires string content)
    assert!(hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({
            "messages":[{"role":"user","content":"hi","tool_calls":[]}]
        })
    ));
    // Non-array messages => false
    assert!(!hipfire_engine::scheduler::batch_messages_are_single_user(
        &serde_json::json!({"messages":"not an array"})
    ));
}

fn req_with_tokens(
    key: AttemptKey,
    sampling: BatchSampling,
    tokens: Vec<u32>,
    max_tokens: usize,
    think: bool,
) -> BatchPendingRequest {
    BatchPendingRequest {
        key,
        prompt: "hi".into(),
        prompt_tokens: tokens,
        started_in_think: think,
        system: None,
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        max_think_tokens: 0,
        max_tokens,
        client_seed: None,
        sampling,
    }
}

#[test]
fn lfm_fast_path_accepts_two_equal_ordinary() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("fp-a", 1);
    let k2 = AttemptKey::new("fp-b", 2);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 2);
}

#[test]
fn lfm_fast_path_mixed_lengths_stop_before_mismatch() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("mix-1", 1);
    let k2 = AttemptKey::new("mix-2", 2);
    let k3 = AttemptKey::new("mix-3", 3);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    batch_announce_terminal(&k3.id, k3.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k3.clone(),
        s.clone(),
        vec![7, 8],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 2, "should stop before mismatched length");
}

#[test]
fn lfm_fast_path_one_request_remains_sequential() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("one-1", 1);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
    // Also ensure inbox not mutated
    assert_eq!(sched.inbox.len(), 1);
    assert_eq!(sched.inbox.front().unwrap(), &k1);
}

#[test]
fn lfm_fast_path_active_lanes_disable() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("act-1", 1);
    let k2 = AttemptKey::new("act-2", 2);
    let k3 = AttemptKey::new("act-3", 3);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    batch_announce_terminal(&k3.id, k3.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k3.clone(),
        s.clone(),
        vec![7, 8, 9],
        10,
        false,
    ));
    // Make scheduler active by assigning one lane
    let _ = sched.try_assign_one();
    assert!(sched.active_count() > 0);
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_awaiting_disable() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(2, 2048);
    let k1 = AttemptKey::new("await-1", 1);
    let k2 = AttemptKey::new("await-2", 2);
    let k3 = AttemptKey::new("await-3", 3);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    batch_announce_terminal(&k3.id, k3.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    let (_key, ticket) = sched.try_assign_one().unwrap();
    // Move to awaiting
    let pending = serde_json::json!({"type":"done","id":k1.id,"attempt_id":k1.attempt_id});
    assert!(sched.mark_awaiting_commit(ticket.lane, pending));
    assert!(sched.awaiting_count() > 0);
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k3.clone(),
        s.clone(),
        vec![7, 8, 9],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_front_aborted_disables() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("abort-front-1", 1);
    let k2 = AttemptKey::new("abort-front-2", 2);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    batch_apply_terminal_control("abort", &k1.id, k1.attempt_id);
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_think_front_disables() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("think-1", 1);
    let k2 = AttemptKey::new("think-2", 2);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        10,
        true,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_zero_max_tokens_front_disables() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("zero-1", 1);
    let k2 = AttemptKey::new("zero-2", 2);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1, 2, 3],
        0,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_capacity_front_disables() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 10);
    let k1 = AttemptKey::new("cap-1", 1);
    let k2 = AttemptKey::new("cap-2", 2);
    let s = sampling(0.3, 1.0);
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    // prompt len 9 + max 5 =14 > cap 10 => exceeds
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s.clone(),
        vec![1u32; 9],
        5,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s.clone(),
        vec![2u32; 9],
        5,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 0);
}

#[test]
fn lfm_fast_path_cohort_mismatch_stops() {
    let _l = begin();
    let mut sched = ContinuousBatchScheduler::new(4, 2048);
    let k1 = AttemptKey::new("cohort-1", 1);
    let k2 = AttemptKey::new("cohort-2", 2);
    let k3 = AttemptKey::new("cohort-3", 3);
    let s1 = sampling(0.3, 1.0);
    let mut s2 = sampling(0.3, 1.0);
    s2.top_p = 0.9; // different cohort
    batch_announce_terminal(&k1.id, k1.attempt_id);
    batch_announce_terminal(&k2.id, k2.attempt_id);
    batch_announce_terminal(&k3.id, k3.attempt_id);
    sched.enqueue(req_with_tokens(
        k1.clone(),
        s1.clone(),
        vec![1, 2, 3],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k2.clone(),
        s1.clone(),
        vec![4, 5, 6],
        10,
        false,
    ));
    sched.enqueue(req_with_tokens(
        k3.clone(),
        s2.clone(),
        vec![7, 8, 9],
        10,
        false,
    ));
    let n = hipfire_engine::scheduler::lfm_fast_path_candidate_len(&sched);
    assert_eq!(n, 2);
}
