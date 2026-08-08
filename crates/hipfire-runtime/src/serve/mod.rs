// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Concurrent serve: several agents served at once by the multi-slot engine.
//
// `hipfire serve` already exposes an OpenAI-compatible /v1/chat/completions
// over HTTP, and HTTP is already concurrent. What serialises requests today is
// behind it: one `Engine` (a single daemon process) guarded by a
// `Mutex<ServeRuntime>`. `SlotEngine` is an alternative backend that lives
// OUTSIDE that mutex — which is the entire point — so requests overlap.

// The engine LOOP lives in `hipfire-arch-qwen35::serve_engine`, not here:
// it needs `forward_batch_slots_graphed`, and that crate already depends on
// this one, so putting the loop here would be a dependency cycle. This module
// owns the protocol types both sides share — the same layering reason
// `swap::snapshot` takes an opaque buffer slice instead of a `DeltaNetState`.

use std::sync::mpsc::Sender;

/// One request handed to the engine.
///
/// `reply` is the client's own channel: the engine streams this request's
/// events down it and nothing else. A dropped receiver is how the engine
/// learns the client is gone.
pub struct SubmitRequest {
    /// Continue an existing session, or `None` to open a new one.
    pub session: Option<u64>,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub reply: Sender<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoneReason {
    Eos,
    MaxTokens,
    /// The client's receiver was dropped. The session is closed and its slot
    /// freed rather than generating into a void.
    ClientGone,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Accepted { session: u64 },
    Token { id: u32 },
    Done { reason: DoneReason },
    Rejected { reason: String },
}

/// Post an event to a client, reporting a vanished receiver as an error.
///
/// The engine must be able to see this: a request whose client has gone still
/// holds a slot, and on a 4-slot box that is 25% of capacity generating output
/// nobody will read.
pub fn send_event(tx: &Sender<Event>, e: Event) -> Result<(), String> {
    tx.send(e)
        .map_err(|_| "client receiver dropped".to_string())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub admitted: usize,
    pub rejected: usize,
    pub evictions: usize,
    pub restores: usize,
}

impl EngineStats {
    pub fn note_admitted(&mut self) {
        self.admitted += 1;
    }
    pub fn note_rejected(&mut self) {
        self.rejected += 1;
    }
    pub fn note_eviction(&mut self) {
        self.evictions += 1;
    }
    pub fn note_restore(&mut self) {
        self.restores += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn a_request_whose_client_vanished_is_detected_before_work_is_done() {
        let (tx, rx) = channel::<Event>();
        drop(rx);
        assert!(
            send_event(&tx, Event::Token { id: 1 }).is_err(),
            "a dropped receiver must be visible so the engine can free the slot"
        );
    }

    #[test]
    fn a_live_client_receives_its_events_in_order() {
        let (tx, rx) = channel::<Event>();
        send_event(&tx, Event::Accepted { session: 4 }).unwrap();
        send_event(&tx, Event::Token { id: 7 }).unwrap();
        send_event(
            &tx,
            Event::Done {
                reason: DoneReason::Eos,
            },
        )
        .unwrap();
        assert_eq!(rx.recv().unwrap(), Event::Accepted { session: 4 });
        assert_eq!(rx.recv().unwrap(), Event::Token { id: 7 });
        assert_eq!(
            rx.recv().unwrap(),
            Event::Done {
                reason: DoneReason::Eos
            }
        );
    }

    #[test]
    fn stats_start_at_zero_and_count_each_outcome() {
        let mut s = EngineStats::default();
        s.note_admitted();
        s.note_admitted();
        s.note_rejected();
        assert_eq!(s.admitted, 2);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.restores, 0);
    }
}
