// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SessionTable — which session holds which slot.
//
// Two properties prevent whole classes of bug:
//
// 1. Admission runs BEFORE the slot is taken, so a rejected request leaves
//    the pool untouched. Otherwise a rejected client silently consumes
//    capacity.
// 2. Session ids are never reused: a monotonically increasing counter, not
//    a free-list index. A stale id from a closed session resolves to
//    `None`, never silently addressing whoever now holds that slot.

use std::collections::HashMap;

use crate::admission::{AdmissionController, AdmitError};
use crate::prefix::{plan_turn, TurnPlan};
use rdna_compute::slot_pool::{SlotId, SlotPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

pub struct Session {
    pub slot: SlotId,
    pub granted_ctx: usize,
    pub tokens: Vec<u32>,
    pub next_pos: usize,
}

#[derive(Default)]
pub struct SessionTable {
    sessions: HashMap<u64, Session>,
    next_id: u64,
}

impl SessionTable {
    /// Admit a session and assign it a slot.
    ///
    /// Admission runs FIRST: only a successful `adm.admit` takes a slot from
    /// `pool`, so a rejected request leaves the pool untouched. If admission
    /// succeeds but the pool has no free slot, the admitted budget is handed
    /// straight back so the two stay consistent.
    pub fn open(
        &mut self,
        pool: &mut SlotPool,
        adm: &mut AdmissionController,
        requested_ctx: usize,
    ) -> Result<SessionId, AdmitError> {
        let granted_ctx = adm.admit(requested_ctx)?;
        let slot = match pool.acquire() {
            Some(slot) => slot,
            None => {
                adm.release(granted_ctx);
                return Err(AdmitError::PoolFull);
            }
        };
        // Monotonically increasing, never reused: a stale id from a closed
        // session must resolve to `None`, never silently address whoever now
        // holds that slot.
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(
            id,
            Session {
                slot,
                granted_ctx,
                tokens: Vec::new(),
                next_pos: 0,
            },
        );
        Ok(SessionId(id))
    }

    /// Close a session, returning its slot and budget together.
    pub fn close(&mut self, pool: &mut SlotPool, adm: &mut AdmissionController, id: SessionId) {
        if let Some(session) = self.sessions.remove(&id.0) {
            pool.release(session.slot);
            adm.release(session.granted_ctx);
        }
    }

    pub fn get(&self, id: SessionId) -> Option<&Session> {
        self.sessions.get(&id.0)
    }

    pub fn get_mut(&mut self, id: SessionId) -> Option<&mut Session> {
        self.sessions.get_mut(&id.0)
    }

    /// Start a turn: decide how much of `prompt` this session's slot already
    /// holds, rewind to that point, and report what remains to prefill.
    ///
    /// The rewind moves no data. Lowering `seq_len` is enough because KV past
    /// that point is overwritten by the next prefill, and `positions[]` -- not
    /// `seq_len` -- is what bounds attention per row.
    ///
    /// The caller must then prefill `prompt[plan.reused..]` and append those
    /// tokens to `session.tokens`.
    ///
    /// Only ever *lowers* `seq_len` and truncates tokens, so the worst
    /// outcome of a wrong answer here is a full re-prefill, never stale KV
    /// being read as valid.
    pub fn begin_turn(
        &mut self,
        pool: &mut SlotPool,
        id: SessionId,
        prompt: &[u32],
    ) -> Result<TurnPlan, String> {
        let session = self
            .sessions
            .get_mut(&id.0)
            .ok_or_else(|| format!("begin_turn: unknown session {}", id.0))?;
        let held = pool.descriptors()[session.slot.0].seq_len as usize;
        let plan = plan_turn(&session.tokens, held, prompt);
        pool.set_seq_len(session.slot, plan.reused)
            .map_err(|e| format!("begin_turn: {e}"))?;
        session.tokens.truncate(plan.reused);
        session.next_pos = plan.reused;
        Ok(plan)
    }

    pub fn active(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{AdmissionController, ModelFootprint};
    use rdna_compute::slot_pool::SlotPool;

    const GIB: u64 = 1024 * 1024 * 1024;
    const PPB: usize = 1088;

    fn rig(n_slots: usize) -> (SlotPool, AdmissionController, SessionTable) {
        let pool = SlotPool::new(n_slots, 4096, PPB).unwrap();
        let adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: GIB,
                kv_bytes_per_token: 1024,
            },
            32 * GIB,
        );
        (pool, adm, SessionTable::default())
    }

    #[test]
    fn open_assigns_a_slot_and_close_returns_it() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        assert_eq!(t.active(), 1);
        // The single slot is taken.
        assert!(t.open(&mut pool, &mut adm, 1024).is_err());
        t.close(&mut pool, &mut adm, id);
        assert_eq!(t.active(), 0);
        // And is reusable.
        t.open(&mut pool, &mut adm, 1024)
            .expect("slot must be reusable");
    }

    #[test]
    fn a_rejected_admission_does_not_consume_a_slot() {
        let (mut pool, mut adm, mut t) = rig(2);
        // Far beyond the budget.
        assert!(t.open(&mut pool, &mut adm, 100_000_000).is_err());
        assert_eq!(
            t.active(),
            0,
            "a rejected session must leave the pool untouched"
        );
        t.open(&mut pool, &mut adm, 1024)
            .expect("pool must still have both slots");
    }

    #[test]
    fn sessions_keep_independent_token_history() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.get_mut(a).unwrap().tokens.extend_from_slice(&[1, 2, 3]);
        t.get_mut(b).unwrap().tokens.push(9);
        assert_eq!(t.get(a).unwrap().tokens, vec![1, 2, 3]);
        assert_eq!(t.get(b).unwrap().tokens, vec![9]);
    }

    #[test]
    fn closing_frees_budget_for_a_later_session() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let before = adm.used_bytes();
        t.close(&mut pool, &mut adm, a);
        assert!(adm.used_bytes() < before, "close must return budget");
    }

    #[test]
    fn begin_turn_rewinds_the_slot_and_reports_the_suffix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        // Turn 1: four tokens are prefilled and recorded.
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot, 4).unwrap();

        // Turn 2 continues the same conversation.
        let plan = t.begin_turn(&mut pool, id, &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(plan.reused, 4);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.next_pos, 4, "next_pos must resume at the reuse point");
        assert_eq!(
            s.tokens,
            vec![1, 2, 3, 4],
            "tokens truncated to the reuse point"
        );
        assert_eq!(pool.descriptors()[s.slot.0].seq_len, 4);
    }

    #[test]
    fn begin_turn_on_divergence_rewinds_to_the_common_prefix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot, 4).unwrap();

        let plan = t.begin_turn(&mut pool, id, &[1, 2, 7, 8]).unwrap();
        assert_eq!(plan.reused, 2);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.tokens, vec![1, 2], "diverged tokens are dropped");
        assert_eq!(s.next_pos, 2);
        assert_eq!(
            pool.descriptors()[s.slot.0].seq_len,
            2,
            "the slot must forget the diverged KV"
        );
    }

    #[test]
    fn begin_turn_on_an_unknown_session_is_an_error_not_a_panic() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, id);
        assert!(t.begin_turn(&mut pool, id, &[1, 2]).is_err());
    }

    #[test]
    fn a_closed_session_id_is_not_reusable_by_accident() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, a);
        assert!(t.get(a).is_none(), "a closed session must not resolve");
    }
}
