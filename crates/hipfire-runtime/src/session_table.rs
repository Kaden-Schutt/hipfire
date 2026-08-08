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
            Session { slot, granted_ctx, tokens: Vec::new(), next_pos: 0 },
        );
        Ok(SessionId(id))
    }

    /// Close a session, returning its slot and budget together.
    pub fn close(
        &mut self,
        pool: &mut SlotPool,
        adm: &mut AdmissionController,
        id: SessionId,
    ) {
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
            ModelFootprint { weights_bytes: GIB, kv_bytes_per_token: 1024 },
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
        t.open(&mut pool, &mut adm, 1024).expect("slot must be reusable");
    }

    #[test]
    fn a_rejected_admission_does_not_consume_a_slot() {
        let (mut pool, mut adm, mut t) = rig(2);
        // Far beyond the budget.
        assert!(t.open(&mut pool, &mut adm, 100_000_000).is_err());
        assert_eq!(t.active(), 0, "a rejected session must leave the pool untouched");
        t.open(&mut pool, &mut adm, 1024).expect("pool must still have both slots");
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
    fn a_closed_session_id_is_not_reusable_by_accident() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, a);
        assert!(t.get(a).is_none(), "a closed session must not resolve");
    }
}
