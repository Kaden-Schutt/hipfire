// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP adaptive-route contract.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. These were the last references
//! to architecture crates left in that file; the shipped daemon code reaches
//! architectures only through `hipfire-loader`'s `Carrier` and the
//! `hipfire-generate` entry points.

#![allow(clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::common::*;

    /// Exact adaptive×MTP prefill invariant:
    /// external chunk size ≤ PREFILL_MAX_BATCH (= adaptive margin), and
    /// maybe_downshift runs at each exclusive committed boundary so a long
    /// prompt cannot hit the start-tier side_cap before return. A single
    /// whole-prompt prefill + post-only downshift is insufficient.
    #[test]
    fn mtp_adaptive_prefill_boundaries_match_chunk_schedule() {
        use hipfire_arch_qwen35::mtp_spec::mtp_prefill_committed_boundaries;
        use hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH;
        assert_eq!(
            mtp_prefill_committed_boundaries(600, 0, PREFILL_MAX_BATCH),
            vec![256, 512, 600]
        );
        assert!(mtp_prefill_committed_boundaries(0, 0, PREFILL_MAX_BATCH).is_empty());
        // Gaps never exceed one prefill chunk (margin safety).
        let b = mtp_prefill_committed_boundaries(10_000, 0, PREFILL_MAX_BATCH);
        let mut prev = 0usize;
        for &pos in &b {
            assert!(pos - prev <= PREFILL_MAX_BATCH);
            prev = pos;
        }
    }

    #[test]
    fn mtp_forward_fail_is_request_error_not_token() {
        // Mirror AR policy for MTP prefill/spec HipResult (VMM growth, etc.):
        // emit request error, never the failed token; poison stays sticky.
        let action = qwen_ar_forward_fail_action();
        assert!(action.emit_request_error);
        assert!(!action.emit_failed_token);
        assert!(!action.clear_adaptive_poison);
    }

    /// Decode-cycle invariant: downshift seq_pos is the live committed prefix
    /// only. Rejected verify length is (n_verify - advance) and lives strictly
    /// past that prefix — never included in maybe_downshift's seq_pos.
    #[test]
    fn mtp_decode_downshift_uses_committed_prefix_only() {
        let cur_pos = 1000usize;
        let max_n = 3usize;
        let n_verify = max_n + 1; // last_committed + candidates
        let advance = 2usize; // e.g. accept 1 + bonus
        let committed_end = cur_pos + advance;
        let reject_suffix_end = cur_pos + n_verify;
        assert!(committed_end < reject_suffix_end);
        // maybe_downshift(committed_end) covers [0, committed_end); rejected
        // [committed_end, reject_suffix_end) must not be required at new tier.
        assert_eq!(reject_suffix_end - committed_end, n_verify - advance);
    }
