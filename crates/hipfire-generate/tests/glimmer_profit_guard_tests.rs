// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer profit guard tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    use hipfire_generate::{dense::glimmer_profit_ledger_after_bonus_decode, dense::glimmer_profit_ledger_post_window, dense::glimmer_profit_ledger_route_prediction, dense::GlimmerProfitGuardStatus, dense::GlimmerProfitProbeKind, dense::GlimmerSpecProfitGuard};

    /// Drive four identical measured windows that sum to (s_total, p_total), then
    /// apply ar_probe_ns. Returns the guard after observe_probe.
    fn eval_group(g: &mut hipfire_generate::dense::GlimmerSpecProfitGuard, s_total: u128, p_total: u128, ar_probe_ns: u128) {
        // Split evenly across four windows; remainder on the last.
        let s_each = s_total / 4;
        let p_each = (p_total / 4) as usize;
        let s_last = s_total - s_each * 3;
        let p_last = (p_total - (p_each as u128) * 3) as usize;
        for i in 0..4 {
            let s = if i == 3 { s_last } else { s_each };
            let p = if i == 3 { p_last } else { p_each };
            let kind = g.observe_full_window(s, p);
            if i < 3 {
                assert_eq!(kind, hipfire_generate::dense::GlimmerProfitProbeKind::None, "window {i}");
            } else {
                assert_eq!(kind, hipfire_generate::dense::GlimmerProfitProbeKind::Measured, "window {i}");
            }
        }
        g.observe_probe(ar_probe_ns);
    }

    fn warmup(g: &mut hipfire_generate::dense::GlimmerSpecProfitGuard) {
        assert_eq!(
            g.observe_full_window(1_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
        g.observe_probe(999); // discarded
    }

    #[test]
    fn disabled_never_probes_or_retires() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(false);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Disabled);
        assert!(!g.enabled());
        for _ in 0..20 {
            assert_eq!(
                g.observe_full_window(10_000, 8),
                hipfire_generate::dense::GlimmerProfitProbeKind::None
            );
            g.observe_probe(1);
        }
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.eligible_windows(), 0);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::None);
    }

    #[test]
    fn first_window_is_warmup_and_excluded() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);
        assert_eq!(
            g.observe_full_window(50_000, 16),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
        assert_eq!(g.eligible_windows(), 1);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::Warmup);
        // Warmup probe discarded — no evaluation, no S/P carried.
        g.observe_probe(1);
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.bad_evaluations(), 0);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Monitoring);
        // Next four windows: only the 4th requests Measured.
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Measured
        );
        // Completing the measured probe with S=40k, P=16, A=2500:
        // ratio = 40000/(16*2500) = 1.0 — deadband; one evaluation counted.
        g.observe_probe(2_500);
        assert_eq!(g.evaluations(), 1);
        assert_eq!(g.last_spec_ns(), 40_000);
        assert_eq!(g.last_productive(), 16);
        assert_eq!(g.last_ar_probe_ns(), 2_500);
    }

    #[test]
    fn four_window_cadence() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        // Two full evaluation groups: only every 4th window is Measured.
        let mut measured = 0u32;
        let mut none = 0u32;
        for i in 0..8 {
            let k = g.observe_full_window(1_000, 2);
            match k {
                hipfire_generate::dense::GlimmerProfitProbeKind::Measured => {
                    measured += 1;
                    g.observe_probe(1_000); // ratio = 4000/(8*1000)=0.5 good
                }
                hipfire_generate::dense::GlimmerProfitProbeKind::None => none += 1,
                hipfire_generate::dense::GlimmerProfitProbeKind::Warmup => panic!("unexpected warmup at {i}"),
            }
        }
        assert_eq!(measured, 2);
        assert_eq!(none, 6);
        assert_eq!(g.evaluations(), 2);
    }

    #[test]
    fn boundary_1049_deadband_105_bad_098_reset() {
        // Choose A=1000, P=100 so A*P = 100_000.
        // bad:  S*100 >= 100_000*105 = 10_500_000  => S >= 105_000  (ratio >= 1.05)
        // good: S*100 <= 100_000*98  =  9_800_000  => S <=  98_000  (ratio <= 0.98)
        // deadband: 98_001 ..= 104_999
        // Exactly 1.049: S = 104_900 => left=10_490_000 < 10_500_000 and > 9_800_000.

        // --- 1.049 deadband retains ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        // Seed one bad so deadband retention is observable.
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());
        // 1.049: retain bad_evaluations == 1
        eval_group(&mut g, 104_900, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 2);

        // --- exactly 1.05 is bad ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // --- exactly 0.98 resets ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        eval_group(&mut g, 105_000, 100, 1_000); // bad -> 1
        assert_eq!(g.bad_evaluations(), 1);
        eval_group(&mut g, 98_000, 100, 1_000); // good -> 0
        assert_eq!(g.bad_evaluations(), 0);
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 2);
    }

    #[test]
    fn two_bad_retires_sticky_good_resets_deadband_retains() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);

        // bad #1
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // deadband retains
        eval_group(&mut g, 100_000, 100, 1_000); // ratio = 1.0
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // good resets
        eval_group(&mut g, 98_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 0);

        // two consecutive bads retire
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 2);
        assert!(g.is_retired());
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Retired);
        assert_eq!(g.retire_evaluation(), g.evaluations());
        assert!(g.retire_cycle() > 0);

        // sticky: further windows/probes are inert
        assert_eq!(
            g.observe_full_window(200_000, 1),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        let evals = g.evaluations();
        g.observe_probe(1);
        assert_eq!(g.evaluations(), evals);
        assert!(g.is_retired());
    }

    #[test]
    fn fresh_object_after_retirement_starts_warmup() {
        let mut old = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut old);
        eval_group(&mut old, 105_000, 100, 1_000);
        eval_group(&mut old, 105_000, 100, 1_000);
        assert!(old.is_retired());

        let mut fresh = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        assert_eq!(fresh.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);
        assert!(!fresh.is_retired());
        assert_eq!(
            fresh.observe_full_window(1_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
    }

    #[test]
    fn zero_progress_and_zero_time_ignored() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        // Zero time
        assert_eq!(g.observe_full_window(0, 8), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        // Zero rows
        assert_eq!(
            g.observe_full_window(10_000, 0),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(g.eligible_windows(), 0);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);

        warmup(&mut g);
        // Build three of four measured windows, then inject zeros (ignored).
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(g.observe_full_window(0, 2), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        assert_eq!(
            g.observe_full_window(1_000, 0),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        // Fourth real window still completes the group.
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::Measured
        );
        // Zero probe is not evidence: evaluation not counted.
        g.observe_probe(0);
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.bad_evaluations(), 0);
        // Cadence recovered — next four-window group works.
        eval_group(&mut g, 4_000, 8, 1_000); // ratio 0.5 good
        assert_eq!(g.evaluations(), 1);
    }

    #[test]
    fn bonus_decode_aligns_mirror_prediction_unpushed_until_route() {
        // Post full window: bonus already on mirror, not in KV/capture.
        let commit_end = 100usize;
        let post = hipfire_generate::dense::glimmer_profit_ledger_post_window(commit_end);
        assert_eq!(post.mirror_len, commit_end + 1);
        assert_eq!(post.state_n_tokens, commit_end);

        // Decoding the pending bonus advances state only — prediction not mirrored.
        let after = hipfire_generate::dense::glimmer_profit_ledger_after_bonus_decode(post);
        assert_eq!(after.mirror_len, commit_end + 1);
        assert_eq!(after.state_n_tokens, commit_end + 1);
        assert_eq!(after.mirror_len, after.state_n_tokens);

        // Retire/AR tail keeps prediction unpushed (same ledger).
        assert_eq!(after, hipfire_generate::dense::glimmer_profit_ledger_after_bonus_decode(post));

        // Continue-spec routes the returned prediction once.
        let cont = hipfire_generate::dense::glimmer_profit_ledger_route_prediction(after);
        assert_eq!(cont.mirror_len, commit_end + 2);
        assert_eq!(cont.state_n_tokens, commit_end + 1);
        // Prediction is one-token-ahead again, not yet in state.
        assert_eq!(cont.mirror_len, cont.state_n_tokens + 1);
    }
