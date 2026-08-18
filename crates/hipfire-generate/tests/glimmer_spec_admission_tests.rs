// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer spec admission tests.
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

    use hipfire_generate::dense::{glimmer_spec_admission, GlimmerSpecMode};

    #[test]
    fn greedy_at_temp_zero() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Greedy);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.01, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Greedy);
    }

    #[test]
    fn chain_sampled_at_temp_one_with_defaults() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
    }

    #[test]
    fn off_when_min_p_present() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, Some(0.05), true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        // zero and None are allowed
        let ok0 = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, Some(0.0), true, false, true);
        assert_eq!(ok0, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
        let ok_none = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, true);
        assert_eq!(ok_none, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
    }

    #[test]
    fn off_when_fast_sample_off() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, false, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_temp_spec_env_off() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, true, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_batched_logits_unavailable() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, false);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        // greedy does NOT require batched logits (still Greedy)
        let g = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.005, None, true, false, false);
        assert_eq!(g, hipfire_generate::dense::GlimmerSpecMode::Greedy);
    }

    #[test]
    fn off_when_max_tokens_one() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 1, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(true, 1, 1.0, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_no_drafter() {
        let m = hipfire_generate::dense::glimmer_spec_admission(false, 16, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(false, 16, 1.0, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn temp_boundary() {
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.01, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.02, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::ChainSampled
        );
        // just above greedy threshold but at/under 1e-6 should be Off, not sampled
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 1e-6, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 5e-7, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
    }
