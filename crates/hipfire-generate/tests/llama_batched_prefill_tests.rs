// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Llama batched prefill tests.
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

    
    use hipfire_runtime::llama::ModelArch;

    #[test]
    fn route_stays_inside_validated_qwen3_q8_envelope() {
        let cases = [
            ("gfx1100", ModelArch::Qwen3, true, true, false, 256, true),
            ("gfx1201", ModelArch::Qwen3, true, true, false, 4, true),
            ("gfx1200", ModelArch::Qwen3, true, true, false, 256, false),
            ("gfx1100", ModelArch::Llama, true, true, false, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, false, false, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, true, true, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, true, false, 3, false),
            ("gfx1100", ModelArch::Qwen3, false, true, false, 256, false),
        ];
        for (arch, model, enabled, q8, eviction, tokens, expected) in cases {
            assert_eq!(
                llama_qwen3_batched_prefill_eligible(arch, model, enabled, q8, eviction, tokens,),
                expected,
                "arch={arch} model={model:?}",
            );
        }
    }

    #[test]
    fn sampled_prefill_preserves_discarded_xorshift_draws() {
        assert_eq!(llama_prefill_sample_seed(42, 4, 0.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 1, 1.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 4, 1.0), 476_557_059);
    }
