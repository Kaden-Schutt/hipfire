// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Adaptive eviction prefill-chunk contract.
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

    use qwen_ar_eviction_prefill_chunk_limit;
    use hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH;

    #[test]
    fn staging_uses_adaptive_boundaries_until_handoff() {
        let window = 2048 + 128;
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(0, window, true),
            PREFILL_MAX_BATCH
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(8192 - PREFILL_MAX_BATCH, window, true),
            PREFILL_MAX_BATCH
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(2048, window, false),
            128
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(window, window, false),
            1
        );
    }
