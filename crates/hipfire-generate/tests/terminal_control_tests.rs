// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Terminal control tests.
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

    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::time::Duration;



    /// `hipfire_generate::dense::glimmer_longest_marker_suffix` byte-slices from the end of the pending
    /// buffer looking for a split Harmony marker. It must skip offsets that
    /// land inside a multibyte character.
    ///
    /// Regression: the first version did `&s[s.len() - len..]` unguarded and
    /// panicked with "byte index N is not a char boundary" the moment Glimmer
    /// emitted a non-ASCII character — `×` in an arithmetic reasoning span took
    /// the whole daemon down mid-generation. Markers are pure ASCII, so an
    /// offset inside a multibyte char can never start one.
    #[test]
    fn glimmer_marker_suffix_is_char_boundary_safe() {
        // Each of these ends in (or contains) a multibyte char at a position the
        // reverse scan would probe.
        for s in [
            "17 × 23",
            "café",
            "—",
            "reasoning ×",
            "emoji 😀",
            "mixed ×<|eo",
        ] {
            let n = hipfire_generate::dense::glimmer_longest_marker_suffix(s);
            assert!(
                s.is_char_boundary(s.len() - n),
                "returned len {n} splits a char in {s:?}"
            );
        }
        // Still detects a genuine split marker.
        assert_eq!(hipfire_generate::dense::glimmer_longest_marker_suffix("abc<|eo"), 4);
        assert_eq!(hipfire_generate::dense::glimmer_longest_marker_suffix("abc"), 0);
    }

