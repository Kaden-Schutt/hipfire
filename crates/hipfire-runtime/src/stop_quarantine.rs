// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded raw-byte quarantine for configured stop markers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuarantineTerminal {
    Open,
    Stopped,
    Finished,
}

/// Scans raw bytes for stop markers without exposing a marker prefix.
///
/// Only a proper suffix of the longest configured marker is retained between
/// calls. Once a marker is found, the quarantine is terminal and all later
/// input is discarded.
#[derive(Debug, Clone)]
pub(crate) struct StopQuarantine {
    markers: Vec<Vec<u8>>,
    pending: Vec<u8>,
    max_pending: usize,
    terminal: QuarantineTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QuarantineOutcome {
    Continue { bytes: Vec<u8> },
    Stop { bytes: Vec<u8> },
}

impl StopQuarantine {
    pub(crate) fn new(markers: Vec<Vec<u8>>) -> Self {
        let mut unique_markers = Vec::new();
        for marker in markers.into_iter().filter(|m| !m.is_empty()) {
            if !unique_markers.iter().any(|existing| existing == &marker) {
                unique_markers.push(marker);
            }
        }
        let markers = unique_markers;
        let max_pending = markers
            .iter()
            .map(|marker| marker.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self {
            markers,
            pending: Vec::new(),
            max_pending,
            terminal: QuarantineTerminal::Open,
        }
    }

    /// Feed raw bytes, returning bytes proven not to be part of a stop
    /// marker and whether a stop marker was found.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> QuarantineOutcome {
        if self.terminal != QuarantineTerminal::Open {
            return if self.terminal == QuarantineTerminal::Stopped {
                QuarantineOutcome::Stop { bytes: Vec::new() }
            } else {
                QuarantineOutcome::Continue { bytes: Vec::new() }
            };
        }

        let mut candidate = Vec::with_capacity(self.pending.len() + bytes.len());
        candidate.extend_from_slice(&self.pending);
        candidate.extend_from_slice(bytes);

        if let Some(start) = self.find_stop(&candidate) {
            self.pending.clear();
            self.terminal = QuarantineTerminal::Stopped;
            return QuarantineOutcome::Stop {
                bytes: candidate[..start].to_vec(),
            };
        }

        let unresolved = self.longest_prefix_suffix(&candidate);
        let safe_len = candidate.len() - unresolved;
        self.pending.clear();
        self.pending.extend_from_slice(&candidate[safe_len..]);
        QuarantineOutcome::Continue {
            bytes: candidate[..safe_len].to_vec(),
        }
    }

    /// Complete normally, recovering a pending suffix that turned out not to
    /// be a stop marker. This is consuming and idempotent.
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        if self.terminal != QuarantineTerminal::Open {
            return Vec::new();
        }
        self.terminal = QuarantineTerminal::Finished;
        std::mem::take(&mut self.pending)
    }

    /// Discard pending bytes for an abort/error path. There is no recovery
    /// after this operation.
    pub(crate) fn discard(&mut self) {
        self.pending.clear();
        self.terminal = QuarantineTerminal::Finished;
    }

    pub(crate) fn reset(&mut self) {
        self.pending.clear();
        self.terminal = QuarantineTerminal::Open;
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    fn max_pending_len(&self) -> usize {
        self.max_pending
    }

    fn find_stop(&self, bytes: &[u8]) -> Option<usize> {
        let mut best = None;
        for marker in &self.markers {
            if let Some(start) = memmem(bytes, marker) {
                if best
                    .map(|(best_start, best_len)| {
                        start < best_start || (start == best_start && marker.len() > best_len)
                    })
                    .unwrap_or(true)
                {
                    best = Some((start, marker.len()));
                }
            }
        }
        best.map(|(start, _len)| start)
    }

    fn longest_prefix_suffix(&self, bytes: &[u8]) -> usize {
        let mut longest = 0;
        for marker in &self.markers {
            let max = bytes.len().min(marker.len().saturating_sub(1));
            for len in 1..=max {
                if bytes[bytes.len() - len..] == marker[..len] {
                    longest = longest.max(len);
                }
            }
        }
        longest.min(self.max_pending)
    }
}

fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if haystack.len() < needle.len() {
        return None;
    }
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()] == *needle {
            return Some(start);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain_chunks(markers: &[&[u8]], input: &[u8], cuts: u32) -> (Vec<u8>, bool) {
        let mut q = StopQuarantine::new(markers.iter().map(|m| m.to_vec()).collect());
        let mut safe = Vec::new();
        let mut offset = 0;
        for end in 1..=input.len() {
            if cuts & (1 << (end - 1)) != 0 {
                match q.push(&input[offset..end]) {
                    QuarantineOutcome::Continue { bytes } => safe.extend(bytes),
                    QuarantineOutcome::Stop { bytes } => {
                        safe.extend(bytes);
                        return (safe, true);
                    }
                }
                offset = end;
            }
        }
        if offset < input.len() {
            match q.push(&input[offset..]) {
                QuarantineOutcome::Continue { bytes } => safe.extend(bytes),
                QuarantineOutcome::Stop { bytes } => {
                    safe.extend(bytes);
                    return (safe, true);
                }
            }
        }
        safe.extend(q.finish());
        (safe, false)
    }

    #[test]
    fn exhaustive_chunk_splits_keep_only_safe_prefix_and_stop() {
        let input = b"safe<stop>tail";
        let split_count = input.len() - 1;
        for cuts in 0..(1u32 << split_count) {
            assert_eq!(
                drain_chunks(&[b"<stop>"], input, cuts),
                (b"safe".to_vec(), true),
                "cuts={cuts:#b}"
            );
        }
    }

    #[test]
    fn exhaustive_chunk_splits_stop_on_overlapping_markers() {
        let input = b"safeabab-tail";
        let markers = [b"aba".as_slice(), b"bab".as_slice()];
        let split_count = input.len() - 1;
        for cuts in 0..(1u32 << split_count) {
            assert_eq!(
                drain_chunks(&markers, input, cuts),
                (b"safe".to_vec(), true),
                "cuts={cuts:#b}"
            );
        }
    }

    #[test]
    fn exhaustive_chunk_splits_finish_false_prefix_without_dropping_it() {
        let input = b"safeab";
        let markers = [b"aba".as_slice(), b"bab".as_slice()];
        let split_count = input.len() - 1;
        for cuts in 0..(1u32 << split_count) {
            assert_eq!(
                drain_chunks(&markers, input, cuts),
                (input.to_vec(), false),
                "cuts={cuts:#b}"
            );
        }
    }

    #[test]
    fn normal_finish_flushes_false_marker_prefix() {
        let mut q = StopQuarantine::new(vec![b"<stop>".to_vec()]);
        assert_eq!(
            q.push(b"safe<sto"),
            QuarantineOutcome::Continue {
                bytes: b"safe".to_vec()
            }
        );
        assert_eq!(q.finish(), b"<sto");
        assert_eq!(q.finish(), Vec::<u8>::new());
    }

    #[test]
    fn earliest_marker_wins_and_longest_marker_wins_ties() {
        let mut q = StopQuarantine::new(vec![b"!".to_vec(), b"<stop>".to_vec()]);
        assert_eq!(
            q.push(b"safe!later<stop>"),
            QuarantineOutcome::Stop {
                bytes: b"safe".to_vec()
            }
        );

        let mut q = StopQuarantine::new(vec![b"ab".to_vec(), b"abc".to_vec()]);
        assert_eq!(
            q.push(b"safeabc-tail"),
            QuarantineOutcome::Stop {
                bytes: b"safe".to_vec()
            }
        );
    }

    #[test]
    fn post_stop_push_is_inert() {
        let mut q = StopQuarantine::new(vec![b"<stop>".to_vec()]);
        assert_eq!(
            q.push(b"<stop>"),
            QuarantineOutcome::Stop { bytes: Vec::new() }
        );
        assert_eq!(
            q.push(b"after"),
            QuarantineOutcome::Stop { bytes: Vec::new() }
        );
        assert_eq!(q.finish(), Vec::<u8>::new());
    }

    #[test]
    fn discard_does_not_recover_pending_bytes() {
        let mut q = StopQuarantine::new(vec![b"<stop>".to_vec()]);
        assert_eq!(
            q.push(b"<sto"),
            QuarantineOutcome::Continue { bytes: Vec::new() }
        );
        q.discard();
        assert_eq!(q.finish(), Vec::<u8>::new());
        assert_eq!(
            q.push(b"after"),
            QuarantineOutcome::Continue { bytes: Vec::new() }
        );
    }

    #[test]
    fn binary_bytes_are_scanned_without_utf8_assumptions() {
        let mut q = StopQuarantine::new(vec![vec![0, 255, 1]]);
        assert_eq!(
            q.push(&[9, 0]),
            QuarantineOutcome::Continue { bytes: vec![9] }
        );
        assert_eq!(
            q.push(&[255, 1, 8]),
            QuarantineOutcome::Stop { bytes: Vec::new() }
        );
    }

    #[test]
    fn exact_bound_proper_prefix_is_retained() {
        let mut q = StopQuarantine::new(vec![b"abcd".to_vec()]);
        assert_eq!(
            q.push(b"abc"),
            QuarantineOutcome::Continue { bytes: Vec::new() }
        );
        assert!(q.has_pending());
        assert_eq!(q.pending_len(), 3);
        assert_eq!(q.finish(), b"abc");
    }

    #[test]
    fn empty_markers_are_ignored() {
        let mut q = StopQuarantine::new(vec![Vec::new()]);
        assert_eq!(
            q.push(b"plain"),
            QuarantineOutcome::Continue {
                bytes: b"plain".to_vec()
            }
        );
        assert_eq!(q.finish(), Vec::<u8>::new());

        let mut q = StopQuarantine::new(vec![Vec::new(), b"stop".to_vec()]);
        assert_eq!(
            q.push(b"plain"),
            QuarantineOutcome::Continue {
                bytes: b"plain".to_vec()
            }
        );
        assert_eq!(
            q.push(b"stop"),
            QuarantineOutcome::Stop { bytes: Vec::new() }
        );
    }

    #[test]
    fn duplicate_markers_stop_once() {
        let mut q = StopQuarantine::new(vec![b"<stop>".to_vec(), b"<stop>".to_vec()]);
        assert_eq!(q.markers.len(), 1);
        assert_eq!(
            q.push(b"safe<stop>tail"),
            QuarantineOutcome::Stop {
                bytes: b"safe".to_vec()
            }
        );
        assert_eq!(q.finish(), Vec::<u8>::new());
    }

    #[test]
    fn pending_memory_is_bounded_by_longest_marker_minus_one() {
        let markers = vec![b"abc".to_vec(), b"long-stop".to_vec()];
        let mut q = StopQuarantine::new(markers.clone());
        for len in 0..1000 {
            let input = vec![b'a'; len];
            let _ = q.push(&input);
            assert!(q.pending_len() <= q.max_pending_len());
            assert_eq!(q.max_pending_len(), b"long-stop".len() - 1);
        }
    }
}
