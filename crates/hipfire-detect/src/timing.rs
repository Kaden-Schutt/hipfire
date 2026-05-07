//! Per-token step-time spike detector. OPT-IN.
//!
//! Tracks the wall-clock delta between consecutive `Committed` events
//! (one per token decoded by the model — pre-`EosFilter`, so timing is
//! independent of buffer flush behaviour). Computes a rolling median
//! over the trailing `WINDOW` deltas and flags any single delta that
//! is more than `RATIO` × median.
//!
//! Off by default: under DPM, graph capture, stdout flush jitter, and
//! eviction events the signal is noisy. The probe binary opts in via
//! `--detect-timing`.

use crate::{Detector, Event, Verdict};
use std::collections::VecDeque;

const WINDOW: usize = 32;
const RATIO: f64 = 4.0;
/// Below this minimum buffer size, we don't trust the median enough
/// to fire — the early window is noisy.
const MIN_BUFFER_TO_FIRE: usize = 8;

pub struct StepTimeSpike {
    last_t_ms: Option<u64>,
    deltas: VecDeque<u64>,
    biggest_spike: Option<(u64, u64)>, // (delta, median_at_that_point)
}

impl StepTimeSpike {
    pub fn new() -> Self {
        Self {
            last_t_ms: None,
            deltas: VecDeque::with_capacity(WINDOW),
            biggest_spike: None,
        }
    }

    fn current_median(&self) -> Option<f64> {
        if self.deltas.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.deltas.iter().copied().collect();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        Some(if sorted.len() % 2 == 0 {
            (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
        } else {
            sorted[mid] as f64
        })
    }
}

impl Default for StepTimeSpike {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for StepTimeSpike {
    fn name(&self) -> &'static str {
        "step_time_spike"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Committed { t_ms, .. } = ev {
            if let Some(prev) = self.last_t_ms {
                let delta = t_ms.saturating_sub(prev);
                if let Some(median) = self.current_median() {
                    if self.deltas.len() >= MIN_BUFFER_TO_FIRE
                        && median > 0.0
                        && (delta as f64) > RATIO * median
                    {
                        let prev_max = self
                            .biggest_spike
                            .map(|(d, _)| d)
                            .unwrap_or(0);
                        if delta > prev_max {
                            self.biggest_spike = Some((delta, median.round() as u64));
                        }
                    }
                }
                self.deltas.push_back(delta);
                if self.deltas.len() > WINDOW {
                    self.deltas.pop_front();
                }
            }
            self.last_t_ms = Some(*t_ms);
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        if let Some((delta, median)) = self.biggest_spike {
            return Verdict::warn(format!(
                "biggest step-time spike: {}ms (rolling median {}ms, ratio {:.1}×)",
                delta,
                median,
                delta as f64 / median.max(1) as f64
            ));
        }
        Verdict::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(pos: usize, t_ms: u64) -> Event<'static> {
        Event::Committed {
            tok_id: 1,
            pos,
            t_ms,
        }
    }

    #[test]
    fn smooth_timing_ok() {
        let mut d = StepTimeSpike::new();
        for i in 0..50u64 {
            d.observe(&commit(i as usize, i * 10));
        }
        assert!(matches!(d.finalize(), Verdict::Ok));
    }

    #[test]
    fn one_big_spike_warns() {
        let mut d = StepTimeSpike::new();
        // Build a baseline of 10ms ticks for 32 samples.
        for i in 0..32u64 {
            d.observe(&commit(i as usize, i * 10));
        }
        // Now drop one 100ms gap (10× median).
        d.observe(&commit(32, 32 * 10 + 100));
        // A few more normal ticks.
        for i in 33..40u64 {
            d.observe(&commit(i as usize, 32 * 10 + 100 + (i - 32) * 10));
        }
        let v = d.finalize();
        assert!(v.is_warn(), "got {:?}", v);
    }

    #[test]
    fn early_buffer_does_not_fire() {
        // First few samples should not fire even with a "spike" — buffer
        // is too small to trust the median.
        let mut d = StepTimeSpike::new();
        d.observe(&commit(0, 0));
        d.observe(&commit(1, 5));
        d.observe(&commit(2, 5000));
        assert!(matches!(d.finalize(), Verdict::Ok));
    }
}
