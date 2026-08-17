// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prometheus exposition for the serve path.
//!
//! The daemon already emits `ttft_ms`, `prefill_tok_s`, `decode_tok_s` and
//! `latency_ms` on every `done` event. Until this module existed the gateway
//! kept exactly one of them — `ServeMeta::recent_tok_s`, a scalar overwritten by
//! each request — so per-request telemetry flowed through the process and was
//! discarded. `/stats` reported counters but no distribution, which is why the
//! periodic decode-stall burst fixed in `dacce7470` had to be diagnosed with a
//! bespoke harness: a p50 and a p99 would have shown it immediately.
//!
//! No new dependency. Prometheus text format is a few lines to emit, and the
//! histogram is fixed-bucket with atomic counters, so scraping never blocks a
//! request. Buckets are chosen for interactive LLM serving: single-digit
//! milliseconds is meaningless here, tens of seconds is a hang.

use std::sync::atomic::{AtomicU64, Ordering};

/// Upper bounds in milliseconds. `+Inf` is implicit and emitted last.
const LATENCY_BUCKETS_MS: [f64; 12] = [
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0,
];

/// Fixed-bucket histogram over f64 observations, lock-free.
#[derive(Debug, Default)]
pub(crate) struct Histogram {
    buckets: [AtomicU64; LATENCY_BUCKETS_MS.len()],
    inf: AtomicU64,
    sum_milli: AtomicU64,
}

impl Histogram {
    pub(crate) fn observe(&self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        let mut placed = false;
        for (i, bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if value <= *bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                placed = true;
                break;
            }
        }
        if !placed {
            self.inf.fetch_add(1, Ordering::Relaxed);
        }
        // Store the sum scaled by 1000 so it stays integral and lock-free.
        self.sum_milli
            .fetch_add((value * 1000.0) as u64, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.buckets
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .sum::<u64>()
            + self.inf.load(Ordering::Relaxed)
    }

    fn sum(&self) -> f64 {
        self.sum_milli.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Prometheus histogram: cumulative buckets, then `_sum` and `_count`.
    fn render(&self, out: &mut String, name: &str, help: &str) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (i, bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.inf.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!("{name}_sum {}\n", self.sum()));
        out.push_str(&format!("{name}_count {}\n", self.count()));
    }
}

/// Everything the gateway can report without asking the daemon.
#[derive(Debug, Default)]
pub(crate) struct Metrics {
    pub(crate) ttft_ms: Histogram,
    pub(crate) latency_ms: Histogram,
    pub(crate) decode_tok_s: Histogram,
    pub(crate) prefill_tok_s: Histogram,
    pub(crate) requests_total: AtomicU64,
    pub(crate) requests_failed: AtomicU64,
    pub(crate) admission_rejected: AtomicU64,
}

impl Metrics {
    /// Fold one completed request's `done` payload.
    ///
    /// Every field is optional: a stream aborted by the client, or a daemon
    /// build that predates a field, simply contributes nothing rather than a
    /// zero. A zero would drag every percentile toward the floor and make the
    /// histogram lie in the reassuring direction.
    pub(crate) fn observe_done(&self, done: &serde_json::Value) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        // The `done` envelope is PATH-DEPENDENT. The rich builder
        // (`hipfire_generate::common::emit_qwen_dflash_done_terminal`) carries
        // prefill_tok_s, decode_tok_s, ttft_ms, tau and cycles; the simple decode
        // path emits only {type,id,tokens,tok_s}. So `prefill_tok_s` fills on the
        // qwen/DFlash routes and stays empty on a plain lfm2.5 decode -- verified
        // by scraping both.
        //
        // ttft and latency are deliberately NOT taken from here even when present.
        // They come from `observe_timing`, measured at the gateway, so that every
        // path reports them on the same basis and the interval includes admission
        // queueing -- the component an operator needs when latency rises.
        if let Some(v) = done
            .get("decode_tok_s")
            .or_else(|| done.get("tok_s"))
            .and_then(serde_json::Value::as_f64)
        {
            self.decode_tok_s.observe(v);
        }
        if let Some(v) = done.get("prefill_tok_s").and_then(serde_json::Value::as_f64) {
            self.prefill_tok_s.observe(v);
        }
    }

    /// Gateway-measured timing for one attempt.
    ///
    /// `ttft_ms` is `None` when the attempt produced no token at all (an error
    /// or an immediate abort); recording a 0 there would put a fake best-case
    /// sample in the histogram.
    pub(crate) fn observe_timing(&self, ttft_ms: Option<f64>, latency_ms: f64) {
        if let Some(t) = ttft_ms {
            self.ttft_ms.observe(t);
        }
        self.latency_ms.observe(latency_ms);
    }

    pub(crate) fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_admission_rejected(&self) {
        self.admission_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Prometheus text exposition (v0.0.4).
    pub(crate) fn render(
        &self,
        queue_depth: usize,
        queue_capacity: usize,
        uptime_secs: u64,
        model: Option<&str>,
    ) -> String {
        let mut out = String::with_capacity(2048);

        let counter = |out: &mut String, name: &str, help: &str, v: u64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"
            ));
        };
        let gauge = |out: &mut String, name: &str, help: &str, v: f64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}\n"
            ));
        };

        counter(
            &mut out,
            "hipfire_requests_total",
            "Completed generation requests.",
            self.requests_total.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_requests_failed_total",
            "Requests that ended in an error response.",
            self.requests_failed.load(Ordering::Relaxed),
        );
        counter(
            &mut out,
            "hipfire_admission_rejected_total",
            "Requests refused by admission control (queue full or model mismatch).",
            self.admission_rejected.load(Ordering::Relaxed),
        );
        gauge(
            &mut out,
            "hipfire_queue_depth",
            "Requests currently in flight or queued.",
            queue_depth as f64,
        );
        gauge(
            &mut out,
            "hipfire_queue_capacity",
            "Admission control capacity.",
            queue_capacity as f64,
        );
        gauge(
            &mut out,
            "hipfire_uptime_seconds",
            "Seconds since the gateway started.",
            uptime_secs as f64,
        );
        gauge(
            &mut out,
            "hipfire_model_loaded",
            "1 when a model is resident, 0 otherwise.",
            if model.is_some() { 1.0 } else { 0.0 },
        );

        self.ttft_ms.render(
            &mut out,
            "hipfire_ttft_milliseconds",
            "Time to first token, per request.",
        );
        self.latency_ms.render(
            &mut out,
            "hipfire_request_latency_milliseconds",
            "Wall time per request.",
        );
        self.decode_tok_s.render(
            &mut out,
            "hipfire_decode_tokens_per_second",
            "Decode throughput per request.",
        );
        self.prefill_tok_s.render(
            &mut out,
            "hipfire_prefill_tokens_per_second",
            "Prefill throughput per request.",
        );

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(pairs: &[(&str, f64)]) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), serde_json::json!(v));
        }
        serde_json::Value::Object(m)
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_sum_is_exact() {
        let h = Histogram::default();
        for v in [5.0, 30.0, 300.0, 90_000.0] {
            h.observe(v);
        }
        let mut s = String::new();
        h.render(&mut s, "t", "help");
        // 5 -> le=10; 30 -> le=50; 300 -> le=500; 90000 -> +Inf
        assert!(s.contains("t_bucket{le=\"10\"} 1"), "{s}");
        assert!(s.contains("t_bucket{le=\"50\"} 2"), "{s}");
        assert!(s.contains("t_bucket{le=\"500\"} 3"), "{s}");
        assert!(s.contains("t_bucket{le=\"+Inf\"} 4"), "{s}");
        assert!(s.contains("t_count 4"), "{s}");
        assert!(s.contains("t_sum 90335"), "{s}");
    }

    #[test]
    fn absent_fields_do_not_observe_a_zero() {
        // A zero would pull every percentile toward the floor -- a histogram that
        // lies in the reassuring direction is worse than a missing one.
        let m = Metrics::default();
        m.observe_done(&done(&[("prefill_tok_s", 400.0)]));
        assert_eq!(m.prefill_tok_s.count(), 1);
        assert_eq!(m.decode_tok_s.count(), 0, "absent decode must not be recorded");
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
        // ttft/latency come from observe_timing, never from the done payload.
        m.observe_timing(None, 120.0);
        assert_eq!(m.ttft_ms.count(), 0, "no first token => no ttft sample");
        assert_eq!(m.latency_ms.count(), 1);
        m.observe_timing(Some(30.0), 200.0);
        assert_eq!(m.ttft_ms.count(), 1);
        assert_eq!(m.latency_ms.count(), 2);
    }

    #[test]
    fn decode_falls_back_to_tok_s() {
        let m = Metrics::default();
        m.observe_done(&done(&[("tok_s", 37.7)]));
        assert_eq!(m.decode_tok_s.count(), 1);
    }

    #[test]
    fn negative_and_nan_are_ignored() {
        let h = Histogram::default();
        h.observe(-1.0);
        h.observe(f64::NAN);
        h.observe(f64::INFINITY);
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn render_emits_help_and_type_for_every_series() {
        let m = Metrics::default();
        let out = m.render(3, 16, 120, Some("qwen3.8-27b"));
        for name in [
            "hipfire_requests_total",
            "hipfire_queue_depth",
            "hipfire_ttft_milliseconds",
            "hipfire_decode_tokens_per_second",
        ] {
            assert!(out.contains(&format!("# HELP {name} ")), "missing HELP {name}");
            assert!(out.contains(&format!("# TYPE {name} ")), "missing TYPE {name}");
        }
        assert!(out.contains("hipfire_queue_depth 3"));
        assert!(out.contains("hipfire_model_loaded 1"));
    }
}
