//! Report renderers — markdown for human reading, JSON for CI.
//!
//! Every report header includes the prompt md5, per CLAUDE.md's
//! τ-sensitivity rule (266-294): one newline can swing perf 14% on
//! 27B DFlash and detector behaviour can drift similarly with prompt
//! shape. Pinning the md5 lets the report be reproduced exactly.

use crate::{Severity, Verdict};
use serde::Serialize;

/// One row in the final report — detector name + final verdict.
#[derive(Debug, Clone, Serialize)]
pub struct ReportRow {
    pub name: String,
    #[serde(flatten)]
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportHeader {
    pub prompt_md5: String,
    pub prompt_label: String,
    pub model: String,
    pub arch: String,
    pub host: String,
    pub total_tokens: usize,
    /// Wall-clock tok/s as the *probe* measured it (`total_tokens / wall_ms`).
    /// Confused on thinking models because the probe sees a "first visible
    /// token" only after `</think>` closes, so its wall_ms folds prefill +
    /// think into TTFT. Kept for back-compat / UX framing; perf consumers
    /// should read `daemon_*` fields instead.
    pub tok_s: f64,
    /// Probe-derived gen rate: `total_tokens / (wall_ms - ttft_ms)`.
    /// Same caveat as `tok_s` — strips think-as-prefill but doesn't
    /// distinguish real prefill from think.
    pub gen_tok_s: f64,
    pub ttft_ms: u64,
    /// Authoritative timings emitted by the daemon's `done` event:
    /// `prefill_ms` and `prefill_tok_s` are real-prefill-only (forward_prefill
    /// timer, post-DPM-warmup), `decode_tok_s` is the steady-state decode
    /// rate (post-prefill), `ttft_ms` is real prefill time, `tok_s` is the
    /// daemon's own `total_tokens / total_wall_seconds`. These are the
    /// numbers to compare against `bench_qwen35_mq4 prefill_tok_s` and
    /// `gen_tok_s`. Zero for non-Qwen35 paths or older daemons.
    pub daemon_prefill_ms: f64,
    pub daemon_prefill_tok_s: f64,
    pub daemon_decode_tok_s: f64,
    pub daemon_ttft_ms: f64,
    pub daemon_tok_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub header: ReportHeader,
    pub rows: Vec<ReportRow>,
    pub hard_fails: usize,
    pub soft_warns: usize,
}

impl Report {
    pub fn new(header: ReportHeader, finals: Vec<(&'static str, Verdict)>) -> Self {
        let rows: Vec<ReportRow> = finals
            .into_iter()
            .map(|(n, v)| ReportRow {
                name: n.to_string(),
                verdict: v,
            })
            .collect();
        let mut hard_fails = 0;
        let mut soft_warns = 0;
        for r in &rows {
            match &r.verdict {
                Verdict::Fired {
                    severity: Severity::Fail,
                    ..
                } => hard_fails += 1,
                Verdict::Fired {
                    severity: Severity::Warn,
                    ..
                } => soft_warns += 1,
                _ => {}
            }
        }
        Self {
            header,
            rows,
            hard_fails,
            soft_warns,
        }
    }

    pub fn overall_label(&self) -> &'static str {
        if self.hard_fails > 0 {
            "FAIL"
        } else if self.soft_warns > 0 {
            "WARN"
        } else {
            "OK"
        }
    }

    pub fn to_markdown(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "## Coherence Probe Report").unwrap();
        writeln!(out, "prompt md5:  {}", self.header.prompt_md5).unwrap();
        writeln!(out, "prompt:      {}", self.header.prompt_label).unwrap();
        writeln!(out, "model:       {}", self.header.model).unwrap();
        writeln!(
            out,
            "arch/host:   {} / {}",
            self.header.arch, self.header.host
        )
        .unwrap();
        writeln!(
            out,
            "tokens:      {} (probe wall {:.1} tok/s, probe gen {:.1} tok/s, probe ttft {}ms)",
            self.header.total_tokens, self.header.tok_s, self.header.gen_tok_s, self.header.ttft_ms
        )
        .unwrap();
        if self.header.daemon_tok_s > 0.0 {
            writeln!(
                out,
                "daemon perf: prefill {:.1} tok/s ({:.1}ms / real ttft) | decode {:.1} tok/s | overall {:.1} tok/s",
                self.header.daemon_prefill_tok_s,
                self.header.daemon_prefill_ms,
                self.header.daemon_decode_tok_s,
                self.header.daemon_tok_s,
            )
            .unwrap();
        }
        writeln!(
            out,
            "verdict:     {} ({} hard, {} soft)",
            self.overall_label(),
            self.hard_fails,
            self.soft_warns
        )
        .unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| detector             | status | detail |").unwrap();
        writeln!(out, "|----------------------|--------|--------|").unwrap();
        for row in &self.rows {
            let detail = match &row.verdict {
                Verdict::Ok => "—".to_string(),
                Verdict::Skip { reason } => reason.clone(),
                Verdict::Fired { detail, .. } => detail.clone(),
            };
            // Pipes inside details would corrupt the table.
            let detail = detail.replace('|', "\\|");
            writeln!(
                out,
                "| {:<20} | {:<6} | {} |",
                row.name,
                row.verdict.label(),
                detail
            )
            .unwrap();
        }
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }
}

/// Compute the md5 of a prompt's bytes — used for the report header.
pub fn prompt_md5(bytes: &[u8]) -> String {
    let digest = md5::compute(bytes);
    format!("{:x}", digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> ReportHeader {
        ReportHeader {
            prompt_md5: "deadbeef".to_string(),
            prompt_label: "test".to_string(),
            model: "qwen3.5-9b.mq4".to_string(),
            arch: "gfx1100".to_string(),
            host: "k9lin".to_string(),
            total_tokens: 100,
            tok_s: 100.0,
            gen_tok_s: 200.0,
            ttft_ms: 50,
            daemon_prefill_ms: 0.0,
            daemon_prefill_tok_s: 0.0,
            daemon_decode_tok_s: 0.0,
            daemon_ttft_ms: 0.0,
            daemon_tok_s: 0.0,
        }
    }

    #[test]
    fn overall_ok() {
        let r = Report::new(
            header(),
            vec![("a", Verdict::Ok), ("b", Verdict::Ok)],
        );
        assert_eq!(r.overall_label(), "OK");
        assert_eq!(r.hard_fails, 0);
    }

    #[test]
    fn overall_fail() {
        let r = Report::new(
            header(),
            vec![("a", Verdict::fail("bad")), ("b", Verdict::warn("meh"))],
        );
        assert_eq!(r.overall_label(), "FAIL");
        assert_eq!(r.hard_fails, 1);
        assert_eq!(r.soft_warns, 1);
    }

    #[test]
    fn markdown_renders() {
        let r = Report::new(header(), vec![("ok", Verdict::Ok)]);
        let md = r.to_markdown();
        assert!(md.contains("## Coherence Probe Report"));
        assert!(md.contains("ok"));
        assert!(md.contains("deadbeef"));
    }

    #[test]
    fn json_renders() {
        let r = Report::new(header(), vec![("ok", Verdict::Ok)]);
        let j = r.to_json();
        assert!(j.contains("\"prompt_md5\""));
        assert!(j.contains("\"ok\""));
    }

    #[test]
    fn pipe_in_detail_escaped() {
        let r = Report::new(
            header(),
            vec![("x", Verdict::fail("a|b|c"))],
        );
        let md = r.to_markdown();
        // The escaped pipe should not appear as a raw delimiter.
        assert!(md.contains("a\\|b\\|c"));
    }

    #[test]
    fn md5_is_stable() {
        let h = prompt_md5(b"hello world");
        assert_eq!(h.len(), 32);
        assert_eq!(prompt_md5(b"hello world"), h);
    }
}
