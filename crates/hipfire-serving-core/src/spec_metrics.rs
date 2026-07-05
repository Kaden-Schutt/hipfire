// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One canonical spec-decode `done`-event emitter.
//!
//! Every speculative strategy path (dspark, dflash, mtp, deepseek4-mtp) used to
//! hand-roll its own `done` schema with per-path `u64` counters and a divergent
//! `format!`/`json!` shape. [`emit_spec_done`] replaces all of them: it takes the
//! arch-agnostic [`SpecMetrics`] accumulator plus a per-site `ext` object and
//! writes ONE canonical `done` event. The metric block field names
//! (`windows`/`proposed`/`accepted`/`committed`/`tau`/`accept_rate`/
//! `mean_draft_len`/`mean_committed`/`acceptance_hist`) are identical across
//! strategies, so eval tooling parses one schema regardless of drafter.

use std::io::Write;

use hipfire_specdecode::SpecMetrics;
use serde_json::{Map, Value};

/// Emit the single canonical spec-decode `done` event to `stdout`.
///
/// Shape:
/// ```json
/// {"type":"done","id":..,"tokens":..,"tok_s":..,"<strategy>":true,
///  "windows":..,"proposed":..,"accepted":..,"committed":..,"tau":..,
///  "accept_rate":..,"mean_draft_len":..,"mean_committed":..,
///  "acceptance_hist":[..], <ext merged>}
/// ```
///
/// - `strategy` is the strategy flag key (e.g. `"dspark"`, `"dflash"`, `"mtp"`),
///   emitted as `"<strategy>": true`.
/// - `m.to_json()` supplies the arch-agnostic metric block.
/// - `ext`, when `Some(Value::Object(..))`, carries site-specific NON-metric
///   fields (prefill/total timings, `finish_reason`, `pflash`, legacy aliases
///   like `cycles`/`max_n`/`spec_k`) and any strategy `drain_extra_metrics()`
///   block. Ext keys are merged LAST, so a site must not put a canonical metric
///   key in `ext` unless it deliberately means to override it.
///
/// `tok_s` is rounded to 1 dp for stable JSONL. This function flushes `stdout`.
pub fn emit_spec_done(
    stdout: &mut std::io::Stdout,
    id: &str,
    tokens: usize,
    tok_s: f64,
    strategy: &str,
    m: &SpecMetrics,
    ext: Option<Value>,
) {
    let mut obj = Map::new();
    obj.insert("type".to_string(), Value::from("done"));
    obj.insert("id".to_string(), Value::from(id));
    obj.insert("tokens".to_string(), Value::from(tokens));
    obj.insert(
        "tok_s".to_string(),
        Value::from((tok_s * 10.0).round() / 10.0),
    );
    obj.insert(strategy.to_string(), Value::from(true));
    if let Value::Object(metric_fields) = m.to_json() {
        for (k, v) in metric_fields {
            obj.insert(k, v);
        }
    }
    if let Some(Value::Object(extra)) = ext {
        for (k, v) in extra {
            obj.insert(k, v);
        }
    }
    let _ = writeln!(stdout, "{}", Value::Object(obj));
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_shape_merges_metrics_and_ext() {
        let mut m = SpecMetrics::new(4);
        m.record_window(4, 2, 3);
        let v = build_value(
            "abc",
            10,
            12.34,
            "dflash",
            &m,
            Some(serde_json::json!({"cycles": 1})),
        );
        assert_eq!(v["type"], "done");
        assert_eq!(v["id"], "abc");
        assert_eq!(v["tokens"], 10);
        assert_eq!(v["tok_s"], 12.3);
        assert_eq!(v["dflash"], true);
        assert_eq!(v["windows"], 1);
        assert_eq!(v["accepted"], 2);
        assert_eq!(v["cycles"], 1); // ext preserved additively
    }

    // Mirror of `emit_spec_done`'s object build, without the stdout write, so the
    // canonical shape is unit-testable.
    fn build_value(
        id: &str,
        tokens: usize,
        tok_s: f64,
        strategy: &str,
        m: &SpecMetrics,
        ext: Option<Value>,
    ) -> Value {
        let mut obj = Map::new();
        obj.insert("type".to_string(), Value::from("done"));
        obj.insert("id".to_string(), Value::from(id));
        obj.insert("tokens".to_string(), Value::from(tokens));
        obj.insert(
            "tok_s".to_string(),
            Value::from((tok_s * 10.0).round() / 10.0),
        );
        obj.insert(strategy.to_string(), Value::from(true));
        if let Value::Object(metric_fields) = m.to_json() {
            for (k, v) in metric_fields {
                obj.insert(k, v);
            }
        }
        if let Some(Value::Object(extra)) = ext {
            for (k, v) in extra {
                obj.insert(k, v);
            }
        }
        Value::Object(obj)
    }
}
