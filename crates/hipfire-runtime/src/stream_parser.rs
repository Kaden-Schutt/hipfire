//! The per-token OUTPUT layer for the generic ar_generate driver. One StreamParser
//! per request owns text emission, channel routing (visible vs reasoning), tool-call
//! events, and the stop / forced-token decisions. The driver owns streamed_tokens +
//! the byte cursor and passes each committed token's id + the RUNNING-VECTOR byte
//! delta (decode_bytes(&streamed)[bytes_fed..]) — never a per-token decode, because
//! BPE detok is non-local.

/// Actions the driver executes for the parser (in returned order).
#[derive(Debug, Clone, PartialEq)]
pub enum StreamAction {
    /// Emit visible/reasoning text. `reasoning=true` → `"reasoning":true` channel.
    Emit { text: String, reasoning: bool },
    /// Emit a `{"type":"tool_calls","calls":[…]}` event.
    ToolCalls(serde_json::Value),
    /// Emit a `{"type":"info",...}` diagnostic (n-gram loop / force-answer).
    Info(String),
    /// Break the decode loop (n-gram / repeat / pad / stop-seq).
    Stop,
}

/// The eos discipline for the sampled eos token.
#[derive(Debug, Clone, PartialEq)]
pub enum EosDecision {
    /// Commit + forward the eos token, then stop (simple arches: eos enters KV + tape,
    /// display-suppressed by the filter — current byte-identical behavior).
    CommitAndStop,
    /// Stop WITHOUT committing the eos (no KV write, no tape entry).
    Stop,
    /// Do NOT commit the eos; enqueue these tokens (surface via next_forced); continue.
    Inject(Vec<u32>),
}

pub trait StreamParser {
    /// Pre-sample. `Some(tok)` forces a token instead of sampling this iteration.
    /// The driver forwards + feeds a forced token exactly like a sampled one.
    fn next_forced(&mut self) -> Option<u32> {
        None
    }

    /// Called when the *sampled* token is eos. Not called for forced tokens.
    fn on_eos(&mut self) -> EosDecision {
        EosDecision::CommitAndStop
    }

    /// Consume a committed token: its id + the running-vector byte delta.
    fn feed(&mut self, tok: u32, bytes: &[u8]) -> Vec<StreamAction>;

    /// The driver calls this each iteration with the external force-answer signal
    /// (`check_force_answer(id)`) before the forced/sample decision; default no-op.
    fn note_force_answer(&mut self, now: bool) {
        let _ = now;
    }

    /// Whether the force-answer latch is set — the driver's steady-state sampler blocks
    /// the `<think>` re-open token when latched (ar_generate 8523). Default false.
    fn force_answer_latched(&self) -> bool {
        false
    }

    /// Emit a token's bytes through the output filter WITHOUT running the guards
    /// (think-cap / n-gram / stop-seq / budget). Used for the terminal eos token on
    /// `EosDecision::CommitAndStop`: the legacy loop emits+forwards the eos token then
    /// breaks BEFORE the guard block, so running feed's guards on it would diverge
    /// (spurious Info / think-enqueue). Default treats the bytes as a plain visible
    /// emit; `DefaultStreamParser` overrides to route them through its EosFilter (so a
    /// marker-eos is stripped, matching the legacy filter behavior).
    fn emit_only(&mut self, tok: u32, bytes: &[u8]) -> Vec<StreamAction> {
        let _ = tok;
        if bytes.is_empty() {
            return Vec::new();
        }
        vec![StreamAction::Emit {
            text: String::from_utf8_lossy(bytes).into_owned(),
            reasoning: false,
        }]
    }

    /// Push a token onto the parser's forced-injection queue. The driver calls this
    /// for each token an `on_eos()` -> `Inject(..)` returned (cohere2moe's empty-turn
    /// guard) so the tokens surface via `next_forced()` on the next iteration. Default
    /// no-op (DefaultStreamParser never returns `Inject`).
    fn enqueue(&mut self, tok: u32) {
        let _ = tok;
    }

    /// End of generation. Flush pending bytes / recover tool-calls-from-text.
    fn finish(&mut self) -> Vec<StreamAction> {
        Vec::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DefaultStreamParser — the byte-identical default for the simple arches.
//
// Reproduces the current inline ar_generate behavior (daemon.rs 8226–8431): the
// EosFilter emit (running-vector delta), stop-seqs, the think-cap force-close
// (surfaced as forced tokens via next_forced), the n-gram loop guard, and the
// in-think budget-alert injection. The parser keeps its OWN `streamed`/`decoded`
// accumulators (the driver hands it the running-vector byte delta per token) so it
// needs no tokenizer and no trait-signature change.
//
// TWO pieces stay driver-side by construction and are wired in Task 3:
//   * the eos discipline — on_eos()=CommitAndStop; the terminal eos token is
//     emitted+forwarded then the loop breaks BEFORE the guards run (current code
//     breaks at 8254 before 8268+). The driver must emit the terminal token
//     WITHOUT running feed's guard path (or the guards would spuriously run on eos).
//   * `force_answer_now` (daemon `check_force_answer(id)`, an external per-request
//     signal) — the driver calls `note_force_answer()` each iteration.
//   * the budget-alert `!in_think` RE-SAMPLE-and-continue branch (8400–8431) — it
//     resamples the NEXT token with a fresh SamplerConfig, which cannot be expressed
//     as a forced-token injection. TODO(task3): the driver keeps that resample branch.

use crate::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use crate::loop_guard::{LoopGuard, StopReason};
use std::collections::VecDeque;

/// Config for `DefaultStreamParser`, built by the driver from the ar_generate locals.
pub struct DefaultStreamParserConfig {
    /// Output filter (UTF-8 boundary buffering + special-marker strip).
    pub eos_filter: EosFilterConfig,
    /// Stop sequences: break when the decoded suffix ends with any of these.
    pub stop_seqs: Vec<String>,
    /// `max_tokens` — needed to cap the think-cap continuation splice.
    pub max_tokens: usize,
    /// Per-window think-cap (0 = disabled). Mirrors `max_think_tokens`.
    pub max_think_tokens: usize,
    /// Pre-encoded `think_continuation()` tokens (the driver encodes; parser injects).
    pub think_continuation_ids: Vec<u32>,
    /// Total-think hard cap (0 = disabled). Mirrors `max_total_think`.
    pub max_total_think: usize,
    /// Tokens after a force-answer latch before forcing EOS. Mirrors
    /// `post_latch_answer_budget`.
    pub post_latch_answer_budget: usize,
    /// Budget-alert trigger (0 = disabled). Mirrors `budget_alert_at_tok`.
    pub budget_alert_at_tok: usize,
    /// Pre-encoded budget-alert text tokens (in-think inject path).
    pub budget_alert_ids: Vec<u32>,
    /// Whether the assistant prefix opened inside `<think>` (OpenThink). Feeds
    /// `currently_in_think`.
    pub started_in_think: bool,
}

impl DefaultStreamParserConfig {
    /// All guards off — a plain UTF-8-safe passthrough. For tests and the simplest
    /// simple-arch decode.
    pub fn disabled() -> Self {
        Self {
            eos_filter: EosFilterConfig::default(),
            stop_seqs: Vec::new(),
            max_tokens: usize::MAX,
            max_think_tokens: 0,
            think_continuation_ids: Vec::new(),
            max_total_think: 0,
            post_latch_answer_budget: 768,
            budget_alert_at_tok: 0,
            budget_alert_ids: Vec::new(),
            started_in_think: false,
        }
    }
}

pub struct DefaultStreamParser {
    cfg: DefaultStreamParserConfig,
    filter: EosFilter,
    loop_guard: LoopGuard,
    streamed: Vec<u32>,
    decoded: Vec<u8>,
    generated: usize,
    forced: VecDeque<u32>,
    // think-cap state (verbatim from ar_generate 8268–8367).
    think_count: usize,
    prev_in_think: bool,
    total_think_tokens: usize,
    force_answer_latched: bool,
    force_answer_now: bool,
    latch_gen_mark: Option<usize>,
    alert_fired: bool,
}

impl DefaultStreamParser {
    pub fn new(cfg: DefaultStreamParserConfig) -> Self {
        let filter = EosFilter::new(cfg.eos_filter.clone());
        let loop_guard = LoopGuard::from_config(crate::config::get());
        Self {
            cfg,
            filter,
            loop_guard,
            streamed: Vec::new(),
            decoded: Vec::new(),
            generated: 0,
            forced: VecDeque::new(),
            think_count: 0,
            prev_in_think: false,
            total_think_tokens: 0,
            force_answer_latched: false,
            force_answer_now: false,
            latch_gen_mark: None,
            alert_fired: false,
        }
    }

    fn in_think(&self) -> bool {
        let raw = std::str::from_utf8(&self.decoded).unwrap_or("");
        crate::emit_text::currently_in_think(raw, self.cfg.started_in_think)
    }
}

impl StreamParser for DefaultStreamParser {
    fn next_forced(&mut self) -> Option<u32> {
        // Think-cap / budget-alert enqueue happens in feed(); next_forced drains it.
        self.forced.pop_front()
    }

    fn note_force_answer(&mut self, now: bool) {
        // Latches like ar_generate 8269–8272.
        self.force_answer_now = now;
        if now {
            self.force_answer_latched = true;
        }
    }

    fn force_answer_latched(&self) -> bool {
        self.force_answer_latched
    }

    fn feed(&mut self, tok: u32, bytes: &[u8]) -> Vec<StreamAction> {
        let mut acts = Vec::new();
        self.streamed.push(tok);
        self.decoded.extend_from_slice(bytes);

        // ── Emit (running-vector delta through the filter). ──
        match self.filter.observe(bytes) {
            FilterAction::Emit(text_bytes) => {
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                acts.push(StreamAction::Emit {
                    text: text.to_string(),
                    reasoning: false,
                });
            }
        }
            FilterAction::EmitAndStop(emit) => {
                if let Ok(text) = std::str::from_utf8(&emit) {
                    if !text.is_empty() {
                        acts.push(StreamAction::Emit {
                            text: text.to_string(),
                            reasoning: false,
                        });
                    }
                }
                self.generated += 1;
                acts.push(StreamAction::Stop);
                return acts;
            }
            FilterAction::Stop => {
                self.generated += 1;
                acts.push(StreamAction::Stop);
                return acts;
            }
            FilterAction::Hold => {}
        }
        self.generated += 1;

        // ── Stop-seqs (8261). ──
        if !self.cfg.stop_seqs.is_empty() {
            let decoded_suffix = std::str::from_utf8(&self.decoded).unwrap_or("");
            if self
                .cfg
                .stop_seqs
                .iter()
                .any(|s| decoded_suffix.ends_with(s.as_str()))
            {
                acts.push(StreamAction::Stop);
                return acts;
            }
        }

        // ── Think-cap / total-think / force-answer (8268–8367). ──
        if self.cfg.max_think_tokens > 0
            || self.force_answer_now
            || self.force_answer_latched
            || self.cfg.max_total_think > 0
        {
            let in_think = self.in_think();
            if in_think {
                self.total_think_tokens += 1;
            }
            if self.cfg.max_total_think > 0 && self.total_think_tokens >= self.cfg.max_total_think {
                self.force_answer_latched = true;
            }
            if self.force_answer_latched && self.latch_gen_mark.is_none() {
                self.latch_gen_mark = Some(self.generated);
            }
            if self.cfg.max_total_think > 0
                && in_think
                && self.total_think_tokens >= self.cfg.max_total_think + 256
            {
                acts.push(StreamAction::Stop);
                return acts;
            }
            if let Some(mark) = self.latch_gen_mark {
                if self.generated.saturating_sub(mark) >= self.cfg.post_latch_answer_budget {
                    acts.push(StreamAction::Stop);
                    return acts;
                }
            }
            if self.cfg.max_think_tokens > 0 {
                if in_think {
                    if !self.prev_in_think {
                        self.think_count = 1;
                    } else {
                        self.think_count += 1;
                    }
                } else {
                    self.think_count = 0;
                }
                self.prev_in_think = in_think;
            }
            let budget_hit =
                self.cfg.max_think_tokens > 0 && self.think_count >= self.cfg.max_think_tokens;
            if in_think && (budget_hit || self.force_answer_now || self.force_answer_latched) {
                // Force-close: enqueue the continuation splice (capped by remaining
                // budget). The driver forwards+advances+feeds each via next_forced.
                let budget_left = self.cfg.max_tokens.saturating_sub(self.generated);
                let take = self.cfg.think_continuation_ids.len().min(budget_left);
                for &t in &self.cfg.think_continuation_ids[..take] {
                    self.forced.push_back(t);
                }
                self.think_count = 0;
                self.prev_in_think = false;
            }
        }

        // ── N-gram loop detector (8370). ──
        if let Some(StopReason::NgramRepeat { count, .. }) = self.loop_guard.check(&self.streamed) {
            let window_len = self.loop_guard.window_len(self.streamed.len());
            acts.push(StreamAction::Info(format!(
                "ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS",
                count, window_len
            )));
            acts.push(StreamAction::Stop);
            return acts;
        }

        // ── Budget-alert (8384). In-think inject; !in_think resample stays driver-side. ──
        if !self.alert_fired
            && self.cfg.budget_alert_at_tok > 0
            && self.generated >= self.cfg.budget_alert_at_tok
            && !self.cfg.budget_alert_ids.is_empty()
        {
            self.alert_fired = true;
            if self.in_think() {
                for &t in &self.cfg.budget_alert_ids {
                    self.forced.push_back(t);
                }
            } else {
                // TODO(task3): the !in_think branch info-emits then RE-SAMPLES the next
                // token with a fresh SamplerConfig and continues — the driver keeps that
                // resample; the parser only surfaces the info diagnostic here.
                acts.push(StreamAction::Info(
                    "budget_alert skipped: not inside an open <think> block".to_string(),
                ));
            }
        }

        acts
    }

    fn emit_only(&mut self, tok: u32, bytes: &[u8]) -> Vec<StreamAction> {
        // Terminal eos: emit through the filter (a marker-eos is stripped), update the
        // accumulators for consistency, but run NO guards (the legacy loop breaks before
        // the guard block on eos). No `generated`/think-counter mutation.
        self.streamed.push(tok);
        self.decoded.extend_from_slice(bytes);
        let mut acts = Vec::new();
        match self.filter.observe(bytes) {
            FilterAction::Emit(text_bytes) => {
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                acts.push(StreamAction::Emit {
                    text: text.to_string(),
                    reasoning: false,
                });
            }
        }
            FilterAction::EmitAndStop(emit) => {
                if let Ok(text) = std::str::from_utf8(&emit) {
                    if !text.is_empty() {
                        acts.push(StreamAction::Emit {
                            text: text.to_string(),
                            reasoning: false,
                        });
                    }
                }
                acts.push(StreamAction::Stop);
            }
            FilterAction::Stop => {
                acts.push(StreamAction::Stop);
            }
            FilterAction::Hold => {}
        }
        acts
    }

    fn on_eos(&mut self) -> EosDecision {
        EosDecision::CommitAndStop
    }

    fn finish(&mut self) -> Vec<StreamAction> {
        let pending = self.filter.finish();
        if pending.is_empty() {
            return Vec::new();
        }
        match std::str::from_utf8(&pending) {
            Ok(text) if !text.is_empty() => vec![StreamAction::Emit {
                text: text.to_string(),
                reasoning: false,
            }],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_cfg() -> DefaultStreamParserConfig {
        DefaultStreamParserConfig::disabled()
    }

    #[test]
    fn plain_text_passthrough_emits_visible() {
        let mut p = DefaultStreamParser::new(plain_cfg());
        let acts = p.feed(42, b"Hi");
        assert_eq!(
            acts,
            vec![StreamAction::Emit {
                text: "Hi".into(),
                reasoning: false
            }]
        );
    }

    #[test]
    fn filter_stop_emits_safe_prefix_then_stop_without_marker() {
        let mut cfg = plain_cfg();
        cfg.eos_filter = EosFilterConfig {
            strip_think: false,
            started_in_think: false,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: Vec::new(),
        };
        let mut p = DefaultStreamParser::new(cfg);
        assert_eq!(
            p.feed(1, b"safe<stop>leaked"),
            vec![
                StreamAction::Emit {
                    text: "safe".into(),
                    reasoning: false,
                },
                StreamAction::Stop,
            ]
        );
    }

    #[test]
    fn emit_only_returns_filter_stop_prefix_without_marker() {
        let mut cfg = plain_cfg();
        cfg.eos_filter = EosFilterConfig {
            strip_think: false,
            started_in_think: false,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: Vec::new(),
        };
        let mut p = DefaultStreamParser::new(cfg);
        let actions = p.emit_only(1, b"safe<stop>leaked");
        assert_eq!(
            actions,
            vec![
                StreamAction::Emit {
                    text: "safe".into(),
                    reasoning: false,
                },
                StreamAction::Stop,
            ]
        );
        assert!(p.emit_only(2, b"after-stop").is_empty());
    }

    #[test]
    fn filter_stop_without_visible_prefix_returns_only_stop() {
        let mut cfg = plain_cfg();
        cfg.eos_filter = EosFilterConfig {
            strip_think: false,
            started_in_think: false,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: Vec::new(),
        };
        let mut p = DefaultStreamParser::new(cfg);
        assert_eq!(p.feed(1, b"<stop>"), vec![StreamAction::Stop]);
    }

    #[test]
    fn on_eos_default_is_commit_and_stop() {
        let mut p = DefaultStreamParser::new(plain_cfg());
        assert_eq!(p.on_eos(), EosDecision::CommitAndStop);
    }

    #[test]
    fn stop_seq_returns_stop() {
        let mut cfg = plain_cfg();
        cfg.stop_seqs = vec!["END".to_string()];
        let mut p = DefaultStreamParser::new(cfg);
        let _ = p.feed(1, b"the ");
        let acts = p.feed(2, b"END");
        assert!(acts.contains(&StreamAction::Stop));
    }

    #[test]
    fn no_forced_by_default() {
        let mut p = DefaultStreamParser::new(plain_cfg());
        assert_eq!(p.next_forced(), None);
    }

    #[test]
    fn finish_emits_held_visible_suffix_once() {
        let mut cfg = plain_cfg();
        cfg.eos_filter = EosFilterConfig {
            strip_think: true,
            started_in_think: false,
            stop_at: Vec::new(),
            holdback_prefixes: vec![b"<partial>".to_vec()],
        };
        let mut p = DefaultStreamParser::new(cfg);
        // Merged (beta) filter semantics: prose before the think marker is
        // emitted eagerly; the open `<think>` span and its content are stripped.
        assert_eq!(
            p.feed(1, b"visible<think>reason"),
            vec![StreamAction::Emit {
                text: "visible".into(),
                reasoning: false,
            }]
        );
        assert_eq!(p.finish(), Vec::new());
        assert_eq!(p.finish(), Vec::new());
    }

    #[test]
    fn finish_suppresses_open_thinking_content() {
        let mut cfg = plain_cfg();
        cfg.eos_filter = EosFilterConfig {
            strip_think: true,
            started_in_think: false,
            stop_at: Vec::new(),
            holdback_prefixes: Vec::new(),
        };
        let mut p = DefaultStreamParser::new(cfg);
        assert!(p.feed(1, b"<think>secret").is_empty());
        assert_eq!(p.finish(), Vec::new());
        assert_eq!(p.finish(), Vec::new());
    }

    #[test]
    fn think_cap_enqueues_continuation_when_budget_hit() {
        // started_in_think=true so the opener isn't needed; max_think_tokens=2 →
        // the 3rd in-think token trips budget_hit and enqueues the continuation.
        let mut cfg = plain_cfg();
        cfg.started_in_think = true;
        cfg.max_think_tokens = 2;
        cfg.think_continuation_ids = vec![900, 901];
        let mut p = DefaultStreamParser::new(cfg);
        // Feed plain in-think text (no </think>), staying inside the block.
        let _ = p.feed(1, b"a");
        let _ = p.feed(2, b"b");
        let _ = p.feed(3, b"c"); // think_count reaches 3 >= 2 → force-close enqueued
        assert_eq!(p.next_forced(), Some(900));
        assert_eq!(p.next_forced(), Some(901));
        assert_eq!(p.next_forced(), None);
    }
}
