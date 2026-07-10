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

    /// End of generation. Flush pending bytes / recover tool-calls-from-text.
    fn finish(&mut self) -> Vec<StreamAction> {
        Vec::new()
    }
}
