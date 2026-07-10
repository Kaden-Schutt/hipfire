# Task 5 report — Cohere2MoeStreamParser (build-only)

Commit: 61c46e5f — feat(archdispatch): Cohere2MoeStreamParser (arch 12 marker machine), build-only (Axis A, task 5)

## What I ported (verbatim from generate_cohere2moe ~12560-12866)
- `Cohere2MoeStreamParser` struct + `C2mSec{Pre,Think,Text,Action}` in daemon.rs (before generate_cohere2moe). `new(tokenizer, tools, max_tokens, max_think_tokens)` resolves the 6 marker ids + `<PAD>` via special_token_id, builds known_tools/tool_params, computes think_budget (think_reserve clamp).
- `feed(tok,bytes)`: pad/repeat(24×) guards → Stop; marker-id section switch (suppressed); else route by section — Think→Emit{reasoning:true}+think_count, Text/Pre→Emit{reasoning:false}+vis_buf, Action→action_buf; is_marker frag suppression; END_ACTION→parse_cohere_action+snap_call_names→ToolCalls(calls array).
- `next_forced()`: think-budget force-close pre-check (enqueue [END_THINKING,START_TEXT] when in Think past budget, nothing visible) then drain queue.
- `on_eos()`: empty-turn guard → Inject([END_THINKING?,START_TEXT]) up to MAX_EOS_SUPPRESS=3, else Stop (NO eos commit — matches legacy 12708).
- `finish()`: tool-call-from-text recovery from vis_buf.

## Trait addition (affects T3-driver / T7)
- Added `StreamParser::enqueue(&mut self, tok)` (default no-op) in stream_parser.rs. Cohere2MoeStreamParser impls it (push to forced queue). **T7 must**: on the driver's `on_eos()->Inject(v)` arm, call `parser.enqueue(t)` for each t in v, then `continue` (do NOT commit the eos). T3 currently left the Inject arm a bare break — T7 fixes this (the structural "eos-checked-before-commit" change).
- ToolCalls(v) carries the calls ARRAY; the driver's execute_action (daemon.rs:8236, added by T3) already renders `{"type":"tool_calls","id":id,"calls":<v>}`. Confirmed matches.

## Tests
- 2 section-machine unit tests in `mod c2m_stream_parser_tests` (daemon.rs): section routing+marker suppression, empty-turn on_eos Inject-then-Stop. Run: `cargo test --example daemon --features deltanet -p hipfire-runtime c2m_stream_parser` → 2/2 pass. (Example test, not lib — cargo test --lib won't run it.)
- Full: daemon build Finished clean; `cargo test --lib -p hipfire-runtime` 342 passed.

## Notes / uncertainties for T7/T8
- `feed` uses the driver-provided running-vector `bytes` as the frag (via from_utf8_lossy), NOT legacy's single-token `tokenizer.decode(&[tok])`. Equivalent for special markers + normal text; any BPE-boundary difference is caught by T8 event-equivalence (cohere2moe is faithful, not byte-identical).
- think_count only increments on emitted Think tokens (matches legacy 12809).
- T7 wiring: generate_cohere2moe keeps its jinja/batched-prefill preamble, constructs Cohere2MoeStreamParser::new(...) via the stream_parser() hook override on a Cohere2MoeDispatch, and drives ar_generate. The on_eos Inject arm + eos-not-committed is the key structural piece.
