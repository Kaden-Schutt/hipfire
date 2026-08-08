# cohere2moe guard-forcing fixtures

These four prompt files are designed to exercise specific decode-loop reactive guards
in `generate_cohere2moe` (daemon.rs ~12600–12900). They are used by the validation
harness for Axis A task 6 of the StreamParser work. Comments cannot go in the prompt
files themselves because the gate reads them verbatim via `cat` and passes the full
content to the model.

---

## cohere2moe_empty_turn.txt

**Target guard:** Guard 1 — Empty-turn EOS guard  
**md5:** `5ae96a97434ccc8af0b3e26e69068772`

**Mechanism:** North-Mini-Code is prone to "reason internally then emit
`<|END_OF_TURN_TOKEN|>` with nothing visible" on questions that have a simple
yes/no or true/false conclusion the model feels it has already resolved in
`<|START_THINKING|>`. The Goldbach conjecture framing (true/false, famously
unproven) triggers extended reasoning and a higher probability of the model
concluding inside `<think>` without emitting a `<|START_TEXT|>` answer.

**Expected event behavior:** Guard fires (`eos_suppressions` increments), daemon
injects `<|START_TEXT|>` (up to 3×), model then emits visible text. The
`{"type":"done"}` event must have `tokens > 0` and the visible output must be
non-empty.

**Uncertainty:** Medium. Whether North actually takes the empty-turn path depends
on model weights + temperature. At temp=0.0 (greedy) this prompt has a good chance
of eliciting reason-only behavior, but it is not guaranteed. May need live tuning —
if the model answers immediately in the text section, try a subtler binary question.

---

## cohere2moe_think_budget.txt

**Target guard:** Guard 2 — Think-budget force-close  
**md5:** `e76040ffa3f18c7677f0cdbd106f0f7c`

**Mechanism:** The 12-person bridge-crossing problem requires enumerating the
optimal strategy across many combinations, which causes North to emit a very long
`<|START_THINKING|>` reasoning trace. When run with a small `max_think_tokens`
(e.g. 80–120), the model will still be inside `<think>` when `think_count >=
think_budget`. The guard then injects `<|END_THINKING|>` + `<|START_TEXT|>` to
force an answer within the remaining budget.

**How to trigger:** Send with a small `max_think_tokens` override (e.g. 100) and
a total `max_tokens` large enough to still allow a brief answer (e.g. 400). The
gate harness that uses this file must set `max_think_tokens` explicitly; the
standard coherence gate rows do not set it, so this file is intended for a
dedicated guard-test row with a small `max_think` param.

**Expected event behavior:** `[cohere2moe] think-budget guard:` log line; model
emits partial reasoning then a forced answer. Total tokens > 0.

**Uncertainty:** Low for triggering the guard (any sufficiently long thinking trace
+ small budget will trip it). The bridge problem is reliably complex. Actual
answer correctness after force-close may vary — the guard is tested for correct
injection, not answer quality.

---

## cohere2moe_toolcall.txt

**Target guard:** Guard 3 — Tool-call via `<|START_ACTION|>`  
**md5:** `0819dea23832b677e2b970353f5480f2`

**Mechanism:** A direct real-time information request ("current weather in Tokyo
right now") that the model cannot answer from training data. When a tool schema is
supplied by the harness at runtime (e.g. a `get_weather` or `web_search` tool),
North should emit `<|START_ACTION|>...<|END_ACTION|>` containing a JSON tool-call
array. The daemon parses this into a `{"type":"tool_calls"}` event.

**How to use:** This prompt must be sent with a tool schema in the `tools` field of
the generate message. The harness should provide at minimum a `get_weather(location)`
or equivalent function. Without a tool schema, North will likely refuse or hallucinate
an answer rather than calling a tool.

**Expected event behavior:** `{"type":"tool_calls","id":"...","calls":[...]}` event
emitted; `tool_calls_emitted = true`; no `<|START_ACTION|>` leak in visible output.

**Uncertainty:** Medium. Requires the harness to supply a matching tool schema.
Without the schema the guard path is never reached. With a generic schema North is
reasonably reliable at using `<|START_ACTION|>` for real-time queries.

---

## cohere2moe_toolcall_as_text.txt

**Target guard:** Guard 4 — Tool-call-written-as-text recovery  
**md5:** `b0505108b9e68c83fb7c637b5c2c2939`

**Mechanism:** Asking North to "show the function call as a JSON object" primes it
to emit the call as visible text in the `<|START_TEXT|>` section rather than via
`<|START_ACTION|>`. The post-loop recovery in `generate_cohere2moe` (daemon.rs
~12867) runs `parse_cohere_action(&vis_buf)` on the accumulated visible output and,
if it finds a parseable tool-call array, emits a `{"type":"tool_calls"}` event as a
recovery action.

**How to use:** Send with a tool schema (same as `cohere2moe_toolcall.txt`) so the
recovery parser has a real tool name to snap to. Without a schema `known_tools` is
empty and `snap_call_names` has nothing to match against; the recovery still fires
but the tool name may be hallucinated.

**Expected event behavior:** `[cohere2moe] recovered N tool_call(s) written as text`
log line; `{"type":"tool_calls"}` event emitted post-loop; `tool_calls_emitted` was
false during the loop.

**Uncertainty:** Medium-high. Whether North writes the call as text vs using
`<|START_ACTION|>` depends on the exact phrasing and the tool schema. The current
wording ("Show me the exact function call you would make... Write it out as a JSON
object") biases toward text output. May need further tuning if North consistently
chooses `<|START_ACTION|>` even with this phrasing.
