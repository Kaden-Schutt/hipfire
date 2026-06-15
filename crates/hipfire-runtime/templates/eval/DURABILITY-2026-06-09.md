# froggeric chat-template durability verification — 2026-06-09

Gate for task #37 (retire qwen ChatScaffold → jinja). Run on k9lin / gfx1100,
tip `0c65915c`. Harness scripts under `$CLAUDE_JOB_DIR/tmp` (jinja_render_dump
example + hf_render_ref.py + parse_forward.py).

## ④ Render-vs-HF byte audit (GPU-free) — minijinja vs transformers jinja2

Identical context dict fed to both engines (`jinja_render_dump` emits the exact
ctx the daemon's `render_messages` builds; `hf_render_ref.py` replicates
`ImmutableSandboxedEnvironment(trim_blocks,lstrip_blocks)` + HF tojson/raise).

"embedded" = the chat_template actually inside qwen3.5-9b.mq4 (dumped via
`examples/dump_embedded_template`), NOT a fixture.

**CONFOUND CAUGHT + CORRECTED (the embedded jinja WAS confounding):** the daemon's
`Message` struct ALWAYS serializes `tool_calls: []` (serde default) and has NO
`reasoning_content` field. First-pass fixtures omitted `tool_calls` → the embedded
template (`message.tool_calls and …`, NO `is defined` guard, line 105) hit
strict-undefined and MINIJINJA-ERRORED, making it look "fragile". With the
daemon-FAITHFUL shape (tool_calls:[] on every message, reasoning as a `<think>`
block in content) that error vanishes. Corrected table:

| fixture (faithful)   | embedded-9b mj-vs-HF | froggeric mj-vs-HF | embedded==froggeric |
|----------------------|----------------------|--------------------|---------------------|
| plain (sys+user)     | **IDENTICAL**        | **IDENTICAL**      | same render         |
| reasoning-in-history | **IDENTICAL**        | **IDENTICAL**      | same render         |
| tools                | DIFFER (tojson)      | DIFFER (tojson)    | DIFFER (format)     |
| multi-step agentic   | DIFFER (tojson)      | DIFFER (tojson)    | DIFFER (format)     |

**Cache-critical paths (plain / reasoning) are byte-identical to HF AND to each
other** for the daemon's real message shape — neither template is more fragile.
RETRACTED: the earlier "froggeric strictly more robust / official hard-errors on
multi-turn" was a fixture artifact (missing the `tool_calls:[]` default the daemon
always provides). froggeric's `is defined` guards only help a missing-`tool_calls`
shape the daemon never produces — theoretical, not a production differentiator.
The embedded==froggeric byte-equality on non-tool paths also explains why ②③'s
embedded and froggeric multi-turn forward outputs were byte-identical (same render
→ same greedy output; confirms NO ChatScaffold fallback occurred).

froggeric-vs-embedded TOOL-format delta (jinja2 both): same `<tool_call><function=
NAME>` call grammar (what the model trained on) + richer agentic instructions and a
`<think>` scaffold in the system block — NOT a syntax change; ②③ confirms froggeric
emits valid calls. Benign.

### The one real divergence: `| tojson` (tool paths only, NOT froggeric-specific)
minijinja's `tojson` differs from HF jinja2 in **two** ways, both only on
structured `| tojson` output (tool DEFINITIONS at template L85, mapping-args L222):
1. **separators** — minijinja `{"a":"b"}` vs HF `{"a": "b"}` (compact vs spaced)
2. **key order** — keys render alphabetically SORTED vs HF insertion-order

Root cause: (1) minijinja's tojson filter is compact; (2) `serde_json = "1"` has
no `preserve_order` → `Value::Object` is a BTreeMap, so the daemon sorts request
tool-keys at parse time. Tool *results* and string-args tool_calls are pass-through
strings → byte-identical. **This is a pre-existing jinja-tool-render skew that hits
the official template identically — not a reason to prefer ChatScaffold, but it
should be fixed before relying on jinja for agentic/tool workloads.**

Fix: register an HF-compatible `tojson` filter on the render env
(`": "`/`", "` separators) AND build serde_json with `features=["preserve_order"]`
so request tool-key order survives to the template.

## ②③ Real-forward A/B (qwen3.5-9b.mq4, temp 0 greedy, daemon stdin one-shot)

Identical request JSONL, only daemon env (template) differs. mt1-3 = 3-turn
sheep-math with reasoning-in-history; ag1-3 = tool-call → ERROR → retry → success.

| case                   | mt1 | mt2 | mt3 | ag1 (call) | ag2 (retry) | ag3 (synth) |
|------------------------|-----|-----|-----|------------|-------------|-------------|
| embedded-official-jinja| ok  | ok  | ok  | CALL ok    | retry ok    | **EMPTY**   |
| froggeric-interleaved  | ok  | ok  | ok  | **EMPTY**  | retry ok    | synth ok    |
| froggeric-PRESERVE     | ok  | ok  | ok  | CALL ok    | retry ok    | synth ok    |

- Multi-turn: all coherent, no leaks/loops (uniq 0.61-0.75, 3gram_rep ≤0.05).
  embedded ≡ froggeric-interleaved mt1/mt2 text byte-for-byte (confirms ④ render
  byte-identity → identical greedy output on real forward).
- Agentic: each template has exactly ONE fragile cell on 9b greedy (embedded:
  empty ag3 synthesis; froggeric-interleaved: empty ag1 first-call). Deterministic
  (temp 0) but narrow; **froggeric-PRESERVE clean on all 6**. Prior 27B validation
  (memory 2026-06-08) showed froggeric's first tool call works → the 9b ag1-empty
  is a small-model/greedy fragility, symmetric with embedded's ag3-empty, NOT a
  froggeric defect. ag2 correct typo→retry under all 3 templates.

## ① Cache-on ≡ cache-off byte-identity — BLOCKED (the gold-standard gate)
Cannot run for the jinja path: cache is DISABLED under jinja today (daemon
8857-8864) and the daemon doesn't thread `preserve_thinking` (render_messages ctx
omits it) → the cache benefit doesn't exist until item-4 wiring lands. This is THE
test that must pass post-wiring (deterministic DeltaNet state via HIPFIRE_DN_STATE_EF
or FP32 + HIPFIRE_DETERMINISTIC=1, else stochastic Q8 masks corruption).

## Verdict
- froggeric render is **durable for the cache/reasoning path** (byte-identical to
  HF *and* to the model's own embedded template) and **coherent on real forwards**
  (multi-turn + agentic), on par with the embedded/official template — NOT more
  fragile, NOT more robust, in the daemon's real message shape.
- **Not yet flip-the-default ready**, for two reasons orthogonal to the template:
  (1) the cache-under-jinja wiring (item-4) doesn't exist, so flipping gains nothing
  for caching; (2) the `| tojson` tool-render skew should be fixed first.
- Recommend: land item-4 wiring → run ① byte-identity → fix tojson/preserve_order →
  then flip default behind the existing `HIPFIRE_JINJA_CHAT` flag with ChatScaffold
  fallback retained.
