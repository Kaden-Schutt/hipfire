# 2026-08-15 — Qwen3.8 `reasoning_effort` is unreachable; the user-facing dial is a token cap

**Status: diagnosis + proposal. Nothing implemented.**

Ground truth is the upstream clone at
`/mnt/nas/kaden/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
(`README.md`, `chat_template.jinja`). Every template behaviour below was
**rendered**, not read — reference `jinja2` 3.1.6, the model's own template,
one probe per value hipfire can emit.

## 1 · What the model specifies

Three *orthogonal* controls. They are not a ladder.

| control | type | default | mechanism |
|---|---|---|---|
| `enable_thinking` | bool | `true` | template emits a **pre-filled empty** `<think>\n\n</think>\n\n` at the generation prompt (`chat_template.jinja:165-168`) |
| `reasoning_effort` | enum **`xhigh` / `medium` / `low`** | `xhigh` | an **injected system-prompt sentence** (`:47-55`) |
| `preserve_thinking` | bool | `true` | retains prior turns' `reasoning_content` in history (`:116-117`) |

Two things matter and are easy to miss:

- **`reasoning_effort` is prompt text, not a budget.** `xhigh` injects *"Reasoning
  effort is set to xhigh. Please think carefully through the task, validate key
  assumptions…"*; `low` injects a brevity instruction; **`medium` injects
  nothing** — it is the unsteered baseline, not a middle sentence.
- **Anything outside the three-value set is a hard error**, by the model's own
  design (`:48-49` `raise_exception`).

Rendered proof, all values hipfire's `REASONING_EFFORTS` can produce:

```
(unset/auto)           -> xhigh-instr     | OPEN <think>
low                    -> low-instr       | OPEN <think>
medium                 -> NO instruction  | OPEN <think>
high                   -> *** REJECTED: Unexpected reasoning effort high.
xhigh                  -> xhigh-instr     | OPEN <think>
max                    -> *** REJECTED: Unexpected reasoning effort max.
none                   -> *** REJECTED: Unexpected reasoning effort none.
enable_thinking=False  -> NO instruction  | CLOSED <think></think>
```

The card also specifies **different sampling per mode** (`README.md:252-253`):
thinking `temp=1.0 top_p=0.95 top_k=20 presence=0.0`; non-thinking
`temp=0.7 top_p=0.80 top_k=20 presence=1.5`.

## 2 · What hipfire does, layer by layer

Two layers are already correct. The defects are all above them.

| layer | verdict |
|---|---|
| **Renderer** `hipfire-runtime/src/prompt_frame.rs` | **CORRECT.** `raise_exception` is registered (`:1168`) and bubbles up as `Err(String)`. Seven tests already cover low / medium / xhigh / unset / none / unsupported (`:3546-3610`). |
| **Daemon resolver** `hipfire-engine/src/prompt.rs:13-26` `qwen_jinja_reasoning` | **CORRECT.** `none`/`off`/`chat` → `(false, None)` → closed think block; `auto`/absent → undefined → template default; everything else passes through verbatim. |
| **Config enum** `hipfire-config/src/lib.rs:495` | **D1 — accepts what the model rejects.** `REASONING_EFFORTS = ["auto","none","low","medium","high","xhigh","max"]`. `high` and `max` validate fine, then hard-fail at render. |
| **TUI selector** `hipfire-tui/src/hipfire/knobs.rs` | **D2 — the dial is missing.** 23 knobs; **none is `reasoning.effort`.** The only reasoning controls are `thinking` (on/off) and `thinking_budget` — a **token cap** whose options are named `low/med/high/xhigh/max` (512/2048/8192/24576/32768, `hipfire-cli/src/main.rs:6739-6743`). Identical vocabulary to the effort dial, entirely different mechanism. |
| **`ThinkMode`** `prompt_frame.rs:80-121` | **D3 — no `Medium`.** `from_str` folds `"medium"` into `Low` (`:118`), so the two rungs the model *does* distinguish collapse into one. It also carries `High`/`Max`, which this model rejects. |
| **`preserve_thinking`** | **D4 — unimplemented.** `JinjaChatFrame` has no such field; the only occurrences are in the diagnostic example `hipfire-runtime/examples/jinja_render_dump.rs`. Template default `true` therefore always applies. |
| **Registry** `registry/models.json` (`qwen3.8:27b`) | **D5 — one sampling profile.** `recommended_settings` and the single `general` profile carry the card's *thinking-mode* numbers. Disabling thinking leaves `temp=1.0 / top_p=0.95 / presence=0.0` where the card asks for `0.7 / 0.80 / 1.5`. |

## 3 · The reported symptom, mechanised

> *"it just defaults to xhigh and if a user selects an effort lower than xhigh
> it force closes xhigh rather than allowing the user to actually change its
> reasoning effort"*

That is exactly what the code does, and it is a **collision of two dials that
share a vocabulary**:

1. The registry pins `reasoning_effort: "xhigh"` + `thinking_budget: "uncapped"`
   for `qwen3.8:27b`, lowered to config as `reasoning.effort` / `reasoning.budget`
   (`hipfire-registry/src/lib.rs:142-151`).
2. A user wanting less reasoning reaches for the TUI — which offers **only**
   `thinking_budget` — and picks `low`.
3. That sets `max_think_tokens = 512`. **It does not touch `reasoning.effort`,
   which is still `xhigh`.**
4. The model is instructed *"think carefully… validate key assumptions…
   consider plausible alternatives"* and begins a deep trace.
5. At 512 tokens the cap latches and the runtime **force-closes `<think>`**
   (`hipfire-generate/src/qwen.rs:5374`).

The reasoning was never reduced — it was *guillotined mid-thought*. Worse than
either endpoint: the model pays xhigh's verbosity, then loses the conclusion
that justified it.

And the one dial that *would* work, `reasoning.effort`, is CLI/config-only
(`hipfire config set reasoning.effort medium`) and absent from the TUI — while
two of its five accepted values (`high`, `max`) fail the request outright.

## 4 · Proposed changes

Ordered by value. F1+F2 alone resolve the report.

### F1 — Derive the supported effort set from the template, per model

Do **not** hardcode a per-model table; it will drift from the 61-model registry.
The template already states the truth (`:48`), so ask it: at model load, render a
one-message prompt once per candidate effort and keep the set that renders. Cost
is a handful of microseconds, it cannot go stale, and it **degrades correctly** —
a permissive template accepts all seven, yielding no constraint.

Then:
- validation rejects an out-of-set value with a message naming the supported
  set, instead of today's opaque jinja render error;
- the TUI and `hipfire config` **enumerate the model's actual rungs**, so a
  Qwen3.8 user is offered exactly `low / medium / xhigh` (+ off).

For a global `reasoning.effort=high` predating the model, two options — a
maintainer's call:
- **(a) strict:** fail with the supported set. Honest, matches the model, but
  breaks an existing global config on first use.
- **(b) clamp + report:** rank-project `high|xhigh|max → xhigh`, and surface the
  projection in the receipt so it is visible rather than silent. Friendlier;
  makes `high` and `xhigh` synonyms on this model.

I lean **(b)** with the projection in the receipt, since `reasoning.effort` is
frequently a global preference rather than a per-model one.

### F2 — Give the TUI the real dial, and stop the vocabulary collision

- Add a `reasoning.effort` knob populated from F1's per-model set.
- Rename `thinking_budget`'s options away from the effort vocabulary
  (e.g. `512 / 2048 / 8192 / 24576 / 32768` or `tight/…/generous`). Two dials
  offering `low/high/xhigh` with unrelated meanings is the whole bug.
- State the interaction in the knob note: **effort shapes the reasoning; budget
  truncates it.** A budget below what the chosen effort tends to spend will
  force-close mid-thought — the failure the user hit.

### F3 — Add `ThinkMode::Medium`

`prompt_frame.rs:80-121`. Today `"medium"` and `"low"` are the same variant, so
any path routed through `ThinkMode` cannot express two of this model's three
rungs. Update `from_str`, the `reasoning_effort_levels_remain_distinct` test
(`:129`), and audit the arch consumers — `glimmer_reasoning_strength`
(`hipfire-generate/src/dense.rs:2397-2424`) currently *works around* this exact
gap via the token budget, and its comment says so.

### F4 — Plumb `preserve_thinking`

Add the field to `JinjaChatFrame`, a `reasoning.preserve` config key, and pass
it through `qwen_jinja_reasoning`'s siblings. Default `true` to match the card.
Until then the template default silently governs and the documented
"disable preserved thinking" path is unreachable.

### F5 — Mode-dependent sampling

Add a `non_thinking` sampling profile to `qwen3.8:27b` / `:27b-fast` carrying
the card's `temp=0.7 top_p=0.80 top_k=20 min_p=0.0 presence_penalty=1.5`, and
select it when thinking is disabled. `registry/v1.json` is generated — edit
`registry/models.json` and re-run `scripts/registry_gen.py`.

## 5 · Verification owed

`prompt_frame.rs` already has the renderer-level tests. What is missing is a
test at the layer that actually broke:

- `reasoning.effort` ∈ {low, medium, xhigh} each reaches the template and
  produces the expected instruction (or absence, for `medium`);
- an out-of-set effort produces a **named validation error**, not a render
  failure;
- lowering `reasoning.budget` **does not** change `reasoning.effort` — and the
  knob note says so;
- with thinking off, the non-thinking sampling profile is the one applied.

Then a live turn on `qwen3.8:27b` at each rung, reading the decoded output and
comparing think-block length — numbers alone never prove this one, since the
failure mode is a *plausible-looking truncated trace*.
