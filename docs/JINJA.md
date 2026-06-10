# Jinja chat templates

Since **2026-06-09** the daemon renders chat prompts through a Jinja
`chat_template` **by default, for every architecture** — the same
template mechanism HuggingFace's `tokenizer.apply_chat_template` uses.
Before this flip, prompts were built by a hand-rolled ChatML scaffold
that had drifted from what the models were actually trained on (JSON
tool calls where Qwen 3.5/3.6 expects the XML `<function=NAME>` form,
`<|im_start|>user` wrapping for tool responses instead of
`<|im_start|>tool`, missing default system prompts, LFM2.5's
`<|startoftext|>` BOS dropped entirely). Rendering the trained template
fixes that class of bug at the source.

Opt out (revert to the hand-rolled ChatML scaffold) with:

```bash
HIPFIRE_JINJA_CHAT=0 hipfire serve
```

## Template resolution order

At model load the daemon resolves which template to use
(`resolve_chat_template` in `crates/hipfire-runtime/examples/daemon.rs`),
first match wins:

1. **`HIPFIRE_CHAT_TEMPLATE_FILE=<path>`** — operator escape hatch /
   debugging knob. If set and readable, this template wins for any
   model loaded by that daemon.
2. **Per-model override file** at
   `~/.hipfire/templates/<model-file-basename>.j2`
   (e.g. `qwen3.5-9b.mq4` → `~/.hipfire/templates/qwen3.5-9b.mq4.j2`).
   Lets you override one model without env-var globalness.
3. **Arch-default bundled templates:**
   - Qwen 3.5 / 3.6, dense + MoE (arch_id 5 / 6) → the bundled
     **froggeric** template
     (`crates/hipfire-runtime/templates/eval/qwen35-froggeric-v20.jinja`,
     from [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates),
     Apache-2.0). Byte-equivalent to the official Qwen template for
     standard renders, with agentic-loop fixes and engine
     compatibility; render-equivalence is locked by a durability test
     (`templates/eval/DURABILITY-2026-06-09.md`).
   - LFM2.5 (arch_id 11) → the `.hfq`-embedded template when present
     (the 350M ships one), else the bundled **LiquidAI** template
     (`templates/eval/lfm2-liquidai.jinja`) — the 8B-A1B export ships
     no embedded template and renders garbage under plain ChatML, so
     the bundle is what makes it work out of the box.
4. **`.hfq`-embedded `tokenizer_config.chat_template`** for every other
   architecture.
5. **None of the above** → the render path falls back to the
   hand-rolled ChatML scaffold (`ChatFrame::Plain`).

A failed render (template parse error, missing context var, template
`raise_exception`) also falls back to Plain at generate time, with a
`[daemon] jinja render failed (…) — falling back to Plain` note on
stderr — a bad template never hard-fails a request.

**Exception:** DeepSeek V4 (arch_id 9) does not render through Jinja at
all. It has a dedicated DSML prompt builder
(`<｜User｜>…<｜Assistant｜>` framing plus grammar-constrained tool
calls) used by both the single-GPU and expert-parallel paths.

## HF byte-exactness

The render environment is set up to byte-match
`transformers.apply_chat_template` output:

- `trim_blocks=true, lstrip_blocks=true` — matches HF's Jinja
  environment construction. Without these, `{% … %}` tags leak
  surrounding whitespace, and for templates with
  history-length-dependent control flow (MiniMax-M2's
  `last_user_index` scan) the leak *varies by turn*, collapsing the
  prompt cache to LCP=1.
- **Strict undefined** — a missing context variable raises an error
  (and falls back to Plain) instead of silently rendering a malformed
  prompt.
- **pycompat method callback** — Python-style `.startswith` / `.split`
  / `.rstrip` / `|items` etc. work; the Qwen3 family template calls
  these throughout its tool branches.
- **`raise_exception`** is registered so templates can fail fast on
  malformed input (e.g. a system message mid-conversation).
- **HF-spaced `tojson`** — minijinja's builtin `tojson` is compact
  (`,`/`:`); hipfire overrides it with `", "` / `": "` separators
  (Python `json.dumps` defaults) and serializes with `serde_json`'s
  `preserve_order` feature, so tool definitions and mapping-valued
  tool-call arguments render with the exact bytes and key order the
  model saw in training.

Context passed to the template: `messages`,
`add_generation_prompt=true`, `enable_thinking`, `bos_token`, `tools`,
`documents` (empty), `tool_call_kwargs`.

- `enable_thinking` maps from the request's think budget
  (`max_think_tokens != 1`); for Qwen thinking models `false` renders
  the empty-think pattern.
- `bos_token` defaults to decoding the tokenizer's `bos_id`; loaders
  can pass an explicit string when the cosmetic decode doesn't match
  the canonical BOS the template expects.

## Tools and multi-turn

When the request carries a `tools` array or a `messages` history, the
daemon renders through the multi-turn entry point
(`JinjaChatFrame::render_messages`) so the template's `{% if tools %}`
and history branches fire. With neither, the single-turn `render()`
convenience synthesizes `[system?, user]`. Messages carry `role`,
`content`, `tool_calls` (`{name, arguments}`), and optional
`tool_call_id` — all four probed-safe under strict-undefined.

## Prompt cache under Jinja (thinking models)

Jinja renders the **full conversation every turn** (stateless render);
the daemon then takes the longest common prefix against what's already
prefilled and forward-extends. The catch for thinking models: the
template renders a historical assistant turn from its API-visible
`content`, but the model's actual emission included `<think>…</think>`
tokens that the API stripped — a naive re-render diverges at the first
assistant turn and the cache misses.

The fix is **verbatim splice**
(`prompt_frame::build_cached_history_jinja`): each cached assistant
turn's content is substituted with an atomic reserved special-token
sentinel, the conversation is rendered through the trained template,
and the sentinel is then replaced with the verbatim generated tokens of
that turn (including reasoning and the generation primer). The spliced
stream byte-matches what was actually prefilled, so the LCP
forward-extension cache hits. Substitution is verified (exactly one
sentinel occurrence per cached turn); any mismatch falls back to a
plain render — always valid, just uncached.

Debug knobs: `HIPFIRE_QWEN_CACHE_TRACE=1` (cache eligibility + lookup
trace), `HIPFIRE_QWEN_PROMPT_CACHE=0` (kill switch).

## Debugging a template

```bash
# Dump the rendered prompt for a model + message set (byte-compare vs
# transformers' apply_chat_template):
cargo run --release -p hipfire-runtime --example render_chat_template
cargo run --release -p hipfire-runtime --example jinja_render_dump

# Force a candidate template for one daemon:
HIPFIRE_CHAT_TEMPLATE_FILE=/tmp/candidate.j2 hipfire serve

# Prefix-cache behavior of a template (forward-extension eval):
cargo run --release -p hipfire-runtime --example template_cache_eval
```

Bundled template provenance and the official-vs-froggeric
render-equivalence audit live in
`crates/hipfire-runtime/templates/eval/PROVENANCE.md` and
`templates/eval/DURABILITY-2026-06-09.md`.
