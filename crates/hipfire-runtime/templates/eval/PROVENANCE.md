# Chat-template eval fixtures (item #37 — ChatScaffold→jinja for qwen)

Fixtures for `examples/template_cache_eval.rs`, which measures chat-template
prefix-cache behaviour (forward-extension LCP) through hipfire's own minijinja
env (trim_blocks + lstrip_blocks + strict + pycompat). NOT yet wired as the
production qwen render path.

- **qwen35-official-reference.jinja** — the official Qwen3.5/3.6 chat_template,
  extracted verbatim from `qwen3.5-0.8b.tier1.mq4.hfq` metadata. Interleaved
  thinking (drops prior-turn reasoning once a new user turn arrives). Reference
  for comparison.
- **qwen35-froggeric-v20.jinja** — froggeric/Qwen-Fixed-Chat-Templates
  (https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), version
  "qwen3.6-froggeric-v20", **Apache-2.0** (compatible with this repo).
  Adds `preserve_thinking` (keep history reasoning → 100% KV forward-extension),
  agentic-loop fixes (retry/stall/empty-think), and engine compatibility
  (avoids loop.previtem / Python-only filters).

## Measured (qwen3.5-0.8b tokenizer, 2-turn colors conversation)
| template  | preserve_thinking | LCP / turn1_kv | forward-extension |
|-----------|-------------------|----------------|-------------------|
| official  | (interleaved)     | 18/59 = 30.5%  | no                |
| froggeric | false (default)   | 18/59 = 30.5%  | no (≡ official)   |
| froggeric | true              | 59/59 = 100%   | YES               |

preserve_thinking=true gives plain-LCP 100% caching for qwen — no verbatim
token-splice and no DeltaNet rewind. Coherence note: qwen's current ChatScaffold
already feeds reasoning-in-history (verbatim splice) and is validated-coherent,
so froggeric+preserve is the same reasoning-in-history rendered via the trained
template — low coherence risk. NEXT: wire qwen generate() to render via this
(flag-gated) + reasoning_content from asst_turn_cache + run both coherence gates.
