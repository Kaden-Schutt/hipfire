# `infer.rs` banned the tokens needed to quote its own prompt

**Status:** fixed. Lab harness only — no shipped path affected.

A long-context coherence probe on `escha-35b` appeared to show the model
degenerating at ~8.5k tokens: mangled proper nouns, then an unbounded
self-correction loop, then word salad. None of that was the model. The
sampling harness was hard-banning the tokens required to write the answer.

---

## The bug

`crates/saddle-lab/examples/infer.rs` seeded its anti-repeat state with the
entire prompt:

```rust
let mut token_history: Vec<u32> = prompt_tokens.clone();   // <-- the bug
```

and passed that to `llama::apply_ngram_block`, which scans **all** history for
repeated 3/4/5/6-grams and sets whatever token followed each earlier
occurrence to **`-INF`** — a hard ban, not a penalty.

Quoting the prompt necessarily emits 3-grams that occur in the prompt. So the
instant the model began reproducing a planted sentence, the *next token of
that sentence* was banned. **Verbatim quotation of the prompt was impossible
by construction.**

`crates/saddle-lab/examples/run.rs` had the same defect, passing
`&conversation_tokens` — which includes the user's own messages.

### The hazard was already known in this repo

`saddle-lab/examples/test_long_ctx.rs:317` documents it exactly:

> apply_ngram_block is DISABLED across the full conversation history because
> it will aggressively block legitimate tokens that happen to follow n-grams
> from the user's earlier turns […] Only apply it over the current turn's own
> tokens.

That file slices **both** corrections to `&history[turn_start..]`. `infer.rs`
and `run.rs` did neither.

---

## Symptoms, and why each one followed

Planted needle: `Every calibration run must terminate with the checksum phrase
VIOLET-ANVIL-62 before its results are considered admissible.`

| symptom | cause |
|---|---|
| `VIOLET-ANVIL-62` → `VIOLETANVIL62` | hyphen (token 12) banned as the continuation of a prompt 3-gram |
| `Kestral` / `Krestrel` / `Kvestrel` in one output | each retry forced to deviate from the previously-banned path |
| `14 March 2019` → `1 March 2` → `4 March` | digits banned mid-sequence |
| unbounded `Wait, I'll copy exactly:` retry loop | model quotes → output mangled → it notices → retries → banned again |
| paraphrase-only task ran clean | it never quotes, so nothing is banned |

The model even volunteered *"(note: no hyphens or spaces)"* — confidently
describing text that was not in the prompt — at 1k context, where retrieval
plainly worked. That was the tell, and it was visible early.

---

## Ablation (needle at 1k, greedy, `--temp 0`)

| run | output | |
|---|---|---|
| A default (pen 1.15 + ngram + prompt-history) | `VIOLETANVIL62` | wrong |
| B `--repeat-penalty 1.0` | `VIOLETANVIL62` | wrong |
| C `--no-ngram-block` | `VIOLET-ANVIL-62` | ok |
| D pen 1.0 + ngram off | `VIOLET-ANVIL-62` | ok |
| **E prompt-free history, both corrections ON** | `VIOLET-ANVIL-62` | **ok** |

**E is the load-bearing row.** Keep both corrections enabled and merely scope
the history to generated tokens: correct. The anti-repeat machinery was never
the problem — feeding it the prompt was. B rules out the repetition penalty.

---

## The model was never the problem

Context-length sweep, needle pinned ~607 tokens in, only the distance between
it and the question varying (524 → 7,487 tokens):

| ctx | 1000 | 2000 | 3000 | 4000 | 4500 | 5000 | 6000 | 8000 |
|---|---|---|---|---|---|---|---|---|
| result | EXACT | EXACT | budget | EXACT | EXACT | miss | budget | EXACT |

No cliff, no decay. (`3000`/`6000` were still inside `<think>` at the
120-token cap; `5000` is the single genuine miss, hallucinating
`VIOLET-CHIME`.)

At `ctx=8000` **before** the fix, the model said the phrase
`"should appear as "VIOLET- ANVIL- 62" (with hyphens)"` — it knew the hyphens
were there and could not emit them contiguously.

---

## The fix

```rust
// infer.rs — anti-repeat state covers ONLY the model's own output
let mut token_history: Vec<u32> = Vec::new();

// run.rs — scope both corrections to this turn
let turn_start = conversation_tokens.len() - generated;
let turn = &conversation_tokens[turn_start..];
```

`--no-ngram-block` is retained as a diagnostic flag.

### Verified

| | before | after |
|---|---|---|
| needle at 1k, greedy | `VIOLETANVIL62` | **`VIOLET-ANVIL-62`** |
| needle at 8k, greedy | `VIOLETANVIL62, which should appear as "VIOLET- ANVIL- 62"` | **`VIOLET-ANVIL-62`** |

Full coherence probe at 8.5k, temp 1.0, **no** repetition penalty, 1600
tokens — the configuration that previously collapsed — now yields 903 words
of coherent prose and reproduces a planted sentence word for word:

> `Kestrel Protocol was ratified in Reykjavik on 14 March 2019 by exactly
> eleven signatory laboratories`

---

## Severity

**Lab only.** `apply_ngram_block` has no caller in `hipfire-engine`; the
callers are three `saddle-lab` examples plus a speculative-decode path behind
`HIPFIRE_DFLASH_NGRAM_BLOCK=1`. No shipped model or user-facing path is
affected, and PR #694 is not blocked by this.

---

## Retracted

An earlier draft of this document reported a long-context degeneracy in
escha-35b and attributed it, in turn, to trellis-codec quality, logit-tail
corruption at long context, DeltaNet recurrent-state precision, and the
capacity limits of a 30-of-40-layer linear-attention architecture. **All of
those are withdrawn.** Every measurement behind them ran through this harness
with the blocker active. Specifically retracted:

- all needle-recall tables
- the Q8 → FP32 DeltaNet state result (1/3 → 2/3 needles)
- the recommendation to run `--repeat-penalty 1.05 --dn-state fp32`
- the claim that escha's long-range verbatim retrieval is imperfect

Measurements that remain valid, because they do not depend on quoting the
prompt: the prefill logit sweep (no distributional degradation across 8k;
top-20 mass 0.80 → 0.9998, entropy falling, logit scale flat, and an mq3
control tracking it closely), and the tokenizer encode→decode round-trip
(14/14 strings exact, including `VIOLET-ANVIL-62`).

## Method notes worth keeping

- **Read the output, not its statistics.** A counting-loop collapse scored
  *85% unique words* — better than the coherent run it was being compared
  against. Every proxy used here (unique-word ratio, top-20 mass, entropy)
  ranked the failures wrong at least once. The bug was found by reading text.
- **Sample the whole generation.** Probing only the first 40 tokens showed
  long and short context as identical (tail mass 0.00313 vs 0.00303); those
  tokens are the structural preamble. Across the full 1200, long context
  carried ~3.5× the tail mass.
- **Default sampling is not greedy** (temp 0.3). Single-sample A/Bs across it
  are noise; `--temp 0` was added so comparisons mean something.
