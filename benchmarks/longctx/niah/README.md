# NIAH (Needle-in-a-Haystack) fixtures

Single-needle long-context retrieval cases for PFlash validation.

## Layout

Each `niah_<N>k.jsonl` is one JSON Lines record per case (currently 1 case per
file — keep simple for the MVP harness):

```json
{
  "context_tokens": 8192,
  "needle": "The secret pass code is mauve-velociraptor-7741.",
  "question": "What is the secret pass code?",
  "expected_answer_substring": "mauve-velociraptor-7741",
  "filler_text": "..."
}
```

## Generator

`generate_niah.py` writes deterministic fixtures from a fixed seed. The needle
is placed at ~50% depth by default. Token counts are approximate (filler is
char-budgeted and tokenized later); the bench harness reports the actual token
count alongside the requested context.

## Determinism

Generator uses `random.seed(0xPF1A5407)` so re-running produces byte-identical
files. Commit the JSONL outputs so the harness has a stable signal across
machines without requiring Python at bench time.
