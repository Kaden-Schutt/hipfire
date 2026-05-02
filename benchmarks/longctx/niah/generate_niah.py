#!/usr/bin/env python3
"""Generate deterministic NIAH single-needle fixtures.

Writes niah_{8k,16k,32k,64k,128k}.jsonl. Each file holds one JSON record:
  context_tokens: the requested context length (token count is approximate;
                  filler is char-budgeted under the assumption ~3.5 chars per
                  token for English prose, and the bench prints the actual
                  tokenized length).
  needle:         the secret-fact sentence inserted at ~50% depth.
  question:       the question the model must answer after the context.
  expected_answer_substring: substring that must appear in the model's output
                  for the case to count as PASS.
  filler_text:    deterministic filler from a fixed seed.

Re-running with no args produces byte-identical files because all random calls
go through a single seeded Random instance.
"""

import json
import random
import sys
from pathlib import Path

SEED = 0x5407_FFFF
RNG = random.Random(SEED)

# Common English filler words; drawn with replacement to fill the haystack.
WORDS = (
    "the quick brown fox jumps over the lazy dog and then settles down for a "
    "long restful nap on a warm patch of sunlit grass while a gentle breeze "
    "carries the scent of distant pine forests across the meadow where small "
    "wildflowers nod their heads in agreement with the rhythm of the afternoon "
    "and time itself seems to pause for just a moment so that everything alive "
    "can breathe in synchrony with the slow steady pulse of the surrounding "
    "world before the sun begins its descent toward the western horizon "
    "painting clouds in shades of amber rose and indigo as evening approaches"
).split()

NEEDLE = "The secret pass code is mauve-velociraptor-7741."
QUESTION = (
    "Based on the document above, what is the secret pass code? "
    "Answer in one short sentence."
)
EXPECTED = "mauve-velociraptor-7741"

# Approximate chars per token for English prose with the Qwen3 tokenizer.
# Empirically about 3.7 (qwen3-0.6b BPE on our WORDS list); use 3.5 as a
# conservative undershoot so token counts come in slightly above target rather
# than below — the bench reports the actual tokenized length anyway.
CHARS_PER_TOKEN = 3.5

# Fraction of context where the needle goes. 0.5 puts it at the midpoint.
NEEDLE_DEPTH = 0.5

CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072]


def make_filler(approx_tokens: int) -> str:
    """Build deterministic filler text targeting `approx_tokens` tokens."""
    target_chars = int(approx_tokens * CHARS_PER_TOKEN)
    out = []
    chars = 0
    while chars < target_chars:
        w = WORDS[RNG.randrange(len(WORDS))]
        out.append(w)
        chars += len(w) + 1  # +1 for the space we'll join on
    return " ".join(out)


def assemble(context_tokens: int) -> str:
    """Place the needle at ~NEEDLE_DEPTH inside `context_tokens` of filler."""
    pre_tokens = int(context_tokens * NEEDLE_DEPTH)
    post_tokens = context_tokens - pre_tokens
    pre = make_filler(pre_tokens)
    post = make_filler(post_tokens)
    return f"{pre}\n\n{NEEDLE}\n\n{post}"


def main():
    out_dir = Path(__file__).parent
    for ctx in CONTEXT_SIZES:
        # Reset the RNG for each fixture so individual files are independent
        # (re-running with --only 32k won't shift later sizes).
        RNG.seed(SEED + ctx)
        filler_text = assemble(ctx)
        # Sanity: the needle must appear exactly once in the assembled text.
        assert filler_text.count(NEEDLE) == 1, \
            f"needle appears {filler_text.count(NEEDLE)}x at ctx={ctx}"
        record = {
            "context_tokens": ctx,
            "needle": NEEDLE,
            "question": QUESTION,
            "expected_answer_substring": EXPECTED,
            "filler_text": filler_text,
        }
        out_path = out_dir / f"niah_{ctx // 1024}k.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
        approx_chars = len(filler_text)
        print(f"wrote {out_path.name}: {approx_chars} chars (~{approx_chars / CHARS_PER_TOKEN:.0f} tokens)")


if __name__ == "__main__":
    sys.exit(main() or 0)
