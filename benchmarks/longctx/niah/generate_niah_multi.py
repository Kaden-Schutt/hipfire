#!/usr/bin/env python3
"""Generate deterministic multi-needle NIAH fixtures.

Writes niah_multi_{16k,64k}.jsonl per PRD §6 Phase 5 multi-needle gate.
Each file has 3 needles inserted at depths {0.25, 0.50, 0.75} of the
context. Each needle is a (key, value) pair with a unique secret value
so the model has to pull all three out of distinct positions, not just
the most recent.

PASS bar: at least min_recovered=2 of 3 expected substrings appear in
the model's answer. The bench will be extended to support a list of
expected substrings; for now the JSON record carries all three.

Re-running with no args is byte-identical (single seeded RNG).
"""

import json
import random
import sys
from pathlib import Path

SEED = 0x5407_FEEE  # distinct from single-needle seed (0x5407_FFFF)
RNG = random.Random(SEED)

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

NEEDLES = [
    ("courier color",    "indigo-octahedron-9931"),
    ("vault key",        "fenrir-quartz-2247"),
    ("safehouse city",   "saint-petersburg-rotunda-5808"),
]
NEEDLE_DEPTHS = [0.25, 0.50, 0.75]

QUESTION = (
    "Based on the document above, list each of the following: "
    "the courier color secret, the vault key secret, and the safehouse city "
    "secret. Use one short sentence per fact."
)
EXPECTED_SUBSTRINGS = [v for (_, v) in NEEDLES]
MIN_RECOVERED = 2

CHARS_PER_TOKEN = 3.5
CONTEXT_SIZES = [16384, 65536]


def make_filler(approx_tokens: int) -> str:
    target_chars = int(approx_tokens * CHARS_PER_TOKEN)
    out = []
    chars = 0
    while chars < target_chars:
        w = WORDS[RNG.randrange(len(WORDS))]
        out.append(w)
        chars += len(w) + 1
    return " ".join(out)


def assemble(context_tokens: int) -> str:
    """Place 3 needles at NEEDLE_DEPTHS inside context_tokens of filler."""
    sorted_needles = sorted(zip(NEEDLE_DEPTHS, NEEDLES))
    fragments = []
    prev_depth = 0.0
    for depth, (key, value) in sorted_needles:
        slice_tokens = int(context_tokens * (depth - prev_depth))
        fragments.append(make_filler(slice_tokens))
        fragments.append(f"\n\nThe {key} is {value}.\n\n")
        prev_depth = depth
    tail_tokens = int(context_tokens * (1.0 - prev_depth))
    fragments.append(make_filler(tail_tokens))
    return "".join(fragments)


def main():
    out_dir = Path(__file__).parent
    for ctx in CONTEXT_SIZES:
        RNG.seed(SEED + ctx)
        filler_text = assemble(ctx)
        for sub in EXPECTED_SUBSTRINGS:
            assert filler_text.count(sub) == 1, \
                f"needle {sub!r} appears {filler_text.count(sub)}x at ctx={ctx}"
        record = {
            "context_tokens": ctx,
            "needles": [
                {"key": k, "value": v, "depth": d}
                for d, (k, v) in zip(NEEDLE_DEPTHS, NEEDLES)
            ],
            "question": QUESTION,
            "expected_answer_substrings": EXPECTED_SUBSTRINGS,
            "min_recovered": MIN_RECOVERED,
            "filler_text": filler_text,
        }
        out_path = out_dir / f"niah_multi_{ctx // 1024}k.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
        approx_chars = len(filler_text)
        print(f"wrote {out_path.name}: {approx_chars} chars "
              f"(~{approx_chars / CHARS_PER_TOKEN:.0f} tokens, "
              f"3 needles at depths {NEEDLE_DEPTHS})")


if __name__ == "__main__":
    sys.exit(main() or 0)
