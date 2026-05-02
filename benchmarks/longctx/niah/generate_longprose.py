#!/usr/bin/env python3
"""Generate a long-prose multi-doc retrieval fixture for PFlash Phase 5.

Per PRD §6 Phase 5: "Long prose / multi-doc prompt: committed under
benchmarks/prompts/ with md5 recorded."

The fixture concatenates three deterministic prose documents -- each
self-contained narrative on a different topic, each holding a unique
"established fact" -- separated by clear document boundaries. The
question asks for one specific fact from one specific document; the
target must locate the right document among distractors and then
retrieve the right fact within it. This is harder than NIAH because
the document-internal context primes the model toward the local
narrative, away from the buried fact.

Token target: ~13K (well below the 0.8B drafter NaN threshold ~16K
documented in MANUAL_REVIEW.md).

The three documents are:
  1. A monastery rule book (mid-document fact: "the bell rings
     forty-three times at dawn and twenty-one times at dusk").
  2. A trade-route ledger (fact: "the caravan from Volzkar carries
     emerald silk and pickled mountain pears").
  3. A starship maintenance manual (fact: "the auxiliary
     thrust-vector control gain is set to 0.847 millijoules per
     pulse").

Question: "How many times does the bell ring at dusk according to
the monastery rule book?"
Expected substring: "twenty-one"

Output: benchmarks/prompts/longprose_multidoc.jsonl
"""

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
OUT = REPO / "benchmarks" / "prompts" / "longprose_multidoc.jsonl"

SEED = 0x5407_FEE0
RNG = random.Random(SEED)

MONASTERY_FACT = (
    "By long-standing custom, the bell rings forty-three times at dawn and "
    "twenty-one times at dusk, and any monk who miscounts the dusk strokes "
    "must take the silent watch the following night."
)
TRADE_FACT = (
    "The caravan from Volzkar carries emerald silk and pickled mountain pears, "
    "and the spice merchants in Hala-on-the-River pay double the inland price "
    "for the pears in any month bearing the half-moon festival."
)
STARSHIP_FACT = (
    "The auxiliary thrust-vector control gain is set to 0.847 millijoules per "
    "pulse, and any deviation greater than seven percent from this value "
    "triggers the secondary disengagement protocol on Bridge Console 4."
)

WORDS = (
    "the abbey kept its lamp the brother walked the long aisle in the morning "
    "and the carved wood of the doors was polished daily by the youngest "
    "novice who also rang the small chimes whenever a stranger arrived "
    "the trade route through the eastern pass had been used since the time "
    "of the third sovereign and the carts were always counted three times "
    "before they were allowed past the toll gate where the ledger was kept "
    "the bridge console blinked twice when the navigator made an error and "
    "the maintenance log recorded each event with a timestamp in fleet "
    "standard time and a brief description of the corrective action taken "
    "the engine sang a low note when the coolant was within tolerance"
).split()


def make_filler(approx_tokens: int) -> str:
    target_chars = int(approx_tokens * 3.5)
    out = []
    chars = 0
    while chars < target_chars:
        w = WORDS[RNG.randrange(len(WORDS))]
        out.append(w)
        chars += len(w) + 1
    return " ".join(out)


def make_doc(title: str, fact: str, total_tokens: int) -> str:
    pre_tokens = int(total_tokens * 0.5)
    post_tokens = total_tokens - pre_tokens
    pre = make_filler(pre_tokens)
    post = make_filler(post_tokens)
    return f"=== {title} ===\n\n{pre}\n\n{fact}\n\n{post}"


QUESTION = (
    "Read the three documents above and answer this question precisely. "
    "How many times does the bell ring at dusk according to the monastery "
    "rule book? Answer in one short sentence."
)
EXPECTED = "twenty-one"


def main():
    RNG.seed(SEED)
    doc1 = make_doc("MONASTERY RULE BOOK", MONASTERY_FACT, 4200)
    RNG.seed(SEED + 1)
    doc2 = make_doc("VOLZKAR TRADE LEDGER", TRADE_FACT, 4200)
    RNG.seed(SEED + 2)
    doc3 = make_doc("STARSHIP MAINTENANCE MANUAL", STARSHIP_FACT, 4200)
    full = "\n\n\n".join([doc1, doc2, doc3])

    if full.count(EXPECTED) != 1:
        print(f"FAIL: needle {EXPECTED!r} appears {full.count(EXPECTED)}x; "
              f"need exactly 1 for a clean retrieval test", file=sys.stderr)
        return 2

    record = {
        "context_tokens": 12600,
        "needle": EXPECTED,
        "question": QUESTION,
        "expected_answer_substring": EXPECTED,
        "filler_text": full,
        "documents": ["MONASTERY RULE BOOK", "VOLZKAR TRADE LEDGER", "STARSHIP MAINTENANCE MANUAL"],
        "fact_document_index": 0,
        "filler_md5": hashlib.md5(full.encode("utf-8")).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
    fixture_md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.name}: {len(full)} chars (~{len(full) // 4} tokens), "
          f"3 docs, fixture_md5={fixture_md5[:12]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
