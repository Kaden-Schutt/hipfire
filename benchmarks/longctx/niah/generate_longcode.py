#!/usr/bin/env python3
"""Generate a long-code retrieval fixture for PFlash Phase 5.

Per PRD §6 Phase 5: "Long code retrieval prompt: committed under
benchmarks/prompts/ with md5 recorded." This fixture inlines a real
hipfire source file (pflash.rs, truncated to ~13K tokens) as the
filler and asks for a specific identifier embedded mid-file. The
JSONL format matches niah_<N>k.jsonl so pflash_niah_bench can run it
unchanged with --pretok / --write-pretok.

The truncation to TRUNCATE_CHARS (~45K chars) is intentional: the
0.8B drafter goes NaN at very long context (escalated as
ScoringDegenerate in MANUAL_REVIEW.md). 13K source tokens stays
well below the empirical 16K-17K boundary where NaN starts, so the
fixture exercises PFlash on real code without hitting the unrelated
drafter bug.

The "needle" is the value of the TOKENIZER_COMPAT_PROBE constant. It
appears once, ~16% into the file (line 268 of 1634, byte ~11K), in
normal source-code surroundings -- exactly the case where compression
risks dropping it. The expected answer substring is "0xCAFEf00d", a
magic constant unique enough to require actual retrieval (not generic
LLM knowledge).

Output: benchmarks/prompts/longcode_pflash.jsonl
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO / "crates" / "engine" / "src" / "pflash.rs"
OUT = REPO / "benchmarks" / "prompts" / "longcode_pflash.jsonl"

# Truncate to keep token count below the drafter NaN boundary (see
# MANUAL_REVIEW.md "PFlash score kernel produces NaN at ~21K source tokens").
# 45K chars * 1/3.5 = ~13K tokens after chatml wrap, well below ~16K threshold.
TRUNCATE_CHARS = 45000

QUESTION = (
    "Read the source code above and answer this question precisely. "
    "What is the exact byte string assigned to the TOKENIZER_COMPAT_PROBE "
    "constant in pflash.rs? Quote it verbatim."
)
EXPECTED = "0xCAFEf00d"


def main():
    full_code = SOURCE.read_text(encoding="utf-8")
    code = full_code[:TRUNCATE_CHARS]
    if EXPECTED not in code:
        print(f"FAIL: expected needle {EXPECTED!r} not in first {TRUNCATE_CHARS} chars of {SOURCE}; "
              f"adjust TRUNCATE_CHARS or move the needle", file=sys.stderr)
        return 2
    if code.count(EXPECTED) != 1:
        print(f"FAIL: needle {EXPECTED!r} appears {code.count(EXPECTED)}x in truncated code; "
              f"need exactly 1 for a clean retrieval test", file=sys.stderr)
        return 2

    record = {
        "context_tokens": 13000,
        "needle": EXPECTED,
        "question": QUESTION,
        "expected_answer_substring": EXPECTED,
        "filler_text": code,
        "source_file": "crates/engine/src/pflash.rs",
        "source_truncated_to_chars": TRUNCATE_CHARS,
        "source_md5_full": hashlib.md5(full_code.encode("utf-8")).hexdigest(),
        "source_md5_truncated": hashlib.md5(code.encode("utf-8")).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
    fixture_md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.name}: {len(code)} chars (~{len(code) // 4} tokens), "
          f"truncated_md5={record['source_md5_truncated'][:12]}, "
          f"fixture_md5={fixture_md5[:12]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
