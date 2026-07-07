#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""No-GPU unit tests for coherence_arm (the v2 sampled-coherence arm)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import coherence_arm as ca


# ---- attractor detector (token-ids) ----

def test_empty_is_attractor():
    assert ca.detect_attractor([]) == (True, "empty")

def test_clean_stream_passes():
    toks = list(range(200))  # all-unique = maximally coherent
    is_a, reason = ca.detect_attractor(toks)
    assert not is_a and reason == "ok"

def test_short_answer_not_attractor():
    # a 1-word answer ("42") has maxfreq=1.0 but is NOT a loop — the min-length floor (found live)
    assert ca.detect_attractor(["42"]) == (False, "short-ok")
    assert ca.detect_attractor(["Paris"]) == (False, "short-ok")
    assert ca.detect_attractor([5] * 10) == (False, "short-ok")   # 10 repeats, below floor -> not judged
    assert ca.detect_attractor([5] * 40)[0]                       # 40 repeats, above floor -> attractor

def test_first128_single_token_loop():
    toks = [7] * 130  # one token repeated → uniq≈0.008 < 0.15, maxfreq≈1.0 > 0.50
    is_a, reason = ca.detect_attractor(toks)
    assert is_a and reason == "first128"

def test_last128_block_loop():
    # clean first half, then a low-unique tail
    toks = list(range(200)) + [1, 2] * 70  # last-128 window uniq ~ 2/128 < 0.30
    is_a, reason = ca.detect_attractor(toks)
    assert is_a and reason in ("last128", "gram3")

def test_gram3_structural_loop():
    # second half is a repeating 5-gram → high 3-gram density but not single-token
    head = list(range(300))
    loop = [1000, 1001, 1002, 1003, 1004] * 40
    is_a, reason = ca.detect_attractor(head + loop)
    assert is_a  # caught by last128 or gram3


# ---- semantic validators ----

def test_number_match():
    assert ca.validate_number_match("...therefore the total is 210 km.", 210)[0]
    assert not ca.validate_number_match("...the total is 205 km.", 210)[0]
    assert not ca.validate_number_match("no digits here", 210)[0]

def test_sentence_count():
    assert ca.validate_sentence_count("One. Two! Three?", 3)[0]
    assert not ca.validate_sentence_count("One. Two.", 3)[0]

def test_numbered_list():
    txt = "1. a\n2. b\n3. c\n4. d\n5. e"
    assert ca.validate_numbered_list(txt, 5)[0]
    assert not ca.validate_numbered_list("1. a\n2. b", 5)[0]

def test_python_compiles():
    assert ca.validate_python_compiles("```python\ndef f(a, b):\n    return a + b\n```")[0]
    assert not ca.validate_python_compiles("```python\ndef f(a, b)\n    return a + b\n```")[0]

def test_tool_call_json():
    assert ca.validate_tool_call_json(['{"x": 1}'])[0]
    assert ca.validate_tool_call_json([])[0]
    assert not ca.validate_tool_call_json(['{"x": '])[0]

def test_run_validators_routing():
    ok, fails = ca.run_validators("reason", "answer is 42", expect={"number": 42})
    assert ok and not fails
    ok, fails = ca.run_validators("factual", "One. Two.", expect={"sentences": 3})
    assert not ok and fails


# ---- paired seed-set rate test (McNemar exact) ----

def test_base_equals_base_not_worse():
    # identical outcomes → zero discordance → never "worse"
    pairs = [(False, False)] * 10 + [(True, True)] * 3
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert not worse and b == 0 and c == 0

def test_variant_strictly_worse_flags():
    # variant fails 8 prompts the baseline passed, baseline fails none the variant passed
    pairs = [(False, True)] * 8 + [(False, False)] * 8
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert worse and b == 8 and c == 0 and p < 0.05

def test_variant_better_not_flagged():
    # variant FIXES 8 that baseline failed → not "worse"
    pairs = [(True, False)] * 8 + [(False, False)] * 8
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert not worse and c == 8 and b == 0

def test_small_discordance_not_significant():
    # 1 vs 0 discordant is not enough evidence at alpha=0.05 (p = 0.5)
    pairs = [(False, True)] + [(False, False)] * 15
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert not worse and b == 1 and p > 0.05


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
