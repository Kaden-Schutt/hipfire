#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""coherence_arm — the sampled-coherence arm of the v2 serve_harness certify.

Implements the three pieces the gap analysis (docs/autoresearch-gap-analysis-2026-07-07.json)
requires the coherence arm to have and the v1 design lacked:

  1. detect_attractor(token_ids)  — the DFlash tiered detector on TOKEN-IDS (gap #17: the
     detector must run on token-ids at the original calibration, not whitespace words).
  2. semantic validators          — per-genre correctness checks (gap #6: structural-only lets
     fluent-but-wrong through). number-match / sentence-count / numbered-list / py_compile /
     tool-call-json.
  3. mcnemar_worse(pairs)         — the PAIRED seed-set rate test (gap #1: a single seed is not
     variant-vs-baseline comparable at temp 1.0; run a seed-SET, same seeds for base+variant, and
     test whether the variant introduces MORE failures than the baseline). Because base and variant
     use the SAME seed set, the trials are PAIRED → McNemar's exact binomial test is the correct
     comparison (not a two-independent-proportion test).

Pure stdlib, no GPU, no scipy — unit-testable in the no-GPU CI.
"""
import ast, math, re
from collections import Counter

# ---------- 1. token-id attractor detector (DFlash calibration) ----------

def _uniq(toks):
    return len(set(toks)) / len(toks) if toks else 1.0

def _maxfreq(toks):
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)

def _gram3_density(toks):
    if len(toks) < 6:
        return 0.0
    grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    c = Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)

def detect_attractor(token_ids, min_len=24):
    """Tiered attractor detection on TOKEN-IDS. Returns (is_attractor: bool, reason: str).

    Tiers (the DFlash-gate calibration, on token-ids):
      first-128: unique-ratio < 0.15 OR max single-token freq > 0.50   (single-token loop early)
      last-128 : unique-ratio < 0.30 OR max single-token freq > 0.50   (block loop late)
      full 2nd-half: 3-gram repetition density > 0.50                  (structural loop)
    An empty stream is a failure. A SHORT stream (< min_len tokens) is NOT judged: the ratios are
    calibrated for 128-token windows, so a valid short answer ("42", "Paris", a one-line function)
    trivially trips maxfreq (a 1-token answer has maxfreq=1.0). Only outputs long enough to sustain a
    real loop are scored. (Found live: short battery answers false-positived without this floor.)
    """
    n = len(token_ids)
    if n == 0:
        return True, "empty"
    if n < min_len:
        return False, "short-ok"
    first, last, half = token_ids[:128], token_ids[-128:], token_ids[n // 2:]
    if first and (_uniq(first) < 0.15 or _maxfreq(first) > 0.50):
        return True, "first128"
    if last and (_uniq(last) < 0.30 or _maxfreq(last) > 0.50):
        return True, "last128"
    if _gram3_density(half) > 0.50:
        return True, "gram3"
    return False, "ok"

# ---------- 2. per-genre semantic validators ----------
# Each returns (ok: bool, detail: str). A validator only participates when the genre asks for it;
# the certify treats "variant fails a validator the baseline passed" as a COHERENCE_FAIL.

def validate_number_match(text, expected):
    """The reason prompt's final numeric answer must equal `expected` (allow int/float, commas)."""
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return False, "no number found"
    try:
        got = float(nums[-1])
        return (abs(got - float(expected)) < 1e-6, f"got={got} expected={expected}")
    except ValueError:
        return False, f"unparseable {nums[-1]!r}"

_SENT_END = re.compile(r"[.!?](?:\s|$)")

def validate_sentence_count(text, n):
    """Exactly `n` sentences (factual: 'in exactly three sentences')."""
    count = len(_SENT_END.findall(text.strip()))
    return (count == n, f"sentences={count} expected={n}")

def validate_numbered_list(text, n):
    """Exactly `n` numbered lines (instruct: 'exactly five ... numbered list')."""
    lines = [l for l in text.splitlines() if re.match(r"\s*\d+[.)]\s+\S", l)]
    return (len(lines) == n, f"numbered_lines={len(lines)} expected={n}")

def _extract_code(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text

def validate_python_compiles(text):
    """The code output must parse as Python (py_compile-equivalent via ast.parse)."""
    code = _extract_code(text)
    try:
        ast.parse(code)
        return True, "parses"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

def validate_tool_call_json(tool_calls):
    """Every tool-call `arguments` payload must be valid JSON. `tool_calls` = list of raw arg
    strings (or dicts with an 'arguments' field). Empty list = vacuously ok."""
    import json
    for tc in tool_calls or []:
        arg = tc.get("arguments") if isinstance(tc, dict) else tc
        if arg is None:
            continue
        try:
            json.loads(arg) if isinstance(arg, str) else None
        except (ValueError, TypeError) as e:
            return False, f"bad tool-call json: {e}"
    return True, "ok"

# Genre → the validators + their expected args. The certify passes each turn's genre here.
def run_validators(genre, text, tool_calls=None, expect=None):
    """Return (ok: bool, failures: list[str]). `expect` carries per-genre params
    (e.g. {'number': 210, 'sentences': 3, 'numbered': 5})."""
    expect = expect or {}
    checks = []
    if genre == "reason" and "number" in expect:
        checks.append(validate_number_match(text, expect["number"]))
    if genre == "factual" and "sentences" in expect:
        checks.append(validate_sentence_count(text, expect["sentences"]))
    if genre == "instruct" and "numbered" in expect:
        checks.append(validate_numbered_list(text, expect["numbered"]))
    if genre == "code":
        checks.append(validate_python_compiles(text))
    checks.append(validate_tool_call_json(tool_calls))
    failures = [d for ok, d in checks if not ok]
    return (len(failures) == 0, failures)

# ---------- 3. paired seed-set rate test (McNemar exact) ----------

def mcnemar_worse(pairs, alpha=0.05):
    """Paired binary rate test over a seed-SET run on BOTH base and variant with the SAME seeds.

    pairs: list of (base_fail: bool, var_fail: bool), one per (prompt, seed) trial.
    Tests H1: the variant introduces MORE failures than the baseline (one-sided).
    Uses McNemar's EXACT test on the discordant pairs:
        b = base-ok & variant-fail   (variant made it worse)
        c = base-fail & variant-ok   (variant made it better)
    Under H0 (variant no worse), b ~ Binomial(b+c, 0.5). One-sided p = P(X >= b).
    Returns (variant_worse: bool, p_value: float, b: int, c: int).
    """
    b = sum(1 for bf, vf in pairs if (not bf) and vf)
    c = sum(1 for bf, vf in pairs if bf and (not vf))
    nd = b + c
    if nd == 0:
        return (False, 1.0, b, c)
    p = sum(math.comb(nd, k) for k in range(b, nd + 1)) / (2 ** nd)
    return (p < alpha and b > c, p, b, c)
