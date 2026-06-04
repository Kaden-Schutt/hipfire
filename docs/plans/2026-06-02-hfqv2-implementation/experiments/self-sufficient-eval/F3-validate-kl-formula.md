<!-- Copyright (c) 2026 Kaden Schutt -->
# F3 — Final gate: is hipfire's 0.0898 vs GGUF's 0.0710 an apples-to-apples KL?

Branch: `foundation/native-bf16-fp32-eval` (commit 33831d4a). Box: mi300
(gfx942 / MI300X VF). Date: 2026-06-04. Scope: code-read + log/source
cross-check only — NO new GPU run, NO GGUF wired into hipfire.

This closes the last of three F3 gates (tokenizer parity 100% PASSED;
bf16-PPL alignment +0.14% PASSED). Gate here: are the two 4-bit KLD numbers
computed with the **identical KL formula** (direction / reduction / log base),
so the 0.0898-vs-0.0710 verdict is a real quality gap, not a definition
mismatch?

---

## 1. hipfire `eval_hipfire_fullvocab` — the 0.0898 source

File: `crates/hipfire-runtime/examples/eval_hipfire_fullvocab.rs` (lines
146–193). Per-position computation:

```rust
// fp64 log-softmax over all vocab_size logits (max-subtract, sum-exp, log_z)
let lp_o = log_probs_f64(&oracle_logits);   // F32 oracle log-probs
let lp_c = log_probs_f64(&cand_logits);     // quant candidate log-probs
let mut kl = 0.0f64;
for v in 0..config.vocab_size {
    let p = lp_o[v].exp();                   // P_oracle(v)
    if p > 0.0 { kl += p * (lp_o[v] - lp_c[v]); }   // P_o·(logP_o − logP_c)
}
kld_sum += kl.max(0.0);
// ... mean_kld = kld_sum / scored_done
```

| axis | value |
|---|---|
| **Direction** | **KL(P_oracle ‖ P_cand)** — oracle (F32) is the reference P, quant is Q. |
| **Reduction** | **mean over scored tokens** (`kld_sum / scored_done`). |
| **Log base** | **natural log** (fp64 `.ln()`/`.exp()`) → **nats**. |
| **Vocab** | **full** — sums over all `vocab_size` (248,320) entries. |
| **Clamp/eps** | skip terms where `P_oracle(v) == 0.0`; per-position `kl.max(0.0)` floor. No epsilon added to log-probs (full log-softmax, no true zeros). |
| **Precision** | fp64 accumulate; logits downloaded as f32 then promoted. |

Number (from `/tmp/f3-fullvocab-awq.log`, exact): **FULL-VOCAB KLD =
0.089757** (32,640 scored tok, cand PPL 9.6347 — matches eval_hipfire
standalone 9.6442 to 0.1%, so the candidate forward is correct).

---

## 2. llama-perplexity `--kl-divergence` — the 0.0710 source

File (mi300): `/tmp/llama.cpp/tools/perplexity/perplexity.cpp`, `log_softmax`
overload (lines 191–249), reported at lines 1948–1949.

```cpp
// full-vocab log-softmax of the CANDIDATE logits (max_logit + log_sum_exp = log_z)
max_logit += log_sum_exp;                  // = log_z of candidate
double sum = 0;
for (int i = 0; i < n_vocab; ++i) {
    const float p_log_base = scale*base_log_prob[i] + min_log_prob;   // log P_base(i)
    if (p_log_base > -16.f) {                                         // tail clamp
        const float p_base = expf(p_log_base);                       // P_base(i)
        sum += p_base * (p_log_base - logits[i] + max_logit);        // P_base·(logP_base − logP_cand)
    }
}
kld.sum_kld += sum; ++kld.count;
// ... "Mean KLD" = mean_and_uncertainty(sum_kld, sum_kld2, count) = sum_kld / count
```

`-logits[i] + max_logit` (with `max_logit += log_sum_exp` already applied) =
`-(logits[i] − log_z_cand)` = `−log P_cand(i)`. So the per-token term is
`Σ_i P_base(i)·(log P_base(i) − log P_cand(i))`.

| axis | value |
|---|---|
| **Direction** | **KL(P_base ‖ P_cand)** — base (bf16, supplied via `--kl-divergence-base`) is the reference P, quant is Q. |
| **Reduction** | **mean over scored tokens** (line 1948, `sum_kld / count`, printed "Mean KLD"). |
| **Log base** | **natural log** (`log()`/`expf()`) → **nats**. |
| **Vocab** | **full** — loops all `n_vocab` entries. |
| **Clamp/eps** | skip terms where `log P_base(i) ≤ −16` (P_base ≲ 1.1e-7) — a negligible tail truncation; base log-probs stored as quantized uint16 (`scale*stored + min_log_prob`). |
| **Precision** | fp64 accumulate (`double sum`), f32 logits. |

Number (Step 4 of F3-matched-comparison): GGUF Q4_K_S = **0.070983** (full-vocab,
128-chunk span, vs the same llama-bf16 base).

---

## 3. Side-by-side — DO THEY MATCH?  → YES

| axis | hipfire fullvocab | llama `--kl-divergence` | match? |
|---|---|---|---|
| KL direction | KL(P_ref ‖ P_quant), ref = F32 oracle | KL(P_ref ‖ P_quant), ref = bf16 base | **YES** (both reference‖quant, the forward KL) |
| reduction | mean over scored tok | mean over scored tok | **YES** |
| log base | natural (nats) | natural (nats) | **YES** |
| vocab coverage | full (248,320) | full (n_vocab) | **YES** |
| accumulation precision | fp64 | fp64 | **YES** |
| tail clamp | skip P_ref==0 | skip log P_ref ≤ −16 (P ≲ 1e-7) | trivially different, negligible |

All three load-bearing axes (direction, reduction, log base) are **identical**.
The only differences are (a) the reference distribution source — F32 oracle for
hipfire vs llama-bf16 base for GGUF (this is the *matched-harness* design; F2
proved the two oracles agree to ~0.0008 nats, and Step-2 confirmed their PPLs
agree to +0.14% on this exact span), and (b) a negligible tail-probability
truncation threshold. **No definition mismatch.**

Because the definitions match, the empirical forward-vs-reverse-KL recompute
(only required by the task if the directions differed) is **not needed** — both
sides already compute the same forward KL(reference ‖ quant). For the record,
at KLD ~0.07–0.09 nats forward and reverse KL agree to a few %, far inside the
21% verdict gap; but that contingency does not arise here.

---

## 4. Q8-anchor sanity cross-check — both KL scales are compatible

A wrong-by-a-constant-factor formula (e.g. log2 vs ln = 1.44×, or a missing
1/2, or a 5×-off normalization) would throw the near-lossless Q8 anchors off
their expected per-engine values. They land where they should:

| anchor | this harness | expected / historical | sane? |
|---|---|---|---|
| hipfire Q8 full-vocab | **0.029810** (`/tmp/f3-fullvocab-q8.log`, cand PPL 9.3212 ≈ oracle 9.3198) | hipfire Q8 quantizes DeltaNet recurrent state + per-tensor granularity; > top-256 MQ8-G256 ~0.0186 but same order; near-lossless | YES |
| GGUF Q8_0 full-vocab | **0.006168** | llama Q8_0 is canonically near-lossless ~0.004–0.006 nats | YES |

Both Q8 numbers are O(1e-2)/O(1e-3) nats, i.e. near-lossless, consistent with
each engine's known Q8 behavior. Neither is off by a 1.44× (log-base) or
integer factor. The KLD chunk-traces in both logs converge smoothly
(AWQ 0.090, Q8 0.0298 — no per-position blow-up), and candidate PPLs sit right
at the F32 oracle. The two KL scales are mutually compatible.

(Direction corroboration: on the same axis GGUF Q8_0 0.0062 < hipfire Q8
0.0298, the SAME ordering as the 4-bit row GGUF 0.071 < hipfire 0.090 — the
Q8 anchor agrees with the 4-bit verdict direction, it is not an outlier.)

---

## 5. VERDICT

**The KL formulas MATCH** — same direction (forward KL, reference ‖ quant),
same reduction (mean over scored tokens), same log base (nats), both full-vocab,
both fp64-accumulated. The only differences are the (F2-equivalent) reference
oracle and a negligible 1e-7 tail clamp.

**The GGUF-beats-hipfire-at-4-bit verdict STANDS on an identical/compatible KL
definition: GGUF Q4_K_S 0.070983 < hipfire AWQ-GPTQ 0.089757 nats** (GGUF ~21%
lower KLD, at higher bpw 4.76 vs ~4.6). The Q8 anchors (hipfire 0.0298 / GGUF
0.0062) land at plausible per-engine near-lossless values and share the same
ordering, independently confirming the two KL scales are on the same footing.
The 0.0898-vs-0.0710 gap is a real quality difference, not a formula artifact.
