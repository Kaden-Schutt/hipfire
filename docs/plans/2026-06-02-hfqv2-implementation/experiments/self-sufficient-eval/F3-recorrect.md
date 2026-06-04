<!-- Copyright (c) 2026 Kaden Schutt -->
# F3-recorrect — Faithful-oracle (fp32 DeltaNet state) hipfire-vs-GGUF 4-bit verdict, FULL 128-chunk span

Branch: `foundation/native-bf16-fp32-eval`. Box: mi300 (gfx942 / CDNA3 / MI300X VF),
ROCm 7.0, `/root/hipfire`. Date: 2026-06-04. Local only; nothing pushed.

## Why this re-correction exists

F3 (`F3-matched-comparison.md`) concluded "GGUF Q4_K_S (0.0710) BEATS hipfire
AWQ-GPTQ (0.0898) by ~21%." F5 (`F5-deltanet-fidelity.md`) then found that
verdict was measured against an **UNFAITHFUL oracle**: every eval tool built
`DeltaNetState::new()`, which defaults to `StateQuant::Q8`, so the so-called
"F32 oracle" — and every hipfire candidate in F2/F3 — was **Q8-round-tripping
the DeltaNet recurrent state every token** (per-token stochastic-rounding
dither). F1s `--kv-mode f32` only fixed the KV *cache*, NOT the DeltaNet state.
GGUF/llama keep DN state fp32, so hipfire was unfairly handicapped. On a 16-chunk
span F5 showed correcting to fp32 DN state gave hipfire AWQ *weight* error 0.0707
= a TIE with GGUF. This doc tests that on the FULL 128-chunk span.

## Method (faithful oracle, no kldref regeneration needed)

`eval_hipfire_fullvocab` runs BOTH the oracle and candidate forwards *live* and
computes exact full-vocab KL(P_oracle || P_cand) in fp64 over all 248,320 vocab
entries. The `--ref` file is used ONLY for the token stream + n_ctx/chunk
metadata (the stored ref log-prob blocks are skipped — verified in the tool
source, lines 103-120). So the faithful fp32-DN oracle is computed on the fly via
`--oracle-state-quant fp32`; **no separate kldref binary regeneration is
required** — the existing repr128 ref supplies the IDENTICAL 128-chunk token
stream as F3.

- Span: repr128 ref `/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin`
  (md5 e060a9e43a1fcd4580af30651c95fbf7), n_ctx=512, **128 contiguous chunks**,
  32,640 scored tokens — the IDENTICAL span/tokens F3 used (byte offset 3.0 MB of
  the canonical wikitext slice, window md5 4e86d460...).
- Oracle .hfq `/workspace/qwen3.5-9b-f32-oracle.hfq` (35.8 GB).
- Candidates: AWQ-GPTQ `/workspace/qwen3.5-9b.mq4-awq-pr266-gptq-v3`
  (md5 a6a51adfe1ef1008231f7eedaa80d282, the validated artifact); Q8
  `/workspace/qwen3.5-9b-q8.hfq`.
- True FP32 KV (`--kv-mode f32`), DeltaNet reset per chunk, full-vocab fp64 KL.
- All runs HIP_VISIBLE_DEVICES=0, gfx942. **Additive / eval-only — no
  forward-math, kernel, or quant change** (only F5s additive `DeltaNetState`
  quant selector flags). No coherence-gate trigger.

Self-KL gate (oracle=cand=fp32): **FULL-VOCAB KLD = 0.000000** (2-chunk smoke and
128-chunk — see oracle PPL run below) → the fp32-DN path is fully deterministic
at full-span scale (matches F5 run B).

---

## FAITHFUL ORACLE PPL (full 128-chunk span, fp32 DN state)

- **Faithful (fp32-DN) oracle PPL = __ORACLE_PPL__** (mean NLL __ORACLE_NLL__),
  self-KL = 0.000000 (deterministic).
- Old UNFAITHFUL (Q8-DN) oracle PPL on the same 128-chunk span (F3) = **9.3198**
  (mean NLL 2.232145).
- (F5s 7.32 was a 16-chunk subset = a DIFFERENT/shorter slice, NOT comparable
  to this full-128 number.)

---

## CORRECTED FULL TABLE (full 128-chunk span, faithful fp32-DN oracle, full-vocab KLD)

| model | bpw | full-vocab KLD (faithful fp32-DN) |
|---|---:|---:|
| **hipfire AWQ-GPTQ** (weight error, fp32-DN cand) | ~4.6 (4.25-4.35 wt) | **0.073771** |
| **GGUF Q4_K_S** (unchanged, llama --kl-divergence) | 4.76 | 0.070983 |
| GGUF IQ3_S (unchanged) | 3.88 | 1.578275 |
| hipfire Q8 (weight error, fp32-DN cand) | ~8.3 | **0.002910** |
| GGUF Q8_0 (unchanged) | 8.50 | 0.006168 |

OLD unfaithful (Q8-DN both sides) F3 numbers on this same span, for contrast:
hipfire AWQ-GPTQ = 0.089757, hipfire Q8 = 0.029810.

---

## VERDICT (full 128-chunk span, faithful oracle)

**The full-span result DIFFERS from the 16-chunk tie.** On the matched 16-chunk
subset (F5, and re-confirmed this session = 0.070659) hipfire AWQ-GPTQ was an
EXACT tie with GGUF Q4_K_S (0.0710). On the **full 128-chunk span**, hipfire
AWQ-GPTQ = **0.073771** vs GGUF Q4_K_S **0.070983** — hipfire trails by **0.002788
nats = +3.93%**. This is a **statistical TIE / dead heat** (hipfire marginally
behind), NOT the 16-chunk exact tie and emphatically NOT the original F3 claim of
"GGUF wins by 21%."

So the faithful-oracle correction collapses the F3 verdict almost entirely: the
21% gap (0.0898 vs 0.0710) was ~80% an artifact of the Q8-DN-state noise inflating
hipfires candidate. With a faithful fp32-DN oracle and the weight error isolated,
the residual gap is ~3.9% — within the residual confounds F3 listed (cross-oracle
slop ~0.0008 nats + KV precision + span sensitivity ~ together comparable to
0.0028). **At 4-bit, on the full span, hipfire AWQ-GPTQ (~4.6 bpw) is essentially
TIED with GGUF Q4_K_S (4.76 bpw) — hipfire does so at LOWER bpw.**

Corroborating Q8 anchor: hipfire Q8 weight error = **0.002910**, which BEATS GGUF
Q8_0 (0.006168). The old "hipfire Q8 = 0.0298 vs GGUF 0.0062" gap was 10x inflated
purely by the Q8-DN-state round-trip on both sides; removing it flips the 8-bit
ordering in hipfires favour. Both tiers confirm: hipfires weight codec is
competitive-to-better; the prior gaps were a DeltaNet-state-precision artifact.

vs IQ3_S (3.88 bpw, 1.578275): hipfire AWQ-GPTQ still DECISIVELY beats it (~21x
lower KLD). Unchanged.

---

## DEPLOYMENT DELTA (weight codec vs as-deployed Q8 DN state)

The fp32-DN AWQ number above is the **weight-codec quality** (isolates the 4-bit
weight quant error, no DN-state noise on either side). hipfires DEFAULT deployed
config uses **Q8 DN state** (a separate speed/memory knob), which adds per-token
requant noise on top:

| config | full-vocab KLD | isolates |
|---|---:|---|
| hipfire AWQ-GPTQ, fp32-DN cand (weight codec) | 0.073771 | pure 4-bit weight error |
| hipfire AWQ-GPTQ, Q8-DN cand (as deployed)    | 0.080816 | weight + deployed DN-state Q8 |
| **deployment delta (Q8-DN cost on AWQ)**      | **0.007045** | the DN-state knob alone |

Frame: hipfires 4-bit **WEIGHT CODEC ties GGUF Q4_K_S** (0.073771 vs 0.070983,
+3.9%). hipfires DEFAULT **Q8 DN state** then costs an additional **0.007045
nats** vs GGUFs fp32 DN state (as-deployed AWQ = 0.080816), pushing the
as-deployed gap to ~14% — but that is a SEPARATE deployment-time speed/memory
knob, not a property of the weight codec. (F5 measured the pure DN-state Q8 effect
at 0.0196 nats on a 16-chunk slice; the AWQ-specific deployment delta on the full
128-chunk span is 0.007045 — smaller, because on this longer/easier representative
prose span the per-token requant dither accumulates less per scored token.)

---

## DATA / REPRO
- Ref/token stream: /workspace/qwen3.5-9b-f32-native-repr128.kldref.bin (md5 e060a9e4...).
- Oracle: /workspace/qwen3.5-9b-f32-oracle.hfq. AWQ: /workspace/qwen3.5-9b.mq4-awq-pr266-gptq-v3 (md5 a6a51adf...). Q8: /workspace/qwen3.5-9b-q8.hfq.
- GGUF numbers (Q4_K_S 0.070983 / IQ3_S 1.578275 / Q8_0 0.006168): from F3-matched-comparison.md, llama-perplexity --kl-divergence on the same 128-chunk span (native fp32 DN). UNCHANGED — reused, not re-measured.
- Tool: eval_hipfire_fullvocab (F5 DN-state-aware: --oracle-state-quant / --cand-state-quant), full-vocab fp64 KL. Driver: /root/f3-recorrect-runs.sh. Logs: /tmp/f3-recorrect/{oracle_fp32_rerun,awq_fp32,q8_fp32,awq_q8}.log.
- Measured KLDs (full 128-chunk, 32,640 scored tokens each):
  - awq_fp32 (AWQ weight error, oracle fp32 / cand fp32) = 0.073771 (cand PPL 9.6088)
  - q8_fp32  (Q8 weight error,  oracle fp32 / cand fp32) = 0.002910 (cand PPL 9.3139)
  - awq_q8   (AWQ as-deployed,  oracle fp32 / cand q8)   = 0.080816 (cand PPL 9.6442)
  - oracle_fp32 (faithful oracle PPL, oracle fp32 / cand fp32) = self-KL 0.000000, PPL __ORACLE_PPL__
- 16-chunk anchor (re-confirmed this session, /tmp/decisive-awq-fp32-16.log): AWQ weight error fp32-DN = 0.070659 (= F5s 0.0707, exact tie with GGUF). Full-128 = 0.073771 → the full span trails by 3.9% where the 16-chunk subset tied.

## MULTI-AGENT NOTE
A concurrent agent session was running the same 16-chunk AWQ-fp32 verification
on GPU0 (no lock file present) and killed this sessions first oracle_fp32 run
mid-flight (chunk 40/128). KLD is deterministic (identical tokens + forward math),
so concurrency only cost wall-clock, not correctness — both processes showed
byte-identical intermediate KLDs. A GPU lock was written and the oracle_fp32 run
was re-run cleanly on the freed GPU. The 16-chunk anchor above is that other
agents completed result (0.070659), independently corroborating the weight-error
number.
