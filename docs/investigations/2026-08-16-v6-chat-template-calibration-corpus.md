# v6 chat-template calibration corpus for qwen3.8-27b

**Date:** 2026-08-16
**Status:** artifacts produced and verified; the A/B that decides whether they are
*better* is separate and not yet reported here.
**Host:** `mi300x` (gfx942), producer-only. **Nothing quantized was scored there** —
see [`2026-08-16-amendment-gfx942-collapse-is-arch-specific.md`](../perf-checkpoints/2026-08-16-amendment-gfx942-collapse-is-arch-specific.md).
**Source:** commit `814d8fd95`, worktree `/scratch/hipfire-v6-814d8fd`, clean.

This record exists because the artifacts live on a rented machine. If that box
goes away, these hashes and commands are what make the work reproducible.

---

## The defect being fixed

Every prior `.calib.hfq` in this campaign — five models, ~85 GB, roughly 12 h of
GPU time — derived from `/scratch/corpus/bartowski_v5.calib.txt`, which is plain
wiki-like prose. **Measured, not assumed:**

| corpus | `im_start` | `im_end` | `tool_call` | `think` |
|---|---|---|---|---|
| v5 (what we had been using) | **0** | **0** | **0** | **0** |
| v6 base render (137 convs + 54 prose) | **789** | **789** | **456** | **332** |

We were calibrating a chat model on prose. The reference recipe renders
calibration text through the target model's own chat template.

Two consequences, both addressed by the same artifact set: calibration/deployment
mismatch, and a **contaminated selector** — scoring on WikiText-2 while
calibrating on wiki-like prose means WT2 is not cleanly out-of-distribution.

## Source material and rendering contract

Source is gist `bartowski1182/e26453c0404e24eb317543ec5360f87a` (2026-08-14):
`v6_prose.txt` (54 pieces selected from `calibration_datav5.txt`) and
`v6_conversations.json` (137 tool-calling conversations from
`interstellarninja/hermes_reasoning_tool_use`, seed 42, explicit `kept_indices`).

Its four `format_notes` are each a silent-failure mode. **All four verified:**

1. `tools=` reached the model template.
2. **192 `function.arguments` values remained dicts** (Qwen templates require
   dicts; a string would break rendering).
3. `reasoning_content` and `role=tool` results rendered.
4. Special tokens were emitted **literally, not escaped** — escaping would
   reproduce the exact zero-`im_start` defect while still yielding a
   plausible-looking corpus.

**Fidelity result: the base render matches the published Qwen3.8-27B v6 corpus on
all four marker counts exactly** (789/789/456/332), with tool byte fraction
63.009% against the published ~63.3%. Bytes land 0.974% below ~1,258,850 and
model-token chunks 16 below ~583 — tokenizer/whitespace level, not structural.

## Artifacts

| role | path | bytes | tokens | sha256 (prefix) |
|---|---|---|---|---|
| full render | `/scratch/corpus/bartowski_v6_814d8fd.split.txt` | 1,344,981 | 314,360 | `36c07c0f…` |
| calibration slice | `…v6_814d8fd.calib.txt` | 1,118,239 | 262,346 | `512cca01…` |
| selector slice | `…v6_814d8fd.eval.txt` | 226,741 | 52,014 | `7dcbeb97…` |
| **v6 selector reference** | `/scratch/work/qwen3.8-27b.ref_v6sel-814d8fd.bin` | 50,675,552 | 24,552 scored | `b3d1fafc…` |
| **v6 calibration** | `/scratch/work/qwen3.8-27b.calibv6-814d8fd.hfq` | 31,481,139,200 | 262,144/tensor | `89a6479c…` |
| WT2 control | `…ref_wt2-control-814d8fd.bin` | — | 24,552 scored | `8c545178…` |
| manifest | `/scratch/work/qwen3.8-27b.v6-producer-814d8fd.manifest.json` | — | — | `dcdd3d85…` |

Calibration verified **from bytes**: 496/496 tensors, uniform 262,144
tokens/tensor, BF16 teacher. Reference verified from its **wire header**: magic
`HFKLDR\0\0`, version 1, `n_ctx` 2048, `n_chunks` 24, `top_k` 256,
`scored_tokens` 24,552, size matching the wire formula.

## The 16 extension conversations — provenance matters

The render needed 16 extension conversations, added in the gist's
`extension_greedy_order`: `6,10,24,34,37,40,74,82,98,100,103,110,120,124,129,143`.

**They were a token-budget accommodation, NOT a fidelity correction.** The base
render already matched the published band on every marker. Arithmetic: 290,751
base tokens − 49,152 reserved for the selector (24×2048) = 241,599, short of the
262,144/tensor budget; with extensions, 314,360 − 49,152 = 265,208, which clears
it. The gist states extensions are **not** det-validated, so the base 137+54
render remains the fidelity-validated artifact and is separately hashed
(`732145ab…`) so it stays citable on its own.

## Split disjointness — proven at two granularities

- conversation-id intersection: **0**
- base partition union: 137 = base source count: 137 (a complete partition)
- identical 512-token chunk-hash intersection: **0**
- identical 2048-token chunk-hash intersection: **0**

The selector is **100% held-out rendered conversations, 0 prose bytes**, using 24
of its 25 complete 2048-token chunks.

## Reference oracle, and a retracted gate

The v6 selector's BF16 teacher oracle is **NLL 2.555774 / PPL 12.8813**.

A "plausible single-digit PPL" gate had been imposed on it and **was wrong** — it
was calibrated on WikiText-2 prose and applied to a pure tool/chat distribution.
It is retracted here so it is not re-imposed. Three things establish the oracle is
real:

1. `ln(248320) = 12.42`, and NLL 2.5558 is a factor of ~5 **below** uniform. For
   contrast the two known failures on record are the gfx942 quantized collapse at
   NLL 14.4–14.8 (at or past uniform) and gemma4 at PPL 1392 against an expected
   ~6.
2. The selector is 0% prose, so scoring materially worse than WT2 prose is the
   **expected** direction. A tool-heavy selector matching a prose baseline would
   be the suspicious result.
3. A same-binary WT2 control reproduced the canonical reference **byte-identically**
   (`cmp`, sha256, md5 `8a21364051d844b97c122e2c895f56d8`, oracle NLL 1.830742 /
   PPL 6.2385), proving the BF16 teacher path is bit-deterministic and the
   `814d8fd95` build carries no regression.

**Labeling requirement.** `ref_v6sel` is a harder, further-OOD instrument than the
calibration mix it accompanies. Its PPL is for **ranking arms on the deployment
distribution**; it is not a general model-quality figure and must never be compared
against a WikiText-2 PPL as though the two measured the same thing. That asymmetry
is deliberate — selector on the distribution v6 targets, WT2 retained as the prose
tripwire — but only works if it is written down.

## Caveat that applies to every number here

Our KLD estimator is **top-k 256**; llama.cpp's `--kl-divergence` is **full-vocab**
(`perplexity.cpp::log_softmax` loops the whole vocabulary). These figures are
internally comparable with our other references and are **not** comparable to any
published table.

## What this record does NOT claim

That v6 calibration is *better*. That requires the 2×2 —
{v5-calibrated, v6-calibrated} × {WT2 tripwire, v6 selector} — quantized with the
byte-proven trunk recipe and scored on **gfx1201**, never gfx942. The result worth
watching is whether the **ranking differs between the tripwire and the selector**:
if it does, earlier prose-scored arm comparisons in this campaign were measuring
the wrong distribution.

## Reproduce

```bash
env -C /scratch/hipfire-v6-814d8fd \
  HIPFIRE_CALIB_BF16=1 HIPFIRE_NORMALIZE_PROMPT=0 \
  HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0 \
  ./target/release/examples/build_kld_ref_native \
    --model /scratch/work/qwen3.8-27b.teacher.bf16.hfq \
    --slice /scratch/corpus/bartowski_v6_814d8fd.eval.txt \
    --output /scratch/work/qwen3.8-27b.ref_v6sel-814d8fd.bin \
    --top-k 256 --n-ctx 2048 --tokenize-mode hipfire --max-chunks 24
```

`HIPFIRE_NORMALIZE_PROMPT=0` is deliberate: reference building must not mutate the
corpus it is scoring.
