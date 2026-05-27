# DFlash prose-τ research note (2026-05-15)

Research note dropped during PR review session. Captures what's relevant to
the prose-τ collapse problem (τ ≈ 10 on code, τ ≈ 3-4 on prose) for future
work on this branch. **Not implementation work** — just findings + source
links so the next pass can start from a real baseline instead of
re-googling.

## Bench data motivating the research

From the 2026-05-13 ChatML/ClosedThink falsification on hipx gfx1151,
27B-3.5-DFlash:

| Path                                        | τ    | tok/s |
|---------------------------------------------|------|-------|
| Bench `--no-chatml` raw                     | 10.0 | 82    |
| Daemon production = ChatML + `prefix=Plain` | 6.83 | 58    |
| Daemon + `prefix=ClosedThink`               | 3.54 | 34    |

Same drafter, same target, same binary md5. ~30% τ collapse from raw →
ChatML wrap, ~50% from ChatML → ClosedThink. CLAUDE.md's
prompt-structure τ rule already documents the principle ("one newline
character can swing τ by 17%"). Open question: how much of the gap is
"architecturally inherent" vs "drafter distribution mismatch we can fix
with retraining"?

## The reframe that matters

**Acceptance Dynamics Across Cognitive Domains** — arXiv:2604.14682
(Apr 2026, 99,768 step-level records). Headline empirical finding: with
a *properly aligned* drafter, **chat has higher acceptance than code**
because RLHF lexical patterns are predictable. Hipfire's inverse
pattern (τ≈10 code, τ≈3-4 prose) is therefore strong diagnostic
evidence the drafter is **misaligned to the production distribution**,
not that prose is inherently hard.

Source: https://arxiv.org/abs/2604.14682

This is the single most important paper to read before any retraining
work — it sets the expected τ-by-domain shape under a correctly-aligned
drafter, so we have a target to hit.

## The mechanism characterization

**Attention Drift in Speculative Decoding** — arXiv:2605.09992
(May 2026). Drafter attention "drifts" off the target's hidden states
and onto its own previously-emitted tokens as a draft chain extends.
Up to 2× regression measured under **template perturbation**.

Source: https://arxiv.org/html/2605.09992v1

ChatML wrapping IS a template perturbation relative to whatever raw
distribution our DeltaNet drafter was trained on. This paper is the
published characterization of the failure mode we observe in the bench
numbers above — the 82 → 58 → 34 ladder reads like the regression
curves they report.

## The fix recipes (training-side)

Both predate the 2026 papers above:

- **DistillSpec** — arXiv:2310.08461 (Zhou et al., Google; ICLR 2024).
  On-policy distillation + task-tailored divergence. Reports 10–45%
  speedup over standard spec decode; combined with target distillation,
  6–10× over standard AR.

- **Direct Alignment of Draft Model for Speculative Decoding with
  Chat-Fine-tuned LLMs** — arXiv:2403.00858 (Mar 2024). Exactly our
  case. Method: pretrain drafter on base text → generate distillation
  set from chat-aligned target → finetune with TVD++ loss. Reports
  2.3× block efficiency / 2.4× AR speedup with a 115M drafter on
  Llama-2 Chat 7B.

Sources:
- https://arxiv.org/abs/2310.08461
- https://arxiv.org/abs/2403.00858

**Quantitative expectation if the alignment hypothesis is right:** the
Direct Alignment paper's 2.3× block-efficiency improvement on a
base-text-pretrained → chat-distilled drafter maps to τ on hipfire's
prose path going from ~3.5 back up toward ~6-7. Not all the way to 10
(parameter gap between 9B drafter and 27B target sets a hard ceiling),
but enough to close most of the bench-vs-production gap.

## Existing infrastructure that helps

Branch `feat/mtp` already has the scaffolding for trunk-argmax
distillation:

- `scripts/sample_prompts.py` — corpus generation
- `scripts/run_distill_parallel.sh` — hiptrx parallel trunk forward
- `scripts/aggregate_argmax.py` — collect target argmax targets

Built originally for MTP head training, the same pipeline produces
exactly what Direct Alignment / DistillSpec need as input — target's
argmax (or full distribution) at every position for a fixed corpus of
ChatML-wrapped sessions.

Adapting to standalone-drafter distillation needs:
1. Replace MTP head architecture with decoder model architecture
   (use existing 0.8B / 9B DeltaNet drafter shape)
2. Adjust loss target from multi-token-ahead to next-token-argmax
   (or KLD against full softmax for higher-fidelity training)
3. Add a convert-PyTorch-state-dict-to-.mq4/.hfq step at the end so
   the trained drafter drops into the existing DFlash dispatch

Estimated wall-clock on hiptrx (4× R9700): **~1 week focused work**.
Target logits collection (6-24 h) + drafter SFT (0.5-3 d) + quantize +
smoke-test (0.5-1 d) + buffer.

## Two small composing runtime upgrades (no retrain)

If runtime tweaks are wanted before committing to a retrain:

- **PLD+** — arXiv:2412.01447 (Dec 2024, NAACL 2025). Re-ranks PLD
  candidates using **early-layer hidden states (layers 9-13)**.
  Tuning-free. Strictly better than vanilla PLD on input-guided
  tasks. Composes with our existing PLD path as a pure addition.

- **Adaptive γ** — PEARL (ICLR 2025) and SpecKV (arXiv:2605.02888,
  May 2026, +56% over fixed γ=4 from draft-entropy signal at 0.34 ms
  decision overhead). Our DDTree budget is currently fixed. Runtime
  knob, no retrain. Composes with DDTree by varying B per-step.

Sources:
- https://arxiv.org/abs/2412.01447
- https://arxiv.org/abs/2605.02888

Lift estimates from published numbers: PLD+ ~+5% on PLD-eligible
prompts; adaptive γ ~+5-10% on mixed traffic. Not the main lever, but
cheap composing wins.

## What's NOT relevant

Filtered out of the "actually relevant" list given hipfire already
integrates DFlash + DDTree + CASK + TriAttn + PLD from their respective
papers:

- **EAGLE-3 / EAGLE-2** — different drafting paradigm than DFlash
  (token-level small drafter, not block-diffusion). Switching means
  abandoning DFlash, not improving it.
- **DDTree** — already integrated.
- **Sequoia, REST, LayerSkip, Lookahead Decoding** — orthogonal
  techniques, none target alignment.
- **VSD / LK Losses / Saguaro / Mirror** — early 2026 spec-decode
  papers, but address symptoms (acceptance objective, parallelism)
  rather than the alignment mismatch.

## Concrete next-step ladder if/when this gets picked up

1. **Validate the alignment hypothesis empirically.** Measure τ on
   identical-content prompts under three wrappings on the same
   drafter+target+binary: raw text, ChatML wrap, ChatML + closed-think
   prefix. If the τ collapse tracks template distance (as the bench
   numbers suggest), the hypothesis is confirmed.

2. **Read Acceptance Dynamics (2604.14682) for the per-domain τ
   profile** under correctly-aligned drafters. Use that as the target
   shape, not "match code prompts."

3. **Adapt feat/mtp distill pipeline to standalone-drafter
   distillation** per Direct Alignment recipe. Corpus = real daemon
   session logs (ChatML-wrapped), distillation = target argmax-or-KLD,
   loss = TVD++.

4. **Quantize trained drafter back to .mq4 / DeltaNet arch=20.**
   Validate via coherence-gate + canonical bench. Expected: prose τ
   recovers from ~3.5 to ~6-7. Code τ probably unchanged (was already
   near ceiling).

5. **Optional composing wins:** PLD+ early-layer reranking + adaptive
   γ from draft entropy. Both are runtime-only, do not require the
   retrain.

## Sources

- DFlash core: https://arxiv.org/abs/2602.06036 (z-lab, Feb 2026)
- DDTree (already integrated): https://arxiv.org/abs/2604.12989
- Acceptance Dynamics Across Cognitive Domains: https://arxiv.org/abs/2604.14682
- Attention Drift: https://arxiv.org/abs/2605.09992
- DistillSpec: https://arxiv.org/abs/2310.08461
- Direct Alignment of Draft Model: https://arxiv.org/abs/2403.00858
- PLD+: https://arxiv.org/abs/2412.01447
- SpecKV: https://arxiv.org/abs/2605.02888
- DeltaNet: https://arxiv.org/abs/2406.06484
- Component-Aware Self-Speculative (Qwen3.5 hybrid): https://arxiv.org/abs/2605.01106

## Empirical validation on 7900 XTX (2026-05-15, late afternoon)

Bench run on k9lin (Sapphire Nitro+ 7900 XTX, gfx1100) directly testing
the hypothesis above. Same `qwen3.5-27b.mq4` target +
`qwen35-27b-dflash.mq4` drafter, max=120, kv-mode=asym3.

### Code prompt (`benchmarks/prompts/lru_cache_pep8_strict.txt`, md5 df5dedc)

| Config | tok/s | τ |
|---|---|---|
| DFlash `--no-chatml` (bench peak) | 155-166 | 7.79-8.46 |
| DFlash `--chatml` (production) | 50-99 (median ~75) | 2.6-3.7 |
| DFlash `--chatml --ddtree-batched` | 42-48 | 4.0-4.6 |
| AR baseline `--chatml` | 44.83 | — |

### Prose prompt (200-word essay request, md5 6fa90245a4d9b03f)

| Config | tok/s | τ |
|---|---|---|
| DFlash `--no-chatml` | 41.74 | **1.07** |
| DFlash `--chatml` (production) | 41.99 | **1.09** |
| DFlash `--chatml --ddtree-batched` | 25.10 | 1.86 |
| **AR baseline `--chatml`** | **45.37** | — |

### What the data says vs the hypothesis

1. **Hypothesis confirmed.** On code prompts the drafter operates near
   its training distribution (τ ≈ 8). On prose the drafter is at floor
   (τ ≈ 1.07) — accepts barely one token per cycle, basically AR with
   drafter-overhead tax.

2. **Stronger than expected.** I'd predicted "drafter performs poorly
   on prose"; the bench shows DFlash is actually **slightly slower than
   AR on prose** (42 vs 45 tok/s, -7%). This is not "marginal win
   reduced"; it's a net regression.

3. **ChatML wrapping is a code-prompt-specific failure mode.** On
   code, ChatML drops τ from 8 → 3 (-63%). On prose, ChatML changes
   τ from 1.07 → 1.09 (zero difference). The drafter on prose was
   already producing near-random argmax matches; ChatML noise on top
   doesn't matter.

4. **DDTree-batched recovers τ but kernel cost dominates on
   gfx1100.** Tree expansion lifts prose τ from 1.07 → 1.86 (+74%
   relative) but kernel slowdown drops tok/s to 25 (-40% vs default).
   Net: still worse than AR. (May behave differently on gfx1201 /
   gfx1151 — not tested today.)

### Implications for the retrain plan

- **Drafter retrain is no longer a "potentially worth it" optimization
  — it is required for DFlash to be net-positive on chat use.**
  Without retrain, the daemon is paying drafter overhead for a slight
  regression on any prose request.

- **The Direct Alignment 2403.00858 recipe's 2.3× block-efficiency
  improvement, applied to a baseline of τ=1.07, would target τ ≈ 2.5
  on prose.** That converts a -7% regression vs AR to roughly a +30%
  speedup — still well below the code-prompt regime, but at least
  net-positive.

- **The drafter we ship today is well-tuned for code prompts.** Don't
  throw out the existing drafter weights — distill from them as the
  initialization for the retrain, not from base text.

### Open questions raised by the bench

- **Does the prose τ collapse hold on gfx1201 / gfx1151?** Different
  arches may have different kernel cost/benefit ratios for DDTree.
- **Where's the cutover from "code regime" to "prose regime"?** Tool
  calls, structured JSON output, multi-turn chat history — these sit
  between pure code and pure prose. A workload-cutover bench across
  these shapes would tell us where the retrain effort earns out most.
- **Online τ rolling-avg as cheap mitigation?** Skip DFlash dispatch
  when last-N-cycle τ falls below ~2.5. PEARL-style adaptive but
  applied at the dispatch level, not within DFlash. Runtime-only
  fix, no retrain required.
