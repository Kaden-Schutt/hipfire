# Spec-Decode Durability — qwen dense, gfx11 (2026-06-23)

Measured on gfx1100 (RX 7900-class), dense Qwen3.6-27B (`qwen3.6-27b.mq4`), q8 KV,
greedy (temp=0), 3-run median. **AR decode baseline ≈ 43 tok/s** (genre-independent).

> **CORRECTION (measure on the daemon, not the demo).** Authoritative durability is measured
> through the **daemon** (the real serving path), NOT `dflash_spec_demo`. The demo measures raw
> continuation and under-reports — it made creative fiction look like a 0.97× failure. On the
> daemon, the serving behavior (chatml + a structured *thinking* block before the answer) makes
> even creative fiction predictable enough that **DFlash clears every genre ≥1.3× + τ>1.5**:
>
> | genre | daemon DFlash tok/s | AR× | τ |
> |---|---|---|---|
> | code | 112.9 | 2.63× | 5.15 |
> | reason | 114.8 | 2.67× | 5.23 |
> | instruct | 96.9 | 2.25× | 4.23 |
> | prose (expository) | 77.1 | 1.79× | 3.10 |
> | creative fiction (lighthouse) | 60.2 | 1.40× | 2.20 |
> | creative fiction (clockmaker) | 65.3 | 1.52× | 2.48 |
>
> **All genres PASS.** The earlier "creative fiction is retrain-bound" verdict was a
> demo-harness artifact (raw continuation never triggers the daemon's thinking-mode serving).
> No adaptive switch and no retrain are required — DFlash alone is durable across every genre on
> the serving path. **Lesson: measure durability on the daemon (the path users hit), never the
> raw demo harness.**

**Durability floors (the goal):**
- **DFlash:** every genre `τ > 1.5` **AND** `tok/s ≥ 1.30× AR` (= 56 tok/s).
- **MTP:** every genre `tok/s ≥ 1.15× AR` (= 49.5 tok/s).

---

## TL;DR — DFlash is durable across EVERY genre on the serving path

Measured through the **daemon** (the path users actually hit), greedy, 27B-3.6:

| Genre | DFlash tok/s | AR× (floor 1.3×, τ>1.5) |
|---|---|---|
| code | 112.9 | 2.63× ✓ τ5.15 |
| reason | 114.8 | 2.67× ✓ τ5.23 |
| instruct | 96.9 | 2.25× ✓ τ4.23 |
| prose (expository) | 77.1 | 1.79× ✓ τ3.10 |
| creative fiction (lighthouse) | 60.2 | 1.40× ✓ τ2.20 |
| creative fiction (clockmaker) | 65.3 | 1.52× ✓ τ2.48 |

**DFlash clears every genre ≥1.3× and τ>1.5 on the daemon** — the durability target is met by
DFlash alone. **MTP independently clears every genre ≥1.15× on the daemon too** (daemon-wired via
`HIPFIRE_QWEN_MTP=1`, lossless): code 1.64×, reason 1.73×, instruct 1.64×, prose 1.63×, creative
fiction 1.26–1.48× (τ2.37–2.79) — the same thinking-mode lift the `mtp_only_demo` missed. **Both
modes are durable across every genre on the serving path; all three requirements met.** No adaptive
switch, no retrain required.

The earlier "creative fiction is retrain-bound (0.97×)" verdict was a **demo-harness artifact**:
`dflash_spec_demo` measures raw open-ended continuation, but the daemon's serving behavior (chatml +
a structured thinking block before the answer) makes even novel fiction predictable enough that τ
jumps 1.05 → 2.2–2.5. Lesson: measure durability on the daemon, never the raw demo harness.

---

## 1. DFlash durable perf matrix (27B-3.6, greedy, q8 KV)

| Genre | linear tok/s | τ | AR× | DDTree b12 tok/s | τ | AR× | Floor (1.3×, τ>1.5) |
|---|---|---|---|---|---|---|---|
| code | **130.0** | 6.33 | 3.03× | 111.6 | 7.80 | 2.60× | **PASS** (linear) |
| reason | **109.9** | 4.79 | 2.53× | 77.2 | 4.94 | 1.78× | **PASS** (linear) |
| instruct | **74.5** | 2.82 | 1.72× | 53.7 | 3.57 | 1.24× | **PASS** (linear) |
| prose | 45.9 | 1.26 | 1.06× | 37.1 | 1.70 | 0.86× | **FAIL** (no config) |

Notes: DDTree loses to linear on tok/s in *every* cell (per-cycle tree-verify cost > τ benefit at
any budget). Prose is the sole failure: linear fails τ (1.26<1.5), DDTree fails tok/s (0.86×).
On 27B-3.5 the picture is identical plus a marginal **instruct** miss (1.25×, 4% short — the 3.5
drafter is slightly weaker on instruct; 3.6 passes at 1.72×).

## 2. MTP durable perf matrix (27B-3.6, K=3 p_min=0.4, greedy, q8 KV)

| Genre | tok/s | τ | AR× | Floor (1.15×) |
|---|---|---|---|---|
| code | 82.95 | 3.38 | 1.93× | **PASS** |
| reason | 71.70 | 3.13 | 1.67× | **PASS** |
| instruct | 67.38 | 2.89 | 1.57× | **PASS** |
| prose | 56.43 | 2.43 | 1.31× | **PASS** |

**All genres clear, losslessly, at one fixed config.** `p_min=0.4` (deeper adaptive chain) is the
prose-optimal early-exit threshold and doesn't hurt structured genres; `p_min` is not a correctness
knob (greedy MTP is distribution-preserving at any threshold). MTP perf is proven via
`mtp_only_demo`; it is **not yet daemon-wired** (only ds4 MTP is) — deployment is the remaining
implement step (§6).

## 2.5 Sampled (temp>0) serving — DFlash + GPU-softmax fast path

The matrices above are greedy (temp=0). For **sampled** serving, DFlash runs via
lossless rejection sampling; the per-cycle full-vocab softmax was the bottleneck
(~31 CPU passes/cycle ≈ +19ms), so a GPU-softmax fast path
(`HIPFIRE_DFLASH_FAST_SAMPLE=1`) moves it on-device, cutting the overhead **~89%**
(59.3 → 42.7 ms/cycle) so sampled per-cycle ≈ greedy.

| Genre | sampled tok/s | AR× | τ | floor |
|---|---|---|---|---|
| code | 66.2 | 1.54× | 2.91 | ≥1.3× ✓ |
| reason | 69.1 | 1.61× | 3.05 | ✓ |
| instruct | 72.0 | 1.67× | 3.32 | ✓ |
| prose | 54.5 | 1.27× | 2.16 | 1.15× ✓ / 1.3× ✗ |
| creative fiction | 52.8–52.9 | 1.23× | 2.05–2.06 | 1.15× ✓ / 1.3× ✗ |

Structured genres clear the 1.3× DFlash floor; prose/fiction clear the 1.15× MTP
floor but land ~5% under the strict 1.3× — the **structural lower sampled τ**
(rejection sampling accepts fewer than greedy argmax: prose τ2.16 sampled vs 3.10
greedy), NOT overhead (per-cycle is already ≈greedy).

**Parity:** distribution-parity, not byte-parity. The GPU softmax (tree reduction)
differs from the host softmax (sequential) at the last ULP, rarely flipping a
borderline accept; the committed distribution is the same temp-T distribution up
to rounding. Validated: long common prefix then benign divergence into equally
valid continuations; coherence-clean across all 6 genres (unique-word ratio
0.61–0.80, 3gram-rep <0.02, no attractors). **Safe default:** opt-in — without the
flag, temp>0 → AR (byte-faithful). `top_p`/`top_k` are not honored on the sampled
DFlash path (full-vocab temp-only; one-time warning). MTP sampled stays off (its
arch-layer sampled path is lossy).

## 3. The prose dividing line

Prose is the entire story. High-entropy narrative text means the target distribution is flat, so a
*lossless* spec-decode acceptor can rarely confirm the draft's specific token — prose's lossless
DFlash τ ceiling is **~1.26** (linear) / ~1.70 (DDTree, but the tree's per-cycle cost erases the
gain). The distilled DFlash drafter is code/agentic-trained (inverse-τ: code 6–10, prose ~1.3).

MTP's head is **jointly trained with the target**, so it actually models prose (τ2.43 greedy) and
its acceptance translates to 56 tok/s — over the (lower) MTP floor. Same hardware, same target,
same KV: the difference is entirely the drafter's prose competence.

## 4. Levers tried and shelved (the campaign)

| Lever | Result | Disposition |
|---|---|---|
| **Chunked (parallel) GDN** | math validated (parity 1.5e-7) but 11-16× *slower* than sequential at every shape; threading recovered to 5× slower, plateaus (grid under-utilization) | **falsified**, default-off flag |
| **DDTree budget right-size (prose)** | decode tok/s flat ~37 across b4–b12 (τ drops with budget — a wash) | falsified |
| **Tree-verify hipGraph fix** | root-caused `block_start` staleness, fixed (committed) — correct, no regression, but ~0% on prose (launch overhead tiny vs the tree's host mask-build) | **fixed & banked**, not a prose lever |
| **Rejection sampling (lossless, temp>0)** | prose τ1.14 — *worse* than greedy (flat target dist) | shelved |
| **CACTUS bumped acceptance (lossy)** | τ>1.5 but visibly corrupts prose (garbled tokens, repeats) at the δ that clears τ; also D2H-capped ~45 tok/s | rejected (quality) |
| **Verify-cost cuts (compressed lm_head, drop topk sync)** | ~+15% ceiling; the 27B forward is the floor, can't reach 56 losslessly | insufficient for prose |
| **MTP (native head)** | clears all genres incl. prose, lossless | **the answer for prose** |
| **DFlash prose drafter retrain** | the only lossless DFlash-prose lever; infra broken 3 ways (trainer won't compile, `load_target_init` unbuilt, mi300 torn down), correct d=5120 arch never run, multi-week, 30–40% odds | **deferred** (dedicated effort) |

## 5. State of spec-decode for qwen in hipfire (report)

**What works, durably:**
- **DFlash (linear, greedy, q8 KV)** is the production fast path for structured genres — 2–4× AR
  on code/reason/instruct, lossless. This is the recommended default for code/agentic/reasoning.
- **MTP (compressed-serial, K=3, p_min=0.4)** is the durable *all-genre* option, including prose,
  at 1.3–1.9× AR, lossless. It is the only mode that clears prose.
- The **tree-verify hipGraph** is now correct (was silently broken via `block_start` staleness)
  and available behind `HIPFIRE_VERIFY_GRAPH_TREE`, though its perf benefit is marginal on current
  workloads.

**What's shelved / why:**
- **DDTree** out-accepts linear on τ but never wins tok/s (per-cycle verify cost) — not a perf win
  on this hardware; useful only where acceptance matters more than throughput.
- **Chunked GDN** — algebraically exact but a perf regression at every shape; the sequential GDN
  is already an optimal kernel for this op. Banked behind a dead flag; do not re-chase.
- **Lossy acceptance (CACTUS)** corrupts prose at the τ it needs; not durable.

**The dividing line:** it's *creative-fiction* generation, not "prose," that's drafter-bound.
Expository/factual/reflective prose under the serving config is predictable enough that DFlash
clears it (τ2.3–2.84). Only *novel narrative* — where the target invents content the drafter can't
anticipate — collapses τ; that's a fundamental spec-decode property, and MTP's native head narrows
it enough to clear the 1.15× floor.

**Highest *durable* perf — chatml serving config, EVERY genre clears (gfx11 dense 27B-3.6):**
- code **98 tok/s** (DFlash, 2.28× AR, τ4.05)
- reason **124 tok/s** (DFlash, 2.89×, τ5.59)
- instruct **89 tok/s** (DFlash, 2.07×, τ3.55)
- prose (expository/reflective/descriptive) **66–76 tok/s** (DFlash, 1.55–1.77×, τ2.3–2.84)
- prose (creative fiction) **50 tok/s** (MTP, 1.16× — the fundamental-limit tail DFlash can't clear)

(`--no-chatml` raw continuation is faster for structured genres — code 130, reason 110 — but is the
canonical *code*-bench config, not the serving mode, and its prose-continuation fails; chatml is the
all-genre-durable config.)

## 6. Open items / recommendations

1. **Deploy MTP — DONE** (commit `fd717e5d`). Wired into the daemon: `LoadedModel.qwen35_mtp_head`
   (bundled `.mq4-mtp` trailer or `<trunk>.mtp` sidecar), `generate_qwen35_mtp`, gated
   `HIPFIRE_QWEN_MTP=1` + greedy + arch 5/6 + single-GPU (default path unchanged), generation-local
   `MtpSpecState` freed at every exit (state-bleed guard), defaults K=3/p_min=0.4. **Validated
   gfx11 27B-3.6:** routes through MTP (`"mtp":true`), **lossless** (byte-identical to AR at
   temp=0), **no state-bleed** (same prompt at positions b & d in a 4-request session →
   byte-identical output + τ), perf over floor (prose 65 / code 78 / capital 93 tok/s decode).
2. **Genre-aware mode selection** (DFlash for structured, MTP for prose) — or simply run MTP
   everywhere (it clears all genres; DFlash is faster on structured but MTP is durable everywhere).
3. **DFlash prose retrain** (lossless, to lift DFlash prose τ→2+): a dedicated effort — build the
   `load_target_init` d=5120 loader, a prose-balanced ChatML corpus, validate on available GPU.
   30–40% odds; only worth it if DFlash-on-prose throughput (~91 tok/s projected) is needed beyond
   what MTP already delivers (56).
4. **3.5 instruct** (1.25×, 4% short) — a small verify-cost lever (compressed lm_head ~+9%) closes
   it; 3.6 instruct already passes.
