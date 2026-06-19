# MoE Mixed-Quant Experts — Findings (2026-06-11)

Target model: **Qwen3.6-35B-A3B** (qwen3_5_moe, 256 experts/top-8, 40 layers, hidden
2048, moe_inter 512). All KLD vs an f32-native oracle (`q36a3b-f32-oracle.hfq`); refs
`q36a3b-{wt2,agentic}-f32.kldref.bin`. Branch `feat/moe-awq-experts` on mi300.
Decode = per-token scoring; "q8/fwht4" = KV mode.

---

## 1. Per-expert AWQ — DEAD END for quality (shipped + measured)
Down-proj AWQ wired end-to-end (commit 3e5f2e9c; indexed silu-rotate `x/s` before FWHT,
gated on per-expert `down.awq_scale` sidecars). Validated correct: down-AWQ file
forwards coherent (PPL 4.78) vs kill-switch garbage (PPL 164.9). **But the quality A/B
(dense-AWQ baseline vs +down-expert-AWQ, isolated):**

| corpus / KV | baseline KLD | +down-AWQ KLD | Δ |
|---|---|---|---|
| wt2 / q8 | 0.03385 | 0.03498 | +3.3% (hurts) |
| wt2 / fwht4 | 0.03572 | 0.03659 | +2.4% (hurts) |
| agentic / q8 | 0.16299 | 0.15961 | **−2.1% (helps)** |
| agentic / fwht4 | 0.16647 | 0.16814 | +1.0% (hurts) |

**Verdict:** expert-AWQ is a wash — helps only agentic/q8, ~free but redundant once you
spend real bits. Dense-AWQ MQ4 is the best-of-MQ4 floor. Quality comes from bits, not
activation-aware 4-bit scaling.

---

## 2. Expert precision ladder (uniform, all 256 experts same format)
Commits: MQ6 wiring 7b71833a (mixed gate_up/down dispatch by individual dtype, gfx942 —
no kernel port needed), MQ5 f7efb940 (full decode parity, ultracode workflow).

**KLD (q8 KV, max-chunks 32):**

| variant (gate_up/down) | bpw | size | wt2 KLD | agentic KLD |
|---|---|---|---|---|
| MQ4 / MQ4 | 4.25 | 19.7 GB | 0.03385 | 0.16299 |
| d6 = MQ4 / **MQ6** | — | 22.3 GB | 0.02739 | 0.13519 |
| **MQ5 / MQ5** | 5.25 | 23.7 GB | 0.01910 | 0.10603 |
| **+P = MQ6 / MQ6** | 6.25 | 27.7 GB | 0.01593 | 0.08677 |

**PPL** (f32 oracle: wt2 5.350, agentic 5.902): MQ4 wt2 5.433/ag 6.099 → MQ5 5.413/6.051
→ +P 5.396/5.967.

Key results:
- **gate_up is the DOMINANT lever, not down.** +P vs d6 (= adding gate_up MQ6 on top of
  down MQ6) buys *another* −32..−42% KLD AND is what moves PPL toward f32. **d6 (down-only
  MQ6) is a half-measure: −14..−19% KLD but PPL-FLAT** (down improves distribution
  fidelity, not next-token likelihood). The PPL win lives in gate_up.
- **KLD ≫ PPL gap at short ctx (512):** MQ6-down cuts KLD a lot but PPL barely moves;
  matches the old kmap bench (the MQ6 PPL win needs >3K ctx; Q8 KV masks it). Use asym4/
  fwht4 KV + longer ctx to surface the PPL win.
- **MQ5** captures ~80% of the MQ4→+P win; only 4 GB under +P → a *measured reference*,
  not a SKU (MQ4-Lloyd at 5.0 bpw would likely dominate it).
- kmap (full promotion) bench (gfx1151, 2026-05-08): MoE +1.7% PPL @ctx2048 Q8 but
  **−19.8% @ctx8192 asym4** (unmasked). kmap'd files need gfx12 for the dense MQ6 GEMV
  (`gemv_mq6g256_prerotated`) — won't forward on gfx942; experts-only promotion does.

---

## 3. REAP importance gate (commit 6381592d) — PASSES
Instrumented per-(layer,expert) `count / Σgate / Σ‖out‖ / Σ(gate×‖out‖)` capture
(`HIPFIRE_MOE_EXPERT_STATS`); TSVs `expert_stats_{agentic,wt2}.tsv`.

- **freq ≈ contribution: Spearman 0.92–0.93** → routing frequency (free from imatrix
  `.counts`) is a fine grader. The "freq≠contribution" worry that parked REAP is resolved.
- **Concentration:** Gini(contribution) 0.76–0.77; top-20% units = ~80% of contribution,
  top-50% = 96%. Deepens with depth (early Gini 0.68 → late 0.75).
- **NEW: hot set is DOMAIN-SPECIFIC** — agentic vs wt2 top-10% overlap only **24%**
  (Jaccard 0.14); top-20% 37%; top-30% 47%. A single graded quant needs the UNION hot-set
  (top-10% union ≈ 17.5% of units) or per-domain builds.

---

## 4. Tier size/perf model (A3B, per-expert graded by importance percentile)
- **Size (exact):** `2.55 GB fixed + 4.026 GB/bpw × avg_bpw`. Reproduces MQ4 19.66 /
  MQ5 23.69 / +P 27.71.
- **Perf (BW-bound est, anchor MQ4 = 150 tok/s gfx11):** per-token read ≈ 0.96 GB fixed
  (lm_head + shared + attn, constant) + ~0.54 GB routed-experts (MQ4) → routed = ~36% of
  read, so bpw swings give modest tok/s swings. Routed weighted by route-freq quintiles
  `[0.52,0.22,0.13,0.08,0.05]`. **Size is cold-driven; perf is hot-driven (the experts
  that fire are the high-bit ones).**
- **Quality est:** hot-dominated; **cold-MQ2-Lloyd tail is the unmeasured risk**
  (coherence-gate, not KLD, is the gate).

| Tier | blend (hot 20% → cold 20%) | avg bpw | size | est tok/s | est ag KLD | character |
|---|---|---|---|---|---|---|
| 1 | MQ4×5 | 4.25 | 19.7 GB | 150 | 0.163 (meas) | uniform baseline |
| 2 | MQ4/MQ4/MQ2L/MQ2L/MQ2L | 3.05 | 14.8 GB | ~157 | ~0.17 | compress (smaller+faster) |
| 3 | MQ6/MQ4/MQ4/MQ2L/MQ2L | 3.85 | 18.0 GB | ~141 | ~0.12 | balance |
| 4 | MQ6/MQ5/MQ4/MQ2L/MQ2L | 4.05 | 18.9 GB | ~138 | ~0.11 | quality-lean |
| 5 | MQ6/MQ5/MQ4/MQ3L/MQ2L | 4.25 | 19.7 GB | ~137 | ~0.10–0.11 | max-grade, **iso-MQ4-size** |
| ref | +P (all MQ6) | 6.25 | 27.7 GB | ~128 | 0.087 (meas) | uniform high |

Headlines: graded tiers 3–5 **dominate +P** (smaller + faster + near-same quality);
**Tier 5 = MQ4 footprint, near-+P quality, ~9% slower**; Tier 2 is the *fastest*
(compress). gfx11 all-resident only — the 5700XT-with-GTT perf model differs (cold = PCIe).

---

## 5. Mixed-precision decode kernel — scoping → **dtype-tag**
Need: routed gate_up + down GEMVs handle per-expert dtype (silu-rotate + combine are
weight-agnostic, unchanged; per-expert dtype already in `ExpertWeights[i].gpu_dtype`,
just collapsed to `[0]` today). Grid is **block-per-expert** (`blockIdx.y = krank`).

- **dtype-tag (one merged kernel, per-block branch): RECOMMENDED.** Block-uniform dtype →
  **no divergence**; 1 launch (matters for the launch-sensitive 5700XT); silu-rotate +
  combine untouched; just a per-expert u8 tag table + the merged kernel. Cost: union of
  two dequant paths (BW-bound → modest occupancy hit).
- two-pass (bucket by dtype, reuse existing MQ6 + MQ2-Lloyd indexed kernels): more
  launches + needs topk **permutation** threaded through silu/combine (bug-prone). The
  "kernel reuse" win is undercut by the permutation glue.

MQ2-Lloyd MoE indexed kernels already exist (ds4: `gemv_mq2g256_lloyd_moe_down_indexed*`).

---

## 6. 5700XT / gfx1010 (RDNA1) — origin loop CLOSED
qwen3.5-0.8B mq4 runs **coherent, native gfx1010** (no GFX override) at **256.8 tok/s
decode** on hipx (HIP_VISIBLE_DEVICES=0, 8.6 GB). RDNA1 forward-correctness gate passes.
A3B-on-5700XT plan: **GTT cold-expert offload** — host-map cold experts
(`hipHostMalloc...Mapped`); the indexed MoE GEMV ptr-table is location-agnostic, so no
pager — heavy-tail masks GTT slowness. Moved to a dedicated worktree
(hipx `~/hipfire-gfx1010`, branch `gfx1010-opt`).

---

## 7. Open — the matrix to run next
Gating question: **does a graded quant hold quality once the cold MQ2-Lloyd tail is real?**
Recommended order:
1. **Cold-tier floor (cheapest, NO new kernel):** uniform all-MQ2-Lloyd-GPTQ + all-MQ3-Lloyd
   experts (existing quant fns) → eval via the CPU-top-K fallback (how the f32 oracle
   already forwards) → KLD + **coherence-gate**. Completes the Lloyd ladder + de-risks the
   cold tier before any kernel.
2. If cold holds → build the **dtype-tag mixed kernel** (Tier 5 or 3) + confirm the graded
   quant matches the model. If it derails → cap cold at MQ3-Lloyd, re-table.

---

## 8. Mixed-precision merged kernel — BUILT + WIRED (down) + MEASURED (commit 687e181b)
Workflow w5nmrl7xh. Two merged dtype-tag GEMVs `gemv_mixed_moe_{down,gate_up}_*`: per-block
branch on `dtype_tags[expert_id]` (0 = MQ6 6-bit affine 200 B/group, 1 = MQ2-Lloyd 2-bit
codebook 72 B/group). Grid is block-per-(row,krank,token) → all 32 threads share one
expert → one tag → **no warp divergence**. Single shared accumulator + shared shfl-reduce;
**expanded write** so one shared `moe_down_combine` serves both dtypes (do NOT use the Lloyd
atomic self-combine in mixed mode — `routed_down_self_combines` is forced off when
`expert_dtype_tags.is_some()`, else Lloyd experts double-count or zero out).

Wiring (decode only): `[n_exp]` u8 tag table (`DType::Raw`, 1 B/elem) built at load iff down
dtypes are not all identical → `MoeResolution.routed_indexable_mixed_per_expert` admits the
k=8 gpu-topk path → `run_moe_decode` merged-down branch. Uniform files are byte-identical
(every branch gated on `expert_dtype_tags.is_some()`, None unless the .hfq carries mixed
per-expert down dtypes). gate_up merged kernel is built + compiles clean but **intentionally
not wired** (first target keeps gate_up uniform MQ4).

Quantizer `HIPFIRE_MOE_GRADED=1` `HIPFIRE_MOE_HOT_FRAC=0.2`: per-layer, rank experts by
imatrix `.counts` DESC, top-frac → MQ6 else MQ2-Lloyd. Gated to **down_proj only** (a bug
where grading both gate_up+down while only down was wired read MQ6 gate_up bytes as MQ4 →
NaN logits; caught + fixed mid-flight). File `/workspace/q36a3b.graded` = 18.05 GB,
51 hot (MQ6) / 205 cold (MQ2-Lloyd) down experts per layer, gate_up uniform MQ4.

**Verdict — does graded down-only beat uniform MQ4? NO.**

| file (size) | wt2 KLD | wt2 PPL | ag KLD | ag PPL | coherence |
|---|---|---|---|---|---|
| MQ4 uniform (19.7 GB) | 0.0339 | — | 0.163 | — | — |
| **graded down-only (18.05 GB)** | **0.0770** | 5.557 | **0.2536** | 5.976 | OK 0h/0s 8/8 |
| MQ2-Lloyd uniform (11.6 GB) | 0.238 | — | 0.466 | — | coherent |

It sits ~midway between MQ2L and MQ4: 2.3× worse than MQ4 on wt2, 1.6× on agentic, only
~8% smaller. **Expected** — matches §2's gate_up-dominant finding: putting 80% of *down*
into 2-bit while gate_up stays MQ4 spends bits on the weak lever. Cross-domain: the
agentic-graded hot-set does NOT transfer (wt2 degrades *more* relative to MQ4), consistent
with the corpus-specific hot-set in §3.

**Kernel proven correct/firing:** grading hot-20% down→MQ6 cut agentic KLD from the
all-MQ2L control's 0.560 to 0.347 (−38%) — only achievable if the tag-0 MQ6 branch actually
dequants those experts as MQ6. Combine/tag-table/coherence all validated. (Daemon path
needs a fresh `daemon` binary — a stale one falls to the CPU per-expert path and panics
`no impl for GemvMq2G256LloydPrerotated`; rebuild resolves, no source change.)

**The one variant that could beat MQ4: full-graded (gate_up too) ~14.8 GB.** The gate_up
merged kernel is already built; the remaining work is (a) wire `expert_dtype_tags` for
gate_up in `run_moe_decode`, (b) un-gate the quantizer to grade gate_up, (c) re-quantize the
~14.8 GB full-graded file, (d) re-run this KLD/coherence battery. That puts the cold bits on
the dominant lever and is the real test of the graded-quant thesis.

---

## 9. N-tier graded sweep — T3-2L / T3-3L (commit 45a3c166, workflow ww41rkaie)
Extended the merged kernel to **4 branches** (tag0=MQ6 200B, tag1=MQ2-Lloyd 72B, tag2=MQ4
136B, tag3=MQ3-Lloyd 112B; block-uniform, single-acc, expanded write; MQ3L uses an 8-slot
`cb_lds` codebook + per-group `__syncthreads()`). **Wired gate_up** (was down-only) — one
shared `[n_exp]` u8 tag table, 4 tags, used by both projections. Quantizer
`HIPFIRE_MOE_TIER_MAP=<file>` grades BOTH gate_up + down per-expert; `scripts/gen_tier_map.py`
builds the map from a **union (agentic ∪ wt2, per-layer-normalized sum_contrib)** REAP ranking
(top-20%→MQ6, next-30%→MQ4, bottom-50%→cold). Both kernels compile gfx942: VGPR=33, zero
spill, LDS=32B, occupancy 8 waves (BW-bound, fine).

**Results (q8 KV, max-chunks 32, vs f32 oracle; MQ4 anchor 19.7 GB 0.0339/0.163):**

| SKU | size | wt2 KLD | ag KLD | coherent | vs MQ4 |
|---|---|---|---|---|---|
| MQ3L-uniform | 16.6 GB | 0.071 | 0.242 | yes | — |
| **T3-2L** (20%MQ6/30%MQ4/50%MQ2L) | 16.85 GB | 0.0352 | 0.1631 | **✅** | wt2 +3.8% (CI tie), ag tie, −2.85 GB |
| MQ4-uniform (anchor) | 19.7 GB | 0.0339 | 0.1630 | yes | — |
| **T3-3L** (…/50%MQ3L) | 19.76 GB | 0.0249 | 0.1332 | **❌ decode-only** | **wt2 −26.5%, ag −18.3%** |

**Verdict:**
- **T3-2L = shippable ~17 GB "MQ4-lite" SIZE SKU.** Ties MQ4 on quality (within bootstrap CI
  0.0294–0.0435 on wt2; dead-even on agentic), coherent, −14% size. **Pareto-dominates uniform
  MQ3L** — half the KLD on both corpora for +0.25 GB. The one genuine frontier point graded
  earns today.
- **T3-3L = the real quality win, blocked.** Beats MQ4 on both corpora at iso-size, but is
  **decode-only**: the MoE batched-**prefill** path hardcodes the MQ4 group stride (136 B/grp)
  and `is_batchable_la()` excludes MQ3G256Lloyd, so prefill would silently corrupt the 112 B/grp
  cold tier → it correctly guards with a panic. The KLD comes from per-token scoring mode (valid,
  routes through the working decode kernels). Not a kernel bug — a parity gap. Unblock = port the
  MQ3L/MQ2L branches into the MoE batched-prefill GEMM (same per-block `dtype_tag` dispatch as the
  decode kernels) + add MQ3G256Lloyd to `is_batchable_la()`.

**Two meta-findings:**
1. **Union ranking held both corpora** — no corpus collapse (T3-3L improved wt2 *and* agentic
   simultaneously). One shared per-expert union tier table is robust; this is the methodology to
   keep (contrast §1's corpus-specific down-AWQ wash).
2. **First-order additive model was optimistic ~20–26%** (T3-2L pred 0.028 → 0.035; T3-3L pred
   0.021 → 0.025). Consistent sign ⇒ the 50% cold mass contributes **super-additively** (routing
   puts non-trivial probability through cold experts; error doesn't fully average out). Use the
   model for *ranking* blends, not absolute targets; ×1.2–1.3 the cold-tier term. This recalibrates
   the §8 <16 GB estimates up ~25% (the 15.5 GB MQ5/MQ3L/MQ2L blend ≈ 0.045 wt2 — beats uniform
   MQ3L, not MQ4).

**Recommended next:** (1) ship T3-2L as the 17 GB SKU; (2) **port MQ3L/MQ2L into MoE
batched-prefill to unlock T3-3L** (highest ROI — turns a measured iso-size win into a shippable
one + makes both graded SKUs daemon-serveable); (3) tune the 20/30/50 boundary (25/35/40, or cold
MQ2L→MQ3L) to convert T3-2L's tie into a clean sub-MQ4 win; (4) do NOT add more tiers until the
prefill port lands.

---

## 10. CONSOLIDATED variant table (canonical reference)
All KLD: per-token, q8 KV, max-chunks 32, vs f32 oracle `q36a3b-f32-oracle.hfq`. Corpora:
wt2 = wikitext-2 (general), ag = agentic/code. Sorted by size.

| # | Variant | gate_up / down | size GB | wt2 KLD | ag KLD | coherent | verdict |
|---|---|---|---|---|---|---|---|
| — | f32 oracle | f32 / f32 | 138 | 0 | 0 | yes | ref (PPL 5.350/5.902) |
| 1 | MQ2-Lloyd uniform | MQ2L / MQ2L | 11.6 | 0.238 | 0.466 | yes | frontier (floor) |
| 2 | MQ3-Lloyd uniform | MQ3L / MQ3L | 16.6 | 0.071 | 0.242 | yes | DOMINATED by T3-2L |
| 3 | **T3-2L graded** | 20/30/50 MQ6/MQ4/MQ2L both | 16.85 | 0.0352 | 0.1631 | yes | **frontier — ship (MQ4-lite, -14% size)** |
| 4 | down-only binary | MQ4 / hot20%MQ6·cold80%MQ2L | 18.05 | 0.0770 | 0.2536 | yes | DOMINATED (wrong lever) |
| 5 | MQ4 uniform (anchor) | MQ4 / MQ4 | 19.7 | 0.0339 | 0.1630 | yes | frontier (dense-AWQ baseline) |
| 6 | down-AWQ | MQ4 / MQ4+AWQ | 19.7 | 0.0350 | 0.1596 | yes | wash vs MQ4 |
| 7 | **T3-3L graded** | 20/30/50 MQ6/MQ4/MQ3L both | 19.76 | 0.0249 | 0.1332 | NO (decode-only) | beats MQ4 -26%/-18% iso-size; blocked on prefill |
| 8 | d6 (down-only MQ6) | MQ4 / MQ6 | 22.3 | 0.0274 | 0.1352 | yes | KLD-helps PPL-flat (down=weak lever) |
| 9 | MQ5 uniform | MQ5 / MQ5 | 23.7 | 0.0191 | 0.1060 | yes | frontier |
| 10 | +P all-MQ6 | MQ6 / MQ6 | 27.7 | 0.0159 | 0.0868 | yes | frontier (high end) |

**Pareto frontier (coherent only):** MQ2L(1) -> T3-2L(3) -> MQ4(5) -> MQ5(9) -> +P(10).
T3-2L is the one new point graded ADDS to the frontier. T3-3L(7) would dominate MQ4 at
19.7GB if serveable (blocked on MoE batched-prefill MQ3L parity).

**Uniform-ladder PPL** (oracle wt2 5.350 / ag 5.902): MQ4 5.433/6.099, MQ5 5.413/6.051,
+P 5.396/5.967. Graded SKUs are KLD-only (PPL not captured).

**Levers learned:** (a) bits win — quant fidelity >> block size; (b) gate_up is the
DOMINANT lever (down-only MQ6/AWQ = KLD-helps/PPL-flat; the +P PPL win lives in gate_up);
(c) graded multi-tier beats uniform at iso-size ONLY when the warm 20-50% band stays >=MQ4
and only the 4%-contribution cold tail drops to MQ2L (binary 80/20 dumps the warm band ->
loses); (d) union (agentic∪wt2) ranking holds both corpora; (e) first-order additive KLD
model is optimistic ~20-26% (cold tier super-additive).
