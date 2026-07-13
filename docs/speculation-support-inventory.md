# Speculation Support Inventory (per architecture)

**Status:** living document — updated while the n-gram seam work proceeds.
**Last updated:** 2026-06-27
**Branch:** `feature/speculator-ddtree`
**Scope:** what speculative-decode mechanism each model architecture supports in
hipfire today, plus — for arches with no native drafter — what (if anything)
exists upstream. Compiled from a per-architecture audit (one agent per arch crate
+ web search), 2026-06-23.

> This doc is the **status** register. For the **interface contract + how to add
> speculative decode to a new arch** (the `SpecTarget` trait, the registry
> wiring, pitfalls), see `.agents/skills/hipfire-arch-port/speculation.md`
> (step 7 of the arch-port skill).

## Vocabulary

- **n-gram drafter** — model-free, arch-generic (`crates/hipfire-runtime/src/spec_ngram.rs`),
  opt-in `HIPFIRE_NGRAM_DRAFT=1`. Any arch can opt in by implementing the
  `SpecTarget` verify seam (`crates/hipfire-runtime/src/spec.rs`).
- **MTP** — learned multi-token-prediction head shipped *with* the model weights.
  Hipfire currently serves the DeepSeek-V3/V4 form; Qwen3.5/3.6 native MTP
  artifacts remain disabled pending SPEC-003.
- **DFlash** — hipfire's block-diffusion drafter (published technique, arXiv
  2602.06036 / Z Lab). Qwen35-specific bespoke path
  (`crates/hipfire-arch-qwen35/src/dflash_spec.rs`); also available for any
  dense-attention target (LLaMA / plain Qwen3, arch 0/1) via the target-generic
  chain speculator (`crates/hipfire-runtime/src/dflash_generic.rs`).
- **SpecTarget verify seam** — arch-generic verify interface
  (`SpecTarget`/`Speculator`, shared `accept_greedy_prefix`). An arch's
  `spec_impl.rs` plugs in as the verify target. Sequential `verify_block`
  (one `forward_step` per token) is a correct byte-identical baseline; a
  block-parallel kernel is an optional perf optimization on top.

## Master table

| Arch crate | arch_id | Model family | In-repo spec support | Native drafter? | Daemon-wired? |
|---|---|---|---|---|---|
| qwen35 | 5/6 | Qwen3.5/3.6 (DeltaNet hybrid) | DFlash + n-gram + SpecTarget verify; native MTP deferred | ✅ DFlash (default greedy); native MTP head is not served | ✅ DFlash & n-gram default-wired; native MTP disabled pending SPEC-003 |
| deepseek4 | 9 | DeepSeek-V4 (MLA+MoE) | MTP + SpecTarget verify | ✅ MTP head (ships in weights) | ✅ auto at temp=0 if MTP weights present (load-resolved `mtp_k`, default 3, range 1–8; greedy-only) |
| llama | 0/1 | Llama/Mistral/Qwen3 dense | DFlash (block-diffusion, z-lab-style draft) + n-gram + SpecTarget verify | ✅ DFlash via external arch_id=20 HFQ draft (see below) | ✅ DFlash auto if `params.draft` is set to an arch_id=20 HFQ; n-gram opt-in `HIPFIRE_NGRAM_DRAFT=1` |
| qwen2 | 7 | Qwen2/2.5, VibeThinker | n-gram + SpecTarget verify (block-parallel) | ❌ (model-free n-gram only) | ✅ n-gram opt-in |
| qwen35-vl | 5 | Qwen3.5/3.6-VL | none (VL path is AR, CPU-sampled) | ❌ | ❌ (text backbone *is* qwen35 — reusable) |
| minimax | 10 | MiniMax-M2 (MoE) | n-gram + SpecTarget verify | ❌ (model-free n-gram only) | ✅ n-gram opt-in `HIPFIRE_NGRAM_DRAFT=1` |
| lfm2moe | 11 | LFM2.5-MoE (Liquid) | n-gram + SpecTarget verify (conv-state rollback) | ❌ (model-free n-gram only) | ✅ n-gram opt-in |
| cohere2moe | 12 | Cohere2-MoE / North-Mini-Code | n-gram + SpecTarget verify (sliding-window seq) | ❌ (model-free n-gram only) | ✅ n-gram opt-in (+ `Cohere2MoeEmit`) |
| dots-ocr | 8 | rednote dots.ocr (Qwen2-1.5B decoder) | n-gram + SpecTarget verify (VL decode-phase) | ❌ (model-free n-gram only) | ✅ n-gram opt-in (image-conditioned prefill unchanged) |

**Has real speculation today:** qwen35 (DFlash + n-gram), deepseek4 (MTP),
llama/qwen3 (DFlash via generic chain speculator + n-gram), qwen2 (n-gram).
Everything else is plain autoregressive.

## Per-arch detail

### qwen35 (arch 5/6, DeltaNet) — richest
- **DFlash** diffusion drafter: `dflash_spec.rs` (`DflashSpeculator`/`DflashState`),
  default production path for greedy generation, daemon-wired via `generate_dflash`→`generate_spec`.
- **Native MTP** head: implementation artifacts remain in `mtp_head.rs`/
  `mtp_speculator.rs`, but serving is disabled pending SPEC-003. Qwen
  `mtp_mode=on` rejects before native-head preflight, open, or GPU upload;
  `auto` and `off` remain AR-only and do not inspect a head.
- **n-gram**: opt-in `HIPFIRE_NGRAM_DRAFT=1`.
- **SpecTarget verify**: `spec_impl.rs` (`ModelSlot`), with DeltaNet snapshot/rollback.
- **Gap:** DFlash+MTP *composite* (`mtp_compose.rs`) validated in demos only, not
  promoted into the `Speculator`/`generate_spec` loop.

### deepseek4 (arch 9, MLA+MoE)
- **MTP** drafter: `mtp_speculator.rs` (`Deepseek4MtpDrafter`), daemon-wired via
  `generate_deepseek4_spec`→`generate_spec`. Auto-activates at `temp=0` when MTP
  weights present (`mtp_mode=auto`). The daemon resolves MTP K once at load: valid
  `HIPFIRE_DEEPSEEK4_SPEC_K` → `HIPFIRE_MTP_K` → load parameter → default 3
  (all 1–8); generation reads the resulting metadata.
- **Greedy-only** (`requires_greedy()=true`); **no n-gram fallback** (`spec_impl.rs`
  n-gram primitives intentionally return `Err`).
- Upstream: DeepSeek-V3/V4 ship 1 MTP module in public weights (`num_nextn_predict_layers=1`).

### llama (arch 0/1, LLaMA / Mistral / plain Qwen3 dense)
- **DFlash** (block-diffusion, z-lab-style): target-generic chain speculator
  (`crates/hipfire-runtime/src/dflash_generic.rs`). Requires an external
  arch_id=20 HFQ draft produced by `dflash_convert` from a z-lab `-DFlash`
  safetensors checkpoint. Supplied via the daemon `load` message `params.draft`
  field (path to the `.hfq` file). Auto-wired in `LlamaCarrier::load` when
  `draft_path` is set to a valid arch_id=20 HFQ.
  - **DDTree tree-SWOR** (shipped default): one tree-masked target forward per
    cycle; lossless / token-identical to AR at temp 0, distribution-exact at
    temp>0. Knobs: `HIPFIRE_DDTREE_BUDGET` (default 8), `HIPFIRE_DDTREE_TOPK`
    (default 2).
  - **Greedy chain** (`HIPFIRE_DFLASH_TREE=0` to opt out of the tree arm):
    lossless / token-identical to AR; temp>0 uses SpecInfer NAIVE sampling
    (distribution-exact).
  - **Empirical note (gfx1151):** break-even acceptance τ ≈ 2.5–3; the win is
    drafter-acceptance-bound. Batched GEMMs at B=1 limit verify-side gains.
- **n-gram**: opt-in `HIPFIRE_NGRAM_DRAFT=1`. **Ceiling:** unbatched MQ4G256/HFQ4
  projection GEMMs mean spec doesn't beat AR on every workload yet.
- Verify is `verify_block_argmax` (sequential); `verify_block_logits` and
  `verify_tree_logits` implemented for the DFlash chain and tree arms.
- Daemon wired: `generate_dflash`→`generate_spec` when `m.speculator.is_some()`.

### qwen2 (arch 7, Qwen2/2.5, VibeThinker)
- Implements `SpecTarget` and routes to `generate_dflash`→`generate_spec` when
  `m.speculator.is_some()` (daemon arm at `daemon.rs:5836`, the template).
- Verify is block-parallel (`forward_verify_block_batched`,
  `attention_decode_batched_history`).
- **n-gram only** (opt-in `HIPFIRE_NGRAM_DRAFT=1`). **Ceiling:** unbatched
  MQ4G256/HFQ4 projection GEMMs mean spec doesn't beat AR on every workload yet.

## Models missing a native drafter — what exists upstream

### A learned drafter exists upstream (adoptable)
- **llama/qwen3/mistral (0/1):** EAGLE-3 heads ship pretrained
  (`nvidia/Llama-3.3-70B-Instruct-Eagle3`, `AngelSlim/Qwen3-32B_eagle3`);
  Qwen3-0.6B/1.7B + Llama-3.2-1B work as same-family draft models.
- **qwen2/2.5 (7):** EAGLE-3 heads (`ruipeterpan/Qwen2.5-14B-Instruct_EAGLE3_UltraChat`,
  2.06–2.39×); Qwen2.5-0.5B/1.5B as draft models. No MTP head (predates it).
- **minimax-M2 (10):** MiniMax *declined* to ship MTP ("no bandwidth"), but
  community EAGLE-3 heads exist (`thoughtworks/MiniMax-M2.5-Eagle3` 2.11×;
  Together `Aurora-Spec-Minimax-M2.5`/`M2.1`).
- **qwen35-vl (VLM):** no VL-specific MTP, but VLM research drafters exist —
  SpecVLM (2.5–2.9×), ViSpec (Qwen2.5-VL, 1.49–1.87×), Spec-LLaVA (3.28×). Require training.

### Nothing exists upstream (n-gram is the only path)
- **lfm2moe (11):** Liquid says EAGLE-3 not worth it at LFM2 scale; no MTP head.
- **cohere2moe (12):** no public MTP/EAGLE; Cohere's Command-A spec-decode is proprietary.
- **dots-ocr (8):** no MTP/EAGLE for the Qwen2-1.5B decoder. But structured layout-JSON
  output makes n-gram a likely cheap win.

## n-gram seam — opportunity & cost

Enabling n-gram on a pure-AR arch requires (mirrors the qwen2 template):
1. `crates/hipfire-arch-<arch>/src/spec_impl.rs` — `impl SpecTarget for <Bundle>`
   (NEW; sequential `verify_block` is a correct baseline).
2. `lib.rs` — register `mod spec_impl;`.
3. `crates/hipfire-loader/src/carriers.rs` — the arch's `Carrier::spec_target_guard`
   returns `InPlaceGuard { bundle }`; ensure `build_speculator(...)` is called in `load`.
4. `crates/hipfire-loader/src/spec_build.rs` — extend n-gram arch_id gate
   (currently `matches!(arch_id, 0|1|5|6|7)`) to include the new arch.
5. `crates/hipfire-runtime/examples/daemon.rs` — insert
   `if m.arch_id == X && m.speculator.is_some() { generate_dflash(...); return; }`
   **before** that arch's bespoke `generate_<arch>` short-circuit.

Per-arch wrinkles:
- **minimax (10):** pure GQA AR — straight port of the qwen2 arm.
- **cohere2moe (12):** sliding-window attention — a block-parallel verify must
  replicate the windowed mask (sliding layers clip to last 4096; global/NoPE = full causal).
  Sequential verify sidesteps this for the baseline.
- **dots-ocr (8):** Qwen2-1.5B decoder, but decode is the **VL** path
  (`generate_vl_dots_ocr`, CPU-sampled) — routing differs from the text arches.
- **lfm2moe (11):** recurrent **conv-state** (`conv_states: Vec<GpuTensor>`, one
  `[hidden, K-1]` ring per conv layer) needs GPU snapshot/restore in
  `verify_block`/`commit_prefix` — same shape of problem as qwen35 DeltaNet. (Complex.)

## n-gram seam — work log

Dispatch snapshot from the one-agent-per-arch-crate phase (compile-check only).
The `Status` column records that phase; all four are now **fully wired + GPU-validated**
— see the completion table in the next section for the authoritative state.

| Arch | Agent | spec_impl.rs | Bundle type | Status |
|---|---|---|---|---|
| minimax (10) | sonnet | ✅ compiles `-p hipfire-arch-minimax` | `hipfire_arch_minimax::MiniMaxBundle` (new) | ✅ wired + GPU-validated (see below) |
| cohere2moe (12) | sonnet | ✅ compiles `-p hipfire-arch-cohere2moe` | `hipfire_arch_cohere2moe::Cohere2MoeBundle` (new) | ✅ wired + GPU-validated (see below) |
| dots-ocr (8) | sonnet | ✅ compiles `-p hipfire-arch-dots-ocr` | `hipfire_arch_dots_ocr::DotsOcrBundle` (new) | ✅ wired (VL decode-phase) + GPU-validated (see below) |
| lfm2moe (11) | opus | ✅ compiles `-p hipfire-arch-lfm2moe` | `hipfire_arch_lfm2moe::Lfm2MoeBundle` (new) | ✅ wired (conv-state rollback) + GPU-validated (see below) |

Agents were scoped to their own arch crate (`spec_impl.rs` + `lib.rs` mod) and
compile-check only; the shared wiring is integrated afterward.

### Wiring + validation status (2026-06-23, after shared-wiring pass)

| Arch | Loader+daemon wired? | Emitter | GPU validation |
|---|---|---|---|
| **minimax (10)** | ✅ (re-export bundle, carrier guard+emitter, build_speculator, spec_build gate +10, daemon arm) | `Qwen35Emit` (ChatML-clean) | ✅ **token-identical** generation (AR vs n-gram, greedy, `MiniMax-M2.7.mq2` 74GB on gfx1151/96GB): `AR[8:] == n-gram` exactly. Sole delta = the leading `<think>\n` delimiter (8 chars) that the bespoke AR path emits raw and `Qwen35Emit` consumes — cosmetic emitter rendering, NOT a generation divergence |
| **lfm2moe (11)** | ✅ (same pattern; conv-state rollback) | `Qwen35Emit` (ChatML-clean) | ✅ **AR == n-gram byte-identical** (575 chars greedy, `lfm2.5-8b-a1b.mq4`); detectors all clean → conv-state snapshot/rollback correct |
| **cohere2moe (12)** | ✅ (bundle re-export, carrier guard+emitter, build_speculator, spec_build gate +12, daemon arm) | `Cohere2MoeEmit` (ported marker state machine + guards) | ✅ **AR == n-gram byte-identical** (669 chars greedy, `North-Mini-Code-1.0.mq4.hfq`); zero marker leaks |
| **dots-ocr (8)** | ✅ (carrier `build_speculator`, spec_build gate +8, VL decode-phase routing in `generate_vl_dots_ocr`) | — (plain UTF-8 text stream; no `SpecEmit` — unframed layout-JSON) | ✅ **AR == n-gram byte-identical** (562 chars greedy, `dots-ocr.q8.hfq` + `dots_ocr_smoke_001.jpg`, table-heavy page, max_seq 8192). Currently *slower* (45.5 vs 60.5 decode tok/s) — sequential-acceptance vs batched-verify overhead on the 1.5B Qwen2 GEMMs; perf is the deferred follow-up |

**minimax + lfm2moe** loader + daemon both build clean (`-p hipfire-loader`,
`--example daemon`). The integration mirrors the qwen2 template exactly:
`pub use <arch>::<Bundle>` (delete loader-local struct) + carrier
`spec_target_guard`(`InPlaceGuard`)+`make_spec_emitter`(`Qwen35Emit`) +
`build_speculator(arch_id, None, None, true, max_seq)` in `load` + spec_build
n-gram gate `0|1|5|6|7|10|11` + daemon arm `if arch_id==X && speculator.is_some()
{ generate_dflash(); return; }` before the bespoke `generate_<arch>`.

**minimax emitter nuance (cosmetic, not blocking):** the n-gram path renders
`<think>` via `Qwen35Emit`'s think-state-machine (delimiter consumed), whereas the
bespoke `generate_minimax` AR path emits the `<think>` tag inline. Generation is
token-identical; only the delimiter surface differs. If exact AR-emitter match is
ever required, minimax would need its own `SpecEmit`; for now the shared ChatML
emitter is correct and arguably cleaner (reasoning-channel handling).

**cohere2moe DONE — both blockers resolved + a latent bug fixed:**
1. *Generation-intervention hook* (the architectural piece): `SpecEmit::take_forced()`
   (default empty → byte-identical no-op for all other emitters). When non-empty
   the `generate_spec` loop advances the target over each forced token, re-feeds it
   through `observe`, sets it as the next draft seed, and continues without honoring
   the suppressed terminator. Validated as a true no-op (lfm2moe + minimax re-ran
   byte-identical after the loop change).
2. *Emitter*: `Cohere2MoeEmit` (`crates/hipfire-arch-cohere2moe/src/spec_emit.rs`)
   ports the `Sec` state machine (marker ids resolved from tokenizer w/ North
   fixed-id fallback), reasoning-channel routing, defense-in-depth `<|MARKER|>`
   suppression, END_ACTION→`tool_calls`, finish-time tool-call-as-text recovery,
   AND both generation guards via `take_forced` (empty-turn guard force-injects
   `<|START_TEXT|>`; think-budget force-close injects `<|END_THINKING|><|START_TEXT|>`,
   sized off the new `SpecEmitCtx.max_tokens`). The four tool-call helpers
   (`parse_cohere_action`/`snap_*`) moved to the arch crate; the daemon AR path
   now imports them (single source).
3. *Latent bug found + fixed*: the agent's cohere2moe `spec_impl` `spec_advance`/
   `verify_block` did `state.n_tokens += 1` AFTER `decode_step` — but `decode_step`
   (via `decode_step_body`) ALREADY sets `n_tokens = position + 1`, so the cursor
   double-advanced, scattering prefill KV across positions 0,2,4,… → coherent-but-
   off-topic output. Removed the redundant `+= 1`. Also switched the bulk prefill
   to `forward_batch` (mirroring AR) so the KV is bit-identical to the batched AR
   path (per-token vs batched GEMM accumulation otherwise drifts greedy decode).

**dots-ocr DONE — VL decode-phase routing:** arch_id 8 decodes via
`generate_vl_dots_ocr` (image-conditioned), NOT the generic text `generate_spec`
loop, and its decoder state lives in **flat `LoadedModel` fields**
(`dots_ocr_config`/`dots_ocr_weights`/`qwen2_state`), not a `ModelState` bundle —
so the `Carrier::spec_target_guard` path the text arches use does not apply. The
resolution keeps the bespoke vision prefill untouched and routes only the
**decode phase**: after `forward_prefill_batch_embeds` leaves the Qwen2 KV warm,
when a speculator was built at load the daemon branches to a new
`decode_vl_dots_ocr_ngram`/`run_dots_ocr_ngram_loop` pair that (a) moves the flat
fields into a `DotsOcrBundle` for the `&mut dyn SpecTarget` borrow (restored on
return), (b) primes the n-gram drafter + fetches the first token WITHOUT
re-running the vision-conditioned prefill — `ChainSpeculator::prefill` with
`cache_hit=true` + empty suffix makes `spec_advance(&[], prompt_len, reset=false)`
just argmax the live logits and only `drafter.prefill_seed(prompt_ids)` — and (c)
runs the `prefill→step` contract with **plain UTF-8 text streaming** (no
`SpecEmit`: OCR output is unframed layout-JSON, no reasoning/marker/tool
channels). Because the n-gram verify always falls back to the same Qwen2 target
greedy argmax, the spec output is byte-identical to AR by construction — only τ
differs. Validated: **562-char OCR == AR exactly**. NOTE: currently *slower*
(45.5 vs 60.5 decode tok/s) — the batched verify of a wide K=12 draft against the
small 1.5B Qwen2 GEMMs costs more than the ~1–2 tokens it commits per window
(same MQ4 GEMM-batching ceiling noted for qwen2). **Perf is the deferred
follow-up**; correctness/wiring is complete.

### Perf follow-up — batched verify FALSIFIED as the fix (2026-06-24 τ measurement)

The deferred "batched verify" perf follow-up was **measured and falsified**
before building. n-gram acceptance (τ = accepted drafts / window) on the wired
no-drafter arches, greedy, on gfx1151:

| Arch | Workload | Verify path | τ | decode tok/s vs AR |
|---|---|---|---|---|
| cohere2moe (12) | free-form code (LRUCache) | sequential | **0.16** | slower (AR done has no tok/s; cycles=172 → 1.16 tok/cycle ≈ AR) |
| cohere2moe (12) | verbatim copy | sequential | **0.42** | slower |
| dots-ocr (8) | structured OCR (table page) | **batched** (qwen2 kernel) | **0.48** | **0.55×** (28.7 vs 52.6) |
| minimax (10) | free-form code (LRUCache) | **sequential** | **0.30** | slower (17.9 tok/s n-gram; cycles=153 → 1.3 tok/cycle) |

**Conclusion (nuanced — splits by whether the model is weight-BW-bound):**

For **small / compute-bound** decoders (dots-ocr 1.5B), batched verify does NOT
help: a B-token verify costs ~B× the compute (GEMM FLOPs scale with B; the model
is launch/compute-bound, not BW-bound), so there is no amortization. dots-ocr is
decisive: it *already* runs the block-parallel qwen2 batched verify, on the
highest-repetition workload of the set, and is still 0.55×. Here acceptance
(τ≪1) AND the lack of BW-amortization both sink it. cohere2moe (mid, sequential
verify) is the same story.

For **large / weight-BW-bound** decoders (minimax 79 GB MoE), the calculus
differs: `forward_batch` reads each weight ~once for all B tokens, so a batched
verify costs ~1 weight-read/cycle while committing (τ+1) tokens/cycle. At the
measured **τ=0.30** that is ~1.3 tokens per weight-read vs AR's 1.0 → a *plausible*
~1.3× decode win, where the current **sequential** verify (≈B forwards/cycle) is
a loss. **Two caveats keep this modest + uncertain:** (1) ceiling is only ~1.3×
at τ=0.30 (not the 2–5× the `forward_batch` docstring cites for high-τ DFlash);
(2) **MoE erodes the amortization** — B tokens route to up to B×k *different*
experts, so the batch reads more expert weight than one token does (partial, not
full, amortization). Realistic outcome: break-even to ~1.3×.

**Decision:** minimax is the ONE arch where a batched verify *could* pay off
(BW-bound), but the upside is modest and MoE-uncertain; the effort is porting
`forward_batch` to return per-row logits/argmax (a verify variant). The other
three arches are falsified (compute-bound and/or τ≪1) — do NOT build batched
verify for them. The general path to a real spec win remains a *learned* drafter
(EAGLE-3 / MTP, see upstream survey), not a verify kernel. n-gram stays correct
+ opt-in (`HIPFIRE_NGRAM_DRAFT=1`); a niche win gated on high-literal-repetition
traffic or (minimax only) a BW-bound batched-verify port.

Diagnostic added: `run_dots_ocr_ngram_loop`'s done envelope now reports
`tau`/`cycles` (parity with the text spec path), so spec-vs-AR acceptance is
visible per request. Probes: `/home/bjoern/hipfire-ngram-validate/tau_probe.py`,
`tau_copy_probe.py`.

### Key integration finding (all 4 arches)
The orphan rule forced each agent to define a NEW bundle struct *in the arch crate*
(parallel to the existing `hipfire_loader::*Bundle`), because `SpecTarget` is a
foreign trait. Wiring therefore requires reconciling the two: `ModelState::X` must
hold the **arch-crate** bundle so the carrier's `spec_target_guard` can return
`InPlaceGuard { bundle }`. This is exactly the pattern qwen2 already uses
(`crate::carrier::Qwen2Bundle` in the arch crate ↔ `ModelState::Qwen2`). The four
new bundles are field-identical to the loader bundles, so the swap is mechanical.

### Remaining shared wiring (per arch)
1. `ModelState::X` → hold the arch-crate bundle (replace loader bundle / flat fields).
2. `<Arch>Carrier::load` → construct the arch-crate bundle + call `build_speculator`.
3. `<Arch>Carrier::spec_target_guard` → return `InPlaceGuard { bundle }`.
4. `spec_build.rs` → add arch_id to the n-gram gate (`matches!(arch_id, 0|1|5|6|7)`).
5. `daemon.rs` → insert `if arch_id==X && speculator.is_some() { generate_dflash(); return; }` before the bespoke `generate_<arch>` short-circuit.

Per-arch wrinkles confirmed by the agents:
- **minimax (10):** `decode_step` returns host logits (argmax host-side); `eos_tok` baked at load; `ctx_capacity = state.max_seq`. Cleanest swap.
- **cohere2moe (12):** `decode_step` takes explicit `position: u32`; `reset(gpu)` zeros device KV. Sequential verify sidesteps the sliding-window batched-mask problem (a windowed `attention_decode_batched_history` is the perf follow-up).
- **dots-ocr (8):** `State = Qwen2State`; reuses `qwen2::forward_step`/`forward_verify_block_batched` directly. BUT decode is the VL path (`generate_vl_dots_ocr`, CPU-sampled), so step 5 needs a VL-aware routing decision, not the plain text arm.
- **lfm2moe (11):** conv-state snapshot/restore implemented (`Lfm2MoeSpecScratch` owns one F32 snapshot buffer per conv ring; `memcpy_dtod`; `commit_prefix` restores+replays on partial accept). `kv_cache_mut=None` (FlashCASK eviction unsound on hybrid). Needs GPU partial-accept byte-parity validation (keep `HIPFIRE_LFM2_GRAPH` off).
