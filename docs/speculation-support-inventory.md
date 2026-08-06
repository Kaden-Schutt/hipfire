# Speculation support inventory

Inventory of speculative-decode **capability** per architecture in hipfire
source. Capability is not product admission, not a performance floor, and not a
validation pass.

| Field | Value |
|---|---|
| Page state | **historical** inventory snapshot (see [`INDEX.md`](INDEX.md)) |
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Comparison base | `origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e` |
| Arch id table | [`architecture-ids.md`](architecture-ids.md) |
| Validation routes | [`VALIDATION.md`](VALIDATION.md) only |

**Fail closed:** before any product claim, re-read the symbols in
[Source anchors](#source-anchors). If this page disagrees with source, source
wins. Empty [`admissions.yml`](admissions.yml) means no inferred admissions.

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

| Term | Meaning |
|---|---|
| **n-gram drafter** | Model-free, arch-generic (`crates/hipfire-runtime/src/spec_ngram.rs`). Opt-in via `HIPFIRE_NGRAM_DRAFT=1` (env wins) or CLI/config `ngram` fields folded into `SpecLoadCfg`. Requires a `SpecTarget` verify seam. |
| **MTP** | Learned multi-token-prediction head shipped with model weights (DeepSeek-V3/V4 style; Qwen3.5/3.6 bundled `.mq4-mtp` / sibling `.mtp`). On qwen35, distinguish the strict `build_speculator` MTP arm from the native serve path (see Mechanism notes). |
| **DFlash** | Block-diffusion drafter (arXiv 2602.06036 / Z Lab). Qwen35 path: `crates/hipfire-arch-qwen35/src/dflash_spec.rs`. Dense llama/plain-Qwen3 targets: generic chain/tree path `crates/hipfire-runtime/src/dflash_generic.rs` with external arch_id=20 HFQ draft. |
| **DSpark** | Draft-sidecar path (`<stem>-dspark.<ext>`). Implemented for llama/plain-Qwen3 (arch 0/1), qwen35 (5/6), and deepseek4 (9). Preferred over DFlash/MTP/n-gram (or in-trunk MTP on deepseek4) when the sidecar loads and `ctx.spec.dspark != Some(false)`. |
| **DDTree** | Tree-masked verify + SWOR sampling. Dense generic path opt-in (`HIPFIRE_DFLASH_TREE=1`); knobs `HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK`. |
| **SpecTarget verify seam** | Arch-generic verify interface (`crates/hipfire-runtime/src/spec.rs`). Arch `spec_impl.rs` plugs the target. Sequential `verify_block` is a correct baseline; block-parallel verify is optional perf. |
| **Speculator registry** | Load-time `build_speculator` in `crates/hipfire-loader/src/spec_build.rs` (DFlash → MTP → n-gram cascade when each is eligible). Carriers may short-circuit **before** this with DSpark (llama 0/1, qwen35 5/6, deepseek4 9). |

How to add speculation to a new arch (interface contract): executable skill
[`.agents/skills/hipfire-arch-port/`](../.agents/skills/hipfire-arch-port/) (speculation step).

## Master table (source-derived)

n-gram arch gate in `build_speculator`:
`matches!(arch_id, 0 | 1 | 5 | 6 | 7 | 8 | 10 | 11 | 12)`.

| Arch crate | arch_id | Family | In-repo mechanisms | Native learned drafter? | Daemon / load wiring (summary) |
|---|---|---|---|---|---|
| llama | 0 / 1 | LLaMA / Mistral / plain Qwen3 dense | **DSpark** sidecar + DFlash generic (arch_id=20 draft) + n-gram + SpecTarget | External DSpark (`-dspark`) and/or DFlash HFQ draft when present and mode allows | Carrier precedence **DSpark > DFlash > n-gram**; daemon routes via `generate_dflash`→`generate_spec` when `speculator.is_some()` |
| qwen35 | 5 / 6 | Qwen3.5/3.6 DeltaNet hybrid | **DSpark** sidecar + DFlash + dual MTP surfaces + n-gram + SpecTarget | DSpark; DFlash draft; native MTP head (bundled trailer or sibling `.mtp`); strict `build_speculator` MTP arm under separate gates | Load precedence **DSpark > DFlash/`build_speculator` MTP/n-gram**. Serve: `HIPFIRE_QWEN_MTP=1` + loaded `qwen35_mtp_head` routes **native MTP before DFlash** (sampled also needs `HIPFIRE_MTP_SAMPLED=1`). Default CLI `dflash_mode=off` |
| qwen2 | 7 | Qwen2/2.5, VibeThinker | n-gram + SpecTarget (block-parallel verify available) | No | n-gram opt-in; daemon `arch_id==7 && speculator` → `generate_dflash` |
| dots-ocr | 8 | rednote dots.ocr (Qwen2-1.5B decoder) | n-gram on **decode phase only** after vision prefill | No | `generate_vl_dots_ocr` → n-gram decode loop when speculator built; vision prefill unchanged |
| deepseek4 | 9 | DeepSeek-V4 MLA+MoE | **DSpark** and/or in-trunk MTP + SpecTarget | Yes — `-dspark` sidecar and/or in-trunk MTP | Precedence **DSpark > MTP**; `mtp_mode` load param (default `auto`); generate gate `deepseek4_spec_requested` + temp policy; **n-gram not in the n-gram arch gate** |
| minimax | 10 | MiniMax-M2 MoE | n-gram + SpecTarget | No | n-gram opt-in; daemon arm before bespoke `generate_minimax` |
| lfm2moe | 11 | LFM2.5 hybrid conv+attn (+MoE variants) | n-gram + SpecTarget (conv-state snapshot/rollback) | No | n-gram opt-in; daemon arm before `generate_lfm2moe`. **Not** a Redline or batched-prefill admission |
| cohere2moe | 12 | Cohere2-MoE / North-Mini-Code | n-gram + SpecTarget + `Cohere2MoeEmit` | No | n-gram opt-in; daemon arm before `generate_cohere2moe` |
| qwen35-vl | 5 | Qwen3.5/3.6-VL | none (VL path is AR, CPU-sampled) | ❌ | ❌ (text backbone *is* qwen35 — reusable) |

**Has a learned or diffusion drafter path in-tree today:** qwen35 (DSpark ± DFlash ± MTP),
deepseek4 (DSpark/MTP), llama/plain-Qwen3 dense (DSpark and/or generic DFlash when
draft/sidecar loaded). **n-gram-capable with SpecTarget:** arch ids in the gate above.
Everything else remains plain autoregressive unless source gains a new arm.

## Mechanism notes

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

### DFlash (qwen35 and dense generic)

- **CLI default:** `speculation.dflash = "off"` in `crates/hipfire-config/src/lib.rs`. `"auto"` turns dense
  Qwen3.5-style targets on and A3B/MoE targets off unless a CASK sidecar or
  explicit `on` overrides.
- **Daemon:** `params.dflash_mode == "off"` skips draft load even if a draft
  path is supplied.
- **qwen35:** production greedy path can use `DflashSpeculator` when a draft is
  loaded; temp>0 uses distribution-preserving verifies (SWOR / fast-sample
  paths) with documented env opt-outs (`HIPFIRE_DFLASH_TEMP_SPEC=0`,
  `HIPFIRE_DFLASH_CHAT=0`, etc.).
- **llama 0/1:** `dflash_generic` requires arch_id=20 HFQ from `dflash_convert`.
  Chain is default; tree arm opt-in `HIPFIRE_DFLASH_TREE=1`.
- **Genre conditionality** is a measured property (see historical tables in
  [`BENCHMARKS.md`](BENCHMARKS.md)), not a universal win.

### DSpark

- Sidecar discovery: `<stem>-dspark.<ext>` next to the trunk when
  `ctx.spec.dspark != Some(false)` (`None` = auto keeps default-on load).
- **llama 0/1:** `LlamaCarrier` loads and builds DSpark **before**
  `build_speculator`. Precedence **DSpark > DFlash > n-gram**.
- **qwen35 5/6:** `finish_qwen35_load` builds `dspark_speculator` first; it wins
  over DFlash and the `build_speculator` cascade. Precedence
  **DSpark > DFlash / build_speculator-MTP / n-gram**.
- **deepseek4 9:** `-dspark` sidecar preferred over in-trunk MTP at load.
  Precedence **DSpark > MTP**.

### MTP

Two distinct qwen35 surfaces — do **not** collapse them:

1. **Strict `build_speculator` MTP arm** (`crates/hipfire-loader/src/spec_build.rs`
   via the gated head load in `finish_qwen35_load`): head is handed to
   `build_speculator` only when `HIPFIRE_QWEN35_MTP=1`, trunk path ends with
   `.mq4-mtp`, no DFlash draft, no DSpark speculator, eviction is None, arch
   5/6. Cascade inside `build_speculator` is DFlash → this MTP → n-gram.
   K via `HIPFIRE_QWEN35_MTP_K` (default 4 at build).
2. **Native qwen35 MTP serve path** (independent of the arm above): loader
   always attempts `qwen35_mtp_head` from a bundled `.mq4-mtp` trailer **or** a
   sibling `.mtp` sidecar (`<trunk>.mtp`). Daemon generate routes
   `generate_qwen35_mtp` when `HIPFIRE_QWEN_MTP=1` **and** the head is loaded
   **before** DFlash/AR; greedy (`temp≈0`) always, sampled (`temp>0`) only with
   additional `HIPFIRE_MTP_SAMPLED=1`. Default serve without those envs stays
   DFlash/AR.

- **deepseek4:** `Deepseek4MtpDrafter` reports `requires_greedy() == true`.
  Spec request from `HIPFIRE_DEEPSEEK4_SPEC_DECODE=1` or
  `mtp_mode`/`HIPFIRE_MTP_MODE` auto/on with weights present. `spec_k` from
  `HIPFIRE_DEEPSEEK4_SPEC_K` → `HIPFIRE_MTP_K` → default 2 on the daemon arm.
  DSpark sidecar, when present, is preferred over this in-trunk MTP at load.

### n-gram

- Enable: `HIPFIRE_NGRAM_DRAFT=1` or CLI-resolved `ngram_draft`.
- Window: `HIPFIRE_NGRAM_DRAFT_K` / config (default K=12 in `build_speculator`).
- deepseek4 is **intentionally outside** the n-gram gate; its SpecTarget n-gram
  primitives are not the production draft path (MTP/DSpark instead).
- lfm2moe verify must snapshot/restore conv rings; keep graph/capture modes
  consistent with the campaign contract when measuring parity.
- Historical τ/perf probes (2026-06-23/24) showed n-gram often slower than AR on
  compute-bound small decoders and only modest upside on large BW-bound MoE at
  low τ — correctness wiring ≠ speedup admission.

## Validation (mandatory routing)

Use [`VALIDATION.md`](VALIDATION.md) as the sole claim → route selector.

| Claim class | Minimum route (from VALIDATION) | Not sufficient |
|---|---|---|
| Spec wiring / compile-only | No-GPU routes when no GPU behavior claimed | Assuming GPU parity |
| Numerical / state parity for a verify path | Path-specific oracle for that arch/surface; **blocked** if none exists | `serve_harness.py` alone |
| User-facing serve semantics | `scripts/serve_harness.py` with the exact model | Numerical proof |
| LFM2.5 chat framing / thinking | `scripts/serve_harness.py` with the exact `lfm2.5:*` tag | Numerical parity |
| Perf improvement from a drafter | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) + stationary matched runs | Single warm run; genre-mismatched prompts |
| Product default / admission | Row in [`admissions.yml`](admissions.yml) | Speculator present at load; harness exit 0 |
| Redline-attributed claim | [`REDLINE.md`](REDLINE.md) ladder | Spec-decode success |

### Retired coherence gates (historical only)

`scripts/coherence-gate-*.sh` (including `coherence-gate-dflash.sh` and
family-specific batteries) are **retired as current acceptance evidence**.
They must not be required for merge, promotion, or benchmark claims. They may
be used only for historical reproduction. See
[`VALIDATION.md`](VALIDATION.md) § Retired coherence-gate scripts.

Campaign-specific omissions of coherence gates (for example an LFM effort) are
**not** a universal rule for every arch.

## Explicit non-claims

| Anti-pattern | Disposition |
|---|---|
| “n-gram is wired ⇒ ship as default” | **Rejected** — opt-in; no admission row |
| “LFM SpecTarget ⇒ LFM Redline / batched prefill promoted” | **Rejected** — separate surfaces; see [`REDLINE.md`](REDLINE.md) §11.4 |
| Coherence-gate green as current acceptance | **Rejected** |
| Stitching Redline product bench + daemon harness into route proof | **Rejected** — [`REDLINE.md`](REDLINE.md) §5/§7 |
| Treating [`BENCHMARKS.md`](BENCHMARKS.md) DFlash genre table as current baseline | **Rejected** — superseded methodology |
| Inferred `admissions.yml` row from this inventory | **Rejected** |

## Upstream learned-drafter survey (context only)

Not in-tree support. Useful when planning; does not admit a route.

| Family | Upstream notes (as of original 2026-06 survey) |
|---|---|
| llama / qwen3 / mistral (0/1) | EAGLE-3 heads and small same-family draft models exist publicly |
| qwen2/2.5 (7) | EAGLE-3 heads; no native MTP in base weights |
| minimax (10) | Vendor declined MTP; community EAGLE-3 heads exist |
| qwen35-vl | VL research drafters (SpecVLM, ViSpec, …) need training |
| lfm2moe (11), cohere2moe (12), dots-ocr (8) | No public MTP/EAGLE suitable as drop-in; n-gram remains the only model-free path |

## Source anchors

| Concern | Path / symbol |
|---|---|
| SpecTarget + accept loop | `crates/hipfire-runtime/src/spec.rs` |
| n-gram drafter | `crates/hipfire-runtime/src/spec_ngram.rs` |
| Generic DFlash + DDTree | `crates/hipfire-runtime/src/dflash_generic.rs` |
| DSpark core / llama body | `crates/hipfire-runtime/src/dspark_core.rs`; `crates/hipfire-arch-llama/src/dspark_body.rs` |
| build_speculator / n-gram gate | `crates/hipfire-loader/src/spec_build.rs` |
| Carriers (per-arch load + DSpark/build_speculator) | `crates/hipfire-loader/src/carriers.rs` (llama 0/1 DSpark>DFlash>n-gram; deepseek4 DSpark>MTP) |
| qwen35 DSpark / DFlash / dual MTP load | `crates/hipfire-loader/src/lib.rs` (`dspark_speculator`; `HIPFIRE_QWEN35_MTP` + `.mq4-mtp` → `build_speculator` MTP; native `qwen35_mtp_head` from bundled trailer or sibling `.mtp`) |
| Daemon generate routing | `crates/hipfire-runtime/examples/daemon.rs` (`generate_dflash`, `generate_spec`, `generate_qwen35_mtp` + `HIPFIRE_QWEN_MTP`/`HIPFIRE_MTP_SAMPLED`, deepseek4/qwen/llama/minimax/lfm/cohere/dots arms) |
| CLI dflash_mode default | `crates/hipfire-config/src/lib.rs` (`speculation.dflash = "off"`) |
| Per-arch SpecTarget | `crates/hipfire-arch-*/src/spec_impl.rs` |
| deepseek4 MTP greedy requirement | `crates/hipfire-arch-deepseek4/src/mtp_speculator.rs` — `requires_greedy` |
| Arch ids | [`architecture-ids.md`](architecture-ids.md) |

## Historical appendix — n-gram seam work log (2026-06)

Retained for provenance of the four-arch n-gram bring-up. **Not** current
procedure and **not** acceptance evidence.

| Arch | Wiring outcome (2026-06-23) | Notes retained from that campaign |
|---|---|---|
| minimax (10) | Loader + daemon + GPU greedy AR vs n-gram token-identical on fixture | Emitter delimiter cosmetic difference vs bespoke AR possible |
| lfm2moe (11) | Loader + daemon + conv-state rollback; AR == n-gram byte-identical on fixture | Keep graph flags consistent with parity contract |
| cohere2moe (12) | Loader + daemon + `Cohere2MoeEmit`; AR == n-gram byte-identical | Fixed double `n_tokens` advance bug during bring-up |
| dots-ocr (8) | VL decode-phase n-gram after vision prefill; AR == n-gram byte-identical | Then slower than AR on measured OCR fixture (low τ / small GEMMs) |

2026-06-24 τ probes (gfx1151, campaign-local) falsified “batched verify fixes
all n-gram losses” for compute-bound small decoders; large BW-bound MoE remained
a modest/uncertain case only. Decision at that time: keep n-gram correct +
opt-in; prefer learned drafters for real speedups.

Shared integration pattern: arch-crate bundle owns `SpecTarget` (orphan rule);
loader re-exports; carrier `spec_target_guard` returns `InPlaceGuard`; daemon
spec arm sits **before** the bespoke `generate_<arch>` short-circuit.

For any new arch, re-derive steps from source and the arch-port skill — do not
treat this appendix as an executable checklist that bypasses
[`VALIDATION.md`](VALIDATION.md).
