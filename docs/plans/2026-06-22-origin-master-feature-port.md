# Goal: Port the 7 origin/master feature stacks into chaingun

**Status:** active worklist (created 2026-06-22). Drive this top-down; each
item is independently shippable. This file is the source of truth for the
`/loop` — update the checkboxes and the per-stack "Progress" notes as work
lands.

## Context (read first)

`chaingun` is the reference branch (NOT `master`). `origin/master`
(Kaden Schutt) diverged ~115 commits from the merge-base `e2d21ae2`
(2026-06-09). chaingun went the other way with a large modularization
refactor (per-arch crates: `hipfire-arch-*`, `hipfire-dispatch` families,
`hipfire-serving-core`, `hipfire-prompt`, `hipfire-quantize`). Because of
that, **none of these are `git cherry-pick`s** — each is a re-implementation
into chaingun's layout, compared against the master commits as reference.

### Already done (do not redo)
- Correctness fixes ported/closed: chat-template (n/a by design + shipped
  defensive warn & loader-spelling widening), unscatter grid.y OOB + rccl
  `unsafe`, gfx942 pair (verified n/a), MTP `fp16_x` stale-cache
  (`49881383`, ported + verified on gfx1151).

### What chaingun ALREADY has (don't re-add — extend/reconcile)
Quant formats present: `MFP4G32`, `HFP4G32`, `MQ2/3/4G256Lloyd`, `Oq4G256`,
`Qtip3G256`, full `HFQ2..6`, `MQ2..6G256`, `ParoQ4G128`. MoE: per-format
grouped-WMMA kernels (hfq4, mq2-lloyd variants, paro), `down_combine_grouped`.
Has: streaming/EP expert load, AWQ-from-hessian, MTP path (+ rollback).
Lacks: mfp4-**E8** variant, **MQ5**, merged graded MoE kernel + `TIER_MAP`,
registry v1/dynamic fetch, **any hipGraph infra**, MTP-perf primitives.

## Working protocol (every iteration)
- Work on `chaingun`. Pull/push go to the `fork` remote (`xynexus/hipfire`).
  Rebase onto `fork/chaingun` before pushing (it moves often). origin is
  upstream-only (no write access).
- **Formatting:** use `cargo fmt` (edition 2021), NOT the standalone `rustfmt`
  binary (it defaults to edition 2015 and over-formats). The pre-commit hook
  blocks any staged `.rs` that `cargo fmt --all --check` flags, so a touched
  file pulls in its pre-existing fmt debt — stage only the files you changed.
- **Gates:** kernel/quant/dispatch/forward changes trip the GPU coherence +
  speed gates via the pre-commit hook (this box is gfx1151, MoE models
  present). Quant-quality changes ALSO need astrea/KLD eval (`hipfire-eval`,
  `.agents/skills/astrea`). Don't `--no-verify`.
- Commit per coherent state; reference the source master hash in the message
  (`Ported from origin/master <hash>`). Co-author trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Document non-applicability explicitly (a verified "n/a" is a valid result).

## Priority order (value ÷ effort)
~~1 Registry v1~~ (BLOCKED — see below) → **2 MQ5 (active head)** → 3 MoE-AWQ
down → 4 MTP-perf (non-hipGraph subset) → 5 mfp4-E8 → 6 Graded N-tier →
7 gfx11-e8 / MTP hipGraph (defer).

---

## [BLOCKED] 1. Registry v1 + dynamic fetch (#47)  — needs product decision
Source: `25fbb01a` (Python `scripts/registry_gen.py` + v1.json + daily
HF-probe workflow), `828a5bc7` (TS `cli/registry_loader.ts` dynamic fetch +
24h cache + bundled fallback), `e86841cd` (readme).
**Verified non-portable (2026-06-22):** the entire stack targets the **bun/TS
distribution CLI** (`cli/*.ts`) + a Python generator. chaingun **deleted the
bun CLI** (converged to Rust `hipfire-cli`) and has **no model pull/download
or remote-registry subsystem at all** — only `hipfire list` of LOCAL models;
distribution relies on the local model dir + `/srv/huggingface` mount. No
`cli/`, no `registry/*.json`, no `HIPFIRE_REGISTRY*`, no HF/raw.githubusercontent
fetch in the Rust tree (reqwest exists only for server/admin).
There is nothing to port onto. Two paths, both a PRODUCT DECISION (not loop
work):
  (a) Build a NEW Rust model-pull/distribution feature in `hipfire-cli`
      (`hipfire pull <id>`, registry fetch + 24h cache + bundled fallback,
      sha256/size verification) — a sizable greenfield feature, not a port.
  (b) Skip — chaingun's local+mount workflow doesn't need a distribution
      registry. (Likely the right call given the Rust-native/converge-and-delete
      stance.)
Progress: BLOCKED pending user decision (a) vs (b). Loop SKIPS this; resume
only if the user picks (a). Active head moved to #2 (MQ5).

## [ ] 2. MQ5G256 (5-bit FWHT MagnumQuant)  — MEDIUM, fills MQ4↔MQ6 gap
Source: `f7efb940` (168 B/group, 5.25 bpw, full MoE decode parity).
chaingun: absent.
Approach: add `MQ5G256` DType + FWHT codec + decode/MoE kernels + dispatch
family wiring + quantizer `--format mq5` path, following the `MQ6G256`
shape. 22-file but single-commit reference.
Done when: quantize→serve a model at mq5; parity test vs scalar passes;
coherence gate green; astrea/KLD shows it sits between mq4 and mq6.
Progress: _not started_

## [ ] 3. MoE-AWQ per-expert down-proj  — MEDIUM, MoE quality
Source: `6198851e` (quantizer per-expert AWQ), `459a9eb4` (down-proj AWQ
kernel: indexed silu-mul-rotate), `3e5f2e9c` (dispatch wiring), `7b71833a`,
`853fe8ea`, `a0c17266`, `6381592d` (REAP importance capture).
chaingun: has AWQ-from-hessian + generic MoE-AWQ plumbing → RECONCILE, don't
duplicate.
Approach: add the per-expert down-proj AWQ kernel + quantizer path on top of
chaingun's existing AWQ; gate down-only quant.
Done when: per-expert down AWQ produces a coherent MoE file; coherence +
KLD eval show quality gain vs non-AWQ down at equal size.
Progress: _not started_

## [ ] 4. MTP perf — non-hipGraph subset  — SMALL cheap wins
Source: `5ac96a8f` + `1495be04` (MTP-head lm_head WMMA routing),
`bc5d005d` (decouple + adaptive-K MTP defaults), `becc0610` (gated small-B
verify decouple). (Skip the hipGraph commits here — see item 7.)
chaingun: MTP path exists (fp16_x just fixed); no hipGraph.
Approach: route MTP-head lm_head through WMMA; add decouple/adaptive-K
levers (per-arch, opt-out). These don't need hipGraph.
Done when: MTP decode tok/s improves on gfx1151 with no τ/coherence
regression (`coherence-gate-dflash`).
Progress: _not started_

## [ ] 5. mfp4-E8 + GPTQ/LDLQ-on-E8  — LARGE, HIGHEST quality value
Source (key anchors; see `git log chaingun..origin/master`): `f8fe55d5`
(runtime infra: E8/Lloyd/P GEMV+dequant kernels + dispatch + `e8.rs` codec),
`2fc54fce` (GPTQ/LDLQ-on-E8 quantizer), `1ec39d8a` + `e4975462` + `63c27582`
(native per-expert Hessian capture), `3e1cc712`/`8d383a0a`/`3b09ba9a`/
`05a030ac` (mfp3/mfp2-E8 cold tiers + GPTQ gating), `c1efafd4` (gfx12 port),
gfx1151 series `a0b78e07`/`b04eb9d0`/`232855eb`/`269091ee`/`0723d805`/etc.
chaingun: has base `MFP4G32`, NOT the E8-lattice variant or calibration.
Approach: dedicated project. Port the `e8.rs` lattice codec, the
Hessian-aware GPTQ/LDLQ rounding, per-expert Hessian capture, and the E8
GEMV/dequant kernels onto chaingun's existing mfp4 base; wire into
hipfire-dispatch + per-arch load. Validate heavily.
Done when: an mfp4-E8 (GPTQ-calibrated) MoE file serves coherently;
astrea/KLD beats plain mfp4 at equal size; coherence + speed gates green on
gfx1151 (and gfx12 path compiles).
Progress: _not started_

## [ ] 6. Graded N-tier MoE (TIER_MAP)  — MED-HIGH, depends on #5
Source: `45a3c166`/`687e181b`/`5cbb010b`/`f26f3e7e` (merged multi-branch
decode + batched-prefill grouped-WMMA), `4d44dddf`/`2682e764`/`40e49684`/
`f689db7a` (wiring/verdicts), `57e847a6`/`466e8e7c` (TIER_MAP).
chaingun: has cold-tier FORMATS but uses separate per-format kernels — needs
the MERGED graded kernel architecture + `HIPFIRE_MOE_TIER_MAP`. E8 tier
depends on #5.
Approach: port the merged dtype-tag MoE kernel (hot MQ6 / cold MQ2-3-Lloyd
[/ E8 once #5 lands]) for decode + batched prefill; add TIER_MAP.
Done when: a graded MoE file (e.g. hot-MQ6/cold-Lloyd) serves coherently at
smaller size than uniform MQ4 with comparable KLD; gates green.
Progress: _not started_

## [ ] 7. gfx11-e8 GEMV / MTP hipGraph  — DEFER
gfx11-e8 GEMV (`9908ba97`/`637341f5`/`0b137bdc`/`691cbb8b`/`b62fff1a`/etc.):
**most are documented WASHES** (default-off, no win on small per-expert MoE
shapes). Only revisit AFTER #5, and re-benchmark on this gfx1151 (and any
gfx1100) before porting — don't port a wash.
MTP hipGraph (`d2700109` AR-forward + MoE hipGraph, `3085f0df` GDN-tape
replay rollback): the +4-40% A3B wins ride on hipGraph capture, which
chaingun has ZERO of. Requires building hipGraph capture infra first — a
large prerequisite. Defer until/unless that infra is wanted on its own.
Progress: _deferred by design_

---

## Done-definition for the whole goal
All of #1–#6 landed on `chaingun` (pushed to fork), each passing its gates,
with #7 explicitly deferred (and gfx11-e8 re-benchmarked, not blind-ported).
Update this file's checkboxes as each lands.
