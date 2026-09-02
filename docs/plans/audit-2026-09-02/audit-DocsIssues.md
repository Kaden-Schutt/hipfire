<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: DocsIssues

# Audit DocsIssues FULL REPORT

Slice DocsIssues. Checkout /home/kaden/ClaudeCode/warpfront/hipfire @ origin/master 8cd15a62b. gh issues/PRs 2026-09-02. No local:// written (no Write tool).

## Broken

1. Dead --example daemon/test_kernels (verified). CONTRIBUTING.md:58-59 GETTING_STARTED.md:129 VALIDATION.md:84-97 prescribe cargo build --release --features deltanet --example daemon -p hipfire-runtime and --example test_kernels. Product: crates/hipfire-daemon Cargo.toml:8-10 [[bin]] daemon. AGENTS/CLAUDE already correct. No test_kernels example/source on master (workspace grep empty). Impact: contributor dead ends; open #645 NixOS no example named.

2. Skills path .skills/ missing (verified). CONTRIBUTING.md:33-44,139-145,269-273 → .skills/hipfire-*. Actual .agents/skills/ (INDEX+CLAUDE+disk). .skills/ absent.

3. MODELS DeepSeek default wrong (verified). MODELS.md:148-149 bare deepseek-v4-flash → mq2lloyd 86.2GB. registry/models.json:255-257 default file mq2r 82GB; Lloyd is deepseek-v4-flash:mq2lloyd :214-216. MQ2R default since 2026-08-14 per registry desc.

4. README counts disagree (verified). Badge 61 curated; body 77 pullable; models.json ~79 keys (maple-preview, bonsai bq1/tq2, ornith, Qwen3.8 ladder). README family table omits maple/bonsai/ornith. hipfire list -r is correct authority; marketing numbers false.

5. ARCHITECTURE force_local wrong (verified). ARCHITECTURE.md:107: HIPFIRE_LOCAL|--kv-mode|--json|--no-stream. CLI.md:70: JSON/non-stream do NOT force local; HIPFIRE_LOCAL|--kv-mode|--image. Source main.rs:1911-1918: HIPFIRE_LOCAL|image|kv_mode|kv_backend|speculation|model_draft|draft_max|dspark_conf_threshold — neither json nor no_stream. CLI closer but incomplete.

6. ARCHITECTURE carriers incomplete (verified). Table ends Cohere2Moe(12). Missing Gemma4(13/22), Muse Glimmer(14), Maple(15). architecture-ids + workspace crates present.

7. INDEX ownership/pin drift (verified). INDEX.md:8-12 inventory date 2026-07-22 branch beta audited ref 202282de…. Release line master. Truth collision: INDEX perf-checkpoints=measured BENCHMARKS=historical; perf-checkpoints/README.md:3-9 lifecycle historical every file AND calls BENCHMARKS current product claims; BENCHMARKS.md:1-10 self-historical. Missing from INDEX top-level set while linked: crate-maps.md GLOSSARY.md GEMMA4_ESERIES_* qwen35-vl-mq4v2-spec.md lfm2-vl-mq4v2-spec.md.

8. CONTRIBUTING active asks closed (verified). CONTRIBUTING.md:286-295 Issue #57 CLOSED 2026-04-27; #58 CLOSED 2026-05-06; #50 OPEN title Test on gfx1152 (not crash prose). Crate topology omits daemon/loader/engine/generate/config/registry/cli/most arch crates.

9. Validation retired harness (verified). VALIDATION.md kernel channel still test_kernels; deepseek4-pr-body.md:87-90 historical coherence-gate+test_kernels easy to re-copy. ci.yml agentic GPU gates removed; GPU manual per VALIDATION.

## Missing

1. No docs drift CI beyond scripts/check-env-docs.py (VALIDATION). No link/example/INDEX completeness gate.
2. MODELS/README incomplete vs registry; no single Gemma status owner across #672/#667/#678/#614/#270.
3. Issue hygiene: 88 open; ~50+ stale >30d (#30–#560 band). #683 undoc EP generate non-route. #681 G1 DeviceMesh merged 2026-09-02; multi-gpu.md may lag. #648 overlaps #682 G2.
4. PR garden 14 open: #682 G2 MERGEABLE hot; #680 ornith alias; #679 hw-gate MERGEABLE rewrites VALIDATION/CONTRIBUTING (doc ownership conflict); #677/#675 draft CONFLICTING historical post-revert do-not-merge; #670 maple CONFLICTING large; #667 gemma draft CONFLICTING; #652 dflash draft stale fixes #640; #643 Windows VMM draft stale fixes #635; #595 UTF-8 draft base beta CONFLICTING stale touches old examples/daemon; #593 tool_calls draft base beta; #563 DS4 paging draft base beta CONFLICTING partial supersession; #527 mesh megabranch superseded forensic. Same-surface #682/#683: #682 #670 #667 #595 #563 #677 #675; #679 docs-only.

## Would change (ranked, cost)

1. hours — Fix dead entrypoints: CONTRIBUTING+GETTING_STARTED+VALIDATION → cargo build -p hipfire-daemon; drop/replace test_kernels or mark kernel channel blocked; .agents/skills/; replace asks with #666/#683/#669/#645.
2. hours — MODELS.md:148 mq2r default; README drop fake 61/77 or generate from models.json; add maple/bonsai/ornith or mark partial.
3. hours — force_local: CLI.md sole owner match main.rs:1911-1918; ARCHITECTURE link not invent --json/--no-stream.
4. hours-day — INDEX refresh master pin; list missing pages; checkpoints=measured evidence; BENCHMARKS=historical only; fix perf-checkpoints README.
5. day — ARCHITECTURE carriers Gemma/Muse/Maple; document #683 EP generate known non-route; sync #681 DeviceMesh if public.
6. day — Issue triage: close #155 announcement; stale label; collapse dups #537↔#646 multi-slot, #623↔#669 GPU, #162↔#645 NixOS, #270↔#672↔#614↔#678↔#667 Gemma, #640↔#652↔#459 DFlash, #448↔#563 pager, #50↔#353 gfx1152.
7. days — Docs drift CI: forbid dead --example and .skills/ links; INDEX completeness; optional models.json badge gen.
8. week+ — PR garden: close #527/#675/#677 historical; rebase/close beta #593/#595/#563; sequence #682 then mesh G3/G5 per #666; land #679 with VALIDATION ownership handoff.

## Open issues table

| # | Cluster | Age | Status | Recommendation |
|---|---|---|---|---|
| 683 | bug/EP-MoE generate | hours | open hot | P0 document+fix archetype |
| 678 | bug/gemma4 | ~1d | open | triage #614/#667/#672 |
| 672 | feat/gemma publish | ~1d | open | keep block artifact |
| 669 | bug/GPU select | ~1d | open | dup-check #623 |
| 666 | tracking/device-mesh | active | open | sole mesh authority post-revert |
| 655 | bug/Windows dump | ~1d | open | split actionable |
| 651 | feat/quant PARO | ~6d | open | backlog |
| 650 | research/KV | ~6d | open | research |
| 649 | research/redline | ~6d | open | research |
| 648 | bug/EP constructors | ~6d | open | align #682 G2 |
| 647 | roadmap/adaptive KV | ~6d | open | roadmap |
| 646 | feat/multi-slot serve | ~6d | open | merge #537 |
| 645 | bug/NixOS examples | ~7d | open | same CONTRIBUTING rot |
| 644 | bug/MQ4V2 decode | ~7d | open | P0 correctness |
| 642 | feat/IQ3_S | ~7d | open | backlog |
| 640 | bug/DFlash leak | ~6d | open | PR #652 |
| 639 | bug/Windows paths | ~9d | open | Windows cluster |
| 635 | bug/Windows VMM | ~7d | open | PR #643 |
| 623 | bug/Windows iGPU | ~11d | open | cluster #669 |
| 614 | bug/gemma hipGraph | ~10d | open | gemma cluster |
| 605 | RFC/discovery API | ~13d | open | RFC |
| 604 | RFC/response_format | ~13d | open | RFC |
| 588 | bug/pp=2 exactness | ~18d | open | multi-GPU |
| 587 | bug/redline GC12 | ~17d | open | needs-triage |
| 577 | bug/physical_cap | ~22d | open | long-ctx |
| 569 | bug/redline-rocr | ~27d | open | redline |
| 568 | bug/RefCell | ~27d | open | runtime |
| 560 | call/regression testers | ~1mo | open | community |
| 558 | feat/gfx1030 branch | ~1mo | open | stale? |
| 540 | feat/tokenizer | ~1mo | open | backlog |
| 537 | feat/multi-slot | ~27d | open | dup #646 |
| 533 | feat/MQ2 prefill | ~1mo | open | backlog |
| 526 | fix/CLI extensions | ~1mo | open | check fixed |
| 499 | bench/W7900 | ~1mo | open | accept or close |
| 491 | bug/instrument | ~1mo | open | tool |
| 490 | bug/instrument | ~2mo | open | stale |
| 486 | bug/DFlash OOM | ~27d | open | spec-decode |
| 478 | bug/gfx1030 default | ~1mo | open | RDNA2 |
| 475 | bench/gfx1201 | ~2mo | open | stale |
| 469 | feat/DFlash trainer | ~2mo | open | research |
| 462 | bug/DeltaNet serve | ~2mo | open | check post-revert |
| 461 | bug/llama dir dispatch | ~2mo | open | verify |
| 459 | bug/DFlash draft load | ~2mo | open | cluster #640 |
| 456 | tracking/MoE lattice | ~2mo | open | research |
| 448 | tracking/weight pager | ~2mo | open | #563 |
| 443 | proposal/AUR | ~2mo | open | community |
| 433 | question/status | ~2mo | open | answer+close |
| 392 | feat/AWQ+GPTQ | ~2mo | open | relabel if shipped |
| 354 | research/HBW-KV | ~3mo | open | stale |
| 353 | bench/gfx1152 | ~3mo | open | pairs #50 |
| 346 | feat/Orthus MTP | ~3mo | open | backlog |
| 345 | question/Nemotron | ~2mo | open | answer+close |
| 344 | research/FeatherOps | ~3mo | open | stale |
| 343 | roadmap/AWQ trunk | ~3mo | open | partial ship? |
| 341 | research/AWQ calib | ~3mo | open | research |
| 328 | refactor/rdna-compute | ~3mo | open | chore |
| 305 | governance/dual-license | ~13d touch | open | governance |
| 301 | RDNA1 follow-ups | ~3mo | open | stale |
| 289 | MQ3 follow-ups | ~3mo | open | partial obsolete |
| 272 | tracking/DDTree+CASK | ~3mo | open | research |
| 271 | tracking/ZAYA1 | ~3mo | open | hold |
| 270 | tracking/Gemma4 hold | ~2mo | open | supersede #672 cluster |
| 252 | bug?/Windows perf | ~3mo | open | needs repro |
| 223 | bug/string garbage | ~3mo | open | may #595 |
| 217 | bug/missing Drop | ~3mo | open | leak w/ #640 |
| 213 | bug/attractor 27B | ~3mo | open | quality |
| 209 | feat/MQ3 MoE prefill | ~3mo | open | backlog |
| 207 | feat/gfx906 MoE | ~3mo | open | backlog |
| 188 | research/mq3-lloyd | ~3mo | open | research |
| 162 | bug/NixOS install | ~3mo | open | cluster #645 |
| 155 | meta/modularization shipped | ~4mo | open | CLOSE announcement |
| 116 | research/Lloyd-MQ3 gates | ~3mo | open | stale vs V2 |
| 114 | research/MQ3 sub-9B | ~4mo | open | registry still cites |
| 113 | research/MQ3 ppl | ~4mo | open | same |
| 105 | question/CPU+GPU | ~3mo | open | answer+close |
| 92 | feat/MoE DFlash drafts | ~3mo | open | backlog |
| 89 | bug/DFlash A3B attractor | ~4mo | open | help wanted |
| 78 | feat/sliding FA | ~4mo | open | backlog |
| 77 | design/NVMe paging | ~3mo | open | design |
| 76 | design/3-tier KV | ~2mo | open | design |
| 61 | feat/gfx1151 baselines | ~1mo | open | relevant |
| 50 | test/gfx1152 | long | open | keep; rewrite CONTRIBUTING |
| 45 | roadmap/living index | ~4mo | open | refresh or close |
| 43 | roadmap/SSM persist | ~4mo | open | stale |
| 42 | roadmap/mutable hipGraph | ~3mo | open | backlog |
| 39 | research/custom DFlash train | ~4mo | open | research |
| 31 | follow-ups/PR28 | ~4mo | open | stale |
| 30 | bug/gemm_qkvza multiblock | ~3mo | open | kernels |

Stale >30d low traffic: majority #30–#560; bulk stale label + close #155 and answered questions #433 #105 #345.

## Confidence

Did not re-read all docs/** (~1495). Sampled top-level owners INDEX VALIDATION BENCHMARKS MODELS ARCHITECTURE CLI GETTING_STARTED perf-checkpoints README admissions CONTRIBUTING README AGENTS CLAUDE models.json force_local daemon Cargo CI. Did not verify every issue body for already-fixed-on-master. Did not deep-diff #681 vs multi-gpu.md. PR mergeable at fetch time. Model count ~79 structural scan not jq length. JSON summary fields: broken 9 verified; missing 5; changes 8 ranked hours→week+.
