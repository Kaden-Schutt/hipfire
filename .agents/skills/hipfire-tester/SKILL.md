---
name: hipfire-tester
description: Guide a tester through hipfire bring-up, smoke tests, DFlash opt-in checks, MQ format sanity, multi-turn recall, CLI surface checks, and benchmark reporting on AMD RDNA/CDNA GPUs. Use when the user wants a standard test matrix or upstream-ready tester report.
---

# hipfire-tester

Use this skill to produce a tester workflow and upstream-ready report. The
detailed checklist lives in `guide.md`; load it when the user is actively
running bring-up or asks for the full matrix.

## Current Baseline

- The repo-wide hard rules in `AGENTS.md` apply, especially prompt md5
  discipline and `./scripts/coherence-gate-dflash.sh` as the canonical
  correctness gate for kernel/quant/dispatch/fusion/rotation/rmsnorm/spec
  decode changes.
- DFlash is opt-in by config: use `hipfire config set dflash_mode auto` or a
  per-model setting before expecting paired drafts to fire.
- Treat benchmark numbers as reportable only with fresh-process runs, prompt
  md5, binary md5, and decoded-output eyeball checks.

## Output Contract

For tester reports include GPU model, gfx arch, ROCm version, hipfire version,
model and draft filenames, config relevant to DFlash/KV, command lines, prompt
md5, binary md5, and pass/fail notes for coherence and CLI smoke tests.
