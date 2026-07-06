---
title: Measure spec-decode durability on the daemon, not the demo harness
date: 2026-06-23
tags: [spec-decode, methodology, lesson]
---

`dflash_spec_demo` / `mtp_only_demo` measure raw open-ended continuation and
**under-report by ~40%** — they never trigger the chatml + structured-thinking
serving behavior that makes even novel content predictable (τ jumps 1.05 → 2.2+). A
"fails on prose/fiction" verdict from a demo is a **harness artifact, not a product
verdict**.

This cost ~20 cycles: under the demo, DFlash creative-fiction looked retrain-bound
(0.97×/τ1.05) and I nearly chased a multi-day retrain. On the **daemon** (the path
users actually hit) it clears every genre — DFlash 1.40–2.67×, MTP 1.26–1.73×.

Rule: confirm τ / tok-s through the daemon (`serve`) before calling a genre
drafter-bound. Also promoted to `CLAUDE.md` → "Hard-won measurement & debugging
rules". Related: `spec-decode-verify-kernel-ceiling`.
