---
title: A3B retained PM4 has no suballocation-only wait candidates
date: 2026-07-12
tags: [redline,pm4,hazards,suballocation,a3b,gfx1201,negative-result]
---

The 833-launch FWHT3 A3B tape on hiptrx reports 832/832 pointer-covered
boundaries, 130 resource-independent edges, and zero `suballocation_candidates`.
The census preserves each pointer start inside its HIP allocation and asks
whether any allocation-wide conflict would disappear under exact-start
identity. None do: all 702 retained inter-dispatch waits include a real
read/write or write/write dependency at the same device pointer.

Do not build a kernel-specific byte-extent catalog for this tape; it cannot
remove a current wait. Keep allocation-wide scheduling fail-closed and retain
the exact-start map only as a diagnostic for future sequences. The 15-position
PM4 gate remained exact with 833 launches, 27 kernels, and sequence hash
`6f56f88512659cba`.
