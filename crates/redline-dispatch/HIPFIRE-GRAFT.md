# Hipfire graft provenance and enablement boundary

This crate is copied from the standalone Redline repository at commit
`50f59ca` after local gfx1201 and remote R9700 certification. The low-level
public ROCr ABI provenance is retained in `../redline-rocr/PROVENANCE.md`.

The graft is intentionally default-off. `HIPFIRE_REPLAY_BACKEND` remains
`hip` unless explicitly set to `shadow` or `auto`; neither mode may replace a
HIP launch until warmup shadow validation proves shared-artifact identity,
byte-exact output, intact guards, automatic clocks, GPU timing, and two
independent speedup samples above the configured threshold. Any ABI,
capability, parity, timeout, queue-fault, or cache-poison failure falls back to
HIP for the process.

The source repository's certified results are:

- real token DAG: 1.076x local and 1.059-1.060x R9700 on GPU timestamps;
- expanded independent set: 1.378-1.381x local and 1.265-1.292x R9700 by
  uninstrumented host-total timing, with separately profiled GPU lower bounds
  still above the historical RADV 1.14x result.

No raw PM4/KMD path from Hipfire's older `redline` crate is used by this graft.

This branch does not transparently route `rdna-compute`'s raw `void **` launch
surface to AQL. The central launch hook records kernel/grid/block metadata only;
it cannot safely infer pointer aliasing, read/write intent, allocation lifetime,
or the kernarg ABI. A model adapter must supply those declarations, build a
`CompiledPlan`, run the two shadow certifications, and explicitly install the
prepared plan before `auto` can report itself ready. Until that adapter exists,
all real launches continue through HIP in every mode.
