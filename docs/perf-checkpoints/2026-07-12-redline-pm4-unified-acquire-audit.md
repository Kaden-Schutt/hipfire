# Retained PM4 unified-acquire and lazy-barrier audit

## Verdict

RADV methods #4/#14 (unified `ACQUIRE_MEM`) and #1/#50 (lazy barriers) are
already structural invariants of the retained single-stream tape. There is no
second per-hazard cache packet stream to combine, so these items require no code
change or performance A/B.

## Unified acquire (#4/#14)

The tape builder makes one Boolean cache-boundary decision between adjacent
nodes. A true decision calls exactly one acquire emitter, which writes one
eight-dword `ACQUIRE_MEM` containing the complete GCR action word. Cache actions
are bits accumulated in that one `GCR_CNTL`; there is no loop over resource
hazards and no path that emits separate invalidate/writeback packets for one
boundary.

For the stable 833-dispatch A3B capture, `required-only` produces:

- one ownership acquire at tape entry;
- 80 required inter-node acquires around repeat/rope boundaries;
- zero boundaries with more than one acquire.

That is 81 `ACQUIRE_MEM` packets total, already the unified form. The gfx12 GCR
trim changes the action word (`0x1c1d1` at entry, `0x10180` inter-node), not the
packet count.

## Lazy barriers (#1/#50)

Compute-idle waits are emitted only when the allocation-wide resource frontier
finds a read/write or write/write conflict. On the same capture, the audit sees
832 boundaries and proves 130 independent. Those 130 boundaries emit neither a
compute-idle wait nor an acquire unless the separate, explicit cache-boundary
policy requires one. A single terminal idle remains before the tape completes.

Unknown kernel ABIs, unresolved pointers, and incomplete resource coverage fail
closed by retaining the wait.

## Consequence

Adding an accumulator abstraction here would encode the same one-word decision
and leave the emitted tape byte-identical. The useful cache-action change was
therefore the separately measured gfx12 GCR trim; unified acquisition and lazy
barrier emission are closed as already implemented.
