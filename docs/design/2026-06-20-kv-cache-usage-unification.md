# KV-cache usage unification — three-layer design

**Status:** design only, not yet implemented. Pick up in a later session.
**Date:** 2026-06-20
**Branch context:** `feature/transparent-loading-all-models` (the branch that
unified KV-cache *instantiation*).

---

## Why this exists

This branch already cleaned up how a KV cache is **created**. There are now
exactly two steps for instantiation:

1. `kv_mode::resolve()` — turns a raw mode string (`"asym3"`, `"q8"`, …) into
   one validated `KvMode` value. It replaced six copy-pasted string-matching
   ladders.
2. `KvCache::from_mode()` — a single dispatcher that turns that `KvMode` into
   the right one of ~30 constructors.

So creation funnels through one clean path. **Using** the cache does not. Once
a cache exists, the rest of the engine has to figure out "what kind of cache is
this?" all over again, in several different places, each with its own copy of
the logic. This document describes how to give *usage* the same single-funnel
cleanliness that creation now has.

### The two leaks we are fixing

The cache records its "kind" as a bag of about eight independent on/off flags
(is it 8-bit? is it 3-bit-rotated? is it FWHT-rotated? is it a legacy llama
format? …). These flags are public, and consumers read them directly. Two
problems follow:

- **Leak 1 — hand-copying.** Four places in the code copy those flags one by
  one into a request object before asking the shared "what kernel do I run"
  helper. If someone adds a new flag, every one of those four places has to be
  updated by hand, and nothing forces it.

- **Leak 2 — bypassing.** Four *other* places (the speculative-decoding and
  long-context-compaction paths) skip the shared helper entirely and re-derive
  the cache's kind from the raw flags with their own private if/else chains.
  These paths genuinely need different GPU kernels, but they all start by
  re-answering the same "what kind of cache is this" question — and they answer
  it in their own slightly-different way each time.

The danger both leaks share: someone adds a new cache kind, updates the
creation path and one or two usage sites, and **silently forgets the others**.
The cache then gets read with the wrong kernel and produces garbage output that
only shows up much later as a quality regression. We want adding a new kind to
be *impossible to half-finish*.

### Hard constraints discovered while exploring

Three facts shape the whole design. They are not negotiable:

1. **Crate boundary.** The cache type lives in one crate (`hipfire-runtime`);
   the request/plan types live in a lower crate (`hipfire-dispatch`) that, by
   design, does **not** depend on the runtime crate. This means the obvious
   "automatic conversion" trick (a Rust `From` impl) is structurally
   impossible. The conversion has to live as a plain method on the cache.

2. **The flags carry more than one decision.** The bypass paths don't just pick
   a kernel — they also use the flags to assert memory-layout preconditions and
   to compute byte sizes. So whatever single representation we introduce must
   preserve *every* distinction any consumer currently cares about (in
   particular the "rotated with FWHT vs rotated with Givens" distinction). If
   we flatten that away, we trade a loud compile error for a silent wrong
   answer.

3. **Not everything is fixed for the cache's lifetime.** Most of the cache's
   "kind" never changes once it's created. But three pieces of the final
   decision genuinely depend on the current token step: whether this is a
   boundary layer, whether we're in the high-precision "flash" regime yet, and
   whether this is a batched/tree step. So we can pre-compute the *stable* part
   once, but the final assembly has to happen per step.

---

## The shape of the solution

Three layers. Each is independently shippable, each has its own pass/fail
test, and they stack: you can stop after Layer 1 and still have a real
improvement, or go all the way to Layer 3 for the full mirror of the creation
path.

```
Layer 1  →  one accessor on the cache         (fixes hand-copying)
Layer 2  →  one sealed "what kind" decision    (fixes bypassing)
Layer 3  →  pre-compute the stable part        (speed + clean pipeline hook)
```

Recommended order is 1 → 2 → 3, because each builds on the last and the early
ones are the lowest-risk.

---

## Layer 1 — one accessor (fixes the hand-copying leak)

**Idea in one line:** give the cache a single method that produces the request
object, so the four hand-copy sites stop copying flags one by one.

**What changes.** Today each of the four sites writes out "flag A = cache.flag
A, flag B = cache.flag B, …" by hand, then fills in the per-step fields, then
asks the shared helper. After this layer, each site instead says "give me the
request object for this cache" (one call), then fills in only the per-step
fields. All the flag-reading lives in exactly one place. We also stop exposing
the raw flags publicly, so nobody *can* hand-copy them anymore.

**Why it's safe but not trivial.** Two of the four sites today pass some flags
as hardcoded "off" (because that arch doesn't support those modes). The new
accessor reads the real flags instead of those hardcoded values. That's only a
true no-op if those flags are *guaranteed* off for that arch's cache. So the
work isn't "write the method" — it's "prove the method produces byte-for-byte
the same request the hand-written code produced, for every arch." There's also
a visibility wrinkle: a couple of the bypass paths live in separate arch crates
and currently read the flags across the crate boundary, so "make the flags
fully private" isn't possible without giving those readers a deliberate, narrow
accessor instead.

**The discipline that makes it trustworthy.** Before swapping any caller, write
a test that asserts the new accessor's output equals the current hand-copied
values, site by site. Make that test green *first*. Then the caller swap is a
verified no-op rather than a hopeful one. This matters because a subtle
difference here routes a different kernel and shows up as a quality regression,
not a crash.

**What "done" looks like.** The four hand-copy sites each shrink to one accessor
call plus the per-step fields. The raw flags are no longer public. A test pins
the accessor to the old behavior.

**Open choice for later (don't decide now):** whether the request object stays
one flat struct (smallest change) or splits into a "cache kind" part and a
"this step" part (stronger guarantee that you can't forget a per-step field,
but a bigger change touching the shared helper's signature). Layer 3 leans
toward the split, so if Layer 3 is likely, do the split here.

---

## Layer 2 — one sealed decision (fixes the bypassing leak)

**Idea in one line:** make "what kind of cache is this" a single function that
only it can answer, and require the bypass paths to take that answer instead of
re-deriving it.

**What changes.** Today the four bypass paths each have their own if/else chain
over the raw flags. After this layer there is one shared function that turns a
cache into a single "kind" value (a closed list of named possibilities). The
raw flags become private, so the hand-rolled if/else chains *won't compile
anymore* — they have to call the shared function. The bypass paths keep their
own special kernels; they just stop re-answering the kind question their own
way.

**The catch that drives the design.** The bypass paths use the flags for more
than picking a kernel — one asserts a memory-layout precondition, another
computes byte sizes per position. So the single "kind" value must keep every
distinction those paths rely on, especially the "FWHT-rotated vs Givens-rotated"
difference. If the kind value blurs that, the precondition check silently goes
weak and a later step mishandles the layout. The rule: the kind value must
preserve every distinction *some* consumer branches on, and ideally expose
those as named questions ("is this layout valid for the long-context path?")
rather than making callers re-inspect internals.

**Handling the "different paths need different kernels" reality.** The bypass
paths exist because speculative decoding and compaction need different kernels
than ordinary decoding. So the shared helper grows a notion of *which phase is
asking* — ordinary decode, prefill, speculative-tree, or compaction — and hands
back the correct kernel set for that phase. This is what lets the bypass paths
rejoin the funnel without losing their specialness. It also turns today's
implicit "is this a batched/tree step" flag-juggling into an explicit,
compiler-checked set of cases.

**The discipline that makes it trustworthy.** Do it as a pure extraction first:
pull the flag-reading half out of the shared helper into its own function, have
the helper call it, and confirm the existing tests still pass with no behavior
change. Only *then* point the bypass paths at it and tighten the flag
visibility. The first commit should change nothing observable.

**What "done" looks like.** There is exactly one function that reads the cache's
flags to decide its kind. The four bypass paths call it instead of re-deriving.
A new cache kind is one new entry in one list, and every consumer picks it up
automatically (or fails to compile, which is the point).

---

## Layer 3 — pre-compute the stable part (speed + clean pipeline hook)

**Idea in one line:** since most of the "kind" decision never changes after the
cache is created, compute that part once at creation and stamp it into the
cache, leaving only the genuinely per-step bits to assemble during inference.

**What changes.** Today the full kind-to-kernel decision runs on every single
token, even though almost all of its inputs are fixed for the cache's whole
life. After this layer, creation (`from_mode`, the same place that already sets
the cache's kind) also computes and stores the stable part of the plan. During
inference, each step does a cheap final assembly: take the stored stable part,
apply the few per-step adjustments, get the final plan. The expensive
flag-reading never runs in the hot loop. This is the direct mirror of the
creation path: the constructor sets up the plan, and a matching method consumes
it.

**The pipeline bonus.** The dispatch pipeline already has a "list of steps"
pattern where one step runs attention from a prepared plan. With the stable part
stored on the cache, that step can become "ask the cache for its attention
step" — the cache hands back a ready-to-run unit that already knows its kind and
only needs the current step's context. This collapses the remaining per-call
plan-building boilerplate and makes the cache the single source of truth for its
own attention, fitting the existing pipeline pattern without bending it.

**The catch that drives the design.** The stored part is *not* a finished plan.
Three things still depend on the current step: whether this is a boundary layer,
whether we've crossed into the high-precision flash regime, and whether this is
a batched/tree step (which can even be an unsupported combination that must be
refused). The split between "stable, stored at creation" and "per-step, applied
during inference" has to be exactly right. Get it wrong — say, store a decision
that actually depends on the step — and you ship a mismatch between the write
and read kernels that the existing internal consistency check catches in debug
builds but could let through in release. There's also a runtime wrinkle: one
operation can change the cache's value-precision mode mid-life, so if that
happens the stored stable part must be re-computed through the same single
helper, never patched ad-hoc.

**The discipline that makes it trustworthy.** Mechanically split the existing
shared helper into two halves — "decide the stable part from the flags" and
"finish the plan with the per-step inputs" — keep the original as a thin wrapper
that calls both, and prove the kernel choices are byte-identical across every
combination before storing anything on the cache. Write down, as a two-column
table, exactly which fields are stable and which are per-step; make that table a
test so future fields have to declare which side they're on.

**What "done" looks like.** The per-token hot path no longer re-reads the cache
flags. Creation stamps the stable plan; a matching method finishes it per step.
The pipeline can ask the cache for a ready-to-run attention step. The
stable/per-step split is pinned by a test.

---

## How the three relate

- **Layer 1** is the symmetric accessor — the smallest thing that makes usage
  look like creation. Ship it alone if that's all you want.
- **Layer 2** makes drift *uncompilable* — it's the part that actually prevents
  the "forgot to update a site" failure.
- **Layer 3** is the performance + pipeline-elegance payoff and the closest
  mirror of `from_mode`.

They're meant to land in order, each behind its own green test, each a no-op
refactor at the moment it lands. If a future session only has appetite for one,
Layer 1 is the right one to take; Layer 2 is the one with the most correctness
value; Layer 3 is the nicest to read but the most delicate to get right.

## A provocation worth ten minutes before starting

Every layer above treats the **cache** as the thing consumers query. There's a
more radical inversion: make the **plan** the object that owns both writing to
and reading from the cache, with the cache demoted to plain storage the plan
operates on. Then "what kind is this" stops being a question anyone asks at all
— it's baked into a plan value minted exactly once at creation, and the four
bypass paths become four methods on that one plan type instead of four funnels.
This might collapse all three layers into a single type — or the per-step fields
might make it untenable. Worth a short spike to find out before committing to
the three-layer path, but **not** worth blocking on.

## Validation note (applies to every layer)

This whole area feeds the attention kernels, so any change here is exactly the
kind the coherence gate exists to protect. Every layer's landing must pass the
coherence gate, and the "prove it's a no-op first" discipline in each layer is
there specifically because a subtle wrong turn here produces a quality
regression, not a crash — the most expensive kind of bug to find later.
