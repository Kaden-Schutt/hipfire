# 2026-08-15 — AMENDMENT: the 3.6-27B DFlash chain checkpoint used a non-pinned trunk

**Lifecycle: `historical`.** Amends
[`2026-08-15-dflash-dense-3.6-27b-chain-decay.md`](2026-08-15-dflash-dense-3.6-27b-chain-decay.md),
which remains **unchanged** per this directory's append-only rule.

## What is wrong with the original

The trunk was selected **by local filename**, which
[`AGENTS.md`](../../AGENTS.md) § "Pinned Hugging Face bench fixture" explicitly
forbids: *"do not identify the canonical trunk by local filename. Local
filenames drift and lookalike AWQ/MQ4 files are not comparable."*

| | bytes |
|---|---:|
| pinned canonical `qwen3.6-27b.mq4` | **14,984,158,208** |
| artifact actually used on `hiptrx` | **14,979,312,640** |
| difference | −4,845,568 |

The digests of what was actually run, recorded so the artifacts are
identifiable later:

```
trunk  70dcd063a493af20a519e3afd0f341910b97bfd1af76aba45fe4742aed14fd15  qwen3.6-27b.mq4
draft  bd8c4f07ae80fe1385bf2606af9a7ba0daa18ca8daec50916f2a489054c44e70  qwen36-27b-dflash-mq4.hfq
```

The trunk digest `70dcd063…` is **not** the pinned `86a5f80f…82442dc`. **No artifact on `hiptrx` matches the pinned
size at all** — the nearest others are `qwen3.6-27b.mq4-q8conv1d-flat` and its
gptq sibling at 14,980,357,120, also non-matching. The canonical trunk is not
present on that host.

AGENTS.md's disposition for this case is unambiguous: *"Reports that use a
trunk with a different digest are not comparable and should be discarded."*

## What survives and what does not

**Does not survive** — any comparison against:

- the pinned dense-3.6 AWQ MTP/DFlash line,
- the τ 4.4–6.1 @ ctx 20695 figure in `dflash_spec.rs:142`,
- the registry's own `qwen3.6:27b` claim of **"44 tok/s AR / 185 tok/s w/ draft
  on code"**. That claim is 4.7× the DFlash throughput measured, and the gap is
  now unattributable: it could be the trunk, the draft, the card, the genre, or
  a stale desc. The original checkpoint should not be read as evidence against
  it.

**Survives, with the caveat stated** — the *internal* A/B. Both arms ran the
same trunk, same draft, same fixture, same config, back to back on one host, so
the relative comparison is self-consistent:

- DFlash beat AR on exactly one of eight turns.
- AR was flat in context; DFlash fell 49.5% from its own peak.
- τ did not decay (`r = +0.278`), and the highest-τ turn (2.83) was still
  33.8% slower than AR.

That last point — acceptance flat while throughput collapses — is a statement
about *this* pair on *this* host, and it is the observation that motivated
looking at drafter cost rather than draft quality. It is a hypothesis-shaping
result, **not** an admissible perf claim.

## What was correct

Sampling **was** registry-resolved, confirmed in the run header:
`temperature 1.0, top_p 0.95, top_k 20, min_p 0.0` — byte-matching
`registry/models.json` → `qwen3.6:27b.recommended_settings`. The fixture was
the committed `benchmarks/prompts/session_coding.json`
(md5 `c0d470288bde3f1e54e4bba04da8f8a2`). The harness was the user-facing
`serve_harness.py --mode session`, per `docs/VALIDATION.md`.

The defect is isolated to trunk identity.

## Required before any re-run is quotable

1. Obtain the canonical trunk — `hipfire-models/qwen3.6-27b`, file
   `qwen3.6-27b.mq4`, HF commit `f9b326a657f14cbc400e384ff84a4b9b4b726ba2`,
   size `14984158208`, SHA-256 `86a5f80f…82442dc` — or refresh the HF headers
   and re-pin if HF has published a newer artifact.
2. `sha256sum` the candidate and require the digest **before** running.
3. Pin the draft the same way. The draft used here
   (`qwen36-27b-dflash-mq4.hfq`, 919,401,472 B) has an md5 recorded in
   AGENTS.md § "Verify md5s after pull" (`204c4c4c…`) that was **not** checked
   either.

## Process note

Both the trunk and the draft were taken on filename faith. The repo already
carries the machinery to prevent this — a pinned digest in AGENTS.md, sizes and
`sha256` in `registry/models.json`, and an md5 table — and none of it was
consulted before running. Prefer resolving through the registry tag
(`--tag qwen3.6:27b`) over a raw `--model` path, so identity is checked rather
than assumed.
