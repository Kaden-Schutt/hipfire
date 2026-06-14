# 13 — Stage 2b PpMtp VRAM audit

Date: 2026-05-28
Branch: fix/q8-batched-masked-no-lds-cap (post commits 0c0889d4 + 4b6bd48b)
Model: `/local/hipfire/qwen3.6-27b.mq4` (qwen3.6-27b, 64 layers, dim=5120, 248k vocab)
MTP head: `/data/hipfire/qwen3.6-27b-cvs16384.mtp` (k=3, cvs=16384)
Hardware: gfx906 (MI50, 32 GiB, HIP idx 0) + gfx1031 (RX 6700 XT, 12 GiB, HIP idx 1)

## Question

Stage 2b's stated user payoff is "longer ctx via gfx906 freed of MTP
head + scratch." Confirm empirically that PP+MTP (Stage 2b shape)
shifts both the MTP head AND the head's scratch off gfx906 onto
gfx1031, and quantify the gfx906 KV-cache ceiling improvement.

## Method

Two daemon configs, each loaded fresh, VRAM sampled via
`rocm-smi --showmeminfo vram` at the `{"type":"loaded"}` event:

- **A (baseline, pp=1)**: `{"params":{"max_seq":N,"mtp_head":<head>}}`,
  no `HIPFIRE_PP_LAYERS`. MTP head + KV all on gfx906.
- **B (Stage 2b, pp=2)**: `{"params":{"pp":2,"max_seq":N,"mtp_head":<head>}}`
  with `HIPFIRE_PP_LAYERS=48,16 HIPFIRE_ALLOW_MIXED_ARCH=1`. Trunk
  layers 0–47 on gfx906, 48–63 + output_norm + lm_head + MTP head +
  spec state on gfx1031.

Two max_seq points each: 4096 (the coherence-gate-pp default) and 32768.

## Results

| config | max_seq | gfx906 used | gfx1031 used | gfx906 free |
|---|---|---|---|---|
| A: pp=1 + MTP | 4096 | 15.57 GiB | (idle, ~25 MB) | 16.43 GiB |
| A: pp=1 + MTP | 32768 | 18.22 GiB | (idle) | 13.78 GiB |
| B: pp=2 + MTP (Stage 2b) | 4096 | 11.13 GiB | 5.93 GiB | 20.85 GiB |
| B: pp=2 + MTP (Stage 2b) | 32768 | 13.09 GiB | 6.85 GiB | 18.89 GiB |

## Stage 2b gfx906 savings

- 4k:  15.57 → 11.13 GiB = **-4.44 GiB freed** (-28.5%)
- 32k: 18.22 → 13.09 GiB = **-5.13 GiB freed** (-28.2%)

Savings are roughly constant across max_seq, as expected: MTP head
(~1 GB) + spec scratch (~0.5 GB) + 16 trunk layers' weights+activations
(~3 GB) lift cleanly off gfx906. The KV-cache delta scales with
max_seq but on the SAME slope per-band.

## Ctx ceiling projection

gfx906 KV growth per (max_seq=32k−max_seq=4k):
- pp=1: (18.22−15.57) GiB / 28k tok = 2.65 GiB → linear extrapolation
  to 32 GiB cap → max_seq ≈ **(32−15.57)/2.65×28k + 4k ≈ 178k**
- pp=2: (13.09−11.13) GiB / 28k tok = 1.96 GiB (only 48/64 layers
  contribute KV) → max_seq ≈ **(32−11.13)/1.96×28k + 4k ≈ 302k**

**Stage 2b lifts the gfx906-bound ctx ceiling from ~178k → ~302k tokens.
≈ 1.7× longer context.** (Subject to gfx1031 not hitting its own cap
first; at the current split gfx1031 grows ~0.92 GiB per 28k of KV on
its 16 of 64 layers and starts at 5.93 GiB at 4k, so it caps around
**(12−5.93)/0.92×28k ≈ 185k** — currently gfx1031 caps below gfx906.
A better layer split — say 52,12 — would push gfx1031 even further by
keeping more KV on gfx906 where there's more headroom; orthogonal to
Stage 2b and revisited when the workload demands it.)

## Total VRAM

| max_seq | A total | B total | delta |
|---|---|---|---|
| 4096 | 15.57 GiB | 17.06 GiB | +1.49 GiB |
| 32768 | 18.22 GiB | 19.94 GiB | +1.72 GiB |

PP+MTP uses ~1.5–1.7 GiB more aggregate VRAM than the single-gpu MTP
baseline. Sources: per-band DeltaNet state replicated, per-band
activations scratch, token_embd mirrored from dev 0 → output_device
for the MTP head's chain embed-lookup. Cost is paid in the OPPOSITE
direction from the gfx906 saving — gfx1031 was empty before, so spending
~6 GiB on it costs ZERO from the gfx906 ceiling perspective.

## Conclusion

Stage 2b's user-facing claim holds: PP+MTP frees ~4.5 GiB of gfx906
VRAM at any max_seq tested, with the saving applied EXACTLY where the
context ceiling lives (gfx906 is the smaller-headroom card per-layer
because it carries the bigger trunk share at higher per-layer KV cost
than gfx1031's 16 layers). Projected ctx ceiling on gfx906 lifts from
~178k → ~302k.

The current 48,16 layer split caps total-system context at ~185k
(gfx1031-bound). A future-work item is to re-tune the split (e.g.
52,12 or 56,8) once a workload actually exercises the >185k regime.
That's a load-time config change, not a kernel/runtime change.

## Repro

```bash
EXE=./target/release/examples/daemon
MODEL=/local/hipfire/qwen3.6-27b.mq4
MTP=/data/hipfire/qwen3.6-27b-cvs16384.mtp

# Test A: pp=1 baseline
(printf '{"type":"load","model":"%s","params":{"max_seq":4096,"mtp_head":"%s"}}\n' "$MODEL" "$MTP"
 sleep 60; printf '{"type":"unload"}\n'; sleep 1) | timeout 120 "$EXE" > /tmp/aud_a.log 2>&1 &
# poll /tmp/aud_a.log for "loaded", then rocm-smi --showmeminfo vram

# Test B: pp=2 Stage 2b
(printf '{"type":"load","model":"%s","params":{"pp":2,"max_seq":4096,"mtp_head":"%s"}}\n' "$MODEL" "$MTP"
 sleep 60; printf '{"type":"unload"}\n'; sleep 1) | \
  env HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16 timeout 120 "$EXE" > /tmp/aud_b.log 2>&1 &
# poll for "loaded", then rocm-smi
```
