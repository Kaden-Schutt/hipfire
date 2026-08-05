# DS4 beta staging golden performance rerun

Date: 2026-08-05  
Branch: `ds4-beta-staging`  
Commit: `8e9d5f5eca828eaa26bd74b4046dd7aeb7f46855`  
Host/device: `hipx`, Radeon 8060S, `gfx1151`, ROCm 7.14

This rerun checks that the accumulated DS4 prefill work still reproduces the
short-code golden decode waterlines established at `743a8d475`. It is a
performance rerun, not a replacement for the 15-position correctness
certification already recorded in
`2026-08-04-ds4-beta-staging-golden-recert.md`.

## Fixture

- Target: `deepseek-v4-flash-0731.mq2r`, preserved P3 artifact
- DSpark sidecar: `deepseek-v4-flash-0731-dspark.mq2r`
- Prompt: `benchmarks/prompts/ds4_dspark_genre_code.json`
- Prompt bytes MD5: `d782138f5bc8bbbd234ca8e4b17cace9`
- 25 prompt tokens, 128 generated tokens, batch 1, greedy
- Q8 KV request, contiguous backend, top-k 6
- Thinking, MTP, and DFlash off
- One accepted fresh process per arm
- DSpark explicitly forced to direct HIP
- AR explicitly admitted to retained PM4 because automatic `.mq2r` admission
  remains a separate open gate

## Results

| Route | Golden | Reproduced | Delta |
|---|---:|---:|---:|
| DSpark direct HIP | 37.3264 tok/s | 37.3101 tok/s | -0.044% |
| AR retained PM4 | 28.8678 tok/s | 28.7640 tok/s | -0.360% |

Both accepted arms emitted byte-identical decoded output, MD5
`e49b9893a207d8a698eb17fdca13db51`.

DSpark retained the expected adaptive behavior: 42 verify windows,
`tau=2.0238095`, and 67% acceptance. Its measured prefill was 353 ms.

The AR route retained the certified identity:

- 2,320 launches
- 32 unique symbols
- 32/32 certified code-object contracts
- zero fallback or unknown launches
- 2,319/2,319 covered boundaries
- sequence hash `c11845041c3101e7`
- one PM4 packet on queue 3, 57,663 dwords
- 126 retained replays for the measured request

## Excluded but preserved

The first DSpark process measured 37.0059 tok/s with the same decoded bytes and
the same `tau`, but its 1,010 ms prefill was a cold-JIT outlier versus the
historical 362 ms and the warm-cache 353 ms. Its JSON and serve log are
preserved in the evidence directory and excluded from the accepted row.

## Binary identity

- CLI SHA-256:
  `3a1291b628427cb00c2651fc682cf4548e22fee785c9fb98497195a0a6f9162c`
- Daemon SHA-256:
  `547bd426819d4583d9dd35531066f62402421a4b4c0dafb7c27cc1d57a31a379`

## Evidence

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-beta-staging-golden-rerun/`

Accepted evidence hashes:

- `dspark-warm-result.json`:
  `09b6d37a62d6fec3c9efafca60b566c97ae0dce075d3f54e5e00155be8b4bd86`
- `dspark-warm-serve.log`:
  `498f85d8a1950ff23ef5ed5025761b7c203c3a4864d892b19f4293098c113db3`
- `ar-pm4-result.json`:
  `f0be07f09682e675036409bdb0e22c84a2e053db7ec25186e5cd4d0c09ab7a3c`
- `ar-pm4-serve.log`:
  `801b69b10048bb214574e38c4298faae31e84a85c484353a52666dc585fe6881`

Skipped: no second 15-position shadow battery because the task was to
remeasure the already-certified performance waterlines and the decode route
identity did not change; no 2,048/512 product run, long-context run, thinking
run, quant change, or automatic `.mq2r` admission change was included.

Verdict: the DSpark and retained-PM4 AR golden waterlines both reproduce on the
current accumulated branch within 0.4%, with byte-identical output and the
unchanged retained route.
