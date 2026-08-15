# DS4 DSpark LocalMaxxing k6/k4 production comparison

Date: 2026-08-06  
Branch: `ds4-beta-staging`  
Host/device: `hipx`, Radeon 8060S, `gfx1151`, ROCm 7.14

> **Superseded as the k6 baseline (2026-08-08).** The 37.3165 tok/s figure below
> remains the correct golden *for the commit it was measured at*, and this
> document's k6/k4 comparison stands. It is no longer the number to benchmark
> against: the tiled LDS top-K gather (`3bc2b47ee`) moved the shipping k6 path
> to **38.97192 tok/s** median. See
> [`2026-08-08-ds4-gfx1151-decode-roofline.md`](2026-08-08-ds4-gfx1151-decode-roofline.md)
> § "Current k6 golden" for the current reference, and § 6 of that document for
> the verify cost model, which corrects the assumption that decode throughput
> scales with tau.

This comparison reproduces the shipping top-k6 DSpark golden three times and
then measures the same production serve path three times with runtime routed
expert fanout set to four. It supersedes the earlier `dspark_bench` k4 smoke
for product performance: that direct benchmark framed the prompt as 24 tokens,
whereas the serving fixture frames the committed prompt as 25 tokens.

## Fixture

- Target: `deepseek-v4-flash-0731.mq2r`, preserved P3 artifact
- Model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- DSpark sidecar: `deepseek-v4-flash-0731-dspark.mq2r`
- Sidecar SHA-256:
  `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e`
- Tool: `scripts/serve_harness.py`, production CLI and daemon lifecycle
- Prompt: entry 0 of `benchmarks/prompts/ds4_dspark_genre_code.json`
- Prompt MD5: `d782138f5bc8bbbd234ca8e4b17cace9`
- 25 prompt tokens, 128 generated tokens, batch 1, greedy
- Q8 KV request, contiguous KV backend
- DSpark direct HIP, adaptive B, sidecar maximum block 5
- Thinking, MTP, and DFlash off
- Three fresh processes per expert-fanout arm

Top-k6 is the shipping checkpoint default. Top-k4 is a runtime model-load
choice over the same weights, not a separately baked artifact and not a change
to the shipping default.

## Results

| Routed experts/token | Samples (tok/s) | Median | Range / median | Tau | Acceptance | Windows |
|---:|---|---:|---:|---:|---:|---:|
| 6 (shipping) | 37.3010 / 37.3214 / 37.3165 | **37.3165** | 0.055% | 2.02381 | 67% | 42 |
| 4 (matched config) | 39.1688 / 39.1808 / 39.1907 | **39.1808** | 0.056% | 1.64583 | 55% | 48 |

The k4 median is +1.8643 tok/s, or +4.996%, over the reproduced shipping k6
median. The throughput gain survives a lower proposal-acceptance rate because
the target moves two fewer routed experts per token.

Decoded output was byte-identical within each arm:

- k6 MD5: `e49b9893a207d8a698eb17fdca13db51`
- k4 MD5: `0f6363c0da1396377ba85881e9061c3b`

The arms are not expected to be byte-identical to one another because changing
expert fanout changes target arithmetic and is a quality/configuration choice.
All six outputs were coherent Python merge-function completions and stopped at
the deliberate 128-token cap. The harness's `runaway` label denotes that cap
hit; no accepted row was empty or an attractor.

## Production k4 plumbing

Commit `95297149e6c1f9ed18a2487c668ccd152f4e7489` adds the typed nullable config
field `model.deepseek4_experts_per_token`. Null preserves the checkpoint
default. Values are schema-bounded to 1 through 6, forwarded in the daemon load
request, and applied only by `Deepseek4Carrier`; Qwen carriers and Qwen-owned
function bodies are unchanged.

The k4 logs prove the selected route for every process:

- `GPU dev 0: gfx1151 (103.1 GB VRAM, HIP 7.14)`
- `deepseek4: runtime experts-per-token override 6 -> 4`
- P3 MQ2R artifact verification
- P3-aligned DSpark sidecar identity verification
- `deepseek4 DSpark speculator enabled (sidecar, block=5)`
- request-level `drafter=dspark`

Both accepted arms used these exact binaries:

- CLI SHA-256:
  `abc1489026b449ed052954f32b9bed09a2ef896df27acf4b64784c86ded683ab`
- Daemon SHA-256:
  `bade329fe6799bae317a5981a82fa8a8866188c2deb66482401ac1641262b798`

Validation before the GPU runs:

- `cargo test -p hipfire-config`: 52 passed
- focused `hipfire-cli` load-parameter test: passed
- `cargo check -p hipfire-loader -p hipfire-cli`: passed
- `cargo check -p hipfire-runtime --example daemon`: passed
- `scripts/serve_harness.py --self-test`: passed

## Excluded but preserved

- The earlier `dspark_bench` k4 smoke and interrupted repetitions are not
  product-serving evidence because that benchmark reported a 24-token prompt
  and bypassed production serve/load behavior.
- The first valid production k6 trio at the immediate parent commit remains a
  historical recovery (37.3181 tok/s median), but the primary table uses the
  subsequent `k6-matched-run1..3` trio on the exact k4 binary hashes. This
  removes the otherwise inert optional-selector source delta from the A/B.
- `k4-production-run1.*` is excluded because singular `--prompt-file` treated
  the JSON fixture itself as a 44-token prose prompt. It measured 32.2269 tok/s
  and is retained solely as an audit of the fixture error.
- The accepted runs use plural `--prompts-file`, selecting the embedded code
  row and reproducing the historical 25-token framing.

## Evidence

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-06-localmaxxing/`

Accepted result JSON SHA-256 values:

- k6: `fb5d9a4b474dbadb5d7496caf19c00bda51297f9f8dc4f4a13150fdc1d8bfc9e`,
  `6a0a96a62fe32ce20d27f8e8e4b138d04582c092ba40dbf47d5bfac6290985a6`,
  `d91af249916c841ce02a5821721bf1e6649b777a291af32433d661b41e1c3c36`
- k4: `5f8f8d9460bfc00b03ceac97cab598bd176ea1ef88f63d9f03089136596b01f8`,
  `ab7f3a13305878051a1fc6ee057b28f47c5da7aac1a5e12e3217b849610c2325`,
  `28ec3391cabc1e0e1893e0ad498b8374e9ca0418fcf3fc87eee95557c6bc8682`

Skipped: no AR arm in this comparison; no quality promotion of k4; no weight,
kernel, PM4, KV, sampling, or shipping-default change; no 2,048/512 or
long-context run.

Verdict: the production shipping k6 DSpark golden is recovered at 37.3165 tok/s
median, while the same production fixture at user-selectable k4 reaches 39.1808
tok/s median. Keep k6 as the quality/default row and report k4 separately as the
matched-config performance row.
