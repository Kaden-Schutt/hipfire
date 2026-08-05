# DS4 gfx1151 cooperative E8 prefill staging

Date: 2026-08-05 UTC  
Branch: `ds4-beta-staging`  
Probe commit: `0fe09b84f0ccd65eeed4aa3af4d28729ccb9ea53`  
Product-route commit: `7e3ae27fd886326a73addd58b7eb77cc2a3fa4cf`

## Result

DeepSeek V4 Flash 0731 MQ2R now shares each decoded 16-row by 128-K
MFP4G32E8SOA weight slab across four wave32 consumers during gfx1151 cold
prefill. The established B4 kernel decoded the same slab independently in
each wave. The cooperative kernel retains the B4 arithmetic order within each
wave and changes only where the decoded F16 weight tile is produced.

The product selector is deliberately narrow. It is default-on only for exact
gfx1151, `B=1024`, and the four shapes that cleared the isolated screen:

- `(M=32768, K=1024)`;
- `(M=4096, K=8192)`;
- `(M=2048, K=4096)`;
- `(M=4096, K=2048)`.

The flat `(M=1024, K=4096)` projection, prompt tails, other batch sizes,
Qwen, gfx1100, and every non-gfx1151 architecture retain their prior route.

## End-to-end prefill

The 2K screen used `hipfire bench`'s native synthetic PP/TG matrix. Each arm
was a fresh process with one warmup, automatic clocks, a ten-second DPM
stabilization, speculation and thinking disabled, and the same preserved CLI,
daemon, artifact, and gfx1151 device. Decode `tg1@128` was retained only as a
control; this kernel is unreachable from decode.

| Fixture | Previous route | Cooperative E8 | Delta |
|---|---:|---:|---:|
| PP 2,048, n=3 fresh processes | 220.70 / 220.40 / 220.00 tok/s | 233.30 / 233.60 / 233.50 tok/s | median **+5.94%** |
| 21,349-token NIAH | 110,182 ms / 193.76 tok/s | 105,648 ms / **202.08 tok/s** | **+4.29%** |
| 85,693-token NIAH | 568,628 ms / 150.70 tok/s | 549,637 ms / **155.91 tok/s** | **+3.46%** |

The three paired 2K deltas were +5.71%, +5.99%, and +6.14%. Candidate spread
was 0.13%; baseline spread was 0.32%. The decode controls remained 27.22,
27.22, and 27.28 tok/s on the candidate, versus 27.22 tok/s on all three
baseline processes within display precision.

The long-context fixtures were the committed `niah_32k.jsonl` and
`niah_128k.jsonl` prompts, MD5 `2e311623a082f6850a45b2ceefee9d9b` and
`1328229814512e36c4743aa3f9df0e33`. Both retained recall 1/1, coherent text,
`finish=stop`, and zero empty, runaway, or attractor failures. Their decoded
answers are byte-identical to the previous promoted controls:

- 21K answer SHA-256 `e020b8bdbe04790ddd1aeb27d7eca787d3f4c303b9998753ed61d0d0557925b1`;
- 85K answer SHA-256 `0d4c212bcfa3829c127a726a63e39d3736b5220a004b663ba7e9c22ce66b5144`.

## Isolated mechanism and resources

The actual-shape ABBA microbench compared all output values as raw F32 bits.
It covered 45,088,768 values with no mismatch.

| Shape | B4 | Cooperative | Speedup |
|---|---:|---:|---:|
| wq_a / wo_a, 1024x4096 | 1763.4 us | 1746.0 us | 1.010x; not routed |
| wq_b, 32768x1024 | 8332.9 us | 5588.6 us | 1.491x |
| wo_b, 4096x8192 | 8567.0 us | 7200.5 us | 1.190x |
| shared up, 2048x4096 | 2443.5 us | 1833.0 us | 1.333x |
| shared down, 4096x2048 | 1835.6 us | 1132.3 us | 1.621x |

Radiowave inspected the actual JIT code objects:

| Resource | Existing B4 | Cooperative E8 |
|---|---:|---:|
| VGPR | 81 | 107 |
| SGPR | 20 | 20 |
| LDS | 0 B | 4096 B |
| VGPR / SGPR spills | 0 / 0 | 0 / 0 |
| private scratch | 0 B | 0 B |
| static wait instructions | 150 | 132 |
| delay-ALU instructions | 137 | 111 |
| global stores | 32 | 8 |

The higher VGPR count is a real residency cost, but the isolated and
end-to-end measurements include that cost and remain positive at every tested
depth. Radiowave classifies both kernels `vmem_only`; the candidate introduces
no hidden scratch traffic. The cooperative HSACO SHA-256 is
`d248eaf168992884d7213e2534e00279c3f92fb13940763f7e66656329414d39`.

## Identity and evidence

Model SHA-256:
`cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.

Preserved baseline binaries:

- CLI SHA-256 `a0dfc56a117da77c7313e105eea73c1db64a176f4b9b912ef78ccf5b287b7ae4`;
- daemon SHA-256 `d609976fd0db1f3b4d5c490083939d49d2e0e750bf9bacb52dab2d9b979d18c1`.

Preserved candidate binaries:

- CLI SHA-256 `dfe0a338cd0f07f8f57d0620bf7711dfd8930afdc0f16c20d72e0e8c785da38f`;
- daemon SHA-256 `df0fbcdc0ab5f0d6c677142bcaea96ce0d6248724ca655cf2a2bf5a100fb52ef`.

All raw evidence is under:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-prefill-e8-coop/`

Important paths:

- `micro-coop4/stderr`;
- `product-2k-{baseline,candidate}{,-2,-3}.log`;
- `model-transfer-21k/{stdout,rows.json,serve.log}`;
- `model-transfer-85k/{stdout,rows.json,serve.log}`;
- `kernel-artifacts/`.

## Scope not claimed

This is a cold-prefill improvement. It does not claim a decode, PM4,
speculation, quantized-KV, sampling, weight, or quality change. The ordinary AR
decode route was not re-profiled because the new symbol is selected only by
the `B=1024` batched prefill dispatcher and is unreachable from the retained
decode tape. Correctness for the changed path is established by raw-bit kernel
parity and byte-identical user-facing outputs at both long-context depths.

## Follow-up: query-width reuse

The first interpretation of `coop16` put sixteen waves in one 512-thread
workgroup. That was the wrong analogue to Qwen's query16 kernel: Qwen uses one
wave to retain sixteen query rows. The 512-thread candidate was raw-bit exact
over 45,088,768 outputs and introduced no spills, but was 1.22x to 5.33x
slower than cooperative-4. It is rejected at commit
`58b90e699dcd231940b2290362f324943c3e4f2c` and was never routed.

The faithful analogue retains eight or sixteen 16-token WMMA accumulators in
one wave. It has no LDS and no cross-wave barrier. The full five-shape screen
at B=1024 was raw-bit exact:

| Shape | Cooperative-4 | B8 | B16 | Selected |
|---|---:|---:|---:|---|
| 1024x4096 | 1747.3 us | 1619.9 us | **1440.2 us** | B16 |
| 32768x1024 | 5591.7 us | 5521.6 us | 6199.8 us | cooperative-4 |
| 4096x8192 | 7400.4 us | **6408.1 us** | 6328.7 us | B8 |
| 2048x4096 | 1798.9 us | 2585.1 us | 2813.6 us | cooperative-4 |
| 4096x2048 | 1171.9 us | 1675.6 us | 1817.2 us | cooperative-4 |

B16 is 21.3% faster than cooperative-4 on 1024x4096. B8 is 15.5% faster on
4096x8192 and is within 1.3% of B16 there while using 125 rather than 213
VGPRs. The product selector therefore routes B16 only for 1024x4096, B8 only
for 4096x8192, and leaves every other shape on its prior route. Both generated
code objects are `vmem_only`, use zero LDS, private scratch, or register
spills, and are exact-gfx1151/B1024-only.

The native 2K product fixture used three fresh candidate processes against the
preserved three-process cooperative-4 baseline:

| Route | Samples | Median |
|---|---|---:|
| cooperative-4 | 233.30 / 233.60 / 233.50 tok/s | 233.50 tok/s |
| B16/B8 bundle | 236.80 / 236.30 / 236.40 tok/s | **236.40 tok/s** |

The median gain is **+1.242%**, and all three candidate samples exceed all
three baseline samples. Candidate spread is 0.212%. Decode remains outside
the changed route; its emitted one-token control was not used as a prefill
claim. This shape bundle is promoted at
`96be792329f8909541ebbbe0829014a7bc4404f8`.

Follow-up evidence is under:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-prefill-e8-wide/`

It contains the raw micro log, all three product logs, both Radiowave reports
and HSACOs, and the exact candidate CLI/daemon binaries.
