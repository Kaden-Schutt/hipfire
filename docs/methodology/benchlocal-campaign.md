# BenchLocal nine-pack quality campaign

How hipfire runs and interprets the BenchLocal capability battery. This file
is **methodology and reproduction**, not a gate and not a number store.

| Concern | Owner |
|---|---|
| Authoritative topology, pins, sampling, exclusions | [`tools/benchlocal/manifest.json`](../../tools/benchlocal/manifest.json) |
| Campaign plan / verify / run / Hermes recovery | `scripts/benchlocal_campaign.py` |
| Results tree → summary JSON | `scripts/benchlocal_score.py` |
| Sampling forcing proxy | `tools/benchlocal/sampling_proxy.mjs` |
| Runtime validation routes (mandatory gates) | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Serve-path harness | [`bench-suite.md`](bench-suite.md), `scripts/serve_harness.py` |
| Public-comparison interpretation (NAS) | `/mnt/nas/kaden/hipfire/benchlocal/2026-07-31-quality/PUBLISHED-COMPARISON.md` |

## What this is and is not

**Is:** a capability / quality campaign against the upstream
[BenchLocal](https://github.com/stevibe/BenchLocal) packs. Four model routes ×
nine runnable packs = 165 scenarios per route and 660 canonical attempts per
campaign realization. The 2026-07-31 pair is the reference realization:

| Campaign id | Thinking | Local tree |
|---|---|---|
| `benchlocal-full-20260731T091750Z` | disabled | `~/.cache/benchlocal-full-20260731T091750Z/` |
| `benchlocal-medium-20260731T194729Z` | `reasoning_effort=medium` | `~/.cache/benchlocal-medium-20260731T194729Z/` |

**Is not:** a mandatory validation gate. Project runtime validation remains
`scripts/serve_harness.py` and `scripts/redline_daemon_harness.py` as routed by
[`docs/VALIDATION.md`](../VALIDATION.md). BenchLocal scores do not admit a
model, replace coherence checks, or redefine validation policy.

## Artifact locations

### NAS preservation

Directory: `/mnt/nas/kaden/hipfire/benchlocal/2026-07-31-quality/`

| File | Purpose | SHA-256 |
|---|---|---|
| `MANIFEST.txt` | Archive inventory, campaign ids, summary digests | — |
| `SHA256SUMS` | Checksums for the two tarballs + `PUBLISHED-COMPARISON.md` | — |
| `PUBLISHED-COMPARISON.md` | Public-comparison interpretation and source links | `f874bb284fc62d5f7591afa5d7cb435ae88e9370423afb17c4c90a2c14d58f9c` |
| `benchlocal-canonical-runs.tar.gz` | Complete local campaign trees (packs, results, summaries, privileged Hermes scratch recovered with sudo); 14303 entries | `216face118f3b346eb42824612d3f809158d7d226c80bda29ccf08592efa8618` |
| `hiptrx-scratch-homes.tar.gz` | Remote GPU sandbox homes, binaries, logs, and result dirs from host `hiptrx`; 89 entries | `c8068c569a663d8e5b9db31009862b6360b9bdcf51fa4cdabbe95b37bc987349` |

Verified archived score summaries (from `MANIFEST.txt`):

| Summary | SHA-256 |
|---|---|
| `full-battery-summary.json` (baseline) | `b47890257e39678d71156ed9e49f0e4b7b6df4483a858c72e6c07be85b72eaef` |
| `full-battery-medium-summary.json` | `46c2a94ec88970d7ac5e9eafe14b0a1649e70451ac882805711aad533cc67f49` |

### Live trees

| Path | Contents |
|---|---|
| `~/.cache/benchlocal-full-20260731T091750Z/` | Baseline packs/, results/, `full-battery-summary.json`, `sampling-proxy.mjs` (+ pre-thinkfix backup) |
| `~/.cache/benchlocal-medium-20260731T194729Z/` | Medium packs/, results/, `full-battery-medium-summary.json`, medium proxy |
| `~/benchlocal-*-20260731T*/` on `hiptrx` | Remote sandbox homes (also in `hiptrx-scratch-homes.tar.gz`) |

Campaign roots are always caller-supplied. Do not hardcode `/tmp` as a
canonical artifact root.

## How to run it

Everything is developer-only orchestration. Python is stdlib-only; Node is
required because the upstream packs are Node/TS. The user-facing control plane
stays Rust-only ([`AGENTS.md`](../../AGENTS.md) §0.6).

### Authoritative inputs

- [`tools/benchlocal/manifest.json`](../../tools/benchlocal/manifest.json) —
  provenance, sampling contract, port topology, four routes, nine pinned packs,
  scoring definitions, exclusion globs. Do not fork a second source of truth.
- `tools/benchlocal/sampling_proxy.mjs` — OpenAI-compatible forcing proxy
  (`THINKING_MODE=disabled|medium`).
- `scripts/benchlocal_campaign.py` — campaign driver.
- `scripts/benchlocal_score.py` — results tree → summary JSON.

### Subcommand flow

```text
# 1. Materialize packs at pinned revisions under <campaign-root>/packs/
#    (clone each packs[].repo at packs[].revision; build pack CLIs / images
#    as the pack READMEs require).

# 2. Plan pure runner commands (no network, no Docker, no GPU):
python3 scripts/benchlocal_campaign.py plan \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --route all \
  --thinking disabled \
  --out <ABS_PATH>/plans

# 3. Verify prerequisites against the plan / host (binaries, ports, images,
#    pack checkouts) before spending GPU time:
python3 scripts/benchlocal_campaign.py verify \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH>

# 4. Run the four-route matrix (hiptrx, 4 GPUs, Docker sidecars, isolated
#    HOME per GPU). Starts daemons + sampling proxies as required by
#    route.proxy_scope, then executes pack argv from the plan:
python3 scripts/benchlocal_campaign.py run \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --thinking disabled   # or medium

# 5. When Hermes --all dies mid-pack, recover per scenario and stitch:
python3 scripts/benchlocal_campaign.py recover-hermes \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --route <slug>

# 6. Score the results tree (optionally against the baseline summary):
python3 scripts/benchlocal_score.py \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --baseline <PATH-to-full-battery-summary.json> \
  --out <ABS_PATH>/full-battery-summary.json
```

`plan` without `--out` prints JSON on stdout. `--route <slug>` emits one plan
object; `--route all` emits `{"<slug>": <plan>, ...}`. With `--out DIR` it
writes `DIR/<slug>.runner-commands.json` per route.

### Subset runs and partial scoring

A full route is nine packs and hours of GPU time, and CLI-40 and HermesAgent-20
dominate that cost. Per-pack reruns were routine in the 2026-07-31 campaigns —
every `hermesagent-20-*` and `cli-40.*infra-fail.*` artifact is one — so both
tools support working a subset:

```text
# Run only selected packs (repeatable or comma-separated), still in manifest order:
python3 scripts/benchlocal_campaign.py run \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --route <slug> \
  --pack reasonmath-15 \
  --skip-sidecars          # legal only for the five sidecar-free packs

# Score a tree that is deliberately incomplete:
python3 scripts/benchlocal_score.py \
  --manifest tools/benchlocal/manifest.json \
  --campaign-root <ABS_PATH> \
  --allow-partial
```

`--allow-partial` scores the packs that are present, lists the absent
`*.stdout.log` artifacts under `raw_artifact_audit.missing`, and sets
`aggregate.incomplete = true` with `packs_present` / `packs_expected` while
forcing `macro_score` and `scenario_weighted_reported_score` to `null`. A mean
over fewer than nine packs is not a macro score and must never be published as
one. Without the flag the scorer fails loudly on the first missing artifact,
which is the correct default for a campaign run.

### Four-route / nine-pack matrix

| Route slug | Family | GPU | Mode highlights | Proxy scope |
|---|---|---:|---|---|
| `qwen27-ar` | qwen27 | 0 | Qwen3.6-27B AR, KV q8, DFlash off | hermes-only |
| `qwen27-dflash` | qwen27 | 1 | same weights, DFlash on | hermes-only |
| `a3b-mq4r` | a3b | 2 | Qwen3.6-35B-A3B MQ4R (speed-max) | all packs |
| `a3b-mq4p` | a3b | 3 | Qwen3.6-35B-A3B MQ4P | all packs |

Nine runnable packs (see [Pinned packs](#pinned-packs)): seven `benchlocal-cli`
15-scenario packs, CLI-40 (40), HermesAgent-20 (20). Unsupported registry packs
`formsight` and `pixelate` are excluded (Electron / multimodal; hardcoded
temperatures that cannot honor the campaign sampling contract).

## Sampling contract and why the proxy exists

### Contract

| Parameter | Value |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `min_p` | 0.0 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` (qwen27) | 0.0 (direct runners omit; server default 0; Hermes proxy forces 0.0) |
| `presence_penalty` (a3b) | 1.5 (proxy-injected on every request) |
| request timeout | 300 s |

Thinking variants (`THINKING_MODE` on the proxy):

| Mode | Forced fields |
|---|---|
| `disabled` (default) | `enable_thinking: false`; no reasoning_effort fields |
| `medium` | `enable_thinking: true`, `reasoning_effort: "medium"`, `max_think_tokens: 1024` |

Reference proxy digests from the preserved campaigns:

| Variant | `sampling-proxy.mjs` SHA-256 |
|---|---|
| baseline (disabled) | `578f2be2b530ad4fb7ed76459ff76ba81c538376ee296af26369cf2af72099a2` |
| medium | `8a221d7c3ad11abc66302876d5c0d0ebf441d36c4d71d1b055009e4bc1af4c3d` |
| pre-thinkfix backup | `b87fb8658860d6fe971ae2c9a5b750e304278411c1edc100bc510906ab1b918f` |

### Why the proxy exists

The pinned Hermes OpenAI SDK rejects `top_k` (and related non-SDK kwargs)
**before the request reaches HTTP**. Debug runs under the baseline tree record
`Completions.create() got an unexpected keyword argument 'top_k'`. Hermes
traffic must therefore always go through the forcing proxy, which:

1. accepts the runner's temperature / top_p only,
2. rewrites every intercepted `POST .../v1/chat/completions` to the full
   campaign contract (including presence_penalty and the thinking variant),
3. appends one attestation JSONL row per rewrite:
   `{timestamp, modelSlug, path, requestModel, temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty}`,
4. exposes `/__sampling_proxy/health` and prints
   `sampling-proxy: listening on http://<host>:<port> -> <target> (<slug>)`.

Proxy env contract: `LISTEN_HOST` (campaign used `0.0.0.0`), `LISTEN_PORT`,
`TARGET_ORIGIN`, `MODEL_SLUG`, `ATTESTATION_LOG`, `PRESENCE_PENALTY` (default
`1.5`), `THINKING_MODE=disabled|medium`.

Attestation audits on the reference realizations:

| Campaign | Attestation rows | Bad rows |
|---|---:|---:|
| baseline (`full-battery-summary.json` per-route audits) | 1858 total (905+400+204+349) | 0 |
| medium (`full-battery-medium-summary.json`) | 1591 | 0 |

a3b routes send **all** pack traffic through the proxy (`proxy_scope: "all"`).
qwen27 routes proxy Hermes only; non-Hermes packs talk to the serve port
directly with the sampling flags on the CLI argv (see archived
`results/qwen27-*/qwen27-*.runner-commands.json`).

## Topology

Four concurrent daemons on host **`hiptrx`**, one per GPU, each with an
isolated `HOME` (`homes/gpu0..gpu3` in the remote scratch tree).

Port rule (from the manifest):

```text
serve_port = 12180 + gpu
proxy_port = serve_port + 100
```

Containers reach the host via Docker bridge **`172.17.0.1`**. Proxy listens on
`0.0.0.0`.

| Route | GPU | serve_port | proxy_port | non_hermes_base (host) | hermes_docker_base |
|---|---:|---:|---:|---|---|
| qwen27-ar | 0 | 12180 | 12280 | `http://127.0.0.1:12180` | `http://172.17.0.1:12280/v1` |
| qwen27-dflash | 1 | 12181 | 12281 | `http://127.0.0.1:12181` | `http://172.17.0.1:12281/v1` |
| a3b-mq4r | 2 | 12182 | 12282 | `http://127.0.0.1:12282` (proxy) | `http://172.17.0.1:12282/v1` |
| a3b-mq4p | 3 | 12183 | 12283 | `http://127.0.0.1:12283` (proxy) | `http://172.17.0.1:12283/v1` |

**Inferred:** a3b serve ports `12182` / `12183` are derived from the port rule
rather than directly observed in a runner-commands artifact. The evidenced
value is the a3b-mq4p Hermes proxy at `http://172.17.0.1:12283/v1`
(`results/a3b-mq4p/hermesagent-20-provenance.json`); serve ports follow from
`proxy_port = serve_port + 100`.

### Engine identity

| Binary | md5 |
|---|---|
| `hipfire` | `2638db7b487a43745c55fb39c345534e` |
| `daemon` | `e4af3aa26e79aa07a45fe696104dbf50` |

The medium campaign deliberately reused the baseline campaign's immutable
binaries (`full-battery-medium-summary.json` → `engine.binary_source` points at
the baseline remote `bin/`).

Smoke health gate uses `max_tokens=1152` and expects `finish_reason=stop` with
answer `42`. A 96-token smoke cap is not a valid health gate on reasoning
routes (open think spans).

## Pinned packs

Nine of eleven BenchLocal registry packs are runnable under the campaign
sampling contract → **165 scenarios / route**, **660 attempts / campaign**.

| Pack id | Dir | Upstream | Revision | Version | Scenarios |
|---|---|---|---|---|---:|
| `dataextract-15` | DataExtract-15 | https://github.com/stevibe/DataExtract-15.git | `00d90bf7506a1d7ffe98943d9ffd8c6eb795dbdb` | 1.0.0 | 15 |
| `instructfollow-15` | InstructFollow-15 | https://github.com/stevibe/InstructFollow-15.git | `187af97cb2b892ad57de176b16a254aba7565a65` | 1.0.0 | 15 |
| `reasonmath-15` | ReasonMath-15 | https://github.com/stevibe/ReasonMath-15.git | `b97632020fa373c52ba92373dbe5dc58b744ce48` | 1.0.0 | 15 |
| `toolcall-15` | ToolCall-15 | https://github.com/stevibe/ToolCall-15.git | `edd6cefe4261b67e8166e9f6d77d671042560294` | 1.0.1 | 15 |
| `promptauthority-15` | PromptAuthority-15 | https://github.com/stevibe/PromptAuthority-15.git | `6ed261b42fcca0df15e4794eeb4da628b1666952` | 1.0.0 | 15 |
| `structoutput-15` | StructOutput-15 | https://github.com/stevibe/StructOutput-15.git | `00de86e9bfc9dd3d86ba397d0cf35bcbc04efd1c` | 1.0.0 | 15 |
| `bugfind-15` | BugFind-15 | https://github.com/stevibe/BugFind-15.git | `eea2224a531f0937a8fffb4949abcd6db81f4d11` | 1.0.1 | 15 |
| `cli-40` | CLI-40 | https://github.com/stevibe/CLI-40.git | `3b95f86e6edac47183348381a9bb211ffaf09404` | 1.0.2 | 40 |
| `hermesagent-20` | HermesAgent-20 | https://github.com/stevibe/HermesAgent-20.git | `fa40ab9fb84a329421bbdfc3062cf28f1670de71` | 1.0.0 | 20 |

Unsupported (not scored): **`formsight`**, **`pixelate`**.

Notes:

- Only `toolcall-15` and `promptauthority-15` are `tool_pack: true` (they accept
  `--tools-format` / `--repetition-penalty` on the pack CLI).
- StructOutput host validator port is remapped to **4011** so it does not
  collide with BugFind's sandbox on 4010.
- CLI-40 sampling is hardcoded in the campaign dist (same base contract; no
  presence_penalty). Hermes sampling beyond temperature/top_p is entirely
  proxy-injected.

## Landmines

| Symptom | Real cause | What to do |
|---|---|---|
| Hermes `--all` dies mid-pack (baseline: after HA-03 on qwen27-ar, after HA-16 on a3b-mq4p; medium: after HA-16 on both a3b routes) | Official runner transport / fetch failure, not a scored scenario result | Run `recover-hermes`; stitch via `hermesagent-20-provenance.json` (or `hermesagent-20.provenance.json` on baseline qwen27-ar). Recovery is the norm, not the exception: the baseline campaign stitched 3 of 4 routes and only `a3b-mq4r` was a single corrected full pass, while in the medium campaign the one clean pass was `qwen27-dflash`. Per-route methods are recorded in each summary's `hermes_recovery` / `recovery_provenance` block. |
| Baseline qwen27-ar HA-17 carries a methodology deviation | Official runner transport failed repeatedly, so HA-17 was driven by a direct POST to the identical verifier `/run-scenario` handler | Legitimate but must stay disclosed: verifier image/core, model route, sampling proxy, raw trace, and deterministic scorer were unchanged — only `run-scenarios.mjs` transport was bypassed. Do not silently normalize this away when re-running |
| HA-09 OOM; socket hang; preload fail; prepare EACCES | Container / host infra during Hermes isolation | Preserve `hermesagent-20*{oom,socket-hang,preload-fail,prepare-eacces}*` artifacts; exclude from scoring; retry the affected scenarios in isolation |
| Docker bind collision on CLI-40 retry | Port / bind clash between verifier containers | Preserve as `cli-40.*infra-fail.*`; exclude; keep the later complete `cli-40.*` retry as canonical (medium a3b-mq4p) |
| Smoke returns open think spans / non-stop finish | `max_tokens=96` is too small on reasoning routes | Use smoke `max_tokens=1152`. Smoke observations are **excluded** from the 660 canonical attempts |
| Temptation to score `results/*-no-presence` | a3b runs missing `presence_penalty=1.5` | Exclude entire `*-no-presence` trees; they are not comparable to the campaign contract |
| Baseline summary shows a 63-hex sha256 for a3b-mq4p | Archive truncated the digest (dropped trailing `f`) | Use repaired value `8eb3be6912aa9db2dcc89c3233b05ccdfe81a84a8d6ef43202f2098d7f2fc78f` (re-hashed 2026-08-02). Sibling a3b-mq4r matched byte-for-byte, confirming the local weights are what the campaign served |
| Hermes fails immediately with `unexpected keyword argument 'top_k'` | Pinned Hermes OpenAI SDK rejects kwargs before HTTP | Point Hermes at the sampling proxy, not the raw serve port. Preserve `*openai-kwargs-failure*` as infra, not quality |
| Archived runner-commands disagree with each other | Historical one-off argv drift (e.g. missing `--provider-model`, flag order, `--model=v` vs `--model v`) | Emit the **canonical** form from `benchlocal_campaign.py plan` / the contract in the manifest driver — do not byte-reproduce every archive inconsistency |

## Scoring definitions and comparability limits

### Definitions

| Metric | Definition |
|---|---|
| Macro score | Unweighted mean of the nine pack headline scores |
| Scenario-weighted score | Pack headline scores weighted by scenario count (seven×15 + Hermes 20 + CLI 40) |
| Status totals | Sum of canonical pack pass / partial / fail counts |

Exclusions applied before scoring are listed in
`tools/benchlocal/manifest.json` → `exclusions` (no-presence trees, Hermes
openai-kwargs failures, Hermes infra diagnostics, CLI-40 infra-fail retries).

### Headline results (disabled → medium)

Source: `PUBLISHED-COMPARISON.md` and the two verified summary JSON files.

| Route | Thinking disabled | Medium thinking | Δ macro |
|---|---:|---:|---:|
| qwen27-ar | 82.9056 | 83.1250 | +0.2194 |
| qwen27-dflash | 84.3139 | 82.3250 | −1.9889 |
| a3b-mq4p | 80.6806 | 68.6750 | −12.0056 |
| a3b-mq4r | 69.4139 | 62.6472 | −6.7667 |

Operational reading (from the medium summary + published comparison):

- DFlash preserved Qwen27 quality vs AR on the disabled run (macro lead ~1.41
  with the same 115/165 hard-pass count).
- Medium thinking did not improve the battery overall: neutral for AR, mildly
  harmful for DFlash, substantially harmful for both A3B routes.
- MQ4R is the speed-max profile — read as retained quality at maximum
  throughput, not as a quality competitor to MQ4P.

### Comparability limits

State these every time numbers are cited externally:

1. **There is no official BenchLocal leaderboard** and no official nine-pack
   macro. See [BugFind-15#1](https://github.com/stevibe/BugFind-15/issues/1).
2. Hipfire ran **one controlled realization** at temperature 1.0 / top_p 0.95 /
   top_k 20. InferRank publishes **three-run averages** at temperature 0.6
   (same top_p / top_k) over an eight-pack subset.
3. Community matrices (Kovalov seven-pack, InferRank eight-pack) used matching
   semantic pack versions but not necessarily the same git revisions or a
   uniform sampling contract.
4. Quantization, serving stack, chat template, reasoning mode, context, retries,
   and pack revisions all move scores.

These campaigns establish **capability tiers and workload profiles**, not a
formal global ranking. Full public-source list and normalized comparison tables
live in NAS `PUBLISHED-COMPARISON.md`:

- https://github.com/stevibe/BenchLocal
- https://github.com/stevibe/BugFind-15/issues/1
- https://sergeykovalov.com/debugging-local-llm-benchmarks/
- https://github.com/Rhonstin/inferrank/blob/master/docs/FULLBENCH_ANALYSIS.md
- https://inferrank.selfcloud.pp.ua/
- https://x.com/stevibe/status/2066563724375376195
- https://x.com/stevibe/status/2060370310030053380

## Related

- [`bench-suite.md`](bench-suite.md) — production serve harness map
- [`perf-benchmarking.md`](perf-benchmarking.md) — measurement identity protocol
- [`docs/VALIDATION.md`](../VALIDATION.md) — which route a change owes
