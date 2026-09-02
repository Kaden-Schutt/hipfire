<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: ConfigTopology

# Audit: ConfigTopology

Checkout: `/home/kaden/ClaudeCode/warpfront/hipfire` @ `origin/master` `8cd15a62b` (read-only).

Scope: `hipfire-config`; runtime `config.rs` / `loader_api.rs` / `multi_gpu.rs` / `device_mesh.rs` / `ep.rs`; `hipfire-registry` + `registry/`; CLI config/registry/pull/serve/bench.

## Architecture (orientation)

Ladder: `resolve(layers)` lowest→highest. Docs operator precedence: CLI one-shot > env > per-model > global TOML > registry > built-in. `load_env_layer` maps schema `env_compat` into typed keys; unknown `HIPFIRE_*` → `developer.<snake>` (`hipfire-config/src/lib.rs:3911-3953`). `ProcessConfig::from_resolved` keeps env_compat fields + developer keys; `legacy_value` / `developer_var` / `process_value` read the freeze only (`:3080-3088`, `:3224-3236`). `RuntimeConfig` OnceLock lowers once; `devices` rewritten to logical `0..N-1`. Device visibility: `hardware.devices`/`HIPFIRE_DEVICES` wins; else HIP/ROCR pair normalized (ROCR physical, HIP logical); `apply_device_visibility` `set_var`s both (`:3135-3145`). Multi-GPU: `Gpus` owns devices + `DeviceMesh`. `init_uniform`/`init_layers` → `from_parts` → mesh `Pp:N`; `init_tp` → mesh `Tp:N`; `single` → empty axes. EP loads all call `init_tp` (mesh labeled Tp, not Ep). `size_of(missing)=1`. Registry: v1 + curated `models.json`; dangling aliases silently retained-away; pull verifies sha/size when present.

## Broken

### 1. Production ambient HIPFIRE_* reads bypass ProcessConfig freeze (verified)
**path_line:** `docs/env-vars.md:51-69`; `crates/hipfire-daemon/src/main.rs:497-499,1199-1202,1575-1589,1801-1804,3294`; `crates/hipfire-loader/src/carriers.rs:409-413,2574-2578`; `crates/hipfire-runtime/src/calibration.rs:44-48,132-133`; `scripts/check-env-docs.py:31-76`; `scripts/no-gpu-ci.sh:29-30`.
**How known:** Docs claim production only reads bootstrap env. Grepped production `std::env::var("HIPFIRE_…")`. Checker fails any non-bootstrap literal outside CENTRAL_CONFIG_READERS/examples/tests.
**Impact:** Installed ProcessConfig can disagree with live ambient after daemon start. PP experimental compose and page-cache policy break the freeze story. Checker success string is false for this tree (CI status not re-executed).

### 2. EP loads advertise Tp mesh; size_of(Ep)==1 (verified)
**path_line:** `crates/hipfire-loader/src/lib.rs:2844-2847,2993-2994,3221-3222,3360-3361,3491-3492`; `crates/hipfire-runtime/src/multi_gpu.rs:265-300`; `crates/hipfire-runtime/src/device_mesh.rs:192-200`.
**How known:** Every `load_model_ep*` uses `Gpus::init_tp`; init_tp sets `mesh = DeviceMesh::rect([(Tp, tp_size)])`; no Ep constructor; size_of defaults missing to 1.
**Impact:** EP still shards via `devices.len()` today. Mesh-aware #681 consumers asking Ep get single-rank. Topology twin of loader-admits / wrong-shape bugs.

### 3. SpecLoadCfg documents loader env fallback that does not exist (verified)
**path_line:** `crates/hipfire-runtime/src/loader_api.rs:92-95`; `crates/hipfire-loader/src/spec_build.rs:201-202`.
**How known:** Comment says HIPFIRE_NGRAM_DRAFT* fallback and env wins; spec_build says never ambient, `unwrap_or(false)`. Daemon fills from RuntimeConfig/params (~1672) so CLI path OK — contract text is false.

### 4. Ornith curated sampling pins disagree with registry test (verified)
**path_line:** `scripts/test_registry_gen_ornith.py:77-88`; `registry/models.json:1466-1512`.
**How known:** Test exact-equality without reasoning_effort; curated has xhigh on recommended/general/coding and none on instruct. Tag/files/aliases look correct; sampling pin drifted.

### 5. Gpus::single vs from_parts(n=1) mesh shapes disagree (verified)
**path_line:** `crates/hipfire-runtime/src/multi_gpu.rs:238-262` (empty axes); `:828-847` (always Pp:n_devices).
**Impact:** Low today (size_of(Pp)=1; has_axis requires size>1). axes() length 0 vs 1 still diverges.

### 6. CLI bench continuous_batch process-env side channel (verified)
**path_line:** `crates/hipfire-cli/src/main.rs:3925-3928,4007-4010`; schema `serve.continuous_batch_size` `hipfire-config/src/lib.rs:1013-1018`.
**Impact:** set_var/get bypasses ProcessConfig; checker-visible; schema already owns the knob.

### 7. Experimental PP gates snapshotted then re-read live (verified)
**path_line:** `load_env_layer` maps HIPFIRE_PP_* → developer.*; daemon still `std::env::var` at `:1575-1589`.
**Impact:** Freeze contract broken for multi-GPU experimental compose.

### 8. check-env-docs.py cannot truthfully pass on this tree (verified logic)
**path_line:** `scripts/check-env-docs.py:14-76` + Broken #1 sites.
Also misses non-literal reads, HIP_VISIBLE_DEVICES/ROCR_*, docs under docs/ not in REFERENCE_DOCS.

## Missing

### 1. No Gpus::init_ep / DimKind::Ep on EP admission (verified)
DeviceMesh supports Ep; constructors only emit Pp/Tp. EP overloads init_tp. Incomplete #681 cutover.

### 2. hipfire registry undocumented in docs/CLI.md (verified)
CLI RegistryAction Status/List/Show/Update/Verify (`main.rs:303-325,642-643`). CLI.md config tables only; grep registry empty.

### 3. init_vram_weighted stub (verified, documented)
`multi_gpu.rs:228-235`; `docs/multi-gpu.md:62`.

### 4. Dangling registry aliases dropped silently (verified)
`hipfire-registry/src/lib.rs:328-336` retain without warning.

### 5. Windows device-visibility hardening/docs (partial)
Unix-shaped set_var only (`:3135-3145`). Client child-env path better. No in-tree #669 string. Not hardware-proven broken.

### 6. ModelEntry.sampling dual/inert vs recommended_settings (verified)
`lib.rs:196-199`; docs say sampling inert. Half-migration dead write surface.

### Notes (not broken)
- No `parallel_capability.rs` on master loader tree — dual path is Tp-labeled EP, not two frameworks.
- Bare `hipfire config` → TUI (`main.rs:686-687`); docs/CLI.md:82 matches. `config <tag>` defaults List (`:1076`).
- `hardware.devices` nullable string / HIPFIRE_DEVICES → ROCR physical + HIP logical — aligned.
- `HIPFIRE_PP_LAYERS` **is** wired via `developer_var` in carriers.rs (not dead).

## Would change (ranked, with cost)

1. **Route residual production env through ProcessConfig/developer_var/typed keys** (daemon PP/DFlash/DPM/log; loader PAGE_EVICTION; calibration). Honest bootstrap list in env-vars.md. Green check-env-docs = true. **Cost: 1–2 days.**
2. **Gpus::init_ep (or mesh-kind on init_tp) on all load_model_ep*** — DimKind::Ep; unit-test size_of(Ep). **Cost: hours–1 day.**
3. **Fix SpecLoadCfg/ngram contract** (+ optional RuntimeConfig default). **Cost: hours.**
4. **Reconcile ornith test with reasoning_effort** (separate instruct pin). **Cost: hours.**
5. **Delete bench HIPFIRE_BENCH_CONTINUOUS_BATCH bridge** — pass param. **Cost: hours.**
6. **Document hipfire registry in CLI.md; warn on dropped aliases.** **Cost: hours.**
7. **Unify Gpus::single mesh with from_parts(1).** **Cost: hours.**
8. **Windows visibility docs + prefer child-env injection.** **Cost: hours** docs; **days+** hardware matrix.

## Confidence

**Did:** config resolve/env/ProcessConfig/visibility; multi_gpu+device_mesh; EP loads; SpecLoadCfg vs spec_build; daemon/loader ambient reads; CLI config/registry/bench; registry parse+ornith; check-env-docs algorithm; docs env-vars/CLI/multi-gpu; production HIPFIRE greps.

**Did not:** run checker/tests/builds; live HF sha/size; full gh novelty (hub capture incomplete); Windows hardware; deep ep.rs mesh consumers; every serve/run flag→ProcessConfig mapping; quant register beyond existence.

**Novelty:** Did not confirm open-issue absence for env-bypass or EP-mesh labeling. #683 is generate-routing (peer). This EP finding is post-#681 **constructor labeling**, not the reverted device-mesh PR body.

**Suspicious not verified:** whether no-gpu-ci is red on check-env-docs; whether generate already calls size_of(Ep); Windows #669 failure mode without issue body/hardware.

## JSON summary (for parent synthesis)
```json
{
  "slice": "ConfigTopology",
  "broken": [
    {"title": "Production ambient HIPFIRE_* bypass ProcessConfig", "path_line": "docs/env-vars.md:51-69; hipfire-daemon/src/main.rs:497,1199,1575-1589,1801; carriers.rs:409; calibration.rs:44; check-env-docs.py:31-76", "verified": true},
    {"title": "EP loads build Tp mesh; size_of(Ep)=1", "path_line": "hipfire-loader/src/lib.rs:2993,3221,3360; multi_gpu.rs:300; device_mesh.rs:192", "verified": true},
    {"title": "SpecLoadCfg env-fallback lie", "path_line": "loader_api.rs:92-95; spec_build.rs:201-202", "verified": true},
    {"title": "Ornith test vs curated reasoning_effort", "path_line": "test_registry_gen_ornith.py:77-88; registry/models.json:1482-1511", "verified": true},
    {"title": "Gpus::single vs from_parts mesh shape", "path_line": "multi_gpu.rs:238-262,847", "verified": true},
    {"title": "Bench continuous_batch set_var side channel", "path_line": "hipfire-cli/src/main.rs:3925,4007", "verified": true},
    {"title": "PP gates snapshotted then re-read live", "path_line": "hipfire-config load_env_layer:3911; daemon:1575-1589", "verified": true},
    {"title": "check-env-docs cannot truthfully pass", "path_line": "scripts/check-env-docs.py:57-76", "verified": true}
  ],
  "missing": [
    {"title": "No init_ep / Ep mesh admission", "path_line": "multi_gpu.rs:265-300; loader lib.rs:2844", "verified": true},
    {"title": "hipfire registry missing from CLI.md", "path_line": "cli main.rs:303-325; docs/CLI.md", "verified": true},
    {"title": "init_vram_weighted stub", "path_line": "multi_gpu.rs:228-235", "verified": true},
    {"title": "Silent dangling alias drop", "path_line": "hipfire-registry/src/lib.rs:328-336", "verified": true},
    {"title": "Windows visibility docs/hardening", "path_line": "hipfire-config/src/lib.rs:3135-3145", "verified": true},
    {"title": "Dual inert ModelEntry.sampling", "path_line": "hipfire-registry/src/lib.rs:196-199", "verified": true}
  ],
  "changes": [
    {"title": "Route residual env via ProcessConfig", "cost": "1-2 days", "path_line": "daemon/loader/runtime ambient sites"},
    {"title": "Gpus::init_ep for EP loads", "cost": "hours-1 day", "path_line": "multi_gpu.rs; loader EP"},
    {"title": "Fix SpecLoadCfg contract", "cost": "hours", "path_line": "loader_api.rs; spec_build.rs"},
    {"title": "Ornith reasoning_effort pins", "cost": "hours", "path_line": "test_registry_gen_ornith.py"},
    {"title": "Delete bench env side channel", "cost": "hours", "path_line": "cli main.rs:3925-4010"},
    {"title": "CLI.md registry + alias warnings", "cost": "hours", "path_line": "docs/CLI.md; registry parse"},
    {"title": "Unify single-device mesh", "cost": "hours", "path_line": "multi_gpu.rs:238-262"},
    {"title": "Windows visibility docs/child-env", "cost": "hours-days", "path_line": "config visibility; docs"}
  ]
}
```
