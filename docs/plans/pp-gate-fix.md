# pp-gate iGPU + heterogeneity fix (issue #216) — v2

Branch: `fix/pp-gate-igpu-detection`
Target: `scripts/pp-gate.sh`
Scope: parts (1) + (2) + (3) + (4) from the issue. (5) — per-device VRAM probe — deferred.

**v2 incorporates findings from three adversarial reviews** (Claude /
GLM-5 / Gemini). Notable design changes from v1:

- Primary device-arch source is sysfs (`/sys/class/kfd/kfd/topology/`),
  not `rocm-smi --showhw`. Stable across ROCm versions, no column-parsing
  fragility. rocm-smi becomes a fallback, with a loud warning when it
  fires.
- `HIP_VISIBLE_DEVICES` is derived from the *filtered* index set, not
  the hardcoded `0,1` default at line 85 — fixes a 3+ GPU host where
  the bug recurred even with the filter in place.
- Empty-SHA equality is explicitly REJECTED, not just compared against
  pp=1's SHA — fixes "both runs emit 0 bytes → false PASS".
- Heterogeneity check compares ISA *family*, sourced from
  `crates/rdna-compute/src/kernels.rs:80-139` — same-family dGPU pairs
  (gfx1100 + gfx1101) no longer false-skip.
- All-iGPU-host carve-out **dropped**; iGPU runs require explicit
  `HIPFIRE_PP_GATE_INCLUDE_IGPU=1`.
- Windows hosts skip cleanly at the top, before any probing.
- iGPU filter list updated for accuracy (gfx1103 added; gfx1035/1036/1037
  labels corrected against LLVM AMDGPU target list).
- Adds `--dry-run` flag for parser verification without GPU work.

## Problem (recap)

On a host with one usable dGPU and one iGPU (MI50 + Renoir gfx90c on
this dev box — `k9lin`, the issue's repro environment), `pp-gate.sh`
counts both as GPUs, exports `HIP_VISIBLE_DEVICES=0,1`, runs the
parity check, and reports `pp=2 ≢ pp=1 byte-identical FAIL` when the
daemon at pp=2 actually emitted **zero bytes** (sha256 prefix
`e3b0c44298fc1c14` = empty string). The iGPU can't host the model;
load fails silently; gate misrepresents it as a parity bug.

## Touched code

Only `scripts/pp-gate.sh`. No engine / kernel / Rust changes.
Estimated +120 lines, -15 lines net (v1 said "+60 / -15" — revised
upward after review-driven additions: sysfs parser, family lookup,
dry-run, stderr handling, Windows skip, validation).

## Platform: Linux only

The probing logic is Linux-specific (sysfs `/sys/class/kfd/` is a
Linux-kernel ioctl surface; ROCm-for-Windows has no equivalent).
The pre-existing script also assumes POSIX `/tmp`, bash arrays, and
a Linux-only daemon binary. On Windows, hipfire runs through Git Bash
/ MSYS2 / WSL at most; dual-GPU pipeline parallelism on Windows ROCm
isn't a supported AMD config. WSL passes through as Linux per
`uname -s` so it's covered by the normal path.

```bash
# ── Platform gate ───────────────────────────────────────────────────────
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "pp-gate: Windows host ($(uname -s)) — skipping (Linux ROCm only)"
        exit 0
        ;;
esac
```

Goes immediately after `set -u`, before `rocm-env.sh` sourcing.

## Device probing

### Primary: sysfs

Read `/sys/class/kfd/kfd/topology/nodes/*/properties`. Each node has
both `simd_count` (0 = CPU host node) and `gfx_target_version`
(integer encoding: `90006` = gfx906, `90012` = gfx90c, `100100` =
gfx1010, `110000` = gfx1100, `110501` = gfx1151, etc.).

Verified on the repro box (this dev box):
```
node 0: cpu_cores_count 12, simd_count 0,   gfx_target_version 0      → CPU host (skip)
node 1: cpu_cores_count 0,  simd_count 240, gfx_target_version 90006  → MI50 (gfx906)
node 2: cpu_cores_count 0,  simd_count 28,  gfx_target_version 90012  → Renoir (gfx90c)
```

KFD node indices ≠ HIP device indices. After filtering out CPU host
nodes (those with `simd_count == 0`), the remaining nodes in node-id
order map 1:1 to HIP device indices in encounter order (matches
HIP/ROCr behavior). Verify this mapping with `rocm-smi --showid`'s
`Node ID:` field cross-reference if any operator reports a mismatch.

Pseudo-code:
```bash
parse_sysfs_devices() {
    local idx=0
    for nodedir in /sys/class/kfd/kfd/topology/nodes/*/; do
        local props="$nodedir/properties"
        [ -r "$props" ] || continue
        local simd gfx
        simd=$(awk '/^simd_count/ {print $2}' "$props")
        gfx=$(awk '/^gfx_target_version/ {print $2}' "$props")
        [ "$simd" = "0" ] && continue          # CPU host node
        [ "$gfx" = "0" ] && continue           # belt-and-suspenders
        # Convert integer encoding (90006) → gfx string (gfx906)
        local gfx_str
        gfx_str=$(gfx_int_to_str "$gfx")
        printf '%d:%s\n' "$idx" "$gfx_str"
        idx=$((idx + 1))
    done
}
```

`gfx_int_to_str` converts integer encoding to canonical gfx string.
The integer encoding is `major * 10000 + minor * 100 + stepping`:
- `90006` → `gfx906` (9.0.6)
- `90012` → `gfx90c` (9.0.12 → hex stepping `c` for gfx90c)
- `100103` → `gfx1013` (10.1.3)
- `110000` → `gfx1100` (11.0.0)
- `110003` → `gfx1103` (11.0.3)
- `110501` → `gfx1151` (11.5.1)

This integer is not the decimal concatenation of the gfx suffix:
`gfx1151` is encoded as 11.5.1 → `110501`, not `115100`.

The stepping-as-hex case (gfx90c, gfx90a) requires a small lookup;
hand-code the handful of known cases plus a fallback that prints the
raw integer for diagnostic purposes if an unknown encoding shows up.

### Fallback: rocm-smi

If `/sys/class/kfd/kfd/topology/` is missing (rare — pre-ROCm-5 host,
no amdkfd module loaded), fall back to `rocm-smi --showhw` parsing.
Per-data-row regex `^([0-9]+)\s+[0-9]+\s+0x[0-9a-f]+\s+[0-9]+\s+(gfx[0-9a-z]+)\s` — the only columns we need (GPU index, GFX VER) are
non-whitespace and stably positioned in data rows, even though the
header has embedded spaces (`GFX VER`, `GFX RAS`).

When fallback fires, emit a loud warning so future regressions are
visible:
```
pp-gate: WARNING — sysfs topology unavailable, falling back to rocm-smi.
                   If filtering misbehaves, this is the first thing to check.
                   Set HIPFIRE_PP_GATE_REQUIRE_SYSFS=1 to hard-fail instead.
```

If both sysfs and rocm-smi fail (no AMD GPUs visible, no ROCm
installed), preserve the existing `HIP_VISIBLE_DEVICES` parse path at
old line 71-73 as a last resort, then the `<2 GPU → skip` early-exit
at line 75-78. This keeps CI / containers / no-GPU dev boxes working
unchanged.

## Filter pipeline

After parsing the device list, apply filters in this order:

### 1. PP_GATE_DEVICES filter (operator intent)

If `PP_GATE_DEVICES` is set, intersect the parsed list with its
indices. Validate that every requested index exists in the parsed
list — if any don't, error (don't silently filter):

```
pp-gate: PP_GATE_DEVICES=0,3 includes index 3, but only 2 device(s)
         are visible. Set PP_GATE_DEVICES to a subset of {0,1}.
```

This honors operator intent for (1) AND catches typos that would
otherwise silently degrade gate coverage. Skip the remaining filters
if `PP_GATE_DEVICES` is set — the operator has explicitly chosen the
device set.

### 2. iGPU filter (default behavior)

Filter out devices whose gfx arch is in the iGPU allowlist. Skip this
filter entirely if `HIPFIRE_PP_GATE_INCLUDE_IGPU=1`.

iGPU gfx archs to filter (verified against LLVM AMDGPU target list +
ROCm device-libs):

| gfx | APU codenames | Source |
|---|---|---|
| `gfx902` | Raven Ridge / Picasso (Ryzen 2000G/3000G) | LLVM AMDGPU.td |
| `gfx909` | Raven2 / Dali (low-end mobile) | LLVM AMDGPU.td |
| `gfx90c` | Renoir / Lucienne / Cezanne / Barcelo (4xxx-5xxx APU) ← **this dev box** | LLVM + verified locally |
| `gfx1013` | Van Gogh (Steam Deck APU); also BC-250 desktop (16 GB UMA, edge case) | LLVM AMDGPU.td |
| `gfx1033` | Rembrandt (6xxx mobile APU) | LLVM AMDGPU.td |
| `gfx1034` | Rembrandt-R refresh | LLVM AMDGPU.td |
| `gfx1035` | Rembrandt-R / 6xxx-refresh mobile APU | LLVM AMDGPU.td |
| `gfx1036` | Raphael desktop iGPU (2 CU, AM5 7xxx) | LLVM AMDGPU.td |
| `gfx1103` | Phoenix / Phoenix2 / Hawk Point mobile APU | LLVM AMDGPU.td |

**Not in the filter:**
- `gfx1150` (Strix Point) — APU but reviewers diverged on classification.
  Conservative call: keep OUT of v1 filter, since gfx1150's 16-CU iGPU
  + LPDDR5x bandwidth profile is the closest any APU has come to being
  a meaningful PP partner. Revisit if a Strix Point user files a
  follow-up.
- `gfx1151` (Strix Halo) — APU but with dGPU-class memory (32+ GB
  unified, 256 GB/s). Keep OUT. UMA bandwidth-sharing concerns
  raised in review are real, but the gate's job is correctness, not
  perf gating — let it run, surface 0-byte detection if it fails.
- `gfx1152` — status unclear (RDNA3.5 family in `dispatch.rs:140` but
  excluded from WMMA paths in `kernels.rs:80-139`). Keep OUT until
  classification confirmed; the heterogeneity check will catch a
  gfx1152 + dGPU mismatch, and 0-byte detection will catch a load
  failure if it's iGPU-class.

The filter list lives as a bash array near the top of the script with
a comment pointing at this plan AND at `crates/rdna-compute/src/kernels.rs`
as the engine's upstream-of-truth for arch handling. **Adding a new
APU arch is a one-line change** to this array; reviewers all flagged
the maintenance concern, and the answer is "make it cheap".

The carve-out for "all devices filtered" is **removed** in v2. If the
filter zeros out the device list:
- `PP_GATE_DEVICES` set → already exited at step 1.
- `PP_GATE_DEVICES` unset → exit with "all visible GPUs are iGPUs;
  set `HIPFIRE_PP_GATE_INCLUDE_IGPU=1` to test anyway".

This eliminates the silent-re-include path identified in three reviews.

### 3. Homogeneity check (family-aware)

If ≥2 devices remain and they belong to different ISA families, skip
with a clear message. **Family**, not raw arch — sourced from the WMMA
groupings already encoded in `crates/rdna-compute/src/kernels.rs:80-139`:

| Family | Archs |
|---|---|
| `cdna1` | gfx906 (also gfx908, gfx90a — group with caution; PP across CDNA gens not validated) |
| `rdna1` | gfx1010, gfx1011, gfx1012 |
| `rdna2` | gfx1030, gfx1031, gfx1032 |
| `rdna3` | gfx1100, gfx1101, gfx1102, gfx1150, gfx1151 (per kernels.rs WMMA groups) |
| `rdna4` | gfx1200, gfx1201 (per kernels.rs WMMA groups) |

iGPU archs are not in any family bucket here — they're filtered out
before this step under default flags, and bypass to "all archs are
heterogeneous" if `INCLUDE_IGPU=1` is set, which is the intended
behavior (override + iGPU + dGPU = expect a noisy run).

Skip message includes a hint about index pinning, per Gemini 4.1:
```
pp-gate: heterogeneous ISA families (rdna3 + cdna1) — skipping.
         Kernel cache is keyed per-family; pp>1 across mismatched
         families isn't supported.
         Hints:
           - pin a homogeneous pair via PP_GATE_DEVICES=0,2
           - or set HIPFIRE_PP_GATE_HETEROGENEOUS=1 to force-run
```

### Precedence summary

```
PP_GATE_DEVICES set       → use those indices verbatim, skip iGPU+homogeneity
PP_GATE_DEVICES unset:
  HIPFIRE_PP_GATE_INCLUDE_IGPU=1 → skip iGPU filter, run homogeneity
  HIPFIRE_PP_GATE_HETEROGENEOUS=1 → skip homogeneity check (iGPU filter still applies)
  default                       → both filters apply
```

The precedence matrix is documented in a header comment at the top of
the script.

### HIP_VISIBLE_DEVICES wiring (B6 fix)

After the filter pipeline produces the final device list, **derive
`HIP_VISIBLE_DEVICES` from those indices**:

```bash
# Replace pp-gate.sh:85 (`export HIP_VISIBLE_DEVICES="${PP_GATE_DEVICES:-0,1}"`)
# with:
export HIP_VISIBLE_DEVICES=$(printf '%s' "$filtered_indices" | paste -sd,)
```

`filtered_indices` is the comma-separated list output by the filter
pipeline. If `PP_GATE_DEVICES` was set, it's already that value. If
the filter ran, it's the filtered indices. **The hardcoded `0,1`
default is removed entirely** — it's the root of the 3-GPU-host
regression case (Gemini 1.1).

## Zero-byte / load-failure detection (B5 + B7)

`gen_sha` is restructured to emit both a text-event count and a hash,
not just a hash. Callers compare both.

```bash
gen_summary() {
    local pp_arg="$1"
    local logfile
    logfile=$(mktemp -t pp-gate-pp${pp_arg}.XXXXXX.log)
    local params='{"max_seq":2048}'
    [ "$pp_arg" = "2" ] && params='{"max_seq":2048,"pp":2}'
    (printf '%s\n' \
        '{"type":"load","model":"'"$MODEL"'","params":'"$params"'}' \
        '{"type":"generate","id":"r1","prompt":"Write a one-sentence greeting.","temperature":0.0,"max_tokens":40}' \
        '{"type":"unload"}'
    ) | "$EXE" 2>"$logfile" \
      | grep '"text"' \
      | python3 -c '
import sys, json, hashlib
toks = []
for line in sys.stdin:
    obj = json.loads(line.strip())
    toks.append(obj["text"])
joined = "".join(toks)
print(f"{len(toks)} {hashlib.sha256(joined.encode()).hexdigest()[:16]} {logfile_path}")
' "logfile_path=$logfile"   # passed via env to keep python heredoc clean
}
```

Callers parse `count sha logfile` and apply three checks in order:

1. **Either count == 0 → load failure.** Print:
   ```
   pp-gate: pp=$pp_arg emitted 0 text events — daemon load or dispatch
            failed. See $logfile for daemon stderr.
            Hints:
              - PP_GATE_DEVICES=0 to skip pp=2
              - HIPFIRE_PP_GATE_INCLUDE_IGPU=1 for deliberate iGPU testing
              - check $logfile for OOM / ISA mismatch / missing model
   ```
   Fires for pp=1 OR pp=2 (GLM-5 H4) — both runs are equally diagnostic.

2. **Either sha == `e3b0c44298fc1c14`** (sha256-prefix of empty string)
   **regardless of equality** → load failure. Catches the "daemon
   emitted text events but they were all empty strings" pathological
   case AND the "both runs failed identically" false-PASS (Gemini 1.2).

3. **Only if both pass (1) and (2): compare counts and SHAs.**
   - Mismatched count → "pp=2 produced N tokens vs pp=1's M" (real
     divergence, but more diagnostic than the old framing).
   - Matching count, mismatched SHA → the old "pp=2 ≢ pp=1
     byte-identical" message (this is the actual parity-bug signal
     the gate was built for).
   - Matching both → PASS.

stderr is redirected per-run to a per-PID tempfile (`mktemp`),
addressing Gemini 3.1's multi-user concern. Both logs are mentioned
in failure messages so the operator doesn't have to guess which file
to read.

## --dry-run flag (GLM-5 L2)

New `--dry-run` flag prints the parsed device list, filter decisions,
and the resulting `HIP_VISIBLE_DEVICES` without spawning the daemon
or running the gate. Doubles as a test oracle:

```
$ ./scripts/pp-gate.sh --dry-run
pp-gate: dry-run mode
  source: sysfs (/sys/class/kfd/kfd/topology/)
  parsed: 0:gfx906 1:gfx90c
  filter pipeline:
    PP_GATE_DEVICES: unset
    iGPU filter:     gfx90c (1) → removed
    homogeneity:     n/a (1 device remaining)
  decision: 1 GPU(s) usable — would skip
  HIP_VISIBLE_DEVICES: (not exported)
```

Costs ~25 lines. Enables operator self-debug and lets the validation
matrix below be exercised without a working ROCm install.

## Validation

Run on this dev box (gfx906 + gfx90c MI50+Renoir, the issue's repro
environment per CLAUDE.md and project memory):

1. **Default invocation** — `./scripts/pp-gate.sh`
   Expected (post-fix): "pp-gate: filtered iGPU gfx90c (index 1); only
   1 GPU(s) usable — skipping". Exit 0, <5 sec.
   Pre-fix actual: ~50 min run → FAIL on empty pp=2 hash.

2. **Forced iGPU inclusion** — `HIPFIRE_PP_GATE_INCLUDE_IGPU=1 ./scripts/pp-gate.sh`
   Expected: dispatches to gfx906+gfx90c, pp=2 emits 0 text events,
   new error message: "pp=2 emitted 0 text events — daemon load or
   dispatch failed. See /tmp/pp-gate-pp2.XXXXXX.log". Exit 1.

3. **PP_GATE_DEVICES single-device** — `PP_GATE_DEVICES=0 ./scripts/pp-gate.sh`
   Expected: filter pipeline skipped (PP_GATE_DEVICES wins precedence),
   1 device → "only 1 GPU(s) visible — skipping". Exit 0.

4. **PP_GATE_DEVICES invalid index** — `PP_GATE_DEVICES=0,5 ./scripts/pp-gate.sh`
   Expected: error "PP_GATE_DEVICES=0,5 includes index 5, but only 2
   device(s) are visible". Exit 2.

5. **Forced iGPU + PP_GATE_DEVICES=0,1** —
   `HIPFIRE_PP_GATE_INCLUDE_IGPU=1 PP_GATE_DEVICES=0,1 ./scripts/pp-gate.sh`
   Expected: runs (PP_GATE_DEVICES wins, includes both); homogeneity
   check fires (gfx906 ≠ gfx90c family) → "heterogeneous ISA families
   — skipping" UNLESS `HIPFIRE_PP_GATE_HETEROGENEOUS=1` also set.

6. **Dry-run** — `./scripts/pp-gate.sh --dry-run`
   Expected: prints parsed/filtered/decision without GPU work. Exit 0.

7. **Dry-run + simulate dual-7900 XTX** — stub sysfs path or feed
   `--dry-run --simulate=0:gfx1100,1:gfx1100` (if simulate flag added;
   otherwise verify by inspection that homogeneity-check path doesn't
   fire on family-identical devices).

8. **Heterogeneous bypass simulation** —
   `--dry-run --simulate=0:gfx1100,1:gfx1101` should reach decision
   "would run" (same family, no skip). Confirms the family-grouping
   fix from B4.

9. **CI / 0-GPU sanity** — invoke in a directory without sysfs nodes
   and with rocm-smi unavailable (e.g. inside a container without
   `/sys/class/kfd/`). Expected: fallback to existing
   `HIP_VISIBLE_DEVICES` parse + skip path. Exit 0.

10. **Windows skip** — `bash -c 'function uname { echo MINGW64_NT-10.0; }; export -f uname; ./scripts/pp-gate.sh'`
    Expected: immediate exit 0 with "Windows host — skipping (Linux
    ROCm only)". No probing, no daemon spawn.

11. **Sysfs missing, rocm-smi present** — `HIPFIRE_PP_GATE_TEST_NO_SYSFS=1`
    (test env var that forces the sysfs probe to fail) → fallback
    warning fires; rocm-smi parse takes over; default behavior matches
    case 1.

## Out of scope (for this PR)

- Part (5) of the issue: per-device VRAM pre-flight. Bigger lift,
  requires either a tiny prober binary or `rocm-smi --showmeminfo vram`
  parsing + free-VRAM threshold logic. Reviews flagged this as
  "probably the real fix and the rest is bandaid"; counter-argument:
  the gfx-arch filter is cheaper, covers the immediate issue, and the
  load-failure detection (B5 + B7) is the safety net that covers gaps
  in the filter list. Revisit if a real VRAM-limited case hits in
  production (e.g. a 16 GB dGPU + heavily-used iGPU sharing RAM).

- Tests for `pp-gate.sh` itself as a separate suite. The `--dry-run`
  flag is the v2 substitute — it covers the parser logic without a
  daemon. A `bats` test suite could come later if the parser grows
  further; for now the dry-run is enough.

## Risks

- **gfx_target_version encoding edge cases.** The integer→string
  conversion is hand-coded for ~15 known archs. Unknown encodings
  print as `gfx_unknown_<int>` and bypass the iGPU filter (treated as
  potentially-dGPU). Acceptable: load-failure detection catches the
  case where it's actually an iGPU.

- **Family grouping divergence.** `kernels.rs` is the source of truth
  for WMMA family, but PP doesn't use WMMA — it uses the basic
  forward-pass kernels which have their own per-arch dispatching in
  `dispatch.rs`. If `kernels.rs:80-139` and `dispatch.rs` ever
  disagree on family membership, this script will side with
  `kernels.rs`. Plan: leave a comment pointing at both files so a
  future maintainer can re-source if needed.

- **Carve-out for gfx1150 / gfx1151 / gfx1152 untested on this host.**
  None of the Strix variants are in the dev-box matrix. Conservative
  default (kept OUT of iGPU filter, so they'd run) is what the v2
  picks; if a Strix user files a real follow-up, adjust.

- **sysfs format drift.** The `gfx_target_version` and `simd_count`
  field names have been stable since amdkfd's introduction in Linux
  3.19 (2015). Lower risk than rocm-smi format drift, but not zero.
  Fallback to rocm-smi with a loud warning covers a sysfs format
  change; the warning ensures the operator notices.

- **Validation gaps.** Cases 7, 8 require a `--simulate` flag that's
  marginal scope. If we don't ship `--simulate`, cases 7/8 fall back
  to inspection-only verification. Decision: ship `--simulate` (~20
  more lines) — it makes the dry-run actually testable, and the
  v1→v2 review process made clear that "verify by inspection" is
  unreliable.

## Commit plan

Single commit. Validation matrix MUST be run before composing the
commit message — replace the "Tested on …" line with actual results,
not aspirational copy. Suggested template:

```
fix(pp-gate): skip iGPUs and heterogeneous ISA families; surface
              zero-output as load failure

Resolves issue#216 false-positive 'pp=2 ≢ pp=1' on MI50 + Renoir
iGPU hosts. Replaces rocm-smi-only device probing with sysfs
gfx_target_version as the primary source.

Changes:

1. PP_GATE_DEVICES now filters the device list used for the count
   check, validates indices exist, and derives HIP_VISIBLE_DEVICES
   from the filtered set (replaces the hardcoded 0,1 default).
2. Default iGPU filter for known APU gfx archs (gfx902, gfx909,
   gfx90c, gfx1013, gfx1033, gfx1034, gfx1035, gfx1036, gfx1103).
   Bypass with HIPFIRE_PP_GATE_INCLUDE_IGPU=1. gfx1150/1151/1152
   excluded from filter — let homogeneity check or load-failure
   detection handle them.
3. Heterogeneous-family skip (cdna1 vs rdna3 etc.) with
   HIPFIRE_PP_GATE_HETEROGENEOUS=1 override. Family grouping sourced
   from crates/rdna-compute/src/kernels.rs:80-139 — same-family dGPUs
   (gfx1100 + gfx1101) no longer falsely skip.
4. Zero-text-event detection + empty-SHA rejection. Either pp=1 or
   pp=2 producing 0 events surfaces as "daemon load or dispatch
   failed", with per-PID stderr log path. Equal empty-SHAs no
   longer report as PASS.
5. Linux-only: clean skip on MINGW/MSYS/CYGWIN at top of script.
6. New --dry-run + --simulate flags for parser verification without
   GPU work.

Validation matrix (11 cases including dry-run, Windows, and
PP_GATE_DEVICES validation) covered in docs/plans/pp-gate-fix.md.

Verified on MI50 (gfx906) + Ryzen 4650G iGPU (gfx90c). [Fill in
actual case-by-case results here from validation run.]
```

After landing locally, open a parallel PR to
`Kaden-Schutt/hipfire#216` referencing this commit's SHA. The script
file is shared between the upstream and this fork; merging upstream
without this commit landed there will create conflicts on the next
sync.

## Open questions

1. **gfx1152 classification.** Reviews flagged the inconsistency
   between `dispatch.rs:140` (groups with RDNA3.5) and
   `kernels.rs:80-139` (excluded from WMMA). Worth a separate
   investigation, but not blocking this PR.
2. **simulate flag scope.** Should `--simulate` accept arbitrary
   device strings, or only a few canned topologies (dual-7900XTX,
   gfx906+gfx1100, etc.)? Canned is simpler; arbitrary is more
   useful for ad-hoc what-ifs. Recommend canned for v2, expand if
   needed.
3. **Family list maintenance.** Decision deferred to v2 review:
   should the bash family table cross-reference `kernels.rs` at
   commit time (a `make sync-arch-tables` pre-commit hook), or is
   the "comment points at the file" approach enough? v1 of this
   review consensus was "comment + TODO" — keep that.
