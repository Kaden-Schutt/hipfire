#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# spec_decode_genre_sweep.sh — AR-vs-spec-decode decode-throughput matrix across
# a genre battery, for 27B-dense (arch 5) and A3B-MoE (arch 6), measured on the
# DAEMON (the serving path users hit — see docs/spec-decode-durability-*.md:
# "measure durability on the daemon, never the raw demo harness"). For each model
# it loads ONE daemon session and runs AR / DFlash / MTP over every genre, then
# prints decode tok/s + ×AR + tau + coherence per genre.
#
# The per-model invariant this exists to guard (do NOT mis-read A3B):
#   * 27B dense — DFlash AND MTP decode tok/s are ABOVE AR on EVERY genre
#     (baseline AR ~43 t/s; DFlash ~1.4-2.2x, MTP ~1.4-1.8x). AR < spec always.
#   * A3B MoE   — MTP is the spec win; DFlash is NET-NEGATIVE BY DESIGN (the A3B
#     draft is weak, R-bar ~0.39, tau ~1.0), so a3b_dflash < a3b_ar is EXPECTED,
#     not a regression. A3B's spec lever is MTP, not DFlash.
#
# Runs whatever daemon is built in the CURRENT worktree
# (./target/release/daemon), so run it from spec-graph and from a merge
# branch and diff the two matrices to prove a merge preserved spec-decode tok/s.
#
# Usage:
#   scripts/spec_decode_genre_sweep.sh [--temp T] [--models 27b,a3b] [--max N]
#                                      [--out FILE] [--label NAME]
# Env: HIPFIRE_MODELS_DIR (default ~/.hipfire/models)
#
# temp 0.0 (default) = greedy (the canonical lossless durability measure).
# temp >0 = sampled spec-decode (HIPFIRE_DFLASH_FAST_SAMPLE / HIPFIRE_MTP_SAMPLED
# engage the lossless rejection-sampling paths). Compare branches at the SAME temp.
set -u
cd "$(dirname "$0")/.."

TEMP=0.0
MODELS_SEL="27b,a3b"
MAXTOK=200
LABEL=""
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --temp)   TEMP="$2"; shift 2;;
    --models) MODELS_SEL="$2"; shift 2;;
    --max)    MAXTOK="$2"; shift 2;;
    --out)    OUT="$2"; shift 2;;
    --label)  LABEL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

M="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
DAEMON="./target/release/daemon"
[ -x "$DAEMON" ] || { echo "no daemon at $DAEMON — build: cargo build --release -p hipfire-daemon" >&2; exit 2; }
RUNDIR="${HIPFIRE_SWEEP_OUT:-$PWD/target/genre_sweep}"; mkdir -p "$RUNDIR"
RES="${OUT:-$RUNDIR/genre_results.jsonl}"; : > "$RES"
[ -n "$LABEL" ] || LABEL="$(git rev-parse --short HEAD 2>/dev/null || echo local)"

# Genre battery (committed prompts). name -> prompt file.
GENRES=(
  "code|benchmarks/prompts/lru_cache_pep8_strict.txt"
  "reason|benchmarks/prompts/trains-meet.txt"
  "factual|benchmarks/prompts/bare_factual.txt"
  "prose|benchmarks/prompts/prose_river_short.txt"
  "fiction|benchmarks/prompts/fiction_lighthouse.txt"
  "instruct|benchmarks/prompts/merge_sort_thinking_off.txt"
  "agentic|benchmarks/prompts/agentic_user_multistep.txt"
  "tool|benchmarks/prompts/tool_call_read_file.txt"
)

# Per-config: name|model|draft(none=AR/MTP)|extra-env. MTP auto-loads its head
# sidecar from the base model when HIPFIRE_QWEN_MTP=1 (27b) or via a bundled
# *-mtp model (a3b). HIPFIRE_*_SAMPLED only bite at temp>0.
ALL_CONFIGS=(
  "27b_ar|$M/qwen3.6-27b.mq4|none|"
  "27b_dflash|$M/qwen3.6-27b.mq4|$M/qwen36-27b-dflash-mq4.hf4|HIPFIRE_DFLASH_FAST_SAMPLE=1"
  "27b_mtp|$M/qwen3.6-27b.mq4|none|HIPFIRE_QWEN_MTP=1 HIPFIRE_MTP_SAMPLED=1"
  "a3b_ar|$M/qwen3.6-35b-a3b.mq4r|none|"
  "a3b_dflash|$M/qwen3.6-35b-a3b.mq4r|$M/qwen36-35b-a3b-dflash-mq4.hf4|HIPFIRE_DFLASH_FAST_SAMPLE=1"
  "a3b_mtp|$M/qwen3.6-35b-a3b.mq4r-cvs-mtp|none|HIPFIRE_QWEN_MTP=1 HIPFIRE_MTP_SAMPLED=1"
)

# Filter configs by --models selection.
CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
  pfx="${cfg%%_*}"
  case ",$MODELS_SEL," in *",$pfx,"*) CONFIGS+=("$cfg");; esac
done
[ "${#CONFIGS[@]}" -gt 0 ] || { echo "no configs match --models $MODELS_SEL" >&2; exit 2; }

GENRE_NAMES="$(printf '%s\n' "${GENRES[@]}" | cut -d'|' -f1 | paste -sd, -)"

. ./scripts/gpu-lock.sh
gpu_acquire "genre-sweep" || { echo "GPU lock failed" >&2; exit 2; }
trap 'gpu_release 2>/dev/null || true' EXIT

echo "===================================================================="
echo "spec-decode genre sweep   label=$LABEL   temp=$TEMP   max=$MAXTOK"
echo "daemon=$DAEMON  genres=$GENRE_NAMES"
echo "===================================================================="

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name model draft env <<< "$cfg"
  [ -f "$model" ] || { echo "[$name] SKIP — model missing: $model"; continue; }
  if [ "$draft" != "none" ] && [ ! -f "$draft" ]; then echo "[$name] SKIP — draft missing: $draft"; continue; fi
  sess="$RUNDIR/sess_$name.jsonl"; log="$RUNDIR/log_$name.log"

  # --- build the daemon session (load -> warm -> one generate per genre -> unload)
  TEMP="$TEMP" MAXTOK="$MAXTOK" python3 - "$model" "$draft" "$sess" <<'PY' "${GENRES[@]}"
import sys, json, os
model, draft, out = sys.argv[1], sys.argv[2], sys.argv[3]
genres = [g.split("|", 1) for g in sys.argv[4:]]
temp = float(os.environ["TEMP"]); mx = int(os.environ["MAXTOK"])
params = {"max_seq": 2048, "kv_mode": "q8"}
if draft != "none":
    params["draft"] = draft
L = [json.dumps({"type": "load", "model": model, "params": params})]
L.append(json.dumps({"type": "generate", "id": "warm",
                     "prompt": open(genres[0][1]).read(),
                     "temperature": temp, "top_p": 0.8, "max_tokens": 16}))
for gname, gp in genres:
    L.append(json.dumps({"type": "generate", "id": gname, "prompt": open(gp).read(),
                         "temperature": temp, "top_p": 0.8, "max_tokens": mx}))
L.append(json.dumps({"type": "unload"}))
open(out, "w").write("\n".join(L) + "\n")
PY

  echo "[$name] running ($model${draft:+ + draft})..."
  # shellcheck disable=SC2086
  env $env HIPFIRE_EMIT_TOKEN_IDS=1 timeout 900 "$DAEMON" < "$sess" > "$log" 2>&1
  ec=$?

  # --- parse done events -> per-genre tok/s, tau, spec-engaged, coherence
  python3 - "$log" "$name" "$RES" "$GENRE_NAMES" "$ec" <<'PY'
import json, sys, collections
log, name, res, gn, ec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5]
done = {}; txt = collections.defaultdict(list)
for line in open(log, errors="replace"):
    line = line.strip()
    if not line.startswith("{"): continue
    try: m = json.loads(line)
    except Exception: continue
    i = str(m.get("id", ""))
    if m.get("type") == "done" and i in gn:
        done[i] = {"dec": m.get("decode_tok_s") or m.get("tok_s"),
                   "tau": m.get("tau") or m.get("decode_tau"),
                   "spec": bool(m.get("dflash") or m.get("mtp"))}
    if m.get("type") in ("token", "committed"):
        txt[i].append(m.get("text", ""))
n_ok = 0
for g in gn:
    d = done.get(g, {}); s = "".join(txt.get(g, [])); w = s.split(); n = len(w)
    if n >= 128:
        last = w[-128:]; u2 = len(set(last)) / 128
        f2 = max((last.count(x) for x in set(last)), default=0) / 128
        coh = u2 >= 0.3 and f2 <= 0.5
    else:
        u2 = (len(set(w)) / n) if n else 0; coh = u2 >= 0.3
    if d.get("dec"): n_ok += 1
    with open(res, "a") as fh:
        fh.write(json.dumps({"cfg": name, "genre": g, "dec": d.get("dec"),
                             "tau": d.get("tau"), "spec": d.get("spec"),
                             "toks": n, "coh": coh, "uniq": round(u2, 2)}) + "\n")
print(f"[{name}] ec={ec} parsed {n_ok}/{len(gn)} genres")
PY
done

echo
echo "=== GENRE SWEEP MATRIX  (label=$LABEL, temp=$TEMP, daemon) ==="
python3 - "$RES" "$GENRE_NAMES" <<'PY'
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
gn = sys.argv[2].split(",")
by = collections.defaultdict(dict)
for r in R:
    model = r["cfg"].split("_")[0]; mode = r["cfg"].split("_", 1)[1]
    by[(model, r["genre"])][mode] = r
models = []
for r in R:
    mm = r["cfg"].split("_")[0]
    if mm not in models: models.append(mm)
for model in models:
    print(f"\n--- {model.upper()}  decode tok/s  (xAR | tau | coh) ---")
    print(f"{'genre':9s} | {'AR':>7s} | {'DFlash':>24s} | {'MTP':>24s}")
    print("-" * 74)
    fails = []
    for g in gn:
        ar = by[(model, g)].get("ar", {}); arv = ar.get("dec") or 0
        def cell(mode):
            r = by[(model, g)].get(mode)
            if not r or not r.get("dec"): return f"{'-':>24s}"
            sp = r["dec"] / arv if arv else 0
            tau = r.get("tau") or 0
            return f"{r['dec']:6.1f} ({sp:.2f}x t{tau:.1f} {'OK' if r['coh'] else 'LOW'})".rjust(24)
        # invariant check: 27b => DFlash AND MTP must beat AR; a3b => MTP only.
        for mode in ("dflash", "mtp"):
            r = by[(model, g)].get(mode)
            if r and r.get("dec") and arv:
                ratio = r["dec"] / arv
                must_beat = (model != "a3b") or (mode == "mtp")
                if must_beat and ratio < 1.0:
                    fails.append(f"{model}/{g}/{mode} {ratio:.2f}x < AR")
        print(f"{g:9s} | {arv:7.1f} | {cell('dflash')} | {cell('mtp')}")
    note = " (a3b_dflash<AR is BY DESIGN, excluded)" if model == "a3b" else ""
    if fails:
        print(f"  ** INVARIANT MISS{note}: " + "; ".join(fails))
    else:
        print(f"  invariant OK: spec > AR where required{note}")
PY
echo "[genre-sweep] DONE — results: $RES"
