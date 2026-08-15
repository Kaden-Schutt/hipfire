#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Length sweep: parent vs MQ2R vs MQ2-Lloyd at 128/256/512/1024 on shared prefixes.
set -euo pipefail
export PATH="${PATH}:/root/.cargo/bin:/opt/rocm/core-7.14/bin"
ROOT=/root/hipfire-work/ds4-parent-gate
BASE=/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline
OUT=$BASE/length_sweep
mkdir -p "$OUT"
cd "$ROOT"
WALL0=$(date +%s)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

PARENT_BIN=$ROOT/target/release/examples/ds4_parent_forward_gate
QUANT_MULTI=$ROOT/target/release/examples/ds4_quant_plog_multi
QUANT_ONE=$ROOT/target/release/examples/ds4_quant_plog
KLD=$ROOT/target/release/examples/ds4_parent_kld
POS_SCAN=/root/plog_pos_scan.py

MQ2R=/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r
MQ2R_SHA=cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce
LLOYD=/mnt/scratch/quantization/deepseek-v4-flash-0731-mq2lloyd/artifacts/deepseek-v4-flash-0731.mq2lloyd
LLOYD_SHA=a6195336fc5505e5612678d16dde68ec66cc0b4011e0526d491f774613f9e075

LENGTHS=(128 256 512 1024)

wait_gpu_free() {
  local tag=$1
  while true; do
    local busy=0
    for p in /proc/[0-9]*; do
      exe=$(readlink "$p/exe" 2>/dev/null || true)
      case "$exe" in
        *ds4_parent_forward_gate*|*ds4_quant_plog*|*ds4_quant_plog_multi*) busy=1 ;;
      esac
    done
    local vram
    vram=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/{print $NF; exit}')
    if [[ "$busy" = "0" && "${vram:-0}" -lt 2000000000 ]]; then
      log "GPU free for $tag (vram=$vram)"
      return 0
    fi
    log "waiting GPU for $tag busy=$busy vram=$vram"
    sleep 15
  done
}

# ── Token prefixes ──────────────────────────────────────────────────────────
log "=== token prefixes ==="
{
  echo "token_files"
  for n in "${LENGTHS[@]}"; do
    if [[ $n -eq 1024 ]]; then
      f=$BASE/tokens.bin
    else
      f=$BASE/tokens_${n}.bin
    fi
    # ensure prefix of full
    python3 - <<PY
import hashlib
base="$BASE"
n=$n
full=open(f"{base}/tokens.bin","rb").read()
path="$f"
data=open(path,"rb").read()
assert data==full[:n*4], f"prefix fail n={n}"
print(f"n={n} path={path} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()} prefix_ok=1")
PY
  done
} | tee "$OUT/tokens_sha.txt"

# ── Ensure parent plogs ─────────────────────────────────────────────────────
# Reuse verified captures when token sha matches.
declare -A PARENT_PLOG
PARENT_PLOG[128]=$BASE/parent_128.plog
PARENT_PLOG[256]=$BASE/parent_256_diag.plog
PARENT_PLOG[512]=$BASE/parent_512.plog
PARENT_PLOG[1024]=$BASE/parent_1024.plog

need_parent=()
for n in "${LENGTHS[@]}"; do
  p=${PARENT_PLOG[$n]}
  if [[ -f "$p" ]]; then
    log "parent n=$n exists: $p ($(stat -c%s "$p") bytes)"
  else
    need_parent+=("$n")
    log "parent n=$n MISSING"
  fi
done

for n in "${need_parent[@]+"${need_parent[@]}"}"; do
  wait_gpu_free "parent_$n"
  if [[ $n -eq 1024 ]]; then
    tok=$BASE/tokens.bin
  else
    tok=$BASE/tokens_${n}.bin
  fi
  plog=$BASE/parent_${n}.plog
  man=$BASE/parent_${n}.manifest.json
  log "running parent n=$n"
  "$PARENT_BIN" \
    --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
    --token-ids "$tok" \
    --plog "$plog" \
    --manifest "$man" \
    --skip-shard-hashes \
    > "$BASE/parent_${n}.log" 2>&1
  PARENT_PLOG[$n]=$plog
  log "parent n=$n done"
done

# If parent_512 was started externally, wait for it.
if [[ ! -f ${PARENT_PLOG[512]} ]]; then
  log "waiting for external parent_512..."
  while [[ ! -f $BASE/parent_512.plog ]]; do
    # still running?
    if ! pgrep -f ds4_parent_forward_gate >/dev/null 2>&1; then
      if [[ -f $BASE/parent_512.plog ]]; then break; fi
      log "parent_512 process gone without plog — relaunch"
      wait_gpu_free parent_512
      "$PARENT_BIN" \
        --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
        --token-ids $BASE/tokens_512.bin \
        --plog $BASE/parent_512.plog \
        --manifest $BASE/parent_512.manifest.json \
        --skip-shard-hashes \
        > $BASE/parent_512.log 2>&1
      break
    fi
    sleep 10
  done
  PARENT_PLOG[512]=$BASE/parent_512.plog
fi

# ── Quant multi-length (one model at a time) ────────────────────────────────
# Skip 1024 if already present (reuse verified baseline).
run_quant_multi() {
  local name=$1 model=$2 expect=$3
  local args=(--model "$model" --expect-sha256 "$expect")
  local need=0
  for n in "${LENGTHS[@]}"; do
    local plog=$BASE/${name}_${n}.plog
    if [[ -f "$plog" && $n -eq 1024 ]]; then
      log "$name n=$n reuse existing $(stat -c%s "$plog") bytes"
      continue
    fi
    if [[ -f "$plog" && $n -ne 1024 ]]; then
      # re-capture shorter lengths only if missing; keep if present
      log "$name n=$n already present — skip"
      continue
    fi
    need=1
    if [[ $n -eq 1024 ]]; then
      args+=(--token-ids "${n}=$BASE/tokens.bin" --plog "${n}=$plog")
    else
      args+=(--token-ids "${n}=$BASE/tokens_${n}.bin" --plog "${n}=$plog")
    fi
  done
  if [[ $need -eq 0 ]]; then
    log "$name: all lengths present"
    return 0
  fi
  wait_gpu_free "$name"
  log "=== $name multi-length capture ==="
  # If only some lengths missing, build arg list of missing only.
  args=(--model "$model" --expect-sha256 "$expect")
  for n in "${LENGTHS[@]}"; do
    local plog=$BASE/${name}_${n}.plog
    if [[ -f "$plog" ]]; then
      continue
    fi
    if [[ $n -eq 1024 ]]; then
      args+=(--token-ids "${n}=$BASE/tokens.bin" --plog "${n}=$plog")
    else
      args+=(--token-ids "${n}=$BASE/tokens_${n}.bin" --plog "${n}=$plog")
    fi
  done
  # if args only has model+sha, nothing to do
  if [[ ${#args[@]} -le 4 ]]; then
    log "$name: nothing missing after recheck"
    return 0
  fi
  "$QUANT_MULTI" "${args[@]}" 2>&1 | tee "$BASE/${name}_multi.log"
  log "$name multi done"
}

run_quant_multi mq2r "$MQ2R" "$MQ2R_SHA"
run_quant_multi mq2lloyd "$LLOYD" "$LLOYD_SHA"

# ── KLD matrix ──────────────────────────────────────────────────────────────
log "=== KLD matrix ==="
{
  echo "length,ref,cand,kld_mean,kld_p50,kld_p95,kld_max,top1,top5,ppl_ref,ppl_cand,max_abs_dlogit"
  for n in "${LENGTHS[@]}"; do
    if [[ $n -eq 1024 ]]; then tok=$BASE/tokens.bin; else tok=$BASE/tokens_${n}.bin; fi
    pref=${PARENT_PLOG[$n]}
    for cand_name in mq2r mq2lloyd; do
      cand=$BASE/${cand_name}_${n}.plog
      json=$OUT/kld_parent_vs_${cand_name}_${n}.json
      log "KLD parent vs $cand_name n=$n"
      "$KLD" --reference "$pref" --candidate "$cand" --tokens "$tok" --json "$json" \
        | tee "$OUT/kld_parent_vs_${cand_name}_${n}.txt"
      python3 - <<PY
import json
j=json.load(open("$json"))
# report may be nested
r=j.get("report", j)
def g(*keys, default=None):
  for k in keys:
    if k in r: return r[k]
    if k in j: return j[k]
  return default
print(",".join(str(x) for x in [
  $n, "parent", "$cand_name",
  g("kld_mean"), g("kld_p50"), g("kld_p95"), g("kld_max"),
  g("top1_agreement"), g("top5_overlap_mean"),
  g("ppl_reference"), g("ppl_candidate"), g("max_abs_logit_delta"),
]))
PY
    done
    # quant vs quant control
    json=$OUT/kld_mq2r_vs_mq2lloyd_${n}.json
    log "KLD mq2r vs mq2lloyd n=$n"
    "$KLD" --reference $BASE/mq2r_${n}.plog --candidate $BASE/mq2lloyd_${n}.plog \
      --tokens "$tok" --json "$json" | tee "$OUT/kld_mq2r_vs_mq2lloyd_${n}.txt"
    python3 - <<PY
import json
j=json.load(open("$json"))
r=j.get("report", j)
def g(*keys, default=None):
  for k in keys:
    if k in r: return r[k]
    if k in j: return j[k]
  return default
print(",".join(str(x) for x in [
  $n, "mq2r", "mq2lloyd",
  g("kld_mean"), g("kld_p50"), g("kld_p95"), g("kld_max"),
  g("top1_agreement"), g("top5_overlap_mean"),
  g("ppl_reference"), g("ppl_candidate"), g("max_abs_logit_delta"),
]))
PY
  done
} | tee "$OUT/kld_table.csv"

# ── Position bucket scans ───────────────────────────────────────────────────
log "=== position bucket scans ==="
for n in "${LENGTHS[@]}"; do
  if [[ $n -eq 1024 ]]; then tok=$BASE/tokens.bin; else tok=$BASE/tokens_${n}.bin; fi
  pref=${PARENT_PLOG[$n]}
  {
    echo "=== pos_scan n=$n ==="
    python3 "$POS_SCAN" \
      "$pref" \
      $BASE/mq2r_${n}.plog \
      $BASE/mq2lloyd_${n}.plog \
      "$tok"
  } | tee "$OUT/pos_scan_${n}.txt"
done

# ── Final summary table ─────────────────────────────────────────────────────
log "=== summary ==="
python3 - <<'PY' | tee "$OUT/SUMMARY.txt"
import json, os, re, time
from pathlib import Path
base = Path("/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline")
out = base / "length_sweep"
lengths = [128, 256, 512, 1024]
parent_map = {
  128: base/"parent_128.plog",
  256: base/"parent_256_diag.plog",
  512: base/"parent_512.plog",
  1024: base/"parent_1024.plog",
}
print("TOKEN PREFIXES")
print(open(out/"tokens_sha.txt").read())
print()
print("PPL / KLD / TOP1 TABLE")
print(f"{'len':>5} {'model':>10} {'PPL':>12} {'vs':>10} {'KLD_mean':>12} {'top1':>8} {'PPL_gap':>10}")
print("-"*75)
rows = []
for n in lengths:
  cells = {}
  for cand in ("mq2r", "mq2lloyd"):
    jp = out / f"kld_parent_vs_{cand}_{n}.json"
    j = json.load(open(jp))
    r = j.get("report", j)
    cells[cand] = r
  # quant-quant
  jq = out / f"kld_mq2r_vs_mq2lloyd_{n}.json"
  qq = json.load(open(jq)).get("report", json.load(open(jq)))
  # parent ppl from either report
  ppl_p = cells["mq2r"].get("ppl_reference")
  print(f"{n:5d} {'parent':>10} {ppl_p:12.6f}")
  for cand in ("mq2r", "mq2lloyd"):
    r = cells[cand]
    gap = ppl_p - r["ppl_candidate"]
    print(f"{n:5d} {cand:>10} {r['ppl_candidate']:12.6f} {'parent':>10} {r['kld_mean']:12.6f} {r['top1_agreement']:8.4f} {gap:10.4f}")
  print(f"{n:5d} {'q-vs-q':>10} {qq['ppl_candidate']:12.6f} {'mq2r':>10} {qq['kld_mean']:12.6f} {qq['top1_agreement']:8.4f}")
  print()
  rows.append((n, ppl_p, cells, qq))

print("GAP vs LENGTH (parent_ppl - quant_ppl)")
print(f"{'len':>5} {'parent':>12} {'mq2r':>12} {'lloyd':>12} {'gap_mq2r':>12} {'gap_lloyd':>12} {'kld_mq2r':>12} {'kld_lloyd':>12} {'qq_kld':>10}")
for n, ppl_p, cells, qq in rows:
  pr = cells["mq2r"]["ppl_candidate"]
  pl = cells["mq2lloyd"]["ppl_candidate"]
  print(f"{n:5d} {ppl_p:12.6f} {pr:12.6f} {pl:12.6f} {ppl_p-pr:12.6f} {ppl_p-pl:12.6f} {cells['mq2r']['kld_mean']:12.6f} {cells['mq2lloyd']['kld_mean']:12.6f} {qq['kld_mean']:10.6f}")

print()
print("POSITION BUCKETS")
for n in lengths:
  print(f"--- n={n} ---")
  print(open(out/f"pos_scan_{n}.txt").read())
PY

WALL1=$(date +%s)
echo "WALL_SECONDS=$((WALL1-WALL0))" | tee -a "$OUT/SUMMARY.txt"
log "DONE wall=$((WALL1-WALL0))s  artifacts in $OUT"
