#!/usr/bin/env bash
# oracle_profile.sh <arch> <dev> <card> <model_path> [max_gen]
# CALIBRATED decode profiler: kernel-trace wall% (high DPM) + census busy/L2/occ
# ratios (profile_standard) -> per-kernel raw ratios + a DERIVED roofline lens.
# Counters validated to revive under profile_standard on all 5 RDNA gens
# (GL2C_HIT/MISS/MemUnitBusy/OccupancyPercent/SQ_BUSY_CYCLES universal; VALU cycles
# gfx1201-only). Emits JSON to stdout + /tmp/oracle-prof-<arch>.json. Data-not-tags:
# stores raw ratios; bound_class is a query-time lens, kept here only as a hint.
set -u
arch=$1; dev=$2; card=$3; mp=$4; maxg=${5:-24}
cd "${HIPFIRE_REPO:-$HOME/hipfire}"   # profile a worktree's daemon when HIPFIRE_REPO is set
D=/opt/rocm/bin/rocprofv3
BIN=./target/release/examples/daemon
[ -f "$mp" ] || { echo "{\"arch\":\"$arch\",\"error\":\"model absent: $mp\"}"; exit 0; }
PL=/sys/class/drm/card$card/device/power_dpm_force_performance_level
# universal counters (revive on all gens) + gfx1201-only VALU cycles (rocprofv3
# errors on an unknown counter, so add it only for RDNA4).
PMC="GL2C_HIT GL2C_MISS MemUnitBusy OccupancyPercent SQ_BUSY_CYCLES"
[ "$arch" = "gfx1201" ] && PMC="$PMC SQ_INST_CYCLES_VALU GL2C_EA_RDREQ"
req(){ printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}\n{"type":"generate","id":"r","prompt":"Explain how a hash map resolves collisions, in two sentences.","temperature":0.0,"max_tokens":%s}\n{"type":"unload"}\n' "$mp" "$1"; }
pkill -9 -f "examples/daemon" 2>/dev/null; rm -f ~/.hipfire/daemon.pid 2>/dev/null; sleep 1
echo auto | sudo tee "$PL" >/dev/null   # NEVER force high (under-clocks RDNA3/4); auto = representative wall%
req 8 | HIP_VISIBLE_DEVICES=$dev $BIN >/dev/null 2>&1   # warm (JIT)
req "$maxg" > /tmp/req-$arch.jsonl
# 1) kernel-trace under AUTO (wall-time attribution; wall% is relative, auto is representative)
rm -rf /tmp/kt-$arch; mkdir -p /tmp/kt-$arch
HIP_VISIBLE_DEVICES=$dev timeout 400 $D --kernel-trace -f csv -d /tmp/kt-$arch -- \
  bash -c "$BIN < /tmp/req-$arch.jsonl >/dev/null 2>&1" >/dev/null 2>&1
# 2) census under PROFILE_STANDARD (ungates perfmon)
echo profile_standard | sudo tee "$PL" >/dev/null
rm -rf /tmp/pmc-$arch; mkdir -p /tmp/pmc-$arch
HIP_VISIBLE_DEVICES=$dev timeout 900 $D -f csv --pmc $PMC \
  -d /tmp/pmc-$arch -- bash -c "$BIN < /tmp/req-$arch.jsonl >/dev/null 2>&1" >/dev/null 2>&1
echo auto | sudo tee "$PL" >/dev/null
# 3) parse (column-by-name) -> raw ratios + derived roofline lens
python3 - "$arch" "$mp" <<'PY'
import csv,glob,os,re,collections,json,sys
arch,mp=sys.argv[1],sys.argv[2]
def load(pats):
    for p in pats:
        fs=sorted(glob.glob(p,recursive=True))
        if fs:
            try: return list(csv.DictReader(open(fs[0])))
            except Exception: pass
    return []
def col(hdr,want):
    for k in hdr:
        if k and k.strip().lower()==want: return k
kt=load([f"/tmp/kt-{arch}/**/*kernel_trace*.csv",f"/tmp/kt-{arch}/**/*.csv"])
wall=collections.defaultdict(float); cnt=collections.defaultdict(int); tot=0.0
if kt:
    h=list(kt[0]); kn=col(h,'kernel_name'); st=col(h,'start_timestamp'); en=col(h,'end_timestamp')
    for r in kt:
        try: d=int(r[en])-int(r[st])
        except Exception: d=0
        n=re.sub(r'\(.*','',(r.get(kn,'') or '')).strip()
        if n: wall[n]+=d; cnt[n]+=1; tot+=d
pmc=load([f"/tmp/pmc-{arch}/**/*counter_collection*.csv",f"/tmp/pmc-{arch}/**/*.csv"])
acc=collections.defaultdict(lambda: collections.defaultdict(lambda:[0.0,0]))
meta=collections.defaultdict(dict)  # per-kernel static resources (VGPR/SGPR/LDS/scratch) = the WHY
if pmc:
    h=list(pmc[0]); kn=col(h,'kernel_name'); cn=col(h,'counter_name'); cv=col(h,'counter_value')
    vgc=col(h,'vgpr_count'); sgc=col(h,'sgpr_count'); ldc=col(h,'lds_block_size'); scc=col(h,'scratch_size'); avc=col(h,'accum_vgpr_count')
    def gi(r,c):
        try: return int(float(r.get(c,0) or 0))
        except Exception: return None
    for r in pmc:
        n=re.sub(r'\(.*','',(r.get(kn,'') or '')).strip()
        try: v=float(r.get(cv,'0') or 0)
        except Exception: v=0.0
        if n and cn: acc[n][r[cn]][0]+=v; acc[n][r[cn]][1]+=1
        if n and vgc and n not in meta:
            meta[n]={'vgpr':gi(r,vgc),'accum_vgpr':gi(r,avc),'sgpr':gi(r,sgc),'lds':gi(r,ldc),'scratch':gi(r,scc)}
# MemUnitBusy/OccupancyPercent are per-dispatch percentages -> AVERAGE across
# dispatches; GL2C_*/SQ_* are raw accumulators -> SUM.
PCT={'MemUnitBusy','OccupancyPercent'}
def g(n,k):
    e=acc[n].get(k)
    if not e: return None
    return (e[0]/e[1]) if (k in PCT and e[1]) else e[0]
def lens(n):
    hit=g(n,'GL2C_HIT'); miss=g(n,'GL2C_MISS'); mem=g(n,'MemUnitBusy'); occ=g(n,'OccupancyPercent')
    l2=(100.0*hit/(hit+miss)) if (hit is not None and miss is not None and (hit+miss)>0) else None
    if mem is None and occ is None: return 'unmeasured', l2
    # derived lens (query-time; hint only): occupancy-starved > DRAM-traffic > L2-resident
    if occ is not None and occ<15: return 'latency/occ-starved', l2
    if l2 is not None and l2<45 and (mem or 0)>25: return 'DRAM-traffic-bound', l2
    if l2 is not None and l2>=70 and (mem or 0)>40: return 'L2-resident/mem-busy', l2
    if (mem or 0)>25: return 'mem-active(mixed)', l2
    return 'compute/other', l2
rows=[]
for n,d in sorted(wall.items(),key=lambda x:-x[1])[:12]:
    kls,l2=lens(n)
    m=meta.get(n,{})
    rows.append({'kernel':n[:60],'wall_pct':round(100*d/tot,1) if tot else 0.0,'n':cnt[n],
                 'l2_hit_pct':round(l2,1) if l2 is not None else None,
                 'mem_busy':round(g(n,'MemUnitBusy'),1) if g(n,'MemUnitBusy') is not None else None,
                 'occ':round(g(n,'OccupancyPercent'),1) if g(n,'OccupancyPercent') is not None else None,
                 'dram_miss':g(n,'GL2C_MISS'),'valu_cyc':g(n,'SQ_INST_CYCLES_VALU'),
                 'vgpr':m.get('vgpr'),'accum_vgpr':m.get('accum_vgpr'),'sgpr':m.get('sgpr'),'lds':m.get('lds'),'scratch':m.get('scratch'),
                 'roofline':kls})
out={'arch':arch,'model':os.path.basename(mp),'kt_dispatches':sum(cnt.values()),
     'census_kernels':len(acc),'rows':rows}
json.dump(out,open(f"/tmp/oracle-prof-{arch}.json","w"),indent=1)
print(json.dumps(out))
PY
