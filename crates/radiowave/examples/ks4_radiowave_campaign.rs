// SPDX-License-Identifier: Apache-2.0
// ks4-radiowave campaign driver (Slice S3): candidate matrix -> radiowave
// compile (identity + ISA) -> env re-export -> fresh-process parity ->
// fresh-process bench -> JSONL append.
//
// Bridge strategy (frozen): compile each candidate through the radiowave
// Compiler (arch gfx1100, wave32, scheduler profile, defines) to record
// identity + ISA, then re-export the SAME variant as HIPFIRE_KS4_DEFINES /
// HIPFIRE_SCHED_PROFILE and run the UNMODIFIED claim-scoped batteries in
// fresh processes: parity test_mq4v2_residual_ksplit_gfx1100.rs (gate
// relL2 <= 5e-5) + bench_dflash_verify_shapes (N=16 residual rows).
// No harness forks.
//
// Discipline: fresh process per candidate step (compile, parity, bench are
// all separate processes), batteries already do 32 warmups + 200x3
// interleaved, report MIN + median, thermal noted per row, model md5 +
// binary md5 per row. VALIDATION.md: retired coherence-gate scripts are
// never acceptance.
//
// Usage (on hipx, gfx1100):
//   export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core PATH=/opt/rocm/bin:$PATH
//   campaign run <id> <defines> <sched> <recipe>
//     <defines>: space-separated NAME[=value] tokens for HIPFIRE_KS4_DEFINES
//                ("" for none); <sched>: default|iterative_ilp|memory_clause|
//                pipeline_ilp for HIPFIRE_SCHED_PROFILE; <recipe>: radiowave
//                recipe id or "-" for none.
//   campaign matrix   # print candidate table
//
// Env overrides: KS4_REPO (default $HOME/ks4-sweep), KS4_TARGET (default
// $HOME/ks4-target), KS4_LEDGER (default repo
// docs/investigations/2026-09-04-ks4-radiowave-ledger.jsonl), KS4_EVIDENCE
// (default repo docs/investigations/2026-09-04-ks4-radiowave-evidence),
// KS4_MODEL (default $HOME/.hipfire/models/qwen3.8-27b.mq4).
//
// Sweep order (frozen): load-form 3x3 grid -> vgpr 48/40/32 x best-load ->
// sched ilp/mem/pipe x best-load -> unroll 2/4 x best-load. Early-stop at
// >=15% down_proj win, then go-deep (extra runs of the winner).

use std::process::{Command, Stdio};
use std::path::PathBuf;

const ARCH: &str = "gfx1100";
// Dispatched kw at the benched shapes: out_proj K=6144 -> 4, down_proj
// K=17408 -> 4 (68 groups; want-8 falls back the [8,4,2] ladder). The
// focused ISA symbol is therefore always ks4_lds.
const FOCUS_KERNEL: &str = "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds";
const KSPLIT_HIP_SRC: &str = "kernels/src/gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds.hip";

/// One sweep candidate: (id, HIPFIRE_KS4_DEFINES tokens, sched profile, recipe).
pub fn matrix() -> Vec<(&'static str, &'static str, &'static str, Option<&'static str>)> {
    vec![
        // Load-form 3x3 grid (W x X in {0,1,2}), both-off baseline first.
        ("base", "", "default", None),
        ("w1", "KS4_W_LOADFORM=1", "default", Some("hipfire.ks4residual.buffer_w_b32")),
        ("w2", "KS4_W_LOADFORM=2", "default", Some("hipfire.ks4residual.aligned_w_b128")),
        ("x1", "KS4_X_LOADFORM=1", "default", Some("hipfire.ks4residual.buffer_x_b32")),
        ("x2", "KS4_X_LOADFORM=2", "default", Some("hipfire.ks4residual.aligned_x_b128")),
        (
            "w1x1",
            "KS4_W_LOADFORM=1 KS4_X_LOADFORM=1",
            "default",
            Some("hipfire.ks4residual.buffer_wx_b32"),
        ),
        (
            "w2x2",
            "KS4_W_LOADFORM=2 KS4_X_LOADFORM=2",
            "default",
            Some("hipfire.ks4residual.aligned_wx_b128"),
        ),
        ("w1x2", "KS4_W_LOADFORM=1 KS4_X_LOADFORM=2", "default", None),
        ("w2x1", "KS4_W_LOADFORM=2 KS4_X_LOADFORM=1", "default", None),
        // vgpr x best-load (best-load defines prepended by the operator).
        ("vgpr48", "KS4_VGPR_CAP=48", "default", None),
        ("vgpr40", "KS4_VGPR_CAP=40", "default", Some("hipfire.ks4residual.vgpr_cap")),
        ("vgpr32", "KS4_VGPR_CAP=32", "default", None),
        // scheduler x best-load.
        ("sched_ilp", "", "iterative_ilp", None),
        ("sched_mem", "", "memory_clause", Some("hipfire.ks4residual.sched_profile")),
        ("sched_pipe", "", "pipeline_ilp", None),
        // unroll x best-load.
        ("unroll2", "KS4_UNROLL=2", "default", Some("hipfire.ks4residual.unroll")),
        ("unroll4", "KS4_UNROLL=4", "default", None),
    ]
}

fn repo() -> PathBuf {
    std::env::var_os("KS4_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("ks4-sweep"))
}

fn sh(cmd: &str, args: &[&str], env: &[(String, String)], cwd: &std::path::Path) -> (bool, String) {
    let mut c = Command::new(cmd);
    c.args(args).current_dir(cwd).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().expect("spawn");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), s)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mode = argv.get(1).map(|s| s.as_str()).unwrap_or("");
    if mode == "matrix" {
        for (id, defines, sched, recipe) in matrix() {
            println!(
                "{id}\tdefines=[{defines}]\tsched={sched}\trecipe={}",
                recipe.unwrap_or("-")
            );
        }
        return;
    }
    assert!(
        argv.len() == 6 && mode == "run",
        "usage: campaign run <id> <defines> <sched> <recipe|->"
    );
    let (id, defines, sched, recipe) = (&argv[2], &argv[3], &argv[4], &argv[5]);
    let recipe: Option<&str> = if recipe == "-" { None } else { Some(recipe) };
    let recipe_py = match recipe {
        Some(s) => format!("'{s}'"),
        None => "None".to_owned(),
    };
    let r = repo();
    let target = std::env::var_os("KS4_TARGET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("ks4-target"));
    let model = std::env::var("KS4_MODEL").unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap())
            .join(".hipfire/models/qwen3.8-27b.mq4")
            .display()
            .to_string()
    });
    let evroot = std::env::var_os("KS4_EVIDENCE").map(PathBuf::from).unwrap_or_else(|| {
        r.join("docs/investigations/2026-09-04-ks4-radiowave-evidence")
    });
    let evdir = evroot.join(id);
    std::fs::create_dir_all(&evdir).unwrap();
    let ledger = std::env::var_os("KS4_LEDGER").map(PathBuf::from).unwrap_or_else(|| {
        r.join("docs/investigations/2026-09-04-ks4-radiowave-ledger.jsonl")
    });

    let parity_bin = target.join("release/examples/test_mq4v2_residual_ksplit_gfx1100");
    let bench_bin = target.join("release/examples/bench_dflash_verify_shapes");
    let radiowave_bin = target.join("release/radiowave");
    let bin_md5 = |p: &std::path::Path| {
        let (ok, out) = sh("md5sum", &[&p.display().to_string()], &[], &r);
        assert!(ok);
        out.split_whitespace().next().unwrap().to_owned()
    };
    let parity_md5 = bin_md5(&parity_bin);
    let bench_md5 = bin_md5(&bench_bin);
    // Model md5 cached once per evidence root (multi-GB file; inputs are
    // otherwise code-fixed seeds, so bytes are identical across runs).
    let model_md5_file = evroot.join("model.md5");
    let model_md5 = if model_md5_file.exists() {
        std::fs::read_to_string(&model_md5_file).unwrap().trim().to_owned()
    } else {
        let (ok, out) = sh("md5sum", &[&model], &[], &r);
        assert!(ok, "md5sum model");
        let h = out.split_whitespace().next().unwrap().to_owned();
        std::fs::write(&model_md5_file, format!("{h}  {model}\n")).unwrap();
        h
    };

    // ---- 1. radiowave compile (identity + ISA) ----
    let obj = evdir.join(format!("ks4_{id}.o"));
    let mut cargs: Vec<String> = vec![
        "compile".into(),
        "--source".into(),
        r.join(KSPLIT_HIP_SRC).display().to_string(),
        "--arch".into(),
        ARCH.into(),
        "--wave32".into(),
        "--scheduler-profile".into(),
        sched.clone(),
        "--output".into(),
        obj.display().to_string(),
    ];
    for tok in defines.split_whitespace() {
        cargs.push("--define".into());
        cargs.push(tok.into());
    }
    let cargs_ref: Vec<&str> = cargs.iter().map(|s| s.as_str()).collect();
    let (ok, cout) = sh(&radiowave_bin.display().to_string(), &cargs_ref, &[], &r);
    std::fs::write(evdir.join("compile.log"), &cout).unwrap();
    assert!(ok, "radiowave compile failed for {id}:\n{cout}");
    let manifest = obj.with_extension("radiowave.json");
    std::fs::copy(&manifest, evdir.join("manifest.json")).unwrap();

    // ---- 2. fresh-process parity + bench under the SAME variant ----
    let mut env: Vec<(String, String)> = vec![
        ("ROCM_PATH".into(), "/opt/rocm/core".into()),
        ("HIP_PATH".into(), "/opt/rocm/core".into()),
        ("PATH".into(), "/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin".into()),
    ];
    if !defines.is_empty() {
        env.push(("HIPFIRE_KS4_DEFINES".into(), defines.clone()));
    }
    if sched != "default" {
        env.push(("HIPFIRE_SCHED_PROFILE".into(), sched.clone()));
    }
    let (parity_ok, parity_out) = sh(&parity_bin.display().to_string(), &[&model], &env, &r);
    std::fs::write(evdir.join("parity.log"), &parity_out).unwrap();
    let parity_pass = parity_ok && parity_out.contains("PASS: every runnable");
    let (bench_ok, bench_out) = sh(&bench_bin.display().to_string(), &[&model], &env, &r);
    std::fs::write(evdir.join("bench.log"), &bench_out).unwrap();
    assert!(bench_ok, "bench failed for {id}:\n{bench_out}");

    // ---- 3. row assembly (python3: parse + sha + append) ----
    let row_py = evdir.join("row.py");
    std::fs::write(
        &row_py,
        format!(
            r#"import json, hashlib, re, subprocess, shutil, os
id_ = '{id}'; defines = '{defines}'; sched = '{sched}'; recipe = {recipe_py}
ev = '{evdir}'; ledger = '{ledger}'
manifest = json.load(open(os.path.join(ev, 'manifest.json')))
insp = manifest.get('inspection', manifest)
kern = next(k for k in insp['kernels'] if k['name'] == '{focus}')
ins = kern['instructions']
obj = os.path.join(ev, 'ks4_' + id_ + '.o')
sha = hashlib.sha256(open(obj,'rb').read()).hexdigest()
cfg = 'ks4-radiowave\n%s\n%s\n%s\n' % (id_, defines, sched)
cfg_sha = hashlib.sha256(cfg.encode()).hexdigest()
# vmcnt-in-loop: unbundle, disassemble, widest backward-branch span = group loop.
def addr_of(l):
    m = re.search(r'//\s*([0-9a-f]+):', l)
    if m: return int(m.group(1), 16)
    m = re.match(r'^\s*([0-9a-f]+)\s*<', l)
    return int(m.group(1), 16) if m else None
llvm = '/opt/rocm/core-10.0/lib/llvm/bin'
bundler = shutil.which('clang-offload-bundler') or llvm + '/clang-offload-bundler'
objdump = shutil.which('llvm-objdump') or llvm + '/llvm-objdump'
tgt = insp.get('bundle_target', 'hipv4-amdgcn-amd-amdhsa--gfx1100')
co = os.path.join(ev, 'ks4_' + id_ + '.co')
subprocess.run([bundler, '--type=o', '--unbundle', '--input=' + obj, '--targets=' + tgt, '--output=' + co], check=True, capture_output=True)
dis = subprocess.run([objdump, '--disassemble', '--mcpu=gfx1100', co], capture_output=True, text=True).stdout.splitlines()
def sym_range(name, lines):
    s = next(i for i, l in enumerate(lines) if '<' + name + '>:' in l)
    e = next((i for i, l in enumerate(lines[s+1:], s+1) if re.search(r'^\s*[0-9a-f]+\s*<\w', l)), len(lines))
focus = '{focus}'
body = sym_range(focus, dis)
base = addr_of(body[0])
spans = []
for l in body:
    if 's_cbranch_execnz' in l or re.search(r'\bs_branch\b', l):
        m = re.search(r'\+0x([0-9a-f]+)>', l); a = addr_of(l)
        if m and a is not None:
            t = base + int(m.group(1), 16)
            if t < a: spans.append((t, a))
if spans:
    t, a = max(spans, key=lambda s: s[1] - s[0])
    vmcnt_loop = sum(1 for l in body if addr_of(l) is not None and t <= addr_of(l) <= a and re.search(r's_waitcnt.*vmcnt', l))
    loop_note = ''
else:
    # No backward data-loop edge: the scheduler unrolled/restructured the
    # group loop (seen under KS4_VGPR_CAP with spills). No honest in-loop
    # count exists; record null rather than a misleading 0.
    vmcnt_loop = None
    loop_note = '; no loop back-edge (unrolled/restructured under this variant)'
os.remove(co)
# parity: worst relL2 + N=16 ks4 rows (out_proj/down_proj).
par = open(os.path.join(ev, 'parity.log')).read()
rows = []
for l in par.splitlines():
    m = re.match(r'\s*(out_proj|down_proj)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\[(OK|FAIL)\]', l)
    if m: rows.append(m.groups())
assert rows, 'no parity rows parsed'
worst = max(float(r[3]) for r in rows)
def n16(proj):
    r = next(r for r in rows if r[0] == proj and r[1] == '16' and r[2] == '4')
    return dict(relL2=float(r[3]), maxAbs=float(r[4]), min_us=float(r[11]), med_us=float(r[12]), status=r[13])
# bench: L0 out/down_proj (residual) N=16 min/med/%roof + symbol.
bench = open(os.path.join(ev, 'bench.log')).read()
sym = dict(re.findall(r'(L0 out_proj \(residual\)|L0 down_proj \(residual\))\s+gpu\.\S+.*?->\s+(\S+)', bench))
brows = {{}}
for l in bench.splitlines():
    m = re.match(r'\s*(L0 out_proj \(residual\)|L0 down_proj \(residual\))\s+16\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', l)
    if m: brows[m.group(1)] = [float(m.group(i).rstrip('%')) for i in range(2, 7)]
assert 'L0 out_proj (residual)' in brows and 'L0 down_proj (residual)' in brows, 'bench N=16 residual rows missing'
o = brows['L0 out_proj (residual)']; d = brows['L0 down_proj (residual)']
spill = kern['vgpr_spill_count'] != 0 or kern['sgpr_spill_count'] != 0 or kern['private_segment_fixed_size'] != 0
pass_ = '{pass_}' == 'True'
verdict = 'completed' if pass_ else 'correctness-rejected'
row = dict(campaign='ks4-radiowave', schema=1, candidate=id_, recipe=recipe, defines=defines, sched_profile=sched,
  config_sha256=cfg_sha, code_object_sha8=sha[:8],
  isa_ks4=dict(global_loads=ins['global_loads'], buffer_loads=ins['buffer_loads'], flat_loads=ins['flat_loads'],
    wait_total=ins['wait_instructions'], vmcnt_in_loop=vmcnt_loop, vgpr=kern['vgpr_count'], sgpr=kern['sgpr_count'],
    scratch=kern['private_segment_fixed_size'], spills=dict(vgpr=kern['vgpr_spill_count'], sgpr=kern['sgpr_spill_count'])),
  parity=dict(pass_=pass_, worst_relL2=worst, out_proj_N16_ks4=n16('out_proj'), down_proj_N16_ks4=n16('down_proj')),
  bench=dict(out_proj_N16_min_us=o[0], out_proj_N16_med_us=o[1], out_proj_N16_pct_roof=o[4],
    down_proj_N16_min_us=d[0], down_proj_N16_med_us=d[1], down_proj_N16_pct_roof=d[4],
    symbol_out=sym.get('L0 out_proj (residual)'), symbol_down=sym.get('L0 down_proj (residual)')),
  prompt_md5='{model_md5}', binary_md5=dict(parity='{parity_md5}', bench='{bench_md5}'),
  spill=spill, shippable=(pass_ and not spill), verdict=verdict,
  reason=('parity relL2<=5e-5 gate' if pass_ else 'parity FAIL relL2>5e-5 or non-finite') + ('; spill/scratch nonzero: never shippable' if spill else '') + loop_note,
  evidence=dict(parity_log=os.path.join(ev, 'parity.log'), bench_log=os.path.join(ev, 'bench.log'), manifest=os.path.join(ev, 'manifest.json')))
open(ledger, 'a').write(json.dumps(row, sort_keys=True) + '\n')
print(json.dumps(dict(candidate=id_, verdict=verdict, worst_relL2=worst, spill=spill,
  out_min=o[0], out_med=o[1], down_min=d[0], down_med=d[1], sha8=sha[:8], vmcnt_loop=vmcnt_loop), indent=1))
"#,
            id = id,
            defines = defines,
            sched = sched,
            evdir = evdir.display().to_string(),
            ledger = ledger.display().to_string(),
            focus = FOCUS_KERNEL,
            pass_ = if parity_pass { "True" } else { "False" },
            parity_md5 = parity_md5,
            bench_md5 = bench_md5,
        ),
    )
    .unwrap();
    let (ok, out) = sh("python3", &[&row_py.display().to_string()], &[], &r);
    print!("{out}");
    assert!(ok, "row assembly failed");
    println!("evidence dir: {}", evdir.display());
}
