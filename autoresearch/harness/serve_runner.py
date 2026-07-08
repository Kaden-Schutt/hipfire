#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""serve_runner — the GPU adapter behind ab_certify_serve.ServeRunner (SCAFFOLD, fleet-untested).

Bridges the tested v2 orchestrator (ab_certify_serve, 51 no-GPU tests) to a live `hipfire serve`,
reusing serve_harness's spawn + send machinery so measurement goes through the real CLI path (no
raw-daemon voodoo). It runs ONE model on a per-arm serve (env differs per arm) and returns the
generation dicts the orchestrator consumes.

  STATUS: parity + coherence + perf arms VALIDATED live on gfx1151 (null-variant certify:
  parity byte-exact PASS, coherence base-vs-base PASS, perf +0.46%/f=0.461 ~tied). Remaining fleet
  item:
    - token-ids over HTTP (the OpenAI stream returns TEXT; until the daemon emits token-ids the
      attractor detector runs on a WORD proxy — gap #17's calibration caveat applies). Marked
      `FLEET-TODO(2)` below.
  RESOLVED: parity determinism is q8_ef + HIPFIRE_DETERMINISTIC (NOT fp32 — fp32 isn't a prod/prof
  path); perf is rocprof pinned-clock kernel-DURATION under profile_standard (gap #4) — the HTTP
  kernel_decode_tok_s proxy it replaced showed a +3% sequential-thermal artifact on a null variant.
"""
import os, sys, re, glob, csv as _csvmod, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import serve_harness as sh
import ab_certify_serve as abc


def _col(hdr, want):
    for k in hdr:
        if k and k.strip().lower() == want:
            return k
    return None


def _cfg(model, daemon_bin, sampling, kv, port, seed=None, prompts_file=None,
         mode="battery", max_tokens=1024):
    """Build a serve_harness cfg for a specific daemon binary + arm settings."""
    class A:  # minimal args shim for build_config
        pass
    a = A()
    a.model, a.tag, a.registry = model, None, os.path.join(sh.REPO, "cli/registry.json")
    a.kv, a.mtp, a.thinking = kv, "off", "med"
    a.max_tokens, a.max_seq = max_tokens, 4096
    a.sampling, a.mode, a.port = sampling, mode, port
    a.seed, a.prompts_file = seed, prompts_file
    cfg = sh.build_config(a); cfg["max_seq"] = a.max_seq
    cfg["_daemon_bin"] = daemon_bin
    return cfg


def _toks_proxy(gen_result):
    """FLEET-TODO(2): real token-ids over HTTP. Until then, use whitespace words as the detector
    input (approximate; gap #17). gen_result is serve_harness.send()'s return dict."""
    txt = (gen_result.get("assistant_content") or gen_result.get("ans_preview") or "")
    return re.findall(r"\S+", txt)


class LiveServeRunner(abc.ServeRunner):
    def __init__(self, model, arch, dev, port_base=11540, prompts_file=None,
                 warmed_prompt="Explain hash maps briefly.", n_perf=8, kv="q8", coh_max_tokens=4096,
                 kernel=None, card=None):
        self.model, self.arch, self.dev = model, arch, dev
        self.port_base, self.prompts_file = port_base, prompts_file
        self.warmed_prompt, self.n_perf, self.kv = warmed_prompt, n_perf, kv
        self.coh_max_tokens = int(os.environ.get("SR_COH_MAX_TOKENS", coh_max_tokens))
        self.kernel = kernel      # target kernel (for the rocprof perf arm)
        self.card = card if card is not None else dev  # DRM card for the perf-level pin

    # --- serve lifecycle: one serve per (daemon, arm-env); killed after use ---
    def _run_on_serve(self, cfg, det, gen_fn):
        os.environ["HIPFIRE_DAEMON_BIN"] = cfg["_daemon_bin"]
        os.environ["HIP_VISIBLE_DEVICES"] = str(self.dev)
        if det:
            # Parity runs the DETERMINISTIC production-adjacent path: q8 + error-feedback (q8_ef) +
            # HIPFIRE_DETERMINISTIC (pins the WMMA reduction order). NOT fp32 — fp32 is not a prod/prof
            # path, so parity on it would test a path that doesn't exist in production. q8_ef makes the
            # Q8 DeltaNet state deterministic (error-feedback), so parity is on the real Q8 path.
            os.environ["HIPFIRE_DETERMINISTIC"] = "1"
            os.environ["HIPFIRE_DN_STATE_EF"] = "1"
        else:
            os.environ.pop("HIPFIRE_DETERMINISTIC", None)
        home = os.path.expanduser(f"~/.cache/serve_runner_{self.arch}")
        log = f"/tmp/serve_runner_{self.arch}_{cfg['port']}.log"
        if not sh.spawn_serve(cfg, home, log):
            raise RuntimeError(f"serve failed to warm for {cfg['_daemon_bin']}")
        try:
            return gen_fn(cfg)
        finally:
            sh._kill_serve()

    # --- ARM: parity (q8_ef + HIPFIRE_DETERMINISTIC greedy, battery — byte-exact text vs B_a) ---
    def parity_gens(self, daemon):
        cfg = _cfg(self.model, daemon, "greedy", self.kv, self.port_base,
                   prompts_file=self.prompts_file, mode="battery")
        def go(c):
            gens = []
            for genre, prompt in sh.load_prompt_battery(self.prompts_file):
                r = sh.send(c, [{"role": "user", "content": prompt}])
                gens.append({"prompt_id": genre, "genre": genre, "seed": 0,
                             "text": r.get("assistant_content", ""), "token_ids": _toks_proxy(r),
                             "finish": r.get("finish"), "empty": r.get("empty"), "tool_calls": []})
            return gens
        return self._run_on_serve(cfg, det=True, gen_fn=go)

    # --- ARM: coherence (Q8 registry temp>0, seed-SET × battery, same session=chain) ---
    def coherence_gens(self, daemon, seeds):
        # ONE serve for the WHOLE seed-set (not a serve per seed — that was 2*N model-loads/certify).
        cfg = _cfg(self.model, daemon, "registry", self.kv, self.port_base + 1,
                   prompts_file=self.prompts_file, mode="chain", max_tokens=self.coh_max_tokens)
        def go(c):
            out = []
            for seed in seeds:
                c["seed"] = seed
                messages = []
                for genre, prompt in sh.load_prompt_battery(self.prompts_file):
                    messages.append({"role": "user", "content": prompt})
                    r = sh.send(c, messages)
                    messages.append({"role": "assistant", "content": r.get("assistant_content", "")})
                    out.append({"prompt_id": genre, "genre": genre, "seed": seed,
                                "text": r.get("assistant_content", ""), "token_ids": _toks_proxy(r),
                                "finish": r.get("finish"), "empty": r.get("empty"), "tool_calls": []})
            return out
        return self._run_on_serve(cfg, det=False, gen_fn=go)

    # --- ARM: perf (rocprof pinned-clock kernel DURATION — the gap #4 fix) ---
    # Measure the TARGET kernel's per-dispatch duration under profile_standard (clock PINNED, so no
    # thermal/DPM drift — the sequential base-then-var thermal artifact that made a null variant look
    # +3% slower). Low-variance, arch-general, no 2-serve memory problem. This is NOT "daemon voodoo":
    # coherence/parity go through serve (output behavior); perf is a kernel-TIMING measurement, and
    # rocprof kernel-trace is the right tool for that. One trace = one per-dispatch sample per decode
    # token, so a single run yields many pinned-clock samples.
    def perf_durations(self, daemon):
        if not self.kernel:
            return []
        if not os.path.exists(daemon):
            print(f"[perf] MISSING daemon binary: {daemon}", file=sys.stderr, flush=True)
            return []
        pl = f"/sys/class/drm/card{self.card}/device/power_dpm_force_performance_level"
        reqf = f"/tmp/perf_req_{self.arch}_c{self.card}.jsonl"
        outdir = f"/tmp/perf_kt_{self.arch}_c{self.card}"
        open(reqf, "w").write(
            f'{{"type":"load","model":"{self.model}","params":{{"max_seq":2048,"kv_mode":"{self.kv}"}}}}\n'
            f'{{"type":"generate","id":"r","prompt":"Explain how a hash map resolves collisions, in two sentences.","temperature":0.0,"max_tokens":32}}\n'
            f'{{"type":"unload"}}\n')

        def run(cmd):
            subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           env=dict(os.environ, HIP_VISIBLE_DEVICES=str(self.dev)))
        # Free the GPU first: a lingering serve-daemon holding HVD keeps rocprof's daemon from
        # initializing HIP. Kill by EXACT comm (-x) — a -f cmdline match would also hit the gate's
        # own python process (its argv carries the daemon paths).
        run(f"pkill -9 -x daemon 2>/dev/null; pkill -9 -x {os.path.basename(daemon)[:15]} 2>/dev/null; sleep 2")
        run(f"{daemon} < {reqf} >/dev/null 2>&1")                       # warm (JIT) at auto
        run(f"echo profile_standard | sudo -n tee {pl} >/dev/null")     # PIN the clock
        run(f"rm -rf {outdir}; mkdir -p {outdir}")
        run(f"timeout 400 /opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d {outdir} "
            f"-- bash -c '{daemon} < {reqf} >/dev/null 2>&1' >/dev/null 2>&1")
        run(f"echo auto | sudo -n tee {pl} >/dev/null")                 # restore
        # parse per-dispatch (end-start) for the target kernel (fuzzy leading-token match: rocprof
        # reports the RUNTIME symbol, which can diverge from the source-file name).
        fs = sorted(glob.glob(f"{outdir}/**/*kernel_trace*.csv", recursive=True) or
                    glob.glob(f"{outdir}/**/*.csv", recursive=True))
        durs = []
        if fs:
            rows = list(_csvmod.DictReader(open(fs[0])))
            if rows:
                h = list(rows[0]); kn = _col(h, 'kernel_name'); st = _col(h, 'start_timestamp'); en = _col(h, 'end_timestamp')
                kt = self.kernel.split("_")
                for r in rows:
                    name = re.sub(r'\(.*', '', (r.get(kn, '') or '')).strip()
                    rt = name.split("_"); n = 0
                    for a, b in zip(kt, rt):
                        if a == b: n += 1
                        else: break
                    if name == self.kernel or name.startswith(self.kernel) or self.kernel.startswith(name):
                        n = 999
                    if n >= 4:
                        try: durs.append(int(r[en]) - int(r[st]))
                        except Exception: pass
        print(f"[perf] daemon={os.path.basename(daemon)} trace_files={len(fs)} rows={len(rows) if fs else 0} "
              f"durs={len(durs)}", file=sys.stderr, flush=True)
        return durs

    def clocks(self, daemon):
        return []  # profile_standard PINS the clock, so a separate clock-VOID is unnecessary here
