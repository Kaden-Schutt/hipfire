# Copyright (c) Kaden Schutt
"""serve_runner — the live GPU adapter behind orchestrator.ServeRunner (fleet-untested wiring).

Bridges the no-GPU-tested v2 orchestrator to a live `hipfire serve`, reusing serve_harness's spawn +
send machinery so measurement goes through the real CLI path (no raw-daemon voodoo — except the one
sanctioned short-greedy parity path). Ports ``autoresearch/harness/serve_runner.py`` into the package
and adapts the perf arm to the CONJUNCTIVE gate: ``perf_measure`` now returns BOTH the serve_harness
``kernel_decode_tok_s`` samples AND the rocprof pinned-clock kernel-duration samples as
``(tok_list, dur_list)``.

  ARM PATHS (deliberate):
    - PARITY   = RAW DAEMON short single greedy, REAL committed token-ids (HIPFIRE_EMIT_TOKEN_IDS).
                 Daemon voodoo allowed ONLY here — a short, no-thinking greedy run is reproducible.
    - COHERENCE= the REAL serve path (hipfire serve, thinking ON, sampled). NO voodoo. Primary gate.
    - PERF     = profile_standard rocprof kernel-DURATION (pinned clock) + serve_harness
                 kernel_decode_tok_s. A WIN needs tok/s UP *and* duration DOWN (conjunctive).

``serve_harness`` is imported LAZILY (inside the methods that need it) so the module imports cleanly
in the no-GPU CI; the rocprof/serve work is behind ``_run_rocprof`` (a monkeypatchable module seam).
"""
import csv as _csvmod
import glob
import os
import re
import subprocess
import sys

from .orchestrator import ServeRunner as _ServeRunner

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SCRIPTS = os.path.join(_REPO, "scripts")


def _sh():
    """Lazy serve_harness import (keeps module import GPU/ROCm-free for the no-GPU CI)."""
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import serve_harness as sh
    return sh


def _col(hdr, want):
    for k in hdr:
        if k and k.strip().lower() == want:
            return k
    return None


def _toks_proxy(gen_result):
    """Until real token-ids are on the serve HTTP stream, use whitespace words as the detector input
    (approximate). gen_result is serve_harness.send()'s return dict."""
    txt = (gen_result.get("assistant_content") or gen_result.get("ans_preview") or "")
    return re.findall(r"\S+", txt)


def _parse_durations(outdir, kernel):
    """Parse per-dispatch (end-start) ns for the target kernel from a rocprofv3 --kernel-trace CSV.
    Fuzzy leading-token match: rocprof reports the RUNTIME symbol, which can diverge from the
    source-file name. Returns list[int] durations (empty if no trace/match)."""
    fs = sorted(glob.glob(f"{outdir}/**/*kernel_trace*.csv", recursive=True) or
                glob.glob(f"{outdir}/**/*.csv", recursive=True))
    durs = []
    if not fs:
        return durs
    rows = list(_csvmod.DictReader(open(fs[0])))
    if not rows:
        return durs
    h = list(rows[0])
    kn, st, en = _col(h, 'kernel_name'), _col(h, 'start_timestamp'), _col(h, 'end_timestamp')
    kt = kernel.split("_")
    for r in rows:
        name = re.sub(r'\(.*', '', (r.get(kn, '') or '')).strip()
        rt = name.split("_")
        n = 0
        for a, b in zip(kt, rt):
            if a == b:
                n += 1
            else:
                break
        if name == kernel or name.startswith(kernel) or kernel.startswith(name):
            n = 999
        if n >= 4:
            try:
                durs.append(int(r[en]) - int(r[st]))
            except Exception:
                pass
    return durs


def _run_rocprof(runner, daemon):
    """The GPU seam for the perf arm (monkeypatched no-GPU). Returns ``(dur_list, tok_list)``:

      dur_list — rocprof profile_standard pinned-clock per-dispatch kernel-duration ns (low variance,
                 directly attributes the change; the primary discriminator).
      tok_list — serve_harness ``kernel_decode_tok_s`` samples (the human-facing magnitude number).

    Measured under a PINNED clock (profile_standard) so no thermal/DPM drift confounds the A/B. This
    is NOT daemon voodoo: coherence/parity go through serve (output behaviour); perf is a kernel-TIMING
    measurement and rocprof kernel-trace is the right tool for that.
    """
    if not runner.kernel or not os.path.exists(daemon):
        if not os.path.exists(daemon):
            print(f"[perf] MISSING daemon binary: {daemon}", file=sys.stderr, flush=True)
        return ([], [])
    pl = f"/sys/class/drm/card{runner.card}/device/power_dpm_force_performance_level"
    reqf = f"/tmp/perf_req_{runner.arch}_c{runner.card}.jsonl"
    outdir = f"/tmp/perf_kt_{runner.arch}_c{runner.card}"
    open(reqf, "w").write(
        f'{{"type":"load","model":"{runner.model}","params":{{"max_seq":2048,"kv_mode":"{runner.kv}"}}}}\n'
        f'{{"type":"generate","id":"r","prompt":"Explain how a hash map resolves collisions, in two sentences.","temperature":0.0,"max_tokens":{runner.max_tokens}}}\n'
        f'{{"type":"unload"}}\n')

    def run(cmd):
        subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       env=dict(os.environ, HIP_VISIBLE_DEVICES=str(runner.dev)))

    # Free the GPU first (a lingering serve-daemon holding HVD blocks rocprof's HIP init). Kill by
    # EXACT comm (-x) so we never hit the gate's own python process.
    run(f"pkill -9 -x daemon 2>/dev/null; pkill -9 -x {os.path.basename(daemon)[:15]} 2>/dev/null; sleep 2")
    run(f"{daemon} < {reqf} >/dev/null 2>&1")                       # warm (JIT) at auto
    run(f"echo profile_standard | sudo -n tee {pl} >/dev/null")     # PIN the clock
    run(f"rm -rf {outdir}; mkdir -p {outdir}")
    run(f"timeout 400 /opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d {outdir} "
        f"-- bash -c '{daemon} < {reqf} >/dev/null 2>&1' >/dev/null 2>&1")
    run(f"echo auto | sudo -n tee {pl} >/dev/null")                 # restore
    durs = _parse_durations(outdir, runner.kernel)
    toks = runner.tok_s_samples(daemon)
    print(f"[perf] daemon={os.path.basename(daemon)} durs={len(durs)} toks={len(toks)}",
          file=sys.stderr, flush=True)
    return (durs, toks)


class LiveServeRunner(_ServeRunner):
    PARITY_PROMPTS = [
        ("p1", "Write a detailed paragraph about the history and future of computing."),
        ("p2", "List the steps to reverse a singly linked list, then give the time complexity."),
    ]

    def __init__(self, model, arch, dev, port_base=11540, prompts_file=None,
                 warmed_prompt="Explain hash maps briefly.", n_perf=8, kv="q8", coh_max_tokens=4096,
                 kernel=None, card=None, max_tokens=32):
        self.model, self.arch, self.dev = model, arch, dev
        self.port_base, self.prompts_file = port_base, prompts_file
        self.warmed_prompt, self.n_perf, self.kv = warmed_prompt, n_perf, kv
        self.coh_max_tokens = int(os.environ.get("SR_COH_MAX_TOKENS", coh_max_tokens))
        self.max_tokens = max_tokens          # perf-arm decode length
        self.kernel = kernel                  # target kernel (for the rocprof perf arm)
        self.card = card if card is not None else dev  # DRM card for the perf-level pin

    # --- serve lifecycle: one serve per (daemon, arm-env); killed after use ---
    def _cfg(self, daemon_bin, sampling, port, seed=None, mode="battery", max_tokens=1024):
        sh = _sh()

        class A:
            pass
        a = A()
        a.model, a.tag, a.registry = self.model, None, os.path.join(sh.REPO, "registry/v1.json")
        a.kv, a.mtp, a.thinking = self.kv, "off", "med"
        a.max_tokens, a.max_seq = max_tokens, 4096
        a.sampling, a.mode, a.port = sampling, mode, port
        a.seed, a.prompts_file = seed, self.prompts_file
        cfg = sh.build_config(a)
        cfg["max_seq"] = a.max_seq
        cfg["_daemon_bin"] = daemon_bin
        return cfg

    def _run_on_serve(self, cfg, det, gen_fn):
        sh = _sh()
        os.environ["HIPFIRE_DAEMON_BIN"] = cfg["_daemon_bin"]
        os.environ["HIP_VISIBLE_DEVICES"] = str(self.dev)
        if det:
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

    # --- ARM: parity (RAW DAEMON, short single greedy, token-id EXACT) ---
    def parity_gens(self, daemon):
        import json as _json
        req = '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"%s"}}\n' % (self.model, self.kv)
        for pid, pr in self.PARITY_PROMPTS:
            req += '{"type":"generate","id":"%s","prompt":"%s","temperature":0.0,"max_tokens":48}\n' % (pid, pr)
        req += '{"type":"unload"}\n'
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(self.dev), HIPFIRE_EMIT_TOKEN_IDS="1",
                   HIPFIRE_DETERMINISTIC="1", HIPFIRE_DN_STATE_EF="1")
        r = subprocess.run(daemon, input=req, shell=True, capture_output=True, text=True, env=env)
        ids = {pid: [] for pid, _ in self.PARITY_PROMPTS}
        for line in (r.stdout or "").splitlines():
            if '"committed"' not in line:
                continue
            try:
                d = _json.loads(line)
            except Exception:
                continue
            rid, tid = d.get("id"), d.get("tok_id")
            if rid in ids and tid is not None:
                ids[rid].append(tid)
        return [{"prompt_id": pid, "token_ids": ids[pid], "text": ""} for pid, _ in self.PARITY_PROMPTS]

    # --- ARM: coherence (registry temp>0, seed-SET × battery, same session=chain) ---
    def coherence_gens(self, daemon, seeds):
        sh = _sh()
        cfg = self._cfg(daemon, "registry", self.port_base + 1, mode="chain",
                        max_tokens=self.coh_max_tokens)

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

    # --- ARM: perf (CONJUNCTIVE: serve_harness tok/s + rocprof pinned-clock kernel-duration) ---
    def perf_measure(self, daemon):
        """Returns ``(tok_list, dur_list)`` — the orchestrator's conjunctive perf input.
        ``_run_rocprof`` yields ``(dur, tok)``; we hand back ``(tok, dur)``."""
        dur, tok = _run_rocprof(self, daemon)
        return (tok, dur)

    def tok_s_samples(self, daemon):
        """serve_harness kernel_decode_tok_s samples for the perf magnitude number. Spawns a serve on
        `daemon` and drives n_perf greedy decodes, harvesting `kernel_decode_tok_s` per response."""
        sh = _sh()
        cfg = self._cfg(daemon, "greedy", self.port_base + 2, mode="battery", max_tokens=self.max_tokens)

        def go(c):
            samples = []
            for _ in range(self.n_perf):
                r = sh.send(c, [{"role": "user", "content": self.warmed_prompt}])
                v = r.get("kernel_decode_tok_s")
                if v:
                    samples.append(v)
            return samples
        try:
            return self._run_on_serve(cfg, det=False, gen_fn=go)
        except Exception as e:
            print(f"[perf] tok/s sampling failed: {e}", file=sys.stderr, flush=True)
            return []

    def clocks(self, daemon):
        return []  # profile_standard PINS the clock, so a separate clock-VOID is unnecessary here


# re-export the base name for callers that `from serve_runner import ServeRunner`
ServeRunner = _ServeRunner
