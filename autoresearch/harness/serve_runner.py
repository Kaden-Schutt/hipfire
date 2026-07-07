#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""serve_runner — the GPU adapter behind ab_certify_serve.ServeRunner (SCAFFOLD, fleet-untested).

Bridges the tested v2 orchestrator (ab_certify_serve, 51 no-GPU tests) to a live `hipfire serve`,
reusing serve_harness's spawn + send machinery so measurement goes through the real CLI path (no
raw-daemon voodoo). It runs ONE model on a per-arm serve (env differs per arm) and returns the
generation dicts the orchestrator consumes.

  STATUS: GPU-UNTESTED SCAFFOLD. The orchestration + arm logic it feeds is fully tested; THIS file
  is the plumbing and MUST be validated on a fleet box. Three items need fleet confirmation before
  it's trustworthy — each marked `FLEET-TODO` below:
    1. FP32-DeltaNet-state selector for the parity serve (only HIPFIRE_DETERMINISTIC=1 is confirmed).
    2. token-ids over HTTP (the OpenAI stream returns TEXT; until the daemon emits token-ids the
       attractor detector runs on a WORD proxy — gap #17's calibration caveat applies).
    3. rocprof pinned-clock kernel-DURATION for the perf PRIMARY (gap #4). This scaffold uses the
       HTTP kernel_decode_tok_s counter as an interim; the pinned-clock duration is a separate
       rocprof path (oracle_profile-style) to wire as the primary discriminator.
"""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import serve_harness as sh
import ab_certify_serve as abc


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
                 warmed_prompt="Explain hash maps briefly.", n_perf=8, kv="q8"):
        self.model, self.arch, self.dev = model, arch, dev
        self.port_base, self.prompts_file = port_base, prompts_file
        self.warmed_prompt, self.n_perf, self.kv = warmed_prompt, n_perf, kv

    # --- serve lifecycle: one serve per (daemon, arm-env); killed after use ---
    def _run_on_serve(self, cfg, det, gen_fn):
        os.environ["HIPFIRE_DAEMON_BIN"] = cfg["_daemon_bin"]
        os.environ["HIP_VISIBLE_DEVICES"] = str(self.dev)
        if det:
            os.environ["HIPFIRE_DETERMINISTIC"] = "1"
            # FLEET-TODO(1): also select FP32 DeltaNet state here (memory: "byte-parity MUST use FP32
            # state + HIPFIRE_DETERMINISTIC=1"). Confirm the exact selector on the box before trusting
            # parity — Q8 stochastic rounding would false-PARITY_FAIL baseline against itself.
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

    # --- ARM: parity (FP32-det greedy, battery) ---
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
        gens = []
        for seed in seeds:
            cfg = _cfg(self.model, daemon, "registry", self.kv, self.port_base + 1, seed=seed,
                       prompts_file=self.prompts_file, mode="chain", max_tokens=4096)
            def go(c, _seed=seed):
                out, messages = [], []
                for genre, prompt in sh.load_prompt_battery(self.prompts_file):
                    messages.append({"role": "user", "content": prompt})
                    r = sh.send(c, messages)
                    messages.append({"role": "assistant", "content": r.get("assistant_content", "")})
                    out.append({"prompt_id": genre, "genre": genre, "seed": _seed,
                                "text": r.get("assistant_content", ""), "token_ids": _toks_proxy(r),
                                "finish": r.get("finish"), "empty": r.get("empty"), "tool_calls": []})
                return out
            gens.extend(self._run_on_serve(cfg, det=False, gen_fn=go))
        return gens

    # --- ARM: perf (Q8, interleaved warmed prompt) ---
    def perf_durations(self, daemon):
        cfg = _cfg(self.model, daemon, "greedy", self.kv, self.port_base + 2, max_tokens=128)
        def go(c):
            durs = []
            for _ in range(self.n_perf + 1):  # +1 throwaway warmup for this shape
                r = sh.send(c, [{"role": "user", "content": self.warmed_prompt}])
                # FLEET-TODO(3): use rocprof pinned-clock kernel DURATION as the primary. Interim:
                # invert kernel_decode_tok_s (now forwarded over HTTP) into a per-token "duration".
                tps = r.get("kernel_decode_tok_s") or r.get("decode_tok_s")
                if tps:
                    durs.append(1000.0 / tps)  # ms/token proxy (lower=better)
            return durs[1:]  # drop the warmup
        return self._run_on_serve(cfg, det=False, gen_fn=go)

    def clocks(self, daemon):
        return []  # FLEET-TODO: sample pp_dpm_sclk of the arch's DRM card during the perf arm
