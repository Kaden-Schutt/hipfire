// Self-contained pp-sweep + decode bench driver for the hipfire daemon.
//
// Speaks the daemon's stdio JSON protocol directly (no cli/index.ts version
// coupling), so it runs against any my-branch daemon that supports
// `bench_prefill`. Forces q8 KV + a bounded max_seq, sweeps prefill at the
// given pp lengths (warm + median-of-3), then measures decode tok/s via a
// greedy generate. Emits ONE JSON line on stdout.
//
// Usage:
//   HIPFIRE_DAEMON_BIN=<box>/target/release/examples/daemon \
//     bun bench_sweep.ts <model_path> [max_seq=9216] [pps=128,256,512,1024,4096,8192] [gen=128] [prompt...]
import { spawn } from "bun";
import { existsSync } from "fs";

const argv = process.argv.slice(2);
const modelPath = argv[0];
const maxSeq = parseInt(argv[1] || "9216", 10);
const pps = (argv[2] || "128,256,512,1024,4096,8192").split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0);
const genLen = parseInt(argv[3] || "128", 10);
const prompt = argv.slice(4).join(" ") || "Explain the theory of general relativity in simple terms.";
const daemonBin = process.env.HIPFIRE_DAEMON_BIN || "";
const kvMode = process.env.HIPFIRE_BENCH_KV || "q8"; // diagnostic override; default q8

function fail(msg: string) { console.log(JSON.stringify({ model: modelPath, error: msg })); process.exit(0); }
if (!daemonBin || !existsSync(daemonBin)) fail(`daemon bin not found: ${daemonBin}`);
if (!modelPath || !existsSync(modelPath)) fail(`model not found: ${modelPath}`);

// Daemon env: pin DPM (clock ramp) before load, and use a persistent on-disk
// kernel cache so JIT recompile is paid once across daemon spawns (each SKU
// spawns a fresh daemon). DPM warmup at load happens BEFORE first-forward JIT,
// so we ALSO re-ramp via a throwaway sweep below (the cold-cache recompile
// leaves the GPU idle long enough for clocks to drop again).
const daemonEnv: any = { ...process.env };
if (!daemonEnv.HIPFIRE_DPM_WARMUP_SECS) daemonEnv.HIPFIRE_DPM_WARMUP_SECS = "10";
if (!daemonEnv.HIPFIRE_KERNEL_CACHE) daemonEnv.HIPFIRE_KERNEL_CACHE = `${daemonEnv.HOME}/.cache/hipfire_kernels`;
const proc = spawn([daemonBin], { stdin: "pipe", stdout: "pipe", stderr: "inherit", env: daemonEnv });
const reader = (proc.stdout as ReadableStream<Uint8Array>).getReader();
let buf = "";
const lines: string[] = [];
const dec = new TextDecoder();

async function recvRaw(): Promise<any> {
  while (true) {
    while (lines.length) {
      const ln = lines.shift()!.trim();
      if (!ln) continue;
      try { return JSON.parse(ln); } catch { /* skip non-JSON banner lines */ }
    }
    const { value, done } = await reader.read();
    if (done) throw new Error("daemon EOF");
    buf += dec.decode(value);
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) { lines.push(buf.slice(0, idx)); buf = buf.slice(idx + 1); }
  }
}
function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error(`timeout: ${what} (${ms}ms)`)), ms))]);
}
async function recvType(t: string, ms: number, what: string): Promise<any> {
  while (true) {
    const m = await withTimeout(recvRaw(), ms, what);
    if (m.type === t || m.type === "error") return m;
  }
}
async function send(m: any) { (proc.stdin as any).write(JSON.stringify(m) + "\n"); await (proc.stdin as any).flush(); }

const out: any = { model: modelPath, max_seq: maxSeq, kv: kvMode, pp: {}, decode_tok_s: null, prefill_natural_tok_s: null };
try {
  await send({ type: "load", model: modelPath, params: { max_seq: maxSeq, kv_mode: kvMode } });
  const loaded = await recvType("loaded", 600_000, "load");
  if (loaded.type === "error") fail(`load: ${loaded.message}`);
  out.arch = loaded.arch;

  // ── throwaway warmup pass ──
  // The load-time DPM warmup runs BEFORE first-forward JIT; the cold-cache
  // kernel recompile then leaves the GPU idle long enough for clocks to drop
  // again. So re-ramp here with a full continuous pass: every pp shape once
  // (JIT prefill kernels + ramp clocks), then a SUSTAINED decode (JIT decode
  // kernels + keep clocks pinned). The measured pass runs immediately after,
  // continuously, so DPM stays high — critical for decode, which is
  // bandwidth/clock-bound and tanks if measured DPM-cold.
  for (const pp of pps) {
    if (pp + 32 > maxSeq) continue;
    await send({ type: "bench_prefill", tokens: pp });
    await recvType("prefill_result", 180_000, `warm pp${pp}`);
  }
  await send({ type: "generate", id: "w", prompt, max_tokens: 192, temperature: 0 });
  await recvType("done", 180_000, "warm decode");

  // ── measured pass (continuous → DPM stays warm) ──
  for (const pp of pps) {
    if (pp + 32 > maxSeq) { out.pp[pp] = null; continue; }
    const s: number[] = [];
    let err: string | null = null;
    for (let i = 0; i < 3; i++) {
      await send({ type: "bench_prefill", tokens: pp });
      const r = await recvType("prefill_result", 180_000, `pp${pp} #${i}`);
      if (r.type === "prefill_result") s.push(r.tok_s);
      else { err = r.message; break; }
    }
    s.sort((a, b) => a - b);
    out.pp[pp] = err ? ({ error: err } as any) : (s.length ? +s[Math.floor(s.length / 2)].toFixed(1) : null);
  }
  // decode measured — DPM already warm from the sweep above
  console.error("DECODE_START " + Date.now());
  await send({ type: "generate", id: "d", prompt, max_tokens: genLen, temperature: 0 });
  const done = await recvType("done", 180_000, "decode");
  console.error("DECODE_END " + Date.now());
  out.decode_tok_s = done.decode_tok_s != null ? +done.decode_tok_s.toFixed(1) : (done.tok_s != null ? +done.tok_s.toFixed(1) : null);
  out.prefill_natural_tok_s = done.prefill_tok_s != null ? +done.prefill_tok_s.toFixed(1) : null;
  console.log(JSON.stringify(out));
} catch (e: any) {
  out.error = String(e?.message || e);
  console.log(JSON.stringify(out));
} finally {
  try { proc.kill(); } catch {}
}
process.exit(0);
