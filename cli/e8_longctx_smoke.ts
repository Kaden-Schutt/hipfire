// Long-prefill compounding-loss check for the grouped-WMMA E8 path.
// Prefills a ~1500-token passage (forces a large grouped-WMMA prefill across
// all 40 layers), then asks a question whose answer sits EARLY in the passage.
// If grouped-WMMA error compounded across the long prefill, late-layer hidden
// states would be corrupted → wrong/garbled answer. Coherent + correct recall
// ⇒ the error is bounded, not compounding.
// Run: HIPFIRE_E8_GFX12=1 HIPFIRE_DEBUG_BATCH=1 bun cli/e8_longctx_smoke.ts
import { spawn } from "bun";
import { homedir } from "os";

const MODEL = `${homedir()}/.hipfire/models/q36a3b.mfp4e8-gptq-v2.hfq`;
const DAEMON = `${import.meta.dir}/../target/release/examples/daemon`;
const env: any = { ...process.env, HIPFIRE_KV_MODE: "q8" };
delete env.HIPFIRE_GRAPH;
const proc = spawn({ cmd: [DAEMON], stdin: "pipe", stdout: "pipe", stderr: "inherit", env });
const stdin = proc.stdin;
const stdout = proc.stdout.getReader();
const dec = new TextDecoder(); const enc = new TextEncoder();
let buf = "";
async function send(o: any) { await stdin.write(enc.encode(JSON.stringify(o) + "\n")); }
async function readUntil(p: (m: any) => boolean) {
  const out: any[] = [];
  while (true) {
    if (buf.includes("\n")) { const i = buf.indexOf("\n"); const l = buf.slice(0, i); buf = buf.slice(i + 1);
      if (l.trim()) { const m = JSON.parse(l); out.push(m); if (p(m)) return out; } }
    else { const { value, done } = await stdout.read(); if (done) return out; buf += dec.decode(value); }
  }
}

await send({ type: "load", model: MODEL, params: { max_seq: 8192 } });
const ld = await readUntil((m) => m.type === "loaded" || m.type === "error");
if (ld.some((m) => m.type === "error")) { console.log("LOAD ERROR", ld); process.exit(1); }

// ~1500-token self-contained passage. The KEY fact to recall ("Aldous Finch",
// "1847", "indigo dye") is planted in the FIRST paragraph; everything after is
// filler so the answer depends on early-context survival through a long prefill.
const P1 =
  "In 1847, a little-known chemist named Aldous Finch first synthesized a stable " +
  "indigo dye from coal tar, a discovery that would quietly reshape the textile " +
  "industry. Finch worked in a cramped laboratory in the town of Marlowe, funded " +
  "by a modest grant from the Marlowe Weavers' Guild. ";
const FILLER = (
  "The process he devised relied on careful temperature control: the reaction " +
  "vessel had to be held within a narrow band, and overheating produced a useless " +
  "brown sludge instead of the prized blue pigment. Early batches were small, and " +
  "the yield was inconsistent, but the Guild saw enough promise to continue its " +
  "support. Over the following decade, the technique spread to neighboring towns, " +
  "each adapting it to local materials and water sources. Dye houses competed to " +
  "refine the purity of the blue, and a small ecosystem of suppliers grew up around " +
  "the trade, providing reagents, glassware, and skilled labor. Apprentices learned " +
  "to read the color of the reaction by eye, judging readiness from subtle shifts in " +
  "hue. Records from the period describe elaborate rituals of timing and stirring, " +
  "passed from master to apprentice and rarely written down. Transport of the " +
  "finished dye posed its own challenges, as the pigment was sensitive to moisture " +
  "and had to be sealed in wax-lined crates. Merchants traveled established routes " +
  "to coastal markets, where the blue cloth commanded premium prices. "
).repeat(5);
const PROMPT = P1 + FILLER + "\n\nQuestion: According to the passage, who first " +
  "synthesized the stable indigo dye, and in what year? Answer in one sentence.";

await send({ type: "generate", id: "lc", prompt: PROMPT, temperature: 0.0, max_tokens: 80, repeat_penalty: 1.0 });
let text = "";
const msgs = await readUntil((m) => m.type === "done" && m.id === "lc");
for (const m of msgs) if (m.type === "token" && m.id === "lc") text += m.text;
console.log(`prompt length: ${PROMPT.length} chars`);
console.log("=== ANSWER (must recall 'Aldous Finch' / '1847' from the FIRST paragraph) ===");
console.log(text.trim());

await send({ type: "unload" });
await readUntil((m) => m.type === "unloaded");
stdin.end();
await proc.exited;
