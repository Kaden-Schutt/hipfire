#!/usr/bin/env bun
// Aggregate bench_cache_runner JSONL into a markdown matrix (cells × scenarios).
// Usage: bun scripts/bench_cache_report.ts <results.jsonl>
import { readFileSync } from "node:fs";
const path = process.argv[2];
if (!path) { console.error("usage: bench_cache_report.ts <results.jsonl>"); process.exit(1); }
const rows = readFileSync(path, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l));
const cells = [...new Set(rows.map(r => r.label))];
const get = (label: string, scenario: string) => rows.find(r => r.label === label && r.scenario === scenario) || {};
const n = (x: any, d = 1) => (typeof x === "number" ? x.toFixed(d) : "-");
const pct = (x: any) => (typeof x === "number" ? `${(x * 100).toFixed(0)}%` : "-");

let md = `# Prompt-cache + DFlash bench matrix\n\n`;
md += `Cells: ${cells.length} — ${cells.join(" · ")}\n\n`;

md += `## short (single-turn baseline; ~cold)\n\n`;
md += `| cell | ttft ms | prefill t/s | decode t/s | τ | uniq |\n|---|--:|--:|--:|--:|--:|\n`;
for (const c of cells) { const s = get(c, "short"); md += `| ${c} | ${n(s.ttft_ms, 0)} | ${n(s.prefill_tok_s)} | ${n(s.decode_tok_s)} | ${s.tau ?? "-"} | ${n(s.min_uniq_ratio, 3)} |\n`; }

md += `\n## multiturn (growing chat — prefix-cache reuse)\n\n`;
md += `| cell | avg reuse (t≥1) | final ctx tok | total wall s | last-turn prefill t/s | decode t/s | τ |\n|---|--:|--:|--:|--:|--:|--:|\n`;
for (const c of cells) {
  const m = get(c, "multiturn"); const last = (m.turns || []).at(-1) || {};
  md += `| ${c} | ${pct(m.avg_reuse_t1plus)} | ${m.final_ctx_tokens ?? "-"} | ${n(m.total_wall_s, 2)} | ${n(last.prefill_tok_s)} | ${n(last.decode_tok_s)} | ${last.tau ?? "-"} |\n`;
}
md += `\nPer-turn reuse (shows prefix cache kicking in as context grows):\n\n`;
for (const c of cells) {
  const m = get(c, "multiturn"); if (!m.turns) continue;
  md += `- **${c}**: ` + m.turns.map((t: any) => `t${t.turn} ${pct(t.reuse)}`).join(" → ") + `\n`;
}

md += `\n## divergent (dropped-history render — checkpoint RESUME)\n\n`;
md += `| cell | resume reuse | replay / prompt tok | resume prefill s | cold-est s | prefill speedup | uniq |\n|---|--:|--:|--:|--:|--:|--:|\n`;
for (const c of cells) { const d = get(c, "divergent"); md += `| ${c} | ${pct(d.resume_reuse)} | ${d.replay_tokens ?? "-"} / ${d.resume_prompt ?? "-"} | ${n(d.resume_prefill_s, 2)} | ${n(d.cold_est_s, 2)} | ${d.prefill_speedup ? d.prefill_speedup + "×" : "-"} | ${n(d.resume_uniq_ratio, 3)} |\n`; }

md += `\n## toolcalls (bash/read/write/grep correctness)\n\n`;
md += `| cell | correct | decode t/s | τ |\n|---|--:|--:|--:|\n`;
for (const c of cells) { const t = get(c, "toolcalls"); md += `| ${c} | ${t.correct ?? "-"}/${t.total ?? "-"} | ${n(t.decode_tok_s)} | ${t.tau ?? "-"} |\n`; }

md += `\n_Notes: τ = DFlash mean accepted tokens/verify cycle (—=AR path, no spec-decode). reuse = cached_tokens/prompt_tokens. Thinking-on always routes to AR regardless of dflash setting._\n`;
console.log(md);
