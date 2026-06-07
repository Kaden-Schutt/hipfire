// Bun-native tests for cli/chat_pure.ts.
//
// Direct import: chat_pure.ts is a side-effect-free module (no top-level
// console writes, no fs/spawn). Unlike parse_tool_calls.test.ts, we don't
// need to duplicate the SUT.
//
// Run: bun test cli/chat_pure.test.ts

import { test, expect, describe } from "bun:test";
import {
  graphemes,
  sanitizePaste,
  estimateTokens,
  computeTokPerSec,
  trimTokenWindow,
  trimMessages,
  renderMarkdown,
  stripAnsi,
  detectFenceLine,
  renderFenceOpen,
  renderFenceClose,
  feedPasteParser,
  historyUp,
  historyDown,
  historySubmit,
} from "./chat_pure.ts";

// ─── graphemes ──────────────────────────────────────────────────────────────

describe("graphemes", () => {
  test("empty string returns empty array", () => {
    expect(graphemes("")).toEqual([]);
  });

  test("ASCII splits one char per grapheme", () => {
    expect(graphemes("hello")).toEqual(["h", "e", "l", "l", "o"]);
  });

  test("non-BMP emoji is one grapheme, not two surrogates", () => {
    // 😀 is U+1F600, which is a 2-code-unit surrogate pair in UTF-16.
    // Naive str[i] would return half-pairs; graphemes() must coalesce.
    const g = graphemes("a😀b");
    expect(g.length).toBe(3);
    expect(g[1]).toBe("😀");
  });

  test("CJK characters are one grapheme each", () => {
    const g = graphemes("你好");
    expect(g.length).toBe(2);
    expect(g[0]).toBe("你");
    expect(g[1]).toBe("好");
  });

  test("ZWJ-joined family emoji is a single grapheme (Intl.Segmenter only)", () => {
    // 👨‍👩‍👧 — man + ZWJ + woman + ZWJ + girl. Renders as one glyph.
    // Array.from() fallback would return 5 elements; Intl.Segmenter returns 1.
    const fam = "👨‍👩‍👧";
    const g = graphemes(fam);
    if ((globalThis as any).Intl?.Segmenter) {
      expect(g.length).toBe(1);
      expect(g[0]).toBe(fam);
    } else {
      // Fallback path — at least confirm it doesn't blow up.
      expect(g.length).toBeGreaterThan(0);
    }
  });

  test("combining marks attach to base character", () => {
    // 'e' + combining acute → "é" as one grapheme.
    const s = "é";
    const g = graphemes(s);
    if ((globalThis as any).Intl?.Segmenter) {
      expect(g.length).toBe(1);
    }
  });
});

// ─── sanitizePaste ──────────────────────────────────────────────────────────

describe("sanitizePaste", () => {
  test("pure ASCII passes through", () => {
    expect(sanitizePaste("hello world")).toBe("hello world");
  });

  test("preserves \\n and \\t", () => {
    expect(sanitizePaste("a\nb\tc")).toBe("a\nb\tc");
  });

  test("strips NUL byte", () => {
    expect(sanitizePaste("a\x00b")).toBe("ab");
  });

  test("strips BEL", () => {
    expect(sanitizePaste("a\x07b")).toBe("ab");
  });

  test("strips embedded ANSI escape sequences (the actual bug it fixes)", () => {
    // If a user pastes `\x1b[31mred\x1b[0m`, the ESC bytes must not survive
    // into inputBuf — otherwise they'd be sent to the model verbatim.
    expect(sanitizePaste("a\x1b[31mred\x1b[0mb")).toBe("a[31mred[0mb");
  });

  test("preserves Unicode (multi-byte chars are >= 0x20)", () => {
    expect(sanitizePaste("hello 你好 😀")).toBe("hello 你好 😀");
  });
});

// ─── estimateTokens ─────────────────────────────────────────────────────────

describe("estimateTokens", () => {
  test("empty string returns 0", () => {
    expect(estimateTokens("")).toBe(0);
  });

  test("short string rounds up", () => {
    // "hi" = 2 chars / 3.5 = 0.57 → ceil = 1
    expect(estimateTokens("hi")).toBe(1);
  });

  test("scales linearly with length", () => {
    // 35 chars / 3.5 = 10
    expect(estimateTokens("a".repeat(35))).toBe(10);
  });
});

// ─── computeTokPerSec / trimTokenWindow ────────────────────────────────────

describe("computeTokPerSec", () => {
  test("empty array returns 0", () => {
    expect(computeTokPerSec([])).toBe(0);
  });

  test("single sample returns 0 (no time span)", () => {
    expect(computeTokPerSec([1000])).toBe(0);
  });

  test("two samples 1s apart yields 2 tok/s", () => {
    expect(computeTokPerSec([0, 1000])).toBe(2);
  });

  test("ten samples over 1s yields 10 tok/s", () => {
    const times = Array.from({ length: 10 }, (_, i) => i * 100);
    // span = 900ms = 0.9s, count = 10, rate = 10/0.9 ≈ 11.1
    expect(computeTokPerSec(times)).toBeCloseTo(10 / 0.9, 1);
  });

  test("zero-span samples (all at same time) return 0", () => {
    expect(computeTokPerSec([500, 500, 500])).toBe(0);
  });
});

describe("trimTokenWindow", () => {
  test("removes entries older than windowMs", () => {
    const now = 10_000;
    const times = [5000, 7000, 8500, 9000];
    trimTokenWindow(times, now, 2000); // cutoff = 8000
    expect(times).toEqual([8500, 9000]);
  });

  test("keeps everything if all in window", () => {
    const now = 10_000;
    const times = [9000, 9500, 9999];
    trimTokenWindow(times, now, 2000);
    expect(times).toEqual([9000, 9500, 9999]);
  });

  test("handles empty array", () => {
    const times: number[] = [];
    trimTokenWindow(times, 10_000, 2000);
    expect(times).toEqual([]);
  });

  test("default window is 2000ms", () => {
    const now = 5000;
    const times = [2000, 3500, 4500];
    trimTokenWindow(times, now); // default cutoff = 3000
    expect(times).toEqual([3500, 4500]);
  });

  test("per-turn reset use case: stale entries from previous turn cleared", () => {
    // Regression: GLM-5 finding M2 (session-cumulative tok/s).
    // After a 30s gap between turns, all old entries must be evicted.
    const times = [1000, 1200, 1400]; // turn 1
    const now = 31_400; // 30s later
    trimTokenWindow(times, now, 2000);
    expect(times).toEqual([]);
  });
});

// ─── trimMessages ───────────────────────────────────────────────────────────

describe("trimMessages", () => {
  test("no trim needed when under target", () => {
    const msgs = [
      { role: "user" as const, content: "short" },
      { role: "assistant" as const, content: "ok" },
    ];
    const r = trimMessages(msgs, 32768, 0.5);
    expect(r.dropped).toBe(0);
    expect(r.kept.length).toBe(2);
  });

  test("preserves leading system message (regression: original /trim ate it)", () => {
    const big = "x".repeat(100_000); // ~28571 tokens, exceeds half of 32768
    const msgs = [
      { role: "system" as const, content: "you are helpful" },
      { role: "user" as const, content: big },
      { role: "user" as const, content: big },
      { role: "user" as const, content: "latest" },
    ];
    const r = trimMessages(msgs, 32768, 0.5);
    expect(r.kept[0]?.role).toBe("system");
    expect(r.kept[0]?.content).toBe("you are helpful");
    // Should have dropped at least one of the big user messages.
    expect(r.dropped).toBeGreaterThan(0);
  });

  test("system message NOT at index 0 is treated like a normal message", () => {
    const big = "x".repeat(100_000);
    const msgs = [
      { role: "user" as const, content: big },
      { role: "system" as const, content: "mid-conversation system msg" },
      { role: "user" as const, content: "latest" },
    ];
    const r = trimMessages(msgs, 32768, 0.5);
    // The leading user message is dropped; the system msg in middle has no special protection.
    expect(r.kept[0]?.role).not.toBe("user");
  });

  test("always keeps at least one message after the system slot", () => {
    const huge = "x".repeat(1_000_000); // way over any sane limit
    const msgs = [
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: huge },
    ];
    const r = trimMessages(msgs, 1024, 0.5);
    // Must not drop the only non-system message (would leave conversation with only system).
    expect(r.kept.length).toBeGreaterThanOrEqual(2);
  });

  test("custom targetPct trims more aggressively", () => {
    const msgs = Array.from({ length: 20 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      content: "x".repeat(1000),
    }));
    const lenient = trimMessages(msgs, 1000, 0.9);
    const strict = trimMessages(msgs, 1000, 0.1);
    expect(strict.dropped).toBeGreaterThan(lenient.dropped);
  });

  test("invalid targetPct (NaN) falls back to 0.5", () => {
    const msgs = Array.from({ length: 10 }, () => ({
      role: "user" as const, content: "x".repeat(100),
    }));
    const r = trimMessages(msgs, 200, NaN);
    // Should behave as 0.5; not infinite-loop or pass through unchanged.
    expect(r.dropped).toBeGreaterThan(0);
  });
});

// ─── stripAnsi (NO_COLOR support) ───────────────────────────────────────────

describe("stripAnsi", () => {
  test("plain text passes through unchanged", () => {
    expect(stripAnsi("hello world")).toBe("hello world");
  });

  test("empty string", () => {
    expect(stripAnsi("")).toBe("");
  });

  test("strips simple SGR (bold)", () => {
    expect(stripAnsi("\x1b[1mbold\x1b[0m")).toBe("bold");
  });

  test("strips compound SGR (bold+cyan)", () => {
    expect(stripAnsi("\x1b[1;36mtitle\x1b[0m")).toBe("title");
  });

  test("strips reverse video", () => {
    expect(stripAnsi("\x1b[7mfoo()\x1b[0m")).toBe("foo()");
  });

  test("strips dim + italic", () => {
    expect(stripAnsi("\x1b[2m>\x1b[0m \x1b[3mquoted\x1b[0m")).toBe("> quoted");
  });

  test("strips OSC 8 hyperlink (ST = ESC \\)", () => {
    const link = "\x1b]8;;https://example.com\x1b\\\x1b[4mlabel\x1b[0m\x1b]8;;\x1b\\";
    expect(stripAnsi(link)).toBe("label");
  });

  test("strips OSC 8 hyperlink with BEL terminator", () => {
    const link = "\x1b]8;;https://example.com\x07\x1b[4mlabel\x1b[0m\x1b]8;;\x07";
    expect(stripAnsi(link)).toBe("label");
  });

  test("strips clear-to-EOL CSI", () => {
    expect(stripAnsi("foo\x1b[K")).toBe("foo");
  });

  test("strips cursor positioning CSI", () => {
    expect(stripAnsi("\x1b[2J\x1b[Hhello")).toBe("hello");
  });

  test("strips private-mode set/reset (e.g. cursor hide)", () => {
    expect(stripAnsi("\x1b[?25lhello\x1b[?25h")).toBe("hello");
  });

  test("preserves Unicode and tabs/newlines", () => {
    expect(stripAnsi("\x1b[1m你好\x1b[0m\n\t😀")).toBe("你好\n\t😀");
  });

  test("strips full markdown-rendered output (real-world fragment)", () => {
    // Render a fragment then verify stripping yields the source text minus
    // the rendered transformations.
    const rendered = renderMarkdown("**bold** and `code`");
    expect(stripAnsi(rendered)).toBe("bold and code");
  });

  test("strips a complete markdown link (OSC 8 + underline + dim parens)", () => {
    const rendered = renderMarkdown("[docs](https://example.com)");
    const stripped = stripAnsi(rendered);
    // What survives: the visible text + space + raw URL in parens.
    expect(stripped).toBe("docs (https://example.com)");
  });
});

// ─── renderMarkdown ─────────────────────────────────────────────────────────

describe("renderMarkdown", () => {
  test("plain text passes through unchanged", () => {
    expect(renderMarkdown("hello world")).toBe("hello world");
  });

  test("inline code is wrapped in inverse-video SGR", () => {
    expect(renderMarkdown("call `foo()` here")).toBe("call \x1b[7mfoo()\x1b[0m here");
  });

  test("bold is wrapped in bold SGR", () => {
    expect(renderMarkdown("**bold**")).toBe("\x1b[1mbold\x1b[0m");
  });

  test("italic single-asterisk wrapped in italic SGR", () => {
    expect(renderMarkdown("*em*")).toBe("\x1b[3mem\x1b[0m");
  });

  test("double asterisk does NOT match italic (must stay bold)", () => {
    // Regression: italic regex with negative lookbehind/ahead must skip **.
    const out = renderMarkdown("**foo**");
    expect(out).toBe("\x1b[1mfoo\x1b[0m");
    // No italic SGR injected.
    expect(out).not.toContain("\x1b[3m");
  });

  test("fenced code block with language label", () => {
    const out = renderMarkdown("```python\nprint(1)\n```", 10);
    expect(out).toContain("[python]");
    expect(out).toContain("print(1)");
    expect(out).toContain("─".repeat(10));
  });

  test("fenced code block without language label", () => {
    const out = renderMarkdown("```\nraw\n```", 10);
    expect(out).toContain("[code]");
    expect(out).toContain("raw");
  });

  test("fenceWidth controls border length", () => {
    const out = renderMarkdown("```\nx\n```", 5);
    expect(out).toContain("─".repeat(5));
    expect(out).not.toContain("─".repeat(6));
  });

  test("incomplete fence (no closing) is left raw — important for streaming", () => {
    // Per finding #6: streaming tail with unclosed ``` must not transform.
    // The renderer is only ever called on COMMITTED lines, so this is the
    // contract: no closing fence → no transform → no flicker on commit.
    const out = renderMarkdown("```python\nprint(1)");
    expect(out).toBe("```python\nprint(1)");
  });

  test("incomplete inline code is left raw", () => {
    expect(renderMarkdown("call `foo here")).toBe("call `foo here");
  });

  test("multiple inline codes on one line", () => {
    const out = renderMarkdown("`a` and `b`");
    expect(out).toBe("\x1b[7ma\x1b[0m and \x1b[7mb\x1b[0m");
  });

  test("# heading is bold + cyan", () => {
    expect(renderMarkdown("# Top")).toBe("\x1b[1;36mTop\x1b[0m");
  });

  test("## heading is bold", () => {
    expect(renderMarkdown("## Section")).toBe("\x1b[1mSection\x1b[0m");
  });

  test("### heading is dim-bold", () => {
    expect(renderMarkdown("### Sub")).toBe("\x1b[1;2mSub\x1b[0m");
  });

  test("heading regex anchored to start of line — '#' mid-text untouched", () => {
    expect(renderMarkdown("not a # heading")).toBe("not a # heading");
  });

  test("heading without space after # is not transformed", () => {
    // `#tag` is a hashtag, not a heading.
    expect(renderMarkdown("#nothashtag")).toBe("#nothashtag");
  });

  test("heading with inline code styles both", () => {
    const out = renderMarkdown("## Use `foo()` here");
    // The ## regex captures the full line content; inline code regex then
    // applies inside the bold-wrapped span.
    expect(out).toContain("\x1b[7mfoo()\x1b[0m");
    expect(out).toContain("\x1b[1m");
  });

  test("bullet list with - dims the bullet, replaces with •", () => {
    const out = renderMarkdown("- item one");
    expect(out).toBe("\x1b[2m•\x1b[0m item one");
  });

  test("bullet list with * dims the bullet", () => {
    const out = renderMarkdown("* item two");
    expect(out).toBe("\x1b[2m•\x1b[0m item two");
  });

  test("bullet list preserves indentation (nested lists)", () => {
    const out = renderMarkdown("  - nested");
    expect(out).toBe("  \x1b[2m•\x1b[0m nested");
  });

  test("dash mid-text is NOT a bullet", () => {
    expect(renderMarkdown("a - b - c")).toBe("a - b - c");
  });

  test("numbered list dims the number + period", () => {
    const out = renderMarkdown("1. first");
    expect(out).toBe("\x1b[2m1.\x1b[0m first");
  });

  test("numbered list with multi-digit number", () => {
    const out = renderMarkdown("12. twelfth");
    expect(out).toBe("\x1b[2m12.\x1b[0m twelfth");
  });

  test("numbered list preserves indentation", () => {
    const out = renderMarkdown("  3. nested item");
    expect(out).toBe("  \x1b[2m3.\x1b[0m nested item");
  });

  test("digit + period mid-text is NOT a numbered list", () => {
    expect(renderMarkdown("see fig 3.5 above")).toBe("see fig 3.5 above");
  });

  test("block quote dims the > and italicizes the body", () => {
    const out = renderMarkdown("> a quote");
    expect(out).toBe("\x1b[2m>\x1b[0m \x1b[3ma quote\x1b[0m");
  });

  test("> mid-text is NOT a block quote", () => {
    expect(renderMarkdown("a > b")).toBe("a > b");
  });

  test("markdown link emits OSC 8 hyperlink + underline + dim raw URL", () => {
    const out = renderMarkdown("see [docs](https://example.com)");
    expect(out).toContain("\x1b]8;;https://example.com\x1b\\");
    expect(out).toContain("\x1b[4mdocs\x1b[0m");
    expect(out).toContain("(https://example.com)");
  });

  test("bare URL gets OSC 8 + underline", () => {
    const out = renderMarkdown("visit https://example.com today");
    expect(out).toContain("\x1b]8;;https://example.com\x1b\\");
    expect(out).toContain("\x1b[4mhttps://example.com\x1b[0m");
  });

  test("bare URL stops at trailing whitespace, not at fragment chars", () => {
    const out = renderMarkdown("https://example.com/path?q=1#frag end");
    // Must include the full URL with query + fragment, but stop before " end"
    expect(out).toContain("\x1b[4mhttps://example.com/path?q=1#frag\x1b[0m");
    expect(out).toContain(" end");
  });

  test("URL inside markdown link is NOT double-wrapped as bare URL", () => {
    // The [..](..) regex runs first; the bare-URL regex's negative lookbehind
    // for `(` and `[` skips URLs that follow `(`.
    const out = renderMarkdown("[label](https://example.com)");
    // Should contain exactly one OSC 8 open marker for the URL.
    const matches = out.match(/\x1b\]8;;https:\/\/example\.com\x1b\\/g);
    expect(matches?.length).toBe(1);
  });

  test("file:// URL is supported", () => {
    const out = renderMarkdown("see file:///tmp/log.txt for details");
    expect(out).toContain("\x1b]8;;file:///tmp/log.txt\x1b\\");
  });
});

// ─── detectFenceLine + renderFence{Open,Close} ──────────────────────────────

describe("detectFenceLine", () => {
  test("plain text outside fence is not detected", () => {
    const r = detectFenceLine("hello world", false);
    expect(r.isFenceOpen).toBe(false);
    expect(r.isFenceClose).toBe(false);
  });

  test("```python opens a fence with language", () => {
    const r = detectFenceLine("```python", false);
    expect(r.isFenceOpen).toBe(true);
    expect(r.isFenceClose).toBe(false);
    expect(r.lang).toBe("python");
  });

  test("``` (no lang) opens with empty lang", () => {
    const r = detectFenceLine("```", false);
    expect(r.isFenceOpen).toBe(true);
    expect(r.lang).toBe("");
  });

  test("``` while inside fence closes it", () => {
    const r = detectFenceLine("```", true);
    expect(r.isFenceClose).toBe(true);
    expect(r.isFenceOpen).toBe(false);
  });

  test("```rust while inside fence still treated as close (no nesting)", () => {
    const r = detectFenceLine("```rust", true);
    expect(r.isFenceClose).toBe(true);
    expect(r.isFenceOpen).toBe(false);
  });

  test("indented ``` opens a fence (LLMs sometimes indent)", () => {
    const r = detectFenceLine("    ```typescript", false);
    expect(r.isFenceOpen).toBe(true);
    expect(r.lang).toBe("typescript");
  });

  test("body line that starts with backtick but not ``` is not a fence", () => {
    const r = detectFenceLine("`single backtick line", false);
    expect(r.isFenceOpen).toBe(false);
    expect(r.isFenceClose).toBe(false);
  });
});

describe("renderFenceOpen / renderFenceClose", () => {
  test("renderFenceOpen with language emits border + label", () => {
    const out = renderFenceOpen("python", 10);
    expect(out).toContain("[python]");
    expect(out).toContain("─".repeat(10));
  });

  test("renderFenceOpen without language uses [code]", () => {
    const out = renderFenceOpen("", 10);
    expect(out).toContain("[code]");
  });

  test("renderFenceClose emits just the dim border", () => {
    const out = renderFenceClose(10);
    expect(out).toContain("─".repeat(10));
    expect(out).toContain("\x1b[2m");
  });

  test("fenceWidth controls border length", () => {
    expect(renderFenceOpen("py", 5)).toContain("─".repeat(5));
    expect(renderFenceOpen("py", 5)).not.toContain("─".repeat(6));
  });
});

// ─── feedPasteParser ────────────────────────────────────────────────────────

describe("feedPasteParser (bracketed paste state machine)", () => {
  const initial = { inPaste: false, buf: "" };

  test("non-paste input passes through", () => {
    const r = feedPasteParser(initial, "hello");
    expect(r.passthrough).toBe("hello");
    expect(r.paste).toBeNull();
    expect(r.state.inPaste).toBe(false);
  });

  test("complete paste in single chunk extracts content", () => {
    const r = feedPasteParser(initial, "\x1b[200~hello world\x1b[201~");
    expect(r.paste).toBe("hello world");
    expect(r.passthrough).toBeNull();
    expect(r.state.inPaste).toBe(false);
  });

  test("paste split across two chunks: start-only", () => {
    const r1 = feedPasteParser(initial, "\x1b[200~part1");
    expect(r1.paste).toBeNull();
    expect(r1.state.inPaste).toBe(true);
    expect(r1.state.buf).toBe("part1");

    const r2 = feedPasteParser(r1.state, "part2\x1b[201~");
    expect(r2.paste).toBe("part1part2");
    expect(r2.state.inPaste).toBe(false);
  });

  test("paste split across three chunks (mid-content)", () => {
    let st = initial;
    let acc = "";
    for (const chunk of ["\x1b[200~aaa", "bbb", "ccc\x1b[201~"]) {
      const r = feedPasteParser(st, chunk);
      st = r.state;
      if (r.paste !== null) acc = r.paste;
    }
    expect(acc).toBe("aaabbbccc");
  });

  test("CRLF is normalized to LF", () => {
    const r = feedPasteParser(initial, "\x1b[200~line1\r\nline2\x1b[201~");
    expect(r.paste).toBe("line1\nline2");
  });

  test("embedded control bytes are sanitized", () => {
    const r = feedPasteParser(initial, "\x1b[200~a\x07b\x00c\x1b[201~");
    expect(r.paste).toBe("abc");
  });

  test("non-paste chunk while in-paste is buffered, not passed through", () => {
    const mid = { inPaste: true, buf: "x" };
    const r = feedPasteParser(mid, "y");
    expect(r.passthrough).toBeNull();
    expect(r.paste).toBeNull();
    expect(r.state.buf).toBe("xy");
  });

  test("paste containing tabs preserves them", () => {
    const r = feedPasteParser(initial, "\x1b[200~a\tb\x1b[201~");
    expect(r.paste).toBe("a\tb");
  });
});

// ─── history navigation ────────────────────────────────────────────────────

describe("history navigation with draft preservation", () => {
  const empty = { history: [], index: 0, draft: null };

  test("up-arrow on empty history is a no-op", () => {
    const r = historyUp(empty, "typed");
    expect(r.buffer).toBe("typed");
    expect(r.state).toBe(empty); // unchanged
  });

  test("up-arrow recalls last entry, saves draft", () => {
    const st = historySubmit(empty, "first");
    // st.index is now 1, history.length is 1, so index === length means "at draft slot".
    const r = historyUp(st, "in-progress draft");
    expect(r.buffer).toBe("first");
    expect(r.state.draft).toBe("in-progress draft");
    expect(r.state.index).toBe(0);
  });

  test("down-arrow back to bottom restores the draft (regression: original lost it)", () => {
    let st = historySubmit(empty, "old");
    const upR = historyUp(st, "my draft");
    st = upR.state;
    const downR = historyDown(st, upR.buffer);
    expect(downR.buffer).toBe("my draft");
    expect(downR.state.draft).toBeNull();
    expect(downR.state.index).toBe(st.history.length);
  });

  test("up-arrow at oldest entry stays put", () => {
    let st = historySubmit(empty, "a");
    st = historySubmit(st, "b");
    const r1 = historyUp(st, "");
    const r2 = historyUp(r1.state, r1.buffer);
    const r3 = historyUp(r2.state, r2.buffer);
    expect(r3.state.index).toBe(0);
    expect(r3.buffer).toBe("a");
    // Already at oldest; further up is a no-op.
    const r4 = historyUp(r3.state, r3.buffer);
    expect(r4.state).toBe(r3.state);
  });

  test("down-arrow at bottom is a no-op", () => {
    const st = historySubmit(empty, "a");
    const r = historyDown(st, "draft");
    expect(r.state).toBe(st);
    expect(r.buffer).toBe("draft");
  });

  test("submit appends to history and resets navigation", () => {
    let st = historySubmit(empty, "first");
    st = historySubmit(st, "second");
    expect(st.history).toEqual(["first", "second"]);
    expect(st.index).toBe(2);
    expect(st.draft).toBeNull();
  });

  test("draft is preserved across multiple ups but cleared on submit", () => {
    let st = historySubmit(empty, "a");
    st = historySubmit(st, "b");
    // Type some draft, then up twice
    let r = historyUp(st, "my draft");
    expect(r.state.draft).toBe("my draft");
    r = historyUp(r.state, r.buffer);
    expect(r.state.draft).toBe("my draft"); // still preserved
    // Submit clears it
    const after = historySubmit(r.state, "submitted");
    expect(after.draft).toBeNull();
  });
});
