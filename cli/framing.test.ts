// Framing regression (NO GPU): extractVisibleThinking must route a model's PLAIN
// (no-think) output to `content` and only genuine <think> reasoning to `reasoning`.
// Guards the LFM2.5-instruct bug where the whole answer was trapped in
// reasoning_content because the CLI assumed started-in-think from the requested
// assistant_prefix. The daemon now reports started_in_think from the rendered
// prompt; this function is the shared non-stream split that consumes it.

import { test, expect, describe } from "bun:test";
import { extractVisibleThinking } from "./index.ts";

describe("extractVisibleThinking framing (no-GPU)", () => {
  test("started_in_think=false + plain output → all content, empty reasoning (LFM instruct)", () => {
    const r = extractVisibleThinking("The capital of France is Paris.", false, false);
    expect(r.content).toBe("The capital of France is Paris.");
    expect(r.reasoning).toBe("");
  });

  test("started_in_think=false + model-emitted <think>…</think> → split (LFM thinking / mid-stream tags)", () => {
    const r = extractVisibleThinking("<think>let me think</think>The capital of France is Paris.", false, false);
    expect(r.content).toBe("The capital of France is Paris.");
    expect(r.reasoning).toBe("let me think");
  });

  test("started_in_think=true + unclosed prompt-primed reasoning → all reasoning, empty content (open_think)", () => {
    const r = extractVisibleThinking("just reasoning, no answer yet", false, true);
    expect(r.content).toBe("");
    expect(r.reasoning).toBe("just reasoning, no answer yet");
  });

  test("started_in_think=true + closed reasoning → reasoning split, answer in content", () => {
    const r = extractVisibleThinking("reasoning here</think>Paris.", false, true);
    expect(r.content).toBe("Paris.");
    expect(r.reasoning).toBe("reasoning here");
  });

  test("started_in_think=true but model still emits its own <think> pair → single split, no double-open", () => {
    const r = extractVisibleThinking("<think>real reasoning</think>Paris.", false, true);
    expect(r.content).toBe("Paris.");
    expect(r.reasoning).toBe("real reasoning");
  });

  test("preserveThinking keeps <think> tags inline in content", () => {
    const r = extractVisibleThinking("<think>x</think>answer", true, true);
    expect(r.content).toBe("<think>x</think>answer");
    expect(r.reasoning).toBe("");
  });
});
