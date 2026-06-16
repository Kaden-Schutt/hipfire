# Coherence battery

- commit: b50f5ff4
- branch: feature/paro-transparent-loading
- date:   2026-06-11T11:22:41+02:00
- mode:   short

Review each output for coherence (fluent English, on-topic, not stuck
in verbatim loops). Hard errors fail the gate; soft output changes do not.

## qwen3.5-0.8b.mq4 — cap

- wall: 1.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":204.9,"prefill_tokens":23,"prefill_ms":33.4,"prefill_tok_s":687.9,"decode_tok_s":224.1,"ttft_ms":33.4,"cached_tokens":0,"finish_reason":"length"}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
The user is asking for the capital of France. This is a factual question about the capital of France. The capital of France is Paris. This information can be verified through common knowledge about the country's location and history. The answer should be concise as per the user's request for one short sentence.
</think>

Paris, France's capital, is located in the Île-de-France region of
```

## qwen3.5-4b.mq4 — code

- wall: 3.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":76,"tok_s":63.2,"prefill_tokens":23,"prefill_ms":71.2,"prefill_tok_s":323.1,"decode_tok_s":67.2,"ttft_ms":71.2,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "Write a one-line Python function named square that returns n*n."

**Output:**

```
The user wants a one-line Python function named `square` that returns n*n (n squared).

In Python, this would be:

```python
def square(n):
    return n * n
```

This is a one-line function definition. Let me write it properly.
</think>

```python
def square(n): return n * n
```<|im_end|>
```

## qwen3.5-9b.mq4 — reason

- wall: 8.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":202,"tok_s":43.9,"prefill_tokens":38,"prefill_ms":104.0,"prefill_tok_s":365.3,"decode_tok_s":44.9,"ttft_ms":104.0,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
Thinking Process:

1.  **Analyze the Request:**
    *   Premise: A farmer has 17 sheep.
    *   Event: All but 9 die.
    *   Question: How many are left?
    *   Constraint: Show brief reasoning then state the final number.

2.  **Break Down the Logic:**
    *   "All but X" means everything except X happened (died). So, if all but 9 died, it means $17 - (\text{all} - \text{but } 9) = \text{remaining}$. Wait, let's rephrase.
    *   "All but 9 die" implies that the number of sheep that died is equal to (Total Sheep) - (Sheep Left).
    *   Alternatively, "All but 9" means $17 - 9$ died? No, that's not right.
    *  
```

## qwen3.5-9b.mq4 — tool-call

- wall: 5.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":61,"tok_s":36.4,"prefill_tokens":194,"prefill_ms":304.2,"prefill_tok_s":637.8,"decode_tok_s":44.5,"ttft_ms":304.2,"cached_tokens":0,"finish_reason":"tool_calls"}`
- prompt: "What does the file /tmp/fibonacci.c contain?"

**Output:**

```
The user is asking me to read the contents of a file at /tmp/fibonacci.c. I should use the read function to get the contents of this file.
</think>

<tool_call>
{"name": "read", "arguments": {"path": "/tmp/fibonacci.c"}}
</tool_call><|im_end|>
```

## qwen3.5-9b.mq3 — reason-mq3

- wall: 7.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":54,"tok_s":48.8,"prefill_tokens":38,"prefill_ms":118.4,"prefill_tok_s":320.8,"decode_tok_s":54.7,"ttft_ms":118.4,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
Here's the answer:
</think>

The farmer has 17 sheep. All but 9 die.

This is a riddle question, so I will not provide the answer.

The farmer has 17 sheep. All but 9 die.<|im_end|>
```

## qwen3.5-27b.mq3 — cap-mq3-27b

- wall: 10.7s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":17.1,"prefill_tokens":23,"prefill_ms":242.6,"prefill_tok_s":94.8,"decode_tok_s":18.1,"ttft_ms":242.6,"cached_tokens":0,"finish_reason":"length"}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
Thinking Process:

1.  **Identify the core question:** The user is asking for the capital of France.
2.  **Retrieve knowledge:** The capital of France is Paris.
3.  **Constraint Check:** The user requested a "one short sentence" answer.
4.  **Formulate the response:** "The capital of France is Paris." or simply "Paris
```

## qwen3.5-4b.mq3-lloyd — cap-mq3-lloyd-4b

- wall: 2.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":42,"tok_s":63.4,"prefill_tokens":23,"prefill_ms":70.1,"prefill_tok_s":328.1,"decode_tok_s":70.9,"ttft_ms":70.1,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
Thinking about the user's request, I need to identify the capital of France. The capital is Paris. I should answer in one short sentence as requested.
</think>

The capital of France is Paris.<|im_end|>
```

## qwen3.5-9b.mq3-lloyd — reason-mq3-lloyd-9b

- wall: 6.8s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":207,"tok_s":48.2,"prefill_tokens":38,"prefill_ms":117.7,"prefill_tok_s":322.8,"decode_tok_s":49.6,"ttft_ms":117.7,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
Thinking Process:

1.  **Analyze the Request:**
    *   Premise: "A farmer has 17 sheep."
    *   Event: "All but 9 die."
    *   Question: "How many are left?"

2.  **Interpret the Phrase:**
    *   The phrase is "All but X". This means everything except X remains.
    *   Total initial number = 17.
    *   Number remaining = All except those that died? No, it means "Everything except [X] dies". Wait, let's parse carefully.

3.  **Re-evaluating the Logic:**
    *   The phrase is a riddle or wordplay question: "A farmer has 17 sheep." (Initial state) -> "All but 9 die." (Event) -> "How many are left?" (Final state).

4.  **Wait, let's re-read the prompt:**
    *  
```

## qwen3.5-4b.mq3-lloyd — long-prefill-mq3-lloyd-4b

- wall: 4.8s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":66.0,"prefill_tokens":192,"prefill_ms":147.6,"prefill_tok_s":1301.2,"decode_tok_s":69.0,"ttft_ms":147.6,"cached_tokens":0,"finish_reason":"length"}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Task:** Review a Python module implementing an LRU cache using a doubly linked list and a hash map.
    *   **Structure:** Two short paragraphs.
    *   **Content:** Answer two specific questions:
        1. Why is a doubly linked list necessary instead of a singly linked list?
        2. What is the worst-case time complexity of `put` and `get`, and why is it constant?
    *   **Constraint:** Keep the answer under 120 words total (including both paragraphs). No code.

2.  **Drafting Paragraph 1 (Doubly Linked List):**
    *   The cache uses a doubly linked list to maintain access order (head/tail).
    *   Why doubly? Because `get` moves the node to the head, and `put` might move it to head or tail. A singly linked list only allows traversal from one direction (usually head to tail). To move a specific node
```

## qwen3.5-9b.mq3-lloyd — long-prefill-mq3-lloyd-9b

- wall: 7.4s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":46.5,"prefill_tokens":192,"prefill_ms":236.3,"prefill_tok_s":812.4,"decode_tok_s":49.0,"ttft_ms":236.3,"cached_tokens":0,"finish_reason":"length"}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Topic:** Python module implementing a fixed-capacity LRU cache.
    *   **Data Structures:** Doubly linked list (DLL) + Hash map.
    *   **Operations:** `get` (move to head), `put` (insert at head, evict tail if full).
    *   **Constraints:** No thread-safety guarantees mentioned.
    *   **Output Requirements:** Two short paragraphs, under 120 words total, no code.

2.  **Evaluate Question 1: Why is the DLL necessary instead of a singly linked list?**
    *   In an LRU cache, we need to remove the least recently used item (tail) when capacity is exceeded during `put`. A singly linked list allows O(1) removal of a node only if we have a reference to it. When evicting from the tail (least recently used), we need to remove that specific node from the end without traversing the whole list. A
```

## qwen3.5-9b.mq4-lloyd — reason-mq4-lloyd-9b — SKIPPED (model not present)

## qwen3.5-9b.q8f16 — long-prefill-q8-9b — SKIPPED (model not present)

## qwen3.5-9b.mq6 — reason-mq6

- wall: 15.4s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":281,"tok_s":31.1,"prefill_tokens":38,"prefill_ms":139.2,"prefill_tok_s":272.9,"decode_tok_s":31.6,"ttft_ms":139.2,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
Thinking Process:

1.  **Analyze the Request:** The user is presenting a riddle: "A farmer has 17 sheep. All but 9 die. How many are left?" I need to provide brief reasoning and then state the final number.

2.  **Analyze the Riddle:**
    *   Initial state: 17 sheep.
    *   Event: "All but 9 die".
    *   Interpretation of "All but X": This phrase means everything except X is removed or affected (in this case, died). So, if all *but* 9 die, it means everything except those specific 9 died. Therefore, the ones that didn't die are the ones that remain alive (or at least exist in the count of survivors).

3.  **Calculate:**
    *   Total sheep = 17.
    *   Sheep that died = All - "but" part = Everything except the remaining group. Wait, let's rephrase: "All but 9" means everyone except a specific group of 9 died.
    *   So, if I have a set of items and "all but X are removed", then X remain.
    *   Sheep remaining = The ones that didn't die = 9.

4.  **Verify:**
    *  
```

## qwen3.5-4b.mq3-awq-only — mq3-awq-paris — SKIPPED (model not present)

## qwen3.5-9b.mq4-awq-gptq-f2-lmhead — lmhead-awq-paris — SKIPPED (model not present)

## qwen3.6-35b-a3b-paro.hfq — paro-a3b-cap — SKIPPED (model not present)

## qwen3.6-35b-a3b-paro.hfq — paro-a3b-sheep — SKIPPED (model not present)

