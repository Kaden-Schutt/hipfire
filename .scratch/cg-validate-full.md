# Coherence battery

- commit: b187e792
- branch: feature/paro-transparent-loading
- date:   2026-06-11T14:42:32+02:00
- mode:   full

Review each output for coherence (fluent English, on-topic, not stuck
in verbatim loops). Hard errors fail the gate; soft output changes do not.

## qwen3.5-0.8b.mq4 — cap

- wall: 1.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":206.0,"prefill_tokens":23,"prefill_ms":33.5,"prefill_tok_s":687.0,"decode_tok_s":225.4,"ttft_ms":33.5,"cached_tokens":0,"finish_reason":"length"}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
The user is asking for the capital of France. This is a factual question about the capital of France. The capital of France is Paris. This information can be verified through common knowledge about the country's location and history. The answer should be concise as per the user's request for one short sentence.
</think>

Paris, France's capital, is located in the Île-de-France region of
```

## qwen3.5-4b.mq4 — code

- wall: 3.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":76,"tok_s":63.2,"prefill_tokens":23,"prefill_ms":70.4,"prefill_tok_s":326.7,"decode_tok_s":67.2,"ttft_ms":70.4,"cached_tokens":0,"finish_reason":"stop"}`
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
- stats: `{"type":"done","id":"r1","tokens":202,"tok_s":43.9,"prefill_tokens":38,"prefill_ms":107.4,"prefill_tok_s":353.7,"decode_tok_s":44.9,"ttft_ms":107.4,"cached_tokens":0,"finish_reason":"stop"}`
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

- wall: 5.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":61,"tok_s":36.3,"prefill_tokens":194,"prefill_ms":305.7,"prefill_tok_s":634.5,"decode_tok_s":44.4,"ttft_ms":305.7,"cached_tokens":0,"finish_reason":"tool_calls"}`
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

- wall: 7.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":54,"tok_s":48.9,"prefill_tokens":38,"prefill_ms":118.5,"prefill_tok_s":320.6,"decode_tok_s":54.7,"ttft_ms":118.5,"cached_tokens":0,"finish_reason":"stop"}`
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

- wall: 11.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":17.1,"prefill_tokens":23,"prefill_ms":243.9,"prefill_tok_s":94.3,"decode_tok_s":18.1,"ttft_ms":243.9,"cached_tokens":0,"finish_reason":"length"}`
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

- wall: 2.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":42,"tok_s":63.5,"prefill_tokens":23,"prefill_ms":70.3,"prefill_tok_s":327.3,"decode_tok_s":71.0,"ttft_ms":70.3,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
Thinking about the user's request, I need to identify the capital of France. The capital is Paris. I should answer in one short sentence as requested.
</think>

The capital of France is Paris.<|im_end|>
```

## qwen3.5-9b.mq3-lloyd — reason-mq3-lloyd-9b

- wall: 7.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":207,"tok_s":48.2,"prefill_tokens":38,"prefill_ms":116.8,"prefill_tok_s":325.2,"decode_tok_s":49.6,"ttft_ms":116.8,"cached_tokens":0,"finish_reason":"stop"}`
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

- wall: 5.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":65.9,"prefill_tokens":192,"prefill_ms":148.9,"prefill_tok_s":1289.2,"decode_tok_s":69.0,"ttft_ms":148.9,"cached_tokens":0,"finish_reason":"length"}`
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

- wall: 7.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":46.7,"prefill_tokens":192,"prefill_ms":232.0,"prefill_tok_s":827.7,"decode_tok_s":49.1,"ttft_ms":232.0,"cached_tokens":0,"finish_reason":"length"}`
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

- wall: 15.8s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":281,"tok_s":31.1,"prefill_tokens":38,"prefill_ms":138.2,"prefill_tok_s":274.9,"decode_tok_s":31.6,"ttft_ms":138.2,"cached_tokens":0,"finish_reason":"stop"}`
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

## qwen3.5-35b-a3b.mq4 — moe-sheep — SKIPPED (model not present)

## qwen3.5-35b-a3b.q8f16 — moe-q8-wo-sheep — SKIPPED (model not present)

## qwen3.6-35b-a3b.mq4 — moe36-sheep

- wall: 46.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":740,"tok_s":46.1,"prefill_tokens":38,"prefill_ms":2756.1,"prefill_tok_s":13.8,"decode_tok_s":55.6,"ttft_ms":2756.1,"cached_tokens":0,"finish_reason":"stop"}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
Here's a thinking process:

1.  **Analyze User Input:**
   - Statement: "A farmer has 17 sheep. All but 9 die. How many are left?"
   - Request: "Show brief reasoning then state the final number."

2.  **Identify Key Phrase/Trick:**
   - The phrase "All but 9 die" is a classic riddle structure.
   - "All but X" means everything except X happens to the rest. So, if all but 9 die, it means 9 survive (are left alive).

3.  **Calculate/Reason:**
   - Total sheep = 17
   - "All but 9 die" → The ones that do NOT die = 9
   - Therefore, the number left alive = 9

4.  **Formulate Response:**
   - Need brief reasoning: Explain the meaning of "all but X".
   - State final number clearly.

   Draft: The phrase "all but 9 die" means that every sheep except for nine passed away, leaving nine alive. Therefore, the farmer has **9** sheep left.

5.  **Check against constraints:**
   - Brief reasoning? Yes.
   - Final number stated? Yes.
   - Matches riddle logic? Yes, it's a well-known trick question where the answer is simply the number given in "all but X".

6.  **Final Output Generation:** (matches draft)✅
   - Reasoning: The phrase "all but 9 die" means that every sheep except nine died, so nine remain alive.
   - Final number: 9.✅ All steps verified. Output matches response. 
   - Self-Correction/Refinement during thought: Sometimes people misread it as 17-9=8, but the phrasing explicitly says "all but 9", which directly states how many survive. I'll make sure the reasoning clearly addresses this linguistic trick to avoid confusion.

   Final check of the prompt: "Show brief reasoning then state the final number." -> I will follow exactly.
   Output matches draft.✅
   Proceeds. 
   [Output Generation] -> "The phrase 'all but 9 die' means that every sheep except nine died, leaving nine alive. Therefore, **9** sheep are left." (Matches request)✅
   All good. 
   Output matches response. 
   [Done.] 
   *Self-Correction/Verification during process*: 
   Wait, let's double check the exact wording: "All but 9 die." This is a classic lateral thinking/riddle question. The answer is indeed 9. Some might overthink it as 17-9=8, but "all but X" means X remain. I'll stick with 9 and explain clearly.
   Reasoning: The phrase "all but 9 die" means that every sheep except nine died, so nine survive.
   Final number: 9.
   Matches perfectly. Output matches.✅
   Proceeds. 
   [Final Response Text] -> The phrase "all but 9 die" means that every sheep except nine died, leaving nine alive. Therefore, **9** sheep are left.✅ (Matches request exactly) 
   All steps verified. Output generated. 
   [Done.]
</think>

The phrase "all but 9 die" means that every sheep except nine died, so those nine survive. 

**Final number: 9**<|im_end|>
```

## qwen3.6-27b.mq4 — tool-call-27b

- wall: 25.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":111,"tok_s":12.8,"prefill_tokens":194,"prefill_ms":1037.0,"prefill_tok_s":187.1,"decode_tok_s":14.5,"ttft_ms":1037.0,"cached_tokens":0,"finish_reason":"tool_calls"}`
- prompt: "What does the file /tmp/fibonacci.c contain?"

**Output:**

```
The user is asking for the contents of a specific file: `/tmp/fibonacci.c`.
I need to use the `read` tool to access this file.
The path is provided as `/tmp/fibonacci.c`.

Plan:
1. Call `read` with the path `/tmp/fibonacci.c`.
2. Output the contents of the file to the user.

Let's do it.
</think>

<tool_call>
{"name": "read", "arguments": {"path": "/tmp/fibonacci.c"}}<|im_end|>
```

## deepseek-v4-flash.mq2lloyd — deepseek4-cap

- wall: 170.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":7,"tok_s":19.774011299435028,"prompt_tokens":17,"prefill_tokens":17,"cached_tokens":0,"prefill_ms":379,"total_ms":734,"finish_reason":"stop","spec_k":3,"spec_windows":3,"spec_accept_pct":66.66666666666666}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
The capital of France is Paris.
```

## deepseek-v4-flash.mq2lloyd — deepseek4-reason

- wall: 201.6s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":52,"tok_s":21.532091097308488,"prompt_tokens":31,"prefill_tokens":31,"cached_tokens":0,"prefill_ms":786,"total_ms":3201,"finish_reason":"stop","spec_k":3,"spec_windows":21,"spec_accept_pct":58.730158730158735}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
**Brief reasoning:**  
The phrase "all but 9 die" means that all except 9 of the sheep have died. Since the farmer started with 17 sheep, the 9 that did not die are still alive.

**Final number:**  
9
```

## deepseek-v4-flash.mq2lloyd — deepseek4-long-prefill

- wall: 157.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":148,"tok_s":18.472291562656014,"prompt_tokens":180,"prefill_tokens":180,"cached_tokens":0,"prefill_ms":3516,"total_ms":11528,"finish_reason":"stop","spec_k":3,"spec_windows":68,"spec_accept_pct":47.05882352941176}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
1. A doubly linked list is necessary because the cache must efficiently remove the tail (least-recently-used) node during eviction and also remove an arbitrary node (when it is moved to the head) without needing to traverse the list. A singly linked list would require O(n) to remove the tail or a removed node’s predecessor, breaking the constant-time requirement.

2. Both `put` and `get are O(1) worst-case because the hash map provides O(1) node lookup, and list operations (insert at head, remove a node, and remove tail) are all constant-time operations on a doubly linked list. No traversal or scanning is needed, so the time remains constant regardless of cache size.
```

