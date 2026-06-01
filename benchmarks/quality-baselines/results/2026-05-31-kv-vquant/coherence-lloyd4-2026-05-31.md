# Coherence battery

- commit: 0121e286
- branch: feat/kv-vquant-fwht-lloyd-v
- date:   2026-05-31T00:32:21-07:00
- mode:   short

Review each output for coherence (fluent English, on-topic, not stuck
in verbatim loops). Hard errors fail the gate; soft output changes do not.

## qwen3.5-0.8b.mq4 — cap

- wall: 2.7s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":7,"tok_s":8.0,"prefill_tokens":21,"prefill_ms":849.9,"prefill_tok_s":24.7,"decode_tok_s":256.3,"ttft_ms":849.9}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>

</think>

 Paris.<|im_end|>
```

## qwen3.5-4b.mq4 — code

- wall: 3.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":104,"tok_s":158.3,"prefill_tokens":21,"prefill_ms":27.7,"prefill_tok_s":757.7,"decode_tok_s":165.3,"ttft_ms":27.7}`
- prompt: "Write a one-line Python function named square that returns n*n."

**Output:**

```
<think>
The user wants a one-line Python function named `square` that returns n*n. This is straightforward - I need to create a function that takes an input (n) and returns n squared.

In Python, this can be written as:

```python
def square(n): return n * n
```

This is a one-liner function definition. Let me make sure it's correct and follows the requirements.
</think>

```python
def square(n): return n * n
```<|im_end|>
```

## qwen3.5-9b.mq4 — reason

- wall: 3.7s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":67,"tok_s":114.8,"prefill_tokens":36,"prefill_ms":41.0,"prefill_tok_s":877.9,"decode_tok_s":123.5,"ttft_ms":41.0}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>

</think>

If all but 9 of the sheep die, then 9 sheep are left.

**Reasoning:**
The phrase "all but X" means that X items remain while the rest perish. Therefore, if 17 sheep existed and all but 9 died, exactly **9** are left.<|im_end|>
```

## qwen3.5-9b.mq4 — tool-call

- wall: 3.5s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":28,"tok_s":69.1,"prefill_tokens":193,"prefill_ms":168.4,"prefill_tok_s":1145.9,"decode_tok_s":118.2,"ttft_ms":168.4}`
- prompt: "What does the file /tmp/fibonacci.c contain?"

**Output:**

```
<think>

</think>

<tool_call>
{"name": "read", "arguments": {"path": "/tmp/fibonacci.c"}}
</tool_call><|im_end|>
```

## qwen3.5-9b.mq3 — reason-mq3

- wall: 2.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":25,"tok_s":108.2,"prefill_tokens":36,"prefill_ms":43.1,"prefill_tok_s":835.7,"decode_tok_s":133.0,"ttft_ms":43.1}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>
Thinking: All but 9 die. How many are left? Show brief reasoning then state the final number.<|im_end|>
```

## qwen3.5-27b.mq3 — cap-mq3-27b

- wall: 7.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":47.1,"prefill_tokens":21,"prefill_ms":85.6,"prefill_tok_s":245.2,"decode_tok_s":49.6,"ttft_ms":85.6}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>
The user is asking for the capital of France in a single short sentence. This is a straightforward factual question with a clear answer.

The capital of France is Paris. This is well-established geographical information that doesn't require any complex reasoning or calculation.

I should provide the answer directly and concisely as requested, keeping it to one short sentence.
</think>

The capital of France
```

## qwen3.5-4b.mq3-lloyd — cap-mq3-lloyd-4b

- wall: 2.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":12,"tok_s":110.5,"prefill_tokens":21,"prefill_ms":30.0,"prefill_tok_s":699.0,"decode_tok_s":152.7,"ttft_ms":30.0}`
- prompt: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>

</think>

Paris is the capital of France.<|im_end|>
```

## qwen3.5-9b.mq3-lloyd — reason-mq3-lloyd-9b

- wall: 3.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":30,"tok_s":105.8,"prefill_tokens":36,"prefill_ms":45.5,"prefill_tok_s":791.2,"decode_tok_s":126.1,"ttft_ms":45.5}`
- prompt: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>

</think>

**Reasoning:**
The phrase
"17 - 9 = 8"

**Final Number:**
8<|im_end|>
```

## qwen3.5-4b.mq3-lloyd — long-prefill-mq3-lloyd-4b

- wall: 3.6s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":148.9,"prefill_tokens":190,"prefill_ms":66.6,"prefill_tok_s":2851.4,"decode_tok_s":155.9,"ttft_ms":66.6}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
<think>
The user wants to know why a doubly linked list is used instead of a singly linked list in an LRU cache implementation, and the time complexity of `put` and `get` operations.

1.  **Why doubly linked list?**
    *   A singly linked list requires storing a reference to the previous node (or the next node) to traverse backwards.
    *   In an LRU cache, we need to access nodes by key (O(1) lookup).
    *   If we use a singly linked list, we can only traverse forward. To access the previous node (e.g., for `get` or evicting), we need O(log n) or O(n) traversal if not using a hash map with O(1) lookup.
    *   However, the question asks about `put` and `get`. The user wants to know why doubly linked list is used instead of singly linked list?

2.  **Time complexity**
    *   The user wants to know the time complexity of `put` and
```

## qwen3.5-9b.mq3-lloyd — long-prefill-mq3-lloyd-9b

- wall: 4.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":138,"tok_s":111.9,"prefill_tokens":190,"prefill_ms":103.4,"prefill_tok_s":1837.3,"decode_tok_s":122.2,"ttft_ms":103.4}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
<think>

</think>

1. A singly linked list lacks O(1) lookup because it requires O(n) time to traverse the list. A doubly linked list allows O(1) lookup via a hash map, which is necessary for constant time access.

2. The worstst case time complexity of `put` and `get` is O(1), because the hash map maps keys to list nodes for O(1) lookup. On `get`, the entry is moved to the head; on `put(key, value)`, the entry is inserted at the head and, if capacity is exceeded, the tail is evicted. There are no thread-safety guarantees.<|im_end|>
```

## qwen3.5-9b.mq4-lloyd — reason-mq4-lloyd-9b — SKIPPED (model not present)

## qwen3.5-9b.q8f16 — long-prefill-q8-9b

- wall: 8.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":220,"tok_s":67.0,"prefill_tokens":190,"prefill_ms":132.4,"prefill_tok_s":1435.0,"decode_tok_s":69.8,"ttft_ms":132.4}`
- prompt: `@coherence_lloyd_long.txt` (md5: `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`)

**Output:**

```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Review a Python module implementing an LRU cache (doubly linked list + hash map).
    *   Constraints: Two short paragraphs. Answer specific questions (1. Why doubly linked list? 2. Worst-case time complexity of `put`/`get` and why constant?). Keep under 120 words total. No code.

2.  **Drafting Paragraph 1 (Doubly Linked List):**
    *   Question: Why is the doubly linked list necessary instead of a singly linked list?
    *   Reasoning: To move an accessed node to the head in O(1) time, we need to traverse backwards from its successor or predecessor to remove it efficiently without searching for the previous node. A singly linked list requires traversing from the head to find the previous node, making updates O(n).
    *   Draft: The doubly linked list enables O(1) removal of arbitrary nodes during cache hits and misses. When accessing
```

## qwen3.5-9b.mq6 — reason-mq6 — SKIPPED (model not present)

## qwen3.5-4b.mq3-awq-only — mq3-awq-paris — SKIPPED (model not present)

## qwen3.5-9b.mq4-awq-gptq-f2-lmhead — lmhead-awq-paris — SKIPPED (model not present)

