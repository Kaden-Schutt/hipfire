# Coherence battery — DFlash / DDTree

- commit: 6126d41d
- branch: codex/pr319-forward-train-6126d41d
- date:   2026-05-26T21:34:21+00:00
- mode:   short
- kv_mode: q8
- target: /home/kaden/.hipfire/models/qwen3.5-27b.mq4
- draft:  /home/kaden/.hipfire/models/qwen35-27b-dflash-mq4.hfq

Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.40,
unique_token_ratio < 0.30 (token-attractor detection — see Path A
failure mode in commit 6c84b13).

## 27b-dflash-prose (dflash)

- wall: 29.5s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 128, "unique": 81, "unique_ratio": 0.633, "max_freq": 0.07, "max_tok": 279, "max_count": 9}`
- stats:
  ```
  emitted: 192 tokens in 14.31s  (13.42 tok/s)
  cycles: 85  committed: 276  accepted: 106  τ=1.247  mean_committed=3.247
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to govern effectively. Communication was slow, and the central government in Rome struggled to control distant provinces. This led to a lack of unity and a sense of disconnection between the center and the periphery.
Another major factor was economic instability. The empire relied heavily on slave labor, which discouraged innovation and technological progress. As the empire expanded, it brought in vast amounts of wealth, but this wealth was not distributed evenly. The rich became richer, while the poor struggled to survive. This economic inequality created social unrest and weakened the social fabric of the empire.
Furthermore, political instability played a significant role. The Roman Empire saw a rapid succession of emperors, many of whom were assassinated or overthrown. This constant turnover of leadership created a power vacuum and made it difficult to implement long-term policies. The military, which was supposed to protect the empire, often became a kingmaker,
```

## 27b-dflash-code (dflash)

- wall: 14.1s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4}`
- stats:
  ```
  emitted: 45 tokens in 0.54s  (83.00 tok/s)
  cycles: 4  committed: 48  accepted: 40  τ=10.000  mean_committed=12.000
  ```

**Output:**

```
     for i in range(len(numbers)):
         for j in range(i + 1, len(numbers)):
             if abs(numbers[i] - numbers[j]) < threshold:
                 return True
     return False<|endoftext|>
```

## 27b-ddtree-b12-prose (ddtree-b12-k2)

- wall: 28.4s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 128, "unique": 88, "unique_ratio": 0.688, "max_freq": 0.062, "max_tok": 13, "max_count": 8}`
- stats:
  ```
  emitted: 193 tokens in 15.26s  (12.64 tok/s)
  cycles: 71  committed: 263  accepted: 121  τ=1.704  mean_committed=3.704
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to govern effectively. Communication was slow, and the central government in Rome struggled to control distant provinces. This led to a lack of unity and a sense of disconnection among the people.
Another major factor was economic instability. The empire relied heavily on slave labor, which discouraged innovation and technological advancement. As the empire expanded, it became increasingly difficult to find new slaves, leading to labor shortages. Furthermore, heavy taxation on the middle and lower classes, combined with rampant inflation, caused widespread poverty and discontent. The rich grew richer while the poor grew poorer, creating a deep social divide.
Political corruption and instability were also significant issues. The Roman Senate was often ineffective, and the emperor's power was frequently challenged. There were frequent changes in leadership, with many emperors being assassinated or overthrown. This constant political turmoil weakened the central authority and made it difficult to implement long-term policies. The
```

## 27b-ddtree-b12-code (ddtree-b12-k2)

- wall: 14.2s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4}`
- stats:
  ```
  emitted: 45 tokens in 0.87s  (51.93 tok/s)
  cycles: 5  committed: 49  accepted: 39  τ=7.800  mean_committed=9.800
  ```

**Output:**

```
     for i in range(len(numbers)):
         for j in range(i + 1, len(numbers)):
             if abs(numbers[i] - numbers[j]) < threshold:
                 return True
     return False<|endoftext|>
```

