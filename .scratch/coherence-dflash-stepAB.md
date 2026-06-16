# Coherence battery — DFlash / DDTree

- commit: b50f5ff4
- branch: feature/paro-transparent-loading
- date:   2026-06-11T11:37:10+02:00
- mode:   short
- kv_mode: q8
- target: /home/bjoern/.hipfire/models/qwen3.6-27b.mq4
- draft:  /home/bjoern/.hipfire/models/qwen36-27b-dflash-mq4.hfq

Hard-fail thresholds (three-tier, see CLAUDE.md DFlash Coherence Gate):
  Tier 1 (first 128): unique_token_ratio < 0.15 OR max_single_token_frequency > 0.50
  Tier 2 (last 128):  unique_token_ratio < 0.30 OR max_single_token_frequency > 0.50
  Tier 3 (full, SOFT flag — human eyeball, not commit-blocking):
          consecutive-3gram repetition density > 0.50 in final half
          OR full-output unique_token_ratio < 0.10
Plus: zero tokens / panic. (Path A failure mode — see commit 6c84b13.)

## 27b-dflash-prose (dflash)

- wall: 50.0s  status: **OK**
- detector: `{"ok": true, "t1_hard": false, "t2_hard": false, "soft_warn": false, "tier3_warn": false, "total": 128, "unique": 87, "unique_ratio": 0.68, "max_freq": 0.062, "max_tok": 13, "max_count": 8, "tier2": {"total": 128, "unique_ratio": 0.711, "max_freq": 0.062}, "gram_density": 0.0, "full_unique_ratio": 0.631}`
- stats:
  ```
  emitted: 195 tokens in 13.89s  (14.04 tok/s)
  cycles: 97  committed: 291  accepted: 97  τ=1.000  mean_committed=3.000
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to be effectively governed from a single center. The Roman Empire was so vast that it was difficult to maintain control over distant provinces. Communication and transportation were slow, and it was hard to respond quickly to crises in far-flung regions. This led to a weakening of central authority and a rise in local power.
Another significant factor was the constant threat of invasion from outside. The empire faced numerous attacks from various groups, including the Goths, Vandals, and Huns. These invasions drained the empire's resources and weakened its military. The Roman army, once the most powerful in the world, became overstretched and underfunded. It was difficult to maintain a large standing army across such a vast territory. As a result, the empire became vulnerable to external threats.
The Roman Empire also suffered from internal political instability. There were frequent changes in leadership, with emperors often being assassinated or
```

## 27b-dflash-code (dflash)

- wall: 51.2s  status: **OK**
- detector: `{"ok": true, "t1_hard": false, "t2_hard": false, "soft_warn": false, "tier3_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4, "tier2": {"total": 44, "unique_ratio": 0.75, "max_freq": 0.091}, "gram_density": 0.0, "full_unique_ratio": 0.75}`
- stats:
  ```
  emitted: 45 tokens in 0.64s  (70.20 tok/s)
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

## 27b-ddtree-b12-prose (ddtree-b12-k2)

- wall: 73.0s  status: **OK**
- detector: `{"ok": true, "t1_hard": false, "t2_hard": false, "soft_warn": false, "tier3_warn": false, "total": 128, "unique": 87, "unique_ratio": 0.68, "max_freq": 0.062, "max_tok": 13, "max_count": 8, "tier2": {"total": 128, "unique_ratio": 0.672, "max_freq": 0.07}, "gram_density": 0.032, "full_unique_ratio": 0.609}`
- stats:
  ```
  emitted: 192 tokens in 15.12s  (12.70 tok/s)
  cycles: 74  committed: 265  accepted: 117  τ=1.581  mean_committed=3.581
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to be effectively governed from a single center. The Roman Empire was so vast that it was difficult to maintain control over distant provinces. Communication and transportation were slow, and it was hard to respond quickly to crises in far-flung regions. This led to a weakening of central authority and a rise in local power.
The second factor was the constant threat of invasion from outside. The empire was surrounded by numerous hostile tribes and kingdoms. The Huns, Goths, Vandals, and others repeatedly attacked the empire's borders. These invasions drained the empire's resources and weakened its military. The Roman army, once the most powerful in the world, became overstretched and underfunded. It was unable to defend the empire's vast borders effectively.
The third factor was internal political instability. The Roman Empire suffered from frequent civil wars and power struggles. Emperors were often assassinated or over
```

## 27b-ddtree-b12-code (ddtree-b12-k2)

- wall: 52.2s  status: **OK**
- detector: `{"ok": true, "t1_hard": false, "t2_hard": false, "soft_warn": false, "tier3_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4, "tier2": {"total": 44, "unique_ratio": 0.75, "max_freq": 0.091}, "gram_density": 0.0, "full_unique_ratio": 0.75}`
- stats:
  ```
  emitted: 45 tokens in 0.93s  (48.52 tok/s)
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

