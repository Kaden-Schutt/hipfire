import { describe, expect, test } from "bun:test";
import {
  createPrefixCheckpointManifest,
  prefixCheckpointCacheKey,
  prefixTokensHash,
  prefixCheckpointCompatible,
  spillEligibility,
  touchPrefixCheckpointManifest,
  type PrefixCheckpointFingerprint,
} from "./state_cache";

const fingerprint: PrefixCheckpointFingerprint = {
  modelArtifactDigest: "sha256:model",
  architectureId: "qwen35",
  tokenizerHash: "sha256:tok",
  chatTemplateHash: "sha256:chat",
  runtimeConfigHash: "sha256:runtime",
  stateMode: "q8+deltanet",
  positionPolicy: "rope-yarn",
  featureFlags: ["qwen35", "prefill_batch"],
};

describe("prefix state cache keys", () => {
  test("hashes token prefixes deterministically", () => {
    expect(prefixTokensHash([1, 2, 3])).toBe(prefixTokensHash([1, 2, 3]));
    expect(prefixTokensHash([1, 2, 3])).not.toBe(prefixTokensHash([1, 2, 4]));
    expect(prefixTokensHash([1, 2, 3])).not.toBe(prefixTokensHash([1, 2, 3, 4]));
  });

  test("includes full model/runtime/state fingerprint in cache key", () => {
    const a = createPrefixCheckpointManifest({
      fingerprint,
      prefixTokens: [10, 11, 12],
      stateKinds: ["deltanet_recurrent", "attention_kv"],
      bytes: 4096,
      createdAtMs: 100,
    });
    const b = createPrefixCheckpointManifest({
      fingerprint: {
        ...fingerprint,
        stateMode: "asym3+deltanet",
      },
      prefixTokens: [10, 11, 12],
      stateKinds: ["attention_kv", "deltanet_recurrent"],
      bytes: 4096,
      createdAtMs: 100,
    });

    expect(a.stateKinds).toEqual(["attention_kv", "deltanet_recurrent"]);
    expect(prefixCheckpointCacheKey(a)).not.toBe(prefixCheckpointCacheKey(b));
  });

  test("requires all recurrent state kinds for compatibility", () => {
    const checkpoint = createPrefixCheckpointManifest({
      fingerprint,
      prefixTokens: [10, 11, 12, 13],
      stateKinds: ["attention_kv", "deltanet_recurrent"],
      bytes: 4096,
      createdAtMs: 100,
    });

    expect(prefixCheckpointCompatible(checkpoint, {
      fingerprint,
      prefixTokens: [10, 11, 12, 13],
      requiredStateKinds: ["attention_kv", "deltanet_recurrent"],
    })).toBe(true);
    expect(prefixCheckpointCompatible(checkpoint, {
      fingerprint,
      prefixTokens: [10, 11, 12, 13],
      requiredStateKinds: ["attention_kv", "mamba_ssm"],
    })).toBe(false);
    expect(prefixCheckpointCompatible(checkpoint, {
      fingerprint,
      prefixTokens: [10, 11, 12, 14],
      requiredStateKinds: ["attention_kv", "deltanet_recurrent"],
    })).toBe(false);
  });
});

describe("state cache spill guardrails", () => {
  test("refuses active, pinned, unknown, and incomplete checkpoints", () => {
    const manifest = createPrefixCheckpointManifest({
      fingerprint,
      prefixTokens: [1, 2, 3],
      stateKinds: ["attention_kv", "mamba_ssm", "mamba_conv"],
      bytes: 2048,
      createdAtMs: 1,
      checksums: {
        attention_kv: "sha256:kv",
        mamba_ssm: "sha256:ssm",
        mamba_conv: "sha256:conv",
      },
    });

    expect(spillEligibility(manifest, {
      activeSession: false,
      pinned: false,
      knownArchitecture: true,
    })).toEqual({ spillable: true, reason: "spillable" });
    expect(spillEligibility(manifest, {
      activeSession: true,
      pinned: false,
      knownArchitecture: true,
    }).reason).toBe("active_session");
    expect(spillEligibility(manifest, {
      activeSession: false,
      pinned: true,
      knownArchitecture: true,
    }).reason).toBe("pinned");
    expect(spillEligibility(manifest, {
      activeSession: false,
      pinned: false,
      knownArchitecture: false,
    }).reason).toBe("unknown_architecture");

    const incomplete = createPrefixCheckpointManifest({
      fingerprint,
      prefixTokens: [1, 2, 3],
      stateKinds: ["attention_kv", "mamba_ssm"],
      bytes: 2048,
      createdAtMs: 1,
      checksums: { attention_kv: "sha256:kv" },
    });
    expect(spillEligibility(incomplete, {
      activeSession: false,
      pinned: false,
      knownArchitecture: true,
    }).reason).toBe("missing_checksum");
  });

  test("normalizes keys and kind order in compatibility checks", () => {
    const baseManifest = createPrefixCheckpointManifest({
      fingerprint: {
        ...fingerprint,
        featureFlags: ["prefill_batch", "qwen35"],
      },
      prefixTokens: [1, 2, 3],
      stateKinds: ["mamba_ssm", "attention_kv", "deltanet_recurrent"],
      bytes: 1024,
      createdAtMs: 10,
    });

    const reorderedManifest = createPrefixCheckpointManifest({
      fingerprint: {
        ...fingerprint,
        featureFlags: ["qwen35", "prefill_batch"],
      },
      prefixTokens: [1, 2, 3],
      stateKinds: ["deltanet_recurrent", "attention_kv", "mamba_ssm"],
      bytes: 1024,
      createdAtMs: 10,
    });

    expect(prefixCheckpointCacheKey(baseManifest)).toBe(prefixCheckpointCacheKey(reorderedManifest));
  });

  test("updates touch metadata without mutating compatibility inputs", () => {
    const manifest = createPrefixCheckpointManifest({
      fingerprint,
      prefixTokens: [1, 2, 3],
      stateKinds: ["attention_kv", "deltanet_recurrent"],
      bytes: 2048,
      createdAtMs: 123,
    });

    const touched = touchPrefixCheckpointManifest(manifest, 999);
    expect(touched.hitCount).toBe(manifest.hitCount + 1);
    expect(touched.lastUsedAtMs).toBe(999);
    expect(touched.createdAtMs).toBe(manifest.createdAtMs);
  });
});
