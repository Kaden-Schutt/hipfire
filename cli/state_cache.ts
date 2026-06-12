import {
  normalizeFeatureFlags,
  type SessionStateKind,
} from "./session_state";

export interface PrefixCheckpointFingerprint {
  modelArtifactDigest: string;
  architectureId: number | string;
  tokenizerHash: string;
  chatTemplateHash: string;
  runtimeConfigHash: string;
  cacheNamespaceHash?: string;
  stateMode: string;
  positionPolicy: string;
  featureFlags: readonly string[];
}

export interface PrefixCheckpointManifest {
  fingerprint: PrefixCheckpointFingerprint;
  tokenPrefixHash: string;
  prefixLen: number;
  stateKinds: readonly SessionStateKind[];
  runtimeState: "metadata_only" | "resident" | "attachable";
  runtimeStateHandle?: string;
  runtimeLogicalPosition?: number;
  daemonPrefixHash?: string;
  daemonPrefixLen?: number;
  bytes: number;
  createdAtMs: number;
  lastUsedAtMs: number;
  hitCount: number;
  checksums: Partial<Record<SessionStateKind, string>>;
}

export interface CreatePrefixCheckpointManifestInput {
  fingerprint: PrefixCheckpointFingerprint;
  prefixTokens: readonly number[];
  stateKinds: readonly SessionStateKind[];
  bytes: number;
  runtimeState?: "metadata_only" | "resident" | "attachable";
  runtimeStateHandle?: string;
  runtimeLogicalPosition?: number;
  daemonPrefixHash?: string;
  daemonPrefixLen?: number;
  createdAtMs: number;
  lastUsedAtMs?: number;
  hitCount?: number;
  checksums?: Partial<Record<SessionStateKind, string>>;
}

export interface PrefixCheckpointCompatibilityInput {
  fingerprint: PrefixCheckpointFingerprint;
  prefixTokens: readonly number[];
  requiredStateKinds: readonly SessionStateKind[];
}

export interface PrefixCheckpointDaemonCompatibilityInput {
  fingerprint: PrefixCheckpointFingerprint;
  daemonPrefixHash: string;
  daemonPrefixLen: number;
  requiredStateKinds: readonly SessionStateKind[];
}

export interface SpillEligibilityInput {
  activeSession: boolean;
  pinned: boolean;
  knownArchitecture: boolean;
}

export function normalizeFingerprint(
  fingerprint: PrefixCheckpointFingerprint,
): PrefixCheckpointFingerprint {
  return {
    ...fingerprint,
    featureFlags: normalizeFeatureFlags(fingerprint.featureFlags),
  };
}

export function normalizeStateKinds(
  stateKinds: readonly SessionStateKind[],
): readonly SessionStateKind[] {
  return [...new Set(stateKinds)].sort();
}

export function prefixTokensHash(tokens: readonly number[]): string {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (const token of tokens) {
    let value = BigInt.asUintN(32, BigInt(Math.floor(token)));
    for (let byte = 0; byte < 4; byte += 1) {
      hash ^= value & 0xffn;
      hash = BigInt.asUintN(64, hash * prime);
      value >>= 8n;
    }
  }
  return hash.toString(16).padStart(16, "0");
}

export function createPrefixCheckpointManifest(
  input: CreatePrefixCheckpointManifestInput,
): PrefixCheckpointManifest {
  return {
    fingerprint: normalizeFingerprint(input.fingerprint),
    tokenPrefixHash: prefixTokensHash(input.prefixTokens),
    prefixLen: input.prefixTokens.length,
    stateKinds: normalizeStateKinds(input.stateKinds),
    runtimeState: input.runtimeState ?? "metadata_only",
    runtimeStateHandle: input.runtimeStateHandle,
    runtimeLogicalPosition: input.runtimeLogicalPosition,
    daemonPrefixHash: input.daemonPrefixHash,
    daemonPrefixLen: input.daemonPrefixLen,
    bytes: Math.max(0, Math.floor(input.bytes)),
    createdAtMs: input.createdAtMs,
    lastUsedAtMs: input.lastUsedAtMs ?? input.createdAtMs,
    hitCount: Math.max(0, Math.floor(input.hitCount ?? 0)),
    checksums: input.checksums ?? {},
  };
}

export function prefixCheckpointAttachable(manifest: PrefixCheckpointManifest): boolean {
  return manifest.runtimeState === "attachable"
    && typeof manifest.runtimeStateHandle === "string"
    && manifest.runtimeStateHandle.length > 0
    && typeof manifest.daemonPrefixHash === "string"
    && /^[0-9a-f]{32}$/.test(manifest.daemonPrefixHash)
    && typeof manifest.daemonPrefixLen === "number"
    && Number.isInteger(manifest.daemonPrefixLen)
    && manifest.daemonPrefixLen >= 0;
}

export function prefixCheckpointCacheKey(
  manifest: Pick<PrefixCheckpointManifest, "fingerprint" | "tokenPrefixHash" | "prefixLen" | "stateKinds" | "daemonPrefixHash" | "daemonPrefixLen">,
): string {
  const fingerprint = normalizeFingerprint(manifest.fingerprint);
  return [
    fingerprint.modelArtifactDigest,
    fingerprint.architectureId,
    fingerprint.tokenizerHash,
    fingerprint.chatTemplateHash,
    fingerprint.runtimeConfigHash,
    fingerprint.cacheNamespaceHash ?? "",
    fingerprint.stateMode,
    fingerprint.positionPolicy,
    fingerprint.featureFlags.join("+"),
    manifest.tokenPrefixHash,
    manifest.prefixLen,
    manifest.daemonPrefixHash ?? "",
    manifest.daemonPrefixLen ?? "",
    normalizeStateKinds(manifest.stateKinds).join("+"),
  ].join("|");
}

export function prefixCheckpointCompatible(
  manifest: PrefixCheckpointManifest,
  input: PrefixCheckpointCompatibilityInput,
): boolean {
  const required = normalizeStateKinds(input.requiredStateKinds);
  const available = new Set(normalizeStateKinds(manifest.stateKinds));
  if (!required.every((kind) => available.has(kind))) return false;
  const candidate = createPrefixCheckpointManifest({
    fingerprint: input.fingerprint,
    prefixTokens: input.prefixTokens,
    stateKinds: manifest.stateKinds,
    bytes: manifest.bytes,
    createdAtMs: manifest.createdAtMs,
  });
  return prefixCheckpointCacheKey(candidate) === prefixCheckpointCacheKey(manifest);
}

export function prefixCheckpointDaemonCompatible(
  manifest: PrefixCheckpointManifest,
  input: PrefixCheckpointDaemonCompatibilityInput,
): boolean {
  const required = normalizeStateKinds(input.requiredStateKinds);
  const available = new Set(normalizeStateKinds(manifest.stateKinds));
  if (!required.every((kind) => available.has(kind))) return false;
  if (!prefixCheckpointAttachable(manifest)) return false;
  if (manifest.daemonPrefixHash !== input.daemonPrefixHash) return false;
  if (manifest.daemonPrefixLen !== input.daemonPrefixLen) return false;
  const a = normalizeFingerprint(manifest.fingerprint);
  const b = normalizeFingerprint(input.fingerprint);
  return a.modelArtifactDigest === b.modelArtifactDigest
    && String(a.architectureId) === String(b.architectureId)
    && a.tokenizerHash === b.tokenizerHash
    && a.chatTemplateHash === b.chatTemplateHash
    && a.cacheNamespaceHash === b.cacheNamespaceHash
    && a.stateMode === b.stateMode
    && a.positionPolicy === b.positionPolicy
    && a.featureFlags.join("+") === b.featureFlags.join("+");
}

export function spillEligibility(
  manifest: PrefixCheckpointManifest,
  input: SpillEligibilityInput,
): { spillable: boolean; reason: string } {
  if (input.activeSession) return { spillable: false, reason: "active_session" };
  if (input.pinned) return { spillable: false, reason: "pinned" };
  if (!input.knownArchitecture) return { spillable: false, reason: "unknown_architecture" };
  for (const stateKind of manifest.stateKinds) {
    if (!manifest.checksums[stateKind]) return { spillable: false, reason: "missing_checksum" };
  }
  return { spillable: true, reason: "spillable" };
}

export function touchPrefixCheckpointManifest(
  manifest: PrefixCheckpointManifest,
  nowMs: number,
): PrefixCheckpointManifest {
  return {
    ...manifest,
    hitCount: manifest.hitCount + 1,
    lastUsedAtMs: nowMs,
  };
}

export function selectResidentCheckpointEvictions(
  manifests: Iterable<PrefixCheckpointManifest>,
  maxCheckpoints: number,
): PrefixCheckpointManifest[] {
  const limit = Math.max(0, Math.floor(maxCheckpoints));
  const attachable = [...manifests].filter(prefixCheckpointAttachable);
  if (attachable.length <= limit) return [];
  return attachable
    .sort((a, b) => {
      const byLastUsed = a.lastUsedAtMs - b.lastUsedAtMs;
      if (byLastUsed !== 0) return byLastUsed;
      return a.createdAtMs - b.createdAtMs;
    })
    .slice(0, attachable.length - limit);
}

export function removeAttachableManifestsByRuntimeHandle(
  caches: Iterable<Map<string, PrefixCheckpointManifest>>,
  runtimeStateHandle: string,
): number {
  if (runtimeStateHandle.length === 0) return 0;
  let removed = 0;
  for (const cache of caches) {
    for (const [key, manifest] of cache.entries()) {
      if (manifest.runtimeState === "attachable" && manifest.runtimeStateHandle === runtimeStateHandle) {
        cache.delete(key);
        removed += 1;
      }
    }
  }
  return removed;
}
