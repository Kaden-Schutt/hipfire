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
  stateMode: string;
  positionPolicy: string;
  featureFlags: readonly string[];
}

export interface PrefixCheckpointManifest {
  fingerprint: PrefixCheckpointFingerprint;
  tokenPrefixHash: string;
  prefixLen: number;
  stateKinds: readonly SessionStateKind[];
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
    bytes: Math.max(0, Math.floor(input.bytes)),
    createdAtMs: input.createdAtMs,
    lastUsedAtMs: input.lastUsedAtMs ?? input.createdAtMs,
    hitCount: Math.max(0, Math.floor(input.hitCount ?? 0)),
    checksums: input.checksums ?? {},
  };
}

export function prefixCheckpointCacheKey(
  manifest: Pick<PrefixCheckpointManifest, "fingerprint" | "tokenPrefixHash" | "prefixLen" | "stateKinds">,
): string {
  const fingerprint = normalizeFingerprint(manifest.fingerprint);
  return [
    fingerprint.modelArtifactDigest,
    fingerprint.architectureId,
    fingerprint.tokenizerHash,
    fingerprint.chatTemplateHash,
    fingerprint.runtimeConfigHash,
    fingerprint.stateMode,
    fingerprint.positionPolicy,
    fingerprint.featureFlags.join("+"),
    manifest.tokenPrefixHash,
    manifest.prefixLen,
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
