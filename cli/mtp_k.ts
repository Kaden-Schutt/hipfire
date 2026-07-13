export function parseMtpK(raw: string | undefined, source: string): number | undefined {
  if (raw === undefined) return undefined;
  if (!/^[1-8]$/.test(raw)) {
    throw new Error(`Invalid ${source} '${raw}'; expected an integer from 1 to 8`);
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > 8) {
    throw new Error(`Invalid ${source} '${raw}'; expected an integer from 1 to 8`);
  }
  return value;
}

export function validateMtpDraftMaxFlag(
  draftMax: string | null,
  speculation: string,
  modelIdentity?: string | null,
): void {
  const mtpDraftMax = mtpDraftMaxForSpeculation(draftMax ?? undefined, speculation, modelIdentity);
  if (mtpDraftMax !== undefined) parseMtpK(mtpDraftMax, "--draft-max");
}

export function mtpDraftMaxForSpeculation(
  draftMax: string | undefined,
  speculation: string,
  modelIdentity?: string | null,
): string | undefined {
  return draftMaxForMechanism(draftMax, speculation, modelIdentity, "mtp");
}

export function draftMaxForMechanism(
  draftMax: string | undefined,
  speculation: string,
  modelIdentity: string | null | undefined,
  mechanism: "mtp" | "ngram" | "dflash",
): string | undefined {
  if (speculation === mechanism) return draftMax;
  if (mechanism === "mtp" && speculation === "auto" && /deepseek|ds4/i.test(modelIdentity ?? "")) {
    return draftMax;
  }
  return undefined;
}

export function resolveMtpK(
  envMtpK: string | undefined,
  draftMax: string | undefined,
  configuredMtpK: number | undefined,
  mtpEligible = true,
): number {
  const configured = configuredMtpK ?? 3;
  if (!Number.isInteger(configured) || configured < 1 || configured > 8) {
    throw new Error(`Invalid configured MTP K '${configured}'; expected an integer from 1 to 8`);
  }
  if (!mtpEligible) return configured;
  return parseMtpK(envMtpK, "HIPFIRE_MTP_K")
    ?? parseMtpK(draftMax, "HIPFIRE_DRAFT_MAX")
    ?? configured;
}
