export const DUMMY_MODEL_TAG = "hipfire:dummy";
export const DUMMY_MODEL_SENTINEL_PATH = "__hipfire_dummy_model__";

export function isDummyModelTag(name: string | null | undefined): boolean {
  return name === DUMMY_MODEL_TAG;
}

export function isDummyModelPath(path: string | null | undefined): boolean {
  return path === DUMMY_MODEL_SENTINEL_PATH;
}

export function resolveDummyModelPath(name: string): string | null {
  return isDummyModelTag(name) ? DUMMY_MODEL_SENTINEL_PATH : null;
}

export function buildDummyLoadMessage(): any {
  return {
    type: "load",
    model: DUMMY_MODEL_SENTINEL_PATH,
    params: {
      dummy_model: true,
      max_seq: 4096,
      physical_cap: 4096,
    },
  };
}
