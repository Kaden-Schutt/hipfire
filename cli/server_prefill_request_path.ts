export type ServerPrefillPendingSelectionInput = {
  stream?: boolean;
  eligible: boolean;
  hasScheduler: boolean;
  queuePreviewReason?: string;
};

export function shouldQueueServerPrefillPending(input: ServerPrefillPendingSelectionInput): boolean {
  return input.stream !== true
    && input.eligible
    && input.hasScheduler
    && input.queuePreviewReason !== "selected"
    && input.queuePreviewReason !== "not_eligible";
}
