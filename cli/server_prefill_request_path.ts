export type ServerPrefillPendingSelectionInput = {
  stream?: boolean;
  responsesRequest?: boolean;
  eligible: boolean;
  hasScheduler: boolean;
  queuePreviewReason?: string;
};

export function shouldQueueServerPrefillPending(input: ServerPrefillPendingSelectionInput): boolean {
  return input.stream !== true
    && input.responsesRequest !== true
    && input.eligible
    && input.hasScheduler
    && input.queuePreviewReason !== "selected"
    && input.queuePreviewReason !== "not_eligible";
}
