import {
  schedulerPolicyForPriority,
  shouldDispatchOpportunistic,
  type SchedulerPolicyEnv,
  type SchedulerPriorityPolicy,
} from "./scheduler_policy";
import {
  sessionsCompatibleForPrefill,
  type RequestSessionDraft,
} from "./session_state";

export interface QueuedPrefillRequest {
  session: RequestSessionDraft;
  enqueuedAtMs: number;
}

export interface NextPrefillBatchInput {
  nowMs: number;
}

export interface PrefillBatchSelection {
  sessions: RequestSessionDraft[];
  policy: SchedulerPriorityPolicy;
  totalPromptTokens: number;
  totalSuffixTokens: number;
  maxPromptTokens: number;
}

export class PriorityPrefillScheduler {
  private readonly buckets: QueuedPrefillRequest[][] = Array.from(
    { length: 256 },
    () => [],
  );
  private readonly queuedIds = new Set<string>();
  private queuedCount = 0;

  constructor(private readonly env: SchedulerPolicyEnv = process.env) {}

  get size(): number {
    return this.queuedCount;
  }

  enqueue(session: RequestSessionDraft, enqueuedAtMs: number): void {
    if (this.queuedIds.has(session.id)) {
      throw new Error(`request session is already queued: ${session.id}`);
    }
    this.buckets[session.priority].push({ session, enqueuedAtMs });
    this.queuedIds.add(session.id);
    this.queuedCount += 1;
  }

  cancel(id: string): boolean {
    if (!this.queuedIds.has(id)) return false;
    for (const bucket of this.buckets) {
      const index = bucket.findIndex((entry) => entry.session.id === id);
      if (index >= 0) {
        bucket.splice(index, 1);
        this.queuedIds.delete(id);
        this.queuedCount -= 1;
        return true;
      }
    }
    this.queuedIds.delete(id);
    return false;
  }

  nextPrefillBatch(input: NextPrefillBatchInput): PrefillBatchSelection | undefined {
    for (let priority = 0; priority < this.buckets.length; priority += 1) {
      const bucket = this.buckets[priority];
      if (bucket.length === 0) continue;

      const candidate = this.selectFromBucket(priority, bucket, input.nowMs);
      if (!candidate) return undefined;

      this.removeSelected(candidate.sessions);
      return candidate;
    }

    return undefined;
  }

  previewNextPrefillBatch(
    input: NextPrefillBatchInput & {
      incomingSession?: RequestSessionDraft;
      incomingEnqueuedAtMs?: number;
    },
  ): PrefillBatchSelection | undefined {
    for (let priority = 0; priority < this.buckets.length; priority += 1) {
      const bucket = [...this.buckets[priority]];
      if (input.incomingSession && input.incomingSession.priority === priority) {
        bucket.push({
          session: input.incomingSession,
          enqueuedAtMs: input.incomingEnqueuedAtMs ?? input.nowMs,
        });
      }

      if (bucket.length === 0) continue;

      const candidate = this.selectFromBucket(priority, bucket, input.nowMs);
      if (!candidate) return undefined;
      if (!input.incomingSession) {
        return candidate;
      }
      if (candidate.sessions.some((s) => s.id === input.incomingSession!.id)) {
        return candidate;
      }
      return undefined;
    }

    return undefined;
  }

  private selectFromBucket(
    priority: number,
    bucket: QueuedPrefillRequest[],
    nowMs: number,
  ): PrefillBatchSelection | undefined {
    const first = bucket[0];
    const policy = schedulerPolicyForPriority(first.session.priority, this.env);
    const compatible = bucket
      .filter((entry) => sessionsCompatibleForPrefill({
        a: first.session,
        b: entry.session,
      }))
      .slice(0, policy.maxBatchSize);
    const totalSuffixTokens = compatible.reduce(
      (sum, entry) => sum + entry.session.suffixTokens.length,
      0,
    );

    if (policy.priorityClass === "opportunistic") {
      const dispatch = shouldDispatchOpportunistic({
        compatibleQueuedTokens: totalSuffixTokens,
        scheduleClear: !this.hasQueuedHigherPriority(priority),
        targetPairTokens: policy.targetPairTokens,
      });
      return dispatch ? this.selection(compatible, policy) : undefined;
    }

    const waitedMs = nowMs - first.enqueuedAtMs;
    if (compatible.length >= policy.maxBatchSize || waitedMs >= policy.coalesceWaitMs) {
      return this.selection(compatible, policy);
    }
    return undefined;
  }

  private hasQueuedHigherPriority(priority: number): boolean {
    for (let p = 0; p < priority; p += 1) {
      if (this.buckets[p].length > 0) return true;
    }
    return false;
  }

  private selection(
    entries: QueuedPrefillRequest[],
    policy: SchedulerPriorityPolicy,
  ): PrefillBatchSelection {
    const sessions = entries.map((entry) => entry.session);
    return {
      sessions,
      policy,
      totalPromptTokens: sessions.reduce(
        (sum, session) => sum + session.promptTokens.length,
        0,
      ),
      totalSuffixTokens: sessions.reduce(
        (sum, session) => sum + session.suffixTokens.length,
        0,
      ),
      maxPromptTokens: sessions.reduce(
        (max, session) => Math.max(max, session.promptTokens.length),
        0,
      ),
    };
  }

  private removeSelected(sessions: readonly RequestSessionDraft[]): void {
    for (const session of sessions) {
      this.cancel(session.id);
    }
  }
}
