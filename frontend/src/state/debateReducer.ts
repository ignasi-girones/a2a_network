/**
 * Pure reducer for debate state.
 *
 * Folds one DebateEvent into the running ExtendedState. Extracted from
 * App.tsx into its own module so:
 *
 *   1. The persistence layer can replay a stored event stream by feeding
 *      events one-by-one through `reduceDebateEvent` and arrive at the
 *      *exact same* state the user had before F5.
 *   2. Vitest can assert on the reducer in isolation, without rendering.
 *
 * Two synthetic stages — emitted by the orchestrator's debate_store on
 * `mark_completed` / `mark_failed` — carry the terminal status:
 *
 *   stage = "verdict"  data = { text: string }   →  status: "completed"
 *   stage = "failed"   data = { error: string }  →  status: "failed"
 *
 * This unifies replay and live-follow: there is no separate `onComplete`
 * callback path; the verdict event is just one more event the reducer
 * consumes. Live mode used to surface the verdict via a separate
 * Task(COMPLETED) frame from the A2A SDK; the persistence layer makes
 * that channel obsolete.
 */

import type {
  AgentPositionsSample,
  ConsensusSnapshot,
  DebateEvent,
  DebateState,
  SubtaskRuntime,
  TaskPlan,
} from '../types';

export interface ExtendedState extends DebateState {
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
  positions: AgentPositionsSample[];
  consensusHistory: ConsensusSnapshot[];
}

export const INITIAL_STATE: ExtendedState = {
  status: 'idle',
  events: [],
  verdict: null,
  error: null,
  plan: null,
  runtime: {},
  positions: [],
  consensusHistory: [],
};

/** Fresh state with status='running' — emitted at debate kick-off. */
export function runningState(): ExtendedState {
  return {
    ...INITIAL_STATE,
    status: 'running',
  };
}

/**
 * Update the runtime map for one of the per-subtask events.
 *
 * - subtask_dispatch → mark running, record worker_id
 * - subtask_done     → mark done, store full text output
 * - subtask_failed   → mark failed, store error
 *
 * Other stages are no-ops here (they go to the timeline log).
 */
function applyEventToRuntime(
  prev: Record<string, SubtaskRuntime>,
  event: DebateEvent,
): Record<string, SubtaskRuntime> {
  const id = event.data?.subtask_id;
  if (!id) return prev;

  const existing = prev[id] ?? { status: 'pending' as const };
  if (event.stage === 'subtask_dispatch') {
    return {
      ...prev,
      [id]: {
        ...existing,
        status: 'running',
        worker_id: event.data?.worker_id ?? existing.worker_id,
      },
    };
  }
  if (event.stage === 'subtask_done') {
    return {
      ...prev,
      [id]: {
        ...existing,
        status: 'done',
        output: event.data?.text ?? event.data?.output_preview ?? existing.output,
      },
    };
  }
  if (event.stage === 'subtask_failed') {
    return {
      ...prev,
      [id]: {
        ...existing,
        status: 'failed',
        error: event.data?.error ?? existing.error,
      },
    };
  }
  return prev;
}

/**
 * Fold one event into the state and return the next state.
 * Pure — no side effects, no setState, just (state, event) → state.
 */
export function reduceDebateEvent(
  prev: ExtendedState,
  event: DebateEvent,
): ExtendedState {
  // Synthetic terminal events from the persistence layer.
  if (event.stage === 'verdict') {
    return {
      ...prev,
      events: [...prev.events, event],
      status: 'completed',
      verdict: typeof event.data?.text === 'string' ? event.data.text : prev.verdict,
    };
  }
  if (event.stage === 'failed') {
    return {
      ...prev,
      events: [...prev.events, event],
      status: 'error',
      error: typeof event.data?.error === 'string' ? event.data.error : prev.error,
    };
  }

  // `plan_ready` events arrive multiple times: once for the initial plan,
  // again after each consensus extension (with the merged plan), and
  // finally when the format_verdict step is appended. We always trust the
  // latest emitted plan but PRESERVE runtime state for any subtask whose
  // status we already track.
  const incomingPlan =
    event.stage === 'plan_ready' && event.data?.plan
      ? (event.data.plan as TaskPlan)
      : null;

  let nextPlan = prev.plan;
  let baseRuntime = prev.runtime;
  if (incomingPlan) {
    const knownIds = new Set(prev.plan?.subtasks.map((t) => t.id) ?? []);
    const mergedSubtasks = [
      ...(prev.plan?.subtasks ?? []),
      ...incomingPlan.subtasks.filter((t) => !knownIds.has(t.id)),
    ];
    nextPlan = {
      ...incomingPlan,
      subtasks: mergedSubtasks,
    };
    baseRuntime = Object.fromEntries(
      mergedSubtasks.map((t) => [
        t.id,
        prev.runtime[t.id] ?? { status: 'pending' as const },
      ]),
    );
  }

  // agent_positions samples per round.
  let nextPositions = prev.positions;
  if (
    event.stage === 'agent_positions' &&
    event.data?.positions &&
    typeof event.data?.round === 'number'
  ) {
    const sample: AgentPositionsSample = {
      round: event.data.round,
      positions: event.data.positions,
      agreement_score: event.data.agreement_score,
    };
    const without = prev.positions.filter((p) => p.round !== sample.round);
    nextPositions = [...without, sample].sort((a, b) => a.round - b.round);
  }

  // Consensus snapshots — agent_positions carries the full payload, and
  // consensus_check refines it with the metric breakdown.
  let nextConsensus = prev.consensusHistory;
  if (
    event.stage === 'agent_positions' &&
    typeof event.data?.round === 'number' &&
    typeof event.data?.agreement_score === 'number'
  ) {
    const snap: ConsensusSnapshot = {
      round: event.data.round,
      agreement_score: event.data.agreement_score,
      reason: event.data.reason,
      positions: event.data.positions,
      shared_points: event.data.shared_points ?? [],
      remaining_disagreements: event.data.remaining_disagreements ?? [],
      components: event.data.components,
      movement: event.data.movement,
      concessions: event.data.concessions,
    };
    const without = prev.consensusHistory.filter((s) => s.round !== snap.round);
    nextConsensus = [...without, snap].sort((a, b) => a.round - b.round);
  } else if (
    event.stage === 'consensus_check' &&
    typeof event.data?.agreement_score === 'number'
  ) {
    const round =
      typeof event.data.round === 'number'
        ? event.data.round
        : prev.consensusHistory.length;
    const existing = prev.consensusHistory.find((s) => s.round === round);
    const merged: ConsensusSnapshot = {
      round,
      agreement_score: event.data.agreement_score,
      reason: event.data.reason ?? existing?.reason,
      positions: event.data.positions ?? existing?.positions,
      shared_points:
        event.data.shared_points ?? existing?.shared_points ?? [],
      remaining_disagreements:
        event.data.remaining_disagreements ??
        existing?.remaining_disagreements ??
        [],
      components: event.data.components ?? existing?.components,
      movement: existing?.movement,
      concessions: existing?.concessions,
    };
    const without = prev.consensusHistory.filter((s) => s.round !== round);
    nextConsensus = [...without, merged].sort((a, b) => a.round - b.round);
  }

  return {
    ...prev,
    events: [...prev.events, event],
    plan: nextPlan,
    runtime: applyEventToRuntime(baseRuntime, event),
    positions: nextPositions,
    consensusHistory: nextConsensus,
  };
}

/**
 * Replay a list of events in order, returning the final state.
 * Used after F5 to rebuild the UI from the persisted event stream.
 */
export function replayEvents(
  events: DebateEvent[],
  base: ExtendedState = runningState(),
): ExtendedState {
  return events.reduce(reduceDebateEvent, base);
}
