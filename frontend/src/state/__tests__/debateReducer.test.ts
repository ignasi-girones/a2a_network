/**
 * Critical invariant for the persistence layer:
 *
 *   replayEvents(events) === events.reduce(reduceDebateEvent, base)
 *
 * If F5'ing the page produces a different state than running the live
 * stream, the whole persistence story breaks. This test feeds a realistic
 * sequence of events twice — once via setState-style step-by-step updates,
 * once via the replay helper — and asserts both produce the same final
 * state.
 */

import { describe, expect, it } from 'vitest';

import type { DebateEvent, TaskPlan } from '../../types';
import {
  INITIAL_STATE,
  reduceDebateEvent,
  replayEvents,
  runningState,
} from '../debateReducer';

const PLAN: TaskPlan = {
  goal: '¿Vale la pena la IA?',
  subtasks: [
    { id: 't1', description: 'normalize', required_skill: 'normalize_input', depends_on: [], perspective: null },
    { id: 't2', description: 'pro', required_skill: 'debate', depends_on: ['t1'], perspective: 'ae1: pro' },
    { id: 't3', description: 'con', required_skill: 'debate', depends_on: ['t1'], perspective: 'ae2: con' },
    { id: 't4', description: 'verdict', required_skill: 'format_verdict', depends_on: ['t2', 't3'], perspective: null },
  ],
  max_workers: 3,
};

const EVENTS: DebateEvent[] = [
  { stage: 'discover', message: 'Workers...', data: {} },
  { stage: 'plan', message: 'Generando plan...', data: {} },
  { stage: 'plan_ready', message: 'Plan listo', data: { plan: PLAN } },
  { stage: 'subtask_dispatch', message: 't1', data: { subtask_id: 't1', worker_id: 'normalizer' } },
  { stage: 'subtask_done', message: 't1 ok', data: { subtask_id: 't1', text: '{"topic":"IA"}' } },
  { stage: 'subtask_dispatch', message: 't2', data: { subtask_id: 't2', worker_id: 'ae1' } },
  { stage: 'subtask_dispatch', message: 't3', data: { subtask_id: 't3', worker_id: 'ae2' } },
  { stage: 'subtask_done', message: 't2 ok', data: { subtask_id: 't2', text: 'Pro argument' } },
  { stage: 'subtask_done', message: 't3 ok', data: { subtask_id: 't3', text: 'Con argument' } },
  {
    stage: 'agent_positions',
    message: 'Posiciones',
    data: {
      round: 0,
      positions: { ae1: 0.0, ae2: 1.0, ae3: 0.5 },
      agreement_score: 0.4,
      shared_points: ['costo importa'],
      remaining_disagreements: ['regulación'],
    },
  },
  {
    stage: 'consensus_check',
    message: 'Score 0.4',
    data: {
      round: 0,
      agreement_score: 0.4,
      reason: 'dispersion=0.6',
      components: { dispersion: 0.4, pairwise_similarity: 0.6 },
    },
  },
  { stage: 'subtask_dispatch', message: 't4', data: { subtask_id: 't4', worker_id: 'feedback' } },
  { stage: 'subtask_done', message: 't4 ok', data: { subtask_id: 't4', text: 'El veredicto.' } },
  { stage: 'verdict', message: 'Veredicto', data: { text: 'El veredicto.' } },
];

describe('reduceDebateEvent', () => {
  it('replay reaches the same final state as live step-by-step', () => {
    // Live: step-by-step (mimics setState((prev) => reduceDebateEvent(prev, ev)))
    const live = EVENTS.reduce(reduceDebateEvent, runningState());

    // Replay: same input, helper used by the post-F5 rehydration path.
    const replay = replayEvents(EVENTS);

    expect(replay).toEqual(live);
  });

  it('synthetic verdict event flips status to completed and stores text', () => {
    const events: DebateEvent[] = [
      { stage: 'discover', message: '', data: {} },
      { stage: 'verdict', message: 'fin', data: { text: 'el verdict' } },
    ];
    const final = replayEvents(events);
    expect(final.status).toBe('completed');
    expect(final.verdict).toBe('el verdict');
  });

  it('synthetic failed event flips status to error and stores message', () => {
    const events: DebateEvent[] = [
      { stage: 'discover', message: '', data: {} },
      { stage: 'failed', message: 'crash', data: { error: 'boom' } },
    ];
    const final = replayEvents(events);
    expect(final.status).toBe('error');
    expect(final.error).toBe('boom');
  });

  it('plan_ready merges subtasks across emissions without losing nodes', () => {
    const initialPlan: TaskPlan = {
      goal: 'g',
      subtasks: [
        { id: 't1', description: '', required_skill: 's', depends_on: [], perspective: null },
      ],
      max_workers: 1,
    };
    const extendedPlan: TaskPlan = {
      goal: 'g',
      subtasks: [
        { id: 't2', description: '', required_skill: 's', depends_on: [], perspective: null },
      ],
      max_workers: 2,
    };
    const events: DebateEvent[] = [
      { stage: 'plan_ready', message: '', data: { plan: initialPlan } },
      { stage: 'plan_ready', message: '', data: { plan: extendedPlan } },
    ];
    const final = replayEvents(events);
    const ids = final.plan?.subtasks.map((t) => t.id);
    expect(ids).toEqual(['t1', 't2']);
  });

  it('subtask_done preserves the worker_id assigned during dispatch', () => {
    const events: DebateEvent[] = [
      { stage: 'subtask_dispatch', message: '', data: { subtask_id: 't1', worker_id: 'ae1' } },
      { stage: 'subtask_done', message: '', data: { subtask_id: 't1', text: 'out' } },
    ];
    const final = replayEvents(events);
    expect(final.runtime.t1).toEqual({
      status: 'done',
      worker_id: 'ae1',
      output: 'out',
    });
  });

  it('agent_positions samples are sorted by round and dedup at the same round', () => {
    const events: DebateEvent[] = [
      {
        stage: 'agent_positions',
        message: '',
        data: { round: 1, positions: { ae1: 0.1 }, agreement_score: 0.3 },
      },
      {
        stage: 'agent_positions',
        message: '',
        data: { round: 0, positions: { ae1: 0.0 }, agreement_score: 0.1 },
      },
      // Re-emission at round 1 with newer score — must replace, not duplicate.
      {
        stage: 'agent_positions',
        message: '',
        data: { round: 1, positions: { ae1: 0.5 }, agreement_score: 0.7 },
      },
    ];
    const final = replayEvents(events);
    expect(final.positions.map((p) => p.round)).toEqual([0, 1]);
    expect(final.positions[1].agreement_score).toBe(0.7);
  });

  it('initial state is idle and untouched', () => {
    expect(INITIAL_STATE.status).toBe('idle');
    expect(INITIAL_STATE.events).toEqual([]);
    expect(INITIAL_STATE.plan).toBeNull();
  });
});
