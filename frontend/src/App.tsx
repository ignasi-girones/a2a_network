/**
 * Phase 4 App — multi-round deliberation UI.
 *
 * Layout:
 *   ┌────────────────────────────────────────────────┐
 *   │  Header (logo + brand)                         │
 *   ├────────────────────────────────────────────────┤
 *   │  DebateHeroSection (claim, round, active role) │
 *   ├────────────┬───────────────────────────────────┤
 *   │ Input      │ AgentRosterPanel                  │
 *   │ Verdict    │ BeliefTrajectoryChart             │
 *   ├────────────┴───────────────────────────────────┤
 *   │ DiscussionLedgerView (chat-style threaded)     │
 *   ├────────────────────────────────────────────────┤
 *   │ HabermasTable (when synthesis emits validity)  │
 *   ├────────────────────────────────────────────────┤
 *   │ DAG plan + raw event timeline (collapsible)    │
 *   └────────────────────────────────────────────────┘
 */

import { useCallback, useState } from 'react';
import { Network, ChevronDown, ChevronUp } from 'lucide-react';
import { startDebateStream } from './api/sse';
import { AgentRosterPanel } from './components/AgentRosterPanel';
import { BeliefTrajectoryChart } from './components/BeliefTrajectoryChart';
import { DebateGraph } from './components/DebateGraph';
import { DebateHeroSection } from './components/DebateHeroSection';
import { DebateTimeline } from './components/DebateTimeline';
import { DiscussionLedgerView } from './components/DiscussionLedgerView';
import { HabermasTable } from './components/HabermasTable';
import { PromptInput } from './components/PromptInput';
import { VerdictDisplay } from './components/VerdictDisplay';
import type {
  AporiaDiagnosis,
  BeliefSeries,
  DebateEvent,
  DebateState,
  DeliberationState,
  LedgerEntry,
  RoleId,
  RoundMeta,
  SubtaskRuntime,
  TaskPlan,
  ValidityClaimRow,
} from './types';

interface ExtendedState extends DebateState {
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
  beliefs: Record<string, BeliefSeries>;
  aporia: AporiaDiagnosis | null;
  habermas: ValidityClaimRow[] | null;
  deliberation: DeliberationState;
}

function emptyDeliberation(): DeliberationState {
  return {
    claim: null,
    goal: null,
    max_rounds: 3,
    current_round: 0,
    rounds: {},
    ledger: [],
    active_role: null,
    terminated_reason: null,
  };
}

function applyEventToBeliefs(
  prev: Record<string, BeliefSeries>,
  event: DebateEvent,
): Record<string, BeliefSeries> {
  if (event.stage !== 'belief_update') return prev;
  const agent = event.data?.agent;
  if (!agent) return prev;
  const existing: BeliefSeries = prev[agent] ?? {
    agent,
    role_id: event.data?.role_id ?? null,
    claim: event.data?.claim ?? '',
    samples: [],
  };
  const sample = {
    t: existing.samples.length,
    log_odds: event.data?.log_odds ?? 0,
    delta: event.data?.delta ?? 0,
    llr: event.data?.llr ?? 0,
    rationale: event.data?.rationale ?? '',
    phase: event.data?.phase ?? 'post_response',
  };
  return {
    ...prev,
    [agent]: {
      ...existing,
      role_id: existing.role_id ?? event.data?.role_id ?? null,
      claim: existing.claim || (event.data?.claim ?? ''),
      samples: [...existing.samples, sample],
    },
  };
}

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
        role_id: event.data?.role_id ?? existing.role_id,
        persona: event.data?.persona ?? existing.persona,
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
      [id]: { ...existing, status: 'failed', error: event.data?.error ?? existing.error },
    };
  }
  return prev;
}

function applyEventToDeliberation(
  prev: DeliberationState,
  event: DebateEvent,
): DeliberationState {
  // Anchor claim/goal/max_rounds when the planner emits them
  if (event.stage === 'plan_ready' && event.data?.plan) {
    const p = event.data.plan as TaskPlan;
    return {
      ...prev,
      claim: p.claim ?? prev.claim,
      goal: p.goal ?? prev.goal,
    };
  }
  if (event.stage === 'deliberation_start') {
    return {
      ...prev,
      claim: event.data?.claim ?? prev.claim,
      goal: event.data?.goal ?? prev.goal,
      max_rounds: event.data?.max_rounds ?? prev.max_rounds,
      current_round: 0,
      rounds: {},
      ledger: [],
      active_role: null,
      terminated_reason: null,
    };
  }
  if (event.stage === 'round_start') {
    const rn = event.data?.round_number;
    if (rn === undefined) return prev;
    const meta: RoundMeta = {
      round_number: rn,
      speakers: event.data?.speakers ?? [],
      status: 'active',
    };
    return {
      ...prev,
      current_round: rn,
      rounds: { ...prev.rounds, [rn]: meta },
    };
  }
  if (event.stage === 'round_dispatch') {
    return { ...prev, active_role: (event.data?.role_id as RoleId) ?? prev.active_role };
  }
  if (event.stage === 'ledger_entry' && event.data?.entry) {
    const entry = event.data.entry as LedgerEntry;
    return {
      ...prev,
      ledger: [...prev.ledger, entry],
      active_role: null, // intervention done, agent goes idle
    };
  }
  if (event.stage === 'round_end') {
    const rn = event.data?.round_number;
    if (rn === undefined) return prev;
    const existing = prev.rounds[rn] ?? {
      round_number: rn,
      speakers: [],
      status: 'closed' as const,
    };
    return {
      ...prev,
      rounds: {
        ...prev.rounds,
        [rn]: { ...existing, status: 'closed' as const },
      },
      active_role: null,
    };
  }
  if (event.stage === 'synthesis_start') {
    return { ...prev, active_role: 'synthesizer' };
  }
  if (event.stage === 'deliberation_terminated') {
    return {
      ...prev,
      terminated_reason: event.data?.terminated_reason ?? event.data?.reason ?? prev.terminated_reason,
    };
  }
  if (event.stage === 'deliberation_complete') {
    return {
      ...prev,
      active_role: null,
      terminated_reason:
        event.data?.terminated_reason ?? prev.terminated_reason ?? 'completed',
    };
  }
  return prev;
}

function App() {
  const [state, setState] = useState<ExtendedState>({
    status: 'idle',
    events: [],
    verdict: null,
    error: null,
    plan: null,
    runtime: {},
    beliefs: {},
    aporia: null,
    habermas: null,
    deliberation: emptyDeliberation(),
  });

  const [showSecondary, setShowSecondary] = useState(false);

  const handleSubmit = useCallback(async (prompt: string) => {
    setState({
      status: 'running',
      events: [],
      verdict: null,
      error: null,
      plan: null,
      runtime: {},
      beliefs: {},
      aporia: null,
      habermas: null,
      deliberation: emptyDeliberation(),
    });

    await startDebateStream(
      prompt,
      (event: DebateEvent) => {
        setState((prev) => {
          const nextPlan =
            event.stage === 'plan_ready' && event.data?.plan
              ? (event.data.plan as TaskPlan)
              : prev.plan;
          const baseRuntime =
            nextPlan && nextPlan !== prev.plan
              ? Object.fromEntries(
                  nextPlan.subtasks.map((t) => [
                    t.id,
                    prev.runtime[t.id] ?? { status: 'pending' as const },
                  ]),
                )
              : prev.runtime;

          let aporia = prev.aporia;
          let habermas = prev.habermas;
          if (event.stage === 'aporia_detected') {
            aporia = {
              detected: event.data?.detected ?? true,
              reason: (event.data as { reason?: string })?.reason ?? '',
              n_agents: event.data?.n_agents ?? 0,
              spread: event.data?.spread ?? null,
              mean_total_movement: event.data?.mean_total_movement ?? null,
              agent_finals: event.data?.agent_finals,
            };
          }
          if (event.stage === 'habermas_table' && event.data?.validity_claims) {
            habermas = event.data.validity_claims;
          }

          return {
            ...prev,
            events: [...prev.events, event],
            plan: nextPlan,
            runtime: applyEventToRuntime(baseRuntime, event),
            beliefs: applyEventToBeliefs(prev.beliefs, event),
            aporia,
            habermas,
            deliberation: applyEventToDeliberation(prev.deliberation, event),
          };
        });
      },
      (verdict: string) => {
        setState((prev) => ({ ...prev, status: 'completed', verdict }));
      },
      (error: string) => {
        setState((prev) => ({ ...prev, status: 'error', error }));
      },
    );
  }, []);

  return (
    <div className="min-h-screen relative">
      {/* Header */}
      <header className="relative z-10 border-b border-slate-200/60 bg-white/40 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm">
              <Network size={16} className="text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 leading-tight">
                A2A Deliberative Network
              </h1>
              <p className="text-[10px] text-slate-500 leading-tight">
                Protocolo A2A v1.0.0 · Panel dialéctico de 6 roles · Blackboard multi-ronda
              </p>
            </div>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">Fase 4</span>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-4 md:px-6 py-6">
        {/* Hero with claim + round indicator */}
        <DebateHeroSection state={state.deliberation} status={state.status} />

        {/* Two-column: input/verdict + roster/chart */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <div className="lg:col-span-1 space-y-4">
            <div className="glass rounded-xl p-4">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                Tema de debate
              </h2>
              <PromptInput
                onSubmit={handleSubmit}
                disabled={state.status === 'running'}
              />
            </div>
            <VerdictDisplay
              verdict={state.verdict}
              error={state.error}
              status={state.status}
              lastEvent={
                state.events.length > 0
                  ? state.events[state.events.length - 1]
                  : null
              }
            />
          </div>

          <div className="lg:col-span-2 space-y-4">
            <div className="glass rounded-xl p-4">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                Panel deliberativo
              </h2>
              <AgentRosterPanel state={state.deliberation} />
            </div>

            <div className="glass rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Trayectorias bayesianas
                </h2>
                <span className="text-[10px] text-slate-400">
                  log-odds del claim · evolución por ronda
                </span>
              </div>
              <BeliefTrajectoryChart
                series={Object.values(state.beliefs)}
                claim={state.deliberation.claim ?? state.plan?.claim ?? null}
              />
              {state.aporia?.detected && (
                <div className="mt-3 border border-amber-300 bg-amber-50/60 rounded-lg p-2.5 text-xs text-amber-900">
                  <span className="font-semibold">Aporía detectada:</span>{' '}
                  spread {state.aporia.spread?.toFixed(2) ?? 'n/a'}, movimiento
                  medio {state.aporia.mean_total_movement?.toFixed(2) ?? 'n/a'}.
                  Disruptor erístico lanzado.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Discussion Ledger — primary narrative view */}
        <section className="mb-6">
          <div className="flex items-center justify-between mb-3 px-1">
            <h2 className="text-sm font-semibold text-slate-700">
              Discusión
            </h2>
            <span className="text-[10px] text-slate-400">
              Cada agente lee el ledger completo en cada ronda
            </span>
          </div>
          <DiscussionLedgerView
            entries={state.deliberation.ledger}
            maxRounds={state.deliberation.max_rounds}
            currentRound={state.deliberation.current_round}
            activeRole={state.deliberation.active_role}
          />
        </section>

        {/* Habermas validity table */}
        {state.habermas && (
          <section className="mb-6">
            <HabermasTable claims={state.habermas} />
          </section>
        )}

        {/* Secondary panels (DAG + raw timeline) */}
        <section className="mb-6">
          <button
            onClick={() => setShowSecondary((v) => !v)}
            className="w-full flex items-center justify-between glass rounded-xl px-4 py-3 hover:shadow-md transition-shadow"
          >
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Información técnica · DAG del plan & log de eventos
            </span>
            {showSecondary ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showSecondary && (
            <div className="mt-3 space-y-4">
              <div className="glass rounded-xl p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                  Grafo del plan
                </h3>
                <DebateGraph plan={state.plan} runtime={state.runtime} />
              </div>
              <div className="glass rounded-xl p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                  Log de eventos ({state.events.length})
                </h3>
                <div className="overflow-y-auto max-h-[40vh]">
                  <DebateTimeline events={state.events} plan={state.plan} />
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
