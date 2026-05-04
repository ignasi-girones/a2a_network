/**
 * App.tsx — versión con doble vista.
 *
 *   🎭 Vista Usuario  — mesa redonda con personajes (UserView) — sin info técnica
 *   📊 Vista Técnica  — paneles originales con gráficos y log (TechnicalView)
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Network, Theater, BarChart3 } from 'lucide-react';
import { startDebateStream } from './api/sse';
import { TechnicalView } from './views/TechnicalView';
import { UserView } from './views/UserView';
import type {
  AgentPositionsSample,
  ConsensusSnapshot,
  DebateEvent,
  DebateState,
  SubtaskRuntime,
  TaskPlan,
} from './types';
import { modelLabel } from './utils/modelLabel';

interface ModelMap {
  orchestrator?: string;
  normalizer?: string;
  ae1?: string;
  ae2?: string;
  ae3?: string;
  feedback?: string;
  embedding?: string;
}

interface ExtendedState extends DebateState {
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
  positions: AgentPositionsSample[];
  consensusHistory: ConsensusSnapshot[];
}

type ViewMode = 'user' | 'technical';

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

function App() {
  const [state, setState] = useState<ExtendedState>({
    status: 'idle',
    events: [],
    verdict: null,
    error: null,
    plan: null,
    runtime: {},
    positions: [],
    consensusHistory: [],
  });

  const [viewMode, setViewMode] = useState<ViewMode>('user');
  const [activeTopic, setActiveTopic] = useState('');
  const [models, setModels] = useState<ModelMap>({});
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;
    (async () => {
      try {
        const res = await fetch('/api/models');
        if (!res.ok) return;
        const data = (await res.json()) as ModelMap;
        if (!cancelledRef.current) setModels(data);
      } catch {
        // Silencio
      }
    })();
    return () => {
      cancelledRef.current = true;
    };
  }, []);

  const handleSubmit = useCallback(async (prompt: string) => {
    setActiveTopic(prompt);
    setState({
      status: 'running',
      events: [],
      verdict: null,
      error: null,
      plan: null,
      runtime: {},
      positions: [],
      consensusHistory: [],
    });

    await startDebateStream(
      prompt,
      (event: DebateEvent) => {
        setState((prev) => {
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
            nextPlan = { ...incomingPlan, subtasks: mergedSubtasks };
            baseRuntime = Object.fromEntries(
              mergedSubtasks.map((t) => [
                t.id,
                prev.runtime[t.id] ?? { status: 'pending' as const },
              ]),
            );
          }

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
              shared_points: event.data.shared_points ?? existing?.shared_points ?? [],
              remaining_disagreements:
                event.data.remaining_disagreements ?? existing?.remaining_disagreements ?? [],
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

  const handleReset = useCallback(() => {
    setActiveTopic('');
    setState({
      status: 'idle',
      events: [],
      verdict: null,
      error: null,
      plan: null,
      runtime: {},
      positions: [],
      consensusHistory: [],
    });
  }, []);

  const isUserView = viewMode === 'user';

  return (
    <div className={isUserView ? 'min-h-screen user-shell' : 'min-h-screen bg-gray-50'}>
      <header
        className={
          isUserView
            ? 'sticky top-0 z-30 border-b border-cyan-300/10 bg-slate-950/70 px-6 py-3 backdrop-blur-xl'
            : 'bg-white border-b border-gray-200 px-6 py-3'
        }
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2.5">
            <div
              className={
                isUserView
                  ? 'w-8 h-8 rounded-lg bg-cyan-400/15 ring-1 ring-cyan-300/30 flex items-center justify-center shadow-[0_0_24px_rgba(34,211,238,0.25)]'
                  : 'w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm'
              }
            >
              <Network size={16} className={isUserView ? 'text-cyan-200' : 'text-white'} />
            </div>
            <div>
              <h1 className={`text-sm font-semibold leading-tight ${isUserView ? 'text-slate-100' : 'text-slate-800'}`}>
                A2A Debate Network
              </h1>
              <p className={`text-[10px] leading-tight ${isUserView ? 'text-cyan-100/55' : 'text-slate-500'}`}>
                Tres agentes deliberando hasta llegar a un consenso
              </p>
            </div>
          </div>

          <div
            className={
              isUserView
                ? 'ml-auto flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1 shadow-inner'
                : 'ml-auto flex items-center gap-1 bg-slate-100 rounded-full p-1 shadow-inner'
            }
          >
            <button
              onClick={() => setViewMode('user')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                viewMode === 'user'
                  ? isUserView
                    ? 'bg-cyan-300 text-slate-950 shadow-[0_0_18px_rgba(34,211,238,0.35)]'
                    : 'bg-white text-indigo-600 shadow-sm'
                  : isUserView
                    ? 'text-slate-300 hover:text-white'
                    : 'text-slate-500 hover:text-slate-700'
              }`}
              aria-pressed={viewMode === 'user'}
            >
              <Theater size={13} />
              Usuario
            </button>
            <button
              onClick={() => setViewMode('technical')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                viewMode === 'technical'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : isUserView
                    ? 'text-slate-300 hover:text-white'
                    : 'text-slate-500 hover:text-slate-700'
              }`}
              aria-pressed={viewMode === 'technical'}
            >
              <BarChart3 size={13} />
              Técnica
            </button>
          </div>

          {/* Badges de modelos solo en vista técnica */}
          {viewMode === 'technical' && (
            <div className="flex flex-wrap gap-2 text-[10px]">
              <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded font-medium" title={models.ae1}>
                AE1: {modelLabel(models.ae1)}
              </span>
              <span className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded font-medium" title={models.ae2}>
                AE2: {modelLabel(models.ae2)}
              </span>
              <span className="bg-fuchsia-100 text-fuchsia-700 px-2 py-1 rounded font-medium" title={models.ae3}>
                AE3: {modelLabel(models.ae3)}
              </span>
              <span className="bg-indigo-100 text-indigo-700 px-2 py-1 rounded font-medium" title={models.orchestrator}>
                Orq: {modelLabel(models.orchestrator)}
              </span>
              <span className="bg-amber-100 text-amber-700 px-2 py-1 rounded font-medium" title={models.normalizer}>
                Norm: {modelLabel(models.normalizer)}
              </span>
              <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded font-medium" title={models.feedback}>
                Fb: {modelLabel(models.feedback)}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className={isUserView ? 'mx-auto min-h-[calc(100vh-65px)] max-w-7xl p-4 lg:p-6' : 'max-w-7xl mx-auto p-4 h-[calc(100vh-72px)] overflow-y-auto'}>
        {viewMode === 'user' ? (
          <UserView
            status={state.status}
            events={state.events}
            runtime={state.runtime}
            positions={state.positions}
            consensusHistory={state.consensusHistory}
            verdict={state.verdict}
            error={state.error}
            topic={activeTopic}
            onSubmit={handleSubmit}
            onReset={handleReset}
          />
        ) : (
          <TechnicalView
            status={state.status}
            events={state.events}
            verdict={state.verdict}
            error={state.error}
            plan={state.plan}
            runtime={state.runtime}
            positions={state.positions}
            consensusHistory={state.consensusHistory}
            onSubmit={handleSubmit}
          />
        )}
      </main>
    </div>
  );
}

export default App;
