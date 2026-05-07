import { useCallback, useEffect, useRef, useState } from 'react';
import { startDebateStream } from './api/sse';
import { AgentPositionsChart } from './components/AgentPositionsChart';
import { ConsensusGauge } from './components/ConsensusGauge';
import { DebateGraph } from './components/DebateGraph';
import { DebateTimeline } from './components/DebateTimeline';
import { PromptInput } from './components/PromptInput';
import { VerdictDisplay } from './components/VerdictDisplay';
import type { DebateEvent } from './types';
import type { ExtendedState } from './state/debateReducer';
import {
  INITIAL_STATE,
  reduceDebateEvent,
  runningState,
} from './state/debateReducer';
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

function App() {
  const [state, setState] = useState<ExtendedState>(INITIAL_STATE);

  const [showTimeline, setShowTimeline] = useState(true);
  const [models, setModels] = useState<ModelMap>({});
  const timelineRef = useRef<HTMLDivElement>(null);

  // Load the per-agent LLM model the orchestrator was started with. Falls
  // back to empty silently if the endpoint isn't reachable yet.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/models');
        if (!res.ok) return;
        const data = (await res.json()) as ModelMap;
        if (!cancelled) setModels(data);
      } catch {
        // Quietly ignore — the badges will just show "—".
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const scrollToBottom = () => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  };

  const handleSubmit = useCallback(async (prompt: string) => {
    setState(runningState());

    await startDebateStream(
      prompt,
      (event: DebateEvent) => {
        setState((prev) => reduceDebateEvent(prev, event));
        setTimeout(scrollToBottom, 50);
      },
      (verdict: string) => {
        setState((prev) => ({
          ...prev,
          status: 'completed',
          verdict,
        }));
      },
      (error: string) => {
        setState((prev) => ({
          ...prev,
          status: 'error',
          error,
        }));
      },
    );
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">
              A2A Debate Network
            </h1>
            <p className="text-xs text-gray-500">
              Protocolo A2A v1.0.0 &mdash; Red de agentes con debate estructurado
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[10px]">
            <span
              className="bg-blue-100 text-blue-700 px-2 py-1 rounded font-medium"
              title={models.ae1}
            >
              AE1: {modelLabel(models.ae1)}
            </span>
            <span
              className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded font-medium"
              title={models.ae2}
            >
              AE2: {modelLabel(models.ae2)}
            </span>
            <span
              className="bg-fuchsia-100 text-fuchsia-700 px-2 py-1 rounded font-medium"
              title={models.ae3}
            >
              AE3: {modelLabel(models.ae3)}
            </span>
            <span
              className="bg-indigo-100 text-indigo-700 px-2 py-1 rounded font-medium"
              title={models.orchestrator}
            >
              Orquestador: {modelLabel(models.orchestrator)}
            </span>
            <span
              className="bg-amber-100 text-amber-700 px-2 py-1 rounded font-medium"
              title={models.normalizer}
            >
              Normalizador: {modelLabel(models.normalizer)}
            </span>
            <span
              className="bg-gray-100 text-gray-700 px-2 py-1 rounded font-medium"
              title={models.feedback}
            >
              Feedback: {modelLabel(models.feedback)}
            </span>
            <span
              className="bg-cyan-100 text-cyan-700 px-2 py-1 rounded font-medium"
              title={models.embedding}
            >
              Embeddings: {modelLabel(models.embedding)}
            </span>
          </div>
        </div>
      </header>

      {/* Main content — two panels */}
      <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 h-[calc(100vh-64px)]">
        {/* Left panel: Input + Verdict */}
        <div className="lg:col-span-1 space-y-4 overflow-y-auto">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h2 className="text-sm font-semibold text-gray-700 mb-3">Tema de debate</h2>
            <PromptInput
              onSubmit={handleSubmit}
              disabled={state.status === 'running'}
            />
          </div>
          <VerdictDisplay
            verdict={state.verdict}
            error={state.error}
            status={state.status}
            lastEvent={state.events.length > 0 ? state.events[state.events.length - 1] : null}
          />
        </div>

        {/* Right panel: DAG graph on top, timeline log below */}
        <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto">
          {/* Graph card */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-gray-700">
                Grafo del plan
                {state.plan && (
                  <span className="ml-2 text-[10px] font-normal text-gray-400">
                    {state.plan.subtasks.length} subtareas
                  </span>
                )}
              </h2>
              <span className="text-[10px] text-gray-400">
                Haz clic en un nodo para ver su salida
              </span>
            </div>
            <DebateGraph plan={state.plan} runtime={state.runtime} />
          </div>

          {/* Consensus gauge card */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <ConsensusGauge history={state.consensusHistory} />
          </div>

          {/* Agent positions chart card */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <AgentPositionsChart samples={state.positions} />
          </div>

          {/* Timeline card (collapsible) */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <button
              onClick={() => setShowTimeline((v) => !v)}
              className="w-full flex items-center justify-between text-sm font-semibold text-gray-700 mb-2"
            >
              <span>
                Registro de eventos
                {state.events.length > 0 && (
                  <span className="ml-2 text-[10px] font-normal text-gray-400">
                    {state.events.length} eventos
                  </span>
                )}
              </span>
              <span className="text-gray-400 text-xs">
                {showTimeline ? '▾ ocultar' : '▸ mostrar'}
              </span>
            </button>
            {showTimeline && (
              <div
                ref={timelineRef}
                className="overflow-y-auto max-h-[50vh]"
              >
                <DebateTimeline events={state.events} plan={state.plan} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
