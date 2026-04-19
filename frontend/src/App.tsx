import { useCallback, useRef, useState } from 'react';
import { startDebateStream } from './api/sse';
import { DebateGraph } from './components/DebateGraph';
import { DebateTimeline } from './components/DebateTimeline';
import { PromptInput } from './components/PromptInput';
import { VerdictDisplay } from './components/VerdictDisplay';
import type {
  DebateEvent,
  DebateState,
  SubtaskRuntime,
  TaskPlan,
} from './types';

interface ExtendedState extends DebateState {
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
}

/**
 * Fold a single streaming event into the runtime map.
 *
 * - subtask_dispatch → mark running, record worker_id
 * - subtask_done     → mark done, store full text output
 * - subtask_failed   → mark failed, store error
 *
 * Anything else is a no-op for the runtime map (it goes to the timeline log).
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

function App() {
  const [state, setState] = useState<ExtendedState>({
    status: 'idle',
    events: [],
    verdict: null,
    error: null,
    plan: null,
    runtime: {},
  });

  const [showTimeline, setShowTimeline] = useState(true);
  const timelineRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  };

  const handleSubmit = useCallback(async (prompt: string) => {
    setState({
      status: 'running',
      events: [],
      verdict: null,
      error: null,
      plan: null,
      runtime: {},
    });

    await startDebateStream(
      prompt,
      (event: DebateEvent) => {
        setState((prev) => {
          // Plan arrives once, in the plan_ready event
          const nextPlan =
            event.stage === 'plan_ready' && event.data?.plan
              ? (event.data.plan as TaskPlan)
              : prev.plan;

          // If the plan just arrived, initialise every node as pending
          const baseRuntime =
            nextPlan && nextPlan !== prev.plan
              ? Object.fromEntries(
                  nextPlan.subtasks.map((t) => [
                    t.id,
                    prev.runtime[t.id] ?? { status: 'pending' as const },
                  ]),
                )
              : prev.runtime;

          return {
            ...prev,
            events: [...prev.events, event],
            plan: nextPlan,
            runtime: applyEventToRuntime(baseRuntime, event),
          };
        });
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
          <div className="flex gap-2 text-[10px]">
            <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded font-medium">AE1: Mistral</span>
            <span className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded font-medium">AE2: Cerebras</span>
            <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded font-medium">Orch: Groq</span>
            <span className="bg-amber-100 text-amber-700 px-2 py-1 rounded font-medium">Normalizer: Gemini</span>
            <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded font-medium">Feedback: Ollama</span>
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

          {/* Timeline card (collapsible) */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <button
              onClick={() => setShowTimeline((v) => !v)}
              className="w-full flex items-center justify-between text-sm font-semibold text-gray-700 mb-2"
            >
              <span>
                Log de eventos
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
