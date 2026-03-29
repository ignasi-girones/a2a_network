import { useCallback, useRef, useState } from 'react';
import { startDebateStream } from './api/sse';
import { DebateTimeline } from './components/DebateTimeline';
import { PromptInput } from './components/PromptInput';
import { VerdictDisplay } from './components/VerdictDisplay';
import type { DebateEvent, DebateState } from './types';

function App() {
  const [state, setState] = useState<DebateState>({
    status: 'idle',
    events: [],
    verdict: null,
    error: null,
  });

  const timelineRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  };

  const handleSubmit = useCallback(async (prompt: string) => {
    setState({ status: 'running', events: [], verdict: null, error: null });

    await startDebateStream(
      prompt,
      (event: DebateEvent) => {
        setState((prev) => ({
          ...prev,
          events: [...prev.events, event],
        }));
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

        {/* Right panel: Debate Timeline */}
        <div
          ref={timelineRef}
          className="lg:col-span-2 bg-white rounded-lg border border-gray-200 p-4 overflow-y-auto"
        >
          <h2 className="text-sm font-semibold text-gray-700 mb-3">
            Timeline del debate
            {state.events.length > 0 && (
              <span className="ml-2 text-[10px] font-normal text-gray-400">
                {state.events.length} eventos
              </span>
            )}
          </h2>
          <DebateTimeline events={state.events} />
        </div>
      </main>
    </div>
  );
}

export default App;
