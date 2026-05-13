import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DebateAlreadyActiveError,
  type DebateSummary,
  type PersistedEvent,
  createDebate,
  getActiveDebate,
  getDebate,
  getEvents,
  listDebates,
  persistedToDebateEvent,
  streamDebate,
} from './api/debates';
import { AgentPositionsChart } from './components/AgentPositionsChart';
import { ConsensusGauge } from './components/ConsensusGauge';
import { DebateGraph } from './components/DebateGraph';
import { DebateHistorySidebar } from './components/DebateHistorySidebar';
import { DebateTimeline } from './components/DebateTimeline';
import { PromptInput } from './components/PromptInput';
import { VerdictDisplay } from './components/VerdictDisplay';
import type { ExtendedState } from './state/debateReducer';
import {
  INITIAL_STATE,
  reduceDebateEvent,
  replayEvents,
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

/** Read ?debate=<id> from the URL, or null. */
function readDebateIdFromURL(): string | null {
  const sp = new URLSearchParams(window.location.search);
  const id = sp.get('debate');
  return id && id.trim() ? id : null;
}

/** Update ?debate=<id> in the URL without reloading. */
function setDebateIdInURL(id: string | null): void {
  const url = new URL(window.location.href);
  if (id) {
    url.searchParams.set('debate', id);
  } else {
    url.searchParams.delete('debate');
  }
  window.history.replaceState({}, '', url.toString());
}

function App() {
  const [state, setState] = useState<ExtendedState>(INITIAL_STATE);

  // DIAGNOSTIC: every render bumps a counter so we can correlate with
  // [streamDebate] EVENT logs. If we see EVENTs but no [App render]
  // entries between them, React is dropping updates. If we see renders
  // but the user reports stale UI, the issue is downstream in a child.
  const renderCount = useRef(0);
  renderCount.current += 1;
  console.log(
    `[App render] #${renderCount.current} events=${state.events.length} status=${state.status}`,
  );
  const [debates, setDebates] = useState<DebateSummary[]>([]);
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conflictMsg, setConflictMsg] = useState<string | null>(null);

  const [showTimeline, setShowTimeline] = useState(true);
  const [models, setModels] = useState<ModelMap>({});
  const timelineRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  // ── Load models metadata ─────────────────────────────────────────────
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

  const refreshDebatesList = useCallback(async () => {
    try {
      const list = await listDebates();
      setDebates(list);
      const active = list.find((d) => d.status === 'running')?.id ?? null;
      setActiveId(active);
      return { list, active };
    } catch (e) {
      console.warn('Failed to refresh debates list:', e);
      return { list: [], active: null };
    }
  }, []);

  /**
   * Open a debate (live or terminal) and feed its events through the
   * reducer. Cancels any previously active stream.
   */
  const openDebate = useCallback(
    async (debateId: string, opts: { resetStateFirst?: boolean } = {}) => {
      // Cancel any running stream before starting a new one.
      streamAbortRef.current?.abort();

      if (opts.resetStateFirst !== false) {
        setState(runningState());
      }

      // 1. Replay everything we have so far in one shot — fast initial paint.
      let lastSeq = -1;
      try {
        const persisted = await getEvents(debateId, -1);
        const debateEvents = persisted.map(persistedToDebateEvent);
        if (persisted.length > 0) {
          lastSeq = persisted[persisted.length - 1].seq;
        }
        setState((prev) => replayEvents(debateEvents, prev));
      } catch (e) {
        console.warn('Replay failed:', e);
      }

      // 2. If still running, follow live events from the last seq we replayed.
      const meta = await getDebate(debateId);
      if (meta && meta.status === 'running') {
        const ctrl = new AbortController();
        streamAbortRef.current = ctrl;
        // Don't await — let it run in the background. The reducer keeps
        // state in sync; when the synthetic 'verdict'/'failed' arrives,
        // status flips and the orchestrator closes the stream.
        void streamDebate(
          debateId,
          lastSeq,
          (ev: PersistedEvent) => {
            setState((prev) => reduceDebateEvent(prev, persistedToDebateEvent(ev)));
            setTimeout(() => {
              if (timelineRef.current) {
                timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
              }
            }, 50);
            // When the terminal event arrives, refresh the sidebar so the
            // status badge flips and the "+ Nuevo debate" button unlocks.
            if (ev.stage === 'verdict' || ev.stage === 'failed') {
              void refreshDebatesList();
            }
          },
          (msg) => {
            console.warn('Stream error:', msg);
          },
          ctrl.signal,
        );
      }
    },
    [refreshDebatesList],
  );

  // ── Mount: decide what to show ───────────────────────────────────────
  // Priority:
  //   1. ?debate=<id> in URL → open that one (read-only if terminal)
  //   2. Otherwise the running debate, if any
  //   3. Else welcome screen (idle)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { list, active } = await refreshDebatesList();
      if (cancelled) return;

      const urlId = readDebateIdFromURL();
      const target =
        urlId && list.some((d) => d.id === urlId)
          ? urlId
          : active
            ? active
            : null;

      if (target) {
        setSelectedDebateId(target);
        setDebateIdInURL(target);
        await openDebate(target);
      } else {
        setDebateIdInURL(null);
      }
    })();
    return () => {
      cancelled = true;
      streamAbortRef.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── User actions ─────────────────────────────────────────────────────

  const handleSubmit = useCallback(
    async (prompt: string) => {
      setConflictMsg(null);
      try {
        const { debate_id } = await createDebate(prompt);
        await refreshDebatesList();
        setSelectedDebateId(debate_id);
        setDebateIdInURL(debate_id);
        await openDebate(debate_id);
      } catch (e) {
        if (e instanceof DebateAlreadyActiveError) {
          setConflictMsg(e.message);
          // Auto-open the active one so the user can watch it.
          await refreshDebatesList();
          if (e.active) {
            setSelectedDebateId(e.active.id);
            setDebateIdInURL(e.active.id);
            await openDebate(e.active.id);
          }
        } else {
          setState((prev) => ({
            ...prev,
            status: 'error',
            error: e instanceof Error ? e.message : String(e),
          }));
        }
      }
    },
    [openDebate, refreshDebatesList],
  );

  const handleSelectDebate = useCallback(
    async (id: string) => {
      setConflictMsg(null);
      setSelectedDebateId(id);
      setDebateIdInURL(id);
      await openDebate(id);
    },
    [openDebate],
  );

  const handleNewDebate = useCallback(() => {
    if (activeId !== null) {
      setConflictMsg('Hay un debate en curso — no se puede empezar uno nuevo');
      return;
    }
    streamAbortRef.current?.abort();
    setSelectedDebateId(null);
    setDebateIdInURL(null);
    setState(INITIAL_STATE);
  }, [activeId]);

  // Read-only mode: we're viewing a terminal debate that we didn't just
  // create — the prompt input is disabled.
  const isReadOnly =
    selectedDebateId !== null &&
    selectedDebateId !== activeId &&
    state.status !== 'running';
  const promptDisabled =
    state.status === 'running' || activeId !== null || isReadOnly;

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <DebateHistorySidebar
        debates={debates}
        activeId={activeId}
        selectedId={selectedDebateId}
        onSelect={handleSelectDebate}
        onNew={handleNewDebate}
      />

      {/* ── Main content ───────────────────────────────────────────── */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
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

        {/* 409 banner */}
        {conflictMsg && (
          <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 text-sm text-amber-800 flex items-center justify-between">
            <span>⚠️ {conflictMsg}</span>
            <button
              onClick={() => setConflictMsg(null)}
              className="text-amber-700 hover:text-amber-900 text-xs"
            >
              ✕
            </button>
          </div>
        )}

        {/* Read-only banner */}
        {isReadOnly && (
          <div className="bg-blue-50 border-b border-blue-200 px-6 py-2 text-sm text-blue-800">
            👁️ Viendo un debate pasado (solo lectura)
          </div>
        )}

        {/* Main content — two panels */}
        <main className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left panel: Input + Verdict */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-3">
                {isReadOnly ? 'Tema (solo lectura)' : 'Tema de debate'}
              </h2>
              <PromptInput onSubmit={handleSubmit} disabled={promptDisabled} />
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

          {/* Right panel: DAG graph on top, timeline log below */}
          <div className="lg:col-span-2 flex flex-col gap-4">
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

            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <ConsensusGauge history={state.consensusHistory} />
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <AgentPositionsChart samples={state.positions} />
            </div>

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
                <div ref={timelineRef} className="overflow-y-auto max-h-[50vh]">
                  <DebateTimeline events={state.events} plan={state.plan} />
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
