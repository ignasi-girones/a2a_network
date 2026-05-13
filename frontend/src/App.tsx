/**
 * App.tsx — orchestrates persistence, streaming, and the two views.
 *
 * Layout:
 *
 *   ┌────────────┬─────────────────────────────────────────────┐
 *   │            │  header   [toggle Usuario / Técnica]        │
 *   │  Sidebar   ├─────────────────────────────────────────────┤
 *   │  (debates  │  banner (409 / read-only)                   │
 *   │   history) ├─────────────────────────────────────────────┤
 *   │            │  <UserView/>  or  <TechnicalView/>          │
 *   └────────────┴─────────────────────────────────────────────┘
 *
 * The reducer state (ExtendedState) is shared as-is between both views —
 * UserView and TechnicalView only differ in rendering, not in data shape.
 *
 * The persistence layer (createDebate / streamDebate / replayEvents) and
 * the URL ?debate=<id> sync are owned here and identical regardless of
 * which view is active.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BarChart3, Network, Theater } from 'lucide-react';
import {
  DebateAlreadyActiveError,
  type DebateSummary,
  type PersistedEvent,
  createDebate,
  getDebate,
  getEvents,
  listDebates,
  persistedToDebateEvent,
  streamDebate,
} from './api/debates';
import { DebateHistorySidebar } from './components/DebateHistorySidebar';
import type { ExtendedState } from './state/debateReducer';
import {
  INITIAL_STATE,
  reduceDebateEvent,
  replayEvents,
  runningState,
} from './state/debateReducer';
import { modelLabel } from './utils/modelLabel';
import { TechnicalView } from './views/TechnicalView';
import { UserView } from './views/UserView';

interface ModelMap {
  orchestrator?: string;
  normalizer?: string;
  ae1?: string;
  ae2?: string;
  ae3?: string;
  feedback?: string;
  embedding?: string;
}

type ViewMode = 'user' | 'technical';

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
  const [debates, setDebates] = useState<DebateSummary[]>([]);
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conflictMsg, setConflictMsg] = useState<string | null>(null);
  const [models, setModels] = useState<ModelMap>({});
  const [viewMode, setViewMode] = useState<ViewMode>('user');

  const streamAbortRef = useRef<AbortController | null>(null);

  // ── Models metadata ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/models');
        if (!res.ok) return;
        const data = (await res.json()) as ModelMap;
        if (!cancelled) setModels(data);
      } catch {
        // ignore — badges just show "—"
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
   * Open a debate (live or terminal) and feed events through the reducer.
   * Cancels any previously active stream.
   */
  const openDebate = useCallback(
    async (debateId: string, opts: { resetStateFirst?: boolean } = {}) => {
      streamAbortRef.current?.abort();

      if (opts.resetStateFirst !== false) {
        setState(runningState());
      }

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

      const meta = await getDebate(debateId);
      if (meta && meta.status === 'running') {
        const ctrl = new AbortController();
        streamAbortRef.current = ctrl;
        void streamDebate(
          debateId,
          lastSeq,
          (ev: PersistedEvent) => {
            setState((prev) =>
              reduceDebateEvent(prev, persistedToDebateEvent(ev)),
            );
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

  // ── Derived values ──────────────────────────────────────────────────
  const isReadOnly =
    selectedDebateId !== null &&
    selectedDebateId !== activeId &&
    state.status !== 'running';
  const promptDisabled =
    state.status === 'running' || activeId !== null || isReadOnly;

  // Topic shown in UserView's hero / chip / final scene. We prefer the
  // selected debate's persisted prompt (always correct after F5/replay).
  const topic = useMemo(() => {
    if (selectedDebateId) {
      const found = debates.find((d) => d.id === selectedDebateId);
      if (found) return found.prompt;
    }
    return '';
  }, [debates, selectedDebateId]);

  const isUserView = viewMode === 'user';

  return (
    <div
      className={
        isUserView
          ? 'min-h-screen user-shell flex'
          : 'min-h-screen bg-gray-50 flex'
      }
    >
      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <DebateHistorySidebar
        debates={debates}
        activeId={activeId}
        selectedId={selectedDebateId}
        onSelect={handleSelectDebate}
        onNew={handleNewDebate}
      />

      {/* ── Main column ─────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Header */}
        <header
          className={
            isUserView
              ? 'sticky top-0 z-30 border-b border-cyan-300/10 bg-slate-950/70 px-6 py-3 backdrop-blur-xl'
              : 'bg-white border-b border-gray-200 px-6 py-3'
          }
        >
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-2.5 min-w-0">
              <div
                className={
                  isUserView
                    ? 'w-8 h-8 rounded-lg bg-cyan-400/15 ring-1 ring-cyan-300/30 flex items-center justify-center shadow-[0_0_24px_rgba(34,211,238,0.25)]'
                    : 'w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm'
                }
              >
                <Network
                  size={16}
                  className={isUserView ? 'text-cyan-200' : 'text-white'}
                />
              </div>
              <div className="min-w-0">
                <h1
                  className={`text-sm font-semibold leading-tight ${
                    isUserView ? 'text-slate-100' : 'text-slate-800'
                  }`}
                >
                  A2A Debate Network
                </h1>
                <p
                  className={`text-[10px] leading-tight ${
                    isUserView ? 'text-cyan-100/55' : 'text-slate-500'
                  }`}
                >
                  Tres agentes deliberando hasta llegar a un consenso
                </p>
              </div>
            </div>

            {/* View toggle */}
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

            {/* Model badges — only in technical view */}
            {viewMode === 'technical' && (
              <div className="flex flex-wrap gap-2 text-[10px] basis-full lg:basis-auto">
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
                  Orq: {modelLabel(models.orchestrator)}
                </span>
                <span
                  className="bg-amber-100 text-amber-700 px-2 py-1 rounded font-medium"
                  title={models.normalizer}
                >
                  Norm: {modelLabel(models.normalizer)}
                </span>
                <span
                  className="bg-gray-100 text-gray-700 px-2 py-1 rounded font-medium"
                  title={models.feedback}
                >
                  Fb: {modelLabel(models.feedback)}
                </span>
                <span
                  className="bg-cyan-100 text-cyan-700 px-2 py-1 rounded font-medium"
                  title={models.embedding}
                >
                  Emb: {modelLabel(models.embedding)}
                </span>
              </div>
            )}
          </div>
        </header>

        {/* 409 banner */}
        {conflictMsg && (
          <div
            className={
              isUserView
                ? 'border-b border-amber-400/30 bg-amber-500/10 px-6 py-2 text-sm text-amber-100 flex items-center justify-between'
                : 'bg-amber-50 border-b border-amber-200 px-6 py-2 text-sm text-amber-800 flex items-center justify-between'
            }
          >
            <span>⚠️ {conflictMsg}</span>
            <button
              onClick={() => setConflictMsg(null)}
              className={
                isUserView
                  ? 'text-amber-200 hover:text-amber-50 text-xs'
                  : 'text-amber-700 hover:text-amber-900 text-xs'
              }
            >
              ✕
            </button>
          </div>
        )}

        {/* Read-only banner */}
        {isReadOnly && (
          <div
            className={
              isUserView
                ? 'border-b border-cyan-300/20 bg-cyan-500/10 px-6 py-2 text-sm text-cyan-100'
                : 'bg-blue-50 border-b border-blue-200 px-6 py-2 text-sm text-blue-800'
            }
          >
            👁️ Viendo un debate pasado (solo lectura)
          </div>
        )}

        {/* Main view area */}
        <main
          className={
            isUserView
              ? 'flex-1 mx-auto w-full max-w-7xl p-4 lg:p-6'
              : 'flex-1 max-w-7xl mx-auto w-full p-4 overflow-y-auto'
          }
        >
          {isUserView ? (
            <UserView
              status={state.status}
              events={state.events}
              runtime={state.runtime}
              positions={state.positions}
              consensusHistory={state.consensusHistory}
              verdict={state.verdict}
              error={state.error}
              topic={topic}
              disabled={promptDisabled}
              onSubmit={handleSubmit}
              onReset={handleNewDebate}
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
              disabled={promptDisabled}
              onSubmit={handleSubmit}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
