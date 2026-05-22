/**
 * UserView — vista del usuario final con la mesa redonda.
 *
 * Sin información técnica (no se muestran modelos LLM, ni log_odds,
 * ni nada de eso). Solo: input, mesa con personajes, veredicto.
 */

import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, Bot, CheckCircle2, Loader2, MessageCircle, RotateCcw, Sparkles } from 'lucide-react';
import type {
  AgentPositionsSample,
  ConsensusSnapshot,
  DebateEvent,
  SubtaskRuntime,
} from '../types';
import { PromptInput } from '../components/PromptInput';
import { Markdown } from '../components/Markdown';
import { RoundTable } from '../components/round-table/RoundTable';
import { RoundIndicator } from '../components/round-table/RoundIndicator';
import { ConsensusBurst } from '../components/round-table/ConsensusBurst';

interface Props {
  status: 'idle' | 'running' | 'completed' | 'error';
  events: DebateEvent[];
  runtime: Record<string, SubtaskRuntime>;
  positions: AgentPositionsSample[];
  consensusHistory: ConsensusSnapshot[];
  verdict: string | null;
  error: string | null;
  topic: string;
  // True when the prompt input must be disabled because:
  //   - a debate is running on the server, OR
  //   - the user is viewing a terminal debate in read-only mode.
  // Parent (App.tsx) owns this — the view just renders it.
  disabled: boolean;
  onSubmit: (prompt: string, files?: File[]) => void;
  onReset: () => void;
}

function useTransientBurst(status: Props['status']): boolean {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (status === 'completed') {
      setVisible(true);
      const timer = setTimeout(() => setVisible(false), 5000);
      return () => clearTimeout(timer);
    } else {
      setVisible(false);
    }
  }, [status]);
  return visible;
}

function useStatusMessage(status: Props['status'], events: DebateEvent[]): string {
  return useMemo(() => {
    if (status === 'idle') return '';
    if (status === 'error') return 'Error en el debate.';
    if (status === 'completed') return 'Debate completado.';
    if (events.length === 0) return 'Iniciando debate…';

    const last = events[events.length - 1];
    const stage = last.stage;

    const stageMessages: Record<string, string> = {
      plan_ready: 'Preparando el debate…',
      subtask_dispatch: 'Un agente está pensando…',
      subtask_done: 'Un agente acaba de responder',
      subtask_failed: 'Un agente ha fallado',
      agent_positions: `Evaluando posiciones (ronda ${last.data?.round ?? '?'})`,
      consensus_check: 'Comprobando si hay consenso…',
      tool_use: `Consultando información…`,
      embedding: 'Procesando…',
      embedding_done: 'Procesado',
      synthesize: 'Generando veredicto final…',
      complete: 'Debate completado',
      plan_complete: 'Plan completado',
    };

    return stageMessages[stage] ?? last.message ?? 'Deliberando…';
  }, [status, events]);
}

function useCurrentRound(
  positions: AgentPositionsSample[],
  consensusHistory: ConsensusSnapshot[],
  events: DebateEvent[],
): { current: number; max: number } {
  return useMemo(() => {
    let current = 0;
    let max = 3;

    if (positions.length > 0) {
      current = Math.max(...positions.map((p) => p.round));
    }
    if (consensusHistory.length > 0) {
      current = Math.max(current, ...consensusHistory.map((s) => s.round));
    }

    for (const event of events) {
      if (event.data?.max_rounds) {
        max = event.data.max_rounds;
        break;
      }
    }

    return { current, max };
  }, [positions, consensusHistory, events]);
}

export function UserView({
  status,
  events,
  runtime,
  positions,
  consensusHistory,
  verdict,
  error,
  topic,
  disabled,
  onSubmit,
  onReset,
}: Props) {
  const burstVisible = useTransientBurst(status);
  const statusMessage = useStatusMessage(status, events);
  const { current: currentRound, max: maxRounds } = useCurrentRound(
    positions,
    consensusHistory,
    events,
  );

  const isRunning = status === 'running';
  const isCompleted = status === 'completed';
  const hasStarted = status !== 'idle';
  const showHero = !hasStarted;
  const showFinalConsensus = isCompleted && !!verdict;
  const finalScore =
    consensusHistory.length > 0
      ? consensusHistory[consensusHistory.length - 1].agreement_score
      : null;

  const roundsUsed = positions.length;

  return (
    <div className="relative overflow-hidden rounded-[28px] border border-cyan-200/10 bg-slate-950/80 shadow-2xl shadow-slate-950/40">
      <SceneBackdrop />

      <AnimatePresence mode="wait">
        {showHero ? (
          <motion.section
            key="hero"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -18 }}
            transition={{ duration: 0.35 }}
            className="relative grid min-h-[calc(100vh-130px)] grid-cols-1 items-center gap-8 p-5 md:p-8 lg:grid-cols-[0.9fr_1.1fr]"
          >
            <Moderator mode="hero" statusText="Lista para abrir la mesa" />
            <div className="max-w-2xl">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-200/20 bg-cyan-200/10 px-3 py-1 text-xs font-semibold text-cyan-100">
                <Sparkles size={14} />
                Debate guiado por agentes
              </div>
              <h2 className="max-w-xl text-3xl font-black leading-tight text-white md:text-5xl">
                ¿Sobre qué tema quieres que debatan los agentes?
              </h2>
              <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300">
                Propón una decisión, dilema o estrategia. La moderadora abrirá la ronda y tres agentes intentarán llegar a un consenso claro.
              </p>
              <div className="mt-7 rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-cyan-950/30 backdrop-blur">
                <PromptInput onSubmit={onSubmit} disabled={disabled} variant="hero" acceptFiles />
              </div>
            </div>
          </motion.section>
        ) : showFinalConsensus ? (
          <FinalConsensusScene
            verdict={verdict}
            topic={topic}
            onReset={onReset}
          />
        ) : (
          <motion.section
            key="debate"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="relative grid min-h-[calc(100vh-130px)] grid-cols-1 gap-4 p-4 lg:grid-cols-[300px_1fr]"
          >
            <aside className="space-y-4">
              <Moderator
                mode="side"
                statusText={statusMessage}
              />

              <TopicChip topic={topic} />

              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 backdrop-blur">
                <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase text-cyan-100/70">
                  <MessageCircle size={14} />
                  Nuevo tema
                </div>
                <PromptInput
                  onSubmit={onSubmit}
                  disabled={disabled}
                  variant="compactDark"
                  acceptFiles
                />
              </div>

              <AnimatePresence>
                {isRunning && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center gap-2 rounded-2xl border border-cyan-200/15 bg-cyan-200/10 p-3 text-xs text-cyan-50"
                  >
                    <Loader2 size={15} className="animate-spin text-cyan-200" />
                    <span>{statusMessage}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {status === 'error' && error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex items-start gap-2 rounded-2xl border border-red-300/30 bg-red-500/10 p-3"
                  >
                    <AlertCircle size={16} className="mt-0.5 flex-shrink-0 text-red-300" />
                    <div className="text-xs text-red-100">
                      <div className="mb-0.5 font-semibold">Error</div>
                      <div className="opacity-90">{error}</div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

            </aside>

            <div className="relative overflow-hidden rounded-[24px] border border-cyan-200/10 bg-slate-900/55 shadow-2xl shadow-black/30">
              <div className="absolute left-4 right-4 top-4 z-10 flex items-center justify-between gap-3">
                <div className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-cyan-100/70 backdrop-blur">
                  Mesa de debate
                </div>
                <RoundIndicator
                  currentRound={currentRound}
                  maxRounds={maxRounds}
                  active={isRunning}
                />
                {isCompleted && (
                  <div className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1.5 text-xs font-semibold text-emerald-100">
                    <CheckCircle2 size={14} />
                    Consenso
                  </div>
                )}
              </div>

              <div className="relative pt-12">
                <RoundTable
                  runtime={runtime}
                  positions={positions}
                  consensusHistory={consensusHistory}
                  running={isRunning}
                  hasStarted={hasStarted}
                />

                <ConsensusBurst
                  visible={burstVisible}
                  finalScore={finalScore}
                  roundsUsed={roundsUsed}
                />
              </div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}

function SceneBackdrop() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(34,211,238,0.18),transparent_30%),radial-gradient(circle_at_80%_10%,rgba(168,85,247,0.13),transparent_28%),linear-gradient(135deg,rgba(15,23,42,0.2),rgba(2,6,23,0.9))]" />
      <div className="absolute inset-0 opacity-[0.16] [background-image:linear-gradient(rgba(103,232,249,0.28)_1px,transparent_1px),linear-gradient(90deg,rgba(103,232,249,0.28)_1px,transparent_1px)] [background-size:42px_42px]" />
      <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-slate-950 to-transparent" />
    </div>
  );
}

function Moderator({ mode, statusText }: { mode: 'hero' | 'side' | 'final'; statusText: string }) {
  const isHero = mode === 'hero';
  const isFinal = mode === 'final';
  const isPrimary = isHero || isFinal;

  return (
    <motion.div
      layout
      className={
        isPrimary
          ? 'relative mx-auto flex w-full max-w-md flex-col items-center'
          : 'relative flex items-center gap-3 rounded-[24px] border border-cyan-200/10 bg-white/[0.04] p-3 backdrop-blur'
      }
    >
      <motion.div
        animate={isPrimary ? { y: [0, -8, 0] } : { y: [0, -3, 0] }}
        transition={{ duration: isPrimary ? 4 : 3, repeat: Infinity, ease: 'easeInOut' }}
        className={isPrimary ? 'relative h-80 w-80 flex-none' : 'relative h-24 w-24 flex-none overflow-hidden'}
      >
        <div className={isPrimary ? 'absolute inset-0' : 'absolute left-1/2 top-0 h-80 w-80 -translate-x-1/2 origin-top scale-[0.3]'}>
          <div className="absolute inset-x-10 bottom-5 h-12 rounded-full bg-cyan-300/20 blur-xl" />
          <div className="absolute left-1/2 top-4 h-24 w-24 -translate-x-1/2 rounded-full border border-cyan-100/30 bg-slate-800 shadow-[0_0_50px_rgba(34,211,238,0.25)]" />
          <div className="absolute left-1/2 top-10 h-10 w-14 -translate-x-1/2 rounded-full bg-cyan-100/90">
            <span className="absolute left-3 top-3 h-2 w-2 rounded-full bg-slate-950" />
            <span className="absolute right-3 top-3 h-2 w-2 rounded-full bg-slate-950" />
            <span className="absolute bottom-2 left-1/2 h-1 w-7 -translate-x-1/2 rounded-full bg-cyan-500" />
          </div>
          <div className="absolute left-1/2 top-28 h-28 w-40 -translate-x-1/2 rounded-t-[56px] rounded-b-3xl border border-cyan-100/20 bg-gradient-to-b from-cyan-300/30 to-fuchsia-400/15" />
          <div className="absolute left-[72px] top-32 h-16 w-8 -rotate-12 rounded-full bg-cyan-200/30" />
          <div className="absolute right-[72px] top-32 h-16 w-8 rotate-12 rounded-full bg-cyan-200/30" />
          <Bot className="absolute left-1/2 top-[142px] -translate-x-1/2 text-cyan-100/80" size={34} />
        </div>
      </motion.div>
      <div className={isPrimary ? 'text-center' : 'min-w-0 flex-1'}>
        <div className="text-xs font-semibold uppercase tracking-wider text-cyan-100/60">
          Moderadora
        </div>
        <div className={isPrimary ? 'mt-2 text-xl font-black text-white' : 'mt-1 text-base font-bold text-white'}>
          {isFinal ? 'Ya tenemos una conclusión' : 'Guío la conversación'}
        </div>
        <div className="mt-2 text-sm leading-5 text-slate-300">{statusText}</div>
      </div>
    </motion.div>
  );
}

function FinalConsensusScene({
  verdict,
  topic,
  onReset,
}: {
  verdict: string;
  topic: string;
  onReset: () => void;
}) {
  const cleanVerdict = cleanFinalVerdict(verdict);

  return (
    <motion.section
      key="final-consensus"
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -16, scale: 0.98 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
      className="relative grid min-h-[calc(100vh-130px)] grid-cols-1 items-center gap-6 p-5 md:p-8 lg:grid-cols-[0.82fr_1.18fr]"
    >
      <div className="relative">
        <div className="absolute left-1/2 top-1/2 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-200/20 bg-cyan-300/10 blur-2xl" />
        <Moderator mode="final" statusText="El debate se ha cerrado con una postura compartida." />
      </div>

      <div className="relative mx-auto w-full max-w-3xl">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12, duration: 0.4 }}
          className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1.5 text-xs font-semibold text-emerald-100"
        >
          <CheckCircle2 size={14} />
          Consenso alcanzado
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18, duration: 0.4 }}
          className="text-3xl font-black leading-tight text-white md:text-5xl"
        >
          El consenso final ha sido…
        </motion.h2>

        {topic && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.24, duration: 0.4 }}
            className="mt-4 max-w-2xl text-sm font-semibold leading-6 text-cyan-100/75"
          >
            Tema: {topic}
          </motion.div>
        )}

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.45 }}
          className="mt-7 rounded-[28px] border border-amber-200/30 bg-slate-900/80 p-5 shadow-[0_0_70px_rgba(34,211,238,0.16)] backdrop-blur md:p-7"
        >
          <div className="max-h-[42vh] overflow-y-auto pr-1 text-base leading-7 text-slate-50 md:text-lg md:leading-8">
            <Markdown className="prose-headings:text-white prose-p:text-slate-50 prose-li:text-slate-100 prose-strong:text-white prose-code:bg-slate-800 prose-code:text-cyan-100 prose-blockquote:text-slate-200 prose-blockquote:border-cyan-200/40">
              {cleanVerdict}
            </Markdown>
          </div>
        </motion.div>

        <motion.button
          type="button"
          onClick={onReset}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.38, duration: 0.35 }}
          className="mt-6 inline-flex items-center gap-2 rounded-full border border-cyan-200/30 bg-cyan-300 px-5 py-3 text-sm font-black text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.35)] transition hover:bg-cyan-200 focus:outline-none focus:ring-2 focus:ring-cyan-100 focus:ring-offset-2 focus:ring-offset-slate-950"
        >
          <RotateCcw size={16} />
          Nuevo debate
        </motion.button>
      </div>
    </motion.section>
  );
}

function cleanFinalVerdict(raw: string): string {
  const extracted = extractVerdictText(raw);
  const withoutCodeFence = extracted
    .replace(/^```(?:json|markdown|md|text)?\s*/i, '')
    .replace(/\s*```$/i, '');

  const lines = withoutCodeFence
    .split(/\r?\n/)
    .map((line) =>
      line
        .replace(/^\s*(?:verdict|final_verdict|consensus|consenso|answer|respuesta|summary|resumen)\s*[:=-]\s*/i, '')
        .trimEnd(),
    )
    .filter((line) => {
      const normalized = line.trim().toLowerCase();
      if (!normalized) return true;
      if (/^(log_odds|payload|query|trace|debug|metadata|raw|json|data|event|stage|worker_id|subtask_id)\b/.test(normalized)) {
        return false;
      }
      if (/^\{|\}$|^\[|\]$/.test(normalized)) return false;
      if (/^[",]*(log_odds|payload|query|trace|debug|metadata|raw|event|stage|worker_id|subtask_id)["']?\s*:/.test(normalized)) {
        return false;
      }
      return true;
    });

  const cleaned = lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  return cleaned || 'No se ha podido mostrar una conclusión legible.';
}

function extractVerdictText(raw: string): string {
  const trimmed = raw.trim();

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    const text = findReadableVerdict(parsed);
    if (text) return text;
  } catch {
    // El verdict normalmente llega como texto; si no es JSON, seguimos con limpieza visual.
  }

  return trimmed;
}

function findReadableVerdict(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (!value || typeof value !== 'object') return null;

  if (Array.isArray(value)) {
    const parts = value
      .map(findReadableVerdict)
      .filter((part): part is string => Boolean(part));
    return parts.length > 0 ? parts.join('\n\n') : null;
  }

  const record = value as Record<string, unknown>;
  const preferredKeys = ['verdict', 'final_verdict', 'consensus', 'consenso', 'summary', 'resumen', 'answer', 'respuesta', 'text'];
  for (const key of preferredKeys) {
    const found = findReadableVerdict(record[key]);
    if (found) return found;
  }

  return null;
}

function TopicChip({ topic }: { topic: string }) {
  if (!topic) return null;

  return (
    <div className="rounded-2xl border border-fuchsia-300/20 bg-fuchsia-300/10 p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-fuchsia-100/70">
        Tema activo
      </div>
      <div className="text-sm font-semibold leading-5 text-white">{topic}</div>
    </div>
  );
}
