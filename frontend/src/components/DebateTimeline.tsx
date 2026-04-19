import type { DebateEvent, TaskPlan } from '../types';
import { Markdown } from './Markdown';

interface Props {
  events: DebateEvent[];
  plan?: TaskPlan | null;
}

// ── Per-event visual classification ─────────────────────────────────────────
function getEventStyle(event: DebateEvent) {
  const stage = event.stage;
  const agent = event.data?.agent;
  const workerId = event.data?.worker_id;
  const perspective = event.data?.perspective;

  // Agentic events carry worker_id (orchestrator → worker dispatch / done)
  if (workerId === 'ae1' || agent === 'ae1') {
    return {
      align: 'self-start',
      bg: 'bg-blue-50 border-blue-200',
      label: perspective ? `AE1 · ${perspective}` : 'AE1',
      labelColor: 'bg-blue-600',
    };
  }
  if (workerId === 'ae2' || agent === 'ae2') {
    return {
      align: 'self-end',
      bg: 'bg-emerald-50 border-emerald-200',
      label: perspective ? `AE2 · ${perspective}` : 'AE2',
      labelColor: 'bg-emerald-600',
    };
  }
  if (workerId === 'normalizer') {
    return {
      align: 'self-center',
      bg: 'bg-amber-50 border-amber-200',
      label: 'Normalizador',
      labelColor: 'bg-amber-600',
    };
  }
  if (workerId === 'feedback') {
    return {
      align: 'self-center',
      bg: 'bg-purple-50 border-purple-200',
      label: 'Feedback',
      labelColor: 'bg-purple-600',
    };
  }

  if (stage === 'tool_use') {
    return {
      align: 'self-center',
      bg: 'bg-violet-50 border-violet-200',
      label: 'Tool',
      labelColor: 'bg-violet-500',
    };
  }

  if (
    stage === 'consensus' ||
    stage === 'complete' ||
    stage === 'plan_complete' ||
    stage === 'synthesize'
  ) {
    return {
      align: 'self-center',
      bg: 'bg-amber-50 border-amber-200',
      label: 'Sistema',
      labelColor: 'bg-amber-600',
    };
  }

  return {
    align: 'self-center',
    bg: 'bg-gray-50 border-gray-200',
    label: 'Orquestador',
    labelColor: 'bg-gray-600',
  };
}

// ── Turn a cryptic "Subtarea t2 completada" into a human label ──────────────
function humanizeMessage(event: DebateEvent, plan: TaskPlan | null | undefined): string {
  const raw = event.message;
  const id = event.data?.subtask_id;
  if (!id || !plan) return raw;
  const task = plan.subtasks.find((t) => t.id === id);
  if (!task) return raw;
  const worker = event.data?.worker_id;
  const skill = task.required_skill;
  const persp = task.perspective ? `·${task.perspective}` : '';
  const tag = worker || skill;

  if (event.stage === 'subtask_dispatch') {
    return `→ ${tag}${persp}: ${truncate(task.description, 60)}`;
  }
  if (event.stage === 'subtask_done') {
    return `✓ ${tag}${persp} completado`;
  }
  if (event.stage === 'subtask_failed') {
    return `✗ ${tag}${persp} falló`;
  }
  return raw;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1).trimEnd() + '…';
}

function isDetailEvent(event: DebateEvent): boolean {
  return !!(event.data?.text);
}

export function DebateTimeline({ events, plan }: Props) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-24 text-gray-400 text-xs">
        <p>Los eventos del debate aparecerán aquí en tiempo real...</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-1">
      {events.map((event, i) => {
        const style = getEventStyle(event);
        const hasDetail = isDetailEvent(event);
        const label = humanizeMessage(event, plan);

        return (
          <div key={i} className={`flex flex-col ${style.align} max-w-[85%]`}>
            <div className={`rounded-lg border p-3 ${style.bg} text-sm`}>
              {/* Header badge */}
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`text-[10px] font-bold text-white px-1.5 py-0.5 rounded ${style.labelColor}`}
                >
                  {style.label}
                </span>
                {event.data?.round && (
                  <span className="text-[10px] text-gray-500">
                    Ronda {event.data.round}
                  </span>
                )}
                {event.data?.required_skill && !event.data?.worker_id && (
                  <span className="text-[10px] text-gray-500 font-mono">
                    {event.data.required_skill}
                  </span>
                )}
              </div>

              {/* Message */}
              <p className="text-gray-700 text-xs font-medium">{label}</p>

              {/* Detail text (full subtask output) */}
              {hasDetail && (
                <details className="mt-2">
                  <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-700">
                    Ver salida completa
                  </summary>
                  <div className="mt-1 text-gray-600">
                    <Markdown>{event.data!.text!}</Markdown>
                  </div>
                </details>
              )}

              {/* Tool use info */}
              {event.stage === 'tool_use' && event.data?.tool && (
                <div className="mt-2 flex flex-col gap-1">
                  <div className="flex items-center gap-1.5 text-[10px] text-violet-700">
                    <span className="font-mono bg-violet-100 px-1.5 py-0.5 rounded">
                      {event.data.tool}()
                    </span>
                  </div>
                  {event.data.query && (
                    <p className="text-[10px] text-gray-500 font-mono truncate">
                      &ldquo;{event.data.query}&rdquo;
                    </p>
                  )}
                </div>
              )}

              {/* Error */}
              {event.data?.error && (
                <p className="mt-1 text-[10px] text-red-600 font-mono">
                  {event.data.error}
                </p>
              )}

              {/* Roles info (legacy flow_manager path) */}
              {event.data?.ae1_role && (
                <div className="mt-2 flex gap-2 text-[10px]">
                  <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">
                    AE1: {event.data.ae1_role}
                  </span>
                  <span className="bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded">
                    AE2: {event.data.ae2_role}
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
