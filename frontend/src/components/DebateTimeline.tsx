import type { DebateEvent, RoleId, TaskPlan } from '../types';
import { ROLE_PALETTE } from '../types';
import { Markdown } from './Markdown';

interface Props {
  events: DebateEvent[];
  plan?: TaskPlan | null;
}

// ── Per-event visual classification ─────────────────────────────────────────
//
// Phase 3 routing: every dispatched/done/failed event carries a `role_id`
// (added by PlanExecutor._configure_worker → subtask_dispatch). Events
// without a role fall back to the worker_id-based mapping for the
// non-role agents (normalizer, feedback) and for system events.

function getEventStyle(event: DebateEvent) {
  const stage = event.stage;
  const agent = event.data?.agent;
  const workerId = event.data?.worker_id;
  // Role can come from data.role_id (dispatch/done events) OR from a
  // worker_id that matches a canonical role name (specialized worker
  // tool_use events).
  const role: RoleId | undefined =
    (event.data?.role_id as RoleId | undefined) ||
    (workerId && workerId in ROLE_PALETTE ? (workerId as RoleId) : undefined);

  if (role) {
    const palette = ROLE_PALETTE[role];
    // Alignment heuristic: analyst left, seeker right (so they sit on either
    // side after t1), devil's advocate centred-right, synthesizer centred.
    const align =
      role === 'analyst'
        ? 'self-start'
        : role === 'seeker'
        ? 'self-end'
        : role === 'devils_advocate'
        ? 'self-end'
        : 'self-center';
    return {
      align,
      bg: roleBg(role),
      label: palette.label,
      labelColor: roleLabelColor(role),
    };
  }

  if (workerId === 'normalizer' || agent === 'normalizer') {
    return {
      align: 'self-center',
      bg: 'bg-amber-50 border-amber-200',
      label: 'Normalizador',
      labelColor: 'bg-amber-600',
    };
  }
  if (workerId === 'feedback' || agent === 'feedback') {
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

  if (stage === 'spawn' || stage === 'spawn_failed') {
    return {
      align: 'self-center',
      bg: 'bg-orange-50 border-orange-200',
      label: 'Spawner',
      labelColor: 'bg-orange-500',
    };
  }

  if (
    stage === 'consensus' ||
    stage === 'complete' ||
    stage === 'plan_complete' ||
    stage === 'plan_ready' ||
    stage === 'plan_start' ||
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

function roleBg(role: RoleId): string {
  // Tailwind doesn't compile dynamic class names, so map explicitly.
  switch (role) {
    case 'analyst':
      return 'bg-blue-50 border-blue-200';
    case 'seeker':
      return 'bg-emerald-50 border-emerald-200';
    case 'devils_advocate':
      return 'bg-red-50 border-red-200';
    case 'empiricist':
      return 'bg-orange-50 border-orange-200';
    case 'pragmatist':
      return 'bg-teal-50 border-teal-200';
    case 'synthesizer':
      return 'bg-purple-50 border-purple-200';
  }
}

function roleLabelColor(role: RoleId): string {
  switch (role) {
    case 'analyst':
      return 'bg-blue-600';
    case 'seeker':
      return 'bg-emerald-600';
    case 'devils_advocate':
      return 'bg-red-600';
    case 'empiricist':
      return 'bg-orange-600';
    case 'pragmatist':
      return 'bg-teal-600';
    case 'synthesizer':
      return 'bg-purple-600';
  }
}

function humanizeMessage(event: DebateEvent, plan: TaskPlan | null | undefined): string {
  const raw = event.message;
  const id = event.data?.subtask_id;
  if (!id || !plan) return raw;
  const task = plan.subtasks.find((t) => t.id === id);
  if (!task) return raw;
  const worker = event.data?.worker_id;
  const role = event.data?.role_id ?? task.role_id;
  const personaName =
    event.data?.persona?.display_name ||
    (role ? ROLE_PALETTE[role]?.label : undefined);
  const tag = personaName || worker || task.required_skill;

  if (event.stage === 'subtask_dispatch') {
    return `→ ${tag}: ${truncate(task.description, 60)}`;
  }
  if (event.stage === 'subtask_done') {
    return `✓ ${tag} completado`;
  }
  if (event.stage === 'subtask_failed') {
    return `✗ ${tag} falló`;
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
        const stratagemId =
          event.data?.persona?.stratagem_id ?? event.data?.stratagem_id ?? null;

        return (
          <div key={i} className={`flex flex-col ${style.align} max-w-[85%]`}>
            <div className={`rounded-lg border p-3 ${style.bg} text-sm`}>
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <span
                  className={`text-[10px] font-bold text-white px-1.5 py-0.5 rounded ${style.labelColor}`}
                >
                  {style.label}
                </span>
                {stratagemId != null && (
                  <span className="text-[10px] text-red-700 bg-red-100 border border-red-200 px-1.5 py-0.5 rounded font-medium">
                    Stratagem #{stratagemId}
                  </span>
                )}
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

              <p className="text-gray-700 text-xs font-medium">{label}</p>

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

              {event.data?.error && (
                <p className="mt-1 text-[10px] text-red-600 font-mono">
                  {event.data.error}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
