import { useMemo, useState } from 'react';
import type {
  PersonaMeta,
  PlanSubtask,
  RoleId,
  SubtaskRuntime,
  SubtaskStatus,
  TaskPlan,
} from '../types';
import { ROLE_PALETTE } from '../types';
import { Markdown } from './Markdown';

interface Props {
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
}

// ── Layout constants ─────────────────────────────────────────────────────────
const COL_WIDTH = 240;   // horizontal spacing between levels
const ROW_HEIGHT = 110;  // vertical spacing between nodes in the same level
const NODE_W = 200;
const NODE_H = 86;
const PADDING = 24;

// ── Status badge styles (orthogonal to role color) ──────────────────────────
const STATUS_BADGE: Record<
  SubtaskStatus,
  { label: string; bg: string; ring: string }
> = {
  pending: { label: 'pendiente', bg: '#9ca3af', ring: '#d1d5db' },
  running: { label: 'ejecutando', bg: '#3b82f6', ring: '#3b82f6' },
  done:    { label: 'completado', bg: '#10b981', ring: '#10b981' },
  failed:  { label: 'error',      bg: '#ef4444', ring: '#ef4444' },
};

// Phase 3: every node has a role; if missing (legacy or pre-dispatch state)
// fall back to a neutral grey so the graph still renders.
const NEUTRAL_PALETTE = {
  label: 'Worker',
  fill: '#f9fafb',
  stroke: '#9ca3af',
  badge: 'bg-gray-100 text-gray-700',
  accent: '#9ca3af',
};

function paletteFor(role: RoleId | null | undefined) {
  if (role && role in ROLE_PALETTE) return ROLE_PALETTE[role];
  return NEUTRAL_PALETTE;
}

// ── Topological layout: assign each node to a "level" (column) ──────────────
function computeLayout(subtasks: PlanSubtask[]) {
  const byId = new Map(subtasks.map((t) => [t.id, t]));
  const levelCache = new Map<string, number>();

  function levelOf(id: string, visiting: Set<string>): number {
    if (levelCache.has(id)) return levelCache.get(id)!;
    if (visiting.has(id)) return 0; // cycle guard; shouldn't happen
    visiting.add(id);
    const node = byId.get(id);
    if (!node || node.depends_on.length === 0) {
      levelCache.set(id, 0);
      return 0;
    }
    const lvl = 1 + Math.max(...node.depends_on.map((d) => levelOf(d, visiting)));
    levelCache.set(id, lvl);
    return lvl;
  }

  const levels: Record<number, PlanSubtask[]> = {};
  for (const t of subtasks) {
    const lvl = levelOf(t.id, new Set());
    (levels[lvl] ||= []).push(t);
  }
  for (const lvl of Object.keys(levels)) {
    levels[Number(lvl)].sort((a, b) => a.id.localeCompare(b.id));
  }

  const positions = new Map<string, { x: number; y: number }>();
  const maxLevel = Math.max(...Object.keys(levels).map(Number));
  const maxRows = Math.max(...Object.values(levels).map((l) => l.length));

  for (let l = 0; l <= maxLevel; l++) {
    const nodes = levels[l] || [];
    const offset = ((maxRows - nodes.length) * ROW_HEIGHT) / 2;
    for (let i = 0; i < nodes.length; i++) {
      positions.set(nodes[i].id, {
        x: PADDING + l * COL_WIDTH,
        y: PADDING + offset + i * ROW_HEIGHT,
      });
    }
  }

  const width = PADDING * 2 + (maxLevel + 1) * COL_WIDTH - (COL_WIDTH - NODE_W);
  const height = PADDING * 2 + maxRows * ROW_HEIGHT - (ROW_HEIGHT - NODE_H);

  return { positions, width, height };
}

function edgePath(
  from: { x: number; y: number },
  to: { x: number; y: number },
) {
  const x1 = from.x + NODE_W;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y + NODE_H / 2;
  const dx = (x2 - x1) * 0.5;
  return `M ${x1},${y1} C ${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
}

export function DebateGraph({ plan, runtime }: Props) {
  const [selected, setSelected] = useState<string | null>(null);

  const layout = useMemo(() => {
    if (!plan) return null;
    return computeLayout(plan.subtasks);
  }, [plan]);

  if (!plan || !layout) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400 text-xs border border-dashed border-gray-200 rounded-lg">
        <p>El grafo del plan aparecerá aquí cuando el orquestador lo genere.</p>
      </div>
    );
  }

  const { positions, width, height } = layout;
  const selectedTask = selected ? plan.subtasks.find((t) => t.id === selected) : null;
  const selectedRuntime = selected ? runtime[selected] : undefined;
  const selectedRole =
    selectedRuntime?.role_id ?? selectedTask?.role_id ?? null;
  const selectedPalette = paletteFor(selectedRole);
  const selectedPersona: PersonaMeta | undefined = selectedRuntime?.persona;

  return (
    <div className="flex flex-col gap-2">
      {/* Goal + claim header */}
      <div className="text-xs text-gray-600 space-y-1">
        <div>
          <span className="font-semibold">Objetivo:</span>{' '}
          <span className="italic">{plan.goal}</span>
        </div>
        {plan.claim && plan.claim !== plan.goal && (
          <div>
            <span className="font-semibold">Tesis:</span>{' '}
            <span className="italic text-gray-500">{plan.claim}</span>
          </div>
        )}
      </div>

      {/* Role legend */}
      <div className="flex flex-wrap gap-1.5 text-[10px]">
        {(Object.keys(ROLE_PALETTE) as RoleId[]).map((r) => {
          const p = ROLE_PALETTE[r];
          return (
            <span
              key={r}
              className={`px-1.5 py-0.5 rounded font-medium ${p.badge}`}
            >
              {p.label}
            </span>
          );
        })}
      </div>

      {/* SVG graph */}
      <div className="overflow-auto border border-gray-200 rounded-lg bg-gradient-to-br from-gray-50 to-white">
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          xmlns="http://www.w3.org/2000/svg"
          style={{ display: 'block' }}
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
            </marker>
          </defs>

          {/* Edges: stroke uses the *destination* role accent so the visual
              flow points toward its consumer. */}
          {plan.subtasks.flatMap((t) =>
            t.depends_on.map((dep) => {
              const from = positions.get(dep);
              const to = positions.get(t.id);
              if (!from || !to) return null;
              const toStatus = runtime[t.id]?.status ?? 'pending';
              const toRole =
                runtime[t.id]?.role_id ?? t.role_id ?? null;
              const toPalette = paletteFor(toRole);
              const stroke =
                toStatus === 'running'
                  ? toPalette.accent
                  : toStatus === 'done'
                  ? '#10b981'
                  : toStatus === 'failed'
                  ? '#ef4444'
                  : '#d1d5db';
              return (
                <path
                  key={`${dep}->${t.id}`}
                  d={edgePath(from, to)}
                  stroke={stroke}
                  strokeWidth={2}
                  fill="none"
                  markerEnd="url(#arrow)"
                  opacity={0.8}
                />
              );
            }),
          )}

          {/* Nodes */}
          {plan.subtasks.map((t) => {
            const pos = positions.get(t.id)!;
            const status = runtime[t.id]?.status ?? 'pending';
            const role = runtime[t.id]?.role_id ?? t.role_id ?? null;
            const palette = paletteFor(role);
            const statusBadge = STATUS_BADGE[status];
            const isSelected = selected === t.id;
            const persona = runtime[t.id]?.persona;
            const stratagemId = persona?.stratagem_id ?? null;

            // Top label: persona display name if configured, else role label,
            // else worker_id, else subtask id. Falls back gracefully through
            // the configure → dispatch → done lifecycle.
            const topLabel =
              persona?.display_name ||
              palette.label ||
              runtime[t.id]?.worker_id ||
              t.id;

            return (
              <g
                key={t.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelected(isSelected ? null : t.id)}
              >
                {/* Pulsing aura while running, tinted with the role accent */}
                {status === 'running' && (
                  <rect
                    x={-4}
                    y={-4}
                    width={NODE_W + 8}
                    height={NODE_H + 8}
                    rx={10}
                    fill={palette.accent}
                    opacity={0.18}
                  >
                    <animate
                      attributeName="opacity"
                      values="0.1;0.3;0.1"
                      dur="1.4s"
                      repeatCount="indefinite"
                    />
                  </rect>
                )}

                <rect
                  x={0}
                  y={0}
                  width={NODE_W}
                  height={NODE_H}
                  rx={8}
                  fill={palette.fill}
                  stroke={isSelected ? '#1f2937' : palette.stroke}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                />

                {/* Persona / role label (top) */}
                <text
                  x={12}
                  y={20}
                  fontSize={12}
                  fontWeight={600}
                  fill={palette.accent}
                >
                  {truncate(topLabel, 28)}
                </text>

                {/* Description (middle, truncated) */}
                <text x={12} y={40} fontSize={10} fill="#4b5563">
                  {truncate(t.description, 32)}
                </text>

                {/* Stratagem badge (bottom-left, only for devil's advocate) */}
                {stratagemId !== null && stratagemId !== undefined && (
                  <g transform={`translate(12, 50)`}>
                    <rect
                      x={0}
                      y={0}
                      width={56}
                      height={14}
                      rx={3}
                      fill="#fee2e2"
                    />
                    <text
                      x={28}
                      y={10}
                      fontSize={9}
                      fontWeight={600}
                      textAnchor="middle"
                      fill="#991b1b"
                    >
                      eristic #{stratagemId}
                    </text>
                  </g>
                )}

                {/* Status badge (bottom-right) */}
                <g transform={`translate(${NODE_W - 78}, ${NODE_H - 22})`}>
                  <rect
                    x={0}
                    y={0}
                    width={70}
                    height={16}
                    rx={3}
                    fill={statusBadge.bg}
                  />
                  <text
                    x={35}
                    y={11}
                    fontSize={9}
                    fontWeight={600}
                    textAnchor="middle"
                    fill="#ffffff"
                  >
                    {statusBadge.label}
                  </text>
                </g>

                {/* Subtask ID (bottom-left) */}
                <text
                  x={12}
                  y={NODE_H - 10}
                  fontSize={9}
                  fill="#9ca3af"
                  fontFamily="monospace"
                >
                  {t.id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detail panel for selected node */}
      {selectedTask && (
        <div
          className="border rounded-lg p-3 bg-white text-xs space-y-2"
          style={{ borderColor: selectedPalette.stroke + '55' }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 flex-wrap">
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${selectedPalette.badge}`}
              >
                {selectedPalette.label}
              </span>
              <span className="font-semibold text-gray-800">
                {selectedPersona?.display_name ||
                  selectedRuntime?.worker_id ||
                  selectedTask.required_skill}
              </span>
              <span className="font-mono text-[10px] text-gray-400">
                ({selectedTask.id})
              </span>
              {selectedPersona?.stratagem_id != null && (
                <span className="text-[10px] px-1.5 py-0.5 rounded font-semibold bg-red-100 text-red-700 border border-red-200">
                  Stratagem #{selectedPersona.stratagem_id}
                </span>
              )}
            </div>
            <button
              onClick={() => setSelected(null)}
              className="text-gray-400 hover:text-gray-600"
              aria-label="Cerrar detalle"
            >
              ×
            </button>
          </div>

          {selectedPersona && (
            <div className="text-[10px] text-gray-500 flex flex-wrap gap-x-3 gap-y-1">
              <span>
                <span className="font-medium">temp:</span>{' '}
                {selectedPersona.temperature.toFixed(2)}
              </span>
              <span>
                <span className="font-medium">tools:</span>{' '}
                {selectedPersona.tool_whitelist.length > 0
                  ? selectedPersona.tool_whitelist.join(', ')
                  : '(ninguna)'}
              </span>
              {selectedRuntime?.worker_id && (
                <span>
                  <span className="font-medium">worker:</span>{' '}
                  <span className="font-mono">{selectedRuntime.worker_id}</span>
                </span>
              )}
            </div>
          )}

          <p className="text-gray-600">{selectedTask.description}</p>
          {selectedTask.depends_on.length > 0 && (
            <p className="text-[10px] text-gray-500">
              Depende de:{' '}
              <span className="font-mono">
                {selectedTask.depends_on.join(', ')}
              </span>
            </p>
          )}
          {selectedRuntime?.output && (
            <details open className="mt-2">
              <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-700 font-medium">
                Salida del worker
              </summary>
              <div className="mt-2 text-gray-700 bg-gray-50 p-2 rounded border border-gray-100">
                <Markdown>{selectedRuntime.output}</Markdown>
              </div>
            </details>
          )}
          {selectedRuntime?.error && (
            <div className="mt-2 text-red-700 bg-red-50 border border-red-200 p-2 rounded">
              <p className="font-semibold mb-1">Error:</p>
              <p className="font-mono text-[10px]">{selectedRuntime.error}</p>
            </div>
          )}
          {!selectedRuntime && (
            <p className="text-[10px] text-gray-400 italic">
              Subtarea aún no iniciada.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1).trimEnd() + '…';
}
