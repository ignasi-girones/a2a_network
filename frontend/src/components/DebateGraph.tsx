import { useMemo, useState } from 'react';
import type { PlanSubtask, SubtaskRuntime, SubtaskStatus, TaskPlan } from '../types';
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

// ── Visual style per status ──────────────────────────────────────────────────
const STATUS_STYLE: Record<
  SubtaskStatus,
  { fill: string; stroke: string; label: string; labelBg: string }
> = {
  pending: {
    fill: '#f9fafb',
    stroke: '#d1d5db',
    label: 'pendiente',
    labelBg: '#e5e7eb',
  },
  running: {
    fill: '#eff6ff',
    stroke: '#3b82f6',
    label: 'ejecutando',
    labelBg: '#3b82f6',
  },
  done: {
    fill: '#ecfdf5',
    stroke: '#10b981',
    label: 'completado',
    labelBg: '#10b981',
  },
  failed: {
    fill: '#fef2f2',
    stroke: '#ef4444',
    label: 'error',
    labelBg: '#ef4444',
  },
};

// ── Topological layout: assign each node to a "level" (column) ──────────────
// level(n) = 0 if n has no deps; otherwise max(level(d)) + 1 over deps d.
// Within a level, order nodes by id for deterministic vertical placement.
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
  // Sort each level by id for stability
  for (const lvl of Object.keys(levels)) {
    levels[Number(lvl)].sort((a, b) => a.id.localeCompare(b.id));
  }

  // Positions
  const positions = new Map<string, { x: number; y: number }>();
  const maxLevel = Math.max(...Object.keys(levels).map(Number));
  const maxRows = Math.max(...Object.values(levels).map((l) => l.length));

  for (let l = 0; l <= maxLevel; l++) {
    const nodes = levels[l] || [];
    // Center nodes within the column vertically so short levels don't hug the top
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

// ── Cubic Bezier path between two node centers-right → center-left ──────────
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

  return (
    <div className="flex flex-col gap-2">
      {/* Goal header */}
      <div className="text-xs text-gray-600">
        <span className="font-semibold">Objetivo:</span>{' '}
        <span className="italic">{plan.goal}</span>
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
          {/* Arrow marker */}
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

          {/* Edges (directed) */}
          {plan.subtasks.flatMap((t) =>
            t.depends_on.map((dep) => {
              const from = positions.get(dep);
              const to = positions.get(t.id);
              if (!from || !to) return null;
              const toStatus = runtime[t.id]?.status ?? 'pending';
              const stroke =
                toStatus === 'running'
                  ? '#3b82f6'
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
            const style = STATUS_STYLE[status];
            const isSelected = selected === t.id;
            const workerId = runtime[t.id]?.worker_id;
            const label =
              workerId ||
              (t.required_skill ? t.required_skill : t.id);

            return (
              <g
                key={t.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelected(isSelected ? null : t.id)}
              >
                {/* Pulsing aura while running */}
                {status === 'running' && (
                  <rect
                    x={-4}
                    y={-4}
                    width={NODE_W + 8}
                    height={NODE_H + 8}
                    rx={10}
                    fill="#3b82f6"
                    opacity={0.15}
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
                  fill={style.fill}
                  stroke={isSelected ? '#1f2937' : style.stroke}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                />

                {/* Worker label (top) */}
                <text
                  x={12}
                  y={20}
                  fontSize={12}
                  fontWeight={600}
                  fill="#111827"
                >
                  {label}
                </text>

                {/* Description (middle, truncated) */}
                <text
                  x={12}
                  y={40}
                  fontSize={10}
                  fill="#4b5563"
                >
                  {truncate(t.description, 30)}
                </text>

                {/* Perspective tag (if present) */}
                {t.perspective && (
                  <g transform={`translate(12, 52)`}>
                    <rect
                      x={0}
                      y={0}
                      width={44}
                      height={14}
                      rx={3}
                      fill={t.perspective === 'pro' ? '#dbeafe' : '#fee2e2'}
                    />
                    <text
                      x={22}
                      y={10}
                      fontSize={9}
                      fontWeight={600}
                      textAnchor="middle"
                      fill={t.perspective === 'pro' ? '#1e40af' : '#991b1b'}
                    >
                      {t.perspective}
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
                    fill={style.labelBg}
                  />
                  <text
                    x={35}
                    y={11}
                    fontSize={9}
                    fontWeight={600}
                    textAnchor="middle"
                    fill="#ffffff"
                  >
                    {style.label}
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
        <div className="border border-gray-200 rounded-lg p-3 bg-white text-xs space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-800">
                {selectedRuntime?.worker_id || selectedTask.required_skill}
              </span>
              <span className="font-mono text-[10px] text-gray-400">
                ({selectedTask.id})
              </span>
              {selectedTask.perspective && (
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${
                    selectedTask.perspective === 'pro'
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-red-100 text-red-700'
                  }`}
                >
                  {selectedTask.perspective}
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
