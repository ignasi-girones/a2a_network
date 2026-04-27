import { useMemo } from 'react';
import type { AgentPositionsSample } from '../types';

interface Props {
  samples: AgentPositionsSample[];
}

// ── Layout constants ────────────────────────────────────────────────────────
const WIDTH = 480;
const HEIGHT = 240;
const PADDING_LEFT = 56;
const PADDING_RIGHT = 24;
const PADDING_TOP = 20;
const PADDING_BOTTOM = 36;

const AGENT_STYLE: Record<
  'ae1' | 'ae2' | 'ae3',
  { color: string; label: string; dash?: string }
> = {
  ae1: { color: '#3b82f6', label: 'AE1' },
  ae2: { color: '#10b981', label: 'AE2' },
  ae3: { color: '#a855f7', label: 'AE3', dash: '4 3' },
};

/**
 * Renders the trajectory of every agent's position on the AE1↔AE2 axis
 * across rounds. The x-axis is the round number, the y-axis is the position
 * score (0 = AE1's initial stance, 1 = AE2's initial stance, 0.5 = neutral).
 *
 * Convergence shows up as the three lines moving toward a common y value.
 */
export function AgentPositionsChart({ samples }: Props) {
  const layout = useMemo(() => {
    if (samples.length === 0) return null;

    const innerW = WIDTH - PADDING_LEFT - PADDING_RIGHT;
    const innerH = HEIGHT - PADDING_TOP - PADDING_BOTTOM;
    const maxRound = Math.max(...samples.map((s) => s.round), 0);
    // Always reserve at least 1 step on the x-axis even if there's only
    // one sample, so the points aren't collapsed onto the y-axis.
    const xSteps = Math.max(maxRound, 1);

    const xOf = (round: number) =>
      PADDING_LEFT + (round / xSteps) * innerW;
    const yOf = (pos: number) =>
      PADDING_TOP + (1 - pos) * innerH;

    return { innerW, innerH, xSteps, xOf, yOf };
  }, [samples]);

  if (samples.length === 0 || !layout) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-400 text-xs border border-dashed border-gray-200 rounded-lg">
        <p>
          El gráfico de posicionamiento aparecerá cuando los agentes hayan
          intercambiado su primera ronda.
        </p>
      </div>
    );
  }

  const { xOf, yOf, xSteps } = layout;

  // Build per-agent path strings
  const agentTags: Array<'ae1' | 'ae2' | 'ae3'> = ['ae1', 'ae2', 'ae3'];
  const series = agentTags.map((tag) => {
    const points = samples
      .filter((s) => s.positions[tag] !== undefined)
      .map((s) => ({ x: xOf(s.round), y: yOf(s.positions[tag] as number) }));
    const path = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(' ');
    return { tag, points, path };
  });

  // Y-axis tick labels (0, 0.25, 0.5, 0.75, 1.0)
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
  // X-axis tick labels (round numbers, capped at xSteps + 1 to avoid noise)
  const xTickValues: number[] = [];
  for (let i = 0; i <= xSteps; i++) xTickValues.push(i);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">
          Posicionamiento de los agentes
        </h3>
        <div className="flex items-center gap-3 text-[10px]">
          {agentTags.map((tag) => (
            <span key={tag} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-0.5"
                style={{
                  backgroundColor: AGENT_STYLE[tag].color,
                  borderTop: AGENT_STYLE[tag].dash
                    ? `2px dashed ${AGENT_STYLE[tag].color}`
                    : undefined,
                }}
              />
              <span className="text-gray-600">{AGENT_STYLE[tag].label}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="overflow-auto border border-gray-200 rounded-lg bg-white">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          width="100%"
          height={HEIGHT}
          xmlns="http://www.w3.org/2000/svg"
          style={{ display: 'block' }}
        >
          {/* Background gradient bands: AE1 zone (top) and AE2 zone (bottom) */}
          <defs>
            <linearGradient id="ae1-band" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#dbeafe" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#dbeafe" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="ae2-band" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%" stopColor="#d1fae5" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#d1fae5" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect
            x={PADDING_LEFT}
            y={PADDING_TOP}
            width={WIDTH - PADDING_LEFT - PADDING_RIGHT}
            height={(HEIGHT - PADDING_TOP - PADDING_BOTTOM) / 2}
            fill="url(#ae1-band)"
          />
          <rect
            x={PADDING_LEFT}
            y={PADDING_TOP + (HEIGHT - PADDING_TOP - PADDING_BOTTOM) / 2}
            width={WIDTH - PADDING_LEFT - PADDING_RIGHT}
            height={(HEIGHT - PADDING_TOP - PADDING_BOTTOM) / 2}
            fill="url(#ae2-band)"
          />

          {/* Grid lines + Y-axis labels */}
          {yTicks.map((t) => (
            <g key={t}>
              <line
                x1={PADDING_LEFT}
                x2={WIDTH - PADDING_RIGHT}
                y1={yOf(t)}
                y2={yOf(t)}
                stroke="#e5e7eb"
                strokeDasharray={t === 0.5 ? '4 3' : undefined}
                strokeWidth={1}
              />
              <text
                x={PADDING_LEFT - 6}
                y={yOf(t) + 4}
                fontSize={10}
                fill="#6b7280"
                textAnchor="end"
              >
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {/* Y-axis end-labels */}
          <text
            x={4}
            y={yOf(1) + 4}
            fontSize={10}
            fontWeight={600}
            fill="#1e40af"
          >
            ← AE1
          </text>
          <text
            x={4}
            y={yOf(0) + 4}
            fontSize={10}
            fontWeight={600}
            fill="#047857"
          >
            ← AE2
          </text>
          <text
            x={4}
            y={yOf(0.5) + 4}
            fontSize={9}
            fill="#7c3aed"
          >
            neutral
          </text>

          {/* X-axis labels (round numbers) */}
          {xTickValues.map((r) => (
            <g key={r}>
              <line
                x1={xOf(r)}
                x2={xOf(r)}
                y1={HEIGHT - PADDING_BOTTOM}
                y2={HEIGHT - PADDING_BOTTOM + 4}
                stroke="#9ca3af"
                strokeWidth={1}
              />
              <text
                x={xOf(r)}
                y={HEIGHT - PADDING_BOTTOM + 16}
                fontSize={10}
                fill="#6b7280"
                textAnchor="middle"
              >
                {r === 0 ? 'apertura' : `r${r}`}
              </text>
            </g>
          ))}
          <text
            x={(WIDTH + PADDING_LEFT - PADDING_RIGHT) / 2}
            y={HEIGHT - 6}
            fontSize={10}
            fill="#374151"
            textAnchor="middle"
          >
            ronda
          </text>

          {/* Series — paths and points */}
          {series.map(({ tag, points, path }) => {
            const style = AGENT_STYLE[tag];
            return (
              <g key={tag}>
                {points.length > 1 && (
                  <path
                    d={path}
                    stroke={style.color}
                    strokeWidth={2.5}
                    fill="none"
                    strokeDasharray={style.dash}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}
                {points.map((p, i) => (
                  <g key={i}>
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={4}
                      fill="white"
                      stroke={style.color}
                      strokeWidth={2}
                    />
                  </g>
                ))}
              </g>
            );
          })}
        </svg>
      </div>

      <p className="text-[10px] text-gray-500">
        Cada punto es la posición evaluada del agente tras esa ronda. La
        convergencia hacia un valor común indica consenso emergente.
      </p>
    </div>
  );
}
