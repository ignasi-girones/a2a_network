/**
 * Phase 3 / Pillar 2 — per-agent log-odds trajectory chart.
 *
 * Improvements over v1:
 *  - Y axis spans ±5 (matches the belief_updater clamp) instead of ±3,
 *    so no values are silently hidden at the visual edge.
 *  - Right axis shows the equivalent probability (sigmoid of log-odds)
 *    so the chart is legible to non-statisticians.
 *  - Synthesizer is excluded from debate lines (it evaluates, not debates).
 *  - Each data point carries a coloured delta badge (▲ green / ▼ red) so
 *    direction of change is visible without inspecting the tooltip.
 *  - Endpoint labels print the final value next to the last dot.
 *  - Larger canvas (W=580 H=280) gives more vertical resolution.
 */

import { useMemo } from 'react';
import type { BeliefSample, BeliefSeries, RoleId } from '../types';
import { ROLE_PALETTE } from '../types';

interface Props {
  series: BeliefSeries[];
  claim: string | null;
}

// Canvas dimensions
const W = 580;
const H = 280;
const MARGIN = { top: 20, right: 68, bottom: 32, left: 42 };
const PLOT_W = W - MARGIN.left - MARGIN.right;
const PLOT_H = H - MARGIN.top - MARGIN.bottom;

// Y range matches belief_updater._LOG_ODDS_MAX = ±5
const Y_MIN = -5;
const Y_MAX = 5;

// Semantic zones — recalibrated for ±5 range
const BANDS = [
  { lo: -5, hi: -3, fill: '#fee2e2', label: 'rechazo fuerte' },
  { lo: -3, hi: -1, fill: '#fef2f2', label: 'rechazo' },
  { lo: -1, hi:  1, fill: '#f3f4f6', label: 'neutral' },
  { lo:  1, hi:  3, fill: '#ecfdf5', label: 'aceptación' },
  { lo:  3, hi:  5, fill: '#d1fae5', label: 'aceptación fuerte' },
];

// Roles that participate in the debate belief trajectory.
// Synthesizer deliberates but doesn't take a positional stance.
const DEBATE_ROLES = new Set<RoleId>([
  'analyst',
  'seeker',
  'devils_advocate',
  'empiricist',
  'pragmatist',
]);

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function pctLabel(logOdds: number): string {
  const p = sigmoid(logOdds) * 100;
  return p >= 99.5 ? '>99%' : p <= 0.5 ? '<1%' : `${Math.round(p)}%`;
}

function colorFor(role: RoleId | null | undefined): string {
  if (role && role in ROLE_PALETTE) return ROLE_PALETTE[role].accent;
  return '#6b7280';
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function yScale(value: number): number {
  const v = clamp(value, Y_MIN, Y_MAX);
  return ((Y_MAX - v) / (Y_MAX - Y_MIN)) * PLOT_H;
}

function xScale(idx: number, n: number): number {
  if (n <= 1) return PLOT_W / 2;
  return (idx / (n - 1)) * PLOT_W;
}

function pathFor(samples: BeliefSample[], totalN: number): string {
  if (samples.length === 0) return '';
  return samples
    .map((s, i) => {
      const x = xScale(i, Math.max(totalN, samples.length));
      const y = yScale(s.log_odds);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

// Y-axis ticks: log-odds on left, probability % on right
const Y_TICKS = [-5, -3, -2, -1, 0, 1, 2, 3, 5];

export function BeliefTrajectoryChart({ series, claim }: Props) {
  // Only debate-side agents appear in the chart
  const debateSeries = useMemo(
    () => series.filter((s) => s.role_id && DEBATE_ROLES.has(s.role_id)),
    [series],
  );

  const maxLen = useMemo(
    () => debateSeries.reduce((m, s) => Math.max(m, s.samples.length), 0),
    [debateSeries],
  );

  if (series.length === 0) {
    return (
      <div className="flex items-center justify-center h-28 text-gray-400 text-xs border border-dashed border-gray-200 rounded-lg">
        <p>Las trayectorias bayesianas aparecerán aquí cuando cada agente
        emita su primer belief_update.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {claim && (
        <div className="text-xs text-gray-600">
          <span className="font-semibold">Tesis:</span>{' '}
          <span className="italic">{claim}</span>
        </div>
      )}

      <div className="overflow-auto border border-gray-200 rounded-lg bg-white">
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          xmlns="http://www.w3.org/2000/svg"
          style={{ display: 'block', width: '100%', height: 'auto' }}
        >
          <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>

            {/* ── Semantic background bands ── */}
            {BANDS.map((b) => {
              const yTop = yScale(b.hi);
              const yBot = yScale(b.lo);
              return (
                <rect
                  key={`band-${b.lo}`}
                  x={0} y={yTop}
                  width={PLOT_W}
                  height={Math.max(0, yBot - yTop)}
                  fill={b.fill}
                  opacity={0.6}
                />
              );
            })}

            {/* ── Band labels (right edge, inside plot) ── */}
            {BANDS.map((b) => {
              const midY = (yScale(b.lo) + yScale(b.hi)) / 2;
              return (
                <text
                  key={`blabel-${b.lo}`}
                  x={PLOT_W - 4}
                  y={midY + 3}
                  fontSize={7.5}
                  textAnchor="end"
                  fill="#9ca3af"
                  fontStyle="italic"
                >
                  {b.label}
                </text>
              );
            })}

            {/* ── Zero baseline ── */}
            <line
              x1={0} x2={PLOT_W}
              y1={yScale(0)} y2={yScale(0)}
              stroke="#9ca3af"
              strokeDasharray="4,3"
              strokeWidth={1}
            />

            {/* ── Left Y-axis: log-odds ticks ── */}
            {Y_TICKS.map((tick) => (
              <g key={`ytick-${tick}`} transform={`translate(0, ${yScale(tick)})`}>
                <line x1={-4} x2={0} y1={0} y2={0} stroke="#d1d5db" />
                <text
                  x={-7} y={3.5}
                  fontSize={9}
                  textAnchor="end"
                  fill="#6b7280"
                  fontFamily="monospace"
                >
                  {tick > 0 ? `+${tick}` : tick}
                </text>
              </g>
            ))}

            {/* ── Right Y-axis: probability % ── */}
            {Y_TICKS.map((tick) => (
              <g key={`ptick-${tick}`} transform={`translate(${PLOT_W}, ${yScale(tick)})`}>
                <line x1={0} x2={4} y1={0} y2={0} stroke="#d1d5db" />
                <text
                  x={7} y={3.5}
                  fontSize={8.5}
                  textAnchor="start"
                  fill="#9ca3af"
                >
                  {pctLabel(tick)}
                </text>
              </g>
            ))}

            {/* ── Right axis label ── */}
            <text
              x={PLOT_W + 60}
              y={PLOT_H / 2}
              fontSize={8}
              fill="#9ca3af"
              textAnchor="middle"
              transform={`rotate(90, ${PLOT_W + 60}, ${PLOT_H / 2})`}
            >
              P(tesis)
            </text>

            {/* ── X-axis baseline ── */}
            <line
              x1={0} x2={PLOT_W}
              y1={PLOT_H} y2={PLOT_H}
              stroke="#d1d5db"
              strokeWidth={1}
            />

            {/* ── Series ── */}
            {debateSeries.map((s) => {
              const color = colorFor(s.role_id);
              const n = Math.max(maxLen, s.samples.length);
              const last = s.samples[s.samples.length - 1];
              return (
                <g key={s.agent}>
                  {/* Line */}
                  <path
                    d={pathFor(s.samples, n)}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />

                  {/* Data points with delta-direction colour */}
                  {s.samples.map((sample, i) => {
                    const cx = xScale(i, n);
                    const cy = yScale(sample.log_odds);
                    const dotColor =
                      sample.delta > 0.05
                        ? '#16a34a'
                        : sample.delta < -0.05
                        ? '#dc2626'
                        : '#9ca3af';
                    return (
                      <g key={i}>
                        <circle
                          cx={cx} cy={cy}
                          r={4}
                          fill="white"
                          stroke={color}
                          strokeWidth={1.5}
                        />
                        {/* Delta dot inside */}
                        <circle cx={cx} cy={cy} r={2} fill={dotColor} />
                        <title>
                          {`${s.role_id ?? s.agent} · turno ${sample.t} (${sample.phase})\n` +
                            `log-odds = ${sample.log_odds.toFixed(2)} → P(tesis) ≈ ${pctLabel(sample.log_odds)}\n` +
                            `Δ = ${sample.delta >= 0 ? '+' : ''}${sample.delta.toFixed(2)}\n` +
                            (sample.rationale || '(sin justificación)')}
                        </title>
                      </g>
                    );
                  })}

                  {/* Endpoint value label */}
                  {last && (
                    <text
                      x={xScale(s.samples.length - 1, n) + 6}
                      y={yScale(last.log_odds) + 3.5}
                      fontSize={8.5}
                      fill={color}
                      fontWeight="600"
                      fontFamily="monospace"
                    >
                      {last.log_odds >= 0 ? '+' : ''}
                      {last.log_odds.toFixed(1)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* ── Left axis label ── */}
          <text
            x={11}
            y={MARGIN.top + PLOT_H / 2}
            fontSize={9}
            fill="#6b7280"
            transform={`rotate(-90, 11, ${MARGIN.top + PLOT_H / 2})`}
            textAnchor="middle"
          >
            log-odds
          </text>

          {/* ── X-axis label ── */}
          <text
            x={MARGIN.left + PLOT_W / 2}
            y={H - 6}
            fontSize={9}
            fill="#6b7280"
            textAnchor="middle"
          >
            turno (t)
          </text>
        </svg>
      </div>

      {/* ── Legend ── */}
      <div className="flex flex-wrap gap-2 text-[10px]">
        {debateSeries.map((s) => {
          const last = s.samples[s.samples.length - 1];
          const color = colorFor(s.role_id);
          const label = (s.role_id && ROLE_PALETTE[s.role_id]?.label) || s.agent;
          const prob = last ? ` ≈ ${pctLabel(last.log_odds)}` : '';
          return (
            <div
              key={s.agent}
              className="flex items-center gap-1 px-2 py-1 rounded border bg-white"
              style={{ borderColor: color + '55' }}
            >
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: color }}
              />
              <span className="font-medium text-gray-700">{label}</span>
              {last && (
                <>
                  <span className="font-mono text-gray-500">
                    {last.log_odds >= 0 ? '+' : ''}
                    {last.log_odds.toFixed(2)}
                  </span>
                  <span className="text-gray-400">{prob}</span>
                </>
              )}
              <span className="text-gray-400">· {s.samples.length} pts</span>
            </div>
          );
        })}
        {/* Synthesizer shown separately as non-debate observer */}
        {series
          .filter((s) => s.role_id === 'synthesizer')
          .map((s) => {
            const last = s.samples[s.samples.length - 1];
            return (
              <div
                key={s.agent}
                className="flex items-center gap-1 px-2 py-1 rounded border bg-white opacity-50"
                style={{ borderColor: '#7c3aed55' }}
              >
                <span className="inline-block w-2 h-2 rounded-full" style={{ background: '#7c3aed' }} />
                <span className="text-gray-500 italic">Sintetizador (observador)</span>
                {last && (
                  <span className="font-mono text-gray-400">
                    {last.log_odds.toFixed(2)}
                  </span>
                )}
              </div>
            );
          })}
      </div>

      {/* ── Reading guide ── */}
      <div className="text-[9px] text-gray-400 flex gap-3">
        <span>
          <span className="inline-block w-2 h-2 rounded-full bg-green-600 mr-1" />
          punto subió
        </span>
        <span>
          <span className="inline-block w-2 h-2 rounded-full bg-red-600 mr-1" />
          punto bajó
        </span>
        <span>Hover sobre un punto para ver la justificación del cambio</span>
      </div>
    </div>
  );
}
