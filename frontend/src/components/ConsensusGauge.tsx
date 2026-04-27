import { useMemo } from 'react';
import type { ConsensusSnapshot } from '../types';

interface Props {
  history: ConsensusSnapshot[];
}

// ── Layout constants for the semicircular gauge ─────────────────────────────
const GAUGE_SIZE = 220;
const GAUGE_RADIUS = 88;
const GAUGE_STROKE = 18;
const GAUGE_CX = GAUGE_SIZE / 2;
const GAUGE_CY = GAUGE_SIZE * 0.78;

// Convert a score in [0, 1] into an angle in [180°, 360°] (left → right arc).
function scoreToAngle(score: number): number {
  return 180 + Math.max(0, Math.min(1, score)) * 180;
}

function polar(cx: number, cy: number, r: number, angleDeg: number) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function arcPath(cx: number, cy: number, r: number, fromDeg: number, toDeg: number) {
  const start = polar(cx, cy, r, fromDeg);
  const end = polar(cx, cy, r, toDeg);
  const large = toDeg - fromDeg > 180 ? 1 : 0;
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}

// ── Verdict band classification ─────────────────────────────────────────────
function classify(score: number): {
  label: string;
  color: string;
  textColor: string;
  bg: string;
} {
  if (score >= 0.75)
    return {
      label: 'Consenso alcanzado',
      color: '#10b981',
      textColor: 'text-emerald-700',
      bg: 'bg-emerald-50 border-emerald-200',
    };
  if (score >= 0.5)
    return {
      label: 'Consenso parcial',
      color: '#f59e0b',
      textColor: 'text-amber-700',
      bg: 'bg-amber-50 border-amber-200',
    };
  return {
    label: 'Sin consenso',
    color: '#ef4444',
    textColor: 'text-red-700',
    bg: 'bg-red-50 border-red-200',
  };
}

export function ConsensusGauge({ history }: Props) {
  const latest = history.length > 0 ? history[history.length - 1] : null;

  const score = latest?.agreement_score ?? 0;
  const verdict = classify(score);
  const needleAngle = scoreToAngle(score);

  // Pre-compute the colored arcs for each band of the gauge.
  const arcs = useMemo(
    () => [
      // 0.00 → 0.50 (red)
      { from: 180, to: 270, color: '#fecaca' },
      // 0.50 → 0.75 (amber)
      { from: 270, to: 315, color: '#fde68a' },
      // 0.75 → 1.00 (emerald)
      { from: 315, to: 360, color: '#a7f3d0' },
    ],
    [],
  );

  const needleEnd = polar(GAUGE_CX, GAUGE_CY, GAUGE_RADIUS - 10, needleAngle);

  // Sparkline geometry
  const sparkW = 220;
  const sparkH = 60;
  const sparkPath =
    history.length === 0
      ? ''
      : history
          .map((h, i) => {
            const x =
              history.length === 1
                ? sparkW / 2
                : (i / (history.length - 1)) * (sparkW - 8) + 4;
            const y = sparkH - h.agreement_score * (sparkH - 8) - 4;
            return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(' ');

  if (history.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-gray-700">
          Evaluación del consenso
        </h3>
        <div className="flex items-center justify-center h-40 text-gray-400 text-xs border border-dashed border-gray-200 rounded-lg">
          <p>
            La evaluación del consenso aparecerá aquí cuando los agentes hayan
            terminado su primera ronda.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">
          Evaluación del consenso
        </h3>
        <span className="text-[10px] text-gray-400">
          {history.length} {history.length === 1 ? 'evaluación' : 'evaluaciones'}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
        {/* Gauge */}
        <div className="flex flex-col items-center">
          <svg
            viewBox={`0 0 ${GAUGE_SIZE} ${GAUGE_SIZE * 0.86}`}
            width="100%"
            style={{ maxWidth: 260 }}
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Coloured arcs */}
            {arcs.map((a, i) => (
              <path
                key={i}
                d={arcPath(GAUGE_CX, GAUGE_CY, GAUGE_RADIUS, a.from, a.to)}
                stroke={a.color}
                strokeWidth={GAUGE_STROKE}
                fill="none"
                strokeLinecap="butt"
              />
            ))}
            {/* Threshold ticks (0.5 and 0.75) */}
            {[0.5, 0.75].map((t) => {
              const angle = scoreToAngle(t);
              const inner = polar(GAUGE_CX, GAUGE_CY, GAUGE_RADIUS - GAUGE_STROKE / 2 - 2, angle);
              const outer = polar(GAUGE_CX, GAUGE_CY, GAUGE_RADIUS + GAUGE_STROKE / 2 + 2, angle);
              return (
                <line
                  key={t}
                  x1={inner.x}
                  y1={inner.y}
                  x2={outer.x}
                  y2={outer.y}
                  stroke="#6b7280"
                  strokeWidth={1.2}
                />
              );
            })}
            {/* Needle */}
            <line
              x1={GAUGE_CX}
              y1={GAUGE_CY}
              x2={needleEnd.x}
              y2={needleEnd.y}
              stroke={verdict.color}
              strokeWidth={3}
              strokeLinecap="round"
              style={{ transition: 'all 0.6s cubic-bezier(.2,.7,.2,1)' }}
            />
            <circle cx={GAUGE_CX} cy={GAUGE_CY} r={6} fill={verdict.color} />
            {/* Endpoint labels */}
            <text x={GAUGE_CX - GAUGE_RADIUS} y={GAUGE_CY + 24} fontSize={10} fill="#9ca3af" textAnchor="middle">
              0
            </text>
            <text x={GAUGE_CX} y={GAUGE_CY - GAUGE_RADIUS - 8} fontSize={10} fill="#9ca3af" textAnchor="middle">
              0.5
            </text>
            <text x={GAUGE_CX + GAUGE_RADIUS} y={GAUGE_CY + 24} fontSize={10} fill="#9ca3af" textAnchor="middle">
              1
            </text>
            {/* Big score */}
            <text
              x={GAUGE_CX}
              y={GAUGE_CY - 20}
              fontSize={28}
              fontWeight={700}
              textAnchor="middle"
              fill="#111827"
            >
              {score.toFixed(2)}
            </text>
          </svg>
          <span
            className={`mt-1 text-xs font-semibold px-2 py-1 rounded ${verdict.bg} ${verdict.textColor}`}
          >
            {verdict.label}
          </span>
        </div>

        {/* Right-side: history sparkline + reason */}
        <div className="flex flex-col gap-2 text-xs text-gray-700">
          <div>
            <p className="text-[10px] uppercase tracking-wide text-gray-500 mb-1">
              Evolución del score
            </p>
            <svg
              viewBox={`0 0 ${sparkW} ${sparkH}`}
              width="100%"
              height={sparkH}
              className="border border-gray-200 rounded bg-gray-50"
              preserveAspectRatio="none"
            >
              {/* Threshold lines */}
              <line
                x1={0}
                x2={sparkW}
                y1={sparkH - 0.75 * (sparkH - 8) - 4}
                y2={sparkH - 0.75 * (sparkH - 8) - 4}
                stroke="#10b981"
                strokeDasharray="3 3"
                strokeWidth={1}
                opacity={0.5}
              />
              <line
                x1={0}
                x2={sparkW}
                y1={sparkH - 0.5 * (sparkH - 8) - 4}
                y2={sparkH - 0.5 * (sparkH - 8) - 4}
                stroke="#f59e0b"
                strokeDasharray="3 3"
                strokeWidth={1}
                opacity={0.5}
              />
              {history.length > 1 && (
                <path
                  d={sparkPath}
                  stroke="#374151"
                  strokeWidth={2}
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              {history.map((h, i) => {
                const x =
                  history.length === 1
                    ? sparkW / 2
                    : (i / (history.length - 1)) * (sparkW - 8) + 4;
                const y = sparkH - h.agreement_score * (sparkH - 8) - 4;
                const fill = classify(h.agreement_score).color;
                return (
                  <circle key={i} cx={x} cy={y} r={4} fill={fill} stroke="white" strokeWidth={1.5} />
                );
              })}
            </svg>
            <div className="flex justify-between text-[10px] text-gray-400 mt-1 px-1">
              <span>
                {history[0]?.round === 0 ? 'apertura' : `r${history[0]?.round}`}
              </span>
              <span>
                {history[history.length - 1]?.round === 0
                  ? 'apertura'
                  : `r${history[history.length - 1]?.round}`}
              </span>
            </div>
          </div>

          {latest?.reason && (
            <div className="bg-gray-50 border border-gray-100 rounded p-2 text-[11px] italic text-gray-600">
              {latest.reason}
            </div>
          )}
        </div>
      </div>

      {/* Shared points and remaining disagreements */}
      {latest &&
        (latest.shared_points.length > 0 ||
          latest.remaining_disagreements.length > 0) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-1">
            <div className="border border-emerald-200 rounded-lg p-3 bg-emerald-50">
              <div className="flex items-center gap-1 mb-2">
                <span className="text-emerald-600">✓</span>
                <h4 className="text-xs font-semibold text-emerald-800">
                  Puntos compartidos
                </h4>
                <span className="ml-auto text-[10px] text-emerald-600 font-mono">
                  {latest.shared_points.length}
                </span>
              </div>
              {latest.shared_points.length === 0 ? (
                <p className="text-[11px] text-emerald-700/70 italic">
                  Aún no hay puntos en común.
                </p>
              ) : (
                <ul className="space-y-1 text-[11px] text-emerald-900">
                  {latest.shared_points.map((p, i) => (
                    <li key={i} className="flex gap-1.5 leading-snug">
                      <span className="text-emerald-500 select-none">•</span>
                      <span>{p}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="border border-amber-200 rounded-lg p-3 bg-amber-50">
              <div className="flex items-center gap-1 mb-2">
                <span className="text-amber-600">!</span>
                <h4 className="text-xs font-semibold text-amber-800">
                  Desacuerdos pendientes
                </h4>
                <span className="ml-auto text-[10px] text-amber-600 font-mono">
                  {latest.remaining_disagreements.length}
                </span>
              </div>
              {latest.remaining_disagreements.length === 0 ? (
                <p className="text-[11px] text-amber-700/70 italic">
                  Sin desacuerdos sustantivos.
                </p>
              ) : (
                <ul className="space-y-1 text-[11px] text-amber-900">
                  {latest.remaining_disagreements.map((p, i) => (
                    <li key={i} className="flex gap-1.5 leading-snug">
                      <span className="text-amber-500 select-none">•</span>
                      <span>{p}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
    </div>
  );
}
