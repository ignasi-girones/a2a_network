/**
 * Phase 3 / Pillar 3 — Habermasian validity-claims table emitted by the
 * Synthesizer. The row scores are 0.0–1.0; the ``admitted`` flag tells
 * the user which contributions actually made it into the verdict.
 *
 * Citation handle for the TFG memoria: this is the most directly auditable
 * output of the deliberation, mapping each panel role onto Habermas's
 * four claims (truth, rightness, sincerity, comprehensibility).
 */

import type { ValidityClaimRow } from '../types';

interface Props {
  claims: ValidityClaimRow[] | null;
}

const COLUMNS: { key: keyof ValidityClaimRow; label: string }[] = [
  { key: 'truth', label: 'Verdad' },
  { key: 'rightness', label: 'Rectitud' },
  { key: 'sincerity', label: 'Veracidad' },
  { key: 'comprehensibility', label: 'Comprensib.' },
];

function bandColor(score: number): string {
  // 0.0 → red, 0.6 admission threshold, 1.0 → green
  if (score >= 0.8) return 'bg-emerald-100 text-emerald-800';
  if (score >= 0.6) return 'bg-lime-100 text-lime-800';
  if (score >= 0.4) return 'bg-amber-100 text-amber-800';
  return 'bg-red-100 text-red-800';
}

export function HabermasTable({ claims }: Props) {
  if (!claims || claims.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-gray-700">
          Pretensiones de validez (Habermas)
        </h2>
        <span className="text-[10px] text-gray-400">
          umbral de admisión = 0.60
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-200">
              <th className="text-left font-medium py-1.5 pr-2">Agente</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="text-center font-medium py-1.5 px-2">
                  {c.label}
                </th>
              ))}
              <th className="text-center font-medium py-1.5 px-2">Admitido</th>
              <th className="text-left font-medium py-1.5 pl-2">Nota</th>
            </tr>
          </thead>
          <tbody>
            {claims.map((row, i) => (
              <tr
                key={`${row.agent}-${i}`}
                className="border-b border-gray-100 last:border-0"
              >
                <td className="py-1.5 pr-2 font-medium text-gray-800">
                  {row.agent}
                </td>
                {COLUMNS.map((c) => {
                  const score = (row[c.key] as number) ?? 0;
                  return (
                    <td key={c.key} className="text-center py-1.5 px-2">
                      <span
                        className={`inline-block min-w-[2.4rem] px-1.5 py-0.5 rounded font-mono font-medium ${bandColor(score)}`}
                      >
                        {score.toFixed(2)}
                      </span>
                    </td>
                  );
                })}
                <td className="text-center py-1.5 px-2">
                  {row.admitted ? (
                    <span className="text-emerald-600 font-bold">✓</span>
                  ) : (
                    <span className="text-red-500 font-bold">✗</span>
                  )}
                </td>
                <td className="py-1.5 pl-2 text-gray-500 italic">
                  {row.note || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
