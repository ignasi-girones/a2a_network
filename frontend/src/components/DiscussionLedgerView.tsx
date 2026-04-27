/**
 * DiscussionLedgerView — chat-style threaded view of the deliberation.
 *
 * Phase 4 replaces the flat event timeline as the *primary* dynamic view.
 * Each ledger entry is a card grouped under its round; new entries fade
 * in and slide up. Hover-highlight on `references` lets the reader trace
 * cross-turn argumentation.
 */

import { AnimatePresence, motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { useMemo } from 'react';
import type { LedgerEntry, RoleId } from '../types';
import { ROLE_PALETTE } from '../types';
import { RoleAvatar } from './RoleAvatar';

interface Props {
  entries: LedgerEntry[];
  maxRounds: number;
  currentRound: number;
  activeRole: RoleId | null;
}

function groupByRound(entries: LedgerEntry[]): Map<number, LedgerEntry[]> {
  const out = new Map<number, LedgerEntry[]>();
  for (const e of entries) {
    const arr = out.get(e.round_number) ?? [];
    arr.push(e);
    out.set(e.round_number, arr);
  }
  return out;
}

function DeltaBadge({ delta }: { delta: number | null }) {
  if (delta === null || Math.abs(delta) < 0.01) return null;
  const positive = delta > 0;
  return (
    <span
      className={[
        'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full',
        'text-[10px] font-mono font-medium',
        positive
          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
          : 'bg-rose-50 text-rose-700 border border-rose-200',
      ].join(' ')}
      title={`Cambio en log-odds tras esta intervención`}
    >
      {positive ? '▲' : '▼'} {positive ? '+' : ''}
      {delta.toFixed(2)}
    </span>
  );
}

function LedgerEntryCard({
  entry,
  isActive,
}: {
  entry: LedgerEntry;
  isActive: boolean;
}) {
  const palette = ROLE_PALETTE[entry.role_id];
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
      className="glass rounded-xl p-4 hover:shadow-md transition-shadow"
      style={{
        borderLeftWidth: '3px',
        borderLeftColor: palette.accent,
      }}
      data-turn={entry.turn}
    >
      <div className="flex items-start gap-3">
        <RoleAvatar role={entry.role_id} size="md" active={isActive} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1.5">
            <span
              className="font-semibold text-sm"
              style={{ color: palette.accent }}
            >
              {palette.label}
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              turno {entry.turn}
            </span>
            <DeltaBadge delta={entry.delta} />
            {entry.belief_after !== null && (
              <span
                className="text-[10px] font-mono text-slate-500"
                title="log-odds tras la intervención"
              >
                log-odds{' '}
                {entry.belief_after >= 0 ? '+' : ''}
                {entry.belief_after.toFixed(2)}
              </span>
            )}
            {entry.references.length > 0 && (
              <span className="text-[10px] text-slate-400 italic">
                ↳ responde a turnos {entry.references.join(', ')}
              </span>
            )}
          </div>

          <div className="prose prose-sm prose-slate max-w-none text-slate-800 leading-relaxed">
            <ReactMarkdown>{entry.text}</ReactMarkdown>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function RoundDivider({
  round,
  count,
  isCurrent,
}: {
  round: number;
  count: number;
  isCurrent: boolean;
}) {
  return (
    <div className="flex items-center gap-3 my-3">
      <span
        className={[
          'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium',
          isCurrent
            ? 'bg-blue-50 text-blue-700 border border-blue-200'
            : 'bg-slate-100 text-slate-600 border border-slate-200',
        ].join(' ')}
      >
        <span
          className={[
            'w-1.5 h-1.5 rounded-full',
            isCurrent ? 'bg-blue-500 animate-pulse' : 'bg-slate-400',
          ].join(' ')}
        />
        Ronda {round}
        <span className="text-slate-400 ml-1">· {count} intervención{count !== 1 ? 'es' : ''}</span>
      </span>
      <span className="flex-1 h-px bg-gradient-to-r from-slate-200 via-slate-100 to-transparent" />
    </div>
  );
}

export function DiscussionLedgerView({
  entries,
  maxRounds,
  currentRound,
  activeRole,
}: Props) {
  const grouped = useMemo(() => groupByRound(entries), [entries]);

  if (entries.length === 0) {
    return (
      <div className="glass rounded-xl p-8 text-center">
        <p className="text-sm text-slate-500 italic">
          La deliberación aparecerá aquí turno a turno.
        </p>
        <p className="text-xs text-slate-400 mt-2">
          Hasta {maxRounds} rondas con selección de speaker en función del
          movimiento bayesiano de cada agente.
        </p>
      </div>
    );
  }

  // Render rounds in ascending order
  const roundNumbers = Array.from(grouped.keys()).sort((a, b) => a - b);

  return (
    <div className="space-y-2">
      <AnimatePresence initial={false}>
        {roundNumbers.map((round) => {
          const entriesInRound = grouped.get(round)!;
          return (
            <div key={`round-${round}`}>
              <RoundDivider
                round={round}
                count={entriesInRound.length}
                isCurrent={round === currentRound}
              />
              <div className="space-y-2">
                {entriesInRound.map((entry) => (
                  <LedgerEntryCard
                    key={`entry-${entry.turn}`}
                    entry={entry}
                    isActive={
                      activeRole === entry.role_id &&
                      round === currentRound &&
                      entry === entriesInRound[entriesInRound.length - 1]
                    }
                  />
                ))}
              </div>
            </div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
