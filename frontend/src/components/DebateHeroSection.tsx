/**
 * DebateHeroSection — top-of-page banner with the claim, round progress,
 * and the currently-speaking agent.
 *
 * Acts as the visual anchor for the deliberation: large serif typography
 * for the claim, a circular SVG progress ring for the round counter, and
 * a live-pulsing avatar of whoever is generating right now.
 */

import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import type { DeliberationState, RoleId } from '../types';
import { ROLE_PALETTE } from '../types';
import { RoleAvatar } from './RoleAvatar';

interface Props {
  state: DeliberationState;
  status: 'idle' | 'running' | 'completed' | 'error';
}

function RoundProgressRing({
  current,
  total,
}: {
  current: number;
  total: number;
}) {
  const size = 64;
  const strokeWidth = 5;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(1, current / total);
  const offset = circumference * (1 - progress);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="url(#progressGradient)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono font-semibold text-base text-slate-800">
          {current}
        </span>
        <span className="text-[9px] text-slate-400 -mt-1">de {total}</span>
      </div>
    </div>
  );
}

function ActiveSpeakerBadge({ role }: { role: RoleId }) {
  const palette = ROLE_PALETTE[role];
  return (
    <motion.div
      key={role}
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0 }}
      className="flex items-center gap-2.5 px-3 py-1.5 rounded-full glass"
      style={{ borderColor: palette.accent + '55' }}
    >
      <RoleAvatar role={role} size="sm" active />
      <div className="flex flex-col">
        <span className="text-[9px] uppercase tracking-wider text-slate-400 leading-tight">
          generando…
        </span>
        <span
          className="text-xs font-medium leading-tight"
          style={{ color: palette.accent }}
        >
          {palette.label}
        </span>
      </div>
    </motion.div>
  );
}

function StatusPill({
  status,
  reason,
}: {
  status: Props['status'];
  reason: string | null;
}) {
  let label = '';
  let cls = '';
  if (status === 'idle') {
    label = 'En espera';
    cls = 'bg-slate-100 text-slate-600';
  } else if (status === 'running') {
    label = 'Deliberando';
    cls = 'bg-blue-50 text-blue-700 border border-blue-200';
  } else if (status === 'completed') {
    if (reason === 'consensus') {
      label = '✓ Consenso';
      cls = 'bg-emerald-50 text-emerald-700 border border-emerald-200';
    } else if (reason === 'aporia_unhandled') {
      label = '⚠ Aporía no resuelta';
      cls = 'bg-amber-50 text-amber-700 border border-amber-200';
    } else {
      label = 'Concluida';
      cls = 'bg-slate-50 text-slate-700 border border-slate-200';
    }
  } else {
    label = 'Error';
    cls = 'bg-rose-50 text-rose-700 border border-rose-200';
  }
  return (
    <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${cls}`}>
      {label}
    </span>
  );
}

export function DebateHeroSection({ state, status }: Props) {
  const claim = state.claim;
  const total = state.max_rounds || 3;
  const current = Math.max(state.current_round, status === 'idle' ? 0 : 1);

  return (
    <section className="glass-strong rounded-2xl p-6 md:p-8 mb-6">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-slate-400">
          <Sparkles size={12} />
          <span>Tesis bajo deliberación</span>
        </div>
        <StatusPill status={status} reason={state.terminated_reason} />
      </div>

      <h2 className="font-serif italic text-2xl md:text-3xl leading-snug text-slate-800 mb-6 min-h-[2.5em]">
        {claim ? (
          <span>"{claim}"</span>
        ) : (
          <span className="text-slate-400">
            Esperando un tema de debate…
          </span>
        )}
      </h2>

      <div className="flex items-center gap-4 md:gap-6 flex-wrap">
        <div className="flex items-center gap-3">
          <RoundProgressRing current={current} total={total} />
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-400 leading-tight">
              Ronda
            </div>
            <div className="font-mono font-semibold text-slate-700 leading-tight">
              {state.current_round || '—'} / {total}
            </div>
            {state.terminated_reason && (
              <div className="text-[10px] text-slate-400 mt-0.5">
                {state.terminated_reason}
              </div>
            )}
          </div>
        </div>

        <div className="h-12 w-px bg-slate-200" />

        <div className="flex-1 min-w-0">
          {state.active_role ? (
            <ActiveSpeakerBadge role={state.active_role} />
          ) : (
            <div className="text-xs text-slate-400 italic">
              {status === 'running'
                ? 'Sintetizando o seleccionando siguiente speaker…'
                : 'Sin agente activo'}
            </div>
          )}
        </div>

        <div className="text-right">
          <div className="text-[10px] uppercase tracking-wider text-slate-400 leading-tight">
            Intervenciones
          </div>
          <div className="font-mono font-semibold text-slate-700">
            {state.ledger.length}
          </div>
        </div>
      </div>
    </section>
  );
}
