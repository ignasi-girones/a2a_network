/**
 * AgentRosterPanel — grid of cards showing the panel composition.
 *
 * Each card shows a role's avatar, label, model, tool whitelist, and a
 * live "talking…/idle/done" status derived from the deliberation state.
 * The active agent gets a pulsing ring.
 */

import { Check } from 'lucide-react';
import type { DeliberationState, RoleId } from '../types';
import { CANONICAL_ROLES, ROLE_PALETTE } from '../types';
import { RoleAvatar } from './RoleAvatar';

interface Props {
  state: DeliberationState;
}

interface RoleSpec {
  role: RoleId;
  model: string;
  tools: string[];
  hint: string;
}

const ROLE_SPECS: Record<RoleId, RoleSpec> = {
  analyst: {
    role: 'analyst',
    model: 'Mistral',
    tools: ['wikipedia', 'calculator'],
    hint: 'Baseline factual, sin tomar partido',
  },
  seeker: {
    role: 'seeker',
    model: 'Groq',
    tools: ['web_search', 'arxiv'],
    hint: 'Evidencia externa para huecos del análisis',
  },
  devils_advocate: {
    role: 'devils_advocate',
    model: 'Cerebras',
    tools: ['web_search'],
    hint: 'Estratagema de Schopenhauer · ataca la tesis',
  },
  empiricist: {
    role: 'empiricist',
    model: 'Groq',
    tools: ['arxiv', 'calculator'],
    hint: 'Falsacionismo popperiano',
  },
  pragmatist: {
    role: 'pragmatist',
    model: 'Mistral',
    tools: ['web_search', 'wikipedia'],
    hint: 'Casos reales documentados',
  },
  synthesizer: {
    role: 'synthesizer',
    model: 'Mistral',
    tools: [],
    hint: 'Validez habermasiana sobre el panel',
  },
};

function turnsForRole(state: DeliberationState, role: RoleId): number {
  return state.ledger.filter((e) => e.role_id === role).length;
}

function lastBeliefForRole(
  state: DeliberationState,
  role: RoleId,
): number | null {
  const entries = state.ledger.filter((e) => e.role_id === role);
  if (entries.length === 0) return null;
  const last = entries[entries.length - 1];
  return last.belief_after;
}

function AgentCard({ spec, state }: { spec: RoleSpec; state: DeliberationState }) {
  const palette = ROLE_PALETTE[spec.role];
  const turns = turnsForRole(state, spec.role);
  const isActive = state.active_role === spec.role;
  const hasParticipated = turns > 0;
  const lastBelief = lastBeliefForRole(state, spec.role);

  return (
    <div
      className={[
        'glass rounded-xl p-3 transition-all',
        isActive ? 'ring-2 ring-offset-2' : '',
      ].join(' ')}
      style={
        isActive
          ? ({
              '--tw-ring-color': palette.accent,
              borderColor: palette.accent,
            } as React.CSSProperties)
          : { borderLeftWidth: '3px', borderLeftColor: palette.accent }
      }
    >
      <div className="flex items-start gap-3">
        <RoleAvatar role={spec.role} size="md" active={isActive} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span
              className="font-semibold text-sm leading-tight"
              style={{ color: palette.accent }}
            >
              {palette.label}
            </span>
            {hasParticipated && !isActive && (
              <Check
                size={11}
                className="text-emerald-600"
                aria-label="participó"
              />
            )}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">{spec.hint}</div>

          <div className="flex items-center gap-1 mt-1.5 flex-wrap">
            <span className="text-[9px] font-mono text-slate-400 uppercase tracking-wider">
              {spec.model}
            </span>
            {spec.tools.length > 0 && (
              <span className="text-[9px] text-slate-300 mx-0.5">·</span>
            )}
            {spec.tools.map((t) => (
              <span
                key={t}
                className="text-[9px] font-mono text-slate-500 bg-slate-100 px-1 py-0.5 rounded"
              >
                {t}
              </span>
            ))}
          </div>

          <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100">
            <span className="text-[9px] text-slate-400">
              {turns === 0
                ? 'sin turnos aún'
                : `${turns} turno${turns > 1 ? 's' : ''}`}
            </span>
            {lastBelief !== null && (
              <span
                className="text-[10px] font-mono"
                style={{ color: palette.accent }}
              >
                {lastBelief >= 0 ? '+' : ''}
                {lastBelief.toFixed(2)}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AgentRosterPanel({ state }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {CANONICAL_ROLES.map((role) => (
        <AgentCard key={role} spec={ROLE_SPECS[role]} state={state} />
      ))}
    </div>
  );
}
