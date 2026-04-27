/**
 * RoleAvatar — circular badge with role icon + accent ring.
 *
 * Single source of truth for role iconography; reused by the ledger,
 * the agent roster, the timeline, and the consensus gauge.
 */

import {
  BookOpen,
  Compass,
  FlaskConical,
  Gavel,
  Hammer,
  ScrollText,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { RoleId } from '../types';
import { ROLE_PALETTE } from '../types';

const ROLE_ICONS: Record<RoleId, LucideIcon> = {
  analyst:         BookOpen,
  seeker:          Compass,
  devils_advocate: Gavel,
  empiricist:      FlaskConical,
  pragmatist:      Hammer,
  synthesizer:     ScrollText,
};

interface Props {
  role: RoleId;
  size?: 'sm' | 'md' | 'lg';
  active?: boolean;
}

const SIZE_CLASSES: Record<NonNullable<Props['size']>, string> = {
  sm: 'w-7 h-7 text-[12px]',
  md: 'w-10 h-10 text-[16px]',
  lg: 'w-14 h-14 text-[20px]',
};

const ICON_PX: Record<NonNullable<Props['size']>, number> = {
  sm: 14,
  md: 20,
  lg: 28,
};

export function RoleAvatar({ role, size = 'md', active = false }: Props) {
  const palette = ROLE_PALETTE[role];
  const Icon = ROLE_ICONS[role];
  return (
    <div
      className={[
        'rounded-full flex items-center justify-center shrink-0',
        'border-2 transition-all',
        SIZE_CLASSES[size],
        active ? 'pulse-ring' : '',
      ].join(' ')}
      style={{
        background: palette.fill,
        borderColor: palette.stroke,
        color: palette.accent,
      }}
      aria-label={palette.label}
      title={palette.label}
    >
      <Icon size={ICON_PX[size]} strokeWidth={2.2} />
    </div>
  );
}
