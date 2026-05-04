/**
 * RoundIndicator — indicador discreto de la ronda actual.
 */

import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  currentRound: number;
  maxRounds: number;
  active: boolean;
}

export function RoundIndicator({ currentRound, maxRounds, active }: Props) {
  if (!active || currentRound === 0) return null;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={currentRound}
        initial={{ opacity: 0, y: -10, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -10, scale: 0.9 }}
        transition={{ type: 'spring', stiffness: 250, damping: 22 }}
        className="rounded-full border border-cyan-200/15 bg-slate-950/70 px-4 py-1.5 inline-flex items-center gap-2 backdrop-blur"
      >
        <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-100/55">
          Ronda
        </span>
        <span className="text-sm font-bold text-cyan-200">{currentRound}</span>
        <span className="text-xs text-slate-400">de {maxRounds}</span>
        <div className="flex items-center gap-0.5 ml-1">
          {Array.from({ length: maxRounds }).map((_, i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 rounded-full transition-colors ${
                i < currentRound ? 'bg-cyan-300' : 'bg-slate-700'
              }`}
            />
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
