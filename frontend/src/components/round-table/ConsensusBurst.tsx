/**
 * ConsensusBurst — animación de cierre cuando termina el debate.
 *
 * Si hay consenso (score >= 0.5 o no hay datos): cartel dorado + confeti.
 * Si NO hay consenso (score < 0.5): cartel sobrio explicándolo, sin confeti.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Award, Handshake } from 'lucide-react';

interface Props {
  visible: boolean;
  finalScore?: number | null;
  roundsUsed?: number;
}

const CONFETTI_COLORS = [
  '#6366f1', '#a855f7', '#10b981', '#f59e0b',
  '#ef4444', '#06b6d4', '#ec4899', '#2563eb',
];

const CONFETTI_COUNT = 70;

interface ConfettiPiece {
  id: number;
  endX: number;
  endY: number;
  rotateEnd: number;
  color: string;
  delay: number;
  size: number;
  shape: 'rect' | 'circle';
}

function generateConfetti(): ConfettiPiece[] {
  return Array.from({ length: CONFETTI_COUNT }, (_, i) => ({
    id: i,
    endX: (Math.random() - 0.5) * 700,
    endY: 200 + Math.random() * 350,
    rotateEnd: (Math.random() - 0.5) * 720,
    color: CONFETTI_COLORS[Math.floor(Math.random() * CONFETTI_COLORS.length)],
    delay: Math.random() * 0.4,
    size: 6 + Math.random() * 9,
    shape: Math.random() > 0.5 ? 'rect' : 'circle',
  }));
}

export function ConsensusBurst({ visible, finalScore, roundsUsed }: Props) {
  // Si finalScore es null/undefined o >= 0.5 → éxito (consenso suficiente).
  // Si es < 0.5 → "no consensus".
  const isSuccess =
    finalScore === null || finalScore === undefined || finalScore >= 0.5;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="absolute inset-0 pointer-events-none flex items-center justify-center overflow-hidden"
        >
          {/* Flash dorado solo en éxito */}
          {isSuccess && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.3, 0] }}
              transition={{ duration: 1.2 }}
              className="absolute inset-0"
              style={{
                background:
                  'radial-gradient(circle at center, rgba(251, 191, 36, 0.6), transparent 60%)',
              }}
            />
          )}

          {/* Confeti */}
          {isSuccess && (
            <div className="absolute inset-0 flex items-center justify-center">
              {generateConfetti().map((piece) => (
                <motion.div
                  key={piece.id}
                  initial={{ x: 0, y: 0, opacity: 1, rotate: 0, scale: 0 }}
                  animate={{
                    x: piece.endX,
                    y: piece.endY,
                    opacity: [1, 1, 0],
                    rotate: piece.rotateEnd,
                    scale: 1,
                  }}
                  transition={{
                    duration: 2.4,
                    delay: piece.delay,
                    ease: [0.2, 0.8, 0.4, 1],
                    times: [0, 0.7, 1],
                  }}
                  className="absolute"
                  style={{
                    width: piece.size,
                    height: piece.shape === 'rect' ? piece.size * 0.4 : piece.size,
                    background: piece.color,
                    borderRadius: piece.shape === 'circle' ? '50%' : '2px',
                  }}
                />
              ))}
            </div>
          )}

          {/* Cartel central */}
          <motion.div
            initial={{ scale: 0, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 16, delay: 0.2 }}
            className={`relative rounded-2xl px-7 py-5 shadow-2xl border-4 max-w-md ${
              isSuccess
                ? 'bg-gradient-to-br from-amber-400 to-amber-500 border-amber-300 text-white'
                : 'bg-gradient-to-br from-slate-600 to-slate-700 border-slate-400 text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              <motion.div
                animate={
                  isSuccess
                    ? { rotate: [0, -15, 15, 0], scale: [1, 1.2, 1] }
                    : {}
                }
                transition={{ duration: 0.8, repeat: 2 }}
              >
                {isSuccess ? <Award size={40} /> : <Handshake size={36} />}
              </motion.div>
              <div>
                <div className="font-bold text-xl leading-tight font-serif italic">
                  {isSuccess
                    ? '¡Consenso alcanzado!'
                    : 'Sin consenso pleno'}
                </div>
                <div className="text-xs opacity-90 leading-tight mt-1">
                  {isSuccess
                    ? roundsUsed
                      ? `Tras ${roundsUsed} ronda${roundsUsed > 1 ? 's' : ''} de debate`
                      : 'Los agentes han llegado a un acuerdo'
                    : 'Los agentes mantienen posturas distintas'}
                </div>
                <div className="text-[10px] opacity-75 leading-tight mt-1.5 italic flex items-center gap-1">
                  <Sparkles size={11} />
                  Lee el veredicto en el panel izquierdo
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
