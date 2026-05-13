/**
 * ConsensusOrb — orbe luminoso en el centro de la mesa que representa
 * visualmente el agreement_score actual.
 *
 *   - score 0   → orbe pequeño, gris (los agentes están en desacuerdo)
 *   - score 0.5 → orbe medio, amarillo (consenso parcial)
 *   - score 1.0 → orbe grande, dorado y brillante (consenso pleno)
 *
 * Es la metáfora visual del consenso: cuanto más se "encienda" el
 * centro de la mesa, más cerca está el debate de cerrarse.
 */

import { motion } from 'framer-motion';

interface Props {
  /** Último score conocido (0..1). null si aún no hay datos. */
  score: number | null;
  /** Si la deliberación está en curso */
  active: boolean;
}

/**
 * Devuelve un color que va de gris (score=0) a amarillo (0.5) a dorado (1.0).
 */
function colorForScore(score: number): string {
  if (score < 0.33) return '#cbd5e1'; // slate-300
  if (score < 0.66) return '#fbbf24'; // amber-400
  return '#f59e0b'; // amber-500 brillante
}

export function ConsensusOrb({ score, active }: Props) {
  const safeScore = score ?? 0;
  const radius = 14 + safeScore * 22; // 14..36
  const color = colorForScore(safeScore);
  const cx = 400;
  const cy = 430;

  // Si no hay datos todavía, no mostramos nada
  if (score === null) return null;

  return (
    <g>
      {/* Halo exterior animado */}
      {active && (
        <motion.circle
          cx={cx}
          cy={cy}
          r={radius + 8}
          fill={color}
          opacity={0.2}
          animate={{ r: [radius + 8, radius + 16, radius + 8], opacity: [0.2, 0.05, 0.2] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* Orbe principal */}
      <motion.circle
        cx={cx}
        cy={cy}
        animate={{ r: radius }}
        transition={{ type: 'spring', stiffness: 100, damping: 18 }}
        fill={color}
        stroke="#fef3c7"
        strokeWidth="2"
      />

      {/* Brillo interior (efecto 3D) */}
      <motion.circle
        cx={cx - 4}
        cy={cy - 6}
        animate={{ r: radius / 3 }}
        transition={{ type: 'spring', stiffness: 100, damping: 18 }}
        fill="#fef3c7"
        opacity={0.6}
      />

      {/* Etiqueta del score */}
      <text
        x="400"
        y={cy + radius + 18}
        textAnchor="middle"
        fontFamily="var(--font-mono)"
        fontSize="11"
        fontWeight="600"
        fill="#fde68a"
      >
        consenso: {(safeScore * 100).toFixed(0)}%
      </text>
    </g>
  );
}
