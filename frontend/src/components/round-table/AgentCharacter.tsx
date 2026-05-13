/**
 * AgentCharacter - personaje cartoon sentado, dibujado de frente.
 */

import { motion } from 'framer-motion';
import { Check, AlertCircle } from 'lucide-react';
import { AGENT_VISUAL, type AgentId } from '../../lib/round-table-config';

export type AgentVisualState = 'idle' | 'thinking' | 'speaking' | 'consensus' | 'failed';

interface Props {
  agentId: AgentId;
  state: AgentVisualState;
}

export function AgentCharacter({ agentId, state }: Props) {
  const config = AGENT_VISUAL[agentId];
  const isSpeaking = state === 'speaking';
  const isThinking = state === 'thinking';
  const isConsensus = state === 'consensus';
  const isFailed = state === 'failed';
  const skin = agentId === 'ae2' ? '#f5d8b8' : agentId === 'ae3' ? '#fbe2c5' : '#fde7d3';
  const hair = agentId === 'ae1' ? '#2f2118' : agentId === 'ae2' ? '#c2410c' : '#f59e0b';
  const chairWidth = agentId === 'ae3' ? 120 : 132;
  const chairHeight = agentId === 'ae3' ? 150 : 160;
  const scale = agentId === 'ae3' ? 0.9 : 0.94;
  const headY = -96;

  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {/* Pulse-ring para el agente activo */}
      {(isSpeaking || isConsensus) && (
        <motion.ellipse
          cx={config.x}
          cy={config.y - 52}
          rx="78"
          ry="96"
          fill="none"
          stroke={isConsensus ? '#fbbf24' : config.accent}
          strokeWidth="3"
          initial={{ opacity: 0.5, scale: 0.9 }}
          animate={{ opacity: 0, scale: 1.18 }}
          transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
        />
      )}

      <g transform={`translate(${config.x}, ${config.y}) scale(${scale})`}>
        <motion.g
          animate={
            isSpeaking
              ? { y: [0, -2, 0] }
              : isThinking
                ? { rotate: [0, -1, 1, 0] }
                : { y: 0, rotate: 0 }
          }
          transition={
            isSpeaking
              ? { duration: 0.6, repeat: Infinity, ease: 'easeInOut' }
              : isThinking
                ? { duration: 2.5, repeat: Infinity, ease: 'easeInOut' }
                : { duration: 0.3 }
          }
        >
          <g transform={`rotate(${config.rotation})`}>
          {/* Silla con respaldo visible. */}
          <rect
            x={-chairWidth / 2}
            y="-82"
            width={chairWidth}
            height={chairHeight}
            rx="22"
            fill="#172033"
            stroke="#67e8f9"
            strokeWidth="2"
            opacity="0.95"
          />
          <rect
            x={-chairWidth / 2 + 12}
            y="-68"
            width={chairWidth - 24}
            height={chairHeight - 22}
            rx="18"
            fill="#26364f"
            opacity="0.82"
          />

          {/* Piernas y postura sentada, visibles bajo el torso hasta que la mesa los tapa. */}
          <path d="M -36 52 Q -54 92, -42 124 L -12 124 Q -12 88, -4 54 Z" fill="#1e293b" />
          <path d="M 36 52 Q 54 92, 42 124 L 12 124 Q 12 88, 4 54 Z" fill="#1e293b" />

          {/* Torso. */}
          <path
            d="M -54 4 Q -42 -38, 0 -40 Q 42 -38, 54 4 L 44 78 Q 0 98, -44 78 Z"
            fill={config.accent}
            stroke={config.secondary}
            strokeWidth="3"
          />
          <path d="M -14 -38 L 0 -20 L 14 -38" fill={skin} stroke="#5a3a2a" strokeWidth="1.5" />

          {/* Brazos apoyados hacia la mesa. */}
          <path
            d="M -48 18 Q -76 42, -76 76 Q -64 84, -52 76 Q -50 52, -30 34 Z"
            fill={config.accent}
            stroke={config.secondary}
            strokeWidth="2.5"
          />
          <path
            d="M 48 18 Q 76 42, 76 76 Q 64 84, 52 76 Q 50 52, 30 34 Z"
            fill={config.accent}
            stroke={config.secondary}
            strokeWidth="2.5"
          />
          <ellipse cx="-70" cy="80" rx="13" ry="10" fill={skin} stroke="#5a3a2a" strokeWidth="1.5" />
          <ellipse cx="70" cy="80" rx="13" ry="10" fill={skin} stroke="#5a3a2a" strokeWidth="1.5" />

          {/* Cuello, cabeza y rasgos. */}
          <rect x="-14" y="-66" width="28" height="28" rx="12" fill={skin} stroke="#5a3a2a" strokeWidth="1.5" />
          <circle cx="0" cy={headY} r="44" fill={skin} stroke="#5a3a2a" strokeWidth="2.5" />
          {agentId === 'ae1' && (
            <>
              <path d="M -39 -102 Q -32 -145, 4 -142 Q 42 -138, 40 -98 Q 25 -116, 4 -112 Q -18 -121, -39 -102 Z" fill={hair} />
              <path d="M -18 -135 Q -4 -147, 14 -134 Q 2 -125, -18 -135 Z" fill="#15100b" />
            </>
          )}
          {agentId === 'ae2' && (
            <>
              <ellipse cx="48" cy="-90" rx="16" ry="30" fill={hair} stroke="#7c2d12" strokeWidth="1.5" />
              <path d="M -42 -98 Q -42 -142, 2 -144 Q 42 -140, 43 -96 Q 24 -118, 0 -114 Q -22 -120, -42 -98 Z" fill={hair} />
              <path d="M -18 -132 Q -4 -140, 12 -132" stroke="#ea580c" strokeWidth="3" fill="none" strokeLinecap="round" />
            </>
          )}
          {agentId === 'ae3' && (
            <>
              <circle cx="0" cy="-145" r="17" fill={hair} stroke="#92400e" strokeWidth="2" />
              <path d="M -40 -100 Q -36 -138, 0 -139 Q 36 -138, 40 -100 Q 22 -114, 0 -111 Q -22 -114, -40 -100 Z" fill="#fbbf24" />
              <circle cx="-14" cy="-96" r="8" fill="white" stroke="#1e293b" strokeWidth="2" opacity="0.9" />
              <circle cx="14" cy="-96" r="8" fill="white" stroke="#1e293b" strokeWidth="2" opacity="0.9" />
              <line x1="-6" y1="-96" x2="6" y2="-96" stroke="#1e293b" strokeWidth="2" />
            </>
          )}
          <ellipse cx="-15" cy="-96" rx="4" ry="5" fill="#1e293b" />
          <ellipse cx="15" cy="-96" rx="4" ry="5" fill="#1e293b" />
          <path d="M -12 -76 Q 0 -68, 12 -76" stroke="#7f1d1d" strokeWidth="2.2" fill="none" strokeLinecap="round" />
          <circle cx="-29" cy="-82" r="5" fill="#ec4899" opacity="0.26" />
          <circle cx="29" cy="-82" r="5" fill="#ec4899" opacity="0.26" />
          </g>
        </motion.g>
      </g>

      {/* Puntitos de "pensando..." sobre la cabeza */}
      {isThinking && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {[0, 1, 2].map((i) => {
            const dotY = config.y - 145;
            return (
              <motion.circle
                key={i}
                cx={config.x - 12 + i * 12}
                cy={dotY}
                r="4"
                fill={config.accent}
                animate={{ y: [0, -8, 0], opacity: [0.4, 1, 0.4] }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  delay: i * 0.18,
                  ease: 'easeInOut',
                }}
              />
            );
          })}
        </motion.g>
      )}

      {/* Tic dorado de consenso */}
      {isConsensus && (
        <motion.g
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 15 }}
        >
          <circle
            cx={config.x + 64}
            cy={config.y - 132}
            r="11"
            fill="#f59e0b"
            stroke="white"
            strokeWidth="2.5"
          />
          <foreignObject
            x={config.x + 57}
            y={config.y - 139}
            width="14"
            height="14"
          >
            <Check size={14} color="white" strokeWidth={3} />
          </foreignObject>
        </motion.g>
      )}

      {/* Símbolo de error */}
      {isFailed && (
        <motion.g
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 15 }}
        >
          <circle
            cx={config.x + 64}
            cy={config.y - 132}
            r="11"
            fill="#ef4444"
            stroke="white"
            strokeWidth="2.5"
          />
          <foreignObject
            x={config.x + 57}
            y={config.y - 139}
            width="14"
            height="14"
          >
            <AlertCircle size={14} color="white" strokeWidth={2.5} />
          </foreignObject>
        </motion.g>
      )}

      {/* Etiqueta SOLO con el nombre del agente (sin info técnica) */}
      <foreignObject
        x={config.x - 50}
        y={config.y + 88}
        width="100"
        height="28"
        style={{ overflow: 'visible' }}
      >
        <div style={{ textAlign: 'center', pointerEvents: 'none' }}>
          <div
            style={{
              display: 'inline-block',
              fontSize: '14px',
              fontWeight: 700,
              color: 'white',
              backgroundColor: config.accent,
              padding: '3px 12px',
              borderRadius: '12px',
              boxShadow: '0 2px 4px rgba(15,23,42,0.15)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {config.label}
          </div>
        </div>
      </foreignObject>
    </motion.g>
  );
}
