/**
 * RoundTable - escena frontal/lateral de debate con personajes sentados.
 */

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type {
  AgentPositionsSample,
  ConsensusSnapshot,
  SubtaskRuntime,
} from '../../types';
import {
  AGENT_IDS,
  workerIdToAgentId,
  type AgentId,
} from '../../lib/round-table-config';
import { AgentCharacter, type AgentVisualState } from './AgentCharacter';
import { ThoughtBubble } from './ThoughtBubble';
import { ConsensusOrb } from './ConsensusOrb';

interface Props {
  runtime: Record<string, SubtaskRuntime>;
  positions: AgentPositionsSample[];
  consensusHistory: ConsensusSnapshot[];
  running: boolean;
  hasStarted: boolean;
}

function truncateBubble(text: string, maxChars = 220): string {
  if (text.length <= maxChars) return text;
  const cutoff = text.slice(0, maxChars).lastIndexOf(' ');
  return text.slice(0, cutoff > 0 ? cutoff : maxChars).trim() + '…';
}

export function RoundTable({
  runtime,
  positions: _positions,
  consensusHistory,
  running,
  hasStarted,
}: Props) {
  /**
   * Calculamos el estado visual de cada agente y el texto de su bocadillo
   * a partir del runtime (subtasks).
   */
  const { visualStates, bubbleTexts } = useMemo(() => {
    const visualStates: Record<AgentId, AgentVisualState> = {
      ae1: 'idle',
      ae2: 'idle',
      ae3: 'idle',
    };
    const bubbleTexts: Partial<Record<AgentId, string>> = {};

    const subtasksByAgent: Record<AgentId, SubtaskRuntime[]> = {
      ae1: [],
      ae2: [],
      ae3: [],
    };

    for (const subtask of Object.values(runtime)) {
      const agentId = workerIdToAgentId(subtask.worker_id);
      if (agentId) subtasksByAgent[agentId].push(subtask);
    }

    for (const agentId of AGENT_IDS) {
      const tasks = subtasksByAgent[agentId];
      if (tasks.length === 0) continue;

      const hasFailed = tasks.some((t) => t.status === 'failed');
      const hasRunning = tasks.some((t) => t.status === 'running');
      const doneTasks = tasks.filter((t) => t.status === 'done');

      if (hasFailed) {
        visualStates[agentId] = 'failed';
        const failed = tasks.find((t) => t.status === 'failed');
        if (failed?.error) bubbleTexts[agentId] = `Error: ${failed.error}`;
      } else if (hasRunning) {
        visualStates[agentId] = 'thinking';
      } else if (doneTasks.length > 0) {
        visualStates[agentId] = running ? 'speaking' : 'consensus';
        const last = doneTasks[doneTasks.length - 1];
        if (last.output) bubbleTexts[agentId] = truncateBubble(last.output);
      }
    }

    if (hasStarted && running && Object.keys(runtime).length === 0) {
      for (const agentId of AGENT_IDS) {
        visualStates[agentId] = 'thinking';
        bubbleTexts[agentId] = 'Estoy revisando el tema antes de tomar postura.';
      }
    }

    if (hasStarted && !running) {
      for (const agentId of AGENT_IDS) {
        if (visualStates[agentId] !== 'failed') visualStates[agentId] = 'consensus';
        if (!bubbleTexts[agentId]) bubbleTexts[agentId] = 'Me sumo a la conclusión final.';
      }
    }

    return { visualStates, bubbleTexts };
  }, [hasStarted, runtime, running]);

  const latestScore =
    consensusHistory.length > 0
      ? consensusHistory[consensusHistory.length - 1].agreement_score
      : null;

  return (
    <svg
      viewBox="0 0 800 640"
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ display: 'block', maxHeight: '78vh' }}
      role="img"
      aria-label="Mesa de debate frontal con tres personajes sentados"
    >
      <defs>
        <linearGradient id="tableTopGradient" x1="0%" x2="100%" y1="0%" y2="20%">
          <stop offset="0%" stopColor="#172033" />
          <stop offset="50%" stopColor="#334155" />
          <stop offset="100%" stopColor="#111827" />
        </linearGradient>
        <linearGradient id="tableFrontGradient" x1="0%" x2="0%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#1f2937" />
          <stop offset="100%" stopColor="#07111f" />
        </linearGradient>
        <radialGradient id="floorGradient" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="#122036" />
          <stop offset="55%" stopColor="#08111f" />
          <stop offset="100%" stopColor="#020617" />
        </radialGradient>
        <filter id="tableShadow2" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="8" />
          <feOffset dx="0" dy="6" result="offsetblur" />
          <feComponentTransfer>
            <feFuncA type="linear" slope="0.35" />
          </feComponentTransfer>
          <feMerge>
            <feMergeNode />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect x="0" y="0" width="800" height="640" fill="url(#floorGradient)" />
      <g opacity="0.22">
        {Array.from({ length: 12 }).map((_, i) => (
          <line
            key={`grid-h-${i}`}
            x1="0"
            x2="800"
            y1={70 + i * 44}
            y2={70 + i * 44}
            stroke="#67e8f9"
            strokeWidth="1"
          />
        ))}
        {Array.from({ length: 14 }).map((_, i) => (
          <line
            key={`grid-v-${i}`}
            x1={40 + i * 55}
            x2={40 + i * 55}
            y1="0"
            y2="640"
            stroke="#67e8f9"
            strokeWidth="1"
          />
        ))}
      </g>

      <path d="M 70 600 C 185 515, 620 515, 730 600 Z" fill="#020617" opacity="0.55" />

      {/* Personajes detras de la mesa, con sillas visibles. */}
      {(['ae3', 'ae1', 'ae2'] as AgentId[]).map((agentId) => (
        <AgentCharacter
          key={agentId}
          agentId={agentId}
          state={visualStates[agentId]}
        />
      ))}

      {/* Mesa frontal: el tablero y el faldon tapan parcialmente los cuerpos. */}
      <g filter="url(#tableShadow2)">
        <path
          d="M 112 392 C 216 342, 584 342, 688 392 L 642 478 C 535 514, 263 514, 158 478 Z"
          fill="url(#tableTopGradient)"
          stroke="#67e8f9"
          strokeWidth="2.5"
        />
        <path
          d="M 158 478 C 260 518, 538 518, 642 478 L 610 574 C 505 612, 297 612, 190 574 Z"
          fill="url(#tableFrontGradient)"
          stroke="#67e8f9"
          strokeWidth="2"
        />
        <path d="M 160 478 C 265 505, 535 505, 640 478" stroke="#a78bfa" strokeWidth="1.5" opacity="0.45" fill="none" />
        <path d="M 245 424 C 335 404, 465 404, 555 424" stroke="#22d3ee" strokeWidth="1" opacity="0.4" fill="none" />
        <rect x="280" y="407" width="92" height="48" rx="5" fill="#0f172a" stroke="#475569" strokeWidth="2" />
        <rect x="286" y="413" width="80" height="32" rx="3" fill="#172554" />
        <rect x="430" y="408" width="96" height="46" rx="5" fill="#0f172a" stroke="#475569" strokeWidth="2" />
        <rect x="438" y="416" width="32" height="4" rx="2" fill="#10b981" opacity="0.8" />
        <rect x="438" y="426" width="70" height="3" rx="1.5" fill="#94a3b8" opacity="0.55" />
        <rect x="438" y="435" width="56" height="3" rx="1.5" fill="#94a3b8" opacity="0.55" />
        <rect x="375" y="382" width="54" height="38" rx="4" fill="#fef3c7" stroke="#92400e" strokeWidth="1.5" />
        <line x1="386" y1="391" x2="418" y2="391" stroke="#3b82f6" strokeWidth="1" />
        <line x1="386" y1="400" x2="414" y2="400" stroke="#3b82f6" strokeWidth="1" />
        <g transform="translate(566 432)">
          <ellipse cx="16" cy="4" rx="7" ry="10" fill="none" stroke="#78350f" strokeWidth="2" />
          <rect x="-10" y="-8" width="28" height="28" rx="6" fill="#f8fafc" stroke="#1e293b" strokeWidth="2" />
          <ellipse cx="4" cy="-5" rx="11" ry="5" fill="#5d3a1a" />
        </g>
        <ConsensusOrb score={latestScore} active={running} />
      </g>

      {/* Bocadillos por encima, calculados para quedar dentro del lienzo. */}
      {AGENT_IDS.map((agentId) => (
        <ThoughtBubble
          key={`bubble-${agentId}`}
          agentId={agentId}
          text={bubbleTexts[agentId] ?? ''}
          visible={!!bubbleTexts[agentId]}
        />
      ))}

      {/* Mensaje invitando si no se ha empezado */}
      {!hasStarted && (
        <motion.text
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.65 }}
          transition={{ delay: 0.5 }}
          x="400"
          y="50"
          textAnchor="middle"
          fontFamily="var(--font-serif)"
          fontStyle="italic"
          fontSize="20"
          fill="#94a3b8"
        >
          Introduce un tema para que comience el debate…
        </motion.text>
      )}

      {/* Estilos para el typewriter */}
      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </svg>
  );
}
