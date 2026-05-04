/**
 * Configuracion visual para la escena frontal de debate.
 *
 * Coordenadas relativas a un viewBox SVG de 800x640. Los agentes estan
 * sentados detras de una mesa en perspectiva frontal/lateral.
 */

export type AgentId = 'ae1' | 'ae2' | 'ae3';

export const AGENT_IDS: readonly AgentId[] = ['ae1', 'ae2', 'ae3'];

export interface AgentVisualConfig {
  label: string;
  /** Color principal del personaje (jersey/blusa) */
  accent: string;
  /** Color secundario (pelo o detalle) */
  secondary: string;
  /** Posición X del centro del personaje */
  x: number;
  /** Posición Y del centro del personaje */
  y: number;
  /** Lado donde aparece el bocadillo */
  bubbleSide: 'left' | 'right' | 'top';
  /** Inclinacion leve del torso/cabeza en grados */
  rotation: number;
}

/**
 * 3 asientos visibles como reunion frontal: AE3 al centro/fondo,
 * AE1 y AE2 a los lados, todos mirando hacia la conversacion.
 */
export const AGENT_VISUAL: Record<AgentId, AgentVisualConfig> = {
  ae3: {
    label: 'AE3',
    accent: '#c026d3',
    secondary: '#7e22ce',
    x: 400,
    y: 328,
    bubbleSide: 'top',
    rotation: 0,
  },
  ae1: {
    label: 'AE1',
    accent: '#2563eb',
    secondary: '#1e40af',
    x: 232,
    y: 374,
    bubbleSide: 'left',
    rotation: -4,
  },
  ae2: {
    label: 'AE2',
    accent: '#059669',
    secondary: '#047857',
    x: 568,
    y: 374,
    bubbleSide: 'right',
    rotation: 4,
  },
};

/**
 * Mapea worker_id de eventos a AgentId. Tolerante con distintos formatos.
 */
export function workerIdToAgentId(workerId: string | undefined): AgentId | null {
  if (!workerId) return null;
  const n = workerId.toLowerCase();
  if (n.includes('ae1') || n.includes('agent-1') || n === '1') return 'ae1';
  if (n.includes('ae2') || n.includes('agent-2') || n === '2') return 'ae2';
  if (n.includes('ae3') || n.includes('agent-3') || n === '3') return 'ae3';
  return null;
}
