/**
 * TechnicalView — la vista técnica original con paneles detallados:
 *   - Tema de debate + Veredicto
 *   - Grafo del plan (DebateGraph)
 *   - Gauge de consenso
 *   - Gráfico de posiciones de los agentes
 *   - Registro de eventos colapsable
 *
 * Este componente contiene literalmente el JSX que había en el `<main>`
 * del App.tsx original, sin tocar nada. La lógica de stream y de
 * estado se ha subido a App.tsx (que la comparte entre ambas vistas).
 */

import { useEffect, useRef, useState } from 'react';
import type {
  AgentPositionsSample,
  ConsensusSnapshot,
  DebateEvent,
  SubtaskRuntime,
  TaskPlan,
} from '../types';
import { AgentPositionsChart } from '../components/AgentPositionsChart';
import { ConsensusGauge } from '../components/ConsensusGauge';
import { DebateGraph } from '../components/DebateGraph';
import { DebateTimeline } from '../components/DebateTimeline';
import { PromptInput } from '../components/PromptInput';
import { VerdictDisplay } from '../components/VerdictDisplay';

interface Props {
  status: 'idle' | 'running' | 'completed' | 'error';
  events: DebateEvent[];
  verdict: string | null;
  error: string | null;
  plan: TaskPlan | null;
  runtime: Record<string, SubtaskRuntime>;
  positions: AgentPositionsSample[];
  consensusHistory: ConsensusSnapshot[];
  disabled: boolean;
  onSubmit: (prompt: string) => void;
}

export function TechnicalView({
  status,
  events,
  verdict,
  error,
  plan,
  runtime,
  positions,
  consensusHistory,
  disabled,
  onSubmit,
}: Props) {
  const [showTimeline, setShowTimeline] = useState(true);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Auto-scroll del log de eventos (idéntico al App.tsx original)
  useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  }, [events.length]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Left panel: Input + Verdict */}
      <div className="lg:col-span-1 space-y-4 overflow-y-auto">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">
            Tema de debate
          </h2>
          <PromptInput onSubmit={onSubmit} disabled={disabled} />
        </div>
        <VerdictDisplay
          verdict={verdict}
          error={error}
          status={status}
          lastEvent={events.length > 0 ? events[events.length - 1] : null}
        />
      </div>

      {/* Right panel: DAG + gauges + timeline */}
      <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-700">
              Grafo del plan
              {plan && (
                <span className="ml-2 text-[10px] font-normal text-gray-400">
                  {plan.subtasks.length} subtareas
                </span>
              )}
            </h2>
            <span className="text-[10px] text-gray-400">
              Haz clic en un nodo para ver su salida
            </span>
          </div>
          <DebateGraph plan={plan} runtime={runtime} />
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <ConsensusGauge history={consensusHistory} />
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <AgentPositionsChart samples={positions} />
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <button
            onClick={() => setShowTimeline((v) => !v)}
            className="w-full flex items-center justify-between text-sm font-semibold text-gray-700 mb-2"
          >
            <span>
              Registro de eventos
              {events.length > 0 && (
                <span className="ml-2 text-[10px] font-normal text-gray-400">
                  {events.length} eventos
                </span>
              )}
            </span>
            <span className="text-gray-400 text-xs">
              {showTimeline ? '▾ ocultar' : '▸ mostrar'}
            </span>
          </button>
          {showTimeline && (
            <div ref={timelineRef} className="overflow-y-auto max-h-[50vh]">
              <DebateTimeline events={events} plan={plan} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
