import type { DebateEvent } from '../types';
import { Markdown } from './Markdown';

interface Props {
  events: DebateEvent[];
}

function getEventStyle(event: DebateEvent) {
  const stage = event.stage;
  const agent = event.data?.agent;

  if (stage === 'tool_use') {
    if (agent === 'ae1') return { align: 'self-start', bg: 'bg-violet-50 border-violet-200', label: 'AE1 · Tool', labelColor: 'bg-violet-500' };
    if (agent === 'ae2') return { align: 'self-end', bg: 'bg-violet-50 border-violet-200', label: 'AE2 · Tool', labelColor: 'bg-violet-500' };
    return { align: 'self-center', bg: 'bg-violet-50 border-violet-200', label: 'Tool', labelColor: 'bg-violet-500' };
  }
  if (agent === 'ae1') return { align: 'self-start', bg: 'bg-blue-50 border-blue-200', label: 'AE1', labelColor: 'bg-blue-600' };
  if (agent === 'ae2') return { align: 'self-end', bg: 'bg-emerald-50 border-emerald-200', label: 'AE2', labelColor: 'bg-emerald-600' };
  if (stage === 'consensus' || stage === 'complete') return { align: 'self-center', bg: 'bg-amber-50 border-amber-200', label: 'System', labelColor: 'bg-amber-600' };

  return { align: 'self-center', bg: 'bg-gray-50 border-gray-200', label: 'Orchestrator', labelColor: 'bg-gray-600' };
}

function isDetailEvent(event: DebateEvent): boolean {
  return !!(event.data?.text);
}

export function DebateTimeline({ events }: Props) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        <p>El debate aparecera aqui en tiempo real...</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 overflow-y-auto max-h-[calc(100vh-200px)] p-1">
      {events.map((event, i) => {
        const style = getEventStyle(event);
        const hasDetail = isDetailEvent(event);

        return (
          <div key={i} className={`flex flex-col ${style.align} max-w-[85%]`}>
            <div className={`rounded-lg border p-3 ${style.bg} text-sm`}>
              {/* Header badge */}
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[10px] font-bold text-white px-1.5 py-0.5 rounded ${style.labelColor}`}>
                  {style.label}
                </span>
                {event.data?.round && (
                  <span className="text-[10px] text-gray-500">
                    Ronda {event.data.round}
                  </span>
                )}
              </div>

              {/* Message */}
              <p className="text-gray-700 text-xs font-medium">{event.message}</p>

              {/* Detail text (collapsible feel) */}
              {hasDetail && (
                <details className="mt-2">
                  <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-700">
                    Ver argumento completo
                  </summary>
                  <div className="mt-1 text-gray-600">
                    <Markdown>{event.data!.text!}</Markdown>
                  </div>
                </details>
              )}

              {/* Tool use info */}
              {event.stage === 'tool_use' && event.data?.tool && (
                <div className="mt-2 flex flex-col gap-1">
                  <div className="flex items-center gap-1.5 text-[10px] text-violet-700">
                    <span className="font-mono bg-violet-100 px-1.5 py-0.5 rounded">
                      {event.data.tool}()
                    </span>
                  </div>
                  {event.data.query && (
                    <p className="text-[10px] text-gray-500 font-mono truncate">
                      &ldquo;{event.data.query}&rdquo;
                    </p>
                  )}
                </div>
              )}

              {/* Roles info */}
              {event.data?.ae1_role && (
                <div className="mt-2 flex gap-2 text-[10px]">
                  <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">
                    AE1: {event.data.ae1_role}
                  </span>
                  <span className="bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded">
                    AE2: {event.data.ae2_role}
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
