import type { DebateEvent } from '../types';
import { Markdown } from './Markdown';

interface Props {
  verdict: string | null;
  error: string | null;
  status: 'idle' | 'running' | 'completed' | 'error';
  lastEvent?: DebateEvent | null;
}

export function VerdictDisplay({ verdict, error, status, lastEvent }: Props) {
  if (status === 'idle') return null;

  if (status === 'running') {
    return (
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center gap-2">
          <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full" />
          <div>
            <p className="text-sm text-blue-700 font-medium">Debate en curso...</p>
            {lastEvent && (
              <p className="text-xs text-blue-500 mt-1">{lastEvent.message}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
        <h3 className="text-sm font-bold text-red-700 mb-1">Error</h3>
        <p className="text-xs text-red-600">{error}</p>
      </div>
    );
  }

  if (verdict) {
    return (
      <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
        <h3 className="text-sm font-bold text-gray-800 mb-3 flex items-center gap-2">
          <span className="text-lg">&#9878;</span> Veredicto Final
        </h3>
        <Markdown className="text-gray-700">{verdict}</Markdown>
      </div>
    );
  }

  return null;
}
