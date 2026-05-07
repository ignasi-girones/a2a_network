import type { DebateSummary } from '../api/debates';

interface Props {
  debates: DebateSummary[];
  activeId: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}

const STATUS_BADGE: Record<DebateSummary['status'], string> = {
  running: 'bg-amber-100 text-amber-700',
  completed: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
};

const STATUS_LABEL: Record<DebateSummary['status'], string> = {
  running: 'En curso',
  completed: 'Completado',
  failed: 'Fallido',
};

function formatRelative(iso: string): string {
  const d = new Date(iso.replace(' ', 'T') + 'Z');
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return 'hace segundos';
  if (diff < 3600) return `hace ${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `hace ${Math.floor(diff / 3600)}h`;
  return `hace ${Math.floor(diff / 86400)}d`;
}

/**
 * Left-rail list of past + current debates. The active one (if any) is
 * sticky at the top with an animated badge; the rest are sorted newest
 * first by `created_at`.
 */
export function DebateHistorySidebar({
  debates,
  activeId,
  selectedId,
  onSelect,
  onNew,
}: Props) {
  return (
    <aside className="w-64 shrink-0 bg-white border-r border-gray-200 flex flex-col h-screen sticky top-0">
      <div className="px-4 py-3 border-b border-gray-200">
        <button
          onClick={onNew}
          disabled={activeId !== null}
          className={`w-full text-sm font-semibold rounded px-3 py-2 transition ${
            activeId !== null
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
          title={
            activeId !== null
              ? 'Hay un debate en curso — espera a que termine'
              : 'Empezar un debate nuevo'
          }
        >
          + Nuevo debate
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {debates.length === 0 ? (
          <p className="text-xs text-gray-500 px-4 py-6 text-center">
            No hay debates todavía.
          </p>
        ) : (
          <ul className="divide-y divide-gray-100">
            {debates.map((d) => {
              const isSelected = d.id === selectedId;
              const isActive = d.id === activeId;
              return (
                <li key={d.id}>
                  <button
                    onClick={() => onSelect(d.id)}
                    className={`w-full text-left px-4 py-3 transition ${
                      isSelected
                        ? 'bg-blue-50'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span
                        className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded ${STATUS_BADGE[d.status]}`}
                      >
                        {isActive && (
                          <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                        )}
                        {STATUS_LABEL[d.status]}
                      </span>
                      <span className="text-[10px] text-gray-400">
                        {formatRelative(d.created_at)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-800 line-clamp-2">
                      {d.prompt}
                    </p>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
