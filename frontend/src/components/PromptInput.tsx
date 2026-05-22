import { useRef, useState } from 'react';
import { Paperclip, X } from 'lucide-react';

const ACCEPTED_EXTENSIONS = '.pdf,.xlsx,.csv,.txt';
const ACCEPTED_MIME = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'text/csv',
  'text/plain',
];
const MAX_FILES = 5;
const MAX_SIZE_MB = 5;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface Props {
  onSubmit: (prompt: string, files?: File[]) => void;
  disabled: boolean;
  variant?: 'default' | 'hero' | 'compactDark';
  acceptFiles?: boolean;
}

export function PromptInput({ onSubmit, disabled, variant = 'default', acceptFiles = false }: Props) {
  const [prompt, setPrompt] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !disabled) {
      onSubmit(prompt.trim(), files.length > 0 ? files : undefined);
      setFiles([]);
      setFileError(null);
    }
  };

  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    // Reset the input so re-selecting the same file works
    e.target.value = '';
    setFileError(null);

    const combined = [...files, ...selected];

    // Validate count
    if (combined.length > MAX_FILES) {
      setFileError(`Maximo ${MAX_FILES} archivos`);
      return;
    }

    // Validate each new file
    for (const f of selected) {
      if (!ACCEPTED_MIME.includes(f.type)) {
        setFileError(`Tipo no soportado: ${f.name}`);
        return;
      }
      if (f.size > MAX_SIZE_BYTES) {
        setFileError(`${f.name} excede ${MAX_SIZE_MB}MB`);
        return;
      }
    }

    setFiles(combined);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setFileError(null);
  };

  const examples = [
    '¿Deberia una startup invertir en infraestructura IA propia o usar servicios cloud?',
    '¿Es mejor el trabajo remoto o presencial para equipos de desarrollo?',
    '¿Microservicios o monolito para una startup en fase inicial?',
  ];

  const isHero = variant === 'hero';
  const isCompactDark = variant === 'compactDark';
  const isDark = isHero || isCompactDark;

  return (
    <div className={isHero ? 'space-y-5' : 'space-y-4'}>
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Escribe un tema para debatir..."
          disabled={disabled}
          rows={isHero ? 6 : isCompactDark ? 2 : 4}
          className={
            isDark
              ? `w-full resize-none rounded-2xl border border-cyan-200/20 bg-slate-950/70 p-4 text-sm text-slate-100 placeholder:text-slate-500 outline-none shadow-[inset_0_0_0_1px_rgba(255,255,255,0.03)] transition focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/20 disabled:cursor-not-allowed disabled:opacity-60 ${isHero ? 'min-h-40 text-base leading-relaxed' : 'min-h-20'}`
              : 'w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500 text-sm'
          }
        />

        {/* File chips */}
        {acceptFiles && files.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {files.map((f, i) => (
              <span
                key={`${f.name}-${i}`}
                className={
                  isDark
                    ? 'inline-flex items-center gap-1.5 rounded-full border border-cyan-200/20 bg-cyan-200/10 px-2.5 py-1 text-xs text-cyan-100'
                    : 'inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs text-blue-700'
                }
              >
                <Paperclip size={12} />
                <span className="max-w-[140px] truncate">{f.name}</span>
                <span className="opacity-60">{formatSize(f.size)}</span>
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  className="ml-0.5 rounded-full p-0.5 hover:bg-black/10 transition"
                  aria-label={`Quitar ${f.name}`}
                >
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}

        {/* File error */}
        {fileError && (
          <div className={`text-xs ${isDark ? 'text-red-300' : 'text-red-600'}`}>
            {fileError}
          </div>
        )}

        <div className="flex gap-2">
          {/* Attach button */}
          {acceptFiles && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={ACCEPTED_EXTENSIONS}
                onChange={handleFilesSelected}
                disabled={disabled}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || files.length >= MAX_FILES}
                title="Adjuntar archivos (PDF, Excel, CSV, TXT)"
                className={
                  isDark
                    ? 'flex-none rounded-2xl border border-cyan-200/20 bg-slate-950/70 px-3.5 py-3 text-cyan-200 transition hover:border-cyan-300/40 hover:bg-cyan-200/10 disabled:cursor-not-allowed disabled:opacity-40'
                    : 'flex-none rounded-lg border border-gray-300 bg-white px-3 py-2.5 text-gray-500 transition hover:bg-gray-50 hover:text-blue-600 disabled:cursor-not-allowed disabled:opacity-40'
                }
              >
                <Paperclip size={18} />
              </button>
            </>
          )}

          <button
            type="submit"
            disabled={disabled || !prompt.trim()}
            className={
              isDark
                ? 'flex-1 rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-bold text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.28)] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none'
                : 'flex-1 py-2.5 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm'
            }
          >
            {disabled ? 'Debate en curso...' : 'Iniciar debate'}
          </button>
        </div>
      </form>

      {!disabled && (
        <div className="space-y-2">
          <p className={`text-xs font-medium ${isDark ? 'text-cyan-100/50' : 'text-gray-500'}`}>Ejemplos:</p>
          {examples.map((ex, i) => (
            <button
              key={i}
              onClick={() => setPrompt(ex)}
              className={
                isDark
                  ? 'block w-full rounded-xl border border-white/10 bg-white/[0.03] p-2.5 text-left text-xs text-slate-300 transition hover:border-cyan-200/30 hover:bg-cyan-200/10'
                  : 'block w-full text-left text-xs text-gray-600 p-2 rounded hover:bg-gray-100 transition-colors'
              }
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
