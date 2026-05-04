import { useState } from 'react';

interface Props {
  onSubmit: (prompt: string) => void;
  disabled: boolean;
  variant?: 'default' | 'hero' | 'compactDark';
}

export function PromptInput({ onSubmit, disabled, variant = 'default' }: Props) {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !disabled) {
      onSubmit(prompt.trim());
    }
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
        <button
          type="submit"
          disabled={disabled || !prompt.trim()}
          className={
            isDark
              ? 'w-full rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-bold text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.28)] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none'
              : 'w-full py-2.5 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm'
          }
        >
          {disabled ? 'Debate en curso...' : 'Iniciar debate'}
        </button>
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
