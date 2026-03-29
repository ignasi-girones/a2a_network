import { useState } from 'react';

interface Props {
  onSubmit: (prompt: string) => void;
  disabled: boolean;
}

export function PromptInput({ onSubmit, disabled }: Props) {
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

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Escribe un tema para debatir..."
          disabled={disabled}
          rows={4}
          className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500 text-sm"
        />
        <button
          type="submit"
          disabled={disabled || !prompt.trim()}
          className="w-full py-2.5 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm"
        >
          {disabled ? 'Debate en curso...' : 'Iniciar Debate'}
        </button>
      </form>

      {!disabled && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 font-medium">Ejemplos:</p>
          {examples.map((ex, i) => (
            <button
              key={i}
              onClick={() => setPrompt(ex)}
              className="block w-full text-left text-xs text-gray-600 p-2 rounded hover:bg-gray-100 transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
