/**
 * Turn a LiteLLM-style model slug into a short, human-friendly label for the
 * header badges. The slug shape is "<provider>/<model-name>" — examples:
 *   "groq/llama-3.3-70b-versatile"        → "Groq · llama-3.3-70b"
 *   "mistral/mistral-large-latest"         → "Mistral · large-latest"
 *   "ollama/qwen2.5:14b"                   → "Ollama · qwen2.5:14b"
 *   "cerebras/qwen-3-235b-a22b-instruct..." → "Cerebras · qwen-3-235b"
 */

const PROVIDER_NAMES: Record<string, string> = {
  groq: 'Groq',
  gemini: 'Gemini',
  mistral: 'Mistral',
  cerebras: 'Cerebras',
  ollama: 'Ollama',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
};

export function modelLabel(slug: string | undefined | null): string {
  if (!slug) return '—';
  const idx = slug.indexOf('/');
  if (idx === -1) return slug;
  const provider = slug.slice(0, idx).toLowerCase();
  const rest = slug.slice(idx + 1);
  const providerLabel = PROVIDER_NAMES[provider] ?? provider;

  // Trim a verbose model name without losing the family (llama-3.3-70b,
  // qwen-3-235b, etc.).
  const short = rest
    .replace(/-instruct.*$/i, '')
    .replace(/-versatile$/i, '')
    .replace(/-latest$/i, ' latest')
    .replace(/-a\d+b.*$/i, '');

  return `${providerLabel} · ${short}`;
}
