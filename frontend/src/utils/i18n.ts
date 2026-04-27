/**
 * Translate the english round/synthesis/opening labels the planner emits
 * (e.g. "ae1: round 1", "ae1: synthesis 2", "ae1: Mediator, neutral") into
 * Spanish for display purposes only. The backend keeps English internally so
 * the LLM reasons consistently — this is purely a presentation layer.
 */
export function humanizePerspective(perspective: string | null | undefined): string {
  if (!perspective) return '';
  let out = perspective;
  // Round descriptors
  out = out.replace(/\bopening\b/gi, 'apertura');
  out = out.replace(/\bsynthesis\s+(\d+)\b/gi, 'síntesis $1');
  out = out.replace(/\bsynthesis\b/gi, 'síntesis');
  out = out.replace(/\bround\s+(\d+)\s+of\s+(\d+)\b/gi, 'ronda $1 de $2');
  out = out.replace(/\bround\s+(\d+)\b/gi, 'ronda $1');
  // Common neutral / mediator labels
  out = out.replace(/\bneutral\s+mediator\b/gi, 'mediador neutral');
  out = out.replace(/\bMediator,\s*neutral\b/gi, 'Mediador, neutral');
  return out;
}
