/**
 * HTTP client for the orchestrator's /debates persistence endpoints.
 *
 * Two flows:
 *
 *   - Live debate
 *       1. createDebate(prompt)      → POST /debates  → { debate_id }
 *       2. streamDebate(debate_id)   → GET  /debates/<id>/stream  (SSE)
 *
 *   - Open a past debate (sidebar click)
 *       1. getEvents(debate_id)      → GET  /debates/<id>/events  → list
 *       2. feed events through reduceDebateEvent in App.tsx for replay
 *
 * The stream endpoint serves both purposes: when the debate is still
 * running it tail-follows live events; when it's already terminal it
 * just plays back the persisted log and closes. The frontend never has
 * to ask "is this live or replay?" — the same handler works for both.
 */

import type { DebateEvent } from '../types';

export interface DebateSummary {
  id: string;
  prompt: string;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  verdict: string | null;
  error: string | null;
}

export interface PersistedEvent {
  seq: number;
  stage: string;
  message: string | null;
  data: Record<string, unknown> | null;
  created_at: string;
}

export class DebateAlreadyActiveError extends Error {
  active: DebateSummary | null;
  constructor(active: DebateSummary | null) {
    super(
      active
        ? `Another debate is already running: "${active.prompt.slice(0, 60)}"`
        : 'Another debate is already running',
    );
    this.name = 'DebateAlreadyActiveError';
    this.active = active;
  }
}

const API = '/api';

async function jsonOrThrow(res: Response): Promise<unknown> {
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // ignore
    }
    throw new Error(
      `HTTP ${res.status} ${res.statusText}: ${JSON.stringify(body)}`,
    );
  }
  return res.json();
}

/** GET /debates — paginated list, newest first. */
export async function listDebates(): Promise<DebateSummary[]> {
  const res = await fetch(`${API}/debates`);
  const body = (await jsonOrThrow(res)) as { debates: DebateSummary[] };
  return body.debates;
}

/** GET /debates/active — null when nothing is running. */
export async function getActiveDebate(): Promise<DebateSummary | null> {
  const res = await fetch(`${API}/debates/active`);
  return (await jsonOrThrow(res)) as DebateSummary | null;
}

/** GET /debates/<id> — metadata, or null on 404. */
export async function getDebate(
  id: string,
): Promise<DebateSummary | null> {
  const res = await fetch(`${API}/debates/${id}`);
  if (res.status === 404) return null;
  return (await jsonOrThrow(res)) as DebateSummary;
}

/** GET /debates/<id>/events?since=N — one-shot replay. */
export async function getEvents(
  id: string,
  since: number = -1,
): Promise<PersistedEvent[]> {
  const res = await fetch(`${API}/debates/${id}/events?since=${since}`);
  const body = (await jsonOrThrow(res)) as { events: PersistedEvent[] };
  return body.events;
}

/** Convert a PersistedEvent into the DebateEvent shape the reducer expects. */
export function persistedToDebateEvent(p: PersistedEvent): DebateEvent {
  return {
    stage: p.stage,
    message: p.message ?? '',
    // The reducer reads .data?.field; null and undefined behave the same way
    // for `?.`, so it's fine to forward null.
    data: (p.data ?? undefined) as DebateEvent['data'],
  };
}

/** POST /debates — kick off a new debate. Throws on 409. */
export async function createDebate(prompt: string): Promise<{
  debate_id: string;
}> {
  const res = await fetch(`${API}/debates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (res.status === 409) {
    let active: DebateSummary | null = null;
    try {
      const body = (await res.json()) as { active?: DebateSummary | null };
      active = body.active ?? null;
    } catch {
      // ignore
    }
    throw new DebateAlreadyActiveError(active);
  }
  const body = (await jsonOrThrow(res)) as { debate_id: string };
  return body;
}

/**
 * GET /debates/<id>/stream — SSE catch-up + tail-follow.
 *
 * Returns when the stream closes (terminal event or disconnect). Errors
 * raised by the network are reported via ``onError``.
 */
export async function streamDebate(
  id: string,
  since: number,
  onEvent: (e: PersistedEvent) => void,
  onError: (msg: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await fetch(`${API}/debates/${id}/stream?since=${since}`, {
      signal,
      headers: { Accept: 'text/event-stream' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No readable stream');
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by blank lines; each line within a frame
      // starts with "data:".
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';
      for (const frame of frames) {
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload) as PersistedEvent);
          } catch {
            // ignore malformed frames — heartbeats etc.
          }
        }
      }
    }
  } catch (error) {
    if ((error as Error).name === 'AbortError') return;
    onError(error instanceof Error ? error.message : String(error));
  }
}
