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
 * Uses the native EventSource API. fetch+reader.read() also works in
 * principle, but the browser sometimes coalesces small chunks before
 * yielding them to userland; EventSource dispatches an `onmessage` event
 * per SSE frame, which gives React a chance to commit each setState
 * between events instead of batching them all into one render at the end.
 *
 * Resolves when the stream closes (terminal event, server disconnect, or
 * the AbortSignal fires).
 */
export function streamDebate(
  id: string,
  since: number,
  onEvent: (e: PersistedEvent) => void,
  onError: (msg: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve) => {
    const url = `${API}/debates/${id}/stream?since=${since}`;
    const es = new EventSource(url);
    let eventCount = 0;
    let closed = false;

    const close = () => {
      if (closed) return;
      closed = true;
      es.close();
      resolve();
    };

    // The orchestrator emits SSE frames with `event: progress`. EventSource
    // dispatches named-event types to specific listeners — `onmessage` only
    // catches *unnamed* frames, so we attach to "progress" explicitly.
    const handle = (msg: MessageEvent) => {
      const payload = msg.data;
      if (!payload) return;
      try {
        const parsed = JSON.parse(payload) as PersistedEvent;
        eventCount += 1;
        onEvent(parsed);
        // The orchestrator closes the stream right after the terminal
        // event, which surfaces as an `error` here — but in some browsers
        // a clean server close arrives as readyState=CLOSED without an
        // error event. Detect it here so we resolve promptly.
        if (parsed.stage === 'verdict' || parsed.stage === 'failed') {
          setTimeout(close, 0);
        }
      } catch {
        // ignore malformed frames — heartbeats etc.
      }
    };

    es.addEventListener('progress', handle);
    es.onmessage = handle; // fallback for any unnamed frames

    es.onerror = () => {
      // EventSource fires `error` both for transient blips and for the
      // final close. readyState distinguishes them:
      //   - CONNECTING (0): browser will retry → leave it
      //   - OPEN       (1): transient, will recover
      //   - CLOSED     (2): terminal, we're done
      if (es.readyState === EventSource.CLOSED) {
        close();
      } else if (eventCount === 0) {
        // No events yet AND we got an error → likely the initial connect
        // failed. Surface it so the UI can show something useful.
        onError('Stream connection failed');
        close();
      }
    };

    if (signal) {
      if (signal.aborted) {
        close();
        return;
      }
      signal.addEventListener('abort', close, { once: true });
    }
  });
}
