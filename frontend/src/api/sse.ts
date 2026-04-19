import type { DebateEvent } from '../types';

/**
 * Generate a RFC4122-ish UUID without relying on crypto.randomUUID().
 *
 * crypto.randomUUID() is only available in "secure contexts" (HTTPS or
 * localhost). The VM deployment is served over plain HTTP
 * (http://nattech.fib.upc.edu:40536), where calling crypto.randomUUID
 * throws TypeError and kills the request before it's sent.
 */
function uuid(): string {
  // Prefer crypto.randomUUID when available (HTTPS / localhost dev).
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.randomUUID === 'function') {
    return c.randomUUID();
  }
  // Fallback: Math.random-backed v4 UUID. Not cryptographically strong,
  // which is fine — we only use these IDs as JSON-RPC correlation keys.
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    const v = ch === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Send a debate request to the orchestrator via A2A JSON-RPC
 * and stream SSE events back.
 *
 * The A2A `message/stream` method uses POST with SSE response,
 * so we can't use EventSource (GET only). We use fetch + ReadableStream.
 */
export async function startDebateStream(
  prompt: string,
  onEvent: (event: DebateEvent) => void,
  onComplete: (verdict: string) => void,
  onError: (error: string) => void,
): Promise<void> {
  try {
    const requestBody = {
      jsonrpc: '2.0',
      id: uuid(),
      method: 'SendStreamingMessage',
      params: {
        message: {
          message_id: uuid(),
          role: 'ROLE_USER',
          parts: [{ text: prompt }],
        },
      },
    };

    const response = await fetch('/api/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No readable stream');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (!data) continue;

          try {
            const parsed = JSON.parse(data);
            console.debug('[SSE]', JSON.stringify(parsed).slice(0, 200));
            processSSEData(parsed, onEvent, onComplete, onError);
          } catch {
            // Not valid JSON yet, might be partial
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

function processSSEData(
  data: any,
  onEvent: (event: DebateEvent) => void,
  onComplete: (verdict: string) => void,
  onError: (error: string) => void,
): void {
  // Handle JSON-RPC response wrapper — SSE events are wrapped in StreamResponse:
  //   data.result.task.status       (Task — completed/failed)
  //   data.result.statusUpdate.status (TaskStatusUpdateEvent — progress)
  const result = data.result || data;
  const status = result.task?.status || result.statusUpdate?.status;
  const artifacts = result.task?.artifacts;

  if (!status) return;

  const state = status.state;
  const parts = status.message?.parts || [];

  // TaskStatusUpdateEvent — progress updates during debate
  if (state === 'TASK_STATE_WORKING') {
    for (const part of parts) {
      if (part.text) {
        try {
          const event: DebateEvent = JSON.parse(part.text);
          onEvent(event);
        } catch {
          onEvent({ stage: 'info', message: part.text });
        }
      }
    }
  }

  // Task completed — final verdict
  if (state === 'TASK_STATE_COMPLETED') {
    let verdict = '';
    if (artifacts) {
      for (const artifact of artifacts) {
        for (const part of artifact.parts || []) {
          if (part.text) verdict += part.text;
        }
      }
    }
    if (!verdict) {
      for (const part of parts) {
        if (part.text) verdict += part.text;
      }
    }
    if (verdict) onComplete(verdict);
  }

  // Task failed
  if (state === 'TASK_STATE_FAILED') {
    const errorMsg = parts[0]?.text || 'Debate failed';
    onError(errorMsg);
  }
}
