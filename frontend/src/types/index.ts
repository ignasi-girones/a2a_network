export interface DebateEvent {
  stage: string;
  message: string;
  data?: {
    agent?: string;
    round?: number;
    text?: string;
    ae1_role?: string;
    ae2_role?: string;
    max_rounds?: number;
    verdict?: string;
  };
}

export interface DebateState {
  status: 'idle' | 'running' | 'completed' | 'error';
  events: DebateEvent[];
  verdict: string | null;
  error: string | null;
}
