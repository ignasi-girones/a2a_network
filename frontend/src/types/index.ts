// ── Plan shape (mirror of common.models.TaskPlan on the backend) ──
export interface PlanSubtask {
  id: string;
  description: string;
  required_skill: string;
  depends_on: string[];
  perspective: string | null;
}

export interface TaskPlan {
  goal: string;
  subtasks: PlanSubtask[];
  max_workers?: number;
}

// ── Per-subtask runtime state derived from streaming events ──
export type SubtaskStatus = 'pending' | 'running' | 'done' | 'failed';

export interface SubtaskRuntime {
  status: SubtaskStatus;
  worker_id?: string;
  output?: string;
  error?: string;
}

// ── Events emitted by the orchestrator's ProgressCallback ──
export interface DebateEvent {
  stage: string;
  message: string;
  data?: {
    // Agentic flow
    plan?: TaskPlan;
    subtask_id?: string;
    subtask_ids?: string[];
    worker_id?: string;
    required_skill?: string;
    perspective?: string | null;
    description?: string;
    depends_on?: string[];
    output_preview?: string;
    text?: string;
    error?: string;

    // Legacy / generic
    agent?: string;
    round?: number;
    ae1_role?: string;
    ae2_role?: string;
    max_rounds?: number;
    verdict?: string;
    tool?: string;
    query?: string;
  };
}

export interface DebateState {
  status: 'idle' | 'running' | 'completed' | 'error';
  events: DebateEvent[];
  verdict: string | null;
  error: string | null;
}
