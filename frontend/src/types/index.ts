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

// ── Per-round agent position scores (0..1 axis where 0 = AE1's stance,
//     1 = AE2's stance, 0.5 = neutral). Emitted by the consensus loop ──
export interface AgentPositions {
  ae1?: number;
  ae2?: number;
  ae3?: number;
}

export interface AgentPositionsSample {
  round: number;
  positions: AgentPositions;
  agreement_score?: number;
}

// ── Empirical components feeding the agreement_score ──
export interface ConsensusComponents {
  dispersion?: number;          // raw [0,1], higher = more apart
  dispersion_score?: number;    // 1 - dispersion, contributes positively
  pairwise_similarity?: number; // mean cosine similarity across agent pairs
  movement_score?: number;      // 0..1 — agents reacting between rounds
  concession_score?: number;    // 0..1 — explicit "you changed my mind" signals
  weights?: {
    dispersion?: number;
    similarity?: number;
    movement?: number;
    concessions?: number;
  };
  fallback?: boolean;
}

// ── Snapshot of the consensus evaluation after a given round ──
export interface ConsensusSnapshot {
  round: number;
  agreement_score: number;
  reason?: string;
  positions?: AgentPositions;
  shared_points: string[];
  remaining_disagreements: string[];
  components?: ConsensusComponents;
  movement?: AgentPositions;     // per-agent movement since previous round
  concessions?: { ae1?: number; ae2?: number; ae3?: number };
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

    // Consensus / position tracking
    round?: number;
    positions?: AgentPositions;
    agreement_score?: number;
    reason?: string;
    shared_points?: string[];
    remaining_disagreements?: string[];
    extension_attempt?: number;
    components?: ConsensusComponents;
    movement?: AgentPositions;
    concessions?: { ae1?: number; ae2?: number; ae3?: number };

    // Legacy / generic
    agent?: string;
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
