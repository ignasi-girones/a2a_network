// ── Phase 3 dialectic role catalog ──────────────────────────────────────────
// Mirrors `RoleId` in common/models.py. Keep both in sync.
export type RoleId =
  | 'analyst'
  | 'seeker'
  | 'devils_advocate'
  | 'empiricist'
  | 'pragmatist'
  | 'synthesizer';

export const CANONICAL_ROLES: readonly RoleId[] = [
  'analyst',
  'seeker',
  'devils_advocate',
  'empiricist',
  'pragmatist',
  'synthesizer',
];

// Persona metadata embedded in `subtask_dispatch` events. Mirrors the dict
// returned by `PlanExecutor._configure_worker`.
export interface PersonaMeta {
  display_name: string;
  role_id: RoleId;
  stratagem_id: number | null;
  tool_whitelist: string[];
  temperature: number;
}

// ── Phase 3 / Pillar 2: Bayesian belief trajectories ───────────────────────
// One sample per agent per LLM intervention; the BeliefTrajectoryChart
// stitches consecutive samples per agent into a line series.
export interface BeliefSample {
  t: number;          // monotonic sample index inside the run
  log_odds: number;   // posterior log-odds after this update
  delta: number;      // signed magnitude of this update
  llr: number;        // raw log-likelihood ratio reported by the lateral LLM
  rationale: string;  // short human-readable justification
  phase: string;      // 'post_initial' | 'post_refine' | 'post_aporia'
}

export interface BeliefSeries {
  agent: string;          // agent_id (worker)
  role_id: RoleId | null;
  claim: string;
  samples: BeliefSample[];
}

// ── Phase 4: Discussion Ledger entries ────────────────────────────────────
// Each entry is one intervention by one agent in one round of the
// multi-round deliberation. The DiscussionLedgerView component renders
// them as a chat-style threaded list grouped by round.
export interface LedgerEntry {
  turn: number;            // monotonic across the run
  round_number: number;    // 1-indexed
  role_id: RoleId;
  agent_id: string;
  text: string;
  belief_after: number | null;
  delta: number | null;
  references: number[];    // turn-ids this entry reacts to
  timestamp: string;
}

// Round lifecycle metadata derived from round_start/round_end events
export interface RoundMeta {
  round_number: number;
  speakers: RoleId[];
  status: 'active' | 'closed';
  n_entries_at_start?: number;
  n_entries_at_end?: number;
}

// Deliberation state aggregated from streaming events
export interface DeliberationState {
  claim: string | null;
  goal: string | null;
  max_rounds: number;
  current_round: number;
  rounds: Record<number, RoundMeta>;
  ledger: LedgerEntry[];
  active_role: RoleId | null;
  terminated_reason: string | null;
}

// ── Phase 3 / Pillar 3: Habermasian validity claims table ──────────────────
// The Synthesizer emits one row per panel member it evaluated.
export interface ValidityClaimRow {
  agent: string;
  truth: number;
  rightness: number;
  sincerity: number;
  comprehensibility: number;
  admitted: boolean;
  note?: string;
}

// ── Phase 3 / Pillar 3: DRTAG aporia diagnosis ─────────────────────────────
export interface AporiaDiagnosis {
  detected: boolean;
  reason: string;
  n_agents: number;
  spread: number | null;
  mean_total_movement: number | null;
  agent_finals?: Record<string, number>;
}

// ── Plan shape (mirror of common.models.TaskPlan on the backend) ──
export interface PlanSubtask {
  id: string;
  description: string;
  role_id?: RoleId | null;
  required_skill: string;
  depends_on: string[];
  perspective: string | null;
}

export interface TaskPlan {
  goal: string;
  claim?: string;
  subtasks: PlanSubtask[];
  max_workers?: number;
}

// ── Per-subtask runtime state derived from streaming events ──
export type SubtaskStatus = 'pending' | 'running' | 'done' | 'failed';

export interface SubtaskRuntime {
  status: SubtaskStatus;
  worker_id?: string;
  role_id?: RoleId;
  persona?: PersonaMeta;
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
    role_id?: RoleId | null;
    persona?: PersonaMeta;
    required_skill?: string;
    perspective?: string | null;
    description?: string;
    depends_on?: string[];
    output_preview?: string;
    text?: string;
    error?: string;

    // Spawn / capacity events
    skill?: string;
    role?: RoleId;
    needed?: number;
    have?: number;
    agent_id?: string;

    // Belief update events (Pillar 2)
    claim?: string;
    log_odds?: number;
    delta?: number;
    llr?: number;
    rationale?: string;
    phase?: string;

    // Pillar 3 — DRTAG aporia + Habermasian validity
    detected?: boolean;
    spread?: number | null;
    mean_total_movement?: number | null;
    n_agents?: number;
    agent_finals?: Record<string, number>;
    stratagem_id?: number | null;
    stratagem_name?: string;
    excluded?: number[];
    validity_claims?: ValidityClaimRow[];
    source_subtask?: string;

    // Phase 4 — Multi-round deliberation
    entry?: LedgerEntry;
    round_number?: number;
    speakers?: RoleId[];
    n_entries?: number;
    rounds_used?: number;
    terminated_reason?: string;
    max_rounds?: number;
    beliefs?: Record<RoleId, {
      log_odds: number;
      total_movement: number;
      last_delta: number;
      turns: number;
    }>;

    // Legacy / generic
    agent?: string;
    round?: number;
    ae1_role?: string;
    ae2_role?: string;
    verdict?: string;
    tool?: string;
    query?: string;

    // Other event fields used by the orchestrator's progress channel
    goal?: string;
    reason?: string;
  };
}

export interface DebateState {
  status: 'idle' | 'running' | 'completed' | 'error';
  events: DebateEvent[];
  verdict: string | null;
  error: string | null;
}

// ── UI palette per dialectic role ───────────────────────────────────────────
// Centralised so DebateGraph, timeline badges, and the header legend agree.
export interface RolePalette {
  label: string;        // Spanish display label
  fill: string;         // node body fill
  stroke: string;       // node border (and edge color when running)
  badge: string;        // tailwind classes for inline role badges
  accent: string;       // primary tint (e.g. selected ring)
}

export const ROLE_PALETTE: Record<RoleId, RolePalette> = {
  analyst: {
    label: 'Analista',
    fill: '#eff6ff',
    stroke: '#2563eb',
    badge: 'bg-blue-100 text-blue-800 border border-blue-200',
    accent: '#2563eb',
  },
  seeker: {
    label: 'Buscador',
    fill: '#ecfdf5',
    stroke: '#059669',
    badge: 'bg-emerald-100 text-emerald-800 border border-emerald-200',
    accent: '#059669',
  },
  devils_advocate: {
    label: 'Abogado del Diablo',
    fill: '#fef2f2',
    stroke: '#dc2626',
    badge: 'bg-red-100 text-red-800 border border-red-200',
    accent: '#dc2626',
  },
  empiricist: {
    label: 'Empírico',
    fill: '#fff7ed',
    stroke: '#ea580c',
    badge: 'bg-orange-100 text-orange-800 border border-orange-200',
    accent: '#ea580c',
  },
  pragmatist: {
    label: 'Pragmático',
    fill: '#f0fdfa',
    stroke: '#0d9488',
    badge: 'bg-teal-100 text-teal-800 border border-teal-200',
    accent: '#0d9488',
  },
  synthesizer: {
    label: 'Sintetizador',
    fill: '#f5f3ff',
    stroke: '#7c3aed',
    badge: 'bg-purple-100 text-purple-800 border border-purple-200',
    accent: '#7c3aed',
  },
};
