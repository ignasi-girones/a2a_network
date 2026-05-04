"""Central metric definitions — the single source of truth for all
Prometheus objects used across the A2A network.

Every metric is created on a custom ``CollectorRegistry`` so that the
default process-level collectors (gc, platform) don't leak into the
/metrics endpoint, keeping cardinality predictable.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)

REGISTRY = CollectorRegistry()

# ── HTTP layer ─────────────────────────────────────────────────────────────────

HTTP_REQUESTS = Counter(
    "a2a_http_requests_total",
    "Total HTTP requests received",
    ["agent_id", "method", "path", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "a2a_http_request_duration_seconds",
    "HTTP request latency",
    ["agent_id", "method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

# ── LLM calls ─────────────────────────────────────────────────────────────────

LLM_CALLS = Counter(
    "a2a_llm_calls_total",
    "Total LLM completion calls",
    ["agent_id", "model", "status"],
    registry=REGISTRY,
)
LLM_DURATION = Histogram(
    "a2a_llm_call_duration_seconds",
    "LLM call latency",
    ["agent_id", "model"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "a2a_llm_tokens_total",
    "LLM tokens processed",
    ["agent_id", "model", "direction"],
    registry=REGISTRY,
)

# ── MCP tool calls ─────────────────────────────────────────────────────────────

MCP_CALLS = Counter(
    "a2a_mcp_calls_total",
    "Total MCP tool invocations",
    ["agent_id", "tool", "status"],
    registry=REGISTRY,
)
MCP_DURATION = Histogram(
    "a2a_mcp_call_duration_seconds",
    "MCP tool call latency",
    ["agent_id", "tool"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0),
    registry=REGISTRY,
)

# ── Debate lifecycle ───────────────────────────────────────────────────────────

DEBATES_TOTAL = Counter(
    "a2a_debates_total",
    "Total debates started",
    ["agent_id"],
    registry=REGISTRY,
)
DEBATES_ACTIVE = Gauge(
    "a2a_debates_active",
    "Currently running debates",
    ["agent_id"],
    registry=REGISTRY,
)
DELIBERATION_ROUNDS = Counter(
    "a2a_deliberation_rounds_total",
    "Total deliberation rounds executed",
    ["agent_id"],
    registry=REGISTRY,
)
DELIBERATION_DURATION = Histogram(
    "a2a_deliberation_duration_seconds",
    "End-to-end deliberation duration",
    ["agent_id", "terminated_reason"],
    buckets=(5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)
SUBTASK_DISPATCH = Counter(
    "a2a_subtask_dispatch_total",
    "Subtask dispatches by role",
    ["agent_id", "role_id"],
    registry=REGISTRY,
)

# ── Belief dynamics ────────────────────────────────────────────────────────────

BELIEF_UPDATES = Counter(
    "a2a_belief_updates_total",
    "Belief state updates",
    ["agent_id", "role_id", "phase"],
    registry=REGISTRY,
)
BELIEF_DELTA_ABS = Histogram(
    "a2a_belief_delta_abs",
    "Absolute magnitude of belief updates",
    ["agent_id", "role_id"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
    registry=REGISTRY,
)
APORIA_DETECTIONS = Counter(
    "a2a_aporia_detections_total",
    "Aporia events detected by DRTAG",
    ["agent_id"],
    registry=REGISTRY,
)

# ── Worker health ──────────────────────────────────────────────────────────────

REGISTRY_WORKERS = Gauge(
    "a2a_registry_workers",
    "Workers currently registered",
    ["agent_id"],
    registry=REGISTRY,
)
WORKER_SPAWNS = Counter(
    "a2a_worker_spawns_total",
    "Dynamic worker spawn events",
    ["agent_id", "role"],
    registry=REGISTRY,
)
WORKER_CONFIGURE_DURATION = Histogram(
    "a2a_worker_configure_duration_seconds",
    "Worker persona configure latency",
    ["agent_id", "role_id"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

# ── Info ───────────────────────────────────────────────────────────────────────

AGENT_INFO = Info(
    "a2a_agent",
    "Static agent metadata",
    ["agent_id"],
    registry=REGISTRY,
)
PROCESS_START = Gauge(
    "a2a_process_start_time_seconds",
    "Unix timestamp when the agent process started",
    ["agent_id"],
    registry=REGISTRY,
)
