# Agentes del sistema

La red consta de **6 agentes A2A** y un servidor MCP de herramientas. Cada agente es un servidor HTTP independiente que publica una **Agent Card** en `/.well-known/agent-card.json` y procesa mensajes vía JSON-RPC tal como define A2A v1.0.0.

El orquestador NO conoce de antemano qué agentes existen. Cada worker se auto-registra al arrancar (POST `/registry/register`) y publica un **catálogo de skills auto-descritas** — la `description` de cada skill es la fuente de verdad sobre cuándo y cómo usarla. Un planner LLM razona sobre ese catálogo para decidir el plan.

---

## Orchestrator (Puerto 8080)

**Modelo (default):** `groq/llama-3.3-70b-versatile`
**Agent Card:** Estática
**Rol:** Coordinador agéntico de la deliberación.

Es el único agente con capacidad de **decisión autónoma**: el resto son workers reactivos que ejecutan su skill cuando reciben un A2A SendMessage.

### Componentes internos

| Módulo | Responsabilidad |
|---|---|
| `agent_registry.py` | Mapa thread-safe `agent_id → WorkerEntry`. Endpoints `/registry/register`, `/registry/agents`, `/registry/by-skill/{id}`. |
| `planner.py` | Llamadas LLM que producen un `TaskPlan` (DAG). Función `create_plan` para apertura, `extend_for_consensus` para extensiones de ronda. |
| `plan_executor.py` | Recorre el DAG, hace `_assign_workers` (pinning por `perspective` + round-robin) y dispatcha cada subtarea vía A2A. |
| `worker_spawner.py` | Lanza workers extra como subprocess si el plan demanda más concurrencia que la registrada. Pool de puertos `9010+`. |
| `agentic_orchestrator.py` | El director. Implementa `run()`, `_consensus_loop()`, `_finalize_with_feedback()`, `_synthesize()`. |
| `models_routes.py` | `GET /models` — devuelve el modelo configurado por agente, leído de `settings`. |
| `executor.py` | El `OrchestratorExecutor` A2A: recibe el prompt del usuario y arranca `AgenticOrchestrator.run()`, emitiendo eventos SSE. |
| `flow_manager.py` | **Legacy**, no se usa en el path agéntico actual. |

### Llamadas LLM internas (no A2A)

El orquestador hace 3 tipos de llamadas LLM directas:

1. **Plan creation** — recibe el prompt + catálogo y emite el DAG inicial.
2. **Consensus check** — tras cada exchange evalúa convergencia, devolviendo `agreement_score`, posiciones por agente, `shared_points` y `remaining_disagreements`. El score se *capa* en código si la dispersión de posiciones lo contradice (defensa contra incoherencias del LLM).
3. **Plan extension** — si no hay consenso y queda presupuesto, pide al planner una mini-extensión: una nueva ronda paralela (una subtask por agente).

### SSE streaming

El orquestador emite `TaskStatusUpdateEvent` con metadata JSON. El frontend acumula estos eventos:

| `stage` | Cuándo se emite |
|---|---|
| `discover` / `plan` / `plan_ready` | Antes / durante / después de generar el DAG |
| `plan_start` / `plan_batch` | El executor arranca un batch ready |
| `subtask_dispatch` / `subtask_done` / `subtask_failed` | Por cada subtarea |
| `tool_use` | Worker invoca un tool MCP (relayed por el executor) |
| `agent_positions` | Tras cada exchange, scores 0..1 por agente |
| `consensus_check` / `consensus` / `no_consensus` | Resultado del LLM evaluador |
| `extend_plan` / `extend_failed` | Apertura/fallo de una extensión |
| `synthesize` / `complete` | Cierre del flow |

---

## Normalizer (Puerto 8081)

**Modelo (default):** `gemini/gemini-2.5-flash`
**Skill:** `normalize_input`
**Rol:** Transforma el texto plano del usuario en JSON estructurado.

### Skill auto-descrito

La `description` que el normalizer publica le dice al planner *cuándo* y *cómo* usarlo:

> *"Converts a raw free-text user prompt into a structured JSON object with topic, domain, question type, constraints, and suggested perspectives. Use as the FIRST step of any plan whose user input arrives as raw natural language."*

### Output

```json
{
  "topic": "...",
  "domain": "finance | tech | hr | ...",
  "question_type": "opinion | decision | comparison | analysis",
  "constraints": [],
  "suggested_perspectives": ["...", "..."]
}
```

### Tolerancia a fallos

- 2 reintentos con corrective message si el LLM no devuelve JSON válido.
- Fallback a una estructura mínima si los reintentos fallan o si el provider falla por completo (timeout/rate limit). El subtask se completa con datos sensatos en lugar de tirar todo el debate.

### Archivos

- `agents/normalizer/__main__.py` — servidor A2A + auto-registro.
- `agents/normalizer/executor.py` — `NormalizerExecutor` con prompt JSON-mode.

---

## Specialized Agents — AE1, AE2, AE3 (Puertos 8082, 8083, 8087)

**Modelos (default):** Mistral / Cerebras / Groq (configurables vía `AE1_MODEL`, `AE2_MODEL`, `AE3_MODEL` en `.env`).
**Skill:** `debate` (los 3 publican el mismo skill).
**Rol:** Participan en deliberaciones multi-agente con identidades persistentes a través de las rondas.

### Roles típicos

- **AE1** — abre advocando una postura.
- **AE2** — abre advocando la postura opuesta.
- **AE3** — *evaluador independiente*: no tiene postura asignada, pondera evidencias y se decanta hacia el lado mejor fundamentado (incluso endorsando totalmente AE1 o AE2 si la evidencia lo justifica). NO es un mediador centrista.

### Convención de `perspective` (clave del enrutado)

El planner asigna `perspective` con el formato `"<agent_id>: <role + stance>"`. El executor extrae el `agent_id` y **pina** la subtask al worker correspondiente. Esto garantiza que la trayectoria de cada agente queda en el mismo proveedor LLM ronda tras ronda. Ejemplos:

- `"ae1: DevOps Engineer, pro-remote"` → pinned a worker `ae1`
- `"ae3: Independent evaluator"` → pinned a worker `ae3`
- `"ae1: synthesis 1"` (rondas posteriores) → sigue pinned a `ae1`

### System prompt (en `agent_state.py`)

`DEFAULT_SYSTEM_PROMPT` es un prompt universal compartido entre AE1/AE2/AE3 que:

- Explica las identidades (ae1, ae2, ae3) y que ae3 es evaluador independiente.
- Anima explícitamente a **cambiar de bando** si la otra parte tiene mejores argumentos.
- Penaliza el centrismo forzado ("both have a point" sin compromiso).
- Pide formato `AGREEMENTS:` / `REFINEMENT:` en cada respuesta.

El rol específico de cada subtask llega en la `description` (ROLE, ROUND, GOAL, etc.) — el system prompt no necesita reconfigurarse, lo que hace que `/internal/configure` sea innecesario en el path agéntico.

### Web search vía MCP

Cada agente puede invocar `web_search` en el MCP server tras formular su argumento inicial, integrando evidencia para refinarlo. La extracción de la query es defensiva: salta los headings (`AGREEMENTS:`, `REFINEMENT:`) para no buscar la propia etiqueta de sección.

### Archivos

- `agents/specialized/__main__.py` — servidor A2A; selecciona el modelo según `agent_id` desde `settings`.
- `agents/specialized/executor.py` — `SpecializedExecutor`: argumenta → busca web → refina → emite Task con metadata MCP.
- `agents/specialized/agent_state.py` — `AgentState` thread-safe con el system prompt deliberativo.
- `agents/specialized/config_api.py` — endpoints `/internal/configure` y `/internal/state`. Existen pero **no se usan en el path agéntico** (el planner pasa los roles por `description`).

---

## Feedback (Puerto 8084)

**Modelo (default):** `ollama/qwen2.5:14b` con fallback automático a Groq
**Skill:** `format_verdict`
**Rol:** Genera el informe final legible en castellano.

### Skill auto-descrito

> *"Produces the final human-readable report of a deliberation. Output is structured Markdown in Spanish. Use as the LAST step of any plan that needs a polished user-facing answer."*

### Estructura del informe

1. **Resumen ejecutivo** (2-3 frases).
2. **Participantes** (rol y perspectiva de los 3 agentes, identificando AE3 como mediador-evaluador independiente).
3. **Argumentos clave** por agente.
4. **Puntos de acuerdo**.
5. **Puntos de desacuerdo**.
6. **Veredicto final**.
7. **Estado del debate** — etiqueta clara: **Consenso alcanzado**, **Consenso parcial**, o **Sin consenso** (no es un "nivel de confianza" — es un análisis explícito de si hubo acuerdo real).

### Tolerancia a fallos

Si Ollama no responde (typically: container down, modelo sin descargar), el agente reintenta con el modelo del orquestador (Groq) automáticamente. Si ambos fallan, el orquestador cae al synth multi-sink interno (`SYNTHESIZE_PROMPT`).

### Archivos

- `agents/feedback/__main__.py` — servidor A2A.
- `agents/feedback/executor.py` — `FeedbackExecutor` con fallback Ollama → Groq.

---

## MCP Tools Server (Puerto 8085)

**Protocolo:** Model Context Protocol via `streamable-http`
**Rol:** Tools accesibles por los specialized agents.

### Tools

| Tool | Descripción | Ejemplo |
|---|---|---|
| `calculator` | Eval seguro de expresiones matemáticas | `calculator("sqrt(144) + 2**3")` → `20.0` |
| `web_search` | Búsqueda DuckDuckGo | `web_search("GDP Spain 2025")` |

### Seguridad de `calculator`

- Sólo permite dígitos, operadores, paréntesis y un set blanco de funciones (`sqrt`, `abs`, `round`, `min`, `max`, `log`, `pow`, `sin`, `cos`, `tan`, `pi`, `e`).
- Bloquea `import`, `__`, `exec`, `eval`, `open`.

### Archivos

- `agents/mcp_tools/server.py` — FastMCP server con los 2 tools.

---

## Cómo añadir un agente nuevo

1. Crea `agents/<nombre>/__main__.py` siguiendo el patrón de `normalizer` o `feedback`.
2. Define un `build_skill(...)` con una **`description` rica** que diga al planner: *cuándo usarlo*, *qué input espera*, *qué output produce*, y cualquier convención (input format, parallelism, etc.).
3. El proceso se auto-registra en el `AgentRegistry` al arrancar — no toca código del orquestador.
4. Añade el servicio a `docker-compose.yml` y al `start_all.sh`.
5. Si el modelo es configurable, añade el campo en `common/config.py` (`<agent>_model`) y exponlo en `.env.example`.

El planner verá el nuevo skill en el catálogo y razonará si usarlo basándose en la `description`.
