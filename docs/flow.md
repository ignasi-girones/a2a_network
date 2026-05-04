# Flujo agéntico del debate

El orquestador NO sigue un pipeline fijo. Cada decisión la toma o un LLM (planner / consensus-evaluator) o un cálculo basado en el estado real del registry. Lo único hardcoded es la **forma del flow** — discover → plan → execute → consensus loop → finalize → synthesize.

## Diagrama de secuencia

```
Frontend                Orchestrator                Workers
   |                         |                         |
   |--SendStreamingMessage-->|                         |
   |                         |                         |
   |                         | discover                |
   |                         |  registry.all_workers() |
   |                         |                         |
   |                         | plan (LLM call)         |
   |<--plan_ready-------------|                         |
   |                         |                         |
   |                         | _ensure_capacity        |
   |                         |  spawn extras if needed |
   |                         |                         |
   |                         | execute(plan)           |
   |                         |  for batch in DAG:      |
   |                         |    pin/round-robin      |
   |                         |--SendMessage----------->| Normalizer
   |<--subtask_dispatch-------|<--Task(COMPLETED)------|
   |<--subtask_done-----------|                         |
   |                         |                         |
   |                         |--SendMessage par.------>| AE1
   |                         |--SendMessage par.------>| AE2
   |                         |--SendMessage par.------>| AE3
   |                         |<--Task(COMPLETED) ×3----|
   |<--subtask_done × 3-------|                         |
   |                         |                         |
   |                         | consensus loop          |
   |                         |  LLM scores positions   |
   |<--agent_positions--------|                         |
   |<--consensus_check--------|                         |
   |                         |                         |
   |                         | if score < 0.75:        |
   |                         |   extend_for_consensus  |
   |                         |   execute extension     |
   |                         |                         |
   |                         | _finalize_with_feedback |
   |                         |--SendMessage----------->| Feedback
   |<--subtask_done-----------|<--Task(COMPLETED)------|
   |                         |                         |
   |                         | synthesize              |
   |<--Task(COMPLETED) verdict|                         |
```

## Fases detalladas

### Fase 1 — Discover

Actor: `AgenticOrchestrator.run()` → `AgentRegistry`.

El orquestador construye un catálogo con todos los workers que se han auto-registrado al arrancar:

```python
workers = await self.registry.all_workers()
catalog = [{"agent_id": w.agent_id, "url": w.url, "skills": w.card["skills"]} for w in workers]
```

Cada `skill` lleva su `id`, `name`, `description` y `tags`. Las descripciones son la **fuente de verdad** sobre cuándo y cómo usar cada skill.

**Evento SSE:** `{stage: "discover", message: "Consultando registry de workers..."}`

---

### Fase 2 — Plan

Actor: `Planner.create_plan(prompt, catalog)`.

LLM call con `PLANNER_SYSTEM_PROMPT` + el catálogo serializado. El prompt **no menciona ninguna skill por nombre** — instruye al LLM a leer las descripciones del catálogo y razonar sobre ellas. El output es un `TaskPlan` JSON validado por Pydantic:

```json
{
  "goal": "Compare ...",
  "subtasks": [
    {"id":"t1", "description":"...", "required_skill":"normalize_input", "depends_on":[], "perspective":null},
    {"id":"t2", "description":"ROLE: ...", "required_skill":"debate", "depends_on":["t1"], "perspective":"ae1: ..."},
    {"id":"t3", "description":"ROLE: ...", "required_skill":"debate", "depends_on":["t1"], "perspective":"ae2: ..."},
    {"id":"t4", "description":"ROLE: ...", "required_skill":"debate", "depends_on":["t1"], "perspective":"ae3: Independent evaluator"}
  ],
  "max_workers": 3
}
```

**Validación**: hasta 3 reintentos si el JSON es inválido o si `required_skill` no existe en el catálogo. Fallback determinista (linear normalize → debate × 2 → format_verdict) si los reintentos fallan.

**Evento SSE:** `{stage: "plan_ready", data: {plan: {...}}}`

---

### Fase 3 — Capacity check

Actor: `_ensure_capacity(plan)` + `WorkerSpawner`.

Calcula la **demanda concurrente máxima por skill** recorriendo el DAG (todas las subtareas ready a la vez en algún punto). Si excede los workers registrados, llama a `spawner.spawn(agent_id)` por cada déficit, asignando un puerto del pool dinámico (9010+) y lanzando `python -m agents.specialized` como subprocess. El nuevo proceso se auto-registra y entra al pool.

Al final del run, los workers spawneados se descartan (`teardown`).

**Evento SSE (si spawn):** `{stage: "spawn", message: "Spawneando 1 worker(s) extra para skill 'debate'"}`

---

### Fase 4 — Execute

Actor: `PlanExecutor.execute(plan)`.

Bucle topológico:

```python
while pending:
    ready = [t for t in pending if all(d in results for d in t.depends_on)]
    assignments = await self._assign_workers(ready)
    outputs = await asyncio.gather(*[self._execute_subtask(t, assignments[t.id], ...) for t in ready])
```

**Asignación de workers** (`_assign_workers`):
- Si `perspective` empieza por un `agent_id` registrado (ej: `"ae1: ..."`), se *pina* la subtask a ese worker concreto. Esto garantiza que cada AE mantiene su identidad LLM ronda tras ronda.
- Si no hay pinning, round-robin sobre los workers disponibles para ese skill.
- Si no hay ningún worker para el skill → `PlanExecutionError` (el orquestador reportará la falla y abortará).

**Construcción del prompt** (`_build_subtask_prompt`):
- Para subtareas no-debate: dump plano de `[dep_id]\n<output>` por cada dependencia.
- Para subtareas debate con `aeN:` perspective: las deps se clasifican en `[Your previous arguments]` / `[OTROS_AGENTES's arguments]` / `[Shared context]`. Esto le da a cada agente la trayectoria estructurada de su propio razonamiento + lo que dijeron los demás.

**A2A dispatch**: cada subtask invoca `send_and_get_text(client, prompt)` que abre un cliente A2A JSON-RPC al worker, manda `SendMessage` y devuelve el texto de la `Task(COMPLETED)`.

**Eventos SSE por subtask:** `subtask_dispatch` → `tool_use` (si el worker invoca MCP) → `subtask_done` con el texto completo.

---

### Fase 5 — Consensus loop (evaluación empírica)

Actor: `_consensus_loop(plan, results, catalog)` + `consensus_metrics.compute_metrics`.

A diferencia de versiones anteriores, **el `agreement_score` y las posiciones por agente NO los decide un LLM**. Se calculan algorítmicamente desde los textos:

#### Anchoring (una sola vez, al entrar al loop)

Los textos de **apertura** de AE1 y AE2 se embedean (vía LiteLLM `aembedding`) y se usan como anchors fijos:
- AE1 opening = posición `0.0` por definición.
- AE2 opening = posición `1.0` por definición.

#### Por iteración (hasta `MAX_CONSENSUS_EXTENSIONS = 3`)

1. **Identifica las latest debates por agente** (`_latest_debate_per_agent`).
2. **Embedea los 3 textos actuales** y calcula posiciones empíricas:
   ```
   position_i = sim(text_i, AE2_anchor) / (sim(text_i, AE1_anchor) + sim(text_i, AE2_anchor))
   ```
   - Texto idéntico a apertura AE1 → posición ≈ 0.0.
   - Texto idéntico a apertura AE2 → posición ≈ 1.0.
   - Texto equidistante → 0.5.

   Esta fórmula garantiza que si AE1 cambia de bando hacia el lado de AE2, su posición se mueve hacia 1.0 — sin que el LLM necesite "darse cuenta".

3. **Calcula 4 métricas en `[0, 1]`** ([consensus_metrics.py](agents/orchestrator/consensus_metrics.py)):
   - **Cohesión** (`1 − dispersión`): cuán cerca están los 3 agentes en el eje.
   - **Similitud**: similitud coseno media de los embeddings entre cada par de agentes.
   - **Movimiento**: cuánto se desplazó cada agente desde la ronda anterior, normalizado y saturado a ~0.25.
   - **Concesiones**: marcadores explícitos como "me has convencido", "tienes razón en X", "you changed my mind", contados con regex bilingüe ES+EN.

4. **`agreement_score` combinado**:
   ```
   agreement_score = 0.40 · cohesión
                   + 0.35 · similitud
                   + 0.15 · movimiento
                   + 0.10 · concesiones
   ```

5. **LLM call separada** — solo para extraer los textos de `shared_points` y `remaining_disagreements` (que no tienen forma cerrada algorítmica). El LLM **no opina** sobre el score ni sobre las posiciones.

6. **Decisión**:
   - Si `score ≥ 0.75` → consenso, salir del loop.
   - Si `attempt + 1 ≥ MAX` → cap agotado, salir.
   - Si no, `Planner.extend_for_consensus` emite una mini-extensión paralela (un subtask por agente con deps idénticos a las latest del round anterior).

7. La extensión se ejecuta y se emite `plan_ready` con el plan mergeado.

**Tolerancia a fallos:** si la API de embeddings falla (ej: Ollama caído), el orquestador degrada a un fallback con posiciones por defecto (AE1=0, AE2=1, AE3=0.5) y score 0 — el debate seguirá hasta agotar el budget de extensiones sin declarar consenso, que es el comportamiento seguro.

**Eventos SSE por iteración:** `agent_positions` (con `components`, `movement`, `concessions`) → `consensus_check` → (si extiende) `extend_plan` → `plan_ready` → batches de la extensión → vuelta al inicio.

---

### Fase 6 — Finalize

Actor: `_finalize_with_feedback(plan, results)`.

Si hay un worker con skill `format_verdict` registrado y el plan tuvo un debate real (latest ae1 + ae2 existen), el orquestador añade una subtask sintética `final_verdict` con:
- `required_skill: "format_verdict"`
- `depends_on: [latest_ae1, latest_ae2, latest_ae3]`
- una descripción que pide un veredicto bien estructurado en castellano

Y la ejecuta como un mini-plan vía `PlanExecutor.execute`. Esto:
- Hace que el feedback agent sea visible en el grafo del frontend (un nodo más).
- Convierte el sink del DAG en un único nodo (en vez de 3 sinks dispersos).
- Si el feedback agent falla (rate limit, Ollama caído), captura la excepción y cae al `_synthesize` interno.

**Evento SSE:** `plan_ready` (con la subtask final añadida) → `subtask_dispatch` → `subtask_done` para `final_verdict`.

---

### Fase 7 — Synthesize

Actor: `_synthesize(user_input, plan, results)`.

Estrategia:
- Si hay **un solo sink** (típicamente el `final_verdict` del paso anterior), su output es el veredicto.
- Si hay **múltiples sinks** (porque el feedback falló o no se ejecutó), llama al LLM del orquestador con `SYNTHESIZE_PROMPT` para combinar todos los outputs en un answer en castellano.

El texto resultante se devuelve como `Task(COMPLETED)` con un `Artifact` etiquetado como `verdict`.

**Eventos SSE final:** `synthesize` → `complete` (con el verdict).

---

## Comunicación entre agentes

| De → A | Protocolo | Método |
|---|---|---|
| Frontend → Orchestrator | A2A JSON-RPC | `SendStreamingMessage` |
| Orchestrator → Workers | A2A JSON-RPC | `SendMessage` |
| Workers → Orchestrator | HTTP POST | `/registry/register` (al arrancar) |
| Frontend → Orchestrator | HTTP GET | `/models` (al cargar) |
| Specialized agents → MCP server | MCP streamable-http | `call_tool` |
| Orchestrator → LLM (planner / consensus / synthesis) | LiteLLM | Llamada directa (no A2A) |

---

## Tolerancia a fallos

| Punto de fallo | Estrategia |
|---|---|
| LLM del planner devuelve JSON inválido | 3 reintentos con corrective message |
| Planner inventa un skill no registrado | Validación post-parse, retry |
| Extensión con deps externos | `external_ids` permite referenciar IDs del plan original |
| Extensión secuencial (x2 depende de x1) | Validación post-parse, retry exigiendo paralelismo |
| Score del consenso incoherente con positions | No aplica — el score se deriva de las positions algorítmicamente |
| Embeddings caen (Ollama/provider) | Fallback a positions por defecto (AE1=0, AE2=1, AE3=0.5), score 0; debate sigue hasta agotar extensiones |
| LLM del normalizer falla (rate limit/timeout) | Estructura mínima como fallback |
| Ollama no disponible (Feedback) | Fallback automático a Groq |
| Feedback agent falla en finalize | Cae al synth multi-sink interno |
| Worker no responde (timeout) | `PlanExecutionError`, propagada al frontend como Task FAILED |
| Demanda concurrente > workers registrados | `WorkerSpawner` lanza extras dinámicamente |
