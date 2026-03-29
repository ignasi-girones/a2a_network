# Agentes del sistema

La red consta de 5 agentes A2A y un servidor MCP de herramientas. Cada agente es un servidor HTTP independiente que expone una **Agent Card** (descriptor de capacidades) y procesa mensajes mediante el protocolo JSON-RPC definido por A2A v1.0.0.

---

## Orchestrator (Puerto 9000)

**Modelo:** Groq / Llama 3.3 70B Versatile
**Agent Card:** Estática
**Rol:** Coordinador central del flujo de debate.

Es el **único agente agentic** del sistema — tiene capacidad de toma de decisiones autónoma respaldada por un LLM. El resto de agentes son reactivos (ejecutan una tarea concreta cuando reciben un mensaje).

### Responsabilidades

1. Recibir el prompt del usuario desde el frontend via `SendStreamingMessage`
2. Delegar la normalización al agente Normalizer via A2A
3. Decidir roles y perspectivas para AE1 y AE2 usando su propio LLM
4. Configurar dinámicamente los agentes especializados via API interna
5. Recoger opiniones iniciales en paralelo (A2A a AE1 y AE2)
6. Gestionar las rondas de debate (máximo configurable)
7. Evaluar consenso tras cada ronda
8. Generar un resumen y enviarlo al agente Feedback
9. Emitir eventos SSE de progreso al frontend durante todo el proceso

### Flujo de comunicación

```
Frontend ──SSE──> Orchestrator ──A2A──> Normalizer
                      │
                      ├──A2A──> AE1 (opinión / debate)
                      ├──A2A──> AE2 (opinión / debate)
                      │
                      └──A2A──> Feedback (resumen → veredicto)
```

### Llamadas LLM internas (no A2A)

El orquestador usa su LLM directamente (sin pasar por A2A) para:

- **Decisión de roles:** Analiza el tema normalizado y asigna roles contrastantes a AE1/AE2 (ej: "DevOps Engineer" vs "Team Lead")
- **Evaluación de consenso:** Tras cada ronda, evalúa si las posiciones convergen
- **Generación de resumen:** Sintetiza el debate completo antes de enviarlo a Feedback

### Streaming SSE

El orquestador emite `TaskStatusUpdateEvent` con metadata JSON en cada hito del debate. El frontend recibe estos eventos en tiempo real para mostrar el timeline.

```json
{
  "stage": "ae1_argues",
  "message": "Ronda 1: AE1 responde",
  "data": {"agent": "ae1", "round": 1, "text": "...argumento..."}
}
```

### Archivos

| Archivo | Descripción |
|---------|-------------|
| `agents/orchestrator/__main__.py` | Servidor A2A con CORS para frontend |
| `agents/orchestrator/executor.py` | `OrchestratorExecutor` + `SSEProgressCallback` |
| `agents/orchestrator/flow_manager.py` | `FlowManager` — lógica completa del debate |

---

## Normalizer (Puerto 9001)

**Modelo:** Google Gemini 2.5 Flash
**Agent Card:** Estática
**Rol:** Transformar input libre del usuario en JSON estructurado.

### Entrada / Salida

**Entrada:** Texto libre del usuario (ej: *"¿Es mejor invertir en acciones o bonos?"*)

**Salida:** JSON normalizado:

```json
{
  "topic": "Comparación entre inversión en acciones y bonos",
  "domain": "finance",
  "question_type": "comparison",
  "constraints": [],
  "suggested_perspectives": [
    "Las acciones ofrecen mayor rentabilidad a largo plazo...",
    "Los bonos proporcionan estabilidad y menor riesgo..."
  ]
}
```

### Tolerancia a fallos

- 2 reintentos si el LLM no devuelve JSON válido
- Fallback a estructura mínima: `{"topic": "<input original>", "domain": "general", ...}`

### Archivos

| Archivo | Descripción |
|---------|-------------|
| `agents/normalizer/__main__.py` | Servidor A2A en puerto 9001 |
| `agents/normalizer/executor.py` | `NormalizerExecutor` con prompt de extracción JSON |

---

## Specialized Agents — AE1 y AE2 (Puertos 9002, 9003)

**Modelos:** Mistral Large (AE1) / Cerebras Qwen 3 235B (AE2)
**Agent Card:** Dinámica (cambia según rol asignado)
**Rol:** Debatir desde perspectivas opuestas asignadas por el orquestador.

### Agent Cards dinámicas

Son los únicos agentes con **agent cards dinámicas** en el sistema. Usan el callback `card_modifier` nativo del SDK A2A:

```python
async def card_modifier(card: AgentCard) -> AgentCard:
    role = await state.get_role()
    card.name = f"Specialized Agent (AE1) - {role}"
    card.description = f"Agent configured as: {role}"
    card.skills = [...]  # Skills generadas por el LLM del orquestador
    return card
```

Antes de la configuración, la card dice *"Awaiting role assignment"*. Después del `/internal/configure`, refleja el rol asignado (ej: *"Financial Analyst — conservative, risk-averse"*).

### API interna de configuración

Endpoint **fuera del protocolo A2A** para inyectar el rol dinámicamente:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/internal/configure` | POST | Asigna rol, perspectiva y skills |
| `/internal/state` | GET | Consulta estado actual del agente |

**Payload de configuración:**

```json
{
  "role": "Financial Analyst",
  "perspective": "Conservative, prioritizes risk management over growth",
  "skills": [
    {"id": "risk_assessment", "name": "Risk Assessment"},
    {"id": "market_analysis", "name": "Market Analysis"}
  ]
}
```

### System prompt

El system prompt se construye automáticamente a partir del rol y perspectiva:

> *"You are a Financial Analyst participating in a structured debate. Your perspective: Conservative, prioritizes risk management over growth. Make clear, well-reasoned arguments. Be concise (3-5 paragraphs max). Engage directly with the opposing argument when responding."*

### Estado mutable

`AgentState` protege el estado con `asyncio.Lock` para seguridad en concurrencia:

```python
class AgentState:
    role: str          # Rol asignado (ej: "DevOps Engineer")
    perspective: str   # Perspectiva del debate
    skills: list       # Skills dinámicas
    system_prompt: str # Prompt construido automáticamente
    ready: bool        # True tras configuración
```

### Archivos

| Archivo | Descripción |
|---------|-------------|
| `agents/specialized/__main__.py` | Servidor A2A con `card_modifier` y rutas internas |
| `agents/specialized/executor.py` | `SpecializedExecutor` — lee prompt de AgentState |
| `agents/specialized/agent_state.py` | Estado mutable thread-safe |
| `agents/specialized/config_api.py` | API REST interna (`/internal/configure`) |

---

## Feedback (Puerto 9004)

**Modelo:** Ollama / Qwen 2.5 14B (local) con fallback a Groq
**Agent Card:** Estática
**Rol:** Generar un informe final legible para humanos.

### Formato del informe

El agente recibe el `FlowResult` completo (JSON con todo el debate) y produce un informe Markdown con:

1. **Resumen ejecutivo** (2-3 frases)
2. **Participantes** (roles y perspectivas de cada agente)
3. **Argumentos clave** (los más fuertes de cada lado)
4. **Puntos de acuerdo**
5. **Puntos de desacuerdo**
6. **Veredicto final** (conclusión equilibrada)
7. **Nivel de confianza** (Alto / Medio / Bajo)

### Tolerancia a fallos

Si Ollama no está disponible o falla, el agente automáticamente usa Groq como proveedor de respaldo:

```python
except Exception:
    # Fallback to orchestrator model (Groq)
    result = await llm_complete(model=settings.orchestrator_model, ...)
```

### Archivos

| Archivo | Descripción |
|---------|-------------|
| `agents/feedback/__main__.py` | Servidor A2A en puerto 9004 |
| `agents/feedback/executor.py` | `FeedbackExecutor` con prompt de formateo |

---

## MCP Tools Server (Puerto 8100)

**Protocolo:** Model Context Protocol (MCP) via `streamable-http`
**Rol:** Proveer herramientas estáticas accesibles por los agentes.

### Herramientas disponibles

| Tool | Descripción | Ejemplo |
|------|-------------|---------|
| `calculator` | Evaluación segura de expresiones matemáticas | `calculator("sqrt(144) + 2**3")` → `20.0` |
| `web_search` | Búsqueda web via API de DuckDuckGo | `web_search("GDP Spain 2025")` → resultados |

### Seguridad

La herramienta `calculator` implementa sanitización de input:
- Solo permite: dígitos, operadores, paréntesis, funciones matemáticas
- Bloquea: `import`, `__`, `exec`, `eval`, `open`
- Funciones permitidas: `sqrt`, `abs`, `round`, `min`, `max`, `log`, `pow`, `sin`, `cos`, `tan`, `pi`, `e`

### Archivos

| Archivo | Descripción |
|---------|-------------|
| `agents/mcp_tools/server.py` | Servidor FastMCP con 2 tools |
