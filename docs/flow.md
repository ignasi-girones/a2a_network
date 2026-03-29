# Flujo del debate

## Diagrama de secuencia

```
 Usuario        Frontend       Orchestrator     Normalizer      AE1           AE2          Feedback
    |               |               |               |             |             |             |
    |--"tema"------>|               |               |             |             |             |
    |               |--SSE POST---->|               |             |             |             |
    |               |               |               |             |             |             |
    |               |  [normalize]  |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |--SendMessage->|             |             |             |
    |               |               |<---JSON-------|             |             |             |
    |               |               |               |             |             |             |
    |               |  [roles]      |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |---LLM call--->| (Groq)      |             |             |
    |               |               |<--roles JSON--|             |             |             |
    |               |               |               |             |             |             |
    |               | [configure]   |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |--POST /internal/configure-->|             |             |
    |               |               |--POST /internal/configure-------------->|             |
    |               |               |               |             |             |             |
    |               | [opinions]    |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |--SendMessage-------------->|             |             |
    |               |               |--SendMessage---------------------------->|             |
    |               |               |<---opinion----|------------|             |             |
    |               |               |<---opinion-------------------------|-----|             |
    |               |               |               |             |             |             |
    |               | [debate]      |               |             |             |             |
    |           +-->|<--progress----|               |             |             |             |
    |           |   |               |--"respond to AE1"---------->|             |             |
    |           |   |               |<---argument----|------------|             |             |
    |  Rondas   |   |<--ae2_argues--|               |             |             |             |
    |  1..N     |   |               |--"respond to AE2"------------------------->|           |
    |           |   |               |<---argument----|---------------------------|           |
    |           |   |<--ae1_argues--|               |             |             |             |
    |           |   |               |               |             |             |             |
    |           |   |               |---LLM: consensus check?    |             |             |
    |           +---|               |<--{consensus: false}        |             |             |
    |               |               |               |             |             |             |
    |               | [summary]     |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |---LLM: generate summary    |             |             |
    |               |               |               |             |             |             |
    |               | [feedback]    |               |             |             |             |
    |               |<--progress----|               |             |             |             |
    |               |               |--SendMessage------------------------------------------->|
    |               |               |<---verdict report----------------------------------------|
    |               |               |               |             |             |             |
    |               | [complete]    |               |             |             |             |
    |               |<--COMPLETED---|               |             |             |             |
    |<--veredicto---|               |               |             |             |             |
```

## Fases detalladas

### Fase 1 — Normalización

**Actor:** Orchestrator -> Normalizer (via A2A `SendMessage`)

El orquestador envía el texto libre del usuario al Normalizer, que lo transforma en JSON estructurado usando Gemini.

**Input:** `"¿Es mejor el trabajo remoto o presencial para equipos de desarrollo?"`

**Output:**
```json
{
  "topic": "Comparación entre trabajo remoto y presencial para equipos de desarrollo",
  "domain": "technology",
  "question_type": "comparison",
  "constraints": [],
  "suggested_perspectives": [
    "El trabajo remoto mejora productividad y calidad de vida...",
    "El trabajo presencial fortalece comunicación y cultura de equipo..."
  ]
}
```

**Evento SSE:** `{stage: "normalize", message: "Normalizando entrada..."}`

---

### Fase 2 — Decisión de roles

**Actor:** Orchestrator (llamada LLM directa a Groq, sin A2A)

El orquestador analiza el topic normalizado y decide qué roles profesionales y perspectivas asignar a cada agente. Los roles deben ser **contrastantes** para producir un debate productivo.

**Output ejemplo:**
```json
{
  "ae1_config": {
    "role": "DevOps Engineer",
    "perspective": "Advocates remote work for technical productivity",
    "skills": [{"id": "infra_analysis", "name": "Infrastructure Analysis"}]
  },
  "ae2_config": {
    "role": "Team Lead",
    "perspective": "Values in-person collaboration for team cohesion",
    "skills": [{"id": "team_management", "name": "Team Management"}]
  },
  "max_rounds": 2
}
```

**Evento SSE:** `{stage: "roles_decided", message: "AE1: DevOps Engineer | AE2: Team Lead", data: {ae1_role: ..., ae2_role: ...}}`

---

### Fase 3 — Configuración dinámica de agentes

**Actor:** Orchestrator -> AE1, AE2 (via HTTP POST, fuera de A2A)

El orquestador envía la configuración a cada agente especializado a través del endpoint interno `/internal/configure`. Este paso:

1. Actualiza el `AgentState` (rol, perspectiva, skills)
2. Construye el system prompt automáticamente
3. Modifica la Agent Card dinámica que sirve el agente

Las dos configuraciones se envían **en paralelo** (`asyncio.gather`).

**Evento SSE:** `{stage: "configure", message: "Configurando agentes especializados..."}`

---

### Fase 4 — Opiniones iniciales

**Actor:** Orchestrator -> AE1, AE2 (via A2A `SendMessage`, en paralelo)

Ambos agentes reciben el mismo prompt basado en el topic normalizado y formulan su posición inicial desde su perspectiva asignada.

Las dos llamadas se ejecutan **en paralelo** para reducir latencia.

**Prompt enviado:**
> *"Analyze this topic from your assigned perspective and formulate your initial position: {normalized_json}"*

**Eventos SSE:**
```
{stage: "ae1_opinion", data: {agent: "ae1", text: "..."}}
{stage: "ae2_opinion", data: {agent: "ae2", text: "..."}}
```

---

### Fase 5 — Rondas de debate

**Actor:** Orchestrator -> AE2 -> AE1 (secuencial, via A2A)

En cada ronda:
1. **AE2 responde** al último argumento de AE1
2. **AE1 responde** al argumento de AE2

Los agentes reciben el argumento del oponente con el prompt:
> *"Respond to this argument from the opposing side: {argumento}"*

**Eventos SSE por ronda:**
```
{stage: "debate_round", data: {round: 1}}
{stage: "ae2_argues", data: {agent: "ae2", round: 1, text: "..."}}
{stage: "ae1_argues", data: {agent: "ae1", round: 1, text: "..."}}
```

---

### Fase 6 — Evaluación de consenso

**Actor:** Orchestrator (llamada LLM directa, sin A2A)

Tras cada ronda, el orquestador evalúa si las posiciones de ambos agentes convergen sustancialmente:

```json
{"consensus": false, "reason": "AE1 still prioritizes remote productivity while AE2 insists on in-person culture"}
```

Si hay consenso, el debate se detiene. Si no, continúa hasta `max_rounds`.

**Evento SSE (si consenso):** `{stage: "consensus", message: "Consenso alcanzado en ronda 2"}`

---

### Fase 7 — Resumen

**Actor:** Orchestrator (llamada LLM directa)

El orquestador genera un resumen textual de todo el debate: opiniones iniciales, argumentos de cada ronda, y si se alcanzó consenso.

**Evento SSE:** `{stage: "summary", message: "Generando resumen del debate..."}`

---

### Fase 8 — Veredicto final

**Actor:** Orchestrator -> Feedback (via A2A `SendMessage`)

El orquestador envía el `FlowResult` completo (serializado como JSON) al agente Feedback, que genera un informe estructurado legible para humanos.

El informe incluye: resumen ejecutivo, participantes, argumentos clave, puntos de acuerdo/desacuerdo, veredicto y nivel de confianza.

**Evento SSE final:** `{stage: "complete", message: "Debate completado", data: {verdict: "..."}}`

---

## Comunicación entre agentes

### Protocolos utilizados

| Comunicación | Protocolo | Método |
|-------------|-----------|--------|
| Frontend -> Orchestrator | A2A JSON-RPC | `SendStreamingMessage` (SSE) |
| Orchestrator -> Normalizer | A2A JSON-RPC | `SendMessage` |
| Orchestrator -> AE1/AE2 (debate) | A2A JSON-RPC | `SendMessage` |
| Orchestrator -> Feedback | A2A JSON-RPC | `SendMessage` |
| Orchestrator -> AE1/AE2 (config) | HTTP REST | `POST /internal/configure` |
| Orchestrator -> LLM (roles/consenso) | LiteLLM | Llamada directa |

### Resolución de Agent Cards

Antes de cada comunicación A2A, el cliente resuelve la Agent Card del destino:

```
GET /.well-known/agent-card.json → AgentCard (JSON)
```

La card contiene: nombre, descripción, capabilities, skills, y el `supportedInterfaces` con `protocolBinding: "JSONRPC"` y `protocolVersion: "1.0"`.

Para los agentes especializados, la card se modifica dinámicamente según el rol asignado.

---

## Tolerancia a fallos

| Punto de fallo | Estrategia |
|----------------|-----------|
| LLM no devuelve JSON válido | Reintentos (2) con prompt correctivo |
| Rate limit del proveedor LLM | Retry con backoff (10s, 25s) |
| Normalizer falla parsing | Fallback a estructura mínima |
| Ollama no disponible (Feedback) | Fallback automático a Groq |
| Agente no responde (timeout) | Timeout configurable de 120s |
| JSON-RPC parse fail (AE) | Error devuelto como texto con contexto |
