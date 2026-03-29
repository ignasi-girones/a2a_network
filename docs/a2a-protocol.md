# Protocolo A2A v1.0.0 — Decisiones de diseño

## Sobre el protocolo A2A

**Agent-to-Agent (A2A)** es un protocolo abierto que permite la comunicación entre agentes de IA de forma estandarizada, independientemente del framework o proveedor que los respalde. Fue publicado el 12 de marzo de 2026.

Este proyecto implementa la **versión 1.0.0** del protocolo usando el SDK alpha `a2a-sdk==1.0.0a0` para Python.

Referencia oficial: https://a2a-protocol.org

---

## Conceptos clave usados

### Agent Card

Cada agente publica un descriptor JSON en `/.well-known/agent-card.json` que describe sus capacidades:

```json
{
  "name": "Normalizer Agent",
  "description": "Analyzes and normalizes user prompts...",
  "supportedInterfaces": [{
    "url": "http://localhost:9001",
    "protocolBinding": "JSONRPC",
    "protocolVersion": "1.0"
  }],
  "provider": {"organization": "A2A Network - University Project"},
  "version": "1.0.0",
  "capabilities": {"streaming": false},
  "skills": [{
    "id": "normalize_input",
    "name": "Normalize Input",
    "description": "Transforms free-text into structured JSON..."
  }],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"]
}
```

### Agent Cards dinámicas

Los agentes AE1 y AE2 usan el callback `card_modifier` del SDK para modificar su card en cada request:

```python
app = A2AStarletteApplication(
    agent_card=initial_card,
    http_handler=handler,
    card_modifier=card_modifier,  # Callback nativo del SDK
)
```

Esto permite que la card refleje el rol actual asignado por el orquestador sin necesidad de reiniciar el agente.

### JSON-RPC Methods (v1.0.0)

La v1.0.0 usa **PascalCase** para los nombres de los métodos:

| Método | Descripción | Uso en este proyecto |
|--------|-------------|---------------------|
| `SendMessage` | Enviar mensaje y esperar respuesta completa | Orchestrator -> Normalizer/AE/Feedback |
| `SendStreamingMessage` | Enviar mensaje con respuesta SSE | Frontend -> Orchestrator |
| `GetTask` | Consultar estado de una tarea | No utilizado |
| `CancelTask` | Cancelar una tarea en curso | Disponible pero no expuesto en UI |

### Tipos protobuf (v1.0.0)

| Tipo | Uso |
|------|-----|
| `AgentProvider(organization=...)` | Identificar el proveedor en la card |
| `AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0")` | Declarar transporte soportado |
| `Role.ROLE_USER` / `Role.ROLE_AGENT` | Distinguir mensajes humanos de agente |
| `TaskState.TASK_STATE_WORKING` | Estado intermedio durante procesamiento |
| `TaskState.TASK_STATE_COMPLETED` | Tarea finalizada con éxito |
| `TaskState.TASK_STATE_FAILED` | Tarea fallida |
| `Part(text=...)` | Contenido textual de un mensaje |

### Streaming (SSE)

El frontend envía `SendStreamingMessage` via POST. El servidor responde con `Content-Type: text/event-stream` y emite eventos SSE progresivamente:

```
data: {"jsonrpc":"2.0","id":"...","result":{"statusUpdate":{"taskId":"...","status":{"state":"TASK_STATE_WORKING","message":{"role":"ROLE_AGENT","parts":[{"text":"{\"stage\":\"normalize\",\"message\":\"...\"}"}]}}}}}

data: {"jsonrpc":"2.0","id":"...","result":{"task":{"id":"...","status":{"state":"TASK_STATE_COMPLETED",...},"artifacts":[...]}}}
```

Cada evento SSE está envuelto en un `StreamResponse` con un campo discriminador:
- `statusUpdate` → `TaskStatusUpdateEvent` (progreso intermedio)
- `task` → `Task` (resultado final o error)

---

## Cambios respecto a v0.3

Este proyecto se construyó directamente sobre v1.0.0, pero durante el desarrollo se encontraron varios cambios respecto a la documentación basada en v0.3:

| Aspecto | v0.3 | v1.0.0 |
|---------|------|--------|
| Nombres de métodos | `message/send`, `message/stream` | `SendMessage`, `SendStreamingMessage` |
| Agent Card path | `/.well-known/agent.json` | `/.well-known/agent-card.json` |
| Provider field | `AgentProvider(name=...)` | `AgentProvider(organization=...)` |
| Protocol version | Campo en `AgentCapabilities` | Campo en `AgentInterface.protocol_version` |
| Protocol binding | `"jsonrpc"` (lowercase) | `"JSONRPC"` (uppercase, enum) |
| Part format | `{"kind": "text", "text": "..."}` | `{"text": "..."}` (sin campo `kind`) |
| Serialización | Pydantic models + JSON | Protobuf + `MessageToDict` |

---

## Decisiones de diseño

### 1. Por qué v1.0.0 y no v0.3

- v1.0.0 es la primera versión estable del protocolo
- Demuestra capacidad de trabajar con tecnología reciente (SDK alpha)
- Las Agent Cards dinámicas tienen soporte nativo via `card_modifier`
- Protobuf ofrece tipado más estricto que los modelos Pydantic de v0.3
- Mayor valor académico al documentar una implementación pionera

### 2. Por qué endpoint interno para configuración

Los agentes AE1/AE2 necesitan ser configurados con roles antes de participar en el debate. A2A no define un mecanismo para "configurar" un agente externamente — solo para enviarle mensajes de trabajo.

Se decidió crear endpoints REST internos (`/internal/configure`) **fuera del protocolo A2A** por:
- Separación clara entre comunicación A2A (debate) y gestión interna (configuración)
- El orquestador necesita configurar antes de enviar el primer mensaje A2A
- Evita contaminar el flujo A2A con mensajes de control

### 3. Por qué LiteLLM en lugar de SDKs nativos

LiteLLM actúa como capa de abstracción sobre 5 proveedores distintos con una interfaz OpenAI-compatible:

```python
# Misma función para cualquier proveedor
await acompletion(model="groq/llama-3.3-70b-versatile", messages=[...])
await acompletion(model="gemini/gemini-2.5-flash", messages=[...])
await acompletion(model="mistral/mistral-large-latest", messages=[...])
```

Ventajas:
- Código uniforme independiente del proveedor
- Cambiar modelo es cambiar un string en `.env`
- Retry y fallback centralizados
- Demuestra el principio de agnosticismo de modelo

### 4. Por qué SSE y no WebSocket

A2A v1.0.0 define `SendStreamingMessage` como un POST que retorna `text/event-stream`. El protocolo no utiliza WebSocket.

El frontend usa `fetch` + `ReadableStream` en lugar de `EventSource` porque:
- `EventSource` solo soporta GET, y A2A requiere POST con body JSON-RPC
- `ReadableStream` permite parsear chunks SSE incrementalmente
- Compatible con el proxy de Vite en desarrollo

### 5. Skills generadas por LLM vs skills estáticas

Las skills de los agentes AE1/AE2 se generan dinámicamente por el LLM del orquestador (fase de decisión de roles). Esto permite:

- Skills contextuales al tema del debate
- Demostración de que las agent cards pueden evolucionar en runtime
- Diferente de las skills estáticas de Normalizer y Feedback

Las skills no ejecutan código — son **descriptores de capacidades a nivel de razonamiento** que informan al LLM del agente sobre qué tipo de análisis puede realizar.

---

## Bibliotecas y componentes del SDK utilizados

| Componente | Import | Uso |
|-----------|--------|-----|
| `A2AStarletteApplication` | `a2a.server.apps` | Servidor A2A HTTP con rutas automáticas |
| `DefaultRequestHandler` | `a2a.server.request_handlers` | Handler por defecto para JSON-RPC |
| `InMemoryTaskStore` | `a2a.server.tasks` | Almacén de tareas en memoria |
| `AgentExecutor` | `a2a.server.agent_execution` | Clase base para la lógica de cada agente |
| `EventQueue` | `a2a.server.events` | Cola de eventos para streaming SSE |
| `A2ACardResolver` | `a2a.client` | Resuelve agent cards remotas |
| `ClientFactory` / `ClientConfig` | `a2a.client` | Crea clientes A2A para comunicación inter-agente |
| `create_text_message_object` | `a2a.client.helpers` | Helper para construir mensajes |
| Tipos protobuf | `a2a.types` | `AgentCard`, `Task`, `Part`, `Role`, `TaskState`, etc. |
