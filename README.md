# A2A Debate Network

Red de **6 agentes autónomos** que deliberan en torno a una pregunta del usuario usando el protocolo **A2A (Agent-to-Agent) v1.0.0**. Cada agente puede usar un proveedor LLM distinto, demostrando agnosticismo de modelo e interoperabilidad entre agentes heterogéneos.

A diferencia de un pipeline rígido, el sistema es **agéntico de extremo a extremo**: el orquestador no tiene los pasos hardcodeados. Pregunta a un *AgentRegistry* qué agentes están vivos, le pide a un *Planner* (LLM) que descomponga la pregunta en un grafo de subtareas, y un *PlanExecutor* dispatcha cada subtarea al agente que toque vía A2A.


## Arquitectura

```
                     +------------+
                     |  Frontend  |  React 19 + Tailwind, SSE streaming
                     |  :8086     |
                     +-----+------+
                           | A2A SendStreamingMessage
                           v
                     +-----+--------+
                     | Orchestrator |  Coordina:
                     |  :8080       |   - AgentRegistry (workers vivos)
                     |              |   - Planner (LLM → DAG)
                     |              |   - PlanExecutor (DAG → A2A calls)
                     |              |   - WorkerSpawner (dynamic scale-out)
                     |              |   - Consensus loop (LLM-graded)
                     +--+--+--+--+--+
                        |  |  |  |
                  A2A   |  |  |  |   A2A
              ┌─────────┘  |  |  └────────────┐
              v            v  v               v
        +------------+ +-----+ +-----+ +------+ +-----------+
        | Normalizer | | AE1 | | AE2 | | AE3  | | Feedback  |
        | :8081      | |:8082| |:8083| |:8087 | | :8084     |
        | Gemini     | |Mist.| |Cere.| |Groq  | | Ollama    |
        +------------+ +--+--+ +--+--+ +--+---+ +-----------+
                          |       |       |
                          └──MCP──┴──MCP──┘
                                  v
                            +-----------+
                            | MCP Tools |
                            |  :8085    |
                            +-----------+
```

| Agente | Puerto | Modelo (default) | Rol |
|---|---|---|---|
| Orchestrator | 8080 | `groq/llama-3.3-70b-versatile` | Planner LLM + ejecución del DAG + consensus loop |
| Normalizer | 8081 | `gemini/gemini-2.5-flash` | Convierte texto plano en JSON estructurado |
| AE1 | 8082 | `mistral/mistral-large-latest` | Debate (perspectiva 1) |
| AE2 | 8083 | `cerebras/qwen-3-235b-a22b-instruct-2507` | Debate (perspectiva opuesta) |
| AE3 | 8087 | `groq/llama-3.1-8b-instant` | Debate como **evaluador independiente** |
| Feedback | 8084 | `ollama/qwen2.5:14b` | Veredicto final formateado en castellano |
| MCP Tools | 8085 | — | `web_search`, `calculator` |

Los modelos se configuran en `.env` (ver `.env.example`). El frontend los carga en runtime vía `GET /models` y los muestra en la cabecera; si cambias `AE3_MODEL` y reinicias, el badge del frontend se actualiza solo.

## Cómo funciona el flujo agéntico

Cuando el usuario manda un prompt:

1. **Discover.** El orquestador consulta `AgentRegistry` (workers que se han auto-registrado al arrancar) y construye un catálogo `{agent_id, url, skills}`.
2. **Plan.** Llama al `Planner` (LLM) con el prompt + catálogo. El planner emite un `TaskPlan`: un DAG de subtareas en JSON donde cada subtarea declara su `required_skill`, sus `depends_on` y, si aplica, una `perspective` para enrutado por agente.
3. **Capacity check.** Si el plan necesita más workers concurrentes de un skill que los registrados, `WorkerSpawner` levanta workers extra como subprocess y espera a que se registren.
4. **Execute.** `PlanExecutor` recorre el DAG topológicamente: cada batch ready corre en paralelo (`asyncio.gather`), pinneando cada subtarea al worker que su `perspective` indique (o haciendo round-robin si no).
5. **Consensus loop.** Para preguntas deliberativas, después de la primera ronda de debate el orquestador llama a un LLM evaluador que devuelve `agreement_score`, `positions` por agente (eje 0..1) y listas de `shared_points` y `remaining_disagreements`. Si el score < 0.75 y queda presupuesto, pide al `Planner` una *extensión* — una nueva ronda paralela en la que cada agente re-evalúa honestamente; iterando hasta `MAX_CONSENSUS_EXTENSIONS = 3`.
6. **Finalize.** El orquestador añade un subtarea final de `format_verdict` que dispatcha al feedback agent — visible como un nodo más en el grafo del frontend.
7. **Synthesize.** El output del feedback (markdown en castellano con etiqueta de estado del debate) es el veredicto que se devuelve al usuario.

Nada de esto está hardcoded en el orquestador: si añades un worker nuevo con una skill nueva y una `description` clara, el planner la incorporará al razonar sobre nuevos prompts. Las skills se autodocumentan en sus AgentCards (qué hacen, cuándo usarlas, qué inputs esperan, etc.).

## Visualización en el frontend

- **Grafo del plan**: el DAG completo en SVG, con estado por nodo (pendiente / ejecutando / completado / fallido) y la `perspective` de cada subtask. Click en un nodo muestra el output completo del worker.
- **Evaluación del consenso**: gauge tipo velocímetro con el `agreement_score`, etiqueta (Sin consenso / Consenso parcial / Consenso alcanzado), sparkline de evolución, lista de puntos compartidos y desacuerdos.
- **Posicionamiento de los agentes**: gráfico de líneas con la trayectoria de cada agente en el eje AE1↔AE2 ronda a ronda.
- **Registro de eventos**: timeline completo de los eventos SSE emitidos por el orquestador (dispatch, done, consensus_check, plan_ready, etc.).
- **Veredicto**: el output del feedback agent en markdown.

## Requisitos

- Python 3.12+
- Node.js 20+ y npm
- (Opcional) Ollama corriendo localmente con `qwen2.5:14b` para el Feedback agent
- API keys gratuitas: [Groq](https://console.groq.com/), [Google AI Studio](https://aistudio.google.com/), [Mistral](https://console.mistral.ai/), [Cerebras](https://cloud.cerebras.ai/)

## Instalación rápida

```bash
git clone <repo-url>
cd a2a_network

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -e ".[dev]"

cd frontend && npm install && cd ..

cp .env.example .env
# Edita .env y rellena las API keys + modelos que quieras
```

`.env.example` ya lista las variables necesarias: API keys, puertos (8080–8087), y modelos de cada agente. Si no especificas un modelo, se usa el default que figura en `common/config.py`.

## Ejecución

### Linux/macOS/WSL — local (sin Docker)
```bash
bash start_all.sh
```

Cada agente loguea en `logs/<service>.log`. Para tail en vivo del orquestador:
```bash
tail -f logs/orchestrator.log
```

### Docker Compose
```bash
docker compose up -d --build
```

### Docker Compose — modo producción con mTLS
Toda la red interna se securiza con certificados autofirmados (CA `PTI_12.1`)
y mTLS bidireccional. El servicio `cert-init` genera la CA y un cert por
agente en `./certs/` la primera vez (idempotente):

```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up -d --build
```

Para que el navegador no muestre warning, importa `certs/ca.pem` en tu
almacén de Autoridades de Certificación de confianza (ver `certs/README.md`).

Tras eso, accede a **https://localhost:8086** (frontend) y **https://localhost:3000**
(Grafana) sin alertas de certificado.

### Acceso
Una vez arrancado, abre **http://localhost:8086** en el navegador.

## Estructura del proyecto

```
a2a_network/
├── agents/
│   ├── orchestrator/       # Planner LLM + executor agéntico
│   │   ├── agent_registry.py     # Registro dinámico de workers
│   │   ├── planner.py            # LLM → TaskPlan DAG
│   │   ├── plan_executor.py      # Recorre DAG, dispatch A2A
│   │   ├── agentic_orchestrator.py  # Director: junta todo
│   │   ├── worker_spawner.py     # Spawneo dinámico de workers
│   │   ├── models_routes.py      # GET /models (modelos por agente)
│   │   └── flow_manager.py       # legacy, no se usa en el path activo
│   ├── normalizer/         # Skill: normalize_input
│   ├── specialized/        # Skill: debate (instanciado como AE1, AE2, AE3)
│   ├── feedback/           # Skill: format_verdict
│   └── mcp_tools/          # Servidor MCP (web_search, calculator)
├── common/                 # config, llm_provider, models, registry_client
├── frontend/               # React 19 + Vite + Tailwind
├── docs/
│   ├── agents.md           # Detalle por agente
│   ├── flow.md             # Flujo agéntico paso a paso
│   └── a2a-protocol.md     # Decisiones sobre A2A v1.0.0
├── docker-compose.yml
├── start_all.sh
└── .env.example
```

## Documentación

| Documento | Contenido |
|---|---|
| [docs/agents.md](docs/agents.md) | Cada agente, su skill auto-descrito, modelo, archivos |
| [docs/flow.md](docs/flow.md) | Flujo agéntico: discover → plan → execute → consensus → finalize |
| [docs/a2a-protocol.md](docs/a2a-protocol.md) | Uso del protocolo A2A v1.0.0 |

## Stack

| Capa | Tecnología |
|---|---|
| Protocolo inter-agente | A2A v1.0.0 (`a2a-sdk 1.0.0a0`) — JSON-RPC sobre HTTP |
| Herramientas externas | MCP via FastMCP (`web_search`, `calculator`) |
| Abstracción LLM | LiteLLM — cualquier provider compatible |
| Orquestación | Pure Python — `asyncio.gather` sobre el DAG, sin frameworks |
| Servidores | Starlette + Uvicorn (ASGI) |
| Frontend | React 19 + Vite 8 + TailwindCSS 4 + SVG (sin chart libs) |
| Streaming | SSE via `SendStreamingMessage` + `ReadableStream` |
| Validación | Pydantic v2 |
| Contenedores | Docker Compose (8 servicios) |

## Licencia

Proyecto académico — Universidad.
