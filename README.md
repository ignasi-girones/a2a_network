# A2A Debate Network

Red de 5 agentes autónomos que debaten temas estructurados usando el protocolo **A2A (Agent-to-Agent) v1.0.0**. Cada agente utiliza un proveedor LLM distinto, demostrando agnosticismo de modelo e interoperabilidad entre agentes heterogéneos.

Proyecto académico desarrollado como Trabajo de Fin de Grado en Ingeniería Informática.

## Arquitectura

```
                          +-----------+
                          |  Frontend |  React 19 + Tailwind
                          |  :5173    |  SSE streaming
                          +-----+-----+
                                | SendStreamingMessage (JSON-RPC)
                                v
                    +-----------+-----------+
                    |    Orchestrator       |  Groq (Llama 3.3 70B)
                    |    :9000             |  Coordina todo el flujo
                    +-+-------+-------+----+
                      |       |       |
            A2A       |  A2A  |       |  A2A
                      v       v       v
              +-------+  +---+---+  +-+--------+
              |Normalizer| | AE1  |  |  AE2    |
              |:9001     | |:9002 |  | :9003   |
              |Gemini    | |Mistral|  |Cerebras |
              +----------+ +------+  +---------+
                                          |
                            A2A           v
                          +-------+  +---------+
                          | MCP   |  |Feedback |
                          |Tools  |  |:9004    |
                          |:8100  |  |Ollama   |
                          +-------+  +---------+
```

| Agente | Puerto | Modelo | Proveedor | Agent Card |
|--------|--------|--------|-----------|------------|
| Orchestrator | 9000 | `llama-3.3-70b-versatile` | Groq | Estática |
| Normalizer | 9001 | `gemini-2.5-flash` | Google Gemini | Estática |
| AE1 (Especializado) | 9002 | `mistral-large-latest` | Mistral AI | Dinámica |
| AE2 (Especializado) | 9003 | `qwen-3-235b` | Cerebras | Dinámica |
| Feedback | 9004 | `qwen2.5:14b` | Ollama (local) | Estática |
| MCP Tools | 8100 | - | - | - |

## Requisitos previos

- **Python 3.12+**
- **Node.js 20+** y npm
- **Ollama** ejecutando localmente con el modelo `qwen2.5:14b` (para el agente Feedback)
- Claves API gratuitas de: [Groq](https://console.groq.com/), [Google AI Studio](https://aistudio.google.com/), [Mistral](https://console.mistral.ai/), [Cerebras](https://cloud.cerebras.ai/)

## Instalación

### 1. Clonar y configurar entorno

```bash
git clone <repo-url>
cd a2a_network

# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instalar dependencias Python
pip install -e ".[dev]"
```

### 2. Instalar frontend

```bash
cd frontend
npm install
cd ..
```

### 3. Configurar API keys

```bash
cp .env.example .env
```

Edita `.env` y añade tus claves API:

```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=csk-...
```

### 4. Instalar modelo de Ollama (para Feedback agent)

```bash
ollama pull qwen2.5:14b
```

> Si tu máquina no soporta 14B, puedes cambiar `FEEDBACK_MODEL=ollama/qwen2.5:7b` en `.env`.

## Ejecución

### Windows

```cmd
start.bat
```

Opciones:
- `start.bat` — Todos los procesos en una terminal
- `start.bat --split` — Cada agente en su propia terminal
- `start.bat --stop` — Detener todos los agentes

### Linux / macOS

```bash
chmod +x start_all.sh
bash start_all.sh
```

### Docker Compose

```bash
docker compose up --build
```

### Acceso

Una vez iniciado, abre **http://localhost:5173** en el navegador.

## Uso

1. Escribe un tema de debate en el panel izquierdo (ej: *"¿Es mejor el trabajo remoto o presencial para equipos de desarrollo?"*)
2. El timeline de la derecha muestra el progreso en tiempo real vía SSE:
   - Normalización del input
   - Asignación de roles a los agentes
   - Opiniones iniciales de AE1 y AE2
   - Rondas de debate con argumentos cruzados
   - Evaluación de consenso
3. El veredicto final aparece en el panel izquierdo

## Estructura del proyecto

```
a2a_network/
├── agents/
│   ├── orchestrator/     # Coordinador del flujo de debate
│   ├── normalizer/       # Transforma input en JSON estructurado
│   ├── specialized/      # Agentes de debate (AE1, AE2) con roles dinámicos
│   ├── feedback/         # Genera informe final legible
│   └── mcp_tools/        # Servidor MCP con calculator y web_search
├── common/
│   ├── a2a_helpers.py    # Utilidades A2A: cards, clientes, comunicación
│   ├── config.py         # Configuración centralizada (Pydantic Settings)
│   ├── llm_provider.py   # Abstracción LLM via LiteLLM
│   └── models.py         # DTOs internos del sistema
├── frontend/             # React 19 + Vite + TailwindCSS
├── docs/                 # Documentación técnica detallada
│   ├── agents.md         # Descripción de cada agente
│   ├── flow.md           # Flujo del debate paso a paso
│   └── a2a-protocol.md   # Uso del protocolo A2A v1.0.0
├── docker-compose.yml
├── start.bat             # Launcher Windows
├── start_all.sh          # Launcher Linux/macOS
└── .env.example          # Plantilla de configuración
```

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [docs/agents.md](docs/agents.md) | Arquitectura y responsabilidades de cada agente |
| [docs/flow.md](docs/flow.md) | Flujo completo del debate con diagramas de secuencia |
| [docs/a2a-protocol.md](docs/a2a-protocol.md) | Decisiones de diseño sobre el protocolo A2A v1.0.0 |

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Protocolo inter-agente | A2A v1.0.0 (`a2a-sdk 1.0.0a0`) — JSON-RPC sobre HTTP |
| Herramientas MCP | FastMCP (`calculator`, `web_search`) |
| Abstracción LLM | LiteLLM — 5 proveedores simultáneos |
| Servidores | Starlette + Uvicorn (ASGI) |
| Frontend | React 19 + Vite 8 + TailwindCSS 4 |
| Streaming | SSE via `SendStreamingMessage` + `ReadableStream` |
| Validación | Pydantic v2 + Protocol Buffers |
| Contenedores | Docker Compose (7 servicios) |

## Licencia

Proyecto académico — Universidad.
