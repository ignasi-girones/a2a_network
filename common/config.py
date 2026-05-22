from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # LLM Provider API Keys
    groq_api_key: str = ""
    gemini_api_key: str = ""
    mistral_api_key: str = ""
    cerebras_api_key: str = ""

    # Agent Ports.
    #
    # Mapped to match the UPC FIB VM external port tunnel:
    #   nattech.fib.upc.edu:40530 → 172.16.4.53:8080 (orchestrator)
    #   nattech.fib.upc.edu:40531 → 172.16.4.53:8081 (normalizer)
    #   nattech.fib.upc.edu:40532 → 172.16.4.53:8082 (ae1)
    #   nattech.fib.upc.edu:40533 → 172.16.4.53:8083 (ae2)
    #   nattech.fib.upc.edu:40534 → 172.16.4.53:8084 (feedback)
    #   nattech.fib.upc.edu:40535 → 172.16.4.53:8085 (mcp-tools)
    #   nattech.fib.upc.edu:40536 → 172.16.4.53:8086 (frontend)
    #   nattech.fib.upc.edu:40537-40539 → 8087-8089 (spare)
    orchestrator_port: int = 8080
    normalizer_port: int = 8081
    ae1_port: int = 8082
    ae2_port: int = 8083
    feedback_port: int = 8084
    mcp_port: int = 8085
    frontend_port: int = 8086
    ae3_port: int = 8087

    # Hostnames — in local dev everything is on localhost, but when deploying
    # to Docker Compose / Kubernetes each agent reaches the orchestrator via
    # the service name (e.g. "orchestrator") and advertises itself to the
    # registry using its own service name (e.g. "ae1", "normalizer"). Both
    # default to "localhost" so `python -m agents.orchestrator` still works
    # unchanged on a developer machine.
    orchestrator_host: str = "localhost"
    self_host: str = "localhost"
    mcp_host: str = "localhost"

    # LLM Models
    orchestrator_model: str = "gemini/gemini-2.5-flash"
    normalizer_model: str = "gemini/gemini-2.5-flash"
    ae1_model: str = "mistral/mistral-large-latest"
    ae2_model: str = "cerebras/llama3.1-8b"
    ae3_model: str = "groq/llama-3.1-8b-instant"
    feedback_model: str = "ollama/qwen2.5:14b"

    # Embedding model used by the empirical consensus metrics. The orchestrator
    # uses this to anchor each agent's position on the AE1↔AE2 axis to the
    # cosine similarity between the agent's current text and the opening
    # texts of AE1/AE2 — instead of asking an LLM to subjectively place each
    # agent on the axis. Any LiteLLM-supported embedding model works.
    #
    # Default `gemini/gemini-embedding-2` is the optimal Gemini embedding
    # available on AI Studio: 8192-token input window (debate texts can grow
    # to several paragraphs × 3 agents in late rounds), stable (not preview),
    # and the newest non-preview version. Only needs GEMINI_API_KEY which
    # the rest of the stack already requires.
    embedding_model: str = "gemini/gemini-embedding-2"

    # Ollama
    ollama_api_base: str = "http://localhost:11434"

    # Debate
    max_debate_rounds: int = 5

    # Dynamic worker pool (sub-phase 2c). When the Planner produces a task that
    # requires more concurrent workers of a given skill than are currently
    # registered, the WorkerSpawner allocates a port from this pool and
    # launches a new specialized worker subprocess.
    #
    # These workers are spawned inside the orchestrator container and advertise
    # themselves via Docker-internal DNS (http://orchestrator:<port>), so they
    # don't need to be published on the host — any port range not already used
    # by a published service is fine. We pick 9010+ so dynamic workers never
    # clash with the 8080-8089 external tunnel range.
    worker_port_pool_start: int = 9010
    worker_port_pool_size: int = 20

    # Telemetry (Prometheus). When enabled, each agent exposes a /metrics
    # endpoint and wraps LLM/MCP calls with timing instrumentation.
    telemetry_enabled: bool = True

    # TLS (mTLS in production). When `tls_enabled=true`, every agent serves
    # HTTPS using the cert at `tls_cert_dir/<service>.pem` and *requires*
    # peer client certs signed by `tls_cert_dir/ca.pem`. Outgoing httpx
    # calls also present the agent's own cert. The matching cert is picked
    # up from `TLS_SERVICE_NAME` env (set per container in the production
    # overlay) or `SELF_HOST` as a fallback.
    #
    # Default is `false` so local dev (start.bat, base docker-compose.yml)
    # keeps using plain HTTP — TLS only activates with the production
    # overlay (`docker-compose.production.yml`), which sets these env vars
    # explicitly.
    tls_enabled: bool = False
    tls_cert_dir: str = "/certs"

    # Attachments (file uploads as debate context).
    attachments_max_files: int = 5
    attachments_max_size_mb: int = 5
    attachments_allowed_mime: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
        "text/plain",
    ]
    attachments_storage_dir: str = "/data/uploads"

    # Debate persistence. SQLite file written by the orchestrator only — every
    # SSE event of every debate gets persisted there so the frontend can
    # replay them on F5 or jump back to a past debate from the sidebar.
    # Default is relative ("data/debates.db") so start.bat works without
    # extra env. The Docker compose overrides DEBATES_DB_PATH=/data/debates.db
    # to land on the host-mounted `debates_data` volume.
    debates_db_path: str = "data/debates.db"

    # CORS origins for the frontend. Comma-separated list in env.
    # Defaults cover common local dev ports; set explicitly in production.
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def url_scheme(self) -> str:
        return "https" if self.tls_enabled else "http"

    def agent_url(self, port: int) -> str:
        """URL used *by the orchestrator* to reach another agent.

        In the legacy flow, agents find each other at localhost; in
        Docker Compose the orchestrator reaches them at their service-name.
        Workers in the current (agentic) design register their own
        advertised URL (built via `own_url`), so the orchestrator reads
        that from the registry rather than constructing it here.
        """
        return f"{self.url_scheme}://{self.orchestrator_host}:{port}"

    def orchestrator_url(self) -> str:
        """URL used by workers to reach the orchestrator (e.g. to register)."""
        return f"{self.url_scheme}://{self.orchestrator_host}:{self.orchestrator_port}"

    def mcp_url(self) -> str:
        """URL specialized agents use to reach the MCP tools server."""
        return f"{self.url_scheme}://{self.mcp_host}:{self.mcp_port}/mcp"

    def own_url(self, port: int) -> str:
        """URL this process advertises to the registry.

        Defaults to `http://localhost:{port}` for local dev. In Docker
        Compose, each service sets SELF_HOST to its own service name so
        peer containers can resolve it.
        """
        return f"{self.url_scheme}://{self.self_host}:{port}"


settings = Settings()
