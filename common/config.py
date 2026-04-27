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
    orchestrator_model: str = "groq/llama-3.3-70b-versatile"
    normalizer_model: str = "gemini/gemini-2.5-flash"
    ae1_model: str = "mistral/mistral-large-latest"
    ae2_model: str = "cerebras/qwen-3-235b-a22b-instruct-2507"
    feedback_model: str = "ollama/qwen2.5:14b"

    # Ollama
    ollama_api_base: str = "http://localhost:11434"

    # Debate
    max_debate_rounds: int = 5

    # Phase 4: maximum number of deliberation rounds in the multi-round
    # blackboard loop (DeliberationLoop). Round 1 is opening statements
    # (everyone speaks), rounds 2..N are speaker-selected follow-ups, plus
    # one extra final round for the Synthesizer. With 3 rounds the typical
    # debate produces ≈10-14 LLM calls — well under Groq's 30 RPM cap.
    deliberation_max_rounds: int = 3

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

    # CORS origins for the frontend. Comma-separated list in env.
    # Defaults cover common local dev ports; set explicitly in production.
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def agent_url(self, port: int) -> str:
        """URL used *by the orchestrator* to reach another agent.

        In the legacy flow, agents find each other at localhost; in
        Docker Compose the orchestrator reaches them at their service-name.
        Workers in the current (agentic) design register their own
        advertised URL (built via `own_url`), so the orchestrator reads
        that from the registry rather than constructing it here.
        """
        return f"http://{self.orchestrator_host}:{port}"

    def orchestrator_url(self) -> str:
        """URL used by workers to reach the orchestrator (e.g. to register)."""
        return f"http://{self.orchestrator_host}:{self.orchestrator_port}"

    def mcp_url(self) -> str:
        """URL specialized agents use to reach the MCP tools server."""
        return f"http://{self.mcp_host}:{self.mcp_port}/mcp"

    def own_url(self, port: int) -> str:
        """URL this process advertises to the registry.

        Defaults to `http://localhost:{port}` for local dev. In Docker
        Compose, each service sets SELF_HOST to its own service name so
        peer containers can resolve it.
        """
        return f"http://{self.self_host}:{port}"


settings = Settings()
