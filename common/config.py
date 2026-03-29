from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # LLM Provider API Keys
    groq_api_key: str = ""
    gemini_api_key: str = ""
    mistral_api_key: str = ""
    cerebras_api_key: str = ""

    # Agent Ports
    orchestrator_port: int = 9000
    normalizer_port: int = 9001
    ae1_port: int = 9002
    ae2_port: int = 9003
    feedback_port: int = 9004

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

    def agent_url(self, port: int) -> str:
        return f"http://localhost:{port}"


settings = Settings()
