from dataclasses import dataclass
from functools import lru_cache
import json
import os


@dataclass(frozen=True)
class Settings:
    api_key: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_reasoning_effort: str | None
    llm_extra_body: dict[str, object]
    llm_connect_timeout: float
    llm_first_token_timeout: float
    llm_timeout: float
    llm_max_retries: int
    fallback_base_url: str | None
    fallback_api_key: str | None
    fallback_model: str | None
    max_concurrent_generations: int
    max_queue_wait: float
    rpm_limit: int
    tpm_limit: int
    daily_request_budget: int
    max_tokens: int
    hard_max_tokens: int
    history_turns: int



def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None



def _extra_body() -> dict[str, object]:
    raw = os.getenv("LLM_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("LLM_EXTRA_BODY_JSON must contain a JSON object")
    return value


@lru_cache
def settings() -> Settings:
    return Settings(
        api_key=os.getenv("INTERNAL_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_reasoning_effort=_optional("LLM_REASONING_EFFORT"),
        llm_extra_body=_extra_body(),
        llm_connect_timeout=float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "5")),
        llm_first_token_timeout=float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "30")),
        llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
        llm_max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", "2"))),
        fallback_base_url=_optional("LLM_FALLBACK_BASE_URL"),
        fallback_api_key=_optional("LLM_FALLBACK_API_KEY"),
        fallback_model=_optional("LLM_FALLBACK_MODEL"),
        max_concurrent_generations=max(1, int(os.getenv("MAX_CONCURRENT_GENERATIONS", "3"))),
        max_queue_wait=max(0.0, float(os.getenv("MAX_QUEUE_WAIT_SECONDS", "10"))),
        rpm_limit=max(0, int(os.getenv("LLM_RPM_LIMIT", "0"))),
        tpm_limit=max(0, int(os.getenv("LLM_TPM_LIMIT", "0"))),
        daily_request_budget=max(0, int(os.getenv("LLM_DAILY_REQUEST_BUDGET", "0"))),
        max_tokens=max(1, int(os.getenv("DEFAULT_MAX_TOKENS", "256"))),
        hard_max_tokens=max(1, int(os.getenv("HARD_MAX_TOKENS", "384"))),
        history_turns=max(1, int(os.getenv("HISTORY_TURNS", "4"))),
    )
