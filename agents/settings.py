"""Настройки языковой модели из переменных окружения (.env, в репозитории — только .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "gigachat")   # gigachat | openai (OpenAI-совместимый API)
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_model: str = os.environ.get("LLM_MODEL", "")
    llm_scope: str = os.environ.get("LLM_SCOPE", "GIGACHAT_API_PERS")
    llm_price_per_1k_rub: float = _f("LLM_PRICE_PER_1K_RUB", 0.0)
    llm_timeout_s: float = _f("LLM_TIMEOUT_S", 60)
    llm_concurrency: int = int(_f("LLM_CONCURRENCY", 1))
    agent_max_tool_calls: int = int(_f("AGENT_MAX_TOOL_CALLS", 8))
    agent_timeout_s: float = _f("AGENT_TIMEOUT_S", 90)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
