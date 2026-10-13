"""Клиент языковой модели: GigaChat (Сбер) или любой OpenAI-совместимый API (например, Yandex AI Studio).

Провайдер, адрес, ключ и модель — в .env: LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Вызов инструментов:
у GigaChat — functions/function_call, у OpenAI-совместимых — tools/tool_calls. Наружу — одинаковый ответ.
"""
from __future__ import annotations

import asyncio
import json
import ssl
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import certifi
import httpx

from agents.settings import settings

CERTS = Path(__file__).resolve().parent / "certs"


class LLMError(Exception):
    pass


@dataclass
class Reply:
    text: str
    calls: list = field(default_factory=list)        # [{"id", "name", "args"}]
    raw: dict = field(default_factory=dict)           # сообщение ассистента в формате провайдера
    tokens: int = 0


def _ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=certifi.where())
    for f in ("russian_trusted_root_ca.pem", "russian_trusted_sub_ca.pem"):
        if (CERTS / f).exists():
            ctx.load_verify_locations(str(CERTS / f))
    return ctx


class LLMClient:
    def __init__(self):
        if not settings.llm_configured:
            raise LLMError("языковая модель не настроена")
        self.provider = settings.llm_provider
        self.key = settings.llm_api_key
        self.model = settings.llm_model or ("GigaChat-2-Max" if self.provider == "gigachat" else "")
        self.base = (settings.llm_base_url or ("https://gigachat.devices.sberbank.ru/api/v1" if self.provider == "gigachat" else "")).rstrip("/")
        self.http = httpx.AsyncClient(timeout=settings.llm_timeout_s, verify=_ssl())
        self._tok, self._exp = None, 0.0
        self._tok_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(settings.llm_concurrency)   # бесплатный GigaChat — один запрос за раз

    async def close(self):
        await self.http.aclose()

    async def _token(self) -> str:
        if self.provider != "gigachat":
            return self.key
        async with self._tok_lock:
            if self._tok and time.time() < self._exp - 60:
                return self._tok
            return await self._new_token()

    async def _new_token(self) -> str:
        r = await self.http.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth", data={"scope": settings.llm_scope},
                                 headers={"Authorization": "Basic " + self.key, "RqUID": str(uuid.uuid4()),
                                          "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        if r.status_code != 200:
            raise LLMError(f"вход в GigaChat не удался ({r.status_code})")
        d = r.json()
        self._tok = d["access_token"]
        self._exp = float(d.get("expires_at", 0)) / 1000 or time.time() + 1500
        return self._tok

    async def chat(self, messages: list[dict], functions: list[dict], temperature: float = 0.1) -> Reply:
        if self.provider == "gigachat":
            body = {"model": self.model, "temperature": temperature, "max_tokens": 900, "messages": messages,
                    "functions": functions, "function_call": "auto"}
        else:
            body = {"model": self.model, "temperature": temperature, "max_tokens": 900, "messages": messages,
                    "tools": [{"type": "function", "function": f} for f in functions], "tool_choice": "auto"}
        for attempt in range(4):
            try:
                async with self._sem:
                    r = await self.http.post(f"{self.base}/chat/completions", json=body,
                                             headers={"Authorization": "Bearer " + await self._token()})
            except httpx.HTTPError as e:
                raise LLMError(f"нет связи с моделью: {type(e).__name__}") from e
            if r.status_code == 429:
                await _sleep(2 + 3 * attempt)
                continue
            if r.status_code != 200:
                raise LLMError(f"модель ответила {r.status_code}")
            d = r.json()
            msg = d["choices"][0]["message"]
            tokens = int((d.get("usage") or {}).get("total_tokens") or 0)
            return self._parse(msg, tokens)
        raise LLMError("модель занята — лимит запросов")

    def _parse(self, msg: dict, tokens: int) -> Reply:
        calls = []
        if self.provider == "gigachat":
            fc = msg.get("function_call")
            if fc:
                calls.append({"id": msg.get("functions_state_id") or str(uuid.uuid4()), "name": fc.get("name"),
                              "args": _args(fc.get("arguments"))})
        else:
            for tc in msg.get("tool_calls") or []:
                f = tc.get("function", {})
                calls.append({"id": tc.get("id"), "name": f.get("name"), "args": _args(f.get("arguments"))})
        return Reply(text=msg.get("content") or "", calls=calls, raw=msg, tokens=tokens)

    def assistant_message(self, reply: Reply) -> dict:
        m = {k: v for k, v in reply.raw.items() if k in ("role", "content", "function_call", "functions_state_id", "tool_calls")}
        m.setdefault("role", "assistant")
        m.setdefault("content", "")
        return m

    def tool_message(self, call: dict, result) -> dict:
        content = json.dumps(result, ensure_ascii=False)
        if self.provider == "gigachat":
            return {"role": "function", "name": call["name"], "content": content}
        return {"role": "tool", "tool_call_id": call["id"], "content": content}


def _args(a):
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except ValueError:
        return {}


async def _sleep(s: float):
    import asyncio
    await asyncio.sleep(s)
