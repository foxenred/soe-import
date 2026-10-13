"""Запись разбора контейнера агентами (GigaChat, если задан ключ в .env; иначе — по правилам):
agents/recordings/<контейнер>.json. Запуск: python -m agents.record K-382"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from agents.llm import LLMClient
from agents.orchestrator import run_case
from agents.settings import ROOT, settings
from agents.tools import Box
from agents.world import world


async def main(container: str) -> None:
    box = Box(world("general_2026", 11))
    events, t0 = [], time.time()

    async def emit(kind, agent, data):
        events.append({"seq": len(events) + 1, "t": round(time.time() - t0, 2), "type": kind, "agent": agent, "data": data})

    llm = LLMClient() if settings.llm_configured else None
    try:
        card = await run_case(box, container, emit, llm)
    finally:
        if llm:
            await llm.close()
    out = ROOT / "agents" / "recordings" / f"{container}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(events, ensure_ascii=False, default=float, indent=1), encoding="utf-8")
    print(out, "| GigaChat" if llm else "| правила", "| числа сверены:", card.get("numbers_check", {}).get("ok"))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "K-382"))
