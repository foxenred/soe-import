"""Разбор груза агентами v5 (GigaChat): внешняя доставка → таможня → внутренняя доставка → регионы → оркестратор.

Каждый агент вызывает свои инструменты (backend/v5/tools.py), итог — 1–2 фразы со ссылками [cNN] на расчеты.
Числа в тексте сверяются с результатами расчетов (backend/agents/numbers.py). Нет модели или ошибка — агент работает
по правилу (те же инструменты, текст по шаблону). Решение утверждает человек.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

from agents import numbers
from agents.llm import LLMClient, LLMError
from agents.text import plain
from agents.settings import settings
from agents.tools import AGENTS, Box, ToolError, schemas_for

Emit = Callable[[str, str | None, dict], Awaitable[None]]

BASE = ("Ты — {title} в мультиагентной системе S&OE-цикла импорта сети магазинов. Отвечай по-русски, коротко: "
        "не больше двух предложений и 40 слов. Сначала вызови нужные инструменты, потом дай итог. Каждое число в итоге "
        "бери только из результатов инструментов и сразу после него ставь ссылку на расчет в квадратных скобках, например "
        "[c03]. Не считай новых чисел сам. Суммы — в рублях с пробелами между разрядами. Даты пиши как «29 ноября». "
        "Не используй обозначения P10, P50, P90 — пиши «вероятно … , не позже …». Решение утверждает человек.")
TASK = {
    "ext": "Контейнер {cid}: где он и когда придет на границу (вероятно и не позже). Если он уже пришел — скажи, когда и насколько позже плана. Вызови get_tracking.",
    "customs": "Контейнер {cid}: проверь документы (check_documents) и статус таможни (customs_status). Скажи, есть ли "
               "проверка, когда выпуск и сколько стоит выпуск под обеспечение.",
    "internal": "Контейнер {cid}: что с вывозом с СВХ (svh_status) и когда он придет в федеральный ТЦ (rail_eta).",
    "regions": "Каким регионам (РЦ) задержка обходится дороже всего? Вызови top_regions. Назови 2–3 РЦ, их стоимость "
               "задержки и когда у них кончится товар.",
    "orchestrator": "Контейнер {cid}. Вызови evaluate_levers и allocation_plan, выбери рычаг с наибольшей разницей "
                    "«сохраненная прибыль минус цена» и создай карточку решения create_decision (lever, why со ссылками, "
                    "regions — приоритетные РЦ). Если ни один рычаг не окупается — lever «ждать».",
}
ORDER = ["ext", "customs", "internal", "regions", "orchestrator"]


async def run_case(box: Box, container_id: str, emit: Emit, llm: LLMClient | None) -> dict:
    usage = {"tokens": 0}
    out = {}
    for agent in ORDER:
        out[agent] = await run_agent(agent, box, container_id, emit, llm, usage)
    if box.card is None:
        rule_decision(box, container_id)
    card = dict(box.card)
    card["why"] = plain(re.sub(r"[{}]", "", card.get("why", ""))).strip()
    card["numbers_check"] = numbers.check(card["why"], box.calls)
    card["mode"] = "llm" if all(o["mode"] == "llm" for o in out.values()) else ("rule" if all(o["mode"] == "rule" for o in out.values()) else "mixed")
    card["tokens"] = usage["tokens"]
    await emit("decision_card", "orchestrator", card)
    await emit("done", None, {"tokens": usage["tokens"]})
    return card


async def run_agent(agent: str, box: Box, container_id: str, emit: Emit, llm: LLMClient | None, usage: dict) -> dict:
    await emit("agent_started", agent, {"title": AGENTS[agent]})
    mode, note, text = "rule", None, None
    if llm is not None:
        try:
            text = await asyncio.wait_for(_llm(agent, box, container_id, emit, llm, usage), timeout=settings.agent_timeout_s)
            mode = "llm"
            if not numbers.check(text, box.calls)["ok"]:
                note, text, mode = "в тексте ИИ было число не из расчета — показан текст расчетного правила", None, "rule"
        except (LLMError, ToolError, asyncio.TimeoutError) as e:
            note = f"ИИ недоступен ({e or type(e).__name__}) — расчетное правило"
    if text is None:
        text = await _rule(agent, box, container_id, emit)
    text = plain(text)
    await emit("agent_message", agent, {"text": text, "refs": sorted(set(re.findall(r"c\d{2}", text)))})
    await emit("agent_finished", agent, {"mode": mode, "note": note, "numbers_check": numbers.check(text, box.calls)})
    return {"mode": mode, "text": text}


async def _call(box: Box, emit: Emit, agent: str, name: str, args: dict):
    cid_ = box.reserve()
    await emit("tool_call", agent, {"call_id": cid_, "tool": name, "args": args})
    try:
        c = box.call(agent, name, args, call_id=cid_)
    except ToolError as e:
        await emit("tool_result", agent, {"call_id": cid_, "tool": name, "result": {"error": str(e)}})
        raise
    await emit("tool_result", agent, {"call_id": c.call_id, "tool": name, "result": c.result})
    return c


async def _llm(agent: str, box: Box, container_id: str, emit: Emit, llm: LLMClient, usage: dict) -> str:
    msgs = [{"role": "system", "content": BASE.format(title=AGENTS[agent])},
            {"role": "user", "content": TASK[agent].format(cid=container_id)}]
    fns = schemas_for(agent)
    for _ in range(6):
        rep = await llm.chat(msgs, fns)
        usage["tokens"] += rep.tokens
        if not rep.calls:
            if not rep.text.strip():
                raise LLMError("пустой ответ")
            return rep.text.strip()
        msgs.append(llm.assistant_message(rep))
        for call in rep.calls:
            args = call["args"] or {}
            if "container_id" in [p for f in fns if f["name"] == call["name"] for p in f["parameters"]["properties"]]:
                args.setdefault("container_id", container_id)
            try:
                c = await _call(box, emit, agent, call["name"], args)
                res = c.result
                res = {"call_id": c.call_id, **res} if isinstance(res, dict) else {"call_id": c.call_id, "result": res}
            except ToolError as e:
                res = {"error": str(e)}
            msgs.append(llm.tool_message(call, res))
    raise LLMError("слишком много шагов")


def _r(x) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _d(iso: str | None) -> str:
    if not iso:
        return "—"
    m = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    y, mm, dd = iso.split("-")
    return f"{int(dd)} {m[int(mm) - 1]}"


async def _rule(agent: str, box: Box, cid_: str, emit: Emit) -> str:
    if agent == "ext":
        c = await _call(box, emit, agent, "get_tracking", {"container_id": cid_})
        r = c.result
        if r.get("arrived_border"):
            return f"{r['container']} пришел на точку входа {r['gateway']} {_d(r['arrived_border'])} [{c.call_id}], сейчас — {r['stage']}."
        return (f"{r['container']} — {r['stage']}, на границе вероятно {_d(r['border_likely'])}, не позже {_d(r['border_latest'])} "
                f"[{c.call_id}].")
    if agent == "customs":
        a = await _call(box, emit, agent, "check_documents", {"container_id": cid_})
        b = await _call(box, emit, agent, "customs_status", {"container_id": cid_})
        docs = "ошибка в документах найдена заранее" if a.result["found_before_filing"] else (
            "расхождение в документах" if a.result["errors"] else "документы сходятся")
        s = f"{docs} [{a.call_id}]; выпуск {_d(b.result['release_forecast'])} [{b.call_id}]"
        g = b.result.get("guarantee_option")
        if g:
            s += f", выпуск под обеспечение стоит {_r(g['cost_rub'])} ₽ и экономит {str(g['saved_days']).replace('.', ',')} сут [{b.call_id}]"
        return s[0].upper() + s[1:] + "."
    if agent == "internal":
        a = await _call(box, emit, agent, "svh_status", {"container_id": cid_})
        b = await _call(box, emit, agent, "rail_eta", {"container_id": cid_})
        s = f"В федеральном ТЦ вероятно {_d(b.result['eta_p50'])}, не позже {_d(b.result['eta_p90'])} [{b.call_id}]"
        pr = a.result.get("priority_option")
        if pr:
            s += f"; приоритетный вывоз с СВХ стоит {_r(pr['cost_rub'])} ₽ и экономит {str(pr['saved_days']).replace('.', ',')} сут [{a.call_id}]"
        return s + "."
    if agent == "regions":
        c = await _call(box, emit, agent, "top_regions", {"n": 3})
        rs = c.result["regions"]
        parts = [f"{r['short']} — {_r(r['C'])} ₽, товар кончится {_d(r['stockout'])}" for r in rs if r["C"] > 0]
        return ("Дороже всего задержка обходится: " + "; ".join(parts) + f" [{c.call_id}].") if parts else f"Дефицита в регионах не ожидается [{c.call_id}]."
    # оркестратор
    return await _rule_orch(box, cid_, emit)


async def _rule_orch(box: Box, cid_: str, emit: Emit) -> str:
    a = await _call(box, emit, "orchestrator", "evaluate_levers", {"container_id": cid_})
    b = await _call(box, emit, "orchestrator", "allocation_plan", {})
    best = a.result["best"]
    opt = next((o for o in a.result["options"] if o["id"] == best), None)
    if opt:
        why = (f"{opt['title']}: сохраняет {_r(opt['benefit_rub'])} ₽ при цене {_r(opt['cost_rub'])} ₽ [{a.call_id}]; "
               f"товар — сначала в {', '.join(b.result['priority_regions'][:3])} [{b.call_id}].")
    else:
        why = f"Рычаги не окупаются — ждать [{a.call_id}]."
    c = await _call(box, emit, "orchestrator", "create_decision",
                    {"container_id": cid_, "lever": best if opt else "ждать", "why": why, "regions": b.result["priority_regions"][:3]})
    return why


def rule_decision(box: Box, cid_: str):
    ev = box.evaluate_levers(cid_)
    best = ev["best"]
    box.create_decision(cid_, best, "Карточка подготовлена расчетным правилом.", [])
