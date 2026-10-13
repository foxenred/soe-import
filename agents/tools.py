"""Инструменты агентов v5 над миром демонстрации. Каждое число в ответе агента — из этих расчетов.

Агенты (схема владельца): внешней доставки, таможни, внутренней доставки, регионов, оркестратор S&OE-цикла.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agents.world import DemoWorld, cid, parse_cid, short_name

AGENTS = {
    "ext": "Агент внешней доставки", "customs": "Агент таможни", "internal": "Агент внутренней доставки",
    "regions": "Агенты регионов", "orchestrator": "Оркестратор S&OE-цикла",
}


class ToolError(Exception):
    pass


@dataclass
class Call:
    call_id: str
    tool: str
    args: dict
    result: Any
    agent: str


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    fn: Callable
    agents: tuple


@dataclass
class Box:
    w: DemoWorld
    calls: list = field(default_factory=list)
    n: int = 0
    card: dict | None = None

    def reserve(self) -> str:
        self.n += 1
        return f"c{self.n:02d}"

    def call(self, agent: str, name: str, args: dict, call_id: str | None = None) -> Call:
        t = TOOLS.get(name)
        if t is None or agent.split(":")[0] not in t.agents:
            raise ToolError(f"нет инструмента {name}")
        try:
            res = t.fn(self, **(args or {}))
        except KeyError as e:
            raise ToolError(str(e).strip("'")) from e
        c = Call(call_id or self.reserve(), name, args or {}, res, agent)
        self.calls.append(c)
        return c

    def i(self, container_id: str) -> int:
        i = parse_cid(container_id)
        if not 0 <= i < self.w.s.N:
            raise KeyError(f"нет контейнера {container_id}")
        return i

    # --- внешняя доставка ---
    def get_tracking(self, container_id: str) -> dict:
        w, i = self.w, self.i(container_id)
        s = w.s
        c = w.container(i)
        gw_plan = s.plan[i] + s.d.pay_shift[i]
        q = s.q_ext[i]
        out = {"container": c["id"], "gateway": c["gateway"], "stage": c["stage_text"], "border_plan": w.day(s.plan[i]),
               "port_load": "высокая" if s.cong_dep[i] > 0.7 else ("низкая" if s.cong_dep[i] < -0.7 else "обычная"),
               "payment_shift_days": round(float(s.d.pay_shift[i]), 1)}
        if w.t >= s.t_gw[i]:      # уже на точке входа — только факт, без прогноза
            out["arrived_border"] = w.day(s.t_gw[i])
            out["late_vs_plan_days"] = round(float(s.t_gw[i] - s.plan[i]), 1)
        else:
            out.update({"border_likely": w.day(gw_plan + q[1]), "border_latest": w.day(gw_plan + q[2]),
                        "border_earliest": w.day(gw_plan + q[0]),
                        "model": "квантильная регрессия по истории перевозок (синтетическая история)"})
        return out

    # --- таможня ---
    def check_documents(self, container_id: str) -> dict:
        """Сверка инвойса, упаковочного листа и декларации (OCR и разбор полей). Ошибки — из модели."""
        w, i = self.w, self.i(container_id)
        s = w.s
        h = int(hashlib.md5(container_id.encode()).hexdigest(), 16)
        gross = 6000 * 0.65 + (h % 400)
        fields = [{"field": "Код ТН ВЭД", "invoice": "условный код", "declaration": "условный код", "ok": True},
                  {"field": "Количество, шт.", "invoice": 6000, "packing_list": 6000, "ok": True}]
        if s.doc[i]:
            fields.append({"field": "Вес брутто, кг", "invoice": round(gross), "packing_list": round(gross * 1.08), "ok": False})
        else:
            fields.append({"field": "Вес брутто, кг", "invoice": round(gross), "packing_list": round(gross), "ok": True})
        return {"container": cid(i), "fields": fields, "errors": int(s.doc[i]),
                "found_before_filing": bool(s.caught[i]), "fix_days_if_missed": round(float(s.d.doc_days[i]), 1) if s.doc[i] else 0}

    def customs_status(self, container_id: str) -> dict:
        w, i = self.w, self.i(container_id)
        s = w.s
        st = w.stage(i)
        out = {"container": cid(i), "stage": w.container(i)["stage_text"], "duty_rate_pct": round(100 * s.cu["duty_rate"]),
               "vat_pct": round(100 * s.cu["vat"])}
        if s.chk[i]:
            out["check"] = "запрошены дополнительные документы по таможенной стоимости (ТК ЕАЭС, ст. 325)"
            out["check_days"] = round(float(s.chk_days[i]), 1)
            out["release_without_guarantee"] = w.day(s.t_gw[i] + s.cu["release_days"] + s.chk_days[i])
        else:
            out["check"] = "нет"
        out["release_forecast"] = w.day(s.t_rel[i])
        for lv in w.levers(i):
            if lv["id"] == "guarantee":
                out["guarantee_option"] = lv
        return out

    # --- внутренняя доставка ---
    def svh_status(self, container_id: str) -> dict:
        w, i = self.w, self.i(container_id)
        s = w.s
        sv = s.sv
        st = w.stage(i)
        waited = max(0.0, w.t - s.t_gw[i])
        q = s.q_svh[i]
        out = {"container": cid(i), "terminal": s.net.gateways[s.gw[i]].name, "stage": w.container(i)["stage_text"],
               "days_at_terminal": round(waited, 1), "free_days": sv["free_days"],
               "storage_rub_day": sv["storage_rub_day"], "wait_p10": round(float(q[0]), 1), "wait_p50": round(float(q[1]), 1),
               "wait_p90": round(float(q[2]), 1), "dispatch_forecast": w.day(s.t_disp[i])}
        for lv in w.levers(i):
            if lv["id"] == "priority":
                out["priority_option"] = lv
        return out

    def rail_eta(self, container_id: str) -> dict:
        w, i = self.w, self.i(container_id)
        lo, mid, hi = w.eta_interval(i)
        return {"container": cid(i), "to": w.net.names[w.net.fed], "eta_p10": w.day(lo), "eta_p50": w.day(mid),
                "eta_p90": w.day(hi), "plan": w.container(i)["plan_fed"], "late_days": w.container(i)["late_days"]}

    # --- регионы ---
    def top_regions(self, n: int = 5) -> dict:
        rs = [r for r in sorted(self.w.regions(), key=lambda r: -r["C"]) if r["C"] > 0][:int(n)]
        return {"date": self.w.today.isoformat(), "regions": [{k: r[k] for k in ("short", "fd", "stock", "demand_per_day", "cover_days",
                                                                                     "stockout", "deficit_days", "C", "P", "U", "R")} for r in rs],
                "formula": "C = П + У + Р; П — дни дефицита × спрос в день × маржа, У — уценка излишка, Р — доп. перевозка"}

    def region(self, name: str) -> dict:
        nm = (name or "").lower()
        for r in self.w.regions():
            if nm in r["short"].lower() or nm in r["name"].lower():
                return {k: r[k] for k in ("short", "fd", "share", "stock", "sales_per_day", "demand_per_day", "cover_days",
                                          "stockout", "deficit_days", "C", "P", "U", "R", "from_fed_days", "mode")}
        raise KeyError(f"нет РЦ {name}")

    # --- оркестратор ---
    def evaluate_levers(self, container_id: str) -> dict:
        w, i = self.w, self.i(container_id)
        opts = w.levers(i)
        for o in opts:
            o["net_rub"] = o["benefit_rub"] - o["cost_rub"]
        best = max(opts, key=lambda o: o["net_rub"]) if opts else None
        return {"container": cid(i), "cost_of_delay_per_day_rub": round(w.cost_of_delay(i, 1.0)), "options": opts,
                "best": best["id"] if best and best["net_rub"] > 0 else "ждать", "limit_rub": int(w.p["v4"]["corridor"]["limit_rub"])}

    def allocation_plan(self) -> dict:
        w = self.w
        rs = sorted(w.regions(), key=lambda r: -r["C"])[:5]
        hub = w.hub()
        return {"federal_rc": short_name(hub["name"]), "hub_stock": hub["stock"], "inflow_next_weeks": hub["inflow_weeks"][:4],
                "priority_regions": [r["short"] for r in rs if r["C"] > 0],
                "rule": "отгрузка регионам по потребности: спрос на срок доставки и пересмотра плюс страховой запас 5 дней; "
                        "при нехватке — сначала регионы с наибольшей стоимостью задержки"}

    def create_decision(self, container_id: str, lever: str, why: str, regions: list | None = None) -> dict:
        w, i = self.w, self.i(container_id)
        ev = self.evaluate_levers(container_id)
        opt = next((o for o in ev["options"] if o["id"] == lever), None)
        self.card = {"container": cid(i), "lever": lever, "option": opt, "options": ev["options"], "why": why,
                     "regions": regions or [], "cost_of_delay_per_day_rub": ev["cost_of_delay_per_day_rub"],
                     "in_limit": bool(opt and opt["cost_rub"] <= ev["limit_rub"]), "limit_rub": ev["limit_rub"]}
        return {"ok": True, "card": self.card}


def _p(**props):
    req = [k for k, v in props.items() if not v.pop("optional", False)]
    return {"type": "object", "properties": props, "required": req}


S = {"type": "string"}
C = {"type": "string", "description": "номер контейнера, например K-321"}
TOOLS = {t.name: t for t in [
    Tool("get_tracking", "Где контейнер, плановая дата и интервал прибытия на границу (P10–P50–P90) по ML-модели.", _p(container_id=dict(C)), Box.get_tracking, ("ext",)),
    Tool("check_documents", "Сверить инвойс, упаковочный лист и декларацию: поля и расхождения.", _p(container_id=dict(C)), Box.check_documents, ("customs",)),
    Tool("customs_status", "Статус таможни: проверка, срок выпуска, вариант выпуска под обеспечение с ценой.", _p(container_id=dict(C)), Box.customs_status, ("customs",)),
    Tool("svh_status", "СВХ: сколько суток стоит, прогноз ожидания вывоза (P10–P90), хранение, вариант приоритетного вывоза.", _p(container_id=dict(C)), Box.svh_status, ("internal",)),
    Tool("rail_eta", "Срок прихода в федеральный ТЦ (P10–P50–P90) и опоздание к плану.", _p(container_id=dict(C)), Box.rail_eta, ("internal",)),
    Tool("top_regions", "Регионы (РЦ) с наибольшей стоимостью задержки C = П + У + Р.", _p(n={"type": "integer", "optional": True}), Box.top_regions, ("regions", "orchestrator")),
    Tool("region", "Показатели одного РЦ: запас, спрос, когда кончится, стоимость задержки.", _p(name=dict(S)), Box.region, ("regions",)),
    Tool("evaluate_levers", "Рычаги по контейнеру: цена, сколько суток экономит, сколько прибыли сохраняет.", _p(container_id=dict(C)), Box.evaluate_levers, ("orchestrator",)),
    Tool("allocation_plan", "План отгрузок из федерального ТЦ и приоритетные регионы.", _p(), Box.allocation_plan, ("orchestrator",)),
    Tool("create_decision", "Карточка решения: рычаг, объяснение со ссылками [cNN], регионы-приоритет.",
         _p(container_id=dict(C), lever={"type": "string", "enum": ["guarantee", "priority", "ждать"]},
            why={"type": "string", "description": "2–3 предложения; каждое число — из результатов инструментов, со ссылкой [cNN]"},
            regions={"type": "array", "items": S, "optional": True}), Box.create_decision, ("orchestrator",)),
]}


def schemas_for(agent: str) -> list[dict]:
    kind = agent.split(":")[0]
    import copy
    return [{"name": t.name, "description": t.description, "parameters": copy.deepcopy(t.params)} for t in TOOLS.values() if kind in t.agents]
