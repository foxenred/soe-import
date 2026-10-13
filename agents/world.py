"""Мир демонстрации v5: сезон модели (src/v5) остановлен на дате демонстрации, состояние — для агентов и сайта.

Контейнеры в пути по этапам (у производителя, до границы, таможня, СВХ, ж/д по России), интервалы прибытия ML-агентов,
статусы таможни и СВХ, запас 41 регионального РЦ и федерального ТЦ, стоимость задержки C = П + У + Р по регионам.
Все числа — из модели; данные условные (магазины и РЦ — открытые данные о сети сети).
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

from src.config import load_params
from src.v5.eta import QS, train
from src.v5.network import RCNet
from src.v5.sim import RULES, Scenario, Season, calibrate_staff, draw

STAGE = {"china": "у производителя: отгрузка", "sea": "в пути до границы", "customs": "на таможне", "svh": "на СВХ: ждет вывоза",
         "rail": "в пути по России", "fed": "в федеральном ТЦ"}


def cid(i: int) -> str:
    return f"K-{i + 1:03d}"


def parse_cid(s: str) -> int:
    s = (s or "").strip().upper().replace("К", "K")
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise KeyError(f"нет контейнера {s}")
    return int(digits) - 1


class DemoWorld:
    def __init__(self, scenario: str = "general_2026", seed: int = 11, demo_day: int | None = None, volume: int | None = None):
        p = load_params()
        self.p = p
        self.net = RCNet.load(p)
        self.eta = _eta()
        v5 = p["v5"]
        base = Scenario("base", volume=volume, wave=tuple(p["experiments"]["wave"]))
        G = v5["general"]
        gen = (("season.end", G["end"]), ("season.peak_multiplier", G["peak_multiplier"]), ("demand.safety_stock", G["safety_stock"]),
               ("sim.supply_window", G["supply_window"]), ("product.markdown", G["markdown"]))
        scs = {"general": replace(base, name="general", over=gen),
               "general_2026": replace(base, name="general_2026", over=gen + (("v5.eta.svh.base", [12.0, 15.0, 12.0, 4.0, 4.0]),)),
               "base": base,
               "stress": replace(base, name="stress", shock=p["experiments"]["stress"]["shock"], wave_add=p["experiments"]["stress"]["wave_delay"]),
               "late": replace(base, name="late", shock=p["experiments"]["stress"]["shock"],
                               over=(("sim.supply_window", [-15, 40]),))}
        self.sc = scs[scenario]
        from src.common import apply_over
        self.pp = apply_over(p, self.sc.over)
        n = int(self.sc.volume or p["v4"]["volume"]["containers_base"])
        self.d = draw(self.net, self.pp, self.sc, n, np.random.default_rng(seed))
        staff = calibrate_staff(self.net, self.pp, self.sc, self.eta, runs=2)
        self.t = int(demo_day if demo_day is not None else v5.get("demo_day", 24))
        self.s = Season(self.net, self.pp, self.sc, self.d, RULES["P2"], staff, self.eta)
        self.s.run(until=self.t)
        self.start = date.fromisoformat(p["season"]["start"])
        self.today = self.start + timedelta(days=self.t)

    # --- даты ---
    def day(self, t: float) -> str:
        return (self.start + timedelta(days=math.floor(t))).isoformat()

    # --- контейнеры ---
    def stage(self, i: int) -> str:
        s, t = self.s, self.t
        if s.arrived[i]:
            return "fed"
        if t < s.t_dep[i]:
            return "china"
        if t < s.t_gw[i]:
            return "sea"
        if t < s.t_rel[i]:
            return "customs"
        if t < s.t_disp[i]:
            return "svh"
        return "rail"

    def eta_interval(self, i: int) -> tuple[float, float, float]:
        """P10, P50, P90 прихода в федеральный ТЦ по знаниям агентов на сегодня."""
        s, t = self.s, self.t
        base_tr = self.net.gw_km[s.gw[i]] / self.net.speed + s.hub_hand
        st = self.stage(i)
        if st == "fed":
            return (s.t_fed[i],) * 3
        if st == "rail":
            x = s.t_disp[i] + s.transit[i] + s.hub_hand
            return (x - 0.5, x, x + 0.5)
        if st == "svh":
            rel = s.t_rel[i]
            w = s.q_svh[i]
            done = t - rel
            return tuple(max(t, rel + max(wq, done)) + base_tr for wq in w)
        if st == "customs":
            rel = s.t_rel[i]
            return tuple(rel + wq + base_tr for wq in s.q_svh[i])
        gw = s.plan[i] + s.d.pay_shift[i] + s.q_ext[i]
        cust = s.cu["release_days"] + (s.d.doc_days[i] if (s.doc[i] and not s.caught[i]) else 0.0)
        return tuple(g + cust + w + base_tr for g, w in zip(gw, s.q_svh[i]))

    def container(self, i: int) -> dict:
        s = self.s
        g = self.net.gateways[s.gw[i]]
        st = self.stage(i)
        lo, mid, hi = self.eta_interval(i)
        plan_fed = s.plan[i] + s.cu["release_days"] + s.plan_wait[i] + self.net.gw_km[s.gw[i]] / self.net.speed + s.hub_hand
        out = {"id": cid(i), "gateway": g.name, "stage": st, "stage_text": STAGE[st],
               "plan_fed": self.day(plan_fed), "eta_p10": self.day(lo), "eta_p50": self.day(mid), "eta_p90": self.day(hi),
               "late_days": round(mid - plan_fed, 1), "units": int(s.U),
               "customs": {"check": bool(s.chk[i]), "guarantee": bool(s.guar[i]), "doc_error": bool(s.doc[i]),
                           "doc_caught": bool(s.caught[i])},
               "svh": {"wait_forecast_days": [round(x, 1) for x in s.q_svh[i]], "priority": bool(s.prio[i])}}
        if st == "svh":
            out["svh"]["waiting_days"] = round(self.t - s.t_rel[i], 1)
        return out

    def pipeline(self) -> list[dict]:
        s = self.s
        rows = [self.container(i) for i in range(s.N) if not s.arrived[i] and s.t_dep[i] <= self.t + 30]
        return sorted(rows, key=lambda x: x["eta_p50"])

    # --- регионы и стоимость задержки ---
    def regions(self) -> list[dict]:
        """Регионы и стоимость задержки C = П + У + Р. Та же проекция, что у стоимости задержки груза: недобор каждого
        РЦ к дню, когда товар из федерального ТЦ еще успеет доехать; нехватка сети в самый напряженный день (потребность
        минус запас федерального ТЦ и приходы по срокам агентов) раскладывается по РЦ пропорционально их недобору."""
        s, t = self.s, self.t
        net = self.net
        rate = s.rate(t)
        start = t + 1
        D = max(1, s.T - start)
        need = s.need(t, start, D)
        i0 = start - s.t0
        inb = s.inb[:, i0:i0 + D]
        X = np.cumsum(need - inb, axis=1)
        gap = np.maximum(0.0, np.maximum.accumulate(X, axis=1) - s.stock[:, None])
        lead = np.ceil(net.from_fed_days).astype(int)
        G = np.stack([gap[r, np.minimum(np.arange(D) + lead[r], D - 1)] for r in range(s.R)])
        Nh = G.sum(axis=0)
        eta = s.view_fed(t)
        arr = np.zeros(D)
        for j in np.nonzero(~s.arrived)[0]:
            k = int(math.floor(eta[j])) - start
            if k < D:
                arr[max(k, 0)] += s.U
        S = s.hub + np.cumsum(arr)
        k_star = int(np.argmax(Nh - S))
        L = max(0.0, float(Nh[k_star] - S[k_star]))
        share = G[:, k_star] / Nh[k_star] if Nh[k_star] > 0 else np.zeros(s.R)
        out = []
        for r in range(s.R):
            q = s.stock[r] + np.cumsum(inb[r]) - np.cumsum(need[r])
            neg = np.nonzero(q < 0)[0]
            stockout = self.day(start + neg[0]) if neg.size else None
            lost = L * share[r]
            P = lost * s.m
            left = max(0.0, s.stock[r] + inb[r].sum() - need[r].sum())
            U_ = left * s.price * s.delta
            R_ = net.from_fed_rub[r] / s.U * lost
            out.append({"id": r, "name": net.names[r], "short": short_name(net.names[r]), "fd": net.fd[r],
                        "lat": net.node[r][0], "lon": net.node[r][1], "share": round(float(net.share[r]), 4),
                        "stock": round(float(s.stock[r])), "sales_per_day": round(float(s.sales[r, max(0, t - 7):t].mean()) if t > 0 else 0.0),
                        "demand_per_day": round(float(rate[r])), "cover_days": round(float(s.stock[r] / max(rate[r], 1e-9)), 1),
                        "stockout": stockout, "deficit_days": round(lost / max(rate[r], 1e-9), 1),
                        "C": round(P + U_ + R_), "P": round(P), "U": round(U_), "R": round(R_),
                        "from_fed_days": round(float(net.from_fed_days[r]), 1), "mode": "авто" if net.truck[r] else "ж/д"})
        self.critical_day = self.day(start + k_star) if L > 0 else None
        return out

    def next_inflow_days(self) -> float:
        s = self.s
        eta = s.view_fed(self.t)
        pend = eta[~s.arrived]
        return float(max(0.0, pend.min() - self.t)) if pend.size else 30.0

    def hub(self) -> dict:
        s = self.s
        eta = s.view_fed(self.t)
        weeks = {}
        for i in np.nonzero(~s.arrived)[0]:
            w = int((eta[i] - self.t) // 7)
            if 0 <= w < 8:
                weeks[w] = weeks.get(w, 0) + 1
        return {"name": self.net.names[self.net.fed], "stock": round(float(s.hub)), "inflow_weeks": [weeks.get(w, 0) for w in range(8)]}

    # --- стоимость задержки контейнера и рычаги ---
    def cost_of_delay(self, i: int, days: float = 1.0) -> float:
        return float(self.s.network_gain(i, days, self.t))

    def levers(self, i: int) -> list[dict]:
        s, t = self.s, self.t
        out = []
        st = self.stage(i)
        cu, sv = s.cu, s.sv
        # о дополнительной проверке известно только после подачи декларации — вариант есть только на таможне
        if st == "customs" and s.chk[i] and not s.guar[i]:
            amount = s.U * cu["customs_value_rub"] * (cu["duty_rate"] + cu["vat"] * (1 + cu["duty_rate"]))
            cost = amount * cu["guarantee_rate_year"] * max(s.chk_days[i], cu["guarantee_min_days"]) / 365 + cu["guarantee_fee_rub"]
            saved = max(0.0, s.t_rel[i] - max(t, s.t_gw[i]) - cu["guarantee_days"])      # как в модели: остаток проверки
            if saved > 0:
                out.append({"id": "guarantee", "title": "Выпуск под обеспечение уплаты пошлин", "cost_rub": round(cost),
                            "saved_days": round(saved, 1), "benefit_rub": round(self.cost_of_delay(i, saved)),
                            "note": f"обеспечение {round(amount):,} ₽ банковской гарантией на {max(int(s.chk_days[i]), cu['guarantee_min_days'])} дней".replace(",", " ")})
        if st in ("svh", "customs"):
            saved = max(0.0, s.t_disp[i] - max(t, s.t_rel[i]) - sv["priority_wait_days"])   # как в модели
            if saved > 0:
                out.append({"id": "priority", "title": "Приоритетный вывоз с СВХ", "cost_rub": int(sv["priority_fee_rub"]),
                            "saved_days": round(saved, 1), "benefit_rub": round(self.cost_of_delay(i, saved)),
                            "note": "надбавка за ускоренную отправку (оценка), хранение сверх 8 суток не платим"})
        return out


@lru_cache(maxsize=1)
def _eta():
    p = load_params()
    net = RCNet.load(p)
    return train(p, len(net.gateways), net.gw_share)


def short_name(name: str) -> str:
    """«РЦ «Тюмень»» → «Тюмень»."""
    import re
    m = re.search(r"«([^»]+)»", name)
    return m.group(1) if m else name


@lru_cache(maxsize=4)
def world(scenario: str = "general_2026", seed: int = 11) -> DemoWorld:
    return DemoWorld(scenario, seed)
