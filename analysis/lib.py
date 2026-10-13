"""Вспомогательные расчеты отчета: модель с записью состояния по дням, проекция карты, интервалы по бутстрепу.

Ничего не меняет в модели: TracedSeason — та же имитация src/v5/sim.py, которая дополнительно записывает запас,
продажи и решения по рычагам для графиков.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from analysis.style import ROOT
from src.config import load_params
from src.common import apply_over
from src.v5.eta import train
from src.v5.experiments import scenarios
from src.v5.network import RCNet
from src.v5.sim import RULES, Season, calibrate_staff, draw


@lru_cache(maxsize=1)
def world():
    """Параметры, сеть РЦ, обученные ML-агенты сроков, режимы и штат (как в src/v5/experiments.py)."""
    p = load_params()
    net = RCNet.load(p)
    eta = train(p, len(net.gateways), net.gw_share)
    S = scenarios(p)
    staff = calibrate_staff(net, apply_over(p, S["general"].over), S["general"], eta)
    return p, net, eta, S, staff


class TracedSeason(Season):
    """Имитация с записью по дням: запас федерального ТЦ и регионов, решения по рычагам с выгодой и ценой."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace = {"t": [], "hub": [], "stock": []}
        self.lever_log = []

    def work(self, t: int):
        super().work(t)
        self.trace["t"].append(t)
        self.trace["hub"].append(self.hub)
        self.trace["stock"].append(self.stock.copy())

    def decide_guarantee(self, i: int, t: int):
        n0, c0 = self.levers["guarantee"], self.cost["guarantee"]
        saved = max(0.0, self.t_rel[i] - max(t, self.t_gw[i]) - self.cu["guarantee_days"])
        super().decide_guarantee(i, t)
        if self.levers["guarantee"] > n0:
            self.lever_log.append({"t": t, "i": i, "lever": "guarantee", "saved": saved, "cost": self.cost["guarantee"] - c0})

    def decide_priority(self, i: int, t: int):
        n0 = self.levers["priority"]
        saved = max(0.0, self.t_disp[i] - max(t, self.t_rel[i]) - self.sv["priority_wait_days"])
        super().decide_priority(i, t)
        if self.levers["priority"] > n0:
            self.lever_log.append({"t": t, "i": i, "lever": "priority", "saved": saved, "cost": self.sv["priority_fee_rub"]})


def run_traced(scenario: str, seed: int, rule: str) -> tuple[TracedSeason, dict]:
    p, net, eta, S, staff = world()
    sc = S[scenario]
    pp = apply_over(p, sc.over)
    n = int(sc.volume or p["v4"]["volume"]["containers_base"])
    d = draw(net, pp, sc, n, np.random.default_rng(seed))
    s = TracedSeason(net, pp, sc, d, RULES[rule], staff, eta)
    r = s.run()
    return s, r


def season_for(scenario: str, seed: int, rule: str = "P2") -> Season:
    """Один вариант без прогона по дням: сроки контейнеров, прогнозы ML-агентов."""
    p, net, eta, S, staff = world()
    sc = S[scenario]
    pp = apply_over(p, sc.over)
    n = int(sc.volume or p["v4"]["volume"]["containers_base"])
    d = draw(net, pp, sc, n, np.random.default_rng(seed))
    return Season(net, pp, sc, d, RULES[rule], staff, eta)


def lcc(lon, lat, lon0: float = 82.0, lat1: float = 50.0, lat2: float = 66.0, lat0: float = 60.0):
    """Коническая равноугольная проекция Ламберта (как на картах России): градусы → условные единицы."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    f1, f2, f0, l0 = map(math.radians, (lat1, lat2, lat0, lon0))
    n = math.log(math.cos(f1) / math.cos(f2)) / math.log(math.tan(math.pi / 4 + f2 / 2) / math.tan(math.pi / 4 + f1 / 2))
    F = math.cos(f1) * math.tan(math.pi / 4 + f1 / 2) ** n / n
    rho = F / np.tan(np.pi / 4 + lat / 2) ** n
    rho0 = F / math.tan(math.pi / 4 + f0 / 2) ** n
    x = rho * np.sin(n * (lon - l0))
    y = rho0 - rho * np.cos(n * (lon - l0))
    return x, y


def fd_polygons():
    """Контуры федеральных округов (Natural Earth) в проекции карты: [(округ, [массивы x, y])]."""
    g = json.loads((ROOT / "data" / "geo" / "russia_fd.geojson").read_text(encoding="utf-8"))
    out = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = []
        for poly in polys:
            ring = np.array(poly[0])
            lon = np.where(ring[:, 0] < 0, ring[:, 0] + 360, ring[:, 0])      # Чукотка за 180°
            rings.append(lcc(lon, ring[:, 1]))
        out.append((f["properties"]["fd"], rings))
    return out


def boot_ci(x, n: int = 4000, q=(0.025, 0.975), seed: int = 1) -> tuple[float, float]:
    """Интервал для среднего по бутстрепу (95 %)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, q[0])), float(np.quantile(m, q[1]))


def agents_cost(p: dict, days: int, events: float) -> float:
    """Стоимость работы агентов за окно (как в src/v5/report.py)."""
    from src.v5.report import agents_cost as ac
    return ac(p, days, events)


def param_comments() -> dict:
    """Комментарии к параметрам config/params.yaml по полному пути («v5.svh.free_days» → «источник: …»)."""
    out, stack = {}, []
    for line in (ROOT / "config" / "params.yaml").read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].rstrip()
        if not body.strip() or body.strip().startswith("-"):
            continue
        ind = len(body) - len(body.lstrip())
        if ":" not in body:
            continue
        key = body.strip().split(":", 1)[0].strip().strip('"')
        while stack and stack[-1][0] >= ind:
            stack.pop()
        stack.append((ind, key))
        if "#" in line:
            out[".".join(k for _, k in stack)] = line.split("#", 1)[1].strip()
    return out


def get(p: dict, path: str):
    for k in path.split("."):
        p = p[k]
    return p


def rel_runs(runs, scenario: str, p: dict, res: dict, extra=None):
    """Результаты по вариантам для относительных показателей: издержки, потери и доля продаж по правилам,
    маржа спроса (одинакова у всех правил — общие случайные числа), стоимость работы агентов в варианте."""
    import pandas as pd
    from src.v5.report import agents_cost, money
    d = runs[runs.scenario == scenario] if extra is None else pd.concat([runs, extra])[lambda x: x.scenario == scenario]
    C = d.pivot(index="seed", columns="rule", values="total_cost")
    L = d.pivot(index="seed", columns="rule", values="lost_profit")
    F = d.pivot(index="seed", columns="rule", values="fill_rate")
    k = L.idxmax(axis=1)
    dm = pd.Series([L.loc[s, k[s]] / (1 - F.loc[s, k[s]]) for s in C.index], index=C.index)
    ev = d[d.rule == "P2"].set_index("seed")["events"].loc[C.index]
    days = money(p, res, scenario)["days"]
    cost = ev.map(lambda e: agents_cost(p, days, e))
    return C, L, F, dm, cost
