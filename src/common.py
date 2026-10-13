"""Общие функции модели: расстояния, точки входа, округление долей, окно продаж, подстановка параметров,
квантили результатов, числа по-русски."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from src.config import ROOT

NB = '\xa0'


def gc_km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


@dataclass
class Gateway:
    name: str
    kind: str            # rail | port
    lat: float
    lon: float
    share: float         # доля контейнеров через эту точку входа
    lead_days: float     # от отгрузки у производителя до точки входа, сут (P50)


def load_gateways(p: dict) -> list[Gateway]:
    """Точки входа из data/gateways.csv (с источниками); доли и сроки — колонки share, lead_days."""
    import csv
    path = ROOT / "data" / "gateways.csv"
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if not row.get("use") or row["use"].strip() != "1":
                continue
            out.append(Gateway(row["gateway"], row["kind"], float(row["lat"]), float(row["lon"]),
                               float(row["share"]), float(row["lead_days"])))
    s = sum(g.share for g in out)
    for g in out:
        g.share /= s
    return out


def largest_remainder(x: np.ndarray) -> np.ndarray:
    base = np.floor(x).astype(int)
    rest = int(round(x.sum())) - base.sum()
    order = np.argsort(-(x - base))
    base[order[:rest]] += 1
    return base


def net_window(p: dict) -> int:
    s = p["season"]
    return (date.fromisoformat(s["end"]) - date.fromisoformat(s["start"])).days + 1


def apply_over(p: dict, over: tuple) -> dict:
    if not over:
        return p
    q = copy.deepcopy(p)
    for key, val in over:
        node = q
        *path, last = key.split(".")
        for k in path:
            node = node[k]
        node[last] = val
    return q


def q(xs) -> dict:
    a = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if a.size == 0:
        return {"p10": None, "p50": None, "p90": None, "mean": None}
    return {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean())}


def num(x, d=0):
    if x is None:
        return "—"
    s = f"{x:,.{d}f}".replace(",", NB).replace(".", ",")
    return s.replace("-", "−")


def mln(x, d=1):
    return "—" if x is None else f"{num(x / 1e6, d)}{NB}млн"


def pct(x, d=1):
    return "—" if x is None else f"{num(100 * x, d)}{NB}%"

