"""Модельная сеть v5: федеральный ТЦ и 41 региональный РЦ (data/rc_network.json — расположение РЦ по открытым
данным, доля РЦ в спросе — доля магазинов, для которых он ближайший)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from src.config import ROOT
from src.common import gc_km, load_gateways

OUT = ROOT / "data" / "rc_network.json"


@dataclass
class RCNet:
    """Сеть для расчета: регионы — региональные РЦ, хаб — федеральный ТЦ."""
    p: dict
    names: list = field(default_factory=list)
    fd: list = field(default_factory=list)
    share: np.ndarray = None
    node: list = field(default_factory=list)
    gateways: list = field(default_factory=list)

    @classmethod
    def load(cls, p: dict) -> "RCNet":
        d = json.loads(OUT.read_text(encoding="utf-8"))
        n = cls(p)
        n.raw = d
        rcs = d["rcs"]
        n.names = [r["name"] for r in rcs]
        n.fd = [r["fd"] for r in rcs]
        n.share = np.array([r["share"] for r in rcs])
        n.node = [(r["lat"], r["lon"]) for r in rcs]
        n.fed = next(i for i, r in enumerate(rcs) if r["is_federal"])
        v5 = p["v5"]
        speed = p["v4"]["sim"]["rail_speed_km_day"]
        n.speed = speed
        km = np.array([r["from_fed_km"] for r in rcs])
        truck = np.array([r["mode"] == "авто" for r in rcs])
        n.from_fed_km = km
        n.truck = truck
        # сроки и стоимость доставки от федерального ТЦ (за 6 000 шт. — машину или контейнер)
        n.from_fed_days = np.where(km <= 0, 0.0, np.where(truck, p["rates"]["truck_load_days"] + km / p["rates"]["truck_speed_km_day"],
                                                         v5["rail_handling_days"] + km / speed))
        n.from_fed_rub = np.where(truck, km * p["rates"]["truck_rub_km"], km * p["sim"]["rail_rub_per_km"])
        n.gateways = load_gateways(p)
        n.gw_share = np.array([g.share for g in n.gateways])
        n.lead = np.array([g.lead_days for g in n.gateways])
        fed = n.node[n.fed]
        n.gw_km = np.array([gc_km((g.lat, g.lon), fed) * p["geography"]["tortuosity"] for g in n.gateways])
        rk = p["rates"]["road_tortuosity"]
        R = len(n.names)
        n.road_km = np.array([[gc_km(n.node[a], n.node[b]) * rk for b in range(R)] for a in range(R)])
        return n
