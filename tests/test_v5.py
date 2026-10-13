"""Модель v5: баланс штук, общие случайные числа, калибровка ML-сроков, заявки регионов, рычаги не во вред, мир демонстрации."""
import numpy as np
import pytest

from src.config import load_params
from src.v5.eta import train
from src.v5.network import RCNet
from src.v5.sim import RULES, Scenario, draw, run_one

P = load_params()
NET = RCNet.load(P)
ETA = train(P, len(NET.gateways), NET.gw_share)
SC = Scenario("stress", volume=300, shock=0.3, wave_add=10)


@pytest.fixture(scope="module")
def rows():
    out = []
    for s in (1, 2):
        d = draw(NET, P, SC, 300, np.random.default_rng(s))
        out.append({k: run_one(NET, P, SC, d, RULES[k], 4.0, ETA) for k in ("P1", "P2a", "P2")})
    return out


def test_balance(rows):
    """Пришло = продано + остаток + в пути, у всех правил."""
    for r in rows:
        assert all(x["balance_ok"] for x in r.values())


def test_same_demand(rows):
    """Общие случайные числа: истинный спрос одинаков у всех правил одного прогона."""
    for r in rows:
        dem = {round(x["lost_units"] / (1 - x["fill_rate"])) for x in r.values() if x["fill_rate"] < 1}
        assert len(dem) <= 1


def test_eta_calibration():
    """В интервал P10–P90 ML-агента сроков попадает 80 ± 5 % прибытий на границу."""
    sc = Scenario("base", volume=500)
    cov = []
    for s in range(1, 5):
        d = draw(NET, P, sc, 500, np.random.default_rng(s))
        cov.append(run_one(NET, P, sc, d, RULES["P2"], 6.0, ETA)["eta_cover"])
    assert 0.75 <= float(np.mean(cov)) <= 0.85


def test_orders_inflated_only_without_agents():
    """Заявки регионов завышают только планировщики: у агентов отгрузки из федерального ТЦ — по прогнозу без завышения."""
    d = draw(NET, P, SC, 300, np.random.default_rng(5))
    assert d.overorder.min() >= P["sim"]["p0_overorder_range"][0] and d.overorder.max() <= P["sim"]["p0_overorder_range"][1]
    assert not RULES["P1"].agents and RULES["P2"].agents


def test_levers_cost_counted(rows):
    """Цена рычагов входит в издержки и не больше пользы: агенты не хуже S&OE без агентов больше чем на 1 %."""
    for r in rows:
        assert r["P2"]["total_cost"] <= r["P1"]["total_cost"] * 1.01


def test_demo_levers_no_lookahead():
    """Мир демонстрации: выпуск под обеспечение — только на таможне, экономия рычага не больше остатка проверки и ожидания."""
    from agents.world import DemoWorld
    w = DemoWorld("general_2026", 3, demo_day=72, volume=200)
    s, t = w.s, w.t
    for i in range(s.N):
        for lv in w.levers(i):
            if lv["id"] == "guarantee":
                assert w.stage(i) == "customs"
                assert lv["saved_days"] <= s.t_rel[i] - t + 0.05
            else:
                assert lv["saved_days"] <= s.t_disp[i] - max(t, s.t_rel[i]) + 0.05
