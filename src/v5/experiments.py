"""Эксперименты модели v5: эффект агентов в S&OE-цикле импорта по сравнению с S&OE без агентов.

Общие случайные числа: в одном прогоне у всех правил одинаковые спрос, задержки, проверки на таможне, ожидание на СВХ.
1. Общая система импорта (обычный режим, загрузка 2026 года, сбой) и для сравнения сезонный товар: P1, P2a, P2
   и P2 без каждого рычага — разложение эффекта по рычагам схемы.
2. Чувствительность: насколько впритык приходит товар, страховой запас, задержки, проверки таможни.

Запуск: .venv/bin/python -m src.v5.experiments [--quick] [--runs-only]
--runs-only — только результаты по вариантам (outputs/model_v5/runs.csv) для отчета: P1, P2a, P2 во всех режимах.
--staff-check — проверка: S&OE без агентов со штатом ×10 (решения без очереди людей), outputs/model_v5/runs_staff.csv.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date

import numpy as np

from src.config import ROOT, load_params
from src.common import q
from src.common import apply_over
from src.v5.eta import train
from src.v5.network import RCNet
from src.v5.sim import EXTRA, LEVERS, RULES, Scenario, calibrate_staff, draw, run_one

OUT = ROOT / "outputs" / "model_v5"
KEEP = ["total_cost", "lost_profit", "markdown", "logistics", "levers_cost", "money", "fill_rate", "hours_total",
        "rho", "react_days", "events", "eta_cover"]
_W: dict = {}


def _init():
    p = load_params()
    _W["p"] = p
    _W["net"] = RCNet.load(p)
    _W["eta"] = train(p, len(_W["net"].gateways), _W["net"].gw_share)


def rules_all():
    # P0 нужен только для подбора штата (calibrate_staff); в сравнении — S&OE без агентов (P1) и агенты
    out = {k: v for k, v in RULES.items() if k != "P0"} | dict(EXTRA)
    for k in LEVERS:
        out[f"P2 без: {k}"] = replace(RULES["P2"], name=f"P2-{k}", **{k: False})
    return out


def _one(args):
    sc, seed, names, staff = args
    p = apply_over(_W["p"], sc.over)
    net, eta = _W["net"], _W["eta"]
    n = int(sc.volume or p["v4"]["volume"]["containers_base"])
    d = draw(net, p, sc, n, np.random.default_rng(seed))
    allr = rules_all()
    out = {}
    for name in names:
        r = run_one(net, p, sc, d, allr[name], staff, eta)
        out[name] = {k: r[k] for k in KEEP}
        out[name]["hours"] = r["hours"]
        out[name]["levers"] = r["levers"]
        out[name]["lost_profit_fd"] = r["lost_profit_fd"]
        out[name]["balance_ok"] = r["balance_ok"]
    return out


def scenarios(p: dict) -> dict:
    """Режимы расчета: общая система импорта (главный расчет) и сезонный товар для сравнения."""
    season = Scenario("season", wave=tuple(p["experiments"]["wave"]))
    G = p["v5"]["general"]
    gen_over = (("season.end", G["end"]), ("season.peak_multiplier", G["peak_multiplier"]),
                ("demand.safety_stock", G["safety_stock"]), ("sim.supply_window", G["supply_window"]),
                ("product.markdown", G["markdown"]))
    base = replace(season, name="general", over=gen_over)
    return {
        "general": base,
        "general_2026": replace(base, name="general_2026", over=gen_over + (("v5.eta.svh.base", [12.0, 15.0, 12.0, 4.0, 4.0]),)),
        "general_shock": replace(base, name="general_shock", shock=p["experiments"]["stress"]["shock"],
                                 wave_add=p["experiments"]["stress"]["wave_delay"]),
        "season": season,
        "season_late": replace(season, name="season_late", over=(("sim.supply_window", [-15, 40]),)),
    }


RUN_COLS = ["total_cost", "lost_profit", "markdown", "logistics", "levers_cost", "money", "fill_rate", "events", "eta_cover"]


def rows_frame(name: str, rows: list, seed: int, rules: list):
    """Результаты по каждому варианту (для распределений и интервалов в отчете): сценарий, вариант, правило, метрики."""
    import pandas as pd
    out = []
    for k, r in enumerate(rows):
        for rule in rules:
            x = r[rule]
            out.append({"scenario": name, "seed": seed + k, "rule": rule, **{c: x[c] for c in RUN_COLS},
                        **{f"lever_{a}": b for a, b in x["levers"].items()}})
    return pd.DataFrame(out)


def mc(pool, sc, runs, seed, names, staff):
    return list(pool.map(_one, [(sc, seed + i, names, staff) for i in range(runs)], chunksize=max(1, runs // 64)))


def paired(rows, a, b):
    d = [r[a]["total_cost"] - r[b]["total_cost"] for r in rows]
    return {"rub": q(d), "pct_of_p1": q([x / r["P1"]["total_cost"] for x, r in zip(d, rows)]) if "P1" in rows[0] else None,
            "win_share": float(np.mean([x > 0 for x in d]))}


def summarize(rows, names):
    out = {}
    for k in names:
        out[k] = {m: q([r[k][m] for r in rows]) for m in KEEP}
        out[k]["hours"] = {h: float(np.mean([r[k]["hours"].get(h, 0.0) for r in rows])) for h in rows[0][k]["hours"]}
        out[k]["levers"] = {h: float(np.mean([r[k]["levers"][h] for r in rows])) for h in rows[0][k]["levers"]}
        out[k]["lost_profit_fd"] = {f: float(np.mean([r[k]["lost_profit_fd"][f] for r in rows])) for f in rows[0][k]["lost_profit_fd"]}
        out[k]["balance_ok"] = all(r[k]["balance_ok"] for r in rows)
    return out


def main(quick: bool = False, runs_only: bool = False, staff_check: bool = False):
    p = load_params()
    seed = p["seed"]
    runs = 60 if quick else p["v4"]["experiments"]["runs"]
    grid = 30 if quick else p["v4"]["experiments"]["grid_runs"]
    S = scenarios(p)
    base = S["general"]
    net = RCNet.load(p)
    eta = train(p, len(net.gateways), net.gw_share)
    staff = calibrate_staff(net, apply_over(p, base.over), base, eta)
    if staff_check:
        # проверка на устойчивость: S&OE без агентов, у которого людей сколько угодно (штат ×10, решения без очереди)
        import pandas as pd
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
            frames = []
            for name in ("general", "general_2026", "general_shock"):
                rows = mc(pool, S[name], runs, seed, ["P1"], staff * 10)
                fr = rows_frame(name, rows, seed, ["P1"])
                fr["rule"] = "P1_full"
                frames.append(fr)
                print(f"{name}: {time.time() - t0:.0f} с", flush=True)
        path = OUT / ("runs_staff_quick.csv" if quick else "runs_staff.csv")
        pd.concat(frames).to_csv(path, index=False)
        print(f"готово за {time.time() - t0:.0f} с → {path}")
        return
    if runs_only:
        # только результаты по вариантам для отчета: четыре правила, все режимы
        import pandas as pd
        main4 = ["P1", "P2a", "P2", "P2 одна точка"]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
            frames = []
            for sc in S.values():
                frames.append(rows_frame(sc.name, mc(pool, sc, runs, seed, main4, staff), seed, main4))
                print(f"{sc.name}: {time.time() - t0:.0f} с", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / ("runs_quick.csv" if quick else "runs.csv")
        pd.concat(frames).to_csv(path, index=False)
        print(f"готово за {time.time() - t0:.0f} с → {path}")
        return
    t0 = time.time()
    res = {"generated": date.today().isoformat(), "seed": seed, "runs": runs, "grid_runs": grid, "staff": staff,
           "rcs": len(net.names), "levers": LEVERS}
    names = list(rules_all())
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
        res["scenarios"] = {}
        frames = []
        for sc in S.values():
            rows = mc(pool, sc, runs, seed, names, staff)
            frames.append(rows_frame(sc.name, rows, seed, ["P1", "P2a", "P2", "P2 одна точка"]))
            res["scenarios"][sc.name] = {
                "params": asdict(sc), "policies": summarize(rows, names),
                "chain": {"agents_rules": paired(rows, "P1", "P2a"), "llm": paired(rows, "P2a", "P2"), "total": paired(rows, "P1", "P2"),
                          "milp_vs_rule": paired(rows, "P2", "P2+MILP"),
                          "risk_logic": paired(rows, "P2 одна точка", "P2")},
                "levers": {k: paired(rows, f"P2 без: {k}", "P2") for k in LEVERS},
            }
            print(f"{sc.name}: {time.time() - t0:.0f} с", flush=True)
        # чувствительность (обычный режим общей системы): запас в пути, задержки, таможня, спрос, цена рычага
        sens_list = [
            ("buffer_days", "sim.supply_window", [[-48, 83], [-41, 90], [-35, 96], [-30, 101]]),
            ("svh_base_shift", "v5.eta.svh.base", [[5, 6, 5, 2, 2], [8, 10, 8, 3, 3], [12, 15, 12, 4, 4], [16, 20, 16, 5, 5]]),
            ("check_share", "v5.customs.check_share", None),
            ("doc_error_share", "v5.customs.doc_error_share", None),
            ("level_sigma", "v5.demand.level_sigma", [0.10, 0.20, 0.35]),
            ("priority_fee", "v5.svh.priority_fee_rub", [20000, 50000, 150000]),
            ("structured_share", "v5.channels.structured_share", [0.2, 0.4, 0.7]),
            ("overorder", "sim.p0_overorder_range", [[0.0, 0.0], [0.10, 0.20], [0.20, 0.40]]),
        ]
        cu = p["v5"]["customs"]
        sens = {}
        for name, path, vals in sens_list:
            if vals is None:
                key = name
                lo, hi = cu.get(f"{key}_range", [cu[key] / 2, cu[key] * 2])
                vals = [lo, cu[key], hi]
            sens[name] = []
            for x in vals:
                scx = replace(base, name=f"{name}={x}", over=base.over + ((path, x),))
                lever_names = [f"P2 без: {k}" for k in LEVERS] if name == "buffer_days" else []
                rows = mc(pool, scx, grid, seed, ["P1", "P2a", "P2"] + lever_names, staff)
                sens[name].append({"value": x, "agents": paired(rows, "P1", "P2"),
                                   "llm": paired(rows, "P2a", "P2"),
                                   "levers": {k: paired(rows, f"P2 без: {k}", "P2") for k in LEVERS} if lever_names else None,
                                   "lost_profit": {k: float(np.mean([r[k]["lost_profit"] for r in rows])) for k in ("P1", "P2a", "P2")}})
            print(f"чувствительность {name}: {time.time() - t0:.0f} с", flush=True)
        res["sensitivity"] = sens
    res["duration_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.concat(frames).to_csv(OUT / ("runs_quick.csv" if quick else "runs.csv"), index=False)
    path = OUT / ("results_quick.json" if quick else "results.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"готово за {res['duration_s']} с → {path}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, runs_only="--runs-only" in sys.argv, staff_check="--staff-check" in sys.argv)
