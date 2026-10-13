"""Сводка модели v5: outputs/model_v5/summary.md — прибыль от агентного S&OE-цикла в общей системе импорта.

Штат людей одинаковый во всех правилах: эффект — дополнительная прибыль от более ранних и точных решений,
а не сокращение персонала. Из прибыли вычитается стоимость работы агентов (сервер, поддержка, токены GigaChat).

Запуск: .venv/bin/python -m src.v5.report [results_quick.json]
"""
from __future__ import annotations

import json
import sys
from datetime import date

from src.config import ROOT, load_params
from src.common import mln, num, pct

TITLE = {"P1": "S&OE без агентов: общий запас в федеральном ТЦ, отгрузка по заявкам регионов, решения людей",
         "P2a": "Агенты без GigaChat", "P2": "Агенты с GigaChat (вся схема)",
         "P2 одна точка": "Агенты, выгода рычага по одной точке срока (прежняя оценка)",
         "P2+MILP": "Вся схема + оптимизатор на 21 день вперед (проверка)"}
MAIN = {"general": "Обычный режим",
        "general_2026": "Загрузка 2026 года: ожидание вывоза с СВХ в 1,5 раза дольше",
        "general_shock": "Сбой: спрос Урала +30 %, часть контейнеров задержана на 10 суток"}
APPX = {"season": "Сезонный товар: пик в декабре, остаток уценивается",
        "season_late": "Сезонный товар, поставки впритык"}
SENS = {"buffer_days": ("Окно прихода на границу, дни от 1 ноября (чем позже начало, тем меньше запас в пути)", lambda v: f"{v[0]}…{v[1]}"),
        "svh_base_shift": ("Ожидание вывоза с СВХ, база по точкам входа, сут", lambda v: "/".join(str(x) for x in v)),
        "check_share": ("Доля деклараций с дополнительной проверкой", lambda v: pct(v, 0)),
        "doc_error_share": ("Доля поставок с ошибкой в документах", lambda v: pct(v, 1)),
        "level_sigma": ("Разброс спроса РЦ к плану", lambda v: pct(v, 0)),
        "priority_fee": ("Надбавка за приоритетный вывоз с СВХ, ₽", lambda v: num(v)),
        "structured_share": ("Доля новостей в структурированном виде", lambda v: pct(v, 0)),
        "overorder": ("Завышение заявок региональными планировщиками", lambda v: f"{pct(v[0], 0)}…{pct(v[1], 0)}")}


def window_days(p: dict, sc: dict) -> int:
    """Окно продаж сценария, дни: общий режим — 120 дней, сезон — 61 день."""
    end = dict((k, v) for k, v in sc["params"]["over"]).get("season.end", p["season"]["end"])
    return (date.fromisoformat(end) - date.fromisoformat(p["season"]["start"])).days + 1


def agents_cost(p: dict, days: int, events: float) -> float:
    """Стоимость работы агентов за окно: сервер и доля разработчика на поддержке по месяцам + токены GigaChat по событиям."""
    a = p["v4"]["agents_cost"]
    months = days / (365 / 12)
    return months * (a["server_rub_month"] + a["support_fte"] * a["developer_rub_month"]) + events * a["tokens_per_event"] * a["llm_rub_per_1k"] / 1000


def money(p: dict, r: dict, key: str) -> dict:
    s = r["scenarios"][key]
    days = window_days(p, s)
    gain = s["chain"]["agents_rules"]["rub"]["mean"] + s["chain"]["llm"]["rub"]["mean"]
    cost = agents_cost(p, days, s["policies"]["P2"]["events"]["mean"])
    n = s["params"]["volume"] or p["v4"]["volume"]["containers_base"]
    return {"days": days, "gain": gain, "cost": cost, "net": gain - cost, "per_container": (gain - cost) / n,
            "year": (gain - cost) * 365 / days}


def scenario_block(r: dict, key: str, title: str) -> list[str]:
    s = r["scenarios"][key]
    pol = s["policies"]
    o = [f"\n### {title}\n",
         "| Правило | Издержки | Упущенная прибыль | Уценка | Перевозки | Рычаги | Деньги в запасе | Продано от спроса |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ("P1", "P2a", "P2", "P2 одна точка", "P2+MILP"):
        x = pol[k]
        o.append(f"| {TITLE[k]} | {mln(x['total_cost']['mean'])} | {mln(x['lost_profit']['mean'])} | {mln(x['markdown']['mean'])} | "
                 f"{mln(x['logistics']['mean'])} | {mln(x['levers_cost']['mean'], 2)} | {mln(x['money']['mean'])} | {pct(x['fill_rate']['mean'])} |")
    c = s["chain"]
    o.append(f"\nРазложение: агенты без GigaChat (P1 → P2a) {mln(c['agents_rules']['rub']['mean'], 2)}; GigaChat (P2a → P2) "
             f"{mln(c['llm']['rub']['mean'], 2)}; итого {mln(c['total']['rub']['mean'])} ({pct(c['total']['pct_of_p1']['mean'], 1)} издержек P1). "
             f"Оценка рычага по сценариям вместо одной точки: {mln(c['risk_logic']['rub']['mean'], 2)}. "
             f"Агенты лучше S&OE без агентов в {pct(c['agents_rules']['win_share'], 0)} вариантов (P1 → P2a). "
             f"Оптимизатор на 21 день {'дороже' if c['milp_vs_rule']['rub']['mean'] < 0 else 'дешевле'} простого правила на "
             f"{mln(abs(c['milp_vs_rule']['rub']['mean']), 2)}.")
    o.append("\nВклад рычагов (насколько дороже P2 без рычага): " +
             "; ".join(f"{r['levers'][k]} — {mln(v['rub']['mean'], 2)}" for k, v in s["levers"].items()) + ".")
    lv = pol["P2"]["levers"]
    o.append(f"\nСколько раз агенты применили рычаги (P2): выпуск под обеспечение — {num(lv['guarantee'], 1)}, "
             f"приоритетный вывоз — {num(lv['priority'], 1)}, переброска — {num(lv['transfer'], 1)}, "
             f"ошибок в документах найдено до подачи — {num(lv['doc_caught'], 1)}.")
    return o


def build(r: dict, p: dict) -> str:
    a = p["v4"]["agents_cost"]
    G = p["v5"]["general"]
    o = ["# Модель v5: прибыль от агентного S&OE-цикла импорта\n",
         f"Общая система импорта: контейнеры от производителя → граница → таможня → СВХ → ж/д → федеральный ТЦ → "
         f"{r['rcs']} региональный РЦ сети → магазины (каждый — к ближайшему РЦ). Поток круглый год, "
         f"{num(p['v4']['volume']['containers_base'])} контейнеров за 120 дней, {num(r['runs'])} вариантов на сценарий "
         f"({num(r['grid_runs'])} — на точку чувствительности), общие случайные числа: у всех правил одного варианта одинаковые "
         f"спрос, задержки, проверки и ожидание на СВХ.\n",
         f"**Штат людей одинаковый во всех правилах ({num(r['staff'])} чел.).** Эффект агентов — дополнительная прибыль: "
         "раньше видят задержку, заранее находят ошибки в документах, включают выпуск под обеспечение и приоритетный вывоз, "
         "когда это дешевле потерь, и отдают товар тем РЦ, где он кончится раньше. Экономия на персонале не считается.\n",
         "## Главное: чистая прибыль от агентов\n",
         "Сравнение — с тем же S&OE без агентов (P1 → P2). Из прибыли вычтена стоимость работы агентов: сервер "
         f"{num(a['server_rub_month'])} ₽ в месяц, поддержка — {pct(a['support_fte'], 0)} ставки разработчика "
         f"({num(a['developer_rub_month'])} ₽ в месяц), токены GigaChat — {num(a['tokens_per_event'])} на событие по "
         f"{str(a['llm_rub_per_1k']).replace('.', ',')} ₽ за 1 000 (без НДС).\n",
         "| Режим | Агенты сверх S&OE без агентов | Стоимость работы агентов | **Чистая прибыль от агентов** | На контейнер | В год при том же потоке |",
         "|---|---|---|---|---|---|"]
    for k, title in MAIN.items():
        if k not in r["scenarios"]:
            continue
        m = money(p, r, k)
        o.append(f"| {title} | {mln(m['gain'])} | {mln(m['cost'], 2)} | **{mln(m['net'])}** | "
                 f"{num(m['per_container'] / 1e3, 1)} тыс. ₽ | {mln(m['year'])} |")
    o.append("\nЗа 120 дней, кроме последнего столбца. «В год» — тот же поток круглый год (×365/120), это пересчет, а не отдельный расчет.")
    o.append("\n## Предпосылки общей системы (раздел `v5.general` в `config/params.yaml`)\n")
    o.append(f"- Импорт круглого года: без сезонного пика, остаток не уценивается, а стоит денег в запасе ({pct(p['economics']['money_rate'], 0)} годовых).")
    o.append(f"- Бережливый запас: закупка равна прогнозу, контейнеры приходят на границу равномерно (дни {G['supply_window'][0]}…{G['supply_window'][1]} от 1 ноября) — "
             "в пути и на складе около 10 дней продаж. Так работает поток, где деньги не замораживают в запасе.")
    o.append("- Задержки, проверки таможни, ожидание на СВХ, рычаги и их цены — те же, что в остальной модели, с источниками (`docs/sources.md`).")
    o.append("- Сезонный товар (условная категория) — в приложении: там запас на сезон большой, и агенты нужны только при поставках впритык.")
    o.append("\n## Подробно по режимам\n")
    for k, title in MAIN.items():
        if k in r["scenarios"]:
            o += scenario_block(r, k, title)
    if r.get("sensitivity"):
        o.append("\n## Когда агенты нужны: чувствительность (обычный режим)\n")
        o.append("| Условие | Значение | Агенты сверх S&OE без агентов | GigaChat | Упущенная прибыль P1 / P2a / P2 |")
        o.append("|---|---|---|---|---|")
        for k, rows in r["sensitivity"].items():
            name, f = SENS.get(k, (k, str))
            for x in rows:
                lp = x["lost_profit"]
                o.append(f"| {name} | {f(x['value'])} | {mln(x['agents']['rub']['mean'], 2)} | {mln(x['llm']['rub']['mean'], 2)} | "
                         f"{mln(lp['P1'])} / {mln(lp['P2a'])} / {mln(lp['P2'])} |")
                if x.get("levers"):
                    o.append(f"| — вклад рычагов | {f(x['value'])} | | " +
                             "; ".join(f"{r['levers'][k2]}: {mln(v['rub']['mean'], 2)}" for k2, v in x["levers"].items()) + " | |")
    if any(k in r["scenarios"] for k in APPX):
        o.append("\n## Приложение: сезонный товар\n")
        o.append("| Режим | Агенты сверх S&OE без агентов | Стоимость работы агентов за окно продаж | Чистая прибыль |")
        o.append("|---|---|---|---|")
        for k, title in APPX.items():
            if k in r["scenarios"]:
                m = money(p, r, k)
                o.append(f"| {title} | {mln(m['gain'])} | {mln(m['cost'], 2)} | {mln(m['net'])} |")
        for k, title in APPX.items():
            if k in r["scenarios"]:
                o += scenario_block(r, k, title)
    return "\n".join(o) + "\n"


def main(name: str = "results.json"):
    r = json.loads((ROOT / "outputs" / "model_v5" / name).read_text(encoding="utf-8"))
    out = ROOT / "outputs" / "model_v5" / ("summary_quick.md" if "quick" in name else "summary.md")
    out.write_text(build(r, load_params()), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json")
