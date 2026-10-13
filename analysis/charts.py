"""Сложные рисунки отчета: схема решения, карта сети, диаграмма сроков контейнеров."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analysis.lib import fd_polygons, lcc
from analysis.style import ACCENT, ACCENT_L, G1, G2, G3, G4, INK, num


def _box(ax, x, y, w, h, text, fc="white", ec=G3, color=INK, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight, linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, color=G2, style="-|>", ls="-", lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def scheme(figsize=(13, 6.2)):
    """Схема: цепочка поставки снизу, агенты над своими этапами, оркестратор, человек и S&OP."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    chain = ["Зарубежное\nпроизводство", "Граница:\n4 точки входа", "Таможня", "СВХ", "Внутренняя\nперевозка", "Федеральный\nТЦ",
             "41\nрегиональный РЦ", "Магазины"]
    w, gap, y0, h = 0.108, 0.0151, 0.05, 0.13
    xs = [0.005 + k * (w + gap) for k in range(len(chain))]
    for k, (x, t) in enumerate(zip(xs, chain)):
        _box(ax, x, y0, w, h, t, fc="#f4f4f4", ec=G3, size=8.8)
        if k:
            _arrow(ax, xs[k - 1] + w, y0 + h / 2, x, y0 + h / 2, color=G2)
    ax.text(0.005, y0 + h + 0.018, "Цепочка поставки", fontsize=8.8, color=G2)
    agents = [("Агент внешней\nдоставки", 1, "срок прихода\nна границу P10–P90"), ("Агент таможни", 2, "ошибки до подачи,\nпроверка, выпуск"),
              ("Агент внутренней\nдоставки", 3.5, "ожидание вывоза,\nсрок до федерального ТЦ"), ("Агенты регионов", 6, "41 РЦ: запас, спрос,\nстоимость задержки")]
    ya, ha = 0.36, 0.19
    cx = []
    for t, k, sub in agents:
        x = xs[int(k)] + (k - int(k)) * (w + gap)
        ax.add_patch(FancyBboxPatch((x, ya), w, ha, boxstyle="round,pad=0,rounding_size=0.012", fc="#eef3ff", ec=ACCENT, lw=1.1))
        ax.text(x + w / 2, ya + ha - 0.045, t, ha="center", va="center", fontsize=8.1, weight="bold", color=INK, linespacing=1.15)
        ax.text(x + w / 2, ya + 0.05, sub, ha="center", va="center", fontsize=7.0, color=G1, linespacing=1.15)
        _arrow(ax, x + w / 2, y0 + h + 0.004, x + w / 2, ya - 0.004, color=ACCENT_L, style="-", ls=(0, (2, 2)))
        cx.append(x + w / 2)
    # LLM (GigaChat): открывает агентам письма, PDF и документы любой формы
    ax.add_patch(FancyBboxPatch((0.005, ya), w, ha, boxstyle="round,pad=0,rounding_size=0.012", fc=ACCENT, ec=ACCENT, lw=1.1))
    ax.text(0.005 + w / 2, ya + ha - 0.045, "LLM", ha="center", va="center", fontsize=8.6, weight="bold", color="white")
    ax.text(0.005 + w / 2, ya + 0.06, "письма, PDF,\nдокументы любой\nформы, объяснения", ha="center", va="center", fontsize=7.0,
            color="white", linespacing=1.15)
    _arrow(ax, 0.005 + w, ya + ha / 2, cx[0] - w / 2, ya + ha / 2, color=ACCENT)
    xo, yo, wo, ho = 0.30, 0.71, 0.40, 0.15
    _box(ax, xo, yo, wo, ho, "Оркестратор S&OE-цикла\nстоимость задержки C = П + У + Р\nвыбор рычага и приоритет регионов", fc=ACCENT, ec=ACCENT,
         color="white", size=10, weight="bold")
    for x in cx:
        _arrow(ax, x, ya + ha, xo + wo * (0.12 + 0.76 * (x - cx[0]) / (cx[-1] - cx[0])), yo, color=ACCENT)
    _box(ax, 0.745, yo, 0.25, ho, "Человек утверждает:\nв лимите — сразу,\nдороже — на S&OE-встрече", fc="white", ec=INK, size=9.2)
    _arrow(ax, xo + wo, yo + ho / 2, 0.745, yo + ho / 2, color=INK)
    _box(ax, 0.005, yo, 0.235, ho, "S&OP: месячный план,\nзакупки, хабы", fc="#f4f4f4", ec=G3, size=9.2)
    _arrow(ax, 0.24, yo + ho / 2, xo, yo + ho / 2, color=G2, style="<|-|>")
    ax.text(0.5, 0.95, "Рычаги: выпуск под обеспечение пошлин · приоритетный вывоз с СВХ · распределение запаса федерального ТЦ · "
            "переброска между РЦ · оповещение регионов", ha="center", fontsize=9, color=G1)
    return fig


def network_map(rcs: list[dict], gateways: list[dict], value=None, value_label: str = "", figsize=(12, 6.4)):
    """Карта: округа, 41 РЦ (размер — магазины), федеральный ТЦ, точки входа; value — цвет кружков (доля акцента)."""
    fig, ax = plt.subplots(figsize=figsize)
    for fd, rings in fd_polygons():
        for x, y in rings:
            ax.fill(x, y, color="#efefef", ec="white", lw=1.0, zorder=1)
    fed = next(r for r in rcs if r["is_federal"])
    fx, fy = lcc(fed["lon"], fed["lat"])
    smax = max(r["share"] for r in rcs)
    for r in rcs:
        x, y = lcc(r["lon"], r["lat"])
        ax.plot([fx, x], [fy, y], color=ACCENT, alpha=0.10, lw=0.8, zorder=2)
    for r in rcs:
        x, y = lcc(r["lon"], r["lat"])
        s = 25 + 420 * r["share"] / smax
        if value is not None:
            a = value[r["name"]]
            ax.scatter(x, y, s=s, color=ACCENT, alpha=0.15 + 0.85 * a, ec=ACCENT, lw=0.8, zorder=3)
        else:
            ax.scatter(x, y, s=s, color="white" if r["mode"] != "авто" else ACCENT_L, ec=ACCENT, lw=1.0, zorder=3)
    ax.scatter(fx, fy, s=160, color=INK, zorder=4)
    ax.annotate("Федеральный ТЦ", (fx, fy), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=10, weight="bold")
    for g in gateways:
        if g["share"] <= 0:
            continue
        x, y = lcc(g["lon"], g["lat"])
        ax.scatter(x, y, s=70, marker="D", color=INK, zorder=4)
        ax.annotate(f"{g['name']} · {num(100 * g['share'])} %", (x, y), xytext=(7, -3), textcoords="offset points", fontsize=9)
    big = sorted(rcs, key=lambda r: -r["share"])[:9]
    for r in big:
        if r["is_federal"]:
            continue
        x, y = lcc(r["lon"], r["lat"])
        ax.annotate(r["short"], (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8, color=G1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def gantt(s, idx, start, ax, label_fmt=None):
    """Этапы контейнеров: производство → граница, таможня, СВХ, внутренняя перевозка до федерального ТЦ; черта — плановый приход в федеральный ТЦ."""
    segs = [("t_dep", "t_gw", G4, "до границы"), ("t_gw", "t_rel", ACCENT_L, "таможня"), ("t_rel", "t_disp", G2, "ожидание на СВХ"),
            ("t_disp", "t_fed", G3, "внутренняя перевозка")]
    plan_fed = s.plan + s.cu["release_days"] + s.plan_wait + s.net.gw_km[s.gw] / s.net.speed + s.hub_hand
    for row, i in enumerate(idx):
        for a, b, c, _ in segs:
            x0, x1 = getattr(s, a)[i], getattr(s, b)[i]
            ax.barh(row, x1 - x0, left=x0, color=c, height=0.72, lw=0)
        if s.chk[i]:
            ax.text(s.t_gw[i] + 0.3, row, "проверка", va="center", fontsize=7, color=INK)
        ax.plot([plan_fed[i]] * 2, [row - 0.42, row + 0.42], color=INK, lw=1.4)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([f"K-{i + 1:03d} · {s.net.gateways[s.gw[i]].name}" for i in idx], fontsize=8)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=c, label=t) for _, _, c, t in segs] + [Line2D([], [], color=INK, lw=1.4, label="плановый приход в федеральный ТЦ")]
    ax.legend(handles=h, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.08), fontsize=8.5)
    ax.grid(axis="y", visible=False)
