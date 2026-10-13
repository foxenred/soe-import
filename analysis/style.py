"""Оформление графиков отчета: шрифт Onest, один акцентный цвет, серые для остального, числа по-русски.

Правила: разряды — неразрывным пробелом, десятичная запятая, без градиентов и теней. Агенты (P2) — акцент,
остальные правила — оттенки серого.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
NB = " "
ACCENT = "#2f6bff"
ACCENT_L = "#9db8ff"
INK = "#111111"
G1, G2, G3, G4 = "#3d3d3d", "#6e6e6e", "#a3a3a3", "#d9d9d9"
RULE_COLOR = {"P1": G2, "P2a": ACCENT_L, "P2": ACCENT, "P2+MILP": G1, "P2 одна точка": G1}
RULE_NAME = {"P1": "S&OE без агентов", "P2a": "Агенты без LLM",
             "P2": "Агенты с LLM", "P2+MILP": "Агенты + оптимизатор", "P2 одна точка": "Агенты, оценка по одной точке"}
SCEN_NAME = {"general": "Обычный режим", "general_2026": "Загрузка 2026 года", "general_shock": "Сбой",
             "season": "Сезонный товар", "season_late": "Сезонный, впритык"}


def setup():
    for f in sorted((ROOT / "assets" / "fonts").glob("Onest-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "Onest", "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10.5, "axes.labelcolor": G1, "axes.edgecolor": G3,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#ececec", "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": G1, "ytick.color": G1,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 110,
        "savefig.dpi": 130, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 9.5,
        "axes.formatter.use_locale": False, "figure.autolayout": False,
    })


def num(x: float, d: int = 0) -> str:
    """12345.6 → «12 346», d знаков после запятой."""
    s = f"{x:,.{d}f}".replace(",", NB).replace(".", ",")
    return s.replace("-", "−")


def mln(x: float, d: int = 1) -> str:
    return f"{num(x / 1e6, d)}{NB}млн{NB}₽"


def pct(x: float, d: int = 0) -> str:
    return f"{num(100 * x, d)}{NB}%"


def fmt_mln(d: int = 0):
    return FuncFormatter(lambda v, _: num(v / 1e6, d))


def fmt_num(d: int = 0):
    return FuncFormatter(lambda v, _: num(v, d))


def fmt_pct(d: int = 0):
    return FuncFormatter(lambda v, _: f"{num(100 * v, d)}{NB}%")


MODEL_NOTE = "Модельные данные"


def spct(x: float, d: int = 1) -> str:
    """Процент со знаком: +2,2 % / −1,0 %."""
    v = 100 * x
    return f"{'+' if v > 0 else ('−' if v < 0 else '')}{num(abs(v), d)}{NB}%"


def pp(x: float, d: int = 1) -> str:
    """Процентные пункты со знаком."""
    v = 100 * x
    return f"{'+' if v > 0 else ('−' if v < 0 else '')}{num(abs(v), d)}{NB}п.{NB}п."


def times(x: float) -> str:
    """«в 21 раз», «в 894 раза»."""
    n = int(round(x))
    word = "раза" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "раз"
    return f"в{NB}{num(n)}{NB}{word}"


def show(fig=None):
    """Показать рисунок с подписью «Модельные данные»."""
    fig = fig or plt.gcf()
    fig.text(0.995, -0.01, MODEL_NOTE, ha="right", va="top", fontsize=8.5, color=G2)
    plt.show()


def table(df):
    """Показать таблицу с подписью «Модельные данные»."""
    from IPython.display import Markdown, display
    display(df)
    display(Markdown(f"<sub><i>{MODEL_NOTE}.</i></sub>"))


def anon(text: str) -> str:
    """Федеральный ТЦ в текстах — «федеральный ТЦ»."""
    return (text.replace("Федеральный ТЦ", "федеральный ТЦ").replace("федеральный ТЦ", "федеральный ТЦ")
            .replace("федерального ТЦ", "федерального ТЦ").replace("федеральном ТЦ", "федеральном ТЦ").replace("федеральный ТЦ", "федеральный ТЦ"))


def note(ax, text: str, y: float = -0.16):
    """Подпись-источник под графиком."""
    ax.text(0, y, text, transform=ax.transAxes, fontsize=8.5, color=G2, va="top")


def legend_rules(ax, rules, **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RULE_COLOR[r], label=RULE_NAME[r]) for r in rules], **kw)


plt.close("all")
