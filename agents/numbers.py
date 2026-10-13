"""Проверка: в тексте агента нет чисел, которых нет в результатах инструментов.

Число из текста засчитывается, если в результатах (или аргументах) вызовов есть значение, которое при округлении
до той же точности дает то же число; учитываются «тыс.» и «млн». Номера контейнеров и ссылки [cNN] не числа.
Если после фрагмента стоит ссылка [cNN], числа фрагмента сверяются именно с этим расчетом: число, взятое из другого
расчета, считается несверенным.
"""
from __future__ import annotations

import json
import re

NUM = re.compile(r"(?<![\w-])([−-]?\d{1,3}(?:[   ]\d{3})+|[−-]?\d+)(?:[,.](\d+))?\s*(тыс|млн|млрд)?")
SKIP = re.compile(r"\[c\d+\]|\bc\d{2}\b|K-\d{2}|К-\d{2}|P\d\b|P10|P50|P90|\d{4}-\d{2}-\d{2}")
SCALE = {"тыс": 1e3, "млн": 1e6, "млрд": 1e9}


ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
MONTH_BEFORE = re.compile(r"(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s*$", re.I)
MONTH = re.compile(r"\s*(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)", re.I)


def _values(obj, out: set, days: set | None = None):
    """Числа из результатов расчета; дни месяца из дат (2026-12-16 → 16) — отдельно, для «16 декабря»."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        out.add(float(obj) * 100)  # доли как проценты
        return
    if isinstance(obj, str):
        if days is not None:
            for m in ISO.finditer(obj):
                days.add(float(m.group(3)))
                days.add(float(m.group(1)) + 0.0001)   # год даты — только после названия месяца
        for m in re.finditer(r"\d+(?:[.,]\d+)?", ISO.sub(" ", obj)):
            out.add(float(m.group(0).replace(",", ".")))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _values(v, out, days)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _values(v, out, days)


def allowed_from(calls, days: set | None = None) -> set:
    vals: set = set(float(i) for i in range(0, 11))
    for c in calls:
        _values(c.result, vals, days)
        _values(c.args, vals, days)
    return vals


REF = re.compile(r"\[(c\d{2}(?:\s*,\s*c\d{2})*)\]")


def _segments(text: str):
    """Фрагменты текста и ссылки, которыми они заканчиваются: [(фрагмент, [c05, c06] | None)].
    Ссылки подряд ([c17][c18] или [c17], [c18]) — одна группа."""
    out, pos, refs, start = [], 0, [], None
    for m in REF.finditer(text):
        ids = [x.strip() for x in m.group(1).split(",")]
        if refs and text[last_end:m.start()].strip(" ,;") == "":
            refs += ids
        else:
            if refs:
                out.append((text[pos:start], refs))
                pos = last_end
            refs, start = ids, m.start()
        last_end = m.end()
    if refs:
        out.append((text[pos:start], refs))
        pos = last_end
    out.append((text[pos:], None))
    return out


def check(text: str, calls) -> dict:
    all_days: set = set()
    allowed = allowed_from(calls, all_days)
    by_id = {getattr(c, "call_id", None): c for c in calls}
    bad, n = [], 0
    for seg, ids in _segments(text or ""):
        vals, days = allowed, all_days
        if ids and any(i in by_id for i in ids):
            vals, days = set(float(i) for i in range(0, 11)), set()
            for i in ids:
                c = by_id.get(i)
                if c is not None:
                    _values(c.result, vals, days)
                    _values(c.args, vals, days)
        clean = SKIP.sub(" ", seg)
        for m in NUM.finditer(clean):
            whole, frac, unit = m.group(1), m.group(2), m.group(3)
            x = float(re.sub(r"[\s\u00a0\u202f\u2009]", "", whole).replace("\u2212", "-") + (f".{frac}" if frac else ""))
            scale = SCALE.get(unit, 1.0)
            tol = 0.5 * 10 ** (-(len(frac) if frac else 0)) * scale
            n += 1
            target = abs(x * scale)
            if not frac and not unit and MONTH.match(clean, m.end()):          # «16 декабря» — день даты
                pool = vals | {d for d in days if d < 32}
            elif not frac and not unit and MONTH_BEFORE.search(clean[:m.start()]):   # «декабря 2026» — год даты
                pool = vals | {round(d) for d in days if d > 1000}
            else:
                pool = vals
            if not any(abs(abs(v) - target) <= tol + 1e-9 for v in pool):
                bad.append(m.group(0).strip())
    return {"ok": not bad, "checked": n, "unverified": bad}


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
