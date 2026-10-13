"""Модель v5: агентное управление S&OE-циклом импорта.

Цепочка: зарубежное производство → точка входа (граница или порт) → таможня → СВХ (ожидание вывоза) → внутренняя
перевозка до федерального ТЦ → 41 региональный РЦ (магазин — к ближайшему РЦ) → магазины. Контейнер не закреплен за
регионом: товар лежит общим запасом в федеральном ТЦ и уходит регионам по их заявкам.

Варианты управления:
  P1  «S&OE без агентов»: заявки составляют региональные планировщики по продажам и завышают их, каждый регион
      по-своему; сроки — план и статусы, которые люди прочитали; события разбирают люди в очереди; рычаги — по их решению.
  P2a «агенты без GigaChat»: заявку составляет агент региона по прогнозу спроса с оценкой потерянных продаж; сроки —
      ML-агенты (P10–P90); оркестратор считает выгоду рычага по сценариям прихода и включает его в тот же день;
      агенты читают только структурированные новости (трекинг, EDI, статусы терминалов) и документы стандартной формы.
  P2  «агенты с GigaChat»: то же, плюс языковая модель читает письма, поток статусов и документы любой формы (с ошибками).
  P0  только для подбора штата (calibrate_staff): обычная работа по контейнерам; в сравнение не входит.

Стоимость задержки C = П + У + Р: П — упущенная прибыль, У — уценка, Р — дополнительные перевозки и хранение.
Метрики — по истинному спросу, решения — по оценке своего варианта управления.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from src.common import largest_remainder, net_window
from src.v5.eta import QS, ETAModels, congestion, ext_delay, svh_wait
from src.v5.network import RCNet


@dataclass(frozen=True)
class Rule:
    name: str
    hub_buffer: bool = True        # федеральный ТЦ держит общий запас и отгружает по потребности (иначе — по плану)
    track: bool = True             # статусы отслеживаются, события разбираются
    agents: bool = False           # ML-сроки, оценка потерянных продаж, решения в тот же день, расчеты — агенты
    llm: bool = False              # есть языковая модель (часы людей на разбор ее ошибок)
    llm_news: bool = False         # языковая модель читает письма, PDF и сообщения экспедиторов и брокеров
    llm_docs: bool = False         # языковая модель сверяет документы в любой форме (правила — только стандартную)
    risk: bool = False             # выгода рычага — ожидаемая по сценариям прихода из ML-сроков (иначе — по одной точке)
    optimizer: bool = False        # распределение запаса федерального ТЦ по минимуму стоимости задержки
    doc_check: bool = False        # таможенный агент заранее проверяет документы
    guarantee: bool = True         # рычаг «выпуск под обеспечение уплаты пошлин»
    priority: bool = True          # рычаг «приоритетный вывоз с СВХ»
    transfer: bool = True          # рычаг «пересчет распределения текущих запасов» (переброска между РЦ)
    eta_ml: bool = True            # интервальные сроки ML-агентов (иначе — план и статусы)


A = dict(agents=True, doc_check=True, risk=True)
LLM = dict(llm=True, llm_news=True, llm_docs=True)
RULES = {
    "P0": Rule("P0", hub_buffer=False, track=False, guarantee=False, priority=False, transfer=False, eta_ml=False),
    "P1": Rule("P1", eta_ml=False),
    "P2a": Rule("P2a", **A),
    "P2": Rule("P2", **LLM, **A),
}
# P2 без одного рычага — для разложения эффекта по рычагам схемы
LEVERS = {"eta_ml": "Интервальные сроки ML-агентов (раннее оповещение)", "doc_check": "Проверка документов агентом таможни",
          "guarantee": "Выпуск под обеспечение уплаты пошлин", "priority": "Приоритетный вывоз с СВХ",
          "transfer": "Переброска текущих запасов между РЦ",
          "llm_news": "LLM читает письма и сообщения экспедиторов и брокеров",
          "llm_docs": "LLM сверяет документы в любой форме"}
# проверенный и отклоненный вариант: распределение запаса федерального ТЦ линейной задачей на 21 день (HiGHS)
EXTRA = {"P2+MILP": Rule("P2+MILP", optimizer=True, **LLM, **A),
         # прежняя оценка выгоды рычага — по одной точке срока (первый расчет 11.10.2026), для сравнения
         "P2 одна точка": Rule("P2-point", **LLM, **{**A, "risk": False})}


@dataclass(frozen=True)
class Scenario:
    name: str = "base"
    volume: int | None = None
    shock: float = 0.0              # отклонение спроса РЦ региона шока
    shock_fd: str = "Уральский"
    wave_add: float = 0.0           # прибавка к ожиданию вывоза с СВХ на основной точке входа в «волну», сут
    wave: tuple = (-5.0, 10.0)
    over: tuple = ()
    perfect: bool = False


@dataclass
class Draws:
    level: np.ndarray
    eps: np.ndarray
    overorder: np.ndarray
    plan: np.ndarray
    gw: np.ndarray
    pay_shift: np.ndarray
    e_noise: np.ndarray
    w_noise: np.ndarray
    cong: np.ndarray
    chk: np.ndarray
    chk_days: np.ndarray
    doc: np.ndarray
    doc_days: np.ndarray
    ocr_u: np.ndarray
    transit_noise: np.ndarray
    llm_u: np.ndarray
    lhat_eps: np.ndarray
    loss_u: np.ndarray
    rep_z: np.ndarray
    chan_u: np.ndarray = None       # канал новости: структурированный (трекинг, EDI) или письмо — сравнивается с долей
    tmpl_u: np.ndarray = None       # форма документов: стандартная или свободная
    scen_u: np.ndarray = None       # сценарии прихода для ожидаемой выгоды рычага, [K, n, 4]


def draw(net: RCNet, p: dict, sc: Scenario, n: int, rng: np.random.Generator) -> Draws:
    R, T = len(net.names), net_window(p)
    v5 = p["v5"]
    sig, sig_d = v5["demand"]["level_sigma"], p["sim"]["daily_noise_sigma"]
    lo, hi = p["sim"]["p0_overorder_range"]
    a, b = p["sim"]["supply_window"]
    plan = np.sort(rng.uniform(a, b, n))
    counts = largest_remainder(net.gw_share * n)
    gw = rng.permutation(np.repeat(np.arange(len(net.gateways)), counts))
    pay = rng.random(n) < p["payments"]["p_pay"]
    plo, phi = p["payments"]["delay_days"]
    cu = v5["customs"]
    return Draws(
        level=np.exp(rng.normal(-sig ** 2 / 2, sig, R)),
        eps=np.exp(rng.normal(-sig_d ** 2 / 2, sig_d, (R, T))),
        overorder=rng.uniform(lo, hi, R), plan=plan, gw=gw,
        pay_shift=np.where(pay, rng.uniform(plo, phi, n), 0.0),
        e_noise=rng.normal(0, 1, n), w_noise=rng.normal(0, 1, n),
        cong=congestion(rng, len(net.gateways), 400, v5["eta"]["congestion_rho"]),
        chk=rng.random(n) < cu["check_share"], chk_days=rng.uniform(*cu["check_days"], n),
        doc=rng.random(n) < cu["doc_error_share"], doc_days=rng.uniform(*cu["doc_error_days"], n),
        ocr_u=rng.random(n), transit_noise=rng.normal(0, 1, n), llm_u=rng.random((n, 3)),
        lhat_eps=rng.normal(0, 1, (R, T)), loss_u=rng.random((n, 3, 6)), rep_z=rng.normal(0, 1, (n, 3)),
        # новые поля — в конце: прежние случайные числа не меняются
        chan_u=rng.random((n, 3)), tmpl_u=rng.random(n), scen_u=rng.random((v5["risk"]["scenarios"], n, 4)))


def _qsample(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Значение по квантилям P10, P50, P90 [n, 3] и равномерным u [K, n]: кусочно-линейно, хвосты — с тем же наклоном,
    не меньше нуля."""
    q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    lo = q50 - (0.5 - u) / 0.4 * (q50 - q10)
    hi = q50 + (u - 0.5) / 0.4 * (q90 - q50)
    return np.maximum(0.0, np.where(u < 0.5, lo, hi))


class Season:
    """Сезон для одного правила. Время t — дни от начала окна продаж; массивы дней сдвинуты на t0."""

    def __init__(self, net: RCNet, p: dict, sc: Scenario, d: Draws, rule: Rule, staff: float, eta: ETAModels):
        self.net, self.p, self.sc, self.d, self.rule, self.eta = net, p, sc, d, rule, eta
        v5, v4 = p["v5"], p["v4"]
        self.v5 = v5
        R = self.R = len(net.names)
        T = self.T = net_window(p)
        n = self.N = len(d.plan)
        self.U = float(p["product"]["units_per_container"])
        price = self.price = float(p["product"]["price_rub"])
        self.m = price * p["product"]["margin"]
        self.unit_cost = price * (1 - p["product"]["margin"])
        self.delta = p["product"]["markdown"]
        self.start = date.fromisoformat(p["season"]["start"])
        peak = (date.fromisoformat(p["season"]["peak_from"]) - self.start).days
        prof = np.where(np.arange(T) >= peak, p["season"]["peak_multiplier"], 1.0)
        self.profile = prof / prof.sum()
        F = n * self.U / (1 + p["demand"]["safety_stock"])
        self.plan_rt = np.outer(net.share * F, self.profile)
        shock = np.ones(R)
        shock[[i for i, f in enumerate(net.fd) if f == sc.shock_fd]] += sc.shock
        self.demand = self.plan_rt * (d.level * shock)[:, None] * d.eps
        self.orders = net.share * F * (1 + d.overorder)
        self.money_rate = p["economics"]["money_rate"]
        self.rail_rate = p["sim"]["rail_rub_per_km"]
        self.truck_rate, self.truck_speed, self.truck_load = (p["rates"][k] for k in ("truck_rub_km", "truck_speed_km_day", "truck_load_days"))
        # контейнеры: этапы и сроки (истина)
        e, cu, sv = v5["eta"], v5["customs"], v5["svh"]
        self.gw = d.gw.astype(int)
        self.plan = d.plan.copy()
        self.lead = net.lead[self.gw]
        self.t_dep = self.plan - self.lead + d.pay_shift                 # выход от производителя (оплата сдвигает отгрузку)
        dep_i = np.clip(np.floor(self.t_dep - 1).astype(int) + 200, 0, 399)
        self.cong_dep = d.cong[self.gw, dep_i]
        self.ext = ext_delay(self.cong_dep, self.gw, d.e_noise, e["ext"])
        self.t_gw = self.plan + d.pay_shift + self.ext                   # прибытие на точку входа
        self.doc = d.doc.copy()
        self.caught = np.zeros(n, dtype=bool)                            # ошибку нашли заранее
        ch = v5["channels"]
        self.struct_share = ch["structured_share"]
        if rule.doc_check:
            # правила сверяют только стандартную форму документов, языковая модель — любую
            readable = np.ones(n, dtype=bool) if rule.llm_docs else d.tmpl_u < ch["doc_template_share"]
            self.caught = self.doc & readable & (d.ocr_u < cu["ocr_catch"])
        self.chk = d.chk.copy()
        self.chk_days = d.chk_days.copy()
        self.guar = np.zeros(n, dtype=bool)                              # выпуск под обеспечение
        gw_i = np.clip(np.floor(self.t_gw).astype(int) + 200, 0, 399)
        self.cong_gw = d.cong[self.gw, gw_i]
        wave = (self.gw == 0) & (self.plan >= sc.wave[0]) & (self.plan <= sc.wave[1])
        self.wait = svh_wait(self.cong_gw, self.gw, d.w_noise, e["svh"]) + np.where(wave, sc.wave_add, 0.0)
        self.prio = np.zeros(n, dtype=bool)                              # приоритетный вывоз
        self.plan_wait = np.array(e["svh"]["base"])[self.gw]             # обычное ожидание вывоза — в плане экспедитора
        self.rel_floor = np.full(n, -np.inf)                             # выпуск не раньше дня решения
        self.disp_floor = np.full(n, -np.inf)
        self.transit = net.gw_km[self.gw] / net.speed * (1 + e["internal_sigma"] * d.transit_noise)
        self.transit = np.maximum(self.transit, 0.5 * net.gw_km[self.gw] / net.speed)
        self.hub_hand = v4["network_a"]["hub_handling_days"]
        self.cu, self.sv = cu, sv
        self._times()
        self.arrived = np.zeros(n, dtype=bool)
        # время
        self.t0 = int(math.floor(min(self.t_dep.min(), -5))) - 1
        self.Dn = T + 160 - self.t0
        # регионы и федеральный ТЦ
        self.stock = np.zeros(R)
        self.inb = np.zeros((R, self.Dn))
        self.hub = 0.0
        self.hub_hold = np.zeros(R)        # P0: штук в федеральном ТЦ, закрепленных за регионом по плану
        self.sales = np.zeros((R, T))
        self.lost = np.zeros((R, T))
        self.lhat = np.zeros((R, T))
        self.stockout = [None] * R
        lh = v4["lost_sales_estimate"]
        self.lh_bias, self.lh_sig = lh["bias"], lh["sigma"]
        hb = v5["hub"]
        self.review, self.safety = hb["review_days"], hb["safety_days"]
        self.cost = {"fed": 0.0, "truck": 0.0, "guarantee": 0.0, "priority": 0.0, "money": 0.0, "rail_in": 0.0}
        # люди и очередь
        ppl = v4["people"]
        self.staff = staff
        self.hours_day, self.workdays = ppl["hours_per_day"], set(ppl["workdays"])
        self.ev_hours = ppl["event_hours_llm"] if rule.llm else (ppl["event_hours_rules_agents"] if rule.agents else ppl["event_hours"])
        self.st_min = ppl["status_minutes"]
        self.routine = ppl["routine_hours_per_container"]
        st_ = v4["statuses"]
        self.interval = 7.0 / st_["per_week"]
        self.per_week = st_["per_week"]
        self.q, self.sender = st_["loss_q"], st_["sender_delay_days"]
        self.sigma_rep = st_["reported_delay_sigma"]
        llm = v4["llm"]
        self.err = llm["error_rate"]
        mixe = llm["error_mix"]
        self.err_h = self.err * (mixe["false_event"] * llm["false_event_hours"] + mixe["unsure"] * llm["unsure_hours"]) / sum(mixe.values())
        self.heap: list = []
        self.seq = 0
        self.backlog = 0.0
        self.routine_backlog = 0.0
        self.hours = {"routine": 0.0, "status": 0.0, "event": 0.0, "llm_errors": 0.0, "docs": 0.0}
        self.work_day = np.zeros(self.Dn)
        self.cap_total = 0.0
        self.events = []
        self.limit = v4["corridor"]["limit_rub"]
        self.eta_thr = v4["corridor"]["eta_threshold_days"]
        self.cooldown = v4["corridor"]["event_cooldown_days"]
        self.protect = v4["corridor"]["donor_protect_days"]
        self.cool = np.full(R, -np.inf)
        self.levers = {"guarantee": 0, "priority": 0, "transfer": 0, "doc_caught": int(self.caught.sum())}
        self.kn_delay = np.zeros(n)          # известная правилу задержка (план + статусы) — для P1
        self.kn = np.zeros((n, 3), dtype=bool)
        self.cover = []                      # калибровка интервала прибытия (P2a, P2)
        self.today = self.t0
        self._ml()
        self._news()

    # --- сроки ---
    def _times(self):
        cu, sv = self.cu, self.sv
        self.custom_days = cu["release_days"] + np.where(self.doc & ~self.caught, self.d.doc_days, 0.0) \
            + np.where(self.chk & ~self.guar, self.chk_days, 0.0) + np.where(self.chk & self.guar, cu["guarantee_days"], 0.0)
        self.t_rel = np.maximum(self.t_gw + self.custom_days, self.rel_floor)
        w = np.where(self.prio, np.minimum(self.wait, sv["priority_wait_days"]), self.wait)
        self.t_disp = np.maximum(self.t_rel + w, self.disp_floor)
        self.t_fed = self.t_disp + self.transit + self.hub_hand

    def _ml(self):
        """Прогнозы ML-агентов: интервал задержки внешней доставки (при выходе от производителя) и ожидания вывоза с СВХ
        (по загрузке на дату прихода по прогнозу). Для правил без ML — план."""
        week = ((self.plan + 300) // 7) % 52
        self.q_ext = self.eta.predict_ext(self.gw, self.cong_dep, week, (self.d.pay_shift > 0).astype(float))
        self.q_svh = self.eta.predict_svh(self.gw, self.cong_gw, week)
        self.q_ext = np.sort(np.maximum(self.q_ext, 0.0), axis=1)
        self.q_svh = np.sort(np.maximum(self.q_svh, 0.0), axis=1)

    def view_fed(self, t: float) -> np.ndarray:
        """Срок прихода в федеральный ТЦ по взгляду правила (P50); для прибывших — факт."""
        base_tr = self.net.gw_km[self.gw] / self.net.speed + self.hub_hand
        rel_plan = self.cu["release_days"]
        if self.sc.perfect:
            return self.t_fed.copy()
        if self.rule.eta_ml and self.rule.agents:
            gw_eta = np.where(t >= self.t_gw, self.t_gw, self.plan + self.d.pay_shift + self.q_ext[:, 1])
            cust = np.where(t >= self.t_gw, self.custom_days, rel_plan + np.where(self.doc & ~self.caught, self.d.doc_days, 0.0))
            svh = np.where(t >= self.t_rel, np.where(self.prio, np.minimum(self.wait, self.sv["priority_wait_days"]), self.wait),
                           self.q_svh[:, 1])
            tr = np.where(t >= self.t_disp, self.transit + self.hub_hand, base_tr)        # GPS после отправки
            return np.where(self.arrived, self.t_fed, gw_eta + cust + svh + tr)
        # план (с обычным ожиданием вывоза) + известные из статусов сдвиги (оплата, задержка до границы, таможня, СВХ)
        plan_fed = self.plan + rel_plan + self.plan_wait + base_tr
        return np.where(self.arrived, self.t_fed, np.maximum(plan_fed + self.kn_delay, t))

    # --- статусы и новости ---
    def _news(self):
        """Новости в статусах: задержка до границы (в плановый день прихода), проверка таможни (в день прихода),
        долгое ожидание вывоза с СВХ (при выпуске). Агенты видят их сразу (структурные данные, API, письма — модель),
        люди — через очередь статусов."""
        self.news_day: dict[int, list] = {}
        if not self.rule.track:
            return
        late = np.maximum(0.0, self.t_gw - (self.plan + self.d.pay_shift)) + self.d.pay_shift
        items = [(self.plan, 0, late), (self.t_gw, 1, np.where(self.chk & ~self.guar, self.chk_days, 0.0)),
                 (self.t_rel, 2, self.wait - self.plan_wait)]
        for when, w, val in items:
            for i in np.nonzero(val > self.eta_thr)[0]:
                self.news_day.setdefault(int(math.floor(when[i])), []).append((int(i), w, float(val[i])))

    def news_today(self, t: int):
        for i, w, val in self.news_day.pop(t, []):
            # агенты сразу видят структурированные новости (трекинг, EDI, статусы терминалов и РЖД); письма, PDF и
            # сообщения — только если их читает языковая модель (и она не пропустила новость); остальное — людям
            structured = self.d.chan_u[i, w] < self.struct_share
            read_llm = self.rule.llm_news and self.d.llm_u[i, w] >= self.err * 0.25
            if self.rule.agents and (structured or read_llm):
                self.learn(i, w, val, t, instant=True)
            else:                          # люди: статус в очереди, может потеряться
                lost = 0
                while lost < 6 and self.d.loss_u[i, w, lost] < self.q:
                    lost += 1
                day = int(math.floor(t + self.sender + lost * self.interval))
                self.human_due.setdefault(day, []).append((i, w, val))

    def human_news_today(self, t: int):
        for i, w, val in self.human_due.pop(t, []):
            self.push(self._dl(i), "news", self.st_min / 60.0, (i, w, val))

    def learn(self, i: int, w: int, val: float, t: int, instant: bool = False):
        if self.kn[i, w]:
            return
        self.kn[i, w] = True
        rep = val * math.exp(self.sigma_rep * self.d.rep_z[i, w])
        self.kn_delay[i] += rep
        kind = ("eta", "check", "svh")[w]
        self.events.append({"kind": kind, "i": i, "t_true": float(t), "t_known": float(t), "t_dec": None})
        self.push(self._dl(i), "event", self.ev_hours, (kind, i, len(self.events) - 1))

    def _dl(self, i: int) -> float:
        return float(self.view_fed(self.today)[i])

    # --- очередь людей ---
    def push(self, key: float, kind: str, hours: float, payload=None):
        if hours <= 0:
            return
        if self.rule.agents and kind == "event":
            # агенты готовят вариант сразу, человек утверждает в тот же день
            self.hours["event"] += hours
            self.work_day[self.today - self.t0] += hours
            self.done(kind, payload, self.today)
            return
        self.seq += 1
        heapq.heappush(self.heap, (max(key, float(self.today)), self.seq, kind, hours, payload))
        self.backlog += hours
        self.work_day[self.today - self.t0] += hours

    def work(self, t: int):
        wd = (self.start + timedelta(days=t)).weekday()
        cap = self.staff * self.hours_day if wd in self.workdays else 0.0
        self.cap_total += cap
        take = min(cap, self.routine_backlog)
        self.routine_backlog -= take
        self.hours["routine"] += take
        cap -= take
        while cap > 1e-9 and self.heap:
            key, sq, kind, h, payload = heapq.heappop(self.heap)
            use = min(h, cap)
            cap -= use
            self.backlog -= use
            self.hours["status" if kind in ("news", "status") else kind] = self.hours.get("status" if kind in ("news", "status") else kind, 0.0) + use
            if use < h - 1e-12:
                heapq.heappush(self.heap, (key, sq, kind, h - use, payload))
                break
            self.done(kind, payload, t)

    def done(self, kind: str, payload, t: int):
        if kind == "news":
            i, w, val = payload
            self.learn(i, w, val, t)
        elif kind == "event":
            ev_kind, i, k = payload
            self.events[k]["t_dec"] = float(t)
            if ev_kind == "check":
                self.decide_guarantee(i, t)
            elif ev_kind == "svh":
                self.decide_priority(i, t)
        elif kind == "region":
            self.decide_transfer(payload, t)

    def statuses_today(self, t: int):
        live = (self.t_dep <= t) & (t < self.t_fed)
        k = int(live.sum())
        if not k:
            return
        cnt = k * self.per_week / 7.0
        if self.rule.llm_news:            # поток статусов и писем читает языковая модель; людям — только ее ошибки
            self.push(float(t) + 30, "llm_errors", cnt * self.err_h)
        else:
            self.push(float(t) + 30, "status", cnt * self.st_min / 60.0)

    # --- стоимость задержки для сети: сколько продаж теряется, если товар придет позже ---
    def rate(self, t: int) -> np.ndarray:
        if self.sc.perfect:
            return self.demand[:, min(max(t, 0), self.T - 1)].copy()
        if t <= 0:
            return self.plan_rt[:, 0].copy()
        a = max(0, t - 7)
        s = self.sales[:, a:t].sum(axis=1)
        if self.rule.agents:
            s = s + self.lhat[:, a:t].sum(axis=1)
        r = s / (t - a)
        if not self.rule.agents:
            # регион без продаж за неделю (товара не было) — планировщик заказывает по плану, а не ноль
            r = np.where(s <= 1e-9, self.plan_rt[:, min(t - 1, self.T - 1)], r)
        return r

    def need(self, t: int, start: int, D: int) -> np.ndarray:
        days = np.arange(start, start + D)
        ok = (days >= 0) & (days < self.T)
        prof = np.where(ok, self.profile[np.clip(days, 0, self.T - 1)], 0.0)
        if self.sc.perfect:
            return np.where(ok, self.demand[:, np.clip(days, 0, self.T - 1)], 0.0)
        ref = self.profile[min(max(t - 1, 0), self.T - 1)]
        return self.rate(t)[:, None] * prof[None, :] / ref

    def eta_scen(self, t: float) -> np.ndarray:
        """Сценарии срока прихода в федеральный ТЦ [K, n] по взгляду ML-агентов: неизвестное еще — по квантилям
        P10, P50, P90 (кусочно-линейно), проверка таможни — с ее долей и длительностью; известное — факт."""
        Us = self.d.scen_u
        cu = self.cu
        base_tr = self.net.gw_km[self.gw] / self.net.speed + self.hub_hand
        gw_eta = np.where(t >= self.t_gw, self.t_gw, self.plan + self.d.pay_shift + _qsample(self.q_ext, Us[:, :, 0]))
        chk = np.where(Us[:, :, 2] < cu["check_share"], cu["check_days"][0] + Us[:, :, 3] * (cu["check_days"][1] - cu["check_days"][0]), 0.0)
        cust = np.where(t >= self.t_gw, self.custom_days,
                        cu["release_days"] + np.where(self.doc & ~self.caught, self.d.doc_days, 0.0) + chk)
        svh = np.where(t >= self.t_rel, np.where(self.prio, np.minimum(self.wait, self.sv["priority_wait_days"]), self.wait),
                       _qsample(self.q_svh, Us[:, :, 1]))
        tr = np.where(t >= self.t_disp, self.transit + self.hub_hand, base_tr)
        return np.where(self.arrived, self.t_fed, gw_eta + cust + svh + tr)

    def network_gain(self, i: int, days_earlier: float, t: int) -> float:
        """Стоимость задержки контейнера i: сколько прибыли сеть сохранит, если он придет в федеральный ТЦ на
        days_earlier раньше. Запас одного РЦ не закрывает дефицит другого: для каждого региона — недобор к дню,
        когда товар из федерального ТЦ еще успеет доехать (с учетом срока доставки), сумма по регионам — потребность
        к федеральному ТЦ; ее закрывают запас федерального ТЦ и приходы. Потерянные продажи — наибольший разрыв
        потребности и прихода. Агенты (risk) считают ожидаемую выгоду по сценариям прихода из ML-сроков: потеря
        продаж выпуклая, и оценка по одной точке ее занижает. Остальные — по одной точке своего взгляда на сроки."""
        net = self.net
        start = t + 1
        D = max(1, self.T - start)
        need = self.need(t, start, D)
        i0 = start - self.t0
        inb = self.inb[:, i0:i0 + D]
        X = np.cumsum(need - inb, axis=1)
        gap = np.maximum(0.0, np.maximum.accumulate(X, axis=1) - self.stock[:, None])   # недобор региона к дню k
        lead = np.ceil(net.from_fed_days).astype(int)
        Nh = np.zeros(D)
        for r in range(self.R):
            k = np.minimum(np.arange(D) + lead[r], D - 1)
            Nh += gap[r, k]
        if self.rule.risk and self.rule.eta_ml and self.rule.agents and not self.sc.perfect:
            E = self.eta_scen(t)
        else:
            E = self.view_fed(t)[None, :]
        live = ~self.arrived
        idx = np.floor(E).astype(int) - start

        def shortfall(ix):
            out = np.zeros(len(ix))
            for k in range(len(ix)):
                m = live & (ix[k] < D)
                arr = np.bincount(np.maximum(ix[k][m], 0), minlength=D)[:D] * self.U
                out[k] = max(0.0, float((Nh - (self.hub + np.cumsum(arr))).max()))
            return out

        L0 = shortfall(idx)
        if not live[i]:
            return 0.0
        ix2 = idx.copy()
        ix2[:, i] = np.floor(E[:, i] - days_earlier).astype(int) - start
        L1 = shortfall(ix2)
        return float((L0 - L1).mean()) * self.m

    # --- рычаги таможни и СВХ ---
    def decide_guarantee(self, i: int, t: int):
        """Выпуск под обеспечение: товар выпускают сразу, пока идет проверка; цена — стоимость обеспечения."""
        if not self.rule.guarantee or self.guar[i] or not self.chk[i] or t >= self.t_rel[i]:
            return
        cu = self.cu
        amount = self.U * cu["customs_value_rub"] * (cu["duty_rate"] + cu["vat"] * (1 + cu["duty_rate"]))
        cost = amount * cu["guarantee_rate_year"] * max(self.chk_days[i], cu["guarantee_min_days"]) / 365 + cu["guarantee_fee_rub"]
        saved = max(0.0, self.t_rel[i] - max(t, self.t_gw[i]) - cu["guarantee_days"])
        gain = self.network_gain(i, saved, t)
        if gain > cost and (self.rule.agents or cost <= self.limit):
            self.guar[i] = True
            self.cost["guarantee"] += cost
            self.levers["guarantee"] += 1
            self._retime(i, t)

    def decide_priority(self, i: int, t: int):
        """Приоритетный вывоз с СВХ: ожидание сокращается до priority_wait_days за плату."""
        if not self.rule.priority or self.prio[i] or t >= self.t_disp[i]:
            return
        sv = self.sv
        saved = max(0.0, self.t_disp[i] - max(t, self.t_rel[i]) - sv["priority_wait_days"])
        if saved <= 0:
            return
        gain = self.network_gain(i, saved, t)
        if gain > sv["priority_fee_rub"]:
            self.prio[i] = True
            self.cost["priority"] += sv["priority_fee_rub"]
            self.levers["priority"] += 1
            self._retime(i, t)

    def _retime(self, i: int, t: int):
        old = self.t_fed[i]
        if self.guar[i]:
            self.rel_floor[i] = max(self.rel_floor[i], t)
        if self.prio[i]:
            self.disp_floor[i] = max(self.disp_floor[i], t)
        self._times()
        self.kn_delay[i] -= max(0.0, old - self.t_fed[i])

    # --- федеральный ТЦ: отгрузки регионам ---
    def hub_dispatch(self, t: int):
        """Пополнение региональных РЦ из запаса федерального ТЦ: потребность — спрос на срок доставки, пересмотра и
        страхового запаса минус остаток и товар в пути. Не хватает запаса — P1 делит пропорционально потребности,
        оркестратор (P2a, P2) — по стоимости задержки: сначала тем, где дефицит наступит раньше и в пик, с учетом
        стоимости доставки."""
        if self.hub <= 1e-9:
            return
        net = self.net
        lead = net.from_fed_days
        horizon = lead + self.review + self.safety
        D = int(math.ceil(horizon.max())) + 1
        start = t
        need_d = self.need(t, start, D)
        cum = np.cumsum(need_d, axis=1)
        i0 = t - self.t0
        inb = np.cumsum(self.inb[:, i0:i0 + D], axis=1)
        rows = np.arange(self.R)
        idx = np.minimum(np.ceil(horizon).astype(int), D - 1)
        idx_c = np.minimum(np.ceil(lead + self.review).astype(int), D - 1)
        want = np.maximum(0.0, cum[rows, idx] - inb[rows, idx] - self.stock)            # до следующей отгрузки + страховой
        if not self.rule.agents:
            # федеральный ТЦ отгружает по заявкам регионов: региональные планировщики завышают заявку, каждый по-своему
            # (sim.p0_overorder_range); агенты регионов формируют заявку по прогнозу спроса без завышения
            want = want * (1 + self.d.overorder)
        tot = want.sum()
        if self.rule.optimizer:
            if (t - self.t0) % self.review:
                return
            x = self.lp_dispatch(t, want)
        elif tot <= 1e-9:
            return
        elif tot <= self.hub:
            x = want
        else:
            x = want * self.hub / tot
        for r in np.nonzero(x > 1e-6)[0]:
            self._ship_from_fed(r, x[r], t)
        self.hub -= x.sum()

    def lp_dispatch(self, t: int, want: np.ndarray) -> np.ndarray:
        """Оркестратор: минимум стоимости задержки C = П + У + Р на горизонте H дней (скользящий план, HiGHS).
        Переменные: отгрузки из федерального ТЦ s[r,k] в день t+k (приходят через срок доставки), потерянные продажи
        l[r,k], запас регионов I[r,k], запас федерального ТЦ h[k]. Ожидаемые приходы в федеральный ТЦ — по срокам
        правила (у агентов — P50 ML-агентов). Цель: маржа × потери + доставка + малый штраф за запас в регионах в конце
        горизонта (запас в федеральном ТЦ гибче); исполняются отгрузки сегодняшнего дня."""
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        R, H = self.R, self.v5["hub"]["lp_horizon_days"]
        net = self.net
        L = np.ceil(net.from_fed_days).astype(int)
        D = self.need(t, t, H)                                   # спрос по взгляду правила [R, H]
        i0 = t - self.t0
        a = self.inb[:, i0:i0 + H]                               # уже отгруженное в регионы
        eta = self.view_fed(t)
        inflow = np.zeros(H)
        for j in np.nonzero(~self.arrived)[0]:
            k = int(math.floor(eta[j])) - t
            if 0 <= k < H:
                inflow[k] += self.U
        nS, nL, nI = R * H, R * H, R * H
        oS, oL, oI, oH = 0, nS, nS + nL, nS + nL + nI
        oU = oH + H                                              # недобор до страхового запаса u[r,k] ≥ ss − I
        nv = oU + R * H
        ss = self.rate(t) * self.safety
        sid = lambda r, k: oS + r * H + k            # noqa: E731
        lid = lambda r, k: oL + r * H + k            # noqa: E731
        iid = lambda r, k: oI + r * H + k            # noqa: E731
        c = np.zeros(nv)
        tr = net.from_fed_rub / self.U
        for r in range(R):
            c[oS + r * H: oS + (r + 1) * H] = tr[r]
            c[oL + r * H: oL + (r + 1) * H] = self.m
            c[iid(r, H - 1)] = 1.0                                # запас в федеральном ТЦ гибче, чем в регионе
            c[oU + r * H: oU + (r + 1) * H] = self.m * self.v5["hub"]["lp_safety_weight"]
        # баланс региона: I[r,k] − I[r,k−1] − s[r,k−L] + D − l = a  →  I − I_prev − s + (−l) = a − D
        rows, cols, vals, rhs = [], [], [], []
        e = 0
        for r in range(R):
            for k in range(H):
                rows += [e, e]
                cols += [iid(r, k), lid(r, k)]
                vals += [1.0, -1.0]
                if k > 0:
                    rows.append(e); cols.append(iid(r, k - 1)); vals.append(-1.0)
                if k - L[r] >= 0:
                    rows.append(e); cols.append(sid(r, k - L[r])); vals.append(-1.0)
                rhs.append(a[r, k] - D[r, k] + (self.stock[r] if k == 0 else 0.0))
                e += 1
        # баланс федерального ТЦ: h[k] − h[k−1] + Σ_r s[r,k] = inflow[k] (+ запас сегодня при k = 0)
        for k in range(H):
            rows.append(e); cols.append(oH + k); vals.append(1.0)
            if k > 0:
                rows.append(e); cols.append(oH + k - 1); vals.append(-1.0)
            for r in range(R):
                rows.append(e); cols.append(sid(r, k)); vals.append(1.0)
            rhs.append(inflow[k] + (self.hub if k == 0 else 0.0))
            e += 1
        A = coo_matrix((vals, (rows, cols)), shape=(e, nv)).tocsr()
        # страховой запас (мягко): −I − u ≤ −ss
        ur, uc, uv, ub = [], [], [], []
        q = 0
        for r in range(R):
            for k in range(H):
                ur += [q, q]; uc += [iid(r, k), oU + r * H + k]; uv += [-1.0, -1.0]; ub.append(-ss[r]); q += 1
        A_ub = coo_matrix((uv, (ur, uc)), shape=(q, nv)).tocsr()
        bounds = [(0, None)] * nv
        for r in range(R):                                        # потери не больше спроса
            for k in range(H):
                bounds[lid(r, k)] = (0, D[r, k])
        res = linprog(c, A_ub=A_ub, b_ub=np.array(ub), A_eq=A, b_eq=np.array(rhs), bounds=bounds, method="highs")
        if not res.success:
            return want * self.hub / max(want.sum(), 1e-9)
        x = np.maximum(0.0, res.x[oS:oS + nS].reshape(R, H)[:, 0])
        tot = x.sum()
        return x * min(1.0, self.hub / tot) if tot > 0 else x

    @staticmethod
    def _short_day(need: np.ndarray, stock: float, inb: np.ndarray) -> float:
        q = stock + np.cumsum(inb) - np.cumsum(need)
        k = np.nonzero(q < 0)[0]
        return float(k[0]) if k.size else 999.0

    def _ship_from_fed(self, r: int, units: float, t: int):
        net = self.net
        arr = t + net.from_fed_days[r]
        k = min(max(int(math.floor(arr)) - self.t0, t - self.t0 + 1), self.Dn - 1)
        self.inb[r, k] += units
        self.cost["fed"] += units / self.U * net.from_fed_rub[r]

    def p0_plan(self):
        """P0: при заказе контейнеры закреплены за регионами по заявкам (наибольший остаток долей)."""
        shares = self.orders / self.orders.sum()
        self.p0_share = shares

    def arrive_fed(self, t: int):
        ready = np.nonzero(~self.arrived & (np.floor(self.t_fed).astype(int) <= t))[0]
        if not ready.size:
            return
        self.arrived[ready] = True
        self.cost["rail_in"] += float(self.net.gw_km[self.gw[ready]].sum()) * self.rail_rate
        units = self.U * ready.size
        if self.rule.eta_ml and self.rule.agents and not self.sc.perfect:
            for i in ready:
                lo = self.plan[i] + self.d.pay_shift[i] + self.q_ext[i, 0]
                hi = self.plan[i] + self.d.pay_shift[i] + self.q_ext[i, 2]
                self.cover.append(lo <= self.t_gw[i] <= hi)
        if self.rule.hub_buffer:
            self.hub += units
        else:
            # P0: отгрузка по плану сразу, долями по заявкам
            for r in range(self.R):
                self._ship_from_fed(r, units * self.p0_share[r], t)

    # --- переброска запасов между РЦ ---
    def detect_regions(self, t: int):
        if t < 0 or not self.rule.transfer:
            return
        rt = self.rate(t)
        H = 10
        need = self.need(t, t + 1, H)
        i0 = t + 1 - self.t0
        q = self.stock[:, None] + np.cumsum(self.inb[:, i0:i0 + H], axis=1) - np.cumsum(need, axis=1)
        short = (q.min(axis=1) < -0.5 * rt) & (self.hub < need.sum(axis=1)) & (t >= self.cool)
        for r in np.nonzero(short)[0]:
            self.cool[r] = t + self.cooldown
            self.push(float(t), "region", self.ev_hours, int(r))

    def decide_transfer(self, r: int, t: int):
        """Переброска из РЦ с излишком: донору остается спрос на min(21 день, до конца окна), выгода больше перевозки."""
        if self.hub > 1e-6:
            return
        H = min(self.protect, max(1, self.T - t - 1))
        need = self.need(t, t + 1, max(H, 1))
        surplus = self.stock - need.sum(axis=1)
        surplus[r] = 0
        dn = int(np.argmax(surplus))
        u = min(self.U, surplus[dn])
        if u < 0.1 * self.U:
            return
        km = float(self.net.road_km[dn, r])
        cost = km * self.truck_rate
        H2 = 14
        need_r = self.need(t, t + 1, H2)[r]
        short = max(0.0, need_r.sum() - self.stock[r] - self.inb[r, t + 1 - self.t0:t + 1 - self.t0 + H2].sum())
        gain = min(u, short) * self.m
        if gain <= cost or (not self.rule.agents and cost > self.limit):
            return
        self.stock[dn] -= u
        a = t + self.truck_load + km / self.truck_speed
        self.inb[r, min(max(int(math.floor(a)) - self.t0, t - self.t0 + 1), self.Dn - 1)] += u
        self.cost["truck"] += cost
        self.levers["transfer"] += 1

    def p0_react(self, t: int):
        """P0: переброска только после дефицита, с задержкой реакции."""
        lag = self.p["v4"]["sim"]["p0_reaction_lag_days"]
        for r in range(self.R):
            so = self.stockout[r]
            if so is None or t < so + lag or t < self.cool[r]:
                continue
            self.cool[r] = t + 14
            rate = self.plan_rt[:, min(max(t, 0), self.T - 1)]
            cover = self.stock / np.maximum(rate, 1e-9)
            cover[r] = -1
            dn = int(np.argmax(cover))
            if cover[dn] <= self.p["sim"]["donor_cover_days"]:
                continue
            u = min(self.U, self.stock[dn] - 14 * rate[dn])
            if u <= 0:
                continue
            km = float(self.net.road_km[dn, r])
            self.stock[dn] -= u
            a = t + self.truck_load + km / self.truck_speed
            self.inb[r, min(max(int(math.floor(a)) - self.t0, t - self.t0 + 1), self.Dn - 1)] += u
            self.cost["truck"] += km * self.truck_rate
            self.hours["event"] += self.p["v4"]["people"]["event_hours"]

    # --- день ---
    def run(self, until: int | None = None) -> dict | None:
        """Сезон день за днем; until — остановиться перед этим днем (состояние для демонстрации)."""
        self.human_due: dict[int, list] = {}
        if not self.rule.hub_buffer:
            self.p0_plan()
        routine = np.zeros(self.Dn)
        np.add.at(routine, np.clip(np.floor(self.t_dep).astype(int) - self.t0, 0, self.Dn - 1), self.routine / 2)
        np.add.at(routine, np.clip(np.floor(self.t_gw).astype(int) - self.t0, 0, self.Dn - 1), self.routine / 2)
        self.routine_day = routine
        T = self.T
        for t in range(self.t0, T if until is None else until):
            self.today = t
            self.routine_backlog += routine[t - self.t0]
            self.work_day[t - self.t0] += routine[t - self.t0]
            self.arrive_fed(t)
            # приход в региональные РЦ
            di = t - self.t0
            self.stock += self.inb[:, di]
            self.inb[:, di] = 0.0
            # продажи
            if 0 <= t < T:
                dem = self.demand[:, t]
                s = np.minimum(dem, self.stock)
                self.sales[:, t] = s
                self.lost[:, t] = dem - s
                self.stock -= s
                self.lhat[:, t] = self.lost[:, t] * np.maximum(0.0, 1 + self.lh_bias + self.lh_sig * self.d.lhat_eps[:, t])
                for r in np.nonzero(self.lost[:, t] > 1e-9)[0]:
                    if self.stockout[r] is None:
                        self.stockout[r] = t
                self.cost["money"] += float(self.stock.sum() + self.hub) * self.unit_cost * self.money_rate / 365
            if self.rule.track:
                self.news_today(t)
                self.human_news_today(t)
                self.statuses_today(t)
            self.work(t)
            if self.rule.hub_buffer:
                self.hub_dispatch(t)
            if self.rule.track:
                self.detect_regions(t)
            else:
                self.p0_react(t)
        return self.results() if until is None else None

    def results(self) -> dict:
        T, U = self.T, self.U
        dem = self.demand.sum()
        lost_u = self.lost.sum()
        future = self.inb[:, T - self.t0:].sum()
        left = float(self.stock.sum() + future + self.hub + (~self.arrived).sum() * U)
        balance = abs(self.N * U - (self.sales.sum() + left)) < 1e-6 * self.N * U
        lp = lost_u * self.m
        md = left * self.price * self.delta
        # хранение на СВХ сверх бесплатного срока (ВМТП): от прихода на точку входа до отправки
        sv = self.sv
        days = np.maximum(0.0, np.minimum(self.t_disp, T) - self.t_gw)
        over = np.maximum(0.0, days - sv["free_days"])
        storage = float((np.minimum(over, 11) * sv["storage_rub_day"] + np.maximum(over - 11, 0) * sv["storage_rub_day_after"]).sum())
        self.cost["storage"] = storage
        logi = self.cost["fed"] + self.cost["truck"] + self.cost["rail_in"] + storage
        levers = self.cost["guarantee"] + self.cost["priority"]
        total = lp + md + logi + levers + self.cost["money"]
        evs = [e for e in self.events if e["t_dec"] is not None]
        react = [e["t_dec"] - e["t_true"] for e in evs]
        caps = self.cap_total
        work = sum(self.hours.values())
        a = np.nonzero(self.routine_day)[0]
        return {
            "total_cost": total, "lost_profit": lp, "markdown": md, "logistics": logi, "levers_cost": levers, "storage": storage,
            "money": self.cost["money"], "fill_rate": float(1 - lost_u / dem), "lost_units": float(lost_u),
            "balance_ok": bool(balance), "hours": dict(self.hours), "hours_total": float(work),
            "capacity_hours": float(caps), "rho": float(self.work_day.sum() / caps) if caps else 0.0,
            "events": len(self.events), "react_days": float(np.median(react)) if react else float("nan"),
            "levers": dict(self.levers), "eta_cover": float(np.mean(self.cover)) if self.cover else float("nan"),
            "lost_profit_fd": {fd: float(sum(self.lost[i].sum() for i, f in enumerate(self.net.fd) if f == fd) * self.m)
                               for fd in sorted(set(self.net.fd))},
            "fed_days_mean": float(np.mean(self.t_fed - (self.plan + self.cu["release_days"] + self.plan_wait + self.net.gw_km[self.gw] / self.net.speed + self.hub_hand))),
        }


def run_one(net, p, sc, d, rule, staff, eta) -> dict:
    s = Season(net, p, sc, d, rule, staff, eta)
    r = s.run()
    r["rule"] = rule.name
    return r


def busy_hours_per_day(s: Season) -> float:
    """Обычная работа центра импорта в загруженном окне (80 % работы) на рабочий день."""
    c = np.cumsum(s.routine_day)
    a = int(np.searchsorted(c, 0.1 * c[-1]))
    b = int(np.searchsorted(c, 0.9 * c[-1]))
    wd = sum(1 for k in range(a, b + 1) if (s.start + timedelta(days=k + s.t0)).weekday() in s.workdays)
    return float(s.routine_day[a:b + 1].sum()) / max(wd, 1)


def calibrate_staff(net, p, sc, eta, runs: int = 5, seed: int = 7) -> float:
    """Штат центра импорта под ρ P0 ≈ 0,7 в загруженном окне (как в v4)."""
    n = int(sc.volume or p["v4"]["volume"]["containers_base"])
    load = []
    for k in range(runs):
        d = draw(net, p, sc, n, np.random.default_rng(seed + k))
        s = Season(net, p, sc, d, RULES["P0"], 1.0, eta)
        s.run()
        load.append(busy_hours_per_day(s) / s.hours_day)
    need = float(np.mean(load)) / p["v4"]["people"]["rho_target"]
    return round(need) if need >= 1.5 else max(0.25, round(need * 4) / 4)


def waterfill_cover(avail: np.ndarray, rate: np.ndarray, units: float) -> np.ndarray:
    """Распределить штуки так, чтобы дни продаж на момент прихода (avail + x) / rate выровнялись снизу."""
    c = avail / rate
    order = np.argsort(c)
    cs, rs = c[order], rate[order]
    acc_r = acc_q = 0.0
    lam = cs[0]
    for j in range(len(cs)):
        acc_r += rs[j]
        acc_q += cs[j] * rs[j]
        nxt = cs[j + 1] if j + 1 < len(cs) else np.inf
        lam = (units + acc_q) / acc_r
        if lam <= nxt:
            break
    return np.maximum(0.0, lam * rate - avail)
