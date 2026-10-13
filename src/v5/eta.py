"""Сроки перевозки и ML-агенты сроков (модель v5).

Синтетический «мир сроков»: у каждой точки входа есть индекс загрузки (порт, переход) — он меняется день ото дня
(авторегрессия). Задержка внешней доставки (до границы) и ожидание вывоза с СВХ зависят от загрузки и случая.
Индекс загрузки наблюдаем (в жизни — данные портов, РЖД, экспедиторов), саму задержку — нет.

Агенты внешней и внутренней доставки — квантильная регрессия (градиентный бустинг, scikit-learn) на синтетической
истории прошлых сезонов: P10, P50, P90 задержки по признакам «точка входа, загрузка, неделя, оплата». Калибровка
проверяется на новом сезоне (в интервал P10–P90 должно попадать около 80 % прибытий).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

QS = (0.1, 0.5, 0.9)


def congestion(rng: np.random.Generator, G: int, days: int, rho: float, wave: tuple | None = None) -> np.ndarray:
    """Индекс загрузки точек входа по дням: AR(1) со стационарным N(0, 1). wave = (точка, с, по, прибавка)."""
    c = np.zeros((G, days))
    c[:, 0] = rng.normal(0, 1, G)
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, days):
        c[:, t] = rho * c[:, t - 1] + s * rng.normal(0, 1, G)
    if wave is not None:
        g, a, b, add = wave
        c[g, a:b] += add
    return c


def _soft(z: np.ndarray) -> np.ndarray:
    """Гладкий max(0, z): задержка не бывает отрицательной, но и не «прилипает» к нулю."""
    return np.logaddexp(0.0, z)


def ext_delay(c: np.ndarray, gw: np.ndarray, noise: np.ndarray, e: dict) -> np.ndarray:
    """Задержка внешней доставки до границы, сут: гладкий max(0, база точки + чувствительность × загрузка + шум)."""
    base = np.array(e["base"])[gw]
    return _soft(base + e["beta"] * c + e["sigma"] * noise)


def svh_wait(c: np.ndarray, gw: np.ndarray, noise: np.ndarray, w: dict) -> np.ndarray:
    """Ожидание вывоза с СВХ (погрузки на ж/д), сут."""
    base = np.array(w["base"])[gw]
    return _soft(base + w["beta"] * c + w["sigma"] * noise)


@dataclass
class ETAModels:
    """Обученные квантильные модели: внешняя доставка (до границы) и ожидание вывоза с СВХ."""
    ext: list
    svh: list
    G: int

    def _x(self, gw, cong, week, pay):
        X = np.zeros((len(gw), self.G + 3))
        X[np.arange(len(gw)), gw] = 1.0
        X[:, self.G] = cong
        X[:, self.G + 1] = week
        X[:, self.G + 2] = pay
        return X

    def predict_ext(self, gw, cong, week, pay) -> np.ndarray:
        X = self._x(gw, cong, week, pay)
        return np.stack([m.predict(X) for m in self.ext], axis=1)

    def predict_svh(self, gw, cong, week) -> np.ndarray:
        X = self._x(gw, cong, week, np.zeros(len(gw)))
        return np.stack([m.predict(X) for m in self.svh], axis=1)


def train(p: dict, G: int, gw_share: np.ndarray, seed: int = 2026) -> ETAModels:
    """Обучение на синтетической истории: seasons прошлых сезонов по n контейнеров (v5.eta.history)."""
    h = p["v5"]["eta"]
    rng = np.random.default_rng(seed)
    n = h["history_seasons"] * h["history_per_season"]
    gw = rng.choice(G, size=n, p=gw_share / gw_share.sum())
    days = 200
    cong = congestion(rng, G, days, h["congestion_rho"])
    t = rng.integers(0, days, n)
    week = (t // 7) % 52
    pay = (rng.random(n) < p["payments"]["p_pay"]).astype(float)
    c_obs = cong[gw, t]
    e = ext_delay(c_obs, gw, rng.normal(0, 1, n), h["ext"])
    w = svh_wait(cong[gw, np.minimum(t + 10, days - 1)], gw, rng.normal(0, 1, n), h["svh"])
    m = ETAModels([], [], G)
    X = m._x(gw, c_obs, week, pay)
    Xw = m._x(gw, cong[gw, np.minimum(t + 10, days - 1)], week, np.zeros(n))
    for q in QS:
        kw = dict(loss="quantile", alpha=q, n_estimators=150, max_depth=3, learning_rate=0.1, random_state=1)
        m.ext.append(GradientBoostingRegressor(**kw).fit(X, e))
        m.svh.append(GradientBoostingRegressor(**kw).fit(Xw, w))
    return m
