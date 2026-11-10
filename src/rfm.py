"""Годовой робастный метод подгонки (RFM) и поиск магнитуды полноты Mc.

Реализованы основные расчётные шаги.
Годовые доверительные интервалы берутся из сохранённого результата статьи:
их пересчёт отдельным бутстрепом при каждом запуске не требуется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .config import START_YEAR, END_YEAR


def cumulative_bins(magnitudes: np.ndarray, step: float = 0.1):
    """Кумулятивное число землетрясений в магнитудных корзинах."""
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]
    if mags.size < 5:
        return np.array([]), np.array([])
    m_min = float(mags.min())
    m_max = float(mags.max())
    # Та же схема округления границ корзин, что использована в исходном коде.
    bins = np.round(np.arange(m_min, m_max + step, step), 2)
    # Исходный код применял np.digitize: значения на крайней правой
    # границе не включаются. Это важно для точного повтора статьи.
    positions = np.digitize(mags, bins) - 1
    valid = (positions >= 0) & (positions < len(bins) - 1)
    hist = np.bincount(positions[valid], minlength=len(bins) - 1)
    centers = np.round((bins[:-1] + bins[1:]) / 2, 2)
    cumulative = np.cumsum(hist[::-1])[::-1]
    return centers[cumulative > 0], cumulative[cumulative > 0]


def robust_fit(magnitudes: np.ndarray, cumulative: np.ndarray,
               max_iter: int = 30, c: float = 6.946, tol: float = 1e-6):
    """Биквадратная IRLS-подгонка логарифма кумулятивных частот."""
    selected = np.asarray(cumulative, dtype=float) > 1e-10
    m = np.asarray(magnitudes, dtype=float)[selected]
    y = np.log10(np.asarray(cumulative, dtype=float)[selected])
    if len(m) < 3:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(m)), m])
    try:
        slope, intercept = np.polyfit(m, y, 1)
        leverage = np.diag(design @ np.linalg.pinv(design.T @ design) @ design.T)
        for _ in range(max_iter):
            residual = y - (intercept + slope * m)
            adjusted = residual / np.sqrt(np.maximum(1 - leverage, 1e-10))
            med = np.median(adjusted)
            mad = np.median(np.abs(adjusted - med))
            scaled = adjusted / (c * mad if mad > 0 else 1.0)
            weights = np.where(np.abs(scaled) < 1, (1 - scaled**2)**2, 0.0)
            coefficients = np.linalg.solve(
                design.T @ (weights[:, None] * design),
                design.T @ (weights * y),
            )
            new_intercept, new_slope = coefficients
            if (abs(new_intercept - intercept) < tol and
                    abs(new_slope - slope) < tol):
                break
            intercept, slope = new_intercept, new_slope
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan
    return float(-slope), float(intercept)


def estimate_mc(magnitudes: np.ndarray, min_mc: float = 3.5):
    """Поиск Mc по локальному минимуму нормированного абсолютного остатка R.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]
    centers, observed = cumulative_bins(mags)
    if len(centers) < 5:
        return {'mc': np.nan, 'b': np.nan, 'a': np.nan, 'r': np.nan}
    candidates = np.arange(max(float(mags.min()), min_mc), mags.max() - 0.5, 0.1)
    residuals = []
    fitted = []
    for candidate in candidates:
        mask = centers >= candidate
        if mask.sum() < 3:
            residuals.append(np.inf)
            fitted.append((np.nan, np.nan))
            continue
        b, a = robust_fit(centers[mask], observed[mask])
        if not np.isfinite(b) or not np.isfinite(a):
            residuals.append(np.inf)
            fitted.append((np.nan, np.nan))
            continue
        prediction = 10 ** (a - b * centers[mask])
        log_obs = np.log10(observed[mask])
        denominator = np.sum(np.abs(log_obs))
        r = (100 * np.sum(np.abs(log_obs - np.log10(prediction))) / denominator
             if denominator > 0 else np.inf)
        residuals.append(r)
        fitted.append((b, a))
    vals = np.asarray(residuals, dtype=float)
    if not np.isfinite(vals).any():
        return {'mc': np.nan, 'b': np.nan, 'a': np.nan, 'r': np.nan}
    if len(vals) > 1 and vals[0] < vals[1]:
        best = 0
    else:
        local = [i for i in range(1, len(vals)-1)
                 if vals[i] < vals[i-1] and vals[i] < vals[i+1]]
        best = local[0] if local else int(np.argmin(vals))
    mc = float(candidates[best])
    tail = centers >= mc
    b, a = robust_fit(centers[tail], observed[tail])
    return {'mc': mc, 'b': b, 'a': a, 'r': float(vals[best])}


def annual_rfm(catalog: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать Mc, a и b для каждого календарного года 1965–2021."""
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        magnitudes = catalog.loc[(catalog['year'] == year) &
                                 (catalog['magnitude'] >= 3.5), 'magnitude'].to_numpy()
        if len(magnitudes) < 50:
            raise ValueError(f'Год {year}: недостаточно событий для RFM')
        result = estimate_mc(magnitudes)
        rows.append({'year': year, 'n_events': len(magnitudes),
                     'n_above_mc': int(np.count_nonzero(magnitudes >= result['mc'])),
                     **result})
    return pd.DataFrame(rows)
