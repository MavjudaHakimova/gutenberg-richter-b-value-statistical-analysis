"""Классический метод максимального правдоподобия Аки (MLE)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from .config import START_YEAR, END_YEAR


def estimate_yearly(catalog: pd.DataFrame, threshold: float = 5.0,
                    min_events: int = 30) -> pd.DataFrame:
    """Оценить b и a по индивидуальным магнитудам M >= M0 для каждого года.

    Порог M0 — правило отбора событий, не оценка фактической полноты Mc.
    Предполагается непрерывная экспоненциальная модель без верхнего усечения.
    """
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        values = catalog.loc[(catalog['year'] == year) &
                             (catalog['magnitude'] >= threshold), 'magnitude'].to_numpy()
        n = len(values)
        if n < min_events:
            raise ValueError(f'Для {year} года и M0={threshold} найдено лишь {n} событий')
        excess = float(np.sum(values - threshold))
        if excess <= 0:
            raise ValueError(f'Для {year} года нет положительного превышения над M0')
        b = n / (np.log(10.0) * excess)
        rows.append({
            'year': year, 'm0': threshold, 'n_events': n,
            'b': b, 'a': np.log10(n) + b * threshold,
            'se': b / np.sqrt(n),
            'ci_low': chi2.ppf(0.025, 2*n) / (2*np.log(10.0)*excess),
            'ci_high': chi2.ppf(0.975, 2*n) / (2*np.log(10.0)*excess),
        })
    return pd.DataFrame(rows)


def all_thresholds(catalog: pd.DataFrame) -> pd.DataFrame:
    """Получить ряды для трёх порогов M0 из таблицы 1 статьи."""
    return pd.concat([estimate_yearly(catalog, m0) for m0 in (5.0, 5.5, 6.0)],
                     ignore_index=True)
