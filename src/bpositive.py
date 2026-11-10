"""Оценивание b-positive по последовательным положительным разностям магнитуд."""
from __future__ import annotations

import numpy as np
import pandas as pd


def b_positive(magnitudes: np.ndarray, *, step: float = 0.1,
               difference_threshold: float = 0.2, min_differences: int = 15):
    """Дискретная оценка b+ из положительных разностей по ван дер Элсту.

    Магнитуды перед разностным расчётом округляются до шага 0.1.
    События должны быть строго упорядочены по времени. Результат не гарантирует независимости от полноты.
    """
    raw = np.asarray(magnitudes, dtype=float)
    bins = np.floor(raw / step + 0.5 + 1e-9).astype(np.int64)
    diff = np.diff(bins)
    limit = int(round(difference_threshold / step))
    chosen = diff[diff >= limit]
    n_positive = int(len(chosen))
    if n_positive < min_differences:
        return np.nan, n_positive
    excess = float(np.mean(chosen - limit))
    if excess <= 0:
        return np.nan, n_positive
    b = np.log1p(1 / excess) / (step * np.log(10))
    return float(b), n_positive


def regional_yearly(catalog: pd.DataFrame, *, ref: float | None = None,
                    difference_threshold: float = 0.2) -> pd.DataFrame:
    """Вычислить регионально-годовые оценки с весом по числу разностей.

    Для полноценных региональных тестов используются лишь годы с >= 20
    землетрясениями и >= 15 положительными разностями.
    """
    work = catalog[['time','fe_large','magnitude']].copy()
    if 'year' in catalog.columns:
        work['year'] = pd.to_numeric(catalog['year'], errors='coerce')
    else:
        work['year'] = pd.to_datetime(work['time'], errors='coerce').dt.year
    work = work.loc[work['year'].between(1965, 2021)].dropna(subset=['time','fe_large','magnitude'])
    if ref is not None:
        # Порог применяется к исходной магнитуде до округления.
        work = work.loc[work['magnitude'] >= ref]
    work = work.sort_values(['fe_large','year','time'], kind='mergesort')
    rows = []
    for (region, year), sample in work.groupby(['fe_large', 'year'], sort=True):
        if len(sample) < 20:
            continue
        b, count = b_positive(sample['magnitude'].to_numpy(),
                              difference_threshold=difference_threshold)
        if np.isfinite(b):
            rows.append({'region': int(region), 'year': int(year),
                         'b_positive': b, 'n_pos': count, 'n_events': len(sample)})
    return pd.DataFrame(rows)


def aggregate(regional: pd.DataFrame) -> pd.DataFrame:
    """Годовая медиана региональных b+ и среднее с весом n_pos."""
    rows = []
    for year, group in regional.groupby('year'):
        rows.append({
            'year': int(year), 'median': float(group['b_positive'].median()),
            'weighted': float(np.average(group['b_positive'], weights=group['n_pos'])),
            'n_regions': int(len(group)),
        })
    return pd.DataFrame(rows).sort_values('year')
