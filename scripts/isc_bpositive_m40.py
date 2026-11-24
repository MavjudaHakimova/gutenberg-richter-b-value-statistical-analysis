#!/usr/bin/env python3
"""Пересчёт агрегированного ряда b-positive для ISC (Mw >= 4.0), 1964–2021.

Использует те же правила, что и остальные варианты: все зоны Flinn–Engdahl,
не менее 20 событий в паре (регион, год), порог положительных разностей 0.2,
не менее 15 отобранных разностей. Обновляет столбцы median_40, weighted_40,
n_regions_40, n_pos_used_40 в data/bpositive_isc_итоги.csv.

Запуск из корня репозитория:
  python scripts/isc_bpositive_m40.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bpositive import aggregate, b_positive  # noqa: E402

FIRST, LAST, REF, THRESHOLD = 1964, 2021, 4.0, 0.2


def main() -> None:
    catalog = pd.read_csv(ROOT / 'data/isc_mw_1960_2021.csv.gz')
    work = catalog[catalog.year.between(FIRST, LAST) & (catalog.magnitude >= REF)]
    work = work.sort_values(['fe_large', 'year', 'time'], kind='mergesort')
    rows = []
    for (region, year), sample in work.groupby(['fe_large', 'year']):
        if len(sample) < 20:
            continue
        b, n_pos = b_positive(sample.magnitude.to_numpy(), difference_threshold=THRESHOLD)
        if np.isfinite(b):
            rows.append({'region': region, 'year': year, 'b_positive': b,
                         'n_pos': n_pos, 'n_events': len(sample)})
    regional = pd.DataFrame(rows)
    agg = aggregate(regional).set_index('year')
    n_pos = regional.groupby('year').n_pos.sum()

    path = ROOT / 'data/bpositive_isc_итоги.csv'
    table = pd.read_csv(path).set_index('year')
    table['median_40'] = agg['median']
    table['weighted_40'] = agg['weighted']
    table['n_regions_40'] = agg['n_regions'].astype('Int64')
    table['n_pos_used_40'] = n_pos.astype('Int64')
    # Стандартные ошибки этого варианта прежним расчётом не воспроизводятся;
    # в коде они не используются, поэтому оставлены пустыми.
    table['se_median_40'] = np.nan
    table['se_weighted_40'] = np.nan
    table.reset_index().to_csv(path, index=False, float_format='%.10g')
    print(f'Обновлено {path.name}: {len(agg)} лет, {FIRST}–{LAST}')


if __name__ == '__main__':
    main()
