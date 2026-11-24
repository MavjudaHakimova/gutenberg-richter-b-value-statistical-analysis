#!/usr/bin/env python3
"""supF-тесты для агрегированных рядов b-positive (NEIC и ISC), как в статье.

Схема бутстрепа та же, что для ряда RFM: случайные блоки со средней длиной
3 года, 19999 повторов. Для NEIC медиана анализируется невзвешенной
регрессией, взвешенное среднее — с весами 1/se²; для ISC оба ряда —
невзвешенной регрессией. Результат: results/aggregated_supf.csv.

Запуск из корня репозитория:
  python scripts/aggregated_supf.py
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.statistics import supf_wild_block  # noqa: E402

RUNS = (
    ('NEIC', 'bpositive_neic_итоги.csv', ('35', '40', '45'), {'median': False, 'weighted': True}),
    ('ISC', 'bpositive_isc_итоги.csv', ('all', '40'), {'median': False, 'weighted': False}),
)


def main() -> None:
    rows = []
    for catalog, name, suffixes, weighting in RUNS:
        table = pd.read_csv(ROOT / 'data' / name)
        for suffix in suffixes:
            for series, weighted in weighting.items():
                column = f'{series}_{suffix}'
                data = table[['year', column, f'se_{column}']].dropna(subset=[column])
                se = data[f'se_{column}'].to_numpy() if weighted else None
                r = supf_wild_block(data.year, data[column], se, weighted=weighted)
                rows.append({'catalog': catalog, 'threshold': suffix, 'series': series,
                             'weighted_regression': weighted, 'break_year': r['break_year'],
                             'supF': round(r['max_f'], 3), 'p': round(r['p'], 4),
                             'critical_95': round(r['critical'], 3), 'n_years': len(data)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'results/aggregated_supf.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
