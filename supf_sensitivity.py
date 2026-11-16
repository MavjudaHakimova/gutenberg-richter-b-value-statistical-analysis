#!/usr/bin/env python3
"""Чувствительность бутстреп-оценки p для supF-теста (ряд RFM).

Перебираются:
  * схема блоков: 'random' (основная, случайное разбиение на блоки
    с геометрическими длинами) и 'fixed' (фиксированные блоки со всеми
    возможными сдвигами границ 0..L-1 лет);
  * средняя или фиксированная длина блока L = 2..6 лет;
  * взвешенная (веса 1/max(se, 0.05)^2) и невзвешенная схема.

Требуется results/rfm_годовые_оценки.csv (создаётся командой python run.py).
Результат: results/supf_чувствительность.csv и results/supf_чувствительность.md.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import RESULTS
from src.statistics import supf_wild_block

BLOCKS = (2, 3, 4, 5, 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-boot', type=int, default=19999,
                        help='Число бутстреп-повторов (по умолчанию 19999)')
    args = parser.parse_args()

    path = RESULTS / 'rfm_годовые_оценки.csv'
    if not path.is_file():
        raise SystemExit(f'Нет {path}. Сначала выполните: python run.py')
    rfm = pd.read_csv(path)
    se = (rfm.ci_high - rfm.ci_low) / (2 * 1.96)

    rows = []
    for weighted in (True, False):
        for block in BLOCKS:
            r = supf_wild_block(rfm.year, rfm.b, se, n_boot=args.n_boot,
                                block_years=block, weighted=weighted,
                                block_scheme='random')
            rows.append({'scheme': 'random', 'weighted': weighted, 'block_years': block,
                         'offset': '', 'break_year': r['break_year'],
                         'sup_f': r['max_f'], 'p': r['p'], 'critical': r['critical']})
            for offset in range(block):
                r = supf_wild_block(rfm.year, rfm.b, se, n_boot=args.n_boot,
                                    block_years=block, weighted=weighted,
                                    block_scheme='fixed', offset=offset)
                rows.append({'scheme': 'fixed', 'weighted': weighted, 'block_years': block,
                             'offset': offset, 'break_year': r['break_year'],
                             'sup_f': r['max_f'], 'p': r['p'], 'critical': r['critical']})
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / 'supf_чувствительность.csv', index=False, float_format='%.6g')

    lines = [
        '# Чувствительность supF-теста (ряд RFM)',
        '',
        f'Число бутстреп-повторов: {args.n_boot}. Год максимума F во всех вариантах: '
        f'{", ".join(map(str, sorted(table.break_year.unique())))}.',
        '',
        '| L, лет | random, с весами | random, без весов | fixed, с весами (мин–макс по сдвигам) | fixed, без весов (мин–макс по сдвигам) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for block in BLOCKS:
        cells = []
        for scheme in ('random', 'fixed'):
            for weighted in (True, False):
                sub = table[(table.scheme == scheme) & (table.weighted == weighted)
                            & (table.block_years == block)]
                if scheme == 'random':
                    cells.append(f'{sub.p.iloc[0]:.3f}')
                else:
                    cells.append(f'{sub.p.min():.3f}–{sub.p.max():.3f}')
        lines.append(f'| {block} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        'random — основная схема статьи: разбиение на блоки заново в каждой псевдовыборке,',
        'длины блоков геометрические со средним L. fixed — фиксированные блоки длины L;',
        'для неё показан разброс p по всем сдвигам границ блоков (0..L-1 лет).',
    ]
    (RESULTS / 'supf_чувствительность.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
