#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEIC b-positive: воспроизводимые региональные supF (3 порога x 20 регионов).

Запуск из корня репозитория:
  python scripts/neic_regional_supF_20x3.py \
      --input results/neic_regions/all_regions_b_values.csv \
      --output results/neic_regions/regional_supf_fixed_unweighted.csv \
      --protocol original

Для сопоставления с основной моделью supF из src/statistics.py:
  python scripts/neic_regional_supF_20x3.py \
      --input results/neic_regions/all_regions_b_values.csv \
      --output results/neic_regions/regional_supf_weighted_random.csv \
      --protocol repo-default

Выходы: CSV на 60 строк и .meta.json с методикой и SHA256 входного файла.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

REGIONS = (19, 12, 8, 24, 15, 23, 1, 14, 18, 22,
           16, 13, 46, 30, 5, 48, 33, 32, 10, 29)
THRESHOLDS = (
    ('3.5', 'b_positive', 'b_se', 1),
    ('4.0', 'b_positive_m40', 'b_se_m40', 2),
    ('4.5', 'b_positive_m45', 'b_se_m45', 3),
)
NAMES = {
    19: 'Япония—Курилы—Камчатка', 12: 'Кермадек—Тонга—Самоа',
    8: 'Андийская Южная Америка', 24: 'Зондская дуга',
    15: 'Бисмарк и Соломоновы острова', 23: 'Борнео—Сулавеси',
    1: 'Аляска—Алеутская дуга', 14: 'Острова Вануату',
    18: 'От Гуама до Японии', 22: 'Филиппинские острова',
    16: 'Новая Гвинея', 13: 'Острова Фиджи',
    46: 'Андаманские острова—Суматра', 30: 'Ближний Восток—Крым—Балканы',
    5: 'Мексика—Гватемала', 48: 'Гиндукуш и Памир',
    33: 'Индийский океан', 32: 'Атлантический океан',
    10: 'Южные Антильские острова', 29: 'Западная Азия',
}


def holm_adjust(ps: list[float]) -> np.ndarray:
    """Step-down Holm: монотонные скорректированные p, один семейство гипотез."""
    p = np.asarray(ps, dtype=float)
    if len(p) == 0 or not np.all(np.isfinite(p)):
        raise ValueError('Для Holm нужны конечные p всех тестов семейства')
    order = np.argsort(p, kind='stable')
    adjusted_sorted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def single_supf(years: np.ndarray, values: np.ndarray, se: np.ndarray, *,
                weighted: bool, scheme: str, seed: int, boot: int,
                block_len: int, trim: float, min_side_years: int) -> dict:
    good = np.isfinite(years) & np.isfinite(values)
    years, y, se = years[good], values[good], se[good]
    if len(years) < 10:
        raise ValueError('Меньше 10 годовых оценок')
    if np.any(np.diff(years) <= 0):
        raise ValueError('Годы должны быть уникальны и отсортированы')
    n = len(y)
    min_side = max(min_side_years, int(np.ceil(trim * n)))
    if n < max(2 * min_side, 10):
        raise ValueError('Недостаточно наблюдений с каждой стороны разрыва')

    if weighted:
        positive = se[np.isfinite(se) & (se > 0)]
        fallback = float(np.median(positive)) if len(positive) else 0.15
        errors = np.where(np.isfinite(se) & (se > 0), se, fallback)
        weights = 1.0 / np.maximum(errors, 0.05)**2
        weights /= weights.mean()
    else:
        weights = np.ones(n)
    sw = np.sqrt(weights)
    x = years - years.mean()
    null_design = np.column_stack([np.ones(n), x]) * sw[:, None]
    q0, _ = np.linalg.qr(null_design, mode='reduced')
    yw = y * sw
    yhat0 = q0 @ (q0.T @ yw)
    residual = yw - yhat0
    breaks = years[min_side-1:n-min_side]
    q_all = []
    for year in breaks:
        after = (years > year).astype(float)
        full = np.column_stack([np.ones(n), x, after, x*after]) * sw[:, None]
        q, _ = np.linalg.qr(full, mode='reduced')
        q_all.append(q)
    qs = np.stack(q_all, axis=0)

    def fvalues(m):
        m = np.atleast_2d(m)
        yy = np.einsum('bn,bn->b', m, m)
        p0 = m @ q0
        rss0 = np.maximum(yy - np.sum(p0*p0, axis=1), 0.0)
        projection = np.einsum('bn,kni->bki', m, qs, optimize=True)
        rss1 = np.maximum(yy[:, None] - np.sum(projection**2, axis=2), 1e-12)
        return np.maximum((rss0[:, None]-rss1)/2.0, 0) / (rss1/(n-4))

    curve = fvalues(yw)[0]
    best = int(np.argmax(curve))
    max_f = float(curve[best])
    rng = np.random.default_rng(seed)
    if scheme == 'fixed-calendar':
        # Именно календарные блоки, а НЕ блоки каждых 3 имеющихся наблюдений.
        blocks = ((years - np.min(years)) // block_len).astype(int)
        signs = rng.choice([-1.0, 1.0], size=(boot, int(blocks.max())+1))
        multipliers = signs[:, blocks]
    elif scheme == 'random':
        multipliers = np.empty((boot, n))
        for k in range(boot):
            starts = rng.random(n) < 1.0/block_len
            starts[0] = True
            ids = np.cumsum(starts)-1
            signs = rng.choice([-1.0, 1.0], size=int(ids[-1])+1)
            multipliers[k] = signs[ids]
    else:
        raise ValueError(scheme)
    pseudodata = yhat0[None, :] + residual[None, :] * multipliers
    boot_max = np.concatenate([
        fvalues(pseudodata[i:i+2000]).max(axis=1)
        for i in range(0, boot, 2000)
    ])
    p = float((1 + int(np.count_nonzero(boot_max >= max_f))) / (1 + boot))
    return {
        'break_year': int(breaks[best]),
        'supF': max_f,
        'p': p,
        'n_years': n,
        'min_side': min_side,
    }


def make_table(source: Path, *, protocol: str, n_boot: int,
               block_years: int, seed0: int, seed_policy: str) -> list[dict]:
    if protocol == 'original':
        weighted, scheme = False, 'fixed-calendar'
    elif protocol == 'repo-default':
        weighted, scheme = True, 'random'
    elif protocol == 'weighted-fixed':
        weighted, scheme = True, 'fixed-calendar'
    elif protocol == 'unweighted-random':
        weighted, scheme = False, 'random'
    else:
        raise ValueError(protocol)
    with source.open(newline='', encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
    needed = {'region', 'year'} | {x for _, b, se, _ in THRESHOLDS for x in (b, se)}
    if not data or not needed.issubset(data[0]):
        raise ValueError(f'Нет столбцов: {sorted(needed - set(data[0]) if data else needed)}')
    records = []
    for threshold, column, se_col, tid in THRESHOLDS:
        for region in REGIONS:
            a = sorted((x for x in data if int(x['region']) == region),
                       key=lambda row: int(row['year']))
            years = np.array([float(x['year']) for x in a])
            b = np.array([float(x[column]) if x[column].strip() else np.nan for x in a])
            se = np.array([float(x[se_col]) if x[se_col].strip() else np.nan for x in a])
            seed = seed0 if seed_policy == 'same' else np.random.SeedSequence([seed0, region, tid])
            r = single_supf(years, b, se, weighted=weighted, scheme=scheme,
                            seed=seed, boot=n_boot, block_len=block_years,
                            trim=.15, min_side_years=5)
            records.append(dict(
                m_min=threshold, region=region, region_name=NAMES[region],
                **r, weighted=weighted, bootstrap_scheme=scheme,
                bootstrap_reps=n_boot, block_years=block_years,
                seed_policy=seed_policy, trimming=.15,
            ))
    all_adj = holm_adjust([r['p'] for r in records])
    for i, r in enumerate(records):
        r['p_holm_60'] = float(all_adj[i])
        r['significant_holm_60'] = bool(all_adj[i] <= .05)
    for threshold, _, _, _ in THRESHOLDS:
        idx = [i for i, r in enumerate(records) if r['m_min'] == threshold]
        adjusted = holm_adjust([records[i]['p'] for i in idx])
        for i, val in zip(idx, adjusted):
            records[i]['p_holm_20'] = float(val)
            records[i]['significant_holm_20'] = bool(val <= .05)
    return records


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--protocol', choices=['original','repo-default','weighted-fixed','unweighted-random'],
                     default='original')
    cli.add_argument('--n-boot', type=int, default=9999)
    cli.add_argument('--block-years', type=int, default=3)
    cli.add_argument('--seed', type=int, default=20261025)
    cli.add_argument('--seed-policy', choices=['same','per-test'], default=None)
    arg = cli.parse_args()
    if arg.n_boot < 99 or arg.block_years < 1:
        cli.error('Нужно >=99 бутстреп-реплик и длина блока >=1')
    seed_policy = arg.seed_policy or ('per-test' if arg.protocol == 'original' else 'same')
    rows = make_table(arg.input, protocol=arg.protocol, n_boot=arg.n_boot,
                      block_years=arg.block_years, seed0=arg.seed,
                      seed_policy=seed_policy)
    arg.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ['m_min','region','region_name','n_years','break_year','supF','p','p_holm_20',
            'p_holm_60','significant_holm_20','significant_holm_60',
            'weighted','bootstrap_scheme','bootstrap_reps','block_years',
            'seed_policy','trimming','min_side']
    with arg.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda d: (float(d['m_min']), int(d['region']))))
    sha = hashlib.sha256(arg.input.read_bytes()).hexdigest()
    meta = dict(source_path=str(arg.input), source_sha256=sha,
                protocol=arg.protocol, weighted=(arg.protocol in ['repo-default','weighted-fixed']),
                n_boot=arg.n_boot, block_years=arg.block_years,
                bootstrap_scheme=('random' if 'random' in arg.protocol or arg.protocol == 'repo-default' else 'fixed-calendar'),
                seed=arg.seed, seed_policy=seed_policy,
                trimming=0.15, min_side_years=5, alpha=0.05,
                holm_groups='separately 20 per threshold and jointly 60',
                tests=60, numpy_version=np.__version__, python_version=platform.python_version())
    arg.output.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Таблица:', arg.output)
    print('SHA256 входного CSV:', sha)
    for threshold, _, _, _ in THRESHOLDS:
        r = [x for x in rows if x['m_min'] == threshold]
        print(f'M>={threshold}: nominal={sum(x["p"]<.05 for x in r)}, '
              f'Holm20={sum(x["significant_holm_20"] for x in r)}, '
              f'Holm60={sum(x["significant_holm_60"] for x in r)}')

if __name__ == '__main__':
    main()
