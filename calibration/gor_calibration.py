"""
Калибровка GOR-коэффициентов для приведения магнитуд к Mw.

Этот скрипт использовался для получения коэффициентов, вшитых в src/gor.py
(константа CONVERSIONS). Требует доступа к исходной PostgreSQL-базе
earthquakes_gr_low; без неё неприменим, но служит документацией
происхождения коэффициентов.

Перед запуском задайте переменную окружения EARTHQUAKE_DB_URL, например:
    export EARTHQUAKE_DB_URL='postgresql://user:password@host:port/dbname'
"""
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


def deming_regression(x: np.ndarray, y: np.ndarray, eta: float = 0.24) -> Dict:
    """GOR регрессия Castellaro (формула 9)"""
    n = len(x)

    sxx = np.var(x, ddof=0)
    syy = np.var(y, ddof=0)
    sxy = np.cov(x, y, ddof=0)[0, 1]

    denom = syy - eta * sxx
    beta = (denom + np.sqrt(denom**2 + 4 * eta * sxy**2)) / (2 * sxy)
    alpha = np.mean(y) - beta * np.mean(x)

    return {
        'alpha': alpha,
        'beta': beta,
        'n': n,
        'eta': eta,
        'equation': f"Mw = {alpha:.3f} + {beta:.3f} * X"
    }


if __name__ == "__main__":
    db_url = os.environ.get('EARTHQUAKE_DB_URL')
    if not db_url:
        raise SystemExit(
            'Не задана переменная окружения EARTHQUAKE_DB_URL. '
            'Пример: export EARTHQUAKE_DB_URL="postgresql://user:password@host:port/dbname"'
        )
    engine = create_engine(db_url)

    plot_folder = "calibration_plots_2"
    os.makedirs(plot_folder, exist_ok=True)

    # 6 РАБОТАЮЩИХ КАЛИБРОВОК
    calibrations = {
        'mb_isc': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as mb_isc, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'mb' AND m1.author_nm = 'ISC'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.24,
            'label': 'mb_ISC'
        },
        'mb_idc': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as mb_idc, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'mb' AND m1.author_nm = 'IDC'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.24,
            'label': 'mb_IDC'
        },
        'mb_bji': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as mb_bji, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'mB' AND m1.author_nm = 'BJI'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.24,
            'label': 'mB_BJI'
        },
        'ml_idc': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as ml_idc, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'ML' AND m1.author_nm = 'IDC'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.67,
            'label': 'ML_IDC'
        },
        'ms_idc': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as ms_idc, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'MS' AND m1.author_nm = 'IDC'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.41,
            'label': 'MS_IDC'
        },
        'ms_isc': {
            'query': """
                SELECT DISTINCT e.event_id, e.event_dttm, m1.mag_value as ms_isc, m2.mag_value as mw_nied
                FROM earthquakes_gr_low.events e
                JOIN earthquakes_gr_low.magnitudes m1 ON e.event_id = m1.event_id
                JOIN earthquakes_gr_low.magnitudes m2 ON e.event_id = m2.event_id
                WHERE m1.mag_type = 'MS' AND m1.author_nm = 'ISC'
                  AND m2.mag_type = 'Mw' AND m2.author_nm = 'NIED'
                """,
            'eta': 0.41,
            'label': 'MS_ISC'
        }
    }

    all_results = {}

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    print("\n" + "=" * 90)
    print("6 GOR КАЛИБРОВОК + OLS (обычная линейная регрессия)")
    print("=" * 90)

    for i, (name, config) in enumerate(calibrations.items()):
        print(f"\n[{i + 1}/6] {config['label']}")

        # Загружаем данные
        df = pd.read_sql_query(config['query'], engine)

        # Убираем пропуски сразу по обеим колонкам
        x_col = df.columns[2]
        df = df.dropna(subset=[x_col, 'mw_nied']).copy()

        x = df[x_col].values
        y = df['mw_nied'].values

        n_events = len(df)

        print(f"   Загружено событий: {n_events:,}")

        if n_events > 10:
            # GOR
            gor = deming_regression(x, y, config['eta'])

            # OLS
            ols = stats.linregress(x, y)
            ols_alpha = ols.intercept
            ols_beta = ols.slope

            all_results[name] = {
                **gor,
                'ols_alpha': ols_alpha,
                'ols_beta': ols_beta,
                'n_events': n_events,
                'x_col': x_col
            }

            # График
            ax = axes[i]
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line_gor = gor['alpha'] + gor['beta'] * x_line
            y_line_ols = ols_alpha + ols_beta * x_line

            ax.scatter(x, y, alpha=0.4, s=18, color='black', label='Данные')
            ax.plot(x_line, y_line_gor, 'r-', linewidth=3, label='GOR')
            ax.plot(x_line, y_line_ols, 'b--', linewidth=3, label='OLS')

            title_text = (
                f"{config['label']}\n"
                f"n={n_events:,}\n"
                f"GOR: Mw = {gor['alpha']:.3f} + {gor['beta']:.3f}×{x_col}"
            )
            ax.set_title(title_text, fontsize=11)

            eq_text = (
                f"GOR: Mw = {gor['alpha']:.3f} + {gor['beta']:.3f}×{x_col}\n"
                f"OLS: Mw = {ols_alpha:.3f} + {ols_beta:.3f}×{x_col}\n"
                f"n = {n_events:,}"
            )
            ax.text(
                0.05, 0.95, eq_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
            )

            ax.grid(True, alpha=0.3)
            ax.set_xlabel(x_col)
            ax.set_ylabel('Mw_NIED')
            ax.legend()

            print(f"   GOR: Mw = {gor['alpha']:.3f} + {gor['beta']:.3f} × {x_col}")
            print(f"   OLS: Mw = {ols_alpha:.3f} + {ols_beta:.3f} × {x_col}")
        else:
            print(f"   Недостаточно данных: {n_events} событий")
            axes[i].text(
                0.5, 0.5,
                f'Недостаточно данных\nn={n_events}',
                ha='center', va='center',
                transform=axes[i].transAxes
            )

    # Если графиков меньше 6, пустые оси скрываем
    for j in range(len(calibrations), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'all_6_gor_ols_calibrations.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 100)
    print("ИТОГОВЫЕ КОЭФФИЦИЕНТЫ")
    print("=" * 100)
    print(f"{'Источник':<12} {'α_GOR':<10} {'β_GOR':<10} {'α_OLS':<10} {'β_OLS':<10} {'n':<12} {'η':<6}")
    print("-" * 100)

    for name in sorted(all_results.keys(), key=lambda k: all_results[k]['n_events'], reverse=True):
        result = all_results[name]
        label = calibrations[name]['label']
        print(
            f"{label:<12} "
            f"{result['alpha']:<10.3f} "
            f"{result['beta']:<10.3f} "
            f"{result['ols_alpha']:<10.3f} "
            f"{result['ols_beta']:<10.3f} "
            f"{result['n_events']:<12,} "
            f"{result['eta']:<6.2f}"
        )

    # Сохранение коэффициентов
    results_df = pd.DataFrame([
        {
            'source': k,
            'label': calibrations[k]['label'],
            'x_col': v['x_col'],
            'alpha_gor': v['alpha'],
            'beta_gor': v['beta'],
            'alpha_ols': v['ols_alpha'],
            'beta_ols': v['ols_beta'],
            'n_events': v['n_events'],
            'eta': v['eta']
        }
        for k, v in all_results.items()
    ])
    results_df.to_csv(os.path.join(plot_folder, 'all_6_gor_ols_coefficients.csv'), index=False)

    # Сохраняем все пары с датами
    all_pairs = pd.concat([
        pd.read_sql_query(config['query'], engine).assign(source=config['label'])
        for config in calibrations.values()
    ], ignore_index=True)
    all_pairs.to_csv(os.path.join(plot_folder, 'all_6_calibration_pairs_with_dates.csv'), index=False)

    print(f"\nСохранено:")
    print(f"   all_6_gor_ols_coefficients.csv ({len(all_results)} источников)")
    print(f"   all_6_gor_ols_calibrations.png")
    print(f"   all_6_calibration_pairs_with_dates.csv ({len(all_pairs):,} строк)")

    total_events = sum(v['n_events'] for v in all_results.values())
    print(f"\nИТОГО: {total_events:,} реально загруженных калибровочных пар")

    # SQL-шаблон
    print("\n" + "=" * 100)
    print("SQL ШАБЛОН MASTER Mw каталога (подставь коэффициенты):")
    print("=" * 100)
    for name, result in all_results.items():
        label = calibrations[name]['label']
        print(f"   -- GOR {label}: Mw = {result['alpha']:.3f} + {result['beta']:.3f} × {label.lower()}")
        print(f"   -- OLS {label}: Mw = {result['ols_alpha']:.3f} + {result['ols_beta']:.3f} × {label.lower()}")