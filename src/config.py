"""Пути, параметры и оформление графиков для всего проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'

ISC_CATALOG = DATA / 'isc_mw_1960_2021.csv.gz'
NEIC_CATALOG = DATA / 'neic_mb_1973_2021.csv.gz'
START_YEAR, END_YEAR = 1965, 2021
BREAK_YEAR = 1994

# Цветовая палитра используется согласованно во всех 11 рисунках.
BLUE = '#1D4ED8'
TEAL = '#0D9488'
RED = '#DC2626'
ORANGE = '#EA580C'
GREEN = '#15803D'
PURPLE = '#7C3AED'
GRAY = '#64748B'


def prepare_style() -> None:
    """Включить читаемое оформление и кириллические подписи."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.23,
        'grid.linestyle': '--',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'semibold',
        'legend.frameon': False,
        'savefig.dpi': 155,
        'savefig.bbox': 'tight',
    })
