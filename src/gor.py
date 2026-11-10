
"""Гомогенизация магнитуд землетрясений к шкале Mw.

Метод общей ортогональной регрессии (GOR).
Раздел 2.3 статьи.

Коэффициенты преобразования и приоритеты магнитуд
соответствуют SQL-запросу создания master_mw_catalog.

В основном анализе используется уже подготовленный
гомогенизированный каталог ISC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Conversion:
    """Коэффициенты преобразования магнитуд."""

    source_type: str
    agency: str
    intercept: float
    slope: float


# Коэффициенты преобразования Mw = alpha + beta * Mx.
# Порядок определяет приоритет выбора магнитуды.

CONVERSIONS = (
    Conversion("mb", "ISC", -0.130, 1.036),
    Conversion("mb", "IDC", -1.277, 1.398),
    Conversion("mB", "BJI", -2.412, 1.435),
    Conversion("ML", "IDC", -4.158, 2.163),
    Conversion("MS", "IDC", 0.998, 0.911),
    Conversion("MS", "ISC", 1.338, 0.792),
)


def convert_to_mw(
    magnitude: float | np.ndarray,
    mag_type: str,
    agency: str,
):
    """Преобразовать магнитуду в шкалу Mw."""

    if mag_type == "Mw" and agency == "NIED":
        return magnitude

    for conv in CONVERSIONS:
        if (
            conv.source_type == mag_type
            and conv.agency == agency
        ):
            return (
                conv.intercept
                + conv.slope * np.asarray(magnitude)
            )

    raise ValueError(
        f"Нет коэффициентов для {mag_type} / {agency}"
    )


def prepare_homogenized_catalog(
    events: pd.DataFrame,
    magnitudes: pd.DataFrame,
) -> pd.DataFrame:
    """Сформировать каталог с магнитудами Mw.

    events:
        event_id, event_dttm

    magnitudes:
        event_id, mag_type, author_nm, mag_value

    Приоритет:
        1. Прямая Mw от NIED.
        2. mb ISC.
        3. mb IDC.
        4. mB BJI.
        5. ML IDC.
        6. MS IDC.
        7. MS ISC.

    При нескольких оценках одной шкалы и агентства
    сохраняется первая запись входной таблицы.
    """

    required_events = {"event_id", "event_dttm"}
    required_magnitudes = {
        "event_id",
        "mag_type",
        "author_nm",
        "mag_value",
    }

    if missing := required_events - set(events.columns):
        raise ValueError(
            f"Отсутствуют поля событий: {sorted(missing)}"
        )

    if missing := required_magnitudes - set(magnitudes.columns):
        raise ValueError(
            f"Отсутствуют поля магнитуд: {sorted(missing)}"
        )

    mags = magnitudes.copy()

    mags["mag_value"] = pd.to_numeric(
        mags["mag_value"],
        errors="coerce",
    )

    mags = mags.dropna(
        subset=["event_id", "mag_value"]
    )

    indexed = mags.drop_duplicates(
        ["event_id", "mag_type", "author_nm"]
    )

    result = (
        events.drop_duplicates("event_id")
        .set_index("event_id")
        .copy()
    )

    result["mw_homogenized"] = np.nan
    result["mw_source"] = pd.NA

    candidates = [
        ("Mw", "NIED", 0.0, 1.0, "Mw_NIED")
    ]

    candidates += [
        (
            c.source_type,
            c.agency,
            c.intercept,
            c.slope,
            f"GOR_{c.source_type}_{c.agency}",
        )
        for c in CONVERSIONS
    ]

    for scale, agency, alpha, beta, label in candidates:

        subset = indexed[
            (indexed["mag_type"] == scale)
            & (indexed["author_nm"] == agency)
        ]

        if subset.empty:
            continue

        values = subset.set_index(
            "event_id"
        )["mag_value"]

        eligible = result.index.intersection(
            values.index
        )

        eligible = eligible[
            result.loc[
                eligible, "mw_homogenized"
            ].isna().to_numpy()
        ]

        if len(eligible):
            result.loc[
                eligible, "mw_homogenized"
            ] = (
                alpha
                + beta * values.loc[eligible].to_numpy()
            )

            result.loc[
                eligible, "mw_source"
            ] = label

    result = result.reset_index()

    result = result.loc[
        result["mw_homogenized"].between(0, 10)
    ].copy()

    return result


def fit_gor_deming(
    x: np.ndarray,
    y: np.ndarray,
    variance_ratio: float,
) -> tuple[float, float]:
    """Оценить коэффициенты общей ортогональной регрессии.

    x — исходные магнитуды.
    y — соответствующие магнитуды Mw.

    variance_ratio:
        Var(ошибка_y) / Var(ошибка_x)

    Возвращает:
        alpha, beta

    Для повторной калибровки необходимы парные
    измерения магнитуд и отношение дисперсий
    их ошибок.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError(
            "Необходимо не менее трёх пар магнитуд"
        )

    if variance_ratio <= 0:
        raise ValueError(
            "Отношение дисперсий должно быть положительным"
        )

    sxx = np.var(x, ddof=1)
    syy = np.var(y, ddof=1)

    sxy = np.cov(x, y, ddof=1)[0, 1]

    if abs(sxy) < 1e-12:
        raise ValueError(
            "Ковариация магнитуд равна нулю"
        )

    lam = variance_ratio

    slope = (
        syy
        - lam * sxx
        + np.sqrt(
            (syy - lam * sxx) ** 2
            + 4 * lam * sxy**2
        )
    ) / (2 * sxy)

    intercept = float(
        np.mean(y) - slope * np.mean(x)
    )

    return intercept, float(slope)
