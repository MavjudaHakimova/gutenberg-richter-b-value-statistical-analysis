#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
import numpy as np
import pandas as pd

from src.config import (DATA, FIGURES, RESULTS, ISC_CATALOG, START_YEAR,
                        END_YEAR, prepare_style)
from src.rfm import annual_rfm
from src.mle import all_thresholds
from src.statistics import chow_fixed, supf_wild_block


def _load_catalog(path):
    """Прочитать сжатый CSV, отфильтровать годы и привести типы данных."""
    if not path.exists():
        raise FileNotFoundError(f'Отсутствует каталог {path}. Распакуйте ZIP полностью.')
    frame = pd.read_csv(path,compression='gzip')
    if 'year' not in frame.columns:
        frame['year'] = pd.to_datetime(frame['time'],errors='coerce',utc=True).dt.year
    frame['year'] = pd.to_numeric(frame['year'],errors='coerce')
    frame['magnitude'] = pd.to_numeric(frame['magnitude'],errors='coerce')
    return frame.loc[frame['year'].between(START_YEAR,END_YEAR) &
                     (frame['magnitude']>=3.5),['year','magnitude']].dropna().copy()


def _generate_statistics(n_boot: int, rfm, mle):
    """Проверить неизвестную дату RFM и фиксированную дату MLE."""
    rfm_se=(rfm.ci_high-rfm.ci_low)/(2*1.96)
    supf = supf_wild_block(rfm.year,rfm.b,rfm_se,n_boot=n_boot)
    chow={}
    for m0,rows in mle.groupby('m0',sort=True):
        chow[float(m0)] = chow_fixed(rows.year,rows.b)
    return supf,chow


def _write_report(rfm, mle, supf, chow):
    """Записать итоги тестов и пояснить границы интерпретации."""
    lines=[
        '# Расчётные результаты',
        '',
        'Расчёты выполнены из включённого в проект каталога ISC Mw.',
        'В данных уже проведена гомогенизация магнитуд; её коэффициенты не переоцениваются.',
        '',
        '## RFM: supF при неизвестной дате',
        '',
        f'- Год максимума: **{supf["break_year"]}**.',
        f'- F-статистика: **{supf["max_f"]:.4f}**.',
        f'- Бутстреп p: **{supf["p"]:.4f}** ({supf["n_boot"]} реплик).',
        f'- Критическое значение (95%): **{supf["critical"]:.4f}**.',
        f'- Разность BIC (нулевая минус альтернативная модель): **{supf["delta_bic"]:+.4f}**.',
        '',
        '## MLE: критерий Чоу для фиксированного 1994 года',
        '',
        '| Порог M0 | F | Номинальное p | Скачок b | Наклон до 1994 | Наклон после 1994 |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for m0,c in sorted(chow.items()):
        lines.append(f'| {m0:.1f} | {c["F"]:.4f} | {c["p"]:.6g} | '
                     f'{c["jump"]:+.4f} | {c["slope_before"]:+.5f} | '
                     f'{c["slope_after"]:+.5f} |')
    lines += [
        '',
        'Важно: M0 не равен обязательно магнитуде полноты Mc.',
        'Критерий Чоу для MLE даёт номинальную значимость, поскольку дата 1994 была выбрана по связанным RFM-данным.',
        'В статье региональные ряды b-positive не подтверждают сопоставимый устойчивый разрыв после поправки Холма.',
        'Это не является доказательством физической перестройки глобального сейсмического режима.',
        '',
        'Доверительные интервалы RFM использованы из сохранённых результатов исходного эксперимента,',
        'а не получены повторным полным бутстрепом при этом запуске.',
    ]
    (RESULTS/'ИТОГИ_АНАЛИЗА.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (RESULTS/'статистические_тесты.json').open('w',encoding='utf-8') as fh:
        serial={
            'rfm_supf':{key:float(supf[key]) if isinstance(supf[key],(int,float,np.floating)) else supf[key]
                        for key in ('max_f','p','critical','delta_bic')},
            'rfm_break_year':int(supf['break_year']),
            'n_boot':int(supf['n_boot']),
            'mle_chow':{str(key):{k:float(v) for k,v in val.items()
                                   if k in ('F','p','jump','slope_before','slope_after')}
                        for key,val in chow.items()},
        }
        json.dump(serial,fh,ensure_ascii=False,indent=2)


def main():
    parser=argparse.ArgumentParser(description='Воссоздать все 11 рисунков статьи')
    parser.add_argument('--quick',action='store_true',help='Использовать ранее посчитанные RFM/MLE-таблицы')
    parser.add_argument('--n-boot',type=int,default=1999,
                        help='Число бутстреп-повторов для supF (по умолчанию 1999)')
    args=parser.parse_args()
    if args.n_boot < 99:
        parser.error('Для supF необходимо не менее 99 бутстреп-повторов')
    prepare_style()
    FIGURES.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('1/4 · Чтение данных и расчёт годовых оценок...')
    isc_catalog=_load_catalog(ISC_CATALOG)
    rfm_path=RESULTS/'rfm_годовые_оценки.csv'
    mle_path=RESULTS/'mle_годовые_оценки.csv'
    if args.quick and rfm_path.is_file() and mle_path.is_file():
        rfm=pd.read_csv(rfm_path)
        mle=pd.read_csv(mle_path)
    else:
        rfm=annual_rfm(isc_catalog)
        mle=all_thresholds(isc_catalog)
        # Сохраняем авторские интервалы, поскольку отдельный повтор 1000
        # бутстреп-реплик по 57 годам был бы излишне долгим.
        published=pd.read_csv(DATA/'rfm_доверительные_интервалы.csv')
        diff=rfm.set_index('year')['b'].sub(published.set_index('year')['b']).abs().max()
        if diff > 0.005:
            warnings.warn(f'Ряд RFM отличается от архивного: max |Δb| = {diff:.4f}')
        rfm=rfm.merge(published[['year','ci_low','ci_high']],on='year',validate='1:1')
        rfm.to_csv(rfm_path,index=False,float_format='%.10g')
        mle.to_csv(mle_path,index=False,float_format='%.10g')
    neic=pd.read_csv(DATA/'bpositive_neic_итоги.csv')
    isc=pd.read_csv(DATA/'bpositive_isc_итоги.csv')
    print('2/4 · Проверка структурных изменений (supF и Чоу)...')
    supf,chows=_generate_statistics(args.n_boot,rfm,mle)
    _write_report(rfm,mle,supf,chows)
    print(f'       RFM: supF={supf["max_f"]:.3f}, год {supf["break_year"]}, p={supf["p"]:.4f}')
    for m,c in sorted(chows.items()):
        print(f'       MLE, M0={m:.1f}: F={c["F"]:.4f}, p={c["p"]:.6g}')
    print('3/4 · Построение рисунков 1–11 с русскими подписями...')
    from src.plots import plot_all
    plot_all(isc_catalog,rfm,mle,neic,isc,supf,chows[5.0])
    print('4/4 · Готово. Графики: figures/; годовые оценки и тесты: results/')


if __name__=='__main__':
    main()
