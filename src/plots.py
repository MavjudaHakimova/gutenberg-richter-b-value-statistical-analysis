from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from .config import (FIGURES, BLUE, TEAL, RED, ORANGE, GREEN, GRAY, PURPLE,
                     BREAK_YEAR)
from .rfm import cumulative_bins


def _new(title: str, *, height: float = 5.1):
    fig, ax = plt.subplots(figsize=(11.8, height), layout='constrained')
    ax.set_title(title, pad=14, loc='left')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    return fig, ax


def _finish(fig, filename: str, caption: str):
    """Сохранить PNG; описание находится в файле РИСУНКИ.md."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename)
    plt.close(fig)


def fig01_frequency(catalog: pd.DataFrame, rfm: pd.DataFrame):
    """Рис. 1: кумулятивное частотно-магнитудное распределение за 1965 г."""
    mags = catalog.loc[catalog.year==1965,'magnitude'].to_numpy()
    row = rfm.set_index('year').loc[1965]
    m, counts = cumulative_bins(mags)
    fig,ax=_new('Рис. 1 · Частотно-магнитудное распределение, 1965 год',height=5.5)
    ax.semilogy(m,counts,'o',color=RED,markersize=3.2,label='Наблюдаемое число событий')
    line = np.linspace(row.mc,min(max(m),9),160)
    ax.semilogy(line,10**(row.a-row.b*line),'-',color=GREEN,lw=2,
                label='Робастная аппроксимация закона Гутенберга–Рихтера')
    ax.axvspan(3.5,row.mc,color=GREEN,alpha=.10,label='Непредставительная область')
    ax.axvline(row.mc,color=BLUE,ls='--',label=f'Порог полноты Mc = {row.mc:.2f}')
    ax.set(xlabel='Магнитуда M',ylabel='Кумулятивное число событий N(M)')
    ax.set_xlim(3.5,9)
    ax.legend(fontsize=9)
    _finish(fig,'01_частотно_магнитудное_распределение.png',
            'Кумулятивные наблюдения за 1965 год и RFM-аппроксимация выше Mc.')


def fig02_mc(rfm):
    fig,ax=_new('Рис. 2 · Изменение магнитуды полноты каталога')
    ax.plot(rfm.year,rfm.mc,'-o',color=BLUE,lw=1.6,markersize=3.5)
    ax.set(xlabel='Год',ylabel='Магнитуда полноты Mc',xlim=(1965,2021))
    _finish(fig,'02_магнитуда_полноты_mc.png',
            'Годовой порог полноты Mc по RFM; снижение после начала 1990-х может влиять на b.')


def fig03_a(rfm):
    fig,ax=_new('Рис. 3 · Динамика параметра сейсмической активности a')
    ax.plot(rfm.year,rfm.a,'-o',color=BLUE,markersize=3.5,lw=1.6)
    ax.set(xlabel='Год',ylabel='Параметр a',xlim=(1965,2021))
    _finish(fig,'03_параметр_a.png',
            'Изменение рассчитанного за год параметра a в законе Гутенберга–Рихтера.')


def _piecewise(years, test):
    t=np.asarray(years,dtype=float)
    x=t-test['mean_year']
    after=(t>test['break_year']).astype(float)
    c=test['coefs']
    return c[0]+c[1]*x+c[2]*after+c[3]*x*after


def fig04_rfm(rfm, rfm_test):
    x=rfm.year.to_numpy(float)
    fig,ax=_new('Рис. 4 · Изменение параметра b: робастный метод (RFM)',height=5.7)
    ax.fill_between(x,rfm.ci_low,rfm.ci_high,color=GREEN,alpha=.14,
                    label='95% доверительный интервал (бутстреп)')
    ax.plot(x,rfm.b,'-o',color=GREEN,markersize=3.5,lw=1.6,label='Годовые оценки b')
    line=np.linspace(x.min(),x.max(),260)
    fit=_piecewise(line,rfm_test)
    pre=line<=rfm_test['break_year']
    ax.plot(line[pre],fit[pre],color=RED,lw=2.6,label='Тренд до 1994 года')
    ax.plot(line[~pre],fit[~pre],color=BLUE,lw=2.6,label='Тренд после 1994 года')
    ax.axvline(rfm_test['break_year'],color=ORANGE,ls='--',lw=1.6,label='Предполагаемый разрыв: 1994 год')
    ax.set(xlabel='Год',ylabel='Параметр b (RFM)',xlim=(1965,2021))
    ax.legend(fontsize=9,ncol=2)
    _finish(fig,'04_динамика_b_rfm.png',
            'Годовые RFM-оценки, модельные доверительные интервалы и две линейные ветви (1994).')


def fig05_supf(supf):
    fig,ax=_new('Рис. 5 · Поиск неизвестного года структурного изменения')
    ax.plot(supf['breaks'],supf['f_curve'],'-o',color=BLUE,markersize=3.3,lw=1.8,
            label='F-статистика для каждой даты')
    ax.axhline(supf['critical'],color=GRAY,ls='--',lw=1.6,
               label=f'95% критическое значение = {supf["critical"]:.2f}')
    ax.axvline(supf['break_year'],color=RED,ls=':',lw=1.8)
    ax.scatter([supf['break_year']],[supf['max_f']],color=RED,s=65,zorder=8,
               label=f'Максимум F = {supf["max_f"]:.2f} в {supf["break_year"]} г.')
    ax.set(xlabel='Предполагаемый год разрыва',ylabel='F-статистика Чоу')
    ax.text(.02,.96,f'Бутстреп-калибровка: p = {supf["p"]:.4f}',
            transform=ax.transAxes,va='top',fontsize=10,color=GRAY)
    ax.legend(loc='upper right',fontsize=9)
    _finish(fig,'05_тест_supf.png',
            'Кривая F для перебора дат, максимум в 1994 году и критическая граница Wild-Block Bootstrap.')


def _neic_panel(ax,df,mode,thresholds,title):
    colors={'35':BLUE,'40':ORANGE,'45':RED}
    labels={'35':'M ≥ 3,5','40':'M ≥ 4,0','45':'M ≥ 4,5'}
    for suffix in thresholds:
        y=df[f'{mode}_{suffix}']
        ax.plot(df.year,y,'-o',color=colors[suffix],markersize=3.1,lw=1.5,
                label=labels[suffix])
    if '35' in thresholds:
        mask=df[f'{mode}_35'].notna()
        coefs=np.polyfit(df.loc[mask,'year'],df.loc[mask,f'{mode}_35'],1)
        ax.plot(df.year,np.polyval(coefs,df.year),'--',color=GRAY,alpha=.85,
                label='Единый линейный тренд (M ≥ 3,5)')
    ax.set(title=title,ylabel=f'b-positive ({"медиана" if mode=="median" else "взвешенное среднее"})',
           xlim=(1973,2021))
    ax.legend(ncol=2,fontsize=9)


def fig06_neic_median(neic):
    fig,axs=plt.subplots(2,1,figsize=(11.8,8.3),layout='constrained',sharex=True)
    fig.suptitle('Рис. 6 · NEIC (mb): медиана региональных оценок b-positive',fontweight='semibold',fontsize=14)
    _neic_panel(axs[0],neic,'median',('35','40'),'Сравнение порогов M ≥ 3,5 и M ≥ 4,0')
    _neic_panel(axs[1],neic,'median',('45',),'Отдельный анализ для M ≥ 4,5')
    axs[1].set_xlabel('Год');axs[1].xaxis.set_major_locator(MultipleLocator(5))
    _finish(fig,'06_bpositive_neic_медиана.png',
            'Годовые медианы региональных оценок b-positive по каталогу NEIC в шкале mb, три порога.')


def fig07_neic_weighted(neic):
    fig,axs=plt.subplots(2,1,figsize=(11.8,8.3),layout='constrained',sharex=True)
    fig.suptitle('Рис. 7 · NEIC (mb): взвешенное среднее b-positive',fontweight='semibold',fontsize=14)
    _neic_panel(axs[0],neic,'weighted',('35','40'),'Сравнение порогов M ≥ 3,5 и M ≥ 4,0')
    _neic_panel(axs[1],neic,'weighted',('45',),'Отдельный анализ для M ≥ 4,5')
    axs[1].set_xlabel('Год');axs[1].xaxis.set_major_locator(MultipleLocator(5))
    _finish(fig,'07_bpositive_neic_взвешенное.png',
            'Средние региональные b-positive с весом по числу положительных разностей (NEIC mb).')


def fig08_isc_bpositive(isc):
    df=isc.loc[isc.year>=1973].copy()
    fig,axs=plt.subplots(2,1,figsize=(11.8,8.3),layout='constrained',sharex=True)
    fig.suptitle('Рис. 8 · ISC (Mw): сравнение региональных b-positive',fontweight='semibold',fontsize=14)
    for ax,mode,name in zip(axs,('median','weighted'),('Медиана','Взвешенное среднее')):
        ax.plot(df.year,df[f'{mode}_all'],'-o',color=BLUE,lw=1.5,markersize=3,
                label='Все доступные события')
        ax.plot(df.year,df[f'{mode}_40'],'-s',color=ORANGE,lw=1.5,markersize=3,
                label='Mw ≥ 4,0')
        c=np.polyfit(df.year,df[f'{mode}_all'],1)
        ax.plot(df.year,np.polyval(c,df.year),'--',color=GRAY,label='Единый тренд')
        ax.set_ylabel(f'b-positive ({name.lower()})')
        ax.legend(ncol=3,fontsize=9)
    axs[1].set_xlabel('Год');axs[1].xaxis.set_major_locator(MultipleLocator(5))
    _finish(fig,'08_bpositive_isc_mw.png',
            'Медиана и взвешенное среднее для гомогенизированного каталога ISC Mw; сравнение порога 4,0.')


def fig09_mle(mle):
    data=mle.loc[mle.m0==5.0].sort_values('year')
    fig,ax=_new('Рис. 9 · Годовые MLE-оценки b при M0 = 5,0')
    ax.fill_between(data.year,data.ci_low,data.ci_high,color=BLUE,alpha=.12,
                    label='95% модельный доверительный интервал')
    ax.plot(data.year,data.b,'-o',color=BLUE,lw=1.6,markersize=3.3,label='Оценка b по Аки')
    ax.set(xlabel='Год',ylabel='Параметр b (MLE)',xlim=(1965,2021))
    ax.legend(fontsize=9)
    _finish(fig,'09_динамика_b_mle.png',
            'Годовые оценки b методом максимального правдоподобия и модельный 95% доверительный интервал.')


def fig10_mle_chow(mle, chow):
    data=mle.loc[mle.m0==5.0].sort_values('year')
    years=data.year.to_numpy()
    fig,ax=_new('Рис. 10 · Критерий Чоу для фиксированного 1994 года (MLE)',height=5.6)
    ax.plot(years,data.b,'-o',color=BLUE,lw=1.3,markersize=3.2,label='Годовые MLE-оценки b')
    ax.plot(years,chow['fit_single'],'--',color=GRAY,lw=1.5,label='Единый линейный тренд')
    pre=years<=BREAK_YEAR
    ax.plot(years[pre],chow['fit_two'][pre],color=RED,lw=2.5,label='Тренд до 1994 г.')
    ax.plot(years[~pre],chow['fit_two'][~pre],color=RED,lw=2.5,label='Тренд после 1994 г.')
    ax.axvline(BREAK_YEAR,ls='--',color=ORANGE,label='Фиксированная дата: 1994 г.')
    ax.set(xlabel='Год',ylabel='Параметр b (MLE)',xlim=(1965,2021))
    ax.text(.02,.95,f'F = {chow["F"]:.2f}; p = {chow["p"]:.6f} (номинальное)',
            transform=ax.transAxes,va='top',color=GRAY,fontsize=10)
    ax.legend(ncol=2,fontsize=9)
    _finish(fig,'10_критерий_чоу_mle.png',
            'Одна прямая сопоставлена с двумя ветвями при заранее выбранной дате 1994 года.')


def fig11_mle_thresholds(mle):
    fig,ax=_new('Рис. 11 · Чувствительность MLE к фиксированному порогу M0',height=6.1)
    colors={5.0:BLUE,5.5:ORANGE,6.0:GREEN}
    for value in (5.0,5.5,6.0):
        df=mle.loc[mle.m0==value].sort_values('year')
        years=df.year.to_numpy()
        color=colors[value]
        if value in (5.0,5.5):
            ax.fill_between(years,df.ci_low,df.ci_high,color=color,alpha=.1)
        ax.plot(years,df.b,'-o',color=color,markersize=2.9,lw=1.25,
                label=f'Годовые оценки при M0 = {value:.1f}'.replace('.',','))
        side=(years>BREAK_YEAR).astype(float)
        X=np.column_stack([np.ones(len(years)),years-BREAK_YEAR,side,(years-BREAK_YEAR)*side])
        fit=X@np.linalg.lstsq(X,df.b.to_numpy(),rcond=None)[0]
        ax.plot(years[side==0],fit[side==0],color=color,lw=2.0,alpha=.9)
        ax.plot(years[side==1],fit[side==1],color=color,lw=2.0,alpha=.9)
    ax.axvline(BREAK_YEAR,color=RED,ls='--',lw=1.7,label='1994 год')
    ax.set(xlabel='Год',ylabel='Параметр b (MLE)',xlim=(1965,2021))
    ax.legend(ncol=2,fontsize=9)
    _finish(fig,'11_mle_разные_пороги.png',
            'Годовые MLE-оценки при M0=5,0;5,5;6,0 с кусочно-линейными трендами и границей 1994 г.')


def plot_all(isc_catalog, rfm, mle, neic_series, isc_series, supf, chow):
    """Создать все одиннадцать рисунков в порядке их нумерации в статье."""
    fig01_frequency(isc_catalog,rfm)
    fig02_mc(rfm)
    fig03_a(rfm)
    fig04_rfm(rfm,supf)
    fig05_supf(supf)
    fig06_neic_median(neic_series)
    fig07_neic_weighted(neic_series)
    fig08_isc_bpositive(isc_series)
    fig09_mle(mle)
    fig10_mle_chow(mle,chow)
    fig11_mle_thresholds(mle)
