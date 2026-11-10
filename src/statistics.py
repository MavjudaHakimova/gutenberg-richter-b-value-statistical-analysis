"""Критерий Чоу, supF с Wild-Block Bootstrap и поправка Холма."""
from __future__ import annotations

import numpy as np
from scipy.stats import f as fisher_f


def chow_fixed(years, values, break_year: int = 1994) -> dict:
    """Тест Чоу для *заранее фиксированной* даты: две прямые против одной."""
    t = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    x = t - break_year
    side = (t > break_year).astype(float)
    design0 = np.column_stack((np.ones(len(x)), x))
    design1 = np.column_stack((np.ones(len(x)), x, side, x*side))
    c0 = np.linalg.lstsq(design0, y, rcond=None)[0]
    c1 = np.linalg.lstsq(design1, y, rcond=None)[0]
    fitted0, fitted1 = design0 @ c0, design1 @ c1
    rss0 = np.sum((y-fitted0)**2)
    rss1 = np.sum((y-fitted1)**2)
    n = len(y)
    if n < 7 or rss1 <= 0:
        raise ValueError('Недостаточно годов для критерия Чоу')
    f_value = float(((rss0-rss1)/2) / (rss1/(n-4)))
    return {'F': f_value, 'p': float(fisher_f.sf(f_value, 2, n-4)),
            'jump': float(c1[2]), 'slope_before': float(c1[1]),
            'slope_after': float(c1[1]+c1[3]), 'fit_single': fitted0,
            'fit_two': fitted1, 'years': t}


def supf_wild_block(years, values, se=None, *, n_boot: int = 1999,
                    block_years: int = 3, seed: int = 20261025,
                    trimming: float = 0.15, weighted: bool = True) -> dict:
    """Значимость неизвестной даты: supF и знаковый бутстреп блоками лет.

    В отличие от обычного теста Чоу, бутстреп учитывает перебор кандидатов
    и использует то же правило максимизации для каждой псевдовыборки.
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    if se is None:
        se = np.ones(len(y))
    errors = np.asarray(se, dtype=float)
    valid = np.isfinite(years) & np.isfinite(y)
    years, y, errors = years[valid], y[valid], errors[valid]
    n = len(y)
    min_side = max(5, int(np.ceil(trimming*n)))
    if n < 2*min_side + 4:
        raise ValueError('Недостаточно годов для supF-теста')
    good = errors[np.isfinite(errors) & (errors > 0)]
    fallback = float(np.median(good)) if len(good) else 0.15
    errors = np.where(np.isfinite(errors) & (errors > 0), errors, fallback)
    weights = 1.0 / np.maximum(errors, 0.05)**2 if weighted else np.ones(n)
    weights = weights / weights.mean()
    sw = np.sqrt(weights)
    x = years-years.mean()
    null = np.column_stack([np.ones(n), x])*sw[:, None]
    q0, _ = np.linalg.qr(null, mode='reduced')
    weighted_y = y*sw
    fit0 = q0 @ (q0.T@weighted_y)
    residual = weighted_y - fit0
    breaks = years[min_side-1:n-min_side]
    bases = []
    designs = []
    for year in breaks:
        after = (years>year).astype(float)
        design = np.column_stack([np.ones(n), x, after, x*after])*sw[:, None]
        q, _ = np.linalg.qr(design, mode='reduced')
        bases.append(q)
        designs.append(design)
    q_all = np.stack(bases)

    def fvalues(matrix):
        matrix = np.atleast_2d(matrix)
        yy = np.einsum('bn,bn->b',matrix,matrix)
        nullp = matrix@q0
        rss0 = np.maximum(yy - np.sum(nullp**2,axis=1), 0)
        projection = np.einsum('bn,kni->bki', matrix, q_all, optimize=True)
        rss1 = np.maximum(yy[:,None]-np.sum(projection**2,axis=2), 1e-12)
        f = np.maximum((rss0[:,None]-rss1)/2,0)/(rss1/(n-4))
        return f, rss0, rss1

    obs, rss0, rss1 = fvalues(weighted_y)
    f_curve = obs[0]
    best = int(np.argmax(f_curve))
    nblock = ((years-years.min())//block_years).astype(int)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0,1.0], size=(n_boot,int(nblock.max())+1))
    sims = fit0[None,:]+residual[None,:]*signs[:,nblock]
    maxima = np.max(fvalues(sims)[0],axis=1)
    max_f = float(f_curve[best])
    p = (1+int(np.count_nonzero(maxima>=max_f)))/(1+n_boot)
    critical = float(np.quantile(maxima,0.95))
    c1 = np.linalg.lstsq(designs[best],weighted_y,rcond=None)[0]
    return {'break_year':int(breaks[best]), 'max_f':max_f,
            'p':float(p), 'critical':critical,
            'breaks':breaks.astype(int), 'f_curve':f_curve,
            'delta_bic':float(n*np.log(rss0[0]/rss1[0,best])-2*np.log(n)),
            'coefs':c1, 'mean_year':float(years.mean()),
            'n_boot':int(n_boot)}


def holm_adjust(p_values):
    """Скорректировать региональные p-значения методом Холма (FWER)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p,np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        adjusted = np.maximum.accumulate((len(order)-np.arange(len(order)))*p[order])
        out[order] = np.minimum(1.0, adjusted)
    return out
