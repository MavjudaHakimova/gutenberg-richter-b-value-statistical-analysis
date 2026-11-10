"""Проверки, подтверждающие воспроизведение чисел из статьи."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mle import all_thresholds
from src.rfm import estimate_mc
from src.statistics import chow_fixed,holm_adjust,supf_wild_block


class TestPaperResults(unittest.TestCase):
    """Сравнить вычисления на исходных данных с опубликованными результатами."""

    @classmethod
    def setUpClass(cls):
        catalog=pd.read_csv(ROOT/'data/isc_mw_1960_2021.csv.gz',
                            usecols=['year','magnitude'])
        cls.catalog=catalog[catalog['year'].between(1965,2021)]

    def test_rfm_1965(self):
        """Проверка первого годового Mc и b в архивном RFM-эксперименте."""
        mag=self.catalog.loc[self.catalog.year==1965,'magnitude'].to_numpy()
        result=estimate_mc(mag)
        self.assertAlmostEqual(result['mc'],5.22708,places=4)
        self.assertAlmostEqual(result['b'],1.51244548,places=4)

    def test_mle_three_thresholds(self):
        """Три строки таблицы 1 должны воспроизводиться численно."""
        yearly=all_thresholds(self.catalog)
        reference={5.0:(9.68781226945,0.000259486010811),
                   5.5:(6.03726573788,0.00434427911711),
                   6.0:(4.27480707171,0.0190037441878)}
        for threshold,(f_expected,p_expected) in reference.items():
            with self.subTest(threshold=threshold):
                sub=yearly.loc[yearly.m0==threshold]
                test=chow_fixed(sub.year,sub.b)
                self.assertAlmostEqual(test['F'],f_expected,places=6)
                self.assertAlmostEqual(test['p'],p_expected,places=8)

    def test_supf_rfm_1994(self):
        """Неизвестная дата и p должны совпадать с результатом статьи."""
        # Берём рассчитанный с полной точностью ряд: таблица интервалов
        # из исходного архива содержит округлённые значения b.
        original=pd.read_csv(ROOT/'results/rfm_годовые_оценки.csv')
        se=(original.ci_high-original.ci_low)/(2*1.96)
        result=supf_wild_block(original.year,original.b,se,n_boot=1999)
        self.assertEqual(result['break_year'],1994)
        self.assertAlmostEqual(result['max_f'],6.8992,places=3)
        self.assertAlmostEqual(result['p'],0.0485,places=4)

    def test_holm_multiple_comparisons(self):
        """Последовательная поправка Холма должна сохранять монотонность."""
        adjusted=holm_adjust([0.005,0.03,0.04,0.30])
        np.testing.assert_allclose(adjusted,[0.02,0.09,0.09,0.3])


if __name__=='__main__':
    unittest.main()
