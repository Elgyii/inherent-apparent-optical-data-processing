# -------------------------------------------------------------------------
# (c) 2015 MaureER Aug-7
# Calculate CDOM absorption
# -------------------------------------------------------------------------
# clear
# clc
# 1-- read ASCII to one XLS file
# dID = {'ref','blank','sample'};
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.optimize import curve_fit

from autils import get_name, merge_pd, read_csv, strip
from figure import Figure


def ay_absorbance(ref: pd, blank: pd, sample: pd):
    """
       (c) 2015 MaureER Aug-7, Tested on Matlab 2015a

       Process CDOM data according to Babin et al, 2003 doi:10.1029/2001JC000882
       However, for baseline correction it uses 700 nm.

       The function also assumes that for each file name there are two data
       files, one ending with '1' and the other with '2'. It then finds 1 and 2
       with same filename and use the average of the two. ANAP data is processed
       according to Babin et al, 2003 doi:10.1029/2001JC000882

       It is a good idea to always distinguish characters used for blanks and
       samples. If they all have same initial character, this Software won't be
       able to make the distinction

        Parameters
        ----------
        ref: pd.DataFrame:
        blank: pd.DataFrame:
        sample: pd.DataFrame:

        Returns
        -------
    """
    # ---------------------------------------------
    # Calculate absorption averaging all the blanks

    ay = None
    cls = get_name(names=sample.columns, sta=True)

    # Youhei Yamashita, Hokkaido University procedure
    # correct the baseline (e.g. effect of small particles) by subtracting
    # average value ranging from 590-600 nm from the entire spectrum.
    # bsl_cor = (sample.index >= 590) & (sample.index <= 600)

    # Babin et al, 2003, doi:10.1029/2001JC000882
    # correct the baseline
    # bsl_cor = (sample.index >= 683) & (sample.index <= 687)
    # Hirawake's method to correct (remove 700 nm to all absorption spectra)
    bsl_cor = sample.index == 700
    shift = sample.index == 800

    # Convert spectrophotometre absorbance to ay absorption coefficient
    # L - cell length 10 cm
    # C - spectrophotometre absorbance constant 2.303

    for name in cls:
        idx = sample.columns.isin([f'{name}1', f'{name}2'])
        # remove the reference blank and the milli-Q blank
        tmp = sample.loc[:, idx].copy()
        bln = list(blank.columns)[0]
        rfn = list(ref.columns)[0]

        for sc in sample.loc[:, idx]:
            ay_abs = tmp.loc[:, sc] - blank.loc[:, bln]
            ay_abs = ay_abs - ref.loc[:, rfn]
            # absorbance to ay absorption coefficient
            ay_abs = ay_abs * (2.303 / 0.1)
            # correct the baseline
            tmp.loc[:, sc] = ay_abs - ay_abs[bsl_cor].values.mean()

        # take the mean of the two spectrum
        mean = tmp.mean(axis=1).to_frame(name=(name, 'mean'))
        std = tmp.std(axis=1).to_frame(name=(name, 'std'))

        tmp = merge_pd(left=mean, right=std)
        ay = merge_pd(left=ay, right=tmp)
    return ay


@dataclass
class SlopeCI:
    """
    Function return values
     r:  float
        correlation coefficient
     p:  float
        p-value for the test of the correlation coefficient
    """
    ay412: list
    ay440: list
    ay443: list
    rsq: float
    rsq_adj: float


def ay_slope(ayd: pd.DataFrame, columns: dict):
    """calculate CDOM slope
    from excel file in NAS2 370-440
    """
    wl = ayd.index.values
    idx = (wl >= 350) & (wl <= 500)
    # idx = ayData.lambda >= 370 & ayData.lambda <= 440
    # ay_lambda0 = 443  # const
    ayd_slope = ayd.loc[idx, :]  # range 350-500 nm
    xs = ayd_slope.index.values

    ay443 = ayd.loc[wl == 443, :].values[0]  # line matrix of ay at 443

    # ay440 = ayd.loc[wl == 440, :]

    # func = lambda x, a, b: a * np.exp(-b * (x - 443))
    def func(x, a, b):
        return a * np.exp(-b * (x - 443))

    rng = np.random.default_rng()
    fields = ["value", "lower", "upper"]
    rows = ['ay412', 'ay440', 'ay443', 'Rsq', 'Rsq_adj']
    result = pd.DataFrame([])

    for j, col in enumerate(columns.keys()):
        # ay(443) for each sample
        yf = ayd_slope.loc[:, (col, 'mean')].values
        # ys = ayd_slope.loc[:, (col, 'std')].values[::-1]
        p0 = [ay443[j], 0.015]
        # bounds = (0, [ay443[j] + ay443[j] / 2, 1.])
        opt, cov = curve_fit(func
                             , xdata=xs
                             , ydata=yf
                             , p0=p0
                             # , sigma=ys
                             , method='lm')
        # , bounds=bounds)
        # print(f'opt: {opt}\ncov: {cov}')
        # cov - estimated variance-covariance matrix for the estimated　coefficients
        # MSE - an estimate of the variance of the error term
        # yp = func(xs, *opt)
        yp = func(wl, *opt)
        yi = ayd.loc[:, (col, 'mean')]
        # plt.plot(wl, yi, label='original')
        # plt.plot(wl, yp, 'r-',
        #          label='fit: a=%5.3f, b=%5.3f' % tuple(opt))
        # plt.legend()
        # plt.show()

        # confidence intervals at 95%
        # https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit
        sigma_ab = np.sqrt(np.diag(cov))
        # # confidence interval of the fit params
        # s = ufloat(opt[0], sigma_ab[0])
        # m = ufloat(opt[1], sigma_ab[1])

        # upper and lower bounds
        bound_upper = func(wl, *(opt + sigma_ab))
        bound_lower = func(wl, *(opt - sigma_ab))

        # goodness of fit
        # ri = yi − f(xi, b)
        residual = yi - yp
        # sum of squares of the residuals
        sse = np.sum(residual ** 2)
        # sum of squares between the data points and their mean2
        sst = np.sum((yi - yi.mean() ** 2))
        rsq = 1 - (sse / sst)
        # R-square adjusted
        dfe = yi.size - len(opt)
        rsq_adj = 1 - (((yi.size - 1) / dfe) * (sse / sst))
        # print(f'NUM: {(yi.size - 1) / dfe}\nDEN: {sse/sst}\n'
        #       f'SST: {sst}\nRsq: {rsq}\nRadj: {rsq_adj}\n')

        idx = wl == 412
        ay412 = [yp[idx][0], bound_lower[idx][0], bound_upper[idx][0]]
        idx = wl == 440
        ay440 = [yp[idx][0], bound_lower[idx][0], bound_upper[idx][0]]
        idx = wl == 443
        ay443_ = [yp[idx][0], bound_lower[idx][0], bound_upper[idx][0]]

        tmp = pd.DataFrame([ay412, ay440, ay443_,
                            [rsq, np.nan, np.nan],
                            [rsq_adj, np.nan, np.nan]]
                           , columns=[[col, col, col], fields]
                           , index=pd.Index(rows))
        result = merge_pd(left=result, right=tmp)
    return result


if __name__ == '__main__':
    CRUISE = '202204'
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\ay')
    IMAGE_PATH = HOME_DIR.joinpath('Figures')

    TODAY = datetime.today().strftime('%Y%m%d')
    SAVE = IMAGE_PATH.joinpath(f'ay_hayatsuki{CRUISE}_{TODAY}.png')
    if not IMAGE_PATH.is_dir():
        IMAGE_PATH.mkdir()

    SHEETS = ['blank', 'reference', 'sample']
    FILE_PATTERN = ('AY_BLK*.ASC', 'AY_REF*.ASC', 'AY_ST*.ASC')
    XLS_FILE = HOME_DIR.joinpath(
        r'OpticalData', f'ay_absorption_toyama{CRUISE}.xlsx')

    # clr = {'8': {'c': '#1f77b4', 'l': '-'},
    #        '10.1': {'c': '#ff7f0e', 'l': '-'},
    #        '10.2': {'c': '#ff7f0e', 'l': '--'},
    #        '11': {'c': '#2ca02c', 'l': '--'},
    #        '13': {'c': '#d62728', 'l': '-'},
    #        'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}
    # clr = {'8': {'c': 'k', 'l': '-'},
    #        '10': {'c': '#ff7f0e', 'l': '-'},
    #        '11': {'c': '#2ca02c', 'l': '-'},
    #        '12': {'c': 'b', 'l': '-'},
    #        '13': {'c': 'r', 'l': '-'},
    #        'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

    bsl, blk, smp = [], [], pd.array([])

    with pd.ExcelWriter(XLS_FILE) as writer:
        for i, pat in enumerate(FILE_PATTERN):
            data = None
            for f in DATA_PATH.glob(pat):
                print(f.name)
                temp = read_csv(file=f)
                data = merge_pd(left=data, right=temp)
            # 1 -- save data
            data.to_excel(writer, sheet_name=SHEETS[i])

            # 2 -- read XLS file and data
            if pat == FILE_PATTERN[0]:
                cols = get_name(names=data.columns)
                blk = data.mean(axis=1).to_frame(cols[0])
                # blk.rename_axis(mapper=cols[0], inplace=True)

            if pat == FILE_PATTERN[1]:
                cols = get_name(names=data.columns)
                bsl = data.mean(axis=1).to_frame(cols[0])
                # bsl.rename_axis(mapper=cols[0], inplace=True)

            if pat == FILE_PATTERN[2]:
                smp = data
            # print(smp)
        cdom = ay_absorbance(ref=bsl, blank=blk, sample=smp)
        # save ay and sample-blank to xls file
        cdom.to_excel(writer, sheet_name='ay')
        col_names = strip(cdom.columns)
        Figure(dataset=cdom,
               columns=col_names,
               filename=SAVE,
               fig_size=(16, 11),
               font_size=20).save(case='ay')
        cdom = ay_slope(ayd=cdom, columns=col_names)
        print(cdom.T)
        cdom.T.to_excel(writer, sheet_name='slope-ci')
