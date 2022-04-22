import re
import sys
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from scipy.stats import linregress


def toyama_pj_dset(file: str = 'ToyamaBay.DatabaseUp2date.txt'):
    dset = pd.read_csv(INSITU_PATH.joinpath(file), sep='\t')
    nan = dset.isin([-999, '-999'])
    dset.mask(nan, inplace=True)
    return dset


def kozuka_ctd_dset(file: str = 'ToyamaPJ_CTD_2014_2021.txt'):
    dset = pd.read_csv(INSITU_PATH.joinpath(file), sep='\t')
    rename = {'Longitude [degrees_east]': 'Longitude [°E]',
              'Latitude [degrees_north]': 'Latitude [°N]',
              'Temperature [~^oC]': 'Temperature [°C]',
              'Salinity [g kg~^-~^1]': 'Salinity [g kg$^{-1}$]',
              'Chl-a [mg/m~^3]': 'Chla [mg m$^{-3}$]'}
    dset.rename(columns=rename, inplace=True)
    return dset


def kozuka_bottle_dset(file: str = 'ToyamaPJ_bottle_data.dtal.txt'):
    dset = pd.read_csv(INSITU_PATH.joinpath(file), sep='\t')
    rename = {'date': 'Date',
              'point': 'Station',
              'tomeido': 'Transparency [m]',
              'sal': 'Salinity [g kg$^{-1}$]',
              'chl': 'Chla [mg m$^{-3}$]',
              'ss': 'SS [mg L$^{-1}$]',
              'cdom440': 'CDOM$_{440}$ [m$^{-1}$]',
              'cdom412': 'CDOM$_{412}$ [m$^{-1}$]',
              'do': 'DO [mg L$^{-1}$]',
              'dip': 'DIP [mg L$^{-1}$]',
              'din': 'DIN [mg L$^{-1}$]',
              'temp': 'Temperature [°C]',
              'sal_mix': 'Salinity$_{(0.5+2)}$ [g kg$^{-1}$]',
              'chl_mix': 'Chla$_{(0.5+2)}$ [mg m$^{-3}$]',
              'ss_mix': 'SS$_{(0.5+2)}$ [mg L$^{-1}$]',
              'cdom440_mix': 'CDOM$_{440(0.5+2)}$ [m$^{-1}$]',
              'cdom412_mix': 'CDOM$_{412(0.5+2)}$ [m$^{-1}$]',
              'do_mix': 'DO$_{(0.5+2)}$ [mg L$^{-1}$]',
              'dip_mix': 'DIP$_{(0.5+2)}$ [mg L$^{-1}$]',
              'din_mix': 'DIN$_{(0.5+2)}$ [mg L$^{-1}$]',
              'sal05': 'Salinity$_{(0.5)}$ [g kg$^{-1}$]',
              'year': 'Year',
              'month': 'Month',
              'day': 'Day'}
    dset.rename(columns=rename, inplace=True)
    return dset


def cdom_dset(file: str = 'ay_absorption_toyama202109.xlsx'):
    dset = pd.read_excel(OPTICS_PATH.joinpath(file),
                         sheet_name='ay',
                         skiprows=[1, 2],
                         usecols=[0, 1, 3, 5, 7])
    rename = {'C_STA1': 'STA. 13', 'C_STA2': 'STA. 8',
              'C_STA3': 'STA. 10', 'C_STA4': 'STA. 11'}
    dset.rename(columns=rename, inplace=True)
    return dset


def ctd_profiles(file: Path, lon: float = None, lat: float = None):
    rename = {'深度 [m]': 'Depth [m]',
              '水温 [℃]': 'Temperature [℃]',
              '塩分 [ ]': 'Salinity [g kg$^{-1}$]',
              'Chl-a [μg/l]': 'Chl-a [mg m^${-3}$]'}

    dset = pd.read_csv(file, skiprows=43, encoding='shift_jis')
    dset.rename(columns=rename, inplace=True)
    if lon and lat:
        slab = 'Salinity [g kg$^{-1}$]'
        pres = dset['Depth [m]'].values
        sal = dset[slab].values
        # print(lon, lat, sal.shape, pres.shape)
        dset.loc[:, slab] = gsw.SA_from_SP(sal, -pres, lon, lat)

    return dset


def insitu_chl(file: str = 'chl_202109-Toyama.xlsx'):
    file = OPTICS_PATH.joinpath(file)
    return pd.read_excel(file, sheet_name='chla-comp')


def kozuka_vs_maure_dset(dkz, dmr, xlabs: list, ylabs: list):
    # xlabels0 = 'CDOM$_{412}$ [m$^{-1}$]', 'CDOM$_{440}$ [m$^{-1}$]'
    # xlabels1 = 'CDOM$_{412(0.5+2)}$ [m$^{-1}$]', 'CDOM$_{440(0.5+2)}$ [m$^{-1}$]'
    # ylabels = 'Salinity [g kg$^{-1}$]', 'Salinity$_{(0.5+2)}$ [g kg$^{-1}$]'

    fg, xs = plt.subplots(nrows=2, ncols=2)
    gf, xa = plt.subplots(nrows=2, ncols=2)

    for i, yl in enumerate(ylabs):
        for j, xl in enumerate(xlabs):
            dset = dkz.loc[:, [x, y]].dropna()
            print(f'{y} | {x} | {dset.shape}', end='')

            if dset.shape[0] > 5:
                dset.plot.scatter(x=x, y=y, ax=xs[i, j])

            dset = dmr.loc[:, [x, y]].dropna()
            print(f' {dset.shape}')
            if dset.shape[0] > 5:
                dset.plot.scatter(x=x, y=y, ax=xa[i, j])
    plt.show()


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama')
    INSITU_PATH = BASE.joinpath(r'Insitu\txt')
    OPTICS_PATH = BASE.joinpath(r'Hayatsuki\202109\OpticalData')
    IMAGE_PATH = BASE.joinpath(r'Hayatsuki\202109\Figures')
    H202109_CTD = BASE.joinpath(r'Hayatsuki\202109\CTD')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (16, 13)  # (12, 7)  # (14, 9)
        plt.rcParams['font.size'] = 40

        clr = {'8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#2ca02c', 'l': '-'},
               '12': {'c': 'b', 'l': '-'},
               '13': {'c': 'r', 'l': '-'},
               'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

        fig, ax = plt.subplots()
        XLABEL = 'CDOM$_{440(0.5+2)}$ [m$^{-1}$]'  # 'CDOM$_{440}$ [m$^{-1}$]'
        YLABEL = 'Salinity$_{(0.5)}$ [g kg$^{-1}$]'  # , 'Salinity [g kg$^{-1}$]'
        s = 300

        # ======== Datasets ========
        BOTTLE = kozuka_bottle_dset()
        INSITU_CHL = insitu_chl()
        CDOM = cdom_dset()
        # ==========================

        BOTTLE['Date'] = pd.to_datetime(BOTTLE['Date'], format='%Y-%m-%d')
        idx = BOTTLE['Date'] > pd.to_datetime('2015-12-31', format='%Y-%m-%d')
        df_bottle = BOTTLE.loc[idx, :]
        # After 2015 salinity is from CTD. Bottle salinity has issues

        msk = (df_bottle.loc[:, XLABEL] > 0.4) | \
              (df_bottle.loc[:, XLABEL] > 0.2) & \
              (df_bottle.loc[:, YLABEL] > 30)

        df_suspect = df_bottle.loc[msk, :]
        df_suspect.plot.scatter(x=XLABEL, y=YLABEL,
                                s=s, c='c',
                                alpha=0.3,
                                ax=ax,
                                edgecolors='k',
                                facecolors='w')

        df_bottle.loc[msk, :] = np.nan
        df_bottle.plot.scatter(x=XLABEL, y=YLABEL, s=s,
                               alpha=0.5, c='0.9', ax=ax,
                               edgecolors='k',
                               facecolors='w')

        x, y = df_bottle.loc[:, XLABEL], df_bottle.loc[:, YLABEL]
        idx = ~(np.isnan(x) | np.isnan(y))
        x, y = x.values[idx], y.values[idx]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print("slope: %f, intercept: %f" % (slope, intercept))
        print("R-squared: %f" % r_value ** 2)
        idx = np.argsort(x)
        ax.plot(x[idx], intercept + slope * x[idx], 'r', lw=2)

        # ====== Add Hayatsuki CDOM =======
        wl = CDOM['Wavelength [nm]'].values
        wl_idx = (wl >= 438) & (wl <= 442)
        new_lab = 'Salinity [g kg$^{-1}$]'

        for f in sorted(H202109_CTD.glob('*.csv')):
            sta = ''.join(re.findall(r'\((\d+)\)', f.name))
            if sta == '9':
                continue
            sta = 'SP' if '(12)総' in f.name else sta
            print(f'{sta:2}: {f.name:50} ', end='')

            if f'STA. {sta}' not in CDOM.columns:
                print()
                continue

            # Lon/lat from CHL dataset
            idx = INSITU_CHL['Sta'].isin([f'STA. {sta}'])
            nol = INSITU_CHL.loc[idx, 'lon'].values[0]
            tal = INSITU_CHL.loc[idx, 'lat'].values[0]

            sds = ctd_profiles(file=f, lon=nol, lat=tal)

            idx = (sds['Depth [m]'].values >= .25) & \
                  (sds['Depth [m]'].values <= .75)
            y = sds[new_lab].values[idx].mean()
            x = CDOM.loc[wl_idx, f'STA. {sta}'].values.mean()

            print(f'| X: {x:.3f} Y: {y:.3f}')
            ax.scatter(x, y,
                       c=clr[sta.strip()]['c'],
                       marker='s',
                       edgecolors='k',
                       alpha=0.6,
                       s=s,
                       label=f'STA. {sta} (0 m)',
                       linewidths=1)

        ax.set_yticks((15, 20, 25, 30, 35))
        ax.legend()

        fmt = ticker.FormatStrFormatter('%g')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        fig_name = 'fig_salinity_cdom_compare.png'
        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        save = IMAGE_PATH.joinpath(fig_name)
        plt.savefig(save)

    except KeyboardInterrupt:
        sys.exit(0)
