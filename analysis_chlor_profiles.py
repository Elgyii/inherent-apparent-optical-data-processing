import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker


def smooth(data, window_length: int = 3):
    """
    1-D data filtering or smoothing using `savgol_filter`.
    In general `savgol_filter` produces good results compared to a few other methods.
    For more, please check https://docs.scipy.org/doc/scipy/reference/signal.html

    Parameters
    ----------
    data: np.array
        Input 1-D array-like
    window_length: int

    Returns
    -------
        y: ndarray
            filtered data, same shape as input
    """
    from scipy.signal import savgol_filter

    data = np.asarray(data)

    return savgol_filter(data,
                         window_length=window_length,
                         polyorder=1)
    # ax.plot(wavelength, tdata, label='savgol_filter')
    # ax.legend()
    # tdata[:] = res
    #
    # plt.close()
    # return True


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    DATA_PATH = BASE.joinpath('CTD')
    IMAGE_PATH = BASE.joinpath('Figures')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = 27, 24  # (10, 9)
        plt.rcParams['font.size'] = 40

        cols = {'深度 [m]': 'Depth [m]',
                '水温 [℃]': 'Temperature [℃]',
                '塩分 [ ]': 'Salinity [psu]',
                'Chl-a [μg/l]': 'Chl-a [mg m^${-3}$]'}

        clr = {'8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#2ca02c', 'l': '-'},
               '12': {'c': 'b', 'l': '-'},
               '13': {'c': 'r', 'l': '-'},
               'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

        fig, ax = plt.subplots(nrows=1, ncols=3,
                               sharex='col', sharey='all')

        for f in sorted(DATA_PATH.glob('*.csv')):
            # f = list(path.glob(f'({sta})*.csv'))
            sta = ''.join(re.findall(r'\((\d+)\)', f.name))
            if sta == '9':
                continue
            sta = 'SP' if '(12)総' in f.name else sta
            print(f'{sta}: {f.name}')
            # print(sds, sds.columns, sta)
            # break

            sds = pd.read_csv(f, skiprows=43, encoding='shift_jis')
            sds.rename(columns=cols, inplace=True)
            x = smooth(data=sds['Chl-a [mg m^${-3}$]'])
            lw = 3
            # lw = 9 if sta in ('SP', '10') else 3

            if len(ax) == 3:
                sds.plot(x='Temperature [℃]', y='Depth [m]', label=f'STA. {sta}',
                         ax=ax[0], color=clr[f'{sta}']['c'], lw=lw)
                ax[1].plot(sds['Salinity [psu]'], sds['Depth [m]'],
                           color=clr[f'{sta}']['c'], lw=lw)
                ax[2].plot(x, sds['Depth [m]'], color=clr[f'{sta}']['c'], lw=lw)
            else:
                ax[0].plot(x, sds['Depth [m]'], color=clr[f'{sta}']['c'], lw=lw)
                sds.plot(x='Salinity [psu]', y='Depth [m]', label=f'STA. {sta}',
                         ax=ax[1], color=clr[f'{sta}']['c'], lw=lw)
            # print(sds['Depth [m]'].min())

        ax[0].set_yscale('log')
        if len(ax) == 3:
            ax[1].set_xlim([10, 35])
            ax[2].set_xlim([-0.01, 4])
            ax[2].set_ylim([0.09, 100])

            ax[0].set_xlabel('Temperature [℃]', fontdict={'weight': 'bold'})
            ax[1].set_xlabel('Salinity [psu]', fontdict={'weight': 'bold'})
            ax[2].set_xlabel(r'Chl-$\ita$ [mg m$^{-3}$]', fontdict={'weight': 'bold'})
        else:
            ax[0].set_xlim([-0.01, 4])
            ax[0].set_ylim([0.09, 100])
            ax[1].set_xlim([10, 35])

            ax[0].set_xlabel(r'Chl-$\ita$ [mg m$^{-3}$]', fontdict={'weight': 'bold'})
            ax[1].set_xlabel('Salinity [psu]', fontdict={'weight': 'bold'})

        ax[0].set_ylabel('Depth [m]', fontdict={'weight': 'bold'})
        ax[0].invert_yaxis()
        # plt.show()
        fmt = ticker.FormatStrFormatter('%g')
        ax[0].yaxis.set_major_formatter(fmt)

        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        save = IMAGE_PATH.joinpath(f'fig_ctd_profiles_{len(ax)}.png')
        fig.savefig(save)
        # plt.show()
        print(save)
        # ==================================================

    except KeyboardInterrupt:
        sys.exit(0)
