import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker


def skip(file):
    if 'KdRrs.csv' in file.name:
        return False
    return True


def get_ed0(data):
    diff = data['Depth'] - data['Depth'].at[0]
    select = np.where(diff == diff.max())[0]
    temp = data.loc[:select[0], :]
    diff = diff[:select[0] + 1]
    loc = abs(diff) < 0.4

    # plt.plot(temp['Depth'])
    # plt.plot(temp['Depth'].loc[loc])
    # plt.show()
    temp.drop(columns=['Depth'], inplace=True)
    return temp.loc[loc, :]


def get_num(numstr: str):
    res = re.findall(r'(\(.*)([8]|[0-9][0-3])\)', numstr)
    if len(res) == 0:
        res = re.findall(r'(\(.*) (.*)\)', numstr)
    return ''.join(res[0][1])


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    DATA_PATH = BASE.joinpath(r'OpticalData\RAMSES')
    IMAGE_PATH = BASE.joinpath('Figures')
    RRS_FILE = DATA_PATH.joinpath('data_Spectrum_Calibrated_2021-09-29_08-21-06_671.xlsx')
    CHL_FILE = BASE.joinpath('chl_202109-Toyama.xlsx')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        # plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (16, 13)  # (12, 7)  # (14, 9)
        plt.rcParams['font.size'] = 40

        # clr = {'8': {'c': '#1f77b4', 'l': '-'},
        #        '10.1': {'c': '#ff7f0e', 'l': '-'},
        #        '10.2': {'c': '#ff7f0e', 'l': '--'},
        #        '11': {'c': '#2ca02c', 'l': '-'},
        #        '13': {'c': '#d62728', 'l': '-'},
        #        'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}
        clr = {'8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#2ca02c', 'l': '-'},
               '12': {'c': 'b', 'l': '-'},
               '13': {'c': 'r', 'l': '-'},
               'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

        PRR_DSET = pd.read_excel(RRS_FILE, sheet_name='PRR_Rrs')
        RAMSES_DSET = pd.read_excel(RRS_FILE, sheet_name='RAMSES_Rrs')

        XLIM = (PRR_DSET['Wavelength [nm]'].min() - 10,
                PRR_DSET['Wavelength [nm]'].max() + 10)
        print(XLIM)
        # ==================================================
        # PRR Rrs image
        fig_name = 'fig_analysis_Rrs_PRR.png'
        fig, ax = plt.subplots()

        for col in PRR_DSET.columns:
            if 'Wavelength' in col:
                continue

            s = get_num(numstr=col)
            print(f'{col}: {s}')
            ax.plot(PRR_DSET['Wavelength [nm]'],
                    PRR_DSET[col], clr[s]['l'],
                    color=clr[s]['c'],
                    label=col[col.index('(') + 1:-1],
                    lw=3)

        for vl in PRR_DSET['Wavelength [nm]'].values:
            ax.axvline(x=vl, **{'linestyle': ':', 'color': 'gray'})

        ax.set_ylabel(r'$\itR$$_{rs}$ [sr$^{-1}$]')
        ax.set_xlabel(r'$\lambda$ [nm]')
        # ax.set_xlim(xlm)
        ax.set_xlim(400, 700)
        ax.set_ylim(-0.0001, 0.0045)

        ax.set_yticks((0, 0.001, 0.002, 0.003, 0.004))
        fmt = ticker.FormatStrFormatter('%g')
        ax.yaxis.set_major_formatter(fmt)
        ax.legend()

        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        save = IMAGE_PATH.joinpath(fig_name)
        fig.savefig(save)
        # plt.show()
        plt.close(fig)
        print(f'Image: {save}')
        # ==================================================

        # PRR vs RAMSES
        fig_name = 'fig_analysis_Rrs_compare_PRR_vs_RAMSES.png'
        fig, axs = plt.subplots(nrows=3, ncols=2,
                                sharex='all', sharey='all',
                                figsize=(18, 24))
        # axs[-1, -1].remove()

        idx = RAMSES_DSET['Wavelength [nm]'].isin(PRR_DSET['Wavelength [nm]'])
        dset = RAMSES_DSET.loc[idx, :]
        sta = 1

        for i, (clp, clt) in enumerate(zip(
                PRR_DSET.columns, dset.columns)):
            if 'Wavelength [nm]' == clp:
                continue

            s = get_num(numstr=clp)
            ax = axs.flat[i - 1]
            print(f'{i}: {clp} vs. {clt}')

            line = '--' if '4_T1042' in clt else '-'
            ax.plot(PRR_DSET['Wavelength [nm]'], PRR_DSET[clp], 'k', label='PRR-800', lw=4)
            ax.plot(dset['Wavelength [nm]'], dset[clt], line, color='k', label='RAMSES', lw=2)
            ax.set_title(clt[clt.index('(') + 1:-1], color=clr[s]['c'], fontweight='bold')  # , loc='left')
            ax.set_xlim(XLIM)

            if i in (1, 3, 5):
                ax.set_ylabel(r'$\itR$$_{rs}$ [sr$^{-1}$]')
            if i in (5, 6):
                ax.set_xlabel(r'$\lambda$ [nm]')
            if i == 6:
                ax.legend()

            for vl in PRR_DSET['Wavelength [nm]'].values:
                ax.axvline(x=vl, **{'linestyle': ':', 'color': 'gray'})

        ax.set_xlim(400, 700)
        ax.set_ylim(-0.0001, 0.0045)
        ax.set_yticks((0, 0.001, 0.002, 0.003, 0.004))
        fmt = ticker.FormatStrFormatter('%g')
        ax.yaxis.set_major_formatter(fmt)

        save = IMAGE_PATH.joinpath(fig_name)
        plt.savefig(save)
        print(save)
        # ==================================================

    except KeyboardInterrupt:
        sys.exit(0)
