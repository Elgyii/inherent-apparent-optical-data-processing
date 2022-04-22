import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker

if __name__ == '__main__':
    HOME_DIR = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki')
    DATA_PATH = HOME_DIR.joinpath('OpticalData')
    IMAGE_PATH = HOME_DIR.joinpath(r'202204\Figures')

    if not IMAGE_PATH.is_dir():
        IMAGE_PATH.mkdir()
    FONT_SIZE = 20

    try:
        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        # plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (16, 11)
        plt.rcParams['font.size'] = FONT_SIZE

        clr = {'4': {'c': '#1f77b4', 'l': '-'},
               '8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#d62728', 'l': '--'},
               '12': {'c': '#2ca02c', 'l': '--'},
               '13': {'c': '#9467bd', 'l': '-'},
               'SP': {'c': 'b', 'l': '-'}, '': '#8c564b'}

        FILE = HOME_DIR.joinpath('RrsToyama.xlsx')
        PRR_DSET = pd.read_excel(FILE, sheet_name='PRR800', skiprows=[0])
        # PRR Rrs image
        FIG_NAME = f'fig_analysis_Rrs_PRR_{datetime.today().strftime("%Y%j")}.png'
        fig, ax = plt.subplots()

        start = min([i for i, col in enumerate(PRR_DSET.columns)
                     if 'Rrs' in col])
        df = PRR_DSET.iloc[:, start:]
        # # sorted
        # idx = np.argsort(df.iloc[0, :])
        # df = df.iloc[:, idx]
        # unique sorted
        x, idx = np.unique(df.iloc[0, :],
                           return_index=True)
        df = df.iloc[:, idx]
        df.insert(0, 'station', PRR_DSET['station'])
        df.insert(1, 'date', PRR_DSET['date'])
        df.insert(2, 'time', PRR_DSET['time'])
        stas = PRR_DSET.iloc[1:, 0]

        lines = ['--', '-']
        uns = np.unique([f'{s}' for s in stas])

        XLIM = (x.min() - 10,
                x.max() + 10)
        mxy = 0

        for i, st in enumerate(uns):
            st = int(st) if st != 'SP' else st
            loc = df['station'].isin([st])
            dfi = df.loc[loc, :].reset_index()

            for r, row in dfi.iterrows():
                print(type(row['date']), type(row['time']))
                label = f"STA. {st}: {row['date'].strftime('%Y%m')}" \
                        f"T{row['time'].strftime('%H%M')}"
                line = '-' if '2022' in label else '--'

                # print(f'{r}: {label}')
                y = row.iloc[4:]
                mxy = max([mxy, y.max()])
                ax.plot(x, y, line,
                        color=clr[f'{st}']['c'],
                        label=label,
                        lw=3)

        for vl in x:
            ax.axvline(x=vl, **dict(linestyle=':', color='gray'))

        ax.set_ylabel(r'$\itR$$_{rs}$ [sr$^{-1}$]',
                      size=FONT_SIZE + 20)
        ax.set_xlabel(r'$\lambda$ [nm]',
                      size=FONT_SIZE + 20)
        ax.set_xlim(*XLIM)
        # ax.set_xlim(400, 700)
        ax.set_ylim(-0.0001, mxy + 0.001)

        # ax.set_yticks((0, 0.001, 0.002, 0.003, 0.004))
        fmt = ticker.FormatStrFormatter('%g')
        ax.yaxis.set_major_formatter(fmt)
        # ax.legend(loc='right', bbox_to_anchor=[1.4, 0.5])
        ax.legend()

        save = IMAGE_PATH.joinpath(FIG_NAME)
        fig.savefig(save)
        # plt.show()
        plt.close(fig)
        print(f'PRR800: {save}')
        # ======================

    except KeyboardInterrupt:
        sys.exit(0)
