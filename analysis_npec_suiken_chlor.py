import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    DATA_PATH = BASE.joinpath(r'OpticalData\RAMSES')
    IMAGE_PATH = BASE.joinpath('Figures')
    RRS_FILE = DATA_PATH.joinpath('data_Spectrum_Calibrated_2021-09-29_08-21-06_671.xlsx')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (11, 9)
        plt.rcParams['font.size'] = 20

        fig, ax = plt.subplots()

        CHL_FILE = DATA_PATH.parent.joinpath('chl_202109-Toyama.xlsx')
        CHL_DF = pd.read_excel(CHL_FILE, sheet_name='chla-comp')

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

        # [s.replace(f'sta. {k+1} (', f'Sta. ').strip(')')
        #                                      for k, s in enumerate(df['Sta'].tolist())]

        for j, (st, pn, mz) in enumerate(zip(CHL_DF['Sta'],
                                             CHL_DF['NPEC'],
                                             CHL_DF['MIZUKEN'])):
            s = ''.join(re.findall(r'(\d+)', st))
            # c = clr[f"{s}.1"]['c'] \
            #     if '10' in st else clr[s]['c']
            if np.isnan(mz):
                continue
            print(st, s)
            sta = 'SP' if s == '12' else s
            ax.plot(pn, mz, marker='o', markersize=20, color=clr[sta]['c'],
                    label=st.replace(f'sta. {j + 1} (', f'STA. ').strip(')'))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.plot([0, 10], [0, 10], '-k')
        ax.set_xlabel('CHL$_{NPEC}$ [mg m$^{-3}$]')
        ax.set_ylabel(u'CHL$_{SUIKEN}$ [mg m$^{-3}$]')
        ax.legend()

        fig_name = 'fig_chla_npec_mizuken.png'
        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        save = IMAGE_PATH.joinpath(fig_name)
        fig.savefig(save)
        # plt.show()
        print(save)
        # ==================================================
        # Location String
        #
        # Location Code
        #
        # 'best'            0
        # 'upper right'     1
        # 'upper left'      2
        # 'lower left'      3
        # 'lower right'     4
        # 'right'           5
        # 'center left'     6
        # 'center right'    7
        # 'lower center'    8
        # 'upper center'    9
        # 'center'          10

    except KeyboardInterrupt:
        sys.exit(0)
