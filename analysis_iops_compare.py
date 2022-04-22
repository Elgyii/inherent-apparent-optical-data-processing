import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_data(key):
    if key == 'chla':
        file = DATA_PATH.joinpath('chl_202109-Toyama.xlsx')
        df = pd.read_excel(file, sheet_name='chla-comp')
        return pd.DataFrame([
            df['NPEC'].tolist()], index=[key],
            columns=[s.replace(f'sta. {k + 1} (', f'STA. ').strip(')')
                     for k, s in enumerate(df['Sta'].tolist())])

    if key == 'ay':
        file = DATA_PATH.joinpath('ay_absorption_toyama202109.xlsx')
        df = pd.read_excel(file, sheet_name='ay', skiprows=[1, 2],
                           usecols=[0, 1, 3, 5, 7])
        df.rename(columns={'C_STA1': 'STA. 13',
                           'C_STA2': 'STA. 8',
                           'C_STA3': 'STA. 10',
                           'C_STA4': 'STA. 11'}, inplace=True)
        return df

    if key in ('ap', 'ad', 'aph'):
        file = DATA_PATH.joinpath('aph_absorption_toyama202109.xlsx')
        df = pd.read_excel(file, sheet_name=key, skiprows=[1, 2],
                           usecols=[0, 1, 3, 5, 7, 9])
        df.rename(columns={'AP_SP': 'STA. SP',
                           'AP_STA1': 'STA. 13',
                           'AP_STA2': 'STA. 8',
                           'AP_STA3': 'STA. 10',
                           'AP_STA4': 'STA. 11'}, inplace=True)
        return df


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki')
    DATA_PATH = BASE.joinpath('OpticalData')
    IMAGE_PATH = BASE.joinpath('Figures')

    if not IMAGE_PATH.is_dir():
        IMAGE_PATH.mkdir()

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        # plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (14, 9)
        plt.rcParams['font.size'] = 20

        clr = {'4': {'c': '#1f77b4', 'l': '-'},
               '8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#d62728', 'l': '--'},
               '12': {'c': '#2ca02c', 'l': '--'},
               '13': {'c': '#9467bd', 'l': '-'},
               'SP': {'c': 'b', 'l': '-'}, '': '#8c564b'}

        FILE = BASE.joinpath('RrsToyama.xlsx')
        PRR_DSET = pd.read_excel(FILE, sheet_name='PRR800', skiprows=[0])
        PRR_DSET.set_index('Wavelength [nm]', inplace=True)

        # Get all max indices
        ind, vals, cols = [], [], {}
        for col in PRR_DSET.columns:
            # SP and the second sta 10 has ni IOP data
            if ('STA. SP' in col) or \
                    (col == '4_T1049 (STA. 10)'):
                continue
            # Max rrs-band
            idx = PRR_DSET[col] == PRR_DSET[col].max()
            # remove parenthesis
            cols.update({col: col[col.index('(') + 1:-1]})
            ind.append(PRR_DSET.loc[idx, :].index.values[0])
            # vals.append(prr.loc[idx, col].values[0])
        ind.extend([412, 565])
        dfr = pd.DataFrame([], columns=cols.values(),
                           index=np.unique(ind))

        # # Fill the dataframe with max vals
        # for idx, val, col in zip(ind, vals, cols.values()):
        #     dfr.loc[idx, col] = val

        # Fill NaN values in non max index locations
        for idx in dfr.index:
            for name, col in cols.items():
                dfr.loc[idx, col] = PRR_DSET.loc[idx, name]

        # PRR vs RAMSES
        # frames = [dfr, df]
        # pd.concat(frames)
        keys = 'chla', 'ay', 'ad', 'ap', 'aph'
        for j, var in enumerate(keys):
            fig_name = f'fig_PRR_Rrs_vs_iops_{var}.png'
            if var != 'chla':
                continue
            fig, ax = plt.subplots(nrows=1, ncols=4,
                                   sharex='col', sharey='all',
                                   figsize=(20, 10))

            dfi = get_data(key=var)
            for i, idx in enumerate(dfr.index):
                col, x, y = None, [], []
                for col in dfi.columns:
                    x.append(dfi.loc[var, col])
                    y.append(dfr.loc[idx, col])
                    c = clr[f"{col.split('. ')[1]}.1"]['c'] \
                        if '10' in col else clr[col.split('. ')[1]]['c']
                    ax[i].scatter(dfi.loc[var, col],
                                  dfr.loc[idx, col],
                                  s=200, label=col, color=c)
                ix = np.argsort(x)
                ax[i].plot(np.asarray(x)[ix], np.asarray(y)[ix], '-k')
                ax[i].set_title(f'@{idx}')
                if var == 'chla':
                    ax[i].set_xlabel('Chla [mg m$^{-3}$]')
                else:
                    ax[i].set_xlabel(f"{var.replace('a', 'a_{')}}} [m$^{{-1}}$]")
            ax[0].set_ylabel('Rrs [sr$^{-1}$]')
            ax[-1].legend()

            save = IMAGE_PATH.joinpath(fig_name)
            fig.savefig(save)
        plt.show()
        print(fig_name)
        # ==================================================

    except KeyboardInterrupt:
        sys.exit(0)
