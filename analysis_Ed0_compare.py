import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd
from dateutil.parser import parse
from matplotlib import pyplot as plt

from rutils import loadmat


def keep(file):
    if 'KdRrs.csv' in file.name:
        return False
    if file.is_file() and file.name.endswith('.xlsx'):
        return False
    if file.name in ('RAW-CAL', 'Air'):
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


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    DATA_PATH = BASE.joinpath('OpticalData')
    IMAGE_PATH = BASE.joinpath('Figures')
    PRR_PATH = BASE.joinpath(r'OpticalData\PRR')
    RAMSES_PATH = BASE.joinpath(r'OpticalData\RAMSES')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (11, 9)
        plt.rcParams['font.size'] = 20

        clr = {'8': {'c': '#1f77b4', 'l': '-'},
               '10.1': {'c': '#ff7f0e', 'l': '-'},
               '10.2': {'c': '#8c564b', 'l': '--'},
               '11': {'c': '#2ca02c', 'l': '--'},
               '13': {'c': '#d62728', 'l': '-'},
               'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

        # PRR data µW/(cm²nm)
        dataset, prr_time = pd.DataFrame([]), []

        for i, f in enumerate(filter(keep, PRR_PATH.glob('*.csv'))):
            print(f'{f.parent.name}: {f.name}')
            sds = pd.read_csv(f, usecols=[0, 17] + list(range(38, 51)),
                              skiprows=[1], parse_dates=True)
            dta = get_ed0(data=sds)

            mean = dta.mean(axis=0).to_frame()
            prr_time.append(parse(dta['Time/Date'].at[0]))
            t = parse(dta['Time/Date'].at[0]).strftime('%H%M%S_PRR')
            vals = (mean.values / 1000) * 10000  # mW/(m²nm)
            tmp = pd.DataFrame(vals, columns=[t], index=mean.index)
            if i == 0:
                dataset = tmp
                continue
            dataset = pd.merge(left=dataset, right=tmp, left_index=True, right_index=True)

        idx = dataset.index
        bands = [int(j.split('_')[1]) for j in idx]
        print(f'\n{bands}\n')

        # RAMSES mW/(m^2 nm)
        t_diff, i = [], 0
        for p in filter(keep, RAMSES_PATH.iterdir()):

            f = [f for f in p.glob('data_Spectrum_Calibrated_*.mat')]
            if len(f) == 0:
                continue
            print(f'{p.name}: {f[0].name, i}')

            sds = loadmat(file=str(f[0]))
            tmp = sds['es_mean']
            wvl = sds['wavelength']

            trios_t = xlrd.xldate_as_datetime(
                sds['ES'].time[0], 0
            )
            t_diff.append(prr_time[i] - trios_t)
            print(f'PRR: {prr_time[i]} | RAMESES: {trios_t} | Diff: {t_diff[-1]}\n')
            i += 1

            trios = []
            for band in bands:
                ix = wvl == band
                trios.append(tmp[ix][0])

            tmp = pd.DataFrame(trios, columns=[f'{trios_t.strftime("%H%M%S")}_TRIOS'], index=idx)
            dataset = pd.merge(left=dataset, right=tmp, left_index=True, right_index=True)

        print(f'\n{dataset}')
        # ==================================================
        fig_name = IMAGE_PATH.joinpath('fig_PRR_RAMSES_Ed0.png')
        fig, ax = plt.subplots()
        labels = {'092954_TRIOS': 'STA. 8', '101405_TRIOS': 'STA. 10',
                  '104232_TRIOS': 'STA. 10', '105548_TRIOS': 'STA. 11',
                  '082109_TRIOS': 'STA. 13', '085012_TRIOS': 'STA. SP'}
        # vals = sorted([val.split('.')[0] for val in labels.values()])
        cols = sorted(dataset.columns)
        # cols = [labels[col] for col in dataset.columns]
        sta = 1

        for k, col in enumerate(cols):
            if 'PRR' in col:
                continue
            t = f'{t_diff[int(k / 2)]}'.split('.')[0]
            s = labels[col].split('. ')[1]
            if s == '10':
                s = f'{s}.{sta}'
                sta += 1
            print(f'{cols[k + 1]} vs. {col}')
            ax.scatter(dataset[cols[k + 1]], dataset[col], s=200,
                       alpha=0.5, edgecolors='k',
                       label=f'{labels[col]}, TDiff: {t}',
                       color=clr[s]['c'],
                       )
        ax.plot([100, 1000], [100, 1000], '-k')
        ax.legend()
        ax.set_xlim([100, 1000])
        ax.set_xlabel('PRR-800 Ed$_0$ [mW m$^{-2}$ nm$^{-1}$]')
        ax.set_ylim([100, 1000])
        ax.set_ylabel('RAMSES Ed$_0$ [mW m$^{-2}$ nm$^{-1}$]')
        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        plt.savefig(fig_name)
        # plt.show()
        print(fig_name)
        # ==================================================
        filename = fig_name.name.replace('.png', '.xlsx').replace('fig', 'data')
        filename = RAMSES_PATH.joinpath(filename)
        if not filename.is_file():
            dataset.to_excel(filename, sheet_name='Ed0', header=True, index=True)

    except KeyboardInterrupt:
        sys.exit(0)
