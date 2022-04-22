import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt, ticker

from rutils import loadmat


def to_excel():
    return


def keep(filename: Path):
    if 'KdRrs.csv' in filename.name:
        return False
    if filename.is_file() and filename.name.endswith('.xlsx'):
        return False
    if filename.name in ('RAW-CAL', 'Air'):
        return False
    return True


def step6_save_rrs(data_path: Path = Path(),
                   ratio: bool = False,
                   lambda0: int = 555,
                   prr_file: Path = None,
                   clr: dict = None,
                   image_file: Path = None,
                   excel_file: Path=None,
                   columns: dict = None):
    font_size = 20
    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['figure.figsize'] = (16, 11)  # (12, 7)  # (14, 9)
    plt.rcParams['font.size'] = font_size

    files = []
    for p in filter(keep, data_path.iterdir()):

        f = [f for f in p.glob('data_Spectrum_Calibrated_*.mat')]
        if len(f) == 0:
            continue
        print(f'{p.name}: {f[0].name}')
        files.append(f[0])

    if len(files) == 0:
        sys.exit(-1)

    sds = loadmat(file=str(files[0]), step=4)

    wl = sds['LW1']['wavelength']
    dataset = np.ones((wl.size, len(files) + 1)) * np.nan
    dataset[:, 0] = wl
    cols = ['Wavelength [nm]']

    # idx = (wl > 401) & (wl < 689)
    # idx = (wl > 349) & (wl < 751)
    # mnx, mxx = 340, 760
    idx = (wl > 300) & (wl < 1000)
    mnx, mxx = 300, 950
    idr = wl == lambda0
    fig, ax = plt.subplots()
    ymx = 0.0001

    # cycler(color='cmykbgr')
    cc = (plt.rcParams['axes.prop_cycle'] *
          cycler(linestyle=['-', '-.', '--']))

    for i, (f, opts) in enumerate(zip(files, cc)):
        print(opts)
        data = loadmat(file=str(f), key='Rrs')
        dataset[:, i + 1] = data
        name = f.parent.name
        cols.append(name)

        if clr is None:
            if ratio:
                ax.plot(wl[idx], data[idx] / data[idr],
                        label=name, lw=3)
            else:
                ax.plot(wl[idx],
                        data[idx],
                        label=name,
                        lw=3,
                        **opts)
        else:
            res = re.findall(r'(\(.*)(8|\d[0-3])\)', name)
            if len(res) == 0:
                res = re.findall(r'(\(.*) (.*)\)', name)

            s = ''.join(res[0][1])
            c, ls = clr[s]['c'], clr[s]['l']
            lw = 3  # 9 if ('STA. SP' in name) or ('STA. 10' in name) else 3
            if ratio:
                ax.plot(wl[idx], data[idx] / data[idr], ls, color=c,
                        label=name[name.index('(') + 1:-1], lw=lw)
            else:
                ax.plot(wl[idx], data[idx], ls, color=c,
                        label=name[name.index('(') + 1:-1], lw=lw)
        ymx = max(ymx, np.nanmax(data[idx]))
    dataset = pd.DataFrame(data=dataset,
                           columns=cols)
    dataset.set_index('Wavelength [nm]', inplace=True)
    print(dataset)

    # print(columns, cols)
    # dataset.rename(columns=columns, inplace=True)
    # ==================================================
    # fig_Spectrum_Calibrated_2021-09-29_08-21-06_671
    ax.set_xlabel(r'$\lambda$ [nm]', size=font_size + 20)
    if ratio:
        if image_file is None:
            image_file = IMAGE_PATH.joinpath(
                files[0].name.replace(
                    'data_Spectrum', f'fig_Spectrum_Rrs_ratio{lambda0}'
                ).replace('.mat', '.png'))
        ax.set_ylabel(r'$\it{R}\rm(\lambda)_{rs}/\it{R}\rm_{rs}$' + f'({lambda0})',
                      size=font_size + 20)
    else:
        if image_file is None:
            image_file = IMAGE_PATH.joinpath(
                files[0].name.replace(
                    'data_Spectrum', f'fig_Spectrum_step6_Rrs'
                ).replace('.mat', '.png'))
        ax.set_ylabel(r'$\itR\rm_{rs}$ [sr$^{-1}$]', size=font_size + 20)
        if ymx < 0.005:
            ax.set_ylim(-0.0001, 0.0045)
            ax.set_yticks((0, 0.001, 0.002, 0.003, 0.004))
        else:
            dec = len(f'{ymx}') - 1
            ax.set_ylim(-0.0001, ymx + eval(f'1e-{dec}/2'))
            # ax.set_yticks((0, 0.001, 0.002, 0.003, 0.004))
    ax.set_xlim(mnx, mxx)
    fmt = ticker.FormatStrFormatter('%g')
    ax.yaxis.set_major_formatter(fmt)
    ax.legend()

    fig.savefig(image_file)
    print(f'Image: {image_file}')

    if excel_file is None:
        excel_file = data_path.joinpath(
            files[0].name.replace('.mat', '.xlsx'))
    if prr_file is None:
        with pd.ExcelWriter(excel_file) as writer:
            dataset.to_excel(
                writer,
                sheet_name='RAMSES_Rrs',
                header=True, index=True)
            print(f'ExcelData: {excel_file}')
    if prr_file:
        join_prr(ramses_dset=dset, prr_dset=PRR_DSET,
                 ramses_file=excel_file, prr_file=prr_file)
    return dataset, ''


def join_prr(prr_dset: pd, prr_file: Path, ramses_dset: pd, ramses_file: Path):
    # ==================================================
    # filename = files[0].name.replace('.mat', '.xlsx')
    # ==================================================
    # ADD PRR Rrs to RAMSES File
    data = pd.read_excel(io=prr_file,
                         sheet_name='PRR800',
                         skiprows=[0, 2, 9, 10, 11, 12, 13, 14, 15],
                         usecols=[0, 11, 32, 33] + list(range(13, 19)) + list(range(22, 27)))
    data.reset_index(drop=True, inplace=True)

    names = [f'{c + 1}_T{name.split("_")[-1]} (STA. {sta})'
             for c, (name, sta) in enumerate(zip(data['PRR-File'], data['Station']))]

    bands = np.asarray([int(col.split('_')[-1])
                        for col in data.columns if 'Rrs' in col])
    idx = np.argsort(bands)

    index = pd.Index(bands[idx], name='Wavelength [nm]')
    data.drop(columns=['Station', 'PRR-File'], inplace=True)

    for i in range(data.shape[0]):
        tmp = pd.DataFrame(data.iloc[i, :].values[idx],
                           columns=[names[i]], index=index)
        if i == 0:
            prr_dset = tmp
            continue
        prr_dset = pd.merge(left=prr_dset, right=tmp,
                            left_index=True, right_index=True)
    prr_dset.reset_index(drop=False, inplace=True)

    # print(prr_dset, names, cols, sep='\n')
    if not ramses_file.is_file():
        with pd.ExcelWriter(ramses_file) as writer:
            ramses_dset.to_excel(writer, sheet_name='RAMSES_Rrs', header=True, index=False)
            prr_dset.to_excel(writer, sheet_name='PRR_Rrs', header=True, index=False)
    print(f'ExcelData: {ramses_file}')
    pass


if __name__ == '__main__':
    ADD_PRR2RAMSES_FILE = False
    CRUISE = '202204'  # '202109'
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')
    IMAGE_PATH = HOME_DIR.joinpath('Figures')
    COLUMNS = CLR = None
    # CLR = {'8': {'c': 'k', 'l': '-'},
    #        '10': {'c': '#ff7f0e', 'l': '-'},
    #        '11': {'c': '#2ca02c', 'l': '-'},
    #        '12': {'c': 'b', 'l': '-'},
    #        '13': {'c': 'r', 'l': '-'},
    #        'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}
    #
    # COLUMNS = {'1_T0821': '1_T0821 (STA. 13)',
    #            '2_T0849': '2_T0849 (STA. SP)',
    #            '3_T0929': '3_T0929 (STA. 8)',
    #            '4_T1013': '4_T1013 (STA. 10)',
    #            '5_T1042': '5_T1042 (STA. 11)',
    #            '6_T1055': '6_T1055 (STA. 12)'}

    PRR_FILE = DATA_PATH.joinpath('Rrs_Toyama.2021.PRR800.xlsx')
    PRR_DSET = pd.DataFrame()

    try:
        dset, file = step6_save_rrs(data_path=DATA_PATH, clr=CLR, columns=COLUMNS)
        if ADD_PRR2RAMSES_FILE:
            join_prr(ramses_dset=dset, prr_dset=PRR_DSET,
                     ramses_file=file, prr_file=PRR_FILE)

    except KeyboardInterrupt:
        sys.exit(0)
