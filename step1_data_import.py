import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as tkf

import pandas as pd

from figure import Figure
from rutils import savemat, get_meta, get_attr

root = tk.Tk()
root.withdraw()


def step1_data_import(data_path: Path = Path()):
    """Import RAMSES data files"""
    dataset, dset = {}, {}
    file_path = Path('.').absolute()
    # LW_BAND, ES_BAND, IDS, IDE = 182, 62, 4, 258

    for sen in range(3):
        if sen == 1:
            window_title, sds = 'Select Lw2 (with dome) data file', 'LW2'
        elif sen == 2:
            window_title, sds = 'Select Es (surface reference) data file', 'ES'
        else:
            window_title, sds = 'Select Lw1 (no dome) data file', 'LW1'

        file_path = tkf.askopenfilename(
            filetypes=[("text file", '*.mlb')]
            , title=window_title
            , initialdir=data_path, )

        if file_path in ('', None):
            sys.exit(-1)

        meta = get_meta(filename=Path(file_path))
        attr = get_attr(filename=Path(file_path)
                        , att_name='%DateTime')

        df = pd.read_csv(
            file_path
            , delim_whitespace=True
            , na_values=['-NAN', '%;;;']
            , skiprows=attr[list(attr.keys())[0]])
        df.rename(columns={col: col.strip('%').strip()
                           for col in df.columns},
                  inplace=True)

        col = 'IntegrationTime'
        idx = df.loc[:, col].isin([int(meta[col])])
        # Keep wavelength row
        # idx.iloc[0] = True

        # Create output variable
        dataset.update({
            sds: {'Step1': {'meta': meta,
                            # 'data': df.loc[idx, :]},
                            'data': df.iloc[1:, :]},
                  'columns': list(df.columns),
                  'lambda0': df.iloc[0, :].to_numpy()}
        })
        dset.update({
            sds: {'data': df.iloc[1:, :],
                  'wavelength': df.iloc[0, :].to_numpy()}
        })
    # =================================
    fig_name = file_path[:file_path.index('_SAM_')]
    fig_name = fig_name.replace('_Spectrum', 'step1_fig_Spectrum')

    Figure(
        dataset=dset,
        filename=Path(fig_name),
        start_index=1
    ).save(case='step1')

    # ============
    # Save to file
    # ============
    for key in dataset.keys():
        dataset[key]['Step1']['data'] = dataset[key]['Step1']['data'].to_numpy()
    save = f"{file_path[:file_path.index('_SAM_')]}.mat"
    save = Path(save.replace('_Spectrum', 'data_Spectrum'))
    savemat(file=save, data=dataset)

    print(f'DataFile: {save}\nStep1: Data import done...!!')
    return


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')
    try:
        step1_data_import(data_path=DATA_PATH)
    except KeyboardInterrupt:
        sys.exit(0)
    except InterruptedError:
        sys.exit(0)
