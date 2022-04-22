import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as tkf

import pandas as pd
import xlrd

from figure import Figure
from rutils import (loadmat, savemat)

root = tk.Tk()
root.withdraw()


def step2_data_select(data_path: Path = Path()):
    """Step2 Check each data during Observation"""

    dataset = {}
    # lw_band, es_band, ids, ide = 182, 62, 4, 258

    file_path = tkf.askopenfilename(
        filetypes=[("mat file", '*.mat')],
        title='Load step1 data file',
        initialdir=data_path, )

    if file_path in ('', None):
        sys.exit(-1)

    data = loadmat(file=file_path, step=1)
    # ==================================================
    fig_name = file_path.replace(
        'data_Spectrum', 'step2_fig_Spectrum').replace('.mat', '.png')
    select = Figure(dataset=data,
                    filename=Path(fig_name)
                    ).save('step2')
    # print(select)

    # ============
    # Save to file
    # ============
    for key in select.keys():
        sds = data[key.upper()]['data']
        dates = pd.DatetimeIndex(
            [xlrd.xldate_as_datetime(d, 0) for d in sds.iloc[:, 0]]
        )
        idx = dates.isin([p[0][0] for p in select[key]])

        # Create output variable
        dataset.update({
            key.upper(): {'Step2': {'data': sds.loc[idx, :].to_numpy()}},
        })
        print(f'\nKey: {key}\n{"=" * 9}\nSelectDates: {dates[idx]}\n')

    savemat(file=Path(file_path), data=dataset, step=2)
    print(f'File: {file_path}\nStep2: Select data, done!')
    return


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')

    try:
        step2_data_select(data_path=DATA_PATH)
    except KeyboardInterrupt:
        sys.exit(0)
