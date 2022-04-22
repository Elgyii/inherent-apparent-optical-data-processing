import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as tkf

import numpy as np

import rutils as util
from figure import Figure

root = tk.Tk()
root.withdraw()


def step4_data_calibrate(data_path: Path = Path()):
    """Step4 Calculating mean Es data and Calibrating Lw data"""

    lw_band, es_band, ids, ide = 182, 62, 4, 258
    dataset, im_data = {}, {}
    file_path = tkf.askopenfilename(
        filetypes=[("mat file", '*.mat')]
        , title='Load step3 data file'
        , initialdir=data_path, )

    if file_path in ('', None):
        sys.exit(-1)

    data = util.loadmat(file=file_path, step=3)
    step2 = util.loadmat(file=file_path, step=2)

    # band1 = band1_Es = 556 # [nm]
    # band2 = band2_Es = 900 # [nm]

    es_mean = np.asarray(data['ES']['data'].mean(axis=0))

    for key in filter(util.skip, data.keys()):
        im_temp = {key: []}

        temp = data[key]['data']
        t = data[key]['time']
        if type(t) == float:
            t = [t]
        m = temp.shape[0]

        if key == 'ES':
            dataset.update({
                key: {'Step4': {'data': temp.to_numpy()}}
            })
            continue

        for j in range(m):
            # ES data at the time of LW
            temp.iloc[j, :] = data[key]['data'].iloc[j, :] * es_mean / util.interp_1d(
                xi=data['ES']['time'],
                yi=data['ES']['data'],
                xq=t[j],
                axis=0
            )

        im_data.update({
            key: {'time': t,
                  'x': step2[key]['wavelength'][ids:ide],
                  'y': step2[key]['data'].iloc[:, ids:ide].to_numpy(),
                  'xi': data[key]['wavelength'],
                  'yi': temp.to_numpy()}
        })

        dataset.update({
            key: {'Step4': {'data': temp.to_numpy()}}
        })

    # ==================================================
    fig_name = file_path.replace(
        'data_Spectrum', 'step4_fig_Spectrum').replace('.mat', '.png')
    # save_fig(dataset=im_data, step=4, fig_name=fig_name)
    Figure(dataset=im_data,
           filename=Path(fig_name)
           ).save('step4')
    # ============
    # Save to file
    # ============
    util.savemat(file=Path(file_path), data=dataset, step=4)
    print(f'File: {file_path}\nStep4: Calibrate data, done!')
    return


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')

    try:
        step4_data_calibrate(data_path=DATA_PATH)
    except KeyboardInterrupt:
        sys.exit(0)
