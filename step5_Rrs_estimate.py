import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as tkf

import numpy as np

import rutils as util
from figure import Figure

root = tk.Tk()
root.withdraw()


def step5_rrs_estimate(data_path: Path = Path()):
    """Step5 Rrs Estimate"""

    file_path = tkf.askopenfilename(
        filetypes=[("mat file", '*.mat')],
        title='Load step4 data file',
        initialdir=data_path, )

    if file_path in ('', None):
        sys.exit(-1)

    data = util.loadmat(file=file_path, step=4)

    rs = 0.0250
    rl = 0.1035
    # RuntimeWarning: Mean of empty slice: All values are nan
    sds = np.squeeze(data['LW1']['data'])
    lw1_mean = np.ma.masked_where(np.isnan(sds), sds)
    if len(lw1_mean.shape) > 1:
        lw1_mean = lw1_mean.mean(axis=0)
    sds = np.squeeze(data['LW2']['data'])
    lw2_mean = np.ma.masked_where(np.isnan(sds), sds)
    if len(lw2_mean.shape) > 1:
        lw2_mean = lw2_mean.mean(axis=0)
    sds = np.squeeze(data['ES']['data'])
    es_mean = np.ma.masked_where(np.isnan(sds), sds)
    if len(es_mean.shape) > 1:
        es_mean = es_mean.mean(axis=0)
    # print(data['LW1']['data'],
    #       data['LW2']['data'],
    #       data['ES']['data'])

    a = lw1_mean / lw2_mean
    a = np.ma.masked_where(a < 0, a)

    ka = np.ma.log(a) / (rl - rs)
    eps = 1.0 - np.ma.exp(-ka * rs)
    lw1_estimate = lw1_mean / (1.0 - eps)
    rrs = lw1_estimate / es_mean
    diff = (lw1_mean - lw2_mean) / lw1_mean

    dataset = {'Step5': {
        'Rrs': rrs,
        'ka': ka,
        'eps': eps,
        'diff': diff,
        'lw1_estimate': lw1_estimate,
        'lw1_mean': lw1_mean,
        'lw2_mean': lw2_mean,
        'es_mean': es_mean,
        'wavelength': data['LW1']['wavelength']}}

    im_data = {'Rrs': rrs,
               'lw1_estimate': lw1_estimate,
               'lw1_mean': lw1_mean,
               'lw2_mean': lw2_mean,
               'wavelength': data['LW1']['wavelength'],
               'es_mean': es_mean}
    # ==================================================
    fig_name = file_path.replace(
        'data_Spectrum', 'step5_fig_Spectrum').replace('.mat', '.png')
    # save_fig(dataset=im_data, step=5, fig_name=fig_name)
    Figure(dataset=im_data,
           filename=Path(fig_name)
           ).save('step5')
    # ============
    # Save to file
    # ============
    util.savemat(file=Path(file_path), data=dataset, step=5)
    print(f'File: {file_path}\nStep5: Rrs estimate, done!')
    return


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')
    try:
        step5_rrs_estimate(data_path=DATA_PATH)

    except KeyboardInterrupt:
        sys.exit(0)
