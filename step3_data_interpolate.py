import sys
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import filedialog as tkf

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import rutils as utils
from figure import Figure

root = tk.Tk()
root.withdraw()


def callback(event):
    plt.close()
    return False


def smooth_ui(event, tdata, wavelength, ax):
    """
    1-D data filtering or smoothing using `savgol_filter`.
    In general `savgol_filter` produces good results compared to a few other methods.
    For more, please check https://docs.scipy.org/doc/scipy/reference/signal.html

    Parameters
    ----------
    wavelength:
    ax:
    event:
    tdata: np.array
        Input 1-D array-like
    window_length: int
        The length of the filter window (i.e., the number of coefficients).
        Must be a positive odd integer.
    polyorder: int
        The order of the polynomial used to fit the samples. Must be less than window_length.
    disp: str
        Whether to visualise the results or not.
        Specified as either being 'x' or 'y' to indicate how input array is treated in the visualization

    Returns
    -------
        y: ndarray
            filtered data, same shape as input
    """
    from scipy.signal import savgol_filter

    res = savgol_filter(tdata,
                        window_length=3,
                        polyorder=1)
    ax.plot(wavelength, tdata, label='savgol_filter')
    ax.legend()
    tdata[:] = res

    plt.close()
    return True


def step3_data_interpolate(data_path: Path = Path()):
    """Step3 Removing spike noise and Interpolating every 1nm"""

    dataset = {}
    lw_band, es_band, ids, ide = 182, 62, 4, 258
    button_colour, hover_yes, hover_no = "gray", 'g', "r"

    file_path = tkf.askopenfilename(
        filetypes=[("mat file", '*.mat')]
        , title='Load step2 data file'
        , initialdir=data_path, )

    if file_path in ('', None):
        sys.exit(-1)

    sds = utils.loadmat(file=file_path, step=2)

    wl = range(310, 1001)

    for key in sds.keys():

        data = sds[key]['data']
        wavelength = sds[key]['wavelength'][ids:ide]
        shape = data.shape[0], len(wl)
        temp_array, t = np.empty(shape=shape), []

        for i in range(data.shape[0]):
            tdata = data.iloc[i, ids:ide]
            if key in ('LW1', 'LW2'):
                fig, ax = plt.subplots()
                ax.plot(wavelength, tdata, label='original')
                ax.set_title(f'{key}\nclick YES to apply a smoothing '
                             f'filter, NO to keep original')
                func = partial(smooth_ui, wavelength=wavelength, ax=ax, tdata=tdata)

                ybx = plt.axes([0.75, 0.8, 0.05, 0.06])
                nbx = plt.axes([0.80, 0.8, 0.05, 0.06])
                yes = Button(ybx, 'YES'
                             , color=button_colour
                             , hovercolor=hover_yes)
                no = Button(nbx, 'NO'
                            , color=button_colour
                            , hovercolor=hover_no)
                yes.on_clicked(func=func)
                no.on_clicked(callback)
                plt.show()

            # remove spike noize
            dif = np.concatenate(([0], np.diff(tdata))).astype(np.float32)
            dif = np.ma.masked_where(np.isnan(dif), dif)
            sign = np.concatenate((dif[1:] * dif[:-1], [0]))
            sign[np.isnan(sign)] = 0
            negative = sign < 0
            percent_larger = dif > np.nanmax(tdata.iloc[35:96]) * 0.1

            idx = negative & percent_larger

            ydata = tdata.loc[~idx]
            xdata = wavelength[~idx]

            # if key == 'LW1':
            #     print(min(xdata), max(xdata))
            print(f'Key: {key:3} | Data: {i} | '
                  f'OShape: {ydata.shape} '
                  f'| IDX: {all(idx)} {wavelength.min(), wavelength.max()}')

            # interpolation in 350-900 nm
            interp_data = utils.interp_1d(xi=xdata, yi=ydata, xq=wl)
            temp_array[i, :] = interp_data
            t.append(data.iloc[i, 0])

        print('=' * 30)
        dataset.update({
            key: {'time': t,
                  'x': wavelength,
                  'y': data.iloc[:, ids:ide].to_numpy(),
                  'xi': wl,
                  'yi': temp_array}
        })
    # ==================================================
    fig_name = file_path.replace(
        'data_Spectrum', 'step3_fig_Spectrum').replace('.mat', '.png')
    # save_fig(dataset=dataset, step=3, fig_name=fig_name)
    Figure(dataset=dataset,
           filename=Path(fig_name)
           ).save('step3')
    # ============
    # Save to file
    # ============
    save = {}
    for key in dataset.keys():
        # Create output variable
        save.update({
            key: {'Step3': {'data': dataset[key]['yi']},
                  'time': dataset[key]['time'],
                  'lambda1': dataset[key]['xi']}
        })
    utils.savemat(file=Path(file_path), data=save, step=3)
    print(f'File: {file_path}\nStep3: Interpolate data, done!')
    return


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')

    try:
        step3_data_interpolate(data_path=DATA_PATH)
    except KeyboardInterrupt:
        sys.exit(0)
