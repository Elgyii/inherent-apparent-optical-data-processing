from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(file: Path):
    name = file.name.split('.ASC')[0]
    sds = pd.read_csv(file, names=['wavelength', name])
    sds = sds.set_index('wavelength')
    return sds


def merge_pd(left: pd, right: pd):
    if left is None:
        return right
    if len(left) == 0:
        return right
    return left.merge(right, how='outer',
                      left_index=True,
                      right_index=True)


def is_number(num: str):
    return num.isdigit()


def get_name(names: list, sta: bool = False):
    keep = []
    append = keep.append
    for col in names:
        k = -1
        while is_number(col[k]):
            k -= 1

        if sta:
            if k == -2:
                append(col[:k + 1])
            if k == -3:
                append(col[:k + 2])
            if k == -4:
                append(col[:k + 3])
            continue
        append(col[:k + 1])
    return np.unique(keep)


def smooth(data, window_length: int = 3, index=None):
    """
    1-D data filtering or smoothing using `savgol_filter`.
    In general `savgol_filter` produces good results compared to a few other methods.
    For more, please check https://docs.scipy.org/doc/scipy/reference/signal.html

    Parameters
    ----------
    data: np.array
        Input 1-D array-like
    index: np.array
    window_length: int

    Returns
    -------
        y: ndarray
            filtered data, same shape as input
    """
    from scipy.signal import savgol_filter

    data = np.asarray(data)
    sf = savgol_filter(data,
                       window_length=window_length,
                       polyorder=1)
    if index is None:
        return sf
    window_length = max(window_length, max(11, window_length))
    st = index.min()
    en = index.max()
    tmp = savgol_filter(sf[st:en],
                        window_length=window_length,
                        polyorder=1)
    sf[st:en] = tmp
    return sf


def strip(columns):
    ret_cols = {}
    for col in columns:
        # stations have (mean, std)
        if 'std' in col:
            continue
        name = col[0] if type(col) == tuple else col
        ret_cols.update({
            name: name.split('_')[1]
        })
    return ret_cols
