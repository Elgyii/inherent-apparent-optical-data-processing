from pathlib import Path

import numpy as np
import pandas as pd
import xlrd
from scipy.interpolate import interp1d, interp2d
import scipy.io as sio


def to_datetime(value):
    if type(value) in (float, np.float32, np.float64):
        return xlrd.xldate_as_datetime(value, 0)
    return [xlrd.xldate_as_datetime(d, 0) for d in value]


def get_meta(filename: Path, attrs: dict = None):
    if attrs is None:
        attrs = {}
    with open(filename, 'r') as txt:
        for r, line in enumerate(txt.readlines()):
            if '=' not in line:
                break
            ky, val = line.split('=')
            attrs.update({
                ky.strip('%').strip(): val.strip('\n').strip()
            })
    print(attrs)
    return attrs


def get_attr(filename: Path, att_name: str):
    with open(filename, 'r') as txt:
        for r, line in enumerate(txt.readlines()):
            if att_name in line:
                break
    return {att_name: r}


def skip(val: str):
    if val not in ('LW1', 'LW2', 'ES'):
        return False
    return True


def interp_1d(xi, yi, xq, axis: int = -1):
    """
        Interpolate a 1-D function.
        x and y are arrays of values used to approximate some function f: y = f(x).
        This class returns a function whose call method uses interpolation to find the value of new points.

    """
    xi = np.asarray(xi)
    yi = np.asarray(yi)
    # xq = np.asarray(xq)
    # print(xi.shape, yi.shape)

    f = interp1d(xi, yi
                 , fill_value=(np.nan, np.nan)
                 , bounds_error=False
                 , axis=axis)
    return f(xq)


def interp_2d(xi, yi, zi, xq, yq):
    """
        Interpolate over a 2-D grid.
        x, y and z are arrays of values used to approximate some function f: z = f(x, y) which
        returns a scalar value z. This class returns a function whose call method uses spline
        interpolation to find the value of new points.
    """
    xi = np.asarray(xi)
    yi = np.asarray(yi)

    if len(xi.shape) == 1:
        shape = yi.shape

        if xi.size == shape[0]:
            xi = np.repeat(xi.reshape(-1, 1), shape[1], axis=1)

        if xi.size == shape[1]:
            xi = np.repeat(xi.reshape(1, -1), shape[0], axis=0)

    f = interp2d(xi, yi, zi
                 , fill_value=np.nan
                 , kind='linear')
    return f(xq, yq)


def savemat(file: Path, data: dict = None, step: int = 1):
    """
    Handles i-o of mat-file.
    Can save, read or update existing mat-file.
    Loads key-variable from mat-file defined by file

    Parameters
    ----------
    file: str
        String specifier of the file location
    data: dict
        Input dictionary of data in case of saving
    step: int

    Returns
    -------
    sds: dict
        Either a bool (if load and file does not exist) or a dictionary of saved/loaded data
    """
    if data is None:
        return
    if step == 1:
        sio.savemat(file_name=file, mdict=data, appendmat=True)
        return

    if file.is_file():
        sds = loadmat(file=str(file))
        contents = {}

        for key in filter(skip, sds.keys()):

            for s in range(1, step):
                if (s == 1) and (step == 5):
                    contents.update({
                        key: {'columns': sds[key].columns,
                              'lambda0': sds[key].lambda0,
                              'lambda1': sds[key].lambda1,
                              'time': sds[key].time,
                              **{f'Step{s}': getattr(sds[key], f'Step{s}')}},
                        **data['Step5']
                    })
                else:
                    if s == 1:
                        try:
                            contents.update({
                                key: {'columns': sds[key].columns,
                                      'lambda0': sds[key].lambda0,
                                      'lambda1': sds[key].lambda1,
                                      'time': sds[key].time,
                                      **{f'Step{s}': getattr(sds[key], f'Step{s}')},
                                      **data[key]}
                            })
                        except AttributeError:
                            contents.update({
                                key: {'columns': sds[key].columns,
                                      'lambda0': sds[key].lambda0,
                                      **{f'Step{s}': getattr(sds[key], f'Step{s}')},
                                      **data[key]}
                            })
                    else:
                        contents.update({
                            key: {**contents[key],
                                  **{f'Step{s}': getattr(sds[key], f'Step{s}')}}
                        })
        sio.savemat(file_name=file, mdict=contents, appendmat=True)
        return contents

    sio.savemat(file_name=file, mdict=data, appendmat=True)
    return data


def loadmat(file: str, key: str = None, step: int = None):
    """
    Loads key-variable from mat-file defined by file

    Parameters
    ----------
    file: str
        String specifier of the file location
    key: str
        String specifier of the key being read
    step: int

    Returns
    -------
    sds: dict
        loaded key array
    """
    if key:
        sds = sio.loadmat(file,
                          variable_names=[key],
                          squeeze_me=True).get(key)
        sds = np.where(sds > 10, np.nan, sds)
        return sds

    sds = sio.loadmat(file,
                      struct_as_record=False,
                      squeeze_me=True)
    if step:
        data, keys = {}, sds.keys()
        for key in filter(skip, keys):
            temp = getattr(sds[key], f"Step{step}").data
            if len(temp.shape) == 1:
                temp = temp.reshape(1, -1)
            try:
                data.update({
                    key: {'data': pd.DataFrame(
                        data=temp,
                        columns=sds[key].columns),
                        'wavelength': sds[key].lambda0}
                })
            except ValueError:
                # Shape problem. Original and interpolated
                cols = [f'c{i:03}' for i in range(temp.shape[1])]
                wavelength = sds[key].lambda1
                tm = sds[key].time
                data.update({key: {
                    'data': pd.DataFrame(data=temp, columns=cols),
                    'wavelength': wavelength,
                    'time': tm}
                })
        return data
    return sds
