import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as tkf

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

import rutils as util
from autils import smooth
from figure import Figure
from step6_save_Rrs import step6_save_rrs

root = tk.Tk()
root.withdraw()


class GetRrs(Figure):
    lw_band, es_band, ids, ide = 182, 62, 4, 258
    button_colour, hover_yes, hover_no = "gray", 'g', "r"
    fc = 'lightgoldenrodyellow'

    def __init__(self, data_path: Path):
        super().__init__(dataset=None, filename=data_path)
        self.no = None
        self.yes = None
        self.close = None
        self.path = data_path
        self.spectra = None
        self.wavelength = None
        self.ax = None

    def fclose(self, event):
        plt.close(self.fig)
        return 0

    def ui(self, event):
        """
        1-D data filtering or smoothing using `savgol_filter`.
        In general `savgol_filter` produces good results compared to a few other methods.
        For more, please check https://docs.scipy.org/doc/scipy/reference/signal.html

        Returns
        -------
            y: ndarray
                filtered data, same shape as input
        """
        idx, = np.where(self.wavelength >= 700)
        res = smooth(data=self.spectra, index=idx)
        self.ax.plot(self.wavelength, res, label='savgol_filter')
        self.ax.legend()
        plt.draw()
        self.spectra[:] = res
        return 0

    def get_button(self, bbox, label):
        hover_color = self.hover_no
        if label == 'yes':
            hover_color = self.hover_yes
        return Button(bbox,
                      label=label,
                      color=self.button_colour,
                      hovercolor=hover_color)

    def get_ui(self, key: str):
        self.ax.plot(self.wavelength, self.spectra,
                     label='original')
        self.ax.set_title(f'{key}\nclick YES to apply a smoothing '
                          f'filter, NO to keep original')
        labels = ('yes', 'no', 'close')
        for i, label in enumerate(labels):
            h = 0.05 * i
            ax = plt.axes([0.70, 0.80 - h, 0.07, 0.05],
                          facecolor=self.fc)
            but = self.get_button(bbox=ax, label=label)
            setattr(self, label, but)

        self.no.on_clicked(self.fclose)
        self.yes.on_clicked(self.ui)
        self.close.on_clicked(self.fclose)
        plt.show()
        return 0

    def data_import(self):
        """Import RAMSES data files"""
        dataset = {}
        file_path = Path('.').absolute()
        self.dset = {}
        # LW_BAND, ES_BAND, IDS, IDE = 182, 62, 4, 258

        for sen in range(3):
            if sen == 1:
                window_title, name = 'Select Lw2 (with dome) data file', 'LW2'
            elif sen == 2:
                window_title, name = 'Select Es (surface reference) data file', 'ES'
            else:
                window_title, name = 'Select Lw1 (no dome) data file', 'LW1'

            file_path = tkf.askopenfilename(
                filetypes=[("text file", '*.mlb')]
                , title=window_title
                , initialdir=self.path)

            if file_path in ('', None):
                sys.exit(-1)

            meta = util.get_meta(filename=Path(file_path))
            attr = util.get_attr(filename=Path(file_path)
                                 , att_name='%DateTime')

            df = pd.read_csv(
                file_path
                , delim_whitespace=True
                , na_values=['-NAN', '%;;;']
                , skiprows=attr[list(attr.keys())[0]])
            df.rename(columns={col: col.strip('%').strip()
                               for col in df.columns},
                      inplace=True)

            # col = 'IntegrationTime'
            # idx = df.loc[:, col].isin([int(meta[col])])

            # Create output variable
            dataset.update({
                name: {'Step1': {'meta': meta,
                                 # 'data': df.loc[idx, :]},
                                 'data': df.iloc[1:, :]},
                       'columns': list(df.columns),
                       'lambda0': df.iloc[0, :].to_numpy()}
            })
            # Image data
            self.dset.update({
                name: {'data': df.iloc[1:, :],
                       'wavelength': df.iloc[0, :].to_numpy()}
            })
            # =================================
        fig_name = file_path[:file_path.index('_SAM_')]
        fig_name = fig_name.replace('_Spectrum', 'step1_fig_Spectrum')
        self.filename = Path(fig_name)
        self.sidx = 1
        self.save(case='step1')
        # ============
        # Save to file
        # ============
        for key in dataset.keys():
            dataset[key]['Step1']['data'] = dataset[key]['Step1']['data'].to_numpy()
        save = f"{file_path[:file_path.index('_SAM_')]}.mat"
        save = Path(save.replace('_Spectrum', 'data_Spectrum'))
        util.savemat(file=save, data=dataset)
        print(f'DataFile: {save}\nStep1: Data import done...!!')
        return save

    def data_select(self, file: Path):
        """Step2 Check each data during Observation"""
        # ==================================================
        self.fig_size = None
        fig_name = str(self.filename).replace('step1', 'step2')
        self.filename = Path(fig_name)
        self.sidx = 0
        select = self.save('step2')
        data = util.loadmat(file=str(file), step=1)

        # print(select)
        # ============
        # Save to file
        # ============
        dataset, self.dset = {}, {}
        for key in select.keys():
            sds = data[key.upper()]['data']
            dates = pd.DatetimeIndex(
                util.to_datetime(value=sds.iloc[:, 0]))
            idx = dates.isin([p[0][0] for p in select[key]])

            # Create output variable
            dataset.update({
                key.upper(): {'Step2': {'data': sds.loc[idx, :].to_numpy()}},
            })
            days = '\n\t'.join([f'{d}' for d in dates[idx]])
            print(f'\nKey: {key}\n{"=" * 9}\nSelectDates\n\t{days}\n')
        util.savemat(file=file, data=dataset, step=2)
        print(f'File: {file}\nStep2: Select data, done!')
        return file

    def data_interpolate(self, file: Path):
        """Step3 Removing spike noise and Interpolating every 1nm"""

        dataset = {}
        self.fig_size = None
        sds = util.loadmat(file=str(file), step=2)

        wl = range(310, 1001)

        for key in sds.keys():
            data = sds[key]['data']
            self.wavelength = sds[key]['wavelength'][self.ids:self.ide]
            shape = data.shape[0], len(wl)
            temp_array, t = np.empty(shape=shape), []

            for i in range(data.shape[0]):
                self.spectra = data.iloc[i, self.ids:self.ide]
                if key in ('LW1', 'LW2'):
                    self.fig, self.ax = plt.subplots()
                    self.get_ui(key=key)
                # remove spike noize
                dif = np.concatenate(([0], np.diff(self.spectra))).astype(np.float32)
                dif = np.ma.masked_where(np.isnan(dif), dif)
                sign = np.concatenate((dif[1:] * dif[:-1], [0]))
                sign[np.isnan(sign)] = 0
                negative = sign < 0
                percent_larger = dif > np.nanmax(self.spectra.iloc[35:96]) * 0.1

                idx = negative & percent_larger

                ydata = self.spectra.loc[~idx]
                xdata = self.wavelength[~idx]

                print(f'Key: {key:3} | Data: {i} | '
                      f'OShape: {ydata.shape} '
                      f'| IDX: {all(idx)} {self.wavelength.min(), self.wavelength.max()}')

                # interpolation in 350-900 nm
                interp_data = util.interp_1d(xi=xdata, yi=ydata, xq=wl)
                temp_array[i, :] = interp_data
                t.append(data.iloc[i, 0])

            print('=' * 30)
            dataset.update({
                key: {'time': t,
                      'x': self.wavelength,
                      'y': data.iloc[:, self.ids:self.ide].to_numpy(),
                      'xi': wl,
                      'yi': temp_array}
            })
        # ==================================================
        fig_name = str(self.filename).replace('step2', 'step3')
        self.filename = Path(fig_name)
        self.dset = dataset
        self.save(case='step3')
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
        util.savemat(file=Path(file), data=save, step=3)
        print(f'File: {file}\nStep3: Interpolate data, done!')
        return file

    def data_calibrate(self, file: Path):
        """Step4 Calculating mean Es data and Calibrating Lw data"""

        dataset, self.dset = {}, {}
        self.fig_size = None
        data = util.loadmat(file=str(file), step=3)
        step2 = util.loadmat(file=str(file), step=2)

        es_mean = np.asarray(data['ES']['data'].mean(axis=0))

        for key in filter(util.skip, data.keys()):

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

            self.dset.update({
                key: {'time': t,
                      'x': step2[key]['wavelength'][self.ids:self.ide],
                      'y': step2[key]['data'].iloc[:, self.ids:self.ide].to_numpy(),
                      'xi': data[key]['wavelength'],
                      'yi': temp.to_numpy()}
            })

            dataset.update({
                key: {'Step4': {'data': temp.to_numpy()}}
            })

        # ==================================================
        fig_name = str(self.filename).replace('step3', 'step4')
        self.filename = Path(fig_name)
        self.save(case='step4')

        # ============
        # Save to file
        # ============
        util.savemat(file=file, data=dataset, step=4)
        print(f'File: {file}\nStep4: Calibrate data, done!')
        return file

    def rrs_estimate(self, file: Path):
        """Step5 Rrs Estimate"""

        self.fig_size = None
        data = util.loadmat(file=str(file), step=4)

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

        self.dset = {'Rrs': rrs,
                     'lw1_estimate': lw1_estimate,
                     'lw1_mean': lw1_mean,
                     'lw2_mean': lw2_mean,
                     'wavelength': data['LW1']['wavelength'],
                     'es_mean': es_mean}
        # ==================================================
        fig_name = str(self.filename).replace('step4', 'step5')
        self.filename = Path(fig_name)
        self.save(case='step5')

        # ============
        # Save to file
        # ============
        util.savemat(file=file, data=dataset, step=5)
        print(f'File: {file}\nStep5: Rrs estimate, done!')
        return file

    def get(self):
        # step1 - data import
        file = self.data_import()
        # step2 - select
        self.data_select(file=file)
        # step3 - interp
        # with plt.ion():
        self.data_interpolate(file=file)
        # step4 - Es calibrate
        self.data_calibrate(file=file)
        # step5 - rrs estimate
        self.rrs_estimate(file=file)
        return file


if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')
    IMAGE_PATH = HOME_DIR.joinpath('Figures')

    try:
        # for path in DATA_PATH.iterdir():
        #     main = GetRrs(data_path=DATA_PATH)
        #     main.get()
        excel_file = DATA_PATH.parent.joinpath(
            'data_Spectrum_Calibrated_2022-04-06.xlsx')
        image_file = IMAGE_PATH.joinpath(
            excel_file.name.replace('.xlsx', '.png'))
        step6_save_rrs(data_path=DATA_PATH,
                       image_file=image_file,
                       excel_file=excel_file)

    except KeyboardInterrupt:
        sys.exit(0)
