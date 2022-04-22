import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from tkinter import BOTH, LEFT, END, X, ttk, Tk

import numpy as np
import pandas as pd
import xlrd
from matplotlib import pyplot as plt, ticker
from matplotlib.widgets import Button
from screeninfo import screeninfo

from autils import smooth
from rutils import to_datetime

LW_BAND, ES_BAND, IDS, IDE = 62, 62, 4, 258
BUTTON_COLOUR, HOVER_COLOUR = "gray", "r"


class Placeholder(ttk.Entry):
    """
        https://stackoverflow.com/questions/27820178/how-to-add-placeholder-to-an-entry-in-tkinter
    """
    def __init__(self, master=None,
                 placeholder='',
                 fg='black',
                 fg_placeholder='grey50', *args, **kw):
        super().__init__(master=master, *args, **kw)
        self.fg = fg
        self.fg_placeholder = fg_placeholder
        self.placeholder = placeholder
        self.bind('<FocusOut>', lambda event: self.fill_placeholder())
        self.bind('<FocusIn>', lambda event: self.clear_box())
        self.fill_placeholder()

    def clear_box(self):
        if not self.get() and super().get():
            self.config()
            self.delete(0, END)

    def fill_placeholder(self):
        if not super().get():
            self.config()
            self.insert(0, self.placeholder)

    def get(self):
        content = super().get()
        if content == self.placeholder:
            return ''
        return content


class SimpleDialog(ttk.Frame):
    """"
    https://stackoverflow.com/questions/50573260/simpledialog-simpledialog-in-python-with-2-inputsh
        Various Options are:
            anchor: This options is used to control the positioning of the text if the widget has more space than
                    required for the text. The default is anchor=CENTER, which centers the text in the available space.

            bg: This option is used to set the normal background clior displayed behind the label and indicator.
                height:This option is used to set the vertical dimension of the new frame.

            width: Width of the label in characters (not pixels!).
                   If this option is not set, the label will be sized to fit its contents.

            bd: This option is used to set the size of the border around the indicator.
                Default bd value is set on 2 pixels.

            font: If you are displaying text in the label (with the text or textvariable option),
                  the font option is used to specify in what font that text in the label will be displayed.

            cursor: It is used to specify what cursor to show when the mouse is moved over the label.
                    The default is to use the standard cursor.

            textvariable: As the name suggests it is associated with a Tkinter variable (usually a StringVar)
                          with the label. If the variable is changed, the label text is updated.

            bitmap:It is used to set the bitmap to the graphical object specified so that, the label can
            represent the graphics instead of text.

            fg:The label clior, used for text and bitmap labels. The default is system specific.
            If you are displaying a bitmap, this is the clior that will appear at the position of the 1-bits in the bitmap.

            image: This option is used to display a static image in the label widget.

            padx:This option is used to add extra spaces between left and right of the text within the label.
            The default value for this option is 1.

            pady:This option is used to add extra spaces between top and bottom of the text within the label.
            The default value for this option is 1.

            justify:This option is used to define how to align multiple lines of text.
            Use LEFT, RIGHT, or CENTER as its values. Note that to position the text inside the widget, use the anchor option.
            Default value for justify is CENTER.

            relief: This option is used to specify appearance of a decorative border around the label.
            The default value for this option is FLAT.

            underline:This

            wraplength:Instead of having only one line as the label text it can be broken itno to the number of
            lines where each line has the number of characters specified to this option."""
    prompt = 'Select observation time.'

    def __init__(self, master=None, times: dict = None):
        self.root = master
        super().__init__(master=master)
        # self allow the variable to be used anywhere in the class
        self.lw1 = ""
        self.lw2 = ""
        self.es = ""
        self.entry0 = None
        self.entry1 = None
        self.entry2 = None
        self.times = times

        self.master.title(self.prompt)
        self.pack(fill=BOTH, expand=True)

        for i, name in enumerate(('lw1', 'lw2', 'es')):
            frame = ttk.Frame(self)
            frame.pack(fill=X)

            label = ttk.Label(frame, text=name.upper(), width=5)
            label.pack(side=LEFT, padx=10, pady=10)

            entry = Placeholder(frame, placeholder=times[name])
            setattr(self, f'entry{i}', entry)
            val = getattr(self, f'entry{i}')
            val.pack(fill=X, padx=10, expand=True)

        frame = ttk.Frame(self)
        frame.pack(fill=X)
        label = ttk.Label(
            frame, text='Push "Submit" to use '
                        'default values.\nOtherwise enter '
                        'comma-separated values.\nEnter values as'
                        ' "start, end" time.')
        label.pack(side=LEFT, padx=10, pady=10)

        frame = ttk.Frame(self)
        frame.pack(fill=X)
        # Command tells the form what to do when the button is clicked
        btn = ttk.Button(frame, text="Submit",
                         command=self.on_submit)
        btn.pack(padx=10, pady=10)

    def on_submit(self):
        for i, name in enumerate(('lw1', 'lw2', 'es')):
            value = getattr(self, f'entry{i}').get()
            if len(value) == 0:
                setattr(self, name, self.times[name].split(','))
            else:
                setattr(self, name, value.split(','))
        self.quit()


def select_data(times: dict):
    # This part triggers the dialog
    root = Tk()
    monitor = screeninfo.get_monitors()[0]
    x, y = (int(monitor.width * 0.2),
            int(monitor.width * 0.15))
    root.geometry(f"{x}x{y}+{x}+{x}")
    app = SimpleDialog(master=root,
                       times=times)
    root.mainloop()
    # Here we can act on the form components or
    # better yet, copy the output to a new variable
    user_input = {'lw1': app.lw1,
                  'lw2': app.lw2,
                  'es': app.es}
    print(user_input)
    # Get rid of the error message if the user clicks the
    # close icon instead of the submit button
    # Any component of the dialog will no longer be available
    # past this point
    try:
        root.destroy()
    except ValueError:
        pass
    # To use data outside of function
    # Can either be used in __main__
    # or by external script depending on
    # what calls main()
    return user_input


def extract_data(times: dict, dataset: dict):
    lw1 = lw2 = es = None
    this_day = to_datetime(
        value=dataset['LW1']['data'].iloc[1, 0]
    )
    y, m, d = this_day.year, this_day.month, this_day.day

    for i, (key, val) in enumerate(times.items()):
        # print(f'{key}: {val} | {val[0].split(":")}')
        sds = dataset[key.upper()]['data']

        try:
            t, h, s = val[0].split(':')
        except ValueError:
            t, h, s = val[0].split(':') + [0]
        start = datetime(y, m, d, int(t), int(h), int(s))
        try:
            t, h, s = val[1].split(':')
        except ValueError:
            t, h, s = val[1].split(':') + [0]
        end = datetime(y, m, d, int(t), int(h), int(s))

        dates = get_dates(df=sds)
        idx, = np.where((dates >= start) & (dates <= end))
        if key == 'lw1':
            lw1 = sds.iloc[idx, :]
        if key == 'lw2':
            lw2 = sds.iloc[idx, :]
        if key == 'es':
            es = sds.iloc[idx, :]
    return lw1, lw2, es


def get_timerange(values: list):
    lower, upper = [], []

    for val in values:
        lower.append(
            to_datetime(value=val.iloc[1, 0])
        )
        upper.append(
            to_datetime(value=val.iloc[-1, 0])
        )
    return lower, upper


def initialise(rows: int, cols: int, monitor, dpi=100):
    """
        https://stackoverflow.com/questions/27861916/
        update-refresh-matplotlib-plots-on-second-monitor
    """
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex='col'
    )
    # mgr = plt.get_current_fig_manager()
    # px, py = mgr.window.wm_maxsize()
    # x, y = 0, monitor.y
    # mgr.window.wm_geometry(f"{y}+{x}")
    py = monitor.height / dpi
    px = monitor.width / dpi
    fig.set_size_inches(px, py, forward=True)
    return fig, axs


def get_dates(df):
    return np.array([
        xlrd.xldate_as_datetime(
            day, 0) for day in df.iloc[1:, 0]
    ])


class Figure:

    def __init__(self, dataset,
                 filename: Path,
                 picker: bool = False,
                 step: int = None,
                 dpi: int = 100,
                 columns: dict = None,
                 fig_size: tuple = None,
                 font_size: float = 12,
                 start_index: int = 0):
        self.init = None
        self.init_fig = None
        self.lines = None
        self.append = None
        self.axes = None
        self.fig = None
        self.es_points = None
        self.lw2_points = None
        self.lw1_points = None
        self.tlu = None
        self.tll = None
        self.lw2 = None
        self.es = None
        self.lw1 = None
        self.fig_size = fig_size
        self.columns = columns
        self.cols = None
        self.rows = None
        self.dset = dataset
        self.picker = picker
        self.rrs_step = step
        self.sidx = start_index
        self.filename = filename
        self.font_size = font_size
        self.dpi = dpi

    def initialise(self, case: str):
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'

        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 2
        plt.rcParams['xtick.minor.width'] = 1

        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 2
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams.update({'font.size': self.font_size})

        rows, cols = None, None
        if self.fig_size:
            plt.rcParams['figure.figsize'] = self.fig_size
        else:
            rows = 3 if case == 'step5' else len(self.dset.keys())
            cols = 1 if case == 'step5' else 2
            height = rows * 5.333333333333333
            width = height * 1.575
            plt.rcParams['figure.figsize'] = width, height
            self.fig_size = width, height

        if case != 'step2':
            plt.rcParams.update({'font.size': self.font_size + 10})
            plt.rcParams['figure.constrained_layout.use'] = True

        if case == 'step2':
            # monitor = info[0]
            # if len(info) > 1:
            #     print(plt.rcParams["backend"])
            #     # plt.rcParams["backend"] = 'QT4Agg'
            #     monitor = [m for m in info if not m.is_primary][0]
            monitor = screeninfo.get_monitors()[0]
            self.init = partial(initialise,
                                rows=rows,
                                cols=cols,
                                monitor=monitor)
            return [None] * 2

        if case == 'step1':
            return plt.subplots(
                nrows=rows,
                ncols=cols,
                sharex='col')

        if case in ('step3', 'step4'):
            return plt.subplots(
                nrows=rows, ncols=cols,
                sharex='col', sharey='row'
            )

        if case == 'step5':
            return plt.subplots(
                nrows=rows,
                ncols=cols,
                sharex='col'
            )
        return plt.subplots(dpi=self.dpi)

    def metadata(self):
        return {'Title': self.filename.name,
                'Author': 'Eligio',
                'Email': 'eligiomaure@gmail.com',
                'date_created': time.ctime()}

    def save(self, case: str):
        self.fig, self.axes = self.initialise(case=case)
        if case in ('step1', 'step2'):
            self.lw1 = self.dset['LW1']['data']
            self.lw2 = self.dset['LW2']['data']
            self.es = self.dset['ES']['data']
            return self.save12(case=case)

        if case in ('step3', 'step4'):
            return self.figure34(case=case)

        if case == 'step5':
            return self.figure5()

        if case in ('ap', 'ad', 'ay', 'aph'):
            return self.figure_iops(var=case[1:])

    def save12(self, case: str):
        start_dates = self.get_timerange()
        end_dates = self.get_timerange('end')
        self.tll = min(start_dates) - timedelta(seconds=2)
        self.tlu = max(end_dates) + timedelta(seconds=2)
        # key = 'lambda0' if case == 'step1' else 'wavelength'

        if case == 'step2':
            self.picker = True
            return self.ui2(sd=start_dates, ed=end_dates)
        else:
            self.figure12()
            self.save_fig()
            plt.close(self.fig)
        return self.dset

    def save_fig(self, tight: bool = False):
        func = partial(self.fig.savefig,
                       fname=self.filename,
                       dpi=self.dpi,
                       metadata=self.metadata())
        if tight:
            func(bbox_inches='tight')
            return 0
        func()
        return 0

    def get_timerange(self, case: str = 'start'):
        values = []
        for val in [self.lw1, self.lw2, self.es]:
            if case == 'start':
                values.append(
                    to_datetime(value=val.iloc[1, 0])
                )
            if case == 'end':
                values.append(
                    to_datetime(value=val.iloc[-1, 0])
                )
        return values

    def extract(self, times: dict):
        lw1 = lw2 = es = None
        this_day = to_datetime(
            value=self.dset['LW1']['data'].iloc[1, 0]
        )
        y, m, d = this_day.year, this_day.month, this_day.day

        for i, (key, val) in enumerate(times.items()):
            # print(f'{key}: {val} | {val[0].split(":")}')
            sds = self.dset[key.upper()]['data']

            try:
                t, h, s = val[0].split(':')
            except ValueError:
                t, h, s = val[0].split(':') + [0]
            start = datetime(y, m, d, int(t), int(h), int(s))
            try:
                t, h, s = val[1].split(':')
            except ValueError:
                t, h, s = val[1].split(':') + [0]
            end = datetime(y, m, d, int(t), int(h), int(s))

            dates = get_dates(df=sds)
            idx, = np.where((dates >= start) & (dates <= end))
            if key == 'lw1':
                lw1 = sds.iloc[idx, :]
            if key == 'lw2':
                lw2 = sds.iloc[idx, :]
            if key == 'es':
                es = sds.iloc[idx, :]
        return lw1, lw2, es

    def callback(self, event):
        self.fig.set_figwidth(self.fig_size[0])
        self.fig.set_figheight(self.fig_size[1])
        plt.rcParams.update({'font.size': self.font_size + 10})
        plt.draw()
        self.save_fig(tight=True)
        plt.close(self.fig)
        return event

    def on_pick(self, event):
        if event.artist not in self.lines:
            print(f'{event.artist} not in {self.lines}')
            return
        n = len(event.ind)
        # lines = '\n\t'.join([f'{li}' for li in self.lines])

        if not n:
            return
        this_line = event.artist
        xd = this_line.get_xdata()
        yd = this_line.get_ydata()
        ind = event.ind
        cax = [ix for ix, ln in enumerate(self.lines)
               if ln == this_line][0]

        info = f'Artist: {this_line}'
        print(f'{info}\nAxis: {cax, 1}')
        name = {0: 'lw1_points',
                1: 'lw2_points',
                2: 'es_points'}[cax]
        points = getattr(self, name)

        xy = tuple(zip(xd[ind], yd[ind]))
        this_ax = self.axes[cax, 1]
        if xy in points['points']:
            # deselect
            idx = [kx for kx, p in enumerate(points['points'])
                   if xy == p][0]
            points['points'].pop(idx)
            points['axes'][idx].set_marker(None)
            points['axes'].pop(idx)
            # this_line.set_marker(None)
        else:
            # select
            mk = this_ax.plot(
                xd[ind],
                yd[ind],
                'or',
                fillstyle='full',
                **dict(markersize=10),
                picker=self.picker
            )
            points['points'].append(xy)
            points['axes'].append(mk[0])
        # setattr(self, name, points)

        plt.draw()
        # fig.canvas.flush_events()
        print(f'onPick point:\n\t'
              f'Time: {xy[0][0]}\n\t'
              f'Data: {xy[0][1]}\n'
              f'{"=" * len(info)}')
        return True

    def ui2(self, sd, ed):
        self.lw1_points = {'points': [], 'axes': []}
        self.lw2_points = {'points': [], 'axes': []}
        self.es_points = {'points': [], 'axes': []}

        times = {}
        for lab, st, en in zip(('lw1', 'lw2', 'es'), sd, ed):
            times.update({
                lab: f'{st.strftime("%H:%M:%S")}, '
                     f'{en.strftime("%H:%M:%S")}'
            })
        with plt.ion():
            # Get figure
            self.fig, self.axes = self.init()
            self.figure12()
            # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            # select time range
            times = select_data(times=times)
            # Extract time range data
            self.lw1, self.lw2, self.es = self.extract(times=times)
            # update figure limits
            self.tll = min(self.get_timerange('start')) - timedelta(seconds=2)
            self.tlu = max(self.get_timerange('end')) + timedelta(seconds=2)

        # Get new figure
        plt.close(self.fig)
        self.lines = []
        append = self.lines.append
        self.fig, self.axes = self.init()
        self.figure12(append=append)
        okx = plt.axes([0.5, 0.01, 0.03, 0.04])
        ok = Button(okx,
                    'OK',
                    color=BUTTON_COLOUR,
                    hovercolor=HOVER_COLOUR)
        ok.on_clicked(self.callback)
        self.fig.canvas.mpl_connect(
            'pick_event', self.on_pick)
        plt.show()
        return {'lw1': self.lw1_points['points'],
                'lw2': self.lw2_points['points'],
                'es': self.es_points['points']}

    def figure12(self, append=None):
        title = None
        ylab, ylim = None, None
        xlab, xlim = None, None

        for i, ax in enumerate(self.axes.flatten()):
            if i in (0, 1):
                # LW1 wavelength, LW1 time
                label, data, wavelength, vl_band = (
                    'Lw1 [mW m$^{-1}$ nm$^{-1}$ sr$^{-1}$]', self.lw1,
                    self.dset['LW1']['wavelength'][IDS:IDE],
                    self.dset['LW1']['wavelength'][LW_BAND]
                )
            elif i in (2, 3):
                # LW2 wavelength, LW2 time
                label, data, wavelength, vl_band = (
                    'Lw2 [mW m$^{-1}$ nm$^{-1}$ sr$^{-1}$]', self.lw2,
                    self.dset['LW2']['wavelength'][IDS:IDE],
                    self.dset['LW2']['wavelength'][LW_BAND]
                )
            else:
                # ES wavelength, ES time
                label, data, wavelength, vl_band = (
                    'Es [mW m$^{-2}$ nm$^{-1}$]', self.es,
                    self.dset['ES']['wavelength'][IDS:IDE],
                    self.dset['ES']['wavelength'][ES_BAND]
                )

            if i in (0, 2):
                ymx = max([np.nanmax(self.lw1.iloc[self.sidx:, IDS:IDE]),
                           np.nanmax(self.lw2.iloc[self.sidx:, IDS:IDE])])
            else:
                ymx = max([np.nanmax(self.es.iloc[self.sidx:, IDS:IDE]),
                           np.nanmax(self.es.iloc[self.sidx:, IDS:IDE])])

            if i in (0, 2, 4):
                ydata = data.iloc[self.sidx:, IDS:IDE]
                if wavelength.shape != ydata.shape:
                    for j in range(ydata.shape[0]):
                        ax.plot(wavelength, ydata.iloc[j, :])
                else:
                    ax.plot(wavelength, ydata)
                ax.axvline(x=vl_band, ymin=0, ymax=1, lw=2, color='b')

                xlim, ylim = [300, 1000], [0, ymx]
                title, ylab, xlab = 'data (raw)', label, 'Wavelength [nm]'

            if i in (1, 3, 5):
                # convert excel num to dates
                dates = pd.DatetimeIndex(
                    [to_datetime(value=d)
                     for d in data.iloc[self.sidx:, 0]]
                )

                band = LW_BAND if i in (1, 3) else ES_BAND
                ydata = data.iloc[self.sidx:, band]

                mx1 = np.ceil(max(
                    [np.nanmax(self.lw1.iloc[self.sidx:, band]),
                     np.nanmax(self.lw2.iloc[self.sidx:, band])]
                ))
                mx2 = np.ceil(max(
                    [np.nanmax(self.es.iloc[self.sidx:, band]),
                     np.nanmax(self.es.iloc[self.sidx:, band])]
                ))
                ymx = [-0.2, mx1] if i in (1, 3) else [0, mx2]

                xlim, ylim, xlab = [self.tll, self.tlu], None, None
                title, ylab = f'data at #{band} band ({vl_band:.4f} nm)', label

                # ================================
                line, = ax.plot(
                    dates,
                    ydata,
                    '-ob',
                    fillstyle='none',
                    **dict(markersize=15),
                    picker=self.picker
                )
                if append:
                    append(line)
            ax.tick_params(axis='x', labelrotation=45)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if title:
                if self.picker and (i in (1, 3, 5)):
                    ax.set_title(
                        f'{title}\nclick on point to select/deselect data')
                else:
                    ax.set_title(title)
            if ylab:
                ax.set_ylabel(ylab)
            if xlab:
                ax.set_xlabel(xlab)
        return 0

    def figure34(self, case: str):
        sup_title = 'despike noise and interpolate' \
            if case == 'step3' \
            else None
        title = '1 nm interpolated' \
            if case == 'step3' \
            else '1 nm interpolated & Es calibrated'
        labels = {'Lw1': '[mW m$^{-1}$ nm$^{-1}$ sr$^{-1}$]',
                  'Lw2': '[mW m$^{-1}$ nm$^{-1}$ sr$^{-1}$]',
                  'Es': '[mW m$^{-2}$ nm$^{-1}$]'}
        keys = [k.lower() for k in self.dset.keys()]
        for k, (key, unit) in enumerate(labels.items()):

            if key.lower() not in keys:
                continue
            print(f'KEY: {key}')
            try:
                sds = self.dset[key]
            except KeyError:
                sds = self.dset[key.upper()]
            ax = self.axes[k, :]
            ax[0].set_ylabel(f'{key} {unit}')
            ax[0].set_title('original')
            ax[1].set_title(title)

            for j in range(sds['yi'].shape[0]):
                print(f'\tData: {j}')
                label = xlrd.xldate_as_datetime(
                    sds['time'][j], 0
                ).strftime('%H:%M:%S')
                ax[0].plot(sds['x'], sds['y'][j, :], label=label)
                ax[1].plot(sds['xi'], sds['yi'][j, :])
            ax[0].legend()

        if sup_title:
            self.fig.suptitle(sup_title)
        self.axes[-1, 0].set_xlim([300, 1000])
        self.axes[-1, 0].tick_params(axis='x', labelrotation=45)
        self.axes[-1, 0].set_xlabel('Wavelength [nm]')

        self.axes[-1, -1].set_xlim([300, 1000])
        self.axes[-1, -1].tick_params(axis='x', labelrotation=45)
        self.axes[-1, -1].set_xlabel('Wavelength [nm]')
        self.save_fig()
        plt.close(self.fig)
        return self.filename

    def figure5(self):
        self.axes[0].plot(self.dset['wavelength'],
                          self.dset['Rrs'])
        self.axes[0].set_ylabel('Rrs [sr$^{-1}$]')

        self.axes[1].plot(self.dset['wavelength'],
                          self.dset['lw1_estimate'],
                          label='Lw1 estimate')
        self.axes[1].plot(self.dset['wavelength'],
                          self.dset['lw1_mean'],
                          label='Lw1 mean')
        self.axes[1].plot(self.dset['wavelength'],
                          self.dset['lw2_mean'],
                          label='Lw2 mean')
        self.axes[1].set_ylabel('Lw [mW m$^{-1}$ nm$^{-1}$ sr$^{-1}$]')
        self.axes[1].legend()

        self.axes[2].plot(self.dset['wavelength'],
                          self.dset['es_mean'])
        self.axes[2].set_title('Es mean')
        self.axes[2].set_ylabel('Es [mW m$^{-2}$ nm$^{-1}$]')

        self.axes[2].set_xlim([300, 1000])
        self.axes[2].set_xlabel('Wavelength [nm]')
        self.axes[2].tick_params(axis='x', labelrotation=45)
        self.save_fig()
        plt.show()
        return self.filename

    def figure_iops(self, var: str):
        for val in self.columns.values():
            key = f'A{var.upper()}_{val}'
            print(f'{key}: {val}')

            x = self.dset.loc[:, (key, 'mean')]
            if var == 'ph':
                x = smooth(data=x, window_length=5)
            self.axes.plot(self.dset.index, x,
                           label=val.replace('ST', 'STA. '),
                           lw=3)
        self.axes.legend()
        self.axes.set_xlim(min(self.dset.index) - 10,
                           max(self.dset.index) + 10)
        self.axes.set_xlabel(r'$\lambda$ [nm]',
                             size=self.font_size + 20)

        if var == 'p':
            self.axes.set_ylabel(r'$\ita$$\rm_{p}$ [m$^{-1}$]',
                                 size=self.font_size + 20)
        if var == 'd':
            self.axes.set_ylabel(r'$\ita$$\rm_{nap}$ [m$^{-1}$]',
                                 size=self.font_size + 20)
            ymx = self.dset.values.max() + .01
            ticks = (0, 0.5, 1, 1.5, 2) \
                if ymx > 1.5 \
                else np.arange(0, 1.1, 0.1) \
                if ymx <= 1 \
                else (0, 0.25, .5, .75, 1, 1.25, 1.5)
            self.axes.set_yticks(ticks)
            self.axes.set_ylim(0, ymx)
            print(0, ymx)

        if var == 'y':
            self.axes.set_ylabel(r'$\ita$$\rm_{y}$ [m$^{-1}$]',
                                 size=self.font_size + 20)
            self.axes.set_ylim(-.05, 1.76)
            self.axes.set_yticks((0, 0.5, 1, 1.5))

        if var == 'ph':
            self.axes.set_ylabel(r'$\ita$$\rm_{\phi}$ [m$^{-1}$]',
                                 size=self.font_size + 20)
        if var not in ('d', 'y'):
            self.axes.set_yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        self.axes.grid('minor')
        self.axes.grid('major', linewidth=2)

        fmt = ticker.FormatStrFormatter('%g')
        self.axes.yaxis.set_major_formatter(fmt)
        self.save_fig()
        plt.close(self.fig)
        return self.filename
