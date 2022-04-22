from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from autils import get_name, merge_pd, read_csv, strip
from figure import Figure

"""
% =========================================================================
% (c) 2015 MaureER Nov-21, Tested on Matlab 2015b
% R1. 2016.Jul. ECS (include JAXA optical dataset)
% =========================================================================
% Process Absorption of total Particulate (AP) matter
% -------------------------------------------------------------------------
% 1 - It is good to separe AP data according to measuremet day or station.
%     
% 2 - This function assumes that for each file name there are two data
%     files, one ending with '1' and the other with '2'. It then finds 1
%     and 2 with same filename and use the average of the two. 
% 
% 3 - All blanks inside one AP folder are used to get a single blank mean
%  
% 4 - The header of Excel file to put data is fixed, so copy the old JAXA
%     optical data file and delete all data leaving the header information.
%     Also don't change the sheet-name
"""


def absorbance(ref: pd, blank: pd, sample: pd, vol: pd):
    """
    """
    print(f'\n{sample}\n')

    # ---------------------------------------------
    # Calculate absorption averaging all the blanks
    def get_stdata(df, sample_vol, data, ask_names: str):
        # Filtration values are saved by the AP names
        inputs = sample_vol.loc[:, col.replace('AD', 'AP')].values if not ask else []

        for sc in sample.loc[:, idx]:
            odf = df.loc[:, sc] - blank.loc[:, bln]
            odf = odf - ref.loc[:, rfn]
            # correct the baseline
            odf = odf - odf[bsl_cor].values.mean()

            # The absorption coefficient of all suspended particles ap(lambda, m^-1)
            # and anap(lambda, m^-1) need to be corrected for the path length amplification
            # in the glass fiber filter using Cleveland and Weidemann (1993)

            count = 0
            if ask and len(inputs) == 0:
                inputs = []
                append = inputs.append
                info = 'Please enter values for the following parameters:'
                print(f'\n{"=" * len(info)}\n{info}\n{"=" * len(info)}')
                ask_name = f'{ask_names}\n{"=" * len(ask_names)}'
                while count < 4:
                    ipt = input(f'{ask_name} {fields[count]} ')
                    if ipt == "":
                        break
                    ask_name = ''
                    count += 1
                    append(float(ipt))

                fv = pd.DataFrame(inputs
                                  , columns=[col]
                                  , index=pd.Index(['F.Vol. [mL]', 'CD1 [mm]', 'CD2 [mm]', 'CD3 [mm]']))
                sp_vol = merge_pd(left=sample_vol, right=fv)

            di = np.mean(inputs[1:]) * 10 ** -3  # mm to m
            s = np.pi * (di / 2) ** 2  # clearance area in m^2
            vf = inputs[0] * 10 ** -6  # mL to m^3
            ods = odf * 0.378 + (odf ** 2) * 0.523
            df.loc[:, sc] = 2.303 * ods * s / vf

        # take the mean of the two spectrum
        mean = df.mean(axis=1).to_frame(name=(col, 'mean'))
        std = df.std(axis=1).to_frame(name=(col, 'std'))

        df = merge_pd(left=mean, right=std)
        data = merge_pd(left=data, right=df)
        return data, sample_vol

    cls = get_name(names=sample.columns, sta=True)
    # correct the baseline
    bsl_cor = sample.index == 750

    fields = ['\n\t Filtration Volume [mL] FV:',
              '\tClearance diameter [mm] d1:',
              '\tClearance diameter [mm] d2:',
              '\tClearance diameter [mm] d3:']
    # Calculate absorption subtracting average of all the blanks
    sds = None
    ask = True if vol.size == 0 else False
    filtrate = None if ask is True else vol

    bln = list(blank.columns)[0]
    rfn = list(ref.columns)[0]

    for col in cls:
        # in case the sample was measure twice
        idx = sample.columns.isin([f'{col}1', f'{col}2', f'{col}3', f'{col}4'])
        tmp = sample.loc[:, idx].copy()
        sx, sy = tmp.shape

        if sy == 4:
            for i in (1, 3):
                idx = sample.columns.isin([f'{col}{i}', f'{col}{i + 1}'])
                # remove the reference blank and the FSW blank
                tmp = sample.loc[:, idx].copy()
                sds, filtrate = get_stdata(df=tmp,
                                           sample_vol=filtrate,
                                           ask_names=' | '.join(tmp.columns),
                                           data=sds)

        if (sy == 2) or (sy == 3):
            sds, filtrate = get_stdata(df=tmp,
                                       sample_vol=filtrate,
                                       ask_names=' | '.join(tmp.columns),
                                       data=sds)
    if 'Vf-Cd' not in writer.sheets:
        filtrate.T.to_excel(writer, sheet_name='Vf-Cd')
    return sds


def od_absorbance(pattern: tuple, path: Path, fil_vol: pd, var: str):
    ref, blank, sample = pd.array([]), [], pd.array([])
    for i, pat in enumerate(pattern):
        data = None
        pat = pat.format(f'A{var.upper()}')
        for f in path.joinpath(f'a{var}').glob(pat):
            print(f.name)
            temp = read_csv(file=f)
            data = merge_pd(left=data, right=temp)
        # 1 -- save data
        data.to_excel(writer, sheet_name=SHEETS[i].format(var))

        # 2 -- read XLS file and data
        if 'BLK' in pat:
            cols = get_name(names=data.columns)
            blank = data.mean(axis=1).to_frame(cols[0])

        if 'REF' in pat:
            cols = get_name(names=data.columns)
            ref = data.mean(axis=1).to_frame(cols[0])

        if '_S' in pat:
            sample = data

    return absorbance(ref=ref, sample=sample, blank=blank, vol=fil_vol)


def ad_absorbance(pattern: tuple, path: Path, fil_vol: pd, columns: dict):
    odd = od_absorbance(pattern=pattern, path=path, fil_vol=fil_vol, var='d')
    odd.to_excel(writer, 'ad')

    # from the xls sheet the slope is calculated using lambda 592-650 but
    # this is not working for me here

    # Following Babin et al. 2003
    # The fit was done for data between 380 and 730 nm, excluding the
    # 400–480 and 620–710 nm ranges to avoid any residual pigment
    # absorption that might still have been present after sodium
    # hypochlorite treatment, in our case methanol

    wl = odd.index.values
    idx = (wl >= 380) & (wl <= 730)
    # idx = (wl >= 592) & (wl <= 650)
    ad_slope = odd.loc[idx, :]  # range 380-730 nm
    xs = ad_slope.index.values[::-1]

    # func = lambda x, a, b: a * np.exp(-b * (x - 443))
    def func(x, a, b):
        return a * np.exp(-b * (x - wl0))

    # Now remove 400-480
    idx = (xs >= 400) & (xs <= 480)
    ad_slope = ad_slope.loc[~idx, :]
    xs = xs[~idx]

    # Then remove 620-710
    idx = (xs >= 620) & (xs <= 710)
    ad_slope = ad_slope.loc[~idx, :]
    xs = xs[~idx]

    # lambda0 = 443; % reference_const
    # from the xls sheet lambda0 = 440
    wl0 = 440
    ad440 = odd.loc[wl == 440, :].values[0]  # line matrix of ay at 440
    rng = np.random.default_rng()
    fields = ["value", "lower", "upper"]
    rows = ['ad412', 'ad440', 'ad443', 'Rsq', 'Rsq_adj']
    result = pd.DataFrame([])

    # print(columns.keys())
    for j, col in enumerate(columns.keys()):
        # ay(443) for each sample
        name = col.replace('AP', 'AD')
        yf = ad_slope.loc[:, (name, 'mean')].values[::-1]
        # ys = ayd_slope.loc[:, (col, 'std')].values[::-1]
        p0 = [ad440[j], 0.0015]
        # bounds = (0, [ay443[j] + ay443[j] / 2, 1.])
        opt, cov = curve_fit(func
                             , xdata=xs
                             , ydata=yf
                             , p0=p0
                             # , sigma=ys
                             , method='lm')
        # , bounds=bounds)
        # print(f'opt: {opt}\ncov: {cov}')
        # cov - estimated variance-covariance matrix for the estimated　coefficients
        # MSE - an estimate of the variance of the error term
        yp = func(wl, *opt)
        # plt.plot(wl, odd.loc[:, (name, 'mean')])
        # plt.plot(wl, yp, 'r-',
        #          label='fit: a=%5.3f, b=%5.3f' % tuple(opt))
        # plt.legend()
        # plt.show()

        # confidence intervals at 95%
        # https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit
        sigma_ab = np.sqrt(np.diag(cov))
        # # confidence interval of the fit params
        # s = ufloat(opt[0], sigma_ab[0])
        # m = ufloat(opt[1], sigma_ab[1])

        # upper and lower bounds
        bound_upper = func(wl, *(opt + sigma_ab))
        bound_lower = func(wl, *(opt - sigma_ab))

        # goodness of fit
        # ri = yi − f(xi, b)
        yf = odd.loc[:, (name, 'mean')].values[::-1]
        residual = yf - yp
        # sum of squares of the residuals
        sse = np.sum(residual ** 2)
        # sum of squares between the data points and their mean2
        sst = np.sum((yf - yf.mean() ** 2))
        rsq = 1 - (sse / sst)
        # R-square adjusted
        dfe = yf.size - len(opt)
        rsq_adj = 1 - ((yf.size - 1) / dfe) * (sse / sst)

        idx = wl == 412
        a412 = [
            yp[idx][0],
            bound_lower[idx][0],
            bound_upper[idx][0]
        ]
        idx = wl == 440
        a440 = [
            yp[idx][0],
            bound_lower[idx][0],
            bound_upper[idx][0]
        ]
        idx = wl == 443
        a443_ = [
            yp[idx][0],
            bound_lower[idx][0],
            bound_upper[idx][0]
        ]

        tmp = pd.DataFrame([a412, a440, a443_,
                            [rsq, np.nan, np.nan],
                            [rsq_adj, np.nan, np.nan]]
                           , columns=[[name, name, name], fields]
                           , index=pd.Index(rows))
        result = merge_pd(left=result, right=tmp)
    result.T.to_excel(writer, sheet_name='slope-ci')
    return odd


def particles(fil_vol: pd):
    ap = od_absorbance(
        pattern=FILE_PATTERN,
        path=DATA_PATH,
        fil_vol=fil_vol,
        var='p')
    # save ay and sample-blank to xls file
    ap.to_excel(writer, sheet_name='ap')
    columns = strip(columns=ap.columns)

    image_file = IMAGE_PATH.joinpath(f'ap_{PNG}_{TODAY}.png')
    Figure(dataset=ap,
           columns=columns,
           filename=image_file,
           fig_size=(16, 11),
           font_size=20).save(case='ap')
    # figure(sds=pa, columns=columns, var='p',
    #        file=image_file)
    print(f'Particles: {image_file}')
    return ap


def detritus(columns: dict, fil_vol: pd):
    det = ad_absorbance(pattern=FILE_PATTERN,
                        path=DATA_PATH,
                        columns=columns,
                        fil_vol=fil_vol)
    # save ay and sample-blank to xls file
    det.to_excel(writer, sheet_name='ad')
    image_file = IMAGE_PATH.joinpath(f'ad_{PNG}_{TODAY}.png')
    Figure(dataset=det,
           columns=columns,
           filename=image_file,
           fig_size=(16, 11),
           font_size=20).save(case='ad')
    print(f'Detritus: {image_file}')
    return det


def phytoplankton(ap: pd.DataFrame,
                  det: pd.DataFrame,
                  columns: dict):
    aph = None
    for col_name in det.columns:
        # if 'std' in col_name:
        #     print(f'{col_name}: Continue')
        name = col_name[0].replace('AD', 'AP')
        phy = ap.loc[:, (name, col_name[1])] - det.loc[:, col_name]
        aph = merge_pd(left=aph, right=phy.to_frame(
            (name.replace('AP', 'APH'), col_name[1])))

    aph.to_excel(writer, sheet_name='aph')
    image_file = IMAGE_PATH.joinpath(f'aph_{PNG}_{TODAY}.png')
    Figure(dataset=aph,
           columns=columns,
           filename=image_file,
           fig_size=(16, 11),
           font_size=20).save(case='aph')
    # figure(sds=aph,
    #        columns=columns,
    #        var='ph',
    #        file=image_file)
    print(f'Phytoplankton: {image_file}')
    return


def get_data(ap: pd.DataFrame = None,
             ad: pd.DataFrame = None,
             vf: pd.DataFrame = None):
    if vf is None:
        vf = pd.DataFrame([])

    # Read excel file
    with pd.ExcelFile(XLS_FILE) as excel:
        # Filtration volume
        if 'Vf-Cd' in excel.sheet_names:
            vf = excel.parse(
                sheet_name='Vf-Cd',
                index_col=[0]
            ).T

        if 'ap' in excel.sheet_names:
            ap = excel.parse(
                sheet_name='ap',
                header=[0, 1],
                index_col=[0]
            )

        if 'ad' in excel.sheet_names:
            ad = excel.parse(
                sheet_name='ad',
                header=[0, 1],
                index_col=[0]
            )
    return ap, ad, vf


if __name__ == '__main__':
    cruise = '202204'
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki')
    DATA_PATH = BASE.joinpath(fr'{cruise}\OpticalData')
    IMAGE_PATH = BASE.joinpath(fr'{cruise}\Figures')
    PNG, TODAY = f'hayatsuki{cruise}', datetime.today().strftime('%Y%m%d')
    XLS_FILE = DATA_PATH.joinpath(f'aph_absorption_toyama{cruise}.xlsx')

    if not IMAGE_PATH.is_dir():
        IMAGE_PATH.mkdir()
    GET_VAR = 'ap', 'ad', 'aph'

    SHEETS = ['OD{}_blank', 'OD{}_reference', 'OD{}_sample']
    FILE_PATTERN = ('{}_BLK*.ASC', '{}_REF*.ASC', '{}_S*.ASC')
    MODE = 'a' if XLS_FILE.is_file() else 'w'
    col_names, apa, ada = {}, None, None
    filtration_vol = pd.DataFrame([])

    if MODE == 'a':
        apa, ada, filtration_vol = get_data()
        col_names = strip(columns=apa.columns)

    with pd.ExcelWriter(XLS_FILE, mode=MODE) as writer:
        writer.if_sheet_exists = 'replace'

        for get_var in GET_VAR:

            # =============================
            # Absorption of total particles
            # =============================
            if (get_var == 'ap') and (apa is None):
                apa = particles(
                    fil_vol=filtration_vol)
                col_names = strip(columns=apa.columns)
                continue

            # ======================
            # Absorption of detritus
            # ======================
            if get_var == 'ad':
                ada = detritus(columns=col_names,
                               fil_vol=filtration_vol)
                continue

            if get_var == 'aph':
                phytoplankton(columns=col_names, det=ada, ap=apa)
