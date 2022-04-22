import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rutils import interp_1d
from matplotlib import pyplot as plt, ticker


def min_dif(array, value: float):
    dif = np.abs(array - value)
    ind, = np.where(dif == dif.min())
    return ind[0]


def chl_estimate(coeffs: dict, r: float):
    a3 = r * (coeffs['c3'] + r * coeffs['c4'])
    a2 = r * (coeffs['c2'] + a3)
    a1 = r * (coeffs['c1'] + a2)
    return coeffs['c0'] + a1


if __name__ == '__main__':
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    DATA_PATH = BASE.joinpath('OpticalData', 'RAMSES')
    IMAGE_PATH = BASE.joinpath('Figures')
    RRS_FILE = DATA_PATH.joinpath('data_Spectrum_Calibrated_2021-09-29_08-21-06_671.xlsx')
    CHL_FILE = BASE.joinpath('OpticalData', 'chl_202109-Toyama.xlsx')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        # plt.style.use('ggplot')
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (16, 13)  # (12, 7)  # (14, 9)
        plt.rcParams['font.size'] = 40

        plt.rcParams['xtick.major.size'] = 10
        plt.rcParams['ytick.major.size'] = 10
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.width'] = 2

        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1

        # clr = {'8': {'c': '#1f77b4', 'l': '-'},
        #        '10.1': {'c': '#ff7f0e', 'l': '-'},
        #        '10.2': {'c': '#ff7f0e', 'l': '--'},
        #        '11': {'c': '#2ca02c', 'l': '-'},
        #        '13': {'c': '#d62728', 'l': '-'},
        #        'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}
        clr = {'8': {'c': 'k', 'l': '-'},
               '10': {'c': '#ff7f0e', 'l': '-'},
               '11': {'c': '#2ca02c', 'l': '-'},
               '12': {'c': 'b', 'l': '-'},
               '13': {'c': 'r', 'l': '-'},
               'SP': {'c': '#9467bd', 'l': '-'}, '': '#8c564b'}

        PRR_DSET = pd.read_excel(RRS_FILE, sheet_name='PRR_Rrs')
        RAMSES_DSET = pd.read_excel(RRS_FILE, sheet_name='RAMSES_Rrs')
        INSITU_CHL = pd.read_excel(CHL_FILE, sheet_name='chla-comp')

        XLIM = (PRR_DSET['Wavelength [nm]'].min() - 10,
                PRR_DSET['Wavelength [nm]'].max() + 10)
        print(XLIM)

        # OC3M	MODIS	Y	443>488	    547	0.2424	-2.7423	    1.8017	0.0015      -1.2280
        # OC3V	VIIRS	Y	443>486	    550	0.2228	-2.4683	    1.5867	-0.4275	    -0.7768
        # OC4   SGLI    Y   443>490>530 555 0.39747 -3.42876    5.33109 -5.39966    1.73379

        # print(trios)
        SENSORS = {'OC3M': {'c0': 0.2424, 'c1': -2.7423, 'c2': 1.8017, 'c3': 0.0015, 'c4': -1.2280,
                            'bands': [443, 488], 'b0': 547, 'm': 'o', 'f': 'w', 'sen': 'MODIS/Aqua', 'l': '-k'},
                   'OC3V': {'c0': 0.2228, 'c1': -2.4683, 'c2': 1.5867, 'c3': -0.4275, 'c4': -0.7768,
                            'bands': [443, 486], 'b0': 550, 'm': '^', 'f': 'w', 'sen': 'VIIRS/SNPP', 'l': '--k'},
                   'OC4': {'c0': 0.39747, 'c1': -3.42876, 'c2': 5.33109, 'c3': -5.39966, 'c4': 1.73379,
                           'bands': [443, 490, 530], 'b0': 565, 'm': 's', 'f': 'w', 'sen': 'SGLI/GCOM-C', 'l': '-.k'}}

        RWL = RAMSES_DSET['Wavelength [nm]'].values
        PWL = PRR_DSET['Wavelength [nm]'].values

        fig, ax = plt.subplots()
        MBR, legend = False, {}
        mbr = np.arange(-1, 1, 0.1)

        for key, val in SENSORS.items():
            print(key)
            for k, col in enumerate(RAMSES_DSET.columns):
                if col == 'Wavelength [nm]':
                    continue
                name = col[col.index('(') + 1:-1]
                idx = INSITU_CHL['Sta'].isin([name])
                print(f'\t{col}:', end='')
                try:
                    isc = INSITU_CHL.loc[idx, "NPEC"].values[0]
                except IndexError:
                    print()
                    continue

                # ========== RAMSES ===========
                rrs = [RAMSES_DSET.loc[RWL == lda, col].values[0]
                       for lda in val['bands']]
                rb0 = RAMSES_DSET.loc[RWL == val['b0'], col].values[0]

                # ============== PRR ==============
                pol = [cl for cl in PRR_DSET.columns if name in cl]
                yi = np.squeeze(PRR_DSET.loc[:, pol].values)
                pb0 = interp_1d(xi=PWL, yi=yi, xq=val['b0'])
                prs = [interp_1d(xi=PWL, yi=yi, xq=xq) for xq in val['bands']]

                # id0 = min_dif(array=PWL, value=val['b0'])
                # ids = [min_dif(array=PWL, value=xq) for xq in val['bands']]
                #
                # print(f'PB0: Interp = {pb0:.4f} | MinDif: {yi[id0]:.4f}\n'
                #       f'PRS: {prs} | {yi[ids]}')

                rbr = np.log10(max(rrs) / rb0)
                pbr = np.log10(max(prs) / pb0)

                chl_ram = chl_estimate(coeffs=val, r=rbr)
                chl_prr = chl_estimate(coeffs=val, r=pbr)

                sta = name.split('.')[1]

                if MBR:
                    rx, px = 10 ** rbr, 10 ** pbr
                    ry = py = isc
                else:
                    rx = px = isc
                    ry, py = 10 ** chl_ram, 10 ** chl_prr

                rrs_txt = ' '.join([f' {r:.6f}' for r in rrs + [rb0]])
                prr_txt = ' '.join([f' {r:.6f}' for r in prs + [pb0]])
                print(f'\n\t{rrs_txt} | {rx:.3f} vs. {ry:.3f} | OCx: {10**chl_ram:.3f}'
                      f'\n\t{prr_txt} | {px:.3f} vs. {py:.3f} | OCx: {10**chl_prr:.3f}\n')
                # xx.append(rx)
                # yy.append(ry)
                ax.scatter(rx, ry,
                           c=clr[sta.strip()]['c'],
                           marker=val['m'],
                           facecolors=val['f'],
                           edgecolors='k',
                           alpha=0.6,
                           s=500,
                           linewidths=1,
                           )
                ax.scatter(px, py,
                           # c=clr[sta.strip()]['c'],
                           marker=val['m'],
                           facecolors=val['f'],
                           edgecolors=clr[sta.strip()]['c'],
                           alpha=1,
                           s=400,
                           linewidths=2,
                           )
                if k == 1:
                    h = ax.scatter(12, 12, c='k', marker=val['m'], s=500)
                    legend.update({f'{key} ({val["sen"]})': h})

            if MBR:
                chl_est = 10 ** (val['c0'] + mbr * (val['c1'] + mbr * (
                        val['c2'] + mbr * (val['c3'] + mbr * val['c4']))))
                h = ax.plot(10 ** mbr, chl_est, val['l'], lw=3)
                legend.update({f'{key} ({val["sen"]})': h[0]})
            # print(f'{key}: {chl_hypo.min():.3f} {chl_hypo.max():.3f}')

        # ===============================================
        # fig_Spectrum_Calibrated_2021-09-29_08-21-06_671
        ax.set_yscale('log')
        ax.set_xscale('log')
        if MBR:
            ax.set_xlabel(r'$\it{in}$-$\it{situ\ R}\rm_{rs}(\lambda_{blue})/'
                          r'\it{R}\rm_{rs}(\lambda_{green})$')
            ax.set_ylabel(r'$\it{in}$-$\it{situ}$ Chl-$\it{a}]$ [mg m$^{-3}]$')
            fig_name = RRS_FILE.name.replace(
                'data_Spectrum', f'fig_insitu_mbr').replace(
                '.xlsx', f'_{datetime.today().strftime("%Y%j")}')
            # ax.axhline(y=0.1, **{'linestyle': ':', 'color': 'gray'})
            # ax.axvline(x=0, **{'linestyle': ':', 'color': 'gray'})

            ax.set_xticks((0.4, 0.6, 1, 2, 4, 6, 10))
            ax.set_yticks((0.5, 1, 2, 5, 10))
            ax.set_xlim(0.3, 11)
            ax.set_ylim(0.6, 11)

            # ax.set_xlim(-1.04575749, 1.04139269)
            # ax.set_ylim(-1.04575749, 1.04139269)
        else:
            ax.set_ylabel(r'OCx Chl-$\it{a}$ [mg m$^{-3}$]')
            ax.set_xlabel(r'$\it{in}$-$\it{situ}$ Chl-$\it{a}$ [mg m$^{-3}$]')
            # ax.legend(handles=legend.values(),
            #           labels=legend.keys())
            fig_name = RRS_FILE.name.replace(
                'data_Spectrum', f'fig_insitu_ocx_chlor').replace(
                '.xlsx', f'_{datetime.today().strftime("%Y%j")}.png')
            ax.plot([0.09, 11], [0.09, 11], '-k')
            ax.set_xticks((0.5, 1, 2, 5, 10))
            ax.set_yticks((0.5, 1, 2, 5, 10))
            ax.set_xlim(0.3, 11)
            ax.set_ylim(0.3, 11)

        ax.grid(color='k', linestyle=':', linewidth=1.5)
        ax.grid(which='minor', color='r', linestyle=':', linewidth=0.5)
        ax.legend(handles=legend.values(),
                  labels=legend.keys())
        fmt = ticker.FormatStrFormatter('%g')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        if not IMAGE_PATH.is_dir():
            IMAGE_PATH.mkdir()
        save = IMAGE_PATH.joinpath(fig_name)
        plt.savefig(save)

        # plt.show()
        print(f'Image: {save}')

    except KeyboardInterrupt:
        sys.exit(0)
