import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, colors, ticker

if __name__ == '__main__':
    # R.J.W. Brewin et al. / Ecological Modelling 221 (2010) 1472–1483
    BASE = Path(r'C:\Users\Eligio\Documents\NPEC\Toyama\Hayatsuki\202109')
    IMAGE_PATH = BASE.joinpath('Figures')

    try:

        plt.rcParams['savefig.facecolor'] = "0.8"
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 20

        fig, ax = plt.subplots()

        s12 = 0.851  # S1,2 is the initial slope
        s1 = 6.801
        c12m = 1.057  # mg m−3 Cm 1,2 is the asymptotic maximum value for C1,2 and
        c1m = 0.107  # mg m−3

        # total = c1 + c2 + c3
        # pico (1), nano (2), micro (3)
        total = np.logspace(-2, 1, 1000)
        pico_nano = c12m * (1 - np.exp(-s12 * total))
        micro = total - pico_nano
        pico = c1m * (1 - np.exp(-s1 * total))
        nano = pico_nano - pico

        ax.plot(total, pico / total, label='Pico', lw=3,
                color=colors.to_hex((130 / 255, 97 / 255, 161 / 255)))
        ax.plot(total, nano / total, label='Nano', lw=3,
                color=colors.to_hex((75 / 255, 172 / 255, 198 / 255)), )
        ax.plot(total, micro / total, lw=3, label='Micro',
                color=colors.to_hex((247 / 255, 150 / 255, 70 / 255)))

        ax.legend()
        ax.set_xscale('log')
        ax.set_xticks((0.01, 0.1, 1, 10))
        majors = ["0.01", "0.1", "1", "10"]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(majors))
        ax.set_xlim(min(total), max(total))
        ax.set_ylim(0, 1)

        ax.set_xlabel(r'Chl-$\it{a}$ [mg m$^{-3}$]')
        ax.set_ylabel(r'Chl-$\it{a}$ size-class')

        save = IMAGE_PATH.joinpath('fig_phyto_size_class_model.png')
        fig.savefig(save)
        # plt.show()

    except KeyboardInterrupt:
        sys.exit(0)
