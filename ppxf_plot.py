###############################################################################
#
#   File:       ppxf_plot.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Make nice plots from a ppxf instance.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
import matplotlib.pyplot as plt
import numpy as np


def ppxf_plot(pp, ax=None):
    """
    Produces a plot of the pPXF best fit.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if pp.lam is None:
        ax.set_xlabel("Pixels")
        x = np.arange(pp.galaxy.size)
    else:
        ax.set_xlabel(r"Observer-frame wavelength (\AA)")
        x = pp.lam

    resid = pp.galaxy - pp.bestfit
    mn = min(np.min(pp.bestfit[pp.goodpixels]),
             np.min(pp.galaxy[pp.goodpixels]))
    mn -= np.percentile(np.abs(resid[pp.goodpixels]), 99)
    # mx = max(np.max(pp.bestfit[pp.goodpixels]),np.max(pp.galaxy[pp.goodpixels]))
    # mx = np.max(pp.galaxy[pp.goodpixels])
    resid += mn   # Offset residuals to avoid overlap
    mn1 = np.min(resid[pp.goodpixels])
    ax.set_ylabel("Relative Flux")

    ax.plot(x, pp.galaxy, 'k', label='Input spectrum')
    ax.fill_between(x, pp.galaxy + pp.noise, pp.galaxy -
                    pp.noise, alpha=0.6, linewidth=0, color='darkgray')
    ax.plot(x[pp.goodpixels], resid[pp.goodpixels], 'd',
            color='LimeGreen', mec='LimeGreen', ms=4, label='Residual')
    w = np.flatnonzero(np.diff(pp.goodpixels) > 1)
    for wj in w:
        a, b = pp.goodpixels[wj: wj + 2]
        ax.axvspan(x[a], x[b], facecolor='lightgray', edgecolor='none')
        ax.plot(x[a: b + 1], resid[a: b + 1], 'b')
    for k in pp.goodpixels[[0, -1]]:
        ax.plot(x[[k, k]], [mn, pp.bestfit[k]], 'lightgray')

    if pp.gas_any:
        if pp.sky is None:
            stars_spectrum = pp.bestfit - pp.gas_bestfit
            ymin = mn
        else:
            nsky = pp.sky.shape[1]
            if nsky == 1:
                sky_spectrum = np.squeeze(pp.sky) * pp.weights[-nsky:]
            else:
                sky_spectrum = np.sum(pp.sky * pp.weights[-nsky:], axis=1)
            stars_spectrum = pp.bestfit - sky_spectrum
            ax.plot(x, sky_spectrum, 'c', linewidth=2, label='Best fit (sky)')
            ax.plot(x, pp.galaxy - sky_spectrum, 'gray',
                    label='Input spectrum - sky')
            ymin = min(np.nanmin(
                pp.galaxy[pp.goodpixels] - sky_spectrum[pp.goodpixels]), np.nanmin(stars_spectrum))
        ax.plot(x, pp.gas_bestfit + mn, c='magenta',
                linewidth=2, label='Best fit (gas)')
        ax.plot(x, pp.bestfit, c='orange',
                linewidth=2, label='Best fit (total)')
        ax.plot(x, stars_spectrum, 'r', linewidth=2,
                label='Best fit (stellar)')

    else:
        ax.plot(x, pp.bestfit, 'r', linewidth=2, label='Best fit')
        ax.plot(x[pp.goodpixels], pp.goodpixels * 0 + mn, '.k', ms=1)

    # Axis limits
    xmin = x[min(pp.goodpixels)]
    xmax = x[max(pp.goodpixels)]
    ax.set_xlim([xmin, xmax] + np.array([-0.02, 0.02]) * (xmax - xmin))

    if pp.sky is None:
        ymin = mn1
    else:
        ymin = np.min((pp.galaxy - sky_spectrum)[pp.goodpixels])
    ymax = max(np.nanmax(pp.bestfit[pp.goodpixels]),
               np.nanmax(pp.galaxy[pp.goodpixels]))
    ax.set_ylim([ymin, ymax] + np.array([-0.05, 0.05]) * (ymax - ymin))

    ax.legend(loc='upper left', numpoints=1)
