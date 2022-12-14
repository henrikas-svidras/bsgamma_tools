import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import b2plot as bp

from uncertainties.unumpy import uarray
from uncertainties.unumpy import nominal_values as nvs
from uncertainties.unumpy import std_devs as sds

from .constants import var_to_string

def equal_binned_colz(values, x_edges, y_edges,
                      nbins=20, nlabels=20, 
                      x_formatter="%.2g",
                      y_formatter="%.2g",
                      col_min=None, col_max=None, cmap='viridis',
                      xname='X', yname='Y', colname='N',
                      col_rot = 270,
                      ax=None):

    equal_bins = np.linspace(0,100, nbins)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    else:
        fig = plt.gcf()

    binsx = pd.unique(np.percentile(x_edges, equal_bins))
    binsy = pd.unique(np.percentile(y_edges, equal_bins))

    colz = ax.imshow(values.T,
                     vmin=col_min, vmax=col_max,
                     cmap=cmap,
                     interpolation='nearest', origin='lower')

    ax.set_xticks(np.linspace(*ax.get_xlim(), nlabels))
    ax.set_xticklabels([x_formatter%f for f in np.percentile(x_edges, np.linspace(0,100, nlabels))])

    ax.set_yticks(np.linspace(*ax.get_ylim(), nlabels))
    ax.set_yticklabels([y_formatter%f for f in np.percentile(y_edges, np.linspace(0,100, nlabels))])

    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    fig.colorbar(colz, ax=ax, label=colname, fraction=0.046, pad=0.04)

def equal_bins(x, nbin):
    """
    Divides a sample x in nbin equal frequencybins
    Input:
        - x:    the sample to divide
        - nbin: the desired number of equal frequency bins
    Returns:
        - numpy array of equal bins
    """
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

def make_datamc_ratio_plot(numerator, denominator, var,
                           weight_col = None,
                           weight_err_col = None,
                           lumi_weight = None, lumi=None,
                           bins=None,
                           text=None, textplace=0.7,
                           axes=None, density=False, show_discrepancy=True,
                           data_label="data", uncertainty_label="Uncertainty band",
                           mc_label="MC", legend_loc="best"):

    if bins is None:
        bins = np.array([1.4,1.6,1.8,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,5.0])
    centers = 0.5*(bins[:-1]+bins[1:])
    widths = -(bins[:-1]-bins[1:])

    if axes is not None:
        ax = axes
    else:
        fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.07}, sharex=True,)

    if len(numerator) == 0 and len(denominator)==0:
        ax[0].text(0.2,0.4, "NO DATA", transform=ax[0].transAxes)
        if not text is None:
            ax[0].text(textplace,1.02, text, transform=ax[0].transAxes)
        return

    #### Calculations
    n_cont_up, _ = np.histogram(denominator[var], 
                            bins = bins, density=density,
                            weights=denominator[lumi_weight]*(denominator[weight_col]+denominator[weight_err_col]))
    n_cont_down, _ = np.histogram(denominator[var], 
                            bins = bins, density=density,
                            weights=denominator[lumi_weight]*(denominator[weight_col]-denominator[weight_err_col]))

    n_cont, _ = np.histogram(denominator[var], 
                             bins = bins, density=density,
                             weights=denominator[lumi_weight]*denominator[weight_col])

    stat_uncertainty_sq, _ = np.histogram(denominator[var], 
                             bins = bins, density=density, 
                             weights=(denominator[lumi_weight]*denominator[weight_col])**2)

    stat_uncertainty = np.sqrt(stat_uncertainty_sq)

    uncertainty_sources = np.array([
        stat_uncertainty,
        0.5*(n_cont_up-n_cont_down),
    ])
    
    if not density:
        uncertainty = np.sqrt((np.sum(uncertainty_sources**2, axis=0)))
        n_off, _, _, _ = bp.errorhist(numerator[var], uncertainty_mode="regular", marker=".",color="k",
                                      bins = bins, ax=ax[0], label=data_label)
    else:
        uncertainty = np.zeros(len(bins)-1)
        n_off, _, _ = bp.hist(numerator[var], density=density, marker=".",color="k",
                                      bins = bins, ax=ax[0], label=data_label)

    ########

    ax[0].bar(centers,
              uncertainty*2,
              bottom=n_cont-uncertainty,
              width=widths, color='gray',
              alpha=0.5, label=uncertainty_label)

    ax[0].hlines(y = n_cont, xmin = bins[:-1], xmax=bins[1:], color='black', ls='dashed', label=mc_label)

    data_distribution = uarray(n_off, np.sqrt(n_off))
    mc_distribution = uarray(n_cont, uncertainty)
    
    mc_distribution[mc_distribution==0] = 1e-9

    data_mc_ratio = data_distribution/mc_distribution
    
    if not density:
        up_errors = sds(data_mc_ratio)
        down_errors = sds(data_mc_ratio)
    else:
        up_errors = np.zeros(len(data_mc_ratio))
        down_errors = np.zeros(len(data_mc_ratio))


    #Draw pull distribution and color area under each

    line,caps,_ = ax[1].errorbar(centers, nvs(data_mc_ratio), xerr = 0.5*widths, yerr=(down_errors, up_errors),
                     fmt='ko',
                     markersize=3,
                     ecolor = 'black')
    _,_,_ = ax[1].errorbar(bins[-2]+0.05,n_off[-1]/n_cont[-1], xerr=0, yerr=[[down_errors[-1]], [up_errors[-1]]], fmt='ko', markersize=3, ecolor='k')

    ax[1].bar(centers,  n_off/n_cont-1, width=widths, color='gray',alpha=0.5, bottom=1)
    ax[1].set_xlim(bins[0]*0.998,bins[-2]*1.002)
    ax[1].set_ylim(0.7,1.3)
    ax[0].set_ylim(-5,1.1*np.max([n_cont_up, n_off]))

    ax[1].set_yticks([0.75,1,1.25])
    if not text is None:
        ax[0].text(textplace,1.02, text, transform=ax[0].transAxes)
    if not lumi is None:
        lumi = rf"$\int \mathcal{{L}}dt =$ {lumi}"
        ax[0].legend(title=lumi, loc=legend_loc)
    else:
        ax[0].legend(loc="legend_loc")

    if not density and show_discrepancy:
        discrepancy = np.sum(np.nan_to_num(n_off/n_cont * (n_off+n_cont)/np.sum(n_off+n_cont), neginf=0, posinf=0))
        ax[0].text(0.05,1.02, f"discrepancy: {discrepancy:.3f}", transform=ax[0].transAxes)
    #discrepancy = scipy.spatial.distance.jensenshannon(n_off, n_cont)
    #discrepancy = np.average(n_off/n_cont)

    ax[1].set_xlabel(var_to_string(var))
    ax[0].set_ylabel("Candidates per bin")
    ax[1].set_ylabel(r"$\frac{\mathrm{DATA}}{MC}$")

