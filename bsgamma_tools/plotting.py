import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    colz = ax.imshow(values[::-1,:],
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

