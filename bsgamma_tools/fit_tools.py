import zfit
import numpy as np
import matplotlib.pyplot as plt

def plot_comp_model(model, data, bins=50, figsize=None, axs = None, spit_vals=False):
    if axs is None:
        fig, (ax, ax_pull) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05}, sharex=True, figsize=figsize)
    else:
        ax = axs[0]
        ax_pull = axs[1]
    for mod, frac in zip(model.pdfs, model.params.values()):
        plot_model(mod, data, bins=bins, scale=frac, plot_data=False, add_pulls=False, axs=ax)
    return plot_model(model, data, bins=bins, axs=(ax,ax_pull), spit_vals=spit_vals)

def plot_model(model, data, bins=50, scale=1, plot_data=True, add_pulls=True, axs=None, figsize=None, spit_vals=False):

    if not add_pulls and axs is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    elif add_pulls and axs is None:
        fig, (ax, ax_pull) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05}, sharex=True, figsize=figsize)
    elif isinstance(axs, list) or isinstance(axs, tuple):
        ax=axs[0]
        ax_pull=axs[1]
    else:
        ax = axs


    nbins = bins

    lower, upper = data.data_range.limit1d
    x = np.linspace(lower, upper, num=1000)  # np.linspace also works
    y = model.pdf(x) * int(data.nevents) / (nbins) * data.data_range.area()
    y *= scale

    ax.plot(x, y)
    data_plot = zfit.run(zfit.z.unstack_x(data))  # we could also use the `to_pandas` method
    #print(len(data_plot))
    #print(data.data_range.area())
    if plot_data:
        n, bins, _ = ax.hist(data_plot, bins=nbins)

        bin_centers = 0.5 * (bins[1:]+ bins[:-1])
        bin_hwidth = 0.5*(bins[1:]- bins[:-1])

        interped_values = model.pdf(bin_centers) * int(data.nevents) / (nbins) * data.data_range.area()
        pulls = (n - interped_values) / (interped_values)**0.5

        #Draw pull distribution and color area under each
        line,caps,_ = ax_pull.errorbar(bin_centers,pulls, xerr = bin_hwidth,
                         fmt='ko',
                         markersize=3,
                         ecolor = 'black')
        ax_pull.bar(bin_centers,pulls,width=2*bin_hwidth,color='gray',alpha=0.5)
        ax.set_xlim(lower, upper)

        if spit_vals:
            return interped_values, n
        else:
            return None

