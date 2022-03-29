import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hepstats.splot import compute_sweights

import b2plot as bp

from scipy.spatial.distance import jensenshannon
import yaml

from bsgamma_tools.helpers import pdg_to_name
from bsgamma_tools.fit_tools import SWeightFit, MbcFit
from bsgamma_tools.constants import safe

plt.style.use('belle2')
golden = (1 + 5 ** 0.5) / 2
fig_width = 10
fig_height = fig_width/golden


def mad(data):
    median = np.nanmedian(data)
    diff = np.abs(data - median)
    mad = np.nanmedian(diff)
    return mad


def calculate_bounds(data, z_thresh=3.5):
    MAD = mad(data)
    median = np.nanmedian(data)
    const = z_thresh * MAD / 0.6745
    return (max(median - const, min(data)), min(median + const, max(data)))

def ratio_error(a,b):
    return np.sqrt(a/b**2 + (a**2/b**3))

def ratio_to_signal(pddf_list, xname='',
                    label_list = None,
                    scale=None, overal_scale = 1,
                    n_bins=30, ranges=(1.4,4),
                    color_list=None, edgecolor='black',
                    yunit='', xunit='', saveas=None, ax=None,
                    annotate_bins=False,
                    secondary_ax=None, secondary_label=''):
    """
     Creates a stacked plot for ratio with signal
    Inputs:
     - pddf (pandas DataFrame): must contain signal, continuum and bbar backgrounds
     - comparison_var (str): variable to plot against, must be acessible in pddf
     - signal_var (str): variable to discriminate signal, must ve acessible in pddf
     - bb_var (str): variable to discriminate bb bar background from continuum, must be acessible in pddf
     - continuum_sup_var_cat (str): an additional cut that can be introduced, for example, a continuum supression cut
     - ranges (float, float): drawing range
     - yunit (str): unit to appear on Y axis title
     - xunit (str): unit to appear on X axis title
     - saveas (str): if not None then name to save as.
     Outputs:
     - plots a figure of a ration of stacked signal, bbar, continuum and ratio between signal/(bbar+cont) bin by bin
    """

    golden = (1 + 5 ** 0.5) / 2
    if ax is None:
        _, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1,golden],'hspace':0.2})
    f = plt.gcf()

    if color_list is None:
        from b2plot.colors import b2helix
        n_stacks = len(pddf_list)
        if n_stacks < 20:
            color_list = b2helix(n_stacks)
        else:
            print('Way too many colours for this palette, consider supplying the colors manually!')
            color_list = b2helix(n_stacks)

    if not isinstance(overal_scale, int) or isinstance(overal_scale, float):
        print("Overal scale is supposed to be a number. Unexpected results might occur when it is not.")

    if isinstance(scale,list):
        if len(scale) == len(pddf_list):
            print("Scaling each pddf differently")
            weights = [[overal_scale * cs] * len(df) for cs, df in zip(scale, pddf_list)]
        else:
            print("Scaling with given weights")
            weights = [[overal_scale * c for c in cs] for cs in scale]
    else:
        print("Scaling all the pddf the same")
        weights = [[overal_scale] * len(df) for df in pddf_list]

    n0,bins0, patches0 = ax[0].hist(pddf_list, n_bins, histtype='stepfilled',stacked=True,
                                    lw=.5, color=color_list, edgecolor=edgecolor, label=label_list,
                                    weights=weights)

    divided_stacks = [n0[0][i]/(np.sqrt(n0[-1][i])) if (n0[-1][i])>0 else 0 for i in range(len(n0[1]))]

    stack_errors = [ratio_error(n0[0][i]*overal_scale*scale[0],np.sqrt(n0[-1][i])) if (n0[-1][i])>0 else 0 for i in range(len(n0[1]))]

    bin_centers = (bins0[1:] + bins0[:-1])/2
    bin_widths  = (bins0[1:] - bins0[:-1])/2
    ax[1].errorbar(bin_centers,divided_stacks,yerr=stack_errors,xerr=bin_widths,fmt='.k')

    if annotate_bins:
        for count, binx, binw in zip(n0.T, bin_centers, bin_widths):
            ax[0].text(binx, count[-1], f'{count[0]:.0f}$\pm$ {np.sqrt(count[0]*scale[0]*overal_scale):.0f}', ha='center', va='bottom')

    if secondary_ax is not None:
        color = np.random.random(3)
        secondary_ax.errorbar(bin_centers,divided_stacks,yerr=stack_errors,fmt='o',color=color,label=secondary_label)

    ax[0].legend(loc='best')
    ranges = (bins[0],bins[-1]) if not ranges else ranges
    ax[0].set_xlim(*ranges)
    ax[0].set_ylabel(f'Candidates/{(ranges[1]-ranges[0])/(len(bins0)-1):.2f} {yunit}')
    ax[1].set_xlabel(f'{xname}, {xunit}')
    ax[1].set_ylabel(f'Sig/Bkg ratio')
    ax[1].set_xlim(*ranges)
    ax[1].set_ylim(bottom=-0.001)
    if secondary_ax is not None:
        secondary_ax.set_xlabel(f'{xname}, {xunit}')
        secondary_ax.set_ylabel(f'Some signficance parameter')
        secondary_ax.set_xlim(*ranges)
        secondary_ax.set_ylim(bottom=-0.001,top=max(max(divided_stacks),secondary_ax.get_ylim()[-1]))
        secondary_ax.legend(loc='best')

    f.align_ylabels(ax)

    if saveas:
        f.savefig(saveas,bbox_inches='tight')

    outputs = {}
    outputs['S'] = n0[0]
    outputs['B'] = n0[-1] - n0[0]
    outputs['S/B'] = outputs['S']/outputs['B']
    outputs['S/sqrt(S+B)'] = np.array(divided_stacks)

    return outputs

def stacked_background(df, by, max_by=10, reverse = True, colors = None, pdgise=True,
                       isolate=None, isolate_name=None, include_other = False, other_name='other', legend_suffix=''):

    pdg_codes = df[by].value_counts().keys()
    if len(pdg_codes)<max_by:
        max_by=len(pdg_codes)+1
    if pdgise:
        pdg_names = [legend_suffix+pdg_to_name(i,True) for i in pdg_codes[:max_by-1]]
    else:
        pdg_names = [legend_suffix+f'{i}' for i in pdg_codes[:max_by-1]]

    top = pdg_codes[:max_by-1]

    if isolate is not None:
        df_isolate = df.query(f'{isolate}')
        df_to_split = df.query(f'~({isolate})')
    else:
        df_to_split = df

    if include_other:
        draw_stack = [df_to_split.query(f'{by}=={code}') for code in top] + [df_to_split[~df_to_split[by].isin(top)]]
        pdg_names += [legend_suffix+other_name]
    else:
        draw_stack = [df_to_split.query(f'{by}=={code}') for code in top]

    if isolate is not None:
        draw_stack.append(df_isolate[draw_col])
        if isolate_name is not None:
            pdg_names.append(legend_suffix+isolate_name)
        else:
            pdg_names.append(legend_suffix+isolate)
    if reverse:
        draw_stack.reverse()
        pdg_names.reverse()
    return draw_stack, pdg_names



def calculate_peaky_codes(sig_df, threshold=0.3, plot=False,
                          save_path=None,
                          suffix=''):
    
    df_to_break_down = sig_df
    stack, names = stacked_background(df_to_break_down, 
                                      by="Btag_mcErrors", 
                                      include_other=False, 
                                      max_by=100, 
                                      pdgise=False)
    
    if plot:
        fig, ax = plt.subplots(1,3, figsize=(20,6))
    
        draw_stack = [s.Btag_Mbc for s in stack]
        draw_stack_unpeaky = []
        draw_stack_peaky = []
        
    reference_stack = df_to_break_down.query('Btag_mcErrors==0').Btag_Mbc
    
    jsdists = {}
    
    names_peaky = []
    names_unpeaky = []
    jsdist_peaky = []
    jsdist_unpeaky = []
    
    bins = np.linspace(5.245,5.291,30)
    reference, _ = np.histogram(reference_stack, bins=bins)

    names = [name.replace('.0','') for name in names]
    
    for n, s in enumerate(stack):
        
        if len(stack)<50:
            continue
        
        test = np.histogram(s.Btag_Mbc, bins=bins)[0]
        distance = jensenshannon(test, reference)
        
        jsdists[names[n]] = distance

        if plot:
            if distance>threshold:
                draw_stack_unpeaky.append(s.Btag_Mbc)
                names_unpeaky.append(names[n]+f' ({distance:.2f})')
                jsdist_unpeaky.append(distance)
            else:
                draw_stack_peaky.append(s.Btag_Mbc)
                names_peaky.append(names[n]+f' ({distance:.2f})')
                jsdist_peaky.append(distance)
            


    if plot:
        
        draw_stack = [pd.concat(draw_stack[:-9])]+draw_stack[-9:] if len(draw_stack)>9 else draw_stack
        label_all = ['other']+names[-9:] if len(names)>9 else names

        draw_stack_peaky = [pd.concat(draw_stack_peaky[:-9])]+draw_stack_peaky[-9:] if len(draw_stack_peaky)>9 else draw_stack_peaky
        label_peaky = ['other']+names_peaky[-9:] if len(names_peaky)>9 else names_peaky

        draw_stack_unpeaky = [pd.concat(draw_stack_unpeaky[:-9])]+draw_stack_unpeaky[-9:] if len(draw_stack_unpeaky)>9 else draw_stack_unpeaky
        label_unpeaky = ['other']+names_unpeaky[-9:] if len(names_unpeaky)>9 else names_unpeaky

    
        
        _,_,_ = bp.stacked(draw_stack,         label = label_all,     bins=bins, ax=ax[0])
        _,_,_ = bp.stacked(draw_stack_peaky,   label = label_peaky,   bins=bins, ax=ax[1])
        _,_,_ = bp.stacked(draw_stack_unpeaky, label = label_unpeaky, bins=bins, ax=ax[2])

        for axis in ax:

            axis.legend(loc='upper left', title='MC error code')
            axis.set_xlabel('tag-$B~M_{bc}$, GeV/$c^2$ ')
            axis.set_xlim((min(bins),max(bins)))

        ax[0].text(0.7,1.01,r'All',transform=ax[0].transAxes)
        ax[1].text(0.7,1.01,'Peaking',transform=ax[1].transAxes)
        ax[2].text(0.7,1.01,'Less peaking',transform=ax[2].transAxes)
        
        if save_path:
            fig.savefig(f'{save_path}/error_code_splitting{suffix}.pdf', bbox_inches='tight')
    
    if save_path:
        with open(f'{save_path}/error_code_distances.yaml', 'w') as file:
            yaml.dump(jsdists, file)
    
    return jsdists

def return_generic_signal(dataframe, charged, matching_codes = None, inclusive = False, resonant = False, name="truth"):

    if matching_codes is None:
        matching_codes = [0]

    if charged:
        codes = safe["Inclusive Xsu modes"]
    else:
        codes = safe["Inclusive Xsd modes"]

    if inclusive:
        codes = codes[:2]
    elif resonant:
        codes = codes[2:]

    dataframe[(dataframe['Bsig_d0_mcpdg'].isin(codes)) & \
              (dataframe['Btag_mcErrors'].isin(matching_codes)) & \
              (dataframe['isSigSideCorrect'] == 1), name] = 1
    
    return dataframe

#################
#### Classes ####
#################

class EGammaSpectrum:

    subtracted = None
    built = None
    sfitted = False
    binned = False
    custom = False
    asymmetric = False

    def __init__(self, original_dataset=None, bins=None):
        #assert 'Btag_Mbc' in original_dataset and 'gamma_EB' in original_dataset, "please give the original dataset that contains Btag_Mbc and gamma_EB columns"
        self.built = False
        if not original_dataset is None:
            self.original_gamma = original_dataset.gamma_EB.values
            self.original_mbc = original_dataset.Btag_Mbc.values
            self.sorter = self.original_mbc.argsort()
        self.bins = bins
        self.bin_centers = 0.5*(bins[:-1]+bins[1:])
        self.bin_width = -0.5*(bins[:-1]-bins[1:])
        print("E gamma spectrum created, please use self.from_bins() or self.from_sfit() to build")

    def from_sfit(self, sfit):
        assert not self.built, "Spectrum is already built. Create a new one!"
        print(type(sfit))
        print(SWeightFit)
        assert isinstance(sfit, SWeightFit), "if building from sweights, please give a fitted SWeightFit"

        self.weights = compute_sweights(sfit.full_fit, sfit.last_fit_data)
        self.weights_crys = self.weights[sfit.yield_crys]
        self.weights_cheb = self.weights[sfit.yield_cheb]
        self.weights_argus = self.weights[sfit.yield_argus]

        self.n_gamma, _  = np.histogram(self.original_gamma, bins=self.bins, weights=self.weights[sfit.yield_crys])

        self.fit_uncertainty = np.zeros(len(self.bins)-1)
        for gamma, weight in zip(self.original_gamma, self.weights[sfit.yield_crys]):
            for n, (min_e, max_e) in enumerate(zip(self.bins[:-1], self.bins[1:])):
                if max_e>gamma>min_e:
                    self.fit_uncertainty[n] += weight**2

        self.fit_uncertainty = np.sqrt(self.fit_uncertainty)

        self.sfitted = True
        self.built = True
        self.subtracted = False

    def from_binfit(self, binfit):
        assert not self.built, "Spectrum is already built. Create a new one!"
        #print(binfit)
        #print(type(binfit))
        #print(MbcFit)
        #print(isinstance(binfit, MbcFit))
        #print(type(binfit)==MbcFit)
        #assert isinstance(binfit, MbcFit), "if building from binned fit, please give a fitted MbcFit"

        self.n_gamma = np.array([count.numpy() for count in binfit.collector['yield_signal']])
        e_u = np.array([binfit.last_result.params[v]['minuit_minos']['upper'] for v in binfit.collector['yield_signal']])
        e_l = np.array([binfit.last_result.params[v]['minuit_minos']['lower'] for v in binfit.collector['yield_signal']])
        self.fit_uncertainty = (e_l, e_u)

        self.binned = True
        self.built = True
        self.subtracted = False
        self.asymmetric = True
    
    def from_values(self, values, uncertainties):
        self.n_gamma = np.array(values)
        self.fit_uncertainty = np.array(uncertainties)
        self.custom = True

    def peak_scale(self):
        raise NotImplementedError
        peak_mbc = zfit.Space('mbc', (5.27, max_mbc+0.000))

        peak_scales = [pdf.numeric_integrate(peak_mbc).numpy()[0] for pdf in fit.collector['signal']]

    def plot_spectrum(self, target=None):
        if self.sfitted:
            self.plot_sweighted_spectrum(target)

        if self.binned or self.custom:
            self.plot_binned_spectrum(target)

    def plot_binned_spectrum(self, target=None):

        fig, ax = plt.subplots(1, 1)

        ax.errorbar(self.bin_centers, self.n_gamma, yerr=np.array(self.fit_uncertainty), fmt='k.')

        if not target is None:
            ax.hist(target, bins=self.bins, histtype="step", label="target histogram", color='r', lw=1.5)
        ax.set_xlabel("$E^B_{\gamma}$, GeV")
        ax.set_ylabel("Crystal Ball yield")
        ax.set_xlim(1.4, 3.5)
        ax.set_ylim(-5, )
        ax.legend(fontsize=15)

    def plot_sweighted_spectrum(self, target=None):
        fig, axs = plt.subplots(1, 2, figsize=(16,6))

        axs[0].plot(self.original_mbc[self.sorter], self.weights_crys[self.sorter], label = '$w_\\mathrm{CB}$')
        axs[0].plot(self.original_mbc[self.sorter], self.weights_cheb[self.sorter], label = '$w_\\mathrm{Chebyshev}$')
        axs[0].plot(self.original_mbc[self.sorter], self.weights_argus[self.sorter], label = '$w_\\mathrm{Argus}$')
        axs[0].plot(self.original_mbc[self.sorter], self.weights_argus[self.sorter] +\
                                                    self.weights_cheb[self.sorter]  +\
                                                    self.weights_crys[self.sorter], label = '$\Sigma w$')

        axs[0].axhline(0, color="0.5")
        axs[0].legend(fontsize=15)
        axs[0].set_xlabel("$M_{bc}$, GeV")
        axs[0].set_xlim(5.245, np.max(self.original_mbc)) 

        axs[1].bar(self.bin_centers, self.n_gamma, self.bin_width*2, align='center', alpha=.5, label='weighted histogram')
        if not target is None:
            axs[1].hist(target, bins=self.bins, histtype="step", label="target histogram", color='r', lw=1.5)
        axs[1].set_xlabel("$E^B_{\gamma}$, GeV")
        axs[1].set_xlim(1.4, 3.5)
        axs[1].set_ylim(-5, )
        axs[1].legend(fontsize=15);

    def __sub__(self, other):
        print(type(other))
        if isinstance(other, float) or isinstance(other, int):
            print('subtracting constant value')
            raise NotImplementedError
        if isinstance(other, EGammaSpectrum):
            print('subtracting spectrum')
            new_value = np.array(self.n_gamma) - np.array(other.n_gamma)
            if self.asymmetric:
                if other.asymmetric:
                    new_uncertainty_u = np.sqrt(self.fit_uncertainty[1]**2 + other.fit_uncertainty[1]**2)
                    new_uncertainty_l = np.sqrt(self.fit_uncertainty[0]**2 + other.fit_uncertainty[0]**2)
                    new_uncertainty = (new_uncertainty_l, new_uncertainty_u)
                else:
                    new_uncertainty_u = np.sqrt(self.fit_uncertainty[1]**2 + other.fit_uncertainty**2)
                    new_uncertainty_l = np.sqrt(self.fit_uncertainty[0]**2 + other.fit_uncertainty**2)
                    new_uncertainty = (new_uncertainty_l, new_uncertainty_u)
            else:
                if other.asymmetric:
                    new_uncertainty_u = np.sqrt(self.fit_uncertainty**2 + other.fit_uncertainty[1]**2)
                    new_uncertainty_l = np.sqrt(self.fit_uncertainty**2 + other.fit_uncertainty[0]**2)
                    new_uncertainty = (new_uncertainty_l, new_uncertainty_u)
                else:
                    new_uncertainty = np.sqrt(self.fit_uncertainty**2 + other.fit_uncertainty**2)
            new_spectrum = EGammaSpectrum(bins=self.bins)
            new_spectrum.from_values(new_value, new_uncertainty)

            return new_spectrum

        #if isinstance(other, list) or isinstance(other, np.array):
        #    print('subtracting a list of entries')
        #    raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            print('subtracting constant value')
            new_value = np.array(self.n_gamma) * other

            if self.asymmetric:
                new_uncertainty_u = self.fit_uncertainty[1] * other
                new_uncertainty_l = self.fit_uncertainty[0] * other
                new_uncertainty = (new_uncertainty_l, new_uncertainty_u)
            else:
                new_uncertainty = np.sqrt(self.fit_uncertainty**2 * other)

            new_spectrum = EGammaSpectrum(bins=self.bins)
            new_spectrum.from_values(new_value, new_uncertainty)
            if self.asymmetric:
                new_spectrum.asymmetric = True

            return new_spectrum

        if isinstance(other, EGammaSpectrum):
            print('subtracting spectrum')
            raise NotImplementedError
        if isinstance(other, list) or isinstance(other, np.array):
            print('subtracting a list of entries')
            raise NotImplementedError
