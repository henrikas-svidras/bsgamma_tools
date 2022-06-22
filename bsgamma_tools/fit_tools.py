import zfit
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import b2plot as bp
import scipy as sp

import itertools


def plot_comp_model(model, data, nbins=50, as_bp=False, figsize=None, 
                                 axs=None, spit_vals=False, normalise=True, 
                                 add_chi2=False, weights=None, *args,**kwargs):
    if axs is None:
        fig, (ax, ax_pull) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05}, 
                                                sharex=True, figsize=figsize)
    else:
        ax = axs[0]
        ax_pull = axs[1]
    line_styles = ["dashed", "dashdot", "dotted"]
    colors= bp.colors.groovebox
    main_fit_color = bp.colors.b2cm[0]
    for n, (mod, frac) in enumerate(zip(model.pdfs, model.params.values())):
        plot_model(mod, data, nbins=nbins, scale=frac,as_bp=as_bp, plot_data=False, add_pulls=False, normalise=normalise, axs=ax, weights=weights, linestyle=line_styles[n], color=colors[n])
    return plot_model(model, data, nbins=nbins, as_bp=as_bp, normalise = normalise, axs=(ax,ax_pull), spit_vals=spit_vals, weights=weights, add_chi2=add_chi2, color=main_fit_color, *args,**kwargs)

def plot_model(model, data, nbins=50, scale=1, as_bp = False,plot_data=True, add_pulls=True, axs=None, 
               figsize=None, spit_vals=False, normalise=True, add_chi2 = False, weights=None,
               linestyle="solid", color="black"):

    if not add_pulls and axs is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    elif add_pulls and axs is None:
        fig, (ax, ax_pull) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05}, sharex=True, figsize=figsize)
    elif isinstance(axs, list) or isinstance(axs, tuple):
        ax=axs[0]
        ax_pull=axs[1]
    else:
        ax = axs



    lower, upper = data.data_range.limit1d
    if isinstance(nbins, int):
        bins = np.linspace(lower,upper, nbins+1)
    else:
        bins = nbins
    x = np.linspace(lower, upper, num=1000)  # np.linspace also works
    if normalise:
        y = model.pdf(x) * int(data.nevents) / (len(bins)-1) * data.data_range.area()
    else:
        y = model.pdf(x) * int(model.get_yield().numpy()) / (len(bins)-1) * data.data_range.area()

    y *= scale

    ax.plot(x, y, linestyle=linestyle, color=color)

    data_plot = zfit.run(zfit.z.unstack_x(data))  # we could also use the `to_pandas` method
    if plot_data:
        if as_bp:
          n, _, _, _ = bp.errorhist(data_plot, bins=bins, ax=ax, weights=weights, color="black", uncertainty_mode="regular")
        else:
          n, bins, _ = ax.hist(data_plot, bins=bins, weights=weights)

        bin_centers = 0.5 * (bins[1:]+ bins[:-1])
        bin_hwidth = 0.5*(bins[1:]- bins[:-1])

        interped_values = model.pdf(bin_centers) * int(data.nevents) / (len(bins)-1) * data.data_range.area()
        pulls = (n - interped_values) / (np.sqrt(n)+1e-9)

        #Draw pull distribution and color area under each
        line,caps,_ = ax_pull.errorbar(bin_centers,pulls, xerr = bin_hwidth,
                         fmt='ko',
                         markersize=3,
                         ecolor = 'black')
        ax_pull.bar(bin_centers,pulls,width=2*bin_hwidth,color='gray',alpha=0.5)
        ax.set_xlim(lower, upper)
        ax.set_ylim(0,)
        ax.set_xticks([])
        ax_pull.set_xlim(lower, upper)

        if add_chi2:
            st = sp.stats.chisquare(n,interped_values)
            chi2 = st.statistic
            pval = st.pvalue

            ax.text(0.1,0.9,f"$\chi^2={chi2:.2f}$",transform=ax.transAxes)
            ax.text(0.1,0.8,f"$p={pval:.3f}$",transform=ax.transAxes)

        pull_limit = np.nanquantile(np.abs(pulls[n>0]),0.99)
        if np.isnan(pull_limit)  or np.isinf(pull_limit):
            pull_limit = 1

        ax_pull.set_ylim(-pull_limit*1.1, pull_limit*1.1)

        if spit_vals:
            return interped_values, n, bin_hwidth
        else:
            return None

import functools
import os
import bsgamma_tools
from bsgamma_tools.helpers import random_string
from collections import defaultdict
from time import time

from zfit.pdf import CrystalBall
from zfit.pdf import Chebyshev
from zfit_physics.pdf import Argus

import pandas as pd

from prettytable import PrettyTable
def pretty_print_result(result, params=None, quick=False, quiet=False):
    print("creating pretty table")
    x = PrettyTable()
    if quick=='zfit_error':
        print("getting zfit errors")
        errors = result.errors(method='zfit_error')
    elif not quick:
        print("getting minos errors")
        errors = result.errors(method='minuit_minos')
    elif quick=='approx':
        errors = None
        print("skipping minos, because in quick mode")
        #print("getting approx errors")
        #errors_approx = result.hesse(method="approx")
        print("getting approx errors")
        errors_hesse = result.hesse(method="approx", name="quick_error")
        print("filling params")

    elif quick=='hesse_np':
        errors = None
        print("skipping minos, because in quick mode")
        #print("getting approx errors")
        #errors_approx = result.hesse(method="approx")
        print("getting hesse_np errors")
        errors_hesse = result.hesse(method="hesse_np", name="quick_error")
        print("filling params")
    else:
        errors = None
        print("skipping minos, because in quick mode")
        #print("getting approx errors")
        #errors_approx = result.hesse(method="approx")
        print("getting hesse errors")
        errors_hesse = result.hesse(method="minuit_hesse", name="quick_error")
        print("filling params")
    for param, val in result.params.items():
        if quick:
            x.field_names = ["Parameter name", "Parameter value", "hesse error"]
            x.add_row([param.name, val['value'], errors_hesse[param]['error']])
        else:
            x.field_names = ["Parameter name", "Parameter value", "Low error", "Up error"]
            x.add_row([param.name, val['value'], errors[0][param]['lower'], errors[0][param]['upper']])

    x.float_format = '.3'
    print("fit results done and done")
    if not quiet:
        print(x)
    return errors

class MbcFit:

    bin_list = np.array([1.4,1.6,1.8,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,5.0])
    bin_centers = 0.5*(bin_list[1:]+bin_list[:-1])
    bin_strings = None
    cheb_bin_groups = [11]
    argus_bin_groups = [11]

    collector = None

    last_result = None

    full_result = None

    ID = None

    lower=None

    lowsig=None

    c_floaty=False

    sampler = None

    ignored_chebyshev = False

    def __init__(self, df_peak, df_combinatorial, df_continuum, obs, fit_var = "Btag_Mbc",
                 cheb_bin_groups = None, argus_bin_groups = None, weights_col = None, bin_list = None, low_c = 0,
                 lower=None, lowsig=None, c_floaty=False, cheb_floaty=False, m_floaty=True, m0_init=None, minimizer=None, ignore_cheb=False):
        self.df_peak = df_peak
        self.df_combinatorial = df_combinatorial
        self.df_continuum = df_continuum
        self.df_total = pd.concat([df_peak, df_combinatorial, df_continuum])
        self.df_bkg = pd.concat([df_combinatorial, df_continuum])

        self.weights_col = weights_col

        self.fit_var = fit_var
        self.max_mbc = max(self.df_total[fit_var])
        self.m0_init = m0_init if m0_init else self.max_mbc

        self.bin_list = bin_list if bin_list is not None else self.bin_list
        self.bin_strings = self.create_bins(self.bin_list)

        self.ID = random_string()

        self.collector = defaultdict(lambda:[])

        self.lower = lower
        self.lowsig = lowsig


        self.c_floaty = c_floaty
        self.cheb_floaty = cheb_floaty
        self.m_floaty = m_floaty

        self.minimizer = zfit.minimize.Minuit() if minimizer is None else minimizer

        self.obs = obs

        self.argus_bin_groups = [11] if argus_bin_groups is None else argus_bin_groups
        self.cheb_bin_groups = [11] if cheb_bin_groups is None else cheb_bin_groups
        self.low_c = low_c


    def timeit(func):
        functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            ts = time()
            result = func(*args, **kwargs)
            te = time()
            print(f'{func.__name__} took {te-ts:.1f} seconds to run')
            return result
        return wrapper_func


    def create_bins(self, input_list):
        beanlist = []
        self.var = 'gamma_EB'
        for a, b in zip(input_list[:-1], input_list[1:]):
            beanlist.append(f'{b}>{self.var}>={a}')
        self.bin_list = input_list
        self.bin_centers = 0.5*(self.bin_list[1:]+self.bin_list[:-1])
        return beanlist

    def init_cb_params(self):
        self.mu = zfit.Parameter('mu_'+self.ID, 5.275, 5.27, 5.29, step_size=0.001, floating=True)
        self.sigma = zfit.Parameter('sigma_'+self.ID, 1e-3, 3e-5, 3.5e-3, step_size=0.5e-5, floating=True)
        self.alfa = zfit.Parameter('alfa_'+self.ID, 0.2, 0, 20, step_size=0.01, floating=True)
        self.npar = zfit.Parameter('npar_'+self.ID, 15, 0.1, 100, step_size=0.01, floating=True)

    def init_argus_params(self):
        self.p = zfit.Parameter('p_'+self.ID, 0.5, -1, 1, step_size=0.001, floating=False)


    @timeit
    def preparatory_signal_peak_fit(self, quick=False): 
        print("update?")
        self.init_cb_params()

        for n, cut in enumerate(self.bin_strings):
            cut_df = self.df_peak.query(cut)
            deef = cut_df[self.fit_var]
            weights = None if self.weights_col is None else cut_df[self.weights_col]
            data = zfit.Data.from_pandas(obs=self.obs, df=deef, weights=weights)

            crys_unext = CrystalBall(obs=self.obs, mu=self.mu, 
                                     sigma=self.sigma, 
                                     alpha=self.alfa,
                                     n=self.npar,  
                                     name=f'cb_{n}_'+self.ID)

            yield_crys = zfit.Parameter(f'yield_crys_{n}_'+self.ID, len(deef), lower=self.lowsig, floating=True)
            crys = crys_unext.create_extended(yield_crys)
            self.collector['data_signal'].append(data)
            self.collector['signal'].append(crys)
            self.collector['yield_signal'].append(yield_crys)

        crys_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.collector['signal'], 
                                                 data=self.collector['data_signal'])
        crys_result = self.minimizer.minimize(crys_nll)

        errors = pretty_print_result(crys_result, quick=quick)

        if not quick:
            errors, new_result = errors
        else:
            new_result = None

        if crys_result.valid or not new_result:
            self.crys_result = crys_result
        else:
            self.crys_result = new_result

        self.mu.floating = False
        self.sigma.floating = False
        self.alfa.floating = False
        self.npar.floating = False


    @timeit    
    def preparatory_cheb_fit(self, quick=False):
        counter = 0
        for enn, n_bin_group in enumerate(self.cheb_bin_groups):
            c0 = zfit.Parameter(f'c0_{enn}_'+self.ID, 1, step_size=0.001, floating=False)
            c1 = zfit.Parameter(f'c1_{enn}_'+self.ID, 0, step_size=0.001)
            c2 = zfit.Parameter(f'c2_{enn}_'+self.ID, 0, step_size=0.001)
            c3 = zfit.Parameter(f'c3_{enn}_'+self.ID, 0, step_size=0.001)
            c4 = zfit.Parameter(f'c4_{enn}_'+self.ID, 0, step_size=0.001)
            c5 = zfit.Parameter(f'c5_{enn}_'+self.ID, 0, step_size=0.001)
            print(f"Group {enn}")
            for n, cut in enumerate(self.bin_strings[counter:counter+n_bin_group]):
                print(f"Including {cut}")
                cut_df = self.df_combinatorial.query(cut)
                deef = cut_df[self.fit_var]
                weights = None if self.weights_col is None else cut_df[self.weights_col]
                data = zfit.Data.from_pandas(obs=self.obs, df=deef, weights=weights)

                cheb_unext = Chebyshev(obs=self.obs, coeffs=[c1,c2,c3,c4,c5], coeff0=c0,
                                       name=f'cheb_{counter+n}_'+self.ID)

                yield_cheb = zfit.Parameter(f'yield_cheb_{counter+n}_'+self.ID,len(deef), lower=self.lower,
                                            floating=True)
                cheb = cheb_unext.create_extended(yield_cheb)

                self.collector['data_cheb'].append(data)
                self.collector['cheb'].append(cheb)
                self.collector['yield_cheb'].append(yield_cheb)

            counter+=n_bin_group
            self.collector['cheb_pars'].append(c0)
            self.collector['cheb_pars'].append(c1)
            self.collector['cheb_pars'].append(c2)
            self.collector['cheb_pars'].append(c3)
            self.collector['cheb_pars'].append(c4)
            self.collector['cheb_pars'].append(c5)

        cheb_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.collector['cheb'], 
                                                 data=self.collector['data_cheb'])
        cheb_result = self.minimizer.minimize(cheb_nll)

        errors = pretty_print_result(cheb_result, quick=quick)

        if not quick:
            errors, new_result = errors
        else:
            new_result = None

        if cheb_result.valid or not new_result:
            self.cheb_result = cheb_result
        else:
            self.cheb_result = new_result

    @timeit
    def preparatory_argus_fit(self, quick=False):

        self.init_argus_params()

        counter = 0

        for enn, n_bin_group in enumerate(self.argus_bin_groups):
            c = zfit.Parameter(f'cargus_{enn}_'+self.ID, -40, upper=self.low_c, floating=True)
            if not self.m_floaty:
                m0 = zfit.Parameter(f'm0_{enn}_'+self.ID, self.m0_init, floating=self.m_floaty)
            else:
                m0 = zfit.Parameter(f'm0_{enn}_'+self.ID, self.m0_init, self.max_mbc-0.003, self.max_mbc+0.001, floating=self.m_floaty)
            print(f"Group {enn}")
            for n, cut in enumerate(self.bin_strings[counter:counter+n_bin_group]):
                print(f"Including {cut}")

                cut_df = self.df_continuum.query(cut)
                deef = cut_df[self.fit_var]
                weights = None if self.weights_col is None else cut_df[self.weights_col]
                data = zfit.Data.from_pandas(obs=self.obs, df=deef, weights=weights)


                argus_unext = Argus(obs=self.obs,
                                    m0=m0,
                                    p=self.p,
                                    c=c, name=f'argus_{counter+n}_'+self.ID)


                yield_argus = zfit.Parameter(f'yield_argus_{counter+n}_'+self.ID, len(deef), lower=self.lower,
                                             floating=True)

                argus = argus_unext.create_extended(yield_argus)

                self.collector['data_argus'].append(data)
                self.collector['argus'].append(argus)
                self.collector['shared_c'].append(c)
                self.collector['shared_m0'].append(m0)

                self.collector['yield_argus'].append(yield_argus)
                #c_standalone = zfit.Parameter(f'c_{counter+n}_standalone_'+self.ID, -40,step_size=0.001)
                #standalone_argus = Argus(obs=self.obs, m0=self.m0, 
                #                         p=self.p, 
                #                         c=c_standalone, name='Standalone Argus_'+self.ID)
                #standalone_argus_nll = zfit.loss.UnbinnedNLL(model=standalone_argus, data=data)
                #standalone_argus_result = self.minimizer.minimize(standalone_argus_nll)
                #standalone_errors = standalone_argus_result.hesse(method="approx")
                #self.collector['c_standalone'].append(c_standalone.numpy())
                #self.collector['c_standalone_err'].append(standalone_errors[c_standalone]['error'])
            counter += n_bin_group

        argus_nll = zfit.loss.ExtendedUnbinnedNLL(model=np.array(self.collector['argus']),
                                                  data=np.array(self.collector['data_argus']))
        argus_result = self.minimizer.minimize(argus_nll)

        errors = pretty_print_result(argus_result, quick=quick)

        if not quick:
            errors, new_result = errors
        else:
            new_result = None

        if argus_result.valid or not new_result:
            self.argus_result = argus_result
        else:
            self.argus_result = new_result

    def prefit(self, quick=False):
        print('Sig peak')
        self.preparatory_signal_peak_fit(quick)
        print('Argus')
        self.preparatory_argus_fit(quick)
        print('Chebi')
        self.preparatory_cheb_fit(quick)
        print('Done')

        for cpar in self.collector['shared_c']:
            cpar.floating=self.c_floaty

        for mpar in self.collector['shared_m0']:
            mpar.floating=self.m_floaty


        for par in self.collector['cheb_pars']:
            par.floating = self.cheb_floaty


        print('registering prefit values')
        self.save_prefit_values()

    def save_last_values(self): 
        self.collector['last_yield_argus'] = [val.numpy() for val in self.collector['yield_argus']]
        self.collector['last_yield_cheb'] = [val.numpy() for val in self.collector['yield_cheb']]
        self.collector['last_yield_signal'] = [val.numpy() for val in self.collector['yield_signal']]
        self.collector['last_c'] = [val.numpy() for val in self.collector['shared_c']]
        self.collector['last_m0'] = [val.numpy() for val in self.collector['shared_m0']]

    def restore_last_values(self):
        for n, last_val in enumerate(self.collector['last_yield_argus']):
            self.collector['yield_argus'][n].set_value(last_val)
        for n, last_val in enumerate(self.collector['last_yield_cheb']):
            if self.ignored_chebyshev:
                continue
            self.collector['yield_cheb'][n].set_value(last_val)
        for n, last_val in enumerate(self.collector['last_yield_signal']):
            self.collector['yield_signal'][n].set_value(last_val)
        for n, last_val in enumerate(self.collector['last_c']):
            self.collector['shared_c'][n].set_value(last_val)
        for n, last_val in enumerate(self.collector['last_m0']):
            self.collector['shared_m0'][n].set_value(last_val)

    def save_prefit_values(self): 
        self.collector['prefit_yield_argus'] = [val.numpy() for val in self.collector['yield_argus']]
        self.collector['prefit_yield_cheb'] = [val.numpy() for val in self.collector['yield_cheb']]
        self.collector['prefit_yield_signal'] = [val.numpy() for val in self.collector['yield_signal']]
        self.collector['prefit_c'] = [val.numpy() for val in self.collector['shared_c']]
        self.collector['prefit_m0'] = [val.numpy() for val in self.collector['shared_m0']]

    def restore_prefit_values(self):
        for n, prefit_val in enumerate(self.collector['prefit_yield_argus']):
            self.collector['yield_argus'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['prefit_yield_cheb']):
            if self.ignored_chebyshev:
                continue
            self.collector['yield_cheb'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['prefit_yield_signal']):
            self.collector['yield_signal'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['prefit_c']):
            self.collector['shared_c'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['prefit_m0']):
            self.collector['shared_m0'][n].set_value(prefit_val)

    def save_final_values(self): 
        self.collector['final_yield_argus'] = [val.numpy() for val in self.collector['yield_argus']]
        self.collector['final_yield_cheb'] = [val.numpy() for val in self.collector['yield_cheb']]
        self.collector['final_yield_signal'] = [val.numpy() for val in self.collector['yield_signal']]
        self.collector['final_c'] = [val.numpy() for val in self.collector['shared_c']]
        self.collector['final_m0'] = [val.numpy() for val in self.collector['shared_m0']]

    def restore_final_values(self):
        for n, prefit_val in enumerate(self.collector['final_yield_argus']):
            self.collector['yield_argus'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['final_yield_cheb']):
            if self.ignored_chebyshev:
                continue
            self.collector['yield_cheb'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['final_yield_signal']):
            self.collector['yield_signal'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['final_c']):
            self.collector['shared_c'][n].set_value(prefit_val)
        for n, prefit_val in enumerate(self.collector['final_m0']):
            self.collector['shared_m0'][n].set_value(prefit_val)

    def ignore_chebyshev_pdf(self):
        self.ignored_chebyshev = True
        for par in self.collector['yield_cheb']:
            par.floating=False
            par.set_value(0)

    @timeit
    def perform_mbc_fit(self, dataset=None, quick=False):

        for n, cut in enumerate(self.bin_strings):
            print(f"Including {cut}")
            cut_df = self.df_total.query(cut)
            deef = cut_df[self.fit_var].dropna()
            weights = None if self.weights_col is None else cut_df[self.weights_col]
            data = zfit.Data.from_pandas(obs=self.obs, df=deef, weights=weights)

            full = zfit.pdf.SumPDF([self.collector['argus'][n], 
                                    self.collector['cheb'][n], 
                                    self.collector['signal'][n]])

            self.collector['data_full'].append(data)
            self.collector['full'].append(full)


        full_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.collector['full'], 
                                                 data=self.collector['data_full'])
        full_result = self.minimizer.minimize(full_nll)

        errors = pretty_print_result(full_result, quick=quick)

        if not quick:
            errors, new_result = errors
        else:
            new_result = None

        if full_result.valid or not new_result:
            self.last_result = full_result
            self.full_result = full_result
        else:
            self.last_result = new_result
            self.full_result = new_result

        self.save_final_values()

        self.fitted = True
        self.last_result = full_result
        self.full_result = full_result
        self.last_fit_data = self.collector['data_full']
        self.full_fit_data = self.collector['data_full']
        for n, cut in enumerate(self.bin_strings):
            self.collector['sampler'].append(self.collector['full'][n].create_sampler(n=20000, fixed_params=True))

        return full_result, type(full_result)

    @timeit
    def perform_toy_mbc_fit(self, scale=1, dataset = None, weights_col = None, restore_default=True, quick=False, quiet=False, no_error=False):
        fit_data = []
        for n, cut in enumerate(self.bin_strings):
            if dataset is None:
                slice_n = self.collector['data_full'][n].nevents.numpy()
                self.collector['sampler'][n].resample(n=int(slice_n*scale))
                fit_data.append(self.collector['sampler'][n])
            else:
                print('using input data')
                weights = None if weights_col is None else dataset.query(cut)[weights_col]
                fit_data.append(zfit.Data.from_pandas(df=dataset.query(cut)[self.fit_var].dropna(), 
                                                      obs=self.obs,
                                                      weights=weights))
        if restore_default:
            self.restore_prefit_values()

        full_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.collector['full'], 
                                                 data=fit_data)
        full_result = self.minimizer.minimize(full_nll)

        if not no_error:

            errors = pretty_print_result(full_result, quick=quick, quiet=quiet)

            if not quick:
                errors, new_result = errors
            else:
                new_result = None

        if full_result.valid or no_error or not new_result:
            self.last_result = full_result
        else:
            self.last_result = new_result

        self.last_fit_data = fit_data

        return self.last_result, fit_data

    def plot_full_fit(self, bin_number=None, plot_data=True, text_shift=0, write_yields=True):
        assert not self.last_result is None, "Must be fitted to be able to plot full result. Call MbcFit.fit() at least once."
        #assert False, "self.collector['data'] has to be changed"
        if bin_number is None:
            self.general_plotter(self.collector['full'], self.last_fit_data, composite=True, plot_data=plot_data)
        else:
            self.single_bin_plotter(self.collector['full'], self.last_fit_data, composite=True, bin_number=bin_number, plot_data=plot_data, text_shift=text_shift, write_yields=write_yields)

    def plot_peak_pdf(self, normalise=False, bin_number=None, plot_data=True, text_shift=0):
        if bin_number is None:
            self.general_plotter(self.collector['signal'], self.collector['data_signal'], normalise=normalise, plot_data=plot_data)
        else:
            self.single_bin_plotter(self.collector['signal'], self.collector['data_signal'], normalise=normalise, bin_number=bin_number, plot_data=plot_data, text_shift=text_shift)

    def plot_cheb_pdf(self, normalise=False, bin_number=None, plot_data=True, text_shift=0):
        if bin_number is None:
            self.general_plotter(self.collector['cheb'], self.collector['data_cheb'], normalise=normalise, plot_data=plot_data)
        else:
            self.single_bin_plotter(self.collector['cheb'], self.collector['data_cheb'], normalise=normalise, bin_number=bin_number, plot_data=plot_data, text_shift=text_shift)

    def plot_argus_pdf(self, normalise=False, bin_number=None, plot_data=True, text_shift=0):
        if bin_number is None:
            self.general_plotter(self.collector['argus'], self.collector['data_argus'], normalise=normalise, plot_data=plot_data)
        else:
            self.single_bin_plotter(self.collector['argus'], self.collector['data_argus'], normalise=normalise, bin_number=bin_number, plot_data=plot_data, text_shift=text_shift)


    def plot_bkg_pdf(self, normalise=False, bin_number=None, plot_data=True, text_shift=0):
        bkg_pdf = [zfit.pdf.SumPDF([arg, cheb]) for arg, cheb in zip(self.collector['argus'], self.collector['cheb'])]
        bkg_data = [zfit.Data.from_pandas(self.df_bkg.query(cut)[self.fit_var], obs=self.obs) for cut in self.bin_strings]
        if bin_number is None:
            self.general_plotter(bkg_pdf, bkg_data, normalise=normalise, plot_data=plot_data)
        else:
            self.single_bin_plotter(bkg_pdf, bkg_data, normalise=normalise, bin_number=bin_number, plot_data=plot_data, text_shift=text_shift)


    def general_plotter(self, models, datasets, normalise=True, composite=False, plot_data=True):
        fig, axs = plt.subplots(12,2,figsize=(18,30), gridspec_kw={'height_ratios': [1.618**2,1]*6, 'hspace':0.05})
        ax = axs.flatten()
        if composite:
            plotter = plot_comp_model
        else:
            plotter = plot_model
        for n, cut in enumerate(self.bin_strings):
            up_ax, down_ax = (ax[4*(n//2)+n%2], ax[4*(n//2)+n%2+2])

            _,_,bin_widhts = plotter(models[n], 
                                     datasets[n], plot_data=plot_data,
                                     as_bp=True, normalise=normalise,
                                     axs=(up_ax, down_ax), spit_vals=True)
            if self.last_result is not None:
                signal_estimate = self.last_result.params[self.collector['yield_signal'][n]]["value"]

                signal_count = datasets[n].nevents.numpy()
                if not self.ignored_chebyshev:
                    cheb_estimate = self.last_result.params[self.collector['yield_cheb'][n]]["value"]
                argus_estimate = self.last_result.params[self.collector['yield_argus'][n]]["value"]
                if "minuit_minos" in self.last_result.params[self.collector['yield_signal'][n]]:
                    signal_up_error = self.last_result.params[self.collector['yield_signal'][n]]["minuit_minos"]["upper"]
                    signal_low_error = self.last_result.params[self.collector['yield_signal'][n]]["minuit_minos"]["lower"]
                    argus_up_error = self.last_result.params[self.collector['yield_argus'][n]]["minuit_minos"]["upper"]
                    argus_low_error = self.last_result.params[self.collector['yield_argus'][n]]["minuit_minos"]["lower"]
                    if not self.ignored_chebyshev:
                        cheb_up_error = self.last_result.params[self.collector['yield_cheb'][n]]["minuit_minos"]["upper"]
                        cheb_low_error = self.last_result.params[self.collector['yield_cheb'][n]]["minuit_minos"]["lower"]
                    #pull = (signal_estimate - signal_count)/signal_error

                    up_ax.text(0.05, 0.9, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm^{{{signal_up_error:.0f}}}_{{{-signal_low_error:.0f}}}'+'$', 
                               transform=up_ax.transAxes)
                    up_ax.text(0.05, 0.8, 'Argus: $'+rf'{argus_estimate:.0f}\pm^{{{argus_up_error:.0f}}}_{{{-argus_low_error:.0f}}}'+'$', 
                               transform=up_ax.transAxes)
                    if not self.ignored_chebyshev:
                        up_ax.text(0.05, 0.7, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm^{{{cheb_up_error:.0f}}}_{{{-cheb_low_error:.0f}}}'+'$', 
                                   transform=up_ax.transAxes)
                elif "zfit_error" in self.last_result.params[self.collector['yield_signal'][n]]:
                    signal_up_error = self.last_result.params[self.collector['yield_signal'][n]]["zfit_error"]["upper"]
                    signal_low_error = self.last_result.params[self.collector['yield_signal'][n]]["zfit_error"]["lower"]
                    argus_up_error = self.last_result.params[self.collector['yield_argus'][n]]["zfit_error"]["upper"]
                    argus_low_error = self.last_result.params[self.collector['yield_argus'][n]]["zfit_error"]["lower"]
                    if not self.ignored_chebyshev:
                        cheb_up_error = self.last_result.params[self.collector['yield_cheb'][n]]["zfit_error"]["upper"]
                        cheb_low_error = self.last_result.params[self.collector['yield_cheb'][n]]["zfit_error"]["lower"]
                    #pull = (signal_estimate - signal_count)/signal_error

                    up_ax.text(0.05, 0.9, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm^{{{signal_up_error:.0f}}}_{{{-signal_low_error:.0f}}}'+'$', 
                               transform=up_ax.transAxes)
                    up_ax.text(0.05, 0.8, 'Argus: $'+rf'{argus_estimate:.0f}\pm^{{{argus_up_error:.0f}}}_{{{-argus_low_error:.0f}}}'+'$', 
                               transform=up_ax.transAxes)
                    if not self.ignored_chebyshev:
                        up_ax.text(0.05, 0.7, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm^{{{cheb_up_error:.0f}}}_{{{-cheb_low_error:.0f}}}'+'$', 
                                   transform=up_ax.transAxes)

                else:
                    signal_error = self.last_result.params[self.collector['yield_signal'][n]]["quick_error"]["error"]
                    argus_error = self.last_result.params[self.collector['yield_argus'][n]]["quick_error"]["error"]
                    if not self.ignored_chebyshev:
                        cheb_error = self.last_result.params[self.collector['yield_cheb'][n]]["quick_error"]["error"]
                    #pull = (signal_estimate - signal_count)/signal_error

                    up_ax.text(0.05, 0.9, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm {signal_error:.0f}'+'$', \
                               transform=up_ax.transAxes)
                    up_ax.text(0.05, 0.8, 'Argus: $'+rf'{argus_estimate:.0f}\pm {argus_error:.0f}'+'$', \
                               transform=up_ax.transAxes)
                    if not self.ignored_chebyshev:
                        up_ax.text(0.05, 0.7, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm{cheb_error:.0f}'+'$', \
                                   transform=up_ax.transAxes)

                up_ax.text(0.05, 0.55, f'N datapoints: {signal_count:.0f}', 
                           transform=up_ax.transAxes)
            up_ax.text(0.6,0.91, '$'+cut.replace(f'{self.var}','E_{\gamma}^B').replace('>=','\geq').replace('<=','\leq')+'$',
                       transform=up_ax.transAxes)
            up_ax.set_ylabel(f"Events/({bin_width*1000:.0f} MeV)")
            down_ax.set_ylabel(f"Pull")
        print('plot saved')

    def single_bin_plotter(self, models, datasets, bin_number, normalise=True, composite=False, plot_data=True, text_shift=0, write_yields=False):
        fig, axs = plt.subplots(2,1,figsize=(8,5), gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05})
        ax = axs.flatten()
        if composite:
            plotter = plot_comp_model
        else:
            plotter = plot_model

        n = bin_number
        up_ax, down_ax = axs[0], axs[1]

        _,_,bin_widhts = plotter(models[n], 
                   datasets[n], plot_data=plot_data,
                   as_bp=True, normalise=normalise,
                   axs=(up_ax, down_ax), spit_vals=True)

        signal_estimate = self.last_result.params[self.collector['yield_signal'][n]]["value"]

        signal_count = datasets[n].nevents.numpy()
        if not self.ignored_chebyshev:
            cheb_estimate = self.last_result.params[self.collector['yield_cheb'][n]]["value"]
        argus_estimate = self.last_result.params[self.collector['yield_argus'][n]]["value"]
        if write_yields and "minuit_minos" in self.last_result.params[self.collector['yield_signal'][n]]:
            signal_up_error = self.last_result.params[self.collector['yield_signal'][n]]["minuit_minos"]["upper"]
            signal_low_error = self.last_result.params[self.collector['yield_signal'][n]]["minuit_minos"]["lower"]
            argus_up_error = self.last_result.params[self.collector['yield_argus'][n]]["minuit_minos"]["upper"]
            argus_low_error = self.last_result.params[self.collector['yield_argus'][n]]["minuit_minos"]["lower"]
            if not self.ignored_chebyshev:
                cheb_up_error = self.last_result.params[self.collector['yield_cheb'][n]]["minuit_minos"]["upper"]
                cheb_low_error = self.last_result.params[self.collector['yield_cheb'][n]]["minuit_minos"]["lower"]
            #pull = (signal_estimate - signal_count)/signal_error

            up_ax.text(0.05, 0.9-text_shift, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm^{{{signal_up_error:.0f}}}_{{{-signal_low_error:.0f}}}'+'$', 
                       transform=up_ax.transAxes)
            up_ax.text(0.05, 0.8-text_shift, 'Argus: $'+rf'{argus_estimate:.0f}\pm^{{{argus_up_error:.0f}}}_{{{-argus_low_error:.0f}}}'+'$', 
                       transform=up_ax.transAxes)
            if not self.ignored_chebyshev:
                up_ax.text(0.05, 0.7-text_shift, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm^{{{cheb_up_error:.0f}}}_{{{-cheb_low_error:.0f}}}'+'$', 
                           transform=up_ax.transAxes)
        elif write_yields and "zfit_error" in self.last_result.params[self.collector['yield_signal'][n]]:
            signal_up_error = self.last_result.params[self.collector['yield_signal'][n]]["zfit_error"]["upper"]
            signal_low_error = self.last_result.params[self.collector['yield_signal'][n]]["zfit_error"]["lower"]
            argus_up_error = self.last_result.params[self.collector['yield_argus'][n]]["zfit_error"]["upper"]
            argus_low_error = self.last_result.params[self.collector['yield_argus'][n]]["zfit_error"]["lower"]
            if not self.ignored_chebyshev:
                cheb_up_error = self.last_result.params[self.collector['yield_cheb'][n]]["zfit_error"]["upper"]
                cheb_low_error = self.last_result.params[self.collector['yield_cheb'][n]]["zfit_error"]["lower"]
            #pull = (signal_estimate - signal_count)/signal_error

            up_ax.text(0.05, 0.9-text_shift, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm^{{{signal_up_error:.0f}}}_{{{-signal_low_error:.0f}}}'+'$', 
                       transform=up_ax.transAxes)
            up_ax.text(0.05, 0.8-text_shift, 'Argus: $'+rf'{argus_estimate:.0f}\pm^{{{argus_up_error:.0f}}}_{{{-argus_low_error:.0f}}}'+'$', 
                       transform=up_ax.transAxes)
            if not self.ignored_chebyshev:
                up_ax.text(0.05, 0.7-text_shift, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm^{{{cheb_up_error:.0f}}}_{{{-cheb_low_error:.0f}}}'+'$', 
                           transform=up_ax.transAxes)

        elif write_yields:
            signal_error = self.last_result.params[self.collector['yield_signal'][n]]["quick_error"]["error"]
            argus_error = self.last_result.params[self.collector['yield_argus'][n]]["quick_error"]["error"]
            if not self.ignored_chebyshev:
                cheb_error = self.last_result.params[self.collector['yield_cheb'][n]]["quick_error"]["error"]
            #pull = (signal_estimate - signal_count)/signal_error

            up_ax.text(0.05, 0.9-text_shift, 'Crystal Ball: $'+rf'{signal_estimate:.0f}\pm {signal_error:.0f}'+'$', \
                       transform=up_ax.transAxes)
            up_ax.text(0.05, 0.8-text_shift, 'Argus: $'+rf'{argus_estimate:.0f}\pm {argus_error:.0f}'+'$', \
                       transform=up_ax.transAxes)
            if not self.ignored_chebyshev:
                up_ax.text(0.05, 0.7-text_shift, 'Chebyshev: $'+rf'{cheb_estimate:.0f}\pm{cheb_error:.0f}'+'$', \
                           transform=up_ax.transAxes)
        if write_yields:
            up_ax.text(0.05, 0.55-text_shift, f'N datapoints: {signal_count:.0f}', 
                       transform=up_ax.transAxes)
            up_ax.text(0.6,0.91, '$'+self.bin_strings[n].replace(f'{self.var}','E_{\gamma}^B').replace('>=','\geq').replace('<=','\leq')+'$',
                       transform=up_ax.transAxes)
        up_ax.set_ylabel(f"Events/({bin_widhts[0]*1000:.1f} MeV)")
        down_ax.set_ylabel(f"Pull")
        print('plot drawn')

    def _old_calculate_chebyshev_shape_systematics(self, dataset=None, scale=None):

        assert not dataset is None  or not scale is None and (dataset is None or scale is None), "Either specify scale or dataset"

        self.systematic_result = self.last_result

        top_coeffs = []
        bot_coeffs = []

        topbot_coeffs = []
        bottop_coeffs = []

        ### WORK IN PROGRESS
        for enn, n_bin_group in enumerate(self.cheb_bin_groups):
            coeffs = {
                    'normal':[1],
                    'down':[1],
                    'up':[1],
                    }
            for param in self.collector['cheb_pars'][1+6*enn:6+6*enn]:

                print(param.name)
                print(param.value)

                low = self.cheb_result.params[param]['minuit_minos']['lower']
                up = self.cheb_result.params[param]['minuit_minos']['upper']
                normal = self.cheb_result.params[param]['value']

                coeffs['normal'].append(normal)
                coeffs['down'].append(normal+low) # + because its negative
                coeffs['up'].append(normal+up)

            # All possible combinations; For 5th order that is 243 combinations (3^5)
            a = list(np.array([coeffs['normal'], coeffs['down'], coeffs['up']]).T)
            all_coeff_combos = np.unique(list(itertools.product(*a)), axis=0)
            print('This needs to be 243: ', len(all_coeff_combos))

            # Save Chebyshev shape for each

            chebis = []
            space = np.linspace(-1,1,50)
            for combo in all_coeff_combos:
                combo_values = np.polynomial.chebyshev.chebval(space, combo)
                chebis.append(combo_values)

            # Get top/bottom most curve
            minchebi = np.min(chebis, axis=0)
            maxchebi = np.max(chebis, axis=0)

            min_start, min_finish = np.array_split(minchebi,2)
            max_start, max_finish = np.array_split(maxchebi,2)

            minmaxchebi = np.concatenate((min_start, max_finish))
            maxminchebi = np.concatenate((max_start, min_finish))

            # Get coefficients for topmost and bottommost curve`
            mincoeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1,1,50), minchebi, 5)
            maxcoeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1,1,50), maxchebi, 5)
            minmaxcoeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1,1,50), minmaxchebi, 5)
            maxmincoeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1,1,50), maxminchebi, 5)
            top_coeffs.append(maxcoeffs)
            bot_coeffs.append(mincoeffs)
            topbot_coeffs.append(maxmincoeffs)
            bottop_coeffs.append(minmaxcoeffs)
        errors, new_result = pretty_print_result(full_result)

            # This min and max coeff also includes the 0th coefficient, 
        top_coeffs = np.array(top_coeffs).flatten()
        bot_coeffs = np.array(bot_coeffs).flatten()
        topbot_coeffs = np.array(topbot_coeffs).flatten()
        bottop_coeffs = np.array(bottop_coeffs).flatten()

        normal_yields = [yld.value().numpy() for yld in self.collector['yield_signal']]

        # Set to down
        self.restore_prefit_values()
        for n, param in enumerate(self.collector['cheb_pars']):
            param.set_value(bot_coeffs[n])

        # TEST HERE: is it correct to my numpy notebook example
        #self.plot_cheb_pdf()

        down_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False)
        down_yields = [yld.value().numpy() for yld in self.collector['yield_signal']]
        # get yields, and save difference

        # Set to up
        self.restore_prefit_values()
        for n, param in enumerate(self.collector['cheb_pars']):
            param.set_value(top_coeffs[n])

        # TEST HERE: is it correct to my numpy notebook example
        #self.plot_cheb_pdf()

        up_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False)
        up_yields = [yld.value().numpy() for yld in self.collector['yield_signal']]
        # get yields, and save difference

        # Set to downup
        self.restore_prefit_values()
        for n, param in enumerate(self.collector['cheb_pars']):
            param.set_value(bottop_coeffs[n])

        # TEST HERE: is it correct to my numpy notebook example
        #self.plot_cheb_pdf()

        downup_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False)
        downup_yields = [yld.value().numpy() for yld in self.collector['yield_signal']]
        # get yields, and save difference

        # Set to updown
        self.restore_prefit_values()
        for n, param in enumerate(self.collector['cheb_pars']):
            param.set_value(topbot_coeffs[n])

        # TEST HERE: is it correct to my numpy notebook example
        #self.plot_cheb_pdf()

        updown_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False)
        updown_yields = [yld.value().numpy() for yld in self.collector['yield_signal']]
        # get yields, and save difference

        # Reset to nominal
        self.restore_prefit_values()
        for n, param in enumerate(self.collector['cheb_pars']):
            if not param in self.cheb_result.params:
                print(param, ' not in result')
                param.set_value(1)
                continue
            param.set_value(self.cheb_result.params[param]['value'])
            print(self.cheb_result.params[param]['value'])

        return down_yields, up_yields, downup_yields, updown_yields, normal_yields

    def new_calculate_chebyshev_shape_systematics(self, dataset=None, scale=None):

        assert not dataset is None  or not scale is None and (dataset is None or scale is None), "Either specify scale or dataset"

        self.systematic_result = self.last_result

        cov_par_set = []
        eig_vals = [[] for _ in self.cheb_bin_groups]
        eig_vects = [[] for _ in self.cheb_bin_groups]

        normal_yields = np.array([yld.value().numpy() for yld in self.collector['yield_signal']])
        all_variations = []
        covariance_matrix = np.zeros((len(self.bin_list)-1, len(self.bin_list)-1))
        for enn, n_bin_group in enumerate(self.cheb_bin_groups):
                cov_par_set.append(self.collector["cheb_pars"][1+enn*6:6+6*enn])
                cov = self.cheb_result.covariance(params=cov_par_set[enn])
                eig_vals[enn], eig_vects[enn] = la.eig(cov)

        for enn, (eig_val_set, eig_vect_set) in enumerate(zip(eig_vals, eig_vects)):
            print(' - ',enn)
            normal_values = [yld.value().numpy() for yld in cov_par_set[enn]]
            for n in range(5):
                print(' --- ',n)
                print('Upper variation:')
                vector = [0]*n+[eig_val_set[n]**0.5]+[0]*(4-n)
                variations = eig_vect_set.dot(vector)

                for m, variation in enumerate(variations):
                    cov_par_set[enn][m].set_value(normal_values[m]+variation)

                self.restore_prefit_values()
                singular_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False, no_error=True)
                singular_yields = np.array([yld.value().numpy() for yld in self.collector['yield_signal']])
                all_variations.append(singular_yields-normal_yields)
                local_cov = np.abs(np.outer(all_variations[-1], all_variations[-1]))
                covariance_matrix += local_cov


                print('Lower variation:')
                for m, variation in enumerate(variations):
                    cov_par_set[enn][m].set_value(normal_values[m]-variation)

                self.restore_prefit_values()
                singular_result, _ = self.perform_toy_mbc_fit(dataset=dataset, scale=scale, restore_default=False, no_error=True)
                singular_yields = np.array([yld.value().numpy() for yld in self.collector['yield_signal']])
                all_variations.append(singular_yields-normal_yields)
                local_cov = np.abs(np.outer(all_variations[-1], all_variations[-1]))
                covariance_matrix += local_cov

            for n, par in enumerate(cov_par_set[enn]):
                par.set_value(normal_values[n])

        return all_variations, covariance_matrix





class SWeightFit:
    obs = None
    minimizer = None
    collector = None
    ID = None
    def __init__(self, df_peak, df_combinatorial, df_argus,  minimizer=None, obs=None):

        self.df_peak = df_peak
        self.df_combinatorial = df_combinatorial
        self.df_argus = df_argus

        self.df_total = pd.concat([df_peak, df_combinatorial, df_argus])

        self.minimizer = zfit.minimize.Minuit(use_minuit_grad=True) if minimizer is None else minimizer
        self.obs = obs
        self.collector = defaultdict(lambda:[])
        self.max_mbc = np.max(self.df_total.Btag_Mbc)

        self.ID = random_string()

    def init_cb_params(self):
        # Crystal Ball
        self.mu       = zfit.Parameter('mu_'+self.ID,    5.28,   lower=5.275, upper=5.285, step_size=0.001, floating=True)
        self.sigma    = zfit.Parameter('sigma_'+self.ID, 1e-3 , lower=1e-4, upper=1e-2, step_size=0.5e-5 , floating=True)
        self.alfa     = zfit.Parameter('alfa_'+self.ID,  0.2, step_size=0.1 , floating=True)
        self.npar        = zfit.Parameter('n_'+self.ID,    15, step_size=0.1, floating=True)

    def init_cheb_params(self):
        # Polynomial
        self.c1         = zfit.Parameter('c1_'+self.ID, 0.1, step_size=0.001, floating=True)
        self.c2         = zfit.Parameter('c2_'+self.ID, -0.1, step_size=0.001, floating=True)
        self.c3         = zfit.Parameter('c3_'+self.ID, 0.1, step_size=0.001, floating=True)
        self.c4         = zfit.Parameter('c4_'+self.ID, -.01, step_size=0.001, floating=True)
        self.c5         = zfit.Parameter('c5_'+self.ID, 0.01, step_size=0.001, floating=True)

    def init_argus_params(self):
        # Argus
        self.m0 = zfit.Parameter('m0_'+self.ID, self.max_mbc,floating=False)
        self.cpar  = zfit.Parameter('c_'+self.ID, -30,upper=0, step_size=0.1, floating=True)
        self.ppar  = zfit.Parameter('p_'+self.ID,0.5,-5,5,step_size=0.01,floating=False)


    def prefit(self):

        self.init_cb_params()
        self.init_cheb_params()
        self.init_argus_params()

        self.preparatory_crystal_peak_fit()
        self.preparatory_argus_fit()
        self.preparatory_cheb_fit()

    def preparatory_crystal_peak_fit(self, quick=False):

        self.crys_data = zfit.Data.from_pandas(obs=self.obs, df=self.df_peak.Btag_Mbc.dropna())

        crys_unext = CrystalBall(obs=self.obs,
                                 mu=self.mu,
                                 sigma=self.sigma,
                                 alpha=self.alfa,
                                 n=self.npar,
                                 name=f'cb_'+self.ID)

        self.yield_crys = zfit.Parameter(f'yield_crys_'+self.ID, len(self.df_peak), lower=None, floating=True)
        self.crys = crys_unext.create_extended(self.yield_crys)

        crys_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.crys, 
                                                 data=self.crys_data)
        crys_result = self.minimizer.minimize(crys_nll)
        errors, new_result = pretty_print_result(crys_result)

        if crys_result.valid or not new_result:
            self.crys_result = crys_result
        else:
            self.crys_result = new_result

        self.mu.floating = False
        self.sigma.floating = False
        self.alfa.floating = False
        self.npar.floating = False

        return self.crys_result

    def preparatory_cheb_fit(self, quick=False):
        self.cheb_data = zfit.Data.from_pandas(obs=self.obs, df=self.df_combinatorial.Btag_Mbc.dropna())

        cheb_unext = Chebyshev(obs=self.obs, 
                               coeffs=[self.c1, self.c2, self.c3, self.c4, self.c5], 
                               name=f'cheb_'+self.ID)

        self.yield_cheb = zfit.Parameter(f'yield_cheb_'+self.ID, len(self.df_combinatorial), lower=None, floating=True)
        self.cheb = cheb_unext.create_extended(self.yield_cheb)



        cheb_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.cheb, 
                                                 data=self.cheb_data)
        cheb_result = self.minimizer.minimize(cheb_nll)

        errors, new_result = pretty_print_result(cheb_result)

        if cheb_result.valid or not new_result:
            self.cheb_result = cheb_result
        else:
            self.cheb_result = new_result

        self.c1.floating = False
        self.c2.floating = False
        self.c3.floating = False
        self.c4.floating = False
        self.c5.floating = False

        return self.cheb_result

    def preparatory_argus_fit(self, quick=False):

        self.argus_data = zfit.Data.from_pandas(obs=self.obs, df=self.df_argus.Btag_Mbc.dropna())
        argus_unext = Argus(obs=self.obs,
                            m0=self.m0,
                            p=self.ppar,
                            c=self.cpar, 
                            name=f'argus_'+self.ID)

        self.yield_argus = zfit.Parameter(f'yield_argus_'+self.ID, len(self.df_argus), floating=True)
        self.argus = argus_unext.create_extended(self.yield_argus)

        argus_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.argus,
                                                  data=self.argus_data)

        argus_result = self.minimizer.minimize(argus_nll)
        error, new_result = pretty_print_result(argus_result)

        if argus_result.valid or not new_result:
            self.argus_result = argus_result
        else:
            self.argus_result = new_result

        self.m0.floating = False
        self.ppar.floating = False
        self.cpar.floating = True

        return self.argus_result

    def perform_mbc_fit(self):

        self.full_data = zfit.Data.from_pandas(obs=self.obs, df=self.df_total.Btag_Mbc)
        self.last_fit_data = self.full_data

        self.full_fit = zfit.pdf.SumPDF([self.argus, self.cheb, self.crys])
        print(f"is full_fit extended?: {self.full_fit.is_extended}")

        full_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.full_fit,
                                                 data=self.full_data)

        full_result = self.minimizer.minimize(full_nll)
        error, new_result = pretty_print_result(full_result)

        if full_result.valid or not new_result:
            self.full_result = full_result
            if not full_result.valid:
                print("RESULT IS NOT VALID!")
        else:
            self.full_result = new_result

        self.sampler = self.full_fit.create_sampler(n=20000, fixed_params=True)

        return self.full_result

    def perform_toy_mbc_fit(self, scale=1, dataset=None, weights_col=None, restore_default=True):
        if dataset is None:
            total_count = len(df_total)
            self.sampler.resample(n=int(total_count*scale))
            data = self.sampler
        else:
            print('using input data')
            weights = None if weights_col is None else dataset[weights_col]
            data = zfit.Data.from_pandas(df=dataset.Btag_Mbc.dropna(),
                                         obs=self.obs,
                                         weights=weights)
        self.last_fit_data = data
        if restore_default:
            self.restore_prefit_values()

        full_nll = zfit.loss.ExtendedUnbinnedNLL(model=self.full_fit,
                                                 data=data)

        toy_result = self.minimizer.minimize(full_nll)
        error, new_result = pretty_print_result(toy_result)

        if toy_result.valid or not new_result:
            self.toy_result = toy_result
        else:
            self.toy_result = new_result

        return self.toy_result, self.last_fit_data

    def restore_prefit_values(self):
        self.yield_argus.set_value(self.argus_result.params[self.yield_argus]["value"])
        self.yield_cheb.set_value(self.cheb_result.params[self.yield_cheb]["value"])
        self.yield_crys.set_value(self.crys_result.params[self.yield_crys]["value"])
        self.cpar.set_value(self.argus_result.params[self.cpar]["value"])

    def restore_final_values(self):
        self.yield_argus.set_value(self.full_result.params[self.yield_argus]["value"])
        self.yield_cheb.set_value(self.full_result.params[self.yield_cheb]["value"])
        self.yield_crys.set_value(self.full_result.params[self.yield_crys]["value"])
        self.cpar.set_value(self.full_result.params[self.cpar]["value"])

    def general_plotter(self, model, data, normalise=True, composite=False):

        fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [1.618**2,1], 'hspace':0.05}, sharex=True,)

        if composite:
            plotter = plot_comp_model
        else:
            plotter = plot_model

        plotter(model,
                data, add_chi2=True, nbins=75,
                as_bp=True, normalise=normalise,
                axs=(ax[0], ax[1]))

        ax[1].set_xlabel("tag-B $M_{bc}$, $GeV/c^2$")
        ax[1].set_xticks([5.25,5.26,5.27,5.28,5.29])
        ax[1].set_ylim(-2.5,2.5)
        ax[1].set_yticks([-2,2])

    def plot_peak_pdf(self, normalise=True):
        self.general_plotter(self.crys, self.crys_data, normalise)
    def plot_cheb_pdf(self, normalise=True):
        self.general_plotter(self.cheb, self.cheb_data, normalise)
    def plot_argus_pdf(self, normalise=True):
        self.general_plotter(self.argus, self.argus_data, normalise)
    def plot_full_fit(self, normalise=True):
        self.general_plotter(self.full_fit, self.last_fit_data, normalise, composite=True)


def correlation_plot(corr_result, params=None, fig=None, ax=None, fontsize=10):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1, figsize=(18,12))
    shape = corr_result.correlation(params=params).shape[0] 
    yedges = []
    xedges = []
    for parset, val in corr_result.correlation(as_dict=True, params=params).items():
        pruned_name = parset[0].name.replace('yield_', '')[:-6]
        if pruned_name not in yedges:
            yedges.append(pruned_name)
            xedges.append(pruned_name)

    a = ax.imshow(corr_result.correlation(params=params), vmin=-1, vmax=1)
    fig.colorbar(a, ax=ax)

    # Where we want the ticks, in pixel locations
    ticks = np.arange(0, shape)
    ticklabels = xedges

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    for (j,i),label in np.ndenumerate(corr_result.correlation(params=params)):
        ax.text(i,j,f'{label:.2f}',ha='center',va='center', fontsize=fontsize, color='red')


def covariance_plot(cov_result, params=None, fig=None, ax=None, fontsize=10):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1, figsize=(18,12))
    shape = cov_result.covariance(params=params).shape[0] 
    yedges = []
    xedges = []
    for parset, val in cov_result.covariance(as_dict=True,params=params).items():
        pruned_name = parset[0].name.replace('yield_', '')[:-6]
        if pruned_name not in yedges:
            yedges.append(pruned_name)
            xedges.append(pruned_name)

    a = ax.imshow(cov_result.covariance(params=params))
    fig.colorbar(a, ax=ax)

    # Where we want the ticks, in pixel locations
    ticks = np.arange(0, shape)
    ticklabels = xedges

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    for (j,i),label in np.ndenumerate(cov_result.covariance(params=params)):
        ax.text(i,j,f'{label:.2f}',ha='center',va='center', fontsize=fontsize, color='red')
    return cov_result.covariance(params=params)
