import glob
import particle
from uncertainties import ufloat, nominal_value
import pandas as pd
import numpy as np
import b2plot as bp

import matplotlib.pyplot as plt

def make_gamma_axes(ax=None, additional='', var=None, legend='upper left'):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(1.4,2.8)
    ax.set_ylim(0,)
    if var is None:
        ax.set_xlabel(rf"$E_{{\gamma}}^B${additional}, GeV")
    else:
        ax.set_xlabel(rf"{var}{additional}")
    ax.set_ylabel("N candidates per bin")
    ax.legend(loc=legend)

class VariableHybridModel:

    charged_resonances = {}
    mixed_resonances = {}

    charged_inclusive = None
    mixed_inclusive = None

    charged_resonance_all = None
    mixed_resonance_all = None

    charged_variations = []
    mixed_variations = []

    charged_scales = {}
    mixed_scales = {}

    charged_names = []
    mixed_names = []

    # Define constants needed for

    N_BB_tot = 500000

    B0_lifetime = ufloat(1.519, 0.004)
    Bp_lifetime = ufloat(1.638, 0.004)

    tauBpluszero = Bp_lifetime/B0_lifetime

    fpluszero = ufloat(1.058, 0.024)

    inclusive_BR = ufloat(3.49, 0.19) * 1e-4

    BplusBR = inclusive_BR * tauBpluszero * (1 + fpluszero) / (1 + tauBpluszero * fpluszero)
    BzeroBR = inclusive_BR *                (1 + fpluszero) / (1 + tauBpluszero * fpluszero)

#     exclB0_BR = np.sum(list(scales_zero.values()))
#     exclBp_BR = np.sum(list(scales_plus.values()))

    N_inclusive_ch = nominal_value(N_BB_tot   * BplusBR) 
    N_inclusive_md = nominal_value(N_BB_tot   * BzeroBR)

    N_inclusive_ch_up   = nominal_value(N_BB_tot * (BplusBR.nominal_value + BplusBR.std_dev))
    N_inclusive_ch_down = nominal_value(N_BB_tot * (BplusBR.nominal_value - BplusBR.std_dev))
    N_inclusive_md_up   = nominal_value(N_BB_tot * (BzeroBR.nominal_value + BzeroBR.std_dev))
    N_inclusive_md_down = nominal_value(N_BB_tot * (BzeroBR.nominal_value - BzeroBR.std_dev))

    xsu = [30353,-30353]
    xsd = [30343,-30343]
    bs = [521,-521,511,-511]

    xsp_codes = [30353,-30353,
                   323,  10323,  325,  20323,  100323,  30323,
                  -323, -10323, -325, -20323, -100323, -30323]
    xsz_codes = [ 30343, -30343,
                   20313,  10313,  313,  113,  30313,  315,  223,
                  -20313, -10313, -313, -113, -30313, -315, -223,]

    def up_weight(self, ufloat_number):
        return (ufloat_number.nominal_value + ufloat_number.std_dev) / ufloat_number.nominal_value

    def down_weight(self, ufloat_number):
        return (ufloat_number.nominal_value - ufloat_number.std_dev) / ufloat_number.nominal_value

    def register_resonance(self, file_name, resonance_name, scale, charged):

        deff = pd.read_feather(file_name)

        deff.loc[:, 'br_weight']                    = nominal_value(self.N_BB_tot * scale / len(deff))
        deff.loc[:, 'incl_weight']                  = 1
        deff.loc[:, 'up_weight']                    = self.up_weight(scale)
        deff.loc[:,f'{resonance_name}-up']          = self.up_weight(scale)
        deff.loc[:, 'down_weight']                  = self.down_weight(scale)
        deff.loc[:,f'{resonance_name}-down']        = self.down_weight(scale)

        if charged:

            self.charged_resonances[resonance_name]     = deff
            self.charged_scales[resonance_name]         = scale
            self.charged_names.append( resonance_name  )

        else:

            self.mixed_resonances[resonance_name] = deff
            self.mixed_scales[resonance_name]     = scale
            self.mixed_names.append( resonance_name  )

    def register_inclusive_default(self, file_name, charged):

        deff = pd.read_feather(file_name)
        deff.loc[:, 'incl_weight']    = 1
        if charged:
            deff.loc[:,'br_weight']   = self.N_inclusive_ch / len(deff)
            deff.loc[:,'up_weight']   = self.N_inclusive_ch_up / self.N_inclusive_ch 
            deff.loc[:,'down_weight'] = self.N_inclusive_ch_down / self.N_inclusive_ch 
            self.charged_inclusive_default    = deff
        else:
            deff.loc[:,'br_weight']   = self.N_inclusive_md / len(deff)
            deff.loc[:,'up_weight']   = self.N_inclusive_md_up / self.N_inclusive_md
            deff.loc[:,'down_weight'] = self.N_inclusive_md_down / self.N_inclusive_md 
            self.mixed_inclusive_default      = deff

    def register_inclusive(self, file_name, charged):

        deff = pd.read_feather(file_name)
        deff.loc[:, 'incl_weight']    = 1
        if charged:
            deff.loc[:,'br_weight']   = self.N_inclusive_ch / len(deff)
            deff.loc[:,'up_weight']   = self.N_inclusive_ch_up / self.N_inclusive_ch 
            deff.loc[:,'down_weight'] = self.N_inclusive_ch_down / self.N_inclusive_ch 
            self.charged_inclusive    = deff
        else:
            deff.loc[:,'br_weight']   = self.N_inclusive_md / len(deff)
            deff.loc[:,'up_weight']   = self.N_inclusive_md_up / self.N_inclusive_md
            deff.loc[:,'down_weight'] = self.N_inclusive_md_down / self.N_inclusive_md 
            self.mixed_inclusive      = deff

    def register_inclusive_variations(self, file_name, charged):

        deff = pd.read_feather(file_name)
        deff.loc[:, 'incl_weight']  = 1
        if charged:
            deff.loc[:,'br_weight'] = self.N_inclusive_ch / len(deff)
            self.charged_variations.append(deff)
        else:
            deff.loc[:,'br_weight'] = self.N_inclusive_md / len(deff)
            self.mixed_variations.append(deff)

    def register_mc_datasets(self, dataframe, charged):

        dataframe.loc[:,'hweight'] = 1.
        dataframe.loc[:,'up_weight'] = 1.
        dataframe.loc[:,'down_weight'] = 1.
        dataframe.loc[:,'br_weight'] = 1.
        dataframe.loc[:,'rel_weight'] = 1.

        if charged:
            self.charged_inclusive_mc = dataframe[(dataframe['Bsig_d0_mcpdg'].isin(self.xsp_codes[:2])) & \
                                                  (dataframe['Btag_isSignal'] == 1) & \
                                                  (dataframe['isSigSideCorrect'] == 1)]
            self.charged_resonance_mc = dataframe[(dataframe['Bsig_d0_mcpdg'].isin(self.xsp_codes[2:])) & \
                                                  (dataframe['Btag_isSignal'] == 1) & \
                                                  (dataframe['isSigSideCorrect'] == 1)]
        else:
            self.mixed_inclusive_mc = dataframe[(dataframe['Bsig_d0_mcpdg'].isin(self.xsz_codes[:2])) & \
                                                (dataframe['Btag_isSignal'] == 1) & \
                                                (dataframe['isSigSideCorrect'] == 1)]
            self.mixed_resonance_mc = dataframe[(dataframe['Bsig_d0_mcpdg'].isin(self.xsz_codes[2:])) & \
                                                (dataframe['Btag_isSignal'] == 1) & \
                                                (dataframe['isSigSideCorrect'] == 1)]    




    def make_hybrid_weights(self, charged_hybrid_bins,
                                  mixed_hybrid_bins, density=False):

        h_exc, _ = np.histogram(self.charged_resonance_all.g_EB, bins=charged_hybrid_bins, 
                                weights=self.charged_resonance_all.br_weight, density=density)
        h_inc, _ = np.histogram(self.charged_inclusive.g_EB, bins=charged_hybrid_bins, 
                                weights=self.charged_inclusive.br_weight, density=density)

        charged_weights = 1 - h_exc/h_inc


        charged_weights[charged_weights<0] = 0
        charged_weights[np.isnan(charged_weights)] = 1

        h_exc, _ = np.histogram(self.mixed_resonance_all.g_EB, bins=mixed_hybrid_bins, 
                                weights=self.mixed_resonance_all.br_weight, density=density)
        h_inc, _ = np.histogram(self.mixed_inclusive.g_EB, bins=mixed_hybrid_bins, 
                                weights=self.mixed_inclusive.br_weight, density=density)

        mixed_weights = 1 - h_exc/h_inc


        mixed_weights[mixed_weights<0] = 0
        mixed_weights[np.isnan(mixed_weights)] = 1

        self.charged_weights = charged_weights
        self.mixed_weights = mixed_weights

        self.charged_hybrid_bins = charged_hybrid_bins
        self.mixed_hybrid_bins = mixed_hybrid_bins



    def apply_hybrid_weights(self):
        self.charged_inclusive.loc[:,'hweight'] = self.charged_weights[np.digitize(self.charged_inclusive.g_EB, self.charged_hybrid_bins) - 1]
        self.mixed_inclusive.loc[:,'hweight'] =   self.mixed_weights  [np.digitize(self.mixed_inclusive.g_EB,   self.mixed_hybrid_bins) - 1]

        self.charged_resonance_all.loc[:,'hweight'] = 1
        self.mixed_resonance_all.loc[:,'hweight'] = 1

    def apply_reweights(self):
        self.charged_inclusive.loc[:,'reweight'] = self.charged_reweights[np.digitize(self.charged_inclusive.g_EB, self.reweight_bins) - 1]
        self.mixed_inclusive.loc[:,'reweight'] = self.mixed_reweights[np.digitize(self.mixed_inclusive.g_EB, self.reweight_bins) - 1]

        self.charged_resonance_all.loc[:,'reweight'] = 1
        self.mixed_resonance_all.loc[:,'reweight'] = 1

    def apply_variation_weights(self):

        self.resvariation_hweights = {}
        self.sigvariation_hweights = {}
        self.par_variation_hweights = {}

        for resonance in self.charged_names:
            self.charged_resonance_all.loc[:,f'{resonance}-down'].fillna(1, inplace=True)
            self.charged_resonance_all.loc[:,f'{resonance}-up'].fillna(1, inplace=True)
            self.mixed_resonance_all.loc[:,f'{resonance}-down'] = 1
            self.mixed_resonance_all.loc[:,f'{resonance}-up'] = 1

            varied_hw_up = make_hybrid_weights(self.charged_inclusive.g_EB, 
                                               self.charged_resonance_all.g_EB, 
                                               self.charged_hybrid_bins,
                                               self.charged_inclusive["br_weight"], 
                                               self.charged_resonance_all["br_weight"]*self.charged_resonance_all[f"{resonance}-up"])
            
            varied_hw_down = make_hybrid_weights(self.charged_inclusive.g_EB, 
                                                 self.charged_resonance_all.g_EB, 
                                                 self.charged_hybrid_bins,
                                                 self.charged_inclusive["br_weight"], 
                                                 self.charged_resonance_all["br_weight"]*self.charged_resonance_all[f"{resonance}-down"])

            self.resvariation_hweights[resonance+'-up'] = varied_hw_up
            self.resvariation_hweights[resonance+'-down'] = varied_hw_down
    

        for resonance in self.mixed_names:
            self.mixed_resonance_all[f'{resonance}-down'].fillna(1, inplace=True)
            self.mixed_resonance_all[f'{resonance}-up'].fillna(1, inplace=True)
            self.charged_resonance_all.loc[:,f'{resonance}-down'] = 1
            self.charged_resonance_all.loc[:,f'{resonance}-up'] = 1
            
            varied_hw_up = make_hybrid_weights(self.mixed_inclusive.g_EB, 
                                               self.mixed_resonance_all.g_EB, 
                                               self.mixed_hybrid_bins,
                                               self.mixed_inclusive["br_weight"], 
                                               self.mixed_resonance_all["br_weight"]*self.mixed_resonance_all[f"{resonance}-up"])
            
            varied_hw_down = make_hybrid_weights(self.mixed_inclusive.g_EB, 
                                                 self.mixed_resonance_all.g_EB, 
                                                 self.mixed_hybrid_bins,
                                                 self.mixed_inclusive["br_weight"], 
                                                 self.mixed_resonance_all["br_weight"]*self.mixed_resonance_all[f"{resonance}-down"])

            self.resvariation_hweights[resonance+'-up'] = varied_hw_up
            self.resvariation_hweights[resonance+'-down'] = varied_hw_down
          

        

        varied_hw_up = make_hybrid_weights(self.charged_inclusive.g_EB, 
                                           self.charged_resonance_all.g_EB, 
                                           self.charged_hybrid_bins,
                                           self.charged_inclusive["br_weight"]*self.charged_inclusive['up_weight'], 
                                           self.charged_resonance_all["br_weight"])


        varied_hw_down = make_hybrid_weights(self.charged_inclusive.g_EB, 
                                           self.charged_resonance_all.g_EB, 
                                           self.charged_hybrid_bins,
                                           self.charged_inclusive["br_weight"]*self.charged_inclusive['down_weight'], 
                                           self.charged_resonance_all["br_weight"])

        self.sigvariation_hweights['charged-up'] = varied_hw_up
        self.sigvariation_hweights['charged-down'] = varied_hw_down
        
        varied_hw_up = make_hybrid_weights(self.mixed_inclusive.g_EB, 
                                           self.mixed_resonance_all.g_EB, 
                                           self.mixed_hybrid_bins,
                                           self.mixed_inclusive["br_weight"]*self.mixed_inclusive['up_weight'], 
                                           self.mixed_resonance_all["br_weight"])


        varied_hw_down = make_hybrid_weights(self.mixed_inclusive.g_EB, 
                                           self.mixed_resonance_all.g_EB, 
                                           self.mixed_hybrid_bins,
                                           self.mixed_inclusive["br_weight"]*self.mixed_inclusive['down_weight'], 
                                           self.mixed_resonance_all["br_weight"])

        self.sigvariation_hweights['mixed-up'] = varied_hw_up
        self.sigvariation_hweights['mixed-down'] = varied_hw_down
        


        for n, (ch, md) in enumerate(zip(self.charged_variations, self.mixed_variations)):

            varied_ch    = make_hybrid_weights(ch.g_EB, 
                                               self.charged_resonance_all.g_EB, 
                                               self.charged_hybrid_bins,
                                               ch["br_weight"], 
                                               self.charged_resonance_all["br_weight"])


            varied_md    = make_hybrid_weights(md.g_EB, 
                                               self.mixed_resonance_all.g_EB, 
                                               self.mixed_hybrid_bins,
                                               md["br_weight"], 
                                               self.mixed_resonance_all["br_weight"])

            self.par_variation_hweights[f'charged_var_{n}'] = varied_ch
            self.par_variation_hweights[f'mixed_var_{n}'] = varied_md
        
    def make_reweights(self, reweight_bins, density=False):

        h_exc, _ = np.histogram(self.charged_inclusive.g_EB, bins=reweight_bins, 
                                density=density)
        h_inc, _ = np.histogram(self.charged_inclusive_default.g_EB, bins=reweight_bins, 
                                density=density)

        charged_reweights = h_exc/h_inc


        charged_reweights[np.isinf(charged_reweights)] = 0
        charged_reweights[np.isnan(charged_reweights)] = 1
        
        h_exc, _ = np.histogram(self.mixed_inclusive.g_EB, bins=reweight_bins, 
                                density=density)
        h_inc, _ = np.histogram(self.mixed_inclusive_default.g_EB, bins=reweight_bins, 
                                density=density)

        mixed_reweights = h_exc/h_inc


        mixed_reweights[np.isinf(mixed_reweights)] = 0
        mixed_reweights[np.isnan(mixed_reweights)] = 1
        
        self.charged_reweights = charged_reweights
        self.mixed_reweights = mixed_reweights
        self.reweight_bins = reweight_bins
    
    def lock(self):
        self.is_locked = True
        
        self.charged_resonance_all = pd.concat(self.charged_resonances.values())
        self.charged_resonance_all.loc[:,'rel_weight'] = 1
        
        self.mixed_resonance_all = pd.concat(self.mixed_resonances.values())
        self.mixed_resonance_all.loc[:,'rel_weight'] = 1
        
        self.charged_leftover = nominal_value(3.49e-4 - np.sum(list(self.charged_scales.values())))
        self.mixed_leftover = nominal_value(3.49e-4 - np.sum(list(self.mixed_scales.values())))

        self.charged_inclusive.loc[:,'rel_weight']  = self.charged_leftover / 3.49e-4
        self.mixed_inclusive.loc[:,'rel_weight']    = self.mixed_leftover / 3.49e-4
        
        self.charged_inclusive_default.loc[:,'rel_weight']  = self.charged_leftover / 3.49e-4
        self.mixed_inclusive_default.loc[:,'rel_weight']  = self.mixed_leftover / 3.49e-4
        
    
    def lock_mc(self):
        self.mixed_inclusive_mc.loc[:,"hweight"] = self.mixed_weights[ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
        self.charged_inclusive_mc.loc[:,"hweight"] = self.charged_weights[ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]

        self.mixed_inclusive_mc.loc[:,"hweight-up"] = self.sigvariation_hweights['mixed-up'][ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
        self.charged_inclusive_mc.loc[:,"hweight-up"] = self.sigvariation_hweights['charged-up'][ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]

        self.mixed_inclusive_mc.loc[:,"hweight-down"] = self.sigvariation_hweights['mixed-down'][ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
        self.charged_inclusive_mc.loc[:,"hweight-down"] = self.sigvariation_hweights['charged-down'][ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]

        self.mixed_resonance_mc.loc[:, "hweight"] = 1
        self.charged_resonance_mc.loc[:, "hweight"] = 1

        self.mixed_resonance_mc.loc[:, "hweight-up"] = 1
        self.charged_resonance_mc.loc[:, "hweight-up"] = 1

        self.mixed_resonance_mc.loc[:, "hweight-down"] = 1
        self.charged_resonance_mc.loc[:, "hweight-down"] = 1

        self.charged_inclusive_mc.loc[:,"reweight"] = self.charged_reweights[-1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, bins=self.reweight_bins)]
        self.mixed_inclusive_mc.loc[:,"reweight"] = self.mixed_reweights[-1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, bins=self.reweight_bins)]

        self.mixed_resonance_mc.loc[:, "reweight"] = 1
        self.charged_resonance_mc.loc[:, "reweight"] = 1

        self.charged_inclusive_mc.loc[:,"scaling"] = 3.49e-4 / self.charged_leftover
        self.mixed_inclusive_mc.loc[:,"scaling"] = 3.49e-4 / self.mixed_leftover

        self.mixed_resonance_mc.loc[:,"scaling"] = 1
        self.charged_resonance_mc.loc[:,"scaling"] = 1

        for mode, val in self.charged_scales.items():
            print(f'Adding weights for {mode}')
            pdgcode = int(particle.Particle.from_string(mode).pdgid)
            pdgcodes = [pdgcode,-pdgcode]

            self.charged_resonance_mc.loc[self.charged_resonance_mc['Bsig_d0_mcpdg'].isin(pdgcodes),f'{mode}-up'] = self.up_weight(val)
            self.charged_resonance_mc.loc[self.charged_resonance_mc['Bsig_d0_mcpdg'].isin(pdgcodes),f'{mode}-down'] = self.down_weight(val)

            self.charged_resonance_mc.loc[:,f'{mode}-up'].fillna(1, inplace=True)
            self.charged_resonance_mc.loc[:,f'{mode}-down'].fillna(1, inplace=True)
            self.charged_inclusive_mc.loc[:,f'{mode}-up'] =  self.resvariation_hweights[f'{mode}-up'][ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]
            self.charged_inclusive_mc.loc[:,f'{mode}-down'] =  self.resvariation_hweights[f'{mode}-down'][ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]

            self.mixed_resonance_mc.loc[:,f'{mode}-up'] = 1
            self.mixed_resonance_mc.loc[:,f'{mode}-down'] = 1
            self.mixed_inclusive_mc.loc[:,f'{mode}-up'] = self.mixed_weights[ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
            self.mixed_inclusive_mc.loc[:,f'{mode}-down'] = self.mixed_weights[ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
        
        for mode, val in self.mixed_scales.items():
            print(f'Adding weights for {mode}')
            pdgcode = int(particle.Particle.from_string(mode).pdgid)
            pdgcodes = [pdgcode,-pdgcode]

            self.mixed_resonance_mc.loc[self.mixed_resonance_mc['Bsig_d0_mcpdg'].isin(pdgcodes),f'{mode}-up'] = self.up_weight(val)
            self.mixed_resonance_mc.loc[self.mixed_resonance_mc['Bsig_d0_mcpdg'].isin(pdgcodes),f'{mode}-down'] = self.down_weight(val)

            self.mixed_resonance_mc.loc[:,f'{mode}-up'].fillna(1, inplace=True)
            self.mixed_resonance_mc.loc[:,f'{mode}-down'].fillna(1, inplace=True)
            self.mixed_inclusive_mc.loc[:,f'{mode}-up'] = self.resvariation_hweights[f'{mode}-up'][ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]
            self.mixed_inclusive_mc.loc[:,f'{mode}-down'] = self.resvariation_hweights[f'{mode}-down'][ -1 + np.digitize(self.mixed_inclusive_mc.gamma_mcEB, self.mixed_hybrid_bins)]

            self.charged_resonance_mc.loc[:,f'{mode}-up'] = 1
            self.charged_resonance_mc.loc[:,f'{mode}-down'] = 1
            self.charged_inclusive_mc.loc[:,f'{mode}-up'] = self.charged_weights[ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]
            self.charged_inclusive_mc.loc[:,f'{mode}-down'] = self.charged_weights[ -1 + np.digitize(self.charged_inclusive_mc.gamma_mcEB, self.charged_hybrid_bins)]
        
        for n in range(len(self.charged_variations)):
            

            self.charged_inclusive_mc.loc[:,f'par_weight_{n}'] = self.par_variation_hweights[f'charged_var_{n}'][-1 +np.digitize(self.charged_inclusive_mc.gamma_mcEB, bins=self.charged_hybrid_bins)]
            self.mixed_inclusive_mc.loc[:,f'par_weight_{n}'] = self.par_variation_hweights[f'mixed_var_{n}'][-1 +np.digitize(self.mixed_inclusive_mc.gamma_mcEB, bins=self.mixed_hybrid_bins)]

            self.charged_resonance_mc.loc[:, f"par_weight_{n}"] = 1
            self.mixed_resonance_mc.loc[:, f"par_weight_{n}"] = 1

            temp_bp_reweight = make_reweights(ch.g_EB,
                                              self.charged_inclusive_default.g_EB,
                                              self.reweight_bins,
                                              self.charged_inclusive_mc.gamma_mcEB,
                                              True
                      )

            temp_bz_reweight = make_reweights(md.g_EB,
                                              self.mixed_inclusive_default.g_EB,
                                              self.reweight_bins,
                                              self.mixed_inclusive_mc.gamma_mcEB,
                                              True
                          )

            self.charged_inclusive_mc.loc[:, f"par_reweight_{n}"] = temp_bp_reweight[-1 +np.digitize(self.charged_inclusive_mc.gamma_mcEB, bins=self.reweight_bins)]
            self.mixed_inclusive_mc.loc[:,f'par_reweight_{n}'] = temp_bz_reweight[-1 +np.digitize(self.mixed_inclusive_mc.gamma_mcEB, bins=self.reweight_bins)]


            self.charged_resonance_mc.loc[:, f"par_reweight_{n}"] = 1
            self.mixed_resonance_mc.loc[:, f"par_reweight_{n}"] = 1



    def draw_unweighted_evtgen(self, reweighted=False):
        
        bins = np.linspace(1.4,2.8,50)
        bin_width = bins[1]-bins[0]
        fig, non_flat_axs = plt.subplots(1,2, figsize=(18,6))
        axs = non_flat_axs.flatten()
        
        # B plus mode
        
        ax = axs[0]
        
        if reweighted:
            signal = self.charged_inclusive
        else:
            signal = self.charged_inclusive_default
        
        draw_list = [df.g_EB for df in self.charged_resonances.values()] + [signal.g_EB]
        name_list = self.charged_names + ['Xs gamma']


        weights = [df['br_weight'].values for df in self.charged_resonances.values()] + \
                  [signal.br_weight*signal.rel_weight]

        bp.stacked(draw_list, bins=bins, ax=ax, label=name_list, weights=weights)


        bp.hist(signal.g_EB, 
                bins=bins, ax=ax, 
                label='Inclusive signal model', scale=3.49e-4, 
                ls='--', color='orange', lw=2)
        
        # B zero mode
        
        ax = axs[1]
        
        if reweighted:
            signal = self.mixed_inclusive
        else:
            signal = self.mixed_inclusive_default

        draw_list = [df.g_EB for df in self.mixed_resonances.values()] + [signal.g_EB]
        name_list = self.mixed_names + ['Xs gamma']


        weights = [df['br_weight'].values for df in self.mixed_resonances.values()] + \
                  [signal.br_weight*signal.rel_weight]

        bp.stacked(draw_list, bins=bins, ax=ax, label=name_list, weights=weights)


        bp.hist(signal.g_EB, 
                bins=bins, ax=ax, 
                label='Inclusive signal model', scale=3.49e-4, 
                ls='--', color='orange', lw=2)

        for ax in axs:

            ax.legend(loc='upper left')

            ax.set_xlabel("$E^{\mathrm{B}}_{\gamma}$, GeV")
            ax.set_ylabel(f" 500000*BF / {bin_width:.2f} GeV")
            ax.legend(loc='upper left')
            if not reweighted:
                ax.text(0.41,0.95, "Approximately what is used in Generic MC", fontsize=12, transform=ax.transAxes)
        axs[0].text(0.41,0.85, "$B^+$", fontsize=12, transform=axs[0].transAxes)
        axs[1].text(0.41,0.85, "$B^0$", fontsize=12, transform=axs[1].transAxes)

    def draw_reweighted(self):

        bins = np.linspace(1.4,2.8,25)
        bin_width = bins[1]-bins[0]
        fig, non_flat_axs = plt.subplots(1,2, figsize=(18,6))
        axs = non_flat_axs.flatten()

        ax=axs[0]

        reweights = self.charged_reweights[-1 + np.digitize(self.charged_inclusive_default.g_EB, self.reweight_bins)]

        bp.hist(self.charged_inclusive.g_EB, bins=bins, label='generated', density=True, lw=2, ax=ax)
        bp.hist(self.charged_inclusive_default.g_EB, bins=bins, label='generic', density=True, lw=2, ax=ax)
        bp.hist(self.charged_inclusive_default.g_EB, bins=bins, label='reweighted', ls=':', 
                weights=reweights, density=True, lw=5, ax=ax)

        make_gamma_axes(ax=ax,additional=' generated')

        ax=axs[1]
        reweights = self.mixed_reweights[-1 + np.digitize(self.mixed_inclusive_default.g_EB, self.reweight_bins)]
        bp.hist(self.mixed_inclusive.g_EB, bins=bins, label='generated', density=True, lw=2, ax=ax)
        bp.hist(self.mixed_inclusive_default.g_EB, bins=bins, label='generic', density=True, lw=2, ax=ax)
        bp.hist(self.mixed_inclusive_default.g_EB, bins=bins, label='reweighted', ls=':', 
                weights=reweights, density=True, lw=5, ax=ax)

        make_gamma_axes(ax=ax, additional=' generated')


        plt.legend(loc='upper left')

    def draw_reweighted_mc(self, drawbins, var='gamma_mcEB'):
        
        fig, non_flat_axs = plt.subplots(1,2, figsize=(18,6))
        axs = non_flat_axs.flatten()
        
        ax=axs[0]
        
        reweights = self.charged_reweights[-1 + np.digitize(self.charged_inclusive_mc[var], self.reweight_bins)]
        
        bp.hist(self.charged_inclusive.g_EB, bins=drawbins, label='generated', density=True, lw=2, ax=ax)
        bp.hist(self.charged_inclusive_mc[var], bins=drawbins, label='generic', density=True, lw=2, ax=ax)
        bp.hist(self.charged_inclusive_mc[var], bins=drawbins, label='reweighted', ls=':', 
                weights=self.charged_inclusive_mc['reweight'], density=True, lw=5, ax=ax)

        make_gamma_axes(ax=ax,additional=' generated')
        
        ax=axs[1]
        
        #reweights = self.mixed_reweights[-1 + np.digitize(self.mixed_inclusive_mc[var], self.reweight_bins)]
        
        bp.hist(self.mixed_inclusive.g_EB, bins=drawbins, label='generated', density=True, lw=2, ax=ax)
        bp.hist(self.mixed_inclusive_mc[var], bins=drawbins, label='generic', density=True, lw=2, ax=ax)
        bp.hist(self.mixed_inclusive_mc[var], bins=drawbins, label='reweighted', ls=':', 
                weights=self.mixed_inclusive_mc['reweight'], density=True, lw=5, ax=ax)

        make_gamma_axes(ax=ax, additional=' generated')


        plt.legend(loc='upper left')
    
    def draw_hybrid_weighted(self,
                        drawbins,
                        var="g_EB", 
                        save=None,
                        include_non_hweighted=True,
                        ):
    
        fig, axs = plt.subplots(1,3, figsize=(27,6))
        bin_width = drawbins[1]-drawbins[0]
        
        for bin in self.charged_hybrid_bins:
            axs[0].axvline(bin, ls='--', lw='1',color='black')
        
        for bin in self.mixed_hybrid_bins:
            axs[1].axvline(bin, ls='--', lw='1',color='black')
        
        #### Bp
        ax = axs[0]
        
        hybrid_weights_inclusive  = self.charged_inclusive['hweight'] * \
                                    self.charged_inclusive['br_weight'] \

        hybrid_weights_resonances = self.charged_resonance_all['hweight'] * \
                                    self.charged_resonance_all['br_weight']


        n1, _, _ = bp.stacked([self.charged_inclusive[var], 
                               self.charged_resonance_all[var]],
                               bins=drawbins,
                               weights=[hybrid_weights_inclusive, 
                                        hybrid_weights_resonances],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        charged_total = pd.concat([self.charged_inclusive, self.charged_resonance_all])
        n_h, _, _ = bp.hist(charged_total[var],
                            bins=drawbins,
                            weights=charged_total['hweight']*charged_total['br_weight'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(charged_total[var],
                                bins=drawbins,
                                weights=charged_total['rel_weight']*charged_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(self.charged_inclusive[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=self.charged_inclusive["incl_weight"]*self.charged_inclusive['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)
        #### B0
        ax = axs[1]

        hybrid_weights_inclusive  = self.mixed_inclusive['hweight'] * \
                                    self.mixed_inclusive['br_weight'] \

        hybrid_weights_resonances = self.mixed_resonance_all['hweight'] * \
                                    self.mixed_resonance_all['br_weight']


        n1, _, _ = bp.stacked([self.mixed_inclusive[var], 
                               self.mixed_resonance_all[var]],
                               bins=drawbins,
                               weights=[hybrid_weights_inclusive, 
                                        hybrid_weights_resonances],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        mixed_total = pd.concat([self.mixed_inclusive, self.mixed_resonance_all])
        n_h, _, _ = bp.hist(mixed_total[var],
                            bins=drawbins,
                            weights=mixed_total['hweight']*mixed_total['br_weight'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(mixed_total[var],
                                bins=drawbins,
                                weights=mixed_total['rel_weight']*mixed_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(self.mixed_inclusive[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=self.mixed_inclusive["incl_weight"]*self.mixed_inclusive['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)

        #print(f"Difference between areas is {100*abs((np.sum(n1)-np.sum(n2))/np.sum(n2)):.2f}%")

        #### Both
        ax = axs[2]

        inclusive_total = pd.concat([self.charged_inclusive, self.mixed_inclusive])
        resonance_total = pd.concat([self.charged_resonance_all, self.mixed_resonance_all])

        n1, _, _ = bp.stacked([inclusive_total[var], 
                               resonance_total[var]],
                               bins=drawbins,
                               weights=[inclusive_total['hweight']*inclusive_total['br_weight'], 
                                        resonance_total['hweight']*resonance_total['br_weight']],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        total_total = pd.concat([charged_total, mixed_total])
        n_h, _, _ = bp.hist(total_total[var],
                            bins=drawbins,
                            weights=total_total['hweight']*total_total['br_weight'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(total_total[var],
                                bins=drawbins,
                                weights=total_total['rel_weight']*total_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(inclusive_total[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=inclusive_total["incl_weight"]*inclusive_total['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)

        for ax in axs:
            if var=='g_EB' or var=='gamma_mcEB':
                additional=' generated'
            elif var=='gamma_EB':
                additional=' reconstructed'


            make_gamma_axes(ax, additional=additional)

            ax.set_ylabel(rf"$5\times10^5 \times$BF$\times$Weights / {bin_width:.2f} GeV")

        if save:
            fig.savefig(f"{save}", bbox_inches='tight', dpi=300)

    def draw_hybrid_weighted_mc(self,
                                drawbins,
                                var="gamma_mcEB", 
                                save=None,
                                include_non_hweighted=True,
                                ):

        fig, axs = plt.subplots(1,3, figsize=(27,6))
        bin_width = drawbins[1]-drawbins[0]

        for bin in self.charged_hybrid_bins:
            axs[0].axvline(bin, ls='--', lw='1',color='black')

        for bin in self.mixed_hybrid_bins:
            axs[1].axvline(bin, ls='--', lw='1',color='black')

        #### Bp
        ax = axs[0]

        hybrid_weights_inclusive  = self.charged_inclusive_mc['hweight']   * \
                                    self.charged_inclusive_mc['br_weight'] * \
                                    self.charged_inclusive_mc['reweight']  * \
                                    self.charged_inclusive_mc['scaling']

        hybrid_weights_resonances = self.charged_resonance_mc['hweight'] * \
                                    self.charged_resonance_mc['br_weight'] * \
                                    self.charged_resonance_mc['reweight'] * \
                                    self.charged_resonance_mc['scaling']


        n1, _, _ = bp.stacked([self.charged_inclusive_mc[var], 
                               self.charged_resonance_mc[var]],
                               bins=drawbins,
                               weights=[hybrid_weights_inclusive, 
                                        hybrid_weights_resonances],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        charged_total = pd.concat([self.charged_inclusive_mc, self.charged_resonance_mc])
        n_h, _, _ = bp.hist(charged_total[var],
                            bins=drawbins,
                            weights=charged_total['hweight']*charged_total['br_weight']*charged_total['reweight']*charged_total['scaling'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(charged_total[var],
                                bins=drawbins,
                                weights=charged_total['rel_weight']*charged_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(self.charged_inclusive_mc[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=self.charged_inclusive_mc["scaling"]*\
                                   self.charged_inclusive_mc["reweight"]*\
                                   self.charged_inclusive_mc['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)
        #### B0
        ax = axs[1]
        
        hybrid_weights_inclusive  = self.mixed_inclusive_mc['hweight']   * \
                                    self.mixed_inclusive_mc['br_weight'] * \
                                    self.mixed_inclusive_mc['reweight']  * \
                                    self.mixed_inclusive_mc["scaling"]

        hybrid_weights_resonances = self.mixed_resonance_mc['hweight'] * \
                                    self.mixed_resonance_mc['br_weight']


        n1, _, _ = bp.stacked([self.mixed_inclusive_mc[var], 
                               self.mixed_resonance_mc[var]],
                               bins=drawbins,
                               weights=[hybrid_weights_inclusive, 
                                        hybrid_weights_resonances],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        mixed_total = pd.concat([self.mixed_inclusive_mc, self.mixed_resonance_mc])

        n_h, _, _ = bp.hist(mixed_total[var],
                            bins=drawbins,
                            weights=mixed_total['hweight']*\
                                    mixed_total['br_weight']*\
                                    mixed_total['reweight']*\
                                    mixed_total['scaling'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(mixed_total[var],
                                bins=drawbins,
                                weights=mixed_total['rel_weight']*\
                                        mixed_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(self.mixed_inclusive_mc[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=self.mixed_inclusive_mc["scaling"]*\
                                   self.mixed_inclusive_mc["reweight"]*\
                                   self.mixed_inclusive_mc['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)

        #print(f"Difference between areas is {100*abs((np.sum(n1)-np.sum(n2))/np.sum(n2)):.2f}%")
        
        #### Both
        ax = axs[2]
        
        inclusive_total = pd.concat([self.charged_inclusive_mc, self.mixed_inclusive_mc])
        resonance_total = pd.concat([self.charged_resonance_mc, self.mixed_resonance_mc])

        n1, _, _ = bp.stacked([inclusive_total[var], 
                               resonance_total[var]],
                               bins=drawbins,
                               weights=[inclusive_total['hweight']*inclusive_total['br_weight']*inclusive_total['reweight']*inclusive_total['scaling'], 
                                        resonance_total['hweight']*resonance_total['br_weight']*resonance_total['reweight']*resonance_total['scaling']],
                               label=['inclusive', 'resonant'],
                               ax=ax)

        total_total = pd.concat([charged_total, mixed_total])
        self.n_c, _, _ = bp.hist(total_total[var],
                            bins=drawbins,
                            weights=total_total['hweight']*total_total['br_weight']*total_total['reweight']*total_total['scaling'],
                            label='hybrid model',
                            color='k',
                            lw=2,
                            ax=ax)
        if include_non_hweighted:
            n_n, _, _ = bp.hist(total_total[var],
                                bins=drawbins,
                                weights=total_total['rel_weight']*total_total['br_weight'],
                                label='non-reweighted',
                                ls='--', color='blue', lw=2,
                                ax=ax)


        n2, _, _ = bp.hist(inclusive_total[var], ls='--',
                           bins=drawbins, color='red', lw=2,
                           weights=inclusive_total["scaling"]*\
                                   inclusive_total["reweight"]*\
                                   inclusive_total['br_weight'],
                           label='Kagan Neubert',
                           ax=ax)
        
        for ax in axs:
            if var=='g_EB' or var=='gamma_mcEB':
                additional=' generated'
            elif var=='gamma_EB':
                additional=' reconstructed'


            make_gamma_axes(ax, additional=additional)

            ax.set_ylabel(rf"$5\times10^5 \times$BF$\times$Weights / {bin_width:.2f} GeV")

        if save:
            fig.savefig(f"{save}", bbox_inches='tight', dpi=300)
            
    def calculate_BR_variation_uncertainty(self,drawbins):
    
        self.cov_mtx = {}

        merged_varied_inc = pd.concat([self.charged_inclusive_mc, self.mixed_inclusive_mc])
        merged_res = pd.concat([self.charged_resonance_mc, self.mixed_resonance_mc])

        for key in self.charged_names+self.mixed_names:

            print(merged_varied_inc.loc[:, key+'-up'].isna().value_counts())
            print(merged_varied_inc.loc[:, key+'-down'].isna().value_counts())



            n_in_up,_ = np.histogram(merged_varied_inc.gamma_mcEB, 
                                    bins = drawbins, \
                                    weights = merged_varied_inc[key+'-up']*merged_varied_inc['reweight']*merged_varied_inc['scaling'])

            n_ex_up,_ = np.histogram(merged_res.gamma_mcEB, 
                                    bins = drawbins, \
                                    weights = merged_res[f'{key}-up'])

            n_in_down,_ = np.histogram(merged_varied_inc.gamma_mcEB, 
                                    bins = drawbins, \
                                    weights = merged_varied_inc[key+'-down']*merged_varied_inc['reweight']*merged_varied_inc['scaling'])

            n_ex_down,_ = np.histogram(merged_res.gamma_mcEB, 
                                    bins = drawbins, \
                                    weights = merged_res[f'{key}-down'])

            n_up = n_in_up+n_ex_up
            n_down = n_in_down+n_ex_down

            errors_up = n_up - self.n_c
            cov_up = np.outer(errors_up, errors_up)

            errors_down = n_down - self.n_c
            cov_down = np.outer(errors_down, errors_down)

            cov = (cov_up + cov_down) / 2

            self.cov_mtx[key] = cov



        summed_mtx = 0
        for mtx in self.cov_mtx.values():
            summed_mtx +=mtx
        diag = summed_mtx.diagonal()
        self.error_res = np.sqrt(diag) 

    def calculate_inclBR_variation_uncertainty(self,drawbins):

        self.cov_mtx_sig = {}
        

        merged_varied_inc = pd.concat([self.charged_inclusive_mc, self.mixed_inclusive_mc])

        merged_res = pd.concat([self.charged_resonance_mc, self.mixed_resonance_mc])

        n_in_up,_ = np.histogram(merged_varied_inc.gamma_mcEB, 
                                bins = drawbins, \
                                weights = merged_varied_inc['hweight-up']*merged_varied_inc["scaling"]*merged_varied_inc["reweight"])

        n_ex_up,_ = np.histogram(merged_res.gamma_mcEB, 
                                bins = drawbins)

        n_in_down,_ = np.histogram(merged_varied_inc.gamma_mcEB, 
                                    bins = drawbins, \
                                    weights = merged_varied_inc['hweight-down']*merged_varied_inc["scaling"]*merged_varied_inc["reweight"])

        n_ex_down,_ = np.histogram(merged_res.gamma_mcEB, 
                                    bins = drawbins)

        n_up = n_in_up+n_ex_up
        n_down = n_in_down+n_ex_down

        errors_up = n_up - self.n_c
        cov_up = np.outer(errors_up, errors_up)

        errors_down = n_down - self.n_c
        cov_down = np.outer(errors_down, errors_down)

        cov = (cov_up + cov_down) / 2

        diag = cov.diagonal()
        self.error_inc = np.sqrt(diag)

        
    def calculate_KNpar_variation_uncertainty(self, drawbins):

        self.cov_mtx_var = {}

        for n, (ch, md) in enumerate(zip(self.charged_variations, self.mixed_variations)):

            # varied_weight_ch = self.par_variation_hweights[f'charged-var-{n}'][np.add(
            #                                 np.digitize(self.charged_inclusive_mc.gamma_mcEB, 
            #                                             bins=self.charged_hybrid_bins),
            #                                 -1)
            #                                ]

            # varied_weight_md = self.par_variation_hweights[f"mixed-var-{n}"][np.add(
            #                                                                  np.digitize(self.mixed_inclusive_mc.gamma_mcEB, 
            #                                                                             bins=self.mixed_hybrid_bins),
            #                                                                 -1)
            #                                                                 ]

            # temp_bp_reweight = make_reweights(ch.g_EB,
            #                                   self.charged_inclusive_default.g_EB,
            #                                   self.reweight_bins,
            #                                   self.charged_inclusive_mc.gamma_mcEB,
            #                                   True
            #           )

            # temp_bz_reweight = make_reweights(md.g_EB,
            #                                   self.mixed_inclusive_default.g_EB,
            #                                   self.reweight_bins,
            #                                   self.mixed_inclusive_mc.gamma_mcEB,
            #                                   True
            #               )



            # self.charged_inclusive_mc.loc[:,f'par_weight_{n}'] = varied_weight_ch * (3.49e-4 / self.charged_leftover) * temp_bp_reweight

            # self.mixed_inclusive_mc.loc[:,f'par_weight_{n}'] = varied_weight_md * (3.49e-4 / self.mixed_leftover) * temp_bz_reweight

            # self.charged_resonance_mc.loc[:,f'par_weight_{n}'] = 1

            # self.mixed_resonance_mc.loc[:,f'par_weight_{n}'] = 1

            merged_varied_inc = pd.concat([self.charged_inclusive_mc, self.mixed_inclusive_mc])
            merged_res = pd.concat([self.charged_resonance_mc, self.mixed_resonance_mc])
            print(merged_varied_inc[f'par_weight_{n}'].isna().value_counts())
            merged_varied_inc[f'par_weight_{n}'].fillna(1, inplace=True)


            n_sigvar,_ = np.histogram(merged_varied_inc.gamma_mcEB, 
                                      bins = drawbins, \
                                      weights = merged_varied_inc[f'par_weight_{n}']*\
                                                merged_varied_inc[f'scaling']*\
                                                merged_varied_inc[f'par_reweight_{n}'])

            n_resvar,_ = np.histogram(merged_res.gamma_mcEB, 
                                      bins = drawbins)

            n_var = n_sigvar+n_resvar

            errors = n_var - self.n_c
            cov = np.outer(errors, errors)


            self.cov_mtx_var[f'{n}'] = cov

        summed_mtx_var = 0
        for mtx in self.cov_mtx_var.values():
            summed_mtx_var +=mtx
        diag = summed_mtx_var.diagonal()
        self.error_var = np.sqrt(diag)



def make_hybrid_weights(signal, resonance, bins, signal_weights = None, resonance_weights=None, density=False):

    H_exc, _ = np.histogram(resonance, bins=bins, weights=resonance_weights, density=density)
    H_inc, _ = np.histogram(signal, bins=bins, weights=signal_weights, density=density)

    weights = 1 - H_exc/H_inc
    
    
    weights[weights<0] = 0
    weights[np.isnan(weights)] = 1
    
    return weights

def make_reweights(df_target, df_old, bins, df_apply=None, density=False):

    H_target, _ = np.histogram(df_target, bins=bins, density=density)
    H_old, _ = np.histogram(df_old, bins=bins, density=density)
    

    
    weights = H_target/H_old
    
    
    weights[np.isinf(weights)] = 0
    weights[np.isnan(weights)] = 1
    print(weights)
    if df_apply is None:
        weight_column = weights[np.digitize(df_old, 
                                            bins)-1]
    else:
        weight_column = weights[np.digitize(df_apply, 
                                            bins)-1]
    
    return weight_column



