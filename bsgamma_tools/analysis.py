import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math
import random

import b2plot

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
    return (max(median - const,min(data)), min(median + const,max(data)))

def document_cut(dict_with_dfs,drawer, cut, truth_var = None ,n_bins=45,ranges = None, save_fig = False):

    """
     Documents a cut.

    Inputs:
     -
     Outputs:
     -
    """

    keys = list(dict_with_dfs)
    stack_len = sum([len(dict_with_dfs[df][drawer]) for df in keys[:-1]])
    signal_len = len(dict_with_dfs['signal'][drawer])
    signal = dict_with_dfs['signal'][drawer].values

    if truth_var:
        true_signal_len =  len(dict_with_dfs['signal'].query(f'{truth_var}==1')[drawer])
        fake_signal_len =  len(dict_with_dfs['signal'].query(f'{truth_var}==0')[drawer])
        bkg_stack_len = sum([len(dict_with_dfs[df].query(f'{truth_var}==0')[drawer]) for df in keys[:-1]])

    stack = [dict_with_dfs[df][drawer].values for df in keys[:-1]]
    #ranges = calculate_bounds(signal) if not ranges else ranges
    ranges = (min(signal),max(signal))

    split_simbol = ">" if ">" in cut else ""
    split_simbol = "<" if "<" in cut and not split_simbol else split_simbol
    split_simbol = "==" if "==" in cut and not split_simbol else split_simbol

    cut_area = cut.split(split_simbol)
    place = cut_area.index(drawer)
    cut_area.remove(drawer)

    cut_dict = {}
    for key,item in dict_with_dfs.items():
        cut_dict[key]=item.query(cut)

    cut_stack_len = sum([len(cut_dict[df][drawer]) for df in keys[:-1]])
    cut_signal_len = len(cut_dict['signal'][drawer])

    if truth_var:
        cut_true_signal_len =  len(cut_dict['signal'].query(f'{truth_var}==1')[drawer])
        cut_fake_signal_len =  len(cut_dict['signal'].query(f'{truth_var}==0')[drawer])
        cut_bkg_stack_len = sum([len(cut_dict[df].query(f'{truth_var}==0')[drawer]) for df in keys[:-1]])

        true_signal_rate = 1-cut_true_signal_len/true_signal_len
        fake_signal_rate = 1-cut_fake_signal_len/fake_signal_len
        mc_background_rate = 1-cut_bkg_stack_len/bkg_stack_len
    else:
        background_rate = 1-cut_stack_len/stack_len
        signal_rate = 1-cut_signal_len/signal_len


    fig, axis  = plt.subplots(1, 1,figsize=(fig_width,fig_height))

    if split_simbol == '==':
        n_bins = np.linspace(min(signal)-0.25,max(signal)+0.25,2*int(max(signal)))
    n1,bi1,p1 = b2plot.stacked(stack, bins=n_bins, range = ranges,
                                                ax = axis,
                                                label=keys[:-1])

    n2,bi2,p2 = b2plot.hist(signal, bins=n_bins, range = ranges,
                                                    scale=stack_len/signal_len,
                                                    color = 'red',
                                                    ax = axis,
                                                    label=keys[-1])



    if len(cut_area)==1:
        if (split_simbol == '>' and place==0) \
           or (split_simbol == '<' and place==1):
                minx = ranges[0]
                maxx = float(cut_area[0])
                axis.axvspan(minx,maxx,alpha=0.25,hatch='/',color='gray')
        elif (split_simbol == '<' and place == 0) \
             or (split_simbol == '>' and place == 1):
                minx = float(cut_area[0])
                maxx = ranges[1]
                axis.axvspan(minx,maxx,alpha=0.25,hatch='/',color='gray')
        elif (split_simbol == '=='):
                minx1 = n_bins[0]
                maxx1 = minx1 if float(cut_area[0])-0.5 < minx1 else float(cut_area[0])-0.5
                maxx2 = n_bins[-1]
                minx2 = maxx2 if float(cut_area[0])+0.5 > maxx2 else float(cut_area[0])+0.5
                print(minx1,maxx1,minx2,maxx2)
                axis.axvspan(minx1,maxx1,alpha=0.25,hatch='/',color='gray')
                axis.axvspan(minx2,maxx2,alpha=0.25,hatch='/',color='gray')

    elif len(cut_area)==2:
        maxx = max(float(cut_area[0]),float(cut_area[1]))
        minx = min(float(cut_area[0]),float(cut_area[1]))
        axis.axvspan(minx,maxx,alpha=0.25,hatch='/',color='gray')

    if truth_var:
        axis.plot([], [], ' ', label=f"Bkg rej: {mc_background_rate*100:.2f}%")
        axis.plot([], [], ' ', label=f"True Sig rej: {true_signal_rate*100:.2f}%")
        axis.plot([], [], ' ', label=f"Fake Sig rej: {fake_signal_rate*100:.2f}%")
    else:
        axis.plot([], [], ' ', label=f"Bkg rej: {background_rate*100:.2f}%")
        axis.plot([], [], ' ', label=f"Sig rej: {signal_rate*100:.2f}%")

    axis.legend(bbox_to_anchor=(1.04,0.5),loc="center left")
    axis.autoscale(tight='True')
    axis.set_xlabel(drawer)
    if isinstance(n_bins, int):
      axis.set_ylabel(f'Candidates per {(ranges[1]-ranges[0])/n_bins:.2g}')
    else:
      axis.set_ylabel(f'Candidates per 1')
    fig.show()

    if save_fig:
        if isinstance(save_fig, str):
            savename = save_fig
        else:
            savename = cut.replace('==','eq').replace('>','morethan').replace('<','lessthan').replace('.','p')+'.png'
        fig.savefig(savename,bbox_inches='tight')

def ratio_to_signal(pddf,comparison_var='gamma_E',signal_var='isBsgamma',bb_var='isNotContinuumEvent',continuum_sup_var_cut = 'True',ranges=(1.4,4),unit='', saveas=None):
    """
     Creates a stacked plot for ratio with signal
    Inputs:
     - pddf (pandas DataFrame): must contain signal, continuum and bbar backgrounds
     - comparison_var (str): variable to plot against, must be acessible in pddf
     - signal_var (str): variable to discriminate signal, must ve acessible in pddf
     - bb_var (str): variable to discriminate bb bar background from continuum, must be acessible in pddf
     - continuum_sup_var_cat (str): an additional cut that can be introduced, for example, a continuum supression cut
     - ranges (float, float): drawing range
     - unit (float, float): unit to appear on Y axis title
     - saveas (str): if not None then name to save as.
     Outputs:
     - plots a figure of a ration of stacked signal, bbar, continuum and ratio between signal/(bbar+cont) bin by bin
    """
    n_bins = 30
    golden = (1 + 5 ** 0.5) / 2
    fig_width = 13
    fig_height = fig_width/golden

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,golden],'hspace':0.2},
                                      figsize=(fig_width,fig_height))

    hists = [pddf.query('&'.join([f'{signal_var}==1',continuum_sup_var_cut]))[comparison_var],
             pddf.query('&'.join([f'{bb_var}==1 & {signal_var}==0',continuum_sup_var_cut]))[comparison_var],
             pddf.query('&'.join([f'{bb_var}==0',continuum_sup_var_cut]))[comparison_var]]

    n0,bins0, patches0 = a0.hist(hists, n_bins, histtype='bar',stacked=True,
                                 color=[a0._get_lines.get_next_color(),a0._get_lines.get_next_color(),a0._get_lines.get_next_color()], 
                                 label=['B->Xs gamma','BB','Cont'])

    divided_stacks = [n0[0][i]/(n0[1][i]+n0[2][i]) if (n0[1][i]+n0[2][i])>0 else 0 for i in range(len(n0[1]))]

    stack_errors = [ratio_error(n0[0][i],n0[1][i]+n0[2][i]) if (n0[1][i]+n0[2][i])>0 else 0 for i in range(len(n0[1]))]

    bin_centers = (bins0[1]-bins0[0])/2+bins0[0:-1]
    a1.errorbar(bin_centers,divided_stacks,yerr=stack_errors,fmt='.',c=next(a0._get_lines.prop_cycler)["color"])

    a0.legend(loc='best')
    ranges = (min(pddf[comparison_var]),max(pddf[comparison_var])) if not ranges else ranges
    a0.set_xlim(*ranges)
    a0.set_ylabel(f'Candidates/{(ranges[1]-ranges[0])/n_bins:.2f} {unit}')
    a1.set_xlabel(comparison_var)
    a1.set_ylabel(f'Sig/Bkg ratio')
    a1.set_xlim(*ranges)
    a1.set_ylim(bottom=-0.001)
    a1.set_xticks([])

    if saveas:
        f.savefig(saveas,bbox_inches='tight')

    f.show()

def ratio_error(a,b):
    return math.sqrt(math.sqrt(a)/b**2 - math.sqrt(b)*(a/b**2)**2)

def train_and_document(variable_list, cuts, test_dataframe, train_dataframe, depth=2, num_trees=200, spectators=['gamma_E','Btag_Mbc','Btag_deltaE'] ,vars_to_compare = None,save_path='',save_prefix='mva', log_scale = True):
    """
     Documents a cut.
    Inputs:
     -
     Outputs:
     -
    """
    from bsgamma_tools.mva import split_set
    from bsgamma_tools.mva import train_FastBDT
    from bsgamma_tools.mva import show_ROC
    from bsgamma_tools.mva import show_separation
    from bsgamma_tools.mva import equalise_bkg_sig
    from bsgamma_tools.mva import show_feature_importance
    from bsgamma_tools.mva import show_corrmatrix
    from bsgamma_tools.mva import roc_eff_purr_compare

    cut_string = '&'.join(list(cuts.values()))
    print(cut_string)
    #columns_train = train_dataframe.Filter(cut_string).AsNumpy(columns=variable_list)
    #_, _, _, _,train_x,train_y = split_set(columns_train, size=1)
    #columns_test = test_dataframe.Filter(cut_string).AsNumpy(columns=variable_list)
    #_, _, _, _,test_x,test_y = split_set(columns_test, size=1)
    train_dataframe.query(cut_string,inplace=True)
    train_dataframe.fillna(0,inplace=True)
    test_dataframe.query(cut_string,inplace=True)
    test_dataframe.fillna(0,inplace=True)
    train_x = train_dataframe[variable_list].values[:,0:-1]
    train_y = train_dataframe[variable_list].values[:,-1]
    test_x = test_dataframe[variable_list].values[:,0:-1]
    test_y = test_dataframe[variable_list].values[:,-1]

    model_grad = train_FastBDT(train_x, train_y, depth =depth, num_trees=num_trees)
    roc_curve_name = save_prefix+'_ROC.pdf' if save_prefix else None
    separation_curve_name = save_prefix+'_separation.pdf' if save_prefix else None
    feat_importance_name = save_prefix+'_features.pdf' if save_prefix else None

    show_ROC(model_grad, train_x,test_x,train_y ,test_y,saveas=save_path+roc_curve_name)
    show_separation(model_grad, train_x, test_x, train_y, test_y,saveas=save_path+separation_curve_name,log=log_scale)
    ranked_importances = show_feature_importance(model_grad,list(train_dataframe[variable_list].columns),saveas=save_path+feat_importance_name)
    best12 = ranked_importances[:13]
    best12.append(variable_list[-1])

    best_train_x = train_dataframe[best12].values[:,0:-1]
    best_train_y = train_dataframe[best12].values[:,-1]
    best_test_x = test_dataframe[best12].values[:,0:-1]
    best_test_y = test_dataframe[best12].values[:,-1]

    best_model_grad = train_FastBDT(best_train_x, best_train_y, depth =depth, num_trees=num_trees)

    show_ROC(best_model_grad, best_train_x,best_test_x,best_train_y ,best_test_y,saveas=save_path+'best_'+roc_curve_name)
    show_separation(best_model_grad, best_train_x, best_test_x, best_train_y, best_test_y,saveas=save_path+'best_'+separation_curve_name,log=log_scale)
    ranked_importances = show_feature_importance(best_model_grad,list(train_dataframe[best12].columns),saveas=save_path+'best_'+feat_importance_name)

    if spectators:
      columns_corr = train_dataframe[spectators]

    pandas_cols_corr = train_dataframe[best12].join(columns_corr) if spectators else train_dataframe[best12]
    b2plot.flat_corr_matrix(pandas_cols_corr.query('isNotContinuumEvent==1').drop(columns=['isNotContinuumEvent']),
                            n_labels = 0,fontsize = 11,labelsize = 11, label_distance = 30, label_rotation = 45)
    if save_prefix:
        corr_mat = plt.gcf()
        corr_mat.savefig(save_path+save_prefix+'_correlation_BBmatrix.pdf')

    b2plot.flat_corr_matrix(pandas_cols_corr.query('isNotContinuumEvent==0').drop(columns=['isNotContinuumEvent']),
                            n_labels = 0,fontsize = 11,labelsize = 11, label_distance = 30, label_rotation = 45)
    if save_prefix:
        corr_mat = plt.gcf()
        corr_mat.savefig(save_path+save_prefix+'_correlation_CONTmatrix.pdf')

    show_corrmatrix(columns=pandas_cols_corr,emphasise=3,saveas=save_path+save_prefix+'_corrs.pdf')

    if vars_to_compare:
        vars_to_compare.append([best12,'Best 12'])
        roc_eff_purr_compare(vars_to_compare,train_dataframe,test_dataframe,saveas =save_path+ save_prefix+'_trainvar_comparison.pdf')


    test_dataframe['contsup'] = model_grad.predict(test_x)
    ratio_to_signal(test_dataframe,unit='GeV',continuum_sup_var_cut='contsup>0.3',saveas=save_path+save_prefix+'_contsup0p3_signal_ratio.pdf')
    ratio_to_signal(test_dataframe,unit='GeV',saveas=save_path+save_prefix+'_nosup_signal_ratio.pdf')
