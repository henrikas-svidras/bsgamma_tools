import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    return (max(median - const, min(data)), min(median + const, max(data)))


def document_cut(dict_with_dfs, drawer, cut, signal_name=None, truth_var=None, n_bins=45, ranges=None, save_fig=False):

    """
     Documents a cut. Takes as input a dictionary with pandas dfs, e.g. : {'ccbar':dataframe1, 'signal_name':dataframe2....} etc. Returns a plot where
     it compares the amount of signa cut away and the amount of background cut away. Everything except signal_name is stacked together.

    Inputs:
     - dict_with_dfs (dict of pandas DataFrames): the datasets containing signal and other datasets
     - drawer (string): column name to be drawn in from datasets
     - cut (string): cut to document, e.g. 'column_name > 1.5'
     - signal_name (string, default=None): if given then looks for 'signal_name' key in dict_with_dfs. Else, searched for a key 'signal'
     - truth_var (string, default=None): if given then truth selects the signal and stacked backgrounds.
     - n_bins (int, default=45): number of bins in the plot. Automatically overriden if the cut is e.g. "column_name==2".
     - ranges (tuple of 2 floats, default=None): min and max of x axis in the plot. If not given, then min(signal), max(signal) are taken.
     - save_fig (bool or str, default=False): whether to save figure or not. If string is given, then saves it with the given name

     Outputs:
     - .png file of the documented cut.
    """

    ### Get the number of signal events and remove it from the dictionary
    signal_name = 'signal' if signal_name is None else signal_name
    signal_df = dict_with_dfs.pop(signal_name) # This will return an error if "signal" is not specified
    signal_len = len(signal_df)

    ### Get all the keys to stack and calculate number of entries
    keys = list(dict_with_dfs)
    stack_len = sum([len(dict_with_dfs[df][drawer]) for df in keys])

    if truth_var:
        true_signal_len = len(signal_df.query(f'{truth_var}==1')[drawer])
        fake_signal_len = len(signal_df.query(f'{truth_var}==0')[drawer])
        bkg_stack_len = sum([len(dict_with_dfs[df].query(f'{truth_var}==0')[drawer]) for df in keys])

    ### Get all the keys to stack and calculate number of entries
    stack = [df[drawer].values for df in dict_with_dfs.values()]

    ranges = (min(signal_df), max(signal_df)) if not ranges else ranges

    cut_dict = {}
    for key, item in dict_with_dfs.items():
        cut_dict[key] = item.query(cut)

    cut_stack_len = sum([len(cut_dict[df][drawer]) for df in keys[:-1]])
    cut_signal_len = len(signal_df.query(cut)[drawer])

    if truth_var:
        cut_true_signal_len = len(signal_df.query(cut).query(f'{truth_var}==1')[drawer])
        cut_fake_signal_len = len(signal_df.query(cut).query(f'{truth_var}==0')[drawer])
        cut_bkg_stack_len = sum([len(cut_dict[df].query(f'{truth_var}==0')[drawer]) for df in keys[:-1]])

        true_signal_rate = 1-cut_true_signal_len/true_signal_len if true_signal_len > 0 else 0
        fake_signal_rate = 1-cut_fake_signal_len/fake_signal_len if fake_signal_len > 0 else 0
        mc_background_rate = 1-cut_bkg_stack_len/bkg_stack_len if bkg_stack_len > 0 else 0
    else:
        background_rate = 1-cut_stack_len/stack_len if stack_len > 0 else 0
        signal_rate = 1-cut_signal_len/signal_len if signal_len > 0 else 0

    fig, axis = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    ### This is to configure the naming of the output plots
    split_simbol = ">" if ">" in cut else ""
    split_simbol = "<" if "<" in cut and not split_simbol else split_simbol
    split_simbol = "==" if "==" in cut and not split_simbol else split_simbol

    cut_area = cut.split(split_simbol)
    place = cut_area.index(drawer)
    cut_area.remove(drawer)
    ############################################################

    ## For integer value cuts, e.g. "rank" the binning is not continuous, so additional setup is needed
    if split_simbol == '==':
        n_bins = np.linspace(min(signal_df)-0.25, max(signal_df)+0.25, 2*int(max(signal_df)))

    n1, bi1, p1 = b2plot.stacked(stack, bins=n_bins, range=ranges,
                                                     ax=axis,
                                                     label=keys[:-1])

    n2, bi2, p2 = b2plot.hist(signal_df, bins=n_bins, range=ranges,
                                                    scale=len(signal_df)*[stack_len/signal_len],
                                                    color='red',
                                                    ax=axis,
                                                    label=keys[-1])
    ### Setup the naming scheme and axes of the plot
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

    ### Legend to include the amount of data cut away, axes and output formatting.
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

def ratio_error(a,b):
    return np.sqrt(a/b**2 + (a**2/b**3))

def train_and_document(variable_list, test_dataframe, train_dataframe,
                       cuts=None, depth=2, num_trees=200,
                       spectators=['gamma_E','Btag_Mbc','Btag_deltaE'],
                       truth_var=None ,vars_to_compare = None,
                       save_path='',save_prefix='mva',
                       log_scale = True,
                       flatness_loss=-1.0,
                       number_of_flatness_features=0):
    """
     A wrapper to fully train and document a training.

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

    if isinstance(cuts, dict):
        cut_string = '&'.join(list(cuts.values()))
    elif isinstance(cuts, str):
        cut_string = cuts
    elif cuts is None:
        pass
    else:
        print(f'Unrecognized cut specified {cuts}')

    if cuts is not  None:
        train_dataframe_cut = train_dataframe.query(cut_string)
        test_dataframe_cut = test_dataframe.query(cut_string)
    else:
        train_dataframe_cut = train_dataframe
        test_dataframe_cut = test_dataframe
    test_dataframe_cut.fillna(0)
    train_dataframe_cut.fillna(0)
    train_x = train_dataframe_cut[variable_list].values[:,0:-1]
    train_y = train_dataframe_cut[variable_list].values[:,-1]
    test_x = test_dataframe_cut[variable_list].values[:,0:-1]
    test_y = test_dataframe_cut[variable_list].values[:,-1]

    model_grad = train_FastBDT(train_x, train_y, depth =depth, num_trees=num_trees, flatness_loss=flatness_loss, number_of_flatness_features=number_of_flatness_features)
    roc_curve_name = save_prefix+'_ROC.pdf' if save_prefix else None
    separation_curve_name = save_prefix+'_separation.pdf' if save_prefix else None
    feat_importance_name = save_prefix+'_features.pdf' if save_prefix else None

    show_ROC(model_grad, train_x,test_x,train_y ,test_y,saveas=save_path+roc_curve_name)
    show_separation(model_grad, train_x, test_x, train_y, test_y,saveas=save_path+separation_curve_name,log=log_scale)
    ranked_importances = show_feature_importance(model_grad,list(train_dataframe_cut[variable_list].columns),saveas=save_path+feat_importance_name)
    best12 = list(ranked_importances.keys())[:12]
    best12.append(variable_list[-1])

    best_train_x = train_dataframe_cut[best12].values[:,0:-1]
    best_train_y = train_dataframe_cut[best12].values[:,-1]
    best_test_x = test_dataframe_cut[best12].values[:,0:-1]
    best_test_y = test_dataframe_cut[best12].values[:,-1]

    best_model_grad = train_FastBDT(best_train_x, best_train_y, depth =depth, num_trees=num_trees)

    show_ROC(best_model_grad, best_train_x,best_test_x,best_train_y ,best_test_y,saveas=save_path+'best_'+roc_curve_name)
    show_separation(best_model_grad, best_train_x, best_test_x, best_train_y, best_test_y,saveas=save_path+'best_'+separation_curve_name,log=log_scale)
    best_ranked_importances = show_feature_importance(best_model_grad,list(train_dataframe_cut[best12].columns),saveas=save_path+'best_'+feat_importance_name)

    if spectators:
      columns_corr = train_dataframe_cut[spectators]

    pandas_cols_corr = train_dataframe_cut[best12].join(columns_corr) if spectators else train_dataframe_cut[best12]
    #b2plot.flat_corr_matrix(pandas_cols_corr.query('isNotContinuumEvent==1').drop(columns=['isNotContinuumEvent']),
    #n_labels = 0,fontsize = 11,label_size = 11, label_rotation = 45)

    fig = plt.gcf()
    fig.suptitle(r'$B\bar{B}$ events')
    if save_prefix:
        corr_mat = plt.gcf()
        corr_mat.savefig(save_path+save_prefix+'_correlation_BBmatrix.pdf')

    #b2plot.flat_corr_matrix(pandas_cols_corr.query('isNotContinuumEvent==0').drop(columns=['isNotContinuumEvent']),
    #                        n_labels = 0,fontsize = 11,label_size = 11, label_rotation = 45)

    fig = plt.gcf()
    fig.suptitle(r'Continuum events')
    if save_prefix:
        corr_mat = plt.gcf()
        corr_mat.savefig(save_path+save_prefix+'_correlation_CONTmatrix.pdf')


    show_corrmatrix(columns=pandas_cols_corr,emphasise=3,saveas=save_path+save_prefix+'_corrs.pdf')

    if vars_to_compare:
        roc_eff_purr_compare(vars_to_compare+[[best12,'Best 12']],train_dataframe_cut,test_dataframe_cut,saveas =save_path+ save_prefix+'_trainvar_comparison.pdf')


    #test_dataframe_cut['contsup'] = best_model_grad.predict(test_x)
    #ratio_to_signal(test_dataframe_cut,unit='GeV',continuum_sup_var_cut='contsup>0.3',saveas=save_path+save_prefix+'_contsup0p3_signal_ratio.pdf')
    #ratio_to_signal(test_dataframe_cut,unit='GeV',saveas=save_path+save_prefix+'_nosup_signal_ratio.pdf')

    return {'total':model_grad, 'best':best_model_grad}, ranked_importances


def data_mc_compare(mc, data, drawer, 
                    bins=20, ranges=None, 
                    colors=None, labels=None, dataname='Data', 
                    xname=None, yname='Candidates',
                    normalise=True, percentile=5,
                    lumis=None,
                    ax=None):

    mc_stack = {}
    mc_total = 0
    for key, i in mc.items():
        mc_stack[key] = i[drawer]
        mc_total+=len(i)
    if colors is None:
        from b2plot.colors import b2helix
        colors = b2helix(len(mc_stack))

    data_points = data[drawer]

    if ax is None:
        fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios':[10, 3], 'hspace': 0.07},sharex=True)

    weights = []
    for key in mc_stack:
        i = mc_stack[key]
        weights.append(np.ones(len(i)))
        if lumis:
            pass
        if normalise:
            weights[-1] *= len(data_points)/mc_total

    if ranges is not None:
        bins = np.linspace(ranges[0], ranges[1], bins)
    
    y1, bin_corners, stuff = ax[0].hist(list(mc_stack.values()), stacked=True,
           bins=bins, weights=weights,
           histtype='stepfilled', edgecolor='black',
           lw=0.5, color=colors, label=labels)

    if len(mc_stack)==1:
        y1 = y1
    else:
        y1 = y1[-1]

    bin_centers = np.array((bin_corners[:-1] + bin_corners[1:]) / 2)
    bin_hwidth = np.array(bin_corners[1:]  - bin_corners[:-1])

    y2,_, bin_centers, errs = b2plot.errorhist(data_points,
                                       bins=bins,
                                       hatch='/',
                                       color='black',label=dataname, 
                                       ax=ax[0])

    ratio = y2/y1
    ratio_err = ratio_error(y2, y1)

    line,caps,_ = ax[1].errorbar(bin_centers,ratio, yerr = ratio_err, xerr = bin_hwidth*0.5,
                         fmt='ko',
                         markersize=3,
                         ecolor = 'black')
    ax[1].bar(bin_centers,ratio-1,width=bin_hwidth,color='gray',alpha=0.5,bottom=1)
    ax[1].set_xlim(ranges[0],ranges[1])
    yl = np.percentile(ratio, 100-percentile)
    ym = np.percentile(ratio, 0+percentile)
    dev = max(yl-1, 1-ym)
    ax[1].set_ylim(1-dev,1+dev)
    ax[0].legend(loc='best')
    ax[0].set_ylabel(f'{yname}/bin')
    ax[1].set_xlabel(f'{xname}')
    ax[1].set_ylabel(r'$\frac{\mathrm{Data}}{\mathrm{MC}}$')
