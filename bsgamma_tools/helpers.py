from particle import Particle
import numpy as np
import random
import string
import warnings

def pdg_to_name(input_id, latex=False):
    """
    Transforms a pdg id to a name. Can return a Latex name.

    """

    from particle import Particle

    safe = {
        '30353':'Xsu',
        '-30353':'-Xsu',
        '30343':'Xsd',
        '-30343':'-Xsd'
    }

    if np.isnan(input_id):
        return 'nan'
    else:
        input_id = int(input_id)
        if str(input_id) in safe.keys():
            return safe[str(input_id)]
        try:
            part = Particle.from_pdgid(input_id)
            if latex:
                return f'${part.latex_name}$'
            else:
                return part.name
            #safe[str(input_id)] = part.name
        except:
            return f'Unknown{input_id}'

def charged_dmID_to_latex(input_id):
    charged_fei_decays = [
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} \pi^{+} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} \pi^{+} \pi^{0} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} \pi^{+} \pi^{+} \pi^{-}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} \pi^{+} \pi^{+} \pi^{-} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} D^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} D^{+} K_{S}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} D^{+} K_{S}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} D^{+*} K_{S}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} D^{+*} K_{S}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} D^{0} K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} D^{0} K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} D^{0 *} K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} D^{0 *} K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D_{s}^{+} \bar{D}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} \pi^{+} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} \pi^{+} \pi^{0} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} \pi^{+} \pi^{+} \pi^{-}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} \pi^{+} \pi^{+} \pi^{-} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D_{s}^{+*} \bar{D}^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D_{s}^{+} \bar{D}^{0 *}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D^{-} \pi^{+} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D^{-} \pi^{+} \pi^{+} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow J / \psi K^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow J / \psi K^{+} \pi^{+} \pi^{-}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow J / \psi K^{+} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow J / \psi K_{S}^{0} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \Lambda_{c}^{-} p \pi^{+} \pi^{0}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \Lambda_{c}^{-} p \pi^{+} \pi^{-} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0} p \bar{p} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \bar{D}^{0 *} p \bar{p} \pi^{+}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D^{+} p \bar{p} \pi^{+} \pi^{-}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow D^{+*} p \bar{p} \pi^{+} \pi^{-}$",
        r"$B_{\mathrm {had }}^{+} \rightarrow \Lambda_{c}^{-} p \pi^{+}$",
                         ]
    if input_id>len(charged_fei_decays):
        return 'none'
    return charged_fei_decays[input_id]

def charged_dmID_to_latex(input_id):
    return 'not yet implemented'

def punzi_FOM(S, B, S0, gaussian=3):
    """
    Implemented in the spirit of
    https://arxiv.org/pdf/physics/0308063.pdf 

    The Putzi figure of merit. Performs better than S/sqrt(S+B).

    Inputs:
        - S: signal count with your cut introduced.
        - B: background cut with your cut introduced.
        - S0: amount of signal without any cuts.
        - gausssian
    Outputs:
        - punzi figure of merit

    """
    if B>0:
        punzi_figure_of_merit = S/S0 * 1/(gaussian/2 + np.sqrt(B))
    else:
        punzi_figure_of_merit = 0
    return punzi_figure_of_merit

def putzi_FOM(*args):
    warnings.warn("Spelled incorrectly -> please call punzi_FOM() in the future")
    return punzi_FOM(*args)

def apply_cut_to_pandas_dict(df_dict, cut=None):
    if cut is not None:
      cut_df_dict = {key:df.query(cut) for key, df in df_dict.items()}
      return cut_df_dict
    else:
      return df_dict

def stacked_background(df, by, max_by=10, reverse = True, ax=None, colors = None, pdgise=True,
                       isolate=None, isolate_name=None, other_name='other', legend_suffix='', add_fraction=True):

    pdg_codes = df[by].value_counts().keys()
    if len(pdg_codes)<max_by:
        max_by=len(pdg_codes)+1
    if pdgise:
        pdg_names = [legend_suffix+pdg_to_name(i,True) for i in pdg_codes[:max_by-1]] + [legend_suffix+other_name]
    else:
        pdg_names = [legend_suffix+f'{i}' for i in pdg_codes[:max_by-1]] + [legend_suffix+other_name]

    top = pdg_codes[:max_by-1]

    if isolate is not None:
        df_isolate = df.query(f'{isolate}')
        df_to_split = df.query(f'~({isolate})')
    else:
        df_to_split = df 

    draw_stack = [df_to_split.query(f'{by}=={code}') for code in top] + [df_to_split[~df_to_split[by].isin(top)]]

    if isolate is not None:
        draw_stack.append(df_isolate)
        if isolate_name is not None:
            pdg_names.append(legend_suffix+isolate_name)
        else:
            pdg_names.append(legend_suffix+isolate)

    if add_fraction:
        fractions = [len(df.query(f"{by}=={code}"))/len(df) for code in top]
        fractions = fractions + [len(df_to_split)/len(df)-sum(fractions)]
        if isolate is not None:
             fractions = fractions + [df_isolate/df]
        pdg_names = [f'{name} ({fraction:.2f})' for name, fraction in zip(pdg_names, fractions)]

    if reverse:
        draw_stack.reverse()
        pdg_names.reverse()
    return draw_stack, pdg_names

def random_string(length=5):
    random_source = string.ascii_letters + string.digits + string.punctuation
    # select 1 lowercase
    password = random.choice(string.ascii_lowercase)
    # select 1 uppercase
    password += random.choice(string.ascii_uppercase)
    # select 1 digit
    password += random.choice(string.digits)
    # select 1 special symbol
    password += random.choice(string.punctuation)

    # generate other characters
    for i in range(length-4):
        password += random.choice(random_source)

    password_list = list(password)
    # shuffle all characters
    random.SystemRandom().shuffle(password_list)
    password = ''.join(password_list)
    return password

def create_bins(input_list, var = 'gamma_EB'):
    beanlist = []
    for a, b in zip(input_list[:-1], input_list[1:]):
        beanlist.append(f'{b}>{var}>={a}')
    return beanlist

