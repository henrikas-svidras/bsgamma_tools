
import numpy as np

def var_to_string(var):
    """Changes some of the commonly used variables that I often use to a nice string name with the expected unit
    Input:
        - (string) var name from root files
    Output:
        - (string) nicely formatted LaTex string
    """
    
    for swap in safe:
        var = var.replace(swap, safe[swap])

    return var

def cut_to_string(cut):
    """Changes a pandas query cut string to a nicely formatted latex cut
    Input:
        - (string) cut as you would put it in pandas.DataFrame `query` method
    Output:
        - (string) nicely formatted LaTex string
    """
    cut

    for swap in safe:
        cut = cut.replace(swap, safe[swap])
    
    cut = cut.replace(" < ", "<"). replace("<", " < ")

    return cut


safe = {
    "Inclusive Xsu modes": [30353,-30353,
                            323, 10323, 325, 20323, 100323, 30323,
                           -323,-10323,-325,-20323,-100323,-30323
                           ],
    "Inclusive Xsd modes": [30343,-30343,
                            313, 10313, 315, 20313, 113, 223, 30313,
                           -313,-10313,-315,-20313,-113,-223,-30313,
                           ],

    ### For var_to_string

    "gamma_ECMS": r"$E_{\gamma}^{CMS}$, GeV",
    "gamma_E": r"$E_{\gamma}$, GeV",
    "gamma_EB": r"$E_{\gamma}^B$, GeV",

    ### For var_to_string

    "Bsig_pi0Prob":r"$\mathcal{P}_{\pi^0\rightarrow\gamma\gamma}$",
    "Bsig_etaProb":r"$\mathcal{P}_{\eta\rightarrow\gamma\gamma}$",

    "gamma_bins":np.array([1.4,1.6,1.8,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,5.0]),
}