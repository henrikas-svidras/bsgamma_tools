
import numpy as np
from uncertainties import ufloat

def var_to_string(var):
    """Changes some of the commonly used variables that I often use to a nice string name with the expected unit
    Input:
        - (string) var name from root files
    Output:
        - (string) nicely formatted LaTex string
    """
    
    for swap in varsafe:
        var = var.replace(swap, varsafe[swap])

    return var

def cut_to_string(cut):
    """Changes a pandas query cut string to a nicely formatted latex cut
    Input:
        - (string) cut as you would put it in pandas.DataFrame `query` method
    Output:
        - (string) nicely formatted LaTex string
    """

    for swap in varsafe:
        cut = cut.replace(swap, varsafe[swap])
    
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
    "gamma_bins":np.array([1.4,1.6,1.8,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,5.0]),
}
varsafe = {
    ### For var_to_string

    "gamma_ECMS": r"$E_{\gamma}^{CMS}$, GeV",
    "gamma_E": r"$E_{\gamma}$, GeV",
    "gamma_EB": r"$E_{\gamma}^B$, GeV",

    ### For var_to_string

    "Bsig_pi0Prob":r"$\mathcal{P}_{\pi^0\rightarrow\gamma\gamma}$",
    "Bsig_etaProb":r"$\mathcal{P}_{\eta\rightarrow\gamma\gamma}$",


    
}

scales_plus = { 
    'K*(1410)+': ufloat(0.000027100, 0.000007000), 
    'K*(892)+': ufloat(0.000039180, 0.000002200), 
    'K*(1680)+': ufloat(0.000066700, 0.000015500), 
    'K(2)*(1430)+': ufloat(0.000013770, 0.000004000), 
    'K(1)(1270)+': ufloat(0.000043800, 0.000006500),
    'K(1)(1400)+': ufloat(0.000009700, 0.000004500),
    'rho(770)+': ufloat(0.000000980,0.000000250),
}
scales_plus['Xsgamma'] = 3.49e-4 - np.sum(list(scales_plus.values()))

scales_zero = { 
    'K(1)(1400)0':ufloat(0.000006500, 0.000006500), 
    'K(1)(1270)0':ufloat(0.000043000, 0.000015000), 
    'K*(892)0': ufloat(0.000041770,0.000002500), 
    'rho(770)0':ufloat(0.000000860, 0.000000150), 
    'K*(1680)0':ufloat(0.000001700, 0.000001700), 
    'K(2)*(1430)0':ufloat(0.000012370, 0.000002400), 
    'omega(782)':ufloat(0.000000440, 0.000000180),
}
scales_zero['Xsgamma'] = 3.49e-4 - np.sum(list(scales_zero.values()))