import numpy as np

def pdg_to_name(input_id, latex=False):
    """
    Transforms a pdg id to a name. Can return a Latex name.

    """

    from particle import Particle

    safe = {
        '30353':'Xsu',
        '30343':'Xsd'
    }

    if np.isnan(int(input_id)):
        return 'nan'
    else:
        if input_id in safe.keys():
            return safe[input_id]
        input_id = int(input_id)
        try:
            if latex:
                return f'${Particle.from_pdgid(input_id).latex_name}$'
            else:
                return Particle.from_pdgid(input_id).name
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
    return charged_fei_decays[input_id]
