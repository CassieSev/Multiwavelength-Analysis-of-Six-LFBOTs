from   astropy import table
from   astropy.io import ascii, fits
import copy
import corner
import numpy as np
from   plotsettings import *
from   prospect.models.sedmodel import SedModel, PolySpecModel
from   prospect.sources import CSPSpecBasis, FastStepBasis
from   prospect.sources.constants import cosmo
from   prospect.utils.obsutils import fix_obs
from   scipy.special import gamma, gammainc
from   sedpy import observate

filter_systematics={'GALEX_FUV': 0.1, 'GALEX_NUV': 0.1,
                    'UVOT_UVW2': 0.1, 'UVOT_UVM2': 0.1, 'UVOT_UVW1': 0.1, 'UVOT_U': 0.1, 'UVOT_B': 0.1, 'UVOT_V': 0.1,
                    'SDSS_U': 0.05, 'SDSS_G': 0.05, 'SDSS_R': 0.05, 'SDSS_I': 0.05, 'SDSS_Z': 0.05,
                    'PANSTARRS_G': 0.05, 'PANSTARRS_R': 0.05, 'PANSTARRS_I': 0.05, 'PANSTARRS_Z': 0.05, 'PANSTARRS_Y': 0.05,
                    'JOHNSON_U': 0.05, 'JOHNSON_B': 0.05, 'BESSELL_V': 0.05, 'COUSINS_R': 0.05, 'BESSELL_I': 0.05,
                    '2MASS_J': 0.05, '2MASS_H': 0.05, '2MASS_KS': 0.1,
                    'F225W': 0.05, 'F390W': 0.05, 'F606W': 0.05,
                    'F105W': 0.05, 'F110W': 0.05, 'F125W': 0.05, 'F140W': 0.05, 'F160W': 0.05,
                    'IRAC_I1': 0.1, 'IRAC_I2': 0.1, 'IRAC_I3': 0.1, 'IRAC_I4': 0.1,
                    'WISE_W1': 0.1, 'WISE_W2': 0.1}


def build_obs(OBS, SPEC=None, SIGMA=None, VERBOSE=True, SHOW_ALL=True, **EXTRAS):
   
    obs                = {}

    filters            = [f for f in OBS.keys() if 'ERR' not in f and f not in ['REDSHIFT', 'ID', 'OBJECT', 'PANSTARRS_Y']]
    filternames        = [f.replace('PANSTARRS', 'sdss') + '0' for f in filters]

    obs["filters"]     = observate.load_filters(filternames)

    # Process magnitudes
    
    mags               = np.array([OBS[x][0]        for x in filters])
    mag_errs           = np.array([OBS[x+'_ERR'][0] for x in filters])
    
    # Convert mags to maggies
    
    obs["maggies"]     = 10**(-0.4*mags)
    obs["maggies_unc"] = 10**(-0.4*(mags-mag_errs)) - 10**(-0.4*mags)
    obs["maggies_unc"] += 10**(-0.4*mags) - 10**(-0.4*(mags+mag_errs))
    obs["maggies_unc"] /= 2.

    # Modify if upper limits are included
    
    mask_ul            = np.where( (mags > 0) & (mag_errs < 0))[0]
    if len(mask_ul) > 0:
        for idx in mask_ul:
            obs["maggies_unc"][idx] = 10**(-0.4*mags[idx])/3.
            obs["maggies"][idx]     = obs["maggies_unc"][idx]/1000.

    obs["phot_mask"]   = [True for f in obs["filters"]]
    obs["phot_wave"]   = [f.wave_effective for f in obs["filters"]]
    
    # Add mandatory filters
    # Filters are not considered in the fit unless data in those filtes are available

    for f in ['BESSELL_V']:
        if f not in filters:
            obs["filters"]     = obs["filters"]     + observate.load_filters([f])
            obs["maggies"]     = np.append(obs["maggies"], [1.e-9])
            obs["maggies_unc"] = np.append(obs["maggies_unc"], [0])
            obs["phot_mask"]   = obs["phot_mask"]   + [False]
            obs["phot_wave"]   = np.append(obs["phot_wave"], observate.load_filters([f])[0].wave_effective)

    if not (SPEC is None) and SIGMA is not None:
            
        # Spectrum

        # generate observables
        wave_obs = SPEC['wave_obs']
        spec     = SPEC['flambda']
        spec_err = SPEC['flambda_err']

        instrumental_sigma_v = SIGMA / wave_obs * 300000

        # convert flambda to to maggies
        # spectra are in erg s-1 cm-2 Angstrom-1

        c_angstrom = 2.998e18
        factor     = ((wave_obs)**2 / c_angstrom) * 1e23 / 3631.
        spec, spec_err = spec * factor, spec_err * factor

        # create spectral mask
        # approximate cut-off for MILES library outside of 3525-7500 A rest-frame
        # also mask Sodium D absorption
        # also mask Ca H & K absorption
        wave_rest = wave_obs / (1+OBS['REDSHIFT'])
        mask = ((spec_err > 0) &
                (spec != 0) &
                (wave_rest > 3525) &
                (wave_rest < 7500) &
                (np.abs(wave_rest-5892.9) > 25) &
                (np.abs(wave_rest-3935.0) > 10) &
                (np.abs(wave_rest-3969.0) > 20)
                )

        mask_telluric = (np.abs(wave_obs[mask] - 6873) > 15) & (np.abs(wave_obs[mask] - 7610) > 20)
        
        if SHOW_ALL:

            obs['wavelength'] = wave_obs
            obs['spectrum']   = spec
            obs['unc']        = spec_err
            obs['mask']       = np.ones(len(wave_obs), dtype=bool)
            obs["sigma_v"]    = instrumental_sigma_v
            
        else:
            
            obs['wavelength'] = wave_obs[mask]
            obs['spectrum']   = spec[mask]
            obs['unc']        = spec_err[mask]
            obs['mask']       = mask_telluric
            obs["sigma_v"]    = instrumental_sigma_v[mask]
        
    else:
        
        obs["wavelength"]  = None
        obs["spectrum"]    = None
        obs['unc']         = None
        obs['mask']        = None
        
    if VERBOSE:
        
        plt.figure()
        ax=plt.subplot(111)

        if SPEC is not None and SIGMA is not None:
            smask = obs['mask']
            ax.plot(obs['wavelength'][smask], obs['spectrum'][smask], '-', lw=2, color='red')
            ax.plot(obs['wavelength'][smask], obs['unc'][smask],      '-', lw=2, color='pink')

        ax.errorbar(obs['phot_wave'][obs['phot_mask']], obs['maggies'][obs['phot_mask']], obs['maggies_unc'][obs['phot_mask']], marker='o', color='black', ms=8, lw=0, elinewidth=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if not (SPEC is None) and SIGMA != None:
            mask = obs['spectrum'] > 0
            ymin = obs['spectrum'][mask].min()
            ymax = obs['spectrum'][mask].max()
            ax.set_ylim(ymin*0.5, ymax*2)
        else:
            ax.set_ylim(obs['maggies'][obs['phot_mask']].min()*0.5, obs['maggies'][obs['phot_mask']].max()*2)
            
        #plt.xlim(6000,10000)
        
    
    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs

def build_model(object_redshift=None, ldist=10.0, fixed_metallicity=None, add_neb=False,
                nbins_sfh=8, mixture_model=True, jitter_model=True, marginalize_neb=True, free_neb_met=True,
                continuum_order=12, **extras):
    
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
        
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """

    from prospect.models.templates import TemplateLibrary, describe
    from prospect.models import priors, sedmodel

    # --- input basic continuity SFH ---
    model_params = TemplateLibrary["parametric_sfh"]
    #model_params = TemplateLibrary["continuity_sfh"]

    # --- Mass ---
    model_params["mass"]["init"] = 1e10
    model_params["mass"]["disp_floor"] = 1e6
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e5, maxi=1e13)
    
    # Redshift and luminosity distance
    model_params["zred"]['isfree'] = False
    model_params["zred"]["init"] = object_redshift
    model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}

    # --- Use C3K everywhere ---
    model_params["use_wr_spectra"] = dict(N=1, isfree=False, init=0)
    model_params["logt_wmb_hot"]   = dict(N=1, isfree=False, init=np.log10(5e4))

    # --- SFH ---

    model_params["sfh"]['init'] = 4
    model_params["sfh"]['isfree'] = False

    model_params["tage"]["prior"] = priors.LogUniform(mini=0.001, maxi=13.8) # Gyr
    model_params["tage"]["init"] = 1
    model_params["tage"]["disp_floor"] = 0.1

    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2) # Gyr
    model_params["tau"]["init"] = 1
    model_params["tau"]["disp_floor"] = 0.1

    # --- IMF ---
    
    model_params["imf_type"]["init"] = 1 # Chabrier
    
    # --- metallicity (flat prior) ---
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)
    model_params["logzsol"]["init"] = -0.2
    
    # --- Reddening ---
    model_params["dust_type"]["init"] = 2 # Calzetti
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=8.0)
    model_params["dust2"]["init"] = 0.05  # dust absorption

    # --- spectral smoothing ---
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=300.0)

    # --- Nebular emission ---
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params['nebemlineinspec'] = dict(N=1, isfree=False, init=False)
        model_params['gas_logu']['isfree'] = True
        if free_neb_met:
            model_params['gas_logz']['isfree'] = True
            _ = model_params["gas_logz"].pop("depends_on")

        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            #model_params.update(TemplateLibrary['fit_eline_redshift'])
            model_params['eline_prior_width']['init'] = 10.0
            model_params['use_eline_prior']['init'] = True

            # only marginalize over a few (strong) emission lines
            if True:
                to_fit = ['Ba-gamma 4341',
            'Ba-beta 4861', '[O III] 4959', '[O III] 5007',
            '[N II] 6548', 'Ba-alpha 6563', '[N II] 6584', 
            '[S II] 6716', '[S II] 6731'
                         ]
                
                model_params['elines_to_fit']['init'] = to_fit

            # model_params['use_eline_prior']['init'] = False
        else:
            model_params['nebemlineinspec']['init'] = True

    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    if continuum_order > 0:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params['spec_norm']['isfree'] = False
        model_params["polyorder"]["init"] = continuum_order

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    if mixture_model:
        model_params['nsigma_outlier_spec'] = dict(N=1, isfree=False, init=50.)
        model_params['f_outlier_spec'] = dict(N=1, isfree=True, init=0.01,
                                              prior=priors.TopHat(mini=1e-5, maxi=0.1))
        model_params['nsigma_outlier_phot'] = dict(N=1, isfree=False, init=50.)
        model_params['f_outlier_phot'] = dict(N=1, isfree=False, init=0.0,
                                              prior=priors.TopHat(mini=0, maxi=0.5))

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a statistically acceptable fit.
    if jitter_model:
        model_params['spec_jitter'] = dict(N=1, isfree=True, init=1.0,
                                           prior=priors.TopHat(mini=1.0, maxi=3.0))
        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model

def build_sps(zcontinuous=1, compute_vega_mags=False,
              object_redshift=None, smooth_instrument=False, obs=None, **extras):
    
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    
    if (obs is not None) and (smooth_instrument):
        #from exspect.utils import get_lsf
        wave_obs = obs["wavelength"]
        sigma_v = obs["sigma_v"]
        speclib = sps.ssp.libraries[1].decode("utf-8")
        wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, object_redshift=object_redshift, **extras)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.set_lsf(wave, delta_v)

    return sps

def build_noise(jitter_model=False, **extras):
    if jitter_model:
        from prospect.likelihood import NoiseModel
        from prospect.likelihood.kernels import Uncorrelated
        jitter = Uncorrelated(parnames=['spec_jitter'])
        spec_noise = NoiseModel(kernels=[jitter], metric_name='unc', weight_by=['unc'])
    else:
        spec_noise = None

    return spec_noise, None

def calzetti_opacity(WAVE):
    """
    Computes the opacity k(lambda) of the Calzetti model
    Based on Eqs. 8a, 8b in Calzetti (2001)
    
    Input:  wavelength in Angstroem
    Output: opacity  
    """
    
    WAVE /= 10000.
    
    if 0.12 <= WAVE < 0.63:
        return 1.17 * ( -2.156 + 1.509/WAVE - 0.198/WAVE**2 + 0.011/WAVE**3 ) + 1.78
    elif 0.63 <= WAVE <= 2.2:
        return  1.17 * ( -1.857 + 1.040/WAVE ) + 1.78
    else:
        return np.nan
    
def calzetti_ebv_star(TAU):
    """
    Computes the E_star(B-V) of the Calzetti model.
    Baed on Eqs. 9 in Calzetti (2001)
    
    Input: optical depth tau
    Output: E_star(B-V)  
    """
    
    ebv_gas  = TAU / 0.921 / calzetti_opacity(5400)
    ebv_star = 0.44 * ebv_gas
    return ebv_star

def compute_fitquality(RESULT, OBS, MODEL, SPS, TYPE='emcee'):
    
    # Find run with the highest probability
    imax = np.argmax(RESULT['lnprobability'])

    # Get the posterior values for the highest probability
    
    if TYPE == "emcee":
        i, j = np.unravel_index(imax, RESULT['lnprobability'].shape)
        theta_max = RESULT['chain'][i, j, :].copy()

    else:
        theta_max = RESULT["chain"][imax, :]
    
    # Get the corresponding photometry and SED model
    _, mphot_map, _ = MODEL.mean_model(theta_max, OBS, sps=SPS)

    # Observations
    mphot_obs     = OBS['maggies']
    mphot_obs_err = OBS['maggies_unc']
    
    # Remove data that were not included in the fit
    
    mask          = OBS["phot_mask"]
    mphot_map     = mphot_map[mask]
    mphot_obs     = mphot_obs[mask]
    mphot_obs_err = mphot_obs_err[mask]
        
    return {'CHI2': np.round(np.sum(((mphot_map-mphot_obs)/mphot_obs_err)**2), 3), 'NOF': len(mphot_obs)}

def get_lsf(wave_obs, sigma_v, speclib="miles", object_redshift=0.0, **extras):
    """This method takes an instrimental resolution curve and returns the
    quadrature difference between the instrumental dispersion and the library
    dispersion, in km/s, as a function of restframe wavelength
    :param wave_obs: ndarray
        Observed frame wavelength (AA)
    :param sigma_v: ndarray
        Instrumental spectral resolution in terms of velocity dispersion (km/s)
    :param speclib: string
        The spectral library.  One of 'miles' or 'c3k_a', returned by
        `sps.ssp.libraries[1]`
    """
    lightspeed = 2.998e5  # km/s
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    wave_rest = wave_obs / (1 + object_redshift)

    # Get the library velocity resolution function at the corresponding
    # *rest-frame* wavelength
    if speclib == "miles":
        miles_fwhm_aa = 2.54
        sigma_v_lib = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
        # Restrict to regions where MILES is used
        good = (wave_rest > 3525.0) & (wave_rest < 7500)
    elif speclib == "c3k_a":
        R_c3k = 3000
        sigma_v_lib = lightspeed / (R_c3k * 2.355)
        # Restrict to regions where C3K is used
        good = (wave_rest > 2750.0) & (wave_rest < 9100.0)
    else:
        sigma_v_lib = sigma_v
        good = slice(None)
        raise ValueError("speclib of type {} not supported".format(speclib))

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_lib**2, 0, np.inf))

    # return the broadening of the rest-frame library spectra required to match
    # the observed frame instrumental lsf
    return wave_rest[good], dsv[good]

def luminosity_distance(REDSHIFT, MODEL='Planck15'):
   
    # Cosmology
    if MODEL == 'WMAP5':
        from astropy.cosmology import WMAP5 as cosmo 
    elif MODEL == 'WMAP7':
        from astropy.cosmology import WMAP7 as cosmo
    elif MODEL == 'WMAP9':
        from astropy.cosmology import WMAP9 as cosmo   
    elif MODEL == 'Planck13':
        from astropy.cosmology import Planck13 as cosmo
    elif MODEL == 'Planck15':
        from astropy.cosmology import Planck15 as cosmo
    elif MODEL == 'Flat70':
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
    else:
        msg   = 'Model {} not supported. Choose between WMAP5, WMAP7, WMAP9, Planck13, Planck15 and Flat70'.format(MDOEL)
        print(bcolors.BOLD + bcolors.FAIL + msg + bcolors.ENDC)
        sys.exit()
    
    return cosmo.luminosity_distance(REDSHIFT).value

def modify_chain(RESULT, RESULT_TYPE, OBS, MODEL, SPS, SFH='delayed'):
    
    # Get the z-index that correspond to tau, tage and mass
    idx_tau  = [i for i in range(len(RESULT['theta_labels'])) if RESULT['theta_labels'][i] == 'tau'][0]
    idx_tage = [i for i in range(len(RESULT['theta_labels'])) if RESULT['theta_labels'][i] == 'tage'][0]
    idx_mass = [i for i in range(len(RESULT['theta_labels'])) if RESULT['theta_labels'][i] == 'mass'][0]

    # Make copy of sampling matrix
    result = copy.deepcopy(RESULT)

    # Make copy model
    model_copy=copy.deepcopy(MODEL)
    model_copy.params['zred']=0
    model_copy.params['lumdist']=1e-5
    
    if RESULT_TYPE == 'EMCEE':
    
        # Get dimensions of the chains
        x, y, z           = np.shape(result['chain'])

        # Create a new array with columns for mass, SFR, sSFR, magobs, magabs
        temp_chain        = np.empty([x, y, z + 3 + 2*len(OBS['filternames'])]) 
        
        # Add previous data
        temp_chain[:,:,:z]=result['chain']
        
        # Populate array
    
        for idx_x in range(x):
            for idx_y in range(y):

                # Get tau, tage and mass
                tau                           = result['chain'][idx_x, idx_y, idx_tau]
                tage                          = result['chain'][idx_x, idx_y, idx_tage]
                mass_formed                   = result['chain'][idx_x, idx_y, idx_mass]

                # Add mass in living stars

                theta                         = result['chain'][idx_x, idx_y, :]
                _, mphot, mfrac               = MODEL.mean_model(theta, sps=SPS,obs=OBS)
                mass_living                   = mass_formed * mfrac
                temp_chain[idx_x, idx_y, z]   = mass_living

                temp_chain[idx_x, idx_y, z+1] = sfr(tau, tage, mass_formed, SFH)#sfr

                # Add sSFR
                ssfr                          = temp_chain[idx_x, idx_y, z+1] / mass_living#sfr
                temp_chain[idx_x, idx_y, z+2] = ssfr
                temp_chain[idx_x, idx_y, z+2] = np.nan if ssfr == np.inf else ssfr

                # Add model predicted apparent magnitudes

                idx_start                     = z + 3
                idx_stop                      = z + 3 + len(OBS['filternames'])

                temp_chain[idx_x, idx_y, idx_start:idx_stop] = -2.5*np.log10(mphot)

                # Add model predicted absolute magnitudes

                _, mphot_abs, _               = model_copy.mean_model(theta, sps=SPS,obs=OBS)

                idx_start                     = z + 3 +   len(OBS['filternames'])
                idx_stop                      = z + 3 + 2*len(OBS['filternames'])

                temp_chain[idx_x, idx_y, idx_start:] = -2.5*np.log10(mphot_abs)

    else:
        x, z             = np.shape(result['chain'])
        
        # Create a new array with columns for mass, SFR, sSFR, magobs, magabs
        temp_chain       = np.empty([x, z + 3 + 2*len(OBS['filternames'])]) 
        
        # Add previous data
        temp_chain[:,:z] = result['chain']

        # Populate array
    
        for idx_x in range(x):

                # Get tau, tage and mass
                tau                         = result['chain'][idx_x, idx_tau]
                tage                        = result['chain'][idx_x, idx_tage]
                mass_formed                 = result['chain'][idx_x, idx_mass]

                # Add mass in living stars

                theta                       = result['chain'][idx_x, :]
                _, mphot, mfrac             = MODEL.mean_model(theta, sps=SPS,obs=OBS)
                mass_living                 = mass_formed * mfrac
                temp_chain[idx_x, z]        = mass_living

                # Add SFR
                temp_chain[idx_x, z+1]      = sfr(tau, tage, mass_formed, SFH)

                # Add sSFR
                ssfr                        = temp_chain[idx_x, z+1] / mass_living
                temp_chain[idx_x, z+2]      = ssfr
                temp_chain[idx_x, z+2]      = np.nan if ssfr == np.inf else ssfr

                # Add model predicted apparent magnitudes

                idx_start                   = z + 3
                idx_stop                    = z + 3 + len(OBS['filternames'])

                temp_chain[idx_x, idx_start:idx_stop] = -2.5*np.log10(mphot)

                # Add model predicted absolute magnitudes

                _, mphot_abs, _             = model_copy.mean_model(theta, sps=SPS,obs=OBS)

                idx_start                   = z + 3 +   len(OBS['filternames'])
                idx_stop                    = z + 3 + 2*len(OBS['filternames'])

                temp_chain[idx_x, idx_start:]= -2.5*np.log10(mphot_abs)

    result['chain']        = temp_chain

    result['theta_labels'] = [u'mass_formed' if x == u'mass' else x for x in result['theta_labels']]
    result['theta_labels'] = result['theta_labels'] + [u'mass', u'sfr', u'ssfr'] \
                           + ['MAGOBS_' + f.replace('_resampled', '') for f in OBS['filternames']] \
                           + ['MAGABS_' + f.replace('_resampled', '') for f in OBS['filternames']] \

    return result

def process_catalogue(FILE, OBJECT, ADD_SYS=False, ADD_EXTCORR=False):
    

    # Load catalogue
    data = ascii.read(FILE)
    data = data[data['OBJECT'] == OBJECT]

    try:
        data.remove_columns(['CONTEXT', 'MISC'])
    except:
        pass
    
    if 'EXTCORR' in data.keys():
        data.remove_column('EXTCORR')

    if ADD_EXTCORR:
        coords  = SkyCoord(data['RA'], data['DEC'], unit=(u.hour, u.deg))
        ebv_mw  = gal_reddening.ebv(coords)
    
    for filter in [x for x in data.keys() if x not in ['ID', 'REDSHIFT', 'CONTEXT', 'OBJECT', 'MISC', 'RA', 'DEC', 'EXTCORR'] and '_ERR' not in x]:

        # Add systematic error in quadrature
        if ADD_SYS and data[filter + '_ERR'] != -99.:
            data[filter + '_ERR'] = np.sqrt(data[filter + '_ERR']**2 + filter_systematics[filter]**2)
            data[filter + '_ERR'].format='.3f'
        
        # Extinction correction
        #if ADD_EXTCORR:
        #    filter_wave  = observate.Filter(filter + '_resampled').wave_pivot * u.Angstrom
        #    data[filter][0] -= extinction.extinction_ccm89(filter_wave, ebv_mw*3.1, 3.1) if 909.09 < filter_wave.value < 33333.33 else 0

        # Remove unneccary data
        if data[filter] < 0.:
            data.remove_columns([filter, filter+'_ERR'])
    
    # Remove unneccesary data

    for key in ['RA', 'DEC']:
        if key in data.keys():
            data.remove_column(key)
    
    return data

def sfr(TAU, TAGE, MASS, SFH):
    if SFH         == 'delayed':
        temp_sfr   = MASS * (TAGE/TAU**2) * np.exp(-TAGE/TAU) / (gamma(2) * gammainc(2, TAGE/TAU)) * 1e-9

    elif SFH       == 'exponential':
        sfr_eq     = lambda t, tau: np.exp(-t/tau)
        times      = np.linspace(0, TAGE, 1000)
        A          = np.trapz(sfr(times, TAU), times)
        temp_sfr   = MASS * sfr_eq(TAGE, TAU) / A * 1e-9

    return np.nan if temp_sfr == np.inf else temp_sfr     

def traceplot(results, showpars=None, start=0, chains=slice(None),
              figsize=None, truths=None, **plot_kwargs):
    """Plot the evolution of each parameter value with iteration #, for each
    walker in the chain.

    :param results:
        A Prospector results dictionary, usually the output of
        ``results_from('resultfile')``.

    :param showpars: (optional)
        A list of strings of the parameters to show.  Defaults to all
        parameters in the ``"theta_labels"`` key of the ``sample_results``
        dictionary.

    :param chains:
        If results are from an ensemble sampler, setting `chain` to an integer
        array of walker indices will cause only those walkers to be used in
        generating the plot.  Useful for to keep the plot from getting too cluttered.

    :param start: (optional, default: 0)
        Integer giving the iteration number from which to start plotting.

    :param **plot_kwargs:
        Extra keywords are passed to the
        ``matplotlib.axes._subplots.AxesSubplot.plot()`` method.

    :returns tracefig:
        A multipaneled Figure object that shows the evolution of walker
        positions in the parameters given by ``showpars``, as well as
        ln(posterior probability)
    """
    import matplotlib.pyplot as pl


    # Get parameter names
    try:
        parnames = np.array(results['theta_labels'])
    except(KeyError):
        parnames = np.array(results['model'].theta_labels())
    # Restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, lnp, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim ==2:
        trace = trace[None, :]
    trace = trace[chains, start:, :]
    lnp = np.atleast_2d(results['lnprobability'])[chains, start:]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start:]
    nwalk = trace.shape[0]

    # Set up plot windows
    ndim = len(parnames) + 1
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim * 1.0 / nx))
    sz = np.array([nx, ny])
    factor = 3.0           # size of one side of one panel
    lbdim = 0.2 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05 * factor         # w/hspace size
    plotdim = factor * sz + factor * (sz - 1) * whspace
    dim = lbdim + plotdim + trdim

    if figsize is None:
        fig, axes = pl.subplots(nx, ny, figsize=(dim[1], dim[0]), sharex=True)
    else:
        fig, axes = pl.subplots(nx, ny, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    #lb = lbdim / dim
    #tr = (lbdim + plotdim) / dim
    #fig.subplots_adjust(left=lb[1], bottom=lb[0], right=tr[1], top=tr[0],
    #                    wspace=whspace, hspace=whspace)

    # Sequentially plot the chains in each parameter
    for i in range(ndim - 1):
        ax = axes.flat[i]
        for j in range(nwalk):
            ax.plot(trace[j, :, i], **plot_kwargs)
        ax.set_title(parnames[i], y=1.02)
        if parnames[i] == 'mass':
            ax.set_yscale('log')
    # Plot lnprob

    ax = axes.flat[-1]
    for j in range(nwalk):
        ax.plot(lnp[j, :], **plot_kwargs)
    ax.set_title('lnP', y=1.02)
    ax.set_yscale('log')

    [ax.set_xlabel("iteration") for ax in axes[-1,:]]
    #[ax.set_xticklabels('') for ax in axes[:-1, :].flat]

    if truths is not None:
        for i, t in enumerate(truths[ind_show]):
            axes.flat[i].axhline(t, color='k', linestyle=':')

    pl.tight_layout()
    
    return fig

def write_posterior_dynesty(DATA, FILE):
    temp   = np.hstack([DATA['chain'], np.reshape(DATA['weights'], (-1, 1))])
    output = table.Table(temp, names=DATA['theta_labels'] + ['WEIGHTS'])
    output.write(FILE, format='fits', overwrite=True)

def subcorner(RESULTS, OBJECT, SHOWPARS=None, TRUTHS=None,
              START=0, THIN=1, CHAINS=slice(None), MAKEPLOT=True, SHOWPLOT=True, SAVEPLOT=True,
              LOGIFY=["mass_formed", "mass", "tau", "sfr", "ssfr"], OUTDIR='./', **KWARGS):

    # pull out the parameter names and flatten the thinned chains
    # Get parameter names
    try:
        parnames = np.array(RESULTS['theta_labels'], dtype='U20')
    except(KeyError):
        parnames = np.array(RESULTS['model'].theta_labels())
    
    # Restrict to desired parameters

    if SHOWPARS is not None:
        ind_show = np.array([parnames.tolist().index(p) for p in SHOWPARS])
        parnames = parnames[ind_show]
        parnames2 = copy.deepcopy(parnames)
    else:
        ind_show = slice(None)
      
    # Get the arrays we need (trace, wghts)
    trace = RESULTS['chain'][..., ind_show]
    
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[CHAINS, START::THIN, :]
    wghts = RESULTS.get('weights', None)
    if wghts is not None:
        wghts = wghts#[START::THIN]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
        
   
    # logify some parameters
    xx = samples.copy()
    if TRUTHS is not None:
        xx_truth = np.array(TRUTHS).copy()
    else:
        xx_truth = None
    for p in LOGIFY:
        if p in parnames:
            idx = parnames.tolist().index(p)
            xx[:, idx] = np.log10(xx[:,idx])
            parnames2[idx] = "log({})".format(parnames2[idx])
            if TRUTHS is not None:
                xx_truth[idx] = np.log10(xx_truth[idx])

    # Write percentiles to table
    
    output = table.Table()
    output['ID'] = [1000]
                
    for i in range(len(parnames)):
        
        q_16, q_50, q_84 = corner.quantile(xx[:,i], [0.16, 0.5, 0.84], weights=wghts)
        print(parnames[i], q_16, q_50, q_84)
        
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_INF'] = q_16
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_MED'] = q_50
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_SUP'] = q_84
        
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_INF'].format = '.3f'
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_MED'].format = '.3f'
        output[parnames2[i].replace('log', '').replace('(','').replace(')','').upper() + '_SUP'].format = '.3f'
       
    # Make corner plot
    
    if MAKEPLOT:
    
        # Rename labels
        parnames = [p.encode('utf-8') for p in parnames]
        for i in range(len(parnames)):
            if parnames[i]   == 'logzsol':
                parnames[i]  = '$\\log\\,Z/Z_\\odot$'
            elif parnames[i] == 'dust2':
                parnames[i]  = '$\\tau_{\\rm dust}$'
            elif parnames[i] == 'tau':
                parnames[i]  = '$\\log\\,\\tau/{\\rm yr}$'
            elif parnames[i] == 'tage':
                parnames[i]  = '$t_{\\rm age}/{\\rm Gyr}$'
            elif parnames[i] == 'mass':
                parnames[i]  = '$\\log\\,M/M_\\odot$'
            elif parnames[i] == 'sfr':
                parnames[i]  = '$\\log\\,{\\rm SFR}/{M_\\odot\\,{\\rm yr}^{-1}}$'
            elif parnames[i] == 'ssfr':
                parnames[i]  = '$\\log\\,{\\rm sSFR}/{\\rm yr}^{-1}$'
            elif '$' not in str(parnames[i]):
                parnames[i]  = str(parnames[i]).replace('_', '\\_')
#
#
        # mess with corner defaults
        corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                         "fill_contours": True, "show_titles": True, 'verbose': False}
        corner_kwargs.update(KWARGS)
#
        ranges        = []
        for i in range(np.shape(xx)[1]):
            if parnames2[i] in ['log(mass)', 'log(sfr)', 'log(ssfr)']:
                xmin, xmax = corner.quantile(xx[:,i], [0.16, 0.84])
                ranges.append([xmin-1, xmax+1])
            else:
                ranges.append([xx[:,i].min(), xx[:,i].max()])
               
        fig = corner.corner(xx, labels=parnames, truths=xx_truth, range=ranges,
                              quantiles=[0.16, 0.5, 0.84], weights=wghts,
                              title_kwargs={"fontsize": legend_size+4},
                              color='#455EB2', hist2d_kwargs={'contourf_kwargs':{'lw': 8}, 'lw': 8},
                              hist_kwargs={'lw':4},
                              **corner_kwargs)

        fig.set_size_inches(27., 27.)
                
        FIGNAME = OUTDIR + OBJECT+'_corner.pdf'
        if SAVEPLOT:
            plt.savefig(FIGNAME)
    
        if not SHOWPLOT:
            plt.close()
    
    return output
    
def sed_plot(RESULTS, OBS, OBS2, MODEL, SPS, FITRESULTS, SHOWPLOT=False, TYPE='emcee', OUTDIR='./'):
    
    # Data
    
    wobs        = np.array(OBS['phot_wave'])
    mag_obs     = -2.5*np.log10(OBS['maggies'])
    mag_err     = mag_obs + 2.5*np.log10(OBS['maggies']+OBS['maggies_unc'])
    mag_err     += -2.5*np.log10(OBS['maggies'] - OBS['maggies_unc']) - mag_obs 
    mag_err     /= 2.
    mag_mask    = [i for i in range(len(OBS['phot_mask'])) if OBS['phot_mask'][i] == True]
    mag_mask_det= np.where(OBS['maggies_unc'] / OBS['maggies'] < 1e3)[0]
    mag_mask_ul = np.where(OBS['maggies_unc'] / OBS['maggies'] == 1000.)[0]

    # Model
    
    # Get best-fit
    imax = np.argmax(RESULTS['lnprobability'])
    if TYPE == "emcee":
        i, j = np.unravel_index(imax, RESULTS['lnprobability'].shape)
        theta_max = RESULTS['chain'][i, j, :].copy()
    else:
        theta_max = RESULTS["chain"][imax, :]

    # generate models
    redshift                = MODEL.params.get('zred', 0.0)
    mspec_map, mphot_map, _ = MODEL.mean_model(theta_max, OBS2, sps=SPS)
    #wspec                   = SPS.wavelengths * (1+redshift)
    
    # SED plot

    plt.figure(figsize=(9*np.sqrt(2),9))

    ax=plt.subplot(111)

    #try:
    if True:        
            smask = OBS['mask']
            ax.axvspan(OBS['wavelength'][smask][0], OBS['wavelength'][smask][-1], color='0.9', lw=0)
            ax.plot(OBS2['wavelength'], -2.5*np.log10(OBS2['spectrum']), '-', lw=3, color='black', alpha=0.2)
            #ax.plot(OBS['wavelength'][smask], -2.5*np.log10(OBS['unc'][smask]),      '-', lw=2, color='pink')
    #except:
    #    pass
    
    # Observed photometry

    # Detections
    ax.errorbar(wobs[np.intersect1d(mag_mask_det, mag_mask)], mag_obs[np.intersect1d(mag_mask_det, mag_mask)], yerr=mag_err[np.intersect1d(mag_mask_det, mag_mask)], 
         lw=0, marker='o', ms=14, elinewidth=2, capsize=0, color='black', zorder=2, mew=2, mec='white')
       
    ax.errorbar(wobs[np.intersect1d(mag_mask_ul, mag_mask)], -2.5*np.log10(3 * OBS['maggies_unc'][np.intersect1d(mag_mask_ul, mag_mask)]),
         lw=0, marker='v', ms=9, elinewidth=2, capsize=0, color='k', zorder=2)

    # Model mags

    mspec_map, mphot_map, _ = MODEL.mean_model(theta_max, OBS2, sps=SPS)
    mask_left  = OBS2['wavelength'] < OBS['wavelength'][0]
    mask_right = OBS2['wavelength'] > OBS['wavelength'][-1]
    ax.plot(OBS2['wavelength'][mask_left], -2.5*np.log10(mspec_map[mask_left]), lw=1, color='tab:blue')
    ax.plot(OBS2['wavelength'][mask_right], -2.5*np.log10(mspec_map[mask_right]), lw=1, color='tab:blue')
    
    #obs3 = copy.deepcopy(obs)
    
    #obs3['wavelength'] = np.logspace(np.log10(3000.), np.log10(11000), 2000)
    #obs3['mask'] = np.ones_like(obs3['wavelength'], dtype=bool)
    #obs3['unc'] = np.ones_like(obs3['wavelength'])
    #obs3['spectrum'] = np.ones_like(obs3['wavelength'])

    #sed2, phot2, _ = MODEL.predict(theta_max, OBS3, sps=SPS)
    #model_sed = MODEL._sed
    #model_sed[do_not_overplot_mask] = np.nan
    
    #print(model_sed)
    #print(sed2)
    
    #ax.step(obs3['wavelength'], -2.5*np.log10(sed2), color='darkgoldenrod',
    #             ls='--', lw=1.5, alpha=0.5, where='mid',
    #             label='$\mathrm{Delayed}$-$\\tau \; \mathrm{model\;SED}$')
    #
    #print(obs3['wavelength']/1e4, -2.5*np.log10(model_sed))
    
    mspec_map, mphot_map, _ = MODEL.mean_model(theta_max, OBS, sps=SPS)
    wspec                   = SPS.wavelengths * (1+redshift)
    ax.plot(OBS['wavelength'], -2.5*np.log10(mspec_map), lw=3, color='tab:blue')
    #ax.plot(wspec, -2.5*np.log10(mspec_map), lw=2)
    
    """
    for ii, f in enumerate(['PANSTARRS_G', 'PANSTARRS_R', 'PANSTARRS_I', 'PANSTARRS_Z', 'PANSTARRS_Y']):
        ax.errorbar(wobs[ii], FITRESULTS['MAGMOD_' + f + '_MED'][0], 
                    [[FITRESULTS['MAGMOD_' + f + '_MED'][0] - FITRESULTS['MAGMOD_' + f + '_INF'][0]], [FITRESULTS['MAGMOD_' + f + '_SUP'][0] - FITRESULTS['MAGMOD_' + f + '_MED'][0]]],
                    marker='o', lw=0, ms=9, color='crimson', elinewidth=2
                     )
        print(wobs[ii], FITRESULTS['MAGMOD_' + f + '_MED'][0])
    """

    xmin, xmax = 3250, 10250 
    #ymin, ymax = max(mag_obs[np.intersect1d(mag_mask_det, mag_mask)]) + 1, min(min(mag_obs[mag_mask_det]), min(mspec_map[mask_spec])) - 0.5
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(21.75, 17.25)
    ax.set_xlabel(r'Observed wavelength (Å)')
    ax.set_ylabel('Brightness (mag)')
    
    minorLocator		= plt.MultipleLocator(250)
    ax.xaxis.set_minor_locator(minorLocator)

    majorLocator		= plt.MultipleLocator(1)
    ax.yaxis.set_major_locator(majorLocator)

    minorLocator		= plt.MultipleLocator(0.5)
    ax.yaxis.set_minor_locator(minorLocator)

    ax2 = ax.twinx()
    
    labels = [r'$g_{\rm PS1}$', r'$r_{\rm PS1}$', r'$i_{\rm PS1}$', r'$z_{\rm PS1}$', r'$y_{\rm PS1}$']
    
    for f, label in zip(OBS['filters'][:5], labels):
        print(f)
        ax2.plot(f.wavelength, f.transmission, lw=3, color='k')
        plt.text(f.wave_effective, 0.375,
                 label,
                 ha='center', va='center', transform=ax2.transData, fontsize=legend_size,
                 path_effects=[PathEffects.withStroke(linewidth=5, foreground="white")])
    
    ax2.set_ylim(0, 5)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_yticks([])

    ax.yaxis.set_ticks_position('both')

    kwargs = {	'ha': 'left',
                'va': 'top',
                'transform': ax.transAxes,
                'fontsize': label_size-2,
                'path_effects'  : [PathEffects.withStroke(linewidth=5, foreground="white")]
                }

    #ax.text(left + 0.05, top - 0.03,
    #        '\\textbf{{ {object} }}'.format(object=FITRESULTS['OBJECT'][0].replace('_', '\_')),
    #        **kwargs)

    
    kwargs = {	'ha': 'left',
                'va': 'top',
                'transform': ax.transAxes,
                'fontsize': legend_size,
                'path_effects'  : [PathEffects.withStroke(linewidth=5, foreground="white")]
                }

    ax.text(left + 0.025, top - 0.03,
            '$\\log\\,M/M_\\odot= {med:.2f}^{{ +{sup:.2f} }}_{{ -{inf:.2f} }}$'.format(
            med=FITRESULTS['MASS_MED'][0],
            sup=FITRESULTS['MASS_SUP'][0] - FITRESULTS['MASS_MED'][0],
            inf=FITRESULTS['MASS_MED'][0] - FITRESULTS['MASS_INF'][0],
            ),
            **kwargs)

    ax.text(left + 0.025, top - 0.10,
            '${{\\rm SFR}} / M_\\odot\\,{{\\rm yr}}^{{-1}} = {med:.2f}^{{ +{sup:.2f} }}_{{ -{inf:.2f} }}$'.format(
            med=10**FITRESULTS['SFR_MED'][0],
            sup=10**FITRESULTS['SFR_SUP'][0] - 10**FITRESULTS['SFR_MED'][0],
            inf=10**FITRESULTS['SFR_MED'][0] - 10**FITRESULTS['SFR_INF'][0],
            ),
            **kwargs)

    ax.text(left + 0.025, top - 0.17,
            '${{\\rm Age}} / {{\\rm Gyr}} = {med:.01f}^{{ +{sup:.01f} }}_{{ -{inf:.01f} }}$'.format(
            med= FITRESULTS['TAGE_MED'][0],
            sup=(FITRESULTS['TAGE_SUP'][0] - FITRESULTS['TAGE_MED'][0]),
            inf=(FITRESULTS['TAGE_MED'][0] - FITRESULTS['TAGE_INF'][0]),
            ),
            **kwargs)

    ax.text(left + 0.025, top - 0.24,
            '$E(B-V)_{{\\rm star}} = {med:.2f}^{{ +{sup:.2f} }}_{{ -{inf:.2f} }}$ mag'.format(
            med=FITRESULTS['EBVSTAR_MED'][0],
            sup=FITRESULTS['EBVSTAR_SUP'][0] - FITRESULTS['EBVSTAR_MED'][0],
            inf=FITRESULTS['EBVSTAR_MED'][0] - FITRESULTS['EBVSTAR_INF'][0]
            ),
            **kwargs)

    """ax.text(left + 0.05, top - 0.38,
            '$\\chi^2/{{\\rm n.o.f.}} = {chi:.2f}/{nof}$'.format(
            chi=FITRESULTS['CHI2_BEST'][0], 
            nof=FITRESULTS['NOF_BEST'][0]
            ),
            **kwargs)
    """

    FIGNAME = OUTDIR + FITRESULTS['OBJECT'][0] + '_sed.pdf'
    #plt.savefig(FIGNAME, dpi=600, papertype='a4', orientation='landscape')
    plt.savefig(FIGNAME)