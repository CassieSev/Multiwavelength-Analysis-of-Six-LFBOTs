import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from astropy.cosmology import Planck18
#plt.style.use('dark_background')
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import astropy.cosmology.units as cu
from vals import redshifts
import os



def calc_beta(gamma_beta):
    return gamma_beta/np.sqrt(gamma_beta**2+1)


def get_shock(nu_ghz, f_jansky, t_days, redshift, object=None, limit=False):
    """Assumes alpha = 1, epsilon_e = epsilon_B = 1/3
    f_jansky is peak OBSERVED flux in Jy
    nu_ghz is peak OBSERVED frequency on GHz
    t_days is OBSERVED days after t0, estimated from transient discovery
    limit is a boolean that says whether the input values are at an observed peak in the SED (false),
    or if we only have limits on the peak due to incomplete SED (false)

    Returns a dictionary with the following entries:
        radius: Shock radius in cm
        magnetic: Magnetic field in G
        velocity: Returns beta = v/c; assumes R/t = gamma*beta*c
        energy: Returns total energy u_B/epsilon_B, in erg
        density: returns medium density in cm^-3
        luminosity: Luminosity at peak frequency, L_nu=4pi*r^2*F_nu, in erg/s/Hz
        nu_rest: Peak frequency in rest frame of object, in GHz
        t_days: Time after t0 in days
        object: object name
        limit: Do we only have limits (True) or are values actually at peak (false)
    """
    z=redshift*cu.redshift
    d_ang=Planck18.angular_diameter_distance(z=redshift).to(u.Mpc).value
    d_lum=Planck18.luminosity_distance(z=redshift).to(u.Mpc).value
    f_corr = f_jansky/(1+z.value)
    nu_rest = nu_ghz * (1+z.value)
    t_rest = t_days/(1+redshift) # Convert to rest frame time
    f=0.5

    r = 8.8*10**15 * (f/0.5)**(-1/19) * f_corr**(9/19) * d_ang**(18/19) * (nu_rest/5)**(-1) # cm
    b = 0.58 * (f/0.5)**(-4/19) * f_corr**(-2/19) * d_ang**(-4/19) * (nu_rest/5) # G

    L = 1.2*10**27 * f_corr * d_lum**2 # erg /s/Hz

    
    beta = calc_beta(r*(1+z.value)/(t_days*3600*24*3*10**10)) # Divide r by t_peak get gamma*beta*c, calc beta from gamma*beta

    U = 1.9*10**46 * 3 * (f/0.5)**(8/19) * f_corr**(23/19) * d_ang**(46/19) * (nu_rest/5)**(-1) # erg
    n_e = 1.02418 * 3 * (f/0.5)**(-6/19) * (f_corr)**(-22/19) * d_ang**(-44/19) * (nu_rest/5)**4 * t_rest**2 # cm^-3
    m_dot = 2.5*10**(-5) * 3 * (f/0.5)**(-1/19) * (f_corr)**(-4/19) * d_ang**(-8/19) * (nu_rest/5)**2 * t_rest**2 #1e-4 M_sun / 1000 km/s
    mass_swept = (n_e*np.power(u.cm, -3)*const.m_p*np.power(r*u.cm, 3)).to(u.Msun)
    # Check v_a/v_c
    assert 6.2*10**(-3)*(f_corr)**(6/19)*d_ang**(12/19)*(nu_rest/5)**(-3)*(t_rest/100)**2<0.1
    assert 5.9*10**(-4)*(f_corr)**(6/13)*d_ang**(12/13)*(nu_rest/5)**(-63/13)*(t_rest/100)**(-38/13)<0.1

    return {'radius':r, 'magnetic':b, 'velocity':beta, 'energy':U, 'density':n_e, 'luminosity':L, 'nu_rest': nu_rest, 't':t_rest, 'object': object,
         'limit':limit, 'wind':m_dot, 'mass_swept': mass_swept}



def get_shock2(nu_ghz, f_jansky, t_days, redshift, object=None, limit=False):
    """Assumes alpha = 1, epsilon_e = epsilon_B = 1/3
    f_jansky is peak flux in Jy
    nu_ghz is peak OBSERVED frequency on GHz
    t_days is days after t0, estimated from transient discovery
    limit is a boolean that says whether the input values are at an observed peak in the SED (false),
    or if we only have limits on the peak due to incomplete SED (false)

    Returns a dictionary with the following entries:
        radius: Shock radius in cm
        magnetic: Magnetic field in G
        velocity: Returns beta = v/c; assumes R/t = gamma*beta*c
        energy: Returns total energy u_B/epsilon_B, in erg
        density: returns medium density in cm^-3
        luminosity: Luminosity at peak frequency, L_nu=4pi*r^2*F_nu, in erg/s/Hz
        nu_rest: Peak frequency in rest frame of object, in GHz
        t_days: Time after t0 in days
        object: object name
        limit: Do we only have limits (True) or are values actually at peak (false)
    """
    F_p=f_jansky/(1+redshift)
    nu_p = (1+redshift)*nu_ghz*1e9 #Convert to Hz
    t_p = t_days/(1+redshift)
    D_mpc=Planck18.angular_diameter_distance(z=redshift).to(u.Mpc).value
    D_ang_cm=Planck18.angular_diameter_distance(z=redshift).to(u.cm).value
    D_cm=Planck18.luminosity_distance(z=redshift).to(u.cm).value
    def infer_Rp(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Chevalier 1998 (Eq. 13)
        return (8.8e15 * (eps_e/eps_B)**(-1/19) * (f/0.5)**(-1/19) 
                * F_p**(9/19) * D_mpc**(18/19) * (nu_p / 5e9)**(-1))  # cm

    def infer_Bp(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Chevalier 1998 (Eq. 14)
        return (0.58 * (eps_e/eps_B)**(-4/19) * (f/0.5)**(-4/19) 
                * F_p**(-2/19) * D_mpc**(-4/19) * (nu_p / 5e9))  # Gauss

    def infer_Up(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Ho 2019 (Eq. 12)
        return ((1.9e46) * (1/eps_B) * (eps_e/eps_B)**(-11/19)
                * (f/0.5)**(8/19) * (F_p)**(23/19)
                * (D_mpc)**(46/19) * (nu_p/5e9)**(-1))

    def infer_beta(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Ho 2019 (Eq. 13)
        L_p = 4 * np.pi * (F_p/1e23) * D_cm**2
        return ((eps_e/eps_B)**(-1/19) * (f/0.5)**(-1/19)
                * (4 * np.pi * (F_p/1e23) * D_ang_cm**2/ 1e26)**(9/19) * (nu_p/5e9)**(-1) * (t_p)**(-1))

    def infer_n_e(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Ho 2019 (Eq. 16)
        L_p = 4 * np.pi * (F_p/1e23) * D_cm**2
        return ((20) * (1/eps_B) * (eps_e/eps_B)**(-6/19) * (f/0.5)**(-6/19)
                * (4 * np.pi * (F_p/1e23) * D_ang_cm**2/ 1e26)**(-22/19) * (nu_p/5e9)**4 * (t_p)**2)

    def infer_energy_per_rad(F_p, eps_e=1/3, eps_B=1/3, f=0.5): # Ho 2019 (Eq. 22)
        L_p = 4 * np.pi * (F_p/1e23) * D_cm**2
        return ((3e29) * (1/eps_B) * (eps_e/eps_B)**(-10/19) * (f/0.5)**(9/19)
                    * (4 * np.pi * (F_p/1e23) * D_ang_cm**2/ 1e26)**(14/19))

    def infer_Mdot_over_vw(F_p, nu_p, eps_e=1/3, eps_B=1/3, f=0.5): # Ho 2019 (Eq. 23)
        L_p = 4 * np.pi * (F_p/1e23) * D_cm**2
        mdot_over_vw = ((5e-5) * (1/eps_B) * (eps_e/eps_B)**(-8/19)
                        * (f/0.5)**(-1/19) * (4 * np.pi * (F_p/1e23) * D_ang_cm**2/ 1e26)**(-4/19)
                        * (nu_p/5e9)**2 * (t_p)**2)
        print(mdot_over_vw)
        return mdot_over_vw * (1e-4 * u.Msun/u.yr) / (1000 * u.km/u.s)
    
    return {'radius':infer_Rp(F_p, nu_p), 'magnetic':infer_Bp(F_p, nu_p), 'velocity':infer_beta(F_p, nu_p),
            'energy':infer_Up(F_p, nu_p), 'density':infer_n_e(F_p, nu_p), 'luminosity':4 * np.pi * (F_p/1e23) * D_cm**2, 'nu_rest': nu_p, 't':t_p, 'object': object,
         'limit':limit, 'wind':infer_Mdot_over_vw(F_p, nu_p)}
    

#print(get_shock(0.000194*1.24,3.6/1.24,90,redshifts['AT2023fhn'], object='AT2023fhn'))
#print(get_shock2(0.000194*1.24,3.6/1.24,90,redshifts['AT2023fhn'], object='AT2023fhn'))

def bpl_smooth_a1(x, xb, Fb, a2, a1=5/2):
    s = 3; sig = np.sign(a1-a2)
    return Fb*(0.5*(x/xb)**(-sig*s*a1) + 0.5*(x/xb)**(-sig*s*a2))**(-sig/s)

def sharp_peak_from_two_anchors(a2,  xb, Fb, x1=0.1, x2=500, a1=5/2):
        F1 = bpl_smooth_a1(x1, xb, Fb, a2, a1=a1); F2 = bpl_smooth_a1(x2, xb, Fb, a2, a1=a1)
        C1 = F1 / (x1**a1); C2 = F2 / (x2**a2)
        x_p = (C1 / C2)**(1.0 / (a2 - a1))
        F_p = C1 * (x_p**a1)
        return x_p, F_p



def fix_freq(frequencies, object, frame):
    """
    Take a frequency entry, and account for redshift to get into the transient's frame
    `frequencies`: Array of frequencies to correct
    `object`: Used to retrieve redshift
    `frame`: Array, if the value is 'rest' do not divide by (1+z)
    """
    new_frequencies=np.array([])
    for i,freq in enumerate(frequencies):
        # Average a range of frequencyes into a single value
        if '-' in freq:
            new_freq = np.average(np.array(freq.split('-'),dtype=float))
        else:
            new_freq=float(freq)
        if frame[i] != 'rest':
            z = redshifts[object[i]]
            new_freq = new_freq *(1+z)
        new_frequencies=np.append(new_frequencies, new_freq)
    return new_frequencies



def fit_sed(nu_vals, F_obs, F_err):
    """
    Fits an array of frequencies and fluxes to a smoothed broken power law
    Parameters:
        `nu_vals` is frequency array
        `F_obs` is flux array
        `F_err` is flux error array

    Returns:
    A tuple, where the first entry is the list of fitted params from curve_fit, and the second is the list
    of errors


    """
    a2=-3/2; a1=5/2
    idx_max = np.argmax(F_obs)
    p0 = [nu_vals[idx_max], F_obs[idx_max], a2]
    params, pcov = curve_fit(bpl_smooth_a1, nu_vals, F_obs, p0=p0, sigma=F_err, absolute_sigma=True)
    nu_p_fit, F_p_fit, a2_fit = params
    param_err = np.sqrt(np.diag(pcov))
    a1_fit = a1; a1_err = 0
    return (params, param_err)


def find_peak():
    pass
def get_slice(object, time):
    df = raw_df.loc[(raw_df['object']==object)&(raw_df['t0']>=time[0])&(raw_df['t0']<=time[1])&(raw_df['unc'].astype(float)>=0)].sort_values(by=['freq_corr'])
    print(df)
    return df


def get_and_fit(object, time_range, mean_time, plot=False):
    """
    Retrieves and fits a smoothed power law to a single epoch of observations to get the peak frequency and flux
    `object`: Object Name
    `time`: A tuple containing the minimum and maximum time bins (in observer frame) that defines the epoch

    Returns `sync`, a dictionary containing the derived synchrotron parameters and errors
    Optionally plots the SED of the epoch alongside the fit.
    """
    df = get_slice(object, time_range)
    min_fluxErr = np.zeros(len(df['unc']))

    # Add an error floor if data from multiple arrays is being used
    if len(np.unique(df['array'])) > 1:
        min_fluxErr = 0.1 * np.abs(df['uJy'])
    F_err = np.maximum(df['unc'], min_fluxErr)

    # Get the peak frequency and peak flux, in observer frame GHz and Jy
    fit_results=fit_sed(np.array(df['freq_corr']), np.array(df['uJy'])*1e-6, F_err*1e-6)
    nu_p_fit, F_p_fit, a2_fit = fit_results[0]
    nu_p_err, F_p_err, a2_err = fit_results[1]

    # For now, don't do anything special
    nu_p_extra, F_p_extra =sharp_peak_from_two_anchors(a2_fit, nu_p_fit, F_p_fit)

    # Get parameters
    sync = get_shock(nu_p_extra,F_p_extra,mean_time,redshifts[object], object=object)
    # Get errors
    rel_Fp = F_p_err / F_p_fit
    rel_nup = nu_p_err / nu_p_fit

    Rp_rel_err = np.sqrt((9/19 * rel_Fp)**2 + rel_nup**2)
    sync['radius_err'] = "{:e}".format(sync['radius'] * Rp_rel_err)

    Bp_rel_err = np.sqrt((2/19 * rel_Fp)**2 + rel_nup**2)
    sync['magnetic_err'] = "{:e}".format(sync['magnetic'] * Bp_rel_err)

    Up_rel_err = np.sqrt((23/19 * rel_Fp)**2 + (1.0 * rel_nup)**2)
    sync['energy_err'] = "{:e}".format(sync['energy'] * Up_rel_err)

    beta_rel_err = np.sqrt((9/19 * rel_Fp)**2 + (1.0 * rel_nup)**2)
    sync['velocity_err'] = "{:e}".format(sync['velocity'] * beta_rel_err)

    ne_rel_err = np.sqrt((22/19 * rel_Fp)**2 + (4.0 * rel_nup)**2)
    sync['density_err'] = "{:e}".format(sync['density'] * ne_rel_err)
    
    if plot:
        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1)
        x_range=np.logspace(-0.1, 2.5, num=100)
        ax.plot(x_range, bpl_smooth_a1(x_range, nu_p_fit, F_p_fit, a2_fit, a1=5/2))
        ax.errorbar(np.array(df['freq_corr']), np.array(df['uJy'])*1e-6, yerr=F_err*1e-6, linestyle='none', marker='s', ms=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([12e-6, 1700e-6])
        plt.show()
        plt.close()

    return sync

if __name__=='__main__':
    raw_df = pd.read_csv('data/new_radio_data.txt')
    raw_df['freq_corr']=fix_freq(raw_df['freq'], raw_df['object'], raw_df['frame'])
    print(get_and_fit('AT2023fhn', (80, 100)))
    print(get_and_fit('AT2023fhn', (135, 140)))
    print(get_and_fit('AT2023vth', (85, 90)))
    print(get_and_fit('AT2023vth', (100, 130)))
    print(get_and_fit('AT2024aehp', (140, 155)))
