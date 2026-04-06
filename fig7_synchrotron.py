import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
from astropy.table import Table
from scipy.optimize import curve_fit
import astropy.constants as const
import astropy.cosmology.units as cu

import vals
plt.rc('font', size=9)

# uses chevalier, ho et al. 2019 (2018cow)
gamma = 3
c1 = 6.27 * 10**18
f = 0.5 #filling factor
E = (0.511*u.MeV).cgs.value
label_size=(10, 8)

objects = ['AT2022abfc','AT2023fhn', 'AT2023hkw', 'AT2023vth', 'AT2024qfm', 'AT2024aehp']
colors={'AT2022abfc': 'blue','AT2023fhn':'orange', 'AT2023hkw':'brown', 'AT2023vth':'purple'}

redshifts=vals.redshifts



def calc_beta(gamma_beta):
    return gamma_beta/np.sqrt(gamma_beta**2+1)


def get_shock(nu_ghz, f_jansky, t_days, redshift, object=None, limit=False):
    """Assumes alpha = 1, epsilon_e = epsilon_B = 1/3
    f_jansky is peak OBSERVED flux in Jy
    nu_ghz is peak OBSERVED frequency on GHz
    t_days is OBSERVED days after  t_obs, estimated from transient discovery
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
        t_days: Time after  t_obs in days
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

    r = 8.8*10**15 * (f/0.5)**(-1/19) * f_corr**(9/19) * d_lum**(18/19) * (nu_rest/5)**(-1) # cm
    b = 0.58 * (f/0.5)**(-4/19) * f_corr**(-2/19) * d_lum**(-4/19) * (nu_rest/5) # G

    L = 1.2*10**27 * f_corr * d_lum**2 # erg /s/Hz

    
    beta = calc_beta(r*(1+z.value)/(t_days*3600*24*3*10**10)) # Divide r by t_peak get gamma*beta*c, calc beta from gamma*beta

    U = 1.9*10**46 * 3 * (f/0.5)**(8/19) * f_corr**(23/19) * d_lum**(46/19) * (nu_rest/5)**(-1) # erg
    n_e = 1.02418 * 3 * (f/0.5)**(-6/19) * (f_corr)**(-22/19) * d_lum**(-44/19) * (nu_rest/5)**4 * t_rest**2 # cm^-3
    m_dot = 2.5*10**(-5) * 3 * (f/0.5)**(-1/19) * (f_corr)**(-4/19) * d_lum**(-8/19) * (nu_rest/5)**2 * t_rest**2 #1e-4 M_sun / 1000 km/s
    mass_swept = (n_e*np.power(u.cm, -3)*const.m_p*np.power(r*u.cm, 3)).to(u.Msun)
    # Check v_a/v_c

    assert 6.2*10**(-3)*(f_corr)**(6/19)*d_lum**(12/19)*(nu_rest/5)**(-3)*(t_rest/100)**2<0.4
    assert 5.9*10**(-4)*(f_corr)**(6/13)*d_lum**(12/13)*(nu_rest/5)**(-63/13)*(t_rest/100)**(-38/13)<0.4

    return {'radius':r, 'magnetic':b, 'velocity':beta, 'energy':U, 'density':n_e, 'luminosity':L, 'nu_rest': nu_rest, 't':t_rest, 'object': object,
         'limit':limit, 'wind':m_dot, 'mass_swept': mass_swept}



def bpl_smooth_a1(x, xb, Fb, a2, a1=5/2):
    """
    Define the smoothed broken pwoer law equation, with peak at (xb, Fb) and power law slopes of a2, a1.
    `x` is the input variable
    """
    s = 1; sig = np.sign(a1-a2)
    return Fb*((x/xb)**(-sig*s*a1) + (x/xb)**(-sig*s*a2))**(-sig/s)

def sharp_peak_from_two_anchors(a2,  xb, Fb, x1=0.1, x2=500, a1=5/2):
    """
    Find the sharp peak of a smoothed broken power law by extending the two ends of the power law
    towards the center above the smoothed peak
    """
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


def get_slice(object, time):
    """
    Get the radio detections for a particular object and time range
    """
    df = raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']>=time[0])&(raw_df['t_obs']<=time[1])&(raw_df['unc'].astype(float)>=0)].sort_values(by=['freq_corr'])

    return df

raw_df = pd.read_csv('data/new_radio_data.txt')
raw_df['freq_corr']=fix_freq(raw_df['freq'], raw_df['object'], raw_df['frame'])


def get_and_fit(object, time, mean_time, plot=False):
    """
    Retrieves and fits a smoothed power law to a single epoch of observations to get the peak frequency and flux
    `object`: Object Name
    `time`: A tuple containing the minimum and maximum time bins (in observer frame) that defines the epoch

    Returns `sync`, a dictionary containing the derived synchrotron parameters and errors
    Optionally plots the SED of the epoch alongside the fit.
    """
    df = get_slice(object, time)
    min_fluxErr = np.zeros(len(df['unc']))

    # Add an error floor if data from multiple arrays is being used
    if len(np.unique(df['array'])) > 1:
        min_fluxErr = 0.1 * np.abs(df['uJy'])
    F_err = np.maximum(df['unc'], min_fluxErr)

    # Get the peak frequency and peak flux, in observer frame GHz and Jy
    fit_results=fit_sed(np.array(df['freq_corr']), np.array(df['uJy'])*1e-6, F_err*1e-6)

    nu_p_fit, F_p_fit, a2_fit = fit_results[0]
    nu_p_err, F_p_err, a2_err = fit_results[1]
    print(np.array([a2_fit, a2_err]))
    print(np.array([F_p_fit, F_p_err]))
    sync = get_shock(nu_p_fit,F_p_fit,mean_time,redshifts[object], object=object)
    # Get errors
    rel_Fp = F_p_err / F_p_fit
    rel_nup = nu_p_err / nu_p_fit

    sync['nu_err'] = nu_p_err * (1+vals.redshifts[object])
    Rp_rel_err = np.sqrt((9/19 * rel_Fp)**2 + rel_nup**2)
    sync['radius_err'] = float("{:e}".format(sync['radius'] * Rp_rel_err))

    Bp_rel_err = np.sqrt((2/19 * rel_Fp)**2 + rel_nup**2)
    sync['magnetic_err'] = float("{:e}".format(sync['magnetic'] * Bp_rel_err))

    Up_rel_err = np.sqrt((23/19 * rel_Fp)**2 + (1.0 * rel_nup)**2)
    sync['energy_err'] = float("{:e}".format(sync['energy'] * Up_rel_err))

    beta_rel_err = np.sqrt((9/19 * rel_Fp)**2 + (1.0 * rel_nup)**2)
    sync['velocity_err'] = float("{:e}".format(sync['velocity'] * beta_rel_err))

    ne_rel_err = np.sqrt((22/19 * rel_Fp)**2 + (4.0 * rel_nup)**2)
    sync['density_err'] = float("{:e}".format(sync['density'] * ne_rel_err))

    wind_rel_err = np.sqrt((4/19 * rel_Fp)**2 + (2 * rel_nup)**2)
    sync['wind_err'] = float("{:e}".format(sync['wind'] * wind_rel_err))

    mass_swept_rel_err = np.sqrt((ne_rel_err)**2 + (3 * Rp_rel_err)**2)
    sync['mass_swept_err'] = float("{:e}".format(sync['mass_swept'].value * mass_swept_rel_err))
    
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

def print_sync(sync):
    limit=False
    if 'radius_err' in sync.keys():
        print(f"{np.int32(np.round(sync['t'], 0))}&${np.round(sync['nu_rest'], 1)}\\pm{np.round(sync['nu_err'], 1)}$&${np.round(sync['radius']*1e-16, 1)}\\pm{np.round(sync['radius_err']*1e-16, 1)}$&${np.round(sync['magnetic'], 2)}\\pm{np.round(sync['magnetic_err'], 3)}$&${np.round(sync['velocity'], 2)}\\pm{np.round(sync['velocity_err'], 2)}$&${np.round(sync['energy']*1e-48, 2)}\\pm{np.round(sync['energy_err']*1e-48, 2)}$&${np.round(sync['density']*1e-3, 3)}\\pm{np.round(sync['density_err']*1e-3, 3)}$&${np.round(sync['wind'], 3)}\\pm{np.round(sync['wind_err'], 3)}$&${np.round(sync['mass_swept'].value*1e5, 1)}\\pm{np.round(sync['mass_swept_err']*1e5, 1)}$")
    else:
        print(f"""{np.int32(np.round(sync['t'], 0))}&$>{np.round(sync['nu_rest'], 1)}$&$<{np.round(sync['radius']*1e-16, 1)}$&$>{np.round(sync['magnetic'], 2)}$&$<{np.round(sync['velocity'], 2)}$&$<{np.round(sync['energy']*1e-48, 2)}$&$>{np.round(sync['density']*1e-3, 3)}$&$>{np.round(sync['wind'], 3)}$&$>{np.round(sync['mass_swept'].value*1e5, 1)}$""")



#print(get_shock(0.000182,12,120,redshifts['AT2024qfm']))

# Get the shock parameters
sync_abfc=get_shock(16, 0.000052,32,redshifts['AT2022abfc'], object='AT2022abfc', limit=True)  # AT2022abfc, upper limit, 1.5 * 10^16 cm, B = 0.96 G
sync_fhn88=get_and_fit('AT2023fhn', (80, 100), 90)
sync_fhn137=get_and_fit('AT2023fhn', (135, 140), 138)
sync_hkw=get_shock(11, 0.000103,44,redshifts['AT2023hkw'], object='AT2023hkw', limit=True)
sync_hkw117=get_shock(6, 0.000150,117,redshifts['AT2023hkw'], object='AT2023hkw', limit=True)
#sync_vth41=get_shock(15,0.000473,41,redshifts['AT2023vth'], object='AT2023vth')  # AT 2023hkw upper limit, 4.9 * 10^16 cm, B = 0.73 G
sync_vth87=get_and_fit('AT2023vth', (85, 90), 87)
sync_vth118=get_and_fit('AT2023vth', (100, 130), 118)
sync_qfm=get_shock(94,0.000082,12,redshifts['AT2024qfm'], object='AT2024qfm', limit=True)
#sync_qfm2=get_shock(94,0.0000117,34,redshifts['AT2024qfm'], object='AT2024qfm', limit=True)
sync_aehp86=get_shock(15,0.000075,86,redshifts['AT2024aehp'], object='AT2024aehp', limit=True)
sync_aehp141=get_and_fit('AT2024aehp', (140, 157), 149)


## Print the shock parameters to record


#print_sync(sync_abfc)
#print_sync(sync_fhn88)
#print_sync(sync_fhn137)
#print_sync(sync_hkw)
#print_sync(sync_hkw117)
#print_sync(sync_vth87)
#print_sync(sync_vth118)
#print_sync(sync_qfm)
#print_sync(sync_aehp86)
#print_sync(sync_aehp141)
#print(sync_abfc['t'], sync_abfc['mass_swept'])
#print(sync_hkw['t'], sync_hkw['mass_swept'])
#print(sync_hkw117['t'], sync_hkw117['mass_swept'])
#print(sync_vth87['t'], sync_vth87['mass_swept'], sync_vth87['mass_swept_err'])
#print(sync_vth118['t'], sync_vth118['mass_swept'], sync_vth118['mass_swept_err'])
#print(sync_fhn88['t'], sync_fhn88['mass_swept'], sync_fhn88['mass_swept_err'])
#print(sync_fhn137['t'], sync_fhn137['mass_swept'], sync_fhn137['mass_swept_err'])
#print(sync_qfm['t'], sync_qfm['mass_swept'])
##print(sync_qfm2)
#print(sync_aehp86['t'], sync_aehp86['mass_swept'])
#print(sync_aehp141['t'], sync_aehp141['mass_swept'], sync_aehp141['mass_swept_err'])

datasets = [sync_abfc, sync_hkw, sync_hkw117, sync_vth87, sync_vth118, sync_fhn88, sync_fhn137, sync_qfm, sync_aehp86, sync_aehp141]

sync_df = pd.DataFrame.from_records(datasets)


def lfbot(ax):
    col = vals.fbot_col
    m = 'o'
    s = 30

    # Koala
    dcm = Planck18.luminosity_distance(z=0.2714).cgs.value
    tnu = np.array([(81/1.2714)*(10/5), (343)*(1.5/5)])/1.2714
    nu = np.array([10, 5])*1E9
    lpeak = np.array([0.364, 0.089])*1E-3*1E-23*4*np.pi*dcm**2
    ax.scatter(tnu, lpeak, marker=m, c=col, s=s, zorder=10, alpha=0.4)
    ax.plot(tnu, lpeak, color=col, ls='-', zorder=10, alpha=0.4)

    # CSS 161010
    tnu = np.array([69*(5.6/5), 357*0.63/5])/1.033
    nu = np.array([5.6, 0.63])*1E9
    dcm = Planck18.luminosity_distance(z=0.033).cgs.value
    lpeak = np.array([8.8E-3, 1.2E-3])*1E-23*4*np.pi*dcm**2
    ax.scatter(
            tnu, lpeak, marker=m, c=col, s=s, label="_none", zorder=10, alpha=0.4)

    ax.plot(tnu, lpeak, color=col, ls='-', zorder=10, alpha=0.4)


    # AT2020xnd
    x1 = 58*21.6/5
    dcm = Planck18.luminosity_distance(z=0.2442).cgs.value
    y1 = (0.68*1E-3*1E-23*4*np.pi*dcm**2)
    ax.scatter(
            x1, y1, marker=m, s=s, facecolors=col, edgecolors=col, zorder=10, alpha=0.4)

    # AT2020mrf
    x1 = 364
    y1 = 1.03E29
    ax.scatter(
            x1, y1, marker=m, s=s, facecolors=col, edgecolors=col, zorder=10, alpha=0.4)

    # AT2022tsd
    x = [40*(100/5)]
    dcm = Planck18.luminosity_distance(z=0.2567).cgs.value
    yf = np.array([0.3])
    y = yf*1E-3*1E-23*4*np.pi*dcm**2
    ax.scatter(x, y, marker=m, c=col, s=s,
               facecolors=col, zorder=100, alpha=0.4)
    
    #ax.text(
    #        x[0]*1.1, y[0]*1.2,"AT2022tsd",verticalalignment='bottom',
    #        horizontalalignment='center', color=col, zorder=100)

    # AT2018cow
    x1 = 22*100/5
    y1 = 4.4E29
    ax.scatter(
            x1, y1, marker=m, s=s, 
            facecolors=col, edgecolors=col, zorder=10, alpha=0.4)
    #ax.text(
    #        x1*1, y1/1.25, "AT2018cow", 
    #        verticalalignment='center',
    #        horizontalalignment='left', color=col, zorder=100)
    x2 = 91*10/5
    y2 = 4.3E28
    ax.scatter(x2, y2, marker=m, s=s, facecolors=col, edgecolors=col,
               label='LFBOT', zorder=10, alpha=0.4)
    ax.plot([x1,x2], [y1,y2], color=col, ls='-', zorder=10, alpha=0.4)
    plt.arrow(x1,y1,x2-x1,y2-y1, color=col, zorder=10, alpha=0.4)

    # AT2024wpp Nayana et al. 2025
    #dcm = Planck18.luminosity_distance(z=0.0868).cgs.value
    #t = np.array([32.4, 46.1, 72.7, 117.6])
    #nu = np.array([90, 21, 8, 6])
    #x =  t*nu/5
    #y = np.array([2.697, 1.524, 1.673, 0.624])*1E-3*1E-23*4*np.pi*dcm**2
    #ax.scatter(x, y, marker=m, s=s, facecolors=col, edgecolors=col,
    #           zorder=10, alpha=0.4)
    #ax.plot(x, y, color=col, ls='-', zorder=10, alpha=0.4)

    # AT2024wpp Perley et al. 2026
    dcm = Planck18.luminosity_distance(z=0.0868).cgs.value
    t = np.array([30, 40, 65, 107, 176]) # days
    nu = np.array([58, 40, 18, 9, 6.7]) # GHz
    x =  t*nu/5
    y = np.array([1.9, 2.7, 1.7, 0.9, 0.2])*1E-3*1E-23*4*np.pi*dcm**2
    ax.scatter(x, y, marker=m, s=s, facecolors=col, edgecolors=col,
               zorder=10, alpha=0.4)
    ax.plot(x, y, color=col, ls='-', zorder=10, alpha=0.4)


def typeii(ax):
    # 88Z, 79C
    tnu = np.array([1253*5/5, 1400*1.4/5])
    lpeak = np.array([2.2E28, 4.3E27])
    names = ['88Z', '79C']

    for i,tnuval in enumerate(tnu):
        if i==0:
            label='SN'
        else:
            label=None
        ax.scatter(
                tnuval, lpeak[i], marker='+', c=vals.sn_col, 
                s=30, label=label,zorder=1)


def ibc(ax):
    # 2003L
    tnu = (30)*(22.5/5)
    lpeak = 3.3E28
    ax.scatter(
            tnu, lpeak, marker='+', c=vals.sn_col, s=30,
            label=None,zorder=3)

    # 11qcj
    tnu = (100)*(5/5)
    lpeak = 7E28
    ax.scatter(
            tnu, lpeak, marker='+', c=vals.sn_col, s=30,
            label=None,zorder=3)
    # 2007bg
    tnu = (55.9)*(8.46/5)
    lpeak = 4.1E28
    ax.scatter(
            tnu, lpeak, marker='+', c=vals.sn_col, s=30, label=None,zorder=3)

    # SN 2003bg
    tnu = (35)*(22.5/5)
    lpeak = 3.9E28
    ax.scatter(
            tnu, lpeak, marker='+', c=vals.sn_col, s=30, label=None,zorder=3)

    # SN 2009bb
    tnu = (20)*(6/5)
    lpeak = 3.6E28
    ax.scatter(
            tnu, lpeak, marker='+', c=vals.sn_col, s=30,zorder=3)


def tde(ax):
    # ASASSN14li
    tnu = (143)*(8.20/5)
    lpeak = 1.8E28
    ax.scatter(
            tnu, lpeak, marker='o', edgecolor=vals.tde_col, 
            facecolor='white', s=20,
            label='TDE')


def llgrb(ax):
    # SN 1998bw
    tnu = (10)*(10/5)
    lpeak = 8.2E28
    ax.scatter(
            tnu, lpeak, marker='x', s=20,
            facecolor=vals.llgrb_col, label="LLGRB")

    # GRB 171205A
    tnu = (4.3)*(6/5)
    dgrb = Planck18.luminosity_distance(z=0.0368).cgs.value
    # 3 mJy at 6 GHz with the VLA; Laskar et al. 2017
    lpeak = 3E-3 * 1E-23 * 4 * np.pi * dgrb**2
    ax.scatter(
            tnu, lpeak, marker='x', s=20,
            facecolor=vals.llgrb_col, label=None)

    # SN 2006aj
    tnu = (5)*(4/5)
    lpeak = 8.3E27
    ax.scatter(
            tnu, lpeak, marker='x', s=20,
            facecolor=vals.llgrb_col, label=None)

    # SN 2010bh
    tnu = (30)*(5/5)
    lpeak = 1.2E28
    ax.scatter(
            tnu, lpeak, marker='x', s=20,
            facecolor=vals.llgrb_col, label=None)


def vele(ax):
    # only use this to plot the GRBs
    direc = "data/radio"
    inputf = direc + "/Soderberg2009Fig4.1.txt"

    dat = Table.read(inputf, format='ascii')

    
    # AT2018cow
    v = 0.126807067696
    E = 3.68387786116342e+48
    ax.scatter(
            v, E, marker='o', c=vals.fbot_col, s=30, zorder=10, alpha=0.4, label='LFBOT')
    
    # Yao et al. AT2020mrf
    v = 0.08
    E = 1.7e+49
    ax.scatter(
            v, E, marker='o', c=vals.fbot_col, s=30, zorder=10, alpha=0.4)
    
    # Coppenjans CSS161010
    v = [0.53, 0.55, 0.36]
    E = [2.9e+49, 5.6e+49, 2.4e+49]
    ax.scatter(
            v, E, marker='o', c=vals.fbot_col, s=30, zorder=10, alpha=0.4)
    ax.plot(
            v, E, c=vals.fbot_col, lw=2, zorder=10, alpha=0.4)

    # AT2024wpp,  Nayana et al. 2025 
    #v = [0.067, 0.244, 0.418, 0.198]
    #E = np.array([0.76, 11.44, 33.12, 11.68])*1e48
    #ax.scatter(
    #        v, E, marker='o', c=vals.fbot_col, s=30, zorder=10, alpha=0.4)
    #ax.plot(
    #        v, E, c=vals.fbot_col, lw=2, zorder=10, alpha=0.4)
    
    # AT2024wpp,  Perley et al. 2026
    v = [0.15, 0.19, 0.2, 0.18, 0.07]
    E = np.array([3.1, 6.9, 8.4, 7.8, 1.6])*1e48
    ax.scatter(
            v, E, marker='o', c=vals.fbot_col, s=30, zorder=10, alpha=0.4)
    ax.plot(
            v, E, c=vals.fbot_col, lw=2, zorder=10, alpha=0.4)
    
    # SN 2003L
    v = 0.12
    E = 6.969524134247872e+47
    ax.scatter(
            v, E, marker='+', c=vals.sn_col, s=30, label="SN")

    # SN 2006aj
    v = 2.12
    E = 7.463410905948252e+47
    ax.scatter(
            v, E, marker='x', edgecolor=vals.llgrb_col, s=20, facecolor=vals.llgrb_col, label='LLGRB')

        
    # Swift TDE
    # just use one epoch
    # epoch 3: day 18
    # ASASSN14li
    # Using Day 143, the first day the peak is resolved
    v = 0.05
    E = 9.385748112535813e+47
    ax.scatter(
            v, E, marker='o', edgecolor=vals.tde_col, 
            facecolor='white', s=20, label='TDE')

    # SN 2007bg
    v = 0.19
    E = 2.4548568053628734e+48
    ax.scatter(
            v, E, marker='+', c=vals.sn_col, s=30, label=None)


    # SN 2003bg
    v = 0.11
    E = 8.736630286056538e+47
    ax.scatter(
            v, E, marker='+', c=vals.sn_col, s=30, label=None)


    # SN 1998bw
    v = 1.26
    E = 4.7885240374184584e+48

    # SN 2009bb
    v = 0.71
    E = 2.9571483301386266e+48
    ax.scatter(
            v, E, marker='+', c=vals.sn_col, s=30)


    # SN 2010bh
    v = 0.33
    E = 9.092980420991635e+47
    ax.scatter(
            v, E, marker='x', edgecolor=vals.llgrb_col, s=20, facecolor=vals.llgrb_col, label=None)


    # SN 1988Z
    v = 0.011
    E = 1.985096671996441e+48
    ax.scatter(
            v, E, marker='+', c=vals.sn_col, s=30)

    # SN 1979C
    v = 0.016
    E = 1.0369335161200779e+48
    ax.scatter(
            v, E, marker='+',  c=vals.sn_col, s=30)
    




fig, ax = plt.subplots(1, 2, figsize=(8,4))
fig.set_tight_layout({"pad": .0})
ax[0].set_xlabel('$t_{\\text{p}}$/day $\\times$ $\\nu_{\\text{rest}}$/5 GHz', fontsize=12)
ax[0].set_ylabel('$L_{\\nu}$ [erg/s/Hz]', fontsize=12)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].tick_params(labelsize=13)
x = np.linspace(3, 2500)
# Plot lines of constant velocity
for i,vel in enumerate(np.array([0.2, 0.4, 1.0])):
    ax[0].plot(x, 10**26*np.power(vel*x, 19/9), color='lightgrey', ls='-', lw=3)
    coef = [1.6, 0.95, 0.8]
    ax[0].text(coef[i]*np.power(2.8e30*1e-26,9/19)/vel, 2.1e30, "$v={}c$".format(vel) ,verticalalignment='center',
        fontsize=8, horizontalalignment='center', zorder=3)
# Plot lines of constant mass loss rate
for m_dot in np.array([0.01, 1, 100]):
    ax[0].plot(x, (1.2e27)*np.power(m_dot/3/2.5e-5, -19/4) * x **(19/2), color='lightgrey', ls='dotted', lw=3)
    if m_dot>=1: m_dot=int(m_dot)
    coef=[2,1.3,1.1]
    ax[0].text(coef[i]*np.power(m_dot/(2.5*10**(-5) * 3), 1/2)*np.power(1.15e27/(1.2e27), 2/19), 9.8e27, "$\dot M/v_w={}$".format(str(m_dot)) ,verticalalignment='center',
        fontsize=10, horizontalalignment='right', zorder=3)


for object in objects:
    marker=vals.markers[object]
    s=50
    if object =='AT2023fhn': s= 80
    sync_df_object = sync_df.loc[sync_df['object']==object]
    x_data = sync_df_object['t']*sync_df_object['nu_rest']/5
    y_data = sync_df_object['luminosity']
    ax[0].scatter(x_data, y_data, #label=sync_df_object['object'].iloc[0],
                marker=marker, ec='black', fc=vals.colors[object], s=s, zorder=200, label=object)
    ax[0].plot(x_data, y_data, color=vals.colors[object], ls='-')
    # Plot arrows
    if object=='AT2022abfc' or object=='AT2023hkw' or object=='AT2024qfm' or object=='AT2024aehp':
        ax[0].arrow(x_data.iloc[0], y_data.iloc[0], 0, -y_data.iloc[0]/2.6, length_includes_head=True,
                    head_width=x_data.iloc[0]/6, head_length=y_data.iloc[0]/10, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
        ax[0].arrow(x_data.iloc[0], y_data.iloc[0], x_data.iloc[0]*0.7, 0, length_includes_head=True,
                    head_width=y_data.iloc[0]/6, head_length=x_data.iloc[0]/4, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
    if object=='AT2023hkw':
        ax[0].arrow(x_data.iloc[1], y_data.iloc[1], 0, y_data.iloc[1]*1/1.6, length_includes_head=True,
                    head_width=x_data.iloc[1]/6, head_length=y_data.iloc[1]/5, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
        ax[0].arrow(x_data.iloc[1], y_data.iloc[1], -x_data.iloc[1]*0.7/1.7, 0, length_includes_head=True,
                    head_width=y_data.iloc[1]/6, head_length=x_data.iloc[1]/10, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
        

def make_lv_labels(ax, objects):
    # Label the points of the nu vs. Lnu plot
    for object in objects:
        sync_df_object = sync_df.loc[sync_df['object']==object]
        x_data = sync_df_object['t']*sync_df_object['nu_rest']/5
        y_data = sync_df_object['luminosity']
        sync_df_object['t']=np.int32(np.round(sync_df_object['t']))

        #if object=='AT2022abfc':
            #ax.text(x_data.iloc[0]*0.6, y_data.iloc[0]/3.1, object, verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[0], fontweight='bold', zorder=200)
            #ax.text(x_data.iloc[0]*0.6, y_data.iloc[0]/4., '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
        #elif object=='AT2023fhn':
            #ax.text(x_data.iloc[0]*0.3, y_data.iloc[0]/1.45, object, fontsize=label_size[0], fontweight='bold', verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], zorder=200)
            #ax.text(x_data.iloc[0]*0.3, y_data.iloc[0]/1.8, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
            #ax.text(x_data.iloc[1]*2.1, y_data.iloc[1]*1.66, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
        #elif object=='AT2023hkw':
            #ax.text(x_data.iloc[0]*0.3, y_data.iloc[0]*1.25, object, verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object],fontsize=label_size[0],fontweight='bold', zorder=200)
            #ax.text(x_data.iloc[0]*0.3, y_data.iloc[0]*1, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
            #ax.text(x_data.iloc[1]*1.4, y_data.iloc[1]*1.76, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='center',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
        #elif object=='AT2023vth':
            #ax.text(x_data.iloc[1]/1.4, y_data.iloc[1]/1.5, object, verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[0],fontweight='bold', zorder=200)
            #ax.text(x_data.iloc[0]*1.25, y_data.iloc[0]/1.43, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
            #ax.text(x_data.iloc[1]/1.4, y_data.iloc[1]/1.9, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
        #elif object=='AT2024qfm':
            #ax.text(x_data.iloc[0]*3.4, y_data.iloc[0]*1.1, object, verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[0],fontweight='bold', zorder=200)
            #ax.text(x_data.iloc[0]*2.9, y_data.iloc[0]/1.23, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
        #elif object=='AT2024aehp':
            #ax.text(x_data.iloc[0]*1.3, y_data.iloc[0]/2.3, object, verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[0],fontweight='bold', zorder=200)
            #ax.text(x_data.iloc[0]*1.3, y_data.iloc[0]/2.8, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)
            #ax.text(x_data.iloc[1]*2.4, y_data.iloc[1]/1.5, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object], fontsize=label_size[1], zorder=200)

# Add other objects from the literature
typeii(ax[0])
ibc(ax[0])
tde(ax[0])
llgrb(ax[0])
lfbot(ax[0])
make_lv_labels(ax[0], objects)

ax[0].legend(loc='upper left', prop={'size': 8})


ax[0].set_ylim([8.2e27, 2.7e30])
ax[0].set_xlim([4, 2400])


ax[1].set_xlabel('Blastwave velocity $\\Gamma\\beta$', fontsize=12)
ax[1].set_ylabel('Total Shock Energy [erg]', fontsize=12)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].tick_params(labelsize=13)


for object in objects:
    marker=vals.markers[object]
    sync_df_object = sync_df.loc[sync_df['object']==object]
    x_data= sync_df_object['velocity']*np.power(1-sync_df_object['velocity']*sync_df_object['velocity'], -0.5)
    y_data=sync_df_object['energy']
    ax[1].scatter(x_data,
                   y_data, #label=sync_df_object['object'].iloc[0],
                   color=vals.colors[object], marker=marker, s=80,
                   )
    ax[1].plot(x_data, y_data, color=vals.colors[object], ls='-')
    if object=='AT2022abfc' or object=='AT2023hkw' or object=='AT2024qfm' or object=='AT2024aehp':
        ax[1].arrow(x_data.iloc[0], y_data.iloc[0], 0, y_data.iloc[0]*1/1.4, length_includes_head=True,
                    head_width=x_data.iloc[0]/6, head_length=y_data.iloc[0]/5, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
        ax[1].arrow(x_data.iloc[0], y_data.iloc[0], -x_data.iloc[0]/3, 0, length_includes_head=True,
                    head_width=y_data.iloc[0]/6, head_length=x_data.iloc[0]/10, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
    if object=='AT2023hkw':
        ax[1].arrow(x_data.iloc[1], y_data.iloc[1], 0, y_data.iloc[1]*1/1.4, length_includes_head=True,
                    head_width=x_data.iloc[1]/6, head_length=y_data.iloc[1]/5, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)
        ax[1].arrow(x_data.iloc[1], y_data.iloc[1], x_data.iloc[1]*0.8/1.5, 0, length_includes_head=True,
                    head_width=y_data.iloc[1]/6, head_length=x_data.iloc[1]/6, edgecolor=vals.colors[object], fc='white', alpha=0.7, zorder=400)





def make_vele_labels(ax, objects):
    # Make labels for the velocity vs energy figure
    for object in objects:
        sync_df_object = sync_df.loc[sync_df['object']==object]
        x_data= sync_df_object['velocity']*np.power(1-sync_df_object['velocity']*sync_df_object['velocity'], -0.5)
        y_data=sync_df_object['energy']
        sync_df_object['t']=np.int32(np.round(sync_df_object['t']))

        # Just formatting the location of labels
        #if object=='AT2022abfc':
            #ax.text(x_data.iloc[0]*2.35, y_data.iloc[0], object, verticalalignment='bottom',
            #    horizontalalignment='center', fontsize=label_size[0],fontweight='bold',color=vals.colors[object])
            #ax.text(x_data.iloc[0]*2.35, y_data.iloc[0]/1.2, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
        #elif object=='AT2023fhn':
            #ax.text(x_data.iloc[1]/2, y_data.iloc[1]*1.1, object, fontsize=label_size[0],fontweight='bold', verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[0]*1.2, y_data.iloc[0]*1.2, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[1]/2, y_data.iloc[1]*0.91, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
        #elif object=='AT2023hkw':
            #ax.text(x_data.iloc[0]/1.1, y_data.iloc[0]*0.6, object, fontsize=label_size[0],fontweight='bold', verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[0]/1.1, y_data.iloc[0]*0.5, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[1]*1.65, y_data.iloc[1]*1.8, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='center', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
        #elif# object=='AT2023vth':
            #ax.text(x_data.iloc[0]/1.83, y_data.iloc[0]*1.7, object, fontsize=label_size[0],fontweight='bold', verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[0]/1.83, y_data.iloc[0]*1.42, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[1]*1.3, y_data.iloc[1]/1.52, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
        #elif object=='AT2024qfm':
            #ax.text(x_data.iloc[0]*2.7, y_data.iloc[0]/1.15, object, fontsize=label_size[0],fontweight='bold', verticalalignment='bottom',
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[0]*2.7, y_data.iloc[0]/1.4, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
        #elif object=='AT2024aehp':
            #ax.text(x_data.iloc[0]/2.4, y_data.iloc[0]*1.33, object, verticalalignment='bottom',
            #    horizontalalignment='center', fontsize=label_size[0],fontweight='bold',color=vals.colors[object])
            #ax.text(x_data.iloc[0]/2.4, y_data.iloc[0]*1.07, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[0]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])
            #ax.text(x_data.iloc[1]/2, y_data.iloc[1]/1.1, '$\\Delta t \\approx {}$ d'.format(sync_df_object['t'].iloc[1]),
            #        verticalalignment='bottom', fontsize=label_size[1],
            #    horizontalalignment='center', color=vals.colors[object])


vele(ax[1])
make_vele_labels(ax[1], objects)



ax[1].set_xlim([0.01, 1.3])
ax[1].set_ylim([3.1e47, 1.17e50])

fig.tight_layout(pad=3.0)
plt.tight_layout()
plt.savefig('figures/fig7_synchrotron.pdf', dpi=450)
plt.show()
plt.close()