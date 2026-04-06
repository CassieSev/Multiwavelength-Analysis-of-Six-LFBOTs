import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import numpy as np
import sys
from astropy.cosmology import Planck18
from astropy.time import Time
import sys
sys.path.append("..")
import vals
from get_radio import *
import pandas as pd




def ptf11qcj(ax, col, legend):
    """ PTF 11qcj
    Corsi et al. 2014
    """
    d = 124 * 3.086E24

    # HIGH FREQUENCY (Carma)
    t0 = 55857.543-15
    dt = np.array([55884.803-t0, 55891.640-t0])/1.0287
    f = np.array([3.96, 3.62])
    ef = [0.88, 0.75]
    nu = 93E9

    l = f*1E-3*1E-23*4*np.pi*d**2
    nu_l = l*nu
    ax.plot(dt, nu_l, c=col, ls='--', zorder=10, label=legend)
    #ax.scatter(dt, l/1.2, c=col, marker='>', zorder=10)
    


def sn2008d(ax, col, legend):
    """ SN 2008D
    Soderberg et al. 2008
    """
    d = 29.9 * 3.086E24

    # HIGH FREQUENCY (Carma)
    dt = np.array([4.94, 6.84])/1.00606547
    f = np.array([3.2, 0.6])
    ef = [0.7, 0.3]
    nu = 95E9

    l = f*1E-3*1E-23*4*np.pi*d**2
    nu_l = l*nu
    #ax.scatter(dt, l/1.2, c=col, marker='>')
    ax.plot(dt, nu_l, c=col, ls='--', label=legend)




def sn2020oi(ax, col, legend):
    """ Maeda et al. 2021 """
    d = 15.5 * 3.086E24
    dt = np.array([5.4, 8.4, 18.3, 51.3])/1.0029274
    fnu = np.array([1.3, 1.22, 0.196, 0.115])
    l = fnu*1E-3*1E-23*4*np.pi*d**2
    nu_l = l*100e9
    #ax.scatter(dt, l, color=col, marker='>')
    ax.plot(dt, nu_l, c=col, ls='--', label=None)




def grb181201A(ax, col, legend):
    """ Laskar et al. 2018
    """
    z = 0.450
    d = Planck18.luminosity_distance(z=z).cgs.value

    freq = 97e9
    t = np.array([8.847e-01, 1.917e+00, 3.877e+00, 8.661e+00, 2.981e+01])/1.450
    flux = np.array([3.413, 1.987, 1.199, 0.624, 0.259])
    lum = flux * 1E-3 * 1E-23 * 4* np.pi*d**2
    nu_lum = lum*freq
    ax.plot(t, nu_lum, c=col, ls='dotted')


def grb161219B(ax, col, legend):
    """ Laskar et al. 2016 """
    z = 0.1475
    d = Planck18.luminosity_distance(z=z).cgs.value

    freq = 100e9
    t = np.array([1.30, 3.30, 8.31, 24.45, 78.18])/1.1475
    flux = np.array([1244, 897, 500, 285, 51])
    lum = flux * 1E-6 * 1E-23 * 4* np.pi*d**2
    nu_lum = lum*freq
    ax.plot(t, nu_lum, c=col, ls='dotted')


def grb130427A(ax, col, legend):
    """ Perley et al 2013
    They have data from CARMA/PdBI at 90 GHz (3mm)
    and also CARMA at 85.00 GHz, which is close enough
    But by the time they caught it, it was fading
    """
    z = 0.340
    d = Planck18.luminosity_distance(z=z).cgs.value

    freq = 93E9
    obs_t_1 = np.array([0.77, 1, 1.91, 2.8]) 
    obs_flux_1 = np.array([3416, 2470, 1189, 807]) * 1E-6
    obs_lum_1 = obs_flux_1 * 1E-23 * 4 * np.pi * d**2
    obs_nu_lum_1 = obs_lum_1*freq/1.340

    freq = 85E9 
    obs_t_2 = np.array([0.81, 3.58, 6.41, 10.36, 23.52])
    obs_flux_2 = np.array([3000, 903, 588, 368, 197]) * 1E-6
    obs_lum_2 = obs_flux_2 * 1E-23 * 4 * np.pi * d**2
    obs_nu_lum_2 = obs_lum_2*freq

    t = np.hstack((obs_t_1, obs_t_2))/1.340
    lum = np.hstack((obs_lum_1, obs_lum_2))
    nu_lum = np.hstack((obs_nu_lum_1, obs_nu_lum_2))
    order =np.argsort(t)

    ax.plot(t, nu_lum, c=col, label=legend, ls='dotted')


def j1644(ax, col, legend):
    """ Zauderer et al. 2011
    They have data from CARMA at 94 GHz (3mm)
    and also CARMA at 87.00 GHz, which is close enough
    But by the time they caught it, it was fading
    """
    z = 0.354
    d = Planck18.luminosity_distance(z=z).cgs.value

    freq = 94E9
    obs_t_1 = np.array([1.85, 19.06])
    obs_flux_1 = np.array([15.7, 10.7]) * 1E-3
    obs_lum_1 = obs_flux_1 * 1E-23 * 4 * np.pi * d**2
    obs_nu_lum_1 = obs_lum_1*freq

    freq = 87E9 
    obs_t_2 = np.array([5.14, 6.09, 7.18, 9.09, 14.61, 19.06, 22.07])
    obs_flux_2 = np.array([18.6, 21.7, 14.6, 15.1, 10.4, 9.36, 5.49])*1E-3
    obs_lum_2 = obs_flux_2 * 1E-23 * 4 * np.pi * d**2
    obs_nu_lum_2 = obs_lum_2*freq

    t = np.hstack((obs_t_1, obs_t_2))/1.354
    lum = np.hstack((obs_lum_1, obs_lum_2))
    nu_lum = np.hstack((obs_nu_lum_1, obs_nu_lum_2))
    order =np.argsort(t)

    #ax.scatter(t[order], lum[order], marker='o', s=25,
    #        facecolor='white', edgecolor=col, label=legend,zorder=100)
    ax.plot(t[order], nu_lum[order], c=col, label=legend, lw=1,zorder=0, ls='-.')



def igr(ax, col, legend):
    """ IGR J12580+0134
    They have data from Planck at 100 GHz
    Discovered by INTEGRAL (https://www.astronomerstelegram.org/?read=3108)

    First detection: 2011 Jan 2-11
    Last non-detection: 2010 Dec 30 to 2011 Jan 2

    So... the dt is something like 1 day to 12 days?
    """
    d = 17*3.086E24 # Mpc to cm

    t = 10 # estimate
    freq = 100E9
    lum = 640*1E-3*1E-23*4*np.pi*d**2
    nu_lum = lum*100e9
    #ax.scatter(t, nu_lum, marker='o',s=25,
    #        facecolor='white', edgecolor=col, label=legend,zorder=100)
    ax.plot(t, nu_lum, c=col, label=legend, lw=1,zorder=0, ls='-.')



def sn1998bw(ax, col, legend):
    """ SN 1998bw
    
    This is a bit complicated because there are two peaks in the light curve,
    but I am using a frequency that has a main peak rather than a frequecy
    with two clear distinct peaks...
    """
    d = 1.17E26 # cm
    nu = 150E9
    t = np.array([12.4])/1.0085
    f = np.array([39])
    lum = f*1E-3*1E-23*4*np.pi*d**2
    nu_lum=lum*nu
    ax.scatter(t, nu_lum, c=col, label=legend, marker='s', s=15)


def sn2017iuk(ax, col, legend):
    """ SN 2017iuk
    """
    d = Planck18.luminosity_distance(z=0.0368).cgs.value
    nu = 92E9 # Band 3
    t = np.array([6.10])/(1+0.0368)
    f = np.array([28])
    l = f*1E-3*1E-23*4*np.pi*d**2
    nu_l=l*nu
    ax.scatter(t,nu_l,c=col, marker='s', s=15)



def at2018cow(ax, col, legend):
    """ 230 GHz values """
    dcm = Planck18.luminosity_distance(z=0.0141).cgs.value
    df = pd.read_csv("data/radio/at2018cow_sma.csv")
    dt = df['dt'].values/(1+0.0141)
    f = df['f'].values
    ef = df['ef'].values
    ef_comb = np.sqrt(ef**2 + (0.15*f)**2)
    nu = 231.5E9
    lum = f * 1E-3 * 1E-23 * 4 * np.pi * dcm**2
    nu_lum=lum*nu
    ax.scatter(dt, nu_lum, c=col, marker='D', label=legend, zorder=3, s=10, alpha=0.3)
    ax.plot(dt, nu_lum, c=col, ls='-', lw=2, label=None, zorder=3, alpha=0.3)
    #ax.text(
    #        dt[-1]*1.1, lum[-1]/1.5, 'AT2018cow', ha='center', va='top',
    #        color=col, fontsize=8)



def at2020xnd(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=0.2442).cgs.value
    dt = np.array([17,24,31,39,46,67])/(1+0.2442)
    fnu = np.array([305,648,1030,868,558,330])
    efnu = np.array([57,44,44,46,38,48])
    nu = np.array([94,94,94,94,94,79])*1e9
    lum = fnu * 1E-6 * 1E-23 * 4 * np.pi * dcm**2 
    nu_lum=np.multiply(nu, lum)
    ax.scatter(dt, nu_lum, c=col, marker='D', label=legend, zorder=10, s=10, alpha=0.3)
    ax.plot(dt, nu_lum, c=col, ls='-', lw=2, label=None, zorder=10, alpha=0.3)


def at2022tsd(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=0.256).cgs.value
    dt = np.array([22.83, 27.70, 28.56, 30.04, 41.23, 58.58, 79.23])/(1+vals.z)
    fnu = np.array([0.245, 0.284, 0.316, 0.179, 0.299, 0.304, 0.153])
    efnu = np.array([0.065, 0.032, 0.078, 0.037, 0.093, 0.030, 0.024])
    nu = np.array([92e9]*len(fnu))
    lum = fnu * 1E-3 * 1E-23 * 4 * np.pi * dcm**2 
    nu_lum = np.multiply(nu, lum)
    ax.scatter(
            dt, nu_lum, facecolor=col, marker='D', 
            label=legend, edgecolor='k', s=10, zorder=3, alpha=0.3)
    ax.plot(dt, nu_lum, c=col, ls='-', lw=2, label=None, zorder=3, alpha=0.3)


def at2022cmc(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=1.193).cgs.value
    t0 = Time(59621.4, format='mjd')
    dt = Time(
            np.array(['2022-02-18','2022-02-20', '2022-02-22', '2022-02-24',
             '2022-02-25', '2022-02-26', '2022-02-28', '2022-03-02',
             '2022-03-06']), format='isot')-t0
    dt = dt.value/(1+1.193)

    fnu = np.array([9.1, 7.4, 7.9, 7.4, 6.6, 4.65, 5.0, 4.0, 3.0])
    efnu = np.array([0.9]*len(fnu))
    lnu = fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu = lnu*230e9
    #ax.scatter(dt, nu_lnu, marker='o', edgecolor=col, facecolor='white',zorder=100,
    #           s=25)
    ax.plot(dt,nu_lnu,c=col,zorder=0, ls='-.', lw=1)
    #ax.text(
    #        dt[-1]*1.01, lnu[0], '22cmc (230 GHz)', fontsize=10, 
    #        ha='left', va='center', color='red', fontweight='bold')


def at2024wpp(ax, col, legend):
    # Nayana et al. 2025 
    dcm = Planck18.luminosity_distance(z=0.0868).cgs.value
    t0 = 60578.4
    dt = np.array([60597.22, 60614.06]) -t0
    dt = dt/(1+0.0868)

    fnu = np.array([0.076, 1.282])
    efnu = np.array([0.9]*len(fnu))
    lnu = fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu = lnu*97.5e9
    #ax.scatter(dt, nu_lnu, marker='o', edgecolor=col, facecolor='white',zorder=100,
    #           s=25)
    ax.scatter(
            dt, nu_lnu, facecolor=col, marker='D', 
            label=legend, edgecolor='k', s=10, zorder=3, alpha=0.3)
    ax.plot(dt,nu_lnu, c=col, ls='-', lw=2, label=None, zorder=3, alpha=0.3)


def at2022abfc(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=0.212).cgs.value
    dt=53/(1+0.212)
    # in milli-Jansky
    fnu=0.348
    # Convert to L_nu units
    lnu=fnu*1E-3*1E-23*4*np.pi*dcm**2
    # Get nu * L_nu
    nu_lnu=lnu*86e9
    col=vals.colors['AT2022abfc']
    ax.scatter(dt,nu_lnu,c=vals.colors['AT2022abfc'],label=legend,edgecolor='k', s=60, marker=vals.markers['AT2022abfc'])
    ax.arrow(dt,nu_lnu,0,-(nu_lnu)/1.5,
            length_includes_head=True,head_length=(nu_lnu)/8,
            head_width=dt/7,facecolor=col, edgecolor='black')
    ax.text(
            63, nu_lnu*3.1, 'AT2022abfc', ha='center', va='top',
            color=col, fontsize=12, fontweight='bold')
    

def at2023fhn(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=0.2377).cgs.value
    dt=12/(1+0.2377)
    fnu=0.09
    lnu=fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu=lnu*86.25e9
    col=vals.colors['AT2023fhn']
    ax.scatter(dt-0.5,nu_lnu,c=col,label=legend, edgecolor='k', marker=vals.markers['AT2023fhn'])
    ax.arrow(dt-0.5,nu_lnu,0,-(nu_lnu)/1.5,
            length_includes_head=True,head_length=(nu_lnu)/8,
            head_width=dt/7,facecolor=col, edgecolor='k')
    ax.text(
            8.5, nu_lnu/3.5, 'AT2023fhn', ha='center', va='top',
            color=col, fontsize=12, fontweight='bold')
    

def at2023vth(ax, col, legend):
    dcm = Planck18.luminosity_distance(z=0.0747).cgs.value
    # Using 100 GHz rest (upper band 1)
    dt=np.array([33, 46, 71, 104])/(1+0.0747)
    fnu=np.array([0.35, 0.349, 0.257, 0.215])
    lnu=fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu=lnu*100e9
    col=vals.colors['AT2023vth']
    ax.plot(dt,nu_lnu,c=col,label=legend,lw=2,zorder=100)
    ax.scatter(dt,nu_lnu,c=col,label=None,edgecolor='k', marker=vals.markers['AT2023vth'])
    ax.text(
            98, 1.8e39, 'AT2023vth', ha='center', va='top',
            color=col, fontsize=12, fontweight='bold')


def at2024qfm(ax, col):
    dcm = Planck18.luminosity_distance(z=0.227).cgs.value
    dt=np.array([12, 19, 33, 58])/(1+0.227)
    fnu=np.array([0.088, 0.147, 0.117, 0.060]) # In mJy
    lnu=fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu=lnu*94e9
    ax.arrow(dt[-1],nu_lnu[-1],0,-(nu_lnu[-1])/1.3,
            length_includes_head=True,head_length=(nu_lnu[-1])/6,
            head_width=dt[-1]/7,facecolor=col, edgecolor='k')
    ax.plot(dt,nu_lnu,c=col,lw=2,zorder=100)
    ax.scatter(dt,nu_lnu,c=col,label=None,edgecolor='k', marker=vals.markers['AT2024qfm'])
    ax.text(
            dt[-1]*1.63, nu_lnu[-1]*1.15, 'AT2024qfm', ha='center', va='top',
            color=col, fontsize=12, fontweight='bold')

def at2024aehp(ax, col):
    dcm = Planck18.luminosity_distance(z=0.1715).cgs.value
    dt=np.array([155])/(1+0.227)
    fnu=np.array([0.17543]) # In mJy
    lnu=fnu*1E-3*1E-23*4*np.pi*dcm**2
    nu_lnu=lnu*102e9
    ax.plot(dt,nu_lnu,c=col,lw=2,zorder=100)
    ax.scatter(dt,nu_lnu,c=col,label=None,edgecolor='k', marker=vals.markers['AT2024aehp'])
    ax.text(
            dt[-1]/1.2, nu_lnu[-1]*2.3, 'AT2024aehp', ha='center', va='top',
            color=col, fontsize=12, fontweight='bold')
    
def run(ax):
    """Some objects hidden as they are out of range, or do not show up
    well on the plot."""


    #sn2020oi(ax, vals.sn_col, 'SNe')
    ptf11qcj(ax, vals.sn_col, 'SN')
    #sn2008d(ax, vals.sn_col, 'SNe')

    # Category: TDEs
    j1644(ax, vals.tde_col, legend='TDE')
    #igr(ax, vals.tde_col, legend=None)
    #at2022cmc(ax, vals.tde_col, legend=None)

    # First category: long-duration GRBs
    grb130427A(ax, vals.lgrb_col, legend='LGRB')
    grb181201A(ax, vals.lgrb_col, None)
    grb161219B(ax, vals.lgrb_col, None)

    # Second category: low-luminosity GRBs
    sn1998bw(ax, vals.llgrb_col, legend='LLGRB')
    sn2017iuk(ax, vals.llgrb_col, None)

    # Third category: Cow-like
    at2018cow(ax, vals.fbot_col, 'LFBOT')
    at2020xnd(ax, vals.fbot_col, None)
    at2022tsd(ax, vals.fbot_col, None)
    at2024wpp(ax, vals.fbot_col, None)
    # New FBOTs for Cassie
    at2022abfc(ax, vals.colors['AT2022abfc'],None)
    at2023fhn(ax, vals.colors['AT2023fhn'],None)
    at2023vth(ax, vals.colors['AT2023vth'],None)
    at2024qfm(ax, vals.colors['AT2024qfm'])
    at2024aehp(ax, vals.colors['AT2024aehp'])

    

    ax.legend(fontsize=8, loc='lower left',handletextpad=0.1)

fig, ax = plt.subplots(1, 1, figsize=(6,4.5), sharex=True, sharey=True)
run(ax)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([4, 195])
ax.set_ylim([6e38, 2e43])
ax.set_ylabel(
        r"Luminosity $\nu L_{\nu}$ [erg s$^{-1}$] (100 GHz)",
        fontsize=15)
ax.set_xlabel("$t_\mathrm{rest}$ (d)", fontsize=16)

ax.legend()
ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("figures/fig6_mm_collage.pdf", dpi=450)
plt.show()
plt.close()