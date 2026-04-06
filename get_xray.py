""" Get the X-ray data of AT2022tsd """
import pandas as pd
import numpy as np
from astropy.io import fits as pyfits
from astropy.time import Time
import vals

ddir = "data/xray"


def load_swift_counts():
    """ load the unbinned counts """
    df = pd.read_table(
            ddir+"/AT2022tsd_XRT_unbinned.qdp", delimiter='\t',
            skiprows=np.hstack((np.arange(0,14),np.array([25,26]))))
    return df


def load_swift():
    """ Load the full counts, flux, luminosity 
    Sort by MJD """
    df = load_swift_counts()

    # Convert counts to flux
    conv = 5.10E-11 # From the XRT spectrum fit
    df['Flux'] = df['Rate    ']*conv
    df['Fluxpos'] = df['Ratepos ']*conv
    df['Fluxneg'] = df['Rateneg']*conv

    # Now the luminosity
    df['L'] =  df['Flux']*4*np.pi*(vals.dL_cm)**2
    df['Lpos'] =  df['Fluxpos']*4*np.pi*(vals.dL_cm)**2
    df['Lneg'] =  df['Fluxneg']*4*np.pi*(vals.dL_cm)**2

    df = df.sort_values('!MJD    ', ignore_index=True)

    return df


def load_chandra():
    df = pd.read_csv(ddir+"/chandra_flux_summary.txt", delimiter=',')
    # Convert date to MJD
    df['MJD'] = Time(df['Start'].values.astype(str), format='isot').mjd

    # Luminosity
    df['L'] = df['Flux']*4*np.pi*(vals.dL_cm)**2
    df['Lpos'] = df['uFlux']*4*np.pi*(vals.dL_cm)**2
    df['Lneg'] = df['lFlux']*4*np.pi*(vals.dL_cm)**2

    # Sort
    df = df.sort_values('MJD', ignore_index=True)

    return df


def get_exp(i):
    """ Get the exposure time for a given obs ID """
    df = pd.read_table(
            ddir+"/swift_obs_log.txt", delimiter='|',
            skiprows=(0,1))
    obsid = df['obsid      ']
    exp = df['exposure  ']
    ops = df['operation_mode']
    point = df['pointing_mode']

    # Get the exposure times on-source
    choose_obsid = np.array([int(str(o)[-2:])==i for o in obsid])
    choose_mode = point=='pointing     '
    choose = np.logical_and(choose_obsid, choose_mode)

    return sum(exp[choose])


def load_chandra_flares(oid):
    """ Get the X-ray flare LC from Chandra 

    Parameters:
    ----------
    oid: obs ID as a string

    There are seven OIDs so far:
    '26641', '26642', '26643', '26644', '27639', '27643', '26645'
    """
    dd = "data/xray" 
    ff = dd + "/" + oid + "/repro/xray_flare_lc.txt"
    dat = np.loadtxt(ff)
    x = dat[:,0]/86400 # in days
    y = dat[:,1]
    xerr = 250/86400
    yerr = dat[:,2]

    # Get the t0 in MJD and MET
    head = pyfits.open(dd + "/%s/repro/acis_bary_evt2.fits" %(oid))[0].header
    t0_mjd = Time(head['DATE-OBS'], format='isot').mjd
    t = t0_mjd+x

    return t,x,y,xerr,yerr


def load_both():
    # Load the Swift data
    df = load_swift()
    t = Time(df['!MJD    '].values, format='mjd')
    dt = (t.mjd-vals.t0['AT2022tsd'])/(1+vals.z)
    print(dt)
    et = (df['T_+ve   '].values)/(1+vals.z)
    L = df['L'].values/1E43
    eL = df['Lpos'].values/1E43

    # Plot the Swift data
    isdet = eL>0

    # Get Chandra
    # WebPIMMS: the factor for going from 0.5-6 to 0.3-10 is 1.77
    factor = 1.77
    df = load_chandra()
    t = Time(df['MJD'].values, format='mjd')
    exp = (df['Exp'].values*1000)/86400 # days
    dt_start = (t.mjd-vals.t0['AT2022tsd'])/(1+vals.z)
    dt_dur = exp/(1+vals.z)
    dt_ch = dt_start + dt_dur/2 # halfway through
    e_dt_ch = dt_dur/2 # half the exposure
    L_ch = 1.77*df['L'].values/1E43
    uL_ch = 1.77*df['Lpos'].values/1E43
    lL_ch = 1.77*df['Lneg'].values/1E43

    # Concatenate
    dt = np.hstack((dt[isdet], dt_ch))
    L = np.hstack((L[isdet], L_ch))
    eL = np.hstack((eL[isdet], uL_ch))
    e_dt = np.hstack((et[isdet], e_dt_ch))
    print(dt)
    return dt, L*1E43, e_dt, eL*1E43