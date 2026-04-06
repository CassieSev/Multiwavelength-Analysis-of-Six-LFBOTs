""" Get the radio data from SMA, ATCA, ALMA """

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def get_data_all():
    """ 
    Get the full set of data: 

    Returns
    -------
    tel: array telling you which telescope the data point is from
    freq: freq of the data point
    days: day of the data point, from time of explosion
    flux: flux of the data point
    eflux: estimate of uncertainty in flux of the data point
    """
    data_dir = "../data"
    dat = Table.read(
        "radio_lc.dat", delimiter="&", format='ascii.no_header')
    days = np.array(dat['col1'])
    tel = np.array(dat['col2'])
    freq = np.array(dat['col4']).astype(float)
    flux_raw = np.array(dat['col5'])
    flux = np.zeros(len(flux_raw))
    eflux= np.zeros(len(flux_raw))
    islim = np.zeros(len(flux_raw), dtype=bool)

    # Now, go through and get the appropriate uncertainties
    for ii,val in enumerate(flux_raw):
        # Identify an upper limit
        if '<' in val:
            islim[ii] = True
            flux[ii] = float(val[2:].split('$')[0])
        else:
            flux[ii] = float(val.split("pm")[0][1:])
            eflux[ii] = float(val.split("pm")[1][0:-1])
    return islim, tel, freq, days, flux, eflux


def get_spectrum(day):
    """ 
    Get full for a specific day
    For each frequency band, you will have to interpolate.
    """

    # First, get *all* of the data
    islim, tel, freq, days, flux, eflux_form, eflux_sys = get_data_all()
    print(day)

    # Don't generate eflux_tot yet,
    # because the right thing to do depends on the telescope

    # The way we interpolate depends on the telescope.
    choose_tel = np.logical_and(tel == 'ATCA', eflux_form > 0)
    ufreq = np.unique(freq[choose_tel])
    nus = [] # which frequencies to keep
    ms = [] # power-law fit
    bs = [] # power-law fit
    
    for val in ufreq:
        # fit a power law in log-log space
        choose = np.logical_and(choose_tel, freq==val)
        # no upper limits

        # only bother if there's >1 point at this frequency
        if sum(choose) > 1:
            nus.append(val)
            # if you allow weights, then the 34 GHz points look really weird
            m,b = np.polyfit(
                    np.log10(days[choose]), np.log10(flux[choose]), deg=1)#,
                    #w=1/eflux_tot[choose]**2)
            ms.append(m)
            bs.append(b)
    
    # Now, interpolate for that day to get the spectrum
    spec = []
    for ii,val in enumerate(nus):
        # Get light curve for that day
        spec.append(10**(ms[ii] * np.log10(day) + bs[ii]))

    # OK, "nus" and "spec" have been updated for the ATCA band!

    # Next: SMA.
    # This is much more complicated, since the emission is not self-absorbed
    # and the frequencies are not consistent.
    
    # I think the right thing to do is to interpolate the data
    # for each frequency...

    # The frequencies with by far the most data are 215.5 GHz and 231.5 GHz
    # And the one with definitively the most data is 231.5

    # So I think the right thing to do is show the 231.5 GHz light curve.
    # For every day that wasn't observed with the 231.5 GHz receiver,
    # interpolate the spectrum if you can

    # the higher frequencies are more difficult.
    # on each day, there are at least two measurements
    # that span 345 GHz.
    # so, on each day interpolate to estimate the flux at 345 GHz.
    tel, freq, days, flux, eflux_form, eflux_sys = get_data_all()
    eflux_tot = np.sqrt(eflux_form**2 + eflux_sys**2)

    choose_tel = tel == 'SMA' 
    d = [] # day
    f_lo = [] # interpolated flux at 231.5 GHz
    f_hi = [] # interpolated flux at 345 GHz
    uday = np.unique(days)

    for ii,val in enumerate(uday):
        choose = np.logical_and(choose_tel, days==val)
        success = False # at least one point added

        # lower bands
        keep_flux = flux[choose][freq[choose] == 231.5]
        if len(keep_flux) == 1:
            # Then you can save the value directly
            success = True
            f_lo.append(keep_flux[0])
        elif len(keep_flux) > 1:
            print("more than one point on a given day?")

        else:
            # Then you can try to interpolate
            try:
                xfit = freq[choose]
                yfit = flux[choose]
                order = np.argsort(xfit)
                out = np.interp(231.5, xfit[order], yfit[order])
                f_lo.append(out)
                success = True
            except:
                # don't extrapolate
                pass

        # upper bands
        keep_flux = flux[choose][freq[choose] == 345]
        if len(keep_flux) == 1:
            # Then you can save the value directly
            f_hi.append(keep_flux[0])
            success = True
        else:
            # Then you can try to interpolate
            try:
                xfit = freq[choose]
                yfit = flux[choose]
                order = np.argsort(xfit)
                out = np.interp(345, xfit[order], yfit[order])
                f_hi.append(out)
                success = True
            except:
                # don't extrapolate
                pass
        if success:
            d.append(val)

    # Now, interpolate for that day to get the two-point spectrum
    nus.append(231.5)
    d = np.array(d)
    f_lo = np.array(f_lo)
    f_hi = np.array(f_hi)
    order = np.argsort(d)
    spec.append(np.interp(day, d[order], f_lo[order]))
    nus.append(345)
    spec.append(np.interp(day, d[order], f_hi[order]))

    # Finally: ALMA
    # For any ALMA points, you should just return the value on the day
    # if it exists.
    # Nothing to interpolate.
    choose = np.logical_and(days == day, tel == 'ALMA')
    for ii,nuval in enumerate(freq[choose]):
        nus.append(nuval)
        spec.append(flux[choose][ii])

    return np.array(nus), np.array(spec)
