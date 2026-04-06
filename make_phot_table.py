import astropy.units as u
import pandas as pd
from astropy.cosmology import Planck18
from astropy.time import Time
import vals
import numpy as np
import matplotlib.pyplot as plt
from fig1_photometry import objects, redshifts, b_v, get_photo, filt_list, color_list, marker_list, t0, ext_corr

for i, object in enumerate(objects):
    data=pd.read_csv('data/{}/{}_photometry.csv'.format(objects[i], objects[i]), sep=',')
    times, fluxes, flux_errs, flux_limits, color, limits, filters, instruments, mjds = get_photo(data, object)

    fluxes=ext_corr(fluxes, color, object=objects[i])
    flux_limits=ext_corr(flux_limits, color, object=objects[i])
    flux_write = np.where(limits, flux_limits, fluxes)

    flux_errs=np.where(limits, '--', flux_errs)
    info = list(zip(times, instruments, filters, flux_write, flux_errs, limits, mjds))
    with open('data/{}/{}_photometry_latex.csv'.format(objects[i], objects[i]), 'w') as f:
        for line in info:
            time, instrument, filter, flux_write, flux_err, limit, mjd = line
            if time >= -6:
                time=int(np.round(time, 0))
                utc = Time(mjd, format='mjd').to_value('iso', subfmt='date_hm').replace("-", "")
                if not limit:
                    flux_err = np.round(float(flux_err), 2)
                flux_write = np.round(flux_write, 2)
                for filt in filt_list:
                    if filter.endswith(filt):
                        filter = filt
                        break
                if instrument == 'ZTF':
                    instrument = 'P48/ZTF'
                #elif instrument == 'P48//ZTF':
                #    instrument = 'P48/ZTF'
                elif instrument == 'UVOT':
                    instrument = 'SWIFT/UVOT'
                elif instrument == 'IOO':
                    instrument = 'LT/IO:O'
                elif instrument == 'SEDM':
                    instrument = 'P60/SEDM'
                elif instrument == 'Deveny+LMI':
                    instrument = 'Deveny/LMI'

                
                if limit:
                    f.write(f'{object}&{utc}&{time}&{instrument}&{{\\em {filter}}}&$>{flux_write}$\\\\ \n')
                else:
                    f.write(f'{object}&{utc}&{time}&{instrument}&{{\\em {filter}}}&${flux_write}\\pm{flux_err}$\\\\ \n')


