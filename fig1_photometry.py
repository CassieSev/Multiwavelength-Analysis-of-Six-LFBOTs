import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy import units as u
import vals
from extinction import fm07

# Setting Color for i-band, listing objects and the t0
i_color='#ffa900'
objects =['AT2022abfc', 'AT2023fhn', 'AT2023hkw', 'AT2023vth',  'AT2024qfm', 'AT2024aehp']
##t0={'AT2023fhn': 60044.204, 'AT2023vth': 60235.116, 'AT2023hkw': 60065.199,
#    'AT2022abfc': 59904.344, 'AT2021ahuo': 59352.4248, 'AT2018cow': 58285.44141,
#    'AT2024aehp': 60663.35732640, 'AT2024qfm': 60518.34636569} # Discoveries
t0={'AT2023fhn': 60046.224062500100, 'AT2023vth': 60237.13100690020, 'AT2023hkw': 60064.172824100100,
    'AT2022abfc': 59904.34430560000, 'AT2021ahuo': 59352.4248, 'AT2018cow': 58287.1500,
    'AT2024aehp': 60663.41862270000, 'AT2024qfm': 60518.3463657} # Peak Magnitude
b_v=vals.b_v
# First SWIFT uvw2, uvm2, uvw1, u, then b wavelengths from https://www.swift.ac.uk/analysis/uvot/filters.php
# then SDSS-g, r, then i, and z  obtained from SVO Filter Service
filter_wave=np.array([1927, 2246, 2600, 3465, 4392, 4671.78, 6141.12, 7457.89, 8922.78])
color_list=['#9500ff','#6100ff', '#2e00ff', '#007fff', '#00bfff', 'green', 'red', i_color, 'black']
filt_list=['uvw2', 'uvm2', 'uvw1', 'u', 'b', 'g', 'r', 'i', 'z']
marker_list=['s', 'o', 'D', 'P', 'X', 's', 'o', 'D', 'P' ]


#extinctions = {'AT2023fhn': (0.096, 0.067, 0.05, 0.037), 'AT2023vth': (0.682, 0.472, 0.35, 0.261),
#             'AT2023hkw': (0.043, 0.03, 0.022, 0.016), 'AT2022abfc': (0.115, 0.08, 0.059, 0.044),
#               'AT2021ahuo': (0.246, 0.17, 0.127, 0.094), 'AT2018cow': (0.287, 0.198, 0.147, 0.11),
#               'AT2024qfm':(0.162, 0.112, 0.083, 0.062), 'AT2024aehp':(0.095, 0.066, 0.049, 0.036, 0.104, 0.125)}
redshifts=vals.redshifts


def app_to_abs_mag(apparent, redshift, kcorrection=False, unitless=False):
    """
    Converts apparent to absolute magnitude using the Planck18 cosmology.
    Args:
        apparent: apparent magnitude
        redshift: redshift
        kcorrection: if true, applies an approximate cosmological k-correction
            using Eq. 2 from Whitesides et al (2017) ApJ 851 107
    """
    if kcorrection:
        apparent += 2.5*np.log10(1+redshift) #
    if unitless:
        return (apparent*u.mag-Planck18.distmod(redshift))/u.mag
    return apparent*u.mag-Planck18.distmod(redshift)


def abs_to_app_mag(absolute, redshift, kcorrection=False, unitless=False):
    """
    Converts absolute to apparent magnitude using the Planck18 cosmology.
    Args:
        absolute: absolute magnitude
        redshift: redshift
        kcorrection: if true, applies an approximate cosmological k-correction
            using Eq. 2 from Whitesides et al (2017) ApJ 851 107
    """
    if kcorrection:
        absolute -= 2.5*np.log10(1+redshift) 
    if unitless:
        return (absolute*u.mag+Planck18.distmod(redshift))/u.mag
    return absolute*u.mag+Planck18.distmod(redshift)

def ext_corr(flux, colors, object=None):
    corrections=np.zeros_like(flux)
    if object != None:
        extinctions=fm07(filter_wave, a_v=3.1*b_v[object])
        for extinction, color in zip(extinctions, color_list):
            corrections[colors==color]=extinction
    return flux-corrections
    

def get_at2018(filepath='data/AT2018cow/at2018cow_photometry_table.txt', filter='g'):
    """
    Get the at2018cow lightcurve.  Pick one filter to plot the lightcurve for.
    Returns lists of the time after peak, the apparent mag in `ext_flux`, the absolute mag in `abs_flux`, and
    the magnitude error in `Emag`.
    """
    extinctions=fm07(filter_wave, a_v=3.1*b_v['AT2018cow'])
    at2018data = pd.read_csv(filepath, sep='\s+')
    at2018data['t0']= (at2018data['MJD']-t0['AT2018cow'])/(1+redshifts['AT2018cow'])
    fluxes = pd.Series(at2018data['ABMag'])
    # Remove non-detections
    filtered_data = at2018data.loc[(at2018data['Filt']==filter)&(fluxes.str.startswith('>')==False),:]
    # Set the keys for obtaining the proper extinction from the `extinctions`` list
    filter_keys={'g':5, 'r':6, 'i':7, 'z':8}
    filtered_data.loc[:,'ext_flux']=filtered_data['ABMag'].astype(float)-extinctions[filter_keys[filter]]
    filtered_data.loc[:,'abs_flux'] = app_to_abs_mag(filtered_data['ext_flux'],redshifts['AT2018cow'],kcorrection=True)

    return filtered_data['t0'], filtered_data['ext_flux'], filtered_data['abs_flux'], filtered_data['Emag'].astype(float)
    
# Define a line to constrain the rise of AT2018cow's lightcurve from the latest non-detection
at2018cow_limit={'g':(58284.1300-t0['AT2018cow'],
                      app_to_abs_mag(18.9, redshifts['AT2018cow'], kcorrection=True, unitless=True))}


def get_photo(df, object, timecol='mjd', errcol='magerr', fluxcol='mag', limitcol='limiting_mag', filtcol='filter', fluxunccol='fluxerr', rest=True):
    """
    Retrieves photometry from a dataframe.  The keyword arguments state the column names for the mag, limiting mag, mag error, and time.  t0 is the
    transient's t0 in MJD.  Also outputs a color string for plotting, where each entry is assigned a color based on the filter.
    """
    filters=pd.Series(df[filtcol])
    
    # utilizes the global filt_list and color_list to associate each filter name with a given color, where
    # each band has a unique color.
    for filt, color in zip(filt_list, color_list):
        df.loc[filters.str.endswith(filt)==True, 'color'] = color

    df = df.loc[filters.str.startswith('atlas')==False,:].sort_values(timecol)
    filters=pd.Series(df[filtcol])
    limit = np.isnan(df[fluxcol])

    if rest:
        time = (df[timecol] - t0[object])/(1+redshifts[object])
    else:
        time = (df[timecol] - t0[object])

    if object != 'AT2024qfm':
        df['zp'] = df[limitcol]+2.5*np.log10(5*df[fluxunccol])
        df['limiting_col_3sigma']=df['zp']-2.5*np.log10(3*df[fluxunccol])
        df[limitcol]=np.where(filters.str.startswith('ztf'), df['limiting_col_3sigma'], df[limitcol])
        if object == 'AT2023hkw':
            df[limitcol]=np.where(df['instrument_name']=='GITCamera', df['limiting_col_3sigma'], df[limitcol])
    flux = df[fluxcol]
    fluxerr=df[errcol]
    flux_limit=df[limitcol]
    colors = df['color']
    filters=pd.Series(df[filtcol])
    instruments=df['instrument_name']
    mjds=df['mjd']

    return time, flux, fluxerr, flux_limit, colors, limit, filters, instruments, mjds


def plot_spec(ax, object):
    """
    Plots vertical lines when spectra were taken.
    """
    t_specs={'AT2022abfc': [10], 'AT2023hkw': [11], 'AT2023fhn': [9,10,16],
             'AT2023vth': [4,25], 'AT2024qfm': [6,13,35,37], 'AT2024aehp':[5, 13, 19]}
    t_spec_list = t_specs[object]
    ax.vlines(t_spec_list, -24, -15, linestyles='dotted', colors='gray', linewidths=2)



def make_fig():
    # Make the figure
    fig, axs = plt.subplots(3,2, figsize=(8,8), layout='constrained')
    flat_axs=axs.flatten()
    # Get the distance modulus to convert from apparent to absolute magnitude.  Unfortunately,
    # need to manually define a function for each of these
    distmods={'AT2022abfc': 39.95919179, 'AT2023hkw': 41.0140787, 'AT2023fhn': 40.23758014, 'AT2023vth': 37.63855381,
              'AT2024aehp': 39.65635114, 'AT2024qfm': 40.33466909}
    
    for i, object in enumerate(objects):
        # read data
        data=pd.read_csv('data/{}/{}_photometry.csv'.format(objects[i], objects[i]), sep=',')
        x, flux, fluxerr, flux_limit, color, limit, _, _, _ = get_photo(data, object)

        # Correct for extinction, use the color list to identify what band each measurement is in
        flux=ext_corr(flux, color, object=objects[i])
        flux_limit=ext_corr(flux_limit, color, object=objects[i])

        # Get absolute magnitudes for detections + non-detections
        abs_flux = app_to_abs_mag(flux, redshifts[objects[i]], kcorrection=True)
        abs_flux_limit = app_to_abs_mag(flux_limit, redshifts[objects[i]], kcorrection=True)

        # Put labels for legend for one subplot only
        if i == 2: labels=filt_list
        else: labels=[None]*len(filt_list)
        # Upper Limits
        flat_axs[i].scatter(x[limit], abs_flux_limit[limit], color=color[limit],
                   marker='v', s=15, alpha=0.15)
        
        # Detections
        for j, _color in enumerate(color_list):
            flat_axs[i].scatter(x[color==_color], abs_flux[color==_color], color=_color,
                                marker=marker_list[j], s=15, label=labels[j])

        #Errors
        flat_axs[i].errorbar(x[~limit], abs_flux[~limit], yerr=fluxerr[~limit],
                    ecolor=color[~limit], elinewidth=1, fmt='none', alpha=1)

        # Plot AT2018cow light curve
        if i ==0:
            flat_axs[i].plot(at2018_t0, at2018flux_abs, color='black', label='AT2018cow g-band', lw=5, alpha=0.2)
        else:
            flat_axs[i].plot(at2018_t0, at2018flux_abs, color='black', lw=5, alpha=0.2)

        flat_axs[i].plot([at2018cow_limit['g'][0], list(at2018_t0)[0]], [at2018cow_limit['g'][1], list(at2018flux_abs)[0]],
                linestyle='dashed', color='black',lw=5, alpha=0.2)
        
        flat_axs[i].set_ylim([-16.5,-22.3])
        
        plot_spec(flat_axs[i], object)
        
        # Define the transformation to create the apparent magnitude y-axis labels on the right of each plot.
        if object == 'AT2022abfc':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+39.95919179, lambda x: x-39.95919179)) 
        elif object=='AT2023hkw':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+41.0140787, lambda x: x-41.0140787)) 
        elif object=='AT2023vth':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+40.23758014, lambda x: x-40.23758014))
        elif object=='AT2023fhn':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+37.63855381, lambda x: x-37.63855381))
        elif object=='AT2024qfm':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+40.33466909, lambda x: x-40.33466909))
        elif object=='AT2024aehp':
            sec_ax=flat_axs[i].secondary_yaxis('right', functions=(lambda x: x+39.65635114, lambda x: x-39.65635114))
        if i == 3:
            sec_ax.set_ylabel('Apparent AB Mag', fontsize=12)
        
        if i == 2:
            flat_axs[i].set_ylabel('Absolute AB Mag', fontsize=12)
        
          
        
        flat_axs[i].set_xlim([-4.5,52]) 
        flat_axs[i].text(0.98, 0.95, "{}".format(objects[i]), fontsize=9, ha='right', va='top', transform=flat_axs[i].transAxes)

        

    for ax in flat_axs:
        ax.set_xlabel('$t_{\mathrm{rest}}$ (d)', fontsize=14)
        ax.label_outer()
    fig.legend(loc='outside lower center',  fontsize='12', ncol=5)
    fig.legend(bbox_to_anchor=(0.1, -0.1), loc='upper left')
    plt.savefig('figures/fig1_optical.pdf', dpi=450)
    plt.show()
    plt.close()



if __name__=='__main__':
    at2018_t0, at2018flux_app, at2018flux_abs, _ = get_at2018()
    make_fig()
#make_app()