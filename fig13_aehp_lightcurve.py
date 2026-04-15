import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy import units as u
from extinction import fm07
from scipy.optimize import curve_fit
import vals

i_color='#ffa900'
redshift=0.1715
filter_wave=np.array([1927, 2246, 2600, 3465, 4392, 4671.78, 6141.12, 7457.89, 8922.78])
color_list=['#9500ff','#6100ff', '#2e00ff', '#007fff', '#00bfff', 'green', 'red', i_color, 'black']
filt_list=['uvw2', 'uvm2', 'uvw1', 'u', 'b', 'g', 'r', 'i', 'z']
marker_list=['s', 'o', 'D', 'P', 'X', 's', 'o', 'D', 'P' ]

def get_photo(df, timecol='mjd', errcol='magerr', fluxcol='mag', limitcol='limiting_mag', t0=0):
    df=df.loc[df['instrument_name']!='ATLAS']
    filters=pd.Series(df['filter'])

    for filt, color in zip(filt_list, color_list):
        df.loc[filters.str.endswith(filt)==True, 'color'] = color

    limit = np.isnan(df[fluxcol])

    time = df[timecol] - t0
    flux = df[fluxcol]
    fluxerr=df[errcol]
    flux_limit=df[limitcol]
    colors = df['color']

    return time, flux, fluxerr, flux_limit, colors, limit

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


def get_at2018(filepath='data/AT2018cow/at2018cow_photometry_table.txt', filter='g'):
    extinctions=fm07(filter_wave, a_v=3.1*0.0759)
    at2018data = pd.read_csv(filepath, sep='\s+')
    at2018data['t0']= at2018data['MJD']-58285.44141-1.73
    fluxes = pd.Series(at2018data['ABMag'])
    filtered_data = at2018data.loc[(at2018data['Filt']==filter)&(fluxes.str.startswith('>')==False),:]
    filter_keys={'g':0, 'r':1, 'i':2, 'z':3}
    filtered_data.loc[:,'ext_flux']=filtered_data.loc[:,'ABMag'].astype(float)-extinctions[filter_keys[filter]]
    filtered_data.loc[:,'abs_flux'] = app_to_abs_mag(filtered_data.loc[:,'ext_flux'],0.0141,kcorrection=True)

    return filtered_data['t0'], filtered_data['ext_flux'], filtered_data['abs_flux'], filtered_data['Emag'].astype(float)

def ext_corr(flux, colors, object=None):
    corrections=np.zeros_like(flux)
    if object != None:
        extinctions=fm07(filter_wave, a_v=3.1*vals.b_v[object])
        for extinction, color in zip(extinctions, ['green', 'red', i_color, 'black']):
            corrections[colors==color]=extinction
    return flux-corrections



at2018_t0, at2018flux_app, at2018flux_abs, _ = get_at2018()
at2018cow_limit={'g':(58284.1300-58285.44141,
                      app_to_abs_mag(18.9, 0.0141, kcorrection=True, unitless=True))}


data=pd.read_csv('data/AT2024aehp/AT2024aehp_photometry.csv', sep=',')
x, flux, fluxerr, flux_limit, color, limit,= get_photo(data, t0=60663.35732640)
abs_flux=app_to_abs_mag(flux, redshift, kcorrection=True)
abs_flux_limit = app_to_abs_mag(flux_limit, redshift, kcorrection=True)
x=x/1.1715
# Fit to power law
def power_law(times, t0, m0):
    return 2.5*(5/3)*np.log10(times+t0) + m0

#flux=ext_corr(flux, color, object=objects[i])
popt, _ = curve_fit(power_law, x[(color=='green') & (~limit)], abs_flux[(color=='green')& (~limit)], p0=[3,-18], bounds=([-np.inf, -np.inf], [7, np.inf]))
popt2, _ = curve_fit(power_law, x[(color=='green') & (~limit)], abs_flux[(color=='green')& (~limit)], p0=[3,-18])
print(popt2)

# Upper Limits
fig=plt.figure(figsize=(6, 4))
ax1=fig.add_subplot(1,1,1)

#ax1.scatter(x[(limit)*(color=='green')], abs_flux_limit[limit], color=color[limit],)
#                   marker='v', s=50, alpha=0.4)

ax1.errorbar(x[color=='green'], abs_flux[color=='green'],yerr=fluxerr[color=='green'],marker='o',
                            ls='none', color=vals.colors['AT2024aehp'], capsize=3, alpha=0.5, markersize=6, label='AT2024aehp')
#ax1.scatter(x[color=='red'], abs_flux[color=='red'], color=color[color=='red'], marker='s', alpha=0.5, s=40, label='r')
#ax1.scatter(x[color=='green'], abs_flux[color=='green'], color=color[color=='green'], marker='o', alpha=0.5, s=50, label='AT2024aehp')
#ax1.scatter(x[color==i_color], abs_flux[color==i_color], color=color[color==i_color], marker='D', alpha=0.5, s=50, label='i')



offset=1
ax1.plot(at2018_t0/1.014, at2018flux_abs+offset, color='black', label='AT2018cow + 1', lw=5, alpha=0.2)
ax1.plot([at2018cow_limit['g'][0], list(at2018_t0)[0]], [at2018cow_limit['g'][1], list(at2018flux_abs)[0]+offset],
                linestyle='dashed', color='black',lw=5, alpha=0.2)

ax1.set_xlabel('$t_{rest}$ (d)', fontsize=14)
ax1.set_ylabel('$M_g$', fontsize=18)
#ax1.set_xscale('log')
ax1.set_xlim([-2.1,22]) 
ax1.set_ylim([-16.8, -21.1])

mrf_t=np.array([0.029, 3.39, 6.1, 11.2, 14.2, 16.7, 22.1, 25.6, 26.5, 27.3, 33.6])/1.1353-2.96
mrf_abs=np.array([-18.38, -20, -19.63, -18.66, -18.01, -18.27, -17.78, -17.22, -17.02, -17.33, -17.09])
mrf_upp=np.array([-18.48, -20.03, -19.65, -18.74, -18.22, -18.52, -18.19, -17.62, -17.37, -17.59, -17.49])

mrf_err=mrf_abs-mrf_upp
#ax1.errorbar(mrf_t, mrf_abs,yerr=mrf_err,marker='o',
#                            ls='none', color='red', ecolor='red', capsize=3, alpha=0.5, markersize=6, label='AT2020mrf')

# Show 5/3 decay!
# Choose green det to be start
t0 = x[(color=='green') & (~limit)].iloc[6]
tf = x[(color=='green') & (~limit)].iloc[-1]
M0 = abs_flux[(color=='green') & (~limit)].iloc[6]
t = np.linspace(x[(color=='green') & (~limit)].iloc[0],tf,1000)
M_t = (2.5*(5/3)*np.log10(t+popt[0])) + popt[1]
M_t2 = (2.5*(5/3)*np.log10(t+popt2[0])) + popt2[1]

ax1.plot(t,M_t,label=r"$t^{-\frac{5}{3}}$ decay")
ax1.set_yticks([-17, -18, -19, -20, -21])

handles, labels = plt.gca().get_legend_handles_labels()
order=[2,0,1]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize='10')
plt.tight_layout()
plt.savefig('figures/fig13_aehp_lc.pdf', dpi=450)
plt.show()
plt.close()



