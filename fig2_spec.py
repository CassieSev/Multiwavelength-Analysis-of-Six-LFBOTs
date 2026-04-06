
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import vals

# Load all the spectra!
data_2023fhn_p200=pd.read_csv('data/AT2023fhn/AT2023fhn_p200.txt', sep='\s+')
data_2023fhn_gmos=pd.read_csv('data/AT2023fhn/AT2023fhn_GMOS.txt', sep='\s+')
data_2023fhn_lris=pd.read_csv('data/AT2023fhn/AT2023fhn_LRIS.txt',sep='\s+', header=166)

data_2022abfc=pd.read_csv('data/AT2022abfc/AT2022abfc_GMOS_stack.txt',sep='\s+')

data_2023hkw=pd.read_csv('data/AT2023hkw/AT2023hkw_Keck.txt', sep='\s+')

data_2023vth=pd.read_csv('data/AT2023vth/AT2023vth_GMOS.txt', sep='\s+')
data_2023vth_SEDM=pd.read_csv('data/AT2023vth/AT2023vth_SEDM.txt', sep='\s+', header=210)

data_2024qfm_gmos=pd.read_csv('data/AT2024qfm/AT2024qfm_GMOS.tsv', sep='\s+', header=0)
data_2024qfm_lris=pd.read_csv('data/AT2024qfm/AT2024qfm_lris.spec', sep='\s+', header=0)
data_2024qfm_lris_new=pd.read_csv('data/AT2024qfm/AT2024qfm_lris_new.spec', sep='\s+', header=0)
data_2024qfm_binospec=pd.read_csv('data/AT2024qfm/AT2024qfm_binospec.ascii', sep='\s+', header=46)

data_2024aehp=pd.read_csv('data/AT2024aehp/AT2024aehp_GMOS.txt', sep=',', header=0)
data_2024aehp_binospec=pd.read_csv('data/AT2024aehp/AT2024aehp_binospec.txt', sep='\s+', header=0)
data_2024aehp_kcwi=pd.read_csv('data/AT2024aehp/AT2024aehp_KCWI.txt', sep='\s+', header=0)

def fix_spectrum(wavelength, flux, fluxe, redshift, bin_size=3, scale=1, limits=None, var=False):
    """
    Get the wavelength, flux, flux error lists for each spectrum.  Specify a bin size for binning, and a scale
    to scale the y-axis amplitudes to aid the final plots.
    `limits` can put an upper and lower wavelength limit  on the spectrum
    `var` is True if the flux error column is the variance, False if it is the square root of the variance
    """
    redshifts=vals.redshifts
    if type(redshift) == str:
        z=redshifts[redshift]
    elif type(redshift) == float | type(redshift) == int:
        z=redshift
    # Cut out zero fluxes
    if limits != None:
        indices=~np.isnan(flux)&(flux!=0)&(wavelength>=limits[0])&(wavelength<=limits[1])
    else:
        indices=~np.isnan(flux)&(flux!=0)

    wavelength=wavelength[indices]
    flux=flux[indices]
    fluxe=fluxe[indices]

    x_binned = []
    y_binned = []
    ey_binned = []

    for _,x_val in enumerate(wavelength):
        choose = np.abs(wavelength-x_val)<bin_size
        # If only one entry in a bin, just copy values over
        if sum(choose)==1:
            x_binned.append(x_val)
            y_binned.append(flux[choose].iloc[0])
            if fluxe[choose].iloc[0] != np.nan:
                ey_binned.append(fluxe[choose].iloc[0])
        # Over the bin, properly average and get the new weighted error
        elif sum(choose)>1:
            if var:
                mean,wsum = np.average(
                    flux[choose], weights=1/fluxe[choose], returned=True)
            else:

                mean,wsum = np.average(
                        flux[choose], weights=1/fluxe[choose]**2, returned=True)
            efmean = np.sqrt(1/wsum)
            x_binned.append(np.average(wavelength[choose]))
            y_binned.append(mean)
            ey_binned.append(efmean)
    x_binned = np.array(x_binned)
    y_binned = np.array(y_binned)

    ey_binned = np.array(ey_binned)

    mean_flux_binned=np.average(y_binned, weights=1/ey_binned**2)
    
    rest_lambda_binned = x_binned/(1+z)
    norm_flux_binned=y_binned/mean_flux_binned
    # Get the mean flux to normalize all fluxes
    if var:
        mean_flux=np.average(flux, weights=1/fluxe)
    else:
        mean_flux=np.average(flux, weights=1/fluxe**2)
    rest_lambda = wavelength/(1+z)
    norm_flux=flux/mean_flux
    
    rest_lambda_binned*=scale
    rest_lambda*=scale

    return rest_lambda_binned, norm_flux_binned, rest_lambda, norm_flux

# Bin all the spectra
abfc_wb, abfc_fb, abfc_w, abfc_f = fix_spectrum(data_2022abfc['wavelength'], data_2022abfc['flux'], data_2022abfc['unc'], 'AT2022abfc', var=True)
fhn_w1b, fhn_f1b, fhn_w1, fhn_f1 = fix_spectrum(data_2023fhn_p200['wavelength'], data_2023fhn_p200['flux'], data_2023fhn_p200['fluxe'], 'AT2023fhn', scale=1.002) #P200 spectrum
fhn_w2b, fhn_f2b, fhn_w2, fhn_f2 = fix_spectrum(data_2023fhn_gmos['wavelength'], data_2023fhn_gmos['flux'], data_2023fhn_gmos['fluxe'], 'AT2023fhn', scale=1.002) #Gemini, featureless
fhn_wb, fhn_fb, fhn_w, fhn_f = fix_spectrum(data_2023fhn_lris['wavelength'], data_2023fhn_lris['flux'], data_2023fhn_lris['unc'], 'AT2023fhn', scale=1.002)
hkw_wb, hkw_fb, hkw_w, hkw_f=fix_spectrum(data_2023hkw['wavelength'], data_2023hkw['flux'], data_2023hkw['unc'], 'AT2023hkw')
vth_wb, vth_fb, vth_w, vth_f=fix_spectrum(data_2023vth['wavelength'], data_2023vth['flux'], data_2023vth['unc'], 'AT2023vth', var=True) # for 2 only
vth_w1b, vth_f1b, vth_w1, vth_f1=fix_spectrum(data_2023vth_SEDM['wavelength'], data_2023vth_SEDM['flux'], data_2023vth_SEDM['unc'], 'AT2023vth', var=True)

qfm_wb, qfm_fb, qfm_w, qfm_f=fix_spectrum(data_2024qfm_lris['wavelength'], data_2024qfm_lris['flux'], data_2024qfm_lris['unc'], 'AT2024qfm')
qfm_w1b, qfm_f1b, qfm_w1, qfm_f1=fix_spectrum(data_2024qfm_lris_new['wavelength'], data_2024qfm_lris_new['flux'], data_2024qfm_lris_new['unc'], 'AT2024qfm', var=True)
qfm_w2b, qfm_f2b, qfm_w2, qfm_f2=fix_spectrum(data_2024qfm_gmos['wavelength'], data_2024qfm_gmos['flux'], data_2024qfm_gmos['unc'], 'AT2024qfm', var=True)
qfm_w3b, qfm_f3b, qfm_w3, qfm_f3=fix_spectrum(data_2024qfm_binospec['wavelength'], data_2024qfm_binospec['flux'], data_2024qfm_binospec['unc'], 'AT2024qfm', var=False)

aehp_wb, aehp_fb, aehp_w, aehp_f=fix_spectrum(data_2024aehp['wavelength'], data_2024aehp['flux'],data_2024aehp['unc'], 'AT2024aehp', var=True, scale=1.0013)
aehp_w1b, aehp_f1b, aehp_w1, aehp_f1=fix_spectrum(data_2024aehp_kcwi['wavelength'], data_2024aehp_kcwi['flux'],data_2024aehp_kcwi['err'], 'AT2024aehp', var=False)
aehp_w2b, aehp_f2b, aehp_w2, aehp_f2=fix_spectrum(data_2024aehp_binospec['wavelength'], data_2024aehp_binospec['flux'],data_2024aehp_binospec['unc'], 'AT2024aehp', var=False)

fig=plt.figure(figsize=(6,8))
gs=gridspec.GridSpec(1,1)
mosaic_plot=False
if mosaic_plot:
    gs=gridspec.GridSpec(2,3,height_ratios=[4,1.2])
    ax1=fig.add_subplot(gs[1,0])
    ax1.plot(fhn_wb, fhn_fb, color='black', linewidth=0.7)

    ax1.axvspan(6713, 6718, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # S II ? for 2023fhn
    ax1.axvspan(6727, 6733, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # S II ? for 2023fhn

    ax1.set_title("AT2023fhn [S II]")
    ax1.set_xlim([6682, 6765])
    ax1.set_ylim([0, 2])
    ax1.set_yticks([])


    ax2=fig.add_subplot(gs[1,1])
    ax2.plot(fhn_wb, fhn_fb, color='black', linewidth=0.7)

    ax2.axvspan(6540, 6551, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # N II for fhn
    ax2.axvspan(6575, 6585, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # N II

    ax2.set_title("AT2023fhn [N II]")
    ax2.set_xlim([6505, 6615])
    ax2.set_ylim([0, 1.75])
    ax2.set_yticks([])


    ax3=fig.add_subplot(gs[1,2])
    ax3.plot(hkw_wb, hkw_fb, color='black', linewidth=0.7)

    ax3.axvspan(6545, 6550, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # N II for vth
    ax3.axvspan(6580, 6588, alpha=0.2, color='green', ymin=0.05, ymax=0.95) # N II


    ax3.set_title("AT2023hkw [N II]")
    ax3.set_xlim([6530, 6600])
    ax3.set_ylim([0.2, 4.4])
    ax3.set_yticks([])

ax_main=fig.add_subplot(gs[0,:])

# Plot the binned spectra as a line, and the unbinned as a shaded one
# Binned
ax_main.plot(abfc_wb, 3*abfc_fb+45, label='AT2022abfc GMOS', color='blue', linewidth=0.7)
ax_main.plot(vth_w1b, 3*vth_f1b+31, label='AT2023vth SEDM', color='purple', linewidth=0.7)
ax_main.plot(vth_wb, 1.35*vth_fb+22, label='AT2023vth GMOS', color='purple', linewidth=0.7)
ax_main.plot(hkw_wb, 2.5*hkw_fb+11, label='AT2023hkw Keck', color='brown', linewidth=0.7)
ax_main.plot(fhn_w2b, fhn_f2b+3, label='AT2023fhn GMOS', color=vals.colors['AT2023fhn'], linewidth=0.7)
ax_main.plot(fhn_w1b, fhn_f1b-6, label='AT2023fhn P200', color=vals.colors['AT2023fhn'], linewidth=0.7)
ax_main.plot(fhn_wb, fhn_fb-17, label='AT2023fhn LRIS', color=vals.colors['AT2023fhn'], linewidth=0.7)
ax_main.plot(qfm_w2b, 1.3*qfm_f2b-27, label='AT2024qfm GMOS', color=vals.colors['AT2024qfm'], linewidth=0.7)
ax_main.plot(qfm_wb, 4*qfm_fb-40, color=vals.colors['AT2024qfm'], linewidth=0.7)
ax_main.plot(qfm_w3b, 3*qfm_f3b-50, color=vals.colors['AT2024qfm'], label='AT2024qfm Binospec', linewidth=0.7)
ax_main.plot(qfm_w1b, 1.8*qfm_f1b-58, label='AT2024qfm LRIS', color=vals.colors['AT2024qfm'], linewidth=0.7)
ax_main.plot(aehp_wb, 1.4*aehp_fb-71, label='AT2024aehp GMOS', color=vals.colors['AT2024aehp'], linewidth=0.7)
ax_main.plot(aehp_w1b, 1.4*aehp_f1b-80, label='AT2024aehp KCWI', color=vals.colors['AT2024aehp'], linewidth=0.7)
ax_main.plot(aehp_w2b, 1.4*aehp_f2b-90, label='AT2024aehp Binospec', color=vals.colors['AT2024aehp'], linewidth=0.7)

# Unbinned
#ax_main.plot(abfc_w, 3*abfc_f+45, label=None, color='blue', alpha=0.3)
#ax_main.plot(vth_w1, 3*vth_f1+31, label=None, alpha=0.3, color='purple')
#ax_main.plot(vth_w, 1.8*vth_f+22, label=None, alpha=0.3, color='purple')
#ax_main.plot(hkw_w, 2.5*hkw_f+11, label=None, color='brown', alpha=0.3)
#ax_main.plot(fhn_w2, fhn_f2+3, label=None, color=vals.colors['AT2023fhn'], alpha=0.3)
#ax_main.plot(fhn_w1, fhn_f1-6, label=None, color=vals.colors['AT2023fhn'], alpha=0.3)
#ax_main.plot(fhn_w, fhn_f-17, label=None, color=vals.colors['AT2023fhn'], alpha=0.3)
#ax_main.plot(qfm_w2,  1.3*qfm_f2-27, label=None, color=vals.colors['AT2024qfm'], alpha=0.3)
#ax_main.plot(qfm_w, 4*qfm_f-40, label=None, color=vals.colors['AT2024qfm'], alpha=0.3)
#ax_main.plot(qfm_w3,  3*qfm_f3-50, label=None, color=vals.colors['AT2024qfm'], alpha=0.3)
#ax_main.plot(qfm_w1, 1.8*qfm_f1-58, label=None, color=vals.colors['AT2024qfm'], alpha=0.3)
#ax_main.plot(aehp_w, 1.4*aehp_f-71, label=None, color=vals.colors['AT2024aehp'], alpha=0.3)
#ax_main.plot(aehp_w1, 1.4*aehp_f1-80, label=None, color=vals.colors['AT2024aehp'], alpha=0.3)
#ax_main.plot(aehp_w2, 1.4*aehp_f2-90, label=None, color=vals.colors['AT2024aehp'], alpha=0.3)

# Make shaded green boxes
ax_main.axvspan(3930, 3975, alpha=0.2, color='green', ymin=0.90, ymax=0.97) # Ca H&K
ax_main.axvspan(6550, 6572, alpha=0.2, color='green', ymin=0.9, ymax=0.95) # H alpha
ax_main.axvspan(6550, 6572, alpha=0.2, color='green', ymin=0.02, ymax=0.83) # H alpha
ax_main.axvspan(6579, 6593, alpha=0.2, color='green', ymin=0.89, ymax=0.94) # [N II]
ax_main.axvspan(6579, 6593, alpha=0.2, color='green', ymin=0.02, ymax=0.83) # [N II]

ax_main.axvspan(6713, 6718, alpha=0.2, color='green', ymin=0.74, ymax=0.80) # [S II] 
ax_main.axvspan(6733, 6738, alpha=0.2, color='green', ymin=0.74, ymax=0.80)
ax_main.axvspan(6713, 6718, alpha=0.2, color='green', ymin=0.035, ymax=0.47)
ax_main.axvspan(6733, 6738, alpha=0.2, color='green', ymin=0.035, ymax=0.47)
ax_main.axvspan(4855, 4868, alpha=0.2, color='green', ymin=0.035, ymax=0.43) # H beta
ax_main.axvspan(4995, 5015, alpha=0.2, color='green', ymin=0.035, ymax=0.43) # [O III]
ax_main.axvspan(3725, 3735, alpha=0.2, color='green', ymin=0.035, ymax=0.57) # [O II]

# Label the visible lines
ax_main.text(6656, 55.2, "H$\\alpha$ & [N II]",va='center', ha='center', fontsize=9)
ax_main.text(4081, 57.4, "Ca H&K",va='center', ha='center', fontsize=9)
ax_main.text(6730, 30, "[S II]",va='center', ha='center', fontsize=9)
ax_main.text(6730, -95, "[S II]",va='center', ha='center', fontsize=9)
ax_main.text(4800, -95, "H$\\beta$",va='center', ha='center', fontsize=9)
ax_main.text(5078, -95, "[O III]",va='center', ha='center', fontsize=9)
ax_main.text(3730, -95, "[O II]",va='center', ha='center', fontsize=9)

# Label each spectrum
ax_main.text(5435, 55, 'AT2022abfc GMOS +10d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, 40, 'AT2023vth SEDM +2d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, 31, 'AT2023vth GMOS +23d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, 20, 'AT2023hkw Keck/DEIMOS +12d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, +8, 'AT2023fhn GMOS +7d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, 0, 'AT2023fhn P200 +8d', fontsize=10,  va='center', ha='center')
ax_main.text(5435, -11, 'AT2023fhn LRIS +14d', fontsize=10,   va='center', ha='center')
ax_main.text(5435, -22, 'AT2024qfm GMOS +6d', fontsize=10, va='center', ha='center')
ax_main.text(5490, -28.6, 'AT2024qfm LRIS +13d', fontsize=10,  va='center', ha='center')
ax_main.text(5780, -40, 'AT2024qfm Binospec +35d', fontsize=10,  va='center', ha='center')
ax_main.text(5770, -52, 'AT2024qfm LRIS +37d', fontsize=10, va='center', ha='center')
ax_main.text(5780, -62, 'AT2024aehp GMOS +5d', fontsize=10, va='center', ha='center')
ax_main.text(5780, -74, 'AT2024aehp KCWI +13d', fontsize=10, va='center', ha='center')
ax_main.text(5780, -83, 'AT2024aehp Binospec +19d', fontsize=10, va='center', ha='center')



ax_main.set_xlim([3500,7300])
ax_main.set_ylim([-98, 60])
ax_main.set_xlabel("$\lambda_{\\text{rest}} (\AA)$", fontsize=12)
ax_main.set_ylabel("$F_\lambda$ + offset", fontsize=12)
ax_main.set_yticks([])


plt.tight_layout()
plt.savefig('figures/fig2_spec_full.pdf', dpi=450, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
