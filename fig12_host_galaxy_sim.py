import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from fig1_photometry import app_to_abs_mag



band = 'r'
population='single_all'

# Grab the luminosities and masses of the CCSNe galaxies
ccsne_galaxy_pop = pd.read_csv('data/taggart/taggart_host_photometry.dat', header=0, sep='\s+')


ccsne_galaxy_pop2 = pd.read_csv('data/taggart/taggart_host_masses.txt', header=0, sep='\s+')


new_names = []
for name in ccsne_galaxy_pop2['Name']:
    new_name=name
    if name[0] == '1':
        new_name='ASAS-SN'+name
    new_names.append(new_name)
# Standardize Name
ccsne_galaxy_pop2['Host']=new_names

# Select on r-band 
ccsne_galaxy_popall = ccsne_galaxy_pop.loc[(ccsne_galaxy_pop['band']==band)].join(ccsne_galaxy_pop2.set_index('Host'), on='Host')

# Get absolute r-band magnitudes and redshifts
ccsne_galaxy_popall['absmag'] = app_to_abs_mag(np.array(ccsne_galaxy_popall['mag']).astype(float), np.array(ccsne_galaxy_popall['zhost']).astype(float), unitless=True)

# Repeat with LGRB hosts
lgrb_galaxy_pop = pd.read_csv('data/taggart/taggart_host_photometry_lgrb.dat', header=0, sep='\s+')


lgrb_galaxy_pop2 = pd.read_csv('data/taggart/taggart_host_masses_lgrb.txt', header=0, sep='\s+')

new_names = []
for i,name in enumerate(lgrb_galaxy_pop2['Name']):
    if lgrb_galaxy_pop2['Type'][i]=='SN':
        new_name='LGRB-SN_'+name
    else:
        new_name='LGRB_'+name
    new_names.append(new_name)
lgrb_galaxy_pop2['Host']=new_names


lgrb_galaxy_popall = lgrb_galaxy_pop.loc[(lgrb_galaxy_pop['band']==band)].join(lgrb_galaxy_pop2.set_index('Host'), on='Host')

lgrb_galaxy_popall['absmag'] = app_to_abs_mag(np.array(lgrb_galaxy_popall['mag']).astype(float), np.array(lgrb_galaxy_popall['zhost']).astype(float), unitless=True)

# And SLSNE hosts
slsne_galaxy_pop_df1 = pd.read_csv('data/taggart/taggart_host_photometry_slsne.dat', header=0, sep='\s+')


slsne_galaxy_pop_df2 = pd.read_csv('data/taggart/taggart_host_masses_slsne.txt', header=0, sep='\s+')

typeii_names = []; typei_names = []; unknown_names = []
for i,name in enumerate(slsne_galaxy_pop_df2['Name']):
    typeii_name='SLSN-II_'+name
    typei_name='SLSN-I_'+name
    unknown_name='SLSN-I?_'+name
    unknown_names.append(unknown_name)
    typeii_names.append(typeii_name)
    typei_names.append(typei_name)

# Also get the Type II and Type I SLSNe subsamples
slsne_galaxy_pop_df2['typeII']=typeii_names
slsne_galaxy_pop_df2['typeI']=typei_names
slsne_galaxy_pop_df2['typeI?']=unknown_names


slsne_galaxy_pop_2 = slsne_galaxy_pop_df1.loc[(slsne_galaxy_pop_df1['band']==band)].join(slsne_galaxy_pop_df2.set_index('typeII'), on='Host', how='inner')
slsne_galaxy_pop_1 = slsne_galaxy_pop_df1.loc[(slsne_galaxy_pop_df1['band']==band)].join(slsne_galaxy_pop_df2.set_index('typeI'), on='Host', how='inner')
slsne_galaxy_pop_unknown = slsne_galaxy_pop_df1.loc[(slsne_galaxy_pop_df1['band']==band)].join(slsne_galaxy_pop_df2.set_index('typeI?'), on='Host', how='inner')
slsne_galaxy_pop_all = pd.concat([slsne_galaxy_pop_1, slsne_galaxy_pop_2, slsne_galaxy_pop_unknown])


# Create the SLSNe host sample and the two subsamples
slsne_galaxy_pop_2['absmag'] = app_to_abs_mag(np.array(slsne_galaxy_pop_2['mag']).astype(float), np.array(slsne_galaxy_pop_2['zhost']).astype(float), unitless=True)
slsne_galaxy_pop_1['absmag'] = app_to_abs_mag(np.array(slsne_galaxy_pop_1['mag']).astype(float), np.array(slsne_galaxy_pop_1['zhost']).astype(float), unitless=True)
slsne_galaxy_pop_all['absmag'] = app_to_abs_mag(np.array(slsne_galaxy_pop_all['mag']).astype(float), np.array(slsne_galaxy_pop_all['zhost']).astype(float), unitless=True)




if band =='r':
    lfbot_appmags = 23 # Both LS and PS1 have limiting mags around 23 for r-band
lfbot_zs=[0.212, 0.24, 0.339, 0.0747, 0.227, 0.1715, 0.033, 0.2714, 0.014145, 0.1353, 0.2433, 0.2564, 0.0868]
lfbot_masses = [10.795012556560364, 10.06685784294691, 10.661027615967404, 8.945177812786273, 10.153901622152992, 8.876250591959275,
                7.3, # Coppejans
                8.71, # Ho 2020 koala
                9.15, # perley2019
                7.94, #yao 2022
                8, #perley2021
                10, #ho2023tsd
                8.941425622123354, # perley in prep
                ]
# Convert the apparent magnitude detection limit to an absolute mag limit at the different LFBOT redshifts
lfbot_absmags = app_to_abs_mag(lfbot_appmags, lfbot_zs, unitless=True)


fig, ax = plt.subplots(1,1, figsize=(6,4), layout='tight')
    

def sim_single_pop(population, obs_absmags, obs_masses, ax_pdf, ax_cdf):
    """
    Weight a given host galaxy population by using its absolute magnitude, with the weight
    equal to what proportion of actual LFBOT redshifts that host galaxy would have been discovered in.

    Then, plot the PDF and CDF of the weighted host galaxy mass distribution

    `population`: Pandas table containing the population we draw from.  Has columns for the absolute
        magnitude of each galaxy and the log Mass/M_sun
    
    `obs_absmags`: A list with a length equal to the size of each sample, where each entry specifies the
        minimum absolute magnitude for that draw

    `obs_masses`: The actual log Mass/M_sun for the observed LFBOTs

    """
    weights = []; logmasses = []
    # Remove NA values
    population=population.dropna(subset=['absmag', 'logmass'])
    x_array=np.linspace(6, 12.3, 50)
    # Using a galaxy's absolute magnitude, apply weights depending on how many of the LFBOT redshifts we would have detected it
    for mag in population['absmag']:
        weights.append(np.count_nonzero(obs_absmags>=mag))
    # Collect the log masses of the host galaxies
    for i, logmass in enumerate(population['logmass']):
        for _ in range(weights[i]):
            logmasses.append(logmass)
    # Plot the pdf using a gaussian KDE
    sample_kde = gaussian_kde(population['logmass'], weights=weights)
    obs_kde = gaussian_kde(obs_masses)
    ax_pdf.plot(x_array, obs_kde.evaluate(x_array), color='black', linewidth=4, label='LFBOT Distribution')
    ax_pdf.plot(x_array, sample_kde.evaluate(x_array), color='red', linewidth=2, ls='dotted', label='Weighted Comparison Distribution')
    ax_pdf.tick_params(axis='both', labelsize=14)
    ax_pdf.set_ylim([0, 0.538])
    ax_pdf.label_outer()
    ax_pdf.set_yticks([0, 0.2, 0.4])

    # Plot the cdf
    ax_cdf.ecdf(obs_masses, color='black', linewidth=4, label='LFBOT Distribution')
    ax_cdf.ecdf(logmasses, color='red', linewidth=2, ls='dotted', label='Weighted Comparison Distribution')
    ax_cdf.plot([10.8, 11.2],[1,1], color='black', linewidth=4)
    ax_cdf.tick_params(axis='both', labelsize=14)
    ax_cdf.set_ylim([0, 1.08])
    ax_cdf.set_yticks([0, 0.2,0.4, 0.6, 0.8, 1])


if population=='single_all':
    plt.close()
    fig, axs = plt.subplots(2,4, figsize=(12,6.5), layout='constrained')
    flat_axs=axs.flatten()
    # Create panels for the CCSNe results and the SLSNe 
    sim_single_pop(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, flat_axs[0], flat_axs[4]) # p = 0.4318
    sim_single_pop(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, flat_axs[1], flat_axs[5]) # p = 0.9563
    sim_single_pop(slsne_galaxy_pop_1, lfbot_absmags, lfbot_masses, flat_axs[2], flat_axs[6]) # p = 0.06794
    sim_single_pop(slsne_galaxy_pop_2, lfbot_absmags, lfbot_masses, flat_axs[3], flat_axs[7]) # p = 0.39
    flat_axs[0].legend(fontsize='7.5', loc='upper left')
    flat_axs[0].set_title('CCSNe',     fontsize=12)
    flat_axs[1].set_title('SLSNe',     fontsize=12)
    flat_axs[2].set_title('SLSNe-I',   fontsize=12)
    flat_axs[3].set_title('SLSNe-II',  fontsize=12)
    flat_axs[4].text(0.05, 0.927, 'p-value$\,\\approx\,0.43$',  fontsize=10, ha='left', va='center', transform=flat_axs[4].transAxes)
    flat_axs[5].text(0.05, 0.927, 'p-value$\,\\approx\,0.96$',  fontsize=10, ha='left', va='center', transform=flat_axs[5].transAxes)
    flat_axs[6].text(0.05, 0.927, 'p-value$\,\\approx\,0.07$',  fontsize=10, ha='left', va='center', transform=flat_axs[6].transAxes)
    flat_axs[7].text(0.05, 0.927, 'p-value$\,\\approx\,0.39$',  fontsize=10, ha='left', va='center', transform=flat_axs[7].transAxes)
    for ax in flat_axs:
        ax.tick_params(axis='both', labelsize=11)
        ax.label_outer()


    fig.text(0.54, 0.038, 'Host Galaxy $\log \left( M/M_{\odot} \\right)$', ha='center', fontsize=14)
    fig.text(0.027, 0.745, 'Proportion of galaxies per log mass', ha='center', fontsize=10, rotation='vertical', va='center')
    fig.text(0.02, 0.32, 'Cumulative Fraction of galaxies', ha='center', fontsize=10, rotation='vertical', va='center')
    fig.text(0.033, 0.32, 'at or below log mass', ha='center', fontsize=10, rotation='vertical', va='center')

    plt.tight_layout(rect=(0.03, 0.05, 1, 1))
    plt.savefig('figures/fig12_host_galaxy_sim_all.pdf', dpi=450)
    plt.show()


plt.close()
