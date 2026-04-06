import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kstest
from scipy.interpolate import CubicSpline
import pandas as pd
from astropy.cosmology import Planck18
from fig1_photometry import app_to_abs_mag



band = 'r'
population='single_all'

ccsne_galaxy_pop = pd.read_csv('data/taggart/taggart_host_photometry.dat', header=0, sep='\s+')


ccsne_galaxy_pop2 = pd.read_csv('data/taggart/taggart_host_masses.txt', header=0, sep='\s+')


new_names = []
for name in ccsne_galaxy_pop2['Name']:
    new_name=name
    if name[0] == '1':
        new_name='ASAS-SN'+name
    new_names.append(new_name)
ccsne_galaxy_pop2['Host']=new_names


ccsne_galaxy_popall = ccsne_galaxy_pop.loc[(ccsne_galaxy_pop['band']==band)].join(ccsne_galaxy_pop2.set_index('Host'), on='Host')

ccsne_galaxy_popall['absmag'] = app_to_abs_mag(np.array(ccsne_galaxy_popall['mag']).astype(float), np.array(ccsne_galaxy_popall['zhost']).astype(float), unitless=True)


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


slsne_galaxy_pop = pd.read_csv('data/taggart/taggart_host_photometry_slsne.dat', header=0, sep='\s+')


slsne_galaxy_pop2 = pd.read_csv('data/taggart/taggart_host_masses_slsne.txt', header=0, sep='\s+')

typeii_names = []; typei_names = []; unknown_names = []
for i,name in enumerate(slsne_galaxy_pop2['Name']):
    typeii_name='SLSN-II_'+name
    typei_name='SLSN-I_'+name
    unknown_name='SLSN-I?_'+name
    unknown_names.append(unknown_name)
    typeii_names.append(typeii_name)
    typei_names.append(typei_name)

slsne_galaxy_pop2['typeII']=typeii_names
slsne_galaxy_pop2['typeI']=typei_names
slsne_galaxy_pop2['typeI?']=unknown_names


slsne_galaxy_pop_2 = slsne_galaxy_pop.loc[(slsne_galaxy_pop['band']==band)].join(slsne_galaxy_pop2.set_index('typeII'), on='Host', how='inner')
slsne_galaxy_pop_1 = slsne_galaxy_pop.loc[(slsne_galaxy_pop['band']==band)].join(slsne_galaxy_pop2.set_index('typeI'), on='Host', how='inner')
slsne_galaxy_pop_unknown = slsne_galaxy_pop.loc[(slsne_galaxy_pop['band']==band)].join(slsne_galaxy_pop2.set_index('typeI?'), on='Host', how='inner')
slsne_galaxy_pop_all = pd.concat([slsne_galaxy_pop_1, slsne_galaxy_pop_2, slsne_galaxy_pop_unknown])



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
lfbot_absmags = app_to_abs_mag(lfbot_appmags, lfbot_zs, unitless=True)


fig, ax = plt.subplots(1,1, figsize=(6,4), layout='tight')

def sim_pop(population, obs_absmags, obs_masses, n, ax):
    """
    Simulate n samples from a given population, with each item in the sample having an
    absolute magnitude limit to reflect our observational bias

    `population`: Pandas table containing the population we draw from.  Has columns for the absolute
        magnitude of each galaxy and the log Mass/M_sun
    
    `obs_absmags`: A list with a length equal to the size of each sample, where each entry specifies the
        minimum absolute magnitude for that draw

    `obs_masses`: The actual log Mass/M_sun for the observed LFBOTs

    `n`: Number of samples to be drawn.  Each sample's size is equal to the number of observed LFBOTs
    (length of obs_absmags).
    """
    x_array=np.linspace(6, 12.3, 50)
    all_samples = np.empty(shape=(1, 50))
    for _ in range(n):
        sample_mass=[]
        for _, absmag in enumerate(obs_absmags):
            pop_slice = population.loc[(population['absmag']<=absmag)]
            sample_mass.append(np.random.choice(pop_slice['logmass']))
        sample_kde = gaussian_kde(sample_mass).evaluate(x_array)
        all_samples=np.append(all_samples, np.array([sample_kde]), axis=0)
        ax.plot(x_array, sample_kde, color='lightgray', alpha=0.2)
    all_samples=all_samples[1:, :]
    sample_percentiles = np.percentile(all_samples, q=[16, 50, 84], axis=0)

    obs_kde = gaussian_kde(obs_masses).evaluate(x_array)
    ax.plot(x_array, sample_percentiles[0], color='red', linewidth=1.5, ls='dotted')
    ax.plot(x_array, sample_percentiles[1], color='red', linewidth=2, label='Median')
    ax.plot(x_array, sample_percentiles[2], color='red', linewidth=1.5, ls='dotted')
    ax.plot(x_array, obs_kde, color='black', linewidth=4, label='Observed')
    

def sim_single_pop(population, obs_absmags, obs_masses, ax):
    """
    Weight a given host galaxy population by using its absolute magnitude, with the weight
    equal to what proportion of actual LFBOT redshifts that host galaxy would have been discovered in

    `population`: Pandas table containing the population we draw from.  Has columns for the absolute
        magnitude of each galaxy and the log Mass/M_sun
    
    `obs_absmags`: A list with a length equal to the size of each sample, where each entry specifies the
        minimum absolute magnitude for that draw

    `obs_masses`: The actual log Mass/M_sun for the observed LFBOTs

    """
    weights = []; logmasses = []
    population=population.dropna(subset=['absmag', 'logmass'])
    x_array=np.linspace(6, 12.3, 50)
    for mag in population['absmag']:
        weights.append(np.count_nonzero(obs_absmags>=mag))
    for i, logmass in enumerate(population['logmass']):
        for _ in range(weights[i]):
            logmasses.append(logmass)
    sample_kde = gaussian_kde(population['logmass'], weights=weights)
    obs_kde = gaussian_kde(obs_masses)
    ax.plot(x_array, obs_kde.evaluate(x_array), color='black', linewidth=4, label='LFBOT Distribution')
    ax.plot(x_array, sample_kde.evaluate(x_array), color='red', linewidth=2, ls='dotted', label='Weighted Comparison Distribution')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim([0, 0.538])
    ax.label_outer()
    ax.set_yticks([0, 0.2, 0.4])


def sim_single_pop_cdf(population, obs_absmags, obs_masses, ax):
    weights = []; logmasses = []
    population=population.dropna(subset=['absmag', 'logmass'])
    x_array=np.linspace(6, 12.3, 50)
    for mag in population['absmag']:
        weights.append(np.count_nonzero(obs_absmags>=mag))
    for i, logmass in enumerate(population['logmass']):
        for _ in range(weights[i]):
            logmasses.append(logmass)

    ax.ecdf(obs_masses, color='black', linewidth=4, label='LFBOT Distribution')
    ax.ecdf(logmasses, color='red', linewidth=2, ls='dotted', label='Weighted Comparison Distribution')
    ax.plot([10.8, 11.2],[1,1], color='black', linewidth=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim([0, 1.08])
    ax.set_yticks([0, 0.2,0.4, 0.6, 0.8, 1])



if population=='ccsne':
    sim_pop(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, 1000, ax)
    plt.savefig('figures/host_galaxy_sim_ccsne.png', dpi=450)
    plt.show()
elif population=='lgrb':
    sim_pop(lgrb_galaxy_popall, lfbot_absmags, lfbot_masses, 1000, ax)
    plt.savefig('figures/host_galaxy_sim_lgrb.png', dpi=450)
    plt.show()
elif population=='slsne_all':
    sim_pop(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, 1000, ax)
    ax.set_title('All SLSNE', fontsize=19)
    plt.savefig('figures/host_galaxy_sim_slsne_all.png', dpi=450)
    plt.show()
elif population=='slsne_i':
    sim_pop(slsne_galaxy_pop_1, lfbot_absmags, lfbot_masses, 1000, ax)
    ax.set_title('SLSNE-I', fontsize=19)
    plt.savefig('figures/host_galaxy_sim_slsne_i.png', dpi=450)
    plt.show()
elif population=='slsne_ii':
    sim_pop(slsne_galaxy_pop_2, lfbot_absmags, lfbot_masses, 1000, ax)
    ax.set_title('SLSNE-II', fontsize=19)
    plt.savefig('figures/host_galaxy_sim_slsne_ii.png', dpi=450)
    plt.show()
elif population=='final':
    plt.close()
    fig, axs = plt.subplots(1,2, figsize=(6,4), layout='constrained')
    flat_axs=axs.flatten()
    sim_pop(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, 1000, flat_axs[1])
    sim_pop(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, 1000, flat_axs[0])
    flat_axs[0].set_title('CCSNe', fontsize=15)
    flat_axs[0].legend(fontsize='9', loc='upper left')
    flat_axs[1].set_title('SLSNe', fontsize=15)
    for ax in flat_axs:
        ax.tick_params(axis='both', labelsize=11)
        ax.set_yticks([])
    fig.text(0.5, 0.04, 'Host Galaxy $\log \left( M/M_{\odot} \\right)$', ha='center', fontsize=20)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig('figures/host_galaxy_sim.png', dpi=450)
    plt.show()
elif population=='final_all':
    plt.close()
    fig, axs = plt.subplots(1,4, figsize=(12,3), layout='constrained')
    flat_axs=axs.flatten()
    sim_pop(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, 1000, flat_axs[1]) 
    sim_pop(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, 1000, flat_axs[0])
    sim_pop(slsne_galaxy_pop_1, lfbot_absmags, lfbot_masses, 1000, flat_axs[2])
    sim_pop(slsne_galaxy_pop_2, lfbot_absmags, lfbot_masses, 1000, flat_axs[3])
    flat_axs[0].legend(fontsize='9', loc='upper left')
    flat_axs[0].text(0.95, 0.91, 'CCSNe',     fontsize=12, ha='right', va='center', transform=flat_axs[0].transAxes)
    flat_axs[1].text(0.95, 0.91, 'SLSNe',     fontsize=12, ha='right', va='center', transform=flat_axs[1].transAxes)
    flat_axs[2].text(0.95, 0.91, 'SLSNe-I',   fontsize=12, ha='right', va='center', transform=flat_axs[2].transAxes)
    flat_axs[3].text(0.95, 0.91, 'SLSNe-II',  fontsize=12, ha='right', va='center', transform=flat_axs[3].transAxes)
    for ax in flat_axs:
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim([0, 1.07])
        ax.label_outer()
        ax.set_yticks([])
    fig.text(0.5, 0.04, 'Host Galaxy $\log \left( M/M_{\odot} \\right)$', ha='center', fontsize=14)
    fig.text(0.022, 0.5, '# galaxies per log mass', ha='center', fontsize=14, rotation='vertical', va='center')
    plt.tight_layout(rect=(0.03, 0.05, 1, 1))
    #plt.show()
elif population=='single_all':
    plt.close()
    fig, axs = plt.subplots(2,4, figsize=(12,6.5), layout='constrained')
    flat_axs=axs.flatten()
    sim_single_pop(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, flat_axs[0]) # p = 0.4318
    sim_single_pop(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, flat_axs[1]) # p = 0.9563
    sim_single_pop(slsne_galaxy_pop_1, lfbot_absmags, lfbot_masses, flat_axs[2]) # p = 0.06794
    sim_single_pop(slsne_galaxy_pop_2, lfbot_absmags, lfbot_masses, flat_axs[3]) # p = 0.39
    sim_single_pop_cdf(ccsne_galaxy_popall, lfbot_absmags, lfbot_masses, flat_axs[4])
    sim_single_pop_cdf(slsne_galaxy_pop_all, lfbot_absmags, lfbot_masses, flat_axs[5])
    sim_single_pop_cdf(slsne_galaxy_pop_1, lfbot_absmags, lfbot_masses, flat_axs[6])
    sim_single_pop_cdf(slsne_galaxy_pop_2, lfbot_absmags, lfbot_masses, flat_axs[7])
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
    plt.savefig('figures/fig11_host_galaxy_sim_all.pdf', dpi=450)
    plt.show()


plt.close()
