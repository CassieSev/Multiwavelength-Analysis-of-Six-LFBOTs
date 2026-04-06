import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18
from sed_fit import bpl_smooth_a1
marker_list = itertools.cycle(('s', 'o', '*', 'D', 'H'))

plasma = mpl.colormaps['plasma'].resampled(20)

redshifts={'AT2021ahuo': 0.342, 'AT2022abfc':0.212,
               'AT2023fhn': 0.24, 'AT2023hkw': 0.339,
               'AT2023vth': 0.0747, 'AT2018cow': 0.0141, 'AT2024qfm': 0.227, 'AT2024aehp': 0.1715}
objects =['AT2022abfc', 'AT2023fhn', 'AT2023hkw', 'AT2023vth', 'AT2024qfm', 'AT2024aehp']
bins={'AT2023fhn': [0, 10, 14, 48, 70, 97, 140, 177,230, 280, 500],
      'AT2023vth':[0, 20,29, 37, 50, 65, 75, 110, 130, 270, 420],
      'AT2024aehp':[0, 20, 140, 160]}

custom_markers={'AT2023fhn': ['s', 's', '*', 'D', 'o','s', 'o','D', '*', 'D'],
                'AT2023vth':['s', 's','o', 'D', '*', 'o','s', 'D', '*', 'o','s'],
                'AT2024aehp':['s', 's', 'D', '*', '1','s', 'D', '*', 'D']}

custom_labels={'AT2023fhn': ['','10d', '25d', '53d', '71-77d','110-111d', '123d','160-167d','186-201d', '356d'],
                'AT2023vth':['', '27-29d', '30-34d', '38-43d', '53d', '64d','79-97d', '110-120d', '188d', '400d'],
                'AT2024aehp':['', '70-73d', '120-132d'],
                'AT2023hkw':['44d', '64d']}


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


def lum_to_ujy(lum, redshift):
    """converts from L (erg/s/Hz) to flux (uJy)"""
    dcm=Planck18.luminosity_distance(z=redshift).cgs.value
    return lum*(1+redshift)/(1E-6 * 1E-23 * 4 * np.pi * dcm**2)

def plot_sed(df_full, color, marker_params, label=None, ax=None, hide_nondets=False):
    """
    Plot the SED for a given dataframe slice, applying a label , color, and marker style
    """
    marker, ms, alpha = marker_params

    # Split data  to full data points and upper limits
    df_data=df_full.loc[(df_full['unc'].astype(float)>=0)]
    df_upper=df_full.loc[(df_full['unc'].astype(float)<0)]
    df_chrimes=df_data.loc[(df_data['obs']=='chrimes')]
    if ax != None and len(df_data)>0:
        if hide_nondets:
            ax.plot(df_data['freq_corr'], df_data['uJy'].astype(float), marker=None, color=color, alpha=alpha)
            ax.errorbar(df_data['freq_corr'], df_data['uJy'].astype(float), markeredgecolor='k', yerr=df_data['unc'].astype(float),marker=marker,
                            ls='none', color=color, ecolor=color, capsize=None, label=label, ms=1.4*ms, alpha=alpha)
            return None
        # Plot the line for all points
        ax.plot(df_full['freq_corr'], df_full['uJy'].astype(float), ls='dotted', marker=None, color=color, alpha=alpha)
        ax.plot(df_data['freq_corr'], df_data['uJy'].astype(float), marker=None, color=color, alpha=alpha)
        # Plot full data points
        ax.errorbar(df_data['freq_corr'], df_data['uJy'].astype(float),yerr=df_data['unc'].astype(float),marker=marker,
                            ls='none', color=color, markeredgecolor='k', ecolor=color, capsize=None, label=label, ms=1.4*ms, alpha=alpha, elinewidth=1)
        ax.errorbar(df_chrimes['freq_corr'], df_chrimes['uJy'].astype(float),yerr=df_chrimes['unc'].astype(float),marker=marker,
                            ls='none', markerfacecolor='white', markeredgecolor=color, ecolor=color, capsize=None, label=None, ms=1.4*ms, elinewidth=1, alpha=alpha)
        # Plot upper limits
        #ax.scatter(df_upper['freq_corr'], df_upper['uJy'].astype(float), marker=marker, facecolors='white', edgecolors=color)
        # Add arrows for each upper limit
        x_upper, y_upper = df_upper['freq_corr'], df_upper['uJy'].astype(float)
        for i, x in enumerate(x_upper):
            ax.arrow(x, y_upper.iloc[i], 0, -y_upper.iloc[i]/3, length_includes_head=True,
                    head_width=x/10, head_length=y_upper.iloc[i]/10, color=color)
            
    



def make_sed(object, arrays=None, ax=None):
    """Function to make radio SED for a given object, using the dataframe defined under global variable
    `raw_df`.  Can select which observatories to include in plot under `arrays`"""
    # First, get a list of epochs corresponding to the object
    df_binning = raw_df.loc[(raw_df['object']==object)&(raw_df['uJy']!='na')]
    times=np.unique(df_binning['t_obs'])
    norm = LogNorm(vmin=11, vmax=380)
    hide_nondets=False
    # Basic Plot for all arrays
    if arrays == None:
        for t in times:
            df_plot=raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']==t)]
    # Advanced plot for final figure, filtering on arrays
    elif object=='AT2023fhn' or object=='AT2023vth' or object=='AT2024aehp':
        bin_cond=bins[object]
        for i in range(len(bin_cond)-1):
            label=custom_labels[object][i]
            default_marker=custom_markers[object][i]
            marker_params=[default_marker, 3, 1]
            if default_marker=='*' or default_marker=='1': marker_params[1] = 5
            if object=='AT2023fhn' and i == 6: hide_nondets=True
            df_plot=raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']>=bin_cond[i])&(raw_df['t_obs']<=bin_cond[i+1])&(raw_df['array'].isin(arrays))].sort_values(by=['freq_corr'])
            print(df_plot)
            default_color=plasma(norm(float(np.average(df_plot['t_obs'].astype(float)))))
            plot_sed(df_plot, default_color, marker_params, label=label, ax=ax, hide_nondets=hide_nondets)
            add_power_law_fit(ax, object, float(np.average([bin_cond[i:i+2]])), colornorm=norm, color=default_color)
    else:
        prev_index=0
        for t in times:
            # default color is based on a plasma colormap ranging from minimum to maximum t
            if object=='AT2023fhn' or object=='AT2023vth' or object=='AT2024aehp':
                # Find which bin time falls into
                bin_cond=bins[object]
                index=np.argwhere(bin_cond-t>0)[0][0]
                # Set labels and colors manually
                default_color=plasma(norm(float(np.average([bin_cond[index-1:index+1]]))))
                #default_marker=custom_markers[object][index]
                #raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']==t), 't_obs'] = float(np.average([bin_cond[index-1:index+1]]))
                #t=float(np.average([bin_cond[index-1:index+1]]))
                # if in the same bin, do not make new label
                #if index != prev_index: label=custom_labels[object][index]
                #else: label=None

                prev_index=index
            else:
                default_color=plasma(norm(float(t)))
                # Increment default marker to be next from a list
                default_marker=next(marker_list)
                label='{}d'.format(np.int32(np.round(t,0)))
            marker_params=[default_marker, 3, 1]
            if default_marker=='*' or default_marker=='1': marker_params[1] = 5
            # Select rows that have the right object, right arrays, at correct t_obs, and is our new data
            df_plot=raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']==t)&(raw_df['array'].isin(arrays))].sort_values(by=['freq_corr'])
            # First, plot our data
            plot_sed(df_plot, default_color, marker_params, label=label, ax=ax, hide_nondets=hide_nondets)
            # Then, plot any data from other proposals, if it exists, under different color and marker
            #df_plot=raw_df.loc[(raw_df['object']==object)&(raw_df['t_obs']==t)&(raw_df['array'].isin(arrays))&(raw_df['obs']=='chrimes')].sort_values(by=['freq_corr'])
            #plot_sed(df_plot, default_color, ['o', 5, 0.6], label=None, ax=ax)
            add_power_law_fit(ax, object, t, colornorm=norm)

            #add_power_law(ax, object, t)
    add_outside_data(ax, object, colornorm=norm)
                

def add_power_law(ax, object, time):
    """Add power law lines close to interesting points, may be removed in final product.  Dependent on 
    object being examined and the time"""
    if object=='AT2023hkw':
        x_line = np.linspace(9, 12, 50)
        y1_line = np.power(x_line, 1)*79/9
        y2_line = np.power(x_line, 2)*79/81
    elif object == 'AT2022abfc':
        x_line = np.linspace(10, 17, 50)
        y1_line = np.power(x_line, 1)*38/10
        y2_line = np.power(x_line, 2)*38/100
    else: return
    ax.plot(x_line, y1_line, marker=None, color='lightgrey', alpha=0.5)
    ax.plot(x_line, y2_line, marker=None, color='lightgrey', alpha=0.5)


def add_outside_data(ax, object, color=None, colornorm=None):
    """
    Add data from Chrimes et al. 2024 for AT2023fhn
    """
    def get_color(time):
        if color:
            return color
        # If we don't specify a color, then use the color map
        return plasma(colornorm(float(time)))
    # For this data file, the units are Hertz for frequency, erg/s/Hz for specific luminosity
    def extract_data(file, x_units='Hz', y_units='lum'):
        df=pd.read_csv(file, header=None, names=['x', 'y'])
        if x_units=='Hz':
            df['x']=df['x']/10**9
        if y_units=='lum':
            df['y']=lum_to_ujy(df['y'], redshifts[object])
        return (df['x'], df['y'])
    
    if object == 'AT2023fhn':
        data_90 = extract_data('data/radio/at2023fhn_chrimes/23fhn_90.csv',x_units='Hz', y_units='lum')
        data_138 = extract_data('data/radio/at2023fhn_chrimes/23fhn_138.csv',x_units='Hz', y_units='lum')
        #ax.plot(data_90[0],data_90[1], marker=None, color=get_color(90), lw=2, alpha=0.5)
        #ax.plot(data_138[0],data_138[1], marker=None, color=get_color(138), lw=2, alpha=0.5)
        

def add_power_law_fit(ax, object, time, color=None, colornorm=None):
    """
    Add Power Law fits used to derive synchrotron parameters
    """
    params=None
    def get_color(time):
        if color:
            return color
        # If we don't specify a color, then use the color map
        return plasma(colornorm(float(time)))
    # For this data file, the units are Hertz for frequency, erg/s/Hz for specific luminosity
    if object =='AT2023fhn':
        if time >= 80 and time <= 96:
            params=[4.39289359e+00, 1.41134008e-04, -5.45737055e-01]
        if time >= 115 and time <= 139:
            params=[5.30108385e+00, 2.13558316e-04, -5.19405716e-01]
    elif object == 'AT2023vth':
        if time <= 100 and time >=75:
            params=[8.44301292e+00, 1.54505245e-03, -9.32892483e-01]
        if time <= 130 and time >= 100:
            params=[3.33337392e+00, 1.00757645e-03, -5.29400533e-01]
    elif object == 'AT2024aehp':
        if time <=160 and time >=140:
            params=[1.22100167e+01,  4.10992778e-04, -4.63646138e-01]
    if params:
        nu_p_fit, F_p_fit, a2_fit = params
        x_range=np.logspace(0, 2.5, num=100)
        ax.plot(x_range, 1e6*bpl_smooth_a1(x_range, nu_p_fit, F_p_fit, a2_fit, a1=5/2), color=get_color(time), lw=2, alpha=0.6, ls='dashed',
                marker=None)



def add_indices(ax, object):
    "Add power law indices on the graphs, specified for each object"
    if object =='AT2023vth':
        ax.text(0.56, 0.72, 'F$\\sim\\nu^{-0.80}$', fontsize=6, transform=ax.transAxes, rotation=-43)
        ax.text(0.62, 0.77, 'F$\\sim\\nu^{-1.07}$', fontsize=6, transform=ax.transAxes, rotation=-43)
    elif object == 'AT2023hkw':
        ax.text(0.4, 0.5, 'F$\\sim\\nu^{-1.08}$', fontsize=6, transform=ax.transAxes, rotation=-35)
    elif object == 'AT2024aehp':
        ax.text(0.44, 0.58, 'F$\\sim\\nu^{1.14}$', fontsize=6, transform=ax.transAxes, rotation=43)
    


def calc_power_law(df, conditions):
    x = np.array(df.loc[conditions, 'freq_corr'])
    y=np.array(df.loc[conditions, 'uJy'].astype(float))
    def linear(x, m, b):
        return m*x+b

    

raw_df = pd.read_csv('data/new_radio_data.txt')
raw_df['freq_corr']=fix_freq(raw_df['freq'], raw_df['object'], raw_df['frame'])
def all_fig():
    fig, axs = plt.subplots(3, 2, figsize=(6,6), sharex=True, sharey=True)
    flat_axs=axs.flatten()
    for i,ax in enumerate(flat_axs):
        make_sed(objects[i],['VLA', 'NOEMA', 'AMI', 'GMRT'], ax)
        if objects[i]=="AT2023vth":
            loc='lower left'
        else:
            loc='upper left'
        if i == 3: ax.legend(loc=loc, ncols=4, prop={'size':6}, labelspacing=0.1, handletextpad=0.1, columnspacing=1)
        elif i ==1: ax.legend(loc=loc, ncols=2, prop={'size':6}, labelspacing=0.1)
        else: ax.legend(loc=loc, prop={'size':6}, labelspacing=0.1)
        #add_indices(ax, objects[i])

        #ax.set_title('{}'.format(objects[i]), fontsize=12)
        ax.text(0.98, 0.95, '{}'.format(objects[i]), fontsize=10, ha='right', va='top', transform=ax.transAxes)
        #ax.set_xlabel('Rest Frame Frequency (GHz)', fontsize=16)
        if i == 2:
            ax.set_ylabel('$F_\\nu$ ($\mu$Jy)', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1, 5, 10, 50], ['1', '5', '10', '50'])
        ax.set_xlim([0.2, 250])
        ax.set_ylim([12, 1800])
        ax.tick_params(labelsize=10)
    for ax in flat_axs: ax.label_outer()


    fig.text(0.54, 0.022, '$\\nu_{\\text{rest}}$ (GHz)', fontsize=10, ha='center', va='center')
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    #add_power_law(ax, object, None)
    plt.savefig('figures/fig5_radio_sed.pdf', dpi=450)
    plt.show()
    plt.close()


all_fig()

    