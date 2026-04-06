""" Figure showing the position of the host galaxy in the M*-SFR plane """

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import vals
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 7 # The maximum allowed for ED figures

tcol = vals.lgrb_col
tcol = 'k'


if __name__=="__main__":
    figwidth_mm = 89 # Nature standard
    figwidth_in = (figwidth_mm/10)/2.54 # in inches
    s=20
    # Initiate figure
    fig,ax = plt.subplots(1,1,figsize=(figwidth_in,figwidth_in))

    # Plot the Taggart sample (scraped from 20xnd paper)
    dat = np.loadtxt("data/taggart_sample.txt", delimiter=',')
    ax.scatter(dat[:,0], dat[:,1], marker='+', c=vals.sn_col, s=25, label='SN')
    dat = np.loadtxt("data/taggart_lgrb.txt", delimiter=',')
    ax.scatter(dat[:,0], dat[:,1], marker='x', c=vals.lgrb_col, s=20, label='LGRB')

    # Add 24puz (Somalwar et al. 2025)
    ax.scatter(1e8, 0.01, marker='*', c=vals.tde_col, label='TDE')
    # Add 18cow
    x = 1.42E9
    xbot = (1.42-0.29)*1E9
    xtop = (1.42+0.17)*1E9
    y = 0.22
    ybot = 0.22-0.04
    ytop = 0.22+0.03
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label='LFBOT', s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add CSS161010
    x = 2E7
    xbot = 1E7
    xtop = 3E7
    y = 1E-2
    ybot = 0.3E-2
    ytop = 2E-2
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None, s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add Koala
    x = 5.1E8
    xbot = (5.1-2.0)*1E8
    xtop = (5.1+3.4)*1E8
    y = 6.8
    ybot = (6.8-4.6)
    ytop = (6.8+3.7)
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None, s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add AT2020xnd
    x = 8E7
    xbot = 3E7
    xtop = 3E8
    y = 0.02
    ybot = 0.02-0.005
    ytop = 0.02+0.005
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None, s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add 2020mrf
    x = 10**7.94
    xbot = 10**(7.94-0.39)
    xtop = 10**(7.94+0.22)
    y = 6.93*1E-3
    ybot = (6.93-0.27)*1E-3
    ytop = (6.93+3.90)*1E-3
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None,s=s,  zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add AT2022tsd
    x = 10**(9.96)
    xtop = 10**(9.96+0.06)
    xbot = 10**(9.96-0.09)
    y = 0.55
    ytop = 0.55+1.36
    ybot = 0.55-0.19
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None, s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)

    # Add AT2024wpp  Perley et al. 2026
    x = 10**(8.76)
    xtop = 10**(8.76+0.03)
    xbot = 10**(8.76-0.03)
    y = 0.075
    ytop = 0.075+0.01
    ybot = 0.075-0.01
    ax.scatter(x, y, marker='o', facecolor=vals.fbot_col, edgecolor='k', label=None, s=s, zorder=3, alpha=0.5)
    ax.hlines(y, xbot, xtop, color=vals.fbot_col, alpha=0.5)
    ax.vlines(x, ybot, ytop, color=vals.fbot_col, alpha=0.5)


    # Add new FBOTs
    # Format is (Mass_lower, Mass_median, mass_upper, sfr_lower, sfr_median, sfr_upper)
    host_data={'AT2022abfc':(10.764655862531049,10.795012556560364,10.822748026971544, 3.68172858175936,4.3007101302536155,4.921330455197504),
               'AT2023hkw':(10.557749354879311,10.661027615967404,10.776302501570063, 2.002167599575805,2.7521105276805016,4.1073354920445935),
               'AT2023fhn':(10.005189386906897,10.066857842946915,10.134252580214296, 6.57488219426011,7.7378497620322735,9.118187525828919),
               'AT2023vth':(8.914316232983944,8.945177812786273,8.975826838422412, 0.20921337759209255,0.24173693902259238,0.28295552922156614),
               'AT2024qfm':(10.023931048924036,10.153901622152992,10.312751738640161, 2.1674455211214774,3.3068306724670626,4.437417530228622),
               'AT2024aehp':(8.756791596847002,8.876250591959275,8.973255912062829, 1.3555748020033165,1.5812439120344963,1.8344270917784233)}
    
    for object in vals.objects:
        # Extract mass and SFR percentiles
        masses = host_data[object][0:3]
        sfrs=host_data[object][3:]

        color=vals.colors[object]
        marker=vals.markers[object]
        ax.scatter(10**masses[1], sfrs[1], facecolor=color, edgecolor='k', marker=marker, s=50,
                   zorder=100, label=object)
        # Plot error bars
        ax.hlines(sfrs[1], 10**masses[0], 10**masses[2], color=color)
        ax.vlines(10**masses[1], sfrs[0], sfrs[2], color=color)



    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"Stellar Mass ($M_\odot$)", fontsize=10)
    ax.set_ylabel(r"Star formation rate ($M_\odot$ yr${}^{-1}$)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim([10**-3.4, 10**1.3])
    ax.legend(loc='lower right', prop={'size':7.5})
    plt.tight_layout()
    plt.savefig("figures/fig10_host_galaxy.pdf", dpi=450, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
