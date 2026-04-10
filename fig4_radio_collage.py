""" 
Plot of luminosity over time
"""


from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

from astropy.cosmology import Planck18
from astropy.time import Time
from read_table import *
from astropy.table import Table

import vals

small = 9
med = 11
large = 12


def plot_limits(ax, x, y, ratiox, ratioy, col):
    """ Plot two arrows from the point """
    ax.annotate('', xy=(x*ratiox, y), xytext=(x, y),
            arrowprops=dict(
                facecolor=col, headwidth=10, width=1, headlength=7))
    ax.annotate('', xy=(x, y*ratioy), xytext=(x, y),
            arrowprops=dict(
                facecolor=col, headwidth=10, width=1, headlength=7))


def plot_line(ax, d, t, nufnu, name, label, col, legend=False, zorder=1):
    """ Plot a line
    If nu > 90 GHz, make it on the left axis
    If nu < 10 GHz, make it on the right axis
    
    Parameters
    ----------
    nufnu: Hz * mJy 
    name: name of the source
    label: label to use as the legend (which also determines the col)
    """
    lum = nufnu * 1e-23 * 1e-3 * 4 * np.pi * d**2
    fs = 11
    nsize = 10 # normal size for points
    fcol = col
    marker = '*'
    s = 70
    ls='-'
    alpha=1
    if name=='AT2018cow':
        marker='.'
        fcol = col
        s=70
        ls='-'
        alpha=0.3
        ax.scatter(
           t, lum, facecolor=fcol, edgecolor=col, 
           marker=marker, s=s, zorder=zorder, alpha=alpha)
    else:
        if label=='SN':
            marker='o'
            s=nsize
            fcol = col # fill color
            label = 'SN'
            ls='--'
            alpha=0.6
        elif label=='GRB':
            marker='o'
            fcol = 'white' # unfilled
            s=nsize
            label = 'GRB'
            alpha=1
            ls='dotted'
        elif label=='Rel. SN':
            marker='s'
            fcol = col 
            s=nsize
            label = 'Rel. SN'
        elif label=='TDE':
            marker='s'
            fcol = 'white' #unfilled
            s=nsize
            label='TDE'
    if legend:
        ax.plot(t, lum, c=col, ls=ls, label=legend, zorder=zorder, alpha=alpha)
    else:
        ax.plot(t, lum, c=col, ls=ls, label=None, zorder=zorder, alpha=alpha)
    return lum


def plot_points(ax, d, nu, t, f, marker, name=None):
    """ Plot set of two points """
    lums = []
    for ii,nuval in enumerate(nu):
        if nuval > 90E9:
            lum = plot_point(ax, d, nuval, t[ii], f[ii], marker, name=name)
            lums.append(lum)
        else:
            lum = plot_point(ax, d, nuval, t[ii], f[ii], marker)
            lums.append(lum)
    ax.plot(
        t, lums, ls='--', c='k', zorder=0)
    return lums


def koala(ax, col, legend):
    z = 0.2714
    dt = np.array([81,310,352,396,596])/(1+0.2714)
    f = np.array([8.3E39,1.5E39,1.1E39,8.16E38,4.1E38])-2.65E38
    ax.errorbar(dt, f, 0.0006*f, marker='.',alpha=0.3, c=col, ms=7)
    ax.plot(dt, f, c=col, lw=2, alpha=0.3)
    #ax.text(
    #        60, 1.2E40, "AT2018lug", 
    #        fontsize=med, horizontalalignment='center',
    #        verticalalignment='bottom', color=col)


def at2021ahuo(ax, col, legend):
    dt = Time("2023-04-01")-Time("2021-05-18")
    zval = 0.342
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    y = 12*1E-6*1E-23*4*np.pi*dcm**2 * 10E9
    x = dt.value/(1+zval)
    ax.scatter(x, y, edgecolor=col, 
            facecolor='white', marker='.', s=70, zorder=100, alpha=0.3)
    ax.arrow(x, y, 0, -y/4, length_includes_head=True, 
             head_width=x/10, head_length=y/15, color=col, zorder=0, alpha=0.3)
    #ax.text(
    #        x*1.4, y*1.2/(1+zval), "21ahuo", 
    #        fontsize=8, horizontalalignment='right',
    #        verticalalignment='bottom', color=col)


def at2020xnd(ax, col, legend):
    dt = np.array([13, 25, 36, 51.9, 71.9, 94.7, 131.6])/(1+0.2442)
    zval = 0.2442
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([0.02, 0.057, 0.08, 0.15, 0.2, 0.168, 0.109]) * 1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(dt, f, c=col, marker='.', s=70, zorder=5, alpha=0.3)
    ax.plot(dt, f, c=col, lw=2, zorder=5, alpha=0.3)
    #ax.text(
    #        dt[0]/1.2, f[0]/1.2/(1+zval), "AT2020xnd", 
    #        fontsize=med, horizontalalignment='right',
    #        verticalalignment='top', color=col)


def at2020mrf(ax, col, legend):
    dt = np.array([262.9, 417.5])*2/(1+0.1353)
    zval = 0.1353
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([0.271, 0.049]) * 1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(dt, f, c=col, marker='.', s=70, zorder=5, alpha=0.3)
    ax.plot(dt, f, c=col, lw=2, zorder=5, alpha=0.3)


def at2022tsd(ax, col, legend):
    dt = np.array([19.75,26.16,34.04,49.13,68.98,112.69,167.53,513])/(1+0.256)
    zval = 0.256
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([0.023,0.031,0.033,0.031,0.049,0.048,0.038,3*0.003]) * \
            1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 15E9
    ax.scatter(dt, f, c=col, marker='.', s=70, zorder=5, alpha=0.3)
    ax.plot(dt[0:-2], f[0:-2], c=col, lw=2, zorder=5, alpha=0.3)
    ax.plot(dt[-2:], f[-2:], c=col, ls='dotted', lw=2, zorder=5, alpha=0.3)

def at2024wpp(ax, col, legend):
    # Nayana et al. 2025 
    zval=0.0868
    dt = (np.array([60613.47, 60628.37, 60657.27, 60706.13, 60723.13, 60753.04])-60578.4)/(1+zval)
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([0.314,0.477, 0.847, 0.202, 0.337, 0.143]) * \
            1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 15E9
    ax.scatter(dt, f, c=col, marker='.', s=70, zorder=5, alpha=0.3)
    ax.plot(dt, f, c=col, lw=2, zorder=5, alpha=0.3)



def at2022abfc(ax, col, legend):
    # Time in days, convert to transient frame
    dt = np.array([15,32,135,215,438])/(1+0.212)
    dets=np.array([False, True, True, False, False])
    zval = 0.212
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    # last one is a non-detection
    # Fluxes in uJy, then convert to cgs L_nu units
    f = np.array([0.013,0.038,0.043,0.026,0.004*3]) * \
            1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    # Plot detections
    ax.scatter(dt[dets], f[dets], c=col, marker=vals.markers['AT2022abfc'], s=30, zorder=5)
    ax.plot(dt[1:3], f[1:3], c=col, lw=2, zorder=5)
    # Plot upper limits with arrows
    ax.scatter(dt[~dets], f[~dets], edgecolor=col, marker=vals.markers['AT2022abfc'], s=30, zorder=5, facecolor='white')
    for (t_upper, y_upper) in  zip(dt[~dets], f[~dets]):
        ax.arrow(t_upper, y_upper, 0, -y_upper/2, length_includes_head=True, 
             head_width=t_upper/7, head_length=y_upper/8, color=col, zorder=200)
    ax.plot(dt[0:2], f[0:2], c=col, lw=2, zorder=5, ls='dotted'); ax.plot(dt[2:], f[2:], c=col, lw=2, zorder=5, ls='dotted')
    ax.text(
            dt[0]/1.07, f[0], "AT2022abfc", 
            fontsize=8, horizontalalignment='right',
            verticalalignment='top', color=col, fontweight='bold')


def at2023fhn(ax, col, legend):
    """ AT 2023fhn """
    # First epoch was non-detection
    # 23 June 2023: 42 uJy at 10 GHz; 74 days
    # 9 July 2023: 118 uJy at 10 GHz; 90 days
    # 25 August 2023: 143 uJy at 10 GHz; 140 days
    zval = 0.24
    dt = np.array([12, 31, 74, 90, 140, 442]) / (1+zval)
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([3*0.018, 3*0.005, 0.042, 0.118, 0.143,0.033]) * \
            1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(dt, f, c=col, marker=vals.markers['AT2023fhn'], s=70, zorder=100)
    ax.plot(dt[2:], f[2:], c=col, lw=2, zorder=100)
    ax.plot(dt[0:3], f[0:3], c=col, lw=2, zorder=100, ls='dotted')

    ax.text(
            dt[0]*1.04, f[0]*2, "AT2023fhn", 
            fontsize=8, horizontalalignment='right',
            verticalalignment='top', color=col, fontweight='bold')

    # Upper limit from the first observation...Chrimes VLA program
    x = 12 / (1+zval)
    y = 3*18 * 1E-6 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(x, y, edgecolor=col, 
            facecolor='white', marker=vals.markers['AT2023fhn'], s=80, zorder=200)
    ax.arrow(x, y, 0, -y/2, length_includes_head=True, 
             head_width=x/7, head_length=y/8, color=col, zorder=200)

    # Upper limit from my program...
    x = 31 / (1+zval)
    y = 3*5 * 1E-6 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(x, y, edgecolor=col, 
            facecolor='white', marker=vals.markers['AT2023fhn'], s=80, zorder=200)
    ax.arrow(x, y, 0, -y/2, length_includes_head=True, 
             head_width=x/7, head_length=y/8, color=col, zorder=100)


def at2023hkw(ax, col, legend):
    """ AT 2023hkw 
    Epoch 1: 44d: 86 uJy
    Epoch 2 60d: 152 uJy (Ku band)
    Epoch 3 26 Aug; 150d: 83 uJy (X band) """
    zval = 0.339
    dt = np.array([44, 64, 117])/(1+0.339)
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    f = np.array([0.077, 0.152, 0.083]) * \
            1E-3 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(dt, f, c=col, marker=vals.markers['AT2023hkw'], s=30, zorder=5)
    
    ax.plot(dt, f, c=col, lw=2, zorder=5)
    ax.text(
            dt[0]/1.1, f[0], "AT2023hkw", 
            fontsize=8, horizontalalignment='right',
            verticalalignment='top', color=col, fontweight='bold')



def at2023vth(ax, col, legend):
    z = 0.0747
    x = np.array([30, 41, 87, 118, 204, 338, 402])/(1+z)
    dcm = Planck18.luminosity_distance(z).cgs.value
    y_ujy = np.array([145, 399, 1290, 769, 56, 12, 8])
    y = y_ujy * 1E-6 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(x[:-2], y[:-2], fc=col, ec='black', marker=vals.markers['AT2023vth'], s=30, zorder=100)
     # Plot upper limits with arrows
    ax.scatter(x[-2:], y[-2:], edgecolor=col, marker=vals.markers['AT2023vth'], s=30, zorder=5, facecolor='white')
    for (t_upper, y_upper) in  zip(x[-2:], y[-2:]):
        ax.arrow(t_upper, y_upper, 0, -y_upper/2, length_includes_head=True, 
             head_width=t_upper/7, head_length=y_upper/8, color=col, zorder=200)
    ax.plot(x[:-2], y[:-2], c=col, lw=2, zorder=5)
    ax.plot(x[-3:], y[-3:], c=col, lw=2, zorder=5, ls='dotted')
    ax.text(
            x[-3]*1.1, y[-3]/1.5, "AT2023vth", 
            fontsize=8, horizontalalignment='right',
            verticalalignment='top', color=col,zorder=1000, fontweight='bold')


def at2024aehp(ax, col, legend):
    z = 0.1715
    x = np.array([8,82, 86, 140.78])/(1+z)
    dcm = Planck18.luminosity_distance(z).cgs.value
    y_ujy = np.array([12, 26, 36, 321])
    y = y_ujy * 1E-6 * 1E-23 * 4 * np.pi * dcm**2 * 10E9
    ax.scatter(x[1:], y[1:], c=col, marker=vals.markers['AT2024aehp'], s=30, zorder=1000)
    ax.plot(x[1:], y[1:], c=col, lw=2, zorder=1000)
    ax.scatter(x[0], y[0], facecolor='white', edgecolor=col, marker=vals.markers['AT2024aehp'], s=30, zorder=1000)
    ax.plot(x[0:2], y[0:2], c=col, lw=2, zorder=1000, ls='dotted')
    ax.text(
            x[0], y[0]/1.2, "AT2024aehp", 
            fontsize=8, horizontalalignment='center',
            verticalalignment='top', color=col,zorder=1000, fontweight='bold')
    # 

def at2018cow(ax, col, legend):
    """ 231.5 GHz light curve and 9 GHz light curve """
    d = Planck18.luminosity_distance(z=0.014).cgs.value

    # low frequency
    nu = 9E9
    data_dir = "/Users/annaho/Dropbox/astro/papers/papers_complete/AT2018cow/data"
    dat = Table.read(
        "%s/radio_lc.dat" %data_dir, delimiter="&",
        format='ascii.no_header')
    tel = np.array(dat['col2'])
    choose = np.logical_or(tel == 'SMA', tel == 'ATCA')

    days = np.array(dat['col1'][choose])
    freq = np.array(dat['col3'][choose]).astype(float)
    flux_raw = np.array(dat['col4'][choose])
    flux = np.array(
            [float(val.split("pm")[0][1:]) for val in flux_raw])
    eflux_sys = np.array([0.1*f for f in flux])
    eflux_form = np.array(
            [float(val.split("pm")[1][0:-1]) for val in flux_raw])
    eflux = np.sqrt(eflux_sys**2 + eflux_form**2)
    choose = freq == 9

    # add the Margutti point and the Bietenholz point
    margutti_x = np.array([84,287])
    margutti_y = np.array([6E28, 3.2E26])/(4*np.pi*d**2)/1E-23/1E-3
    x = np.hstack((days[choose], margutti_x))/(1+0.014)
    y = np.hstack((flux[choose], margutti_y)) * nu
    lum = plot_line(
            ax, d, x, y,
            'AT2018cow', None, col, legend, zorder=3, alpha=0.3)
    #ax.text(x[-1], lum[-1]/1.5, 'AT2018cow', fontsize=med,
    #        verticalalignment='top',
    #        horizontalalignment='center', c=col)


def css(ax, col, legend):
    """ 6 GHz light curve """
    d = Planck18.luminosity_distance(z=0.034).cgs.value

    # low frequency
    nu = 6E9

    # add the points from Deanne's paper
    x = np.array([69, 99, 162, 357])/(1+0.034)
    y = np.array([4.5, 6.1, 2.3, 0.07])*nu
    lum = plot_line(
            ax, d, x, y,
            'AT2018cow', None, col, legend, zorder=0)



def tde(ax, col, legend):
    """  
    Plot the 4.9 GHz light curve from the VLA
    """
    z = 0.354
    d = Planck18.luminosity_distance(z=z).cgs.value

    # Need to add 3.04 to the Zauderer points
    nu, dt, f, ef, islim = zauderer()
    t = (dt+3.04)/(1+z)

    # Low frequency
    nu_plt = 4.9E9
    choose = np.logical_and(~islim, nu == nu_plt/1E9)
    dt_all = t[choose]
    nufnu_all = nu_plt*f[choose]

    # adding the set from Berger2012
    # and making the same correction as above
    # this is 4.9 GHz
    t = (np.array([3.87, 4.76, 5.00, 5.79, 6.78, 7.77, 9.79, 14.98, 22.78,
        35.86, 50.65, 67.61, 94.64, 111.62, 126.51, 143.62, 164.38, 174.47,
        197.41, 213.32])) / (1+z)
    f = np.array([0.25, 0.34, 0.34, 0.61, 0.82, 1.48, 1.47, 1.80, 2.10, 4.62,
        4.84, 5.86, 9.06, 9.10, 9.10, 11.71, 12.93, 12.83, 13.29, 12.43])

    # adding the set from Zauderer2013
    # they also say it's relative to March 25.5...
    # so I think I need to subtract 3.04 days from here too
    t = (np.array([245.23, 302.95, 383.92, 453.66, 582.31]))/(1+z)
    f = np.array([12.17, 12.05, 12.24, 11.12, 8.90])
    dt_all = np.append(dt_all, t)
    nufnu_all = np.append(nufnu_all, f*nu_plt)

    # adding the set from Eftekhari 2024
    t = np.array([645, 651.1, 787.6, 1032, 1105, 1373, 1894])/(1+z)
    f = np.array([8.24, 8.63, 6.23, 4.21, 3.52, 2.34, 1.47])
    dt_all = np.append(dt_all, t)
    nufnu_all = np.append(nufnu_all, f*nu_plt)

    order = np.argsort(dt_all)
    lum = plot_line(
            ax, d, dt_all[order], nufnu_all[order], 
            'SwiftJ1644+57', 'TDE', col, legend)
    ax.text(dt_all[order][10], lum[10]*1.1, 'Swift J1644+57', fontsize=small,
            verticalalignment='bottom',
            horizontalalignment='left')



def grb030329(ax, col, legend):
    """ 
    Berger 2003
    Van der Horst et al. 2008
    
    Explosion day was obviously 03/29
    """
    z = 0.1686
    d = Planck18.luminosity_distance(z=z).cgs.value

    # LOW FREQUENCY

    # Berger: this is the best frequency to pick from this paper
    t = np.array(
            [0.58, 1.05, 2.65, 3.57, 4.76, 6.89, 7.68, 9.49, 11.90, 
                12.69, 14.87, 16.66, 18.72, 20.58, 25.70, 28.44, 31.51, 
                33.58, 36.52, 42.55, 44.55, 59.55, 66.53]) / (1+z)
    f = np.array(
            [3.50, 1.98, 8.50, 6.11, 9.68, 15.56, 12.55, 13.58, 17.70, 
                17.28, 19.15, 17.77, 15.92, 16.08, 15.34, 12.67, 13.55, 
                13.10, 10.64, 8.04, 8.68, 4.48, 4.92])
    nu = np.array([8.5E9]*len(f))

    # Van der Horst: best frequency is 2.3 GHz
    t = np.append(t, np.array([268.577, 306.753, 365.524, 420.168, 462.078, 
        583.683, 743.892, 984.163]) / (1+z))
    f = np.append(
            f, np.array([1613, 1389, 871, 933, 707, 543, 504, 318]) * 1E-3)
    nu = np.append(nu, np.array([2.3E9]*8))
    lum = plot_line(ax, d, t, nu*f, 'GRB030329', 'GRB', col, legend)

    

def grb111209a(ax, col, legend):
    """ 
    Hancock+ 2012, GCN 12804
    """
    z = 0.677
    d = Planck18.luminosity_distance(z=z).cgs.value

    t = np.array([5.1])/(1+z)
    f = np.array([0.97])
    nu = np.array([9E9]*len(f))

    lum = plot_line(ax, d, t, nu*f, 'GRB111209A', 'GRB', col, legend)
    ax.scatter(t,lum,marker='*',c='k',s=80)


def grb130427A(ax, col, legend):
    """ Perley et al 2013.
    They have data from CARMA/PdBI at 90 GHz (3mm)
    But by the time they caught it, it was fading
    """
    z = 0.340
    d = Planck18.luminosity_distance(z=z).cgs.value

    freq = 5.10E9
    t = np.array([0.677, 2.04, 4.75, 9.71, 17.95, 63.78, 128.34]) / (1+z)
    f = np.array([1290, 1760, 648, 454, 263, 151, 86]) * 1E-3

    freq = 6.8E9
    t = np.array([0.677, 2.04, 4.75, 9.71, 9.95, 12.92, 27.67, 59.8, 128]) / (1+z)
    f = np.array([2570, 1820, 607, 374, 385, 332, 243, 109, 91]) * 1E-3

    lum = plot_line(ax, d, t/(1+0.34), freq*f, 'GRB130427A', 'GRB', col, legend)
    #ax.text(3, 5E39, 'GRB130427A', ha='left', color='grey')


def sn2007bg(ax, col, legend):
    """ Salas et al. 2013
    Peak is resolved for 4.86, 8.46 GHz
    """
    nu = 8.46E9
    d = Planck18.luminosity_distance(z=0.0346).cgs.value
    t = np.array(
            [13.8, 19.2, 26.1, 30.9, 41.3, 55.9, 66.8, 81.8, 98.8, 124, 
                144, 159.8, 189.9, 214.9, 250.9, 286.8, 314.8, 368.8, 
                386.8, 419.9, 566.9, 623.8, 720.8, 775.8, 863.8])/(1+0.0346)
    f = np.array(
            [480, 753, 804, 728, 1257, 1490, 1390, 1325, 1131, 957, 
                621, 316, 379, 404, 783, 1669, 2097, 2200, 
                2852, 3344, 3897, 3891, 3842, 3641, 3408]) * 1E-3
    lum = plot_line(ax, d, t, nu*f, 'SN2007bg', 'SN', col, legend)


def sn2003bg(ax, col, legend):
    """ Soderberg et al. 2006
    Peak is resolved for 22.5, 15, 8.46, 4.86, 1.43
    Again, there are two peaks...
    Let's choose the first peak, 8.46
    """
    nu = 8.46E9
    d = 6.056450393620008e+25

    t = np.array(
                [10, 12, 23, 35, 48, 58, 63, 73, 85, 91, 115, 129,
                132, 142, 157, 161, 181, 201, 214, 227, 242, 255,
                266, 285, 300, 326, 337, 351, 368, 405, 410, 424,
                434, 435, 493, 533, 632, 702, 756, 820, 902, 978])/(1+0.00441483)
    f = np.array(
                [2.51, 3.86, 12.19, 24.72, 40.34, 51.72, 49.64, 46.20,
                38.638, 33.85, 45.74, 53.94, 54.27, 54.83, 48.43,
                47.43, 35.76, 31.35, 28.67, 27.38, 24.57, 22.30,
                21.67, 21.31, 20.88, 20.33, 19.85, 18.84, 17.14,
                14.61, 14.49, 14.16, 13.25, 13.08, 10.04, 8.92,
                6.23, 6.18, 4.62, 3.93, 4.69, 4.48])
    lum = plot_line(ax, d, t, nu*f, 'SN2003bg', 'SN', col, legend, zorder=0)



def sn2009bb(ax, col, legend):
    """ expl date Mar 19, Soderberg et al. 2010"""

    nu = 8.46E9
    d = 1.237517263280789e+26
    t_apr = 11 + np.array([5.2, 8.2, 13.2, 15.1, 23.2, 29.1])
    t_may = 11 + 30 + np.array([3.1, 10.1, 13, 20.1, 27])
    t_jun = 11 + 30 + 31 + np.array([6, 17, 26])
    t_jul = 11 + 30 + 31 + 30 + np.array([18.9])
    t_aug = 11 + 30 + 31 + 30 + 31 + np.array([11.8])
    t = np.hstack((t_apr, t_may, t_jun, t_jul, t_aug))/(1+0.00896492)
    flux = np.array([24.681, 17.568, 16.349, 13.812, 8.881,
        7.714, 8.482, 6.824, 6.327, 3.294, 4.204, 3.203, 2.392,
        1.903, 1.032, 1.084])
    lum = plot_line(ax, d, t, nu*flux, 'SN2009bb', 'SN', col, legend) #Changed from Rel SN.
    #ax.text(t[0]/1.05, lum[0], '2009bb', fontsize=11,
    #        verticalalignment='center',
    #        horizontalalignment='right')


def sn1998bw(ax, col, legend):
    """ SN 1998bw
    
    This is a bit complicated because there are two peaks in the light curve,
    but I am using a frequency that has a main peak rather than a frequecy
    with two clear distinct peaks...
    """
    d = 1.17E26 # cm
    nu = 150E9
    t = np.array([12.4])/(1+0.0087)
    f = np.array([39])
    nu = 2.3E9
    t = np.array([11.7, 14.6, 15.7, 16.5, 17.8, 19.7, 21.6, 23.6, 25.9, 26.8, 28.8, 30.0, 32.9, 34.7, 36.8, 38.8, 40.0, 45.7, 51.7, 57.7, 64.7, 67.7, 80.5])
    f = np.array([19.7, 22.3, 23.5, 23.9, 25.1, 25.3, 20.9, 22.9, 28.0, 28.7, 31.1, 31.3, 27.3, 33.5, 31.8, 31, 31.3, 26.8, 23.1, 18.5, 15.6, 15.6, 9.6])
    lum = plot_line(ax, d, t, nu*f, 'SN1998bw', 'GRB', col, legend)




def limits(ax):
    """ VLASS limits for FBOTs """

    # ZTF19abyjzvd
    z = 0.137
    t = np.array([26, 181])
    f = np.array([5*3,5*3]) #uJy (RMS) times 3
    nu = 10E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(
            t/(1+z), lum, marker='*', c='k', s=100,
            label='Fast-luminous transients', zorder=5)
    ax.plot(
            t/(1+z), lum, c='k', ls='--', zorder=5)
    for ii,tval in enumerate(t):
        ax.arrow(
                tval/(1+z), lum[ii], 0, -lum[ii]/2, color='k',
                length_includes_head=True, head_width=tval/5, head_length=lum[ii]/5)
    ax.text(
            t[0]/1.3, lum[0], "ZTF19abyjzvd", color='k',
            fontsize=11, horizontalalignment='right')

    # iPTF15ul
    z = 0.0657
    t = 5
    f = 12*3 #uJy (RMS) times 3
    nu = 6E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(t/(1+z), lum, marker='*', c='k', label='_none', s=80)
    ax.arrow(
            t/(1+z), lum, 0, -lum/2, color='k', 
            length_includes_head=True, head_width=1, head_length=lum/5)
    ax.text(t/1.2, lum, "iPTF15ul", fontsize=small, horizontalalignment='right')

    # Dougie
    z = 0.19
    t = 3736
    f = 93 # uJy
    nu = 3E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(t/(1+z), lum, marker='*', c='k', label="_none", s=80)
    ax.arrow(
            t/(1+z), lum, 0, -lum/2, color='k', 
            length_includes_head=True, head_width=700, head_length=lum/5)
    ax.text(t/1.2, lum*1.2, "Dougie", fontsize=small, horizontalalignment='center',
            verticalalignment='bottom')


    # 05D2bk
    z = 0.699
    t = 4739
    f = 134 # uJy
    nu = 4E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    print(t/(1+z), lum)
    ax.scatter(t/(1+z), lum, marker='*', c='k', s=80)
    ax.arrow(
            t/(1+z), lum, 0, -lum/2, color='k', 
            length_includes_head=True, head_width=t/7, head_length=lum/5)
    ax.text(t/(1+z), lum*1.2, "05D2bk", fontsize=small, horizontalalignment='center',
            verticalalignment='bottom')

    # DES16X1eho
    z = 0.593
    t = 365
    f = 152 # uJy
    nu = 4E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(t/(1+z), lum, marker='*', c='k', s=80)
    ax.text(t*1.1/(1+z), lum, "DES16X1eho", fontsize=small, 
            horizontalalignment='left')
    ax.arrow(
            t/(1+z), lum, 0, -lum/2, color='k', 
            length_includes_head=True, head_width=50, head_length=lum/5)\

    # 06D1hc
    z = 0.555
    t = 4034
    f = 136 # uJy
    nu = 4E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(t/(1+z), lum, marker='*', c='k', s=80)
    ax.arrow(
            t/(1+z), lum, 0, -lum/2, color='k', 
            length_includes_head=True, head_width=t/7, head_length=lum/5)
    ax.text(t/(1+z), lum/2, "06D1hc", fontsize=small, 
            horizontalalignment='center', verticalalignment='top')

    # 16asu
    z = 0.187
    t = np.array([43.7, 255])
    f = np.array([17, 17])
    nu = 6.2E9
    dcm = Planck18.luminosity_distance(z=z).cgs.value
    lum = f*1E-6 * 1E-23 * 4 * np.pi * dcm**2 * nu
    ax.scatter(t/(1+z), lum, marker='*', c='k', zorder=3, s=80)
    ax.plot(t/(1+z), lum, c='k', ls='--', zorder=3)
    ax.text(t[0]*1.1/(1+z), lum[0], "iPTF16asu", fontsize=small, 
            horizontalalignment='left', va='bottom')
    for ii,tval in enumerate(t):
        ax.arrow(
                tval/(1+z), lum[ii], 0, -lum[ii]/2, color='k', 
                length_includes_head=True, head_width=tval/5, 
                head_length=lum[ii]/5)


if __name__=="__main__":
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5), sharex=True, sharey=True)
    props = dict(boxstyle='round', facecolor='white')

    bk = 'lightgrey'


    ax.axhline(y=1E37, c='k', ls='--', lw=0.5)
    ax.text(
            35,8E36,'Ordinary SNe',
            va='top',style='italic',color='grey',fontsize=10)
    ax.text(35,1.6E37,'Dense CSM', 
            va='top',ha='left', style='italic',color='grey',fontsize=10)
    #ax.axhline(y=1E39, c='k', ls='--', lw=0.5)
    #ax.text(
    #        38,7.5E40,'Relativistic explosions', 
    #        va='bottom', style='italic',color='grey',fontsize=10)

    sn2007bg(ax, vals.sn_col, 'SN')
    sn2003bg(ax, vals.sn_col, None)
    sn2009bb(ax, vals.sn_col, None)

    grb030329(ax, vals.lgrb_col, None)
    grb130427A(ax, vals.lgrb_col, 'LGRB')

    
    sn1998bw(ax, vals.llgrb_col, 'LLGRB')
    
    #at2018cow(ax, vals.fbot_col, None) Currently don't have radio data for this
    css(ax, vals.fbot_col, 'LFBOT')

    koala(ax, vals.fbot_col, None)
    at2020mrf(ax, vals.fbot_col, None)
    at2020xnd(ax, vals.fbot_col, None)
    at2021ahuo(ax, vals.fbot_col, None)

    at2022tsd(ax, vals.fbot_col, None)
    at2024aehp(ax, vals.fbot_col, None)
    at2022abfc(ax, vals.colors['AT2022abfc'], None)
    

    at2023vth(ax, vals.colors['AT2023vth'], None)
    at2023fhn(ax, vals.colors['AT2023fhn'], None)
    at2023hkw(ax, vals.colors['AT2023hkw'], None)
    at2024aehp(ax, vals.colors['AT2024aehp'], None)

    ax.set_ylabel(
            r"Luminosity $\nu L_{\nu}$ [erg s$^{-1}$] (10 GHz)", 
            fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(4, 1800) 
    ax.set_ylim(4E36, 2E41)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("$t_\mathrm{rest}$ (d)", fontsize=16)

    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig4_radio_collage.pdf", dpi=450)
    plt.show()
    plt.close()
