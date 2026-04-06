#!/usr/bin/env python
# coding: utf-8

# X-ray LC compilation (original code from Yuhan Yao) 

import numpy as np
import pandas as pd

from astropy.time import Time
import astropy.constants as const


from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from astropy.cosmology import Planck18
 
import vals


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['font.size']=10
from matplotlib import ticker
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'




from fbot_xdata import cow_xrt_lc




from load_grb_xlc import add_grb_lcs, add_xlc_sn1998bw, add_xlc_sn2010dh, \
                        add_xlc_sn2006aj, add_xlc_sn2003dh



def add_at2022cmc(ax, color = "tab:darkgrey", ms=9):
    z = 1.1933
    a = pd.read_csv("data/xray/at2022cmc_nicer.dat", delimiter=' ')
    ax.plot(a['x'], a['L'], color=color, lw=0.5, zorder=10, ls='-.')#, label='AT2022cmc')
    a = pd.read_csv("data/xray/at2022cmc_xrt.dat", delimiter=' ')
    ax.plot(a['x'], a['L'], color=color, lw=0.5, zorder=10, ls='-.', label='TDE',)


def add_cow(ax, color = "k"):
    tbx = cow_xrt_lc()
    
    t0 = np.hstack([tbx["phase"].data, np.array([78.1, 211.8])])/(1+0.1353)
    L0 = np.hstack([tbx["L_XRT"].data, np.array([1e+40, 7.1e+39])])
    eL0_left = np.hstack([tbx["L_XRT_unc_left"].data, np.array([2e+39, 2e+39])])
    eL0_right = np.hstack([tbx["L_XRT_unc_right"].data, np.array([1e+39, 2e+39])])

    #ax.errorbar(t0, L0, yerr = [eL0_left, eL0_right], 
    #            fmt = "D-", color = color, markersize=3, zorder = 3, linewidth = 1)
    ax.scatter(t0, L0, marker='D', s=10, color=color, zorder=3, alpha=0.3)
    ax.plot(t0, L0, color=color, zorder=3, lw=0.5, alpha=0.3)
    ax.scatter(0,0,marker='D', s=10, label='LFBOTs',color=color, alpha=0.3)
    
#     ix = (t0>34.5)&(t0<37.5)
#     subL0 = L0[ix]
#     subL0_unc = eL0_left[ix]/2 + eL0_right[ix]/2
#     subweight= 1 / subL0_unc**2
#     subl0 = np.average(subL0, weights = subweight)
#     subl0_unc = 1 / np.sqrt(np.sum(subweight))
#     print ("AT2018cow at ~ 36 days luminosity: %.4f +- %.4f e+42 erg/s"%(subl0/1e+42, subl0_unc/1e+42))
    
    
def add_css(ax, color = "cyan"):
    t2 = np.array([99, 130, 291])/(1+0.034)
    # although I don't quite believe in the Coppejans+2020 analysis.... I will use the numbers
    f2 = np.array([1.33e-15, 1.94e-15, 1.31e-15])
    ef2 = np.array([0.76e-15, 0.97e-15, np.nan])
    distance_cm = cosmo.luminosity_distance(0.034).value*1e+6 * const.pc.cgs.value
    L2 = f2 * (4*np.pi*distance_cm**2)
    eL2 = ef2 * (4*np.pi*distance_cm**2)

    ax.scatter(t2[:2], L2[:2], color = color, marker= "D", s=10)
    ax.errorbar(t2[:2], L2[:2], yerr=eL2[:2],
                ecolor=color, elinewidth=0.7, fmt='none', zorder=50)
    #ax.plot(t2[:2], L2[:2], color = color)
    ax.scatter(t2[2:], L2[2:], marker = "v", color = color, s=10)

    dcm = Planck18.luminosity_distance(z=0.034).cgs.value
    dt = np.array([25.14, 30.17, 33.06])/(1+0.034)
    L = np.array([1.8*10**(-13), 1.2*10**(-13), 2*10**(-13)])*4*np.pi*dcm**2

    all_data=(np.append(dt, t2[:1]), np.append(L, L2[:1]))

    # Plot disjointed lines
    ax.plot(all_data[0], all_data[1], ls='--', c=color)
    ax.plot(t2[1:], L2[1:], ls='--', color=color)
    ax.plot(t2[0:2], L2[0:2], ls='-', color=color)

    ax.scatter(dt[1:], L[1:], marker='D', edgecolor=color, s=10, facecolor='white', zorder=50)
    for (t_upper, y_upper) in  zip(dt[1:], L[1:]):
        ax.arrow(t_upper, y_upper, 0, -y_upper/1.4, length_includes_head=True, 
             head_width=t_upper/10, head_length=y_upper/8, color=color, zorder=200)
    ax.scatter(dt[0], L[0], marker='D', color=color, s=10, zorder=4)
    ax.errorbar(dt[0], L[0], yerr=[[2.4e41], [3.6e41]],
                ecolor=color, elinewidth=0.7, fmt='none', zorder=50)
    ax.text(t2[0]/2, L2[0]*1.8, 'CSS161010', c=color, ha='center', fontsize=7, zorder=100, fontweight='bold')



def add_20xnd(ax, color = "orange"):
    tdis = 59132
    mjd1 = np.array([59157.8, 59163.8, 59179.1, 59207.2, 59316.6, 59317.1])
    distance_cm = cosmo.luminosity_distance(0.243).value*1e+6 * const.pc.cgs.value
    t1 = (mjd1-tdis) / (1+0.243)
    f1 = np.array([3.46e-14, 2.79e-14, 0.15e-14, 0.24e-14, 0.20e-14, 0.24e-14])
    ef1_right = np.array([0.96e-14, 0.75e-14, 0.17e-14, np.nan, np.nan, np.nan])
    ef1_left = np.array([1.27e-14, 0.67e-14, 0.11e-14, np.nan, np.nan, np.nan])
    L1 = f1 * (4*np.pi*distance_cm**2)
    eL1_right = ef1_right * (4*np.pi*distance_cm**2)
    eL1_left = ef1_left * (4*np.pi*distance_cm**2)
    
    ax.scatter(t1[:3], L1[:3], marker='D', color=color, zorder=10, s=10, alpha=0.3)
    # Don't show errors for other LFBOTs
    #ax.errorbar(t1[:3], L1[:3], [eL1_left[:3], eL1_right[:3]], fmt = "o-", markersize=4.5, 
    #            color = color, zorder = 4, label = "AT2020xnd")
    ax.plot(t1[:3], L1[:3], color=color, zorder=4, lw=0.5, alpha=0.3)
    ax.plot(t1[3:5], L1[3:5], marker = "v", markersize=3, alpha=0.3, color = color, linestyle = "none", zorder = 2)



def add_2020mrf(ax, color = "tab:red", ms=9):
    z = 0.1353
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    t0 = 59012  
    D_cm = D*const.pc.cgs.value

    #tsrg_e1_mjd = Time(["2020-01-19T10:41:10.661","2020-01-22T14:41:21.682"]).mjd
    #srg1_phase_obs = (tsrg_e1_mjd - t0)/(1+z)
    
    # eRASS2 detection
    tsrg_e2 = np.array([35, 36, 37])
    LX_20mrf = 3.9039e-13 * 4 * np.pi * D_cm**2
    LX_20mrf_unc_right = 4.786e-13 * 4 * np.pi * D_cm**2 - LX_20mrf
    LX_20mrf_unc_left = LX_20mrf - 3.110e-13 * 4 * np.pi * D_cm**2
    frac = np.array([0.05309, 0.32039, 0.05112])
    meancr = 0.10891
    lsrg_e2 = LX_20mrf / meancr * frac
    lsrg_e2_unc_right = LX_20mrf_unc_right/ meancr * frac
    lsrg_e2_unc_left = LX_20mrf_unc_left/ meancr * frac
    
    
    print ("AT2020mrf eRASS2 Luminosity = %.2f - %.2f + %.2f e+43 erg/s"%(LX_20mrf/1e+43, 
                                                                          LX_20mrf_unc_left/1e+43, 
                                                                          LX_20mrf_unc_right/1e+43))
    # Two Chandra obsIDs
    tc1 = 327.4
    tc2 = 328.2
    Lc1 = 4.00e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    Lc2 = 1.57e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    Lc1_unc_right = 0.68e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    Lc1_unc_left = 1.24e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    Lc2_unc_right = 0.27e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    Lc2_unc_left = 0.49e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    
    Lc_mean = np.mean([Lc1, Lc2])
    
    lnL_ratio = np.log(LX_20mrf / Lc_mean)
    lnt_ratio = np.log(328 / 36)
    lnL_ratio  / lnt_ratio

    ts = np.hstack([tsrg_e2, np.array([tc1, tc2]) ])
    Ls = np.hstack([lsrg_e2, np.array([Lc1, Lc2]) ])
    eLs_right = np.hstack([lsrg_e2_unc_right, np.array([Lc1_unc_right, Lc2_unc_right]) ])
    eLs_left = np.hstack([lsrg_e2_unc_left, np.array([Lc1_unc_left, Lc2_unc_left]) ])
    lw = 0.8
    #ax.errorbar(ts[:3], Ls[:3], yerr = [eLs_left[:3], eLs_right[:3]], marker='*', color = color, 
    #            label = "AT2020mrf", markersize = ms+2, linestyle = "-", linewidth = lw,
    #            elinewidth = 1.5,
    #            markeredgecolor = "k", markeredgewidth = 0.5, ecolor = "k")
    ax.scatter([36/(1+0.1353)], [LX_20mrf], marker='D', color=color, s=10, zorder=4, alpha=0.3)
    ax.scatter(ts[3:]/(1+0.1353), Ls[3:], marker='D', color = color, zorder=4, s=10, alpha=0.3)
    
    # eRASS3 upper limit
    tsrg_e3_mjd = Time(["2021-01-15T04:59:44", "2021-01-27T05:00:14"]).mjd
    srg3_phase_obs = (tsrg_e3_mjd - t0)#/(1+z)
    tsrg_e3 = np.median(srg3_phase_obs)
    LXupp_20mrf_e3 = 7.24e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    #ax.scatter(tsrg_e3, LXupp_20mrf_e3, marker = "v", color=color, s=10)
    
    # eRASS4 upper limit
    tsrg_e4_mjd = Time(["2021-07-15T00:00:00", "2021-07-27T00:00:00"]).mjd
    srg4_phase_obs = (tsrg_e4_mjd - t0)#/(1+z)
    tsrg_e4 = np.median(srg4_phase_obs)
    LXupp_20mrf_e4 = 8.26e-14 * 4 * np.pi * (D * const.pc.cgs.value)**2
    #ax.scatter(tsrg_e4, LXupp_20mrf_e4, marker = "v", color=color, s=15)
    
    #ax.plot([36,tsrg_e3,ts[3],ts[4],tsrg_e4], 
    #        [LX_20mrf,LXupp_20mrf_e3,Ls[3],Ls[4],LXupp_20mrf_e4],
    #        ls='-',color=color, lw=0.5)
    ax.plot(np.array([36,ts[3],ts[4]])/(1+0.1353), 
            [LX_20mrf,Ls[3],Ls[4]],
            ls='-',color=color, lw=0.5, alpha=0.3)
        
    ax.plot(np.array([ts[3],ts[4]])/(1+0.1353), [Ls[3],Ls[4]],ls='-',color=color, lw=0.5, alpha=0.3)




def add_afterglows(ax, color='k', ms=9):
    
    # ZTF19abvizsw
    z = 1.2596
    ax.plot(np.array([1.9,9.9])/(1+1.2596),[2.5E46, 3.8E45],c=color, zorder=2)
    ax.scatter(np.array([1.9,9.9])/(1+1.2596),[2.5E46, 3.8E45],marker='o', c=color, zorder=2)
    
    # ZTF20aajnksq
    z = 2.9
    ax.scatter(2.1/(1+2.9),9.9E45,c=color,marker='o')
    ax.plot(np.array([2.1,3.4])/(1+2.9), [9.9E45, 9.9E45], c=color, ls='--', zorder=2)
    ax.scatter(3.4/(1+2.9),9.9E45,marker='v',c=color, zorder=2)
    
    # ZTF21aayokph
    z = 1.0624
    ax.scatter(1.94/(1+1.0624), 2.2E45, marker='o', c=color, zorder=2)
    ax.plot(np.array([1.94,4.39])/(1+1.0624), [2.2E45, 1.0E45], c=color, ls='--', zorder=2)
    ax.scatter(4.39/(1+1.0624), 1.0E45, marker='v', c=color, zorder=2)
    
    # ZTF21aaeyldq
    z = 2.5131
    ax.scatter(0.83/(1+2.5131), 1.7E46, marker='o', c=color, label="ZTF Afterglows", zorder=2)
    ax.plot(np.array([0.83, 3.80])/(1+2.5131), [1.7E46, 1.1E46], c=color, ls='--', zorder=2)
    ax.scatter(3.80/(1+2.5131),1.1E46,c=color,marker='v', zorder=2)




from get_xray import load_swift, load_chandra, load_both
def add_22tsd(ax, color = "red"):
    dt, L, e_dt, eL = load_both()
    dt=dt/(1+0.2564)
    order = np.argsort(dt)
    ratio = L[order]/eL[order]
    choose = ratio > 1
    ax.errorbar(dt[order][choose], L[order][choose], yerr=eL[order][choose], color = color, marker = "D", markersize = 3,
            lw=0.5, zorder=200, alpha=0.3)     

def add_24wpp(ax, color='red'):
    # Nayana et al. 2025 
    zval=0.0868
    dcm = Planck18.luminosity_distance(z=zval).cgs.value
    #dt = np.array([np.average([2.1,5]), np.average([5,9]), np.average([9,15]), np.average([17.4,18.6]),
    #               np.average([21.3, 34.5]), np.average([49.5, 52.8]), np.average([75,76.6]),
    #               np.average([98.9,99.5]), np.average([139.7,140.2])])/(1+zval)
    #L = np.array([7.9, 7.2, 2.9, 0.93, 0.34, 0.67, 0.81, 0.12, 0.14])*10**(-13)*4*np.pi*dcm**2
    

    # Perley et al. 2026
    dt = (np.array([60580.409, 60580.4835, 60580.681500000006, 60580.741500000004, 60582.15, 60583.7135, 60584.45, 60585.5965, 60587.8195, 60589.6605, 60590.4415, 60590.5185, 60590.744, 60591.857, 60592.4755, 60595.423500000004, 60625.7435, 60631.966, 60637.4225, 60663.32])-60578.4)/(1+zval)
    L = np.array([129.1, 85.6, 58.1, 109.6, 135.1, 48.3, 77.5, 71.2, 47.8, 20.9, 30.6, 17.3, 9.4, 17.5, 14.1, 16.9, 14.0, 12.9, 10.1, 11.1])*10**(-14)*4*np.pi*dcm**2

    ax.scatter(dt, L, marker='D', color=color, s=10, zorder=4, alpha=0.3)
    ax.plot(dt, L, color=color, zorder=4, lw=0.5, alpha=0.3)


def add_22abfc(ax, color='blue'):
    dcm = Planck18.luminosity_distance(z=0.212).cgs.value
    dt = 17/(1+0.212)
    # convert from erg/cm/cm/s to luminosity
    L = 2*10**(-13)*4*np.pi*dcm**2
    ax.scatter(dt, L, marker=vals.markers['AT2022abfc'], s=30,facecolor='white', edgecolor=color, zorder=100)
    ax.text(1.1*dt, L/2.2, 'AT2022abfc', c=color, ha='left', fontsize=11, zorder=100, fontweight='bold')
    ax.arrow(dt, L, 0, -L/1.4, length_includes_head=True, 
             head_width=dt/10, head_length=L/8, facecolor='white', edgecolor=color, zorder=200)



def add_23fhn(ax, color='blue'):
    marker=vals.markers['AT2023fhn']
    dcm = Planck18.luminosity_distance(z=0.2377).cgs.value
    dt = np.array([14.78957, 15, 28, 63, 210])/(1+0.2377)
    L = np.array([7.6*10**(-15), 1.7*10**(-13), 4.5*10**(-16), 8.2*10**(-16), 3.5*10**(-16)])*4*np.pi*dcm**2
    # New Values from Nayana et al.
    dt = np.array([12.421751175350732*(1+0.2377), 15, 23.808459838163706*(1+0.2377), 51.84196331927669*(1+0.2377), 210])/(1+0.2377)
    L = np.array([ 3.788e42/(4*np.pi*dcm**2), 1.7*10**(-13), 617.1377204879623e39/(4*np.pi*dcm**2), 605.3003269852896e39/(4*np.pi*dcm**2), 3.5*10**(-16)])*4*np.pi*dcm**2
    low_err = L[[0,2,3]] - np.array([428.6524566284672e39,332.97515343671637e39,350.4361057720496e39])
    up_err = np.array([8963.396428135813e39,895.0755220357021e39,817.9128005984965e39])- L[[0,2,3]]
    print(np.array([8963.396428135813e39,895.0755220357021e39,817.9128005984965e39]))
    print(L[[0,2,3]])
    L_err = [low_err, up_err]
    print(L_err)
    ax.plot(dt, L, marker=None, c=color, ls='--')
    ax.scatter(dt[[1, 4]], L[[1, 4]], s=30, marker=marker, facecolor='white', edgecolor=color, zorder=100)
    for (t_upper, y_upper) in  zip(dt[[1, 4]], L[[1, 4]]):
        ax.arrow(t_upper, y_upper, 0, -y_upper/1.4, length_includes_head=True, 
             head_width=t_upper/10, head_length=y_upper/8, facecolor='white', edgecolor=color, zorder=200)
    ax.scatter(dt[[0,2,3]], L[[0,2,3]], marker=marker, color=color, s=40, edgecolors = "k", zorder=50)
    ax.errorbar(dt[[0,2,3]], L[[0,2,3]], yerr=L_err,
                ecolor=color, elinewidth=0.7, fmt='none', zorder=50)
    ax.text(dt[0]/1.2, L[0]/2.5, 'AT2023fhn', c=color, ha='right', fontsize=11, zorder=100, fontweight='bold')



def add_23hkw(ax, color='blue'):
    marker=vals.markers['AT2023hkw']
    dcm = Planck18.luminosity_distance(z=0.339).cgs.value
    dt = 20/(1+0.339)
    L = 1.1*10**(-13)*4*np.pi*dcm**2
    ax.scatter(dt, L, s=30, marker=marker, facecolor='white', edgecolor=color, zorder=100)
    ax.arrow(dt, L, 0, -L/1.4, length_includes_head=True, 
             head_width=dt/10, head_length=L/8, facecolor='white', edgecolor=color, zorder=200)
    ax.text(dt*1.3,1.5*L, 'AT2023hkw', c=color, ha='left', fontsize=11, zorder=100, fontweight='bold')


def add_23vth(ax, color='blue'):
    marker=vals.markers['AT2023vth']
    dcm = Planck18.luminosity_distance(z=0.0747).cgs.value
    dt = 23/(1+0.0747)
    L = 1.6*10**(-13)*4*np.pi*dcm**2
    ax.scatter(dt, L, marker=marker, s=30, facecolor='white', edgecolor=color, zorder=100)
    ax.text(dt*1.2, L, 'AT2023vth', c=color, ha='left', fontsize=11, zorder=50, fontweight='bold')
    ax.arrow(dt, L, 0, -L/1.4, length_includes_head=True, 
             head_width=dt/10, head_length=L/8, facecolor='white', edgecolor=color, zorder=200)


def add_24qfm(ax, color='blue'):
    marker=vals.markers['AT2024qfm']
    # Give values and which obs are detections
    det_data=pd.read_csv('data/xray/AT2024qfm_Xray_det.csv')
    nondet_data=pd.read_csv('data/xray/AT2024qfm_Xray_nondet.csv')
    det_data['det']=True; nondet_data['det']=False

    dt=(np.concatenate((np.array(det_data['MJD_Mid']), np.array(nondet_data['MJD_Mid'])))-vals.t0['AT2024qfm'])/(1+0.227)
    f=np.concatenate((np.array(det_data['Flux']), np.array(nondet_data['UpperLimit_Flux'])))
    count=np.concatenate((np.array(det_data['Rate']), np.array(nondet_data['UpperLimit'])))
    dets=np.concatenate((np.array(det_data['det']), np.array(nondet_data['det'])))
    sort_ind=np.argsort(dt)
    dt=dt[sort_ind]; f=f[sort_ind]; dets=dets[sort_ind]; count=count[sort_ind]

    dcm = Planck18.luminosity_distance(z=0.227).cgs.value
    L_upper=np.array(det_data['FluxPos'])*4*np.pi*dcm**2; L_lower=np.array(det_data['FluxNeg'])*4*np.pi*dcm**2
    L=f*4*np.pi*dcm**2

    print(dt*(1+0.227))
    #Plot dashed and dotted lines
    det_lines=[[0,1],[8,9,10]]
    for det_line in det_lines:
        ax.plot(dt[det_line], L[det_line], marker=None, c=color)
    ax.plot(dt[1:9], L[1:9], marker=None, c=color, ls='dotted')
    ax.plot(dt[10:], L[10:], marker=None, c=color, ls='dotted')
    # Plot points
    ax.scatter(dt[dets], L[dets], marker=marker, c=color, s=30, edgecolor='k', zorder=200)
    ax.errorbar(dt[dets], L[dets], yerr=(np.abs(L_lower), L_upper),
                ecolor=color, elinewidth=0.7, fmt='none', zorder=50)
    ax.scatter(dt[~dets], L[~dets],marker=marker, s=30, facecolor='white', edgecolor=color, zorder=100)
    for (t_upper, y_upper) in  zip(dt[~dets], L[~dets]):
        ax.arrow(t_upper, y_upper, 0, -y_upper/1.4, length_includes_head=True, 
             head_width=t_upper/10, head_length=y_upper/8, facecolor='white', edgecolor=color, zorder=200)
    ax.text(dt[-2], L[-2]*18, 'AT2024qfm', c=color, ha='center', fontsize=11, zorder=100, fontweight='bold')

def add_24aehp(ax, color='blue'):
    marker=vals.markers['AT2024aehp']
    dcm = Planck18.luminosity_distance(z=0.1715).cgs.value
    dt = np.array([12, 15, 17])/(1+0.1715)
    L = np.array([2.3*10**(-13), 3.0*10**(-13), 2.4*10**(-13)])*4*np.pi*dcm**2
    print(L)
    ax.scatter(dt, L, marker=marker, s=30, facecolor='white', edgecolor=color, zorder=100)
    for (t_upper, y_upper) in  zip(dt, L):
        ax.arrow(t_upper, y_upper, 0, -y_upper/1.4, length_includes_head=True, 
             head_width=t_upper/10, head_length=y_upper/8, facecolor='white', edgecolor=color, zorder=200)
    ax.plot(dt, L, marker=None, c=color, ls='--')
    ax.text(dt[0]/5.3, L[0]/2.2, 'AT2024aehp', c=color, ha='left', fontsize=11, zorder=100, fontweight='bold')



def create_xray_panel(ax):
    """ Create a panel showing the X-ray LC comparison """
    sn_col = vals.sn_col
    llgrb_col = vals.llgrb_col
    lgrb_col = vals.lgrb_col

    # Old LFBOTs

    add_css(ax, color = vals.fbot_col) 
    add_20xnd(ax, color = vals.fbot_col)    
    add_2020mrf(ax, color = vals.fbot_col)
    add_22tsd(ax, color= vals.fbot_col)
    add_24wpp(ax, color= vals.fbot_col)
    # New lcs for Cassie
    add_22abfc(ax, color=vals.colors['AT2022abfc'])
    add_23fhn(ax, color=vals.colors['AT2023fhn'])
    add_23hkw(ax, color=vals.colors['AT2023hkw'])
    add_23vth(ax, color=vals.colors['AT2023vth'])
    add_24qfm(ax, color=vals.colors['AT2024qfm'])
    add_24aehp(ax, color=vals.colors['AT2024aehp'])


    add_grb_lcs(ax, color=lgrb_col)

    # GRB-SN
    add_xlc_sn1998bw(ax, color=llgrb_col) # Pian et al. 2000
    add_xlc_sn2010dh(ax, color=llgrb_col) # Fan et al. 2011
    add_xlc_sn2006aj(ax, color=llgrb_col) # Campana et al. 2006, soderberg et al. 2006
    add_xlc_sn2003dh(ax, color=llgrb_col) # Tiengo et al. 2004


    add_cow(ax, color = vals.fbot_col)

    ymax = 7e+47
    ymin = 8e+38
    xmin = 0.5
    xmax = 600

    xs = np.linspace(xmin, xmax)
    yccmax = 5e+39
    ax.fill_between(xs, ymin, yccmax, color = sn_col, zorder = 0, alpha = 0.8)
    ax.arrow(20, yccmax, 0, -yccmax*0.25, color = 'white', head_length = yccmax*0.15,
             head_width = 2.0)
    ax.text(15, yccmax*0.25, "Normal CCSNe", color = 'white')

    ax.legend(loc = "upper right", #bbox_to_anchor = (0, 0.2)
              ncol = 1, fontsize=10, handletextpad=0.1)

    ax.set_xlabel("$t_\mathrm{rest}$ (d)", fontsize=12)
    ax.set_ylabel("X-ray luminosity [erg s$^{-1}$] ($0.3-10\,$keV)", fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(1,600)
    ax.set_ylim(1E39,2E45)

fig, ax = plt.subplots(1, 1, figsize=(6,4.5), sharex=True, sharey=True)
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

create_xray_panel(ax)

plt.tight_layout()
plt.savefig('figures/fig3_x_collage.pdf', dpi=450)
plt.show()
plt.close()