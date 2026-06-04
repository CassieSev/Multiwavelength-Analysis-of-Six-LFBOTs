#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:57:42 2021

@author: yuhanyao
"""
import numpy as np
import astropy.io.ascii as asci
from astropy.table import Table, vstack
import astropy.constants as const

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

import matplotlib
import matplotlib.pyplot as plt
import vals
fs = 10
matplotlib.rcParams['font.size']=fs



def get_xlc_sn2010jl():
    """
    Table 5 of Chandra + 2015
    """
    filename = "SNe/SN2010jl/Chandra2015_tab5.dat"
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    tt = []
    ff = []
    eff = []
    
    for i in range(len(lines)):
        myline = lines[i]
        idt = 4
        if myline.split(" ")[4][0] in ["S", "N"]:
            idt = 5
        if myline.split(" ")[5][0] in ["X"]:
            idt = 6
        #print (myline.split(" ")[idt])
        tt.append(float(myline.split(" ")[idt]))
        newsubs = myline.split(" ")[idt+1:]
        if newsubs[0] == '±':
            idf = 2
        else:
            idf = 0
        ff.append(float(newsubs[idf][1:])*1e-13)
        eff.append(float(newsubs[idf+2][:-1])*1e-13)
        
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    #t = np.array([43.55. 53.03, 60.34])
    return df


def get_xlc_sn2005kd():
    """
    Dwarkadas+2016, Table 1
    
    conversion from 0.3--8 keV to 0.3--10 keV
    nh = 4e+21, APEC model, solar metalicity, 17 keV
    """
    multi = 1.18
    tt = np.array([440, 479, 504, 784, 
                   1015, 2200, 2419, 2940])
    ff = np.array([26, 49.6, 41.4, 44.6, 
                   19.86, 6.7, 3.35, 1.98]) * 1e-14 * multi
    eff_right = np.array([13, 27, 4.1, 25,
                         1.87, 0, 0.18, 0.44]) * 1e-14 * multi
    eff_left = np.array([11, 16.8, 9.4, 25.3, 
                        6.93, 0, 1.65, 0.36]) * 1e-14 * multi
    
    df = Table(data = [tt, ff, eff_right, eff_left],
               names = ["t", "f", "f+", "f-"])
    return df


def get_xlc_sn2006jd():
    """
    [1] Chandra+2012, Table 7, unabsorbed 0.2--10 keV flux, 
            multiply by 0.87 to get absorbed 0.3--10 keV flux
    [2] Katsuda+2016, Table 6, unabsorbed 0.2--10 keV flux,
            multiply by 0.87 to get absorbed 0.3--10 keV flux
    """
    multi = 0.87
    #z = 0.0186
    #D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    #D_cm = D * const.pc.cgs.value
    #scale2 = 4 * np.pi * D_cm**2
    tt = np.array([403.2, 431.5, 459.8, 496.2, 698.5, 907.7, 1063.6, 1067.5, 1609.8,
                   #564 / 1.015, 2940 / 1.015
                   ])
    ff = np.array([4.51, 4.28, 4.91, 3.98, 5.35, 3.63, 3.01, 3.38, 3.71,
                   #20.5e+40/scale2*1e+13, 5.7e+40/scale2*1e+13
                   ]) * 1e-13 * multi
    eff_right = np.array([0.73, 0.73, 0.79, 0.66, 0.73, 0.33, 0.60, 0.89, 0.60,
                          #1.5e+40/scale2*1e+13, 0.7e+40/scale2*1e+13
                          ]) * 1e-13 * multi
    eff_left = np.array([0.73, 0.73, 0.79, 0.66, 0.73, 0.30, 0.60, 0.93, 0.60,
                         #1.5e+40/scale2*1e+13, 0.6e+40/scale2*1e+13
                         ]) * 1e-13 * multi
    
    df = Table(data = [tt, ff, eff_right, eff_left],
               names = ["t", "f", "f+", "f-"])
    df = df[np.argsort(df["t"])]
    return df


def get_xlc_scp06f6():
    """
    [1] Leven+2013, XMM, unabsorbed 0.2--10 keV flux, nH = 8.85e+19,
            multiply by 0.71 to get absorbed 0.3--10 keV flux
        CXO non-detection
    """
    #z = 1.189
    # xmm
    #(Time("2006-08-02").mjd - 53767)/(1+1.189)
    # cxo
    #(Time("2006-11-04").mjd - 53767)/(1+1.189)
    tt = np.array([83.1, 126.1])
    ff = np.array([1.3e-13*0.71, 1.4e-14])
    eff = np.array([0.18 * (1.3e-13*0.71), np.nan])
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    df = df[np.argsort(df["t"])]
    return df


def get_xlc_15bn():
    # Margutti+2018 Section 2.1.4
    #z = 0.1136
    # (Time("2015-06-01").mjd - 57013)/(1+z)
    # (Time("2015-12-18").mjd - 57013)/(1+z)
    tt = np.array([144.6, 324.2])
    ff = np.array([9.8e-15, 5.3e-15])
    eff = np.array([np.nan, np.nan])
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    df = df[np.argsort(df["t"])]
    return df


def add_SLSNe_xlc(ax):
    color = "plum"

    df = get_xlc_scp06f6()
    z = 1.189
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    t1 = df["t"]
    L1 = df["f"]*multi
    eL1 = df["ferr"]*multi
    ax.errorbar(t1[0], L1[0], eL1[0], fmt = "s-", markersize = 4, 
                elinewidth = 0.6, linewidth = 1,
                color = color, zorder = 4, label = "SLSNe")
    ax.plot(t1[1], L1[1], markersize = 4,
            color = color, zorder = 4, marker = "v", alpha = 0.6)
    ax.plot(t1, L1, linestyle = "-.", color = color, markersize = 0.1, alpha = 0.6)
    #ax.text(90, 7e+43, "SCP 06F6", fontsize = fs-1, color = color)
    
    # PTF 12dam
    z = 0.107
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    #(Time("2012-06-11").mjd - 56022)/(1+z) --> 59.0
    #(Time("2012-06-19").mjd - 56022)/(1+z) --> 66.1
    Lx = 7e-16 * multi
    ax.errorbar(62.5, Lx, xerr = 3, yerr = 1e+40, fmt = "s-", markersize = 4, 
                elinewidth = 0.6, linewidth = 1,
                color = color, zorder = 4)
    #ax.text(45, 7e+39, "PTF12dam", fontsize = fs-1, color = color)
    
    # SN2015bn
    df = get_xlc_15bn()
    z = 0.1136
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    t1 = df["t"]
    L1 = df["f"]*multi
    ax.plot(t1, L1, markersize = 4, color = color, zorder = 4, marker = "v", 
            linestyle = "-.", alpha = 0.6)
    #ax.text(300, 1e+41, "15bn", fontsize = fs-1, color = color)
    

def add_SNeIIn_xlc(ax):
    color = "mediumaquamarine"
    
    df = get_xlc_sn2010jl()
    z = 0.0107
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr = df["ferr"]*multi
    ax.errorbar(xx, yy, yerr, color = color, zorder = 1, marker = ">", 
                markersize = 2, linestyle = "-.",
                elinewidth = 0.5, linewidth = 0.6, label = "SNe IIn")
    #ax.text(1050, 1.5e+40, "10jl", fontsize = fs-1, color = color)
    
    df = get_xlc_sn2005kd()
    df = df[df["t"]!=479]
    z = 0.015040
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    ax.errorbar(xx, yy, [yerr_left, yerr_right],
                color = color, zorder = 1, marker = ">", markersize = 2,
                linestyle = "-.",elinewidth = 0.5, linewidth = 0.6)
    #ax.text(1070, 1e+41, "05kd", fontsize = fs-1, color = color)
    
    df = get_xlc_sn2006jd()
    z = 0.0186
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    ax.errorbar(xx, yy, [yerr_left, yerr_right],
                color = color, zorder = 1, marker = ">", markersize = 2,
                linestyle = "-.",elinewidth = 0.5, linewidth = 0.6)
    #ax.text(1000, 3e+41, "06jd", fontsize = fs-1, color = color)
