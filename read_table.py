""" Read the table from the radio data """

import numpy as np
import pandas as pd


DATA_DIREC = "/Users/annaho/Dropbox/astro/projects_proto/IcBL/old/data/radio_compilations"


def zauderer():
    lines = open(
            "%s/Zauderer2011/table.txt" %DATA_DIREC, "r").readlines()
    dt = []
    nu = []
    f = []
    islim = []
    ef = []

    for line in lines:
        dt.append(float(line.split("&")[1]))
        nu.append(float(line.split("&")[3]))
        if '<' in line:
            islim.append(True)
            f.append(float(
                line.split("&")[4].split("$")[1][1:]))
        else:
            islim.append(False)
            f.append(float(
                line.split("&")[4].split("$")[1].split("pm")[0]))
            ef.append(float(
                line.split("&")[4].split("$")[1].split("pm")[1]))

    dt = np.array(dt)
    nu = np.array(nu)
    f = np.array(f)
    ef = np.array(ef)
    islim = np.array(islim)

    return nu, dt, f, ef, islim


def read_2003L():
    """ this paper doesn't give upper limits """
    dt = []
    nu = []
    f = []
    ef = []
    
    lines = pd.read_table(
            "%s/SN2003L/table.txt" %DATA_DIREC, delimiter='&', 
            names=['Date', 'dt', 'F4.9', 'F8.5', 'F15.0', 'F22.5', 'Config'])

    for freq in ['4.9', '8.5', '15.0', '22.5']:
        choose = ~np.array(['nodata' in line for line in lines['F%s' %freq]])
        [dt.append(val) for val in lines['dt'].values[choose]]
        [nu.append(freq) for val in np.arange(sum(choose))]
        [f.append(float(val.split("$pm$")[0])) for val in lines['F%s' %freq][choose].values]
        [ef.append(float(val.split("$pm$")[1])) for val in lines['F%s' %freq][choose].values]

    dt = np.array(dt).astype(float)
    nu = np.array(nu).astype(float)
    f = np.array(f).astype(float)
    ef = np.array(ef).astype(float)

    return nu, dt, f, ef


def line_1993J(dt, nu, f, ef, islim, line, col, wl):
    temp = str(line.split("&")[col])
    if 'nodata' not in temp:
        dt.append(float(line.split("&")[1]))
        nu.append(3E10/wl) # freq
        if '<' in temp:
            islim.append(True)
            f.append(float(
                temp.split("$")[1].split('<')[1]))
            ef.append(-99)
        else:
            islim.append(False)
            f.append(float(
                temp.split("$")[1].split("pm")[0]))
            ef.append(float(
                temp.split("$")[1].split("pm")[1]))
    return dt, nu, f, ef, islim


def read_1993J_low_freq():
    lines = open(
            "%s/SN1993J/table_low_freq.txt" %DATA_DIREC, "r").readlines()
    dt = []
    nu = []
    f = []
    ef = []
    islim = []

    for line in lines:
        dt, nu, f, ef, islim = line_1993J(dt, nu, f, ef, islim, line, 3, 20)
        dt, nu, f, ef, islim = line_1993J(dt, nu, f, ef, islim, line, 4, 6)
        dt, nu, f, ef, islim = line_1993J(dt, nu, f, ef, islim, line, 5, 3.6)
        dt, nu, f, ef, islim = line_1993J(dt, nu, f, ef, islim, line, 6, 2)
        dt, nu, f, ef, islim = line_1993J(dt, nu, f, ef, islim, line, 7, 1.2)

    nu = np.array(nu)
    f = np.array(f)
    ef = np.array(ef)
    dt = np.array(dt)
    islim = np.array(islim)

    return nu, dt, f, ef, islim


def read_1993J_high_freq():
    lines = open(
            "%s/SN1993J/table_high_freq.txt" %DATA_DIREC, "r").readlines()
    dt = []
    nu = []
    f = []
    ef = []
    islim = []

    for line in lines:
        dt.append(float(line.split("&")[1]))
        nu.append(1E9*float(line.split("&")[4]))
        if '<' not in line:
            islim.append(False)
            f.append(float(line.split("&")[3].split("$")[1].split("pm")[0]))
            ef.append(float(line.split("&")[3].split("$")[1].split("pm")[1]))
        else:
            islim.append(True)
            f.append(float(line.split("&")[3].split("$")[1].split("<")[1]))
            ef.append(-99)

    nu = np.array(nu)
    f = np.array(f)
    ef = np.array(ef)
    dt = np.array(dt)
    islim = np.array(islim)

    return nu, dt, f, ef, islim


def read_2011dh():
    lines = open(
            "%s/SN2011dh/data.txt" %DATA_DIREC, "r").readlines()
    dt = []
    nu = []
    f = []
    ef = []
    islim = []

    for line in lines:
        dt.append(float(line.split('&')[0]))
        nu.append(float(line.split('&')[1]))
        if 'leq' in line:
            islim.append(True)
            f.append(float(line.split('&')[2].split('$')[1].split('leq')[1]))
            ef.append(-99)
        else:
            islim.append(False)
            f.append(float(line.split('&')[2].split('$pm$')[0]))
            ef1 = float(line.split('&')[2].split('$pm$')[1])
            ef2 = float(line.split('&')[2].split('$pm$')[2])
            ef.append(np.sqrt(ef1**2+ef2**2))

    dt = np.array(dt)
    nu = np.array(nu)*1E9
    f = np.array(f)
    ef = np.array(ef)
    islim = np.array(islim)

    return dt, nu, f, ef, islim
