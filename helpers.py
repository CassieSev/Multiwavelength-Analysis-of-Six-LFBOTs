import numpy as np
import pandas as pd
from astropy.time import Time
from penquins import Kowalski
import secrets



def logon():
    """ Log onto Kowalski """
    username = secrets.username
    password = secrets.password
    s = Kowalski(
        protocol='https', host='kowalski.caltech.edu', port=443,
            verbose=False, username=username, password=password)
    return s


def mag_to_flux(mr,freq):
    """ Convert AB mag to flux """
    fnu = 10**((mr+48.6)/(-2.5))
    return fnu*freq


def get_alpha(dt,name):
    """ Get temporal power-law index """
    alphas = np.zeros(len(dt))
    if name=='iPTF14yb':
        alphas[dt>0] = 1.02
    elif name=='AT2019pim':
        alphas[dt<=3] = 0.9
        alphas[dt>8] = 2
    elif name=='AT2020blt':
        alphas[dt<=1] = 0.54
        alphas[dt>1] = 2.62
    elif name=='AT2020kym':
        alphas[dt<0.8] = 1.53
        alphas[dt>=0.8] = 0.8
    elif name=='AT2020sev':
        alphas[dt>0] = 0.68
    elif name=='AT2020yxz':
        alphas[dt>0] = 0.96
    elif name=='AT2021any':
        alphas[dt<0.8] = 0.7
        alphas[dt>=0.8] = 2.3
    elif name=='AT2021buv':
        alphas[dt<2.2] = 0.6
        alphas[dt>=2.2] = 1.7
    elif name=='AT2021lfa':
        alphas[dt>0] = 1.57
    return alphas


def get_dets(s, name):
    q = {"query_type": "find",
         "query": {
             "catalog": "ZTF_alerts",
             "filter": {
                     'objectId': {'$eq': name},
                     'candidate.isdiffpos': {'$in': ['1', 't']},
             },
             "projection": {
                     "_id": 0,
                     "candidate.jd": 1,
                     "candidate.magpsf": 1,
                     "candidate.sigmapsf": 1,
                     "candidate.fid": 1,
                     "candidate.programid": 1,
                     "candidate.field": 1,
                     "candidate.ra": 1,
                     "candidate.dec": 1,
             }
         }
         }
    query_result = s.query(query=q)
    try:
        out = query_result['data']
        return out
    except:
        return []


def get_first_det(s, name):
    out = get_dets(s, name)
    jds = np.array([det['candidate']['jd'] for det in out])
    mag = np.array([det['candidate']['magpsf'] for det in out])
    emag = np.array([det['candidate']['sigmapsf'] for det in out])
    filt = np.array([det['candidate']['fid'] for det in out])
    ind = np.argmin(jds)
    return jds[ind], mag[ind], emag[ind], filt[ind]


def get_pos(s, name):
    """ Take median position from alerts """
    det_alerts = get_dets(s, name)
    ras = [det['candidate']['ra'] for det in det_alerts]
    decs = [det['candidate']['dec'] for det in det_alerts]
    ra = np.median(ras)
    dec = np.median(decs)
    return ra,dec


def get_list():
    """ Get the list of source names """
    dat = pd.read_csv("sources.txt")
    return dat['Name'].values
        

def get_t0(name):
    dat = pd.read_csv("sources.txt")
    t0 = Time(dat['t0'][dat['Name']==name].values[0], format='mjd').jd
    return t0


def get_iau(name):
    dat = pd.read_csv("sources.txt")
    tns = dat['IAU'][dat['Name']==name].values[0]
    return tns


def get_grb(name):
    dat = pd.read_csv("sources.txt")
    grb = dat['GRB'][dat['Name']==name].values[0]
    return grb


def get_z(name):
    dat = pd.read_csv("sources.txt")
    z = dat['z'][dat['Name']==name].values[0]
    return z


def get_t90(grb):
    dat = pd.read_fwf("kw_burst_list_v3.txt")
    t90 = dat['T90'][dat['Name']==grb].values[0]
    et90 = dat['T90Err'][dat['Name']==grb].values[0]
    return (t90,et90)


def get_energetics(grb, quantity):
    """
    grb: name of the burst (e.g., GRB140226A)
    quantity: can be Epi, Dpp, Fl, PF, Eiso, Liso
    """
    dat = pd.read_fwf(ddir+"/kw_burst_list_v3.txt")
    val = float(dat[quantity][dat['Name']==grb].values[0])
    if val == "--":
        return None
    else:
        val_l = float(dat['%sErrDn' %quantity][dat['Name']==grb].values[0])
        val_u = float(dat['%sErrUp' %quantity][dat['Name']==grb].values[0])
        return (val,val_l,val_u)
