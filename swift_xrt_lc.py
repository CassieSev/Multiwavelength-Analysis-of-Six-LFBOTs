""" Make a Swift/XRT light curve for AT2024wpp """

# Package imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.coordinates import Longitude, Latitude
import time
import pickle
from astropy.time import Time
from swifttools.swift_too import ObsQuery
from swifttools.swift_too import Data
import swifttools.ukssdc.xrt_prods as ux
import swifttools.ukssdc as uk
import astropy.units as u
import vals

def get_obsids(ra, dec):
    # Initialize query
    query = ObsQuery()
    query.ra, query.dec = ra, dec 

    if query.submit():
        print("Success!")
    else:
        print(f"Fail or timeout? {query.status}")

    print(query) # groups by orbit/snapshot
    print(query.observations) # groups by obs ID 

    # Number of observations...
    obsdict = query.observations
    len(obsdict.keys()) # number is 54

    # Target name: first two were ZTF24abjjpbo, rest were AT2024wpp

    # Treat the first observation separately because of the cosmic ray issue.
    # OBSID 00016843001: 4080s, 2024-09-27 
    # was done in 6 orbits
    #query[0].download(xrt=True, uvot=False, match="*po_cl.evt*") # clearly has CR

    return obsdict


def lc_snapshot():
    """ Get the LC per snapshot for a given obs ID """
    ##### GENERATE LC
    # Following https://www.swift.ac.uk/API/ukssdc/xrt_prods/README.md
    # https://www.swift.ac.uk/API/ukssdc/xrt_prods/RequestJob.md
    # & advice from Phil Evans via helpdesk

    # Validate email
    myReq = ux.XRTProductRequest('annayqho@berkeley.edu', silent=False)

    # Set the global parameters. I know it's detected, so centroiding on
    myReq.setGlobalPars(name='AT2024wpp', getTargs=True, T0=60578.436,
                    RA=40.5224286, Dec=-16.956104, centroid=True,
                    useSXPS=False, posErr=1)

    # Add a light curve
    myReq.addLightCurve(binMeth='snapshot', useObs=obsid)

    # Submit the job - note, 
    # this can fail so we ought to check the return code in real life
    myReq.submit()

    # Now wait until it's complete
    done=myReq.complete
    while not done:
        time.sleep(60)
        done=myReq.complete

    # And download the products
    lc = myReq.retrieveLightCurve(incbad='yes')
    return lc
    #myReq.downloadProducts('/my/path/', format='zip')


def lc_obsid(name, t0, ra, dec, centroid):
    """ Get LC by obs ID """
    # Validate email
    myReq = ux.XRTProductRequest('annayqho@berkeley.edu', silent=False)
    myReq.deprecate = True

    # The target name you use doesn't matter.
    # Set the global parameters. I know it's detected, so centroiding on
    myReq.setGlobalPars(name=name, getTargs=True, T0=t0,
                    RA=ra, Dec=dec, centroid=centroid,
                    useSXPS=False, posErr=1)

    # Add a light curve
    myReq.addLightCurve(binMeth='obsid')
    # Submit the job - note, 
    # this can fail so we ought to check the return code in real life
    myReq.submit()
    # Now wait until it's complete
    done=myReq.complete
    while not done:
        time.sleep(60)
        done=myReq.complete

    # And download the products
    lc = myReq.retrieveLightCurve(incbad='yes')
    return lc


def spec(name, t0, ra, dec, centroid, z):
    """ Get the spectrum """
    # Validate email
    myReq = ux.XRTProductRequest('annayqho@berkeley.edu', silent=False)
    myReq.deprecate = False
    # The target name you use doesn't matter.
    # Set the global parameters. I know it's detected, so centroiding on
    myReq.setGlobalPars(name=name, getTargs=True, T0=t0,
                    RA=ra, Dec=dec, centroid=centroid,
                    useSXPS=False, posErr=1)

    # Add a spectrum
    myReq.addSpectrum(redshift=z,
                      timeslice='obsid', galactic=True, whichData='all') # default is powerlaw

    # Submit the job - note, 
    # this can fail so we ought to check the return code in real life
    myReq.submit()

    # Now wait until it's complete
    done=myReq.complete
    while not done:
        time.sleep(60)
        done=myReq.complete

    # And download the products
    specfit = myReq.retrieveSpectralFits(returnData=True)
    return specfit


def generate_lc_files():
    """  Generate all the light-curve files, save dictionaries """
    # Get the list of obs IDs
    obsids = list(get_obsids().keys())

    # Get the measurements from the first epoch
    obsid_first = '00016843001'
    lc_first = lc_snapshot(obsid_first)
    with open("lc_first_obsid.pkl", "wb") as f:
        pickle.dump(lc_first, f)

    # Get & save the AT2024wpp LC
    lc_at2024wpp = lc_obsid()
    with open("lc_at2024wpp_obsid_old.pkl", "wb") as f:
        pickle.dump(lc_at2024wpp, f)


def regenerate_lc_file():
    """ There was a bug with the binning. Phil Evans suggested fix """
    myNewReq = ux.XRTProductRequest('annayqho@berkeley.edu')
    myNewReq.copyOldJob(218631, becomeThis=True)
    myNewReq.deprecate = False
    lc = myNewReq.retrieveLightCurve(incbad='yes', returnData=True)
    with open("lc_at2024wpp_obsid.pkl", "wb") as f:
        pickle.dump(lc, f)


def get_swift_lcs():
    """ Generate the dataframe of detections, nondetections, and binned
    nondetections 
    """
    # Concatenate the dictionaries into a single LC
    with open("lc_first_obsid.pkl", "rb") as f:
        first = pickle.load(f) # by snapshot

    with open("lc_at2024wpp_obsid.pkl", "rb") as f:
        at = pickle.load(f) # by obsid

    # Get detection DFs
    first_det = first['PC'].drop(axis=0, index=2) # drop CR-impacted obs.
    first_det['ObsID'] = ['00016843001']*len(first_det) # add column
    first_det['TimePos'] = first_det['T_+ve']
    first_det['TimeNeg'] = first_det['T_-ve']
    first_det['RatePos'] = first_det['Ratepos']
    first_det['RateNeg'] = first_det['Rateneg']
    at_det = at['PC_incbad'].drop(axis=0, index=0) # drop the first epoch
    det = pd.concat((first_det, at_det))

    # Get U.L. DFs
    first_nondet = first['PCUL']
    first_nondet['RatePos'] = first_nondet['Ratepos']
    first_nondet['RateNeg'] = first_nondet['Rateneg']
    first_nondet['TimeNeg'] = first_nondet['T_-ve']
    first_nondet['TimePos'] = first_nondet['T_+ve']
    first_nondet['UpperLimit'] = first_nondet['Rate']
    first_nondet['ObsID'] = ['00016843001']*len(first_nondet) # add column
    at_nondet = at['PCUL_incbad']
    nondet = pd.concat((first_nondet, at_nondet))

    # Drop most of the columns
    det = det[['Time', 'TimeNeg', 'TimePos', 'ObsID', 'Rate', 'RatePos', 'RateNeg']]

    ## Create a LC of non-detections
    nondet = nondet[at_nondet.columns] # new version of column names
    # Note: all binning resulted in non-detections.
    nondet_binned = nondet.copy()

    # Merge some rows
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848011', '00016848012']))) 
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                ['00016848014','00016848015','00016843005','00016843006']))) 
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848016', '00016848017'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848018', '00016848019'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848020', '00016848021'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848022', '00016848023'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848024', '00016848025', '00016848026'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848027', '00016848028', '00016848029'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848031', '00016848032'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848033', '00016848034'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                ['00016848036','00016848040','00016848041','00016848042'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848044','00016848045'])))
    print(new_row)
    new_row = uk.mergeLightCurveBins(nondet_binned, insert=True, remove=True,
                rows=(nondet_binned['ObsID'].isin(
                    ['00016848047','00016848048'])))
    print(new_row)

    return det, nondet, nondet_binned

coords={'AT2023fhn': (152.015625, 21.07305556), 'AT2023vth':(269.143107, 8.043446),
        'AT2022abfc': (72.830058, -26.978075), 'AT2023hkw': (160.573750, 52.487840),
        'AT2024aehp': (125.281134,  28.739489), 'AT2024qfm': (350.347514, 11.942417),
        'AT2024wpp':(40.523307, -16.957198),
        'grb250419a':(202.40537, 7.04076), 'CSS161010': (74.6433, -8.3011)} #only for SWIFT, not VLA localization


t0={'AT2023fhn': 60044.204, 'AT2023vth': 60235.116, 'AT2023hkw': 60065.199,
    'AT2022abfc': 59904.344, 'AT2021ahuo': 59352.4248, 'AT2018cow': 58285.44141,
    'AT2024aehp': 60663.35732640, 'AT2024qfm': 60518.34636569, 'CSS161010': 57671.48}


if __name__=="__main__":
    swift_band=2.34544956e+18*u.Hz
    name='AT2024aehp'
    start_t=t0[name]
    ra=Longitude(f'{coords[name][0]}d')
    dec=Latitude(f'{coords[name][1]}d')
    download=True
    if download:
        get_obsids(ra, dec)
        lc=lc_obsid(name, start_t, ra.degree, dec.degree, True)
        specfit=spec(name, start_t, ra.degree, dec.degree, True, vals.redshifts[name])

        with open("data/{}/lc_{}.pkl".format(name,name), "wb") as f:
            pickle.dump(lc, f)
        with open("data/{}/spec_{}.pkl".format(name,name), "wb") as f:
            pickle.dump(specfit, f)
    
    # Import data from pickle files
    with open("data/{}/lc_{}.pkl".format(name,name), "rb") as f:
        lc = pickle.load(f)
    with open("data/{}/spec_{}.pkl".format(name,name), "rb") as f:
        specfit = pickle.load(f)

    print(lc)
    print(specfit)

    

    # Get detections + non-detections, join together
    det_data=pd.DataFrame(lc["PC"])
    nondet_data=pd.DataFrame(lc["PCUL"])

    dt=np.concatenate((np.array(det_data['Time']), np.array(nondet_data['Time'])))

    cts_to_flux=4E-11  # Can change based on spec observations + WEBPIMMS


    flux=np.concatenate((np.array(det_data['Rate']), np.array(nondet_data['Rate'])))*cts_to_flux  # erg/cm2/s
    
    # Conversion to Jy if you need it
    (flux * u.erg/u.cm**2/u.s/swift_band).to(u.Jy)
    print(dt)
    print(flux)
#
    ## Detections table
    #d['MJD_Mid'] = Time(60578.436 + (d['Time'])/86400, format='mjd').isot
    #d['MJD_Start'] = Time(60578.436 + (d['Time']+d['TimeNeg'])/86400, format='mjd').isot
    #d['MJD_End'] = Time(60578.436 + (d['Time']+d['TimePos'])/86400, format='mjd').isot
    #d['Flux'] = d['Rate'] * fac
    #d['FluxPos'] = d['RatePos'] * fac
    #d['FluxNeg'] = d['RateNeg'] * fac
    #d = d[['MJD_Start', 'MJD_Mid', 'MJD_End', 'ObsID', 
    #       'Rate', 'RatePos', 'RateNeg', 'Flux', 'FluxPos', 'FluxNeg']]
#
    ## Non-detections table
    #nd['MJD_Mid'] = Time(60578.436 + (nd['Time'])/86400, format='mjd').isot
    #nd['MJD_Start'] = Time(60578.436 + (nd['Time']+nd['TimeNeg'])/86400, format='mjd').isot
    #nd['MJD_End'] = Time(60578.436 + (nd['Time']+nd['TimePos'])/86400, format='mjd').isot
    #nd['UpperLimit_Flux'] = nd['UpperLimit'] * fac
    #nd = nd[['MJD_Start', 'MJD_Mid', 'MJD_End', 'ObsID', 'UpperLimit', 'UpperLimit_Flux']]
#
    ## Non-detections table, binned
    #ndb['MJD_Mid'] = Time(60578.436 + (ndb['Time'])/86400, format='mjd').isot
    #ndb['MJD_Start'] = Time(60578.436 + (ndb['Time']+ndb['TimeNeg'])/86400, format='mjd').isot
    #ndb['MJD_End'] = Time(60578.436 + (ndb['Time']+ndb['TimePos'])/86400, format='mjd').isot
    #ndb['UpperLimit_Flux'] = ndb['UpperLimit'] * fac
    #ndb = ndb[['MJD_Start', 'MJD_Mid', 'MJD_End', 'ObsID', 'UpperLimit', 'UpperLimit_Flux']]
#
    #with open("at2024wpp_xrt_for_dan.pkl", "wb") as f:
    #    pickle.dump({'Detections': d, 'Limits': nd, 'Limits_Binned': ndb}, f)
