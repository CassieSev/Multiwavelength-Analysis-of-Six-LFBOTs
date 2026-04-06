### Install dependencies ###
#pip install noaodatalab

### Import packages ###
from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert
import numpy as np

# This script uses TAP queries to retrieve data from various tables.

# Currently, the script is queries data from Legacy Survey DR10,
# which is named by the table 'ls_dr10.tractor'.

# The below will print out the descriptions of each column on the table
#print(qc.schema('ls_dr10.tractor'))

# Alternatively, go to this site: https://datalab.noirlab.edu/query.php?
# and select ls_dr10 -> ls_dr10.tractor to get column info
#To query a different table, replace instances of 'ls_dr10.tractor' with the new table, which
# you can find also from the above site.

# Enter source position and search radius

coords={'AT2023fhn': (152.015625, 21.07305556), 'AT2023vth':(269.143107, 8.043446),
        'AT2022abfc': (72.830058, -26.978075), 'AT2023hkw': (160.573750, 52.487840),
        'AT2024aehp': (125.281134,  28.739489), 'AT2024qfm': (350.347514, 11.942417),
        'AT2024wpp':(40.523307, -16.957198), 'AT2020xnd': (335.008425,  -2.840374)}

object='AT2020xnd'
ra = coords[object][0]
dec =  coords[object][1]
radius = 0.5# in arcseconds

# Get the object ID corresponding to your source position
columns1 = "ra,dec,type,ls_id, ra_ivar, dec_ivar"

# Now, we need to format the TAP query

# Line one, 'SELECT', describes the columns which we want to retrieve
# 'FROM' names the table we are accessing
# 'WHERE' states the conditions.  in this case, q3c_radial_query specifies that we want to
# search in  radius around the given ra and dec.  The first two entries say that we are
# using ra and dec coordinates, the next two gives the center of the search in ra and dec
# , and the last gives search radius in degrees
query1 = """
        SELECT %s
        FROM ls_dr10.tractor
        WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.2f/3600)
        """%(columns1, ra, dec, radius)
result1 = qc.query(sql=query1)
df1 = convert(result1)
print(df1)
print(np.power(df1["ra_ivar"][0], -1/2))
print(np.power(df1["dec_ivar"][0], -1/2))
my_ls_id = df1["ls_id"].values[0]



# For a general query, you use the ls_dr10.tractor table (same table as above)

# If you prefer magnitudes, use as mag_g instead of flux_g (but no corresponding ivar) 
columns_mag="mag_g, mag_r, mag_i, mag_z, mag_w1, mag_w2, mag_w3, mag_w4, snr_g, snr_r, snr_i, snr_z, snr_w1, snr_w2, snr_w3, snr_w4"
# Similar format, but now we look for object that matches the id in earlier query
query_mag = """
        SELECT %s
        FROM ls_dr10.tractor
        WHERE ls_id=%d
        """%(columns_mag, my_ls_id)

result_mag = qc.query(sql=query_mag)
df_mag = convert(result_mag)
print('Mags')
print(df_mag.iloc[:,0:8])

print('SNR')
print(df_mag.iloc[:,8:])
# The host brightness is given in flux and flux ivar (flux_g, flux_ivar_g),
# where flux is in units of nanomaggies and flux ivar in 1/nanomaggies^2.
columns_flux = "flux_g,flux_ivar_g,flux_r,flux_ivar_r,flux_i,flux_ivar_i,flux_z,flux_ivar_z,flux_w1,flux_ivar_w1,flux_w2,flux_ivar_w2,flux_w3,flux_ivar_w3"
query_flux = """
        SELECT %s
        FROM ls_dr10.tractor
        WHERE ls_id=%d
        """%(columns_flux, my_ls_id)

result_flux = qc.query(sql=query_flux)
df_flux = convert(result_flux)
# Can also convert from nanomaggies to apparent magnitude using the following functions:

# The flux is in units of nanomaggies:
# m=22.5−2.5log10(flux)
# em=-2.5*np.log10(np.e)*(1/flux)*ef
# em = -1.0857 * ef/f

def nanomaggie_to_mag(nanomaggie):
    return 22.5 - 2.5*np.log10(nanomaggie)

def ivar_to_mage(ivar, flux):
    fluxe=np.power(ivar, -1/2)
    return -1.0857*fluxe/flux

#Print out magnitudes
for band in ['g', 'r', 'i', 'z', 'w1', 'w2', 'w3']:
    print(f'{band}: '+str(nanomaggie_to_mag(np.array(df_flux['flux_{}'.format(band)]))[0]))
    print(ivar_to_mage(np.array(df_flux['flux_ivar_{}'.format(band)]), np.array(df_flux['flux_{}'.format(band)])))



# There is a separate table for photometric redshifts: ls_dr10.photo_z
columns_z = "ls_id,z_phot_median,z_phot_l68,z_phot_u68"
# Similar as in query 2, but getting different data from another table
query_z = """
        SELECT %s
        FROM ls_dr10.photo_z
        WHERE ls_id=%d
        """%(columns_z, my_ls_id)

result_z = qc.query(sql=query_z)
# Gives the photo-z, as well as 1 sigma limits
df_z = convert(result_z)
print(df_z)