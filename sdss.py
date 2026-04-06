from astroquery.sdss import SDSS
from astropy import coordinates as coords


pos = coords.SkyCoord('125.281134d 28.739489d', frame='icrs')
xid = SDSS.query_region(pos, radius='1 arcsec', fields=['ra', 'dec',
                                                        'u', 'err_u', 'g', 'err_g', 'r', 'err_r', 'i', 'err_i', 'z', 'err_z'])
print(xid)

# For fluxes in nanomaggies + variance in inverse nanomaggies^2:
#column names are: 'modelFlux_u', 'modelFluxIvar_u',