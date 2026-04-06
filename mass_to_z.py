import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
from astropy.table import Table

log_m=np.arange(8.91, 11.70, step=0.2)
log_z=np.array([-0.6, -0.61, -0.65, -0.61, -0.52, -0.41, -0.23, -0.11, -0.01, 0.04, 0.07, 0.1, 0.12, 0.13])


print(np.interp(8.92904885656, log_m, log_z))