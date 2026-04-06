import pandas as pd
import numpy as np

df=pd.read_csv('data/AT2024wpp/perley_xrt_input.txt', sep=' & ')
print(df)

df['t']=0.5*(np.array(df['t_start'])+np.array(df['t_end']))

dummy = []
#df['flux_pos']=np.float32(np.round(df['flux_pos'], 0))

df['flux_corr']=np.where(pd.isna(df['flux_neg']), df['flux'], '$'+df.flux.map(str)+'^{'+df.flux_pos.map(str)+'}_{'+df.flux_neg.map(str)+'}$')

times = df['t'][~pd.isna(df['flux_neg'])]
fluxes = df['flux'][~pd.isna(df['flux_neg'])]


print(list(np.float64(fluxes)))
print(list(np.float64(times)))
#df.to_csv('data/AT2024wpp/perley_xrt_output.txt', sep= '& ', columns=['t', 'flux_corr'], header=None)