import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
import vals
import numpy as np
import matplotlib.pyplot as plt

"""
Creates a figure comparing the cumulative distribution of offsets to a host galaxy
for several classes of extragalactic transients
"""

obj_name='AT2020xnd'

ztf_fbot_coords={'AT2024aehp': ('08:21:07.47', '+28:44:22.16')} # Fritz
fbot_coords={'AT2024aehp': ('8:21:07.4738385633', '28:44:22.1691010108'), 'AT2022abfc':('4:51:19.1997452756', '-26:58:41.5873226548'),
             'AT2023fhn': ('10:08:03.805', '21:04:26.7'), 'AT2023hkw':('10:42:17.7478060336', '52:29:19.0314773950'),
             'AT2023vth': ('17:56:34.4031', '8:02:37.30'), 'AT2024qfm': ('23:21:23.45',  '+11:56:31.99'),
             'AT2020xnd': ('22:20:02.0390675464', '-2:50:25.3982086265')} # vla except for AT2023fhn (chrimes) and 2024qfm (fritz)
host_coords={'AT2024aehp': (125.281102, 28.739465), 'AT2023fhn':(152.015585, 21.072938),
             'AT2022abfc': (72.830053, -26.978074), 'AT2023hkw': (160.573578, 52.487839),
             'AT2024qfm': (350.347507, 11.94241), 'AT2023vth':(269.14376351	, 8.04324255),
             'AT2020xnd': (335.008504, -2.840442)} # From LS DR10, or PS1 DR2 for AT2023vth
host_coords_fhn={'AT2023fhn': ('10:08:03.78', '21:04:22.5')} # For AT2023fhn, using Ashley Chrimes' localization
c1 = SkyCoord(fbot_coords[obj_name][0],  fbot_coords[obj_name][1], unit=(u.hourangle, u.deg), frame='fk5')
c2 = SkyCoord(host_coords[obj_name][0]*u.deg, host_coords[obj_name][1]*u.deg, frame='fk5')  
#c2 = SkyCoord(host_coords_fhn[obj_name][0],  host_coords_fhn[obj_name][1], unit=(u.hourangle, u.deg), frame='fk5')
print(c1.separation(c2).arcsecond)
print(Planck18.angular_diameter_distance(z=vals.redshifts[obj_name])*c1.separation(c2).rad)

    
if __name__ == '__main__':
    ### First figure is Normalized Offsets
    ######## Comparison data
    hostnormedoffset_compare = np.loadtxt('comparison_normed_data.txt',skiprows=2)
    # 0 = FRBs - IR+UV - Bhandari 2022
    # 1 = LGRBs - IR/UVIS - Blanchard et al. 2016
    # 2 = LGRBs - IR - Lyman et al. 2017
    # 3 = SGRBs - UV - Fong & Berger 2013 (inc. Fong et al. 2010)
    # 4 = SGRBs - IR - Fong & Berger 2013 (inc. Fong et al. 2010)
    # 5 = SLSNe
    # 6 = KK2012
    # 7 = Ca-rich SNe, De et al. 2020 
    # 8 = SGRBs - optical - Fong+ 2022

    
    #plt.figure(42)
    #plt.subplot(1,2,2)
    ### Comparison samples
    #lgrbs = np.append(hostnormedoffset_compare[:,1][hostnormedoffset_compare[:,1]<999],hostnormedoffset_compare[:,2][hostnormedoffset_compare[:,2]<999])
    #N,bins,patches = plt.hist(lgrbs,histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='c',linestyle='--')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],'--c',linewidth=2,label='LGRBs')
    #
    #N,bins,patches = plt.hist(hostnormedoffset_compare[:,5][hostnormedoffset_compare[:,5]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='y',linestyle='-')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],'-y',linewidth=2,label='SLSNe')
  #
    #N,bins,patches = plt.hist(hostnormedoffset_compare[:,6][hostnormedoffset_compare[:,6]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='m',linestyle='--')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],'--m',linewidth=2,label='CCSNe')
#
    #N,bins,patches = plt.hist(hostnormedoffset_compare[:,0][hostnormedoffset_compare[:,0]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='b',linestyle='-')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],'-b',linewidth=2,label='FRBs') #now Bhandari 2022, was Mannings 21 before
    #
    #N,bins,patches = plt.hist(hostnormedoffset_compare[:,7][hostnormedoffset_compare[:,7]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='#ff9900',linestyle='--')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],linestyle='--',color='#ff9900',linewidth=2,label='Ca-rich SNe')
    #
#
    #N,bins,patches = plt.hist(hostnormedoffset_compare[:,8][hostnormedoffset_compare[:,8]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='r',linestyle='-')
    #patches[0].set_xy(patches[0].get_xy()[:-1])
    #plt.plot([0,1],[-1,-2],'-r',linewidth=2,label='SGRBs')
   #
#
    #plt.ylim([0,1])
    #plt.xlim([0.02,25])
    #plt.xscale('log') 
    #plt.xlabel(r'Host-normalised offset $r_{n}$',fontsize=16)  
    #plt.ylabel('Cumulative Fraction',fontsize=16) 
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.legend()    
    #
    ##spiral
    #plt.plot([3.67,3.67],[0,1],'-k',linewidth=3) #555
    #plt.plot([4.25,4.25],[0,1],'--k',linewidth=3) #814
#
    ##satellite
    #plt.plot([3.98,3.98],[0,1],'-k',linewidth=1) #555
    #plt.plot([4.66,4.66],[0,1],'--k',linewidth=1) #814
    #
    
    
    # Raw offsets
    plt.figure(86)
    plt.subplot(1,1,1)
    ######## Comparison data
    offset_compare = np.loadtxt('comparison_offset_data.txt',skiprows=2)
    # 0 = FRBs - IR+UV - Bhnadari+22
    # 1 = LGRBs - IR/UVIS - Blanchard et al. 2016
    # 2 = LGRBs - IR - Lyman et al. 2017
    # 3 = SGRBs - IR/UVIS - Fong+22
    # 4 = Type Ia SNe - Wang 2013
    # 5 = All CCSNe - Schulze 2020
    # 6 = SLSNe L15, S20
    # 7 = Ca-rich SNe, De et al. 2020
    # 8 = SGRBs - IR/UVIS - Fong & Berger 2013 (inc. Fong et al. 2010)

    #FBOT offsets
    offset_fbots = np.array([1.7,1.9,0.28,1.19,6,  0.8, 5,   17, 2, 15, 3, 0.47, 3.7]) #in kpc
    N,bins,patches = plt.hist(offset_fbots,histtype='step',density=True,cumulative=True,bins=1000,color='k',linestyle='-',linewidth=3)
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],color='k',linewidth=3,linestyle='-',label='LFBOTs')


    lgrbs = np.append(offset_compare[:,1][offset_compare[:,1]<999],offset_compare[:,2][offset_compare[:,2]<999])
    N,bins,patches = plt.hist(lgrbs,histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='c',linestyle='--')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'--c',linewidth=2,label='LGRBs')
    
    N,bins,patches = plt.hist(offset_compare[:,6][offset_compare[:,6]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='y',linestyle='-')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'-y',linewidth=2,label='SLSNe')

    N,bins,patches = plt.hist(offset_compare[:,5][offset_compare[:,5]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='m',linestyle='--')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'--m',linewidth=2,label='CCSNe')
    
    N,bins,patches = plt.hist(offset_compare[:,0][offset_compare[:,0]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='b',linestyle='-')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'-b',linewidth=2,label='FRBs')
    
    N,bins,patches = plt.hist(offset_compare[:,4][offset_compare[:,4]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='g',linestyle='-')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'-g',linewidth=2,label='SNe Ia')
    
    N,bins,patches = plt.hist(offset_compare[:,7][offset_compare[:,7]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='#ff9900',linestyle='--')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],linestyle='--',color='#ff9900',linewidth=2,label='Ca-rich SNe')
    
    N,bins,patches = plt.hist(offset_compare[:,3][offset_compare[:,3]<999],histtype='step',density=True,cumulative=True,bins=1000,linewidth=2,color='r',linestyle='-')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    plt.plot([0,1],[-1,-2],'-r',linewidth=2,label='SGRBs')


    plt.ylim([0,1])
    plt.xlim([0.05,100])
    plt.xscale('log')
    #plt.yticks([])
    #plt.xticks([])  
    plt.xlabel(r'Offset (kpc)',fontsize=14)  
    plt.ylabel('Cumulative Fraction',fontsize=14)  
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={'size':12})    
    
    #spiral
    #plt.plot([16.51,16.51],[0,1],'-k',linewidth=3) #555
    #plt.plot([16.55,16.55],[0,1],'--k',linewidth=3) #814

    #satellite
    #plt.plot([5.35,5.35],[0,1],'-k',linewidth=1) #555
    #plt.plot([5.34,5.34],[0,1],'--k',linewidth=1) #814
plt.tight_layout()
plt.savefig('figures/fig8_host_offset.pdf', dpi=450)
plt.show()

