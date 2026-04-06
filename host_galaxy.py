""" Figure showing the position of the transient in its host galaxy,
as well as the position of the host galaxy in the M*-SFR plane """

import numpy as np
from astropy.cosmology import Planck18
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
#from reproject import reproject_interp, reproject_adaptive
from astropy import coordinates as coords
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from scipy.interpolate import RectBivariateSpline
import sys
import vals
sys.path.append("..")



def get_host_phot_sdss(ra,dec, imsize):
    """ Get host photometry from SDSS """
    ddir = "data/AT2023hkw/"
    # Position of the transient in the host galaxy
    file_num = "003103-4-0059"
    bands='grizu'
    cuts=[]
    # Figure out pos from header
    for band in bands:
        rim = fits.open(ddir + "frame-{}-{}.fits".format(band, file_num))
        head = rim[0].header
        wcs = WCS(head)
        target_pix = wcs.all_world2pix([(np.array([ra,dec], np.float64))], 1)[0]

        xpos = target_pix[0]
        ypos = target_pix[1]

        # Plot transient in its host galaxy
        data = fits.open( # range 20~200 perhaps
                  ddir + "frame-{}-{}.fits".format(band, file_num))[0].data
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()

        bkg = Background2D(data, (16,16), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        data=data-bkg.background
        cut=data[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]

        # Do 2D Interpolation
        cut_dims=np.shape(cut)
        y_diffs=np.abs(np.diff(cut, n=2, axis=0))<1.7*cut[1:cut_dims[0]-1,:]
        x_diffs=np.abs(np.diff(cut, n=2, axis=1))<1.7*cut[:,1:cut_dims[0]-1]

        y_diffs=np.append(np.insert(y_diffs, obj=0, values=0, axis=0), np.zeros(shape=(1, cut_dims[0])), axis=0)
        x_diffs=np.append(np.insert(x_diffs, obj=0, values=0, axis=1), np.zeros(shape=(cut_dims[0],1)), axis=1)

        mask=(x_diffs==1)&(y_diffs==1)
        filtered_cut=np.where(mask, cut, 0)

        x_array = np.arange(0,cut_dims[0])
        y_array = np.arange(0,cut_dims[1])
        cut_spline = RectBivariateSpline(x_array, y_array, filtered_cut, kx=1, ky=1, s=0.02)
        new_cut=np.array([cut_spline.ev(i, y_array) for i in range(cut_dims[0])])
        cuts.append(new_cut)


    #ucut = u[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]
    #gcut = g[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]
    #rcut = r[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]
    #icut = i[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]
    #zcut = z[int(ypos-imsize):int(ypos+imsize),int(xpos-imsize):int(xpos+imsize)]

    return cuts, (imsize,imsize)


def get_host_ls(ra, dec):
    """ Get host images from Legacy Survey """
    ddir = "../data//"

    # Position of the transient in the host galaxy

    # Open images
    dat = fits.open(ddir + "cutout_180.6551_35.3931.fits")[0].data
    g = dat[0]
    r = dat[1]
    z = dat[2]

    # Get header info
    head = fits.open(ddir + "cutout_180.6551_35.3931.fits")[0].header
    wcs = WCS(head)
    target_pix = wcs.all_world2pix([[ra, dec, 1]], 1)[0]  
    xpos = target_pix[0]
    ypos = target_pix[1]
    print(xpos,ypos)

    return (g,r,z), (xpos,ypos)


if __name__=="__main__":
    """ host galaxy panel """
    object='AT2023hkw'
    img_source='SDSS'
    if img_source=='SDSS':
        fig,ax = plt.subplots(1,1, figsize=(8,8))

        # Get host galaxy images
        gal,pos = get_host_phot_sdss(vals.coords[object][0], vals.coords[object][1], 15)
        u,g,r,i,z = gal
        xpos,ypos = pos

        # Try gri
        #rgb = make_lupton_rgb(z*1.1,r*2,g*4,stretch=0.1, Q=2) # for ls
        rgb = make_lupton_rgb(r*1.45,g*1.7,3.5*u,stretch=1.4, Q=0.5)
        ax.imshow(rgb, origin='lower', vmin=0.01, vmax=0.03)

        # Mark position of transient
        markcol = 'white'
        imsize = rgb.shape[0]
        ax.plot([xpos, xpos], [ypos, ypos-8], c=markcol, lw=1)
        ax.plot([xpos, xpos+8], [ypos, ypos], c=markcol, lw=1)
        ax.text(xpos+1, ypos-2, object, 
                ha='left', va='top', fontsize=20, color=markcol)

        # Mark compass
        imsize = 30
        ax.plot((imsize-3,imsize-3), (imsize-3,imsize-6), color=markcol, lw=2)
        ax.text(
                imsize-3, imsize-7, "S", color=markcol, fontsize=16,
                horizontalalignment='center', verticalalignment='top')
        ax.plot((imsize-3,imsize-6), (imsize-3,imsize-3), color=markcol, lw=2)
        ax.text(
                imsize-7, imsize-3, "E", color=markcol, fontsize=16,
                horizontalalignment='right', verticalalignment='center')
        ax.axis('off')

        ax.text(
            0.01, 0.99,"SDSS $gru$",fontsize=20,transform=ax.transAxes,
            horizontalalignment='left', va='top', color='white')
        
        # For SDSS: 0.396 arcsec per pixel
        x = 25
        y = 2
        x2 = x + 1/0.396
        ax.plot((x,x2), (y,y), color=markcol, lw=2)
        ax.text((x2+x)/2, y*1.1, "1''", color=markcol, fontsize=10,
                verticalalignment='bottom', horizontalalignment='center')
        dA = Planck18.angular_diameter_distance(z=vals.redshifts[object]).value
        scale = (1/206265)*dA*1000
        ax.text((x2+x)/2, y/1.1,"(%s kpc)"%int(scale),color=markcol,fontsize=10,
                verticalalignment='top', horizontalalignment='center')

    # Mark image scale : 0.262 arcsec per pixel, only for LegacySurvey
    #x = 200
    #y = 20
    #x2 = x + 5/0.262
    #ax.plot((x,x2), (y,y), color=markcol, lw=2)
    #ax.text((x2+x)/2, y*1.1, "5''", color=markcol, fontsize=10,
    #        verticalalignment='bottom', horizontalalignment='center')
    #dA = Planck18.angular_diameter_distance(z=vals.z).value
    #scale = (5/206265)*dA*1000
    #ax.text((x2+x)/2, y/1.1,"(%s kpc)"%int(scale),color=markcol,fontsize=10,
    #        verticalalignment='top', horizontalalignment='center')
    
    plt.show()
    #plt.savefig("host_cutout.png",dpi=300,bbox_inches='tight',pad_inches=0.1)
    plt.close()
