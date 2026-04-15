import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from io import BytesIO
import urllib.request
import urllib.error

# Source information from Table 1
sources = {
    'AT2022abfc': {
        'ra': '4:51:19.200', 'dec': '-26:58:41.59', 'z': 0.212,
        'coord_frame': 'hmsdms'
    },
    'AT2023fhn': {
        'ra': '10:08:03.805', 'dec': '21:04:26.70', 'z': 0.24,
        'coord_frame': 'hmsdms'
    },
    'AT2023hkw': {
        'ra': '10:42:17.748', 'dec': '52:29:19.03', 'z': 0.339,
        'coord_frame': 'hmsdms'
    },
    'AT2023vth': {
        'ra': '17:56:34.403', 'dec': '8:02:37.30', 'z': 0.0747,
        'coord_frame': 'hmsdms'
    },
    'AT2024qfm': {
        'ra': '23:21:23.450', 'dec': '+11:56:31.99', 'z': 0.2270,
        'coord_frame': 'hmsdms'
    },
    'AT2024aehp': {
        'ra': '8:21:07.474', 'dec': '28:44:22.17', 'z': 0.1715,
        'coord_frame': 'hmsdms'
    },
}

# Parse coordinates
for name, info in sources.items():
    coord = SkyCoord(
        info['ra'], info['dec'],
        unit=(u.hourangle, u.deg), frame='icrs'
    )
    info['coord'] = coord
    info['ra_deg'] = coord.ra.deg
    info['dec_deg'] = coord.dec.deg
    print(f"{name}: RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:.5f}")


def fetch_legacy_jpg(ra, dec, size_arcmin=0.5, pixscale=0.262):
    """Fetch a color JPG cutout from the Legacy Survey."""
    npix = int(size_arcmin * 60 / pixscale)
    url = (
        f"https://www.legacysurvey.org/viewer/cutout.jpg"
        f"?ra={ra}&dec={dec}&size={npix}&layer=ls-dr10&pixscale={pixscale}"
    )
    print(f"  Trying Legacy Survey: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            if len(data) < 500:
                print("  Legacy Survey: image too small, likely blank")
                return None
            from PIL import Image
            img = Image.open(BytesIO(data))
            arr = np.array(img)
            # Check if image is mostly one color (blank/error)
            if arr.std() < 5:
                print("  Legacy Survey: image appears blank")
                return None
            return arr, 'Legacy Survey DR10 $grz$'
    except Exception as e:
        print(f"  Legacy Survey failed: {e}")
        return None


def fetch_panstarrs_jpg(ra, dec, size_arcmin=0.5):
    """Fetch a color JPG cutout from Pan-STARRS (dec > -30).

    Uses the ps1filenames API to get image paths, then fitscut for cutout.
    """
    size_pix = int(size_arcmin * 60 / 0.25)  # 0.25 arcsec/pix

    # Step 1: get filenames for g, r, i filters
    fn_url = (
        f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        f"?ra={ra}&dec={dec}&filters=gri&type=stack"
    )
    print(f"  Trying Pan-STARRS (querying filenames)...")
    try:
        req = urllib.request.Request(fn_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            lines = response.read().decode().strip().split('\n')

        if len(lines) < 2:
            print("  Pan-STARRS: no images found")
            return None

        # Parse filenames by filter
        filt_files = {}
        for line in lines[1:]:
            parts = line.split()
            filt = parts[4]
            fname = parts[7]
            filt_files[filt] = fname

        if not all(f in filt_files for f in 'gri'):
            print(f"  Pan-STARRS: missing filters, have {list(filt_files.keys())}")
            # Use whatever we have
            if 'r' not in filt_files:
                return None

        # Step 2: build fitscut color URL
        red = filt_files.get('i', filt_files.get('r'))
        green = filt_files.get('r')
        blue = filt_files.get('g', filt_files.get('r'))

        color_url = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
            f"?ra={ra}&dec={dec}&size={size_pix}&format=jpg"
            f"&red={red}&green={green}&blue={blue}"
            f"&autoscale=98.0&output_size={size_pix}"
        )
        print(f"  Pan-STARRS cutout URL: {color_url[:100]}...")
        req = urllib.request.Request(color_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            if len(data) < 1000:
                print("  Pan-STARRS: image too small, likely blank")
                return None
            from PIL import Image
            img = Image.open(BytesIO(data))
            # Determine which filters were actually used
            filters_used = []
            for f in ['g', 'r', 'i']:
                if f in filt_files:
                    filters_used.append(f)
            filt_str = ''.join(filters_used)
            return np.array(img), f'Pan-STARRS ${filt_str}$'
    except Exception as e:
        print(f"  Pan-STARRS failed: {e}")
        return None


def fetch_sdss_jpg(ra, dec, size_arcmin=0.5):
    """Fetch a color JPG cutout from SDSS."""
    scale = 0.396  # arcsec/pix
    npix = max(200, int(size_arcmin * 60 / scale))
    url = (
        f"/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale={https://skyserver.sdss.org/dr17scale}&width={npix}&height={npix}"
    )
    print(f"  Trying SDSS: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            if len(data) < 1000:
                print("  SDSS: image too small, likely blank")
                return None
            from PIL import Image
            img = Image.open(BytesIO(data))
            arr = np.array(img)
            # Check if image is mostly black (outside SDSS footprint)
            if np.mean(arr) < 10:
                print("  SDSS: image is mostly black, outside footprint")
                return None
            return arr, 'SDSS DR17 $gri$'
    except Exception as e:
        print(f"  SDSS failed: {e}")
        return None


def get_cutout(ra, dec, size_arcmin=0.5):
    """Try Legacy Survey, then Pan-STARRS, then SDSS."""
    result = fetch_legacy_jpg(ra, dec, size_arcmin=size_arcmin)
    if result is not None:
        return result

    if dec > -30:
        result = fetch_panstarrs_jpg(ra, dec, size_arcmin=size_arcmin)
        if result is not None:
            return result

    result = fetch_sdss_jpg(ra, dec, size_arcmin=size_arcmin)
    if result is not None:
        return result

    print("  WARNING: No cutout found from any survey!")
    return None


# Fetch all cutouts
print("\nFetching cutouts...")
cutouts = {}
size_arcmin = 0.5  # 30 arcsec on each side of transient

for name, info in sources.items():
    print(f"\n{name}:")
    result = get_cutout(info['ra_deg'], info['dec_deg'], size_arcmin=size_arcmin)
    if result is not None:
        cutouts[name] = {'image': result[0], 'survey': result[1]}
    else:
        cutouts[name] = None

# Make the figure
fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()

names_order = [
    'AT2022abfc', 'AT2023fhn', 'AT2023hkw',
    'AT2023vth', 'AT2024qfm', 'AT2024aehp'
]

for i, name in enumerate(names_order):
    ax = axes[i]
    info = sources[name]

    if cutouts[name] is not None:
        img = cutouts[name]['image']
        survey = cutouts[name]['survey']
        ax.imshow(img, origin='upper')

        # Mark transient position at center with crosshairs
        cy, cx = img.shape[0] / 2, img.shape[1] / 2
        cross_size = img.shape[0] * 0.08
        gap = img.shape[0] * 0.03
        lw = 1.5
        color = 'white'

        # Draw crosshairs with gap in center
        ax.plot([cx - cross_size, cx - gap], [cy, cy],
                '-', color=color, lw=lw)
        ax.plot([cx + gap, cx + cross_size], [cy, cy],
                '-', color=color, lw=lw)
        ax.plot([cx, cx], [cy - cross_size, cy - gap],
                '-', color=color, lw=lw)
        ax.plot([cx, cx], [cy + gap, cy + cross_size],
                '-', color=color, lw=lw)

        # Add scale bar (10 arcsec)
        pix_per_arcsec = img.shape[1] / (size_arcmin * 2 * 60)
        bar_length_pix = 10 * pix_per_arcsec  # 10 arcsec
        bar_y = img.shape[0] * 0.90
        bar_x_start = img.shape[1] * 0.05
        ax.plot([bar_x_start, bar_x_start + bar_length_pix],
                [bar_y, bar_y], '-', color='white', lw=2)
        ax.text(bar_x_start + bar_length_pix / 2, bar_y - img.shape[0] * 0.04,
                '10"', color='white', fontsize=7, ha='center', va='bottom')

        # Add N/E compass arrows (N=up, E=left for standard orientation)
        arrow_x = img.shape[1] * 0.88
        arrow_y = img.shape[0] * 0.15
        arrow_len = img.shape[0] * 0.10
        ax.annotate('', xy=(arrow_x, arrow_y - arrow_len),
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        ax.text(arrow_x, arrow_y - arrow_len - img.shape[0] * 0.005,
                'N', color='white', fontsize=6, ha='center', va='bottom',
                fontweight='bold')
        ax.annotate('', xy=(arrow_x - arrow_len, arrow_y),
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        ax.text(arrow_x - arrow_len - img.shape[1] * 0.02, arrow_y,
                'E', color='white', fontsize=6, ha='right', va='center',
                fontweight='bold')

        # Label with source name and redshift
        z = info['z']
        label = f"{name}"
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                color='white', fontsize=8, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.6, edgecolor='none'))
        ax.text(0.97, 0.03, survey, transform=ax.transAxes,
                color='white', fontsize=6, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.5, edgecolor='none'))
    else:
        ax.text(0.5, 0.5, f'{name}\nNo image available',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
plt.savefig('figures/fig7_host_cutouts.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("\nSaved fig_host_cutouts.pdf")
plt.close()
