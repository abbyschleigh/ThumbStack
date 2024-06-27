import astropy
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import fitsio
import healpy as hp

import sys
from importlib import reload
sys.path.append('../src/')

import catalog
reload(catalog)
from catalog import *

import universe
reload(universe)
from universe import *

import mass_conversion
reload(mass_conversion)
from mass_conversion import *

# DESI catalog location
# fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/catalogs/dr9_lrg_1.1.1_pzbins_20221204.fits'
fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/catalogs/dr9_lrg_pzbins_20230509.fits'
cat = Table(fitsio.read(fn))
print(len(cat))

min_nobs = 2
max_ebv = 0.15
max_stardens = 2500

massConversion = MassConversionKravtsov14()
# Converting a flat virial mass to Stellar Mass, this is what we're using for every object for now.
MStellar = massConversion.fmVirTomStar(2e13)
# We run the code for the four DESI redshift bins in a for loop.
bins = [1, 2, 3, 4]

for pz_bin in bins:
    if pz_bin == 1:
        MStellar = massConversion.fmVirTomStar(10**(13.4))
    elif pz_bin == 2:
        MStellar = massConversion.fmVirTomStar(10**(13.24))
    else:
        MStellar = massConversion.fmVirTomStar(2e13)

    
    cat = Table.read(fn, format='fits')
    
    # Remove area south of DEC=-29  (this was not applied Martin's paper)
    mask = cat['DEC']>-29
    print('Remove area south of DEC=-29', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
    cat = cat[mask]

    # Remove "islands" in the NGC
    mask = ~((cat['DEC']<-10.5) & (cat['RA']>120) & (cat['RA']<260))
    print('Remove islands', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
    cat = cat[mask]

    # NOBS cut
    mask = (cat['PIXEL_NOBS_G']>=min_nobs) & (cat['PIXEL_NOBS_R']>=min_nobs) & (cat['PIXEL_NOBS_Z']>=min_nobs)
    print('NOBS', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
    cat = cat[mask]

    # Apply LRG mask
    mask = cat['lrg_mask']==0
    print('LRG mask', np.sum(mask), np.sum(~mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

    # Martin's EBV cut
    mask = cat['EBV']<max_ebv
    print('EBV', np.sum(mask), np.sum(~mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

    # Martin's STARDENS cut
    stardens = np.load('/global/cfs/cdirs/desi/users/rongpu/useful/healpix_maps/pixweight-dr7.1-0.22.0_stardens_64_ring.npy')
    stardens_nside = 64
    mask = stardens>=max_stardens
    bad_hp_idx = np.arange(len(stardens))[mask]
    cat_hp_idx = hp.pixelfunc.ang2pix(stardens_nside, cat['RA'], cat['DEC'], lonlat=True, nest=False)
    mask_bad = np.in1d(cat_hp_idx, bad_hp_idx)
    print('STARDENS', np.sum(~mask_bad), np.sum(mask_bad), np.sum(mask_bad)/len(mask_bad))
    cat = cat[~mask_bad]

    # Bin Cut
    mask = (cat['pz_bin']==pz_bin)
    print('pz_bin_cut', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
    cat = cat[mask]

    print(len(cat))
    
    # Converting to Pandas and renaming the columns
    df = cat.to_pandas()
    df2 = df.loc[:, ('RA', 'DEC', 'Z_PHOT_MEDIAN')]
    df2.rename(columns={'Z_PHOT_MEDIAN':'Z'}, inplace=True)
    
    
    # New Columns with Mstellar given from above
    colnames = ['coordX', 'coordY', 'coordZ', 'dX', 'dY', 'dZ', 'dXKaiser', 'dYKaiser', 'dZKaiser',
                'vX', 'vY', 'vZ', 'vR', 'vTheta', 'vPhi']

    for col in colnames:
        df2[col]=0

    df2['MStellar'] = MStellar
    # df2['hasM'] = 0
    
    # Saving the fits catalogs as txt files
    df2.to_csv(r'../output/catalog/DESI_pz{}/catalog.txt'.format(pz_bin), header=None, index=None, sep=' ', mode='w')
    print('Saved DESI_pz{}'.format(pz_bin))

    
# Lastly we read & save these again using the built in ThumbStack catalog function, so it adds the last few columns (like mVir) for us.
# cosmological parameters
u = UnivMariana()

for i in range(4):
    j = i+1
    Catalog(u, massConversion, name="DESI_pz{}".format(j), nameLong="DESI pz bin {}".format(j), pathInCatalog='../output/catalog/DESI_pz{}/catalog.txt'.format(j), save=True)