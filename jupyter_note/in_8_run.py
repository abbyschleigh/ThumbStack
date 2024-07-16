import sys
sys.path.append('../src/')
import thumbstack
from thumbstack import *
from my_functions import cutting, mapping, masking, binning
import numpy as np
from pixell import enmap
import mass_conversion
from mass_conversion import *
import matplotlib.pyplot as plt

my_map = enmap.read_map('in_8_test_map.fits')
my_mask = enmap.read_map('in_8_test_mask.fits')

def stacked_plot(catalog_name, cmb_map, mask, n_bins, gal_property, name):
    
    # run binning
    boundaries = binning(catalog_name, n_bins, gal_property, tabled=True)

    # run thumbstack with 
    for i in range(n_bins):
        outName = name + '_bin' + str(i+1)
        this_min = gal_property + '_min'
        this_max = gal_property + '_max'
        prop_dict = {} # clears dictionary from prev loop
        prop_dict = {gal_property: [boundaries[this_min][i], boundaries[this_max][i]]}

        ThumbStack(cutting(catalog_name, prop_dict), 
                   cmb_map, 
                   mask, 
                   name=outName,
                   nameLong=None, 
                   save=True, 
                   nProc=32,
                   filterTypes='ringring2',
                   #doMBins=False, 
                   doBootstrap=True,
                   # doStackedMap=True,
                   #doVShuffle=False, 
                   #cmbNu=cmap.nu, 
                   #cmbUnitLatex=cmap.unitLatex,
                   pathOut='')
    
    # plotting
    plt.figure()
    for i in range(n_bins):
        fits_name = 'figures/thumbstack/' + name + '_bin' + str(i+1) + '/ringring2_tsz_uniformweight.fits'
        fits_file = Table.read(fits_name, format='fits')
        cap_name = 'bin ' + str(i+1)
        plt.errorbar(fits_file['R'], fits_file['T'], yerr = fits_file['error'], alpha=0.4, label=cap_name)
    
    plt.xlabel(r'$R$ [arcmin]')
    plt.ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]')
    plt.legend()
    
    fig_name=name+'_profile.png'
    plt.savefig(fig_name)
    
    plt.show()
    
    return

stacked_plot('in_8_test', my_map, my_mask, 3, 'RA', 'in_8_test_run')