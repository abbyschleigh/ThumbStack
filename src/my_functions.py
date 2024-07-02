########################################################### imports
# all req files must be in a directory within 'Thumbstack' code

import sys
sys.path.append('../src/')

from headers import *

from astropy.table import Table
import astropy

from pixell import enmap, enplot, reproject, utils, curvedsky 
import numpy as np

# from pixell: eshow which allows you to view maps
def eshow(x,**kwargs): 
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "ipython")

########################################################## cutting

# description: inputs fits file, outputs astropy table with imposed constraints

# requirments: my_functions.py, fits file

# arguements:
# - file_name: name of file (without .fits extension ; string)
# - dict = {'Param_1': [lower_bound, upper_bound], 'Param_2': [lower_bound, upper_bound], etc}

# notes: key in 'Param_number' needs to match how it is written in the column name of the data
def cutting(file_name, dict):
    data=Table.read(file_name+'.fits', format='fits')
    for key in dict:
        my_min=dict[key][0]
        my_max=dict[key][1]
        condition_1 = data[key] >= my_min
        new_tab=data[condition_1]
        condition_2=new_tab[key] <= my_max
        data=new_tab[condition_2] #rewrites what the data table so if there are multiple param boundaries it works with the already edited table
        print('- Property boundaries applied')
    return data #returns a viewable table

########################################################## massage (THIS HAS SINCE BEEN REPLACED BY TABLEREADER IN THUMBSTACK.PY)
# description: inputs astropy table, outputs thumbstack friendly catalog

# requirements: my_functions.py, astropy Table

# arguements:
# - data_table: astropy table (in our case would do data_table=cutting(...) )
# - directory_name: name of the output catalog folder (string)


def massage(data_table, directory_name):
    ################## set-up
    filterType = 'diskring'
    T_CIB = '10.7'
    # T_CIB = '24.0'

    nProc = 64  # 1 haswell node on cori

    # cosmological parameters
    #u = UnivMariana()

    # M*-Mh relation
    massConversion = MassConversionKravtsov14()
    MStellar = massConversion.fmVirTomStar(2e13)
    
    ################ the actual process
    fn  = directory_name
    #cat = Table.read(fn+'.fits', format='fits') commented out bc cutting func already does this
    cat = data_table
    
    names = [name for name in cat.colnames if len(cat[name].shape) <= 1]
    df  = cat[names].to_pandas()
    df  = df.loc[:, ('RA', 'DEC', 'Z')]
    colnames = ['coordX', 'coordY', 'coordZ', 'dX', 'dY', 'dZ', 'dXKaiser', 'dYKaiser', 'dZKaiser','vX', 'vY', 'vZ', 'vR', 'vTheta', 'vPhi']
    for col in colnames: df[col]=0
    df['MStellar'] = MStellar
    
    # below func will create a txt file with the directory name, not the original data file name. this file will be in the same directory of the notebook
    df.to_csv(fn+'.txt', header=None, index=None, sep=' ', mode='w')
    
    # creates catalog txt file, can be found in ../ouput/catalog
    Catalog(massConversion, name=fn, nameLong=fn+"cool_desi_bgs_galaxy", pathInCatalog=fn+'.txt', save=True)
    
    # grab catalog object
    ex = Catalog(massConversion, name=fn, nameLong=fn+"cool_desi_bgs_galaxy", save=False)
    
    return ex

########################################################## mapping
# description: inputs ACT CMB map, outputs map w/o poloarization. can specify ra and dec, not req

# requirements: my_functions.py, ACT map

# arguements: 
# - map_name: filename (with .fits and/or .txt extension; string)
# - save: if user wants to save the mask as fits (True/ False)
# - save_name: name of file the map will save as. do not add .fits, function does so. (string)

def mapping(map_name, save=False, save_name=None):
    print('- Map: Importing CMB map')
    result = enmap.read_map(map_name)

    print('- Map: Remove polarization')
    # if the map contains polarization, keep only temperature
    if len(result.shape) > 2:
        result = result[0]

    if save:
        that_name = save_name + '.fits'
        print('- Map: Saving map as '+that_name)
        enmap.write_map(that_name, result)

    print('- Map: Creation Done')
    return result


########################################################## masking
# description: inputs ACT CMB map, output mask that takes all obs array values that >1 become 1. 0 stays 0. also strips map to just color

# requirements: my_functions.py, ACT map

# arguements: 
# - act_map: filename (with .fits and/or .txt extension; string)
# - planck_map: filename (with .fits and/or .txt extension; string)
# - save: if user wants to save the mask as fits (True/ False)
# - save_name: name of file the map will save as. do not add .fits, function does so. (string)

def masking(act_map, planck_map, save=False, save_name=None):
    print('- Mask: Importing CMB Map')
    CMBmap = enmap.read_map(act_map)
    
    # remove polarization and have all int values >1 become 1
    print('- Mask: Remove Polarization and have all int>1 become 1')
    if len(CMBmap.shape) > 2:
        CMBmap = CMBmap[0]
        CMBmap_edit = np.logical_not(CMBmap == 0).astype(int)
    elif len(CMBmap.shape) <= 2:
        CMBmap_edit = np.logical_not(CMBmap == 0).astype(int)

    print('- Mask: Importing Planck mask')
    # planck map mask
    pmap_fname = hp.read_map(planck_map)

    print('- Mask: Changing Planck mask to pixell friendly')
    planck_map = reproject.healpix2map(pmap_fname, CMBmap_edit.shape, CMBmap_edit.wcs, rot=None, method='spline', spin=[0], order=0)

    # result of all processes
    result = planck_map*CMBmap_edit

    if save:
        that_name = save_name + '.fits'
        print('- Mask: Saving mask as '+that_name)
        enmap.write_map(that_name, result)

    print('- Mask: Creation Done')
    return result