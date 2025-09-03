########################################################### imports
# all req files must be in a directory within 'Thumbstack' code

import sys
sys.path.append('../src/')

from headers import *
from thumbstack import ThumbStack

from astropy.table import Table
import astropy
from astropy import units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table, vstack
from scipy.cluster.vq import kmeans, vq

from pixell import enmap, enplot, reproject, utils, curvedsky 
import numpy as np

# from pixell: eshow which allows you to view maps
def eshow(x,**kwargs): 
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "ipython")

########################################################## cutting

# description: inputs fits file, outputs astropy table with imposed constraints

# requirments: fits file

# arguements:
# - file_name: name of file in fits format (string)
# - dict = {'Param_1': [lower_bound, upper_bound], 'Param_2': [lower_bound, upper_bound], etc}

# notes: key in 'Param_number' needs to match how it is written in the column name of the data
def cutting(file_name, dict, save=False, name=None):
    data=Table.read(file_name, format='fits')
    for key in dict:
        my_min=dict[key][0]
        my_max=dict[key][1]
        condition_1 = data[key] >= my_min
        new_tab=data[condition_1]
        condition_2=new_tab[key] <= my_max
        data=new_tab[condition_2] #rewrites what the data table so if there are multiple param boundaries it works with the already edited table
    
    if save:
        if name == None:
            name = 'altered_table'
        table_path=name+".fits"
        data.write(table_path, format='fits', overwrite=True)
    
    print('- Property boundaries applied')
    return data #returns a viewable table

########################################################## massage (THIS HAS SINCE BEEN REPLACED BY TABLEREADER IN THUMBSTACK.PY)
# description: inputs astropy table, outputs thumbstack friendly catalog

# requirements: astropy Table

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

# requirements: ACT map

# arguements: 
# - map_name: filename (with .fits and/or .txt extension; string)
# - save: if user wants to save the mask as fits (True/ False)
# - name: name of file the map will save as. do not add .fits, function does so. (string)

def mapping(map_name, save=False, name=None):
    print('- Map: Importing CMB map')
    result = enmap.read_map(map_name)

    print('- Map: Remove polarization')
    # if the map contains polarization, keep only temperature
    if len(result.shape) > 2:
        result = result[0]

    if save:
        that_name = name + '.fits'
        print('- Map: Saving map as '+that_name)
        enmap.write_map(that_name, result)

    print('- Map: Creation Done')
    return result


########################################################## masking
# description: inputs ACT CMB map, output mask that takes all obs array values that >1 become 1. 0 stays 0. also strips map to just color

# requirements: ACT map, planck map

# arguements: 
# - act_map: CMB footprint; filename (with .fits and/or .txt extension; string)
# - planck_map: galacitc mask; filename (with .fits and/or .txt extension; string)
# - ps_map: point source mask; filename (with .fits and/or .txt extension; string)
# - save: if user wants to save the mask as fits (True/ False)
# - name: name of file the map will save as. do not add .fits, function does so. (string)

def masking(act_map, planck_map=None, ps_map=None, save=False, name=None):
    if save is True and name is None:
        print('Error: save input is True but has no save name. Make a name= to continue')
        return
    
    print('- Mask: Importing CMB Map')
    CMBmap = enmap.read_map(act_map)
    
    # remove polarization and have all int values >1 become 1
    print('- Mask: Remove Polarization and have all int>1 become 1')
    if len(CMBmap.shape) > 2:
        CMBmap = CMBmap[0]
        result = np.logical_not(CMBmap == 0).astype(int)
    elif len(CMBmap.shape) <= 2:
        result = np.logical_not(CMBmap == 0).astype(int)
    
    if ps_map is not None:
        # point source map: no polarization to remove
        print('- Mask: Importing Point Source Map')
        pointSource_map = enmap.read_map(ps_map).astype(np.int64)
        result *= pointSource_map

    if planck_map is not None:
        print('- Mask: Importing Planck mask')
        # planck map mask
        pmap_fname = hp.read_map(planck_map)

        print('- Mask: Changing Planck mask to pixell friendly')
        pix_planck_map = reproject.healpix2map(pmap_fname, CMBmap.shape, CMBmap.wcs, rot=None, method='spline', spin=[0], order=0)
        result *= pix_planck_map

    if save:
        that_name = name + '.fits'
        print('- Mask: Saving mask as '+that_name)
        enmap.write_map(that_name, result)

    print('- Mask: Creation Done')
    return result


########################################################## binning
# description: bins galaxies based on a physical property and into (generally) equal groups, outputs table of several property min/ max of bins

# requirements: fits table

# notes: binning automatically outputs the following properties: 'RA', 'DEC', 'Z', 'LOGM', 'LOGSFR', 'AGNLUM'. adding these values to the add_val list is not necessary, but will not break the func if you do

# arguements: 
# - table_name: name of file in fits format (string)
# - n_bins: number of bins
# - gal_property: property to be binned by (string)
# - add_val: list of other paramaters from the data which can be binned or view binned min/max (list of strings)

def binning(table_name, n_bins, gal_property, add_val=None):
    table = Table.read(table_name, format='fits')

    # sort table based on the property provided
    sorted_tab = table[table[gal_property].argsort()]
    
    # get how many galaxies will be in each bin
    length = len(table)
    base_value = length // n_bins
    remainder = length % n_bins
    bins = [base_value + 1 if i < remainder else base_value for i in range(n_bins)] #list of bin amount

    original = []
    original = ['RA', 'DEC', 'Z', 'LOGM', 'LOGSFR', 'AGNLUM'] # based on DESI column names

    if add_val != None:
        for i in range(len(add_val)):
            while add_val[i] not in original: # added so if user adds already used vals into add_val, there won't be double columns
                original.append(add_val[i])

    if gal_property not in original:
        original.append(gal_property)

    out_add_val = {'n_objects': bins}
    for name in original:
        outMin = name + '_min'
        outMax = name + '_max'
        out_add_val[outMin] = []
        out_add_val[outMax] = []
    
    starting = 0
    for i in range(n_bins):
        ending = starting + bins[i]
        
        cut_table = Table() #clears out table from prev loop
        cut_table = sorted_tab[starting:ending]

        starting += bins[i]

        for name in original:
            outMin = name + '_min'
            outMax = name + '_max'
            out_add_val[outMin].extend([cut_table[name].min()])
            out_add_val[outMax].extend([cut_table[name].max()])

    new_tab = Table(out_add_val)

    return new_tab


########################################################## stacked_plot
# description: runs thumbstack for different bins of a property and returns a plot of the profile for all the bins

# requirements: fits table, map, mask

# notes: run the masking and mapping func seperately and save them so this func is not re-running the functions every loop

# arguements: 
# - table_name: name of file in fits format (string)
# - cmb_map: map already ran through mapping (.fits)
# - mask: mask already ran through mask (.fits)
# - n_bins: number of bins
# - gal_property: property to be binned by (current capabilities are for z, logm, logsfr, or agnlum); string
# - name: run output name (string)
def stacked_plot(table_name, cmb_map, mask, n_bins, gal_property, name):
    
    # run binning
    boundaries = binning(table_name, n_bins, gal_property)
    this_min = gal_property + '_min'
    this_max = gal_property + '_max'

    # run thumbstack with 
    for i in range(n_bins):
        outName = name + '_bin' + str(i+1)
        prop_dict = {} # clears dictionary from prev loop
        prop_dict = {gal_property: [boundaries[this_min][i], boundaries[this_max][i]]}

        ThumbStack(cutting(table_name, prop_dict), 
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
    print('- Stacked Plot: Creating Plot')
    data=Table.read(table_name, format='fits')
    
    ff = 18
    plt.rcParams["figure.figsize"] = (8,6)
    plt.figure()

    if str(data[gal_property].unit) == 'None':
        units = ''
        
    elif str(data[gal_property].unit) != 'None':
        units = ' ' + '[' + str(data[gal_property].unit) + ']'
        
    for i in range(n_bins):
        fits_name = 'figures/thumbstack/' + name + '_bin' + str(i+1) + '/ringring2_tsz_uniformweight.fits'
        fits_file = Table.read(fits_name, format='fits')
        if gal_property == 'AGNLUM':
            cap_name = "{:.4e}".format(boundaries[this_min][i]) + '< '+ gal_property + ' <' + "{:.4e}".format(boundaries[this_max][i])
        else: 
            cap_name = str(round(boundaries[this_min][i], 3)) + '< '+ gal_property + ' <' + str(round(boundaries[this_max][i], 3))
        plt.errorbar(fits_file['R'], fits_file['T'], yerr = fits_file['error'], alpha=0.4, label = cap_name, capsize=4, linewidth=3)

    plt.axhline(y=0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r'$R$ [arcmin]', fontsize=ff)
    plt.ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]', fontsize=ff)
    title = 'tSZ signal binned by ' + gal_property + units
    plt.title(title, fontsize=ff)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    fig_name=name+'_profile.pdf'
    plt.savefig(fig_name)
    plt.close()
    
    return

########################################################## pdf_hist
# description: generates array and pdf histogram for all objects in run

# requirements: name of thumbstack run

# arguements: 
# - table_name: name of thumbstack run (string)
# - column: column of txt file assessed (int; optional)
# - bins: number of bins for histogram (int; optional)
# - profile_plot: if you want an output viewable plot (True/ False; optional)
def pdf_hist(name, column=-1, bins=100, profile_plot=False):
    arr=np.genfromtxt('output/thumbstack/'+name+'/object_profiles.txt')
    plt.hist(arr[:,column],bins=bins)
    
    if profile_plot:
        m = np.mean(arr,axis=0)
        plt.plot(m)
        plt.close()
    
    return arr


########################################################## chi2
# description: creates chi2 value between two thumbstack runs

# requirements: name of thumbstack runs (2)

#notes: if assessing between 2 bins ran via stacked_plot(), the names in the chi2 func would be 'name_bin#' where the name in the '' is the name ran in the stacked_plot func

# arguements: 
# - name1/2: name of thumbstack run (string)
def chi2(name1, name2):
    bin1_cov= np.genfromtxt('output/thumbstack/'+name1+'/cov_ringring2_tsz_uniformweight_bootstrap.txt')
    bin2_cov= np.genfromtxt('output/thumbstack/'+name2+'/cov_ringring2_tsz_uniformweight_bootstrap.txt')
    bin1_prof=np.genfromtxt('output/thumbstack/'+name1+'/ringring2_tsz_uniformweight_measured.txt')[:,1]
    bin2_prof=np.genfromtxt('output/thumbstack/'+name2+'/ringring2_tsz_uniformweight_measured.txt')[:,1]

    v1 = bin1_prof - bin2_prof
    m = bin1_cov + bin2_cov
    m1 = np.linalg.inv(m)
    v2 = np.dot(v1,m1)
    result = np.dot(v2, v1)

    return result

########################################################## snr
# description: creates snr value of a single thumbstack run

# requirements: name of thumbstack run

#notes: this should be done using a thumbstack run of all objects, not divied up bin. this will create a larger snr value

# arguements: 
# - name: name of thumbstack run (string)

def snr(name):
    cov= np.genfromtxt('output/thumbstack/'+name+'/cov_ringring2_tsz_uniformweight_bootstrap.txt')
    prof=np.genfromtxt('output/thumbstack/'+name+'/ringring2_tsz_uniformweight_measured.txt')[:,1]

    m= np.linalg.inv(cov)
    v=np.dot(prof,m)
    
    return np.sqrt(np.dot(v,prof))

########################################################## boxey_feedback

# resulting plots can be found in directory boxed_script/{name}
def boxey_feedback(table, n_boxes, frequency, filterType, weights='uniform', splitMethod='median', boxPlot=False, name='test'):
    '''
    Input:
    
        table (astropy.table): astropy table that has property cuts made to it already. 
        n_boxes (int): number of boxes for the data to split into on the redshift and mass distribution
        frequency (string): 150ghz, 90ghz, kappa
        filterType (string): ringring2, ring
        weights (string): 
            'uniform': uniform weighting system; weights are defined by np.ones
            'mass': mass weighting system; more massive objects have more weight than less massive objects
        splitMethod (string): each box is split into high and low feedback via their log(m) by log(sfr/m) distributions
            'fit': best fit line made by log(m) distribution, then the split is defined by positive or negative difference from the best fit line
            'median': horizontal split based on the median values of log(sfr/m)
            'aglum': split based on median of agnlum, high and low values.
        boxPlot (bool): print what the boxed redshift v mass distribution
        name (string): output name

    Output:

        covariance, chi2
    '''

    ### check if the input are allowed inputs
    weights_allowed = ['uniform', 'mass']
    splitMethod_allowed = ['fit', 'median', 'agnlum']
    
    if weights not in weights_allowed:
        raise ValueError(f"Invalid weights input: '{weights}'. Must be one of {weights_allowed}.")

    if splitMethod not in splitMethod_allowed:
        raise ValueError(f"Invalid split_method input: '{splitMethod}'. Must be one of {splitMethod_allowed}.")

    if type(n_boxes) is not int:
        raise ValueError(f"Invalid n_boxes input type: '{n_boxes}'. Must be an integer.")

    ###

    os.makedirs(f'boxed_script/{name}', exist_ok=True)

    if weights == 'mass':
        calculated_weights = 10**table['LOGM'] / np.sum(10**table['LOGM'])
        table['mass_weights'] = calculated_weights

    table = table[table[f'catmask_{filterType}_{frequency}']==True]

    '''
    Splitting
    '''
    data = np.vstack((table['Z'], table['LOGM'])).T
    
    # k-means to group into `n_boxes` clusters

    print(f'making boxes')
    centroids, _ = kmeans(data, n_boxes)
    labels, _ = vq(data, centroids)
    
    boxes = {}
    for i in range(n_boxes):
        mask = labels == i
        box_data = data[mask]
        boxes[i] = {
            'z': box_data[:, 0],
            'logm': box_data[:, 1],
            'center': centroids[i]
    }

    if boxPlot:
        fig, ax = plt.subplots()
        scatter = ax.scatter(table['Z'], table['LOGM'], c=labels, cmap=f'tab20', s=0.1)
        
        # Show box centers and draw boundaries (approximate as circles)
        for i in range(n_boxes):
            cx, cy = boxes[i]['center']
            ax.plot(cx, cy, 'kx', markersize=12)
            ax.text(cx, cy, f'Box {i}', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))
        
        ax.set_title('Scatter Plot with Clustered Boxes')
        ax.set_xlabel('Z')
        ax.set_ylabel('LOGM')
        plt.grid(True)
        plt.savefig(f'boxed_script/{name}/boxey_plot.png', bbox_inches='tight')
        plt.close()

    '''
    Save box number to table and split into high and low feedback groups for every box
    '''
    completed=Table()
    completed.write(f'boxed_script/{name}/{name}_boxes.fits', overwrite=True)
    
    for i in range(len(boxes)):
        mask = list(zip(table['Z'], table['LOGM']))  # List of (x, y) from table
        targets = set(zip(boxes[i]['z'], boxes[i]['logm']))  # Set of target (x, y)
        
        matched = table[[pair in targets for pair in mask]]
        matched['box']=i

        dd = matched[matched['LOGM'].argsort()]
        v=dd['LOGSFR']-dd['LOGM']
        dd['sfr/m']=v

        if splitMethod == 'fit':
            A = np.polyfit(dd['LOGM'], v, 2)
            pp = np.poly1d(A)
            ff=v-pp(dd['LOGM'])
            
            dd['fdb']=ff
            
            mea = np.median(dd['fdb'])
            lo=dd[dd['fdb']>=mea] #low
            hi=dd[dd['fdb']<mea] #high

        if splitMethod == 'median':
            mea = np.median(10**dd['sfr/m'])
            lo=dd[dd['sfr/m']>=np.log10(mea)] #low
            hi=dd[dd['sfr/m']<np.log10(mea)] #high

        if splitMethod == 'agnlum':
            mea = np.median(dd['AGNLUM'])
            lo=dd[dd['AGNLUM']<mea] #low
            hi=dd[dd['AGNLUM']>=mea] #high

        lo['split']='low'
        hi['split']='high'

        merge = vstack([lo, hi])

        gen = Table.read(f'boxed_script/{name}/{name}_boxes.fits')
        merged_table = vstack([gen, merge])
        
        merged_table.write(f'boxed_script/{name}/{name}_boxes.fits', overwrite=True)

    '''
    Make high and low splits equal
    '''

    def match_table_lengths(table1, table2, seed=None):
        """
        Randomly removes rows from the longer table so both tables have equal length.
    
        Parameters:
        -----------
        table1 : astropy.table.Table
            The first input table.
        table2 : astropy.table.Table
            The second input table.
        seed : int, optional
            Random seed for reproducibility.
    
        Returns:
        --------
        new_table1 : astropy.table.Table
            Table 1 after possible truncation.
        new_table2 : astropy.table.Table
            Table 2 after possible truncation.
        """
        if seed is not None:
            np.random.seed(seed)
    
        len1 = len(table1)
        len2 = len(table2)
    
        if len1 == len2:
            return table1, table2
    
        if len1 > len2:
            indices_to_keep = np.random.choice(len1, len2, replace=False)
            new_table1 = table1[sorted(indices_to_keep)]
            return new_table1, table2
        else:
            indices_to_keep = np.random.choice(len2, len1, replace=False)
            new_table2 = table2[sorted(indices_to_keep)]
            return table1, new_table2

    low, high = match_table_lengths(merged_table[merged_table['split']=='low'], merged_table[merged_table['split']=='high'])

    '''
    Jackknife and Cov
    '''
    def jackknife_weighted_mean_cov_fast(X1, w1, X2, w2):
        """
        Fast jackknife estimation of weighted mean and its covariance.
    
        Parameters:
            X1: (n_samples, d1)
            w1: (n_samples,)
            X2: (n_samples, d2)
            w2: (n_samples,)
    
        Returns:
            mean1: (d1,)
            mean2: (d2,)
            cov11: (d1, d1) covariance of mean1
            cov22: (d2, d2) covariance of mean2
            cov12: (d1, d2) cross-covariance between mean1 and mean2
        """
        n = X1.shape[0]
        d1 = X1.shape[1]
        d2 = X2.shape[1]
    
        # Normalize weights once
        w1 = w1 / np.sum(w1)
        w2 = w2 / np.sum(w2)
    
        # Compute full means
        mean1 = X1.T @ w1  # shape (d1,)
        mean2 = X2.T @ w2  # shape (d2,)
    
        # Precompute full weighted sums
        S1 = X1.T * w1     # (d1, n)
        S2 = X2.T * w2     # (d2, n)
    
        # Leave-one-out weighted sums (efficient)
        total_w1 = np.sum(w1)
        total_w2 = np.sum(w2)
    
        sum1 = np.sum(S1, axis=1, keepdims=True) - S1  # (d1, n)
        sum2 = np.sum(S2, axis=1, keepdims=True) - S2  # (d2, n)
    
        w1_loo = total_w1 - w1  # (n,)
        w2_loo = total_w2 - w2  # (n,)
    
        # LOO means: shape (n, d1) and (n, d2)
        jk_mean1 = (sum1 / w1_loo).T  # (n, d1)
        jk_mean2 = (sum2 / w2_loo).T  # (n, d2)
    
        # Mean of jackknife means
        mean_jk1 = np.mean(jk_mean1, axis=0)
        mean_jk2 = np.mean(jk_mean2, axis=0)
    
        # Demeaned jackknife means
        diff1 = jk_mean1 - mean_jk1  # (n, d1)
        diff2 = jk_mean2 - mean_jk2  # (n, d2)
    
        # Jackknife covariance of the mean
        cov11 = (n - 1) / n * diff1.T @ diff1  # (d1, d1)
        cov22 = (n - 1) / n * diff2.T @ diff2  # (d2, d2)
        cov12 = (n - 1) / n * diff1.T @ diff2  # (d1, d2)
    
        return mean1, mean2, cov11, cov22, cov12


    if weights == 'uniform':
        m1,m2,c11,c22,c12=jackknife_weighted_mean_cov_fast(low[f'profile_{filterType}_{frequency}'], np.ones(len(low)), 
                                                           high[f'profile_{filterType}_{frequency}'], np.ones(len(high)))

    if weights == 'mass':
        m1,m2,c11,c22,c12=jackknife_weighted_mean_cov_fast(low[f'profile_{filterType}_{frequency}'], low['mass_weights'], 
                                                           high[f'profile_{filterType}_{frequency}'], high['mass_weights'])

    '''
    Plot
    '''
    r=np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ])

    if splitMethod == 'agnlum':
        plt.errorbar(r, m1, yerr=np.diag(c11)**0.5, 
                     label=f"{'{:.2e}'.format(min(low['AGNLUM']))} < AGNLUM < {'{:.2e}'.format(max(low['AGNLUM']))} | low")
        plt.errorbar(r, m2, yerr=np.diag(c22)**0.5, 
                     label=f"{'{:.2e}'.format(min(high['AGNLUM']))} < AGNLUM < {'{:.2e}'.format(max(high['AGNLUM']))} | high")
        
    else:
        plt.errorbar(r, m1, yerr=np.diag(c11)**0.5, label='low feedback')
        plt.errorbar(r, m2, yerr=np.diag(c22)**0.5, label='high feedback')
    
    plt.axhline(y=0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r'$R$ [arcmin]')
    plt.ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]')
    plt.title(f'Profile')
    plt.legend()
    
    plt.savefig(f'boxed_script/{name}/{name}_profile.png', bbox_inches='tight')
    
    #plt.show()
    plt.close()

    '''
    Save radius, mean, and error plots
    '''
    np.savetxt(f'boxed_script/{name}/{name}_profile_lowf.txt', np.column_stack((r, m1, np.diag(c11)**0.5)), delimiter="\t")
    np.savetxt(f'boxed_script/{name}/{name}_profile_highf.txt', np.column_stack((r, m2, np.diag(c22)**0.5)), delimiter="\t")

    '''
    Cov and Chi2
    '''
    c_d = c11+c22-c12- c12.T
    np.savetxt(f'boxed_script/{name}/{name}_covariance.txt', c_d)

    chi=np.dot(np.dot((m1-m2),np.linalg.inv(c_d)), (m1-m2))
    np.savetxt(f'boxed_script/{name}/{name}_chi2.txt', np.array([chi]))
    
    return c_d, chi