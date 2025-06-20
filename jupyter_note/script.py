import matplotlib.pyplot as plt
import pandas as pd
from pixell import enmap, utils
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import sys
from astropy.table import Table, vstack
from hep_ml.reweight import GBReweighter

sys.path.append('../ThumbStack/src/')

import my_functions
from my_functions import binning, cutting, binning, stacked_plot, masking, mapping, eshow
import thumbstack
from thumbstack import ThumbStack

class individualWeightedPlot(object):

    def __init__(self, tableName, outName='50n_max0.7', capFilter='ringring2', resolution=1., split=True, save=False, tsCompute=False, outputFile='outputs.txt', mapPath='tsz', maskPath='masks', zLim=0.7):
        
        self.outName = outName
        self.output = np.loadtxt(outputFile)
        self.zLim = zLim
        self.mapPath = mapPath
        self.maskPath = maskPath
        self.capFilter = capFilter
        self.res=resolution

        if save:
            uwTable=self.computeFeedbackSplit(tableName)
            self.table=self.computeIndividualWeights(uwTable, tableName+'_w.fdb_sfrm.fits')

        self.table=Table.read(tableName)

        if tsCompute:
            if split:
                self.table=self.tsSplitRun(tableName)
            if not split:
                self.table=self.tsFullRun(tableName)

        self.outName = outName
        self.plotIndivWeight()


    def computeFeedbackSplit(self, tableName, slope=1.):
        df=Table.read(tableName)
        
        df1=df[df['LOGM']>9.5] #9.5 being the cut off mass for simba, where galaxies>9.5 logm create feedback
        dd = df1[df1['LOGM'].argsort()]
        
        v=dd['LOGSFR']-dd['LOGM']
        A = np.polyfit(dd['LOGM'], v, slope)
        pp = np.poly1d(A)
        ff=v-pp(dd['LOGM'])
        
        dd['fdb']=ff
        dd['sfr/m']=v
        
        dd.write(tableName+'_w.fdb_sfrm.fits', format='fits', overwrite=True)
        print('Individual Weight Plot: split calculated')

        return dd

    def computeIndividualWeights(self, table, n_estimators=100):
        print('Individual Weight Plot: weights calculating...')
        dd = table # sim data
        data=Table.read('../ThumbStack/jupyter_note/data/FujiBGSPhysProp_v1.1.fits') #real data
        data=data[(data["Z"] <0.7)&(data["LOGSFR"] >-3)&(data["LOGM"] >9.5)] # splits
        
        feature_names = ['Z', 'LOGM', 'LOGSFR'] # names of properties to be weighted by
        
        df = dd[feature_names].to_pandas()
        df0 = data[feature_names].to_pandas()
        sim_data= pd.DataFrame(df, columns=feature_names)
        real_data = pd.DataFrame(df0, columns=feature_names)
        
        reweighter = GBReweighter(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=3
        )
        
        reweighter.fit(original=sim_data, target=real_data)
        
        weights = reweighter.predict_weights(sim_data)
        
        dd['weights']=weights
        dd.write(tableName+'_wWeights.fits')

        print('Individual Weight Plot: weights calculation completed')

        return dd

    def tsSplitRun(self, tableName, outliers=[126,151], SIMTYPE='m50n512'):
        outputs = self.output
        mapPath = self.mapPath
        maskPath = self.maskPath
        name = self.outName

        blank=Table()
        blank.write(name+'wProfiles.fits', overwrite=True)

        this_data=self.table

        for i in range(len(outputs[:,0])):
            if outputs[i, 1]>self.zLim:
                continue
            elif int(outputs[i, 2]) in outliers:
                continue
            else:
                num  = int(outputs[i,2])
                gg = enmap.read_map(f"{mapPath}/tmap_{SIMTYPE}_npix-4096_{num}.fits")

                ggf= enmap.fft(gg, normalize='phys')
                modl= gg.modlmap()
                theta=self.res/60*np.pi/180
                
                bmap=np.exp(-(theta*modl)**2/16/np.log(2))
                ggc=np.real(enmap.ifft(ggf*bmap, normalize='phys'))
                
                masked = enmap.read_map(f"{maskPath}/tmap_{SIMTYPE}_npix-4096_{num}_mask.fits")
            
                this_data1 = this_data[(this_data['Z']==outputs[i, 1]) & (this_data['fdb']>=0)]
                this_data2 = this_data[(this_data['Z']==outputs[i, 1]) & (this_data['fdb']<0)]
                this_name1 = f"{name}_lo_{num}"
                this_name2 = f"{name}_hi_{num}"
        
                ts = ThumbStack(this_data1, 
                                ggc, 
                                masked, 
                                name=this_name1,
                                nameLong=None, 
                                save=True,
                                nProc=32,
                                filterTypes=self.capFilter,
                                #doMBins=False, 
                                doBootstrap=True,
                                # doStackedMap=True,
                                #doVShuffle=False, 
                                #cmbNu=cmap.nu, 
                                #cmbUnitLatex=cmap.unitLatex,
                                pathOut='')
        
                filtered1=(ts.catalogMask(overlap=True, psMask=False, filterType='ringring2')).tolist()
                this_data1['filt']=filtered1
                this_data1['profile']=np.loadtxt('output/thumbstack/'+this_name1+'/ringring2_filtmap.txt')*(180*60/np.pi)**2
                this_data1.write('output/thumbstack/'+this_name1+'/objTable.fits', overwrite=True)
                
                ts = ThumbStack(this_data2, 
                                ggc, 
                                masked, 
                                name=this_name2,
                                nameLong=None, 
                                save=True,
                                nProc=32,
                                filterTypes=self.capFilter,
                                #doMBins=False, 
                                doBootstrap=True,
                                # doStackedMap=True,
                                #doVShuffle=False, 
                                #cmbNu=cmap.nu, 
                                #cmbUnitLatex=cmap.unitLatex,
                                pathOut='')
                
                filtered2=(ts.catalogMask(overlap=True, psMask=False, filterType='ringring2')).tolist()
                this_data2['filt']=filtered2
                this_data2['profile']=np.loadtxt('output/thumbstack/'+this_name2+'/ringring2_filtmap.txt')*(180*60/np.pi)**2
                this_data2.write('output/thumbstack/'+this_name2+'/objTable.fits', overwrite=True)
        
                gen = Table.read(name+'wProfiles.fits')
                
                both = vstack([this_data1, this_data2])
                merged_table = vstack([gen, both])
        
                merged_table.write(name+'wProfiles.fits', overwrite=True)

        return merged_table

    def tsFullRun(self, tableName, outliers=[126,151], SIMTYPE='m50n512'):
        outputs = self.output
        mapPath = self.mapPath
        maskPath = self.maskPath
        name = self.outName

        blank=Table()
        blank.write(name+'wProfiles.fits', overwrite=True)

        this_data=self.table

        for i in range(len(outputs[:,0])):
            if outputs[i, 1]>self.zLim:
                continue
            elif int(outputs[i, 2]) in outliers:
                continue
            else:
                num  = int(outputs[i,2])
                gg = enmap.read_map(f"{mapPath}/tmap_{SIMTYPE}_npix-4096_{num}.fits")

                ggf= enmap.fft(gg, normalize='phys')
                modl= gg.modlmap()
                theta=self.res/60*np.pi/180
                
                bmap=np.exp(-(theta*modl)**2/16/np.log(2))
                ggc=np.real(enmap.ifft(ggf*bmap, normalize='phys'))
                
                masked = enmap.read_map(f"{maskPath}/tmap_{SIMTYPE}_npix-4096_{num}_mask.fits")
            
                this_data1 = this_data[(this_data['Z']==outputs[i, 1])]
                this_name1 = f"{name}_{num}"
        
                ts = ThumbStack(this_data1, 
                                ggc, 
                                masked, 
                                name=this_name1,
                                nameLong=None, 
                                save=True,
                                nProc=32,
                                filterTypes=self.capFilter,
                                #doMBins=False, 
                                doBootstrap=True,
                                # doStackedMap=True,
                                #doVShuffle=False, 
                                #cmbNu=cmap.nu, 
                                #cmbUnitLatex=cmap.unitLatex,
                                pathOut='')
        
                filtered1=(ts.catalogMask(overlap=True, psMask=False, filterType='ringring2')).tolist()
                this_data1['filt']=filtered1
                this_data1['profile']=np.loadtxt('output/thumbstack/'+this_name1+'/ringring2_filtmap.txt')*(180*60/np.pi)**2
                this_data1.write('output/thumbstack/'+this_name1+'/objTable.fits', overwrite=True)
        
                gen = Table.read(name+'wProfiles.fits')
                merged_table = vstack([gen, this_data1])
        
                merged_table.write(name+'wProfiles.fits', overwrite=True)

        return merged_table


    def plotIndivWeight(self, filt=True, r=np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ])):
        dd=self.table
        path=self.outName

        if filt:
            dd=dd[dd['filt']==True]

        def weighted_mean_and_jackknife_cov(data, weights=None):
            """
            Fast computation of weighted mean and jackknife covariance.
           
            Parameters
            ----------
            data : array-like of shape (N, D)
                Array of N data vectors of length D.
            weights : array-like of shape (N,), optional
                Weights per data point. Defaults to uniform weights.
           
            Returns
            -------
            mean : ndarray of shape (D,)
                Weighted mean of the data.
            cov : ndarray of shape (D, D)
                Jackknife covariance matrix.
            """
            data = np.asarray(data)   # shape (N, D)
            N, D = data.shape
        
            if weights is None:
                weights = np.ones(N)
            else:
                weights = np.asarray(weights)
        
            # Total weighted sum and mean
            W_total = np.sum(weights)
            sum_weighted = np.sum(data * weights[:, None], axis=0)
            mean = sum_weighted / W_total
        
            # Vectorized jackknife: leave-one-out sums and means
            W_jk = W_total - weights               # shape (N,)
            sum_jk = sum_weighted - data * weights[:, None]  # shape (N, D)
            mean_jk = sum_jk / W_jk[:, None]       # shape (N, D)
        
            # Jackknife estimate of covariance
            jk_mean = np.mean(mean_jk, axis=0)
            diffs = mean_jk - jk_mean              # shape (N, D)
            cov = (N - 1) / N * diffs.T @ diffs    # shape (D, D)
        
            return mean, cov

        m,c=weighted_mean_and_jackknife_cov(np.array(dd['profile']), np.array(dd['weights']))
        plt.errorbar(r, m*(180*60/np.pi)**2, yerr=np.diag(c)**0.5 *(180*60/np.pi)**2)
        plt.axhline(y=0, color='black', linewidth=1, linestyle='--')
        plt.xlabel(r'$R$ [arcmin]')
        plt.ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]')

        plt.savefig(f"{path}_weighted_prof.pdf", bbox_inches='tight')

        plt.show()
        return