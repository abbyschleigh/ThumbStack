# Some notes for Abby
# change pathOut to point to your local ThumbStack directory
pathOut = 'your/path/to/ThumbStack'

from headers import *

##################################################################################
##################################################################################

# meant to replace catalog object
class TableReader(object):
    def __init__(self, table, nObj=None, name=None):
        self.table = table

        if name is None:
            self.name='this_data'
        else:
            self.name=name

        self.read_table(nObj=nObj)
        

    def read_table(self, nObj=None):
        print("- load full table")
        data = self.table
        self.nObj = len(data)
        #
        # sky coordinates and redshift
        
        self.RA = np.array(data['RA']) # [deg]
        self.DEC = np.array(data['DEC'])   # [deg]
        self.Z = np.array(data['Z'])

class ThumbStack(object):

    def __init__(self, table, cmbMap, cmbMask, cmbHit=None, name="test", nameLong=None, save=False, nProc=1, filterTypes='diskring', doStackedMap=False, doBootstrap=False, cmbNu=150.e9, cmbUnitLatex=r'$\mu$K', pathOut=pathOut, rApMinArcmin=2., rApMaxArcmin=6., rApInnerRad=1.):

        self.nProc = nProc
        self.Catalog = TableReader(table,name=name) # table is assumed to be an astropy table
        self.name = name
        if nameLong is None:
            self.nameLong = self.name
        else:
            self.nameLong = nameLong
        self.cmbMap = cmbMap
        self.cmbMask = cmbMask
        self.cmbHit = cmbHit
        self.doBootstrap = doBootstrap
        self.cmbNu = cmbNu
        self.cmbUnitLatex = cmbUnitLatex

        self.rApInnerRad = rApInnerRad
        self.rApMinArcmin = rApMinArcmin
        self.rApMaxArcmin = rApMaxArcmin
        
        # aperture photometry filters to implement
        if filterTypes == 'diskring':
            self.filterTypes = np.array(['diskring'])
        elif filterTypes == 'ringring':
            self.filterTypes = np.array(['ringring'])
        elif filterTypes == 'ringring2':
            self.filterTypes = np.array(['ringring2'])
        elif filterTypes == 'disk':
            self.filterTypes = np.array(['disk'])
        elif filterTypes == 'ring':
            self.filterTypes = np.array(['ring'])
        elif filterTypes == 'all':
            self.filterTypes = np.array(['diskring', 'ringring', 'ringring2', 'disk', 'ring'])

        # estimators (ksz, tsz) and weightings (uniform, hit, var, ...)
        # for stacked profiles, bootstrap cov and v-shuffle cov
        if self.cmbHit is not None:
            # ['tsz_uniformweight', 'tsz_hitweight', 'tsz_varweight', 'ksz_uniformweight', 'ksz_hitweight', 'ksz_varweight', 'ksz_massvarweight']
            self.Est = ['tsz_uniformweight', 'tsz_varweight']
            # ['tsz_varweight', 'ksz_varweight']
            self.EstBootstrap = ['tsz_uniformweight', 'tsz_varweight']
            self.EstVShuffle = []  # ['ksz_varweight']
            # ['tsz_varweight', 'ksz_varweight']
            self.EstMBins = ['tsz_uniformweight', 'tsz_varweight']
        else:
            # ['tsz_uniformweight', 'ksz_uniformweight', 'ksz_massvarweight']
            self.Est = ['tsz_uniformweight']
            # ['tsz_uniformweight', 'ksz_uniformweight']
            self.EstBootstrap = ['tsz_uniformweight']
            self.EstVShuffle = []  # ['ksz_uniformweight']
            # ['ksz_uniformweight'] #['tsz_uniformweight', 'ksz_uniformweight']
            self.EstMBins = ['tsz_uniformweight']

        # resolution of the cutout maps to be extracted
        self.resCutoutArcmin = 0.25   # [arcmin]
        # projection of the cutout maps
        self.projCutout = 'cea'

        # number of samples for bootstraps, shuffles
        self.nSamples = 10000

        # Output path
        self.pathOut = pathOut + "output/thumbstack/"+self.name
        if not os.path.exists(self.pathOut):
            os.makedirs(self.pathOut)

        # Figures path
        self.pathFig = "./figures/thumbstack/"+self.name
        if not os.path.exists(self.pathFig):
            os.makedirs(self.pathFig)
        # test figures path
        self.pathTestFig = self.pathFig+"/tests"
        if not os.path.exists(self.pathTestFig):
            os.makedirs(self.pathTestFig)

        print("- Thumbstack: "+str(self.name))

        self.loadAPRadii()
        #self.loadMMaxBins()

        if save:
            self.saveOverlapFlag(nProc=self.nProc)
        self.loadOverlapFlag()

        if save:
            self.saveFiltering(nProc=self.nProc)
        self.loadFiltering()

        self.measureAllVarFromHitCount(plot=save)

#      self.measureAllMeanTZBins(plot=save, test=False)

        if save:
            self.saveAllStackedProfiles()
        self.loadAllStackedProfiles()

        if save:
            self.plotAllStackedProfiles()
            self.plotAllCov()
            self.computeAllSnr()

        # if save:
        if True:
            if doStackedMap:
                # save all stacked maps
                # self.saveAllStackedMaps()
                # save only the stacked maps for
                # the best tsz and ksz estimators,
                # and for the diskring weighting
                # self.saveAllStackedMaps(filterTypes=['diskring'], Est=['tsz_varweight', 'ksz_varweight'])
                self.saveAllStackedMaps(filterTypes=None, Est=None)

    ##################################################################################
    ##################################################################################

    def loadAPRadii(self):
        # radii to use for AP filter: comoving Mpc/h
        self.nRAp = 9  # 30 #9  #4

        # Aperture radii in Mpc/h
        # self.rApMinMpch = 1.
        # self.rApMaxMpch = 5
        # self.RApMpch = np.linspace(self.rApMinMpch, self.rApMaxMpch, self.nRAp)

        # Aperture radii in arcmin
        # self.rApInnerRad = 1.
        # self.rApMinArcmin = 2.  # 0.1   #1.  # 1.
        # if self.filterTypes == 'ringring':
        #     self.rApMinArcmin = 2.  # 0.1   #1.  # 1.
        # else:
        #     self.rApMinArcmin = 1.  # 0.1   #1.  # 1.
        # self.rApMaxArcmin = 6.  # 6.  #6.  # 4.
        self.RApArcmin = np.linspace(
            self.rApMinArcmin, self.rApMaxArcmin, self.nRAp)

    def cutoutGeometry(self, test=False):
        '''Create enmap for the cutouts to be extracted.
        Returns a null enmap object with the right shape and wcs.
        '''

        # choose postage stamp size to fit the largest ring
        dArcmin = np.ceil(2. * self.rApMaxArcmin * np.sqrt(2.))
        #
        nx = np.floor((dArcmin / self.resCutoutArcmin - 1.) / 2.) + 1.
        dxDeg = (2. * nx + 1.) * self.resCutoutArcmin / 60.
        ny = np.floor((dArcmin / self.resCutoutArcmin - 1.) / 2.) + 1.
        dyDeg = (2. * ny + 1.) * self.resCutoutArcmin / 60.

        # define geometry of small square maps to be extracted
        shape, wcs = enmap.geometry(np.array([[-0.5*dxDeg, -0.5*dyDeg], [0.5*dxDeg, 0.5*dyDeg]])
                                    * utils.degree, res=self.resCutoutArcmin*utils.arcmin, proj=self.projCutout)
        cutoutMap = enmap.zeros(shape, wcs)

        if test:
            print("cutout sides are dx, dy =",
                  dxDeg*60., ",", dyDeg*60., "arcmin")
            print("cutout pixel dimensions are", shape)
            print("hence a cutout resolution of", dxDeg*60. /
                  shape[0], ",", dyDeg*60./shape[1], "arcmin per pixel")
            print("(requested", self.resCutoutArcmin, "arcmin per pixel)")

        return cutoutMap


    ##################################################################################

    def sky2map(self, ra, dec, map):
        '''Gives the map value at coordinates (ra, dec).
        ra, dec in degrees.
        Uses nearest neighbor, no interpolation.
        Will return 0 if the coordinates requested are outside the map
        '''
        # interpolate the map to the given sky coordinates
        # convert from degrees to radians
        sourcecoord = np.array([dec, ra]) * utils.degree
        # use nearest neighbor interpolation
        return map.at(sourcecoord, prefilter=False, mask_nan=False, order=0)

    # def saveOverlapFlag(self, thresh=1.e-5, nProc=1):

    def saveOverlapFlag(self, thresh=0.95, nProc=1):
        '''1 for objects that overlap with the hit map,
        0 for objects that don't.
        '''
        # print("Create overlap flag")
        # find if a given object overlaps with the CMB hit map

        def foverlap(iObj):
            '''Returns 1. if the object overlaps with the hit map and 0. otherwise.
            '''
            ra = self.Catalog.RA[iObj]
            dec = self.Catalog.DEC[iObj]
            hit = self.sky2map(ra, dec, self.cmbMask)
            return float(hit > thresh)

        tStart = time()
        with sharedmem.MapReduceByThread(np=nProc) as pool:
            overlapFlag = np.array(
                pool.map(foverlap, list(range(self.Catalog.nObj))))
        tStop = time()
        np.savetxt(self.pathOut+"/overlap_flag.txt", overlapFlag)

    def loadOverlapFlag(self):
        self.overlapFlag = np.genfromtxt(self.pathOut+"/overlap_flag.txt")

    ##################################################################################

    def examineCmbMaps(self):

        # mask, before re-thresholding
        x = self.cmbMask.copy()
        path = self.pathFig+"/hist_cmbmask_prerethresh.pdf"
        myHistogram(x, nBins=71, lim=(np.min(x), np.max(x)),
                    path=path, nameLatex=r'CMB mask value', semilogy=True)

        # rethreshold the mask
        mask = (self.cmbMask > 0.5)[0]

        # mask, after re-thresholding
        x = 1. * mask.copy()
        path = self.pathFig+"/hist_cmbmask_postrethresh.pdf"
        myHistogram(x, nBins=71, lim=(np.min(x), np.max(x)),
                    path=path, nameLatex=r'CMB mask value', semilogy=True)

        # masked map histogram
        x = self.cmbMap[mask]
        path = self.pathFig+"/hist_cmbmap.pdf"
        myHistogram(x, nBins=71, lim=(-10.*np.std(x), 10.*np.std(x)), path=path,
                    nameLatex=r'CMB map value', semilogy=True, doGauss=True, S2Theory=[110.**2])

        # masked hit count histogram
        x = self.cmbHit[mask]
        path = self.pathFig+"/hist_cmbhit.pdf"
        myHistogram(x, nBins=71, lim=(np.min(x), np.max(x)),
                    path=path, nameLatex=r'CMB hit count', semilogy=True)

    ##################################################################################

    def extractStamp(self, ra, dec, test=False):
        """Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
        ra, dec in degrees.
        Does it for the map, the mask and the hit count.
        """

        stampMap = self.cutoutGeometry()
        stampMask = stampMap.copy()
        stampHit = stampMap.copy()

        # coordinates of the square map (between -1 and 1 deg)
        # output map position [{dec,ra},ny,nx]
        opos = stampMap.posmap()

        # coordinate of the center of the square map we want to extract
        # ra, dec in this order
        # convert from degrees to radians
        sourcecoord = np.array([ra, dec])*utils.degree

        # corresponding true coordinates on the big healpy map
        ipos = rotfuncs.recenter(
            opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]

        # extract the small square map by interpolating the big map
        # these are now numpy arrays: the wcs info is gone

        # Here, I use bilinear interpolation
        stampMap[:, :] = self.cmbMap.at(
            ipos, prefilter=True, mask_nan=False, order=1)
        stampMask[:, :] = self.cmbMask.at(
            ipos, prefilter=True, mask_nan=False, order=1)
        if self.cmbHit is not None:
            stampHit[:, :] = self.cmbHit.at(
                ipos, prefilter=True, mask_nan=False, order=1)

        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1.*(stampMask[:, :] > 0.5)

        if test:
            print("Extracted cutouts around ra=", ra, "dec=", dec)

            print("Map:")
            print("- min, mean, max =", np.min(stampMap),
                  np.mean(stampMap), np.max(stampMap))
            print("- plot")
            plots = enplot.plot(stampMap, grid=True)
            enplot.write(self.pathTestFig+"/stampmap_ra" +
                         np.str(np.round(ra, 2))+"_dec"+np.str(np.round(dec, 2)), plots)

            print("Mask:")
            print("- min, mean, max =", np.min(stampMask),
                  np.mean(stampMask), np.max(stampMask))
            print("- plot")
            plots = enplot.plot(stampMask, grid=True)
            enplot.write(self.pathTestFig+"/stampmask_ra" +
                         np.str(np.round(ra, 2))+"_dec"+np.str(np.round(dec, 2)), plots)

            print("Hit count:")
            print("- min, mean, max =", np.min(stampHit),
                  np.mean(stampHit), np.max(stampHit))
            print("- plot the hit")
            plots = enplot.plot(stampHit, grid=True)
            enplot.write(self.pathTestFig+"/stamphit_ra" +
                         np.str(np.round(ra, 2))+"_dec"+np.str(np.round(dec, 2)), plots)

        return opos, stampMap, stampMask, stampHit

    ##################################################################################

    def aperturePhotometryFilter(self, opos, stampMap, stampMask, stampHit, r0, r1, filterType='diskring',  test=False):
        """Apply an AP filter (disk minus ring) to a stamp map:
        AP = int d^2theta * Filter * map.
        Unit is [map unit * sr]
        The filter function is dimensionless:
        Filter = 1 in the disk, - (disk area)/(ring area) in the ring, 0 outside.
        Hence:
        int d^2theta * Filter = 0.
        r0 and r1 are the radius of the disk and ring in radians.
        stampMask should have values 0 and 1 only.
        Output:
        filtMap: [map unit * sr]
        filtMask: [mask unit * sr]
        filtHitNoiseStdDev: [1/sqrt(hit unit) * sr], ie [std dev * sr] if [hit map] = inverse var
        diskArea: [sr]
        """
        # coordinates of the square map in radians
        # zero is at the center of the map
        # output map position [{dec,ra},ny,nx]
        dec = opos[0, :, :]
        ra = opos[1, :, :]
        radius = np.sqrt(ra**2 + dec**2)
        # exact angular area of a pixel [sr] (same for all pixels in CEA, not CAR)
        pixArea = ra.area() / len(ra.flatten())

        # detect point sources within the filter:
        # gives 0 in the absence of point sources/edges; gives >=1 in the presence of point sources/edges
        filtMask = np.sum((radius <= r1) * (1-stampMask))   # [dimensionless]

        # disk filter [dimensionless]
        inDisk = 1.*(radius <= r0)
        # exact angular area of disk [sr]
        diskArea = np.sum(inDisk) * pixArea
        # ring filter [dimensionless]
        inRing = 1.*(radius > r0)*(radius <= r1)

        if filterType == 'diskring':
            # normalize the ring so that the disk-ring filter integrates exactly to zero
            inRing *= np.sum(inDisk) / np.sum(inRing)
            # disk minus ring filter [dimensionless]
            filterW = inDisk - inRing
            if np.isnan(np.sum(filterW)):
                print("filterW sums to nan", r0, r1, np.sum(radius), np.sum(
                    1.*(radius > r0)), np.sum(1.*(radius > r0)*(radius <= r1)))
        elif filterType == 'ringring':
            # A ringring filter is a diskring filter that also masks out the central pixels (within radius rr). This is done
            # so that we can mask out the CIB from dust contamination at the central pixel in galaxies.
            # ringring is the non equal area ringring filter. Instead of having the two inner and outer rings be equal area,
            # the outer ring is the same area as the inner disk (as if the inner ring is a disk), while both the inner and outer
            # rings are re-scaled to the same area as the disk.
            rr = self.rApInnerRad / 60. * np.pi/180.
            innerRing = 1.*(radius > rr)*(radius <= r0)
            innerRing *= np.sum(inDisk)/np.sum(innerRing)
            inRing *= np.sum(inDisk) / np.sum(inRing)
            filterW = innerRing - inRing
        elif filterType == 'ringring2':
            # ringring2 is the equal-area ringring filter. The outer ring radius is chosen so that the area of the outer ring
            # is exactly the same as the inner ring. Due to pixelization effects we still re-scale the outer ring to be the 
            # same as the inner ring. (important: We don't change the diskarea parameter)
            rr = self.rApInnerRad / 60. * np.pi/180.
            innerRing = 1.*(radius > rr)*(radius <= r0)
            # innerRing *= np.sum(inDisk)/np.sum(innerRing)
            
            r1 = np.sqrt(2*r0**2 - rr**2)
            inRing = 1.*(radius > r0)*(radius <= r1)
            inRing *= np.sum(innerRing) / np.sum(inRing)
            
            filterW = innerRing - inRing
        elif filterType == 'disk':
            # disk filter [dimensionless]
            inDisk = 1.*(radius <= r0)
            filterW = inDisk
        elif filterType == 'ring':
            filterW = inRing
        elif filterType == 'meanring':
            filterW = inRing / np.sum(pixArea * inRing)

        # apply the filter: int_disk d^2theta map -  disk_area / ring_area * int_ring d^2theta map
        filtMap = np.sum(pixArea * filterW * stampMap)   # [map unit * sr]
        # quantify noise std dev in the filter
        if self.cmbHit is not None:
            # to get the std devs [sr / sqrt(hit unit)]
            filtHitNoiseStdDev = np.sqrt(
                np.sum((pixArea * filterW)**2 / (1.e-16 + stampHit)))
        else:
            filtHitNoiseStdDev = 0.

        # print "filtHitNoiseStdDev = ", filtHitNoiseStdDev
        if np.isnan(filtHitNoiseStdDev):
            print("filtHitNoiseStdDev is nan")

        if test:
            print("AP filter with disk radius =",
                  r0 * (180.*60./np.pi), "arcmin")
            # count nb of pixels where filter is strictly positive
            nbPix = len(np.where(filterW > 0.)[0])
            print("- nb of pixels in the cutout: " +
                  str(filterW.shape[0] * filterW.shape[1]))
            print("= nb of pixels where filter=0: " +
                  str(len(np.where(filterW == 0.)[0])))
            print("+ nb of pixels where filter>0: " +
                  str(len(np.where(filterW > 0.)[0])))
            print("+ nb of pixels where filter<0: " +
                  str(len(np.where(filterW < 0.)[0])))
            print("- disk area: "+str(diskArea)+" sr, ie " +
                  str(diskArea * (180.*60./np.pi)**2)+"arcmin^2")
            print("  (from r0, expect "+str(np.pi*r0**2)+" sr, ie " +
                  str(np.pi*r0**2 * (180.*60./np.pi)**2)+"arcmin^2)")
            print("- disk-ring filter sums over pixels to "+str(np.sum(filterW)))
            print("  (should be 0; compared to "+str(len(filterW.flatten()))+")")
            print("- filter on unit disk: " +
                  str(np.sum(pixArea * filterW * inDisk)))
            print("  (should be disk area in sr: "+str(diskArea)+")")
            print("- filter on map: "+str(filtMap))
            print("- filter on mask: "+str(filtMask))
            print("- filter on inverse hit: "+str(filtHitNoiseStdDev))
            print("- plot the filter")
            filterMap = stampMap.copy()
            filterMap[:, :] = filterW.copy()

            vmax = np.max(np.abs(filterW))
            # , color='hotcold')
            plots = enplot.plot(filterMap, grid=True, min=-vmax, max=vmax)
            enplot.write(self.pathTestFig+"/stampfilter_r0" +
                         floatExpForm(r0)+"_r1"+floatExpForm(r1), plots)

        return filtMap, filtMask, filtHitNoiseStdDev, diskArea

    ##################################################################################

    def analyzeObject(self, iObj, test=False):
        '''Analysis to be done for each object: extract cutout map once,
        then apply all aperture photometry filters requested on it.
        Returns:
        filtMap: [map unit * sr]
        filtMask: [mask unit * sr]
        filtHitNoiseStdDev: [1/sqrt(hit unit) * sr], ie [std dev * sr] if [hit map] = inverse var
        diskArea: [sr]
        '''

        # if iObj % 100000 == 0:
        #     print("- analyze object", iObj)

        # create arrays of filter values for the given object
        filtMap = {}
        filtMask = {}
        filtHitNoiseStdDev = {}
        filtArea = {}
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            # create arrays of filter values for the given object
            filtMap[filterType] = np.zeros(self.nRAp)
            filtMask[filterType] = np.zeros(self.nRAp)
            filtHitNoiseStdDev[filterType] = np.zeros(self.nRAp)
            filtArea[filterType] = np.zeros(self.nRAp)

        # only do the analysis if the object overlaps with the CMB map
        if self.overlapFlag[iObj]:
            # Object coordinates
            ra = self.Catalog.RA[iObj]   # in deg
            dec = self.Catalog.DEC[iObj]  # in deg
            z = self.Catalog.Z[iObj]
            # choose postage stamp size to fit the largest ring
            dArcmin = np.ceil(2. * self.rApMaxArcmin * np.sqrt(2.))
            dDeg = dArcmin / 60.
            # extract postage stamp around it
            opos, stampMap, stampMask, stampHit = self.extractStamp(
                ra, dec, test=test)

            for iFilterType in range(len(self.filterTypes)):
                filterType = self.filterTypes[iFilterType]

                # loop over the radii for the AP filter
                for iRAp in range(self.nRAp):
                    # disk radius in comoving Mpc/h
                    # rApMpch = self.RApMpch[iRAp]
                    # convert to radians at the given redshift
                    # r0 = rApMpch / self.U.bg.comoving_transverse_distance(z) # rad

                    # Disk radius in rad
                    r0 = self.RApArcmin[iRAp] / 60. * np.pi/180.
                    # choose an equal area AP filter
                    r1 = r0 * np.sqrt(2.)

                    # perform the filtering
                    filtMap[filterType][iRAp], filtMask[filterType][iRAp], filtHitNoiseStdDev[filterType][iRAp], filtArea[filterType][iRAp] = self.aperturePhotometryFilter(
                        opos, stampMap, stampMask, stampHit, r0, r1, filterType=filterType, test=test)

        if test:
            print(" plot the measured profile")
            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            #
            for iFilterType in range(len(self.filterTypes)):
                filterType = self.filterTypes[iFilterType]
                ax.plot(self.RApArcmin, filtMap[filterType])
            #
            plt.show()

        return filtMap, filtMask, filtHitNoiseStdDev, filtArea

    def saveFiltering(self, nProc=1):

        print("Evaluate all filters on all objects")
        # loop over all objects in catalog
        tStart = time()
        with sharedmem.MapReduceByThread(np=nProc) as pool:
            def f(iObj): return self.analyzeObject(iObj, test=False)
            result = np.array(pool.map(f, list(range(self.Catalog.nObj))))
        tStop = time()
        print("took", (tStop-tStart)/60., "min")

        # unpack and save to file
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]

            filtMap = np.array([result[iObj, 0][filterType][:]
                               for iObj in range(self.Catalog.nObj)])
            filtMask = np.array([result[iObj, 1][filterType][:]
                                for iObj in range(self.Catalog.nObj)])
            filtHitNoiseStdDev = np.array(
                [result[iObj, 2][filterType][:] for iObj in range(self.Catalog.nObj)])
            filtArea = np.array([result[iObj, 3][filterType][:]
                                for iObj in range(self.Catalog.nObj)])

            np.savetxt(self.pathOut+"/"+filterType+"_filtmap.txt", filtMap)
            np.savetxt(self.pathOut+"/"+filterType+"_filtmask.txt", filtMask)
            np.savetxt(self.pathOut+"/"+filterType +
                       "_filtnoisestddev.txt", filtHitNoiseStdDev)
            np.savetxt(self.pathOut+"/"+filterType+"_filtarea.txt", filtArea)

    def loadFiltering(self):
        self.filtMap = {}
        self.filtMask = {}
        self.filtHitNoiseStdDev = {}
        self.filtArea = {}

        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            self.filtMap[filterType] = np.genfromtxt(
                self.pathOut+"/"+filterType+"_filtmap.txt")
            self.filtMask[filterType] = np.genfromtxt(
                self.pathOut+"/"+filterType+"_filtmask.txt")
            self.filtHitNoiseStdDev[filterType] = np.genfromtxt(
                self.pathOut+"/"+filterType+"_filtnoisestddev.txt")
            self.filtArea[filterType] = np.genfromtxt(
                self.pathOut+"/"+filterType+"_filtarea.txt")

    ##################################################################################

    def catalogMask(self, overlap=True, psMask=True, extraSelection=1., filterType=None, outlierReject=True):
        '''Returns catalog mask: 1 for objects to keep, 0 for objects to discard.
        Use as:
        maskedQuantity = Quantity[mask]
        '''

        # Here mask is 1 for objects we want to keep
        mask = np.ones_like(self.Catalog.RA)
        # print("start with fraction", np.sum(mask)/len(mask), "of objects")
        if overlap:
            mask *= self.overlapFlag.copy()
            # print("keeping fraction", np.sum(mask)/len(mask),
            #       "of objects after further overlap cut")
        # PS mask: look at largest aperture, and remove if any point within the disk or ring is masked
        if psMask:
            # The point source mask may vary from one filterType to another
            if filterType is None:
                filterType = list(self.filtMask.keys())[0]
            mask *= 1.*(np.abs(self.filtMask[filterType][:, -1]) < 1.)
            # print("keeping fraction", np.sum(mask) /
            #       len(mask), "of objects after PS mask")
        mask *= extraSelection
        # print "keeping fraction", np.sum(mask)/len(mask), " of objects"
        if outlierReject:
            mask = mask.astype(bool)
            # we reject objects whose filter values are such
            # that the probability to have >=1 object with such a high absolute value
            # in a sample of this size
            # is equivalent to a 5sigma PTE = 5.73e-7.
            # Sample of 1: proba that the value is found outside of [-n*sigma, +n*sigma]
            # called PTE
            # is p = erf( n / sqrt(2))
            # Sample of N independent objects: the proba that at least one of them is outside [-n*sigma, +n*sigma]
            # is 1 - (1-p)^N
            # ie N * p  if p<<1
            # Our threshold: n*p should correspond to a 5 sigma PTE, ie
            # N * p = erf(5/sqrt(2)) = 5.733e-7
            nObj = np.sum(mask)
            if nObj != 0:
                def f(nSigmas): return nObj * special.erfc(nSigmas /
                                                           np.sqrt(2.)) - special.erfc(5. / np.sqrt(2.))
                nSigmasCut = optimize.brentq(f, 0., 1.e2)
                # ts has shape (nObj, nRAp)
                # sigmas has shape  nRAp
                sigmas = np.std(self.filtMap[filterType][mask, :], axis=0)
                # shape is (nObj, nRAp)
                newMask = (
                    np.abs(self.filtMap[filterType][:, :]) <= nSigmasCut * sigmas[np.newaxis, :])
                # take the intersection of the masks
                mask *= np.prod(newMask, axis=1).astype(bool)
                # print("keeping fraction", np.sum(1.*mask)/len(mask),
                #       "of objects after further outlier cut")
        # make sure the mask is boolean
        mask = mask.astype(bool)
        # print("keeping fraction", np.sum(1.*mask)/len(mask), "in the end")
        return mask

    ##################################################################################

    def measureVarFromHitCount(self, filterType,  plot=False):
        """Returns a list of functions, one for each AP filter radius,
        where the function takes filtHitNoiseStdDev**2 \propto [(map var) * sr^2] as input and returns the
        actual measured filter variance [(map unit)^2 * sr^2].
        The functions are expected to be linear if the detector noise is the main source of noise,
        and if the hit counts indeed reflect the detector noise.
        To be used for noise weighting in the stacking.
        """
        # keep only objects that overlap, and mask point sources
        mask = self.catalogMask(overlap=True, psMask=True, filterType=filterType, outlierReject=False)
        # This array contains the true variances for each object and aperture
        filtVarTrue = np.zeros((self.Catalog.nObj, self.nRAp))

        if self.cmbHit is not None:
            # print("Interpolate variance=f(hit count) for each aperture")
            fVarFromHitCount = np.empty(self.nRAp, dtype='object')
            for iRAp in range(self.nRAp):
                # print("Aperture number "+str(iRAp))
                x = self.filtHitNoiseStdDev[filterType][mask, iRAp]**2
                y = self.filtMap[filterType][mask, iRAp].copy()
                y = (y - np.mean(y))**2

                # define bins of hit count values
                nBins = 10  # 21
                binEdges = np.logspace(
                    np.log10(np.min(x)), np.log10(np.max(x)), nBins, 10.)

#            # define bins of hit count values,
#            # with an equal number of objects in each bin
#            nBins = 10
#            binEdges = splitBins(x, nBins)
#            print "bin edges"
#            print binEdges
#            print np.min(x), np.max(x)

                # compute histograms
                binCenters, binEdges, binIndices = stats.binned_statistic(
                    x, x, statistic='mean', bins=binEdges)
                binCounts, binEdges, binIndices = stats.binned_statistic(
                    x, x, statistic='count', bins=binEdges)
                binnedVar, binEdges, binIndices = stats.binned_statistic(
                    x, y, statistic=np.mean, bins=binEdges)
                sBinnedVar, binEdges, binIndices = stats.binned_statistic(
                    x, y, statistic=np.std, bins=binEdges)
                sBinnedVar /= np.sqrt(binCounts)

                # exclude the empty bins, which did not contain data
                I = np.where(np.isfinite(
                    binCenters * binCounts * binnedVar * sBinnedVar))[0]
                binCenters = binCenters[I]
                binCounts = binCounts[I]
                binnedVar = binnedVar[I]
                sBinnedVar = sBinnedVar[I]

                # interpolate, to use as noise weighting
                fVarFromHitCount[iRAp] = interp1d(
                    binCenters, binnedVar, kind='linear', bounds_error=False, fill_value=(binnedVar[0], binnedVar[-1]))

                # evaluate the variance for each object
                filtVarTrue[mask, iRAp] = fVarFromHitCount[iRAp](x)

#            # evaluate the variance for each object
#            # for the objects masked, still give them a weight just in case,
#            # to avoid null weights
#            xFull = self.filtHitNoiseStdDev[filterType][:, iRAp]**2
#            filtVarTrue[:,iRAp] = fVarFromHitCount[iRAp](xFull)

                if plot:
                    # plot
                    fig = plt.figure(0)
                    ax = fig.add_subplot(111)
                    #
                    # measured
                    ax.errorbar(binCenters, binnedVar*(180.*60./np.pi)**2,
                                yerr=sBinnedVar*(180.*60./np.pi)**2, fmt='.', label=r'measured')
                    # interpolated
                    newX = np.logspace(np.log10(
                        np.min(x)/2.), np.log10(np.max(x)*2.), num=np.int(10.*nBins), base=10.)
                    newY = np.array(list(map(fVarFromHitCount[iRAp], newX)))
                    ax.plot(newX, newY*(180.*60./np.pi)
                            ** 2, label=r'interpolated')
                    #
                    ax.set_xscale('log', nonpositive='clip')
                    ax.set_yscale('log', nonpositive='clip')
                    ax.set_xlabel(
                        r'Det. noise var. from combined hit [arbitrary]')
                    ax.set_ylabel(r'Measured var. [$\mu$K.arcmin$^2$]')
                    #
                    path = self.pathFig+"/binned_noise_vs_hit_" + \
                        filterType+"_"+str(iRAp)+".pdf"
                    fig.savefig(path, bbox_inches='tight')
                    fig.clf()

        else:
            # print("Measure var for each aperture (no hit count)")
            meanVarAperture = np.var(self.filtMap[filterType][mask, :], axis=0)
            for iRAp in range(self.nRAp):
                filtVarTrue[mask, iRAp] = meanVarAperture[iRAp] * \
                    np.ones(np.sum(mask))

        return filtVarTrue

    def measureAllVarFromHitCount(self, plot=False):
        self.filtVarTrue = {}
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            self.filtVarTrue[filterType] = self.measureVarFromHitCount(
                filterType, plot=plot)

    ##################################################################################

    def measureMeanTZBins(self, filterType, plot=False, test=False):
        '''Measure the mean filter temperatures in redshift bins,
        in order to subtract it to make the kSZ estimator robust to dust evolution * mean velocities.
        '''
        # print("Measure mean T in z-bins (to subtract for kSZ)")

        # keep only objects that overlap, and mask point sources
        mask = self.catalogMask(overlap=True, psMask=True, filterType=filterType)

        # redshift bins
        nZBins = 10
        zBinEdges = np.linspace(self.Catalog.Z.min(),
                                self.Catalog.Z.max(), nZBins)

        # array of mean t to fill
        meanT = np.zeros((self.Catalog.nObj, self.nRAp))
        fMeanT = np.empty(self.nRAp, dtype='object')
        for iRAp in range(self.nRAp):
            # quantities to be binned
            z = self.Catalog.Z[mask].copy()
            t = self.filtMap[filterType][mask, iRAp].copy()
    #      s2 = self.filtVarTrue[filterType][mask, iRAp].copy()
    #      # inverse-variance weight the temperatures
    #      t  = t/s2 / np.mean(1./s2)
            # compute histograms
            zBinCenters, zBinEdges, zBinIndices = stats.binned_statistic(
                z, z, statistic='mean', bins=zBinEdges)
            zBinCounts, zBinEdges, zBinIndices = stats.binned_statistic(
                z, z, statistic='count', bins=zBinEdges)
            tMean, zBinEdges, zBinIndices = stats.binned_statistic(
                z, t, statistic=np.mean, bins=zBinEdges)
            sTMean, zBinEdges, zBinIndices = stats.binned_statistic(
                z, t, statistic=np.std, bins=zBinEdges)
            sTMean /= np.sqrt(zBinCounts)
            # piecewise-constant interpolation
            fMeanT[iRAp] = interp1d(zBinEdges[:-1], tMean, kind='previous',
                                    bounds_error=False, fill_value=(tMean[0], tMean[-1]))

            # evaluate the interpolation at each object
            meanT[mask, iRAp] = fMeanT[iRAp](z)

            if plot:
                fig = plt.figure(0)
                ax = fig.add_subplot(111)
                factor = (180.*60./np.pi)**2
                #
                ax.axhline(0., color='k')
                # Measured
                ax.errorbar(zBinCenters, factor*tMean,
                            yerr=factor*sTMean, label=r'Measured')
                # Check interpolation
                z = np.linspace(self.Catalog.Z.min(),
                                self.Catalog.Z.max(), 101)
                ax.plot(z, factor*fMeanT[iRAp](z), '--', label=r'Interpolated')
                #
                ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
                ax.set_xlabel(r'$z$ bins')
                ax.set_ylabel(r'Mean T per $z$-bin [$\mu$K$\cdot$arcmin$^2$]')
                #
                path = self.pathFig+"/binned_mean_t_" + \
                    filterType+"_"+str(iRAp)+".pdf"
                fig.savefig(path, bbox_inches='tight')
                if test:
                    plt.show()
                fig.clf()

        return meanT

    def measureAllMeanTZBins(self, plot=False, test=False):
        self.meanT = {}
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            # print(("For "+filterType+" filter:"))
            self.meanT[filterType] = self.measureMeanTZBins(
                filterType, plot=plot, test=test)

    ##################################################################################

    def computeStackedProfile(self, filterType, est, iBootstrap=None, iVShuffle=None, stackedMap=False, ts=None, mask=None):
        """Returns the estimated profile and its uncertainty for each aperture.
        est: string to select the estimator
        iBootstrap: index for bootstrap resampling
        iVShuffle: index for shuffling velocities
        tTh: to replace measured temperatures by a theory expectation
        ts: option to specify another thumbstack object
        """

        # tStart = time()

        # print(("- Compute stacked profile: "+filterType+", "+est+", "+tTh))

        # compute stacked profile from another thumbstack object
        if ts is None:
            ts = self

        # select objects that overlap, and reject point sources
        if mask is None:
            mask = ts.catalogMask(overlap=True, psMask=True,
                                  filterType=filterType)

#      tMean = ts.meanT[filterType].copy()

        # temperatures [muK * sr]
        t = ts.filtMap[filterType].copy()  # [muK * sr]
        t = t[mask, :]
   #     tMean = tMean[mask,:]
        # -v/c [dimless]

        # true filter variance for each object and aperture,
        # valid whether or not a hit count map is available
        s2Full = ts.filtVarTrue[filterType][mask, :]
        # Variance from hit count (if available)
        s2Hit = ts.filtHitNoiseStdDev[filterType][mask, :]**2
        # print "Shape of s2Hit = ", s2Hit.shape
        # halo masses

        if iBootstrap is not None:
            # make sure each resample is independent,
            # and make the resampling reproducible
            np.random.seed(iBootstrap)
            # list of overlapping objects
            nObj = np.sum(mask)
            # print "sample "iBootstrap, ";", nObj, "objects overlap with", ts.name
            I = np.arange(nObj)
            # choose with replacement from this list
            J = np.random.choice(I, size=nObj, replace=True)
            #
            t = t[J, :]
            # tMean = tMean[J,:]
            s2Hit = s2Hit[J, :]
            s2Full = s2Full[J, :]

        if iVShuffle is not None:
            # make sure each shuffling is independent,
            # and make the shuffling reproducible
            np.random.seed(iVShuffle)
            # list of overlapping objects
            nObj = np.sum(mask)
            I = np.arange(nObj)
            # shuffle the velocities
            J = np.random.permutation(I)

        # tSZ: uniform weighting
        if est == 'tsz_uniformweight':
            weights = np.ones_like(s2Hit)
            norm = 1./np.sum(weights, axis=0)
        # tSZ: detector-noise weighted (hit count)
        elif est == 'tsz_hitweight':
            weights = 1./s2Hit
            norm = 1./np.sum(weights, axis=0)
        # tSZ: full noise weighted (detector noise + CMB)
        elif est == 'tsz_varweight':
            weights = 1./s2Full
            norm = 1./np.sum(weights, axis=0)


        # return the stacked profiles
        if not stackedMap:
            stack = norm * np.sum(t * weights, axis=0)
            sStack = norm * np.sqrt(np.sum(s2Full * weights**2, axis=0))
            return stack, sStack

        # or, if requested, compute and return the stacked cutout map
        else:
            # define chunks
            nChunk = ts.nProc
            chunkSize = int(ts.Catalog.nObj / nChunk)
            # list of indices for each of the nChunk chunks
            chunkIndices = [
                list(range(iChunk*chunkSize, (iChunk+1)*chunkSize)) for iChunk in range(nChunk)]
            # make sure not to miss the last few objects:
            # add them to the last chunk
            chunkIndices[-1] = list(range((nChunk-1) *
                                    chunkSize, ts.Catalog.nObj))

            # select weights for a typical aperture size (not the smallest, not the largest)
            # iRAp0 = ts.nRAp / 2
            iRAp0 = int(ts.nRAp / 4)
            norm = norm[iRAp0]
            # need to link object number with weight,
            # despite the mask
            weightsLong = np.zeros(ts.Catalog.nObj)
            weightsLong[mask] = weights[:, iRAp0]

            def stackChunk(iChunk):
                # object indices to be processed
                chunk = chunkIndices[iChunk]

                # start with a null map for stacking
                resMap = ts.cutoutGeometry()
                for iObj in chunk:
                    # if iObj % 100000 == 0:
                    #     print("- analyze object", iObj)
                    if ts.overlapFlag[iObj]:
                        # Object coordinates
                        ra = ts.Catalog.RA[iObj]   # in deg
                        dec = ts.Catalog.DEC[iObj]  # in deg
                        z = ts.Catalog.Z[iObj]
                        # extract postage stamp around it
                        opos, stampMap, stampMask, stampHit = ts.extractStamp(
                            ra, dec, test=False)
                        resMap += stampMap * weightsLong[iObj]
                return resMap

            # dispatch each chunk of objects to a different processor
            with sharedmem.MapReduceByThread(np=ts.nProc) as pool:
                resMap = np.array(pool.map(stackChunk, list(range(nChunk))))

            # sum all the chunks
            resMap = np.sum(resMap, axis=0)
            # normalize by the proper sum of weights
            resMap *= norm

            return resMap


    def plotFilterHistograms(self, filterType, ts=None):
        '''Plot histograms of the AP filter values,
        for each AP filter size,
        and compare with the Gaussian expectation.
        This enables checking for outliers.
        '''
        if ts is None:
            ts = self

        # select objects that overlap, and reject point sources
        mask = ts.catalogMask(overlap=True, psMask=True,
                              filterType=filterType)

        # aperture photometry filters
        t = ts.filtMap[filterType].copy()  # [muK * sr]
        t = t[mask, :]
        # convert from sr to arcmin^2
        factor = (180.*60./np.pi)**2
        t *= factor

        # true filter variance for each object and aperture,
        # valid whether or not a hit count map is available
        s2Full = ts.filtVarTrue[filterType][mask, :]
        # Variance from hit count (if available)
        s2Hit = ts.filtHitNoiseStdDev[filterType][mask, :]**2

        for iRAp in range(ts.nRAp):
            # Raw temperatures
            path = ts.pathFig + "/histogram_t_" + \
                filterType+"_uniformweight_"+str(iRAp)+".pdf"
            myHistogram(t[:, iRAp], nBins=101, lim=None, S2Theory=[], path=path, plot=False,
                        nameLatex=r'$T_i$ [$\mu$K$\cdot$arcmin$^2$]', semilogx=False, semilogy=True, doGauss=True)

            # inverse variance weighted temperatures
            path = ts.pathFig + "/histogram_t_" + \
                filterType+"_varweight_"+str(iRAp)+".pdf"
            myHistogram(t[:, iRAp] / s2Full[:, iRAp], nBins=101, lim=None, S2Theory=[], path=path,
                        plot=False, nameLatex=r'$T_i/\sigma_{T, i}^2$', semilogx=False, semilogy=True, doGauss=True)

    ##################################################################################

    def SaveCovBootstrapStackedProfile(self, filterType, est, nSamples=100, nProc=1):
        """Estimate covariance matrix for the stacked profile from bootstrap resampling
        """
        # print("Performing", nSamples, "bootstrap resamples")
        tStart = time()
        with sharedmem.MapReduceByThread(np=nProc) as pool:
            def f(iSample): return self.computeStackedProfile(
                filterType, est, iBootstrap=iSample)
            result = np.array(pool.map(f, list(range(nSamples))))
            # result = np.array(map(f, range(nSamples)))
        tStop = time()
        # print("took", (tStop-tStart)/60., "min")
        # unpack results
        stackSamples = result[:, 0, :]  # shape (nSamples, nRAp)
        # sStackSamples = result[:,1,:]
        # estimate cov
        covStack = np.cov(stackSamples, rowvar=False)
        # save it to file
        np.savetxt(self.pathOut+"/cov_"+filterType +
                   "_"+est+"_bootstrap.txt", covStack)

    def SaveCovBootstrapTwoStackedProfiles(self, ts2, filterType, est, nSamples=100, nProc=1):
        """Estimate the full covariance for two stacked profiles.
        These need to use the same galaxy catalog. The temperature maps can be different.
        Resamples only the objects in common, then rescales the cov assuming
        that the objects that are not in common are not statistically different
        from the objects in common.
        """
        # print("Performing", nSamples, "bootstrap resamples")
        # for each resample, compute both profiles, and concatenate, before taking the cov
        tStart = time()

        # find the objects that overlap with both maps
        # bootstrap resample only these objects,
        # then rescale the cov mat
        mask1 = self.catalogMask(
            overlap=True, psMask=True, filterType=filterType)
        mask2 = ts2.catalogMask(overlap=True, psMask=True,
                                filterType=filterType)
        mask12 = mask1 * mask2
        n1 = np.sum(mask1)
        n2 = np.sum(mask2)
        n12 = np.sum(mask12)
        # print("Objects that overlap with 1, 2, 1&2:", n1, n2, n12)
        # build the rescaling matrix for the cov
        block = np.ones((self.nRAp, self.nRAp))
        f11 = 1. * n12 / n1 * block
        f12 = 1. * n12**2 / n1 / n2 * block
        f22 = 1. * n12 / n2 * block
        rescaleMat = np.block([[f11, f12], [f12, f22]])

        def f(iSample):
            # print "Joint Bootstrap", iSample
            prof1 = self.computeStackedProfile(
                filterType, est, iBootstrap=iSample, mask=mask12)
            prof2 = self.computeStackedProfile(
                filterType, est, iBootstrap=iSample, ts=ts2, mask=mask12)
            # concatenate the 2 profiles
            jointProf = np.concatenate((prof1[0], prof2[0]))
            return jointProf

        with sharedmem.MapReduceByThread(np=nProc) as pool:
            stackSamples = np.array(pool.map(f, list(range(nSamples))))
            # result = np.array(map(f, range(nSamples)))
        tStop = time()
        # print("took", (tStop-tStart)/60., "min")
        # estimate cov
        covStack = np.cov(stackSamples, rowvar=False)
        # rescale the cov mat, since only the galaxies in common for the two maps
        # were sampled
        covStack *= rescaleMat
        # save it to file
        path = self.pathOut+"/cov_"+filterType+"_"+est + \
            "_joint_"+self.name+"_"+ts2.name+"_bootstrap.txt"
        print("saving to:")
        print(path)
        np.savetxt(path, covStack)

    def saveAllCovBootstrapTwoStackedProfiles(self, ts2):
        # print("- compute full joint cov between stacked profiles from:")
        # print(self.name)
        # print(ts2.name)
        # Compute all filter types
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            # covariance matrices from bootstrap,
            # only for a few select estimators
            for iEst in range(len(self.EstBootstrap)):
                est = self.EstBootstrap[iEst]
                self.SaveCovBootstrapTwoStackedProfiles(
                    ts2, filterType, est, nSamples=self.nSamples, nProc=min(8, self.nProc))

    def SaveCovVShuffleStackedProfile(self, filterType, est, nSamples=100, nProc=1):
        """Estimate covariance matrix for the stacked profile from shuffling velocities
        """
        tStart = time()
        with sharedmem.MapReduceByThread(np=nProc) as pool:
            def f(iSample): return self.computeStackedProfile(
                filterType, est, iVShuffle=iSample)
            result = np.array(pool.map(f, list(range(nSamples))))
        tStop = time()
        # print("took", (tStop-tStart)/60., "min")
        # unpack results
        stackSamples = result[:, 0, :]  # shape (nSamples, nRAp)
        # sStackSamples = result[:,1,:]
        # estimate cov
        covStack = np.cov(stackSamples, rowvar=False)
        # save it to file
        np.savetxt(self.pathOut+"/cov_"+filterType +
                   "_"+est+"_vshuffle.txt", covStack)

        # Null test: measure mean
        data = np.zeros((self.nRAp, 3))
        data[:, 0] = self.RApArcmin
        # mean of the shuffled stacks
        data[:, 1] = np.mean(stackSamples, axis=0)
        data[:, 2] = np.sqrt(np.diag(covStack)) / \
            np.sqrt(nSamples)  # variance of the mean
        np.savetxt(self.pathOut+"/"+filterType +
                   "_"+est+"_vshufflemean.txt", data)

    ##################################################################################

    def saveAllStackedMaps(self, filterTypes=None, Est=None):
        # print("- compute all stacked maps")
        if filterTypes is None:
            filterTypes = self.filterTypes
        if Est is None:
            Est = self.Est

        # Get cutout geometry, for plotting
        cutoutMap = self.cutoutGeometry()
        size = cutoutMap.posmap()[0, :, :].max() - \
            cutoutMap.posmap()[0, :, :].min()
        baseMap = FlatMap(
            nX=cutoutMap.shape[0], nY=cutoutMap.shape[1], sizeX=size, sizeY=size)

        # loop over filter types: only matter
        # because they determine the weights in the stacked map
        for iFilterType in range(len(filterTypes)):
            filterType = filterTypes[iFilterType]
            # Estimators (tSZ, kSZ, various weightings...)
            for iEst in range(len(Est)):
                est = Est[iEst]
                # print("compute stacked map:", filterType, est)
                stackedMap = self.computeStackedProfile(
                    filterType, est, iBootstrap=None, iVShuffle=None, stackedMap=True)

                # save the stacked cutout
                path = self.pathOut + "/stackedmap_"+filterType+"_"+est+".txt"
                np.savetxt(path, stackedMap)

                # for low SNR signals, downgrade the map
                # lowBaseMap = FlatMap(nX=cutoutMap.shape[0]//2, nY=cutoutMap.shape[1]//2, sizeX=size, sizeY=size)
                # nXNew = stackedMap.shape[0] // 2
                # nYNew = stackedMap.shape[1] // 2
                # stackedMap = baseMap.downResolution(nXNew, nYNew, data=stackedMap)

                # plot the stacked map and save it
                path = self.pathFig + "/stackedmap_"+filterType+"_"+est+".png"
                baseMap.plot(data=stackedMap, save=True, path=path)
                # lowBaseMap.plot(data=stackedMap, save=True, path=path)

    ##################################################################################

    def saveAllStackedProfiles(self):
        print("- compute all stacked profiles and their cov")
        tStart = time()
        data = np.zeros((self.nRAp, 3))
        data[:, 0] = self.RApArcmin  # [arcmin]

        # Compute all filter types and estimators
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]

            # check AP filter histograms
#         self.plotFilterHistograms(filterType)

            # Estimators (tSZ, kSZ, various weightings...)
            for iEst in range(len(self.Est)):
                est = self.Est[iEst]
                # measured stacked profile
                data[:, 1], data[:, 2] = self.computeStackedProfile(
                    filterType, est)  # [map unit * sr]
                np.savetxt(self.pathOut+"/"+filterType +
                           "_"+est+"_measured.txt", data)

            # covariance matrices from bootstrap,
            # only for a few select estimators
            if self.doBootstrap:
                for iEst in range(len(self.EstBootstrap)):
                    est = self.EstBootstrap[iEst]
                    self.SaveCovBootstrapStackedProfile(
                        filterType, est, nSamples=self.nSamples, nProc=self.nProc)

        tStop = time()
        print("Computing all stacked profiles (and cov) took",
              (tStop-tStart)/60., "min")

    def loadAllStackedProfiles(self):
        print("- load stacked profiles and their cov")
        self.stackedProfile = {}
        self.sStackedProfile = {}
        self.covBootstrap = {}
        self.covVShuffle = {}

        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]

            # all stacked profiles
            for iEst in range(len(self.Est)):
                est = self.Est[iEst]
                # measured stacked profile
                data = np.genfromtxt(
                    self.pathOut+"/"+filterType+"_"+est+"_measured.txt")
                self.stackedProfile[filterType+"_"+est] = data[:, 1]
                self.sStackedProfile[filterType+"_"+est] = data[:, 2]

            # covariance matrices from bootstrap,
            # only for a few select estimators
            if self.doBootstrap:
                for iEst in range(len(self.EstBootstrap)):
                    est = self.EstBootstrap[iEst]
                    self.covBootstrap[filterType+"_"+est] = np.genfromtxt(
                        self.pathOut+"/cov_"+filterType+"_"+est+"_bootstrap.txt")


    ##################################################################################

    def plotStackedProfile(self, filterType, Est, name=None, pathDir=None, theory=True, tsArr=None, plot=False, legend=True):
        """Compares stacked profiles, and their uncertainties.
        If pathDir is not specified, save to local figure folder.
        """
        if name is None:
            name = Est[0]
        if tsArr is None:
            tsArr = [self]
        if pathDir is None:
            pathDir = self.pathFig

        # stacked profile
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #
        # convert from sr to arcmin^2
        factor = (180.*60./np.pi)**2
        #
        ax.axhline(0., c='k', lw=1)
        #
        # colors = ['r', 'g', 'b', 'm', 'c']
        lineStyles = ['-', '--', '-.', ':']
        for iEst in range(len(Est)):
            est = Est[iEst]
            # c = colors[iEst%len(colors)]

            for iTs in range(len(tsArr)):
                ts = tsArr[iTs]
                ls = lineStyles[iTs % len(tsArr)]

                ax.errorbar(ts.RApArcmin+iTs*0.05, factor * ts.stackedProfile[filterType+"_"+est], factor * ts.sStackedProfile[filterType +
                            "_"+est], fmt=ls, label=filterType.replace('_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
                if theory:
                    ax.plot(ts.RApArcmin+iTs*0.05, factor * ts.stackedProfile[filterType+"_"+est+"_theory_tsz"], ls='--', label="theory tsz, "+filterType.replace(
                        '_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
                    ax.plot(ts.RApArcmin+iTs*0.05, factor * ts.stackedProfile[filterType+"_"+est+"_theory_ksz"], ls='-.', label="theory ksz, "+filterType.replace(
                        '_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
        #
        if legend:
            ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
        ax.set_xlabel(r'$R$ [arcmin]')
        ax.set_ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]')
        # ax.set_ylim((0., 2.))
        #
        path = pathDir+"/"+name+".pdf"
        fig.savefig(path, bbox_inches='tight')
        if plot:
            plt.show()
        fig.clf()

        # uncertainty on stacked profile
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #
        # convert from sr to arcmin^2
        factor = (180.*60./np.pi)**2
        #
        colors = ['r', 'g', 'b', 'm', 'c']
        lineStyles = ['-', '--', '-.', ':']
        for iEst in range(len(Est)):
            est = Est[iEst]
            c = colors[iEst % len(colors)]

            for iTs in range(len(tsArr)):
                ts = tsArr[iTs]
                ls = lineStyles[iTs % len(tsArr)]

                ax.plot(ts.RApArcmin+iTs*0.05, factor * ts.sStackedProfile[filterType+"_"+est], c=c, ls=ls, lw=2, label='analytic, '+filterType.replace(
                    '_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
                if est in ts.covBootstrap:
                    ax.plot(ts.RApArcmin+iTs*0.05, factor * np.sqrt(np.diag(ts.covBootstrap[filterType+"_"+est])), c=c, ls=ls, lw=1.5, label="bootstrap, "+filterType.replace(
                        '_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
                if est in ts.covVShuffle:
                    ax.plot(ts.RApArcmin+iTs*0.05, factor * np.sqrt(np.diag(ts.covVShuffle[filterType+"_"+est])), c=c, ls=ls, lw=1, label="v shuffle, "+filterType.replace(
                        '_', ' ')+' '+est.replace('_', ' ')+' '+ts.name.replace('_', ' '))
        #
        if legend:
            ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
        ax.set_xlabel(r'$R$ [arcmin]')
        ax.set_ylabel(r'$\sigma(T)$ [$\mu K\cdot\mathrm{arcmin}^2$]')
        # ax.set_ylim((0., 2.))
        #
        path = pathDir+"/s_"+name+".pdf"
        fig.savefig(path, bbox_inches='tight')
        if plot:
            plt.show()
        fig.clf()

    def plotAllStackedProfiles(self):
        print("- plot all stacked profiles")

        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]
            # all stacked profiles
            for iEst in range(len(self.Est)):
                est = self.Est[iEst]
                self.plotStackedProfile(
                    filterType, [est], name=filterType+"_"+est)

            # stacked profiles in mass bins
            if self.doMBins:
                self.plotTszKszContaminationMMax()

                for iEst in range(len(self.EstMBins)):
                    est = self.EstMBins[iEst]
                    # measured stacked profiles
                    estArr = [est+"_mmax"+str(iMMax)
                              for iMMax in range(self.nMMax)] + [est]
                    self.plotStackedProfile(
                        filterType, estArr, name=filterType+"_"+est+"_mmax", theory=False, legend=False)

    ##################################################################################

    def plotCov(self, cov, name="", show=False):
        # Covariance matrix
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #
        # pcolor wants x and y to be edges of cell,
        # ie one more element, and offset by half a cell
        dR = (self.rApMaxArcmin - self.rApMinArcmin) / self.nRAp
        RApEdgesArcmin = np.linspace(
            self.rApMinArcmin-0.5*dR, self.rApMaxArcmin+0.5*dR, self.nRAp+1)
        # Covariance  matrix
        X, Y = np.meshgrid(RApEdgesArcmin, RApEdgesArcmin, indexing='ij')
        cp = ax.pcolormesh(X, Y, cov, cmap='YlOrRd')
        #
        ax.set_aspect('equal')
        plt.colorbar(cp)
        ax.set_xlim((np.min(RApEdgesArcmin), np.max(RApEdgesArcmin)))
        ax.set_ylim((np.min(RApEdgesArcmin), np.max(RApEdgesArcmin)))
        ax.set_xlabel(r'R [arcmin]')
        ax.set_ylabel(r'R [arcmin]')
        #
        path = self.pathFig+"/cov_"+name+".pdf"
        fig.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            fig.clf()

        # Correlation coefficient
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        #
        # pcolor wants x and y to be edges of cell,
        # ie one more element, and offset by half a cell
        dR = (self.rApMaxArcmin - self.rApMinArcmin) / self.nRAp
        RApEdgesArcmin = np.linspace(
            self.rApMinArcmin-0.5*dR, self.rApMaxArcmin+0.5*dR, self.nRAp+1)
        X, Y = np.meshgrid(RApEdgesArcmin, RApEdgesArcmin, indexing='ij')
        #
        sigma = np.sqrt(np.diag(cov))
        cor = np.array([[cov[i1, i2] / (sigma[i1]*sigma[i2])
                       for i2 in range(self.nRAp)] for i1 in range(self.nRAp)])
        cp = ax.pcolormesh(X, Y, cor, cmap='YlOrRd', vmin=0., vmax=1.)
        #
        ax.set_aspect('equal')
        plt.colorbar(cp)
        ax.set_xlim((np.min(RApEdgesArcmin), np.max(RApEdgesArcmin)))
        ax.set_ylim((np.min(RApEdgesArcmin), np.max(RApEdgesArcmin)))
        ax.set_xlabel(r'R [arcmin]')
        ax.set_ylabel(r'R [arcmin]')
        #
        path = self.pathFig+"/cor_"+name+".pdf"
        fig.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            fig.clf()

    def plotAllCov(self):
        print("- plot all covariances")
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]

            # covariance matrices from bootstrap,
            # only for a few select estimators
            if self.doBootstrap:
                for iEst in range(len(self.EstBootstrap)):
                    est = self.EstBootstrap[iEst]
                    self.plotCov(
                        self.covBootstrap[filterType+"_"+est], filterType+"_"+est+"_bootstrap")


    ##################################################################################

    def plotCovTwoStackedProfiles(self, cov, name="", show=False):
        '''For the figure captions, it is assumed that the first stack
        is at 150GHz, the second at 90GHz.
        '''

        RApArcmin = np.concatenate((self.RApArcmin, self.RApArcmin))
        X, Y = np.meshgrid(RApArcmin, RApArcmin, indexing='ij')
        #
        # corresponding indices (for masks)
        I = np.arange(len(RApArcmin))
        II, JJ = np.meshgrid(I, I, indexing='ij')

        # Covariance matrix
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #
        # Covariance  matrix
        covLower = np.ma.masked_where(II < JJ, cov)
        cp = ax.imshow(covLower, cmap='YlOrRd')  # vmin=0., vmax=1.
        plt.colorbar(cp)
        #
        ax.axhline(0.5 * I[-1], c='k')
        ax.axvline(0.5 * I[-1], c='k')
        #
        ax.text(0.25, 0.9, r'150 GHz', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.text(0.75, 0.75, r'150 GHz - 90 GHz', ha='center',
                va='center', transform=ax.transAxes, fontsize=14)
        ax.text(0.75, 0.4, r'90 GHz', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        #
        ax.set_xticks(np.arange(len(RApArcmin)))
        ax.set_yticks(np.arange(len(RApArcmin)))
        ax.set_xticklabels(RApArcmin, fontdict={'fontsize': 10}, rotation=45)
        ax.set_yticklabels(RApArcmin, fontdict={'fontsize': 10})
        ax.set_aspect('equal')
        ax.set_xlabel(r'R [arcmin]')
        ax.set_ylabel(r'R [arcmin]')
        ax.set_title(r'Joint covariance matrix')
        #
        path = self.pathFig+"/cov_joint_"+name+".pdf"
        print("Saving cov plot to:")
        print(path)
        fig.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            fig.clf()

        # Correlation coefficients
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #
        # Correlation coefficients
        sigma = np.sqrt(np.diag(cov))
        cor = np.array([[cov[i1, i2] / (sigma[i1]*sigma[i2])
                       for i2 in I] for i1 in I])
        corLower = np.ma.masked_where(II < JJ, cor)
        cp = ax.imshow(corLower, cmap='YlGnBu', vmin=0., vmax=1.)
        cb = plt.colorbar(cp)
        #
        ax.axhline(0.5 * I[-1], c='k')
        ax.axvline(0.5 * I[-1], c='k')
        #
        ax.text(0.25, 0.9, r'150 GHz', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.text(0.75, 0.75, r'150 GHz - 90 GHz', ha='center',
                va='center', transform=ax.transAxes, fontsize=14)
        ax.text(0.75, 0.4, r'90 GHz', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        #
        ax.set_xticks(np.arange(len(RApArcmin)))
        ax.set_yticks(np.arange(len(RApArcmin)))
        ax.set_xticklabels(RApArcmin, fontdict={'fontsize': 10}, rotation=45)
        ax.set_yticklabels(RApArcmin, fontdict={'fontsize': 10})
        ax.set_aspect('equal')
        ax.set_xlabel(r'R [arcmin]')
        ax.set_ylabel(r'R [arcmin]')
        ax.set_title(r'Joint correlation matrix')
        #
        path = self.pathFig+"/cor_joint_"+name+".pdf"
        print("Saving cor plot to:")
        print(path)
        fig.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            fig.clf()

    def plotAllCovTwoStackedProfiles(self, ts2):
        print("- plot all full joint cov between stacked profiles from:")
        print(self.name)
        print(ts2.name)

        # Compute all filter types
        for iFilterType in range(len(self.filterTypes)):
            filterType = self.filterTypes[iFilterType]

            # covariance matrices from bootstrap,
            # only for a few select estimators
            for iEst in range(len(self.EstBootstrap)):
                est = self.EstBootstrap[iEst]

                # read cov from file
                pathCov = self.pathOut+"/cov_"+filterType+"_"+est + \
                    "_joint_"+self.name+"_"+ts2.name+"_bootstrap.txt"
                print("Reading joint cov from:")
                print(pathCov)
                cov = np.genfromtxt(pathCov)
                name = filterType+"_"+est+"_"+self.name+"_"+ts2.name+"_bootstrap"
                self.plotCovTwoStackedProfiles(cov, name=name, show=False)

    ##################################################################################

    def ftheoryGaussianProfile(self, sigma_cluster, filterType='diskring'):
        """Alpha_ksz signal, between 0 and 1.
        Assumes that the projected cluster profile is a 2d Gaussian,
        with sigma_cluster in arcmin
        Assumes equal area disk-ring filter
        """
        if filterType == 'diskring':
            result = (1. - np.exp(-0.5*self.RApArcmin**2/sigma_cluster**2))**2
        elif filterType == 'ringring':
            result = (1. - np.exp(-0.5*self.RApArcmin**2/sigma_cluster**2))**2
        elif filterType == 'ringring2':
            result = (1. - np.exp(-0.5*self.RApArcmin**2/sigma_cluster**2))**2
        elif filterType == 'disk':
            result = 1. - np.exp(-0.5*self.RApArcmin**2/sigma_cluster**2)
        elif filterType == 'ring':
            result = np.exp(-0.5*self.RApArcmin**2/sigma_cluster**2) - \
                np.exp(-0.5*(self.RApArcmin*np.sqrt(2.))**2/sigma_cluster**2)
        return result

    def ftheoryGaussianProfilePixelated(self, sigma_cluster=1.5, filterType='diskring', dxDeg=0.3, dyDeg=0.3, resArcmin=0.25, proj='cea', pixwin=0, test=False):
        """Alpha_ksz signal, between 0 and 1.
        Assumes that the projected cluster profile is a 2d Gaussian,
        with sigma_cluster in arcmin
        Assumes equal area disk-ring filter.
        This version is not analytical, but instead generates a mock cutout map
        with a Gaussian profile, and runs the AP filters on it.
        This takes into account the discreteness of the edges of the AP filters
        due to the pixelation of the map
        """

        ###########################################
        # generate pixellized cluster profile map

        # generate null cutout map
        stampMap = self.cutoutGeometry()
        opos = stampMap.posmap()

        # fill the central pixel
        ra = 0.
        dec = 0.
        # coordinates in [rad]
        sourcecoord = np.array([dec, ra]) * np.pi/180.
        # find pixel indices (float) corresponding to ra, dec
        iY, iX = enmap.sky2pix(shape, wcs, sourcecoord,
                               safe=True, corner=False)
        # nearest pixel
        jY = np.int(round(iY))
        jX = np.int(round(iX))
        # fill in the central pixel
        # and normalize to integrate to 1 over angles in [muK*sr]
        pixSizeMap = stampMap.pixsizemap()
        # divide by pixel area in sr
        stampMap[jY, jX] = 1. / pixSizeMap[jY, jX]

        # convolve map with a Gaussian  profile of given sigma (not fwhm)
        # convert from arcmin to [rad]
        stampMap = enmap.smooth_gauss(stampMap, sigma_cluster * np.pi/180./60.)

        # apply the pixel window function if desired
        if pixwin != 0:
            stampMap = enmap.apply_window(stampMap, pow=pixwin)

        ###########################################
        # perform the AP filtering

        # create arrays of filter values for the given object
        filtMap = np.zeros(self.nRAp)

        # loop over the radii for the AP filter
        for iRAp in range(self.nRAp):
            # Disk radius in rad
            r0 = self.RApArcmin[iRAp] / 60. * np.pi/180.
            # choose an equal area AP filter
            r1 = r0 * np.sqrt(2.)

            # perform the filtering
            filtMap[iRAp], _, _, _ = self.aperturePhotometryFilter(
                opos, stampMap, stampMap, stampMap, r0, r1, filterType=filterType, test=False)

        if test:
            # compare to the non-pixelated theory profile
            nonPixelated = self.ftheoryGaussianProfile(
                sigma_cluster, filterType=filterType)

            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            #
            ax.plot(self.RApArcmin, nonPixelated, 'k-', label=r'Analytical')
            ax.plot(self.RApArcmin, filtMap, 'b--', label=r'Pixelated')
            #
            ax.legend(loc=4, fontsize='x-small', labelspacing=0.1)

            plt.show()

        return filtMap

    ##################################################################################

    def computeSnrStack(self, filterType, est, tTh='', theory=None, name=None):
        """Compute null rejection, SNR (=detection significance)
        for the requested estimator.
        The estimator considered should have a bootstrap covariance.
        """

        print("- compute SNR and significances for "+filterType+" "+est+" "+tTh)

        # replace data with theory if requested
        if tTh == 'tsz':
            tTh = '_theory_tsz'
        elif tTh == 'ksz':
            tTh = '_theory_ksz'
        else:
            tTh = ''

        if name is None:
            name = ''
        else:
            name = '_'+name

        if theory is None:
            sigma_cluster = 3.
            theory = self.ftheoryGaussianProfile(
                sigma_cluster, filterType=filterType)

        path = self.pathFig+"/snr_"+filterType+"_"+est+tTh+name+".txt"
        with open(path, 'w') as f:
            f.write("*** "+est+" SNR ***\n")

            # data and covariance
            d = self.stackedProfile[filterType+"_"+est+tTh].copy()
            cov = self.covBootstrap[filterType+"_"+est].copy()
            dof = len(d)

            # Compute chi^2_null
            chi2Null = d.dot(np.linalg.inv(cov).dot(d))
            # goodness of fit for null hypothesis
            f.write("number of dof:"+str(dof)+"\n")
            f.write("null chi2Null="+str(chi2Null)+"\n")
            pteNull = 1. - stats.chi2.cdf(chi2Null, dof)
            f.write("null pte="+str(pteNull)+"\n")
            # pte as a function of sigma, for a Gaussian random variable

            def fsigmaToPTE(sigma): return special.erfc(
                sigma/np.sqrt(2.)) - pteNull
            sigmaNull = optimize.brentq(fsigmaToPTE, 0., 1.e3)
            f.write("null pte significance="+str(sigmaNull)+"sigmas\n\n")

            # Gaussian model: find best fit amplitude
            sigma_cluster = 1.5  # arcmin

            def fdchi2(p):
                a = p[0]
                result = (d-a*theory).dot(np.linalg.inv(cov).dot(d-a*theory))
                result -= chi2Null
                return result
            # Minimize the chi squared
            p0 = 1.
            res = optimize.minimize(fdchi2, p0)
            abest = res.x[0]
            # sbest= res.x[1]
            f.write("best-fit amplitude="+str(abest)+"\n")
            f.write("number of dof:"+str(dof - 1)+"\n\n")

            # goodness of fit for best fit
            chi2Best = fdchi2([abest])+chi2Null
            f.write("best-fit chi2="+str(chi2Best)+"\n")
            pteBest = 1. - stats.chi2.cdf(chi2Best, dof-1.)
            f.write("best-fit pte="+str(pteBest)+"\n")
            # pte as a function of sigma, for a Gaussian random variable
            def fsigmaToPTE(sigma): return special.erfc(
                sigma/np.sqrt(2.)) - pteBest
            sigma = optimize.brentq(fsigmaToPTE, 0., 1.e3)
            f.write("best-fit pte significance="+str(sigma)+"sigmas\n\n")

            # favour of best fit over null
            f.write("best-fit sqrt(delta chi2)=" +
                    str(np.sqrt(abs(fdchi2([abest]))))+"sigmas\n")

            def fsigmaToPTE(sigma): return special.erfc(sigma/np.sqrt(2.))
            pte = fsigmaToPTE(np.sqrt(abs(fdchi2([abest]))))
            f.write("pte (if Gaussian)="+str(pte)+"\n")

    def computeAllSnr(self):
        print("- compute all SNR and significances")
        if self.doBootstrap:
            for filterType in self.filterTypes:
                for est in self.EstBootstrap:
                    self.computeSnrStack(filterType, est)
                    self.computeSnrStack(filterType, est, tTh='tsz')
                    self.computeSnrStack(filterType, est, tTh='ksz')

    ##################################################################################
