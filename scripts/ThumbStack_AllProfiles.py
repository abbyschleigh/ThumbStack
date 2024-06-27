import sys
sys.path.append('../src/')


from importlib import reload
import universe
reload(universe)
from universe import *

import mass_conversion
reload(mass_conversion)
from mass_conversion import *

import catalog
reload(catalog)
from catalog import *

import thumbstack
reload(thumbstack)
from thumbstack import *

import cmb
reload(cmb)
from cmb import *
# from headers import *
from cmbMap import *
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

##################################################################################
# Parameters

filterType = 'diskring'
T_CIB = '10.7'
# T_CIB = '24.0'

pathFig = "/pscratch/sd/r/rhliu/projects/ThumbStack/figures/summary_plots/"

plot_Path = "./figures/final3/ThumbStack_AllPlots_dbeta_107.pdf"

pathMask = '/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/wide_mask_GAL070_apod_1.50_deg_wExtended.fits'
pathMask2 = '/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/wide_mask_GAL070_apod_1.50_deg_wExtended_srcfree_Will.fits'

##################################################################################

nProc = 64  # 1 haswell node on cori
##################################################################################
##################################################################################

# cosmological parameters
u = UnivMariana()

# M*-Mh relation
massConversion = MassConversionKravtsov14()
# massConversion.plot()

##################################################################################
# Galaxy Catalogs (from DESI)

print("Read galaxy catalogs")
tStart = time()

catalogs = {
    "DESI_pz1": Catalog(u, massConversion, name="DESI_pz1", nameLong="DESI pz bin 1", save=False),
    "DESI_pz2": Catalog(u, massConversion, name="DESI_pz2", nameLong="DESI pz bin 2", save=False),
    "DESI_pz3": Catalog(u, massConversion, name="DESI_pz3", nameLong="DESI pz bin 3", save=False),
    "DESI_pz4": Catalog(u, massConversion, name="DESI_pz4", nameLong="DESI pz bin 4", save=False)
}

tStop = time()
print("took "+str(round((tStop-tStart)/60., 2))+" min")

###################################################################################
# Read CMB maps

CMB_params = [("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + "ilc_SZ_yy.fits", 
               pathMask2, False, 'act_dr6_fiducial', 'ACT DR6 (fiducial)', 93.e9),
              # ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
              #  "ilc_SZ_deproj_cib_yy.fits", 
              #  pathMask2, False, 'act_dr6_nocib', 'ACT DR6 (Deprojected CIB)', 93.e9),
              # ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
              #  "ilc_SZ_deproj_cib_1.2_"+T_CIB+"_yy.fits", 
              #  pathMask2, False, 'act_dr6_nocib_Beta_1.2_'+T_CIB, 
              #  r'ACT DR6 (Deprojected CIB, $\beta = 1.2$)', 93.e9),
              # ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
              #  "ilc_SZ_deproj_cib_1.4_"+T_CIB+"_yy.fits", 
              #  pathMask2, False, 'act_dr6_nocib_Beta_1.4_'+T_CIB, 
              #  r'ACT DR6 (Deprojected CIB, $\beta = 1.4$)', 93.e9),
              # ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
              #  "ilc_SZ_deproj_cib_1.6_"+T_CIB+"_yy.fits", 
              #  pathMask2, False, 'act_dr6_nocib_Beta_1.6_'+T_CIB, 
              #  r'ACT DR6 (Deprojected CIB, $\beta = 1.6$)', 93.e9),
              ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
               "ilc_SZ_deproj_cib_cibdBeta_1.2_"+T_CIB+"_yy.fits", 
               pathMask2, False, 'act_dr6_dBeta_1.2_'+T_CIB, r'ACT DR6 (d$\beta$ 1.2)', 93.e9),
              ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
               "ilc_SZ_deproj_cib_cibdBeta_1.4_"+T_CIB+"_yy.fits",  
               pathMask2, False, 'act_dr6_dBeta_1.4_'+T_CIB, r'ACT DR6 (d$\beta$ 1.4)', 93.e9),
              ("/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
               "ilc_SZ_deproj_cib_cibdBeta_1.6_"+T_CIB+"_yy.fits",  
               pathMask2, False, 'act_dr6_dBeta_1.6_'+T_CIB, r'ACT DR6 (d$\beta$ 1.6)', 93.e9)
             ]

# Path List
# CMB_pathlist = ["/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + "ilc_SZ_yy.fits",
#                 # "/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" +
#                 # "act_planck_s08_s22_daynight_f090_map.fits",
#                 # "/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" +
#                 # "act_planck_s08_s22_daynight_f150_map.fits",
#                 "/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
#                 "ilc_SZ_deproj_cib_yy.fits",
#                 # "/pscratch/sd/r/rhliu/projects/ThumbStack/ACT_DR6/" + 
#                 # "ilc_SZ_deproj_cib_cibdT_1.7_10.7_yy.fits",
#                ]
# CMB_masklist = [pathMask2] * len(CMB_pathlist)
# CMB_convert = [False, True, True]
# CMB_name = ['act_dr6_fiducial', 'act_dr6_f90',
#             'act_dr6_f150',]
#             # 'act_dr6_nocib',  'act_dr6_dTa_10.7']
# CMB_namepublic = ['ACT DR6 (fiducial)', 'ACT DR6 (f90)',
#                   'ACT DR6 (f150)', ]
#                   # 'ACT DR6 (Deprojected CIB)', 'ACT DR6 dT']
# CMB_nu = [93.e9, 93.e9, 150.e9]

CMB_pathlist = []
CMB_masklist = []
CMB_convert = []
CMB_name = []
CMB_namepublic = []
CMB_nu = []
for element in CMB_params:
    CMB_pathlist.append(element[0])
    CMB_masklist.append(element[1])
    CMB_convert.append(element[2])
    CMB_name.append(element[3])
    CMB_namepublic.append(element[4])
    CMB_nu.append(element[5])
    


filterTypes = [filterType] * len(CMB_params)

cmbMap_list = []

for i, path in enumerate(CMB_pathlist):
    cmap = cmbMap(path,
                  pathMask=CMB_masklist[i],
                  # pathMask=pathMask,
                  # pathHit="/pscratch/sd/r/rhliu/projects/ThumbStack/" + \
                  # "act_dr5.01_s08s18_AA_f150_daynight_ivar.fits",
                  pathHit=None,
                  nu=CMB_nu[i], unitLatex=r'y', convert_y=CMB_convert[i],
                  name=CMB_name[i])
    cmbMap_list.append(cmap)

catalogKeys = catalogs.keys()

###################################################################################
# Testing
save = True
# save = False
ts_list = [[] for _ in range(len(CMB_masklist))]


for key in list(catalogKeys):
    catalog = catalogs[key]
    
    for i, cmap in enumerate(cmbMap_list):
    
        ts = ThumbStack(u, catalog, 
                        cmap.map(), 
                        cmap.mask(), 
                        cmap.hit(), 
                        catalog.name + '_' + cmap.name,
                        nameLong=None, 
                        save=save, 
                        nProc=nProc,
                        filterTypes=filterTypes[i],
                        doMBins=False, 
                        doBootstrap=True,
                        # doStackedMap=True,
                        doVShuffle=False, 
                        cmbNu=cmap.nu, 
                        cmbUnitLatex=cmap.unitLatex,
                        rApMinArcmin=1.)
        ts_list[i].append(ts)

###################################################################################

# Next for plotting:


# Parameters
factor = (180.*60./np.pi)**2
est = 'tsz_uniformweight'

# Plotting 


###############################
fig, subplots = plt.subplots(2, 2, figsize=(8,8), sharex='col', sharey='row')
subplots = subplots.ravel()

for i, key in enumerate(list(catalogKeys)):
    ax = subplots[i]
    for j in range(len(ts_list)):
        tsj = ts_list[j][i]
        filterType = filterTypes[j]

        ax.errorbar(tsj.RApArcmin, factor * tsj.stackedProfile[filterType+"_"+est], 
                    factor * tsj.sStackedProfile[filterType+"_"+est], 
                    label=CMB_namepublic[j], lw=2, ls='-.', capsize=6)


    # ax.plot(ts2.RApArcmin, factor * ts2.stackedProfile[filterType+"_"+est+"_theory_tsz"], ls='--', 
    #     label="theory tsz")
    ax.set_title(key)
    ax.grid()
    if i>=2:
        ax.set_xlabel(r'$R$ [arcmin]')
    if i==0 or i==2:
        ax.set_ylabel(r'Compton Y-parameter $[\mathrm{arcmin}^2]$')
    
ax.legend(fontsize=10, labelspacing=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig(plot_Path, dpi=100) # bbox_inches='tight')

print('Done!!!')