from headers import *

##################################################################################

class cmbMap(object):

    def __init__(self, pathMap, pathMask=None, pathHit=None,  name="test", nu=150.e9, unitLatex=r'$\mu$K', convert_y=False):
        self.name = name
        self.pathMap = pathMap
        self.pathMask = pathMask
        self.pathHit = pathHit

        self.nu = nu
        self.unitLatex = unitLatex
        self.convert_y = convert_y

    def map(self):
        result = enmap.read_map(self.pathMap)
        # if the map contains polarization, keep only temperature
        if len(result.shape) > 2:
            result = result[0]
        if self.convert_y:
            Tcmb = 2.726   # K
            h = 6.63e-34   # SI
            kB = 1.38e-23  # SI
            def f(nu):
               """frequency dependence for tSZ temperature
               """
               x = h*nu/(kB*Tcmb)
               return x*(np.exp(x)+1.)/(np.exp(x)-1.) -4.
            result = result / (Tcmb * f(self.nu) * 1.e6)
        return result

    def mask(self):
        if self.pathMask is None:

            CMBmap = enmap.read_map(self.pathMap)
            if len(CMBmap.shape) > 2:
                CMBmap = CMBmap[0]
            result = np.logical_not(CMBmap == 0).astype(int)
            return result
        result = enmap.read_map(self.pathMask)
        # if the map contains polarization, keep only temperature
        if len(result.shape) > 2:
            result = result[0]
        return result

    def hit(self):
        if self.pathHit is None:
            return None
        else:
            result = enmap.read_map(self.pathHit)
            # if the map contains polarization, keep only temperature
            if len(result.shape) > 2:
                result = result[0]
            return result
        

class cmbMap2(object):

    def __init__(self, Map, Mask=None, Hit=None,  name="test", nu=150.e9, unitLatex=r'$\mu$K', convert_y=False):
        self.name = name
        self.Map_ = Map
        self.Mask_ = Mask
        self.Hit_ = Hit

        self.nu = nu
        self.unitLatex = unitLatex
        self.convert_y = convert_y

    def map(self):
        result = self.Map_.copy()
        # if the map contains polarization, keep only temperature
        if len(result.shape) > 2:
            result = result[0]
        if self.convert_y:
            Tcmb = 2.726   # K
            h = 6.63e-34   # SI
            kB = 1.38e-23  # SI
            def f(nu):
               """frequency dependence for tSZ temperature. This is the non-relativistic conversion
               """
               x = h*nu/(kB*Tcmb)
               return x*(np.exp(x)+1.)/(np.exp(x)-1.) -4.
            result = result / (Tcmb * f(self.nu) * 1.e6)
        return result

    def mask(self):
        if self.Mask_ is None:

            CMBmap = self.Map_.copy()
            if len(CMBmap.shape) > 2:
                CMBmap = CMBmap[0]
            result = np.logical_not(CMBmap == 0).astype(int)
            return result
        result = self.Mask_.copy()
        # if the map contains polarization, keep only temperature
        if len(result.shape) > 2:
            result = result[0]
        return result

    def hit(self):
        if self.Hit_ is None:
            return None
        else:
            result = self.Hit_.copy()
            # if the map contains polarization, keep only temperature
            if len(result.shape) > 2:
                result = result[0]
            return result
