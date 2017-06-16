from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.pylab import cm as _cm
from scipy.io import loadmat as _loadmat
import os as _os

''' Create STM colormaps.
HP 11/10/2016

Colormaps created here can be called from within stmpy as
stmpy.cm.name_of_colormap. Primarily the colormaps are created from STMView
.mat files. 

One goal of this module is to only make colormaps accessible so that when a
user uses tab auto-completion they are presented with a list of colormaps and
NOTHING ELSE.  For this reason all other vairables are 'hidden' and begin with
a single underscore. Additionally, all colormaps in the module should be
accompanied by a reversed verion with the same name but '_r' on the end.
'''

_path = _os.path.dirname(__file__) + '/' 

def _make_STMView_colormap(fileName, name='my_cmap'):
    matFile = _loadmat(_path + fileName)
    for key in matFile:
        if key not in ['__version__', '__header__', '__globals__']:
           return _LSC.from_list(name, matFile[key])


def _reverse_LSC(cmap):     
    reverse = []
    k = []   
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
    LinearL = dict(zip(k,reverse))
    my_cmap_r = _LSC(cmap.name + '_r', LinearL) 
    return my_cmap_r

def _make_diverging_colormap(i, f, m=[1,1,1], name='my_cmap'):
    '''
    Creates a three-color diverging colormap by interpolating smoothly between
    the RGB inputs. Output is a LinearSegmentedColormap from Matplotlib, which
    can be easily reversed using: cmap_r = _reverse_LSC(cmap), which is a
    function written in this module. 
    Inputs:
        i - Initial color as RGB or RGBA tuple or list
        f - Final color as RGB or RGBA tuple or list
        m - Middle color as RGB or RGBA tuple or list (default: white)
        name - optional name for colormap (does nothing?)
    Usage:
        BuGy = _make_diverging_colormap(i, f, m=[1,1,1], name='my_cmap')
    '''
    _cdict = {'red':   ((0.0, i[0], i[0]), (0.5, m[0], m[0]), (1.0, f[0], f[0])),
              'green': ((0.0, i[1], i[1]), (0.5, m[1], m[1]), (1.0, f[1], f[1])),
              'blue':  ((0.0, i[2], i[2]), (0.5, m[2], m[2]), (1.0, f[2], f[2]))}
    return _LSC(name, _cdict)

BuGy = _make_diverging_colormap(_cm.RdGy(0.99), _cm.RdBu(0.99))
GnGy = _make_diverging_colormap(_cm.RdGy(0.99), _cm.BuGn(0.99))
bluered = _make_diverging_colormap(i=(0.230, 0.299, 0.754), f=(0.706, 0.016,0.150), 
                                   m=(0.865, 0.865, 0.865), name='bluered')
yanghe = _make_STMView_colormap('YH.mat')
autumn = _make_STMView_colormap('Autumn.mat')
blue1 = _make_STMView_colormap('Blue1.mat')
blue2 = _make_STMView_colormap('Blue2.mat')
blue3 = _make_STMView_colormap('Blue3.mat')
defect0 = _make_STMView_colormap('Defect0.mat')
defect1 = _make_STMView_colormap('Defect1.mat')
defect2 = _make_STMView_colormap('Defect2.mat')
defect4 = _make_STMView_colormap('Defect4.mat')
gray = _make_STMView_colormap('Gray.mat')
sailingMod2 = _make_STMView_colormap('SailingMod2.mat')

BuGy_r = _reverse_LSC(BuGy)
GnGy_r = _reverse_LSC(GnGy)
bluered_r = _reverse_LSC(bluered)
yanghe_r =_reverse_LSC(yanghe)
autumn_r =_reverse_LSC(autumn)
blue1_r = _reverse_LSC(blue1)
blue2_r = _reverse_LSC(blue2)
blue3_r = _reverse_LSC(blue3)
defect0_r = _reverse_LSC(defect0)
defect1_r = _reverse_LSC(defect1)
defect2_r = _reverse_LSC(defect2)
defect4_r = _reverse_LSC(defect4)
gray_r = _reverse_LSC(gray)
sailingMod2_r = _reverse_LSC(sailingMod2)


