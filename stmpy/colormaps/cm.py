from matplotlib.colors import ListedColormap as _ListedColormap
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

def _make_STMView_colormap(fileName):
    matFile = _loadmat(_path + fileName)
    for key in matFile:
        if key not in ['__version__', '__header__', '__globals__']:
            return _ListedColormap(matFile[key])

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

RdBu = _cm.RdBu
RdGy = _cm.RdGy
BuGy = _make_diverging_colormap(_cm.RdGy(0.99), _cm.RdBu(0.99))
GnGy = _make_diverging_colormap(_cm.RdGy(0.99), _cm.BuGn(0.99))
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

RdBu_r = _cm.RdBu_r
RdGy_r = _cm.RdGy_r
BuGy_r = _reverse_LSC(BuGy)
GnGy_r = _reverse_LSC(GnGy)
yanghe_r =_ListedColormap(yanghe.colors[::-1])
autumn_r =_ListedColormap(autumn.colors[::-1])
blue1_r = _ListedColormap(blue1.colors[::-1])
blue2_r = _ListedColormap(blue2.colors[::-1])
blue3_r = _ListedColormap(blue3.colors[::-1])
defect0_r = _ListedColormap(defect0.colors[::-1])
defect1_r = _ListedColormap(defect1.colors[::-1])
defect2_r = _ListedColormap(defect2.colors[::-1])
defect4_r = _ListedColormap(defect4.colors[::-1])
gray_r = _ListedColormap(gray.colors[::-1])

