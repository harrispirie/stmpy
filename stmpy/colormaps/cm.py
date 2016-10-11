from matplotlib.colors import ListedColormap as _ListedColormap
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
a single underscore.
'''

_path = _os.path.dirname(__file__) + '/' 

def _make_STMView_colormap(fileName):
    matFile = _loadmat(_path + fileName)
    for key in matFile:
        if key not in ['__version__', '__header__', '__globals__']:
            return _ListedColormap(matFile[key])

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

