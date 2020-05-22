from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.pylab import cm
from scipy.io import loadmat as _loadmat
import numpy as _np
import os as _os

'''
A collection of nice colormaps combining those from stmview, the default
matplotlib ones and custom stmpy colormaps.

Usage:
    When making a colormap, ensure it is correcly named (i.e. it must have a
    cmap.name attrubute), then add it to the list at the bottom of the file to
    create the reversed colormap.

 History:
    2016-11-10  - HP : Initial commit.
    2017-10-31  - HP : Incorporated all matplotlib colormaps.
'''

_path = _os.path.join(_os.path.dirname(__file__), 'maps', '')

def invert_cmap(cmap, name='my_cmap'):
    '''
    Creates a new colormap from an existing Listed Colormap by implementing the
    mapping:
        (R, G, B, alpha)   -->     (1-R, 1-G, 1-B, alpha)

    Inputs:
        cmap    - Required : Must be a Listed Colormap.

    Returns:
        newCmap - Inverse colormap of cmap.

    History:
        2017-10-31  - HP : Initial commit.
    '''
    colors = cmap(_np.arange(cmap.N))
    colors[:,:3] = 1 - colors[:,:3]
    return cmap.from_list(name, colors, cmap.N)


def _make_STMView_colormap(fileName, name='my_cmap'):
    if fileName.endswith('.mat'):
        matFile = _loadmat(_path + fileName)
        for key in matFile:
            if key not in ['__version__', '__header__', '__globals__']:
                return _LSC.from_list(name, matFile[key])
    elif fileName.endswith('.txt'):
        txtFile = _np.loadtxt(_path + fileName)
        return _LSC.from_list(name, txtFile)

def _write_cmap_to_file(fileName, cmap):
    with open(fileName, 'w') as fileID:
        for ix in range(256):
            val = cmap(ix/256.0)
            for v in val[:-1]:
                fileID.write(str(v))
                fileID.write(', ')
            if ix != 255:
                fileID.write('\n ')

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

cm.BuGy = _make_diverging_colormap(cm.RdGy(0.99), cm.RdBu(0.99), name='BuGy')
cm.GnGy = _make_diverging_colormap(cm.RdGy(0.99), cm.BuGn(0.99), name='GnGy')
cm.redblue = _make_diverging_colormap(i=(0.230, 0.299, 0.754), f=(0.706, 0.016,0.150),
                                      m=(0.865, 0.865, 0.865), name='redblue')
cm.autumn = _make_STMView_colormap('Autumn.mat', name='autumn')
cm.blue1 = _make_STMView_colormap('Blue1.mat', name='blue1')
cm.blue2 = _make_STMView_colormap('Blue2.mat', name='blue2')
cm.blue3 = _make_STMView_colormap('Blue3.mat', name='blue3')
cm.defect0 = _make_STMView_colormap('Defect0.mat', name='defect0')
cm.defect1 = _make_STMView_colormap('Defect1.mat', name='defect1')
cm.defect2 = _make_STMView_colormap('Defect2.mat', name='defect2')
cm.defect4 = _make_STMView_colormap('Defect4.mat', name='defect4')
cm.gray = _make_STMView_colormap('Gray.mat', name='gray')

cm.PuGn = _make_STMView_colormap('PuGn.txt', name='PuGn')

cm.sailingMod2 = _make_STMView_colormap('SailingMod2.mat', name='sailingMod2')
cm.jackyYRK = _make_diverging_colormap([1, 1, 0], [0, 0, 0.5],
                                        m=[0.7, 0.2, 0], name='jackyYRK')
cm.jackyCopper = _make_diverging_colormap([0.2, 0.1, 0], [1, 0.95, 0.6],
                                        m=[1, 0.65, 0.25], name='jackyCopper')
cm.jackyRdGy = _make_diverging_colormap([0.2, 0.2, 0.2], [0.7, 0, 0],
                                        m=[0.95,0.95, 0.95], name='jackyRdGy')
_cdictPSD = {'red':   ((0.00, 0.00, 0.06),
                       (0.25, 0.21, 0.21),
                       (0.45, 0.31, 0.31),
                       (0.65, 1.00, 1.00),
                       (1.00, 1.00, 1.00)),

             'green': ((0.00, 0.00, 0.06),
                       (0.25, 0.24, 0.24),
                       (0.50, 0.38, 0.38),
                       (0.75, 0.83, 0.83),
                       (1.00, 0.98, 1.00)),

             'blue':  ((0.00, 0.00, 0.08),
                       (0.25, 0.47, 0.47),
                       (0.50, 0.33, 0.33),
                       (0.75, 0.27, 0.27),
                       (1.00, 0.95, 1.00))}

cm.jackyPSD = _LSC('jackyPSD', _cdictPSD)
cm.jason = _make_STMView_colormap('Red_Blue.txt', name='jason')
cm.yanghe = invert_cmap(cm.defect0, name='yanghe')
cm.helix = invert_cmap(cm.cubehelix_r, name='helix')
cm.gold = invert_cmap(cm.bone_r, name='gold')
cm.als = _make_STMView_colormap('ALS.txt', name='als')
cm.hpblue = _make_diverging_colormap([0,0,0],
        [0.14901960784313725, 0.5450980392156862, 0.9176470588235294])
cm.mhblue = _make_STMView_colormap('mhblue.mat', name='mhblue')



# Reverse Cmaps: Add new cmap name to the list.
cmaps = [cm.BuGy, cm.GnGy, cm.redblue, cm.autumn, cm.blue1, cm.blue2, cm.blue3,
         cm.defect0, cm.defect1, cm.defect2, cm.defect4, cm.gray,
         cm.sailingMod2, cm.jackyYRK, cm.jackyCopper, cm.jackyRdGy,
         cm.jackyPSD, cm.jason, cm.helix, cm.yanghe, cm.gold, cm.als,
         cm.hpblue, cm.mhblue, cm.PuGn]

for cmap in cmaps:
    rev = _reverse_LSC(cmap)
    setattr(cm, cmap.name + '_r', rev)


# Remove non-cmap attributes and methods from cm
removeall = ['colors', 'absolute_import', 'cmaps_listed', 'cmapname', 'cmap_d', 'datad',
          'division', 'get_cmap', 'LUTSIZE', 'ma', 'mpl', 'np', 'print_function', 'os',
          'register_cmap', 'revcmap', 'ScalarMappable', 'six', 'unicode_literals', 'cbook']

remove = ['colors', 'absolute_import', 'cmaps_listed', 'cmapname', 'cmap_d', 'datad',
          'division', 'get_cmap', 'LUTSIZE', 'ma', 'mpl', 'np', 'print_function', 'os',
          'register_cmap', 'revcmap', 'six', 'unicode_literals', 'cbook']

#for name in remove:
#    delattr(cm, name)
