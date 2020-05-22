import numpy as np
import pylab as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

boxProperties = dict(boxstyle='square', facecolor='w', alpha=0.7, linewidth=0.0)
textOptions = dict(fontsize=12, color = 'k', bbox=boxProperties, ha='right', va='top')

def saturate(level_low=0, level_high=None, im=None):
    '''
    Adjusts color axis of in current handle.  Calculates a probablility density
    function for the data in current axes handle.  Uses upper and lower
    thresholds to find sensible c-axis limits.  Thresholds are between 0 and
    100.  If unspecified the upper threshold is assumed to be 100 - lower
    threshold.

    Usage:  pcolormesh(image)
            saturate(10)
    '''
    level_low = float(level_low) / 200.0
    if level_high is None:
        level_high = 1-level_low
    else:
        level_high = (float(level_high)+100) / 200.0
    if im is not None:
        images = [im]
        data = im.get_array().ravel()
    else:
        imageObjects = mpl.pyplot.gca().get_children()
        data = []
        images = []
        for item in imageObjects:
            if isinstance(item, (mpl.image.AxesImage, mpl.collections.QuadMesh)):
                images.append(item)
                data.append(item.get_array().ravel())
    y = sorted(np.array(data).ravel())
    y_density = np.absolute(y) / sum(np.absolute(y))
    pdf = np.cumsum(y_density)
    y_low = np.absolute(level_low - pdf)
    y_high = np.absolute(level_high - pdf)
    c_low = y[np.argmin(y_low)]
    c_high = y[np.argmin(y_high)]
    for image in images:
        image.set_clim(c_low, c_high)


def write_animation(data, fileName, saturation=2, clims=(0,1), cmap=None,
                    label=None, label_caption='meV', speed=8, zoom=1, **kwargs):
    ''' Create a movie from a 3D data set and save it to the path specified. Intended
    for visualising DOS maps and QPI data. Iterates through the first index in
    the data set to create an animation.

    Notes:
        Make sure you include file extension (eg '.mov') in the file path.
        Currently supported file types are mov and mp4.
        Make sure you have ffmpeg installed (e.g. through Homebrew) before running.
        The MP4 writer is really difficult to get high quality.  I learned that
        it depends on figsize, bitrate and dpi.  Basically, if the combination
        of these parameters falls outside of a narrow window the compression
        session will fail to start.  The defaults used in this script seem to
        work.


    Inputs:
        data    - Required : A 3D numpy array containing each frame in the
                             movie.
        fileName - Required : Full path for output file including extension.
        saturation - Optional : Saturation applied to each frame separately.
        clims   - Optional : If saturation=None, user defined clims are used.
                             Specify as a tuple (cmin, cmax).
        cmap    - Optional : Colormap instance will default to cm.bone_r if not
                             specified.
        label   - Optional : Data to use for labeling frames.  Must have the
                             same length as data.shape[0].
        label_caption - Optional : String to be displayed after label.
        speed   - Optional : Frames per second for output.
        zoom    - Optional : Float for zoom factor (eg. 2 times zoom).
        **kwarg - Optional : Additional keyword arguments are sent to imshow().

    Returns:
        Has no returns.

    Usage:
        write_animation(data, fileName, saturation=2, clims=(0,1), cmap=None,
                    label=None, label_caption='meV', speed=8, zoom=1, **kwargs)

    History:
        2017-06-08  -   HP  : Added **kwargs sent to imshow
        2017-06-23  -   HP  : Added support for MP4 export and made codec
                              choice automatic.
    '''
    boxProperties = dict(boxstyle='square', facecolor='w', alpha=0.8, linewidth=0.0)
    textOptions = dict(fontsize=14, color='k', bbox=boxProperties, ha='right', va='center')

    if cmap is None:
        cmap = cm.bone_r
    fig = plt.figure(figsize=[4,4])
    ax = plt.subplot(111)
    ax.set_xticks([]), ax.set_yticks([])
    im = plt.imshow(data[0], cmap=cmap, extent=[-1,1,-1,1], origin='lower',
                    **kwargs)
    ax.set_xlim(-1.0/zoom, 1.0/zoom)
    ax.set_ylim(-1.0/zoom, 1.0/zoom)
    if saturation is not None:
        saturate(saturation, im=im)
    else:
        plt.clim(clims)

    if label is not None:
        tx = ax.text(0.95,0.95,'{:2.1f} {:}'.format(label[0], label_caption),
                  transform=ax.transAxes, **textOptions)
    def init():
        im.set_array(data[0])
        ax.text(20,200,'')
        return [im]

    def animate(i):
        im.set_array(data[i])
        if saturation is not None:
            saturate(saturation, im=im)
        else:
            plt.clim(clims)
        if label is not None:
            tx.set_text('{:2.1f} {:}'.format(label[i], label_caption))
        return [im]
    fig.tight_layout()
    ani = FuncAnimation(fig, animate, init_func=init, frames=data.shape[0])
    if fileName.endswith('.mov'):
        ani.save(fileName, codec='prores', dpi=200, fps=speed)
    elif fileName.endswith('.mp4'):
        ani.save(fileName, dpi=200, bitrate=1e5, fps=speed)
    else:
        print('ERR: fileName must end with .mov or .mp4')


def add_colorbar(loc=0, label='', fs=12, size='5%', pad=0.05, ax=None, im=None,
        ticks=True):
    '''Add a colorbar to the current axis. Alternatively, use add_cbar().

    Inputs:
        loc     - Optional : Specify the location of the colorbar: 0 (bottom),
                             1 (right), 2 (top) or 3 (left).
        label   - Optional : String containing colorbar label
        fs      - Optional : Float for colorbar label fontsize
        size    - Optional : String describing the width of the colorbar as a
                             percentage fo the image.
        pad     - Optional : Float for colorbar pad from image axes.
        ax      - Optional : Axes to attach the colorbar to.  Uses gca() as
                             default.
        im      - Optional : Image used to get colormap and color limits.

    Returns:
        cbar    - matplotlib.colorbar.Colorbar instance.

    History:
        2017-07-20  - HP : Initial commit.
        2017-08-06  - HP : Added flag for removing ticks.
    '''
    if ax is None:
        ax = mpl.pyplot.gca()
    if im is None:
        elements = ax.get_children()
        for element in elements:
            if isinstance(element, (mpl.image.AxesImage, mpl.collections.QuadMesh)):
                im = element
    divider = make_axes_locatable(ax)
    fig = ax.get_figure()
    if loc == 0:
        cax = divider.new_vertical(size=size, pad=pad, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=fs)
    elif loc == 1:
        cax = divider.new_horizontal(size=size, pad=pad, pack_start=False)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, fontsize=fs)
    elif loc == 2:
        cax = divider.new_vertical(size=size, pad=pad, pack_start=False)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=fs)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
    elif loc == 3:
        cax = divider.new_horizontal(size=size, pad=pad, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    else:
        raise ValueError('loc must be 0 (bottom), 1 (right), 2 (top) or 3 (left).')
    if ticks is False:
        cbar.set_ticks([])
    return cbar

def add_cbar(ax=None, im=None, orient='v', length='45%', thickness='7%',
        hPos=None, vPos=None, sf=2, units='', fs=17, pad=0.12, labelpad=0.05):
    '''Adds colorbar to current axis.

    Inputs:
        ax      - Optional : Axes to attach the colorbar to.  Uses gca() as
                             default.
        im      - Optional : Image used to get colormap and color limits.
        orient  - Optional : 'v' for vertical or 'h' for horizontal.
                             Default: 'v'
        length  - Optional : String containing colorbar length as a percent.
                             Default: '45%' (of ax)
        thickness - Optional : String containing colorbar thickness as a percent.
                             Default: '7%' (of ax)
        hPos    - Optional : Float. Horizontal position from lower left of ax.
                             Default: 1.1 (for vertical) (ie 10% away from RHS)
        vPos    - Optional : Float. Vertical position from lower left of ax.
                             Default: 0.1 (for vertical) (ie 10% away from BOTTOM)
        sf      - Optional : Int. Number of significant figures for tick labels.
                             Default: 2
        units   - Optional : String to put after tick label.
                             Default: ''
        fs      - Optional : Float for label fontsize.
                             Default: 17
        pad     - Optional : Float that pushes the colorbar away from ax.
                             Default: 0.12 (equivalent to 12% of ax)
        labelpad - Optional : Float. Distance from edge of colorbar to start of label.
                              Default: 0.05 (equivalent to 5% of ax)

    Returns:
        cbar    - matplotlib.colorbar.Colorbar instance.

    History:
        2019-11-02  - HP : Initial commit.
        2019-11-24  - HP : Added horizontal compatibility.
    '''
    fm ='%.' + str(sf) + 'f '
    ln = float(length.split('%')[0])/100
    th = float(thickness.split('%')[0])/100
    if ax is None:
        ax = mpl.pyplot.gca()
    if im is None:
        elements = ax.get_children()
        for element in elements:
            if isinstance(element, (mpl.image.AxesImage, mpl.collections.QuadMesh)):
                im = element
    if orient == 'v':
        if hPos is None:
            hPos = 1
        if vPos is None:
            vPos = 0.12
        hPos += pad
        axins = inset_locator.inset_axes(ax, width=thickness, height=length,
                        loc='lower left', bbox_to_anchor=(hPos, vPos, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=0)
        cb = mpl.pyplot.colorbar(im, cax=axins, orientation='vertical')
        low, high = cb.get_clim()
        axins.text(hPos+th/2, vPos+ln+labelpad, fm % high + units,
                transform=ax.transAxes, fontsize=fs, ha='center')
        axins.text(hPos+th/2, vPos-labelpad, fm % low + units,
                transform=ax.transAxes, fontsize=fs, ha='center', va='top')
    elif orient == 'h':
        if hPos is None:
            hPos = 0.5 - ln/2
        if vPos is None:
            vPos = 0
        vPos -= pad
        axins = inset_locator.inset_axes(ax, width=length, height=thickness,
                        loc='lower left', bbox_to_anchor=(hPos, vPos, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=0)
        cb = mpl.pyplot.colorbar(im, cax=axins, orientation='horizontal')
        low, high = cb.get_clim()
        axins.text(hPos-labelpad, vPos+th/2, fm % low + units,
                transform=ax.transAxes, fontsize=fs, ha='right', va='center')
        axins.text(hPos+ln+labelpad, vPos+th/2, fm % high + units,
                transform=ax.transAxes, fontsize=fs, ha='left', va='center')
    cb.set_ticks([])
    return cb


def add_label(label, loc=0, ax=None, fs=20, txOptions=None, bbox=None):
    '''Add text labels to images.

    Inputs:
        label   - Required : String (or formatted string) with contents of
                             label.
        loc     - Optional : Integer to describe location of label:
                                 0: top-right
                                 1: top-left
                                 2: bottom-left
                                 3: bottom-right
        ax      - Optional : Axes to place the label.  Uses mpl.pyplot.gca() if
                            not provided.
        txOptions - Optional : Dictionary containing keyword arguments for text
                               formatting, such as: 'fontsize' and 'color'.
        bbox    - Optional : Dictionary containing box properties, such as:
                             'boxstyle', 'facecolor', 'alpha', 'linewidth'.  If
                             no box is desired set to False.

    Returns:
        tx  -   mpl.pyplot.text opbject.

    History:
        2017-08-15  - HP : Initial commit.
        2017-08-31  - HP : Added new locations for text.
    '''
    if ax is None:
        ax = mpl.pyplot.gca()
    if bbox is None:
        bbox = dict(boxstyle='square', facecolor='w', alpha=0.8, linewidth=0.0)
        if txOptions is None:
            txOptions = dict(fontsize=fs, color='k', bbox=bbox)
    elif bbox is False:
        if txOptions is None:
            txOptions = dict(fontsize=fs, color='k')
    else:
        if txOptions is None:
            txOptions = dict(fontsize=fs, color='k', bbox=bbox)

    if loc == 0:
        tx = ax.text(0.95,0.95, label, va='top', ha='right',
                    transform=ax.transAxes, **txOptions)
    elif loc == 1:
        tx = ax.text(0.05,0.95, label, va='top', ha='left',
                    transform=ax.transAxes, **txOptions)
    elif loc == 2:
        tx = ax.text(0.05,0.05, label, va='bottom', ha='left',
                    transform=ax.transAxes, **txOptions)
    elif loc == 3:
        tx = ax.text(0.95,0.05, label, va='bottom', ha='right',
                    transform=ax.transAxes, **txOptions)
    else:
        raise ValueError('loc must be one of the following options:\n' +
                    '\t\t0: top-right\n\t\t1: top-left\n\t\t2: bottom-left' +
                    '\n\t\t3:bottom-right')

    return tx

def add_scale_bar(length, imgsize, imgpixels, barheight=1.5e-2, axes=None, unit='nm',
        loc='lower right', color='w', fs=20, pad=0.5, tex=True, **kwargs):
    """
    Add scale bar to images.

    Inputs:
        length      - Required : Float. The length of scale bar in the given
                                unit.
        imgsize     - Required : Float. The size of image in the given unit.
        imgpixels   - Required : Float. The size of image in unit of pixels.
        barheight   - Optional : Float. The ratio between height of scale bar
                                and the height of the whole image. Default: 1.5 %.
        axes        - Optional : Axes object. It controls which axis the scale
                                bar is added to. If not given, gca() will be the
                                default value.
        unit        - Optional : String. Unit that will be shown in the text of
                                scale bar. Default: nm
        loc         - Optional : Int or string. Default: 'lower right'.
                                'best'          0
                                'upper right'   1
                                'upper left'    2
                                'lower left'    3
                                'lower right'   4
                                'right'         5
                                'center left'   6
                                'center right'  7
                                'lower center'  8
                                'upper center'  9
                                'center'        10
        color       - Optional : String. It specifies the color of the scale bar
                                and the text on it.
        fs          - Optional : Float. Fontsize for the label.
        pad         - Optional : Float. Distance from edge of image.
        tex         - Optional : Boolean. If false, the font is taken from
                                 rcparams. (If true it uses latex fonts).
        **kwargs    - Optional : Passed to AnchorSizeBar(). Including:
                                 'sep' : separation between bar and label
                                 'label_top' : top put label above bar

    Returns:
        scalebar    - scalebar object, which can be added to axis object Ax with
                                "Ax.add_artist(scalebar)"

    History:
        2018-11-02  - RL : Initial commit.
        2019-10-22  - HP : Add option to *not* use tex formatting!
    """

    if axes is None:
        axes = gca()
    to_distance = imgsize/imgpixels
    fontprops = fm.FontProperties(size=fs)
    if tex:
        scalebar = AnchoredSizeBar(axes.transData,
            length/to_distance, r'$\mathbf{{{}''\ {}}}$'.format(length, unit), loc,
            pad=pad,
            color=color,
            frameon=False,
            size_vertical=int(imgpixels*barheight),
            fontproperties=fontprops,
            **kwargs
            )
    else:
        scalebar = AnchoredSizeBar(axes.transData,
            length/to_distance, '{} {}'.format(length, unit), loc,
            pad=pad,
            color=color,
            frameon=False,
            size_vertical=int(imgpixels*barheight),
            fontproperties=fontprops,
            **kwargs
            )

    axes.add_artist(scalebar)
    return scalebar

def get_cbar(cmap, figsize=[0.36,0.07], color='k', orientation='horizontal',
        nseg=100):
    """
    Plot a basic colorbar for a given colormap.

    Inputs:
        cmap    - Required : Instance of stmpy.cm.
        figsize - Optional : Desired size for the colorbar
        color   - Optional : Border color
        orientation - Optional : 'horizontal' or 'vertical'
        nseg    - Optional : Number of discrete colors to use in plotting.

    Returns:
        cb - matplotlib colorbar instance.

    History:
    2020-04-25  - HP : Initial commit.
    """
    a = np.array([[0,1]])
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    im = ax.imshow(a, cmap=cmap)
    ax.set_visible(False)
    cax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    bounds = np.linspace(0, 1, nseg+1)
    cb = plt.colorbar(im, cax=cax, orientation=orientation, boundaries=bounds)
    cb.set_ticks([])
    cb.outline.set_edgecolor(color)
    return cb
