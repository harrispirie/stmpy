import numpy as np
import pylab as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import cm

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
        tx = ax.text(0.95,0.95,'{:2.2f} {:}'.format(label[0], label_caption), 
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
            tx.set_text('{:2.0f} {:}'.format(label[i], label_caption))
        return [im]
    fig.tight_layout()
    ani = FuncAnimation(fig, animate, init_func=init, frames=data.shape[0])
    if fileName.endswith('.mov'):
        ani.save(fileName, codec='prores', dpi=200, fps=speed)
    elif fileName.endswith('.mp4'):
        ani.save(fileName, dpi=200, bitrate=1e5, fps=speed)
    else:
        print('ERR: fileName must end with .mov or .mp4')



