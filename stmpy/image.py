import numpy as np
import stmpy as sp
import pylab as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

def write_animation(F, fileName, saturation=2, label=None, cmap=None, speed=8,
                    zoom=1, codec='prores', clims=(0,1)):
    '''
    Create a movie from a 3D data set and save it in the working directory.  Intended    for visualising DOS maps.

    Iterates through the first index in the data set to create an animation.
    Make sure you include '.mov' in the file name.  Default codec is prores
    (apple), other codecs can be specified. Make sure you have ffmpeg installed
    (e.g. through Homebrew) before running.

    Usage: write_animation(data.LIY, 'MOV_DOSmap.mov', saturation=2,
    label=data.en, cmap=cm.bone_r, speed=8, zoom=1, codec='prores')
    '''

    boxProperties1 = dict(boxstyle='square', facecolor='w', 
                            alpha=0.8, linewidth=0.0)
    textOptions1 = dict(fontsize=14, color = 'k', 
                        bbox=boxProperties1, ha='right', va='center')
    if cmap is None:
        cmap = cm.bone_r

    x = np.linspace(-1, 1, F.shape[1]+1)
    fig = plt.figure(figsize=[4,4])
    ax = plt.subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(-1.0/zoom, 1.0/zoom)
    plt.ylim(-1.0/zoom, 1.0/zoom)
    im = plt.pcolormesh(x, x, F[0], cmap=cmap)
    if saturation is not None:
        sp.saturate(saturation)
    else:
        plt.clim(clims)

    if label is not None:
        tx = plt.text(0.95,0.95,'{:2.2f} meV'.format(label[0]), 
                  transform=ax.transAxes, **textOptions1)
    def init():
        im.set_array(F[0].ravel())
        plt.text(20,200,'')
        return [im]

    def animate(i):
        im.set_array(F[i].ravel())
        if saturation is not None:
            sp.saturate(saturation)
        else:
            plt.clim(clims)
        if label is not None:
            tx.set_text('{:2.0f} meV'.format(label[i]))
        return [im]
    fig.tight_layout()
    ani = FuncAnimation(fig, animate, init_func=init, frames = F.shape[0])
    ani.save(fileName, codec=codec, dpi=200, fps=speed)






