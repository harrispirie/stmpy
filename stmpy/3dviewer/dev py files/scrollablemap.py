'''
Modidfied based on multi_slice_viewer
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
'''
import matplotlib as mpl
# mpl.use('Qt5Agg')
import numpy as np
from misc import grab_bg, blit_bg, volumetype
# from rangeslider import RangeSlider

class ScrollableMap:

    def __init__(self, volume, ax, ch_names=[], **kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        ### check volume type and channel names ###
        if volumetype(volume) == 'a 3D array':
            self.ax.volume = volume
            self.ax.ch_names = [] # kill channel name for map (you should know the channel name)
        elif volumetype(volume) == 'list of 2D arrays':
            self.ax.volume = np.asarray(volume)
            if ch_names:
                if len(volume) == len(ch_names):
                   self.ax.ch_names = ch_names
                else:
                    raise ValueError('ERROR: number of ch_names does not match number of channels in volume!')
            else:
                if len(volume) == 1:
                    self.ax.ch_names = ['Topo']
                else:
                    raise ValueError('ERROR: please provide %d channel names!'%len(volume))
        elif volumetype(volume) == 'a 2D array':
            self.ax.volume = volume.reshape(1, volume.shape[0], volume.shape[1])
            if ch_names:
                self.ax.ch_names = ch_names[0] # only use the first channel name
            else:
                self.ax.ch_names = ['Topo']
                print('WARNING: channel name not provided, use \'Topo\' as default.')
        else:
            raise ValueError('ERROR: volume argument only supports input types of a 2D/3D array or a list of 2D arrays!')

        ### draw the first image in volume ###
        self.ax.index = 0
        self.im = ax.imshow(self.ax.volume[self.ax.index], **kwargs)

        ### connect to mouse scroll event ###
        if self.ax.volume.shape[0] > 1: # suppress scroll event if voloume has only one image
            self.cid = ax.figure.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)

        ### blitting ###
        self.use_blit = True
        if self.use_blit:
            self._bg = None

    def on_mouse_scroll(self, event):
        if event.inaxes != self.ax:
            return
        else:
            if event.button == 'up':
                index = self.ax.index - 1
                if index >= 0:
                    if self.use_blit:
                        self._bg = grab_bg(self.canvas, [self.im,])
                        self.canvas.restore_region(self._bg)
                    self.im.set_array(self.ax.volume[index])
                    self.im.set_clim(self.ax.volume[index].min(), self.ax.volume[index].max())
                    self.ax.index = index
                    if self.use_blit:
                        blit_bg(self.canvas, self._bg, [self.im,])
                    else:
                        self.canvas.draw()
            elif event.button == 'down':
                index = self.ax.index + 1
                if index < self.ax.volume.shape[0]:
                    if self.use_blit:
                        self._bg = grab_bg(self.canvas, [self.im,])
                    self.im.set_array(self.ax.volume[index])
                    self.im.set_clim(self.ax.volume[index].min(), self.ax.volume[index].max())
                    self.ax.index = index
                    if self.use_blit:
                        blit_bg(self.canvas, self._bg, [self.im,])
                    else:
                        self.canvas.draw()


# import matplotlib.pyplot as plt
# import stmpy

# fname = '201201_FeSeTe_006_001.3ds'
# data = stmpy.load(fname, biasOffset=False)
# data.z = stmpy.tools.lineSubtract(data.Z, n=1) #needs to be passed
# data.LIX = data.grid['LI Demod 1 X (A)'] /2 + data.grid['LI Demod 1 X [bwd] (A)']/2 #needs to be passed. Hoffman lab uses LIY
# fig = plt.figure(figsize=(3, 3), dpi=300)
# fig.set_tight_layout(False)
# gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[8, 1])
# ax0 = fig.add_subplot(gs[0])
# ax1 = fig.add_subplot(gs[1])
# # ax[0].imshow(data.LIX[0], cmap=mpl.cm.bwr, origin='lower', aspect=1)
# sm = ScrollableMap(data.LIX, ax0, cmap=mpl.cm.hot, origin='lower', aspect=1)
# rs = RangeSlider(ax1, '', data.LIX.min(), data.LIX.max(), valinit=(data.LIX[0].min(), data.LIX[0].max()), orientation='vertical', cmap=mpl.cm.hot)
# rs.labeltext.set_fontsize(6)
# rs.valtext.set_fontsize(6)
# def update_clim(vals):
#     vmin, vmax = rs.vals[0], rs.vals[1]
#     ax0.get_images()[-1].set_clim(vmin, vmax)
#     fig.canvas.draw_idle()
# rs.on_changed(update_clim)

# plt.show()
