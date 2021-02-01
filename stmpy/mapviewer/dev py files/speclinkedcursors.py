import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
from misc import format_eng, grab_bg, blit_bg

class SpecLinkedCursors:
    '''
    Two linked cursors in two separate images/axes. Image shapes/extents must be the same.
    '''
    epsilon = 10
    cursor_color = 'gold'
    cursor_lw = 0.5
    title_fontdict = dict(fontsize=6)

    
    def __init__(self, axtopo, axmap, axspec, use_blit=True):
        '''pass data.en to energy, replace with Spy object and use spy.en to pass it'''
        self.axtopo = axtopo
        # axmap has attributes: volume = map array; index = bias index to show; ch_names = list of channel names(default = [])
        self.axmap = axmap
        self.axspec = axspec
        self.fig = axtopo.figure
        self.canvas = self.fig.canvas
        self.im1 = axtopo.get_images()[-1] # image on top
        self.im2 = axmap.get_images()[-1]
        self.spec = axspec.get_lines()[0] # plot line on bottom
        self.en = self.spec.get_data()[0]
        self.en_ind = axmap.index
        self.z_ind = axtopo.index
        extent = self.im1.get_extent() # set by scan info
        shape = self.im1.get_array().shape
        if not np.allclose(extent, self.im2.get_extent()):
            print('WARNING: two images have different extents! Please check input arrays!')
        if not shape == self.im2.get_array().shape:
            print('WARNING: two images have different shapes! Please check input arrays!')
        self.dx = (extent[1]-extent[0])/shape[-1]
        self.dy = (extent[3]-extent[2])/shape[-2]
        self.x = np.arange(extent[0]+self.dx/2, extent[1], self.dx) # half pixel shift in imshow
        self.y = np.arange(extent[2]+self.dy/2, extent[3], self.dy)
        self._last_indices = [int(shape[-1]/2), int(shape[-2]/2)]
        self.spec.set_ydata(axmap.volume[:, self._last_indices[1], self._last_indices[0]])
        self.bline = axspec.axvline(self.en[self.en_ind], color=self.cursor_color, lw=self.cursor_lw)
        self.btext = axmap.set_title('Bias='+format_eng(self.en[self.en_ind])+'V', fontdict=self.title_fontdict)
        self.ttext = axtopo.set_title(self.axtopo.ch_names[self.z_ind], fontdict=self.title_fontdict)
        axspec.relim()
        axspec.autoscale_view()
        xc, yc = self.x[self._last_indices[0]], self.y[self._last_indices[1]]
        self.hline1 = axtopo.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline1 = axtopo.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        self.hline2 = axmap.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline2 = axmap.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        self.drag = False
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_bscroll = self.canvas.mpl_connect('scroll_event', self.bias_scroll)
        if self.axtopo.volume.shape[0] > 1: # suppress scroll event if voloume has only one image
            self.cid_tscroll = self.canvas.mpl_connect('scroll_event', self.topo_scroll)
        self.use_blit = use_blit
        if self.use_blit:
            self._bg = None

    def on_button_press(self, event):
        if event.inaxes not in [self.axtopo, self.axmap]:
            return
        if event.button != 1:
            return
        index = self._last_indices
        xy_disp = event.inaxes.transData.transform((self.x[index[0]], self.y[index[1]]))
        d = np.sqrt((xy_disp[0] - event.x)**2 + (xy_disp[1] - event.y)**2)
        if d <= self.epsilon:
            self.drag = True
            # Only connect to mouse movement when the left mouse button REMAINS pressed
            self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_button_release)
            self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.drag_cursors)

    def on_button_release(self, event):
        # if event.inaxes not in [self.axtopo, self.axmap]:
        #     return
        if event.button != 1:
            return
        self.drag = False
        self.canvas.mpl_disconnect(self.cid_move)
        self.canvas.mpl_disconnect(self.cid_release)

    def bias_scroll(self, event):
        if event.inaxes != self.axmap:
            return
        else:
            if event.button == 'up': 
                ind = self.en_ind - 1
                if ind >= 0:
                    self.bias_scroll_update(ind)
            elif event.button == 'down':
                ind = self.en_ind + 1
                if ind < len(self.en):
                    self.bias_scroll_update(ind)

    def bias_scroll_update(self, ind):
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.bline, self.btext, self.im2, self.vline2, self.hline2])
            self.canvas.restore_region(self._bg)
        self.bline.set_xdata(self.en[ind])
        self.im2.set_array(self.axmap.volume[ind])
        self.btext = self.axmap.set_title('Bias='+format_eng(self.en[ind])+'V', fontdict=self.title_fontdict)
        self.en_ind = ind
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.bline, self.btext, self.im2, self.vline2, self.hline2])
        else:
            self.canvas.draw()

    def topo_scroll(self, event):
        if event.inaxes != self.axtopo:
            return
        else:
            if event.button == 'up':
                ind = self.z_ind - 1
                if ind >= 0:
                    self.topo_scroll_update(ind)
            elif event.button == 'down':
                ind = self.z_ind + 1
                if ind < self.axtopo.volume.shape[0]:
                    self.topo_scroll_update(ind)

    def topo_scroll_update(self, ind):
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.ttext, self.im1, self.vline1, self.hline1])
            self.canvas.restore_region(self._bg)
        self.bline.set_xdata(self.en[ind])
        self.im1.set_array(self.axtopo.volume[ind])
        self.ttext = self.axtopo.set_title(self.axtopo.ch_names[ind], fontdict=self.title_fontdict)
        self.z_ind = ind
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.ttext, self.im1, self.vline1, self.hline1])
        else:
            self.canvas.draw()        


    def drag_cursors(self, event):
        if event.inaxes not in [self.axtopo, self.axmap]:
            self.drag = False
            return
        else:
            for ax in [self.axtopo, self.axmap]:
                navmode = ax.get_navigate_mode()
                if navmode is not None:
                    self.drag = False
                    break
            if self.drag:   
                x, y = event.xdata, event.ydata
                indices = [min(np.searchsorted(self.x+self.dx/2, x), len(self.x) - 1), min(np.searchsorted(self.y+self.dy/2, y), len(self.y) - 1)]
                if np.allclose(indices, self._last_indices):
                    return  # still on the same data point. Nothing to do.
                self._last_indices = indices
                x = self.x[indices[0]]
                y = self.y[indices[1]]
                # update the line positions
                if self.use_blit:
                    self._bg = grab_bg(self.canvas, [self.hline1, self.hline2, self.vline1, self.vline2, self.spec, self.bline])
                    self.canvas.restore_region(self._bg)
                self.hline1.set_ydata(y)
                self.vline1.set_xdata(x)
                self.hline2.set_ydata(y)
                self.vline2.set_xdata(x)
                self.spec.set_ydata(self.axmap.volume[:, indices[1], indices[0]])
                self.axspec.relim()
                self.axspec.autoscale_view()
                if self.use_blit:
                    blit_bg(self.canvas, self._bg, [self.hline1, self.hline2, self.vline1, self.vline2, self.spec, self.bline])
                else:
                    self.canvas.draw()



import matplotlib.pyplot as plt
from scrollablemap import ScrollableMap
from limlinkedaxes import LimLinkedAxes

fig = plt.figure(num='3DS Viewer', figsize=(3.8, 3), dpi=300)
# fig.set_tight_layout(False)
ax1 = plt.subplot2grid((5, 6), (0, 0), rowspan=2, colspan=3) # control area
ax2 = plt.subplot2grid((5, 6), (0, 3), rowspan=2, colspan=3) # spectrum plot
ax3 = plt.subplot2grid((5, 6), (2, 0), rowspan=3, colspan=3) # 2D image
ax4 = plt.subplot2grid((5, 6), (2, 3), rowspan=3, colspan=3)  # 3D array, slices will be showed
fig.suptitle('Spectrum-linked Cursors')
# a typical map size 150x150x51
a = np.linspace(0, 40, 150)
b = np.linspace(0, 40, 150)
c = np.linspace(-1, 1, 5)
z = np.random.rand(len(c), len(b), len(a))
# for iy in range(len(b)):
#     for ix in range(len(a)):
#         offset = ix + iy * len(a) + 1
#         z[:, iy, ix] = c**2 + offset + 0.1*np.random.rand(len(c),)

var1, var2, var3 = np.copy(z[1]), np.copy(z[2]), np.copy(z[3])
var1[1, 1] = 2
var2[75, 75] = 2
var3[-2, -2]  = 2
# ax3.imshow(z[0], cmap=mpl.cm.Blues, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
sm1 = ScrollableMap(var1, ax3, ch_names=[], cmap=mpl.cm.Blues, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
sm2 = ScrollableMap(z, ax4, cmap=mpl.cm.gray, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
inds = (0, 0)
ax2.plot(c, z[:, inds[1], inds[0]], lw=1)
ax1.set_axis_off()
lla = LimLinkedAxes(ax3, ax4)
dc = SpecLinkedCursors(ax3, ax4, ax2)
plt.show()

