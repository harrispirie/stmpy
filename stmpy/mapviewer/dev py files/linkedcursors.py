# built-in solution from matplotlib.widget import MultiCursor

import matplotlib as mpl
import numpy as np


class LinkedCursors:
    '''
    Two linked cursors in two separate images/axes. Image shapes/extents must be the same.
    '''
    epsilon = 5
    cursor_color = 'gold'
    cursor_lw = 0.5
    
    def __init__(self, ax1, ax2):
        self.ax1 = ax1
        self.ax2 = ax2
        canvas = ax1.figure.canvas
        # canvas2 = ax1.figure.canvas
        im1 = ax1.get_images()[-1] # image on top
        im2 = ax2.get_images()[-1]
        extent = im1.get_extent()
        shape = im1.get_array().shape
        # print(shape, im1.get_array().shape)
        if not np.allclose(extent, im2.get_extent()):
            print('Warning: two images have different extents! Please check input arrays!')
        if not shape == im2.get_array().shape:
            print('Warning: two images have different shapes! Please check input arrays!')
        self.dx = (extent[1]-extent[0])/shape[-1]
        self.dy = (extent[3]-extent[2])/shape[-2]
        self.x = np.arange(extent[0]+self.dx/2, extent[1], self.dx) # half pixel shift in imshow
        self.y = np.arange(extent[2]+self.dy/2, extent[3], self.dy)
        self._last_indices = [int(shape[-1]/2), int(shape[-2]/2)]
        xc, yc = self.x[self._last_indices[0]], self.y[self._last_indices[1]]
        self.hline1 = ax1.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline1 = ax1.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        self.hline2 = ax2.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline2 = ax2.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        # print('Initial:%.2f, %.2f'%self._last_indices[:])
        self.drag = False
        # text location in axes coords
        # self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
        # for canvas in [canvas1, canvas2]:
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)

    def on_draw(self, event):
        # self.create_new_background()
        # self.ax.draw_artist(self.im)
        self.ax1.draw_artist(self.hline1)
        self.ax1.draw_artist(self.vline1)
        self.ax2.draw_artist(self.hline2)
        self.ax2.draw_artist(self.vline2)

    def on_button_press(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        index = self._last_indices
        xy_disp = event.inaxes.transData.transform((self.x[index[0]], self.y[index[1]]))
        d = np.sqrt((xy_disp[0] - event.x)**2 + (xy_disp[1] - event.y)**2)
        if d <= self.epsilon:
            self.drag = True

    def on_button_release(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.drag = False
        # self.ax.figure.canvas.draw()

    def on_mouse_move(self, event):
        if not event.inaxes:
            self.drag = False
            return
        else:
            # self.drag = False
        #     self._last_indices = None
            # need_redraw = self.set_cross_hair_visible(False)
            # if need_redraw:
            #     self.ax.figure.canvas.draw()
        # else:
        #     self.set_cross_hair_visible(True)
            if self.drag:   
                x, y = event.xdata, event.ydata
                index = [min(np.searchsorted(self.x+self.dx/2, x), len(self.x) - 1), min(np.searchsorted(self.y+self.dy/2, y), len(self.y) - 1)]
                if np.allclose(index, self._last_indices):
                    return  # still on the same data point. Nothing to do.
                self._last_indices = index
                x = self.x[index[0]]
                y = self.y[index[1]]
                # update the line positions
                self.hline1.set_ydata(y)
                self.vline1.set_xdata(x)
                self.hline2.set_ydata(y)
                self.vline2.set_xdata(x)
                # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax1.figure.canvas.draw()
            self.ax2.figure.canvas.draw()


# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Linked Cursors')
# a = np.linspace(0, 40, 6)
# b = np.linspace(0, 40, 6)
# c = np.linspace(-1, 1, 5)
# z = np.random.rand(len(c), len(b), len(a))
# ax1.imshow(z[0], cmap=mpl.cm.Blues, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
# # ax2.imshow(z[-1], cmap=mpl.cm.gray, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
# sm2 = ScrollableMap(z, ax2, cmap=mpl.cm.gray, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
# dc = LinkedCursors(ax1, ax2)
# plt.show()

