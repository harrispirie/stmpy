
'''
Modidfied based on Matplotlib 3.3.3 Cross hair curosr
https://matplotlib.org/3.3.3/gallery/misc/cursor_demo.html
'''
import matplotlib as mpl
import numpy as np


class ClicknDragCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    epsilon = 5
    
    def __init__(self, ax, im):
        self.ax = ax
        canvas = ax.figure.canvas
        self.im = im
        extent = im.get_extent()
        shape = im.get_array().shape
        dx = (extent[1]-extent[0])/shape[-1]
        dy = (extent[3]-extent[2])/shape[-2]
        self.x = np.arange(extent[0]+dx/2, extent[1], dx)
        self.y = np.arange(extent[2]+dy/2, extent[3], dy)
        self._last_indices = [int(shape[-1]/2), int(shape[-2]/2)]
        xc, yc = self.x[self._last_indices[0]], self.y[self._last_indices[1]]
        self.hline = ax.axhline(yc, color='y', lw=0.5)
        self.vline = ax.axvline(xc, color='y', lw=0.5)
        # print('Initial:%.2f, %.2f'%self._last_indices[:])
        self.drag = False
        # text location in axes coords
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)

    def on_draw(self, event):
        # self.create_new_background()
        # self.ax.draw_artist(self.im)
        self.ax.draw_artist(self.hline)
        self.ax.draw_artist(self.vline)

    # def set_cross_hair_visible(self, visible):
    #     need_redraw = (self.hline.get_visible() != visible) or (self.vline.get_visible() != visible)
    #     self.hline.set_visible(visible)
    #     self.vline.set_visible(visible)
    #     self.text.set_visible(visible)
    #     return need_redraw

    # def create_new_background(self):
    #     if self._creating_background:
    #         # discard calls triggered from within this function
    #         return
    #     self._creating_background = True
    #     self.set_cross_hair_visible(False)
    #     self.ax.figure.canvas.draw()
    #     self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
    #     self.set_cross_hair_visible(True)
    #     self._creating_background = False

    def on_button_press(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        index = self._last_indices
        xy_disp = self.ax.transData.transform((self.x[index[0]], self.y[index[1]]))
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
                index = [min(np.searchsorted(self.x, x), len(self.x) - 1), min(np.searchsorted(self.y, y), len(self.y) - 1)]
                if np.allclose(index, self._last_indices):
                    return  # still on the same data point. Nothing to do.
                self._last_indices = index
                x = self.x[index[0]]
                y = self.y[index[1]]
                # update the line positions
                self.hline.set_ydata(y)
                self.vline.set_xdata(x)
                # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.set_title('Click & Drag Cursor')
# a = np.linspace(0, 40, 16)
# b = np.linspace(0, 40, 16)
# c = np.linspace(-1, 1, 5)
# z = np.random.rand(len(c), len(b), len(a))
# im = ax.imshow(z[0], cmap=mpl.cm.Blues, origin='lower', aspect=1, extent=[a[0], a[-1], b[0], b[-1]])
# dc = ClicknDragCursor(ax, im)
# plt.show()

