import matplotlib as mpl
import numpy as np

class LimLinkedAxes:
    def __init__(self, ax1, ax2):
        self.ax1 = ax1
        self.ax2 = ax2
        self.canvas = ax1.figure.canvas
        self.oxy = [self._get_limits(ax) for ax in [ax1, ax2]]
        self.cid1 = self.canvas.mpl_connect('motion_notify_event', self.re_zoom)  # for right-click pan/zoom
        self.cid2 = self.canvas.mpl_connect('button_release_event',self.re_zoom)  # for rectangle-select zoom

    def _get_limits(self, ax):
        return [list(ax.get_xlim()), list(ax.get_ylim())]

    def _set_limits(self, ax, lims):
        ax.set_xlim(*(lims[0]))
        ax.set_ylim(*(lims[1]))

    def re_zoom(self, event):
        for ax in [self.ax1, self.ax2]:
            navmode = ax.get_navigate_mode()
            if navmode is not None:
                break
        if navmode not in ['PAN', 'ZOOM']:
            return
        else:
            for ix, ax in enumerate([self.ax1, self.ax2]):
                nxy = self._get_limits(ax)
                if self.oxy[ix] != nxy:
                    self._set_limits(self.ax1, nxy)
                    self._set_limits(self.ax2, nxy)
                    self.canvas.draw()
                    self.oxy = [self._get_limits(ax) for ax in [self.ax1, self.ax2]]

# import matplotlib.pyplot as plt
# x = np.linspace(-1, 1, 11)
# y1 = x
# y2 = x ** 2
# extent = [x[0], x[-1], x[0], x[-1]]
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(x, y1, 'ks', ms=3)
# ax[0, 1].plot(x, y2, 'r', lw=1)
# ax[1, 0].imshow(np.random.rand(20, 20), cmap=mpl.cm.gray, origin='lower', aspect=1, extent=extent)
# ax[1, 1].imshow(np.random.rand(20, 20), cmap=mpl.cm.Reds, origin='lower', aspect=1, extent=extent)
# lla = LimLinkedAxes(ax[1, 0], ax[1, 1])
# plt.show()