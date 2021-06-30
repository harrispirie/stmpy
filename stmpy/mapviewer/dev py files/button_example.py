import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
fig.set_tight_layout(False)
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l1, = plt.plot(t, s, lw=2)


class Index:
    ind = 1
    def __init__(self, button, line):
        self.button = button
        self.line = line
    def vis(self, event):
        self.ind = (self.ind + 1) % 2
        bcolors = ['0.85', 'gold']
        self.line.set_visible(self.ind)
        self.button.ax.set_facecolor(bcolors[self.ind])
        self.button.hovercolor = bcolors[self.ind]
        self.line.figure.canvas.draw()


# axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.12, 0.075])
bnext = Button(axnext, 'Cursor', color='gold', hovercolor='gold')
callback = Index(bnext, l1)
bnext.on_clicked(callback.vis)
# bprev = Button(axprev, 'Previous')
# bprev.on_clicked(callback.prev)

plt.show()