import matplotlib
# PyQt5 package required for mac users 
# matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class Index:
    def __init__(self, line, t, freqs):
        self.ind = 0
        self.l = line
        self.freqs = freqs
        self.t =

    def next(self, event):
        self.ind += 1
        i = self.ind % len(self.freqs)
        ydata = np.sin(2*np.pi*self.freqs[i]*self.t)
        self.l.set_ydata(ydata)
        self.l.axes.figure.canvas.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(self.freqs)
        ydata = np.sin(2*np.pi*self.freqs[i]*self.t)
        self.l.set_ydata(ydata)
        self.l.axes.figure.canvas.draw()
        # fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure
        # ax2.plot(1, 1)
        # plt.show()
        # fig2.canvas.manager.window.move(100,100)

def test_call(t, freqs):
    fig, ax = plt.subplots()
    fig.set_tight_layout(False)
    # fig.canvas.mpl_connect('close_event', lambda _: fig.canvas.stop_event_loop()) 
    plt.subplots_adjust(bottom=0.5)
    # t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2*np.pi*freqs[0]*t)
    l, = plt.plot(t, s, lw=2)

    callback = Index(l, t, freqs)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    # plt.show(block=False)
    plt.show(block=True)
    # 

# freqs = np.arange(2, 20, 3)
# t = np.arange(0, 1, 0.001)
# test_call(t, freqs)