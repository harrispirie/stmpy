import matplotlib as mpl
# mpl.use('Qt5Agg')
import numpy as np
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.x = x
        self.y = y
        self.ly = ax.axvline(1, color='g', lw=1)
        self.lx = ax.axhline(1, color='g', lw=1)
        # self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if event.inaxes:
            indx = np.searchsorted(self.x, [event.xdata])[0]
            indy = np.searchsorted(self.y, [event.ydata])[0]
            x = self.x[indx]
            y = self.y[indy]
            self.lx.set_xdata(x)
            self.ly.set_ydata(y)
            self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()
        else:
            pass

class App(QtWidgets.QMainWindow):         
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.figure = plt.figure(figsize=(3, 3), dpi=300)
        self.canvas = FigureCanvas(self.figure)
        self.canvas_ax = self.canvas.figure.subplots()
        x = np.arange(0,4, 40)
        y = np.arange(0,4, 40)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(2*np.pi/20*xx)*np.sin(2*np.pi/40*yy)
        self.canvas_ax.imshow(z, cmap=mpl.cm.gray, origin='lower', aspect=1, extent=[x[0], x[-1], y[0], y[-1]])

        # Layout
        layout = QtWidgets.QVBoxLayout(self._main)
        layout.addWidget(self.canvas)
        # self.showMaximized()
        self.cursor = SnaptoCursor(self.canvas_ax, x, y)
        self.cid = self.canvas.mpl_connect('motion_notify_event', self.cursor.mouse_move)


if __name__ == '__main__':   
    app = QtWidgets.QApplication([])   
    ex = App()
    ex.show()
    app.exec_()