import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class SomeClass():

    def __init__(self):
        self.pause = False

        self.fig, ax = plt.subplots()
        ax.set_aspect("equal")

        self.movie = []
        nt = 10
        X,Y = np.meshgrid(np.arange(16), np.arange(16))

        for t in range(nt):
            data = np.sin((X+t*1)/3.)**2+ 1.5*np.cos((Y+t*1)/3.)**2
            pc = ax.pcolor(data)
            self.movie.append([pc])

        self.ani = animation.ArtistAnimation(self.fig, self.movie, interval=100)    
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        plt.show()

    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True

a = SomeClass()