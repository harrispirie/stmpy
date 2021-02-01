# def multi_slice_viewer(volume, ax):
class MSV:
    def __init__(self, volume, ax):
    # remove_keymap_conflicts({'j', 'k'})
        self.ax = ax
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        ax.figure.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)

    def on_mouse_scroll(self, event):

        if event.inaxes is None:
            return
        else:
            if event.button == 'up':
                self.previous_slice(self.ax)
            elif event.button == 'down':
                self.next_slice(self.ax)
            self.ax.figure.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[-1].set_array(volume[ax.index])

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[-1].set_array(volume[ax.index])

# def remove_keymap_conflicts(new_keys_set):
#     for prop in plt.rcParams:
#         if prop.startswith('keymap.'):
#             keys = plt.rcParams[prop]
#             remove_list = set(keys) & new_keys_set
#             for key in remove_list:
#                 keys.remove(key)

# ----- Load Data -----
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import stmpy

fname = '201201_FeSeTe_006_001.3ds'
data = stmpy.load(fname, biasOffset=False)
data.z = stmpy.tools.lineSubtract(data.Z, n=1) #needs to be passed
data.LIX = data.grid['LI Demod 1 X (A)'] /2 + data.grid['LI Demod 1 X [bwd] (A)']/2 #needs to be passed. Hoffman lab uses LIY
fig, ax = plt.subplots()
# multi_slice_viewer(data.LIX, ax
msv = MSV(data.LIX, ax)
plt.show()
